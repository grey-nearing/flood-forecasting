# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GEOGLOWS ECMWF v2 REST and Zarr client for forecasts and retrospective series."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import requests
import xarray as xr

from maas.config import (
    GEOGLOWS_BASE_URL,
    RETURN_PERIOD_YEARS,
    parse_finite_float,
    parse_int,
)
from maas.thresholds import (
    compute_empirical_weibull_return_periods,
    compute_gumbel_return_periods,
    compute_return_periods,
    extract_annual_maxima,
    gumbel_quantile_from_return_periods,
)


def _opt_rounded(
    seq: Sequence[Any], idx: int, ndigits: int = 2
) -> float | None:
    if idx >= len(seq):
        return None
    val = parse_finite_float(seq[idx])
    return round(val, ndigits) if val is not None else None


def parse_geoglows_forecast_response(
    payload: Mapping[str, Any] | None,
    river_id: int,
) -> dict[str, Any]:
    """Parse GEOGLOWS `/forecaststats/{river_id}` JSON into normalized records.

    Skips intermediate hourly rows where both `flow_med` and `flow_avg` are
    blank strings so blank values are never turned into spurious `0.0` flows.
    """
    records: list[dict[str, Any]] = []
    if isinstance(payload, Mapping) and 'datetime' in payload:
        times = payload.get('datetime') or []
        meds = payload.get('flow_med') or []
        avgs = payload.get('flow_avg') or []
        maxs = payload.get('flow_max') or []
        mins = payload.get('flow_min') or []
        p25s = payload.get('flow_25p') or []
        p75s = payload.get('flow_75p') or []
        high_res = payload.get('high_res') or []

        for i, raw_t in enumerate(times):
            med_val = _opt_rounded(meds, i)
            avg_val = _opt_rounded(avgs, i)
            if med_val is None and avg_val is None:
                continue
            records.append(
                {
                    'time': str(raw_t),
                    'flow_med': med_val,
                    'flow_avg': avg_val,
                    'flow_max': _opt_rounded(maxs, i),
                    'flow_min': _opt_rounded(mins, i),
                    'flow_25p': _opt_rounded(p25s, i),
                    'flow_75p': _opt_rounded(p75s, i),
                    'high_res': _opt_rounded(high_res, i),
                }
            )

    has_live = len(records) > 0
    return {
        'model': 'geoglows',
        'available': has_live,
        'status': 'live' if has_live else 'unavailable',
        'river_id': river_id,
        'unit': 'CUBIC_METERS_PER_SECOND',
        'data': records,
    }


def parse_geoglows_return_periods_payload(
    payload: Any,
    river_id: int,
) -> dict[str, float] | None:
    """Parse GEOGLOWS v2 `/returnperiods` JSON payload variants."""
    if not isinstance(payload, Mapping):
        return None
    out: dict[str, float] = {}

    def _take(period: Any, value: Any) -> None:
        val_inner = value
        if isinstance(val_inner, Mapping):
            val_inner = val_inner.get(
                str(river_id),
                next(iter(val_inner.values()), None) if val_inner else None,
            )
        if isinstance(val_inner, list):
            val_inner = val_inner[0] if val_inner else None
        cleaned = str(period).strip().lower().replace('return_period_', '')
        p_float = parse_finite_float(cleaned)
        if p_float is None or not p_float.is_integer():
            return
        t = int(p_float)
        fv = parse_finite_float(val_inner)
        if fv is not None and t in (2, 5, 10, 20, 25, 50, 100):
            out.setdefault(f'return_period_{t}', round(fv, 2))

    candidates: list[Mapping[str, Any]] = [payload]
    for key in (
        str(river_id),
        'return_periods',
        'returnperiods',
        'gumbel',
        'data',
    ):
        sub = payload.get(key)
        if isinstance(sub, Mapping):
            candidates.append(sub)
    for cand in candidates:
        for key, val in cand.items():
            skey = str(key).strip()
            if skey.startswith('return_period_') or skey.isdigit():
                _take(key, val)
    periods = payload.get('return_period')
    values = (
        payload.get('gumbel')
        if payload.get('gumbel') is not None
        else payload.get('logpearson3')
    )
    if not out and isinstance(periods, list) and isinstance(values, list):
        for period, value in zip(periods, values, strict=False):
            _take(period, value)
    if len(out) < 3:
        return None
    if 'return_period_20' not in out:
        q20 = gumbel_quantile_from_return_periods(out, 20.0)
        if q20 is not None:
            out['return_period_20'] = q20
    return out


def parse_geoglows_retrospective_response(
    payload: Mapping[str, Any] | None,
    river_id: int,
    method: str = 'gumbel',
    official_endpoint_error: str | None = None,
) -> dict[str, Any] | None:
    """Compute return periods from GEOGLOWS `/retrospectivedaily/{river_id}` JSON."""
    if not isinstance(payload, Mapping):
        return None
    times = payload.get('datetime') or []
    values = payload.get(str(river_id))
    if not isinstance(values, list):
        values = next(
            (
                v
                for k, v in payload.items()
                if k not in ('datetime', 'metadata') and isinstance(v, list)
            ),
            None,
        )
    if not isinstance(values, list) or not times:
        return None
    annual_max = extract_annual_maxima(times, values)
    if len(annual_max) < 8:
        return None

    method_key = method.strip().lower()
    if method_key == 'gema':
        rps = compute_return_periods(
            annual_max,
            return_periods=RETURN_PERIOD_YEARS,
        )
        method_label = rps['method']
    elif method_key == 'weibull':
        rps = compute_empirical_weibull_return_periods(
            annual_max,
            return_periods=RETURN_PERIOD_YEARS,
        )
        method_label = rps['method']
    elif method_key in ('gumbel', 'ev1'):
        gumbel_rps = compute_gumbel_return_periods(annual_max)
        if not gumbel_rps:
            return None
        rps = dict(gumbel_rps)
        method_label = (
            'EV1 (Gumbel, method of moments) fit to calendar-year maxima '
            'of retrospectivedaily'
        )
    else:
        raise ValueError(
            f"Unsupported return period method {method!r}. Expected 'gumbel', 'gema', or 'weibull'."
        )

    valid = [
        v
        for v in (parse_finite_float(x) for x in values)
        if v is not None and v >= 0.0
    ]
    meta = (
        payload.get('metadata')
        if isinstance(payload.get('metadata'), Mapping)
        else {}
    )
    rp_fields = {
        k: v for k, v in rps.items() if str(k).startswith('return_period_')
    }
    return {
        'provider': 'geoglows',
        'status': 'computed',
        'river_id': river_id,
        'source': (
            'GEOGLOWS v2 retrospective simulation (EV1 fit computed locally)'
        ),
        'method': method_label,
        'years_of_record': len(annual_max),
        'record_start': meta.get('start_date') or str(times[0])[:10],
        'record_end': meta.get('end_date') or str(times[-1])[:10],
        'mean_flow': round(sum(valid) / len(valid), 3) if valid else None,
        'unit': 'm³/s',
        'official_endpoint_error': official_endpoint_error,
        **rp_fields,
    }


def extract_geoglows_zarr_series(
    zarr_path: Path,
    river_id: int,
    var_name: str = 'Qout',
) -> list[dict[str, Any]]:
    """Extract a reach discharge series from a local GEOGLOWS Zarr/NetCDF store."""
    if not zarr_path.exists():
        raise FileNotFoundError(f'GEOGLOWS archive not found: {zarr_path}')
    open_fn = (
        xr.open_zarr
        if zarr_path.is_dir() or zarr_path.suffix == '.zarr'
        else xr.open_dataset
    )
    with open_fn(zarr_path) as ds:
        if var_name not in ds:
            raise KeyError(
                f'Variable {var_name!r} not found in {zarr_path}; available: {list(ds.data_vars)}'
            )
        riv_dim = 'rivid' if 'rivid' in ds.coords else 'river_id'
        if riv_dim not in ds.coords:
            raise KeyError(
                f'Missing river ID coordinate in {zarr_path}; available: {list(ds.coords)}'
            )
        rivids = ds[riv_dim].to_numpy(dtype=np.int64)
        matches = np.flatnonzero(rivids == int(river_id))
        if len(matches) == 0:
            raise KeyError(f'river_id {river_id} not found in {zarr_path}')
        sub = ds[var_name].isel({riv_dim: int(matches[0])})
        times = sub['time'].to_numpy()
        vals = sub.to_numpy(dtype=float)

    records: list[dict[str, Any]] = []
    for t, v in zip(times.ravel(), vals.ravel(), strict=False):
        t_str = str(np.datetime_as_string(t, unit='s')) + 'Z'
        fval = float(v)
        records.append(
            {
                'time': t_str,
                'discharge': round(fval, 2)
                if np.isfinite(fval)
                else float('nan'),
            }
        )
    return records


class GeoGLOWSClient:
    """Client for GEOGLOWS ECMWF v2 REST endpoints."""

    def __init__(
        self,
        base_url: str = GEOGLOWS_BASE_URL,
        timeout_s: float = 12.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout_s = timeout_s
        self.session = session if session is not None else requests.Session()

    def fetch_river_id(self, lat: float, lon: float) -> int | None:
        """Query GEOGLOWS `/getriverid` to snap `(lat, lon)` to a 9-digit COMID."""
        url = f'{self.base_url}/getriverid'
        resp = self.session.get(
            url, params={'lat': lat, 'lon': lon}, timeout=self.timeout_s
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, Mapping) and 'river_id' in payload:
            return parse_int(payload['river_id'])
        return None

    def fetch_forecast(self, river_id: int) -> dict[str, Any]:
        """Fetch 15-day ensemble forecast statistics for `river_id`."""
        url = f'{self.base_url}/forecaststats/{int(river_id)}'
        resp = self.session.get(
            url, params={'format': 'json'}, timeout=self.timeout_s
        )
        resp.raise_for_status()
        return parse_geoglows_forecast_response(
            resp.json(), river_id=int(river_id)
        )

    def fetch_return_periods(self, river_id: int) -> dict[str, Any] | None:
        """Fetch official GEOGLOWS return periods from `/returnperiods/{river_id}`."""
        url = f'{self.base_url}/returnperiods/{int(river_id)}'
        resp = self.session.get(
            url, params={'format': 'json'}, timeout=self.timeout_s
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        rps = parse_geoglows_return_periods_payload(
            resp.json(), river_id=int(river_id)
        )
        if not rps:
            return None
        return {
            'provider': 'geoglows',
            'status': 'live',
            'river_id': int(river_id),
            'source': 'GEOGLOWS v2 official return periods (/returnperiods)',
            'method': 'Official GEOGLOWS return-period dataset',
            'unit': 'm³/s',
            **rps,
        }

    def fetch_retrospective_return_periods(
        self,
        river_id: int,
        method: str = 'gumbel',
        timeout_s: float = 60.0,
    ) -> dict[str, Any] | None:
        """Fetch daily retrospective series and compute return periods."""
        url = f'{self.base_url}/retrospectivedaily/{int(river_id)}'
        resp = self.session.get(
            url, params={'format': 'json'}, timeout=timeout_s
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return parse_geoglows_retrospective_response(
            resp.json(),
            river_id=int(river_id),
            method=method,
        )


__all__ = [
    'GeoGLOWSClient',
    'extract_geoglows_zarr_series',
    'parse_geoglows_forecast_response',
    'parse_geoglows_retrospective_response',
    'parse_geoglows_return_periods_payload',
]
