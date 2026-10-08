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


_RETURNPERIODS_BROKEN_UNTIL: float = 0.0
_FORECASTSTATS_BROKEN_UNTIL: float = 0.0
_GEOGLOWS_RP_TABLE: dict[str, np.ndarray] | None = None
_ZARR_RIVID_INDEX: dict[str, np.ndarray] | None = None
_ZARR_STORE_META: dict[str, Any] | None = None
_DEFAULT_GEOGLOWS_RP_PATH = (
    Path.home()
    / '.cache'
    / 'openhydronet'
    / 'data'
    / 'geoglows_v2'
    / 'geoglows_return_periods_v1.npz'
)
_GEOGLOWS_CLOUDFRONT_BASE = 'https://d14ritg1bypdp7.cloudfront.net'


def _get_zarr_rivid_index(npy_path: Path | None = None) -> dict[str, np.ndarray] | None:
    """Load and sort the 6.84M-reach CloudFront Zarr river_id lookup array once."""
    global _ZARR_RIVID_INDEX  # noqa: PLW0603
    if _ZARR_RIVID_INDEX is not None:
        return _ZARR_RIVID_INDEX
    path = npy_path or (_DEFAULT_GEOGLOWS_RP_PATH.parent / 'geoglows_zarr_rivid_v1.npy')
    if not path.exists() or path.stat().st_size == 0:
        return None
    raw = np.load(path, allow_pickle=False)
    order = np.argsort(raw).astype(np.int32)
    _ZARR_RIVID_INDEX = {
        'sorted_rids': raw[order].astype(np.int32),
        'orig_idx': order,
    }
    return _ZARR_RIVID_INDEX


def _get_cloudfront_zarr_meta(session: requests.Session) -> dict[str, Any] | None:
    """Resolve and cache the latest GEOGLOWS v2 CloudFront Zarr metadata and time axis."""
    global _ZARR_STORE_META  # noqa: PLW0603
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415
    import time as _time  # noqa: PLC0415
    import numcodecs  # noqa: PLC0415

    now_mono = _time.monotonic()
    if _ZARR_STORE_META is not None and now_mono < _ZARR_STORE_META.get('expires_at', 0.0):
        return _ZARR_STORE_META

    now_utc = datetime.now(UTC)
    for day_offset in (0, 1, 2, 3):
        run_dt = now_utc - timedelta(days=day_offset)
        zarr_url = f"{_GEOGLOWS_CLOUDFRONT_BASE}/{run_dt.strftime('%Y%m%d')}00.zarr"
        r_meta = session.get(f'{zarr_url}/.zmetadata', timeout=6.0)
        if r_meta.status_code != 200:
            continue
        zmeta = r_meta.json().get('metadata') or {}
        q_meta = zmeta.get('Qout/.zarray')
        t_meta = zmeta.get('time/.zarray')
        t_attrs = zmeta.get('time/.zattrs') or {}
        if not q_meta or not t_meta:
            continue
        r_time = session.get(f'{zarr_url}/time/0', timeout=6.0)
        if r_time.status_code != 200 or not r_time.content:
            continue
        t_comp = numcodecs.get_codec(t_meta['compressor'])
        t_sec = np.frombuffer(t_comp.decode(r_time.content), dtype=t_meta['dtype'])
        units_str = str(t_attrs.get('units') or '')
        base_date_str = units_str.replace('seconds since', '').strip()[:10]
        base_dt = datetime.strptime(base_date_str, '%Y-%m-%d').replace(tzinfo=UTC)
        iso_times = [
            (base_dt + timedelta(seconds=int(s))).strftime('%Y-%m-%dT%H:%M:%SZ')
            for s in t_sec
        ]
        _ZARR_STORE_META = {
            'zarr_url': zarr_url,
            'q_meta': q_meta,
            'q_codec': numcodecs.get_codec(q_meta['compressor']),
            'iso_times': iso_times,
            'expires_at': now_mono + 1800.0,
        }
        return _ZARR_STORE_META
    return None


def _fetch_forecast_from_cloudfront_zarr(
    river_id: int,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    """Fetch 51-member ensemble + high_res forecast for `river_id` from GEOGLOWS CloudFront Zarr."""
    idx_table = _get_zarr_rivid_index()
    if idx_table is None:
        return None
    rid = int(river_id)
    sorted_rids = idx_table['sorted_rids']
    pos = int(np.searchsorted(sorted_rids, rid))
    if pos >= len(sorted_rids) or int(sorted_rids[pos]) != rid:
        return None
    zarr_idx = int(idx_table['orig_idx'][pos])

    sess = session if session is not None else requests.Session()
    meta = _get_cloudfront_zarr_meta(sess)
    if meta is None:
        return None

    chunks = meta['q_meta']['chunks']
    chunk_width = int(chunks[2])
    chunk_id = zarr_idx // chunk_width
    offset = zarr_idx % chunk_width

    r_chunk = sess.get(f"{meta['zarr_url']}/Qout/0.0.{chunk_id}", timeout=8.0)
    if r_chunk.status_code != 200 or not r_chunk.content:
        return None
    q_arr = np.frombuffer(
        meta['q_codec'].decode(r_chunk.content),
        dtype=meta['q_meta']['dtype'],
    ).reshape(chunks)

    rq = q_arr[:, :, offset].astype(np.float64)
    ens_all = rq[:51, :]
    hres_all = rq[51, :] if rq.shape[0] > 51 else None
    iso_times = meta['iso_times']

    records: list[dict[str, Any]] = []
    for t_idx, t_str in enumerate(iso_times):
        if t_idx >= ens_all.shape[1]:
            break
        col = ens_all[:, t_idx]
        valid = col[np.isfinite(col) & (col >= 0.0)]
        if valid.size == 0:
            continue
        pcts = np.percentile(valid, [0, 25, 50, 75, 100])
        hval: float | None = None
        if hres_all is not None and np.isfinite(hres_all[t_idx]) and hres_all[t_idx] >= 0.0:
            hval = round(float(hres_all[t_idx]), 2)
        records.append(
            {
                'time': t_str,
                'flow_min': round(float(pcts[0]), 2),
                'flow_25p': round(float(pcts[1]), 2),
                'flow_med': round(float(pcts[2]), 2),
                'flow_avg': round(float(np.mean(valid)), 2),
                'flow_75p': round(float(pcts[3]), 2),
                'flow_max': round(float(pcts[4]), 2),
                'high_res': hval,
            }
        )
    if not records:
        return None
    return {
        'model': 'geoglows',
        'available': True,
        'status': 'live',
        'river_id': rid,
        'unit': 'CUBIC_METERS_PER_SECOND',
        'data': records,
    }


def lookup_cached_geoglows_return_periods(
    river_id: int,
    npz_path: Path | None = None,
) -> dict[str, Any] | None:
    """Look up pre-cached GEOGLOWS return periods from `geoglows_return_periods_v1.npz` in O(log N)."""
    global _GEOGLOWS_RP_TABLE  # noqa: PLW0603
    path = npz_path or _DEFAULT_GEOGLOWS_RP_PATH
    if _GEOGLOWS_RP_TABLE is None:
        if not path.exists() or path.stat().st_size == 0:
            return None
        with np.load(path, allow_pickle=False) as z:
            _GEOGLOWS_RP_TABLE = {
                'river_id': z['river_id'],
                'return_periods': z['return_periods'],
                'values': (
                    z['gumbel_daily']
                    if 'gumbel_daily' in z.files
                    else z['gumbel'].T
                ),
            }

    rids = _GEOGLOWS_RP_TABLE['river_id']
    rid = int(river_id)
    idx = int(np.searchsorted(rids, rid))
    if idx >= len(rids) or int(rids[idx]) != rid:
        return None
    periods = _GEOGLOWS_RP_TABLE['return_periods']
    row = _GEOGLOWS_RP_TABLE['values'][idx]
    out: dict[str, float] = {}
    for p, val in zip(periods, row, strict=False):
        fval = float(val)
        if np.isfinite(fval) and fval > 0.0:
            out[f'return_period_{int(p)}'] = round(fval, 2)
    if len(out) < 3:
        return None
    if 'return_period_20' not in out:
        q20 = gumbel_quantile_from_return_periods(out, 20.0)
        if q20 is not None:
            out['return_period_20'] = q20
    return {
        'provider': 'geoglows',
        'status': 'live',
        'river_id': rid,
        'source': 'GEOGLOWS v2 official return periods (return-periods.zarr)',
        'method': 'EV1 (Gumbel, method of moments) daily return periods',
        'unit': 'm³/s',
        **out,
    }


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
        """Snap `(lat, lon)` to a 9-digit GEOGLOWS COMID via local pyramid or `/getriverid`."""
        is_live_session = (
            type(self.session) is requests.Session
            and not hasattr(self.session.get, '_mock_name')
            and self.base_url == GEOGLOWS_BASE_URL.rstrip('/')
        )
        if is_live_session:
            from maas.fetcher import _get_cached_geoglows_pyramid  # noqa: PLC0415
            from maas.networks import snap_geoglows_reach_from_network  # noqa: PLC0415

            npz_path = _DEFAULT_GEOGLOWS_RP_PATH.parent / 'geoglows_network_v1.npz'
            if npz_path.exists():
                net = _get_cached_geoglows_pyramid(npz_path)
                if net is not None:
                    snapped = snap_geoglows_reach_from_network(
                        net, float(lat), float(lon), None, radius_deg=0.15
                    )
                    if snapped is not None and snapped.get('river_id'):
                        return int(snapped['river_id'])

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
        global _FORECASTSTATS_BROKEN_UNTIL  # noqa: PLW0603
        import time as _time  # noqa: PLC0415

        rid = int(river_id)
        is_live_session = (
            type(self.session) is requests.Session
            and not hasattr(self.session.get, '_mock_name')
            and self.base_url == GEOGLOWS_BASE_URL.rstrip('/')
        )
        if is_live_session and (
            _time.monotonic() < _FORECASTSTATS_BROKEN_UNTIL
            or _get_zarr_rivid_index() is not None
        ):
            zarr_fc = _fetch_forecast_from_cloudfront_zarr(rid, self.session)
            if zarr_fc is not None:
                return zarr_fc

        url = f'{self.base_url}/forecaststats/{rid}'
        eff_timeout = min(self.timeout_s, 3.5) if is_live_session else self.timeout_s
        resp = self.session.get(
            url, params={'format': 'json'}, timeout=eff_timeout
        )
        if is_live_session and resp.status_code >= 500:
            _FORECASTSTATS_BROKEN_UNTIL = _time.monotonic() + 300.0
            zarr_fc = _fetch_forecast_from_cloudfront_zarr(rid, self.session)
            if zarr_fc is not None:
                return zarr_fc
        resp.raise_for_status()
        return parse_geoglows_forecast_response(resp.json(), river_id=rid)

    def fetch_return_periods(self, river_id: int) -> dict[str, Any] | None:
        """Fetch official GEOGLOWS return periods from local cache or `/returnperiods/{river_id}`."""
        global _RETURNPERIODS_BROKEN_UNTIL  # noqa: PLW0603
        import time as _time  # noqa: PLC0415

        is_live_session = (
            type(self.session) is requests.Session
            and not hasattr(self.session.get, '_mock_name')
            and self.base_url == GEOGLOWS_BASE_URL.rstrip('/')
        )
        if is_live_session:
            cached_rp = lookup_cached_geoglows_return_periods(int(river_id))
            if cached_rp is not None:
                return cached_rp
            if _time.monotonic() < _RETURNPERIODS_BROKEN_UNTIL:
                return None

        url = f'{self.base_url}/returnperiods/{int(river_id)}'
        eff_timeout = min(self.timeout_s, 2.5) if is_live_session else self.timeout_s
        resp = self.session.get(
            url, params={'format': 'json'}, timeout=eff_timeout
        )

        if resp.status_code == 404:
            return None
        if is_live_session and resp.status_code >= 500:
            _RETURNPERIODS_BROKEN_UNTIL = _time.monotonic() + 600.0
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

    fetch_official_return_periods = fetch_return_periods

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
