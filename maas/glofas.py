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

"""Copernicus GloFAS v4 Open-Meteo/CDS/Zarr extractor and grid snapper."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
import xarray as xr

from maas.config import (
    GLOFAS_BASE_URL,
    GLOFAS_RES_DEG,
    RETURN_PERIOD_YEARS,
    parse_finite_float,
)
from maas.networks import glofas_cell_center
from maas.thresholds import (
    compute_empirical_weibull_return_periods,
    compute_gumbel_return_periods,
    compute_return_periods,
    extract_annual_maxima,
)

_MIN_REANALYSIS_YEARS: int = 8

GLOFAS_DAILY_VARIABLES = (
    'river_discharge,'
    'river_discharge_mean,'
    'river_discharge_median,'
    'river_discharge_max,'
    'river_discharge_min,'
    'river_discharge_p25,'
    'river_discharge_p75'
)


def _extract_rounded_or_none(
    seq: Sequence[Any],
    idx: int,
    ndigits: int = 2,
) -> float | None:
    if idx >= len(seq):
        return None
    val = parse_finite_float(seq[idx])
    return round(val, ndigits) if val is not None else None


def _validate_daily_sequence(
    daily: Mapping[str, Any],
    key: str,
    expected_len: int | None = None,
) -> Sequence[Any]:
    """Validate `daily[key]` exists, is a sequence, and has `expected_len`."""
    if key not in daily:
        raise KeyError(
            f'Missing required key {key!r} in GloFAS daily forecast payload'
        )
    seq = daily[key]
    if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
        raise TypeError(
            f'Expected sequence for {key!r} in GloFAS daily forecast payload'
        )
    if expected_len is not None and len(seq) != expected_len:
        raise ValueError(
            f'Length mismatch for {key!r}: '
            f'expected {expected_len}, got {len(seq)}'
        )
    return seq


def parse_glofas_forecast_response(
    payload: Mapping[str, Any] | None,
    lat: float,
    lon: float,
) -> dict[str, Any]:
    """Parse an Open-Meteo `/v1/flood` JSON payload into GloFAS records.

    Missing values are preserved as `None` (never replaced with `0.0`).
    Extracts the deterministic control trace (`discharge_deterministic` from
    `river_discharge`), summary statistics (`mean`, `median`, `min`, `max`,
    `p25`, `p75`), and any individual perturbed ensemble members
    (`river_discharge_member01`..`river_discharge_member50`).
    """
    records: list[dict[str, Any]] = []
    ensemble_members: dict[str, list[float | None]] = {}
    if isinstance(payload, Mapping) and isinstance(
        payload.get('daily'), Mapping
    ):
        daily = payload['daily']
        times = _validate_daily_sequence(daily, 'time')
        n_steps = len(times)
        dets = _validate_daily_sequence(daily, 'river_discharge', n_steps)
        means = _validate_daily_sequence(daily, 'river_discharge_mean', n_steps)
        meds = _validate_daily_sequence(
            daily, 'river_discharge_median', n_steps
        )
        maxs = _validate_daily_sequence(daily, 'river_discharge_max', n_steps)
        mins = _validate_daily_sequence(daily, 'river_discharge_min', n_steps)
        p25s = _validate_daily_sequence(daily, 'river_discharge_p25', n_steps)
        p75s = _validate_daily_sequence(daily, 'river_discharge_p75', n_steps)
        for i, raw_t in enumerate(times):
            records.append(
                {
                    'time': str(raw_t),
                    'discharge_deterministic': _extract_rounded_or_none(
                        dets, i
                    ),
                    'discharge_mean': _extract_rounded_or_none(means, i),
                    'discharge_median': _extract_rounded_or_none(meds, i),
                    'discharge_max': _extract_rounded_or_none(maxs, i),
                    'discharge_min': _extract_rounded_or_none(mins, i),
                    'discharge_p25': _extract_rounded_or_none(p25s, i),
                    'discharge_p75': _extract_rounded_or_none(p75s, i),
                }
            )
        member_keys = sorted(
            k for k in daily if str(k).startswith('river_discharge_member')
        )
        for mk in member_keys:
            seq = _validate_daily_sequence(daily, str(mk), n_steps)
            short_key = str(mk).replace('river_discharge_', '')
            ensemble_members[short_key] = [
                _extract_rounded_or_none(seq, i) for i in range(n_steps)
            ]
    has_live = len(records) > 0
    return {
        'model': 'copernicus_glofas',
        'available': has_live,
        'status': 'live' if has_live else 'unavailable',
        'lat': lat,
        'lon': lon,
        'unit': 'CUBIC_METERS_PER_SECOND',
        'data': records,
        'ensemble_members': ensemble_members,
    }


def compute_glofas_reanalysis_return_periods(
    payload: Mapping[str, Any] | None,
    end_year: int,
    method: str = 'gumbel',
) -> dict[str, Any] | None:
    """Compute GloFAS v4 return periods from a daily reanalysis JSON payload."""
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get('daily'), Mapping
    ):
        return None
    daily = payload['daily']
    times = daily.get('time') or []
    values = daily.get('river_discharge') or []
    annual_max = extract_annual_maxima(times, values)
    if len(annual_max) < _MIN_REANALYSIS_YEARS:
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
            'EV1 (Gumbel, method of moments) fit to calendar-year maxima'
        )
    else:
        raise ValueError(
            f'Unsupported return period method {method!r}. '
            "Expected 'gumbel', 'gema', or 'weibull'."
        )

    valid = [
        v
        for v in (parse_finite_float(x) for x in values)
        if v is not None and v >= 0.0
    ]
    rp_fields = {
        k: v for k, v in rps.items() if str(k).startswith('return_period_')
    }
    return {
        'provider': 'glofas',
        'status': 'live',
        'source': (
            f'Copernicus GloFAS v4 reanalysis (Open-Meteo, 1984-{end_year})'
        ),
        'method': method_label,
        'years_of_record': len(annual_max),
        'mean_flow': round(sum(valid) / len(valid), 3) if valid else None,
        'grid_lat': payload.get('latitude'),
        'grid_lon': payload.get('longitude'),
        'unit': 'm³/s',
        **rp_fields,
    }


def extract_glofas_zarr_series(
    zarr_path: Path,
    lat: float,
    lon: float,
    var_name: str = 'dis24',
) -> list[dict[str, Any]]:
    """Extract a point discharge series from a local GloFAS Zarr/NetCDF archive.

    Snaps `(lat, lon)` to the 0.05° LISFLOOD cell center and preserves `NaN`
    for any missing timestep.
    """
    if not zarr_path.exists():
        raise FileNotFoundError(f'GloFAS archive not found: {zarr_path}')
    cell_lat, cell_lon = glofas_cell_center(lat, lon, GLOFAS_RES_DEG)
    open_fn = (
        xr.open_zarr
        if zarr_path.is_dir() or zarr_path.suffix == '.zarr'
        else xr.open_dataset
    )
    with open_fn(zarr_path) as ds:
        if var_name not in ds:
            raise KeyError(
                f'Variable {var_name!r} not found in {zarr_path}; '
                f'available: {list(ds.data_vars)}'
            )
        lat_coord = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_coord = 'longitude' if 'longitude' in ds.coords else 'lon'
        if lat_coord not in ds.coords or lon_coord not in ds.coords:
            raise KeyError(
                f'Missing latitude/longitude coordinates in {zarr_path}: '
                f'{list(ds.coords)}'
            )
        lats = ds[lat_coord].to_numpy(dtype=float)
        lons = ds[lon_coord].to_numpy(dtype=float)
        lat_idx = int(np.argmin(np.abs(lats - cell_lat)))
        lon_idx = int(np.argmin(np.abs(lons - cell_lon)))
        if (
            abs(float(lats[lat_idx]) - cell_lat) > GLOFAS_RES_DEG
            or abs(float(lons[lon_idx]) - cell_lon) > GLOFAS_RES_DEG
        ):
            raise ValueError(
                f'Cell ({cell_lat}, {cell_lon}) is outside archive domain '
                f'in {zarr_path}.'
            )
        sub = ds[var_name].isel({lat_coord: lat_idx, lon_coord: lon_idx})
        times = sub['time'].to_numpy()
        vals = sub.to_numpy(dtype=float)

    records: list[dict[str, Any]] = []
    for t, v in zip(times.ravel(), vals.ravel(), strict=False):
        t_str = str(np.datetime_as_string(t, unit='D'))
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


class GloFASClient:
    """Client for Copernicus GloFAS v4 forecasts and reanalysis."""

    def __init__(
        self,
        base_url: str = GLOFAS_BASE_URL,
        timeout_s: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the GloFAS HTTP client."""
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.session = session if session is not None else requests.Session()

    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 15,
    ) -> dict[str, Any]:
        """Fetch 15-day GloFAS deterministic and 50-member ensemble forecast."""
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': GLOFAS_DAILY_VARIABLES,
            'ensemble': 'true',
            'forecast_days': forecast_days,
        }
        resp = self.session.get(
            self.base_url, params=params, timeout=self.timeout_s
        )
        resp.raise_for_status()
        return parse_glofas_forecast_response(resp.json(), lat=lat, lon=lon)

    def fetch_reanalysis_return_periods(
        self,
        lat: float,
        lon: float,
        start_date: str = '1984-01-01',
        end_year: int | None = None,
        method: str = 'gumbel',
    ) -> dict[str, Any] | None:
        """Fetch GloFAS v4 historical reanalysis and compute return periods."""
        eff_end_year = (
            end_year if end_year is not None else datetime.now(UTC).year - 1
        )
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'river_discharge',
            'start_date': start_date,
            'end_date': f'{eff_end_year}-12-31',
        }
        resp = self.session.get(
            self.base_url, params=params, timeout=self.timeout_s
        )
        resp.raise_for_status()
        return compute_glofas_reanalysis_return_periods(
            resp.json(),
            end_year=eff_end_year,
            method=method,
        )


__all__ = [
    'GLOFAS_DAILY_VARIABLES',
    'GloFASClient',
    'compute_glofas_reanalysis_return_periods',
    'extract_glofas_zarr_series',
    'parse_glofas_forecast_response',
]
