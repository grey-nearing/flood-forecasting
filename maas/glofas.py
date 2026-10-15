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

# Native GloFAS v4 summary series supplied by Open-Meteo `/v1/flood`, keyed by
# the short reference name used in `glofas_reference` API/query parameters.
# Each maps to the per-record field name written by
# `parse_glofas_forecast_response`.
GLOFAS_REFERENCE_FIELDS: dict[str, str] = {
    'control': 'discharge_deterministic',
    'median': 'discharge_median',
    'mean': 'discharge_mean',
}
GLOFAS_REFERENCE_LABELS: dict[str, str] = {
    'control': 'Control (Unperturbed)',
    'median': 'Ensemble Median',
    'mean': 'Ensemble Mean',
}
DEFAULT_GLOFAS_REFERENCE: str = 'control'


def normalize_glofas_reference(reference: str | None) -> str:
    """Validate a GloFAS reference-series name.

    Accepted names are `'control'`, `'median'`, and `'mean'`.

    Args:
        reference: Requested reference name (case-insensitive). `None` selects
            `DEFAULT_GLOFAS_REFERENCE`.

    Returns:
        The canonical lower-case reference key.

    Raises:
        TypeError: If `reference` is not a string.
        ValueError: If `reference` is not one of `GLOFAS_REFERENCE_FIELDS`.
    """
    if reference is None:
        return DEFAULT_GLOFAS_REFERENCE
    if not isinstance(reference, str):
        raise TypeError(
            f'`glofas_reference` must be a string, got {type(reference)!r}'
        )
    key = reference.strip().lower()
    if key not in GLOFAS_REFERENCE_FIELDS:
        raise ValueError(
            f'Unsupported GloFAS reference series {reference!r}; expected one '
            f'of {sorted(GLOFAS_REFERENCE_FIELDS)}.'
        )
    return key


def glofas_reference_field(reference: str | None) -> str:
    """Return the per-record field name for a GloFAS reference series."""
    return GLOFAS_REFERENCE_FIELDS[normalize_glofas_reference(reference)]


def describe_glofas_reference(reference: str | None) -> dict[str, Any]:
    """Describe the selected GloFAS reference series for API payloads.

    Returns:
        Dictionary with the canonical `key`, the per-record `field`, a short
        human-readable `label`, and the list of all available `options`.
    """
    key = normalize_glofas_reference(reference)
    return {
        'key': key,
        'field': GLOFAS_REFERENCE_FIELDS[key],
        'label': GLOFAS_REFERENCE_LABELS[key],
        'options': [
            {
                'key': opt,
                'field': GLOFAS_REFERENCE_FIELDS[opt],
                'label': GLOFAS_REFERENCE_LABELS[opt],
            }
            for opt in GLOFAS_REFERENCE_FIELDS
        ],
    }


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


def _select_glofas_point_series(
    da: xr.DataArray,
    *,
    step_index: int | None,
    member_index: int | None,
    zarr_path: Path,
) -> xr.DataArray:
    """Reduce a point-sliced GloFAS array to a strictly 1-D `time` series.

    Optional forecast lead-time (`step`) and ensemble member (`number`)
    dimensions must be resolved by an explicit index; they are never selected
    or averaged implicitly.

    Raises:
        ValueError: If `step` / `number` is present without an explicit index,
            if an index is out of range, if `time` is missing, or if any other
            dimension remains after selection.
    """
    dims = tuple(str(d) for d in da.dims)
    if 'time' not in dims:
        raise ValueError(
            f'GloFAS archive {zarr_path} has no `time` dimension on the '
            f'point series; found dims {dims}.'
        )
    for dim_name, arg_name, what, index in (
        ('step', 'step_index', 'lead time', step_index),
        ('number', 'member_index', 'ensemble member', member_index),
    ):
        if dim_name not in dims:
            continue
        size = int(da.sizes[dim_name])
        if index is None:
            raise ValueError(
                f'GloFAS archive {zarr_path} has a `{dim_name}` dimension of '
                f'size {size}; pass an explicit `{arg_name}` to select one '
                f'{what}.'
            )
        if not 0 <= index < size:
            raise ValueError(
                f'`{arg_name}`={index} is out of range for the `{dim_name}` '
                f'dimension of {zarr_path} (size {size}).'
            )
        da = da.isel({dim_name: index})
    remaining = tuple(str(d) for d in da.dims)
    if remaining != ('time',):
        raise ValueError(
            f'Unexpected dimensions {remaining} on GloFAS point series in '
            f'{zarr_path}; expected exactly ("time",).'
        )
    return da


def extract_glofas_zarr_series(  # noqa: PLR0913
    zarr_path: Path,
    lat: float,
    lon: float,
    var_name: str = 'dis24',
    *,
    step_index: int | None = None,
    member_index: int | None = None,
) -> list[dict[str, Any]]:
    """Extract a point discharge series from a local GloFAS Zarr/NetCDF archive.

    Snaps `(lat, lon)` to the 0.05° LISFLOOD cell center and preserves `NaN`
    for any missing timestep. Archives that carry a forecast lead-time
    dimension (`step`) or an ensemble member dimension (`number`) require an
    explicit `step_index` / `member_index`; any other extra dimension raises
    `ValueError`.

    Args:
        zarr_path: Local `.zarr` directory or NetCDF file.
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        var_name: Discharge variable name (default `'dis24'`).
        step_index: Positional index along `step` when that dimension exists.
        member_index: Positional index along `number` when that dimension
            exists.

    Returns:
        List of records with a ``time`` (``YYYY-MM-DD``) and a ``discharge``
        in m³/s (``NaN`` for missing timesteps).
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
        lats = np.asarray(ds[lat_coord].to_numpy(), dtype=float)
        lons = np.asarray(ds[lon_coord].to_numpy(), dtype=float)
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
        sub = _select_glofas_point_series(
            ds[var_name].isel({lat_coord: lat_idx, lon_coord: lon_idx}),
            step_index=step_index,
            member_index=member_index,
            zarr_path=zarr_path,
        )
        times = sub['time'].to_numpy()
        vals = np.asarray(sub.to_numpy(), dtype=float)

    records: list[dict[str, Any]] = []
    for t, v in zip(times.ravel(), vals.ravel(), strict=True):
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
    'DEFAULT_GLOFAS_REFERENCE',
    'GLOFAS_DAILY_VARIABLES',
    'GLOFAS_REFERENCE_FIELDS',
    'GLOFAS_REFERENCE_LABELS',
    'GloFASClient',
    'compute_glofas_reanalysis_return_periods',
    'describe_glofas_reference',
    'extract_glofas_zarr_series',
    'glofas_reference_field',
    'normalize_glofas_reference',
    'parse_glofas_forecast_response',
]
