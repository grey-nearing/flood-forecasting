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
"""Real-Time Forecasting & Hot-Start Service for OpenHydroNet.

Connects the Forecasting and Data Assimilation UI tab directly to:
1. ``multimet.timeseries_extractors.realtime.fetch_realtime_multimet``
   (Cold-Start 365-day spin-up fetch and Hot-Start incremental catch-up across
   CPC, IMERG, HRES, ECMWF AIFS, NOAA GFS, NOAA GEFS, and ECMWF IFS-ENS).
2. ``model`` (``MeanEmbeddingForecastLSTM``) for Cold-Start (``forward()`` +
   ``save_state()``) and Hot-Start (``load_state_from_disk()`` + incremental
   hindcast state advancement + forecast rollout) inference.
3. Strict state-date validation (``t0 - seq_length <= state_date <= t0 - 1``)
   and strict ``NaN`` preservation for real-time meteorological latency gaps.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from frontend import profile_manager
from frontend.config import (
    FLOOD_FORECASTING_REPO_DIR,
    ensure_flood_forecasting_on_sys_path,
    extend_multimet_package_path,
)
from model.datasetzoo.caravan import load_caravan_attributes
from model.datautils.scaler import Scaler
from model.modelzoo import get_model
from model.utils.config import Config
from model.utils.configutils import (
    flatten_feature_list,
    product_name_from_feature,
)
from multimet.timeseries_extractors import config as mm_config
from multimet.timeseries_extractors import realtime as mm_realtime
from multimet.timeseries_extractors.hres import find_latest_hres_open_data_date

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_RUN_DIR = (
    FLOOD_FORECASTING_REPO_DIR
    / 'model'
    / 'tutorial'
    / 'model-runs'
    / '5-basin-example'
)
DEFAULT_TUTORIAL_CARAVAN_DIR = (
    FLOOD_FORECASTING_REPO_DIR / 'model' / 'tutorial' / 'Caravan-nc'
)

_MIN_FORECAST_LEAD_STEPS = 2
_SECONDS_PER_DAY = 86400.0
_CMS_PER_MM_DAY_KM2 = 1000.0 / _SECONDS_PER_DAY
_DEFAULT_SPINUP_DAYS = 365
_DEFAULT_LOOKBACK_DAYS = 14
_DEFAULT_LEAD_TIME_DAYS = 7
_HINDCAST_TAIL_DAYS = 14
_NDIM_3D = 3
_DEFAULT_MODEL_ID = '5-basin-example'

NOWCAST_PRODUCTS_METADATA: dict[str, dict[str, Any]] = {
    'IMERG': {
        'id': 'IMERG',
        'label': 'NASA GPM IMERG Final/Late Run',
        'var_name': 'imerg_precipitation',
        'typical_latency_days': 1,
    },
    'CPC': {
        'id': 'CPC',
        'label': 'NOAA CPC Unified Gauge-Based Analysis',
        'var_name': 'cpc_precipitation',
        'typical_latency_days': 2,
    },
    'ERA5_LAND': {
        'id': 'ERA5_LAND',
        'label': 'ECMWF ERA5-Land Nowcast',
        'var_name': 'era5land_total_precipitation',
        'typical_latency_days': 5,
    },
    'CHIRPS': {
        'id': 'CHIRPS',
        'label': 'UCSB CHIRPS v2.0 / v3.0 Daily',
        'var_name': 'chirps_precipitation',
        'typical_latency_days': 2,
    },
}

FORECAST_PRODUCTS_METADATA: dict[str, dict[str, Any]] = {
    'HRES': {
        'id': 'HRES',
        'label': 'ECMWF IFS HRES (0.25° Open Data)',
        'precip_var': 'hres_total_precipitation',
        'temp_var': 'hres_temperature_2m',
    },
    'ECMWF_AIFS': {
        'id': 'ECMWF_AIFS',
        'label': 'ECMWF AIFS Single (AI Weather Model)',
        'precip_var': 'aifs_total_precipitation',
        'temp_var': 'aifs_temperature_2m',
    },
    'AIFS': {
        'id': 'AIFS',
        'label': 'ECMWF AIFS Single (AI Weather Model)',
        'precip_var': 'aifs_total_precipitation',
        'temp_var': 'aifs_temperature_2m',
    },
    'NOAA_GFS': {
        'id': 'NOAA_GFS',
        'label': 'NOAA GFS (Global Forecast System)',
        'precip_var': 'gfs_total_precipitation',
        'temp_var': 'gfs_temperature_2m',
    },
    'GFS': {
        'id': 'GFS',
        'label': 'NOAA GFS (Global Forecast System)',
        'precip_var': 'gfs_total_precipitation',
        'temp_var': 'gfs_temperature_2m',
    },
    'NOAA_GEFS': {
        'id': 'NOAA_GEFS',
        'label': 'NOAA GEFS Ensemble Mean (31-Member)',
        'precip_var': 'gefs_total_precipitation',
        'temp_var': 'gefs_temperature_2m',
    },
    'GEFS': {
        'id': 'GEFS',
        'label': 'NOAA GEFS Ensemble Mean (31-Member)',
        'precip_var': 'gefs_total_precipitation',
        'temp_var': 'gefs_temperature_2m',
    },
    'ECMWF_IFS_ENS': {
        'id': 'ECMWF_IFS_ENS',
        'label': 'ECMWF IFS Ensemble Mean (51-Member)',
        'precip_var': 'ifs_ens_total_precipitation',
        'temp_var': 'ifs_ens_temperature_2m',
    },
    'IFS_ENS': {
        'id': 'IFS_ENS',
        'label': 'ECMWF IFS Ensemble Mean (51-Member)',
        'precip_var': 'ifs_ens_total_precipitation',
        'temp_var': 'ifs_ens_temperature_2m',
    },
}

_KNOWN_PRODUCT_PREFIXES: tuple[str, ...] = (
    'era5land_',
    'ifs_ens_',
    'chirpsgefs_',
    'graphcast_',
    'chirps_',
    'imerg_',
    'hres_',
    'aifs_',
    'gefs_',
    'gfs_',
    'cpc_',
)

_VARIABLE_DESCRIPTORS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        'precip',
        'precipitation',
        'mm/day',
        'Total Precipitation (mm/day)',
        'bar',
    ),
    ('dewpoint', 'dewpoint', '°C', '2m Dewpoint Temperature (°C)', 'line'),
    ('temperature', 'temperature', '°C', '2m Air Temperature (°C)', 'line'),
    (
        'solar_radiation',
        'solar_radiation',
        'W/m²',
        'Surface Net Solar Radiation (W/m²)',
        'line',
    ),
    (
        'thermal_radiation',
        'thermal_radiation',
        'W/m²',
        'Surface Net Thermal Radiation (W/m²)',
        'line',
    ),
    ('pressure', 'pressure', 'kPa', 'Surface Pressure (kPa)', 'line'),
    ('humidity', 'humidity', 'kg/kg', 'Specific Humidity (kg/kg)', 'line'),
    (
        'u_component_of_wind',
        'wind_u',
        'm/s',
        '10m U-Wind Component (m/s)',
        'line',
    ),
    (
        'v_component_of_wind',
        'wind_v',
        'm/s',
        '10m V-Wind Component (m/s)',
        'line',
    ),
    (
        'num_stations',
        'stations',
        'count',
        'Reporting Gauge Stations (count)',
        'bar',
    ),
)

_MODEL_CACHE: dict[str, dict[str, Any]] = {}


@dataclasses.dataclass(frozen=True)
class OverrideConfig:
    """Normalized What-If override configuration."""

    precip_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    temp_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    product_overrides: dict[str, dict[str, dict[str, float]]] = (
        dataclasses.field(default_factory=dict)
    )
    is_what_if: bool = False

    @property
    def has_any(self) -> bool:
        """Return True if any precip, temp, or product override is present."""
        return bool(
            self.precip_overrides
            or self.temp_overrides
            or self.product_overrides
        )


@dataclasses.dataclass(frozen=True)
class ProductSliceContext:
    """Context for extracting nowcast/forecast series from a Zarr dataset."""

    prod_name: str
    store_path: Path
    sub: xr.Dataset
    all_vars: list[str]
    variables_meta: list[dict[str, Any]]
    nowcast_dates: pd.DatetimeIndex
    t0: pd.Timestamp


@dataclasses.dataclass(frozen=True)
class TensorBuildContext:
    """Context for building scaled PyTorch feature tensors for one catchment."""

    catchment_id: str
    hindcast_dates: pd.DatetimeIndex
    forecast_dates: pd.DatetimeIndex
    t0: pd.Timestamp
    scaler: Scaler
    overrides: OverrideConfig


def _is_valid_zarr_store(store_path: Path) -> bool:
    """Return True if ``store_path`` is an existing Zarr v2 or v3 directory."""
    if not store_path.is_dir():
        return False
    return any(
        (store_path / marker).exists()
        for marker in ('.zmetadata', '.zgroup', 'zarr.json')
    )


def _clean_float(val: object) -> float | None:
    """Convert a numeric value to a Python float, turning NaN/Inf into None."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, np.ndarray):
        if val.size != 1:
            return None
        val = val.item()
    if not isinstance(val, (int, float, np.number)):
        return None
    fval = float(val)
    if not math.isfinite(fval):
        return None
    return round(fval, 4)


def _format_date_str(ts: str | pd.Timestamp | np.datetime64) -> str:
    """Format a timestamp-like object as ``YYYY-MM-DD``."""
    return pd.Timestamp(ts).strftime('%Y-%m-%d')


def get_state_file_path(
    username: str | None,
    catchment_id: str,
    model_id: str = _DEFAULT_MODEL_ID,
) -> Path:
    """Return the canonical ``.npz`` state file path for a catchment."""
    pm = profile_manager.get_profile_manager()
    safe_cid = ''.join(
        c for c in str(catchment_id) if c.isalnum() or c in ('_', '-')
    )
    safe_mid = ''.join(
        c for c in str(model_id) if c.isalnum() or c in ('_', '-')
    )
    states_dir = pm.get_model_states_dir(username)
    return states_dir / f'{safe_cid}_{safe_mid}_state.npz'


def inspect_saved_state(
    username: str | None,
    catchment_id: str,
    model_id: str = _DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Read metadata (including ``date``) from a saved ``.npz`` state file."""
    state_path = get_state_file_path(username, catchment_id, model_id=model_id)
    if not state_path.is_file() or state_path.stat().st_size == 0:
        return {
            'exists': False,
            'state_path': str(state_path),
            'state_date': None,
            'hidden_shape': None,
        }
    with np.load(state_path, allow_pickle=False) as data:
        files = set(data.files)
        state_date = str(data['date'].item()) if 'date' in files else None
        h_key = (
            'h_n'
            if 'h_n' in files
            else ('h_hindcast' if 'h_hindcast' in files else None)
        )
        h_shape = list(data[h_key].shape) if h_key is not None else None
    return {
        'exists': True,
        'state_path': str(state_path),
        'state_date': state_date,
        'hidden_shape': h_shape,
    }


def _discover_zarr_stores(username: str | None) -> dict[str, Path]:
    """Find all ``<PRODUCT>/timeseries.zarr`` stores in user dynamics dirs."""
    pm = profile_manager.get_profile_manager()
    stores: dict[str, Path] = {}
    for base_dir in (
        pm.get_realtime_dynamics_dir(username),
        pm.get_dynamics_dir(username),
    ):
        if not base_dir.exists():
            continue
        for child in sorted(base_dir.iterdir()):
            if not child.is_dir():
                continue
            zarr_cand = child / 'timeseries.zarr'
            if _is_valid_zarr_store(zarr_cand) and child.name not in stores:
                stores[child.name] = zarr_cand
            elif _is_valid_zarr_store(child) and child.stem not in stores:
                stores[child.stem] = child
    return stores


def _find_basin_key_in_ds(ds: xr.Dataset, catchment_id: str) -> str | None:
    """Match ``catchment_id`` against ``basin`` coordinate without guessing."""
    if 'basin' not in ds.coords:
        return None
    basins = [str(b) for b in ds.coords['basin'].to_numpy()]
    if catchment_id in basins:
        return catchment_id
    lower_map = {b.lower(): b for b in basins}
    if catchment_id.lower() in lower_map:
        return lower_map[catchment_id.lower()]
    return None


def _extract_latest_issue_from_store(
    store_path: Path, catchment_id: str
) -> pd.Timestamp | None:
    """Return latest valid issue date for ``catchment_id`` in ``store_path``."""
    if not _is_valid_zarr_store(store_path):
        return None
    ds = xr.open_zarr(store_path, consolidated=False)
    bkey = _find_basin_key_in_ds(ds, catchment_id)
    if bkey is None or 'date' not in ds.coords:
        return None
    sub = ds.sel(basin=bkey)
    precip_vars = [
        v
        for v in sub.data_vars
        if 'precipitation' in str(v) or str(v).endswith('_precip')
    ]
    if precip_vars and 'lead_time' in sub.dims:
        da = sub[precip_vars[0]]
        if da.shape[-1] >= _MIN_FORECAST_LEAD_STEPS:
            fc_vals = da.isel(lead_time=slice(1, None)).to_numpy()
            valid_fc_mask = (~np.isnan(fc_vals)).any(axis=-1)
            valid_dates = sub['date'].to_numpy()[valid_fc_mask]
            if len(valid_dates) > 0:
                return (
                    pd.Timestamp(valid_dates[-1]).floor('D').tz_localize(None)
                )
    dates = sub['date'].to_numpy()
    if len(dates) > 0:
        return pd.Timestamp(dates[-1]).floor('D').tz_localize(None)
    return None


def resolve_issue_date(
    username: str | None,
    catchment_id: str,
    requested_issue_date: str | None = None,
    *,
    probe_ecmwf: bool = False,
) -> pd.Timestamp:
    """Determine the forecast issue date ``t0``.

    Raises:
        ValueError: When no valid forecast issue date can be resolved.
    """
    if (
        requested_issue_date
        and str(requested_issue_date).strip().lower() != 'latest'
    ):
        return pd.Timestamp(requested_issue_date).floor('D').tz_localize(None)

    stores = _discover_zarr_stores(username)
    for prod_key in (
        'HRES',
        'ECMWF_AIFS',
        'AIFS',
        'NOAA_GFS',
        'GFS',
        'NOAA_GEFS',
        'GEFS',
        'ECMWF_IFS_ENS',
        'IFS_ENS',
        'GRAPHCAST',
    ):
        if prod_key not in stores:
            continue
        found = _extract_latest_issue_from_store(stores[prod_key], catchment_id)
        if found is not None:
            return found

    if probe_ecmwf:
        ensure_flood_forecasting_on_sys_path()
        extend_multimet_package_path()
        now_utc = pd.Timestamp.now('UTC').floor('D').tz_localize(None)
        latest = find_latest_hres_open_data_date(now_utc)
        if latest is not None:
            return pd.Timestamp(latest).floor('D').tz_localize(None)

    raise ValueError(
        f"No valid forecast issue date found for catchment '{catchment_id}'. "
        'Populate a forecast Zarr store or provide an explicit issue_date.'
    )


def _describe_variable(var_name: str) -> dict[str, Any]:
    """Build UI metadata (label, unit, kind, chart_type) for a Zarr variable."""
    short_name = str(var_name)
    for prefix in _KNOWN_PRODUCT_PREFIXES:
        if short_name.lower().startswith(prefix):
            short_name = short_name[len(prefix) :]
            break

    lower = short_name.lower()
    for keyword, kind, unit, label, chart_type in _VARIABLE_DESCRIPTORS:
        if keyword in lower or (
            keyword == 'temperature' and lower.startswith('temp')
        ):
            return {
                'name': var_name,
                'short_name': short_name,
                'kind': kind,
                'unit': unit,
                'label': label,
                'chart_type': chart_type,
            }
    if 'snow_depth' in lower:
        return {
            'name': var_name,
            'short_name': short_name,
            'kind': 'snow',
            'unit': 'mm',
            'label': f'{short_name.replace("_", " ").title()} (mm)',
            'chart_type': 'line',
        }
    if 'potential_evaporation' in lower:
        return {
            'name': var_name,
            'short_name': short_name,
            'kind': 'pet',
            'unit': 'mm/day',
            'label': f'{short_name.replace("_", " ").title()} (mm/day)',
            'chart_type': 'line',
        }
    return {
        'name': var_name,
        'short_name': short_name,
        'kind': 'other',
        'unit': '',
        'label': short_name.replace('_', ' ').title(),
        'chart_type': 'line',
    }


def _extract_nowcast_product_entry(ctx: ProductSliceContext) -> dict[str, Any]:
    """Extract nowcast series entry from a 2D ``(basin, date)`` Zarr dataset."""
    meta = NOWCAST_PRODUCTS_METADATA.get(
        ctx.prod_name,
        {
            'id': ctx.prod_name,
            'label': ctx.prod_name,
            'var_name': next(
                (v for v in ctx.all_vars if 'precip' in v.lower()),
                ctx.all_vars[0],
            ),
        },
    )
    var_name = _resolve_var_in_ds(
        ctx.sub, str(meta.get('var_name') or '')
    ) or next(
        (v for v in ctx.all_vars if 'precip' in v.lower()),
        ctx.all_vars[0],
    )
    t_var = next(
        (
            v
            for v in ctx.all_vars
            if ('temperature' in v.lower() or v.lower().endswith('_temp'))
            and 'dewpoint' not in v.lower()
            and not v.lower().endswith(('_min', '_max'))
        ),
        None,
    )
    store_dates = pd.to_datetime(ctx.sub['date'].to_numpy()).floor('D')
    date_strs = [_format_date_str(d) for d in store_dates]

    var_maps: dict[str, dict[str, float | None]] = {}
    for v_name in ctx.all_vars:
        vals_1d = ctx.sub[v_name].to_numpy()
        var_maps[v_name] = {
            d_s: _clean_float(val)
            for d_s, val in zip(date_strs, vals_1d, strict=False)
        }

    series_points: list[dict[str, Any]] = []
    latest_valid_date = None
    for d in ctx.nowcast_dates:
        d_str = _format_date_str(d)
        p_val = var_maps.get(var_name, {}).get(d_str)
        t_val = var_maps.get(t_var, {}).get(d_str) if t_var else None
        pt_values = {
            v_name: var_maps[v_name].get(d_str) for v_name in ctx.all_vars
        }
        series_points.append(
            {
                'date': d_str,
                'is_issue_date': False,
                'precip_mm': p_val,
                'temp_c': t_val,
                'temperature_c': t_val,
                'values': pt_values,
            }
        )
        if p_val is not None:
            latest_valid_date = d_str

    missing_recent_days = 0
    for pt in reversed(series_points):
        if pt['precip_mm'] is None:
            missing_recent_days += 1
        else:
            break

    return {
        'id': ctx.prod_name,
        'product': ctx.prod_name,
        'kind': 'nowcast',
        'label': meta.get('label', ctx.prod_name),
        'variable': var_name,
        'precip_var': var_name,
        'temp_var': t_var,
        'has_precip': any(pt['precip_mm'] is not None for pt in series_points),
        'has_temp': any(
            pt['temperature_c'] is not None for pt in series_points
        ),
        'variables': ctx.variables_meta,
        'default_variable_mode': 'precip_and_temp',
        'zarr_path': str(ctx.store_path),
        'issue_date': _format_date_str(ctx.t0),
        'latest_valid_date': latest_valid_date,
        'trailing_latency_gap_days': missing_recent_days,
        'series': series_points,
    }


def _extract_forecast_product_entries(
    ctx: ProductSliceContext,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Extract day-1 hindcast and ``t0`` forecast entries from a Zarr store."""
    meta = FORECAST_PRODUCTS_METADATA.get(
        ctx.prod_name,
        {
            'id': ctx.prod_name,
            'label': ctx.prod_name,
            'precip_var': next(
                (v for v in ctx.all_vars if 'precip' in v.lower()),
                None,
            ),
            'temp_var': next(
                (v for v in ctx.all_vars if 'temperature' in v.lower()),
                None,
            ),
        },
    )
    p_var = _resolve_var_in_ds(
        ctx.sub, str(meta.get('precip_var') or '')
    ) or next(
        (v for v in ctx.all_vars if 'precip' in v.lower()),
        None,
    )
    t_var = _resolve_var_in_ds(
        ctx.sub, str(meta.get('temp_var') or '')
    ) or next(
        (v for v in ctx.all_vars if 'temperature' in v.lower()),
        None,
    )
    if not p_var or p_var not in ctx.sub.data_vars:
        return None, None

    store_dates = pd.to_datetime(ctx.sub['date'].to_numpy()).floor('D')
    store_date_strs = [_format_date_str(d) for d in store_dates]

    hc_var_maps: dict[str, dict[str, float | None]] = {}
    for v_name in ctx.all_vars:
        da_1d_v = ctx.sub[v_name].isel(lead_time=0).to_numpy()
        hc_var_maps[v_name] = {
            d_s: _clean_float(val)
            for d_s, val in zip(store_date_strs, da_1d_v, strict=False)
        }

    hc_points: list[dict[str, Any]] = []
    for d in ctx.nowcast_dates:
        d_str = _format_date_str(d)
        p_val = hc_var_maps.get(p_var, {}).get(d_str)
        t_val = hc_var_maps.get(t_var, {}).get(d_str) if t_var else None
        hc_points.append(
            {
                'date': d_str,
                'lead_time_days': 1,
                'is_issue_date': False,
                'precip_mm': p_val,
                'temp_c': t_val,
                'temperature_c': t_val,
                'values': {
                    v_name: hc_var_maps[v_name].get(d_str)
                    for v_name in ctx.all_vars
                },
            }
        )

    hc_entry: dict[str, Any] | None = None
    if any(
        pt['precip_mm'] is not None or pt['temperature_c'] is not None
        for pt in hc_points
    ):
        hc_key = f'{ctx.prod_name}_1D'
        hc_entry = {
            'id': hc_key,
            'product': hc_key,
            'kind': 'nowcast_hindcast',
            'label': f'{meta.get("label", ctx.prod_name)} (Day-1 Hindcast)',
            'variable': p_var,
            'precip_var': p_var,
            'temp_var': t_var,
            'has_precip': any(pt['precip_mm'] is not None for pt in hc_points),
            'has_temp': any(
                pt['temperature_c'] is not None for pt in hc_points
            ),
            'variables': ctx.variables_meta,
            'default_variable_mode': 'precip_and_temp',
            'zarr_path': str(ctx.store_path),
            'latest_valid_date': next(
                (
                    pt['date']
                    for pt in reversed(hc_points)
                    if pt['precip_mm'] is not None
                ),
                None,
            ),
            'trailing_latency_gap_days': 0,
            'series': hc_points,
        }

    # Require exact forecast issue date match == t0
    if ctx.t0 not in store_dates:
        return hc_entry, None

    fc_slice = ctx.sub.sel(date=ctx.t0)
    lead_coords = fc_slice['lead_time'].to_numpy()
    fc_var_arrays: dict[str, Any] = {
        v_name: fc_slice[v_name].to_numpy() for v_name in ctx.all_vars
    }

    fc_points: list[dict[str, Any]] = []
    for idx, raw_lt in enumerate(lead_coords):
        if isinstance(raw_lt, (np.timedelta64, pd.Timedelta)):
            lt_days = round(
                pd.Timedelta(raw_lt).total_seconds() / _SECONDS_PER_DAY
            )
        else:
            lt_days = int(raw_lt)
        valid_date = ctx.t0 + pd.Timedelta(days=lt_days - 1)
        valid_date_str = _format_date_str(valid_date)
        p_val = (
            _clean_float(fc_var_arrays[p_var][idx])
            if p_var in fc_var_arrays
            else None
        )
        t_val = (
            _clean_float(fc_var_arrays[t_var][idx])
            if (t_var and t_var in fc_var_arrays)
            else None
        )
        pt_values = {
            v_name: _clean_float(fc_var_arrays[v_name][idx])
            for v_name in ctx.all_vars
        }
        fc_points.append(
            {
                'lead_time_days': lt_days,
                'date': valid_date_str,
                'is_issue_date': bool(lt_days == 1),
                'precip_mm': p_val,
                'temp_c': t_val,
                'temperature_c': t_val,
                'values': pt_values,
            }
        )

    fc_entry = {
        'id': ctx.prod_name,
        'product': ctx.prod_name,
        'kind': 'forecast',
        'label': meta.get('label', ctx.prod_name),
        'variable': p_var,
        'precip_var': p_var,
        'temp_var': t_var,
        'has_precip': any(pt['precip_mm'] is not None for pt in fc_points),
        'has_temp': any(pt['temperature_c'] is not None for pt in fc_points),
        'variables': ctx.variables_meta,
        'default_variable_mode': 'precip_and_temp',
        'zarr_path': str(ctx.store_path),
        'issue_date': _format_date_str(ctx.t0),
        'horizon_days': len(fc_points),
        'lead_times_days': [pt['lead_time_days'] for pt in fc_points],
        'series': fc_points,
        'hindcast_series': hc_points,
    }
    return hc_entry, fc_entry


def extract_multi_stream_precipitation_series(
    username: str | None,
    catchment_id: str,
    issue_date: pd.Timestamp,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Read real nowcast and forecast meteorological series from user stores."""
    t0 = pd.Timestamp(issue_date).floor('D').tz_localize(None)
    expected_state_date = t0 - pd.Timedelta(days=1)
    nowcast_start = t0 - pd.Timedelta(days=lookback_days)
    nowcast_dates = pd.date_range(nowcast_start, expected_state_date, freq='D')

    stores = _discover_zarr_stores(username)
    nowcast_streams: dict[str, dict[str, Any]] = {}
    forecast_streams: dict[str, dict[str, Any]] = {}
    input_products: dict[str, dict[str, Any]] = {}

    for prod_name, store_path in stores.items():
        if not _is_valid_zarr_store(store_path):
            continue
        ds = xr.open_zarr(store_path, consolidated=False)
        bkey = _find_basin_key_in_ds(ds, catchment_id)
        if bkey is None:
            continue
        sub = ds.sel(basin=bkey)
        all_vars = [
            str(v)
            for v in sub.data_vars
            if not str(v).endswith('_missing_fraction')
        ]
        if not all_vars:
            continue
        ctx = ProductSliceContext(
            prod_name=prod_name,
            store_path=store_path,
            sub=sub,
            all_vars=all_vars,
            variables_meta=[_describe_variable(v) for v in all_vars],
            nowcast_dates=nowcast_dates,
            t0=t0,
        )
        if 'lead_time' not in sub.dims and 'date' in sub.coords:
            nc_entry = _extract_nowcast_product_entry(ctx)
            nowcast_streams[prod_name] = nc_entry
            input_products[prod_name] = nc_entry
        elif 'lead_time' in sub.dims and 'date' in sub.coords:
            hc_entry, fc_entry = _extract_forecast_product_entries(ctx)
            if hc_entry is not None:
                nowcast_streams[hc_entry['id']] = hc_entry
            if fc_entry is not None:
                forecast_streams[prod_name] = fc_entry
                input_products[prod_name] = fc_entry

    return {
        'issue_date': _format_date_str(t0),
        'expected_state_date': _format_date_str(expected_state_date),
        'nowcast_dates': [_format_date_str(d) for d in nowcast_dates],
        'nowcast_products': nowcast_streams,
        'forecast_products': forecast_streams,
        'input_products': input_products,
    }


def _normalize_catchment_id_list(
    catchment_id: str | None = None,
    catchment_ids: Sequence[str] | None = None,
) -> list[str]:
    """Normalize ``catchment_id`` and ``catchment_ids`` into a unique list."""
    ids: list[str] = []
    if catchment_ids:
        for raw in catchment_ids:
            if raw is None:
                continue
            for part in str(raw).split(','):
                cleaned = part.strip()
                if cleaned and cleaned not in ids:
                    ids.append(cleaned)
    if catchment_id:
        cleaned_single = str(catchment_id).strip()
        if cleaned_single and cleaned_single not in ids:
            ids.insert(0, cleaned_single)
    return ids


def _basin_has_realtime_data(
    stores: dict[str, Path], catchment_id: str
) -> bool:
    """Return True if any valid Zarr store contains ``catchment_id``."""
    for store_path in stores.values():
        if not _is_valid_zarr_store(store_path):
            continue
        ds = xr.open_zarr(store_path, consolidated=False)
        if _find_basin_key_in_ds(ds, catchment_id) is not None:
            return True
    return False


def _is_state_date_valid_for_hotstart(
    saved_state_date: str | None,
    expected_state_ts: pd.Timestamp,
    max_spinup_days: int = _DEFAULT_SPINUP_DAYS,
) -> bool:
    """Return True if ``saved_state_date`` is within the hindcast window."""
    if not saved_state_date:
        return False
    state_ts = pd.Timestamp(saved_state_date).floor('D').tz_localize(None)
    delta_days = int((expected_state_ts - state_ts).days)
    return 0 <= delta_days <= max_spinup_days


def _build_catchments_status_map(
    username: str | None,
    selected_ids: list[str],
    t0: pd.Timestamp,
    model_id: str,
) -> dict[str, dict[str, Any]]:
    """Build per-catchment status map for all known and selected catchments."""
    expected_state_ts = t0 - pd.Timedelta(days=1)
    expected_state_date = _format_date_str(expected_state_ts)
    issue_date_str = _format_date_str(t0)

    pm = profile_manager.get_profile_manager()
    watersheds = pm.load_watersheds(username)
    all_known_ids: list[str] = list(selected_ids)
    for feat in watersheds:
        props = feat.get('properties', {})
        cid = str(props.get('catchment_id') or feat.get('id') or '').strip()
        if cid and cid not in all_known_ids:
            all_known_ids.append(cid)

    stores = _discover_zarr_stores(username)
    catchments_status: dict[str, dict[str, Any]] = {}
    for cid in all_known_ids:
        c_state = inspect_saved_state(
            username=username, catchment_id=cid, model_id=model_id
        )
        c_has_data = _basin_has_realtime_data(stores, cid)
        c_prev = bool(
            c_state.get('exists')
            and c_state.get('state_date') == expected_state_date
        )
        c_valid_hs = bool(
            c_state.get('exists')
            and _is_state_date_valid_for_hotstart(
                c_state.get('state_date'), expected_state_ts
            )
        )
        catchments_status[cid] = {
            'catchment_id': cid,
            'issue_date': issue_date_str,
            'expected_state_date': expected_state_date,
            'saved_state': c_state,
            'is_state_from_previous_day': c_prev,
            'is_state_valid_for_hotstart': c_valid_hs,
            'has_realtime_data': c_has_data,
            'can_fetch_hotstart': c_valid_hs,
            'can_run_hotstart': bool(c_valid_hs and c_has_data),
            'can_run_coldstart': bool(c_has_data),
        }
    return catchments_status


def get_realtime_status(  # noqa: PLR0913
    username: str | None,
    catchment_id: str,
    requested_issue_date: str | None = None,
    model_id: str = _DEFAULT_MODEL_ID,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    *,
    probe_ecmwf: bool = False,
    catchment_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return full status for the Forecasting tab."""
    selected_ids = _normalize_catchment_id_list(catchment_id, catchment_ids)
    primary_id = (
        str(catchment_id).strip()
        if catchment_id
        else (selected_ids[0] if selected_ids else '')
    )

    t0 = resolve_issue_date(
        username=username,
        catchment_id=primary_id,
        requested_issue_date=requested_issue_date,
        probe_ecmwf=probe_ecmwf,
    )
    expected_state_ts = t0 - pd.Timedelta(days=1)
    expected_state_date = _format_date_str(expected_state_ts)
    issue_date_str = _format_date_str(t0)

    state_info = inspect_saved_state(
        username=username, catchment_id=primary_id, model_id=model_id
    )
    is_state_from_previous_day = bool(
        state_info.get('exists')
        and state_info.get('state_date') == expected_state_date
    )

    streams = extract_multi_stream_precipitation_series(
        username=username,
        catchment_id=primary_id,
        issue_date=t0,
        lookback_days=int(lookback_days),
    )
    has_realtime_data = bool(
        streams['nowcast_products'] or streams['forecast_products']
    )

    catchments_status = _build_catchments_status_map(
        username=username,
        selected_ids=selected_ids,
        t0=t0,
        model_id=model_id,
    )
    if primary_id in catchments_status:
        catchments_status[primary_id]['has_realtime_data'] = has_realtime_data
        catchments_status[primary_id]['can_run_coldstart'] = has_realtime_data
        catchments_status[primary_id]['can_run_hotstart'] = bool(
            catchments_status[primary_id]['is_state_valid_for_hotstart']
            and has_realtime_data
        )

    eval_ids = selected_ids or ([primary_id] if primary_id else [])
    missing_or_stale: list[dict[str, Any]] = []
    for cid in eval_ids:
        cs = catchments_status.get(cid, {})
        st = cs.get('saved_state', {})
        if not cs.get('is_state_valid_for_hotstart'):
            st_date = st.get('state_date')
            reason = (
                f'No saved state file (requires state <= {expected_state_date})'
                if not st.get('exists') or not st_date
                else (
                    f'Saved state is from {st_date}, outside valid hindcast '
                    f'window ending {expected_state_date}'
                )
            )
            missing_or_stale.append(
                {
                    'catchment_id': cid,
                    'state_exists': bool(st.get('exists')),
                    'state_date': st_date,
                    'expected_state_date': expected_state_date,
                    'reason': reason,
                }
            )

    all_selected_have_valid_state = bool(
        eval_ids and len(missing_or_stale) == 0
    )
    all_selected_have_previous_day_state = bool(
        eval_ids
        and all(
            catchments_status.get(cid, {}).get(
                'is_state_from_previous_day', False
            )
            for cid in eval_ids
        )
    )
    all_selected_have_realtime_data = bool(
        eval_ids
        and all(
            catchments_status.get(cid, {}).get('has_realtime_data', False)
            for cid in eval_ids
        )
    )
    can_run_hotstart = bool(
        all_selected_have_valid_state and all_selected_have_realtime_data
    )
    can_run_coldstart = bool(all_selected_have_realtime_data)

    if not all_selected_have_valid_state:
        bad_summary = '; '.join(
            f'{item["catchment_id"]} ({item["state_date"] or "no state"})'
            for item in missing_or_stale
        )
        hotstart_msg = (
            'Hot-Start requires a saved state within the hindcast window up to '
            f'{expected_state_date}. Unavailable for: {bad_summary}. '
            'Run Cold-Start first.'
        )
    elif not all_selected_have_realtime_data:
        hotstart_msg = (
            f'Saved state ({expected_state_date}) is valid, but real-time '
            'forcing data has not been fetched for all selected catchments yet.'
        )
    else:
        hotstart_msg = (
            f'Hot-Start ready for {len(eval_ids)} catchment(s) '
            f'(target state date {expected_state_date}).'
        )

    coldstart_msg = (
        f'Cold-Start ready for {len(eval_ids)} selected catchment(s).'
        if can_run_coldstart
        else (
            'Fetch real-time data for the selected catchment(s) before '
            'running the model.'
        )
    )

    return {
        'status': 'success',
        'username': username,
        'basin_id': primary_id,
        'catchment_id': primary_id,
        'selected_catchment_ids': eval_ids,
        'catchments_status': catchments_status,
        'missing_or_stale_state_catchments': missing_or_stale,
        'all_selected_have_previous_day_state': (
            all_selected_have_previous_day_state
        ),
        'all_selected_have_realtime_data': all_selected_have_realtime_data,
        'can_fetch_hotstart': all_selected_have_valid_state,
        'model_id': model_id,
        'issue_date': issue_date_str,
        'expected_state_date': expected_state_date,
        'saved_state': state_info,
        'is_state_from_previous_day': is_state_from_previous_day,
        'can_run_hotstart': can_run_hotstart,
        'can_run_coldstart': can_run_coldstart,
        'has_realtime_data': has_realtime_data,
        'hotstart_status_message': hotstart_msg,
        'coldstart_status_message': coldstart_msg,
        'nowcast_dates': streams['nowcast_dates'],
        'nowcast_products': streams['nowcast_products'],
        'forecast_products': streams['forecast_products'],
        'input_products': streams['input_products'],
    }


def _validate_hotstart_fetch_states(
    username: str | None,
    target_ids: list[str],
    t0_check: pd.Timestamp,
    model_id: str,
    spinup_days: int,
) -> int:
    """Validate Hot-Start states before fetching and return lookback days."""
    expected_state_ts = t0_check - pd.Timedelta(days=1)
    expected_state_date = _format_date_str(expected_state_ts)
    issue_date_str = _format_date_str(t0_check)
    hotstart_lookback = 1

    for cid in target_ids:
        st_info = inspect_saved_state(
            username=username, catchment_id=cid, model_id=model_id
        )
        if not st_info.get('exists'):
            raise ValueError(
                'Cannot fetch in Hot-Start mode: no saved LSTM state file '
                f"exists for catchment '{cid}'. Hot-Start requires a saved "
                f'state up to {expected_state_date}. Please run Cold-Start '
                'first.'
            )
        st_date_str = st_info.get('state_date')
        if not _is_state_date_valid_for_hotstart(
            st_date_str, expected_state_ts, spinup_days
        ):
            raise ValueError(
                'Cannot fetch in Hot-Start mode: saved LSTM state for '
                f"catchment '{cid}' is stamped '{st_date_str}', which is "
                'outside the valid hindcast window ending '
                f"'{expected_state_date}' for issue date '{issue_date_str}'."
            )
        st_ts = pd.Timestamp(st_date_str).floor('D').tz_localize(None)
        delta_days = int((expected_state_ts - st_ts).days)
        hotstart_lookback = max(hotstart_lookback, delta_days + 1)

    return hotstart_lookback


def _resolve_product_enums(
    products: list[str] | None,
) -> list[mm_config.Product] | None:
    """Resolve user-supplied product names into ``mm_config.Product`` enums."""
    if not products:
        return None
    selected_enums: list[mm_config.Product] = []
    alias_map = {
        'ECMWF_AIFS': 'AIFS',
        'NOAA_GFS': 'GFS',
        'NOAA_GEFS': 'GEFS',
        'ECMWF_IFS_ENS': 'IFS_ENS',
    }
    product_members = mm_config.Product.__members__
    for p_str in products:
        key = p_str.upper().strip()
        key = alias_map.get(key, key)
        if key not in product_members:
            raise ValueError(
                f"Unsupported multimet product '{p_str}'. "
                f'Supported: {sorted(product_members.keys())}'
            )
        selected_enums.append(product_members[key])
    return selected_enums


def fetch_realtime_forcing_for_catchment(  # noqa: PLR0913
    username: str | None,
    catchment_id: str | None = None,
    mode: str = 'coldstart',
    reference_date: str | None = None,
    spinup_days: int = _DEFAULT_SPINUP_DAYS,
    *,
    products: list[str] | None = None,
    catchment_ids: Sequence[str] | None = None,
    model_id: str = _DEFAULT_MODEL_ID,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch real-time multimet forcings for the selected catchment(s)."""
    norm_mode = mode.lower().strip().replace('-', '').replace('_', '')
    if norm_mode not in ('coldstart', 'hotstart'):
        raise ValueError(
            f"Invalid mode {mode!r}; expected 'coldstart' or 'hotstart'."
        )

    target_ids = _normalize_catchment_id_list(catchment_id, catchment_ids)
    if not target_ids:
        raise ValueError(
            'At least one catchment_id must be selected to fetch real-time '
            'data.'
        )
    primary_id = str(catchment_id).strip() if catchment_id else target_ids[0]

    pm = profile_manager.get_profile_manager()
    watersheds = pm.load_watersheds(username)
    ws_by_id: dict[str, dict[str, Any]] = {}
    for feat in watersheds:
        props = feat.get('properties', {})
        cid = str(props.get('catchment_id') or feat.get('id') or '').strip()
        if cid:
            ws_by_id[cid] = feat

    missing_cids = [cid for cid in target_ids if cid not in ws_by_id]
    if missing_cids:
        raise ValueError(
            f'Catchment(s) {missing_cids!r} not found in active profile '
            'watersheds.'
        )
    target_feats = [ws_by_id[cid] for cid in target_ids]

    norm_ref_date = (
        None
        if (
            not reference_date
            or str(reference_date).strip().lower() == 'latest'
        )
        else str(reference_date).strip()
    )

    if norm_mode == 'hotstart':
        t0_check = resolve_issue_date(
            username=username,
            catchment_id=primary_id,
            requested_issue_date=norm_ref_date,
            probe_ecmwf=False,
        )
        hotstart_lookback = _validate_hotstart_fetch_states(
            username=username,
            target_ids=target_ids,
            t0_check=t0_check,
            model_id=model_id,
            spinup_days=int(spinup_days),
        )
    else:
        hotstart_lookback = 1

    shapes_dir = pm.get_catchment_shapes_dir(username)
    shapes_dir.mkdir(parents=True, exist_ok=True)
    for cid, feat in zip(target_ids, target_feats, strict=True):
        single_path = shapes_dir / f'{cid}.geojson'
        single_path.write_text(
            json.dumps({'type': 'FeatureCollection', 'features': [feat]}),
            encoding='utf-8',
        )

    if len(target_ids) == 1:
        catchment_geojson_path = shapes_dir / f'{target_ids[0]}.geojson'
    else:
        catchment_geojson_path = (
            shapes_dir / 'selected_catchments_batch.geojson'
        )
        catchment_geojson_path.write_text(
            json.dumps({'type': 'FeatureCollection', 'features': target_feats}),
            encoding='utf-8',
        )

    output_dir = pm.get_realtime_dynamics_dir(username)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = mm_realtime.fetch_realtime_multimet(
        basins=catchment_geojson_path,
        output_dir=output_dir,
        mode=norm_mode,
        lookback_days=(
            int(spinup_days) if norm_mode == 'coldstart' else hotstart_lookback
        ),
        reference_date=norm_ref_date,
        id_column='catchment_id',
        products=_resolve_product_enums(products),
        overwrite=bool(overwrite),
    )

    ref_ts = pd.Timestamp(summary.reference_date).floor('D').tz_localize(None)
    status_payload = get_realtime_status(
        username=username,
        catchment_id=primary_id,
        requested_issue_date=_format_date_str(ref_ts),
        model_id=model_id,
        catchment_ids=target_ids,
    )
    status_payload['fetch_summary'] = {
        'mode': summary.mode,
        'reference_date': _format_date_str(ref_ts),
        'catchment_ids': target_ids,
        'basin_count': len(target_ids),
        'elapsed_seconds': round(float(summary.elapsed_seconds), 2),
        'products_written': {str(k): str(v) for k, v in summary.stores.items()},
        'windows': {
            str(k): [_format_date_str(w[0]), _format_date_str(w[1])]
            for k, w in summary.product_windows.items()
        },
    }
    return status_payload


def _resolve_model_run_dir() -> Path:
    """Resolve the path to the 5-basin-example pretrained model run dir."""
    if DEFAULT_MODEL_RUN_DIR.exists():
        return DEFAULT_MODEL_RUN_DIR
    legacy_dir = (
        FLOOD_FORECASTING_REPO_DIR
        / 'tutorial'
        / 'model-runs'
        / '5-basin-example'
    )
    if legacy_dir.exists():
        return legacy_dir
    return DEFAULT_MODEL_RUN_DIR


def _resolve_tutorial_caravan_dir() -> Path:
    """Resolve the path to the tutorial Caravan attributes directory."""
    if DEFAULT_TUTORIAL_CARAVAN_DIR.exists():
        return DEFAULT_TUTORIAL_CARAVAN_DIR
    zarr_dir = (
        FLOOD_FORECASTING_REPO_DIR / 'model' / 'tutorial' / 'Caravan-zarr'
    )
    if zarr_dir.exists():
        return zarr_dir
    legacy_dir = FLOOD_FORECASTING_REPO_DIR / 'tutorial' / 'Caravan-nc'
    if legacy_dir.exists():
        return legacy_dir
    return DEFAULT_TUTORIAL_CARAVAN_DIR


def _load_forecast_model_bundle(
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Load the trained ``MeanEmbeddingForecastLSTM`` and ``Scaler`` on CPU."""
    target_dir = (
        Path(run_dir) if run_dir is not None else _resolve_model_run_dir()
    )
    cache_key = str(target_dir.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    ensure_flood_forecasting_on_sys_path()
    cfg = Config(target_dir / 'config.yml')
    cfg.update_config(
        {
            'device': 'cpu',
            'run_dir': target_dir,
            'data_dir': target_dir,
        }
    )

    model = get_model(cfg).to(torch.device('cpu'))
    ckpts = sorted(target_dir.glob('model_epoch*.pt'))
    if not ckpts:
        raise FileNotFoundError(
            f'No model_epoch*.pt checkpoint found in {target_dir}'
        )
    state_dict = torch.load(
        str(ckpts[-1]),
        map_location=torch.device('cpu'),
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    scaler = Scaler(scaler_dir=target_dir, calculate_scaler=False)

    bundle = {
        'cfg': cfg,
        'model': model,
        'scaler': scaler,
        'head': model.head,
        'checkpoint_path': str(ckpts[-1]),
        'flatten_feature_list': flatten_feature_list,
        'product_name_from_feature': product_name_from_feature,
        'load_caravan_attributes': load_caravan_attributes,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _to_attr_dataframe(res: object) -> pd.DataFrame | None:
    """Convert ``load_caravan_attributes`` output to a ``pandas.DataFrame``."""
    if res is None:
        return None
    if isinstance(res, pd.DataFrame):
        return res
    if isinstance(res, xr.Dataset):
        return res.to_dataframe()
    return None


def _read_profile_csv_attributes(
    csv_dirs: Sequence[Path],
    catchment_id: str,
) -> pd.Series | None:
    """Search flat CSV files in ``csv_dirs`` for ``catchment_id`` attributes."""
    merged: dict[str, Any] = {}
    for csv_dir in csv_dirs:
        if not csv_dir.is_dir():
            continue
        for csv_path in sorted(csv_dir.glob('*.csv')):
            if not csv_path.is_file() or csv_path.stat().st_size == 0:
                continue
            df_csv = pd.read_csv(csv_path)
            id_col = next(
                (
                    c
                    for c in ('gauge_id', 'basin', 'catchment_id', 'id')
                    if c in df_csv.columns
                ),
                None,
            )
            if id_col is None:
                continue
            matches = df_csv[df_csv[id_col].astype(str) == catchment_id]
            if not matches.empty:
                row = matches.iloc[-1]
                for col in df_csv.columns:
                    if col != id_col and pd.notna(row[col]):
                        merged[col] = row[col]
    return pd.Series(merged) if merged else None


def _lookup_row_in_df(
    df: pd.DataFrame | None, catchment_id: str
) -> pd.Series | None:
    """Extract a single ``pd.Series`` row for ``catchment_id`` from ``df``."""
    if df is None or catchment_id not in df.index:
        return None
    loc_res = df.loc[catchment_id]
    return loc_res.iloc[-1] if isinstance(loc_res, pd.DataFrame) else loc_res


def _resolve_basin_area_km2(
    username: str | None,
    catchment_id: str,
    row: pd.Series,
) -> float:
    """Resolve positive finite drainage area (``km^2``) for ``catchment_id``."""
    pm = profile_manager.get_profile_manager()
    area_km2: float | None = None
    for feat in pm.load_watersheds(username):
        props = feat.get('properties', {})
        cid = props.get('catchment_id') or feat.get('id')
        if cid == catchment_id:
            raw_area = props.get('area_km2') or props.get('area')
            if raw_area is not None:
                area_km2 = float(raw_area)
            break

    if area_km2 is None:
        for area_col in ('area_km2', 'area'):
            if area_col in row.index and pd.notna(row[area_col]):
                area_km2 = float(row[area_col])
                break

    if area_km2 is None or not math.isfinite(area_km2) or area_km2 <= 0.0:
        raise ValueError(
            "Valid positive drainage area ('area_km2' or 'area') not found "
            f"for catchment '{catchment_id}'."
        )
    return area_km2


def _load_static_attributes_for_basin(
    username: str | None,
    catchment_id: str,
    attr_features: list[str],
    load_caravan_attributes_fn: Callable[..., Any],
) -> tuple[np.ndarray, float]:
    """Load static Caravan attributes for ``catchment_id``.

    Raises:
        KeyError: If the catchment or any required static attribute is missing.
        ValueError: If a positive finite ``area_km2`` cannot be found.
    """
    pm = profile_manager.get_profile_manager()
    row: pd.Series | None = None

    statics_zarr = pm.get_statics_zarr_path(username)
    if _is_valid_zarr_store(statics_zarr):
        row = _lookup_row_in_df(
            _to_attr_dataframe(
                load_caravan_attributes_fn(
                    data_dir=pm.get_statics_dir(username)
                )
            ),
            catchment_id,
        )

    if row is None:
        row = _read_profile_csv_attributes(
            [
                pm.get_caravan_attributes_dir(username),
                pm.get_attributes_dir(username),
            ],
            catchment_id,
        )

    tutorial_caravan_dir = _resolve_tutorial_caravan_dir()
    if row is None and tutorial_caravan_dir.exists():
        has_tut_zarr = _is_valid_zarr_store(
            tutorial_caravan_dir / 'attributes.zarr'
        )
        has_tut_dirs = (tutorial_caravan_dir / 'attributes').is_dir() and any(
            d.is_dir() for d in (tutorial_caravan_dir / 'attributes').iterdir()
        )
        if has_tut_zarr or has_tut_dirs:
            row = _lookup_row_in_df(
                _to_attr_dataframe(
                    load_caravan_attributes_fn(data_dir=tutorial_caravan_dir)
                ),
                catchment_id,
            )

    if row is None:
        raise KeyError(
            f"Static attributes for catchment '{catchment_id}' were not found "
            'in the active user profile or Caravan attribute stores.'
        )

    missing_cols = [f for f in attr_features if f not in row.index]
    if missing_cols:
        raise KeyError(
            f"Catchment '{catchment_id}' is missing required static "
            f'attributes: {missing_cols}'
        )

    area_km2 = _resolve_basin_area_km2(username, catchment_id, row)
    vals = np.array(
        [float(row[f]) for f in attr_features],
        dtype=np.float32,
    )
    return vals, area_km2


def _resolve_var_in_ds(ds: xr.Dataset | None, feat: str) -> str | None:
    """Find matching variable name in ``ds`` without cross-model fallback."""
    if ds is None:
        return None
    candidates = [
        feat,
        feat.replace('_precipitation', '_precip'),
        feat.replace('_precip', '_precipitation'),
    ]
    for cand in candidates:
        if cand in ds.data_vars:
            return cand
    return None


def _scale_feature_array(
    scaler: Scaler, feat_name: str, arr: np.ndarray
) -> np.ndarray:
    """Scale ``arr`` using ``scaler.scaler[feat_name]``."""
    scaler_ds = scaler.scaler
    var_key = _resolve_var_in_ds(scaler_ds, feat_name)
    if var_key is None:
        raise KeyError(f"Feature '{feat_name}' is missing from scaler dataset.")
    center = float(scaler_ds[var_key].sel(parameter='center').to_numpy())
    scale = float(scaler_ds[var_key].sel(parameter='scale').to_numpy())
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Invalid scaler parameters for feature '{feat_name}': "
            f'center={center}, scale={scale}.'
        )
    return ((arr - center) / scale).astype(np.float32)


def _unpack_override_payloads(
    precip_overrides: dict[str, Any] | None,
    temp_overrides: dict[str, float] | None = None,
    product_overrides: dict[str, dict[str, dict[str, float]]] | None = None,
    *,
    is_what_if: bool = False,
) -> OverrideConfig:
    """Normalize precip, temp, and per-product override dictionaries."""
    clean_precip: dict[str, float] = {}
    clean_temp: dict[str, float] = dict(temp_overrides or {})
    clean_prod: dict[str, dict[str, dict[str, float]]] = dict(
        product_overrides or {}
    )
    what_if_flag = bool(is_what_if)

    if isinstance(precip_overrides, dict):
        if '__temp_overrides__' in precip_overrides and not clean_temp:
            raw_t = precip_overrides.get('__temp_overrides__')
            if isinstance(raw_t, dict):
                clean_temp = {
                    str(k): float(v) for k, v in raw_t.items() if v is not None
                }
        if '__product_overrides__' in precip_overrides and not clean_prod:
            raw_p = precip_overrides.get('__product_overrides__')
            if isinstance(raw_p, dict):
                clean_prod = raw_p
        if '__is_what_if__' in precip_overrides:
            what_if_flag = what_if_flag or bool(
                precip_overrides.get('__is_what_if__')
            )
        for k, v in precip_overrides.items():
            if (
                not str(k).startswith('__')
                and isinstance(v, (int, float, np.number))
                and not isinstance(v, bool)
                and math.isfinite(float(v))
            ):
                clean_precip[str(k)] = float(v)

    if clean_precip or clean_temp or clean_prod:
        what_if_flag = True

    return OverrideConfig(
        precip_overrides=clean_precip,
        temp_overrides=clean_temp,
        product_overrides=clean_prod,
        is_what_if=what_if_flag,
    )


def _lookup_feature_override(
    prod: str,
    var_kind: str | None,
    d_str: str,
    overrides: OverrideConfig,
) -> float | None:
    """Resolve an override value for ``(prod, var_kind, d_str)``."""
    if var_kind not in ('precip', 'temp'):
        return None

    prod_lower = prod.lower()
    alias_table: dict[str, tuple[str, ...]] = {
        'hres': ('HRES', 'hres'),
        'graphcast': ('GRAPHCAST', 'graphcast'),
        'aifs': ('ECMWF_AIFS', 'AIFS', 'ecmwf_aifs', 'aifs'),
        'ecmwf_aifs': ('ECMWF_AIFS', 'AIFS', 'ecmwf_aifs', 'aifs'),
        'imerg': ('IMERG', 'imerg'),
        'cpc': ('CPC', 'cpc'),
    }
    aliases = alias_table.get(prod_lower, (prod.upper(), prod_lower))

    for alias in aliases:
        p_dict = overrides.product_overrides.get(alias)
        if isinstance(p_dict, dict):
            v_dict = p_dict.get(var_kind)
            if (
                isinstance(v_dict, dict)
                and d_str in v_dict
                and v_dict[d_str] is not None
            ):
                val = float(v_dict[d_str])
                return max(0.0, val) if var_kind == 'precip' else val

    fallback_map = (
        overrides.precip_overrides
        if var_kind == 'precip'
        else overrides.temp_overrides
    )
    if d_str in fallback_map and fallback_map[d_str] is not None:
        val = float(fallback_map[d_str])
        return max(0.0, val) if var_kind == 'precip' else val

    return None


def _apply_date_overrides(
    arr: np.ndarray,
    dates: pd.DatetimeIndex,
    prod: str,
    var_kind: str | None,
    overrides: OverrideConfig,
) -> None:
    """In-place apply any matching date overrides to ``arr``."""
    if var_kind is None or not overrides.has_any:
        return
    for idx, d in enumerate(dates):
        ov = _lookup_feature_override(
            prod, var_kind, _format_date_str(d), overrides
        )
        if ov is not None:
            arr[idx] = ov


def _build_hindcast_feature_tensor(
    feat: str,
    prod: str,
    ds: xr.Dataset | None,
    tctx: TensorBuildContext,
) -> torch.Tensor:
    """Assemble and scale a single hindcast feature tensor ``[1, T, 1]``."""
    seq_len = len(tctx.hindcast_dates)
    lead_time = len(tctx.forecast_dates)
    var_name = _resolve_var_in_ds(ds, feat)
    hc_arr = np.full(seq_len, np.nan, dtype=np.float32)
    fc_pad = np.full(lead_time, np.nan, dtype=np.float32)
    var_kind = (
        'precip' if 'precip' in feat else ('temp' if 'temp' in feat else None)
    )

    if ds is not None and var_name is not None:
        bkey = _find_basin_key_in_ds(ds, tctx.catchment_id)
        if bkey is not None:
            sub = ds.sel(basin=bkey)
            s_dates = pd.to_datetime(sub['date'].to_numpy()).floor('D')
            da = sub[var_name]
            if 'lead_time' in da.dims:
                da = da.isel(lead_time=0)
            v_map = dict(zip(s_dates, da.to_numpy(), strict=False))
            for i, d in enumerate(tctx.hindcast_dates):
                if d in v_map:
                    hc_arr[i] = float(v_map[d])

    _apply_date_overrides(
        hc_arr, tctx.hindcast_dates, prod, var_kind, tctx.overrides
    )
    full_arr = _scale_feature_array(
        tctx.scaler, feat, np.concatenate([hc_arr, fc_pad])
    )
    return torch.from_numpy(full_arr.reshape(1, seq_len + lead_time, 1)).to(
        torch.float32
    )


def _build_forecast_feature_tensor(
    feat: str,
    prod: str,
    ds: xr.Dataset | None,
    tctx: TensorBuildContext,
) -> torch.Tensor:
    """Assemble and scale a single forecast feature tensor ``[1, T, 1]``."""
    seq_len = len(tctx.hindcast_dates)
    lead_time = len(tctx.forecast_dates)
    var_name = _resolve_var_in_ds(ds, feat)
    hc_arr = np.full(seq_len, np.nan, dtype=np.float32)
    fc_arr = np.full(lead_time, np.nan, dtype=np.float32)
    var_kind = (
        'precip' if 'precip' in feat else ('temp' if 'temp' in feat else None)
    )

    if ds is not None and var_name is not None:
        bkey = _find_basin_key_in_ds(ds, tctx.catchment_id)
        if bkey is not None:
            sub = ds.sel(basin=bkey)
            s_dates = pd.to_datetime(sub['date'].to_numpy()).floor('D')
            da_hc = sub[var_name].isel(lead_time=0)
            hc_map = dict(zip(s_dates, da_hc.to_numpy(), strict=False))
            for i, d in enumerate(tctx.hindcast_dates):
                if d in hc_map:
                    hc_arr[i] = float(hc_map[d])
            if tctx.t0 in s_dates:
                fc_vals = sub[var_name].sel(date=tctx.t0).to_numpy()
                n_copy = min(lead_time, len(fc_vals))
                fc_arr[:n_copy] = fc_vals[:n_copy].astype(np.float32)

    _apply_date_overrides(
        hc_arr, tctx.hindcast_dates, prod, var_kind, tctx.overrides
    )
    _apply_date_overrides(
        fc_arr, tctx.forecast_dates, prod, var_kind, tctx.overrides
    )
    full_arr = _scale_feature_array(
        tctx.scaler, feat, np.concatenate([hc_arr, fc_arr])
    )
    return torch.from_numpy(full_arr.reshape(1, seq_len + lead_time, 1)).to(
        torch.float32
    )


def _assemble_inference_tensors(
    username: str | None,
    catchment_id: str,
    issue_date: pd.Timestamp,
    bundle: dict[str, Any],
    overrides: OverrideConfig,
) -> tuple[dict[str, Any], pd.DatetimeIndex, pd.DatetimeIndex, float]:
    """Build scaled PyTorch inputs for ``MeanEmbeddingForecastLSTM``."""
    cfg = bundle['cfg']
    scaler: Scaler = bundle['scaler']
    flatten_fn = bundle['flatten_feature_list']
    prod_from_feat = bundle['product_name_from_feature']

    seq_len = int(cfg.seq_length or _DEFAULT_SPINUP_DAYS)
    lead_time = int(cfg.lead_time or _DEFAULT_LEAD_TIME_DAYS)

    t0 = pd.Timestamp(issue_date).floor('D').tz_localize(None)
    hindcast_dates = pd.date_range(
        t0 - pd.Timedelta(days=seq_len), t0 - pd.Timedelta(days=1), freq='D'
    )
    forecast_dates = pd.date_range(
        t0, t0 + pd.Timedelta(days=lead_time - 1), freq='D'
    )

    fc_features = flatten_fn(cfg.forecast_inputs)
    hc_features = flatten_fn(cfg.hindcast_inputs)
    attr_features = flatten_fn(cfg.static_attributes)

    stores = _discover_zarr_stores(username)
    open_datasets: dict[str, xr.Dataset] = {}
    for p_name, s_path in stores.items():
        if not _is_valid_zarr_store(s_path):
            continue
        ds_obj = xr.open_zarr(s_path, consolidated=False)
        open_datasets[p_name.lower()] = ds_obj
        open_datasets[p_name.upper()] = ds_obj

    tctx = TensorBuildContext(
        catchment_id=catchment_id,
        hindcast_dates=hindcast_dates,
        forecast_dates=forecast_dates,
        t0=t0,
        scaler=scaler,
        overrides=overrides,
    )

    model_x_d_hindcast: dict[str, Any] = {}
    for feat in hc_features:
        prod = str(prod_from_feat(feat)).lower()
        model_x_d_hindcast[feat] = _build_hindcast_feature_tensor(
            feat=feat,
            prod=prod,
            ds=open_datasets.get(prod),
            tctx=tctx,
        )

    model_x_d_forecast: dict[str, Any] = {}
    for feat in fc_features:
        prod = str(prod_from_feat(feat)).lower()
        model_x_d_forecast[feat] = _build_forecast_feature_tensor(
            feat=feat,
            prod=prod,
            ds=open_datasets.get(prod),
            tctx=tctx,
        )

    attr_vals, area_km2 = _load_static_attributes_for_basin(
        username,
        catchment_id,
        attr_features,
        bundle['load_caravan_attributes'],
    )
    scaled_attrs = np.full_like(attr_vals, np.nan, dtype=np.float32)
    for i, a_name in enumerate(attr_features):
        val = attr_vals[i]
        if not np.isnan(val):
            scaled_attrs[i] = float(
                _scale_feature_array(
                    scaler, a_name, np.array([val], dtype=np.float32)
                )[0]
            )
    model_x_s = torch.from_numpy(scaled_attrs.reshape(1, -1)).to(torch.float32)

    model_input = {
        'x_d_hindcast': model_x_d_hindcast,
        'x_d_forecast': model_x_d_forecast,
        'x_s': model_x_s,
    }
    return model_input, hindcast_dates, forecast_dates, area_km2


def _to_3d_state_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a saved numpy LSTM state array to a 3D float32 tensor."""
    t = torch.from_numpy(arr).to(dtype=torch.float32)
    while t.ndim < _NDIM_3D:
        t = t.unsqueeze(0)
    return t


def _execute_hotstart_forward_and_advance(
    model: torch.nn.Module,
    model_input: dict[str, Any],
    state_path: Path,
    expected_state_ts: pd.Timestamp,
    *,
    is_what_if: bool,
) -> dict[str, torch.Tensor]:
    """Run Hot-Start inference and advance state if ``state_date < t0 - 1``."""
    full_seq_len = int(model.cfg.seq_length or _DEFAULT_SPINUP_DAYS)
    lead_time = int(model.cfg.lead_time or _DEFAULT_LEAD_TIME_DAYS)

    with np.load(state_path, allow_pickle=False) as st_npz:
        files = set(st_npz.files)
        state_date_str = str(st_npz['date'].item()) if 'date' in files else ''
        h_hind_arr = (
            st_npz['h_hindcast'] if 'h_hindcast' in files else st_npz['h_n']
        )
        c_hind_arr = (
            st_npz['c_hindcast'] if 'c_hindcast' in files else st_npz['c_n']
        )
        h_fore_arr = (
            st_npz['h_forecast'] if 'h_forecast' in files else h_hind_arr
        )
        c_fore_arr = (
            st_npz['c_forecast'] if 'c_forecast' in files else c_hind_arr
        )

    state_ts = pd.Timestamp(state_date_str).floor('D').tz_localize(None)
    delta_days = int((expected_state_ts - state_ts).days)
    if delta_days < 0 or delta_days > full_seq_len:
        raise ValueError(
            f"Saved state date '{state_date_str}' cannot be used for target "
            f"state date '{_format_date_str(expected_state_ts)}' "
            f'(delta_days={delta_days}, seq_length={full_seq_len}).'
        )

    model.load_state_from_disk(state_path)
    preloaded = model.__dict__.get('_preloaded_state')
    if isinstance(preloaded, dict):
        preloaded.setdefault('h_hindcast', h_hind_arr)
        preloaded.setdefault('c_hindcast', c_hind_arr)
        preloaded.setdefault('h_forecast', h_fore_arr)
        preloaded.setdefault('c_forecast', c_fore_arr)

    model.seq_length = delta_days
    tail_steps = delta_days + lead_time
    hot_input = {
        'x_d_hindcast': {
            k: v[:, -tail_steps:, :]
            for k, v in model_input['x_d_hindcast'].items()
        },
        'x_d_forecast': {
            k: v[:, -tail_steps:, :]
            for k, v in model_input['x_d_forecast'].items()
        },
        'x_s': model_input['x_s'],
    }
    need_advance = delta_days > 0 and not is_what_if
    pred_dict = model(hot_input, return_embeddings=need_advance)

    if need_advance:
        h_hind_init = (
            _to_3d_state_tensor(h_hind_arr),
            _to_3d_state_tensor(c_hind_arr),
        )
        h_fore_init = (
            _to_3d_state_tensor(h_fore_arr),
            _to_3d_state_tensor(c_fore_arr),
        )
        static_rep = (
            pred_dict['static_embedding'].unsqueeze(1).repeat(1, delta_days, 1)
        )
        hind_in = torch.cat(
            [
                pred_dict['hindcast_embedding'][:, :delta_days, :],
                static_rep,
            ],
            dim=-1,
        ).nan_to_num(nan=0.0)
        hind_step_out, (h_hind_new, c_hind_new) = model.hindcast_lstm(
            input=hind_in, hx=h_hind_init
        )
        fore_in = torch.cat(
            [
                pred_dict['forecast_embedding'][:, :delta_days, :],
                hind_step_out,
                static_rep,
            ],
            dim=-1,
        ).nan_to_num(nan=0.0)
        _, (h_fore_new, c_fore_new) = model.forecast_lstm(
            input=fore_in, hx=h_fore_init
        )
        expected_state_date = _format_date_str(expected_state_ts)
        np.savez_compressed(
            state_path,
            h_hindcast=h_hind_new.detach().cpu().numpy(),
            c_hindcast=c_hind_new.detach().cpu().numpy(),
            h_forecast=h_fore_new.detach().cpu().numpy(),
            c_forecast=c_fore_new.detach().cpu().numpy(),
            h_n=h_hind_new.detach().cpu().numpy(),
            c_n=c_hind_new.detach().cpu().numpy(),
            date=np.array(expected_state_date, dtype='U10'),
        )

    model.seq_length = full_seq_len
    model.__dict__.pop('_preloaded_state', None)
    return pred_dict


def _unscale_predictions(
    pred_dict: dict[str, torch.Tensor],
    bundle: dict[str, Any],
    area_km2: float,
    lead_time: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Unscale model predictions to ``mm/day`` and ``m^3/s`` (CMS)."""
    model = bundle['model']
    scaler: Scaler = bundle['scaler']
    target_var = bundle['cfg'].target_variables[0]

    y_hat_scaled = model.point_prediction(pred_dict)
    scaler_ds = scaler.scaler
    if target_var not in scaler_ds.data_vars:
        raise KeyError(
            f"Target variable '{target_var}' is missing from scaler dataset."
        )
    y_center = float(scaler_ds[target_var].sel(parameter='center').to_numpy())
    y_scale = float(scaler_ds[target_var].sel(parameter='scale').to_numpy())
    if (
        not math.isfinite(y_center)
        or not math.isfinite(y_scale)
        or y_scale <= 0.0
    ):
        raise ValueError(
            f"Invalid scaler parameters for target '{target_var}': "
            f'center={y_center}, scale={y_scale}.'
        )
    unscaled_y = y_hat_scaled * y_scale + y_center

    raw_mm = unscaled_y.squeeze(0).squeeze(-1).cpu().numpy()
    mm_per_day = np.where(np.isnan(raw_mm), np.nan, np.maximum(0.0, raw_mm))
    cms_factor = float(area_km2) * _CMS_PER_MM_DAY_KM2
    q_cms = mm_per_day * cms_factor

    prob_quantiles_cms: dict[str, np.ndarray] = {}
    for q_key in ('q05', 'q25', 'q75', 'q95'):
        if q_key in pred_dict and isinstance(pred_dict[q_key], torch.Tensor):
            q_raw = (
                (pred_dict[q_key] * y_scale + y_center)
                .squeeze(0)
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            q_mm = np.where(np.isnan(q_raw), np.nan, np.maximum(0.0, q_raw))
            prob_quantiles_cms[q_key] = (q_mm * cms_factor)[-lead_time:]

    return mm_per_day, q_cms, prob_quantiles_cms


def _format_forecast_and_hindcast_series(
    mm_per_day: np.ndarray,
    q_cms: np.ndarray,
    prob_quantiles_cms: dict[str, np.ndarray],
    hindcast_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], float | None, str | None
]:
    """Format unscaled arrays into forecast and hindcast point lists."""
    lead_time = len(forecast_dates)
    fc_mm = mm_per_day[-lead_time:]
    fc_cms = q_cms[-lead_time:]

    forecast_series: list[dict[str, Any]] = []
    peak_q: float | None = None
    peak_date: str | None = None
    for idx, d in enumerate(forecast_dates):
        q_val = _clean_float(fc_cms[idx])
        mm_val = _clean_float(fc_mm[idx])
        d_str = _format_date_str(d)
        if q_val is not None and (peak_q is None or q_val >= peak_q):
            peak_q = q_val
            peak_date = d_str

        def _get_quantile(q_name: str, step_idx: int) -> float | None:
            if q_name not in prob_quantiles_cms:
                return None
            val = _clean_float(prob_quantiles_cms[q_name][step_idx])
            return round(val, 3) if val is not None else None

        forecast_series.append(
            {
                'lead_time_days': idx + 1,
                'date': d_str,
                'discharge_cms': round(q_val, 3) if q_val is not None else None,
                'discharge_mm_day': (
                    round(mm_val, 4) if mm_val is not None else None
                ),
                'q05_cms': _get_quantile('q05', idx),
                'q25_cms': _get_quantile('q25', idx),
                'q75_cms': _get_quantile('q75', idx),
                'q95_cms': _get_quantile('q95', idx),
            }
        )

    hindcast_series: list[dict[str, Any]] = []
    if len(q_cms) > lead_time:
        hc_cms = q_cms[:-lead_time]
        hc_mm = mm_per_day[:-lead_time]
        tail_len = min(_HINDCAST_TAIL_DAYS, len(hc_cms))
        for d, q_v, m_v in zip(
            hindcast_dates[-tail_len:],
            hc_cms[-tail_len:],
            hc_mm[-tail_len:],
            strict=False,
        ):
            q_clean = _clean_float(q_v)
            m_clean = _clean_float(m_v)
            hindcast_series.append(
                {
                    'date': _format_date_str(d),
                    'discharge_cms': (
                        round(q_clean, 3) if q_clean is not None else None
                    ),
                    'discharge_mm_day': (
                        round(m_clean, 4) if m_clean is not None else None
                    ),
                }
            )
    return forecast_series, hindcast_series, peak_q, peak_date


def _run_forecast_model_single(
    username: str | None,
    catchment_id: str,
    t0: pd.Timestamp,
    bundle: dict[str, Any],
    overrides: OverrideConfig,
) -> dict[str, Any]:
    """Run ``MeanEmbeddingForecastLSTM`` for a single catchment."""
    norm_mode = bundle['active_mode']
    model_id = bundle['active_model_id']
    issue_date_str = _format_date_str(t0)
    expected_state_ts = t0 - pd.Timedelta(days=1)
    expected_state_date = _format_date_str(expected_state_ts)
    state_path = get_state_file_path(
        username=username, catchment_id=catchment_id, model_id=model_id
    )

    cfg = bundle['cfg']
    model = bundle['model']
    lead_time = int(cfg.lead_time or _DEFAULT_LEAD_TIME_DAYS)
    full_seq_len = int(cfg.seq_length or _DEFAULT_SPINUP_DAYS)

    model_input, hindcast_dates, forecast_dates, area_km2 = (
        _assemble_inference_tensors(
            username=username,
            catchment_id=catchment_id,
            issue_date=t0,
            bundle=bundle,
            overrides=overrides,
        )
    )

    with torch.no_grad():
        if norm_mode == 'coldstart':
            model.seq_length = full_seq_len
            model.__dict__.pop('_preloaded_state', None)
            if not overrides.is_what_if:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_state(model_input, state_path)
                with np.load(state_path, allow_pickle=False) as raw_saved:
                    saved_npz = {k: raw_saved[k] for k in raw_saved.files}
                saved_npz['date'] = np.array(expected_state_date, dtype='U10')
                if 'h_hindcast' in saved_npz and 'h_n' not in saved_npz:
                    saved_npz['h_n'] = saved_npz['h_hindcast']
                if 'c_hindcast' in saved_npz and 'c_n' not in saved_npz:
                    saved_npz['c_n'] = saved_npz['c_hindcast']
                np.savez_compressed(state_path, **saved_npz)

            pred_dict = model(model_input)
        else:
            pred_dict = _execute_hotstart_forward_and_advance(
                model=model,
                model_input=model_input,
                state_path=state_path,
                expected_state_ts=expected_state_ts,
                is_what_if=overrides.is_what_if,
            )

        mm_per_day, q_cms, prob_quantiles_cms = _unscale_predictions(
            pred_dict=pred_dict,
            bundle=bundle,
            area_km2=area_km2,
            lead_time=lead_time,
        )

    forecast_series, hindcast_series, peak_q, peak_date = (
        _format_forecast_and_hindcast_series(
            mm_per_day=mm_per_day,
            q_cms=q_cms,
            prob_quantiles_cms=prob_quantiles_cms,
            hindcast_dates=hindcast_dates,
            forecast_dates=forecast_dates,
        )
    )
    saved_state = inspect_saved_state(
        username=username, catchment_id=catchment_id, model_id=model_id
    )

    return {
        'status': 'success',
        'mode': norm_mode,
        'is_what_if': overrides.is_what_if,
        'basin_id': catchment_id,
        'catchment_id': catchment_id,
        'model_id': model_id,
        'checkpoint_path': bundle['checkpoint_path'],
        'issue_date': issue_date_str,
        'expected_state_date': expected_state_date,
        'area_km2': round(float(area_km2), 2),
        'lead_time_days': lead_time,
        'peak_discharge_cms': round(peak_q, 3) if peak_q is not None else None,
        'peak_date': peak_date,
        'hindcast_tail': hindcast_series,
        'forecast': forecast_series,
        'saved_state': saved_state,
    }


def run_forecast_model(  # noqa: PLR0913
    username: str | None,
    catchment_id: str | None = None,
    mode: str = 'coldstart',
    requested_issue_date: str | None = None,
    precip_overrides: dict[str, Any] | None = None,
    *,
    model_id: str = _DEFAULT_MODEL_ID,
    model_run_dir: Path | None = None,
    catchment_ids: Sequence[str] | None = None,
    temp_overrides: dict[str, float] | None = None,
    product_overrides: dict[str, dict[str, dict[str, float]]] | None = None,
    is_what_if: bool = False,
) -> dict[str, Any]:
    """Execute Cold-Start or Hot-Start inference on the forecast LSTM."""
    norm_mode = mode.lower().strip().replace('-', '').replace('_', '')
    if norm_mode not in ('coldstart', 'hotstart'):
        raise ValueError(
            f"Invalid mode {mode!r}; expected 'coldstart' or 'hotstart'."
        )

    target_ids = _normalize_catchment_id_list(catchment_id, catchment_ids)
    if not target_ids:
        raise ValueError(
            'At least one catchment_id must be selected to run the forecast '
            'model.'
        )
    primary_id = str(catchment_id).strip() if catchment_id else target_ids[0]

    t0 = resolve_issue_date(
        username=username,
        catchment_id=primary_id,
        requested_issue_date=requested_issue_date,
        probe_ecmwf=False,
    )
    issue_date_str = _format_date_str(t0)
    expected_state_ts = t0 - pd.Timedelta(days=1)
    expected_state_date = _format_date_str(expected_state_ts)

    bundle = {
        **_load_forecast_model_bundle(run_dir=model_run_dir),
        'active_mode': norm_mode,
        'active_model_id': model_id,
    }
    full_seq_len = int(bundle['cfg'].seq_length or _DEFAULT_SPINUP_DAYS)

    if norm_mode == 'hotstart':
        for cid in target_ids:
            state_info = inspect_saved_state(
                username=username, catchment_id=cid, model_id=model_id
            )
            if not state_info.get('exists'):
                raise ValueError(
                    'Cannot run model in Hot-Start mode: no saved LSTM state '
                    f"file exists for catchment '{cid}'. Please run Cold-Start "
                    'first to initialize the basin state for '
                    f'{expected_state_date}.'
                )
            st_date = state_info.get('state_date')
            if not _is_state_date_valid_for_hotstart(
                st_date, expected_state_ts, full_seq_len
            ):
                raise ValueError(
                    'Cannot run model in Hot-Start mode: saved LSTM state for '
                    f"catchment '{cid}' is stamped '{st_date}', which is "
                    'outside the valid hindcast window ending '
                    f"'{expected_state_date}' for forecast issue date "
                    f"'{issue_date_str}'. Please run Cold-Start first."
                )

    overrides = _unpack_override_payloads(
        precip_overrides,
        temp_overrides,
        product_overrides,
        is_what_if=is_what_if,
    )

    results_by_id: dict[str, dict[str, Any]] = {}
    for cid in target_ids:
        results_by_id[cid] = _run_forecast_model_single(
            username=username,
            catchment_id=cid,
            t0=t0,
            bundle=bundle,
            overrides=overrides,
        )

    primary_res = results_by_id[primary_id]
    updated_status = get_realtime_status(
        username=username,
        catchment_id=primary_id,
        requested_issue_date=issue_date_str,
        model_id=model_id,
        catchment_ids=target_ids,
    )

    return {
        **primary_res,
        'batch': len(target_ids) > 1,
        'basin_count': len(target_ids),
        'catchment_ids': target_ids,
        'results_by_id': results_by_id,
        'saved_state': updated_status['saved_state'],
        'post_run_status': updated_status,
    }


def inspect_realtime_stores_for_basin(
    username: str | None,
    basin_id: str,
    nowcast_lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    *,
    query_upstream_if_empty: bool = False,
    basin_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Server endpoint wrapper for ``GET /api/forecast/status``."""
    del query_upstream_if_empty
    return get_realtime_status(
        username=username,
        catchment_id=str(basin_id),
        lookback_days=int(nowcast_lookback_days or _DEFAULT_LOOKBACK_DAYS),
        probe_ecmwf=False,
        catchment_ids=basin_ids,
    )


def fetch_realtime_for_basin(  # noqa: PLR0913
    username: str | None,
    basin_id: str | None = None,
    mode: str = 'coldstart',
    reference_date: str | None = None,
    products: list[str] | None = None,
    *,
    lookback_days: int | None = None,
    overwrite: bool = False,
    basin_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Server endpoint wrapper for ``POST /api/forecast/fetch-realtime``."""
    spinup = int(lookback_days) if lookback_days else _DEFAULT_SPINUP_DAYS
    return fetch_realtime_forcing_for_catchment(
        username=username,
        catchment_id=str(basin_id) if basin_id else None,
        mode=mode,
        reference_date=reference_date,
        spinup_days=spinup,
        products=products,
        catchment_ids=basin_ids,
        overwrite=overwrite,
    )


def run_hydrological_model_for_basin(  # noqa: PLR0913
    username: str | None,
    basin_id: str | None = None,
    mode: str = 'coldstart',
    model_run_dir: str | Path | None = None,
    requested_issue_date: str | None = None,
    *,
    precip_overrides: dict[str, Any] | None = None,
    basin_ids: Sequence[str] | None = None,
    temp_overrides: dict[str, float] | None = None,
    product_overrides: dict[str, dict[str, dict[str, float]]] | None = None,
    is_what_if: bool = False,
) -> dict[str, Any]:
    """Server endpoint wrapper for ``POST /api/forecast/run-model``."""
    return run_forecast_model(
        username=username,
        catchment_id=str(basin_id) if basin_id else None,
        mode=mode,
        requested_issue_date=requested_issue_date,
        precip_overrides=precip_overrides,
        model_run_dir=Path(model_run_dir) if model_run_dir else None,
        catchment_ids=basin_ids,
        temp_overrides=temp_overrides,
        product_overrides=product_overrides,
        is_what_if=is_what_if,
    )
