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

import calendar
import contextlib
import copy
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import tqdm as tqdm_mod
import xarray as xr
from dask.callbacks import Callback as DaskCallback

from frontend import profile_manager
from frontend.config import (
    FLOOD_FORECASTING_REPO_DIR,
    ensure_flood_forecasting_on_sys_path,
    extend_multimet_package_path,
)
from model.datasetzoo.caravan import load_caravan_attributes
from model.datautils.scaler import Scaler
from model.modelzoo import get_model, load_model_weights
from model.training import (
    get_loss_obj,
    get_optimizer,
    get_regularization_obj,
)
from model.utils.assimilationconfig import AssimilationConfig
from model.utils.config import Config
from model.utils.configutils import (
    flatten_feature_list,
    product_name_from_feature,
)
from multimet.timeseries_extractors import config as mm_config
from multimet.timeseries_extractors import realtime as mm_realtime
from multimet.timeseries_extractors.hres import find_latest_hres_open_data_date
from multimet.timeseries_extractors.zarr_writer import (
    repair_zarr_basin_dimension_mismatch,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = 'google-floodhub-settings-110-epochs'
DEFAULT_MODEL_RUN_DIR = (
    FLOOD_FORECASTING_REPO_DIR
    / 'model'
    / 'pretrained-models'
    / _DEFAULT_MODEL_ID
)

PRETRAINED_FORECAST_MODELS: tuple[dict[str, Any], ...] = (
    {
        'id': 'google-floodhub-settings-110-epochs',
        'label': 'FloodHub Settings (With GraphCast · 110 Epochs)',
        'short_label': 'With GraphCast',
        'forecast_products': ['HRES', 'GRAPHCAST'],
        'hindcast_products': ['HRES', 'GRAPHCAST', 'IMERG', 'CPC'],
        'run_dir': (
            FLOOD_FORECASTING_REPO_DIR
            / 'model'
            / 'pretrained-models'
            / 'google-floodhub-settings-110-epochs'
        ),
    },
    {
        'id': 'google-floodhub-settings-no-graphcast-110-epochs',
        'label': 'FloodHub Settings (Without GraphCast · HRES + IMERG + CPC)',
        'short_label': 'No GraphCast',
        'forecast_products': ['HRES'],
        'hindcast_products': ['HRES', 'IMERG', 'CPC'],
        'run_dir': (
            FLOOD_FORECASTING_REPO_DIR
            / 'model'
            / 'pretrained-models'
            / 'google-floodhub-settings-no-graphcast-110-epochs'
        ),
    },
    {
        'id': 'google-floodhub-settings-aifs-110-epochs',
        'label': (
            'FloodHub Settings (With ECMWF AIFS · HRES + AIFS + IMERG + CPC)'
        ),
        'short_label': 'With ECMWF AIFS',
        'forecast_products': ['HRES', 'AIFS'],
        'hindcast_products': ['HRES', 'AIFS', 'IMERG', 'CPC'],
        'run_dir': (
            FLOOD_FORECASTING_REPO_DIR
            / 'model'
            / 'pretrained-models'
            / 'google-floodhub-settings-aifs-110-epochs'
        ),
    },
)


def list_forecast_models(
    username: str | None = None,
) -> list[dict[str, Any]]:
    """Return available and training forecast models for the Forecasting UI."""
    models: list[dict[str, Any]] = []
    for entry in PRETRAINED_FORECAST_MODELS:
        run_dir: Path = entry['run_dir']
        has_cfg = (run_dir / 'config.yml').is_file()
        ckpts = (
            sorted(run_dir.glob('model_epoch*.pt')) if run_dir.is_dir() else []
        )
        available = bool(has_cfg and len(ckpts) > 0)
        models.append(
            {
                'id': entry['id'],
                'label': entry['label'],
                'short_label': entry['short_label'],
                'forecast_products': list(entry['forecast_products']),
                'hindcast_products': list(entry['hindcast_products']),
                'available': available,
                'status': 'ready' if available else 'training',
                'is_pretrained': True,
            }
        )

    pm = profile_manager.get_profile_manager()
    user_key = _normalize_user_key(username)
    user_runs_dir = pm.get_model_runs_dir(user_key)
    if user_runs_dir.is_dir():
        for run_dir in sorted(user_runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            ckpts = sorted(run_dir.glob('model_epoch*.pt'))
            if (run_dir / 'config.yml').is_file() and ckpts:
                models.append(
                    {
                        'id': f'user_{run_dir.name}',
                        'label': f'User Fine-Tuned: {run_dir.name}',
                        'short_label': run_dir.name,
                        'forecast_products': ['HRES'],
                        'hindcast_products': ['HRES', 'IMERG', 'CPC'],
                        'available': True,
                        'status': 'ready',
                        'is_pretrained': False,
                    }
                )
    return models


def resolve_forecast_model_run_dir(
    model_id: str | None = _DEFAULT_MODEL_ID,
    *,
    username: str | None = None,
    run_dir: Path | str | None = None,
) -> Path:
    """Resolve a ``model_id`` or explicit ``run_dir`` to a model directory."""
    if run_dir is not None:
        return Path(run_dir)
    clean_id = (model_id or _DEFAULT_MODEL_ID).strip() or _DEFAULT_MODEL_ID
    for entry in PRETRAINED_FORECAST_MODELS:
        if entry['id'] == clean_id:
            return Path(entry['run_dir'])
    if clean_id.startswith('user_'):
        run_name = clean_id.removeprefix('user_')
        if (
            not run_name
            or '/' in run_name
            or '\\' in run_name
            or '..' in run_name
        ):
            raise ValueError(f'Invalid user model_id: {model_id!r}')
        pm = profile_manager.get_profile_manager()
        user_key = _normalize_user_key(username)
        runs_dir = pm.get_model_runs_dir(user_key).resolve()
        candidate = (runs_dir / run_name).resolve()
        if candidate.parent != runs_dir:
            raise ValueError(f'Invalid user model_id: {model_id!r}')
        return candidate
    candidate_pretrained = (
        FLOOD_FORECASTING_REPO_DIR / 'model' / 'pretrained-models' / clean_id
    )
    if candidate_pretrained.is_dir():
        return candidate_pretrained
    return DEFAULT_MODEL_RUN_DIR


_MIN_FORECAST_LEAD_STEPS = 2
_SECONDS_PER_DAY = 86400.0
_CMS_PER_MM_DAY_KM2 = 1000.0 / _SECONDS_PER_DAY
_DEFAULT_SPINUP_DAYS = 365
_DEFAULT_LOOKBACK_DAYS = 14
_DEFAULT_LEAD_TIME_DAYS = 7
_HINDCAST_TAIL_DAYS = 14
_NDIM_3D = 3
_MAX_NOWCAST_LAG_DAYS = 5

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
_ZARR_CACHE_LOCK = threading.Lock()
_ZARR_META_CACHE: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
_ZARR_WINDOW_CACHE: dict[
    tuple[tuple[str, int, int, int, int], str, str, int], xr.Dataset
] = {}
_MAX_ZARR_CACHE_ENTRIES = 64
_ASSIMILATION_SYNC_LOCK = threading.Lock()
_OBSERVED_STREAMFLOW_FORECAST_PAD_DAYS = 7

_ALL_REALTIME_PRODUCTS: tuple[str, ...] = (
    'HRES',
    'IMERG',
    'CPC',
    'AIFS',
    'GFS',
    'GEFS',
    'IFS_ENS',
    'DYNAMICAL_IMERG',
)
_FETCH_PROGRESS_LOCK = threading.Lock()
_FETCH_PROGRESS_BY_USER: dict[str, dict[str, Any]] = {}
_FETCH_CANCEL_EVENTS: dict[str, threading.Event] = {}
_RUN_PROGRESS_LOCK = threading.Lock()
_RUN_PROGRESS_BY_USER: dict[str, dict[str, Any]] = {}
_DA_PROGRESS_LOCK = threading.Lock()
_DA_PROGRESS_BY_USER: dict[str, dict[str, Any]] = {}
_DA_CANCEL_EVENTS: dict[str, threading.Event] = {}
_DA_SESSION_LOCK = threading.Lock()
_DA_SESSIONS_BY_KEY: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_FETCH_THREAD_LOCAL = threading.local()
_TQDM_PATCH_LOCK = threading.Lock()
_TQDM_PATCH_INSTALLED = False
_ORIG_TQDM_CLASS: Any = tqdm_mod.std.tqdm


class RealtimeFetchCancelledError(RuntimeError):
    """Raised when an active real-time forcing download is stopped."""


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
    safe_cid = _sanitize_catchment_filename_token(catchment_id)
    safe_mid = _sanitize_catchment_filename_token(model_id)
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


_DAILY_FORECAST_TTL_SECONDS: float = 4 * 3600.0


def get_daily_forecast_cache_path(
    username: str | None,
    catchment_id: str,
    model_id: str = _DEFAULT_MODEL_ID,
) -> Path:
    """Return the per-basin cached daily forecast JSON path."""
    pm = profile_manager.get_profile_manager()
    safe_cid = _sanitize_catchment_filename_token(catchment_id)
    safe_mid = _sanitize_catchment_filename_token(model_id)
    cache_dir = pm.get_profile_dir(username) / 'forecasts' / 'daily_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{safe_cid}_{safe_mid}_forecast.json'


def load_cached_daily_forecast(  # noqa: PLR0911
    username: str | None,
    catchment_id: str,
    model_id: str = _DEFAULT_MODEL_ID,
    *,
    ttl_seconds: float = _DAILY_FORECAST_TTL_SECONDS,
) -> dict[str, Any] | None:
    """Load cached daily forecast for ``catchment_id``, clearing if expired."""
    if not catchment_id:
        return None
    cache_path = get_daily_forecast_cache_path(
        username=username, catchment_id=catchment_id, model_id=model_id
    )
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    now = time.time()
    cached_at = _clean_float(payload.get('cached_at_unix'))
    if cached_at is None:
        try:
            cached_at = float(cache_path.stat().st_mtime)
        except OSError:
            return None
    if ttl_seconds <= 0.0 or (now - cached_at) > float(ttl_seconds):
        with contextlib.suppress(OSError):
            cache_path.unlink(missing_ok=True)
        return None

    payload['cached_at_unix'] = round(cached_at, 3)
    payload['expires_at_unix'] = round(cached_at + float(ttl_seconds), 3)
    payload['ttl_seconds'] = round(float(ttl_seconds), 1)
    return payload


def save_cached_daily_forecast(
    username: str | None,
    catchment_id: str,
    result_payload: dict[str, Any],
    model_id: str = _DEFAULT_MODEL_ID,
    *,
    ttl_seconds: float = _DAILY_FORECAST_TTL_SECONDS,
) -> dict[str, Any]:
    """Persist daily forecast result with TTL for browser refresh restore."""
    now = time.time()
    existing = load_cached_daily_forecast(
        username=username,
        catchment_id=catchment_id,
        model_id=model_id,
        ttl_seconds=ttl_seconds,
    )
    if (
        existing is not None
        and not result_payload.get('hindcast_tail')
        and isinstance(existing.get('hindcast_tail'), list)
        and len(existing['hindcast_tail']) > 0
    ):
        result_payload['hindcast_tail'] = existing['hindcast_tail']

    result_payload['cached_at_unix'] = round(now, 3)
    result_payload['expires_at_unix'] = round(now + float(ttl_seconds), 3)
    result_payload['ttl_seconds'] = round(float(ttl_seconds), 1)

    cache_path = get_daily_forecast_cache_path(
        username=username, catchment_id=catchment_id, model_id=model_id
    )
    tmp_path = (
        cache_path.parent
        / f'.{cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}'
    )
    tmp_path.write_text(json.dumps(result_payload), encoding='utf-8')
    tmp_path.replace(cache_path)
    return result_payload


_MAX_SHORT_UPLOAD_POINTS = 90
_MIN_CSV_COLUMNS = 2
_ACCEPTED_BASIN_COLUMNS: tuple[str, ...] = (
    'basin',
    'basin_id',
    'catchment_id',
    'gauge_id',
    'site_no',
    'station_id',
)
_ACCEPTED_DATE_COLUMNS: tuple[str, ...] = (
    'date',
    'datetime',
    'timestamp',
    'time',
    'valid_time',
    'day',
)
_ACCEPTED_FLOW_COLUMNS: tuple[str, ...] = (
    'streamflow',
    'discharge_cms',
    'streamflow_cms',
    'discharge',
    'q_cms',
    'flow_cms',
    'q',
    'flow',
    'obs',
    'value',
)
_MISSING_VALUE_TOKENS: frozenset[str] = frozenset(
    {
        '',
        'na',
        'nan',
        'null',
        'none',
        '-999',
        '-999.0',
        '-9999',
        '-9999.0',
        '-99999',
        '-99999.0',
    }
)
_SENTINEL_NEGATIVE_TOKENS: frozenset[str] = frozenset(
    {'-999', '-999.0', '-9999', '-9999.0', '-99999', '-99999.0'}
)
_ISO_DATE_PREFIX_RE = re.compile(
    r'^(\d{4})[-/](\d{2})[-/](\d{2})'
    r'(?:[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$'
)
_FLOAT_TOKEN_RE = re.compile(
    r'^[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?$'
)
_MONTHS_PER_YEAR = 12


def _sanitize_catchment_filename_token(catchment_id: str) -> str:
    """Sanitize ``catchment_id`` into a collision-free filename stem."""
    raw = str(catchment_id).strip()
    cleaned = ''.join(c for c in raw if c.isalnum() or c in ('_', '-'))
    stem = cleaned or 'basin'
    if cleaned == raw and raw:
        return stem
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    return f'{stem}_{digest}'


def get_assimilation_csv_path(
    username: str | None,
    catchment_id: str,
) -> Path:
    """Return the per-basin uploaded real-time streamflow CSV path.

    Args:
        username: Target profile username (or None for active profile).
        catchment_id: Basin identifier.

    Returns:
        Filesystem ``Path`` under ``assimilation/uploads/``.
    """
    pm = profile_manager.get_profile_manager()
    safe_cid = _sanitize_catchment_filename_token(catchment_id)
    uploads_dir = pm.get_assimilation_dir(username) / 'uploads'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    return uploads_dir / f'{safe_cid}.csv'


def _clean_streamflow_value(val: object) -> float | None:
    """Validate and clean a single streamflow observation (m^3/s).

    Preserves missing/negative/non-finite values as ``None`` (``NaN``).
    """
    cleaned = _clean_float(val)
    if cleaned is None or cleaned < 0.0:
        return None
    return round(cleaned, 3)


def _parse_iso_calendar_date(raw_val: object) -> pd.Timestamp | None:
    """Parse an ISO calendar date string into a naive daily Timestamp.

    Preserves the local calendar date ``YYYY-MM-DD`` for timezone-qualified
    ISO timestamps and returns ``None`` if ``raw_val`` is not a valid ISO date.
    """
    if isinstance(raw_val, pd.Timestamp):
        ts = raw_val
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.floor('D')
    text = str(raw_val or '').strip()
    if not text:
        return None
    m = _ISO_DATE_PREFIX_RE.match(text)
    if m is None:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= _MONTHS_PER_YEAR):
        return None
    max_day = calendar.monthrange(year, month)[1]
    if not (1 <= day <= max_day):
        return None
    return pd.Timestamp(year=year, month=month, day=day)


def is_valid_issue_date_str(requested_issue_date: str | None) -> bool:
    """Return True if ``requested_issue_date`` is None, ``'latest'``, or ISO."""
    if requested_issue_date is None:
        return True
    cleaned = str(requested_issue_date).strip()
    if not cleaned or cleaned.lower() == 'latest':
        return True
    return _parse_iso_calendar_date(cleaned) is not None


def _read_uploaded_csv_points(
    csv_path: Path,
    expected_catchment_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Read a normalized per-basin upload CSV into ``(basin_id, points)``."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return None
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if df.empty or 'date' not in df.columns:
        return None
    val_col = next(
        (c for c in ('streamflow', 'discharge_cms') if c in df.columns),
        None,
    )
    if val_col is None:
        return None
    if 'basin' in df.columns:
        basins = [str(b).strip() for b in df['basin'] if str(b).strip()]
        cid = basins[0] if basins else csv_path.stem
        if (
            expected_catchment_id is not None
            and cid != str(expected_catchment_id).strip()
        ):
            return None
    else:
        cid = (
            str(expected_catchment_id).strip()
            if expected_catchment_id
            else csv_path.stem
        )

    points: list[dict[str, Any]] = []
    for raw_d, raw_q in zip(df['date'], df[val_col], strict=False):
        ts = _parse_iso_calendar_date(raw_d)
        if ts is None:
            continue
        q_str = str(raw_q).strip()
        q_val: float | None = None
        if q_str.lower() not in _MISSING_VALUE_TOKENS and _FLOAT_TOKEN_RE.match(
            q_str
        ):
            q_val = _clean_streamflow_value(float(q_str))
        points.append(
            {
                'date': _format_date_str(ts),
                'discharge_cms': q_val,
            }
        )
    if not points:
        return None
    points.sort(key=lambda item: item['date'])
    return cid, points


def _can_modify_assimilation_zarr(zarr_path: Path) -> bool:
    """Return True if ``zarr_path`` is missing or owned by CSV uploads."""
    if not zarr_path.exists():
        return True
    if not _is_valid_zarr_store(zarr_path):
        return False
    with xr.open_zarr(zarr_path, consolidated=False) as ds:
        src = ds.attrs.get('source_dataset')
        return src in (None, 'User CSV Upload')


def _sync_assimilation_zarr_from_uploads(username: str | None) -> Path | None:
    """Rebuild ``streamflow_realtime.zarr`` atomically from upload CSVs."""
    pm = profile_manager.get_profile_manager()
    uploads_dir = pm.get_assimilation_dir(username) / 'uploads'
    zarr_path = pm.get_assimilation_zarr_path(username)

    with _ASSIMILATION_SYNC_LOCK:
        if not _can_modify_assimilation_zarr(zarr_path):
            return zarr_path

        basin_frames: dict[str, pd.Series] = {}
        if uploads_dir.is_dir():
            for csv_path in sorted(uploads_dir.glob('*.csv')):
                parsed = _read_uploaded_csv_points(csv_path)
                if parsed is not None:
                    cid, pts = parsed
                    s = pd.Series(
                        [
                            np.nan
                            if pt['discharge_cms'] is None
                            else float(pt['discharge_cms'])
                            for pt in pts
                        ],
                        index=pd.DatetimeIndex([pt['date'] for pt in pts]),
                        dtype=np.float32,
                    )
                    basin_frames[cid] = s[~s.index.duplicated(keep='last')]

        if not basin_frames:
            if zarr_path.exists():
                shutil.rmtree(zarr_path)
            return None

        all_basins = sorted(basin_frames.keys())
        min_dt = min(s.index.min() for s in basin_frames.values())
        max_dt = max(s.index.max() for s in basin_frames.values())
        date_index = pd.date_range(min_dt, max_dt, freq='D')
        arr = np.full(
            (len(all_basins), len(date_index)),
            np.nan,
            dtype=np.float32,
        )
        for b_idx, cid in enumerate(all_basins):
            s_reindexed = basin_frames[cid].reindex(date_index)
            arr[b_idx, :] = s_reindexed.to_numpy(dtype=np.float32)

        ds = xr.Dataset(
            {'streamflow': (('basin', 'date'), arr)},
            coords={'basin': all_basins, 'date': date_index.to_numpy()},
            attrs={
                'title': 'Uploaded Real-Time Streamflow Observations',
                'source_dataset': 'User CSV Upload',
                'temporal_resolution': '1D',
                'units': 'm^3/s',
            },
        )
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_zarr = (
            zarr_path.parent
            / f'.{zarr_path.name}.tmp.{os.getpid()}.{time.time_ns()}'
        )
        ds.to_zarr(tmp_zarr, mode='w', zarr_format=2, consolidated=True)
        backup_zarr = (
            zarr_path.parent
            / f'.{zarr_path.name}.old.{os.getpid()}.{time.time_ns()}'
        )
        if zarr_path.exists():
            zarr_path.replace(backup_zarr)
        tmp_zarr.replace(zarr_path)
        if backup_zarr.exists():
            shutil.rmtree(backup_zarr)
        return zarr_path


def _load_points_from_zarr(
    username: str | None, catchment_id: str
) -> list[dict[str, Any]]:
    """Read uploaded streamflow points from Zarr store fallback."""
    pm = profile_manager.get_profile_manager()
    zarr_path = pm.get_assimilation_zarr_path(username)
    if not _is_valid_zarr_store(zarr_path):
        return []
    with xr.open_zarr(zarr_path, consolidated=False) as ds:
        bkey = _find_basin_key_in_ds(ds, catchment_id)
        if bkey is None or 'date' not in ds.coords:
            return []
        sub = ds.sel(basin=bkey)
        v_name = next(
            (
                str(v)
                for v in ('streamflow', 'discharge_cms', 'q_cms')
                if v in sub.data_vars
            ),
            None,
        )
        if v_name is None:
            return []
        s_dates = pd.to_datetime(sub['date'].to_numpy()).floor('D')
        s_vals = sub[v_name].to_numpy()
    return [
        {
            'date': _format_date_str(d_val),
            'discharge_cms': _clean_streamflow_value(q_val),
        }
        for d_val, q_val in zip(s_dates, s_vals, strict=False)
    ]


def load_realtime_streamflow_series(
    username: str | None,
    catchment_id: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    *,
    max_points: int | None = _MAX_SHORT_UPLOAD_POINTS,
) -> list[dict[str, Any]]:
    """Load uploaded real-time streamflow observations for ``catchment_id``.

    Args:
        username: Target profile username (or None for active profile).
        catchment_id: Basin identifier to load.
        start_date: Optional inclusive lower date bound (``YYYY-MM-DD``).
        end_date: Optional inclusive upper date bound (``YYYY-MM-DD``).
        max_points: Maximum trailing points returned when no date window is
            specified (pass ``None`` to return all points).

    Returns:
        Sorted list of ``{'date': str, 'discharge_cms': float | None}``.
    """
    if not catchment_id:
        return []
    cid_clean = str(catchment_id).strip()
    csv_path = get_assimilation_csv_path(username, cid_clean)
    parsed = _read_uploaded_csv_points(
        csv_path,
        expected_catchment_id=cid_clean,
    )
    if parsed is not None:
        points = parsed[1]
    else:
        points = _load_points_from_zarr(username, cid_clean)

    points.sort(key=lambda item: item['date'])
    if not points:
        return []

    if start_date is not None or end_date is not None:
        start_str = _format_date_str(start_date) if start_date else None
        end_str = _format_date_str(end_date) if end_date else None
        return [
            pt
            for pt in points
            if (start_str is None or pt['date'] >= start_str)
            and (end_str is None or pt['date'] <= end_str)
        ]

    if max_points is not None and max_points > 0:
        return points[-max_points:]
    return points


def _detect_csv_columns_result(
    df: pd.DataFrame,
) -> tuple[tuple[str | None, str, str] | None, str | None]:
    """Identify ``(basin_col, date_col, flow_col)`` from header names."""
    cols = [str(c) for c in df.columns]
    if len(cols) < _MIN_CSV_COLUMNS:
        msg = (
            'Uploaded CSV must contain at least 2 columns (date and '
            'streamflow).'
        )
        return None, msg

    lower_to_col = {c.strip().lower(): c for c in cols}

    basin_col = next(
        (
            lower_to_col[cand]
            for cand in _ACCEPTED_BASIN_COLUMNS
            if cand in lower_to_col
        ),
        None,
    )

    date_col = next(
        (
            lower_to_col[cand]
            for cand in _ACCEPTED_DATE_COLUMNS
            if cand in lower_to_col and lower_to_col[cand] != basin_col
        ),
        None,
    )
    if date_col is None:
        msg = (
            f'Could not find a recognized date column in {cols}. '
            f'Accepted date headers: {list(_ACCEPTED_DATE_COLUMNS)}.'
        )
        return None, msg

    flow_col = next(
        (
            lower_to_col[cand]
            for cand in _ACCEPTED_FLOW_COLUMNS
            if cand in lower_to_col
            and lower_to_col[cand] not in (basin_col, date_col)
        ),
        None,
    )
    if flow_col is None:
        msg = (
            f'Could not find a recognized streamflow column in {cols}. '
            f'Accepted streamflow headers: {list(_ACCEPTED_FLOW_COLUMNS)}.'
        )
        return None, msg

    return (basin_col, date_col, flow_col), None


def _detect_csv_columns(
    df: pd.DataFrame,
) -> tuple[str | None, str, str]:
    """Identify ``(basin_col, date_col, flow_col)`` in an uploaded CSV."""
    res, err = _detect_csv_columns_result(df)
    if err is not None or res is None:
        raise ValueError(err or 'Invalid CSV columns.')
    return res


def _parse_flow_token(
    raw_flow_str: str,
    row_idx: int,
    flow_col: str,
) -> tuple[float, str, str | None]:
    """Parse a single CSV streamflow token into ``(val, kind, err_msg)``."""
    lower_flow = raw_flow_str.lower()
    if lower_flow in _MISSING_VALUE_TOKENS:
        kind = (
            'negative'
            if lower_flow in _SENTINEL_NEGATIVE_TOKENS
            else 'nonfinite'
        )
        return np.nan, kind, None
    if not _FLOAT_TOKEN_RE.match(raw_flow_str):
        msg = (
            f'Invalid numeric streamflow value {raw_flow_str!r} at row '
            f"{row_idx + 1} in column '{flow_col}'."
        )
        return np.nan, 'error', msg
    parsed_float = float(raw_flow_str)
    if not math.isfinite(parsed_float):
        return np.nan, 'nonfinite', None
    if parsed_float < 0.0:
        return np.nan, 'negative', None
    return round(parsed_float, 3), 'valid', None


def _parse_streamflow_rows(
    df: pd.DataFrame,
    basin_col: str | None,
    date_col: str,
    flow_col: str,
    fallback_cid: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse validated CSV rows into per-basin ``(date, flow)`` lists."""
    rows_by_basin: dict[str, list[tuple[str, float]]] = {}
    seen_dates_by_basin: dict[str, set[str]] = {}
    dup_dates_by_basin: dict[str, list[str]] = {}
    counts = {'nonfinite': 0, 'negative': 0, 'valid': 0}

    for row_idx in range(len(df)):
        cid_val = (
            str(df[basin_col].iloc[row_idx]).strip()
            if basin_col is not None
            else fallback_cid
        )
        if not cid_val:
            msg = (
                f'Missing basin identifier at row {row_idx + 1} in '
                f"column '{basin_col}'."
            )
            return None, msg

        raw_date_str = str(df[date_col].iloc[row_idx]).strip()
        ts = _parse_iso_calendar_date(raw_date_str)
        if ts is None:
            msg = (
                f'Could not parse date {raw_date_str!r} at row {row_idx + 1} '
                f"in column '{date_col}'. Expected YYYY-MM-DD."
            )
            return None, msg
        date_str = _format_date_str(ts)

        seen_set = seen_dates_by_basin.setdefault(cid_val, set())
        if date_str in seen_set:
            dups = dup_dates_by_basin.setdefault(cid_val, [])
            if date_str not in dups:
                dups.append(date_str)
        seen_set.add(date_str)

        raw_flow_str = str(df[flow_col].iloc[row_idx]).strip()
        flow_val, kind, flow_err = _parse_flow_token(
            raw_flow_str,
            row_idx,
            flow_col,
        )
        if flow_err is not None:
            return None, flow_err
        counts[kind] += 1
        rows_by_basin.setdefault(cid_val, []).append((date_str, flow_val))

    if dup_dates_by_basin:
        first_cid, dup_list = next(iter(dup_dates_by_basin.items()))
        msg = (
            f"Uploaded CSV contains duplicate dates for basin '{first_cid}': "
            f'{dup_list}. Provide one daily observation per date.'
        )
        return None, msg

    if counts['valid'] == 0:
        msg = (
            f"Uploaded CSV column '{flow_col}' contains no valid non-negative "
            'streamflow observations.'
        )
        return None, msg

    return (
        {
            'basin_col': basin_col,
            'date_col': date_col,
            'flow_col': flow_col,
            'rows_by_basin': rows_by_basin,
            'n_rows_in': len(df),
            'n_dropped_dates': 0,
            'n_nonfinite': counts['nonfinite'],
            'n_negative': counts['negative'],
        },
        None,
    )


def _parse_and_validate_streamflow_csv(
    csv_content: str,
    catchment_id: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and parse raw CSV text into per-basin streamflow tables."""
    raw_text = (csv_content or '').strip()
    if not raw_text:
        return None, 'Uploaded CSV file is empty.'

    non_empty_lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if len(non_empty_lines) < _MIN_CSV_COLUMNS:
        return None, 'Uploaded CSV contains no data rows.'

    df = pd.read_csv(
        io.StringIO(raw_text),
        dtype=str,
        keep_default_na=False,
    )
    if df.empty:
        return None, 'Uploaded CSV contains no data rows.'

    cols_res, cols_err = _detect_csv_columns_result(df)
    if cols_err is not None or cols_res is None:
        return None, cols_err
    basin_col, date_col, flow_col = cols_res

    fallback_cid = str(catchment_id or '').strip()
    if basin_col is None and not fallback_cid:
        msg = (
            'A catchment_id is required when the CSV does not include a '
            'basin column.'
        )
        return None, msg

    return _parse_streamflow_rows(
        df=df,
        basin_col=basin_col,
        date_col=date_col,
        flow_col=flow_col,
        fallback_cid=fallback_cid,
    )


def validate_streamflow_csv_request(
    csv_content: str,
    catchment_id: str | None = None,
) -> str | None:
    """Validate CSV upload parameters and return an error message or None."""
    _, err = _parse_and_validate_streamflow_csv(csv_content, catchment_id)
    return err


def upload_realtime_streamflow_csv(
    username: str | None,
    catchment_id: str | None,
    csv_content: str,
    filename: str = 'streamflow.csv',
) -> dict[str, Any]:
    """Parse, validate, and store an uploaded real-time streamflow CSV.

    Args:
        username: Target profile username (or None for active profile).
        catchment_id: Default catchment ID when the CSV has no ``basin`` column.
        csv_content: Raw UTF-8 CSV content string.
        filename: Original uploaded filename.

    Returns:
        Dictionary summarizing uploaded rows, date bounds, and updated status.

    Raises:
        ValueError: If the CSV is empty, missing required headers, contains
            unparseable dates/numbers, duplicate dates, or no valid values.
    """
    parsed, err = _parse_and_validate_streamflow_csv(csv_content, catchment_id)
    if err is not None or parsed is None:
        raise ValueError(err or 'Invalid streamflow CSV.')

    date_col: str = parsed['date_col']
    flow_col: str = parsed['flow_col']
    rows_by_basin: dict[str, list[tuple[str, float]]] = parsed['rows_by_basin']

    updated_basins: list[str] = []
    for cid_str in sorted(rows_by_basin.keys()):
        sorted_rows = sorted(rows_by_basin[cid_str], key=lambda item: item[0])
        daily = pd.DataFrame(
            {
                'basin': [cid_str] * len(sorted_rows),
                'date': [r[0] for r in sorted_rows],
                'streamflow': [r[1] for r in sorted_rows],
            }
        )
        out_csv = get_assimilation_csv_path(username, cid_str)
        daily.to_csv(out_csv, index=False)
        updated_basins.append(cid_str)

    zarr_path = _sync_assimilation_zarr_from_uploads(username)
    with _DA_SESSION_LOCK:
        _DA_SESSIONS_BY_KEY.clear()
    target_cid = (
        str(catchment_id).strip()
        if (catchment_id and str(catchment_id).strip() in updated_basins)
        else updated_basins[0]
    )
    full_series = load_realtime_streamflow_series(
        username,
        target_cid,
        max_points=None,
    )
    series_for_target = full_series[-_MAX_SHORT_UPLOAD_POINTS:]
    truncated = len(full_series) > len(series_for_target)
    valid_count = sum(
        1 for pt in full_series if pt['discharge_cms'] is not None
    )
    observed_by_id = {
        cid: load_realtime_streamflow_series(username, cid)
        for cid in updated_basins
    }

    status_payload = get_realtime_status(
        username=username,
        catchment_id=target_cid,
    )
    start_date_str = full_series[0]['date'] if full_series else None
    end_date_str = full_series[-1]['date'] if full_series else None
    return {
        'status': 'success',
        'username': username,
        'catchment_id': target_cid,
        'basins_updated': updated_basins,
        'filename': str(filename or 'streamflow.csv'),
        'date_column': date_col,
        'streamflow_column': flow_col,
        'row_count': len(full_series),
        'total_points': len(full_series),
        'returned_points': len(series_for_target),
        'truncated': truncated,
        'valid_count': valid_count,
        'n_rows_in': parsed['n_rows_in'],
        'n_dropped_dates': parsed['n_dropped_dates'],
        'n_nonfinite': parsed['n_nonfinite'],
        'n_negative': parsed['n_negative'],
        'start_date': start_date_str,
        'end_date': end_date_str,
        'message': (
            f'Loaded {valid_count}/{len(full_series)} valid obs '
            f'({start_date_str} → {end_date_str})'
        ),
        'observed_streamflow': series_for_target,
        'observed_by_id': observed_by_id,
        'zarr_path': str(zarr_path) if zarr_path else None,
        'forecast_status': status_payload,
    }


def clear_realtime_streamflow_csv(
    username: str | None,
    catchment_id: str,
) -> dict[str, Any]:
    """Delete uploaded real-time streamflow CSV for ``catchment_id``.

    Args:
        username: Target profile username (or None for active profile).
        catchment_id: Basin identifier whose uploaded CSV should be removed.

    Returns:
        Dictionary with cleared ``observed_streamflow`` and refreshed status.

    Raises:
        ValueError: If ``catchment_id`` is empty.
    """
    cid_str = str(catchment_id or '').strip()
    if not cid_str:
        raise ValueError('catchment_id is required to clear streamflow data.')
    csv_path = get_assimilation_csv_path(username, cid_str)
    if csv_path.exists():
        csv_path.unlink()
    _sync_assimilation_zarr_from_uploads(username)
    with _DA_SESSION_LOCK:
        _DA_SESSIONS_BY_KEY.clear()
    status_payload = get_realtime_status(
        username=username,
        catchment_id=cid_str,
    )
    return {
        'status': 'success',
        'username': username,
        'catchment_id': cid_str,
        'observed_streamflow': [],
        'forecast_status': status_payload,
    }


_RETURN_PERIOD_KEY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        'return_period_2',
        ('return_period_2', 'rp_2', 'rp2', 'q2', '2', '2yr', 'warning_2yr'),
    ),
    (
        'return_period_5',
        ('return_period_5', 'rp_5', 'rp5', 'q5', '5', '5yr', 'danger_5yr'),
    ),
    (
        'return_period_10',
        ('return_period_10', 'rp_10', 'rp10', 'q10', '10', '10yr'),
    ),
    (
        'return_period_20',
        (
            'return_period_20',
            'rp_20',
            'rp20',
            'q20',
            '20',
            '20yr',
            'extreme_20yr',
        ),
    ),
    (
        'return_period_25',
        ('return_period_25', 'rp_25', 'rp25', 'q25', '25', '25yr'),
    ),
    (
        'return_period_50',
        ('return_period_50', 'rp_50', 'rp50', 'q50', '50', '50yr'),
    ),
    (
        'return_period_100',
        ('return_period_100', 'rp_100', 'rp100', 'q100', '100', '100yr'),
    ),
)
_BARE_NUMERIC_RP_ALIASES: frozenset[str] = frozenset(
    {'2', '5', '10', '20', '25', '50', '100'}
)


def _extract_normalized_return_periods(
    raw_dict: dict[str, Any],
    *,
    allow_bare_numeric_keys: bool = True,
) -> dict[str, float]:
    """Normalize return period threshold keys into ``return_period_<yr>``."""
    nested = raw_dict.get('return_periods_cms') or raw_dict.get(
        'return_periods'
    )
    top_lower = {
        str(k).strip().lower(): v
        for k, v in raw_dict.items()
        if allow_bare_numeric_keys
        or str(k).strip().lower() not in _BARE_NUMERIC_RP_ALIASES
    }
    if isinstance(nested, dict):
        for k, v in nested.items():
            top_lower[str(k).strip().lower()] = v
    normalized: dict[str, float] = {}
    for canonical_key, aliases in _RETURN_PERIOD_KEY_ALIASES:
        for alias in aliases:
            if alias in top_lower:
                cleaned = _clean_float(top_lower[alias])
                if cleaned is not None and cleaned > 0.0:
                    normalized[canonical_key] = round(cleaned, 3)
                    break
    return normalized


def load_basin_return_periods(
    username: str | None,
    catchment_id: str,
) -> dict[str, float]:
    """Load return period thresholds (m^3/s) for ``catchment_id`` if present.

    Args:
        username: Target profile username (or None for active profile).
        catchment_id: Basin identifier.

    Returns:
        Dictionary mapping canonical ``return_period_<yr>`` keys to thresholds
        in m^3/s.
    """
    if not catchment_id:
        return {}
    pm = profile_manager.get_profile_manager()
    rp_path = pm.get_return_periods_path(username)
    if rp_path.is_file() and rp_path.stat().st_size > 0:
        rp_data = json.loads(rp_path.read_text(encoding='utf-8'))
        if isinstance(rp_data, dict):
            entry = rp_data.get(catchment_id)
            if isinstance(entry, dict):
                normalized = _extract_normalized_return_periods(
                    entry,
                    allow_bare_numeric_keys=True,
                )
                if normalized:
                    return normalized

    for feat in pm.load_watersheds(username):
        props = feat.get('properties') or {}
        cid = str(props.get('catchment_id') or feat.get('id') or '').strip()
        if cid == catchment_id and isinstance(props, dict):
            normalized = _extract_normalized_return_periods(
                props,
                allow_bare_numeric_keys=False,
            )
            if normalized:
                return normalized
    return {}


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


def _zarr_store_mtime_key(
    store_path: Path,
) -> tuple[str, int, int, int, int]:
    """Return a fast mtime cache key for ``store_path``."""
    resolved = str(store_path.resolve())
    if not store_path.exists():
        return (resolved, 0, 0, 0, 0)
    root_mtime = store_path.stat().st_mtime_ns
    meta_mtime = 0
    for marker in ('.zmetadata', '.zgroup', 'zarr.json'):
        p = store_path / marker
        if p.exists():
            meta_mtime = max(meta_mtime, p.stat().st_mtime_ns)
    date_mtime = 0
    for rel in ('date', 'date/.zarray', 'date/0'):
        p = store_path / rel
        if p.exists():
            date_mtime = max(date_mtime, p.stat().st_mtime_ns)
    basin_mtime = 0
    for rel in ('basin', 'basin/.zarray', 'basin/0'):
        p = store_path / rel
        if p.exists():
            basin_mtime = max(basin_mtime, p.stat().st_mtime_ns)
    return (resolved, root_mtime, meta_mtime, date_mtime, basin_mtime)


def _open_zarr_self_healing(store_path: Path) -> xr.Dataset:
    """Open ``store_path``, repairing partial-append basin shape mismatches."""
    try:
        return xr.open_zarr(store_path, consolidated=False)
    except ValueError:
        if repair_zarr_basin_dimension_mismatch(store_path):
            return xr.open_zarr(store_path, consolidated=False)
        raise


def _get_cached_store_meta(store_path: Path) -> dict[str, Any]:
    """Return cached basin list and issue date map for ``store_path``."""
    mkey = _zarr_store_mtime_key(store_path)
    with _ZARR_CACHE_LOCK:
        cached = _ZARR_META_CACHE.get(mkey)
        if cached is not None:
            return cached
    last_date_str: str | None = None
    has_lead_time = False
    try:
        with _open_zarr_self_healing(store_path) as ds:
            basins = (
                [str(b) for b in ds.coords['basin'].to_numpy()]
                if 'basin' in ds.coords
                else []
            )
            has_lead_time = 'lead_time' in ds.dims
            if 'date' in ds.coords and int(ds.sizes.get('date', 0)) > 0:
                last_ts = pd.Timestamp(ds.coords['date'].to_numpy()[-1]).floor(
                    'D'
                )
                last_date_str = _format_date_str(last_ts)
    except Exception:
        logger.warning(
            'Could not read Zarr store metadata at %s',
            store_path,
            exc_info=True,
        )
        basins = []
    entry: dict[str, Any] = {
        'basins': basins,
        'lower_map': {b.lower(): b for b in basins},
        'latest_issue_by_basin': {},
        'last_date': last_date_str,
        'has_lead_time': has_lead_time,
    }
    with _ZARR_CACHE_LOCK:
        existing = _ZARR_META_CACHE.get(mkey)
        if existing is not None:
            return existing
        if len(_ZARR_META_CACHE) >= _MAX_ZARR_CACHE_ENTRIES:
            _ZARR_META_CACHE.pop(next(iter(_ZARR_META_CACHE)))
        _ZARR_META_CACHE[mkey] = entry
        return entry


def _open_cached_zarr_window(
    store_path: Path,
    nowcast_dates: pd.DatetimeIndex,
    t0: pd.Timestamp,
) -> xr.Dataset:
    """Return an in-memory Zarr Dataset sliced to ``nowcast_dates`` + ``t0``."""
    mkey = _zarr_store_mtime_key(store_path)
    nc_start = _format_date_str(nowcast_dates[0]) if len(nowcast_dates) else ''
    t0_str = _format_date_str(t0)
    wkey = (mkey, nc_start, t0_str, len(nowcast_dates))
    with _ZARR_CACHE_LOCK:
        cached = _ZARR_WINDOW_CACHE.get(wkey)
        if cached is not None:
            return cached
    with _open_zarr_self_healing(store_path) as raw_ds:
        drop_vars = [
            v for v in raw_ds.data_vars if str(v).endswith('_missing_fraction')
        ]
        ds = raw_ds.drop_vars(drop_vars) if drop_vars else raw_ds
        if 'date' in ds.coords:
            store_dates = pd.to_datetime(ds['date'].to_numpy()).floor('D')
            target_dates = set(pd.to_datetime(nowcast_dates).floor('D')) | {t0}
            keep_idx = [
                i for i, d in enumerate(store_dates) if d in target_dates
            ]
            ds_win = (
                ds.isel(date=keep_idx).load()
                if keep_idx
                else ds.isel(date=slice(0, 0)).load()
            )
        else:
            ds_win = ds.load()
    with _ZARR_CACHE_LOCK:
        if len(_ZARR_WINDOW_CACHE) >= _MAX_ZARR_CACHE_ENTRIES:
            _ZARR_WINDOW_CACHE.pop(next(iter(_ZARR_WINDOW_CACHE)))
        _ZARR_WINDOW_CACHE[wkey] = ds_win
    return ds_win


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


def _resolve_basin_latest_issue(
    sub: xr.Dataset,
    precip_var: str | None,
    *,
    fallback_to_last_date: bool = True,
) -> pd.Timestamp | None:
    """Resolve the latest valid forecast issue date for a single-basin slice."""
    if precip_var and 'lead_time' in sub.dims:
        da = sub[precip_var]
        if da.shape[-1] >= _MIN_FORECAST_LEAD_STEPS:
            tail_da = da.isel(date=slice(-_DEFAULT_LOOKBACK_DAYS, None))
            fc_vals = tail_da.isel(lead_time=slice(1, None)).to_numpy()
            valid_dates = tail_da['date'].to_numpy()[
                (~np.isnan(fc_vals)).any(axis=-1)
            ]
            if len(valid_dates) == 0 and da.shape[0] > _DEFAULT_LOOKBACK_DAYS:
                fc_vals = da.isel(lead_time=slice(1, None)).to_numpy()
                valid_dates = sub['date'].to_numpy()[
                    (~np.isnan(fc_vals)).any(axis=-1)
                ]
            if len(valid_dates) > 0:
                return (
                    pd.Timestamp(valid_dates[-1]).floor('D').tz_localize(None)
                )
            if not fallback_to_last_date:
                return None
    dates = sub['date'].to_numpy()
    if len(dates) > 0:
        return pd.Timestamp(dates[-1]).floor('D').tz_localize(None)
    return None


def _extract_latest_issue_from_store(
    store_path: Path, catchment_id: str
) -> pd.Timestamp | None:
    """Return latest valid issue date for ``catchment_id`` in ``store_path``."""
    if not _is_valid_zarr_store(store_path):
        return None
    meta = _get_cached_store_meta(store_path)
    bkey = (
        catchment_id
        if catchment_id in meta['basins']
        else meta['lower_map'].get(catchment_id.lower())
    )
    if bkey is None:
        return None
    latest_map: dict[str, pd.Timestamp | None] = meta['latest_issue_by_basin']
    with _ZARR_CACHE_LOCK:
        if bkey in latest_map:
            return latest_map[bkey]
    with xr.open_zarr(store_path, consolidated=False) as ds:
        if 'date' not in ds.coords:
            with _ZARR_CACHE_LOCK:
                latest_map[bkey] = None
            return None
        precip_var = next(
            (
                str(v)
                for v in ds.data_vars
                if 'precipitation' in str(v) or str(v).endswith('_precip')
            ),
            None,
        )
        tail_ds = ds.isel(date=slice(-_DEFAULT_LOOKBACK_DAYS, None))
        if precip_var:
            tail_ds = tail_ds[[precip_var]].load()
        computed: dict[str, pd.Timestamp | None] = {}
        for bk in meta['basins']:
            res_ts = _resolve_basin_latest_issue(
                tail_ds.sel(basin=bk) if precip_var else ds.sel(basin=bk),
                precip_var,
                fallback_to_last_date=False,
            )
            if res_ts is None:
                res_ts = _resolve_basin_latest_issue(
                    ds.sel(basin=bk),
                    precip_var,
                    fallback_to_last_date=True,
                )
            computed[bk] = res_ts
    with _ZARR_CACHE_LOCK:
        latest_map.update(computed)
        return latest_map.get(bkey)


def _find_latest_issue_date(
    username: str | None,
    catchment_id: str,
    *,
    probe_ecmwf: bool = False,
) -> pd.Timestamp | None:
    """Search forecast Zarr stores (and optionally ECMWF) for issue date."""
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
        ValueError: When ``requested_issue_date`` is malformed or no valid
            forecast issue date can be resolved.
    """
    if (
        requested_issue_date
        and str(requested_issue_date).strip().lower() != 'latest'
    ):
        parsed = _parse_iso_calendar_date(requested_issue_date)
        if parsed is None:
            raise ValueError(
                f'Invalid issue_date {requested_issue_date!r}; '
                'expected YYYY-MM-DD.'
            )
        return parsed

    found = _find_latest_issue_date(
        username=username,
        catchment_id=catchment_id,
        probe_ecmwf=probe_ecmwf,
    )
    if found is not None:
        return found

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
    target_dates_set = set(pd.to_datetime(ctx.nowcast_dates).floor('D'))
    keep_indices = [
        i for i, d in enumerate(store_dates) if d in target_dates_set
    ]
    sub_window = (
        ctx.sub.isel(date=keep_indices)
        if keep_indices
        else ctx.sub.isel(date=slice(0, 0))
    )
    date_strs = [_format_date_str(store_dates[i]) for i in keep_indices]

    var_maps: dict[str, dict[str, float | None]] = {}
    for v_name in ctx.all_vars:
        vals_1d = sub_window[v_name].to_numpy()
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
    target_dates_set = set(pd.to_datetime(ctx.nowcast_dates).floor('D'))
    keep_indices = [
        i for i, d in enumerate(store_dates) if d in target_dates_set
    ]
    hc_sub = (
        ctx.sub.isel(date=keep_indices, lead_time=0)
        if keep_indices
        else ctx.sub.isel(date=slice(0, 0), lead_time=0)
    )
    store_date_strs = [_format_date_str(store_dates[i]) for i in keep_indices]

    hc_var_maps: dict[str, dict[str, float | None]] = {}
    for v_name in ctx.all_vars:
        da_1d_v = hc_sub[v_name].to_numpy()
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
        meta = _get_cached_store_meta(store_path)
        bkey = (
            catchment_id
            if catchment_id in meta['basins']
            else meta['lower_map'].get(catchment_id.lower())
        )
        if bkey is None:
            continue
        ds = _open_cached_zarr_window(store_path, nowcast_dates, t0)
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


def _collect_basins_with_realtime_data(
    stores: dict[str, Path],
) -> set[str]:
    """Return lower-cased set of basin IDs present in any valid Zarr store."""
    present: set[str] = set()
    for store_path in stores.values():
        if not _is_valid_zarr_store(store_path):
            continue
        meta = _get_cached_store_meta(store_path)
        present.update(meta['lower_map'].keys())
    return present


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
    basins_with_data = _collect_basins_with_realtime_data(stores)
    catchments_status: dict[str, dict[str, Any]] = {}
    for cid in all_known_ids:
        c_state = inspect_saved_state(
            username=username, catchment_id=cid, model_id=model_id
        )
        c_has_data = cid.lower() in basins_with_data
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
        c_obs = load_realtime_streamflow_series(
            username=username, catchment_id=cid
        )
        c_has_obs = any(pt['discharge_cms'] is not None for pt in c_obs)
        catchments_status[cid] = {
            'catchment_id': cid,
            'issue_date': issue_date_str,
            'expected_state_date': expected_state_date,
            'saved_state': c_state,
            'is_state_from_previous_day': c_prev,
            'is_state_valid_for_hotstart': c_valid_hs,
            'has_realtime_data': c_has_data,
            'has_observed_streamflow': c_has_obs,
            'can_fetch_hotstart': c_valid_hs,
            'can_run_hotstart': bool(c_valid_hs and c_has_data),
            'can_run_coldstart': bool(c_has_data),
        }
    return catchments_status


def _resolve_status_issue_date(
    username: str | None,
    primary_id: str,
    requested_issue_date: str | None,
    *,
    probe_ecmwf: bool,
) -> tuple[pd.Timestamp, str]:
    """Resolve ``(t0, issue_date_source)`` for status payloads."""
    if (
        requested_issue_date
        and str(requested_issue_date).strip().lower() != 'latest'
    ):
        parsed = _parse_iso_calendar_date(requested_issue_date)
        if parsed is None:
            raise ValueError(
                f'Invalid issue_date {requested_issue_date!r}; '
                'expected YYYY-MM-DD.'
            )
        return parsed, 'requested'

    found = _find_latest_issue_date(
        username=username,
        catchment_id=primary_id,
        probe_ecmwf=probe_ecmwf,
    )
    if found is not None:
        return found, 'forecast_store'

    obs_pts = load_realtime_streamflow_series(
        username=username, catchment_id=primary_id
    )
    if obs_pts:
        return (
            pd.Timestamp(obs_pts[-1]['date']).floor('D').tz_localize(None)
            + pd.Timedelta(days=1),
            'observed_streamflow',
        )
    return pd.Timestamp.now('UTC').floor('D').tz_localize(None), 'today'


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

    t0, issue_date_source = _resolve_status_issue_date(
        username=username,
        primary_id=primary_id,
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

    observed_streamflow = load_realtime_streamflow_series(
        username=username,
        catchment_id=primary_id,
        start_date=(
            t0 - pd.Timedelta(days=int(lookback_days))
            if has_realtime_data
            else None
        ),
        end_date=(
            t0
            + pd.Timedelta(
                days=_DEFAULT_LEAD_TIME_DAYS
                + _OBSERVED_STREAMFLOW_FORECAST_PAD_DAYS
            )
            if has_realtime_data
            else None
        ),
    )
    has_observed_streamflow = any(
        pt['discharge_cms'] is not None for pt in observed_streamflow
    )
    return_periods = load_basin_return_periods(
        username=username, catchment_id=primary_id
    )
    fetch_progress = get_realtime_fetch_progress(
        username=username,
        catchment_id=primary_id,
        catchment_ids=eval_ids,
    )
    run_progress = get_realtime_run_progress(
        username=username,
        catchment_id=primary_id,
        catchment_ids=eval_ids,
    )
    cached_forecasts_by_id: dict[str, dict[str, Any]] = {}
    for cid in catchments_status:
        cf = load_cached_daily_forecast(
            username=username, catchment_id=cid, model_id=model_id
        )
        if cf is not None:
            cached_forecasts_by_id[cid] = cf
    cached_forecast = cached_forecasts_by_id.get(primary_id)

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
        'available_models': list_forecast_models(username),
        'issue_date': issue_date_str,
        'issue_date_source': issue_date_source,
        'expected_state_date': expected_state_date,
        'saved_state': state_info,
        'is_state_from_previous_day': is_state_from_previous_day,
        'can_run_hotstart': can_run_hotstart,
        'can_run_coldstart': can_run_coldstart,
        'has_realtime_data': has_realtime_data,
        'has_observed_streamflow': has_observed_streamflow,
        'observed_streamflow': observed_streamflow,
        'return_periods': return_periods,
        'hotstart_status_message': hotstart_msg,
        'coldstart_status_message': coldstart_msg,
        'nowcast_dates': streams['nowcast_dates'],
        'nowcast_products': streams['nowcast_products'],
        'forecast_products': streams['forecast_products'],
        'input_products': streams['input_products'],
        'fetch_progress': fetch_progress,
        'run_progress': run_progress,
        'products_disk_status': fetch_progress.get('products_disk_status', {}),
        'cached_forecast': cached_forecast,
        'cached_forecasts_by_id': cached_forecasts_by_id,
        'forecast_cache_ttl_seconds': _DAILY_FORECAST_TTL_SECONDS,
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


def _invalidate_zarr_caches() -> None:
    """Clear in-memory Zarr metadata and window caches after store writes."""
    with _ZARR_CACHE_LOCK:
        _ZARR_META_CACHE.clear()
        _ZARR_WINDOW_CACHE.clear()
    with _DA_SESSION_LOCK:
        _DA_SESSIONS_BY_KEY.clear()


def _normalize_user_key(username: str | None) -> str:
    """Return canonical lowercase profile username key."""
    pm = profile_manager.get_profile_manager()
    raw = (username or pm.active_username or 'guest').strip().lower()
    return raw or 'guest'


def _inspect_products_disk_readiness(
    username: str | None,
    catchment_id: str | None = None,
    catchment_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Inspect per-product Zarr store existence and basin coverage on disk."""
    stores = _discover_zarr_stores(username)
    eval_ids = _normalize_catchment_id_list(catchment_id, catchment_ids)
    all_prods = list(_ALL_REALTIME_PRODUCTS)
    for k in stores:
        if k not in all_prods:
            all_prods.append(k)

    metas_by_prod: dict[str, dict[str, Any]] = {}
    for prod_name in all_prods:
        store_path = stores.get(prod_name)
        if store_path is not None and _is_valid_zarr_store(store_path):
            metas_by_prod[prod_name] = _get_cached_store_meta(store_path)

    all_last_dates = [
        str(m['last_date'])
        for m in metas_by_prod.values()
        if m.get('last_date')
    ]
    ref_last_date = max(all_last_dates) if all_last_dates else None

    readiness: dict[str, dict[str, Any]] = {}
    for prod_name in all_prods:
        store_path = stores.get(prod_name)
        meta = metas_by_prod.get(prod_name)
        valid_store = meta is not None and store_path is not None
        has_basins = False
        is_current = True
        basins_count = 0
        last_date = meta.get('last_date') if meta is not None else None
        if valid_store and meta is not None:
            basins_count = len(meta['basins'])
            if eval_ids:
                has_basins = all(
                    cid in meta['basins'] or cid.lower() in meta['lower_map']
                    for cid in eval_ids
                )
            else:
                has_basins = basins_count > 0
            if ref_last_date and last_date:
                if meta.get('has_lead_time'):
                    is_current = str(last_date) >= str(ref_last_date)
                else:
                    delta_days = int(
                        (
                            pd.Timestamp(ref_last_date)
                            - pd.Timestamp(last_date)
                        ).days
                    )
                    is_current = delta_days <= _MAX_NOWCAST_LAG_DAYS
        ready_for_basin = bool(has_basins and is_current)
        stale_on_disk = bool(has_basins and not is_current)
        readiness[prod_name] = {
            'product': prod_name,
            'store_exists': bool(valid_store),
            'ready_on_disk': ready_for_basin,
            'stale_on_disk': stale_on_disk,
            'last_date': last_date,
            'basins_count': basins_count,
            'zarr_path': str(store_path) if valid_store else None,
        }
    return readiness


class _TrackedTqdm(_ORIG_TQDM_CLASS):  # type: ignore[misc]
    """Thread-local progress-reporting subclass of ``tqdm.std.tqdm``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._oh_handler = getattr(
            _FETCH_THREAD_LOCAL, 'progress_handler', None
        )
        self._oh_closed = False
        super().__init__(*args, **kwargs)
        if self._oh_handler is not None:
            self._oh_emit(int(self.n or 0), event='bar_init')

    def _oh_emit(self, current_val: int, *, event: str = 'bar_step') -> None:
        from frontend.jobs import check_current_job_cancelled  # noqa: PLC0415

        check_current_job_cancelled()
        handler = getattr(self, '_oh_handler', None)
        if handler is None:
            return
        total_val = int(self.total) if getattr(self, 'total', None) else 0
        start_t = float(getattr(self, 'start_t', time.time()))
        elapsed = max(0.0, time.time() - start_t)
        rate = (
            current_val / elapsed
            if elapsed > 0.05 and current_val > 0  # noqa: PLR2004
            else None
        )
        if rate and rate > 0 and total_val > current_val:
            eta: float | None = max(0.0, (total_val - current_val) / rate)
        elif total_val > 0 and current_val >= total_val:
            eta = 0.0
        else:
            eta = None
        handler(
            {
                'event': event,
                'source': 'tqdm',
                'current': int(current_val),
                'total': total_val,
                'unit': str(getattr(self, 'unit', 'day') or 'day'),
                'desc': str(getattr(self, 'desc', '') or ''),
                'elapsed_seconds': round(elapsed, 2),
                'rate': round(rate, 3) if rate is not None else None,
                'eta_seconds': round(eta, 1) if eta is not None else None,
                'bar_id': id(self),
            }
        )

    def __iter__(self) -> Iterator[Any]:
        for idx, obj in enumerate(super().__iter__(), 1):
            yield obj
            self._oh_emit(max(int(self.n or 0), idx), event='bar_step')

    def update(self, n: float = 1) -> bool | None:
        res = super().update(n)
        self._oh_emit(int(self.n or 0), event='bar_step')
        return res

    def close(self) -> None:
        if getattr(self, '_oh_handler', None) is not None and not getattr(
            self, '_oh_closed', False
        ):
            self._oh_closed = True
            self._oh_emit(int(self.n or 0), event='bar_close')
        super().close()


def _ensure_tqdm_progress_hook() -> None:
    """Install ``_TrackedTqdm`` on ``tqdm.tqdm`` once in a thread-safe way."""
    global _TQDM_PATCH_INSTALLED  # noqa: PLW0603
    with _TQDM_PATCH_LOCK:
        if not _TQDM_PATCH_INSTALLED:
            tqdm_mod.tqdm = _TrackedTqdm
            tqdm_mod.std.tqdm = _TrackedTqdm
            _TQDM_PATCH_INSTALLED = True


class _DaskFetchProgressCallback(DaskCallback):
    """Dask task callback that forwards chunk completion to ``handler``."""

    def __init__(self, handler: Callable[[dict[str, Any]], None]) -> None:
        super().__init__()
        self._handler = handler
        self._start_t = time.time()

    def _start_state(
        self,
        dsk: dict[Any, Any],
        state: dict[str, Any],
    ) -> None:
        del dsk
        self._start_t = time.time()
        total = (
            len(state.get('ready', ()))
            + len(state.get('waiting', ()))
            + len(state.get('running', ()))
            + len(state.get('finished', ()))
        )
        if total > 0:
            self._handler(
                {
                    'event': 'dask_start',
                    'source': 'dask',
                    'current': 0,
                    'total': total,
                    'unit': 'chunks',
                }
            )

    def _posttask(
        self,
        key: object,
        result: object,
        dsk: dict[Any, Any],
        state: dict[str, Any],
        worker_id: object,
    ) -> None:
        del key, result, dsk, worker_id
        done = len(state.get('finished', ()))
        total = (
            done
            + len(state.get('ready', ()))
            + len(state.get('waiting', ()))
            + len(state.get('running', ()))
        )
        elapsed = max(0.0, time.time() - self._start_t)
        rate = (
            done / elapsed
            if elapsed > 0.05 and done > 0  # noqa: PLR2004
            else None
        )
        eta = (
            max(0.0, (total - done) / rate)
            if rate and rate > 0 and total > done
            else None
        )
        if total > 0:
            self._handler(
                {
                    'event': 'dask_step',
                    'source': 'dask',
                    'current': done,
                    'total': total,
                    'unit': 'chunks',
                    'elapsed_seconds': round(elapsed, 2),
                    'rate': round(rate, 3) if rate is not None else None,
                    'eta_seconds': round(eta, 1) if eta is not None else None,
                }
            )


def _init_user_fetch_progress(
    user_key: str,
    target_ids: list[str],
    norm_mode: str,
    requested_products: list[str],
    spinup_days: int,
) -> dict[str, Any]:
    """Initialize and register the in-memory fetch progress state for a user."""
    now = time.time()
    est_days = int(spinup_days) + 1 if norm_mode == 'coldstart' else 7
    products_map: dict[str, dict[str, Any]] = {}
    for idx, prod_name in enumerate(requested_products):
        products_map[prod_name] = {
            'product': prod_name,
            'index': idx,
            'status': 'queued',
            'stage': 'queued',
            'percent': 0.0,
            'current': 0,
            'total': est_days,
            'unit': 'day',
            'desc': 'Queued',
            'window': None,
            'started_at': None,
            'elapsed_seconds': 0.0,
            'rate': None,
            'eta_seconds': None,
            'ready_on_disk': False,
            '_window_days': est_days,
            '_cum_offset': 0,
            '_bar_id': None,
            '_bar_last': 0,
            '_has_tqdm': False,
        }
    state: dict[str, Any] = {
        'active': True,
        'status': 'running',
        'username': user_key,
        'mode': norm_mode,
        'catchment_id': target_ids[0] if target_ids else None,
        'catchment_ids': list(target_ids),
        'reference_date': None,
        'started_at': now,
        'updated_at': now,
        'elapsed_seconds': 0.0,
        'current_product': (
            requested_products[0] if requested_products else None
        ),
        'completed_products_count': 0,
        'completed_products': [],
        'total_products_count': len(requested_products),
        'overall_percent': 0.0,
        'message': (
            f'Preparing to fetch {", ".join(requested_products)}...'
            if requested_products
            else 'Preparing real-time fetch...'
        ),
        'error': None,
        'products_order': list(requested_products),
        'products': products_map,
    }
    with _FETCH_PROGRESS_LOCK:
        _FETCH_PROGRESS_BY_USER[user_key] = state
    return state


def _recompute_overall_fetch_progress(state: dict[str, Any]) -> None:
    """Recompute aggregate progress fields on ``state`` in-place."""
    now = time.time()
    state['updated_at'] = now
    state['elapsed_seconds'] = round(
        max(0.0, now - float(state.get('started_at') or now)), 1
    )
    order: list[str] = state.get('products_order', [])
    prods: dict[str, dict[str, Any]] = state.get('products', {})
    completed = [
        p for p in order if prods.get(p, {}).get('status') == 'completed'
    ]
    state['completed_products'] = completed
    state['completed_products_count'] = len(completed)
    state['total_products_count'] = len(order)
    if order:
        total_pct = sum(
            float(prods.get(p, {}).get('percent') or 0.0) for p in order
        )
        state['overall_percent'] = round(
            min(100.0, max(0.0, total_pct / len(order))), 1
        )
    else:
        state['overall_percent'] = 0.0


def _apply_product_lifecycle_event(
    state: dict[str, Any],
    ev: dict[str, Any],
    event_type: str,
) -> None:
    """Update ``state`` for ``init`` and ``product_*`` lifecycle events."""
    prods = state['products']
    if event_type == 'init':
        state['reference_date'] = ev.get('reference_date')
        return
    prod_name = str(ev.get('product') or state.get('current_product') or '')
    p_entry = prods.get(prod_name)
    if p_entry is None:
        return
    if event_type == 'product_start':
        state['current_product'] = prod_name
        win_days = max(1, int(ev.get('window_days') or p_entry['total'] or 1))
        p_entry.update(
            {
                'status': 'downloading',
                'stage': 'downloading',
                'percent': 0.0,
                'current': 0,
                'total': win_days,
                'window': [ev.get('start_date'), ev.get('end_date')],
                'started_at': time.time(),
                'desc': (
                    f'{prod_name} [{ev.get("start_date")} .. '
                    f'{ev.get("end_date")}]'
                ),
                '_window_days': win_days,
                '_cum_offset': 0,
                '_bar_id': None,
                '_bar_last': 0,
                '_has_tqdm': False,
            }
        )
        state['message'] = f'Downloading {prod_name} (0 / {win_days} days · 0%)'
    elif event_type == 'product_writing':
        p_entry['status'] = 'writing_zarr'
        p_entry['stage'] = 'writing_zarr'
        p_entry['percent'] = max(float(p_entry.get('percent') or 0.0), 99.0)
        p_entry['current'] = int(p_entry.get('total') or p_entry['current'])
        p_entry['eta_seconds'] = 0.0
        state['message'] = f'Writing {prod_name} Zarr store to disk...'
    elif event_type == 'product_complete':
        started = float(p_entry.get('started_at') or time.time())
        p_entry.update(
            {
                'status': 'completed',
                'stage': 'completed',
                'percent': 100.0,
                'current': int(p_entry.get('total') or 1),
                'eta_seconds': 0.0,
                'elapsed_seconds': round(max(0.0, time.time() - started), 1),
                'ready_on_disk': True,
            }
        )
        _invalidate_zarr_caches()
        state['message'] = f'Completed {prod_name} Zarr store.'


_TQDM_REALTIME_PRODUCTS = frozenset({'HRES', 'IMERG', 'CPC'})


def _apply_step_progress_event(
    state: dict[str, Any],
    ev: dict[str, Any],
    event_type: str,
) -> None:
    """Update active product progress for ``bar_*`` and ``dask_*`` events."""
    active_prod = str(state.get('current_product') or '')
    p_entry = state['products'].get(active_prod)
    if p_entry is None:
        return
    started = float(p_entry.get('started_at') or time.time())
    p_entry['elapsed_seconds'] = round(max(0.0, time.time() - started), 1)

    if event_type in ('bar_init', 'bar_step', 'bar_close'):
        p_entry['_has_tqdm'] = True
        bar_id = ev.get('bar_id')
        bar_cur = max(0, int(ev.get('current') or 0))
        bar_tot = max(0, int(ev.get('total') or 0))
        win_days = max(1, int(p_entry.get('_window_days') or bar_tot or 1))
        if p_entry['_bar_id'] is not None and bar_id != p_entry['_bar_id']:
            p_entry['_cum_offset'] += int(p_entry.get('_bar_last') or 0)
            p_entry['_bar_last'] = 0
        p_entry['_bar_id'] = bar_id
        p_entry['_bar_last'] = max(int(p_entry.get('_bar_last') or 0), bar_cur)
        eff_cur = int(p_entry['_cum_offset']) + bar_cur
        eff_tot = max(win_days, int(p_entry['_cum_offset']) + bar_tot, eff_cur)
        pct = round(min(99.5, (100.0 * eff_cur) / max(1, eff_tot)), 1)
        unit = str(ev.get('unit') or 'day')
        p_entry.update(
            {
                'status': 'downloading',
                'stage': 'downloading',
                'current': eff_cur,
                'total': eff_tot,
                'unit': unit,
                'percent': pct,
                'desc': str(ev.get('desc') or p_entry.get('desc') or ''),
                'rate': ev.get('rate'),
                'eta_seconds': ev.get('eta_seconds'),
            }
        )
        state['message'] = (
            f'Downloading {active_prod}: {eff_cur} / {eff_tot} {unit}s '
            f'({p_entry["percent"]:.1f}%)'
        )
    elif (
        event_type in ('dask_start', 'dask_step')
        and not p_entry.get('_has_tqdm')
        and active_prod not in _TQDM_REALTIME_PRODUCTS
    ):
        d_cur = max(0, int(ev.get('current') or 0))
        d_tot = max(1, int(ev.get('total') or 1))
        pct = round(min(99.5, (100.0 * d_cur) / d_tot), 1)
        p_entry.update(
            {
                'status': 'downloading',
                'stage': 'downloading',
                'current': d_cur,
                'total': d_tot,
                'unit': 'chunks',
                'percent': pct,
                'rate': ev.get('rate'),
                'eta_seconds': ev.get('eta_seconds'),
            }
        )
        state['message'] = (
            f'Downloading {active_prod}: {d_cur} / {d_tot} chunks '
            f'({p_entry["percent"]:.1f}%)'
        )


def _handle_fetch_progress_event(user_key: str, ev: dict[str, Any]) -> None:
    """Process a single progress event from ``RealtimeForcingFetcher``."""
    event_type = str(ev.get('event') or '')
    with _FETCH_PROGRESS_LOCK:
        state = _FETCH_PROGRESS_BY_USER.get(user_key)
        if state is None:
            return
        if event_type in (
            'init',
            'product_start',
            'product_writing',
            'product_complete',
        ):
            _apply_product_lifecycle_event(state, ev, event_type)
        elif event_type in (
            'bar_init',
            'bar_step',
            'bar_close',
            'dask_start',
            'dask_step',
        ):
            _apply_step_progress_event(state, ev, event_type)
        elif event_type == 'complete':
            state['active'] = False
            state['status'] = 'completed'
            state['current_product'] = None
            for p_entry in state['products'].values():
                if p_entry.get('status') != 'error':
                    p_entry['status'] = 'completed'
                    p_entry['stage'] = 'completed'
                    p_entry['percent'] = 100.0
                    p_entry['ready_on_disk'] = True
            _invalidate_zarr_caches()
            state['message'] = (
                f'Completed fetching {len(state.get("products_order", []))} '
                'product(s).'
            )
        _recompute_overall_fetch_progress(state)


def get_realtime_fetch_progress(
    username: str | None,
    catchment_id: str | None = None,
    catchment_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return live per-product download progress and disk readiness."""
    user_key = _normalize_user_key(username)
    disk_status = _inspect_products_disk_readiness(
        username=user_key,
        catchment_id=catchment_id,
        catchment_ids=catchment_ids,
    )
    with _FETCH_PROGRESS_LOCK:
        raw_state = _FETCH_PROGRESS_BY_USER.get(user_key)
        snapshot = copy.deepcopy(raw_state) if raw_state is not None else None

    if snapshot is None:
        ready_prods = [
            p for p, info in disk_status.items() if info.get('ready_on_disk')
        ]
        return {
            'status': 'idle',
            'active': False,
            'username': user_key,
            'mode': None,
            'catchment_id': catchment_id,
            'catchment_ids': _normalize_catchment_id_list(
                catchment_id, catchment_ids
            ),
            'reference_date': None,
            'started_at': None,
            'updated_at': None,
            'elapsed_seconds': 0.0,
            'current_product': None,
            'completed_products_count': len(ready_prods),
            'completed_products': ready_prods,
            'total_products_count': 0,
            'overall_percent': 0.0,
            'message': None,
            'error': None,
            'products_order': [],
            'products': {},
            'products_disk_status': disk_status,
        }

    if snapshot.get('active') and snapshot.get('started_at'):
        snapshot['elapsed_seconds'] = round(
            max(0.0, time.time() - float(snapshot['started_at'])), 1
        )
    cleaned_products: dict[str, dict[str, Any]] = {}
    for prod_name, p_info in snapshot.get('products', {}).items():
        pub = {k: v for k, v in p_info.items() if not str(k).startswith('_')}
        disk_ready = bool(
            disk_status.get(prod_name, {}).get('ready_on_disk', False)
        )
        pub['ready_on_disk'] = bool(pub.get('ready_on_disk') or disk_ready)
        cleaned_products[prod_name] = pub
    snapshot['products'] = cleaned_products
    snapshot['products_disk_status'] = disk_status
    return snapshot


def cancel_realtime_fetch(
    username: str | None = None,
) -> dict[str, Any]:
    """Request immediate cancellation of any active real-time forcing fetch."""
    user_key = _normalize_user_key(username)
    with _FETCH_PROGRESS_LOCK:
        cancel_ev = _FETCH_CANCEL_EVENTS.get(user_key)
        if cancel_ev is None:
            cancel_ev = threading.Event()
            _FETCH_CANCEL_EVENTS[user_key] = cancel_ev
        cancel_ev.set()

        state = _FETCH_PROGRESS_BY_USER.get(user_key)
        was_active = bool(state and state.get('active'))
        if state is None:
            state = {
                'status': 'cancelled',
                'active': False,
                'cancel_requested': True,
                'username': user_key,
                'mode': None,
                'catchment_id': None,
                'catchment_ids': [],
                'reference_date': None,
                'started_at': None,
                'updated_at': time.time(),
                'elapsed_seconds': 0.0,
                'current_product': None,
                'completed_products_count': 0,
                'completed_products': [],
                'total_products_count': 0,
                'overall_percent': 0.0,
                'message': 'Download stopped by user.',
                'error': None,
                'products_order': [],
                'products': {},
            }
            _FETCH_PROGRESS_BY_USER[user_key] = state
        else:
            state['active'] = False
            state['status'] = 'cancelled'
            state['cancel_requested'] = True
            state['current_product'] = None
            state['message'] = 'Download stopped by user.'
            for p_entry in state.get('products', {}).values():
                if p_entry.get('status') in (
                    'downloading',
                    'writing_zarr',
                    'queued',
                ):
                    p_entry['status'] = 'cancelled'
                    p_entry['stage'] = 'cancelled'
                    p_entry['eta_seconds'] = 0.0
            _recompute_overall_fetch_progress(state)

    prog_snapshot = get_realtime_fetch_progress(username=user_key)
    return {
        'status': 'cancelled',
        'cancelled': True,
        'was_active': was_active,
        'username': user_key,
        'message': 'Download stopped by user.',
        'progress': prog_snapshot,
    }


def _record_completed_fetch_summary(
    user_key: str,
    target_ids: Sequence[str],
    summary: mm_realtime.RealtimeFetchResult,
) -> None:
    """Attach completed ``fetch_summary`` metadata to user progress state."""
    ref_ts = pd.Timestamp(summary.reference_date).floor('D').tz_localize(None)
    with _FETCH_PROGRESS_LOCK:
        done_state = _FETCH_PROGRESS_BY_USER.get(user_key)
        if done_state is not None:
            done_state['reference_date'] = _format_date_str(ref_ts)
            done_state['fetch_summary'] = {
                'mode': summary.mode,
                'reference_date': _format_date_str(ref_ts),
                'catchment_ids': list(target_ids),
                'basin_count': len(target_ids),
                'elapsed_seconds': round(float(summary.elapsed_seconds), 2),
                'products_written': {
                    str(k): str(v) for k, v in summary.stores.items()
                },
                'windows': {
                    str(k): [_format_date_str(w[0]), _format_date_str(w[1])]
                    for k, w in summary.product_windows.items()
                },
            }


def _mark_user_fetch_cancelled(user_key: str) -> None:
    """Update user fetch progress state when a download is cancelled."""
    with _FETCH_PROGRESS_LOCK:
        c_state = _FETCH_PROGRESS_BY_USER.get(user_key)
        if c_state is not None:
            c_state['active'] = False
            c_state['status'] = 'cancelled'
            c_state['message'] = 'Download stopped by user.'
            for p_entry in c_state.get('products', {}).values():
                if p_entry.get('status') in (
                    'downloading',
                    'writing_zarr',
                    'queued',
                ):
                    p_entry['status'] = 'cancelled'
                    p_entry['stage'] = 'cancelled'
                    p_entry['eta_seconds'] = 0.0


def _execute_tracked_multimet_fetch(  # noqa: PLR0913
    user_key: str,
    target_ids: list[str],
    catchment_geojson_path: Path,
    output_dir: Path,
    *,
    norm_mode: str,
    eff_lookback: int,
    norm_ref_date: str | None,
    products: list[str] | None,
    overwrite: bool,
) -> mm_realtime.RealtimeFetchResult:
    """Run ``fetch_realtime_multimet`` with live per-product progress hooks."""
    resolved_enums = _resolve_product_enums(products)
    requested_prod_names = (
        [p.value for p in resolved_enums]
        if resolved_enums
        else list(mm_realtime.DEFAULT_REALTIME_PRODUCTS)
    )
    cancel_ev = threading.Event()
    with _FETCH_PROGRESS_LOCK:
        _FETCH_CANCEL_EVENTS[user_key] = cancel_ev
    _init_user_fetch_progress(
        user_key=user_key,
        target_ids=target_ids,
        norm_mode=norm_mode,
        requested_products=requested_prod_names,
        spinup_days=eff_lookback,
    )
    _ensure_tqdm_progress_hook()

    def _on_progress(ev: dict[str, Any]) -> None:
        if cancel_ev.is_set():
            raise RealtimeFetchCancelledError('Download stopped by user.')
        _handle_fetch_progress_event(user_key, ev)

    fetch_kwargs: dict[str, Any] = {
        'basins': catchment_geojson_path,
        'output_dir': output_dir,
        'mode': norm_mode,
        'lookback_days': eff_lookback,
        'reference_date': norm_ref_date,
        'id_column': 'catchment_id',
        'products': resolved_enums,
        'overwrite': bool(overwrite),
    }
    sig_params = inspect.signature(
        mm_realtime.fetch_realtime_multimet
    ).parameters
    if 'progress_callback' in sig_params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
    ):
        fetch_kwargs['progress_callback'] = _on_progress

    _FETCH_THREAD_LOCAL.progress_handler = _on_progress
    try:
        with _DaskFetchProgressCallback(_on_progress):
            summary = mm_realtime.fetch_realtime_multimet(**fetch_kwargs)
        _record_completed_fetch_summary(user_key, target_ids, summary)
        _on_progress({'event': 'complete'})
    except RealtimeFetchCancelledError:
        _mark_user_fetch_cancelled(user_key)
        raise
    except Exception as exc:
        with _FETCH_PROGRESS_LOCK:
            err_state = _FETCH_PROGRESS_BY_USER.get(user_key)
            if err_state is not None:
                err_state['active'] = False
                err_state['status'] = 'error'
                err_state['error'] = str(exc)
                err_state['message'] = f'Fetch failed: {exc}'
                cur_p = err_state.get('current_product')
                if cur_p and cur_p in err_state.get('products', {}):
                    err_state['products'][cur_p]['status'] = 'error'
                    err_state['products'][cur_p]['stage'] = 'error'
        raise
    else:
        return summary
    finally:
        _FETCH_THREAD_LOCAL.progress_handler = None
        _invalidate_zarr_caches()


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

    user_key = _normalize_user_key(username)
    eff_lookback = (
        int(spinup_days) if norm_mode == 'coldstart' else hotstart_lookback
    )
    try:
        summary = _execute_tracked_multimet_fetch(
            user_key=user_key,
            target_ids=target_ids,
            catchment_geojson_path=catchment_geojson_path,
            output_dir=output_dir,
            norm_mode=norm_mode,
            eff_lookback=eff_lookback,
            norm_ref_date=norm_ref_date,
            products=products,
            overwrite=bool(overwrite),
        )
    except RealtimeFetchCancelledError:
        status_payload = get_realtime_status(
            username=username,
            catchment_id=primary_id,
            requested_issue_date=norm_ref_date,
            model_id=model_id,
            catchment_ids=target_ids,
        )
        status_payload['status'] = 'cancelled'
        status_payload['cancelled'] = True
        status_payload['message'] = 'Download stopped by user.'
        return status_payload

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


def _load_forecast_model_bundle(
    run_dir: Path | None = None,
    *,
    model_id: str | None = _DEFAULT_MODEL_ID,
    username: str | None = None,
) -> dict[str, Any]:
    """Load the trained ``MeanEmbeddingForecastLSTM`` and ``Scaler`` on CPU.

    Args:
        run_dir: Model run directory containing ``config.yml``, a
            ``model_epoch*.pt`` checkpoint, and ``scaler.zarr``. Defaults to
            resolving ``model_id`` via ``resolve_forecast_model_run_dir``.
        model_id: Identifier of the pretrained or user fine-tuned model.
        username: Active profile username when resolving ``user_*`` models.

    Raises:
        FileNotFoundError: If ``run_dir`` has no ``config.yml`` or no
            ``model_epoch*.pt`` checkpoint.
    """
    target_dir = resolve_forecast_model_run_dir(
        model_id=model_id,
        username=username,
        run_dir=run_dir,
    )
    cache_key = str(target_dir.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    config_path = target_dir / 'config.yml'
    if not config_path.is_file():
        raise FileNotFoundError(
            f'Model run directory {target_dir} has no config.yml.'
        )

    ensure_flood_forecasting_on_sys_path()
    cfg = Config(config_path)
    cfg.update_config(
        {
            'device': 'cpu',
            'compile': False,
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
    load_model_weights(model, ckpts[-1], torch.device('cpu'))
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

    csv_row = _read_profile_csv_attributes(
        [
            pm.get_caravan_attributes_dir(username),
            pm.get_attributes_dir(username),
        ],
        catchment_id,
    )
    if row is None:
        row = csv_row
    elif csv_row is not None:
        merged = row.to_dict()
        for k, v in csv_row.items():
            if pd.notna(v):
                merged[k] = v
        row = pd.Series(merged)

    if row is None:
        raise KeyError(
            f"Static attributes for catchment '{catchment_id}' were not found "
            'in the active user profile attribute stores.'
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
        ds_obj = _open_cached_zarr_window(s_path, hindcast_dates, t0)
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
    window_start = (
        hindcast_series[0]['date']
        if hindcast_series
        else _format_date_str(t0 - pd.Timedelta(days=_HINDCAST_TAIL_DAYS))
    )
    window_end = (
        forecast_series[-1]['date']
        if forecast_series
        else _format_date_str(t0 + pd.Timedelta(days=lead_time - 1))
    )
    observed_streamflow = load_realtime_streamflow_series(
        username=username,
        catchment_id=catchment_id,
        start_date=window_start,
        end_date=window_end,
    )
    return_periods = load_basin_return_periods(
        username=username, catchment_id=catchment_id
    )

    result_payload = {
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
        'has_observed_streamflow': any(
            pt['discharge_cms'] is not None for pt in observed_streamflow
        ),
        'observed_streamflow': observed_streamflow,
        'return_periods': return_periods,
        'saved_state': saved_state,
    }
    if not overrides.is_what_if:
        result_payload = save_cached_daily_forecast(
            username=username,
            catchment_id=catchment_id,
            result_payload=result_payload,
            model_id=model_id,
        )
    return result_payload


def _init_user_run_progress(  # noqa: PLR0913
    user_key: str,
    target_ids: list[str],
    norm_mode: str,
    model_id: str,
    issue_date_str: str,
    *,
    is_what_if: bool = False,
) -> dict[str, Any]:
    """Initialize per-basin model inference progress for a user."""
    now = time.time()
    basins_map: dict[str, dict[str, Any]] = {}
    for idx, cid in enumerate(target_ids):
        basins_map[cid] = {
            'catchment_id': cid,
            'index': idx,
            'status': 'queued',
            'percent': 0.0,
            'started_at': None,
            'elapsed_seconds': 0.0,
            'peak_discharge_cms': None,
            'peak_date': None,
        }
    mode_label = 'Hot Start' if norm_mode == 'hotstart' else 'Cold Start'
    first_basin = target_ids[0] if target_ids else None
    state: dict[str, Any] = {
        'active': True,
        'status': 'running',
        'username': user_key,
        'mode': norm_mode,
        'is_what_if': bool(is_what_if),
        'model_id': model_id,
        'issue_date': issue_date_str,
        'catchment_id': first_basin,
        'catchment_ids': list(target_ids),
        'basins_order': list(target_ids),
        'started_at': now,
        'updated_at': now,
        'elapsed_seconds': 0.0,
        'current_basin': first_basin,
        'current_basin_index': 0,
        'completed_basins_count': 0,
        'completed_basins': [],
        'total_basins_count': len(target_ids),
        'overall_percent': 0.0,
        'message': (
            f'Running {mode_label} inference (1/{len(target_ids)}): '
            f'{first_basin}...'
            if first_basin
            else f'Preparing {mode_label} inference...'
        ),
        'error': None,
        'basins': basins_map,
    }
    with _RUN_PROGRESS_LOCK:
        _RUN_PROGRESS_BY_USER[user_key] = state
    return state


def _update_user_run_progress_step(  # noqa: C901, PLR0913, PLR0915
    user_key: str,
    *,
    event: str,
    basin_id: str | None = None,
    basin_index: int = 0,
    basin_elapsed: float = 0.0,
    peak_discharge_cms: float | None = None,
    peak_date: str | None = None,
    error: str | None = None,
) -> None:
    """Update per-basin model inference progress state in a thread-safe way."""
    now = time.time()
    with _RUN_PROGRESS_LOCK:
        state = _RUN_PROGRESS_BY_USER.get(user_key)
        if state is None:
            return
        state['updated_at'] = now
        state['elapsed_seconds'] = round(
            max(0.0, now - float(state.get('started_at') or now)), 2
        )
        total = max(1, int(state.get('total_basins_count') or 1))
        mode_label = (
            'Hot Start' if state.get('mode') == 'hotstart' else 'Cold Start'
        )
        basins_map: dict[str, dict[str, Any]] = state.get('basins', {})
        order: list[str] = state.get('basins_order', [])

        if event == 'basin_start' and basin_id:
            state['current_basin'] = basin_id
            state['current_basin_index'] = basin_index
            b_entry = basins_map.get(basin_id)
            if b_entry is not None:
                b_entry['status'] = 'running'
                b_entry['percent'] = 50.0
                b_entry['started_at'] = now
            done_cnt = int(state.get('completed_basins_count') or 0)
            state['overall_percent'] = round(
                min(99.0, 100.0 * (done_cnt + 0.25) / total), 1
            )
            state['message'] = (
                f'Running {mode_label} ({basin_index + 1}/{total}): '
                f'{basin_id}...'
            )
        elif event == 'basin_complete' and basin_id:
            b_entry = basins_map.get(basin_id)
            if b_entry is not None:
                b_entry['status'] = 'completed'
                b_entry['percent'] = 100.0
                b_entry['elapsed_seconds'] = round(basin_elapsed, 2)
                b_entry['peak_discharge_cms'] = peak_discharge_cms
                b_entry['peak_date'] = peak_date
            completed = [
                cid
                for cid in order
                if basins_map.get(cid, {}).get('status') == 'completed'
            ]
            state['completed_basins'] = completed
            state['completed_basins_count'] = len(completed)
            state['overall_percent'] = round(
                min(100.0, 100.0 * len(completed) / total), 1
            )
            state['message'] = (
                f'Completed {len(completed)}/{total} basin(s) '
                f'({basin_id} in {basin_elapsed:.2f}s)'
            )
        elif event == 'complete':
            for cid in order:
                b_entry = basins_map.get(cid)
                if b_entry is not None and b_entry.get('status') != 'completed':
                    b_entry['status'] = 'completed'
                    b_entry['percent'] = 100.0
            state['active'] = False
            state['status'] = 'completed'
            state['current_basin'] = None
            state['completed_basins'] = list(order)
            state['completed_basins_count'] = len(order)
            state['overall_percent'] = 100.0
            state['message'] = (
                f'{mode_label} inference complete for {len(order)} '
                f'basin{"s" if len(order) != 1 else ""} '
                f'({state["elapsed_seconds"]:.2f}s)'
            )
        elif event == 'error':
            state['active'] = False
            state['status'] = 'error'
            state['error'] = error or 'Model execution failed'
            state['message'] = f'Model run failed: {state["error"]}'
            cur_b = state.get('current_basin')
            if cur_b and cur_b in basins_map:
                basins_map[cur_b]['status'] = 'error'


def get_realtime_run_progress(
    username: str | None,
    catchment_id: str | None = None,
    catchment_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return live per-basin model inference progress for a user."""
    user_key = _normalize_user_key(username)
    with _RUN_PROGRESS_LOCK:
        raw_state = _RUN_PROGRESS_BY_USER.get(user_key)
        snapshot = copy.deepcopy(raw_state) if raw_state is not None else None

    if snapshot is None:
        eval_ids = _normalize_catchment_id_list(catchment_id, catchment_ids)
        return {
            'status': 'idle',
            'active': False,
            'username': user_key,
            'mode': None,
            'is_what_if': False,
            'model_id': _DEFAULT_MODEL_ID,
            'issue_date': None,
            'catchment_id': catchment_id,
            'catchment_ids': eval_ids,
            'basins_order': eval_ids,
            'started_at': None,
            'updated_at': None,
            'elapsed_seconds': 0.0,
            'current_basin': None,
            'current_basin_index': 0,
            'completed_basins_count': 0,
            'completed_basins': [],
            'total_basins_count': len(eval_ids),
            'overall_percent': 0.0,
            'message': None,
            'error': None,
            'basins': {},
        }

    if snapshot.get('active') and snapshot.get('started_at'):
        now = time.time()
        snapshot['elapsed_seconds'] = round(
            max(0.0, now - float(snapshot['started_at'])), 2
        )
        cur_b = snapshot.get('current_basin')
        basins_map = snapshot.get('basins', {})
        if cur_b and cur_b in basins_map:
            b_start = basins_map[cur_b].get('started_at')
            if b_start:
                basins_map[cur_b]['elapsed_seconds'] = round(
                    max(0.0, now - float(b_start)), 2
                )
    return snapshot


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
        **_load_forecast_model_bundle(
            run_dir=model_run_dir,
            model_id=model_id,
            username=username,
        ),
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

    user_key = _normalize_user_key(username)
    _init_user_run_progress(
        user_key=user_key,
        target_ids=target_ids,
        norm_mode=norm_mode,
        model_id=model_id,
        issue_date_str=issue_date_str,
        is_what_if=overrides.is_what_if,
    )

    results_by_id: dict[str, dict[str, Any]] = {}
    try:
        for idx, cid in enumerate(target_ids):
            _update_user_run_progress_step(
                user_key,
                event='basin_start',
                basin_id=cid,
                basin_index=idx,
            )
            t_basin_start = time.time()
            single_res = _run_forecast_model_single(
                username=username,
                catchment_id=cid,
                t0=t0,
                bundle=bundle,
                overrides=overrides,
            )
            basin_elapsed = max(0.0, time.time() - t_basin_start)
            single_res['elapsed_seconds'] = round(basin_elapsed, 2)
            results_by_id[cid] = single_res
            _update_user_run_progress_step(
                user_key,
                event='basin_complete',
                basin_id=cid,
                basin_index=idx,
                basin_elapsed=basin_elapsed,
                peak_discharge_cms=single_res.get('peak_discharge_cms'),
                peak_date=single_res.get('peak_date'),
            )
        _update_user_run_progress_step(user_key, event='complete')
    except Exception as exc:
        _update_user_run_progress_step(user_key, event='error', error=str(exc))
        raise

    primary_res = results_by_id[primary_id]
    updated_status = get_realtime_status(
        username=username,
        catchment_id=primary_id,
        requested_issue_date=issue_date_str,
        model_id=model_id,
        catchment_ids=target_ids,
    )
    run_progress = get_realtime_run_progress(
        username=username,
        catchment_id=primary_id,
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
        'run_progress': run_progress,
    }


def inspect_realtime_stores_for_basin(  # noqa: PLR0913
    username: str | None,
    basin_id: str,
    nowcast_lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    *,
    query_upstream_if_empty: bool = False,
    basin_ids: Sequence[str] | None = None,
    requested_issue_date: str | None = None,
    model_id: str = _DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Server endpoint wrapper for ``GET /api/forecast/status``."""
    del query_upstream_if_empty
    return get_realtime_status(
        username=username,
        catchment_id=str(basin_id),
        requested_issue_date=requested_issue_date,
        model_id=model_id or _DEFAULT_MODEL_ID,
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
    model_id: str = _DEFAULT_MODEL_ID,
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
        model_id=model_id or _DEFAULT_MODEL_ID,
        overwrite=overwrite,
    )


def run_hydrological_model_for_basin(  # noqa: PLR0913
    username: str | None,
    basin_id: str | None = None,
    mode: str = 'coldstart',
    model_run_dir: str | Path | None = None,
    requested_issue_date: str | None = None,
    *,
    model_id: str = _DEFAULT_MODEL_ID,
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
        model_id=model_id or _DEFAULT_MODEL_ID,
        model_run_dir=Path(model_run_dir) if model_run_dir else None,
        catchment_ids=basin_ids,
        temp_overrides=temp_overrides,
        product_overrides=product_overrides,
        is_what_if=is_what_if,
    )


_DEFAULT_NUDGING_RHO = 0.65
_DEFAULT_NUDGING_ALPHA = 1.0
_DEFAULT_NUDGING_WINDOW_DAYS = 1
_MAX_NUDGING_WINDOW_DAYS = 14
_MIN_POSITIVE_DISCHARGE_CMS = 1e-6


def compute_nudging_postprocessing(  # noqa: C901, PLR0912, PLR0913, PLR0915
    forecast_series: Sequence[dict[str, Any]],
    hindcast_series: Sequence[dict[str, Any]] | None = None,
    observed_series: Sequence[dict[str, Any]] | None = None,
    *,
    issue_date: str | None = None,
    rho: float = _DEFAULT_NUDGING_RHO,
    alpha: float = _DEFAULT_NUDGING_ALPHA,
    window_days: int = _DEFAULT_NUDGING_WINDOW_DAYS,
) -> dict[str, Any]:
    """Apply autoregressive residual-persistence nudging (Karan DA paper Eq. 6).

    Computes the residual error ``e = Q_obs - Q_base`` at the latest valid
    observation date ``t_anchor <= issue_date`` (optionally averaged over a
    ``window_days`` lookback window ending at ``t_anchor``) and updates
    subsequent discharge steps ``k = (d - t_anchor).days >= 0`` via::

        Q_pp(d) = max(0.0, Q_base(d) + alpha * (rho ** k) * e)

    When ``alpha == 1.0`` and ``window_days == 1``, this matches Equation 6 of
    Karan Ruparell's Data Assimilation paper (``ar1_postprocess``).
    """
    rho_clamped = float(np.clip(float(rho), 0.0, 1.0))
    alpha_clamped = float(np.clip(float(alpha), 0.0, 1.0))
    win_clamped = int(
        np.clip(int(window_days or 1), 1, _MAX_NUDGING_WINDOW_DAYS)
    )

    hc_list = list(hindcast_series or [])
    fc_list = list(forecast_series or [])
    obs_list = list(observed_series or [])

    base_by_date: dict[str, float] = {}
    for pt in hc_list:
        d_str = str(pt.get('date') or '').strip()
        q_val = _clean_float(pt.get('discharge_cms'))
        if d_str and q_val is not None:
            base_by_date[d_str] = q_val
    for pt in fc_list:
        d_str = str(pt.get('date') or '').strip()
        q_val = _clean_float(pt.get('discharge_cms'))
        if d_str and q_val is not None:
            base_by_date[d_str] = q_val

    obs_by_date: dict[str, float] = {}
    for pt in obs_list:
        d_str = str(pt.get('date') or '').strip()
        q_val = _clean_float(pt.get('discharge_cms'))
        if d_str and q_val is not None and q_val >= 0.0:
            obs_by_date[d_str] = q_val

    overlap_dates = sorted(d for d in obs_by_date if d in base_by_date)
    if issue_date:
        pre_issue = [d for d in overlap_dates if d <= str(issue_date).strip()]
        if pre_issue:
            overlap_dates = pre_issue

    if not overlap_dates:
        return {
            'status': 'success',
            'method': 'nudging',
            'applied': False,
            'rho': round(rho_clamped, 4),
            'alpha': round(alpha_clamped, 4),
            'window_days': win_clamped,
            'anchor_date': None,
            'residual_cms': 0.0,
            'obs_at_anchor_cms': None,
            'base_at_anchor_cms': None,
            'half_life_days': None,
            'nudged_series': [],
            'forecast': [dict(pt) for pt in fc_list],
            'peak_discharge_cms': None,
            'peak_date': None,
        }

    anchor_date = overlap_dates[-1]
    anchor_ts = pd.Timestamp(anchor_date)
    win_start_str = _format_date_str(
        anchor_ts - pd.Timedelta(days=win_clamped - 1)
    )
    window_dates = [
        d for d in overlap_dates if win_start_str <= d <= anchor_date
    ] or [anchor_date]

    residuals = [obs_by_date[d] - base_by_date[d] for d in window_dates]
    residual_cms = float(np.mean(residuals))
    obs_at_anchor = obs_by_date[anchor_date]
    base_at_anchor = base_by_date[anchor_date]

    if 0.0 < rho_clamped < 1.0:
        half_life: float | None = round(
            float(-np.log(2.0) / np.log(rho_clamped)), 2
        )
    else:
        half_life = None

    all_ordered_dates = sorted(base_by_date.keys())
    nudged_series: list[dict[str, Any]] = []
    nudged_by_date: dict[str, float] = {}
    delta_by_date: dict[str, float] = {}

    for d_str in all_ordered_dates:
        if d_str < anchor_date:
            continue
        step_k = max(0, int((pd.Timestamp(d_str) - anchor_ts).days))
        weight_k = alpha_clamped * (rho_clamped**step_k)
        delta_q = weight_k * residual_cms
        base_q = base_by_date[d_str]
        nudged_q = max(0.0, base_q + delta_q)
        nudged_by_date[d_str] = round(nudged_q, 3)
        delta_by_date[d_str] = round(delta_q, 3)
        nudged_series.append(
            {
                'date': d_str,
                'step_from_anchor': step_k,
                'discharge_cms': round(nudged_q, 3),
                'baseline_cms': round(base_q, 3),
                'delta_cms': round(delta_q, 3),
                'weight': round(weight_k, 4),
            }
        )

    nudged_forecast: list[dict[str, Any]] = []
    peak_q: float | None = None
    peak_date: str | None = None
    for pt in fc_list:
        item = dict(pt)
        d_str = str(item.get('date') or '').strip()
        base_q = _clean_float(item.get('discharge_cms'))
        if d_str in nudged_by_date and base_q is not None:
            new_q = nudged_by_date[d_str]
            delta_q = delta_by_date[d_str]
            item['discharge_cms'] = new_q
            base_mm = _clean_float(item.get('discharge_mm_day'))
            if base_mm is not None and base_q > _MIN_POSITIVE_DISCHARGE_CMS:
                item['discharge_mm_day'] = round(
                    max(0.0, base_mm * (new_q / base_q)), 4
                )
            for q_key in ('q05_cms', 'q25_cms', 'q75_cms', 'q95_cms'):
                q_orig = _clean_float(item.get(q_key))
                if q_orig is not None:
                    item[q_key] = round(max(0.0, q_orig + delta_q), 3)
        q_cur = _clean_float(item.get('discharge_cms'))
        if q_cur is not None and (peak_q is None or q_cur >= peak_q):
            peak_q = q_cur
            peak_date = d_str
        nudged_forecast.append(item)

    return {
        'status': 'success',
        'method': 'nudging',
        'applied': True,
        'rho': round(rho_clamped, 4),
        'alpha': round(alpha_clamped, 4),
        'window_days': win_clamped,
        'anchor_date': anchor_date,
        'window_dates': window_dates,
        'residual_cms': round(residual_cms, 3),
        'obs_at_anchor_cms': round(obs_at_anchor, 3),
        'base_at_anchor_cms': round(base_at_anchor, 3),
        'half_life_days': half_life,
        'nudged_series': nudged_series,
        'forecast': nudged_forecast,
        'peak_discharge_cms': round(peak_q, 3) if peak_q is not None else None,
        'peak_date': peak_date,
    }


_DEFAULT_VARIATIONAL_WINDOW_DAYS = 14
_MAX_VARIATIONAL_WINDOW_DAYS = 180
_DEFAULT_VARIATIONAL_REG_WEIGHT = 0.5
_DEFAULT_VARIATIONAL_LR = 0.01
_DEFAULT_VARIATIONAL_EPOCHS = 5
_MAX_VARIATIONAL_EPOCHS = 200
_DEFAULT_VARIATIONAL_LOSS = 'MSE'
_DEFAULT_VARIATIONAL_CLIP_NORM = 1.0
_DEFAULT_VARIATIONAL_TOLERANCE: float | None = None
_DA_CLIP_EPS = 1e-6
_ALL_ASSIMILATION_COMPONENTS = (
    'static_embedding',
    'hindcast_embedding',
    'forecast_embedding',
)


def _init_user_da_progress(  # noqa: PLR0913
    user_key: str,
    catchment_id: str,
    model_id: str,
    issue_date_str: str,
    *,
    base_step: int,
    total_steps: int,
    prior_rmse: float | None,
) -> dict[str, Any]:
    """Initialize live per-iteration variational DA progress for user."""
    now = time.time()
    target_step = base_step + total_steps
    state: dict[str, Any] = {
        'active': True,
        'status': 'running',
        'username': user_key,
        'catchment_id': catchment_id,
        'model_id': model_id,
        'issue_date': issue_date_str,
        'started_at': now,
        'updated_at': now,
        'elapsed_seconds': 0.0,
        'base_step': base_step,
        'current_step': base_step,
        'target_step': target_step,
        'steps_completed': 0,
        'total_steps': total_steps,
        'overall_percent': 0.0,
        'total_loss': None,
        'obs_loss': None,
        'reg_loss': None,
        'window_rmse_cms': prior_rmse,
        'prior_window_rmse_cms': prior_rmse,
        'message': (
            f'Starting Variational DA backprop '
            f'(Step {base_step} → {target_step})...'
            if total_steps > 0
            else 'Evaluating Step 0 (Prior)...'
        ),
        'error': None,
    }
    with _DA_PROGRESS_LOCK:
        _DA_PROGRESS_BY_USER[user_key] = state
    return state


def _update_user_da_progress(  # noqa: C901, PLR0913
    user_key: str,
    *,
    event: str,
    current_step: int | None = None,
    steps_completed: int | None = None,
    total_loss: float | None = None,
    obs_loss: float | None = None,
    reg_loss: float | None = None,
    window_rmse_cms: float | None = None,
    error: str | None = None,
) -> None:
    """Update live variational DA progress state in a thread-safe way."""
    now = time.time()
    with _DA_PROGRESS_LOCK:
        state = _DA_PROGRESS_BY_USER.get(user_key)
        if state is None:
            return
        state['updated_at'] = now
        state['elapsed_seconds'] = round(
            max(0.0, now - float(state.get('started_at') or now)), 2
        )
        tot = max(1, int(state.get('total_steps') or 1))
        if current_step is not None:
            state['current_step'] = int(current_step)
        if steps_completed is not None:
            state['steps_completed'] = int(steps_completed)
        if total_loss is not None:
            state['total_loss'] = round(float(total_loss), 6)
        if obs_loss is not None:
            state['obs_loss'] = round(float(obs_loss), 6)
        if reg_loss is not None:
            state['reg_loss'] = round(float(reg_loss), 6)
        if window_rmse_cms is not None:
            state['window_rmse_cms'] = round(float(window_rmse_cms), 3)

        done = int(state.get('steps_completed') or 0)
        cur_s = int(state.get('current_step') or 0)
        tgt_s = int(state.get('target_step') or tot)
        if event == 'step':
            state['overall_percent'] = round(
                min(99.5, 100.0 * done / tot),
                1,
            )
            loss_txt = (
                f' · Loss {state["total_loss"]:.4f}'
                if state.get('total_loss') is not None
                else ''
            )
            rmse_txt = (
                f' · RMSE {state["window_rmse_cms"]:.1f} m³/s'
                if state.get('window_rmse_cms') is not None
                else ''
            )
            state['message'] = (
                f'Step {cur_s}/{tgt_s} ({done}/{tot}){loss_txt}{rmse_txt}'
            )
        elif event == 'complete':
            state['active'] = False
            state['status'] = 'completed'
            state['overall_percent'] = 100.0
            rmse_txt = (
                f' · RMSE {state["window_rmse_cms"]:.1f} m³/s'
                if state.get('window_rmse_cms') is not None
                else ''
            )
            state['message'] = (
                f'Completed Step {cur_s}{rmse_txt} '
                f'({state["elapsed_seconds"]:.2f}s)'
            )
        elif event == 'cancelled':
            state['active'] = False
            state['status'] = 'cancelled'
            state['message'] = f'Stopped at Step {cur_s} by user.'
        elif event == 'error':
            state['active'] = False
            state['status'] = 'error'
            state['error'] = error or 'Variational DA failed'
            state['message'] = f'Variational DA failed: {state["error"]}'


def get_variational_assimilation_progress(
    username: str | None,
    catchment_id: str | None = None,
) -> dict[str, Any]:
    """Return live per-iteration variational DA progress for ``username``."""
    user_key = _normalize_user_key(username)
    with _DA_PROGRESS_LOCK:
        raw_state = _DA_PROGRESS_BY_USER.get(user_key)
        snapshot = copy.deepcopy(raw_state) if raw_state is not None else None
    if snapshot is None:
        return {
            'status': 'idle',
            'active': False,
            'username': user_key,
            'catchment_id': catchment_id,
            'model_id': _DEFAULT_MODEL_ID,
            'issue_date': None,
            'started_at': None,
            'updated_at': None,
            'elapsed_seconds': 0.0,
            'base_step': 0,
            'current_step': 0,
            'target_step': 0,
            'steps_completed': 0,
            'total_steps': 0,
            'overall_percent': 0.0,
            'total_loss': None,
            'obs_loss': None,
            'reg_loss': None,
            'window_rmse_cms': None,
            'prior_window_rmse_cms': None,
            'message': None,
            'error': None,
        }
    if snapshot.get('active') and snapshot.get('started_at'):
        snapshot['elapsed_seconds'] = round(
            max(0.0, time.time() - float(snapshot['started_at'])),
            2,
        )
    return snapshot


def cancel_variational_assimilation(
    username: str | None = None,
) -> dict[str, Any]:
    """Cooperatively stop an active variational DA optimization loop."""
    user_key = _normalize_user_key(username)
    was_active = False
    with _DA_PROGRESS_LOCK:
        ev = _DA_CANCEL_EVENTS.get(user_key)
        if ev is not None:
            ev.set()
        state = _DA_PROGRESS_BY_USER.get(user_key)
        if state is not None and state.get('active'):
            was_active = True
            state['active'] = False
            state['status'] = 'cancelled'
            state['updated_at'] = time.time()
            state['message'] = (
                f'Stopped at Step {state.get("current_step", 0)} by user.'
            )
    return {
        'status': 'cancelled',
        'username': user_key,
        'was_active': was_active,
        'progress': get_variational_assimilation_progress(user_key),
    }


@contextlib.contextmanager
def _freeze_model_for_da(model: torch.nn.Module) -> Iterator[None]:
    """Freeze model weights and disable dropout during latent backprop."""
    requires_grad = {p: p.requires_grad for p in model.parameters()}
    dropout_p = {
        m: m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)
    }
    rnns = {
        m: (m.training, m.dropout)
        for m in model.modules()
        if isinstance(m, torch.nn.RNNBase)
    }
    try:
        for p in requires_grad:
            p.requires_grad_(requires_grad=False)
        for m in dropout_p:
            m.p = 0.0
        for m in rnns:
            m.dropout = 0.0
            m.train(mode=True)
        yield
    finally:
        for p, flag in requires_grad.items():
            p.requires_grad_(flag)
        for m, p in dropout_p.items():
            m.p = p
        for m, (training, dropout) in rnns.items():
            m.dropout = dropout
            m.train(training)


def _clip_da_gradients(
    tensors: Iterable[torch.Tensor],
    max_norm: float,
) -> None:
    """Clip per-sequence gradient norm for latent embedding tensors."""
    grad_tensors = [t for t in tensors if t.grad is not None]
    if not grad_tensors:
        return
    sq_norm = sum(t.grad.flatten(1).pow(2).sum(dim=1) for t in grad_tensors)
    eps = max(_DA_CLIP_EPS, float(torch.finfo(sq_norm.dtype).eps))
    scale = (max_norm / (sq_norm.sqrt() + eps)).clamp(max=1.0)
    for t in grad_tensors:
        t.grad.mul_(scale.view(-1, *([1] * (t.grad.ndim - 1))))


def _build_full_observed_overrides(
    cur_full_embeddings: dict[str, torch.Tensor],
    active_window_tensors: dict[str, torch.Tensor],
    win_start_idx: int,
    win_end_idx: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Assemble full ``[0, seq_len)`` embedding overrides for ``forward()``."""
    overrides: dict[str, torch.Tensor] = {}
    for c_name in _ALL_ASSIMILATION_COMPONENTS:
        if c_name == 'static_embedding':
            overrides[c_name] = active_window_tensors.get(
                c_name, cur_full_embeddings[c_name]
            )
        elif c_name in active_window_tensors:
            base_full = cur_full_embeddings[c_name]
            overrides[c_name] = torch.cat(
                [
                    base_full[:, :win_start_idx],
                    active_window_tensors[c_name],
                    base_full[:, win_end_idx:seq_len],
                ],
                dim=1,
            )
        else:
            overrides[c_name] = cur_full_embeddings[c_name][:, :seq_len]
    return overrides


def _compute_component_shifts(
    prior_embeddings: dict[str, torch.Tensor],
    current_embeddings: dict[str, torch.Tensor],
    win_start_idx: int,
    win_end_idx: int,
) -> dict[str, float]:
    """Compute relative L2 embedding shifts vs Step 0 prior in ``window``."""
    shifts: dict[str, float] = {}
    for c_name in _ALL_ASSIMILATION_COMPONENTS:
        if c_name in prior_embeddings and c_name in current_embeddings:
            p_t = prior_embeddings[c_name].detach().float()
            a_t = current_embeddings[c_name].detach().float()
            if p_t.ndim >= _NDIM_3D:
                p_t = p_t[:, win_start_idx:win_end_idx]
                a_t = a_t[:, win_start_idx:win_end_idx]
            v_mask = torch.isfinite(p_t) & torch.isfinite(a_t)
            if bool(v_mask.any()):
                diff_norm = float(
                    torch.linalg.vector_norm((a_t - p_t)[v_mask]).item()
                )
                base_norm = float(torch.linalg.vector_norm(p_t[v_mask]).item())
                shifts[c_name] = round(diff_norm / max(base_norm, 1e-6), 4)
            else:
                shifts[c_name] = 0.0
    return shifts


def _build_assimilated_series_points(  # noqa: PLR0913, PLR0917
    step_q_cms: np.ndarray,
    prior_q_cms: np.ndarray,
    hindcast_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    win_start_idx: int,
    win_end_idx: int,
    seq_len: int,
    lead_time: int,
) -> list[dict[str, Any]]:
    """Build continuous assimilated series over visible hindcast + forecast."""
    all_dates_ts = list(hindcast_dates) + list(forecast_dates)
    tail_start_idx = max(0, min(win_start_idx, seq_len - _HINDCAST_TAIL_DAYS))
    series: list[dict[str, Any]] = []
    for idx in range(tail_start_idx, seq_len + lead_time):
        d_str = _format_date_str(all_dates_ts[idx])
        a_q = _clean_float(step_q_cms[idx])
        b_q = _clean_float(prior_q_cms[idx])
        if a_q is None or b_q is None:
            continue
        series.append(
            {
                'date': d_str,
                'discharge_cms': round(a_q, 3),
                'baseline_cms': round(b_q, 3),
                'delta_cms': round(a_q - b_q, 3),
                'in_window': bool(win_start_idx <= idx < win_end_idx),
                'is_forecast': bool(idx >= seq_len),
            }
        )
    return series


def _normalize_assimilation_components(  # noqa: C901, PLR0912
    raw_components: Any,  # noqa: ANN401
) -> list[str] | dict[str, dict[str, float]]:
    """Normalize user-specified latent embedding component selection."""
    if raw_components is None:
        return list(_ALL_ASSIMILATION_COMPONENTS)
    if isinstance(raw_components, str):
        key = raw_components.strip().lower().replace('-', '_')
        alias_map: dict[str, list[str]] = {
            'all': list(_ALL_ASSIMILATION_COMPONENTS),
            'embedded_all': list(_ALL_ASSIMILATION_COMPONENTS),
            'dynamic': ['hindcast_embedding', 'forecast_embedding'],
            'embedded_both_dynamic': [
                'hindcast_embedding',
                'forecast_embedding',
            ],
            'static': ['static_embedding'],
            'static_only': ['static_embedding'],
            'static_embedding': ['static_embedding'],
            'hindcast_embedding': ['hindcast_embedding'],
            'forecast_embedding': ['forecast_embedding'],
            'static_hindcast': ['static_embedding', 'hindcast_embedding'],
            'embedded_both': ['static_embedding', 'hindcast_embedding'],
        }
        if key in alias_map:
            return alias_map[key]
        raw_components = [
            c.strip() for c in raw_components.split(',') if c.strip()
        ]

    if isinstance(raw_components, dict):
        cleaned_dict: dict[str, dict[str, float]] = {}
        for name, spec in raw_components.items():
            c_name = str(name).strip()
            if c_name not in _ALL_ASSIMILATION_COMPONENTS:
                raise ValueError(
                    f'Unsupported assimilation component {c_name!r}; '
                    f'supported: {list(_ALL_ASSIMILATION_COMPONENTS)}.'
                )
            sub: dict[str, float] = {}
            if isinstance(spec, dict):
                rw = _clean_float(spec.get('regularization_weight'))
                lr = _clean_float(spec.get('initial_learning_rate'))
                if rw is not None and rw >= 0.0:
                    sub['regularization_weight'] = float(rw)
                if lr is not None and lr > 0.0:
                    sub['initial_learning_rate'] = float(lr)
            cleaned_dict[c_name] = sub
        return cleaned_dict or list(_ALL_ASSIMILATION_COMPONENTS)

    if isinstance(raw_components, (list, tuple, set)):
        names: list[str] = []
        for item in raw_components:
            c_name = str(item).strip()
            if not c_name:
                continue
            if c_name not in _ALL_ASSIMILATION_COMPONENTS:
                raise ValueError(
                    f'Unsupported assimilation component {c_name!r}; '
                    f'supported: {list(_ALL_ASSIMILATION_COMPONENTS)}.'
                )
            if c_name not in names:
                names.append(c_name)
        return names or list(_ALL_ASSIMILATION_COMPONENTS)

    return list(_ALL_ASSIMILATION_COMPONENTS)


def _eval_da_loss_tensors(  # noqa: PLR0913, PLR0917
    model: Any,  # noqa: ANN401
    output: dict[str, torch.Tensor],
    active_window_tensors: dict[str, torch.Tensor],
    prior_window_refs: dict[str, torch.Tensor],
    weights_map: dict[str, float],
    loss_obj: Any,  # noqa: ANN401
    loss_data: dict[str, torch.Tensor],
    win_start_idx: int,
    win_end_idx: int,
    *,
    fits_head_outputs: bool,
) -> tuple[torch.Tensor, float, float, float]:
    """Evaluate Variational DA objective (obs + prior reg) on window slice."""
    point = model.point_prediction(output)[:, win_start_idx:win_end_idx]
    if not fits_head_outputs:
        prediction: dict[str, torch.Tensor] = {'y_hat': point}
    else:
        prediction = {
            k: v[:, win_start_idx:win_end_idx]
            for k, v in output.items()
            if isinstance(v, torch.Tensor)
            and v.ndim >= _NDIM_3D
            and v.shape[1] >= win_end_idx
        }
    n_win = max(1, win_end_idx - win_start_idx)
    total_loss_t, all_losses_t = loss_obj(
        prediction,
        loss_data,
        predict_last_n=n_win,
        other_model_data={
            'optimized_components': active_window_tensors,
            'baseline_components': prior_window_refs,
            'component_weights': weights_map,
        },
    )
    obs_loss_t = all_losses_t.get('loss', total_loss_t)
    reg_loss_t = total_loss_t - obs_loss_t
    return (
        total_loss_t,
        float(total_loss_t.detach().item()),
        float(obs_loss_t.detach().item()),
        float(reg_loss_t.detach().item()),
    )


def _build_da_step_entry(  # noqa: PLR0913
    step_num: int,
    pred_dict: dict[str, torch.Tensor],
    cur_full_embeddings: dict[str, torch.Tensor],
    prior_full_embeddings: dict[str, torch.Tensor],
    *,
    total_loss_val: float,
    obs_loss_val: float,
    reg_loss_val: float,
    prior_q_cms: np.ndarray,
    y_cms_arr: np.ndarray,
    bundle: dict[str, Any],
    area_km2: float,
    hindcast_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    win_start_idx: int,
    win_end_idx: int,
    seq_len: int,
    lead_time: int,
    lr_val: float,
    reg_val: float,
    win_val: int,
    loss_name: str,
    comp_names: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Build per-iteration history entry with hydrograph, RMSE, and shifts."""
    with torch.no_grad():
        step_mm, step_q_cms, step_quantiles = _unscale_predictions(
            pred_dict=pred_dict,
            bundle=bundle,
            area_km2=area_km2,
            lead_time=lead_time,
        )
    step_fc, step_hc, step_peak_q, step_peak_date = (
        _format_forecast_and_hindcast_series(
            mm_per_day=step_mm,
            q_cms=step_q_cms,
            prob_quantiles_cms=step_quantiles,
            hindcast_dates=hindcast_dates,
            forecast_dates=forecast_dates,
        )
    )
    win_obs = y_cms_arr[win_start_idx:win_end_idx]
    win_step = step_q_cms[win_start_idx:win_end_idx]
    valid_win = np.isfinite(win_obs) & np.isfinite(win_step)
    if bool(np.any(valid_win)):
        diff = win_step[valid_win] - win_obs[valid_win]
        step_rmse: float | None = round(
            float(np.sqrt(np.mean(diff**2))),
            3,
        )
    else:
        step_rmse = None

    step_shifts = _compute_component_shifts(
        prior_full_embeddings,
        cur_full_embeddings,
        win_start_idx,
        win_end_idx,
    )
    step_assim_series = _build_assimilated_series_points(
        step_q_cms=step_q_cms,
        prior_q_cms=prior_q_cms,
        hindcast_dates=hindcast_dates,
        forecast_dates=forecast_dates,
        win_start_idx=win_start_idx,
        win_end_idx=win_end_idx,
        seq_len=seq_len,
        lead_time=lead_time,
    )
    entry = {
        'step': int(step_num),
        'total_loss': (
            round(float(total_loss_val), 6)
            if np.isfinite(total_loss_val)
            else None
        ),
        'obs_loss': (
            round(float(obs_loss_val), 6) if np.isfinite(obs_loss_val) else None
        ),
        'reg_loss': (
            round(float(reg_loss_val), 6) if np.isfinite(reg_loss_val) else None
        ),
        'window_rmse_cms': step_rmse,
        'component_shifts': step_shifts,
        'peak_discharge_cms': (
            round(step_peak_q, 3) if step_peak_q is not None else None
        ),
        'peak_date': step_peak_date,
        'learning_rate': round(float(lr_val), 5),
        'regularization_weight': round(float(reg_val), 4),
        'window_days': int(win_val),
        'loss': str(loss_name),
        'assimilation_components': list(comp_names),
        'hindcast_tail': step_hc,
        'forecast': step_fc,
        'assimilated_series': step_assim_series,
        'nudged_series': step_assim_series,
    }
    return entry, step_q_cms


def run_variational_assimilation_for_basin(  # noqa: C901, PLR0912, PLR0913, PLR0915
    username: str | None,
    catchment_id: str,
    *,
    issue_date: str | None = None,
    window_days: int = _DEFAULT_VARIATIONAL_WINDOW_DAYS,
    regularization_weight: float = _DEFAULT_VARIATIONAL_REG_WEIGHT,
    initial_learning_rate: float = _DEFAULT_VARIATIONAL_LR,
    epochs: int = _DEFAULT_VARIATIONAL_EPOCHS,
    loss: str = _DEFAULT_VARIATIONAL_LOSS,
    assimilation_components: Any = None,  # noqa: ANN401
    clip_gradient_norm: float | None = _DEFAULT_VARIATIONAL_CLIP_NORM,
    early_stopping_tolerance: float | None = _DEFAULT_VARIATIONAL_TOLERANCE,
    continue_session: bool = False,
    reset: bool = False,
    from_step: int | None = None,
    observed_series: Sequence[dict[str, Any]] | None = None,
    model_id: str = _DEFAULT_MODEL_ID,
    model_run_dir: Path | None = None,
    precip_overrides: dict[str, Any] | None = None,
    temp_overrides: dict[str, float] | None = None,
    product_overrides: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """Run iterative variational latent-embedding DA with step-by-step history.

    Optimizes the selected latent embeddings of ``MeanEmbeddingForecastLSTM``
    (``static_embedding``, ``hindcast_embedding``, ``forecast_embedding``)
    over an observation window preceding ``issue_date`` while keeping all
    neural network weights frozen. Supports continuing from the current or any
    previously inspected step with updated learning parameters, live progress
    polling, and per-iteration visual inspection (`epoch_history`).
    """
    cid_str = str(catchment_id or '').strip()
    if not cid_str:
        raise ValueError('catchment_id is required for data assimilation.')
    user_key = _normalize_user_key(username)

    t0 = resolve_issue_date(
        username=username,
        catchment_id=cid_str,
        requested_issue_date=issue_date,
        probe_ecmwf=False,
    )
    issue_date_str = _format_date_str(t0)
    session_key = (
        user_key,
        cid_str,
        str(model_id or _DEFAULT_MODEL_ID),
        issue_date_str,
    )
    if reset:
        with _DA_SESSION_LOCK:
            _DA_SESSIONS_BY_KEY.pop(session_key, None)

    bundle = {
        **_load_forecast_model_bundle(
            run_dir=model_run_dir,
            model_id=model_id,
            username=username,
        ),
        'active_mode': 'coldstart',
        'active_model_id': model_id,
    }
    cfg: Config = bundle['cfg']
    model = bundle['model']
    scaler: Scaler = bundle['scaler']
    seq_len = int(cfg.seq_length or _DEFAULT_SPINUP_DAYS)
    lead_time = int(cfg.lead_time or _DEFAULT_LEAD_TIME_DAYS)

    win_clamped = int(
        np.clip(
            int(window_days or _DEFAULT_VARIATIONAL_WINDOW_DAYS),
            1,
            min(_MAX_VARIATIONAL_WINDOW_DAYS, seq_len),
        )
    )
    reg_clamped = float(np.clip(float(regularization_weight), 0.0, 100.0))
    lr_clamped = float(np.clip(float(initial_learning_rate), 1e-5, 1.0))
    epochs_clamped = int(np.clip(int(epochs), 0, _MAX_VARIATIONAL_EPOCHS))
    comp_spec = _normalize_assimilation_components(assimilation_components)
    comp_names = (
        list(comp_spec.keys())
        if isinstance(comp_spec, dict)
        else list(comp_spec)
    )

    overrides = _unpack_override_payloads(
        precip_overrides,
        temp_overrides,
        product_overrides,
        is_what_if=False,
    )
    model_input, hindcast_dates, forecast_dates, area_km2 = (
        _assemble_inference_tensors(
            username=username,
            catchment_id=cid_str,
            issue_date=t0,
            bundle=bundle,
            overrides=overrides,
        )
    )

    obs_list = (
        list(observed_series)
        if observed_series is not None
        else load_realtime_streamflow_series(username, cid_str)
    )
    obs_by_date: dict[str, float] = {}
    for pt in obs_list:
        d_str = str(pt.get('date') or '').strip()
        q_val = _clean_float(pt.get('discharge_cms'))
        if d_str and q_val is not None and q_val >= 0.0:
            obs_by_date[d_str] = q_val

    y_cms_arr = np.full(seq_len + lead_time, np.nan, dtype=np.float32)
    for idx, d_ts in enumerate(hindcast_dates):
        d_str = _format_date_str(d_ts)
        if d_str in obs_by_date:
            y_cms_arr[idx] = obs_by_date[d_str]

    # Fallback when user uploaded a single anchor observation on t0 itself
    if np.isnan(y_cms_arr[:seq_len]).all() and issue_date_str in obs_by_date:
        y_cms_arr[seq_len - 1] = obs_by_date[issue_date_str]

    model.seq_length = seq_len
    model.__dict__.pop('_preloaded_state', None)
    with torch.no_grad():
        prior_dict = model(model_input, return_embeddings=True)
        prior_mm, prior_q_cms, prior_quantiles = _unscale_predictions(
            pred_dict=prior_dict,
            bundle=bundle,
            area_km2=area_km2,
            lead_time=lead_time,
        )

    prior_fc_series, prior_hc_series, prior_peak_q, prior_peak_date = (
        _format_forecast_and_hindcast_series(
            mm_per_day=prior_mm,
            q_cms=prior_q_cms,
            prob_quantiles_cms=prior_quantiles,
            hindcast_dates=hindcast_dates,
            forecast_dates=forecast_dates,
        )
    )

    finite_hc_idx = np.where(np.isfinite(y_cms_arr[:seq_len]))[0]
    if len(finite_hc_idx) == 0:
        return {
            'status': 'success',
            'method': 'variational',
            'applied': False,
            'catchment_id': cid_str,
            'basin_id': cid_str,
            'model_id': model_id,
            'issue_date': issue_date_str,
            'window_days': win_clamped,
            'effective_window_days': win_clamped,
            'window_start_date': None,
            'window_end_date': _format_date_str(hindcast_dates[-1]),
            'anchor_date': None,
            'regularization_weight': round(reg_clamped, 4),
            'initial_learning_rate': round(lr_clamped, 5),
            'epochs': epochs_clamped,
            'base_step': 0,
            'current_step': 0,
            'steps_ran': 0,
            'cancelled': False,
            'loss': str(loss or _DEFAULT_VARIATIONAL_LOSS).upper(),
            'assimilation_components': comp_names,
            'component_shifts': dict.fromkeys(comp_names, 0.0),
            'n_valid_obs': 0,
            'prior_window_rmse_cms': None,
            'assimilated_window_rmse_cms': None,
            'total_loss': None,
            'obs_loss': None,
            'reg_loss': None,
            'epoch_history': [],
            'hindcast_tail': prior_hc_series,
            'forecast': prior_fc_series,
            'baseline_hindcast_tail': prior_hc_series,
            'baseline_forecast': prior_fc_series,
            'nudged_series': [],
            'assimilated_series': [],
            'peak_discharge_cms': (
                round(prior_peak_q, 3) if prior_peak_q is not None else None
            ),
            'peak_date': prior_peak_date,
            'baseline_peak_discharge_cms': (
                round(prior_peak_q, 3) if prior_peak_q is not None else None
            ),
            'elapsed_sec': 0.0,
        }

    last_obs_idx = int(finite_hc_idx[-1])
    lag_steps = max(0, (seq_len - 1) - last_obs_idx)
    effective_window = int(np.clip(win_clamped + lag_steps, 1, max(1, seq_len)))
    win_start_idx = seq_len - effective_window
    win_end_idx = seq_len

    # Check if any finite observation falls inside [win_start_idx, win_end_idx);
    # if all observations are older than win_start_idx, expand window to reach
    # last_obs_idx.
    if not np.isfinite(y_cms_arr[win_start_idx:win_end_idx]).any():
        effective_window = int(
            np.clip((seq_len - last_obs_idx) + win_clamped - 1, 1, seq_len)
        )
        win_start_idx = seq_len - effective_window

    target_var = cfg.target_variables[0]
    scaler_ds = scaler.scaler
    y_center = float(scaler_ds[target_var].sel(parameter='center').to_numpy())
    y_scale = float(scaler_ds[target_var].sel(parameter='scale').to_numpy())
    cms_factor = float(area_km2) * _CMS_PER_MM_DAY_KM2
    y_mm_arr = y_cms_arr / cms_factor
    y_scaled_arr = ((y_mm_arr - y_center) / y_scale).astype(np.float32)

    batch_for_da: dict[str, torch.Tensor] = {
        'x_d_hindcast': model_input['x_d_hindcast'],
        'x_d_forecast': model_input['x_d_forecast'],
        'x_s': model_input['x_s'],
        'y': torch.from_numpy(
            y_scaled_arr.reshape(1, seq_len + lead_time, 1)
        ).to(torch.float32),
    }

    req_loss = str(loss or _DEFAULT_VARIATIONAL_LOSS).strip().upper()
    if req_loss not in ('MSE', 'NSE', 'CMAL'):
        req_loss = 'MSE'
    head_name = str(cfg.head or '').strip().lower()
    if req_loss == 'CMAL' and head_name not in ('cmal', 'cmal_deterministic'):
        req_loss = 'MSE'
    if req_loss == 'NSE':
        valid_scaled = y_scaled_arr[:seq_len][
            np.isfinite(y_scaled_arr[:seq_len])
        ]
        std_val = (
            float(np.std(valid_scaled)) if len(valid_scaled) >= 2 else 1.0  # noqa: PLR2004
        )
        batch_for_da['per_basin_target_stds'] = torch.tensor(
            [[max(std_val, 1e-3)]], dtype=torch.float32
        )

    da_block: dict[str, Any] = {
        'assimilation_components': comp_spec,
        'assimilation_window': int(effective_window),
        'regularization_weight': float(reg_clamped),
        'initial_learning_rate': float(lr_clamped),
        'epochs': max(1, int(epochs_clamped)),
        'optimizer': 'Adam',
        'loss': req_loss,
    }
    if clip_gradient_norm is not None and float(clip_gradient_norm) > 0.0:
        da_block['clip_gradient_norm'] = float(clip_gradient_norm)
    if (
        early_stopping_tolerance is not None
        and float(early_stopping_tolerance) > 0.0
    ):
        da_block['early_stopping_tolerance'] = float(early_stopping_tolerance)

    parent_dict = dict(cfg.as_dict())
    parent_dict['seq_length'] = int(seq_len + lead_time)
    parent_dict['lead_time'] = int(lead_time)
    assim_cfg = AssimilationConfig(da_block, parent_cfg=parent_dict)

    loss_obj = get_loss_obj(assim_cfg, per_sequence=True)
    loss_obj.set_regularization_terms(get_regularization_obj(assim_cfg))
    fits_head_outputs = req_loss == 'CMAL'
    loss_data: dict[str, torch.Tensor] = {
        'y': batch_for_da['y'][:, win_start_idx:win_end_idx],
    }
    if 'per_basin_target_stds' in batch_for_da:
        loss_data['per_basin_target_stds'] = batch_for_da[
            'per_basin_target_stds'
        ]
    weights_map = {
        name: float(spec['regularization_weight'])
        for name, spec in assim_cfg.assimilation_components.items()
    }

    prior_full_embeddings: dict[str, torch.Tensor] = {
        c_name: (
            prior_dict[c_name].detach().float().clone()
            if c_name == 'static_embedding'
            else prior_dict[c_name][:, :seq_len].detach().float().clone()
        )
        for c_name in _ALL_ASSIMILATION_COMPONENTS
    }
    prior_window_refs: dict[str, torch.Tensor] = {
        c_name: (
            prior_full_embeddings[c_name]
            if c_name == 'static_embedding'
            else prior_full_embeddings[c_name][:, win_start_idx:win_end_idx]
        )
        for c_name in comp_names
    }

    # Check if we are continuing an existing session for this (user, basin, t0)
    session_state: dict[str, Any] | None = None
    if continue_session and not reset:
        with _DA_SESSION_LOCK:
            existing = _DA_SESSIONS_BY_KEY.get(session_key)
            if existing is not None:
                session_state = existing

    if session_state is not None:
        step_embs_map: dict[int, dict[str, torch.Tensor]] = session_state.get(
            'step_embeddings', {}
        )
        history_list: list[dict[str, Any]] = list(
            session_state.get('history', [])
        )
        target_from = (
            int(from_step)
            if from_step is not None
            else int(session_state.get('current_step', 0))
        )
        if target_from in step_embs_map:
            base_step = target_from
            cur_full_embeddings = {
                k: v.detach().float().clone()
                for k, v in step_embs_map[target_from].items()
            }
            history_list = [
                h for h in history_list if int(h.get('step', 0)) <= base_step
            ]
            step_embs_map = {
                s: embs for s, embs in step_embs_map.items() if s <= base_step
            }
        else:
            base_step = int(session_state.get('current_step', 0))
            cur_full_embeddings = {
                k: v.detach().float().clone()
                for k, v in session_state['current_embeddings'].items()
            }
    else:
        base_step = 0
        cur_full_embeddings = {
            k: v.detach().float().clone()
            for k, v in prior_full_embeddings.items()
        }
        with torch.no_grad():
            _, prior_tot_l, prior_obs_l, prior_reg_l = _eval_da_loss_tensors(
                model=model,
                output=prior_dict,
                active_window_tensors=prior_window_refs,
                prior_window_refs=prior_window_refs,
                weights_map=weights_map,
                loss_obj=loss_obj,
                loss_data=loss_data,
                win_start_idx=win_start_idx,
                win_end_idx=win_end_idx,
                fits_head_outputs=fits_head_outputs,
            )
        step0_entry, _ = _build_da_step_entry(
            0,
            prior_dict,
            prior_full_embeddings,
            prior_full_embeddings,
            total_loss_val=prior_tot_l,
            obs_loss_val=prior_obs_l,
            reg_loss_val=prior_reg_l,
            prior_q_cms=prior_q_cms,
            y_cms_arr=y_cms_arr,
            bundle=bundle,
            area_km2=area_km2,
            hindcast_dates=hindcast_dates,
            forecast_dates=forecast_dates,
            win_start_idx=win_start_idx,
            win_end_idx=win_end_idx,
            seq_len=seq_len,
            lead_time=lead_time,
            lr_val=lr_clamped,
            reg_val=reg_clamped,
            win_val=win_clamped,
            loss_name=req_loss,
            comp_names=comp_names,
        )
        history_list = [step0_entry]
        step_embs_map = {
            0: {
                k: v.detach().float().clone()
                for k, v in prior_full_embeddings.items()
            }
        }

    # Window prior RMSE diagnostic
    win_obs = y_cms_arr[win_start_idx:win_end_idx]
    win_prior = prior_q_cms[win_start_idx:win_end_idx]
    valid_prior_win = np.isfinite(win_obs) & np.isfinite(win_prior)
    n_valid_obs = int(np.sum(valid_prior_win))
    prior_rmse: float | None = (
        round(
            float(
                np.sqrt(
                    np.mean(
                        (win_prior[valid_prior_win] - win_obs[valid_prior_win])
                        ** 2
                    )
                )
            ),
            3,
        )
        if n_valid_obs > 0
        else None
    )

    cancel_ev = threading.Event()
    with _DA_PROGRESS_LOCK:
        _DA_CANCEL_EVENTS[user_key] = cancel_ev
    _init_user_da_progress(
        user_key=user_key,
        catchment_id=cid_str,
        model_id=model_id,
        issue_date_str=issue_date_str,
        base_step=base_step,
        total_steps=epochs_clamped,
        prior_rmse=prior_rmse,
    )

    t_start = time.perf_counter()
    cancelled = False
    steps_ran = 0
    cur_step = base_step

    if epochs_clamped > 0:
        active_window_tensors: dict[str, torch.Tensor] = {
            c_name: (
                cur_full_embeddings[c_name]
                .detach()
                .float()
                .clone()
                .requires_grad_(requires_grad=True)
                if c_name == 'static_embedding'
                else cur_full_embeddings[c_name][:, win_start_idx:win_end_idx]
                .detach()
                .float()
                .clone()
                .requires_grad_(requires_grad=True)
            )
            for c_name in comp_names
        }
        param_groups: list[dict[str, Any]] = []
        for c_name, tensor in active_window_tensors.items():
            group: dict[str, Any] = {'params': [tensor]}
            c_lr = assim_cfg.assimilation_components[c_name][
                'initial_learning_rate'
            ]
            if c_lr is not None:
                group['lr'] = float(c_lr)
            param_groups.append(group)
        optimizer = get_optimizer(param_groups, assim_cfg)

        # Determine whether any component has prior edits outside current window
        has_prev_edits = base_step > 0
        device_type = batch_for_da['y'].device.type
        if not torch.amp.is_autocast_available(device_type):
            device_type = 'cpu'

        try:
            with (
                torch.autocast(device_type=device_type, enabled=False),
                _freeze_model_for_da(model),
            ):
                for step_idx in range(1, epochs_clamped + 1):
                    if cancel_ev.is_set():
                        cancelled = True
                        break

                    last_finite_win = {
                        k: v.detach().clone()
                        for k, v in active_window_tensors.items()
                    }
                    optimizer.zero_grad()

                    if has_prev_edits:
                        fwd_overrides = _build_full_observed_overrides(
                            cur_full_embeddings,
                            active_window_tensors,
                            win_start_idx,
                            win_end_idx,
                            seq_len,
                        )
                        fwd_slice = (0, seq_len)
                    else:
                        fwd_overrides = active_window_tensors
                        fwd_slice = (win_start_idx, win_end_idx)

                    out_step = model(
                        batch_for_da,
                        assimilation_overrides=fwd_overrides,
                        assimilation_slice=fwd_slice,
                    )
                    loss_t, _, _, _ = _eval_da_loss_tensors(
                        model=model,
                        output=out_step,
                        active_window_tensors=active_window_tensors,
                        prior_window_refs=prior_window_refs,
                        weights_map=weights_map,
                        loss_obj=loss_obj,
                        loss_data=loss_data,
                        win_start_idx=win_start_idx,
                        win_end_idx=win_end_idx,
                        fits_head_outputs=fits_head_outputs,
                    )
                    if not bool(torch.isfinite(loss_t)):
                        break
                    loss_t.backward()
                    if any(
                        t.grad is not None
                        and not bool(torch.isfinite(t.grad).all())
                        for t in active_window_tensors.values()
                    ):
                        break
                    if assim_cfg.clip_gradient_norm is not None:
                        _clip_da_gradients(
                            active_window_tensors.values(),
                            float(assim_cfg.clip_gradient_norm),
                        )
                    optimizer.step()
                    if any(
                        not bool(torch.isfinite(t.detach()).all())
                        for t in active_window_tensors.values()
                    ):
                        for k, v in last_finite_win.items():
                            active_window_tensors[k].data.copy_(v)
                        break

                    # Commit updated window into cur_full_embeddings
                    with torch.no_grad():
                        cur_full_embeddings = _build_full_observed_overrides(
                            cur_full_embeddings,
                            {
                                k: v.detach().clone()
                                for k, v in active_window_tensors.items()
                            },
                            win_start_idx,
                            win_end_idx,
                            seq_len,
                        )
                        eval_overrides = (
                            cur_full_embeddings
                            if has_prev_edits
                            else {
                                k: v.detach().clone()
                                for k, v in active_window_tensors.items()
                            }
                        )
                        eval_slice = (
                            (0, seq_len)
                            if has_prev_edits
                            else (win_start_idx, win_end_idx)
                        )
                        post_dict = model(
                            batch_for_da,
                            assimilation_overrides=eval_overrides,
                            assimilation_slice=eval_slice,
                        )
                        _, step_tot_l, step_obs_l, step_reg_l = (
                            _eval_da_loss_tensors(
                                model=model,
                                output=post_dict,
                                active_window_tensors={
                                    k: v.detach()
                                    for k, v in active_window_tensors.items()
                                },
                                prior_window_refs=prior_window_refs,
                                weights_map=weights_map,
                                loss_obj=loss_obj,
                                loss_data=loss_data,
                                win_start_idx=win_start_idx,
                                win_end_idx=win_end_idx,
                                fits_head_outputs=fits_head_outputs,
                            )
                        )

                    cur_step = base_step + step_idx
                    steps_ran = step_idx
                    step_entry, _ = _build_da_step_entry(
                        cur_step,
                        post_dict,
                        cur_full_embeddings,
                        prior_full_embeddings,
                        total_loss_val=step_tot_l,
                        obs_loss_val=step_obs_l,
                        reg_loss_val=step_reg_l,
                        prior_q_cms=prior_q_cms,
                        y_cms_arr=y_cms_arr,
                        bundle=bundle,
                        area_km2=area_km2,
                        hindcast_dates=hindcast_dates,
                        forecast_dates=forecast_dates,
                        win_start_idx=win_start_idx,
                        win_end_idx=win_end_idx,
                        seq_len=seq_len,
                        lead_time=lead_time,
                        lr_val=lr_clamped,
                        reg_val=reg_clamped,
                        win_val=win_clamped,
                        loss_name=req_loss,
                        comp_names=comp_names,
                    )
                    history_list.append(step_entry)
                    step_embs_map[cur_step] = {
                        k: v.detach().float().clone()
                        for k, v in cur_full_embeddings.items()
                    }
                    _update_user_da_progress(
                        user_key,
                        event='step',
                        current_step=cur_step,
                        steps_completed=steps_ran,
                        total_loss=step_tot_l,
                        obs_loss=step_obs_l,
                        reg_loss=step_reg_l,
                        window_rmse_cms=step_entry['window_rmse_cms'],
                    )
        except Exception as exc:
            _update_user_da_progress(user_key, event='error', error=str(exc))
            raise

    elapsed_sec = time.perf_counter() - t_start
    latest_entry = history_list[-1]
    _update_user_da_progress(
        user_key,
        event='cancelled' if cancelled else 'complete',
        current_step=cur_step,
        steps_completed=steps_ran,
        total_loss=latest_entry.get('total_loss'),
        obs_loss=latest_entry.get('obs_loss'),
        reg_loss=latest_entry.get('reg_loss'),
        window_rmse_cms=latest_entry.get('window_rmse_cms'),
    )

    with _DA_SESSION_LOCK:
        _DA_SESSIONS_BY_KEY[session_key] = {
            'current_step': cur_step,
            'current_embeddings': {
                k: v.detach().float().clone()
                for k, v in cur_full_embeddings.items()
            },
            'step_embeddings': step_embs_map,
            'history': history_list,
            'updated_at': time.time(),
        }

    anchor_date = _format_date_str(hindcast_dates[last_obs_idx])
    window_start_date = _format_date_str(hindcast_dates[win_start_idx])
    window_end_date = _format_date_str(hindcast_dates[win_end_idx - 1])

    return {
        'status': 'success',
        'method': 'variational',
        'applied': True,
        'catchment_id': cid_str,
        'basin_id': cid_str,
        'model_id': model_id,
        'issue_date': issue_date_str,
        'window_days': win_clamped,
        'effective_window_days': effective_window,
        'window_start_date': window_start_date,
        'window_end_date': window_end_date,
        'anchor_date': anchor_date,
        'regularization_weight': round(reg_clamped, 4),
        'initial_learning_rate': round(lr_clamped, 5),
        'epochs': epochs_clamped,
        'base_step': base_step,
        'current_step': cur_step,
        'steps_ran': steps_ran,
        'cancelled': cancelled,
        'loss': req_loss,
        'assimilation_components': comp_names,
        'component_shifts': latest_entry['component_shifts'],
        'n_valid_obs': n_valid_obs,
        'prior_window_rmse_cms': prior_rmse,
        'assimilated_window_rmse_cms': latest_entry['window_rmse_cms'],
        'total_loss': latest_entry['total_loss'],
        'obs_loss': latest_entry['obs_loss'],
        'reg_loss': latest_entry['reg_loss'],
        'epoch_history': history_list,
        'hindcast_tail': latest_entry['hindcast_tail'],
        'forecast': latest_entry['forecast'],
        'baseline_hindcast_tail': prior_hc_series,
        'baseline_forecast': prior_fc_series,
        'nudged_series': latest_entry['assimilated_series'],
        'assimilated_series': latest_entry['assimilated_series'],
        'peak_discharge_cms': latest_entry['peak_discharge_cms'],
        'peak_date': latest_entry['peak_date'],
        'baseline_peak_discharge_cms': (
            round(prior_peak_q, 3) if prior_peak_q is not None else None
        ),
        'elapsed_sec': round(elapsed_sec, 3),
    }


def _resolve_nudging_baseline_series(  # noqa: PLR0913
    username: str | None,
    catchment_id: str,
    *,
    forecast_series: Sequence[dict[str, Any]] | None,
    hindcast_series: Sequence[dict[str, Any]] | None,
    issue_date: str | None,
    model_id: str,
    model_run_dir: Path | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    str | None,
    float | None,
]:
    """Return baseline forecast/hindcast series, using cache or run fallback."""
    fc_in = list(forecast_series or [])
    hc_in = list(hindcast_series or [])
    eff_issue = issue_date
    base_peak: float | None = None
    if forecast_series is None and not fc_in:
        cached_fc = load_cached_daily_forecast(username, catchment_id, model_id)
        if cached_fc and cached_fc.get('forecast'):
            fc_in = list(cached_fc.get('forecast') or [])
            hc_in = list(cached_fc.get('hindcast_tail') or [])
            base_peak = _clean_float(cached_fc.get('peak_discharge_cms'))
            if not eff_issue:
                eff_issue = cached_fc.get('issue_date')
        else:
            try:
                base_run = run_forecast_model(
                    username=username,
                    catchment_id=catchment_id,
                    mode='coldstart',
                    requested_issue_date=issue_date,
                    model_id=model_id,
                    model_run_dir=model_run_dir,
                )
                fc_in = list(base_run.get('forecast') or [])
                hc_in = list(base_run.get('hindcast_tail') or [])
                base_peak = _clean_float(base_run.get('peak_discharge_cms'))
                if not eff_issue:
                    eff_issue = base_run.get('issue_date')
            except (ValueError, FileNotFoundError, KeyError):
                pass
    return fc_in, hc_in, eff_issue, base_peak


def run_data_assimilation_for_basin(  # noqa: PLR0913
    username: str | None,
    catchment_id: str,
    *,
    method: str = 'nudging',
    rho: float = _DEFAULT_NUDGING_RHO,
    alpha: float = _DEFAULT_NUDGING_ALPHA,
    window_days: int = _DEFAULT_NUDGING_WINDOW_DAYS,
    issue_date: str | None = None,
    forecast_series: Sequence[dict[str, Any]] | None = None,
    hindcast_series: Sequence[dict[str, Any]] | None = None,
    observed_series: Sequence[dict[str, Any]] | None = None,
    regularization_weight: float = _DEFAULT_VARIATIONAL_REG_WEIGHT,
    initial_learning_rate: float = _DEFAULT_VARIATIONAL_LR,
    epochs: int = _DEFAULT_VARIATIONAL_EPOCHS,
    loss: str = _DEFAULT_VARIATIONAL_LOSS,
    assimilation_components: Any = None,  # noqa: ANN401
    clip_gradient_norm: float | None = _DEFAULT_VARIATIONAL_CLIP_NORM,
    early_stopping_tolerance: float | None = _DEFAULT_VARIATIONAL_TOLERANCE,
    continue_session: bool = False,
    reset: bool = False,
    from_step: int | None = None,
    model_id: str = _DEFAULT_MODEL_ID,
    model_run_dir: Path | None = None,
    precip_overrides: dict[str, Any] | None = None,
    temp_overrides: dict[str, float] | None = None,
    product_overrides: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """Server endpoint helper for ``POST /api/forecast/assimilate``."""
    cid_str = str(catchment_id or '').strip()
    if not cid_str:
        raise ValueError('catchment_id is required for data assimilation.')

    norm_method = str(method or 'nudging').strip().lower().replace('-', '_')
    if norm_method in ('nudging', 'post_processing', 'postprocessing', 'ar1'):
        obs = (
            list(observed_series)
            if observed_series is not None
            else load_realtime_streamflow_series(username, cid_str)
        )
        fc_in, hc_in, eff_issue, base_peak = _resolve_nudging_baseline_series(
            username=username,
            catchment_id=cid_str,
            forecast_series=forecast_series,
            hindcast_series=hindcast_series,
            issue_date=issue_date,
            model_id=model_id,
            model_run_dir=model_run_dir,
        )
        res = compute_nudging_postprocessing(
            forecast_series=fc_in,
            hindcast_series=hc_in,
            observed_series=obs,
            issue_date=eff_issue,
            rho=rho,
            alpha=alpha,
            window_days=window_days,
        )
        res['catchment_id'] = cid_str
        res['basin_id'] = cid_str
        res['issue_date'] = eff_issue
        res['baseline_forecast'] = fc_in
        res['baseline_hindcast_tail'] = hc_in
        res['baseline_peak_discharge_cms'] = base_peak
        return res

    if norm_method in (
        'variational',
        'embedding',
        'latent_variational',
        '4dvar',
        '4d_var',
    ):
        return run_variational_assimilation_for_basin(
            username=username,
            catchment_id=cid_str,
            issue_date=issue_date,
            window_days=(
                _DEFAULT_VARIATIONAL_WINDOW_DAYS
                if window_days is None
                else window_days
            ),
            regularization_weight=regularization_weight,
            initial_learning_rate=initial_learning_rate,
            epochs=epochs,
            loss=loss,
            assimilation_components=assimilation_components,
            clip_gradient_norm=clip_gradient_norm,
            early_stopping_tolerance=early_stopping_tolerance,
            continue_session=continue_session,
            reset=reset,
            from_step=from_step,
            observed_series=observed_series,
            model_id=model_id,
            model_run_dir=model_run_dir,
            precip_overrides=precip_overrides,
            temp_overrides=temp_overrides,
            product_overrides=product_overrides,
        )

    raise ValueError(
        f'Unsupported data assimilation method {method!r}; expected '
        "'nudging' or 'variational'."
    )
