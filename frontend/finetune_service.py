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

"""Interactive per-basin model fine-tuning service for OpenHydroNet."""

from __future__ import annotations

import io
import json
import logging
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from frontend.profile_manager import get_profile_manager
from model.datasetzoo import get_dataset
from model.datasetzoo.multimet import MultimetDataLoader
from model.modelzoo import get_model, get_model_class, load_model_weights
from model.training import get_loss_obj, get_optimizer, get_regularization_obj
from model.utils.config import Config
from utils.file_paths import SCALER_ZARR_NAME

if TYPE_CHECKING:
    from model.datasetzoo.basedataset import BaseDataset

    from model.datautils.scaler import Scaler
    from model.training.loss import BaseLoss

REPO_ROOT = Path(__file__).resolve().parent.parent


def _users_dir() -> Path:
    return get_profile_manager().profiles_dir
_METRIC_EPS = 1e-12
_DATE_COLUMN_ALIASES = frozenset(
    {'date', 'time', 'datetime', 'timestamp', 'issue_date'}
)
_SUPPORTED_DATA_TYPES = frozenset(
    {
        'precip',
        'precipitation',
        'temp',
        'temperature',
        'streamflow',
        'q',
        'discharge',
    }
)

PRETRAINED_MODEL_DIRS: tuple[tuple[str, str, Path], ...] = (
    (
        'google-floodhub-settings-110-epochs',
        'Google FloodHub Pretrained (110 Epochs, CMAL)',
        REPO_ROOT
        / 'model'
        / 'pretrained-models'
        / 'google-floodhub-settings-110-epochs',
    ),
    (
        '5-basin-example',
        '5-Basin Tutorial Pretrained (5 Epochs, Regression)',
        REPO_ROOT / 'model' / 'tutorial' / 'model-runs' / '5-basin-example',
    ),
)

SUPPORTED_LOSSES: tuple[str, ...] = (
    'CMALLoss',
    'MSE',
    'NSE',
    'RMSE',
    'WMSE',
)

SUPPORTED_LR_STRATEGIES: tuple[str, ...] = (
    'constant',
    'StepLR',
    'ReduceLROnPlateau',
    'ExponentialLR',
)

_SAFE_ID_RE = re.compile(r'[^a-zA-Z0-9._-]+')
_LEAD_COL_RE = re.compile(
    r'^(?:lead[_\s-]*|t\+|d\+|day[_\s-]*|\+)?(\d+)(?:d|day|days|h|hr|hrs)?$',
    re.IGNORECASE,
)


def _sanitize_id(value: str, fallback: str = 'default') -> str:
    cleaned = _SAFE_ID_RE.sub('_', (value or '').strip()).strip('._-')
    return cleaned or fallback


def _session_root(username: str, basin_id: str, model_id: str) -> Path:
    safe_user = _sanitize_id(username, 'default_user')
    safe_basin = _sanitize_id(basin_id, 'custom_basin')
    safe_model = _sanitize_id(model_id, 'google-floodhub-settings-110-epochs')
    return (
        _users_dir()
        / safe_user
        / 'finetune_workspace'
        / safe_basin
        / safe_model
    )


def _basin_data_root(username: str, basin_id: str) -> Path:
    safe_user = _sanitize_id(username, 'default_user')
    safe_basin = _sanitize_id(basin_id, 'custom_basin')
    return _users_dir() / safe_user / 'finetune_data' / safe_basin


def _find_latest_checkpoint(run_dir: Path) -> tuple[Path | None, int]:
    checkpoints = sorted(run_dir.glob('model_epoch*.pt'))
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    match = re.search(r'model_epoch(\d+)\.pt$', latest.name)
    epoch = int(match.group(1)) if match else len(checkpoints)
    return latest, epoch


def _extract_product_groups(
    inputs_cfg: dict[str, list[str]] | list[str] | None,
) -> list[str]:
    if isinstance(inputs_cfg, dict):
        return list(inputs_cfg.keys())
    if isinstance(inputs_cfg, list):
        groups: list[str] = []
        for feat in inputs_cfg:
            prefix = feat.split('_', 1)[0]
            if prefix not in groups:
                groups.append(prefix)
        return groups
    return []


def _inspect_model_from_run_dir(
    model_id: str,
    label: str,
    run_dir: Path,
    *,
    extra_local_products: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    cfg_path = run_dir / 'config.yml'
    ckpt_path, ckpt_epoch = _find_latest_checkpoint(run_dir)
    if not cfg_path.exists() or ckpt_path is None:
        raise FileNotFoundError(
            f'Model directory {run_dir} is missing config.yml or checkpoint.'
        )

    cfg = Config(cfg_path, dev_mode=True)
    cfg_dict = cfg.as_dict().copy()
    cfg_dict['compile'] = False
    cfg_dict['device'] = 'cpu'
    cfg_dict['run_dir'] = str(run_dir)
    cfg_dict['base_run_dir'] = str(run_dir)
    cfg_dict['is_finetuning'] = False

    newly_init: set[tuple[str, str]] = set()
    if extra_local_products:
        hindcast_in = (
            dict(cfg_dict.get('hindcast_inputs', {}))
            if isinstance(cfg_dict.get('hindcast_inputs'), dict)
            else {}
        )
        forecast_in = (
            dict(cfg_dict.get('forecast_inputs', {}))
            if isinstance(cfg_dict.get('forecast_inputs'), dict)
            else {}
        )
        for prod_key, bands in extra_local_products.items():
            hindcast_in[prod_key] = list(bands)
            forecast_in[prod_key] = list(bands)
            newly_init.add(('shared_embeddings_fc', prod_key))
        cfg_dict['hindcast_inputs'] = hindcast_in
        cfg_dict['forecast_inputs'] = forecast_in

    inspect_cfg = Config(cfg_dict, dev_mode=True)
    model_cls = get_model_class(inspect_cfg.model)
    model = get_model(inspect_cfg)
    if newly_init:
        model._newly_initialized_embeddings = newly_init  # noqa: SLF001

    components = model.describe_tunable_components()
    total_params = sum(p.numel() for p in model.parameters())

    head_name = str(cfg.head).lower()
    allowed_losses = (
        ['CMALLoss']
        if head_name in ('cmal', 'cmal_deterministic')
        else ['MSE', 'NSE', 'RMSE', 'WMSE']
    )

    return {
        'model_id': model_id,
        'label': label,
        'run_dir': str(run_dir),
        'checkpoint_path': str(ckpt_path),
        'checkpoint_epoch': ckpt_epoch,
        'architecture': str(cfg.model),
        'architecture_class': model_cls.__name__,
        'head': str(cfg.head),
        'default_loss': str(cfg.loss),
        'allowed_losses': allowed_losses,
        'supported_lr_strategies': list(SUPPORTED_LR_STRATEGIES),
        'default_learning_rate': float(cfg.initial_learning_rate or 5e-4),
        'default_lr_strategy': str(
            getattr(cfg, 'learning_rate_strategy', None) or 'constant'
        ),
        'default_batch_size': int(cfg.batch_size or 32),
        'seq_length': int(cfg.seq_length),
        'lead_time': int(cfg.lead_time),
        'forecast_overlap': int(cfg.forecast_overlap or cfg.seq_length),
        'predict_last_n': int(cfg.predict_last_n or cfg.lead_time),
        'hidden_size': int(cfg.hidden_size),
        'total_parameters': int(total_params),
        'tunable_component_names': model_cls.get_tunable_component_names(),
        'tunable_components': components,
        'base_hindcast_products': _extract_product_groups(cfg.hindcast_inputs),
        'base_forecast_products': _extract_product_groups(cfg.forecast_inputs),
        'static_attributes': list(cfg.static_attributes or []),
        'target_variables': list(cfg.target_variables or ['streamflow']),
    }


def list_finetune_models(
    username: str = 'default_user',
) -> list[dict[str, Any]]:
    """Return inventory of base models and dynamically inspected components."""
    inventory: list[dict[str, Any]] = []
    local_override = {
        'local_forcing': ['local_precipitation', 'local_temperature']
    }

    for model_id, label, run_dir in PRETRAINED_MODEL_DIRS:
        if (run_dir / 'config.yml').exists():
            info = _inspect_model_from_run_dir(
                model_id,
                label,
                run_dir,
                extra_local_products=local_override,
            )
            inventory.append(info)

    safe_user = _sanitize_id(username, 'default_user')
    user_runs_dir = _users_dir() / safe_user / 'runs'
    if user_runs_dir.exists():
        for run_dir in sorted(user_runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            ckpt_path, _ = _find_latest_checkpoint(run_dir)
            if (run_dir / 'config.yml').exists() and ckpt_path is not None:
                model_id = f'user_{run_dir.name}'
                info = _inspect_model_from_run_dir(
                    model_id,
                    f'User Model: {run_dir.name}',
                    run_dir,
                    extra_local_products=local_override,
                )
                inventory.append(info)

    return inventory


def _resolve_model_dir(
    model_id: str, username: str = 'default_user'
) -> tuple[str, Path]:
    for mid, label, run_dir in PRETRAINED_MODEL_DIRS:
        if mid == model_id and (run_dir / 'config.yml').exists():
            return label, run_dir
    if model_id.startswith('user_'):
        safe_user = _sanitize_id(username, 'default_user')
        run_name = model_id.removeprefix('user_')
        candidate = _users_dir() / safe_user / 'runs' / run_name
        if (candidate / 'config.yml').exists():
            return f'User Model: {run_name}', candidate
    for _mid, label, run_dir in PRETRAINED_MODEL_DIRS:
        if (run_dir / 'config.yml').exists():
            return label, run_dir
    raise FileNotFoundError(f'Unknown model_id: {model_id}')


def _identify_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() in _DATE_COLUMN_ALIASES:
            return str(col)
    return str(df.columns[0])


def _extract_daily_dataframe(
    csv_content: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    text = (csv_content or '').strip()
    if not text:
        raise ValueError('Uploaded CSV content is empty.')

    df = pd.read_csv(io.StringIO(text))
    if df.empty or len(df.columns) < 2:  # noqa: PLR2004
        raise ValueError(
            'CSV must contain a date/time column and at least one data column.'
        )

    date_col = _identify_date_column(df)
    raw_dates = pd.to_datetime(df[date_col], errors='coerce', utc=False)
    if raw_dates.isna().all():
        raise ValueError(
            f"Could not parse dates from column '{date_col}'. "
            'Expected ISO dates (YYYY-MM-DD).'
        )
    if hasattr(raw_dates.dt, 'tz') and raw_dates.dt.tz is not None:
        raw_dates = raw_dates.dt.tz_localize(None)
    raw_dates = raw_dates.dt.floor('D')

    valid_date_mask = ~raw_dates.isna()
    df = df.loc[valid_date_mask].copy()
    df['__parsed_date__'] = raw_dates.loc[valid_date_mask]
    df = df.sort_values('__parsed_date__').drop_duplicates(
        subset=['__parsed_date__'], keep='last'
    )

    data_cols = [
        str(c) for c in df.columns if c not in (date_col, '__parsed_date__')
    ]
    if not data_cols:
        raise ValueError('CSV contains no numeric data columns.')

    min_date = df['__parsed_date__'].iloc[0]
    max_date = df['__parsed_date__'].iloc[-1]
    full_dates = pd.date_range(min_date, max_date, freq='D')
    df = df.set_index('__parsed_date__').reindex(full_dates)
    return df, full_dates, data_cols


def _extract_lead_columns(
    df: pd.DataFrame, data_cols: list[str]
) -> tuple[dict[int, np.ndarray], list[np.ndarray]]:
    lead_map: dict[int, np.ndarray] = {}
    unmatched_cols: list[np.ndarray] = []
    for col in data_cols:
        col_str = str(col).strip()
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(
            dtype=np.float32
        )
        m = _LEAD_COL_RE.match(col_str)
        if m:
            lead_idx = int(m.group(1))
            if lead_idx >= 1:
                lead_map[lead_idx] = vals
            elif lead_idx == 0 and 1 not in lead_map:
                lead_map[1] = vals
        else:
            unmatched_cols.append(vals)
    return lead_map, unmatched_cols


def parse_time_by_lead_csv(
    csv_content: str,
    *,
    target_lead_time: int = 7,
) -> tuple[pd.DatetimeIndex, np.ndarray, list[int]]:
    """Parse a CSV file structured as ``time`` by ``lead_time``.

    Parameters
    ----------
    csv_content : str
        Raw CSV text uploaded by the user.
    target_lead_time : int, optional
        Number of forecast lead steps expected by the model (default 7).

    Returns
    -------
    tuple[pd.DatetimeIndex, np.ndarray, list[int]]
        - ``dates``: Sorted daily ``DatetimeIndex`` of length ``T``.
        - ``matrix``: Float32 array of shape ``(T, target_lead_time)`` where
          missing values are preserved as ``np.nan``.
        - ``parsed_leads``: List of integer lead times found in the CSV columns.
    """
    df, full_dates, data_cols = _extract_daily_dataframe(csv_content)
    lead_map, unmatched_cols = _extract_lead_columns(df, data_cols)

    num_dates = len(full_dates)
    matrix = np.full((num_dates, target_lead_time), np.nan, dtype=np.float32)

    if len(data_cols) == 1:
        daily_series = (
            lead_map[next(iter(lead_map))] if lead_map else unmatched_cols[0]
        )
        for k in range(target_lead_time):
            if k == 0:
                matrix[:, 0] = daily_series
            elif k < num_dates:
                matrix[:-k, k] = daily_series[k:]
        parsed_leads = [1]
    elif lead_map:
        parsed_leads = sorted(lead_map.keys())
        for k in range(1, target_lead_time + 1):
            if k in lead_map:
                matrix[:, k - 1] = lead_map[k]
    else:
        parsed_leads = list(range(1, len(unmatched_cols) + 1))
        for idx, vals in enumerate(unmatched_cols[:target_lead_time]):
            matrix[:, idx] = vals

    return full_dates, matrix, parsed_leads


def _resolve_upload_key_and_units(
    dtype_clean: str,
    matrix: np.ndarray,
    units: str,
    area_km2: float | None,
) -> tuple[str, np.ndarray, str]:
    if dtype_clean in ('precip', 'precipitation'):
        if np.nanmin(matrix) < 0.0:
            raise ValueError('Precipitation values cannot be negative.')
        return 'precip', matrix, 'mm/day'
    if dtype_clean in ('temp', 'temperature'):
        return 'temp', matrix, 'degC'

    if units.strip().lower() in ('m3/s', 'cms', 'm³/s'):
        if area_km2 is None or area_km2 <= 0:
            raise ValueError(
                'Basin area_km2 (> 0) is required to convert streamflow '
                'from m³/s to mm/day.'
            )
        matrix = (matrix * (86.4 / float(area_km2))).astype(np.float32)
    return 'streamflow', matrix, 'mm/day'


def upload_finetune_csv(  # noqa: PLR0913
    username: str,
    basin_id: str,
    data_type: str,
    csv_content: str,
    *,
    filename: str = '',
    units: str = 'mm/day',
    area_km2: float | None = None,
    target_lead_time: int = 7,
) -> dict[str, Any]:
    """Parse and store per-basin local CSV data (precip, temp, or streamflow).

    Parameters
    ----------
    username : str
        Active user profile name.
    basin_id : str
        Target basin identifier.
    data_type : str
        One of ``'precip'``, ``'temp'``, or ``'streamflow'``.
    csv_content : str
        Raw CSV string.
    filename : str, optional
        Original filename for display metadata.
    units : str, optional
        For streamflow: ``'mm/day'`` or ``'m3/s'`` (CMS). If ``'m3/s'`` and
        ``area_km2`` is provided, converts to ``mm/day``
        (``cms * 86.4 / area_km2``) to match Caravan/FloodHub scaler units.
    area_km2 : float | None, optional
        Basin drainage area in km² used when converting streamflow from m³/s.
    target_lead_time : int, optional
        Number of lead times to store in the local forcing dataset (default 7).
    """
    dtype_clean = (data_type or '').strip().lower()
    if dtype_clean not in _SUPPORTED_DATA_TYPES:
        raise ValueError(
            f"Unsupported data_type '{data_type}'. "
            "Must be 'precip', 'temp', or 'streamflow'."
        )

    dates, matrix, parsed_leads = parse_time_by_lead_csv(
        csv_content, target_lead_time=target_lead_time
    )

    if not np.isfinite(matrix[:, 0]).any():
        raise ValueError(
            f'Uploaded {data_type} CSV contains no finite values '
            'in the primary lead column.'
        )

    data_root = _basin_data_root(username, basin_id)
    raw_dir = data_root / 'raw_arrays'
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_root / 'upload_meta.json'
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    key, matrix, canonical_units = _resolve_upload_key_and_units(
        dtype_clean, matrix, units, area_km2
    )

    np.savez(
        raw_dir / f'{key}.npz',
        dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
        matrix=matrix,
        parsed_leads=np.array(parsed_leads, dtype=np.int32),
    )

    finite_mask = np.isfinite(matrix[:, 0])
    meta[key] = {
        'filename': filename or f'{key}.csv',
        'uploaded_at': datetime.now(UTC).isoformat(),
        'start_date': str(dates[0].strftime('%Y-%m-%d')),
        'end_date': str(dates[-1].strftime('%Y-%m-%d')),
        'num_days': len(dates),
        'num_valid_days': int(finite_mask.sum()),
        'num_lead_columns': len(parsed_leads),
        'parsed_leads': parsed_leads,
        'units': canonical_units,
        'original_units': units,
        'area_km2': area_km2,
        'min_val': float(np.nanmin(matrix[:, 0])),
        'max_val': float(np.nanmax(matrix[:, 0])),
        'mean_val': float(np.nanmean(matrix[:, 0])),
    }
    meta['basin_id'] = basin_id
    meta_path.write_text(json.dumps(meta, indent=2))

    _rebuild_local_zarr_stores(
        username, basin_id, target_lead_time=target_lead_time
    )
    return meta[key]


def generate_sample_basin_csvs(
    username: str,
    basin_id: str,
    *,
    num_days: int = 240,
    lead_time: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate realistic synthetic ``time`` x ``lead_time`` CSVs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=num_days, freq='D')
    t_idx = np.arange(num_days, dtype=np.float64)

    base_temp = 12.0 + 10.0 * np.sin(2.0 * np.pi * (t_idx - 80.0) / 365.0)
    daily_temp = (base_temp + rng.normal(0.0, 2.2, size=num_days)).astype(
        np.float32
    )

    rain_prob = 0.28 + 0.12 * np.sin(2.0 * np.pi * t_idx / 90.0)
    wet_days = rng.uniform(0.0, 1.0, size=num_days) < rain_prob
    daily_precip = np.where(
        wet_days, rng.gamma(shape=1.6, scale=6.5, size=num_days), 0.0
    ).astype(np.float32)

    streamflow = np.zeros(num_days, dtype=np.float32)
    soil_storage = 25.0
    slow_store = 8.0
    quick_store = 0.0
    for i in range(num_days):
        p = float(daily_precip[i])
        et = max(0.2, 0.08 * max(0.0, float(daily_temp[i]) + 5.0))
        soil_storage = max(0.0, min(80.0, soil_storage + 0.65 * p - et))
        effective_rain = max(0.0, p * (0.15 + 0.55 * (soil_storage / 80.0)))
        quick_store = 0.45 * quick_store + 0.75 * effective_rain
        slow_store = 0.94 * slow_store + 0.25 * effective_rain
        q = 0.35 * quick_store + 0.08 * slow_store + 0.4
        streamflow[i] = max(0.05, q + float(rng.normal(0.0, 0.08)))

    lead_cols = [f'lead_{k}' for k in range(1, lead_time + 1)]

    def _to_matrix_csv(series: np.ndarray, noise_std: float) -> str:
        rows: list[dict[str, Any]] = []
        for i, d in enumerate(dates):
            row: dict[str, Any] = {'date': d.strftime('%Y-%m-%d')}
            for k in range(1, lead_time + 1):
                src_idx = min(num_days - 1, i + k - 1)
                val = float(series[src_idx])
                if k > 1 and noise_std > 0:
                    val += float(rng.normal(0.0, noise_std * math.sqrt(k - 1)))
                    if series is daily_precip:
                        val = max(0.0, val)
                row[f'lead_{k}'] = round(val, 3)
            rows.append(row)
        df_out = pd.DataFrame(rows, columns=['date', *lead_cols])
        return df_out.to_csv(index=False)

    precip_csv = _to_matrix_csv(daily_precip, noise_std=0.35)
    temp_csv = _to_matrix_csv(daily_temp, noise_std=0.25)
    sf_df = pd.DataFrame(
        {
            'date': dates.strftime('%Y-%m-%d'),
            'lead_1': np.round(streamflow, 3),
        }
    )
    sf_csv = sf_df.to_csv(index=False)

    upload_finetune_csv(
        username,
        basin_id,
        'precip',
        precip_csv,
        filename=f'{basin_id}_precip_time_by_lead.csv',
        target_lead_time=lead_time,
    )
    upload_finetune_csv(
        username,
        basin_id,
        'temp',
        temp_csv,
        filename=f'{basin_id}_temp_time_by_lead.csv',
        target_lead_time=lead_time,
    )
    upload_finetune_csv(
        username,
        basin_id,
        'streamflow',
        sf_csv,
        filename=f'{basin_id}_streamflow.csv',
        target_lead_time=lead_time,
    )
    return {
        'precip_csv_preview': '\n'.join(precip_csv.splitlines()[:6]),
        'temp_csv_preview': '\n'.join(temp_csv.splitlines()[:6]),
        'streamflow_csv_preview': '\n'.join(sf_csv.splitlines()[:6]),
    }


def clear_finetune_data(username: str, basin_id: str) -> dict[str, Any]:
    """Delete uploaded local CSVs and fine-tuning sessions for ``basin_id``."""
    data_root = _basin_data_root(username, basin_id)
    if data_root.exists():
        shutil.rmtree(data_root)
    safe_user = _sanitize_id(username, 'default_user')
    safe_basin = _sanitize_id(basin_id, 'custom_basin')
    ws_dir = _users_dir() / safe_user / 'finetune_workspace' / safe_basin
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
    return {'status': 'cleared', 'basin_id': basin_id}


def _load_uploaded_arrays(
    username: str, basin_id: str
) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    raw_dir = _basin_data_root(username, basin_id) / 'raw_arrays'
    loaded: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    if not raw_dir.exists():
        return loaded
    for key in ('precip', 'temp', 'streamflow'):
        p = raw_dir / f'{key}.npz'
        if p.exists():
            with np.load(p, allow_pickle=False) as archive:
                dates = pd.to_datetime(archive['dates'])
                matrix = archive['matrix'].astype(np.float32)
                loaded[key] = (pd.DatetimeIndex(dates), matrix)
    return loaded


def _rebuild_local_zarr_stores(
    username: str,
    basin_id: str,
    *,
    target_lead_time: int = 7,
) -> dict[str, Any] | None:
    """Write ``LOCAL_FORCING/timeseries.zarr`` and ``streamflow.zarr``."""
    loaded = _load_uploaded_arrays(username, basin_id)
    if not all(k in loaded for k in ('precip', 'temp', 'streamflow')):
        return None

    start_date = max(loaded[k][0][0] for k in ('precip', 'temp', 'streamflow'))
    end_date = min(loaded[k][0][-1] for k in ('precip', 'temp', 'streamflow'))
    if start_date >= end_date:
        raise ValueError(
            f'Uploaded CSVs have no overlapping date range '
            f'(max start {start_date.date()} >= min end {end_date.date()}).'
        )

    common_dates = pd.date_range(start_date, end_date, freq='D')
    num_dates = len(common_dates)

    def _align(key: str) -> np.ndarray:
        dates, mat = loaded[key]
        cols = min(mat.shape[1], target_lead_time)
        out = np.full((num_dates, target_lead_time), np.nan, dtype=np.float32)
        idx = dates.get_indexer(common_dates)
        valid = idx >= 0
        out[valid, :cols] = mat[idx[valid], :cols]
        for k in range(cols, target_lead_time):
            if k < num_dates:
                out[:-k, k] = out[k:, 0]
            out[-k:, k] = out[-1, 0]
        for k in range(1, target_lead_time):
            tail_nan = ~np.isfinite(out[:, k]) & np.isfinite(out[:, 0])
            out[tail_nan, k] = out[tail_nan, 0]
        return out

    precip_mat = _align('precip')
    temp_mat = _align('temp')
    sf_dates, sf_mat = loaded['streamflow']
    sf_idx = sf_dates.get_indexer(common_dates)
    sf_series = np.full(num_dates, np.nan, dtype=np.float32)
    valid_sf = sf_idx >= 0
    sf_series[valid_sf] = sf_mat[sf_idx[valid_sf], 0]

    data_root = _basin_data_root(username, basin_id)
    zarr_root = data_root / 'zarr_store'
    local_dyn_dir = zarr_root / 'local_dyn' / 'LOCAL_FORCING'
    if local_dyn_dir.exists():
        shutil.rmtree(local_dyn_dir)
    local_dyn_dir.mkdir(parents=True, exist_ok=True)

    lead_coords = pd.timedelta_range(
        start='1D', periods=target_lead_time, freq='D'
    )
    forcing_ds = xr.Dataset(
        {
            'local_precipitation': (
                ('basin', 'date', 'lead_time'),
                precip_mat[np.newaxis, :, :],
            ),
            'local_temperature': (
                ('basin', 'date', 'lead_time'),
                temp_mat[np.newaxis, :, :],
            ),
        },
        coords={
            'basin': [basin_id],
            'date': common_dates,
            'lead_time': lead_coords,
        },
    )
    forcing_ds.to_zarr(local_dyn_dir / 'timeseries.zarr', mode='w')

    sf_zarr_path = zarr_root / 'streamflow.zarr'
    if sf_zarr_path.exists():
        shutil.rmtree(sf_zarr_path)
    sf_ds = xr.Dataset(
        {
            'streamflow': (
                ('basin', 'date'),
                sf_series[np.newaxis, :],
            )
        },
        coords={'basin': [basin_id], 'date': common_dates},
    )
    sf_ds.to_zarr(sf_zarr_path, mode='w')

    basin_file = zarr_root / 'basin.txt'
    basin_file.write_text(f'{basin_id}\n')

    return {
        'start_date': common_dates[0].strftime('%Y-%m-%d'),
        'end_date': common_dates[-1].strftime('%Y-%m-%d'),
        'num_days': num_dates,
    }


def _ensure_placeholder_global_stores(
    base_cfg: Config,
    base_run_dir: Path,
    basin_id: str,
    dates: pd.DatetimeIndex,
    zarr_root: Path,
) -> tuple[Path, Path]:
    """Create local NaN placeholder Zarr stores for base products & statics."""
    global_dyn_dir = zarr_root / 'global_dyn_placeholders'
    lead_time = int(base_cfg.lead_time)
    lead_coords = pd.timedelta_range(start='1D', periods=lead_time, freq='D')

    hindcast_in = (
        base_cfg.hindcast_inputs
        if isinstance(base_cfg.hindcast_inputs, dict)
        else {}
    )
    forecast_in = (
        base_cfg.forecast_inputs
        if isinstance(base_cfg.forecast_inputs, dict)
        else {}
    )
    all_products = set(hindcast_in.keys()) | set(forecast_in.keys())

    for prod in sorted(all_products):
        prod_upper = prod.upper()
        pdir = global_dyn_dir / prod_upper / 'timeseries.zarr'
        if pdir.exists():
            shutil.rmtree(pdir)
        pdir.parent.mkdir(parents=True, exist_ok=True)
        bands = sorted(
            set(hindcast_in.get(prod, [])) | set(forecast_in.get(prod, []))
        )
        if prod in forecast_in:
            nan_arr = np.full(
                (1, len(dates), lead_time), np.nan, dtype=np.float32
            )
            ds = xr.Dataset(
                dict.fromkeys(
                    bands, (('basin', 'date', 'lead_time'), nan_arr)
                ),
                coords={
                    'basin': [basin_id],
                    'date': dates,
                    'lead_time': lead_coords,
                },
            )
        else:
            nan_arr_2d = np.full((1, len(dates)), np.nan, dtype=np.float32)
            ds = xr.Dataset(
                dict.fromkeys(bands, (('basin', 'date'), nan_arr_2d)),
                coords={'basin': [basin_id], 'date': dates},
            )
        ds.to_zarr(pdir, mode='w')

    attr_path = zarr_root / 'attributes.zarr'
    if attr_path.exists():
        shutil.rmtree(attr_path)
    base_scaler = xr.open_zarr(base_run_dir / SCALER_ZARR_NAME).load()
    static_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for attr_name in base_cfg.static_attributes or []:
        if attr_name in base_scaler.data_vars:
            center_val = float(
                base_scaler[attr_name].sel(parameter='center').values
            )
        else:
            center_val = 0.0
        static_vars[attr_name] = (
            ('basin',),
            np.array([center_val], dtype=np.float32),
        )
    attr_ds = xr.Dataset(static_vars, coords={'basin': [basin_id]})
    attr_ds.to_zarr(attr_path, mode='w')
    return global_dyn_dir, attr_path


def compute_partition_dates(
    dates: pd.DatetimeIndex,
    *,
    base_seq_length: int,
    lead_time: int,
    split_ratio: float = 0.7,
) -> dict[str, Any]:
    """Compute effective sequence length and train/validation date bounds."""
    num_days = len(dates)
    min_required = lead_time + 14
    if num_days < min_required:
        raise ValueError(
            f'Uploaded record has {num_days} overlapping days, but at least '
            f'{min_required} days are required for lead_time={lead_time}.'
        )

    max_allowed_seq = max(7, (num_days - lead_time - 6) // 2)
    eff_seq_length = min(int(base_seq_length), int(max_allowed_seq))

    first_sample_idx = eff_seq_length
    last_sample_idx = num_days - lead_time
    total_sample_days = last_sample_idx - first_sample_idx + 1
    if total_sample_days < 4:  # noqa: PLR2004
        raise ValueError(
            f'Insufficient valid sample dates ({total_sample_days}) after '
            f'accounting for seq_length={eff_seq_length} and '
            f'lead_time={lead_time}.'
        )

    clamped_ratio = max(0.2, min(0.9, float(split_ratio)))
    train_count = max(
        2,
        min(
            total_sample_days - 2,
            round(total_sample_days * clamped_ratio),
        ),
    )
    split_sample_idx = first_sample_idx + train_count - 1
    val_start_idx = split_sample_idx + 1

    return {
        'effective_seq_length': eff_seq_length,
        'lead_time': lead_time,
        'split_ratio': round(train_count / total_sample_days, 3),
        'record_start_date': dates[0].strftime('%Y-%m-%d'),
        'record_end_date': dates[-1].strftime('%Y-%m-%d'),
        'train_start_date': dates[first_sample_idx].strftime('%Y-%m-%d'),
        'train_end_date': dates[split_sample_idx].strftime('%Y-%m-%d'),
        'validation_start_date': dates[val_start_idx].strftime('%Y-%m-%d'),
        'validation_end_date': dates[last_sample_idx].strftime('%Y-%m-%d'),
        'split_date': dates[split_sample_idx].strftime('%Y-%m-%d'),
        'train_sample_days': int(train_count),
        'validation_sample_days': int(total_sample_days - train_count),
        'total_days': int(num_days),
    }


def _compute_metrics(
    obs: np.ndarray, sim: np.ndarray
) -> dict[str, float | None]:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if int(mask.sum()) < 2:  # noqa: PLR2004
        return {'nse': None, 'kge': None, 'rmse': None}
    o = obs[mask].astype(np.float64)
    s = sim[mask].astype(np.float64)
    diff = s - o
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sum((o - np.mean(o)) ** 2))
    nse = (
        float(1.0 - np.sum(diff * diff) / denom)
        if denom > _METRIC_EPS
        else None
    )

    std_o = float(np.std(o))
    std_s = float(np.std(s))
    mean_o = float(np.mean(o))
    mean_s = float(np.mean(s))
    if (
        std_o > _METRIC_EPS
        and std_s > _METRIC_EPS
        and abs(mean_o) > _METRIC_EPS
    ):
        r = float(np.corrcoef(o, s)[0, 1])
        alpha = std_s / std_o
        beta = mean_s / mean_o
        kge = float(
            1.0
            - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        )
    else:
        kge = None
    return {
        'nse': (
            round(nse, 4) if nse is not None and math.isfinite(nse) else None
        ),
        'kge': (
            round(kge, 4) if kge is not None and math.isfinite(kge) else None
        ),
        'rmse': round(rmse, 4) if math.isfinite(rmse) else None,
    }


def _append_subkey_to_spec(spec: dict[str, Any], grp: str, sub: str) -> None:
    existing = spec.get(grp)
    if existing is True:
        return
    sub_list = list(existing) if isinstance(existing, list) else []
    if sub not in sub_list:
        sub_list.append(sub)
    spec[grp] = sub_list


def _build_finetune_modules_spec(
    selected_components: list[str] | dict[str, Any] | None,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Convert UI component selections into ``Config.finetune_modules``."""
    target_model = getattr(model, '_orig_mod', model)
    newly_init: set[tuple[str, str]] = getattr(
        target_model, '_newly_initialized_embeddings', set()
    )

    spec: dict[str, Any] = {}
    if isinstance(selected_components, dict):
        for k, v in selected_components.items():
            spec[str(k)] = v
    elif isinstance(selected_components, list):
        for item in selected_components:
            token = str(item).strip()
            if not token:
                continue
            if '.' in token:
                grp, sub = token.split('.', 1)
                _append_subkey_to_spec(spec, grp, sub)
            else:
                spec[token] = True

    for grp, prod in sorted(newly_init):
        _append_subkey_to_spec(spec, grp, prod)

    if not spec:
        spec['head'] = True
    return spec


def _unfreeze_module_dict(
    module: torch.nn.ModuleDict,
    module_name: str,
    subspec: object,
) -> list[str]:
    unfrozen: list[str] = []
    keys_to_unfreeze: list[str] = []
    if subspec is True:
        keys_to_unfreeze = list(module.keys())
    elif isinstance(subspec, list):
        keys_to_unfreeze = [str(k) for k in subspec if str(k) in module]
    for sub_key in keys_to_unfreeze:
        for p in module[sub_key].parameters():
            p.requires_grad = True
        unfrozen.append(f'{module_name}.{sub_key}')
    return unfrozen


def _apply_freeze_spec(
    model: torch.nn.Module, finetune_modules: dict[str, Any]
) -> list[str]:
    """Freeze all parameters except those enabled in ``finetune_modules``."""
    target_model = getattr(model, '_orig_mod', model)
    for param in target_model.parameters():
        param.requires_grad = False

    unfrozen: list[str] = []
    for module_name, subspec in finetune_modules.items():
        module = getattr(target_model, module_name, None)
        if module is None:
            continue
        if isinstance(module, torch.nn.ModuleDict):
            unfrozen.extend(
                _unfreeze_module_dict(module, module_name, subspec)
            )
        elif subspec is True:
            for p in module.parameters():
                p.requires_grad = True
            unfrozen.append(module_name)
    return unfrozen


def _evaluate_dataset_predictions(  # noqa: PLR0913
    model: torch.nn.Module,
    dataset: BaseDataset,
    loss_obj: BaseLoss,
    scaler: Scaler,
    full_dates: pd.DatetimeIndex,
    *,
    lead_time: int,
) -> tuple[float, list[float | None], list[float | None]]:
    """Evaluate loss and unscaled streamflow predictions across a dataset."""
    model.eval()
    loader = MultimetDataLoader(
        dataset,
        lazy_load=False,
        logging_level=logging.INFO,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    sf_var = scaler.scaler['streamflow']
    sf_center = float(sf_var.sel(parameter='center').values)
    sf_scale = float(sf_var.sel(parameter='scale').values)

    losses: list[float] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for xs_raw in loader:
            xs = model.pre_model_hook(xs_raw, is_train=False)
            output = model(xs)
            loss_val, _ = loss_obj(output, xs)
            if torch.isfinite(loss_val):
                losses.append(float(loss_val.item()))
            point_pred = model.point_prediction(output)
            forecast_slice = point_pred[:, -lead_time:, 0]
            unscaled = (
                forecast_slice.detach().cpu().numpy() * sf_scale + sf_center
            )
            all_preds.append(unscaled)

    mean_loss = float(np.mean(losses)) if losses else float('nan')
    sim_lead1: list[float | None] = [None] * len(full_dates)
    sim_lead_last: list[float | None] = [None] * len(full_dates)

    if not all_preds:
        return mean_loss, sim_lead1, sim_lead_last

    pred_matrix = np.concatenate(all_preds, axis=0)
    date_to_pos = {d: i for i, d in enumerate(full_dates)}

    ds_dates = pd.DatetimeIndex(dataset._dataset.coords['date'].values)  # noqa: SLF001
    for sample_pos in range(len(dataset._sample_index)):  # noqa: SLF001
        if sample_pos >= pred_matrix.shape[0]:
            break
        sample_idx = dataset._sample_index[sample_pos]  # noqa: SLF001
        date_int_idx = int(sample_idx['date'])
        issue_date = ds_dates[date_int_idx]
        pos1 = date_to_pos.get(issue_date)
        if pos1 is not None and np.isfinite(pred_matrix[sample_pos, 0]):
            sim_lead1[pos1] = round(float(pred_matrix[sample_pos, 0]), 4)
        last_lead_date = issue_date + pd.Timedelta(days=lead_time - 1)
        pos_last = date_to_pos.get(last_lead_date)
        if pos_last is not None and np.isfinite(pred_matrix[sample_pos, -1]):
            sim_lead_last[pos_last] = round(
                float(pred_matrix[sample_pos, -1]), 4
            )

    return mean_loss, sim_lead1, sim_lead_last


def get_finetune_workspace_status(
    username: str,
    basin_id: str,
    model_id: str = 'google-floodhub-settings-110-epochs',
    *,
    split_ratio: float | None = None,
) -> dict[str, Any]:
    """Return full status of the per-basin fine-tuning workspace."""
    models = list_finetune_models(username)
    selected_model = next(
        (m for m in models if m['model_id'] == model_id),
        models[0] if models else None,
    )
    if selected_model is None:
        raise FileNotFoundError('No pretrained models found in repository.')

    data_root = _basin_data_root(username, basin_id)
    meta_path = data_root / 'upload_meta.json'
    upload_meta = (
        json.loads(meta_path.read_text()) if meta_path.exists() else {}
    )
    loaded = _load_uploaded_arrays(username, basin_id)
    data_ready = all(k in loaded for k in ('precip', 'temp', 'streamflow'))

    sess_dir = _session_root(username, basin_id, selected_model['model_id'])
    state_path = sess_dir / 'session_state.json'
    session_state: dict[str, Any] = (
        json.loads(state_path.read_text()) if state_path.exists() else {}
    )

    active_split = (
        float(split_ratio)
        if split_ratio is not None
        else float(session_state.get('split_ratio', 0.7))
    )

    timeseries_payload: dict[str, Any] | None = None
    partition_info: dict[str, Any] | None = None

    if data_ready:
        start_d = max(
            loaded[k][0][0] for k in ('precip', 'temp', 'streamflow')
        )
        end_d = min(
            loaded[k][0][-1] for k in ('precip', 'temp', 'streamflow')
        )
        common_dates = pd.date_range(start_d, end_d, freq='D')

        def _extract_lead1(key: str) -> list[float | None]:
            d_idx, mat = loaded[key]
            indexer = d_idx.get_indexer(common_dates)
            out: list[float | None] = []
            for pos in indexer:
                if pos >= 0 and np.isfinite(mat[pos, 0]):
                    out.append(round(float(mat[pos, 0]), 4))
                else:
                    out.append(None)
            return out

        partition_info = compute_partition_dates(
            common_dates,
            base_seq_length=int(selected_model['seq_length']),
            lead_time=int(selected_model['lead_time']),
            split_ratio=active_split,
        )
        timeseries_payload = {
            'dates': [d.strftime('%Y-%m-%d') for d in common_dates],
            'precip': _extract_lead1('precip'),
            'temp': _extract_lead1('temp'),
            'streamflow_obs': _extract_lead1('streamflow'),
        }

    return {
        'basin_id': basin_id,
        'model_id': selected_model['model_id'],
        'selected_model': selected_model,
        'models': models,
        'upload_meta': upload_meta,
        'data_ready': data_ready,
        'partition': partition_info,
        'timeseries': timeseries_payload,
        'session': {
            'current_epoch': int(session_state.get('current_epoch', 0)),
            'status': session_state.get(
                'status', 'ready' if data_ready else 'awaiting_data'
            ),
            'split_ratio': (
                partition_info['split_ratio']
                if partition_info
                else active_split
            ),
            'epoch_history': session_state.get('epoch_history', []),
            'last_hyperparameters': session_state.get(
                'last_hyperparameters',
                {
                    'epochs_per_step': 5,
                    'learning_rate': selected_model['default_learning_rate'],
                    'loss': selected_model['default_loss'],
                    'learning_rate_strategy': selected_model[
                        'default_lr_strategy'
                    ],
                    'batch_size': selected_model['default_batch_size'],
                    'finetune_modules': [
                        'shared_embeddings_fc.local_forcing',
                        'head',
                    ],
                },
            ),
        },
    }


def reset_finetune_session(
    username: str,
    basin_id: str,
    model_id: str = 'google-floodhub-settings-110-epochs',
) -> dict[str, Any]:
    """Reset the interactive fine-tuning session back to Epoch 0."""
    sess_dir = _session_root(username, basin_id, model_id)
    if sess_dir.exists():
        shutil.rmtree(sess_dir)
    return get_finetune_workspace_status(username, basin_id, model_id)


def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_strat_val: str,
    ft_cfg: Config,
) -> Any:  # noqa: ANN401
    if lr_strat_val == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(ft_cfg.learning_rate_epochs_drop or 5)),
            gamma=float(ft_cfg.learning_rate_drop_factor or 0.5),
        )
    if lr_strat_val == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(ft_cfg.learning_rate_drop_factor or 0.5),
            patience=max(1, int(ft_cfg.learning_rate_epochs_drop or 3)),
        )
    if lr_strat_val == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(ft_cfg.learning_rate_drop_factor or 0.95),
        )
    return None


def _build_finetune_config(  # noqa: PLR0913, PLR0917
    base_cfg: Config,
    base_run_dir: Path,
    run_dir: Path,
    zarr_root: Path,
    global_dyn_dir: Path,
    attr_path: Path,
    partition: dict[str, Any],
    *,
    lead_time: int,
    lr_val: float,
    loss_val: str,
    lr_strat_val: str,
    bs_val: int,
) -> Config:
    eff_seq = int(partition['effective_seq_length'])
    cfg_dict = base_cfg.as_dict().copy()
    cfg_dict['is_finetuning'] = True
    cfg_dict['base_run_dir'] = str(base_run_dir)
    cfg_dict['run_dir'] = str(run_dir)
    cfg_dict['compile'] = False
    cfg_dict['device'] = 'cpu'
    cfg_dict['num_workers'] = 0
    cfg_dict['lazy_load'] = False
    cfg_dict['seq_length'] = eff_seq
    cfg_dict['forecast_overlap'] = eff_seq
    cfg_dict['lead_time'] = lead_time
    cfg_dict['predict_last_n'] = lead_time
    cfg_dict['initial_learning_rate'] = lr_val
    cfg_dict['loss'] = loss_val
    cfg_dict['learning_rate_strategy'] = lr_strat_val
    if lr_strat_val in ('StepLR', 'ReduceLROnPlateau'):
        default_drop = 5 if lr_strat_val == 'StepLR' else 3
        cfg_dict['learning_rate_epochs_drop'] = max(
            1, int(cfg_dict.get('learning_rate_epochs_drop') or default_drop)
        )
        cfg_dict['learning_rate_drop_factor'] = float(
            cfg_dict.get('learning_rate_drop_factor') or 0.5
        )
    elif lr_strat_val == 'ExponentialLR':
        cfg_dict['learning_rate_drop_factor'] = float(
            cfg_dict.get('learning_rate_drop_factor') or 0.95
        )
    cfg_dict['batch_size'] = bs_val
    cfg_dict['nan_handling_method'] = 'masked_mean'
    cfg_dict['union_mapping'] = {}

    basin_file = zarr_root / 'basin.txt'
    cfg_dict['train_basin_file'] = str(basin_file)
    cfg_dict['validation_basin_file'] = str(basin_file)
    cfg_dict['test_basin_file'] = str(basin_file)

    fmt = '%d/%m/%Y'
    cfg_dict['train_start_date'] = pd.Timestamp(
        partition['train_start_date']
    ).strftime(fmt)
    cfg_dict['train_end_date'] = pd.Timestamp(
        partition['train_end_date']
    ).strftime(fmt)
    cfg_dict['validation_start_date'] = pd.Timestamp(
        partition['validation_start_date']
    ).strftime(fmt)
    cfg_dict['validation_end_date'] = pd.Timestamp(
        partition['validation_end_date']
    ).strftime(fmt)
    cfg_dict['test_start_date'] = cfg_dict['validation_start_date']
    cfg_dict['test_end_date'] = cfg_dict['validation_end_date']

    cfg_dict['statics_data_dir'] = str(attr_path)
    cfg_dict['targets_data_dir'] = str(zarr_root / 'streamflow.zarr')
    cfg_dict['dynamics_data_dir'] = [
        str(global_dyn_dir),
        str(zarr_root / 'local_dyn'),
    ]

    hindcast_in = dict(cfg_dict.get('hindcast_inputs', {}))
    forecast_in = dict(cfg_dict.get('forecast_inputs', {}))
    local_bands = ['local_precipitation', 'local_temperature']
    hindcast_in['local_forcing'] = list(local_bands)
    forecast_in['local_forcing'] = list(local_bands)
    cfg_dict['hindcast_inputs'] = hindcast_in
    cfg_dict['forecast_inputs'] = forecast_in
    return Config(cfg_dict, dev_mode=True)


def _evaluate_and_build_epoch_record(  # noqa: PLR0913, PLR0917
    epoch: int,
    label: str,
    model: torch.nn.Module,
    train_eval_ds: BaseDataset,
    val_ds: BaseDataset,
    loss_obj: BaseLoss,
    scaler: Scaler,
    common_dates: pd.DatetimeIndex,
    obs_series: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    lead_time: int,
    learning_rate: float,
    unfrozen_parts: list[str],
    override_train_loss: float | None = None,
) -> tuple[dict[str, Any], float]:
    tr_loss, tr_sim1, tr_sim_last = _evaluate_dataset_predictions(
        model,
        train_eval_ds,
        loss_obj,
        scaler,
        common_dates,
        lead_time=lead_time,
    )
    val_loss, val_sim1, val_sim_last = _evaluate_dataset_predictions(
        model,
        val_ds,
        loss_obj,
        scaler,
        common_dates,
        lead_time=lead_time,
    )
    if override_train_loss is not None:
        tr_loss = override_train_loss

    combined_sim1 = [
        v if v is not None else t
        for t, v in zip(tr_sim1, val_sim1, strict=True)
    ]
    combined_sim_last = [
        v if v is not None else t
        for t, v in zip(tr_sim_last, val_sim_last, strict=True)
    ]
    sim_arr = np.array(
        [np.nan if x is None else x for x in combined_sim1],
        dtype=np.float32,
    )
    tr_metrics = _compute_metrics(obs_series[train_mask], sim_arr[train_mask])
    val_metrics = _compute_metrics(obs_series[val_mask], sim_arr[val_mask])
    record = {
        'epoch': epoch,
        'label': label,
        'train_loss': round(tr_loss, 5) if math.isfinite(tr_loss) else None,
        'val_loss': round(val_loss, 5) if math.isfinite(val_loss) else None,
        'train_nse': tr_metrics['nse'],
        'val_nse': val_metrics['nse'],
        'train_kge': tr_metrics['kge'],
        'val_kge': val_metrics['kge'],
        'train_rmse': tr_metrics['rmse'],
        'val_rmse': val_metrics['rmse'],
        'learning_rate': learning_rate,
        'unfrozen_components': unfrozen_parts,
        'streamflow_sim_lead1': combined_sim1,
        'streamflow_sim_lead_last': combined_sim_last,
    }
    return record, val_loss


def _train_single_epoch(
    model: torch.nn.Module,
    train_loader: MultimetDataLoader,
    optimizer: torch.optim.Optimizer,
    loss_obj: BaseLoss,
    clip_grad_norm: float | None,
) -> float:
    model.train()
    batch_losses: list[float] = []
    for xs_raw in train_loader:
        optimizer.zero_grad()
        xs = model.pre_model_hook(xs_raw, is_train=True)
        output = model(xs)
        loss_tensor, _ = loss_obj(output, xs)
        loss_tensor.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if torch.isfinite(loss_tensor):
            batch_losses.append(float(loss_tensor.item()))
    return float(np.mean(batch_losses)) if batch_losses else float('nan')


def _step_lr_scheduler(lr_scheduler: object, val_loss: float) -> None:
    if lr_scheduler is None:
        return
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if math.isfinite(val_loss):
            lr_scheduler.step(val_loss)
    elif hasattr(lr_scheduler, 'step'):
        lr_scheduler.step()  # type: ignore[union-attr]


def run_finetune_epochs(  # noqa: C901, PLR0913, PLR0915
    username: str,
    basin_id: str,
    model_id: str,
    *,
    epochs: int = 5,
    learning_rate: float | None = None,
    loss: str | None = None,
    learning_rate_strategy: str | None = None,
    batch_size: int | None = None,
    finetune_modules: list[str] | dict[str, Any] | None = None,
    split_ratio: float = 0.7,
) -> dict[str, Any]:
    """Run ``epochs`` of interactive per-basin fine-tuning and pause."""
    if epochs < 1 or epochs > 200:  # noqa: PLR2004
        raise ValueError('epochs per step must be between 1 and 200.')

    _, base_run_dir = _resolve_model_dir(model_id, username)
    base_ckpt, _ = _find_latest_checkpoint(base_run_dir)
    if base_ckpt is None:
        raise FileNotFoundError(f'No checkpoint found in {base_run_dir}.')

    base_cfg = Config(base_run_dir / 'config.yml', dev_mode=True)
    lead_time = int(base_cfg.lead_time)

    zarr_meta = _rebuild_local_zarr_stores(
        username, basin_id, target_lead_time=lead_time
    )
    if zarr_meta is None:
        raise ValueError(
            'Please upload local precipitation, temperature, and streamflow '
            'CSVs before fine-tuning.'
        )

    loaded = _load_uploaded_arrays(username, basin_id)
    start_d = max(loaded[k][0][0] for k in ('precip', 'temp', 'streamflow'))
    end_d = min(loaded[k][0][-1] for k in ('precip', 'temp', 'streamflow'))
    common_dates = pd.date_range(start_d, end_d, freq='D')

    partition = compute_partition_dates(
        common_dates,
        base_seq_length=int(base_cfg.seq_length),
        lead_time=lead_time,
        split_ratio=split_ratio,
    )

    data_root = _basin_data_root(username, basin_id)
    zarr_root = data_root / 'zarr_store'
    global_dyn_dir, attr_path = _ensure_placeholder_global_stores(
        base_cfg, base_run_dir, basin_id, common_dates, zarr_root
    )

    sess_dir = _session_root(username, basin_id, model_id)
    run_dir = sess_dir / 'run'
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = sess_dir / 'session_state.json'
    session_state: dict[str, Any] = (
        json.loads(state_path.read_text()) if state_path.exists() else {}
    )

    prev_split = session_state.get('split_date')
    if prev_split is not None and prev_split != partition['split_date']:
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        session_state = {}

    lr_val = float(
        learning_rate
        if learning_rate is not None
        else (base_cfg.initial_learning_rate or 5e-4)
    )
    head_name = str(base_cfg.head).lower()
    loss_val = str(loss) if loss is not None else str(base_cfg.loss)
    if (
        head_name in ('cmal', 'cmal_deterministic')
        and loss_val.lower() != 'cmalloss'
    ):
        loss_val = 'CMALLoss'

    lr_strat_val = str(
        learning_rate_strategy
        if learning_rate_strategy is not None
        else (getattr(base_cfg, 'learning_rate_strategy', None) or 'constant')
    )
    if lr_strat_val not in SUPPORTED_LR_STRATEGIES:
        lr_strat_val = 'constant'

    bs_val = int(
        batch_size if batch_size is not None else (base_cfg.batch_size or 16)
    )

    ft_cfg = _build_finetune_config(
        base_cfg,
        base_run_dir,
        run_dir,
        zarr_root,
        global_dyn_dir,
        attr_path,
        partition,
        lead_time=lead_time,
        lr_val=lr_val,
        loss_val=loss_val,
        lr_strat_val=lr_strat_val,
        bs_val=bs_val,
    )

    train_ds = get_dataset(
        cfg=ft_cfg, is_train=True, period='train', compute_scaler=False
    )
    val_ds = get_dataset(
        cfg=ft_cfg, is_train=False, period='validation', compute_scaler=False
    )
    train_eval_ds = get_dataset(
        cfg=ft_cfg, is_train=False, period='train', compute_scaler=False
    )

    torch.manual_seed(int(ft_cfg.seed or 42))
    model = get_model(ft_cfg).to(torch.device('cpu'))
    latest_sess_ckpt, current_epoch = _find_latest_checkpoint(run_dir)
    if latest_sess_ckpt is not None and current_epoch > 0:
        load_model_weights(
            model, latest_sess_ckpt, device='cpu', allow_new_embeddings=False
        )
        model._newly_initialized_embeddings = {  # noqa: SLF001
            ('shared_embeddings_fc', 'local_forcing')
        }
    else:
        current_epoch = 0
        load_model_weights(
            model, base_ckpt, device='cpu', allow_new_embeddings=True
        )

    ft_spec = _build_finetune_modules_spec(finetune_modules, model)
    unfrozen_parts = _apply_freeze_spec(model, ft_spec)
    ft_cfg.update_config({'finetune_modules': ft_spec})
    cfg_yml_path = run_dir / 'config.yml'
    if cfg_yml_path.exists():
        cfg_yml_path.unlink()
    ft_cfg.dump_config(folder=run_dir)

    loss_obj = get_loss_obj(ft_cfg)
    if ft_cfg.regularization:
        loss_obj.set_regularization_terms(get_regularization_obj(cfg=ft_cfg))

    optimizer = get_optimizer(model, ft_cfg)
    lr_scheduler = _create_lr_scheduler(optimizer, lr_strat_val, ft_cfg)

    sf_dates, sf_mat = loaded['streamflow']
    sf_indexer = sf_dates.get_indexer(common_dates)
    obs_series = np.full(len(common_dates), np.nan, dtype=np.float32)
    valid_obs = sf_indexer >= 0
    obs_series[valid_obs] = sf_mat[sf_indexer[valid_obs], 0]

    train_mask = (
        common_dates >= pd.Timestamp(partition['train_start_date'])
    ) & (common_dates <= pd.Timestamp(partition['train_end_date']))
    val_mask = (
        common_dates >= pd.Timestamp(partition['validation_start_date'])
    ) & (common_dates <= pd.Timestamp(partition['validation_end_date']))

    epoch_history: list[dict[str, Any]] = list(
        session_state.get('epoch_history', [])
    )

    if current_epoch == 0 and not epoch_history:
        rec0, _ = _evaluate_and_build_epoch_record(
            0,
            'Epoch 0 (Pretrained Initial)',
            model,
            train_eval_ds,
            val_ds,
            loss_obj,
            train_ds.scaler,
            common_dates,
            obs_series,
            train_mask,
            val_mask,
            lead_time=lead_time,
            learning_rate=lr_val,
            unfrozen_parts=unfrozen_parts,
        )
        epoch_history.append(rec0)

    train_loader = MultimetDataLoader(
        train_ds,
        lazy_load=False,
        logging_level=logging.INFO,
        batch_size=bs_val,
        shuffle=True,
        num_workers=0,
        collate_fn=train_ds.collate_fn,
    )

    for step_idx in range(1, epochs + 1):
        target_epoch = current_epoch + step_idx
        mean_tr_loss = _train_single_epoch(
            model,
            train_loader,
            optimizer,
            loss_obj,
            ft_cfg.clip_gradient_norm,
        )

        ckpt_out = run_dir / f'model_epoch{target_epoch:03d}.pt'
        torch.save(model.state_dict(), ckpt_out)

        active_lr = float(optimizer.param_groups[0]['lr'])
        rec_ep, val_loss = _evaluate_and_build_epoch_record(
            target_epoch,
            f'Epoch {target_epoch}',
            model,
            train_eval_ds,
            val_ds,
            loss_obj,
            train_ds.scaler,
            common_dates,
            obs_series,
            train_mask,
            val_mask,
            lead_time=lead_time,
            learning_rate=active_lr,
            unfrozen_parts=unfrozen_parts,
            override_train_loss=mean_tr_loss,
        )
        _step_lr_scheduler(lr_scheduler, val_loss)
        epoch_history.append(rec_ep)

    new_current_epoch = current_epoch + epochs
    session_state = {
        'basin_id': basin_id,
        'model_id': model_id,
        'current_epoch': new_current_epoch,
        'status': 'paused',
        'split_ratio': partition['split_ratio'],
        'split_date': partition['split_date'],
        'updated_at': datetime.now(UTC).isoformat(),
        'last_hyperparameters': {
            'epochs_per_step': epochs,
            'learning_rate': lr_val,
            'loss': loss_val,
            'learning_rate_strategy': lr_strat_val,
            'batch_size': bs_val,
            'finetune_modules': unfrozen_parts,
        },
        'epoch_history': epoch_history,
    }
    state_path.write_text(json.dumps(session_state, indent=2))

    return get_finetune_workspace_status(
        username, basin_id, model_id, split_ratio=partition['split_ratio']
    )
