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

import dataclasses
import io
import json
import logging
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import xarray as xr

from frontend.profile_manager import get_profile_manager
from model.datasetzoo import get_dataset
from model.datasetzoo.multimet import MultimetDataLoader
from model.modelzoo import (
    BaseModel,
    get_model,
    get_model_class,
    load_model_weights,
)
from model.training import get_loss_obj, get_optimizer, get_regularization_obj
from model.utils.config import Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from model.datasetzoo.basedataset import BaseDataset

    from model.datautils.scaler import Scaler
    from model.training.loss import BaseLoss

REPO_ROOT = Path(__file__).resolve().parent.parent

_METRIC_EPS = 1e-12
_MIN_CSV_COLUMNS = 2
_MIN_SAMPLE_DAYS = 4
_MIN_METRIC_POINTS = 2
_MAX_EPOCHS_PER_STEP = 200
_MIN_SEQ_LENGTH = 8
_MIN_SPLIT_RATIO = 0.2
_MAX_SPLIT_RATIO = 0.9
_M3S_TO_MM_DAY_FACTOR = 86.4

_DATE_COLUMN_ALIASES = frozenset(
    {'date', 'time', 'datetime', 'timestamp', 'issue_date'}
)
_BASIN_COLUMN_ALIASES = frozenset(
    {'basin', 'basin_id', 'catchment_id', 'gauge_id', 'site_no', 'station_id'}
)
_REALTIME_PRECIP_CANDIDATES: tuple[tuple[str, str], ...] = (
    ('HRES', 'hres_total_precipitation'),
    ('HRES', 'hres_tp'),
    ('GFS', 'gfs_total_precipitation'),
    ('GFS', 'gfs_tp'),
    ('NOAA_GFS', 'gfs_total_precipitation'),
    ('NOAA_GFS', 'gfs_tp'),
    ('AIFS', 'aifs_total_precipitation'),
    ('AIFS', 'aifs_tp'),
    ('ECMWF_AIFS', 'aifs_total_precipitation'),
    ('ECMWF_AIFS', 'aifs_tp'),
    ('IMERG', 'imerg_precipitation'),
    ('DYNAMICAL_IMERG', 'imerg_precipitation'),
    ('CPC', 'cpc_precipitation'),
    ('CPC', 'cpc_precip'),
    ('ERA5_LAND', 'era5land_total_precipitation_sum'),
    ('ERA5', 'era5_total_precipitation'),
)
_REALTIME_TEMP_CANDIDATES: tuple[tuple[str, str], ...] = (
    ('HRES', 'hres_temperature_2m'),
    ('HRES', 'hres_2t'),
    ('GFS', 'gfs_temperature_2m'),
    ('GFS', 'gfs_2t'),
    ('NOAA_GFS', 'gfs_temperature_2m'),
    ('NOAA_GFS', 'gfs_2t'),
    ('AIFS', 'aifs_temperature_2m'),
    ('AIFS', 'aifs_2t'),
    ('ECMWF_AIFS', 'aifs_temperature_2m'),
    ('ECMWF_AIFS', 'aifs_2t'),
    ('ERA5_LAND', 'era5land_temperature_2m'),
    ('ERA5', 'era5_temperature_2m'),
)
SUPPORTED_FORCING_MODES: tuple[str, ...] = (
    'global_only',
    'global_and_local',
    'local_only',
)
_NA_STRINGS = frozenset({'', 'nan', 'null', 'none', 'na', 'n/a'})
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
_PRECIP_UNITS = frozenset({'mm/day', 'mm/d', 'mm'})
_TEMP_UNITS = frozenset({'degc', '°c', 'c', 'celsius'})
_STREAMFLOW_MM_UNITS = frozenset({'mm/day', 'mm/d', 'mm'})
_STREAMFLOW_CMS_UNITS = frozenset({'m3/s', 'cms', 'm³/s'})

PRETRAINED_MODEL_DIRS: tuple[tuple[str, str, Path], ...] = (
    (
        'google-floodhub-settings-110-epochs',
        'Google FloodHub Pretrained (With GraphCast, 110 Epochs)',
        REPO_ROOT
        / 'model'
        / 'pretrained-models'
        / 'google-floodhub-settings-110-epochs',
    ),
    (
        'google-floodhub-settings-no-graphcast-110-epochs',
        'Google FloodHub Pretrained (Without GraphCast, 110 Epochs)',
        REPO_ROOT
        / 'model'
        / 'pretrained-models'
        / 'google-floodhub-settings-no-graphcast-110-epochs',
    ),
    (
        'google-floodhub-settings-aifs-110-epochs',
        'Google FloodHub Pretrained (With ECMWF AIFS, 110 Epochs)',
        REPO_ROOT
        / 'model'
        / 'pretrained-models'
        / 'google-floodhub-settings-aifs-110-epochs',
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
    'ConstantLR',
    'StepLR',
    'ReduceLROnPlateau',
)

_SAFE_ID_RE = re.compile(r'[^a-zA-Z0-9._-]+')
_LEAD_COL_RE = re.compile(
    r'^(?:lead[_\s-]*|t\+|d\+|day[_\s-]*|\+)?([1-9]\d*)(?:d|day|days)?$',
    re.IGNORECASE,
)
_LEAD_LIKE_PREFIX_RE = re.compile(
    r'^(?:lead|t\+|d\+|day|\+|-?\d)', re.IGNORECASE
)
_CATCHMENT_PREFIX_STRIP_RE = re.compile(
    r'^catchment_(?:hydroatlas|merit(?:_hydro)?)_[^\s—-]+\s*[—-]\s*',
    re.IGNORECASE,
)
_CATCHMENT_ID_PREFIX_RE = re.compile(
    r'^catchment_(?:hydroatlas|merit_hydro|merit)_',
    re.IGNORECASE,
)
_AUTO_WRAPPER_NAME_RE = re.compile(
    r'^(?:Delineated Catchment|Uploaded Watershed)\s*\((catchment_.*)\)$',
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class _FinetuneConfigParams:
    """Parameters used to construct a per-basin fine-tuning ``Config``."""

    base_cfg: Config
    base_run_dir: Path
    run_dir: Path
    zarr_root: Path
    global_dyn_dir: Path
    attr_path: Path
    partition: dict[str, object]
    lead_time: int
    lr_val: float
    loss_val: str
    lr_strat_val: str
    bs_val: int
    include_local_forcing: bool = True


@dataclasses.dataclass(frozen=True)
class _EpochEvalContext:
    """Context bundle for evaluating train/validation hydrograph metrics."""

    model: torch.nn.Module
    train_eval_ds: BaseDataset
    val_ds: BaseDataset
    loss_obj: BaseLoss
    scaler: Scaler
    common_dates: pd.DatetimeIndex
    obs_series: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    lead_time: int
    unfrozen_parts: list[str]


def _require_safe_id(value: str, field_name: str) -> str:
    raw = (value or '').strip()
    if not raw:
        raise ValueError(f'{field_name} must be a non-empty string.')
    cleaned = _SAFE_ID_RE.sub('_', raw).strip('._-')
    if not cleaned:
        raise ValueError(
            f'{field_name} {value!r} contains no valid identifier characters.'
        )
    return cleaned


def _validate_finite_float(
    value: object,
    field_name: str,
    *,
    strictly_positive: bool = False,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f'{field_name} must be a finite number.')
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f'{field_name} must be a finite number (got {value}).')
    if strictly_positive and fval <= 0.0:
        raise ValueError(f'{field_name} must be > 0 (got {fval}).')
    if min_val is not None and fval < min_val:
        raise ValueError(f'{field_name} must be >= {min_val} (got {fval}).')
    if max_val is not None and fval > max_val:
        raise ValueError(f'{field_name} must be <= {max_val} (got {fval}).')
    return fval


def _validate_positive_int(
    value: object,
    field_name: str,
    *,
    min_val: int = 1,
    max_val: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{field_name} must be an integer.')
    if value < min_val:
        raise ValueError(f'{field_name} must be >= {min_val} (got {value}).')
    if max_val is not None and value > max_val:
        raise ValueError(f'{field_name} must be <= {max_val} (got {value}).')
    return value


def _session_root(username: str, basin_id: str, model_id: str) -> Path:
    profile_dir = get_profile_manager().get_profile_dir(username)
    safe_basin = _require_safe_id(basin_id, 'basin_id')
    safe_model = _require_safe_id(model_id, 'model_id')
    return profile_dir / 'finetune_workspace' / safe_basin / safe_model


def _basin_data_root(username: str, basin_id: str) -> Path:
    profile_dir = get_profile_manager().get_profile_dir(username)
    safe_basin = _require_safe_id(basin_id, 'basin_id')
    return profile_dir / 'finetune_data' / safe_basin


def _parse_checkpoint_epoch(path: Path) -> int:
    match = re.search(r'model_epoch(\d+)\.pt$', path.name)
    return int(match.group(1)) if match else -1


def _find_latest_checkpoint(run_dir: Path) -> tuple[Path | None, int]:
    candidates = [
        (_parse_checkpoint_epoch(p), p)
        for p in run_dir.glob('model_epoch*.pt')
        if _parse_checkpoint_epoch(p) >= 0
    ]
    if not candidates:
        return None, 0
    candidates.sort(key=lambda item: (item[0], item[1].name))
    epoch, latest = candidates[-1]
    return latest, epoch


def _extract_product_groups(
    inputs_cfg: dict[str, list[str]] | list[str] | list[list[str]] | None,
) -> list[str]:
    if isinstance(inputs_cfg, dict):
        return list(inputs_cfg.keys())
    if isinstance(inputs_cfg, list):
        groups: list[str] = []
        for entry in inputs_cfg:
            items = entry if isinstance(entry, list) else [entry]
            for feat in items:
                prefix = str(feat).split('_', 1)[0]
                if prefix and prefix not in groups:
                    groups.append(prefix)
        return groups
    return []


def _normalize_lr_strategy_name(raw_strategy: object) -> str:
    text = str(raw_strategy or '').strip()
    if not text or text.lower() == 'constant':
        return 'ConstantLR'
    for canonical in SUPPORTED_LR_STRATEGIES:
        if text.lower() == canonical.lower():
            return canonical
    return 'ConstantLR'


def _default_head_component_name(component_names: Sequence[str]) -> str:
    if 'head' in component_names:
        return 'head'
    if 'forecast_head' in component_names:
        return 'forecast_head'
    return component_names[-1] if component_names else 'head'


def _inspect_model_from_run_dir(
    model_id: str,
    label: str,
    run_dir: Path,
    *,
    extra_local_products: dict[str, list[str]] | None = None,
) -> dict[str, object]:
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
    if extra_local_products and isinstance(
        cfg_dict.get('hindcast_inputs'), dict
    ):
        hindcast_in = dict(cfg_dict.get('hindcast_inputs', {}))
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
    if isinstance(model, BaseModel) and newly_init:
        model.set_newly_initialized_embeddings(newly_init)

    components = (
        model.describe_tunable_components()
        if isinstance(model, BaseModel)
        else []
    )
    total_params = sum(p.numel() for p in model.parameters())

    head_name = str(cfg.head).lower()
    allowed_losses = (
        ['CMALLoss']
        if head_name in ('cmal', 'cmal_deterministic')
        else ['MSE', 'NSE', 'RMSE', 'WMSE']
    )
    default_loss = (
        'CMALLoss'
        if head_name in ('cmal', 'cmal_deterministic')
        else str(cfg.loss).upper()
    )
    tunable_names = model_cls.get_tunable_component_names()

    return {
        'model_id': model_id,
        'label': label,
        'run_dir': str(run_dir),
        'checkpoint_path': str(ckpt_path),
        'checkpoint_epoch': ckpt_epoch,
        'architecture': str(cfg.model),
        'architecture_class': model_cls.__name__,
        'head': str(cfg.head),
        'default_loss': default_loss,
        'allowed_losses': allowed_losses,
        'supported_lr_strategies': list(SUPPORTED_LR_STRATEGIES),
        'default_learning_rate': float(cfg.initial_learning_rate or 5e-4),
        'default_lr_strategy': _normalize_lr_strategy_name(
            getattr(cfg, 'learning_rate_strategy', None)
        ),
        'default_batch_size': int(cfg.batch_size or 32),
        'seq_length': int(cfg.seq_length),
        'lead_time': int(cfg.lead_time),
        'forecast_overlap': int(cfg.forecast_overlap or cfg.seq_length),
        'predict_last_n': int(cfg.predict_last_n or cfg.lead_time),
        'hidden_size': int(cfg.hidden_size),
        'total_parameters': int(total_params),
        'tunable_component_names': tunable_names,
        'tunable_components': components,
        'base_hindcast_products': _extract_product_groups(cfg.hindcast_inputs),
        'base_forecast_products': _extract_product_groups(cfg.forecast_inputs),
        'static_attributes': list(cfg.static_attributes or []),
        'target_variables': list(cfg.target_variables or ['streamflow']),
    }


def list_finetune_models(
    username: str = 'guest',
) -> list[dict[str, object]]:
    """Return inventory of base models and dynamically inspected components."""
    inventory: list[dict[str, object]] = []
    local_override = {
        'local_forcing': ['local_precipitation', 'local_temperature']
    }

    for model_id, label, run_dir in PRETRAINED_MODEL_DIRS:
        ckpt_path, _ = _find_latest_checkpoint(run_dir)
        if (run_dir / 'config.yml').exists() and ckpt_path is not None:
            info = _inspect_model_from_run_dir(
                model_id,
                label,
                run_dir,
                extra_local_products=local_override,
            )
            inventory.append(info)

    user_runs_dir = get_profile_manager().get_model_runs_dir(username)
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
    model_id: str, username: str = 'guest'
) -> tuple[str, Path]:
    clean_id = (model_id or '').strip()
    if not clean_id:
        raise ValueError('model_id must be a non-empty string.')

    for mid, label, run_dir in PRETRAINED_MODEL_DIRS:
        ckpt_path, _ = _find_latest_checkpoint(run_dir)
        if (
            mid == clean_id
            and (run_dir / 'config.yml').exists()
            and ckpt_path is not None
        ):
            return label, run_dir

    if clean_id.startswith('user_'):
        run_name = clean_id.removeprefix('user_')
        if (
            not run_name
            or '/' in run_name
            or '\\' in run_name
            or '..' in run_name
        ):
            raise ValueError(f'Invalid user model_id: {model_id!r}')
        runs_dir = get_profile_manager().get_model_runs_dir(username).resolve()
        candidate = (runs_dir / run_name).resolve()
        if candidate.parent != runs_dir:
            raise ValueError(f'Invalid user model_id: {model_id!r}')
        if (candidate / 'config.yml').is_file():
            return f'User Model: {run_name}', candidate

    raise FileNotFoundError(f'Unknown model_id: {model_id}')


def _identify_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() in _DATE_COLUMN_ALIASES:
            return str(col)
    raise ValueError(
        'CSV must contain an explicit date column named one of '
        f'{sorted(_DATE_COLUMN_ALIASES)}.'
    )


def _identify_basin_column(df: pd.DataFrame, date_col: str) -> str | None:
    for col in df.columns:
        if col == date_col:
            continue
        if str(col).strip().lower() in _BASIN_COLUMN_ALIASES:
            return str(col)
    return None


def _list_csv_basins(csv_content: str) -> list[str]:
    text = (csv_content or '').strip()
    if not text:
        return []
    try:
        df = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        return []
    if df.empty or len(df.columns) < _MIN_CSV_COLUMNS:
        return []
    try:
        date_col = _identify_date_column(df)
    except ValueError:
        return []
    basin_col = _identify_basin_column(df, date_col)
    if basin_col is None:
        return []
    basin_series = df[basin_col].astype(str).str.strip()
    return [
        b
        for b in dict.fromkeys(basin_series.tolist())
        if b and b.lower() not in _NA_STRINGS
    ]


def _validate_numeric_column(series: pd.Series, col_name: str) -> None:
    str_vals = series.dropna().astype(str).str.strip()
    non_empty_mask = ~str_vals.str.lower().isin(_NA_STRINGS)
    non_empty_vals = str_vals.loc[non_empty_mask]
    if non_empty_vals.empty:
        return
    coerced = pd.to_numeric(non_empty_vals, errors='coerce')
    if coerced.isna().any():
        bad_sample = str(non_empty_vals.loc[coerced.isna()].iloc[0])
        raise ValueError(
            f"Column '{col_name}' contains non-numeric value {bad_sample!r}."
        )


def _filter_dataframe_by_basin(
    df: pd.DataFrame, basin_col: str, basin_id: str | None
) -> pd.DataFrame:
    basin_series = df[basin_col].astype(str).str.strip()
    unique_basins = [
        b
        for b in dict.fromkeys(basin_series.tolist())
        if b and b.lower() not in _NA_STRINGS
    ]
    target_b = (basin_id or '').strip()
    if target_b and target_b in unique_basins:
        return df.loc[basin_series == target_b].copy()
    if len(unique_basins) == 1:
        return df.loc[basin_series == unique_basins[0]].copy()
    if len(unique_basins) > 1:
        if target_b:
            raise ValueError(
                f"Basin '{target_b}' was not found in CSV column "
                f"'{basin_col}'. Available basins: {unique_basins[:10]}"
            )
        raise ValueError(
            f'Multi-basin CSV contains {len(unique_basins)} basins '
            f'({unique_basins[:5]}); specify basin_id to select one.'
        )
    return df


def _parse_and_validate_daily_dates(
    df: pd.DataFrame, date_col: str
) -> pd.Series:
    raw_dates = pd.to_datetime(df[date_col], errors='coerce', utc=False)
    if raw_dates.isna().any():
        bad_count = int(raw_dates.isna().sum())
        raise ValueError(
            f"Column '{date_col}' contains {bad_count} unparseable date "
            'row(s). Expected ISO dates (YYYY-MM-DD).'
        )
    if hasattr(raw_dates.dt, 'tz') and raw_dates.dt.tz is not None:
        raw_dates = raw_dates.dt.tz_localize(None)
    if (raw_dates != raw_dates.dt.floor('D')).any():
        raise ValueError(
            'Sub-daily timestamps are not supported; expected daily dates '
            '(YYYY-MM-DD).'
        )
    if raw_dates.duplicated().any():
        raise ValueError(
            'CSV contains duplicate dates; each daily timestamp must be unique.'
        )
    return raw_dates


def _extract_daily_dataframe(
    csv_content: str,
    *,
    basin_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    text = (csv_content or '').strip()
    if not text:
        raise ValueError('Uploaded CSV content is empty.')

    df = pd.read_csv(io.StringIO(text))
    if df.empty or len(df.columns) < _MIN_CSV_COLUMNS:
        raise ValueError(
            'CSV must contain a date/time column and at least one data column.'
        )

    date_col = _identify_date_column(df)
    basin_col = _identify_basin_column(df, date_col)
    if basin_col is not None:
        df = _filter_dataframe_by_basin(df, basin_col, basin_id)

    raw_dates = _parse_and_validate_daily_dates(df, date_col)

    excluded_cols = {date_col, '__parsed_date__'}
    if basin_col is not None:
        excluded_cols.add(basin_col)
    data_cols = [str(c) for c in df.columns if c not in excluded_cols]
    if not data_cols:
        raise ValueError('CSV contains no numeric data columns.')
    for col in data_cols:
        _validate_numeric_column(df[col], col)

    df = df.copy()
    df['__parsed_date__'] = raw_dates
    df = df.sort_values('__parsed_date__')

    min_date = df['__parsed_date__'].iloc[0]
    max_date = df['__parsed_date__'].iloc[-1]
    full_dates = pd.date_range(min_date, max_date, freq='D')
    df = df.set_index('__parsed_date__').reindex(full_dates)
    return df, full_dates, data_cols


def _extract_lead_columns(
    df: pd.DataFrame, data_cols: list[str]
) -> dict[int, np.ndarray]:
    lead_map: dict[int, np.ndarray] = {}
    if len(data_cols) == 1:
        col = data_cols[0]
        col_str = str(col).strip()
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(
            dtype=np.float32
        )
        match = _LEAD_COL_RE.match(col_str)
        if match:
            lead_idx = int(match.group(1))
        elif _LEAD_LIKE_PREFIX_RE.match(col_str):
            raise ValueError(
                f"Invalid lead column name '{col_str}'. Expected 'lead_1' .. "
                "'lead_N' with positive daily lead indices."
            )
        else:
            lead_idx = 1
        lead_map[lead_idx] = vals
        return lead_map

    unmatched: list[str] = []
    for col in data_cols:
        col_str = str(col).strip()
        match = _LEAD_COL_RE.match(col_str)
        if not match:
            unmatched.append(col_str)
            continue
        lead_idx = int(match.group(1))
        if lead_idx in lead_map:
            raise ValueError(
                f'Duplicate column for lead_{lead_idx} in uploaded CSV.'
            )
        lead_map[lead_idx] = pd.to_numeric(df[col], errors='coerce').to_numpy(
            dtype=np.float32
        )
    if unmatched:
        raise ValueError(
            f'Unrecognized lead column names {unmatched} in multi-column CSV. '
            "Expected 'lead_1', 'lead_2', ..."
        )
    return lead_map


def parse_time_by_lead_csv(
    csv_content: str,
    *,
    target_lead_time: int = 7,
    basin_id: str | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, list[int]]:
    """Parse a CSV file structured as ``time`` by ``lead_time``.

    Parameters
    ----------
    csv_content : str
        Raw CSV text uploaded by the user.
    target_lead_time : int, optional
        Number of forecast lead steps expected by the model (default 7).
    basin_id : str | None, optional
        Target basin identifier when filtering a multi-basin CSV.

    Returns
    -------
    tuple[pd.DatetimeIndex, np.ndarray, list[int]]
        - ``dates``: Sorted daily ``DatetimeIndex`` of length ``T``.
        - ``matrix``: Float32 array of shape ``(T, target_lead_time)`` where
          un-uploaded leads and missing values remain ``np.nan``.
        - ``parsed_leads``: Sorted list of positive integer lead times found.
    """
    target_leads = _validate_positive_int(
        target_lead_time, 'target_lead_time', min_val=1
    )
    df, full_dates, data_cols = _extract_daily_dataframe(
        csv_content, basin_id=basin_id
    )
    lead_map = _extract_lead_columns(df, data_cols)

    num_dates = len(full_dates)
    matrix = np.full((num_dates, target_leads), np.nan, dtype=np.float32)
    parsed_leads = sorted(lead_map.keys())
    for k in parsed_leads:
        if 1 <= k <= target_leads:
            matrix[:, k - 1] = lead_map[k]

    return full_dates, matrix, parsed_leads


def _resolve_upload_key_and_units(
    dtype_clean: str,
    matrix: np.ndarray,
    units: str | None,
    area_km2: float | None,
) -> tuple[str, np.ndarray, str]:
    u_clean = (units or '').strip().lower()
    if dtype_clean in ('precip', 'precipitation'):
        if u_clean and u_clean not in _PRECIP_UNITS:
            raise ValueError(
                f"Unsupported precipitation units '{units}'. Expected 'mm/day'."
            )
        if np.nanmin(matrix) < 0.0:
            raise ValueError('Precipitation values cannot be negative.')
        return 'precip', matrix, 'mm/day'

    if dtype_clean in ('temp', 'temperature'):
        if u_clean and u_clean not in _TEMP_UNITS:
            raise ValueError(
                f"Unsupported temperature units '{units}'. Expected 'degC'."
            )
        return 'temp', matrix, 'degC'

    if np.nanmin(matrix) < 0.0:
        raise ValueError('Streamflow values cannot be negative.')

    if not u_clean or u_clean in _STREAMFLOW_MM_UNITS:
        return 'streamflow', matrix, 'mm/day'

    if u_clean in _STREAMFLOW_CMS_UNITS:
        if area_km2 is None:
            raise ValueError(
                'Basin area_km2 (> 0) is required to convert streamflow '
                'from m³/s to mm/day.'
            )
        valid_area = _validate_finite_float(
            area_km2, 'area_km2', strictly_positive=True
        )
        converted = (matrix * (_M3S_TO_MM_DAY_FACTOR / valid_area)).astype(
            np.float32
        )
        return 'streamflow', converted, 'mm/day'

    raise ValueError(
        f"Unsupported streamflow units '{units}'. Expected 'mm/day' or 'm3/s'."
    )


def _save_single_basin_finetune_csv(  # noqa: PLR0913
    username: str,
    basin_id: str,
    data_type: str,
    dtype_clean: str,
    csv_content: str,
    *,
    filename: str,
    units: str | None,
    area_km2: float | None,
    target_lead_time: int,
) -> dict[str, object]:
    dates, matrix, parsed_leads = parse_time_by_lead_csv(
        csv_content,
        target_lead_time=target_lead_time,
        basin_id=basin_id,
    )

    if not np.isfinite(matrix[:, 0]).any():
        raise ValueError(
            f'Uploaded {data_type} CSV contains no finite values '
            'in the primary lead_1 column.'
        )

    data_root = _basin_data_root(username, basin_id)
    raw_dir = data_root / 'raw_arrays'
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_root / 'upload_meta.json'
    meta: dict[str, object] = (
        json.loads(meta_path.read_text(encoding='utf-8'))
        if meta_path.exists()
        else {}
    )

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
    entry_meta: dict[str, object] = {
        'basin_id': basin_id,
        'filename': filename or f'{key}.csv',
        'uploaded_at': datetime.now(UTC).isoformat(),
        'start_date': str(dates[0].strftime('%Y-%m-%d')),
        'end_date': str(dates[-1].strftime('%Y-%m-%d')),
        'num_days': len(dates),
        'num_valid_days': int(finite_mask.sum()),
        'num_lead_columns': len(parsed_leads),
        'parsed_leads': parsed_leads,
        'units': canonical_units,
        'original_units': units or canonical_units,
        'area_km2': area_km2,
        'min_val': float(np.nanmin(matrix[:, 0])),
        'max_val': float(np.nanmax(matrix[:, 0])),
        'mean_val': float(np.nanmean(matrix[:, 0])),
    }
    meta[key] = entry_meta
    meta['basin_id'] = basin_id
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    _rebuild_local_zarr_stores(
        username, basin_id, target_lead_time=target_lead_time
    )
    return entry_meta


def upload_finetune_csv(
    username: str,
    basin_id: str,
    data_type: str,
    csv_content: str,
    **options: object,
) -> dict[str, object]:
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
    **options : object
        Optional keyword arguments:
        - ``filename`` (``str``): Original filename for display metadata.
        - ``units`` (``str | None``): Physical units of the uploaded data.
        - ``area_km2`` (``float | None``): Basin drainage area in km² when
          converting streamflow from ``m3/s`` to ``mm/day``.
        - ``target_lead_time`` (``int``): Number of lead times (default 7).
    """
    allowed_opts = {'filename', 'units', 'area_km2', 'target_lead_time'}
    unknown_opts = set(options) - allowed_opts
    if unknown_opts:
        raise TypeError(
            f'Unexpected keyword argument(s): {sorted(unknown_opts)}'
        )

    filename = str(options.get('filename') or '')
    raw_units = options.get('units')
    units = str(raw_units) if raw_units is not None else None
    raw_area = options.get('area_km2')
    raw_lead = options.get('target_lead_time', 7)
    target_lead_time = _validate_positive_int(
        raw_lead, 'target_lead_time', min_val=1
    )

    dtype_clean = (data_type or '').strip().lower()
    if dtype_clean not in _SUPPORTED_DATA_TYPES:
        raise ValueError(
            f"Unsupported data_type '{data_type}'. "
            "Must be 'precip', 'temp', or 'streamflow'."
        )

    csv_basins = _list_csv_basins(csv_content)
    clipped_ids = {
        str(b['basin_id']) for b in list_user_clipped_basins(username)
    }

    if len(csv_basins) > 1:
        target_basins = [
            b for b in csv_basins if b in clipped_ids or b == basin_id
        ] or list(csv_basins)
        results_by_basin: dict[str, dict[str, object]] = {}
        for b_id in target_basins:
            b_area = (
                _validate_finite_float(
                    raw_area, 'area_km2', strictly_positive=True
                )
                if (raw_area is not None and b_id == basin_id)
                else _resolve_basin_area_km2(username, b_id)
            )
            results_by_basin[b_id] = _save_single_basin_finetune_csv(
                username,
                b_id,
                data_type,
                dtype_clean,
                csv_content,
                filename=filename,
                units=units,
                area_km2=b_area,
                target_lead_time=target_lead_time,
            )
        primary_b = (
            basin_id if basin_id in results_by_basin else target_basins[0]
        )
        primary_meta = dict(results_by_basin[primary_b])
        primary_meta['basins_uploaded'] = target_basins
        return primary_meta

    if (
        len(csv_basins) == 1
        and csv_basins[0] != basin_id
        and csv_basins[0] in clipped_ids
    ):
        basin_id = csv_basins[0]
        raw_area = None

    area_km2 = (
        _validate_finite_float(raw_area, 'area_km2', strictly_positive=True)
        if raw_area is not None
        else _resolve_basin_area_km2(username, basin_id)
    )
    entry_meta = _save_single_basin_finetune_csv(
        username,
        basin_id,
        data_type,
        dtype_clean,
        csv_content,
        filename=filename,
        units=units,
        area_km2=area_km2,
        target_lead_time=target_lead_time,
    )
    entry_meta['basins_uploaded'] = [basin_id]
    return entry_meta


def clear_finetune_data(username: str, basin_id: str) -> dict[str, object]:
    """Delete uploaded local CSVs and fine-tuning sessions for ``basin_id``."""
    data_root = _basin_data_root(username, basin_id)
    if data_root.exists():
        shutil.rmtree(data_root)
    profile_dir = get_profile_manager().get_profile_dir(username)
    safe_basin = _require_safe_id(basin_id, 'basin_id')
    ws_dir = profile_dir / 'finetune_workspace' / safe_basin
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
    return {'status': 'cleared', 'basin_id': basin_id}


def _read_zarr_basin_lead_matrix(
    zpath: Path,
    basin_id: str,
    var_name: str,
    target_lead_time: int,
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    if not _is_valid_zarr_dir(zpath):
        return None
    ds = xr.open_zarr(str(zpath), decode_timedelta=False)
    basin_coord = next(
        (
            c
            for c in ('basin', 'basin_id', 'catchment_id', 'gauge_id')
            if c in ds.coords
        ),
        None,
    )
    if (
        basin_coord is None
        or 'date' not in ds.coords
        or var_name not in ds.data_vars
    ):
        return None
    basin_vals = [str(b) for b in ds.coords[basin_coord].to_numpy()]
    if basin_id not in basin_vals:
        return None
    sub = ds[var_name].sel({basin_coord: basin_id}).load()
    raw_dates = pd.to_datetime(sub.coords['date'].to_numpy())
    dates = (
        pd.DatetimeIndex(raw_dates).tz_localize(None).floor('D')
        if getattr(raw_dates, 'tz', None) is not None
        else pd.DatetimeIndex(raw_dates).floor('D')
    )
    if len(dates) == 0:
        return None
    vals = np.asarray(sub.to_numpy(), dtype=np.float32)
    mat = np.full((len(dates), target_lead_time), np.nan, dtype=np.float32)
    if vals.ndim == 2:  # noqa: PLR2004
        cols = min(vals.shape[1], target_lead_time)
        mat[:, :cols] = vals[:, :cols]
    elif vals.ndim == 1:
        mat[:, 0] = vals
    if not np.isfinite(mat[:, 0]).any():
        return None
    lead0 = mat[:, 0]
    idx_base = np.arange(len(dates))
    for k in range(1, target_lead_time):
        nan_mask = ~np.isfinite(mat[:, k])
        if nan_mask.any():
            shifted_idx = np.minimum(idx_base + k, len(dates) - 1)
            mat[nan_mask, k] = lead0[shifted_idx[nan_mask]]
    return dates, mat


_PRODUCT_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    'HRES': ('HRES', 'ECMWF_HRES'),
    'AIFS': ('AIFS', 'ECMWF_AIFS'),
    'IMERG': ('IMERG', 'DYNAMICAL_IMERG'),
    'CPC': ('CPC',),
    'GFS': ('GFS', 'NOAA_GFS'),
    'ERA5': ('ERA5', 'ERA5_LAND'),
    'GRAPHCAST': ('GRAPHCAST',),
}


def _profile_dynamics_dirs(username: str) -> tuple[Path, ...]:
    pm = get_profile_manager()
    profile_dir = pm.get_profile_dir(username)
    return (
        pm.get_realtime_dir(username) / 'dynamics',
        profile_dir / 'realtime' / 'dynamics',
        profile_dir / 'dynamics',
    )


def _extract_realtime_forcing_matrix(
    username: str,
    basin_id: str,
    candidates: Sequence[tuple[str, str]],
    target_lead_time: int = 7,
) -> tuple[pd.DatetimeIndex, np.ndarray, str] | None:
    for base_dir in _profile_dynamics_dirs(username):
        if not base_dir.is_dir():
            continue
        for prod_name, var_name in candidates:
            zpath = base_dir / prod_name / 'timeseries.zarr'
            res = _read_zarr_basin_lead_matrix(
                zpath, basin_id, var_name, target_lead_time
            )
            if res is not None:
                return res[0], res[1], prod_name
    return None


def _attach_overlapping_realtime_forcing(
    loaded: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    username: str,
    basin_id: str,
    target_lead_time: int,
) -> None:
    if 'streamflow' not in loaded:
        return
    sf_dates = loaded['streamflow'][0]
    sf_start, sf_end = sf_dates[0], sf_dates[-1]
    for key, candidates in (
        ('precip', _REALTIME_PRECIP_CANDIDATES),
        ('temp', _REALTIME_TEMP_CANDIDATES),
    ):
        if key in loaded:
            continue
        rt = _extract_realtime_forcing_matrix(
            username,
            basin_id,
            candidates,
            target_lead_time=target_lead_time,
        )
        if rt is not None and max(sf_start, rt[0][0]) < min(sf_end, rt[0][-1]):
            loaded[key] = (rt[0], rt[1])


def _load_raw_uploaded_arrays(
    username: str,
    basin_id: str,
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


def _resolve_forcing_mode(
    requested_mode: str | None,
    *,
    has_local_forcing: bool,
    has_global_forcing: bool,
) -> str:
    clean = (requested_mode or '').strip().lower()
    if clean in SUPPORTED_FORCING_MODES:
        if clean == 'local_only' and not has_local_forcing:
            return 'global_only'
        return clean
    if has_local_forcing and has_global_forcing:
        return 'global_and_local'
    if has_local_forcing:
        return 'local_only'
    return 'global_only'


def _load_uploaded_arrays(
    username: str,
    basin_id: str,
    *,
    target_lead_time: int = 7,
    forcing_mode: str | None = None,
) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    raw = _load_raw_uploaded_arrays(username, basin_id)
    mode_clean = (forcing_mode or '').strip().lower()
    if mode_clean == 'local_only':
        return raw
    if mode_clean == 'global_only':
        loaded: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
        if 'streamflow' in raw:
            loaded['streamflow'] = raw['streamflow']
        _attach_overlapping_realtime_forcing(
            loaded, username, basin_id, target_lead_time
        )
        return loaded
    loaded = dict(raw)
    _attach_overlapping_realtime_forcing(
        loaded, username, basin_id, target_lead_time
    )
    return loaded


def _rebuild_local_zarr_stores(
    username: str,
    basin_id: str,
    *,
    target_lead_time: int = 7,
    forcing_mode: str | None = None,
) -> dict[str, object] | None:
    """Write ``LOCAL_FORCING/timeseries.zarr`` and ``streamflow.zarr``."""
    loaded = _load_uploaded_arrays(
        username,
        basin_id,
        target_lead_time=target_lead_time,
        forcing_mode=forcing_mode,
    )
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
    basin_file.write_text(f'{basin_id}\n', encoding='utf-8')

    return {
        'start_date': common_dates[0].strftime('%Y-%m-%d'),
        'end_date': common_dates[-1].strftime('%Y-%m-%d'),
        'num_days': num_dates,
    }


def _is_valid_zarr_dir(zarr_path: Path) -> bool:
    if not zarr_path.is_dir():
        return False
    return any(
        (zarr_path / marker).exists()
        for marker in ('.zmetadata', '.zgroup', 'zarr.json')
    )


def _read_statics_zarr_for_basin(
    statics_zarr: Path, basin_id: str
) -> dict[str, float]:
    if not _is_valid_zarr_dir(statics_zarr):
        return {}
    ds = xr.open_zarr(str(statics_zarr), decode_timedelta=False).load()
    basin_coord = next(
        (
            c
            for c in ('basin', 'basin_id', 'catchment_id', 'gauge_id')
            if c in ds.coords
        ),
        None,
    )
    if basin_coord is None:
        return {}
    basin_vals = [str(b) for b in ds.coords[basin_coord].to_numpy()]
    if basin_id not in basin_vals:
        return {}
    sub = ds.sel({basin_coord: basin_id})
    out: dict[str, float] = {}
    for var_name in ds.data_vars:
        val = float(np.asarray(sub[var_name].values).item())
        if math.isfinite(val):
            out[str(var_name)] = val
    return out


def _read_statics_csv_for_basin(
    csv_path: Path, basin_id: str
) -> dict[str, float]:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return {}
    df_csv = pd.read_csv(csv_path)
    id_col = next(
        (
            c
            for c in ('gauge_id', 'basin_id', 'catchment_id', 'basin', 'id')
            if c in df_csv.columns
        ),
        None,
    )
    if id_col is None:
        return {}
    matches = df_csv[df_csv[id_col].astype(str) == basin_id]
    if matches.empty:
        return {}
    row = matches.iloc[-1]
    out: dict[str, float] = {}
    for col in df_csv.columns:
        if col == id_col or pd.isna(row[col]):
            continue
        val_num = pd.to_numeric(pd.Series([row[col]]), errors='coerce')
        if not val_num.isna().iloc[0]:
            fval = float(val_num.iloc[0])
            if math.isfinite(fval):
                out[str(col)] = fval
    return out


def _load_profile_static_attributes_for_basin(
    username: str, basin_id: str
) -> dict[str, float]:
    """Load real static attributes for ``basin_id`` from the user profile."""
    pm = get_profile_manager()
    found = _read_statics_zarr_for_basin(
        pm.get_statics_zarr_path(username), basin_id
    )
    for csv_dir in (
        pm.get_caravan_attributes_dir(username),
        pm.get_attributes_dir(username),
    ):
        if csv_dir.is_dir():
            for csv_path in sorted(csv_dir.glob('*.csv')):
                found.update(_read_statics_csv_for_basin(csv_path, basin_id))
    return found


def clean_basin_display_name(
    raw_name: object,
    reach_attrs: object,
    basin_id: str,
) -> str:
    """Return a clean human-readable basin name without internal prefixes."""
    if isinstance(raw_name, str) and raw_name.strip():
        stripped = _CATCHMENT_PREFIX_STRIP_RE.sub('', raw_name.strip()).strip()
        if stripped and not _AUTO_WRAPPER_NAME_RE.match(stripped):
            return _CATCHMENT_ID_PREFIX_RE.sub('', stripped)
    if isinstance(reach_attrs, dict):
        river_name = reach_attrs.get('river_name')
        if isinstance(river_name, str) and river_name.strip():
            return river_name.strip()
    clean_id = _CATCHMENT_ID_PREFIX_RE.sub('', str(basin_id or '').strip())
    return clean_id or str(basin_id)


def _extract_polygon_area_km2(props: dict[str, object]) -> float | None:
    for key in ('area_km2', 'area', 'AREA', 'SUB_AREA', 'upstream_area_km2'):
        val = props.get(key)
        if isinstance(val, (int, float, np.number)) and not isinstance(
            val, bool
        ):
            fval = float(val)
            if math.isfinite(fval) and fval > 0.0:
                return round(fval, 2)
    reach_attrs = props.get('reach_attributes')
    if isinstance(reach_attrs, dict):
        r_area = reach_attrs.get('upstream_area_km2')
        if isinstance(r_area, (int, float, np.number)) and not isinstance(
            r_area, bool
        ):
            fval = float(r_area)
            if math.isfinite(fval) and fval > 0.0:
                return round(fval, 2)
    return None


def list_user_clipped_basins(username: str) -> list[dict[str, object]]:
    """Return clipped/delineated watersheds in the user's account."""
    pm = get_profile_manager()
    features = pm.load_watersheds(username)
    basins: list[dict[str, object]] = []
    seen: set[str] = set()
    for feat in features:
        if not isinstance(feat, dict):
            continue
        raw_props = feat.get('properties')
        props = raw_props if isinstance(raw_props, dict) else {}
        cid = str(props.get('catchment_id') or feat.get('id') or '').strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        area_km2 = _extract_polygon_area_km2(props)
        if area_km2 is None:
            static_attrs = _load_profile_static_attributes_for_basin(
                username, cid
            )
            stat_area = static_attrs.get('area')
            if (
                stat_area is not None
                and math.isfinite(stat_area)
                and stat_area > 0.0
            ):
                area_km2 = round(float(stat_area), 2)
        clean_name = clean_basin_display_name(
            props.get('name'),
            props.get('reach_attributes'),
            cid,
        )
        basins.append(
            {
                'basin_id': cid,
                'name': clean_name,
                'area_km2': area_km2,
                'source': str(props.get('source') or 'delineated'),
            }
        )
    return basins


def _resolve_basin_area_km2(username: str, basin_id: str) -> float | None:
    """Resolve drainage area in km² from clipped polygons or static attrs."""
    clean_id = (basin_id or '').strip()
    if not clean_id:
        return None
    for b in list_user_clipped_basins(username):
        if b['basin_id'] == clean_id and isinstance(
            b.get('area_km2'), (int, float)
        ):
            return float(b['area_km2'])
    static_attrs = _load_profile_static_attributes_for_basin(username, clean_id)
    stat_area = static_attrs.get('area')
    if stat_area is not None and math.isfinite(stat_area) and stat_area > 0.0:
        return round(float(stat_area), 2)
    return None


def _read_profile_product_band(
    username: str,
    prod_upper: str,
    basin_id: str,
    var_name: str,
    target_lead_time: int,
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    aliases = _PRODUCT_DIR_ALIASES.get(prod_upper, (prod_upper,))
    for base_dir in _profile_dynamics_dirs(username):
        if not base_dir.is_dir():
            continue
        for folder_name in aliases:
            zpath = base_dir / folder_name / 'timeseries.zarr'
            res = _read_zarr_basin_lead_matrix(
                zpath, basin_id, var_name, target_lead_time
            )
            if res is not None:
                return res
    return None


def _build_product_dataset_for_finetune(
    user_basin: tuple[str, str],
    prod_bands: tuple[str, list[str]],
    time_coords: tuple[pd.DatetimeIndex, pd.TimedeltaIndex],
    *,
    is_forecast: bool,
    use_global_data: bool,
) -> xr.Dataset:
    username, basin_id = user_basin
    prod, bands = prod_bands
    dates, lead_coords = time_coords
    prod_upper = prod.upper()
    lead_time = len(lead_coords)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for var_name in bands:
        band_res = (
            _read_profile_product_band(
                username,
                prod_upper,
                basin_id,
                var_name,
                lead_time if is_forecast else 1,
            )
            if use_global_data
            else None
        )
        if is_forecast:
            arr_3d = np.full(
                (1, len(dates), lead_time), np.nan, dtype=np.float32
            )
            if band_res is not None:
                src_dates, src_mat = band_res
                idx = src_dates.get_indexer(dates)
                valid = idx >= 0
                cols = min(src_mat.shape[1], lead_time)
                arr_3d[0, valid, :cols] = src_mat[idx[valid], :cols]
            data_vars[var_name] = (('basin', 'date', 'lead_time'), arr_3d)
        else:
            arr_2d = np.full((1, len(dates)), np.nan, dtype=np.float32)
            if band_res is not None:
                src_dates, src_mat = band_res
                idx = src_dates.get_indexer(dates)
                valid = idx >= 0
                arr_2d[0, valid] = src_mat[idx[valid], 0]
            data_vars[var_name] = (('basin', 'date'), arr_2d)

    coords: dict[str, object] = {'basin': [basin_id], 'date': dates}
    if is_forecast:
        coords['lead_time'] = lead_coords
    return xr.Dataset(data_vars, coords=coords)


def _has_native_global_inputs(global_dyn_dir: Path, base_cfg: Config) -> bool:
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
    has_hc = False
    for prod, bands in hindcast_in.items():
        zpath = global_dyn_dir / prod.upper() / 'timeseries.zarr'
        if not _is_valid_zarr_dir(zpath):
            continue
        ds = xr.open_zarr(str(zpath), decode_timedelta=False)
        if bands and all(
            b in ds.data_vars and bool(np.isfinite(ds[b].to_numpy()).any())
            for b in bands
        ):
            has_hc = True
            break

    has_fc = False
    for prod, bands in forecast_in.items():
        zpath = global_dyn_dir / prod.upper() / 'timeseries.zarr'
        if not _is_valid_zarr_dir(zpath):
            continue
        ds = xr.open_zarr(str(zpath), decode_timedelta=False)
        if bands and all(
            b in ds.data_vars and bool(np.isfinite(ds[b].to_numpy()).any())
            for b in bands
        ):
            has_fc = True
            break
    return has_hc and has_fc


def _prepare_finetune_zarr_stores(
    username: str,
    base_cfg: Config,
    basin_id: str,
    dates: pd.DatetimeIndex,
    *,
    use_global_data: bool = True,
) -> tuple[Path, Path]:
    """Write global product Zarr stores and real basin static attributes."""
    zarr_root = _basin_data_root(username, basin_id) / 'zarr_store'
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
        ds = _build_product_dataset_for_finetune(
            (username, basin_id),
            (prod, bands),
            (dates, lead_coords),
            is_forecast=(prod in forecast_in),
            use_global_data=use_global_data,
        )
        ds.to_zarr(pdir, mode='w')

    attr_path = zarr_root / 'attributes.zarr'
    if attr_path.exists():
        shutil.rmtree(attr_path)

    required_attrs = list(base_cfg.static_attributes or [])
    basin_attrs = _load_profile_static_attributes_for_basin(username, basin_id)
    if not basin_attrs and required_attrs:
        raise FileNotFoundError(
            f"Static attributes for basin '{basin_id}' were not found in "
            f"profile '{username}'. Extract or upload catchment static "
            'attributes before fine-tuning.'
        )

    missing_attrs = [a for a in required_attrs if a not in basin_attrs]
    if missing_attrs:
        raise KeyError(
            f"Basin '{basin_id}' is missing required static attribute(s): "
            f'{missing_attrs}'
        )

    static_vars: dict[str, tuple[tuple[str], np.ndarray]] = {
        attr_name: (
            ('basin',),
            np.array([basin_attrs[attr_name]], dtype=np.float32),
        )
        for attr_name in required_attrs
    }
    attr_ds = xr.Dataset(static_vars, coords={'basin': [basin_id]})
    attr_ds.to_zarr(attr_path, mode='w')
    return global_dyn_dir, attr_path


def compute_partition_dates(
    dates: pd.DatetimeIndex,
    *,
    base_seq_length: int,
    lead_time: int,
    split_ratio: float = 0.7,
    predict_last_n: int | None = None,
) -> dict[str, object]:
    """Compute effective sequence length and disjoint train/val date bounds."""
    ratio_val = _validate_finite_float(
        split_ratio,
        'split_ratio',
        min_val=_MIN_SPLIT_RATIO,
        max_val=_MAX_SPLIT_RATIO,
    )
    lead_val = _validate_positive_int(lead_time, 'lead_time', min_val=1)
    base_seq_val = _validate_positive_int(
        base_seq_length, 'base_seq_length', min_val=1
    )
    pred_last = (
        _validate_positive_int(predict_last_n, 'predict_last_n', min_val=1)
        if predict_last_n is not None
        else lead_val
    )
    target_horizon = max(lead_val, pred_last)
    min_seq = max(_MIN_SEQ_LENGTH, target_horizon)

    num_days = len(dates)
    min_required = min_seq + lead_val + target_horizon + (_MIN_SAMPLE_DAYS - 2)
    if num_days < min_required:
        raise ValueError(
            f'Uploaded record has {num_days} overlapping days, but at least '
            f'{min_required} days are required for lead_time={lead_val} and '
            f'disjoint train/validation windows.'
        )

    max_allowed_seq = max(
        min_seq, (num_days - lead_val - target_horizon - 4) // 2
    )
    eff_seq_length = min(base_seq_val, max_allowed_seq)

    first_sample_idx = eff_seq_length
    last_sample_idx = num_days - lead_val
    usable_sample_days = (last_sample_idx - first_sample_idx + 1) - (
        target_horizon - 1
    )
    if usable_sample_days < _MIN_SAMPLE_DAYS:
        raise ValueError(
            f'Insufficient valid sample dates ({usable_sample_days}) after '
            f'accounting for seq_length={eff_seq_length} and '
            f'lead_time={lead_val}.'
        )

    train_count = max(
        2,
        min(
            usable_sample_days - 2,
            round(usable_sample_days * ratio_val),
        ),
    )
    split_sample_idx = first_sample_idx + train_count - 1
    val_start_idx = split_sample_idx + target_horizon
    val_count = last_sample_idx - val_start_idx + 1

    return {
        'effective_seq_length': eff_seq_length,
        'base_seq_length': base_seq_val,
        'seq_length_shortened': eff_seq_length < base_seq_val,
        'lead_time': lead_val,
        'split_ratio': round(train_count / usable_sample_days, 3),
        'record_start_date': dates[0].strftime('%Y-%m-%d'),
        'record_end_date': dates[-1].strftime('%Y-%m-%d'),
        'train_start_date': dates[first_sample_idx].strftime('%Y-%m-%d'),
        'train_end_date': dates[split_sample_idx].strftime('%Y-%m-%d'),
        'validation_start_date': dates[val_start_idx].strftime('%Y-%m-%d'),
        'validation_end_date': dates[last_sample_idx].strftime('%Y-%m-%d'),
        'split_date': dates[split_sample_idx].strftime('%Y-%m-%d'),
        'train_sample_days': int(train_count),
        'validation_sample_days': int(val_count),
        'total_days': int(num_days),
    }


def _compute_metrics(
    obs: np.ndarray, sim: np.ndarray
) -> dict[str, float | None]:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if int(mask.sum()) < _MIN_METRIC_POINTS:
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


def _append_subkey_to_spec(spec: dict[str, object], grp: str, sub: str) -> None:
    existing = spec.get(grp)
    if existing is True:
        return
    sub_list = [str(x) for x in existing] if isinstance(existing, list) else []
    if sub not in sub_list:
        sub_list.append(sub)
    spec[grp] = sub_list


def _prune_missing_local_forcing_from_spec(
    spec: dict[str, object],
    target_model: torch.nn.Module,
) -> None:
    for grp, subspec in list(spec.items()):
        grp_mod = getattr(target_model, grp, None)
        if (
            isinstance(grp_mod, torch.nn.ModuleDict)
            and isinstance(subspec, list)
            and 'local_forcing' in subspec
            and 'local_forcing' not in grp_mod
        ):
            filtered = [k for k in subspec if k != 'local_forcing']
            if filtered:
                spec[grp] = filtered
            else:
                del spec[grp]


def _build_finetune_modules_spec(
    selected_components: list[str] | dict[str, object] | None,
    model: torch.nn.Module,
) -> dict[str, object]:
    """Convert UI component selections into ``Config.finetune_modules``."""
    target_model = getattr(model, '_orig_mod', model)
    newly_init: set[tuple[str, str]] = (
        target_model.get_newly_initialized_embeddings()
        if isinstance(target_model, BaseModel)
        else set()
    )

    spec: dict[str, object] = {}
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

    _prune_missing_local_forcing_from_spec(spec, target_model)

    if not spec:
        module_parts = list(getattr(target_model, 'module_parts', []))
        default_head = _default_head_component_name(module_parts)
        spec[default_head] = True

    for grp, prod in sorted(newly_init):
        _append_subkey_to_spec(spec, grp, prod)

    return spec


def _unfreeze_module_dict(
    module: torch.nn.ModuleDict,
    module_name: str,
    subspec: object,
) -> list[str]:
    if isinstance(subspec, bool):
        if not subspec:
            return []
        keys_to_unfreeze = list(module.keys())
    elif isinstance(subspec, str):
        keys_to_unfreeze = [subspec]
    elif isinstance(subspec, list) and all(isinstance(k, str) for k in subspec):
        keys_to_unfreeze = [str(k) for k in subspec]
    else:
        raise ValueError(
            f'Invalid value {subspec!r} for ModuleDict {module_name!r} in '
            "'finetune_modules'; expected bool, str, or list[str]."
        )

    unfrozen: list[str] = []
    for sub_key in keys_to_unfreeze:
        if sub_key not in module:
            raise KeyError(
                f"Submodule '{sub_key}' not found in '{module_name}'. "
                f'Available keys: {list(module.keys())}'
            )
        for p in module[sub_key].parameters():
            p.requires_grad = True
        unfrozen.append(f'{module_name}.{sub_key}')
    return unfrozen


def _validate_newly_init_unfrozen(target_model: torch.nn.Module) -> None:
    if not isinstance(target_model, BaseModel):
        return
    for grp, prod in sorted(target_model.get_newly_initialized_embeddings()):
        group_obj = getattr(target_model, grp, None)
        if (
            isinstance(group_obj, torch.nn.ModuleDict)
            and prod in group_obj
            and not any(p.requires_grad for p in group_obj[prod].parameters())
        ):
            raise ValueError(
                f"Newly initialized embedding '{grp}.{prod}' must be "
                "unfrozen in 'finetune_modules'."
            )


def _apply_freeze_spec(
    model: torch.nn.Module, finetune_modules: dict[str, object]
) -> list[str]:
    """Freeze all parameters except those enabled in ``finetune_modules``."""
    target_model = getattr(model, '_orig_mod', model)
    allowed_parts = set(getattr(target_model, 'module_parts', []))
    for param in target_model.parameters():
        param.requires_grad = False

    unfrozen: list[str] = []
    for module_name, subspec in finetune_modules.items():
        if module_name not in allowed_parts:
            raise KeyError(
                f"Unknown finetune module '{module_name}'. "
                f'Available module_parts: {sorted(allowed_parts)}'
            )
        module = getattr(target_model, module_name)
        if isinstance(module, torch.nn.ModuleDict):
            unfrozen.extend(_unfreeze_module_dict(module, module_name, subspec))
        else:
            if not isinstance(subspec, bool):
                raise TypeError(
                    f'Invalid value {subspec!r} for non-ModuleDict module '
                    f"'{module_name}'; expected bool."
                )
            if subspec:
                for p in module.parameters():
                    p.requires_grad = True
                unfrozen.append(module_name)

    _validate_newly_init_unfrozen(target_model)
    if not any(p.requires_grad for p in target_model.parameters()):
        raise ValueError(
            'At least one model component must be unfrozen for fine-tuning.'
        )
    return unfrozen


def _evaluate_dataset_predictions(
    model: torch.nn.Module,
    dataset: BaseDataset,
    loss_obj: BaseLoss,
    scaler: Scaler,
    full_dates: pd.DatetimeIndex,
) -> tuple[float, list[float | None], list[float | None]]:
    """Evaluate loss and unscaled streamflow predictions across a dataset."""
    model.eval()
    lead_time = int(getattr(dataset, 'lead_time', 1))
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
    sim_lead1: list[float | None] = [None] * len(full_dates)
    sim_lead_last: list[float | None] = [None] * len(full_dates)
    date_to_pos = {d: i for i, d in enumerate(full_dates)}

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
            batch_dates = np.asarray(xs_raw['date'])
            for row_idx in range(unscaled.shape[0]):
                issue_date = pd.Timestamp(batch_dates[row_idx, -lead_time])
                last_date = pd.Timestamp(batch_dates[row_idx, -1])
                pos1 = date_to_pos.get(issue_date)
                if pos1 is not None and np.isfinite(unscaled[row_idx, 0]):
                    sim_lead1[pos1] = round(float(unscaled[row_idx, 0]), 4)
                pos_last = date_to_pos.get(last_date)
                if pos_last is not None and np.isfinite(unscaled[row_idx, -1]):
                    sim_lead_last[pos_last] = round(
                        float(unscaled[row_idx, -1]), 4
                    )

    mean_loss = float(np.mean(losses)) if losses else float('nan')
    return mean_loss, sim_lead1, sim_lead_last


def _populate_realtime_upload_meta(
    loaded: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    upload_meta: dict[str, object],
) -> None:
    if 'streamflow' not in loaded:
        return
    for k_name, unit_str in (('precip', 'mm/day'), ('temp', 'degC')):
        if k_name in loaded and k_name not in upload_meta:
            d_idx, mat = loaded[k_name]
            upload_meta[k_name] = {
                'filename': 'Profile Forcing',
                'source': 'realtime_dynamics',
                'start_date': d_idx[0].strftime('%Y-%m-%d'),
                'end_date': d_idx[-1].strftime('%Y-%m-%d'),
                'num_days': len(d_idx),
                'num_valid_days': int(np.isfinite(mat[:, 0]).sum()),
                'num_lead_columns': int(mat.shape[1]),
                'units': unit_str,
            }


def _check_model_global_products_for_basin(
    username: str,
    basin_id: str,
    selected_model: dict[str, object],
) -> tuple[list[str], list[str]]:
    hc_groups = selected_model.get('base_hindcast_products')
    fc_groups = selected_model.get('base_forecast_products')
    model_prods = sorted(
        (
            set(hc_groups if isinstance(hc_groups, list) else [])
            | set(fc_groups if isinstance(fc_groups, list) else [])
        )
        - {'local_forcing'}
    )
    products_found: list[str] = []
    products_missing: list[str] = []
    for prod in model_prods:
        prod_upper = str(prod).upper()
        aliases = _PRODUCT_DIR_ALIASES.get(prod_upper, (prod_upper,))
        found_prod = False
        for base_dir in _profile_dynamics_dirs(username):
            if not base_dir.is_dir():
                continue
            for folder_name in aliases:
                zpath = base_dir / folder_name / 'timeseries.zarr'
                if _is_valid_zarr_dir(zpath):
                    ds = xr.open_zarr(str(zpath), decode_timedelta=False)
                    b_coord = next(
                        (
                            c
                            for c in (
                                'basin',
                                'basin_id',
                                'catchment_id',
                                'gauge_id',
                            )
                            if c in ds.coords
                        ),
                        None,
                    )
                    if b_coord is not None and basin_id in [
                        str(x) for x in ds.coords[b_coord].to_numpy()
                    ]:
                        found_prod = True
                        break
            if found_prod:
                break
        if found_prod:
            products_found.append(str(prod))
        else:
            products_missing.append(str(prod))
    return products_found, products_missing


def _inspect_global_forcing_coverage(
    username: str,
    basin_id: str,
    selected_model: dict[str, object],
    raw_uploaded: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
) -> tuple[
    dict[str, object],
    tuple[pd.DatetimeIndex, np.ndarray, str] | None,
    tuple[pd.DatetimeIndex, np.ndarray, str] | None,
]:
    lead_val = int(str(selected_model['lead_time']))
    pred_last_val = int(str(selected_model['predict_last_n']))
    rt_precip = _extract_realtime_forcing_matrix(
        username,
        basin_id,
        _REALTIME_PRECIP_CANDIDATES,
        target_lead_time=lead_val,
    )
    rt_temp = _extract_realtime_forcing_matrix(
        username,
        basin_id,
        _REALTIME_TEMP_CANDIDATES,
        target_lead_time=lead_val,
    )
    archive_exists = rt_precip is not None and rt_temp is not None
    if archive_exists and rt_precip is not None and rt_temp is not None:
        g_start = max(rt_precip[0][0], rt_temp[0][0])
        g_end = min(rt_precip[0][-1], rt_temp[0][-1])
        g_days = max(0, int((g_end - g_start).days) + 1)
        g_start_str: str | None = g_start.strftime('%Y-%m-%d')
        g_end_str: str | None = g_end.strftime('%Y-%m-%d')
    else:
        g_start = pd.Timestamp.min
        g_end = pd.Timestamp.min
        g_days = 0
        g_start_str = None
        g_end_str = None

    products_found, products_missing = _check_model_global_products_for_basin(
        username, basin_id, selected_model
    )

    target_horizon = max(lead_val, pred_last_val)
    min_seq = max(_MIN_SEQ_LENGTH, target_horizon)
    min_req_days = min_seq + lead_val + target_horizon + (_MIN_SAMPLE_DAYS - 2)
    today_utc = pd.Timestamp.now('UTC').tz_localize(None).floor('D')

    if 'streamflow' in raw_uploaded:
        sf_dates = raw_uploaded['streamflow'][0]
        sf_start, sf_end = sf_dates[0], sf_dates[-1]
        sf_days = len(sf_dates)
        sf_start_str: str | None = sf_start.strftime('%Y-%m-%d')
        sf_end_str: str | None = sf_end.strftime('%Y-%m-%d')
        ov_start = max(sf_start, g_start)
        ov_end = min(sf_end, g_end)
        overlap_days = (
            max(0, int((ov_end - ov_start).days) + 1)
            if archive_exists and ov_start <= ov_end
            else 0
        )
        coverage_ratio = (
            round(min(1.0, overlap_days / max(1, sf_days)), 3)
            if sf_days > 0
            else 0.0
        )
        has_sufficient_overlap = overlap_days >= min_req_days
        full_coverage = archive_exists and overlap_days >= max(1, sf_days - 3)
        recommended_lookback_days = max(
            365, int((today_utc - sf_start).days) + 14
        )
    else:
        sf_start_str = None
        sf_end_str = None
        sf_days = 0
        overlap_days = g_days
        coverage_ratio = 1.0 if archive_exists else 0.0
        has_sufficient_overlap = archive_exists and g_days >= min_req_days
        full_coverage = has_sufficient_overlap
        recommended_lookback_days = 365

    status_info: dict[str, object] = {
        'archive_exists_for_basin': archive_exists,
        'has_basin_archive': archive_exists,
        'has_sufficient_overlap': has_sufficient_overlap,
        'full_coverage': full_coverage,
        'fully_covers_streamflow': full_coverage,
        'needs_extraction': not (archive_exists and has_sufficient_overlap),
        'precip_product': rt_precip[2] if rt_precip is not None else None,
        'temp_product': rt_temp[2] if rt_temp is not None else None,
        'global_start_date': g_start_str,
        'global_end_date': g_end_str,
        'global_days': g_days,
        'streamflow_start_date': sf_start_str,
        'streamflow_end_date': sf_end_str,
        'streamflow_days': sf_days,
        'overlap_days': overlap_days,
        'min_required_days': min_req_days,
        'coverage_ratio': coverage_ratio,
        'products_found': products_found,
        'products_missing': products_missing,
        'recommended_lookback_days': recommended_lookback_days,
    }
    return status_info, rt_precip, rt_temp


def _build_workspace_timeseries_and_partition(
    loaded: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    selected_model: dict[str, object],
    *,
    data_ready: bool,
    active_split: float,
    extra_precip: tuple[
        tuple[pd.DatetimeIndex, np.ndarray] | None,
        tuple[pd.DatetimeIndex, np.ndarray, str] | None,
    ]
    | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not loaded:
        return None, None
    seq_len_val = int(str(selected_model['seq_length']))
    lead_val = int(str(selected_model['lead_time']))
    pred_last_val = int(str(selected_model['predict_last_n']))
    active_keys = (
        ('precip', 'temp', 'streamflow')
        if data_ready
        else tuple(k for k in ('precip', 'temp', 'streamflow') if k in loaded)
    )
    start_d = max(loaded[k][0][0] for k in active_keys)
    end_d = min(loaded[k][0][-1] for k in active_keys)
    if start_d > end_d:
        return None, None

    common_dates = pd.date_range(start_d, end_d, freq='D')

    def _align_series(
        pair: tuple[pd.DatetimeIndex, np.ndarray] | None,
    ) -> list[float | None] | None:
        if pair is None:
            return None
        d_idx, mat = pair
        indexer = d_idx.get_indexer(common_dates)
        return [
            round(float(mat[pos, 0]), 4)
            if pos >= 0 and np.isfinite(mat[pos, 0])
            else None
            for pos in indexer
        ]

    def _extract_lead1(key: str) -> list[float | None]:
        res = _align_series(loaded.get(key))
        return res if res is not None else [None] * len(common_dates)

    partition_info: dict[str, object] | None = None
    if data_ready:
        partition_info = compute_partition_dates(
            common_dates,
            base_seq_length=seq_len_val,
            lead_time=lead_val,
            split_ratio=active_split,
            predict_last_n=pred_last_val,
        )
    else:
        try:
            partition_info = compute_partition_dates(
                common_dates,
                base_seq_length=seq_len_val,
                lead_time=lead_val,
                split_ratio=active_split,
                predict_last_n=pred_last_val,
            )
        except ValueError:
            partition_info = None

    raw_local_precip = extra_precip[0] if extra_precip is not None else None
    rt_precip = extra_precip[1] if extra_precip is not None else None
    precip_local = _align_series(raw_local_precip)
    precip_global = (
        _align_series((rt_precip[0], rt_precip[1]))
        if rt_precip is not None
        else None
    )

    timeseries_payload: dict[str, object] = {
        'dates': [d.strftime('%Y-%m-%d') for d in common_dates],
        'precip': _extract_lead1('precip'),
        'precip_local': precip_local if precip_local is not None else [],
        'precip_global': precip_global if precip_global is not None else [],
        'precip_global_product': (
            rt_precip[2] if rt_precip is not None else None
        ),
        'temp': _extract_lead1('temp'),
        'streamflow_obs': _extract_lead1('streamflow'),
    }
    return timeseries_payload, partition_info


def _get_or_compute_baseline_eval(
    username: str,
    basin_id: str,
    model_id: str,
    partition_info: dict[str, object],
    resolved_forcing_mode: str,
) -> dict[str, object] | None:
    common_dates = pd.date_range(
        str(partition_info['record_start_date']),
        str(partition_info['record_end_date']),
        freq='D',
    )
    sess_dir = _session_root(username, basin_id, model_id)
    state_path = sess_dir / 'session_state.json'
    session_state: dict[str, object] = (
        json.loads(state_path.read_text(encoding='utf-8'))
        if state_path.exists()
        else {}
    )
    raw_hist = session_state.get('epoch_history')
    if isinstance(raw_hist, list) and raw_hist:
        first_rec = raw_hist[0]
        if isinstance(first_rec, dict) and first_rec.get('epoch') == 0:
            sim1 = first_rec.get('streamflow_sim_lead1')
            if isinstance(sim1, list) and len(sim1) == len(common_dates):
                return first_rec

    meta_path = _basin_data_root(username, basin_id) / 'upload_meta.json'
    upload_meta: dict[str, object] = (
        json.loads(meta_path.read_text(encoding='utf-8'))
        if meta_path.exists()
        else {}
    )
    cache_path = sess_dir / 'baseline_eval_cache.json'
    sf_up = (
        upload_meta.get('streamflow', {}).get('uploaded_at', '')
        if isinstance(upload_meta.get('streamflow'), dict)
        else ''
    )
    pr_up = (
        upload_meta.get('precip', {}).get('uploaded_at', '')
        if isinstance(upload_meta.get('precip'), dict)
        else ''
    )
    signature = (
        f'{model_id}|{basin_id}|{resolved_forcing_mode}|'
        f'{partition_info.get("record_start_date")}|'
        f'{partition_info.get("record_end_date")}|'
        f'{partition_info.get("split_date")}|'
        f'{partition_info.get("effective_seq_length")}|'
        f'{sf_up}|{pr_up}'
    )
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            rec = cached.get('record') if isinstance(cached, dict) else None
            if (
                isinstance(cached, dict)
                and cached.get('signature') == signature
                and isinstance(rec, dict)
                and isinstance(rec.get('streamflow_sim_lead1'), list)
                and len(rec['streamflow_sim_lead1']) == len(common_dates)
            ):
                return rec
        except (OSError, ValueError, TypeError):
            pass

    _, base_run_dir = _resolve_model_dir(model_id, username)
    base_ckpt, _ = _find_latest_checkpoint(base_run_dir)
    if base_ckpt is None:
        return None
    base_cfg = Config(base_run_dir / 'config.yml', dev_mode=True)
    lead_time = int(base_cfg.lead_time)
    if (
        _rebuild_local_zarr_stores(
            username,
            basin_id,
            target_lead_time=lead_time,
            forcing_mode=resolved_forcing_mode,
        )
        is None
    ):
        return None

    zarr_root = _basin_data_root(username, basin_id) / 'zarr_store'
    use_global = resolved_forcing_mode != 'local_only'
    global_dyn_dir, attr_path = _prepare_finetune_zarr_stores(
        username,
        base_cfg,
        basin_id,
        common_dates,
        use_global_data=use_global,
    )
    has_native_global = use_global and _has_native_global_inputs(
        global_dyn_dir, base_cfg
    )
    include_local = (resolved_forcing_mode != 'global_only') or (
        not has_native_global
    )

    eval_run_dir = sess_dir / 'baseline_eval_run'
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    lr_val = float(base_cfg.initial_learning_rate or 5e-4)
    loss_val = _resolve_and_validate_loss(None, base_cfg)
    lr_strat_val = _resolve_and_validate_lr_strategy(None, base_cfg)
    bs_val = int(base_cfg.batch_size or 16)

    ft_cfg = _build_finetune_config(
        _FinetuneConfigParams(
            base_cfg=base_cfg,
            base_run_dir=base_run_dir,
            run_dir=eval_run_dir,
            zarr_root=zarr_root,
            global_dyn_dir=global_dyn_dir,
            attr_path=attr_path,
            partition=partition_info,
            lead_time=lead_time,
            lr_val=lr_val,
            loss_val=loss_val,
            lr_strat_val=lr_strat_val,
            bs_val=bs_val,
            include_local_forcing=include_local,
        )
    )
    train_ds = get_dataset(
        cfg=ft_cfg, is_train=True, period='train', compute_scaler=False
    )
    torch.manual_seed(int(ft_cfg.seed or 42))
    model = get_model(ft_cfg).to(torch.device('cpu'))
    load_model_weights(
        model, base_ckpt, device='cpu', allow_new_embeddings=True
    )
    eval_ctx = _build_epoch_eval_context(
        model, ft_cfg, train_ds, partition_info, []
    )
    rec0, _ = _evaluate_and_build_epoch_record(
        eval_ctx,
        0,
        'Epoch 0 (Pretrained Initial)',
        lr_val,
    )
    sess_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({'signature': signature, 'record': rec0}, indent=2),
        encoding='utf-8',
    )
    return rec0


def get_finetune_workspace_status(
    username: str,
    basin_id: str,
    model_id: str = 'google-floodhub-settings-110-epochs',
    *,
    split_ratio: float | None = None,
    forcing_mode: str | None = None,
) -> dict[str, object]:
    """Return full status of the per-basin fine-tuning workspace."""
    _resolve_model_dir(model_id, username)
    models = list_finetune_models(username)
    selected_model = next(
        (m for m in models if m['model_id'] == model_id),
        None,
    )
    if selected_model is None:
        raise FileNotFoundError(f'Unknown model_id: {model_id}')

    user_basins = list_user_clipped_basins(username)
    clean_basin_id = (basin_id or '').strip()
    if (
        not clean_basin_id or clean_basin_id == 'custom_basin_01'
    ) and user_basins:
        clean_basin_id = str(user_basins[0]['basin_id'])
    basin_id = clean_basin_id or 'custom_basin_01'

    matched_basin = next(
        (b for b in user_basins if b['basin_id'] == basin_id),
        None,
    )
    basin_name = (
        str(matched_basin['name'])
        if matched_basin is not None
        else clean_basin_display_name(None, None, basin_id)
    )
    basin_area_km2 = (
        float(matched_basin['area_km2'])
        if matched_basin is not None
        and isinstance(matched_basin.get('area_km2'), (int, float))
        else _resolve_basin_area_km2(username, basin_id)
    )

    lead_val = int(str(selected_model['lead_time']))

    data_root = _basin_data_root(username, basin_id)
    meta_path = data_root / 'upload_meta.json'
    upload_meta = (
        json.loads(meta_path.read_text(encoding='utf-8'))
        if meta_path.exists()
        else {}
    )
    raw_uploaded = _load_raw_uploaded_arrays(username, basin_id)
    global_forcing_status, rt_precip, _ = _inspect_global_forcing_coverage(
        username, basin_id, selected_model, raw_uploaded
    )
    has_local_forcing = 'precip' in raw_uploaded and 'temp' in raw_uploaded
    has_global_forcing = bool(
        global_forcing_status.get('archive_exists_for_basin')
    )

    sess_dir = _session_root(
        username, basin_id, str(selected_model['model_id'])
    )
    state_path = sess_dir / 'session_state.json'
    session_state: dict[str, object] = (
        json.loads(state_path.read_text(encoding='utf-8'))
        if state_path.exists()
        else {}
    )

    saved_mode = (
        str(session_state.get('forcing_mode'))
        if session_state.get('forcing_mode')
        else None
    )
    resolved_forcing_mode = _resolve_forcing_mode(
        forcing_mode or saved_mode,
        has_local_forcing=has_local_forcing,
        has_global_forcing=has_global_forcing,
    )

    loaded = _load_uploaded_arrays(
        username,
        basin_id,
        target_lead_time=lead_val,
        forcing_mode=resolved_forcing_mode,
    )
    data_ready = all(k in loaded for k in ('precip', 'temp', 'streamflow'))
    _populate_realtime_upload_meta(loaded, upload_meta)

    active_split = (
        _validate_finite_float(
            split_ratio,
            'split_ratio',
            min_val=_MIN_SPLIT_RATIO,
            max_val=_MAX_SPLIT_RATIO,
        )
        if split_ratio is not None
        else float(str(session_state.get('split_ratio', 0.7)))
    )

    timeseries_payload, partition_info = (
        _build_workspace_timeseries_and_partition(
            loaded,
            selected_model,
            data_ready=data_ready,
            active_split=active_split,
            extra_precip=(raw_uploaded.get('precip'), rt_precip),
        )
    )
    if partition_info is None:
        data_ready = False

    baseline_metrics: dict[str, object] | None = None
    if (
        data_ready
        and partition_info is not None
        and timeseries_payload is not None
    ):
        try:
            rec0 = _get_or_compute_baseline_eval(
                username,
                basin_id,
                str(selected_model['model_id']),
                partition_info,
                resolved_forcing_mode,
            )
            if rec0 is not None:
                timeseries_payload['streamflow_sim_base'] = rec0.get(
                    'streamflow_sim_lead1'
                )
                timeseries_payload['streamflow_sim_base_lead_last'] = rec0.get(
                    'streamflow_sim_lead_last'
                )
                baseline_metrics = {
                    k: rec0.get(k)
                    for k in (
                        'epoch',
                        'label',
                        'train_loss',
                        'val_loss',
                        'train_nse',
                        'val_nse',
                        'train_kge',
                        'val_kge',
                        'train_rmse',
                        'val_rmse',
                    )
                }
        except (FileNotFoundError, KeyError, ValueError, RuntimeError):
            baseline_metrics = None

    comp_names = selected_model.get('tunable_component_names')
    default_head = _default_head_component_name(
        comp_names if isinstance(comp_names, list) else ['head']
    )

    return {
        'basin_id': basin_id,
        'basin_name': basin_name,
        'basin_area_km2': basin_area_km2,
        'user_basins': user_basins,
        'model_id': selected_model['model_id'],
        'selected_model': selected_model,
        'models': models,
        'upload_meta': upload_meta,
        'has_local_forcing': has_local_forcing,
        'forcing_mode': resolved_forcing_mode,
        'global_forcing_status': global_forcing_status,
        'data_ready': data_ready,
        'partition': partition_info,
        'timeseries': timeseries_payload,
        'baseline_metrics': baseline_metrics,
        'session': {
            'current_epoch': int(str(session_state.get('current_epoch', 0))),
            'status': session_state.get(
                'status', 'ready' if data_ready else 'awaiting_data'
            ),
            'split_ratio': (
                partition_info['split_ratio']
                if partition_info
                else active_split
            ),
            'forcing_mode': resolved_forcing_mode,
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
                        default_head,
                    ],
                },
            ),
        },
    }


def reset_finetune_session(
    username: str,
    basin_id: str,
    model_id: str = 'google-floodhub-settings-110-epochs',
) -> dict[str, object]:
    """Reset the interactive fine-tuning session back to Epoch 0."""
    _resolve_model_dir(model_id, username)
    sess_dir = _session_root(username, basin_id, model_id)
    if sess_dir.exists():
        shutil.rmtree(sess_dir)
    return get_finetune_workspace_status(username, basin_id, model_id)


def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_strat_val: str,
    ft_cfg: Config,
) -> (
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
):
    if lr_strat_val == 'ConstantLR':
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=1,
        )
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
    raise ValueError(f'Unsupported learning_rate_strategy: {lr_strat_val!r}')


def _build_finetune_config(params: _FinetuneConfigParams) -> Config:
    eff_seq = int(str(params.partition['effective_seq_length']))
    cfg_dict = params.base_cfg.as_dict().copy()
    cfg_dict['is_finetuning'] = True
    cfg_dict['base_run_dir'] = str(params.base_run_dir)
    cfg_dict['run_dir'] = str(params.run_dir)
    cfg_dict['compile'] = False
    cfg_dict['device'] = 'cpu'
    cfg_dict['num_workers'] = 0
    cfg_dict['lazy_load'] = False
    cfg_dict['seq_length'] = eff_seq
    cfg_dict['forecast_overlap'] = eff_seq
    cfg_dict['lead_time'] = params.lead_time
    base_pred_last = int(params.base_cfg.predict_last_n or params.lead_time)
    cfg_dict['predict_last_n'] = min(eff_seq, base_pred_last)
    cfg_dict['initial_learning_rate'] = params.lr_val
    cfg_dict['loss'] = params.loss_val
    cfg_dict['learning_rate_strategy'] = params.lr_strat_val
    if params.lr_strat_val in ('StepLR', 'ReduceLROnPlateau'):
        default_drop = 5 if params.lr_strat_val == 'StepLR' else 3
        cfg_dict['learning_rate_epochs_drop'] = max(
            1, int(cfg_dict.get('learning_rate_epochs_drop') or default_drop)
        )
        cfg_dict['learning_rate_drop_factor'] = float(
            cfg_dict.get('learning_rate_drop_factor') or 0.5
        )
    cfg_dict['batch_size'] = params.bs_val
    cfg_dict['nan_handling_method'] = 'masked_mean'
    cfg_dict['union_mapping'] = {}

    basin_file = params.zarr_root / 'basin.txt'
    cfg_dict['train_basin_file'] = str(basin_file)
    cfg_dict['validation_basin_file'] = str(basin_file)
    cfg_dict['test_basin_file'] = str(basin_file)

    fmt = '%d/%m/%Y'
    cfg_dict['train_start_date'] = pd.Timestamp(
        str(params.partition['train_start_date'])
    ).strftime(fmt)
    cfg_dict['train_end_date'] = pd.Timestamp(
        str(params.partition['train_end_date'])
    ).strftime(fmt)
    cfg_dict['validation_start_date'] = pd.Timestamp(
        str(params.partition['validation_start_date'])
    ).strftime(fmt)
    cfg_dict['validation_end_date'] = pd.Timestamp(
        str(params.partition['validation_end_date'])
    ).strftime(fmt)
    cfg_dict['test_start_date'] = cfg_dict['validation_start_date']
    cfg_dict['test_end_date'] = cfg_dict['validation_end_date']

    cfg_dict['statics_data_dir'] = str(params.attr_path)
    cfg_dict['targets_data_dir'] = str(params.zarr_root / 'streamflow.zarr')
    hindcast_in = dict(cfg_dict.get('hindcast_inputs', {}))
    forecast_in = dict(cfg_dict.get('forecast_inputs', {}))
    if params.include_local_forcing:
        cfg_dict['dynamics_data_dir'] = [
            str(params.global_dyn_dir),
            str(params.zarr_root / 'local_dyn'),
        ]
        local_bands = ['local_precipitation', 'local_temperature']
        hindcast_in['local_forcing'] = list(local_bands)
        forecast_in['local_forcing'] = list(local_bands)
    else:
        cfg_dict['dynamics_data_dir'] = [str(params.global_dyn_dir)]
    cfg_dict['hindcast_inputs'] = hindcast_in
    cfg_dict['forecast_inputs'] = forecast_in
    return Config(cfg_dict, dev_mode=True)


def _evaluate_and_build_epoch_record(
    ctx: _EpochEvalContext,
    epoch: int,
    label: str,
    learning_rate: float,
    override_train_loss: float | None = None,
) -> tuple[dict[str, object], float]:
    tr_loss, tr_sim1, tr_sim_last = _evaluate_dataset_predictions(
        ctx.model,
        ctx.train_eval_ds,
        ctx.loss_obj,
        ctx.scaler,
        ctx.common_dates,
    )
    val_loss, val_sim1, val_sim_last = _evaluate_dataset_predictions(
        ctx.model,
        ctx.val_ds,
        ctx.loss_obj,
        ctx.scaler,
        ctx.common_dates,
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
    tr_metrics = _compute_metrics(
        ctx.obs_series[ctx.train_mask], sim_arr[ctx.train_mask]
    )
    val_metrics = _compute_metrics(
        ctx.obs_series[ctx.val_mask], sim_arr[ctx.val_mask]
    )
    record: dict[str, object] = {
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
        'unfrozen_components': ctx.unfrozen_parts,
        'streamflow_sim_lead1': combined_sim1,
        'streamflow_sim_lead_last': combined_sim_last,
    }
    return record, val_loss


def _train_single_epoch(
    model: torch.nn.Module,
    train_loader: MultimetDataLoader,
    optimizer: torch.optim.Optimizer,
    loss_obj: BaseLoss,
    ft_cfg: Config,
) -> float:
    model.train()
    clip_grad_norm = ft_cfg.clip_gradient_norm
    allow_nan = int(ft_cfg.allow_subsequent_nan_losses or 0)
    nan_count = 0
    batch_losses: list[float] = []

    for xs_raw in train_loader:
        xs = model.pre_model_hook(xs_raw, is_train=True)
        output = model(xs)
        loss_tensor, _ = loss_obj(output, xs)
        if not torch.isfinite(loss_tensor):
            nan_count += 1
            if nan_count > allow_nan:
                raise RuntimeError(
                    f'Loss was non-finite for {nan_count} consecutive batch '
                    'step(s). Stopped fine-tuning.'
                )
            continue
        nan_count = 0
        optimizer.zero_grad()
        loss_tensor.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        batch_losses.append(float(loss_tensor.item()))

    if not batch_losses:
        raise RuntimeError('All training batches produced non-finite loss.')
    return float(np.mean(batch_losses))


def _step_lr_scheduler(
    lr_scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    ),
    val_loss: float,
) -> None:
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if math.isfinite(val_loss):
            lr_scheduler.step(val_loss)
    else:
        lr_scheduler.step()


def _resolve_and_validate_loss(requested_loss: object, base_cfg: Config) -> str:
    head_name = str(base_cfg.head).lower()
    allowed = (
        ('CMALLoss',)
        if head_name in ('cmal', 'cmal_deterministic')
        else ('MSE', 'NSE', 'RMSE', 'WMSE')
    )
    if requested_loss is None or str(requested_loss).strip() == '':
        return allowed[0] if len(allowed) == 1 else str(base_cfg.loss).upper()
    raw = str(requested_loss).strip()
    for candidate in allowed:
        if raw.lower() == candidate.lower():
            return candidate
    raise ValueError(
        f"Loss '{raw}' is not compatible with model head '{base_cfg.head}'. "
        f'Allowed losses: {list(allowed)}'
    )


def _resolve_and_validate_lr_strategy(
    requested_strategy: object, base_cfg: Config
) -> str:
    if requested_strategy is None or str(requested_strategy).strip() == '':
        return _normalize_lr_strategy_name(
            getattr(base_cfg, 'learning_rate_strategy', None)
        )
    raw = str(requested_strategy).strip()
    for canonical in SUPPORTED_LR_STRATEGIES:
        if raw.lower() == canonical.lower():
            return canonical
    raise ValueError(
        f"Unsupported learning_rate_strategy '{raw}'. "
        f'Supported strategies: {list(SUPPORTED_LR_STRATEGIES)}'
    )


def _validate_uploaded_forecast_leads(
    loaded: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    lead_time: int,
) -> None:
    for key in ('precip', 'temp'):
        _, mat = loaded[key]
        missing_leads = [
            k + 1
            for k in range(lead_time)
            if k >= mat.shape[1] or not np.isfinite(mat[:, k]).any()
        ]
        if missing_leads:
            raise ValueError(
                f"Uploaded '{key}' CSV is missing finite values for forecast "
                f'lead column(s) {missing_leads} required by model '
                f'lead_time={lead_time}.'
            )


def _save_optimizer_state(
    run_dir: Path,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    ),
    unfrozen_parts: list[str],
) -> None:
    opt_path = run_dir / f'optimizer_state_epoch{epoch:03d}.pt'
    payload = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'unfrozen_parts': list(unfrozen_parts),
        'lr_scheduler_cls': type(lr_scheduler).__name__,
    }
    torch.save(payload, opt_path)


def _restore_optimizer_state(
    run_dir: Path,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    ),
    unfrozen_parts: list[str],
) -> None:
    opt_path = run_dir / f'optimizer_state_epoch{epoch:03d}.pt'
    if not opt_path.is_file():
        return
    target_lr = float(optimizer.param_groups[0]['lr'])
    saved = torch.load(str(opt_path), map_location='cpu', weights_only=True)
    if not isinstance(saved, dict):
        return
    if saved.get('unfrozen_parts') == list(unfrozen_parts):
        opt_state = saved.get('optimizer')
        if isinstance(opt_state, dict):
            optimizer.load_state_dict(opt_state)
            for group in optimizer.param_groups:
                group['lr'] = target_lr
        if saved.get('lr_scheduler_cls') == type(lr_scheduler).__name__:
            sched_state = saved.get('lr_scheduler')
            if isinstance(sched_state, dict):
                lr_scheduler.load_state_dict(sched_state)


def _parse_run_options(
    options: dict[str, object], base_cfg: Config
) -> tuple[
    int,
    float,
    str,
    str,
    int,
    list[str] | dict[str, object] | None,
    float,
    str | None,
]:
    allowed_opts = {
        'epochs',
        'learning_rate',
        'loss',
        'learning_rate_strategy',
        'batch_size',
        'finetune_modules',
        'split_ratio',
        'forcing_mode',
    }
    unknown = set(options) - allowed_opts
    if unknown:
        raise TypeError(f'Unexpected keyword argument(s): {sorted(unknown)}')

    epochs = _validate_positive_int(
        options.get('epochs', 5),
        'epochs',
        min_val=1,
        max_val=_MAX_EPOCHS_PER_STEP,
    )
    raw_lr = options.get('learning_rate')
    lr_val = (
        _validate_finite_float(raw_lr, 'learning_rate', strictly_positive=True)
        if raw_lr is not None
        else float(base_cfg.initial_learning_rate or 5e-4)
    )
    loss_val = _resolve_and_validate_loss(options.get('loss'), base_cfg)
    lr_strat_val = _resolve_and_validate_lr_strategy(
        options.get('learning_rate_strategy'), base_cfg
    )
    raw_bs = options.get('batch_size')
    bs_val = (
        _validate_positive_int(raw_bs, 'batch_size', min_val=1)
        if raw_bs is not None
        else int(base_cfg.batch_size or 16)
    )
    raw_modules = options.get('finetune_modules')
    if raw_modules is not None and not isinstance(raw_modules, (list, dict)):
        raise TypeError('finetune_modules must be a list or dict.')
    ft_modules: list[str] | dict[str, object] | None = (
        [str(x) for x in raw_modules]
        if isinstance(raw_modules, list)
        else (dict(raw_modules) if isinstance(raw_modules, dict) else None)
    )
    split_ratio = _validate_finite_float(
        options.get('split_ratio', 0.7),
        'split_ratio',
        min_val=_MIN_SPLIT_RATIO,
        max_val=_MAX_SPLIT_RATIO,
    )
    raw_fmode = options.get('forcing_mode')
    forcing_mode = (
        str(raw_fmode).strip().lower()
        if raw_fmode is not None and str(raw_fmode).strip()
        else None
    )
    if (
        forcing_mode is not None
        and forcing_mode != 'auto'
        and forcing_mode not in SUPPORTED_FORCING_MODES
    ):
        raise ValueError(
            f"Unsupported forcing_mode '{forcing_mode}'. "
            f'Supported modes: {list(SUPPORTED_FORCING_MODES)}'
        )
    return (
        epochs,
        lr_val,
        loss_val,
        lr_strat_val,
        bs_val,
        ft_modules,
        split_ratio,
        forcing_mode,
    )


def _execute_epoch_training_steps(
    eval_ctx: _EpochEvalContext,
    train_ds: BaseDataset,
    ft_cfg: Config,
    current_epoch: int,
    epoch_history: list[dict[str, object]],
) -> list[dict[str, object]]:
    run_dir = Path(str(ft_cfg.run_dir))
    lr_val = float(ft_cfg.initial_learning_rate)
    lr_strat_val = str(ft_cfg.learning_rate_strategy)
    bs_val = int(ft_cfg.batch_size)
    epochs = int(ft_cfg.epochs)

    optimizer = get_optimizer(eval_ctx.model, ft_cfg)
    lr_scheduler = _create_lr_scheduler(optimizer, lr_strat_val, ft_cfg)
    if current_epoch > 0:
        _restore_optimizer_state(
            run_dir,
            current_epoch,
            optimizer,
            lr_scheduler,
            eval_ctx.unfrozen_parts,
        )

    if current_epoch == 0 and not epoch_history:
        rec0, _ = _evaluate_and_build_epoch_record(
            eval_ctx,
            0,
            'Epoch 0 (Pretrained Initial)',
            lr_val,
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
            eval_ctx.model,
            train_loader,
            optimizer,
            eval_ctx.loss_obj,
            ft_cfg,
        )

        ckpt_out = run_dir / f'model_epoch{target_epoch:03d}.pt'
        torch.save(eval_ctx.model.state_dict(), ckpt_out)

        active_lr = float(optimizer.param_groups[0]['lr'])
        rec_ep, val_loss = _evaluate_and_build_epoch_record(
            eval_ctx,
            target_epoch,
            f'Epoch {target_epoch}',
            active_lr,
            override_train_loss=mean_tr_loss,
        )
        _step_lr_scheduler(lr_scheduler, val_loss)
        _save_optimizer_state(
            run_dir,
            target_epoch,
            optimizer,
            lr_scheduler,
            eval_ctx.unfrozen_parts,
        )
        epoch_history.append(rec_ep)

    return epoch_history


def _load_or_init_finetune_model(
    ft_cfg: Config,
    run_dir: Path,
    base_ckpt: Path,
    finetune_modules: list[str] | dict[str, object] | None,
) -> tuple[torch.nn.Module, int, list[str]]:
    torch.manual_seed(int(ft_cfg.seed or 42))
    model = get_model(ft_cfg).to(torch.device('cpu'))
    latest_sess_ckpt, current_epoch = _find_latest_checkpoint(run_dir)
    if latest_sess_ckpt is not None and current_epoch > 0:
        load_model_weights(
            model, latest_sess_ckpt, device='cpu', allow_new_embeddings=False
        )
        shared_fc = getattr(model, 'shared_embeddings_fc', None)
        if (
            isinstance(model, BaseModel)
            and isinstance(shared_fc, torch.nn.ModuleDict)
            and 'local_forcing' in shared_fc
        ):
            model.set_newly_initialized_embeddings(
                {('shared_embeddings_fc', 'local_forcing')}
            )
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
    return model, current_epoch, unfrozen_parts


def _build_epoch_eval_context(
    model: torch.nn.Module,
    ft_cfg: Config,
    train_ds: BaseDataset,
    partition: dict[str, object],
    unfrozen_parts: list[str],
) -> _EpochEvalContext:
    val_ds = get_dataset(
        cfg=ft_cfg, is_train=False, period='validation', compute_scaler=False
    )
    train_eval_ds = get_dataset(
        cfg=ft_cfg, is_train=False, period='train', compute_scaler=False
    )
    loss_obj = get_loss_obj(ft_cfg)
    if ft_cfg.regularization:
        loss_obj.set_regularization_terms(get_regularization_obj(cfg=ft_cfg))

    sf_ds = xr.open_zarr(
        str(ft_cfg.targets_data_dir), decode_timedelta=False
    ).load()
    common_dates = pd.DatetimeIndex(sf_ds.coords['date'].to_numpy())
    obs_series = np.asarray(sf_ds['streamflow'].to_numpy()[0], dtype=np.float32)

    train_mask = (
        common_dates >= pd.Timestamp(str(partition['train_start_date']))
    ) & (common_dates <= pd.Timestamp(str(partition['train_end_date'])))
    val_mask = (
        common_dates >= pd.Timestamp(str(partition['validation_start_date']))
    ) & (common_dates <= pd.Timestamp(str(partition['validation_end_date'])))

    return _EpochEvalContext(
        model=model,
        train_eval_ds=train_eval_ds,
        val_ds=val_ds,
        loss_obj=loss_obj,
        scaler=train_ds.scaler,
        common_dates=common_dates,
        obs_series=obs_series,
        train_mask=train_mask,
        val_mask=val_mask,
        lead_time=int(ft_cfg.lead_time),
        unfrozen_parts=unfrozen_parts,
    )


def run_finetune_epochs(
    username: str,
    basin_id: str,
    model_id: str,
    **options: object,
) -> dict[str, object]:
    """Run ``epochs`` of interactive per-basin fine-tuning and pause."""
    _, base_run_dir = _resolve_model_dir(model_id, username)
    base_ckpt, _ = _find_latest_checkpoint(base_run_dir)
    if base_ckpt is None:
        raise FileNotFoundError(f'No checkpoint found in {base_run_dir}.')

    base_cfg = Config(base_run_dir / 'config.yml', dev_mode=True)
    lead_time = int(base_cfg.lead_time)
    (
        epochs,
        lr_val,
        loss_val,
        lr_strat_val,
        bs_val,
        finetune_modules,
        split_ratio,
        requested_forcing_mode,
    ) = _parse_run_options(options, base_cfg)

    raw_uploaded = _load_raw_uploaded_arrays(username, basin_id)
    has_local_forcing = 'precip' in raw_uploaded and 'temp' in raw_uploaded
    rt_p = _extract_realtime_forcing_matrix(
        username,
        basin_id,
        _REALTIME_PRECIP_CANDIDATES,
        target_lead_time=lead_time,
    )
    rt_t = _extract_realtime_forcing_matrix(
        username,
        basin_id,
        _REALTIME_TEMP_CANDIDATES,
        target_lead_time=lead_time,
    )
    has_global_forcing = rt_p is not None and rt_t is not None
    resolved_forcing_mode = _resolve_forcing_mode(
        requested_forcing_mode,
        has_local_forcing=has_local_forcing,
        has_global_forcing=has_global_forcing,
    )

    if (
        _rebuild_local_zarr_stores(
            username,
            basin_id,
            target_lead_time=lead_time,
            forcing_mode=resolved_forcing_mode,
        )
        is None
    ):
        raise ValueError(
            'Streamflow and input forcing data (global archive or uploaded '
            'local precipitation/temperature CSVs) are required before '
            'fine-tuning. Extract global forcings for this catchment or '
            'upload local CSVs.'
        )

    loaded = _load_uploaded_arrays(
        username,
        basin_id,
        target_lead_time=lead_time,
        forcing_mode=resolved_forcing_mode,
    )
    _validate_uploaded_forecast_leads(loaded, lead_time)

    start_d = max(loaded[k][0][0] for k in ('precip', 'temp', 'streamflow'))
    end_d = min(loaded[k][0][-1] for k in ('precip', 'temp', 'streamflow'))
    common_dates = pd.date_range(start_d, end_d, freq='D')

    partition = compute_partition_dates(
        common_dates,
        base_seq_length=int(base_cfg.seq_length),
        lead_time=lead_time,
        split_ratio=split_ratio,
        predict_last_n=int(base_cfg.predict_last_n or lead_time),
    )

    zarr_root = _basin_data_root(username, basin_id) / 'zarr_store'
    use_global = resolved_forcing_mode != 'local_only'
    global_dyn_dir, attr_path = _prepare_finetune_zarr_stores(
        username,
        base_cfg,
        basin_id,
        common_dates,
        use_global_data=use_global,
    )
    has_native_global = use_global and _has_native_global_inputs(
        global_dyn_dir, base_cfg
    )
    include_local = (resolved_forcing_mode != 'global_only') or (
        not has_native_global
    )

    sess_dir = _session_root(username, basin_id, model_id)
    run_dir = sess_dir / 'run'
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = sess_dir / 'session_state.json'
    session_state: dict[str, object] = (
        json.loads(state_path.read_text(encoding='utf-8'))
        if state_path.exists()
        else {}
    )

    prev_split = session_state.get('split_date')
    prev_mode = session_state.get('forcing_mode')
    if (prev_split is not None and prev_split != partition['split_date']) or (
        prev_mode is not None and prev_mode != resolved_forcing_mode
    ):
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        session_state = {}

    ft_cfg = _build_finetune_config(
        _FinetuneConfigParams(
            base_cfg=base_cfg,
            base_run_dir=base_run_dir,
            run_dir=run_dir,
            zarr_root=zarr_root,
            global_dyn_dir=global_dyn_dir,
            attr_path=attr_path,
            partition=partition,
            lead_time=lead_time,
            lr_val=lr_val,
            loss_val=loss_val,
            lr_strat_val=lr_strat_val,
            bs_val=bs_val,
            include_local_forcing=include_local,
        )
    )
    ft_cfg.update_config({'epochs': epochs})

    train_ds = get_dataset(
        cfg=ft_cfg, is_train=True, period='train', compute_scaler=False
    )
    model, current_epoch, unfrozen_parts = _load_or_init_finetune_model(
        ft_cfg, run_dir, base_ckpt, finetune_modules
    )
    eval_ctx = _build_epoch_eval_context(
        model,
        ft_cfg,
        train_ds,
        partition,
        unfrozen_parts,
    )

    raw_history = session_state.get('epoch_history')
    epoch_history = _execute_epoch_training_steps(
        eval_ctx,
        train_ds,
        ft_cfg,
        current_epoch,
        list(raw_history) if isinstance(raw_history, list) else [],
    )

    split_ratio_out = float(str(partition['split_ratio']))
    session_state = {
        'basin_id': basin_id,
        'model_id': model_id,
        'current_epoch': current_epoch + epochs,
        'status': 'paused',
        'split_ratio': split_ratio_out,
        'split_date': partition['split_date'],
        'forcing_mode': resolved_forcing_mode,
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
    state_path.write_text(json.dumps(session_state, indent=2), encoding='utf-8')

    return get_finetune_workspace_status(
        username,
        basin_id,
        model_id,
        split_ratio=split_ratio_out,
        forcing_mode=resolved_forcing_mode,
    )
