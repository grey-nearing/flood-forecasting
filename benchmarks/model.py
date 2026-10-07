# Copyright 2025 Google LLC
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

"""Canonical benchmark suite for OpenHydroNet (`model`) evaluation.

Provides three core sub-benchmarks with strict zero-imputation, zero-masking,
and zero-fallback guarantees:

1. ``compare_forcings``: Evaluates an identical trained model checkpoint and
   scaler against two MultiMet forcing directories (e.g. canonical
   ``gs://caravan-multimet/v1.1`` vs. locally reconstructed MultiMet archives)
   and quantifies per-basin, per-lead-time prediction and hydrological skill
   differences (NSE, KGE, RMSE, Alpha-NSE, Beta-NSE, Pearson-r, FHV, FMS, FLV).
2. ``benchmark_architectures``: Benchmarks ``mean_embedding_forecast_lstm`` and
   ``handoff_forecast_lstm`` (across regression, probabilistic CMAL, and 1-day
   no-overlap configurations) for training/evaluation throughput, memory
   footprint (``lazy_load=False`` vs. ``lazy_load=True`` and basin-window
   scaling), multi-lead accuracy, and forcing sensitivity.
3. ``benchmark_hot_start``: Benchmarks both (a) LSTM state-handoff hot-start
   inference (``save_state`` + ``load_state_from_disk`` at ``seq_length=0`` vs.
   cold-start ``seq_length=S`` spinup) and (b) checkpoint warm-start
   fine-tuning convergence vs. cold-start training from scratch.
"""

import argparse
import copy
import gc
import json
import resource
import shutil
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from model.datasetzoo import get_dataset
from model.datasetzoo.multimet import MultimetDataLoader, _open_zarr
from model.datautils.utils import load_basin_file
from model.evaluation.evaluate import start_evaluation
from model.evaluation.metrics import calculate_metrics
from model.evaluation.tester import RegressionTester
from model.evaluation.utils import BasinBatchSampler, get_samples_indexes
from model.training.train import start_training
from model.utils.config import Config

DEFAULT_HINDCAST_INPUTS: dict[str, list[str]] = {
    'cpc': ['cpc_precipitation'],
    'imerg': ['imerg_precipitation'],
    'hres': [
        'hres_total_precipitation',
        'hres_temperature_2m',
        'hres_surface_pressure',
        'hres_surface_net_solar_radiation',
        'hres_surface_net_thermal_radiation',
    ],
}

DEFAULT_FORECAST_INPUTS: dict[str, list[str]] = {
    'hres': [
        'hres_total_precipitation',
        'hres_temperature_2m',
        'hres_surface_pressure',
        'hres_surface_net_solar_radiation',
        'hres_surface_net_thermal_radiation',
    ],
}

DEFAULT_STATIC_ATTRIBUTES: list[str] = [
    'area',
    'p_mean',
    'pet_mean_ERA5_LAND',
    'aridity_ERA5_LAND',
    'ele_mt_smn',
    'slp_dg_sav',
]

BENCHMARK_METRICS: list[str] = [
    'NSE',
    'KGE',
    'MSE',
    'RMSE',
    'Alpha-NSE',
    'Beta-NSE',
    'Pearson-r',
    'FHV',
    'FMS',
    'FLV',
]


def _get_current_rss_mb() -> float:
    """Return current resident set size (VmRSS) in MiB from /proc/self/status."""
    status_path = Path('/proc/self/status')
    if status_path.is_file():
        for line in status_path.read_text().splitlines():
            if line.startswith('VmRSS:'):
                parts = line.split()
                return float(parts[1]) / 1024.0
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return float(rusage.ru_maxrss) / 1024.0


def _get_max_rss_mb() -> float:
    """Return peak resident set size (ru_maxrss) in MiB."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return float(rusage.ru_maxrss) / 1024.0


def _assert_no_fallback_or_imputation(cfg: Config) -> None:
    """Enforce hard benchmark rule: zero union_mapping fallback and zero basin skipping."""
    if cfg.union_mapping:
        raise ValueError(
            'Benchmark violation: union_mapping fallback is strictly prohibited '
            f'during benchmark evaluation, got {cfg.union_mapping}.'
        )
    if cfg.tester_skip_obs_all_nan:
        raise ValueError(
            'Benchmark violation: tester_skip_obs_all_nan must be False so no '
            'basins are silently excluded during evaluation.'
        )


def _build_base_config_dict(
    *,
    experiment_name: str,
    run_dir: Path,
    dynamics_data_dir: Path,
    caravan_dir: Path,
    basin_file: Path,
    model_name: str = 'mean_embedding_forecast_lstm',
    head: str = 'regression',
    seq_length: int = 30,
    lead_time: int = 7,
    forecast_overlap: int | None = None,
    predict_last_n: int | None = None,
    hidden_size: int = 32,
    batch_size: int = 32,
    epochs: int = 2,
    seed: int = 42,
    lazy_load: bool = False,
    train_start_date: str = '01/04/2018',
    train_end_date: str = '15/05/2018',
    test_start_date: str = '01/04/2018',
    test_end_date: str = '20/06/2018',
    hindcast_inputs: dict[str, list[str]] | list[str] | None = None,
    forecast_inputs: dict[str, list[str]] | list[str] | None = None,
    static_attributes: list[str] | None = None,
    target_variables: list[str] | None = None,
) -> dict[str, Any]:
    """Build a validated Config dictionary with zero fallback or basin masking."""
    arch_key = model_name.lower()
    effective_model = arch_key
    effective_head = head.lower()
    effective_lead_time = lead_time
    effective_overlap = forecast_overlap
    effective_predict_last_n = predict_last_n

    if arch_key == 'cudalstm':
        effective_model = 'handoff_forecast_lstm'
        effective_head = 'regression'
        effective_lead_time = 1
        effective_overlap = 1
        effective_predict_last_n = 2
    elif arch_key.endswith('_cmal'):
        effective_model = arch_key.removesuffix('_cmal')
        effective_head = 'cmal'

    if effective_overlap is None:
        if effective_model == 'mean_embedding_forecast_lstm':
            effective_overlap = seq_length
        else:
            effective_overlap = min(10, max(1, seq_length // 2))
    elif effective_model == 'handoff_forecast_lstm' and seq_length > 1:
        effective_overlap = min(effective_overlap, seq_length - 1)

    if effective_predict_last_n is None:
        effective_predict_last_n = max(2, effective_lead_time + 1)

    h_inputs = copy.deepcopy(
        DEFAULT_HINDCAST_INPUTS if hindcast_inputs is None else hindcast_inputs
    )
    f_inputs = copy.deepcopy(
        DEFAULT_FORECAST_INPUTS if forecast_inputs is None else forecast_inputs
    )
    s_attrs = list(
        DEFAULT_STATIC_ATTRIBUTES
        if static_attributes is None
        else static_attributes
    )
    t_vars = list(
        ['streamflow'] if target_variables is None else target_variables
    )

    emb_dim = max(16, hidden_size)
    cfg_dict: dict[str, Any] = {
        'experiment_name': experiment_name,
        'run_dir': str(run_dir),
        'dataset': 'multimet',
        'train_basin_file': str(basin_file),
        'validation_basin_file': str(basin_file),
        'test_basin_file': str(basin_file),
        'targets_data_dir': str(caravan_dir),
        'statics_data_dir': str(caravan_dir),
        'dynamics_data_dir': str(dynamics_data_dir),
        'train_start_date': train_start_date,
        'train_end_date': train_end_date,
        'validation_start_date': test_start_date,
        'validation_end_date': test_end_date,
        'test_start_date': test_start_date,
        'test_end_date': test_end_date,
        'hindcast_inputs': h_inputs,
        'forecast_inputs': f_inputs,
        'static_attributes': s_attrs,
        'target_variables': t_vars,
        'model': effective_model,
        'hidden_size': hidden_size,
        'head': effective_head,
        'output_activation': 'linear',
        'statics_embedding': {
            'type': 'fc',
            'hiddens': [emb_dim],
            'activation': ['tanh'],
            'dropout': 0.0,
        },
        'hindcast_embedding': {
            'type': 'fc',
            'hiddens': [emb_dim],
            'activation': ['tanh'],
            'dropout': 0.0,
        },
        'forecast_embedding': {
            'type': 'fc',
            'hiddens': [emb_dim],
            'activation': ['tanh'],
            'dropout': 0.0,
        },
        'seq_length': seq_length,
        'lead_time': effective_lead_time,
        'forecast_overlap': effective_overlap,
        'timestep_counter': True,
        'output_dropout': 0.0,
        'compile': False,
        'device': 'cpu',
        'seed': seed,
        'loss': 'CMAL' if effective_head == 'cmal' else 'MSE',
        'optimizer': 'Adam',
        'epochs': epochs,
        'save_weights_every': 1,
        'batch_size': batch_size,
        'initial_learning_rate': 0.001,
        'clip_gradient_norm': 1.0,
        'metrics': ['NSE', 'KGE', 'RMSE'],
        'predict_last_n': effective_predict_last_n,
        'num_workers': 0,
        'validate_every': None,
        'validate_n_random_basins': 0,
        'log_n_figures': 0,
        'log_tensorboard': False,
        'lazy_load': lazy_load,
        'tester_skip_obs_all_nan': False,
        'inference_mode': True,
        'cache': {'enabled': False},
    }

    if effective_model == 'handoff_forecast_lstm':
        cfg_dict['state_handoff_network'] = {
            'type': 'fc',
            'hiddens': [emb_dim],
            'activation': ['tanh'],
            'dropout': 0.0,
        }
        if effective_head == 'regression' and effective_overlap > 0:
            cfg_dict['regularization'] = ['forecast_overlap']

    if effective_head == 'cmal':
        cfg_dict['n_distributions'] = 3
        cfg_dict['n_samples'] = 20

    return cfg_dict


def _copy_trained_run_for_eval(
    trained_run_dir: Path,
    eval_root_dir: Path,
    dynamics_data_dir: Path,
    epoch: int,
) -> tuple[Config, Path]:
    """Create an isolated copy of trained checkpoint + scaler for a specific forcing dir."""
    if eval_root_dir.exists():
        shutil.rmtree(eval_root_dir)
    eval_root_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(trained_run_dir / 'config.yml', eval_root_dir / 'config.yml')
    ckpt_name = f'model_epoch{epoch:03d}.pt'
    shutil.copy2(trained_run_dir / ckpt_name, eval_root_dir / ckpt_name)
    if (trained_run_dir / 'scaler.zarr').exists():
        shutil.copytree(
            trained_run_dir / 'scaler.zarr', eval_root_dir / 'scaler.zarr'
        )
    if (trained_run_dir / 'train_data').exists():
        shutil.copytree(
            trained_run_dir / 'train_data', eval_root_dir / 'train_data'
        )

    cfg = Config(eval_root_dir / 'config.yml')
    cfg.update_config(
        {
            'run_dir': eval_root_dir,
            'dynamics_data_dir': Path(dynamics_data_dir),
            'tester_skip_obs_all_nan': False,
            'inference_mode': True,
        }
    )
    _assert_no_fallback_or_imputation(cfg)
    return cfg, eval_root_dir


def _reduce_sim_da(da: xr.DataArray) -> xr.DataArray:
    """Reduce probabilistic sample dimension if present, preserving (date,) DataArray."""
    if 'samples' in da.dims:
        return da.mean(dim='samples')
    return da


def _compute_safe_series_metrics(
    obs_da: xr.DataArray, sim_da: xr.DataArray
) -> dict[str, float]:
    """Compute hydrological metrics on 1D (date,) DataArrays without masking failures."""
    valid_obs_mask = np.isfinite(obs_da.values)
    valid_both_mask = valid_obs_mask & np.isfinite(sim_da.values)
    if int(valid_both_mask.sum()) < 2:
        return {m: float('nan') for m in BENCHMARK_METRICS}
    obs_valid = obs_da.values[valid_both_mask]
    sim_valid = sim_da.values[valid_both_mask]
    if float(np.std(obs_valid)) == 0.0 or float(np.std(sim_valid)) == 0.0:
        mse_val = float(np.mean((sim_valid - obs_valid) ** 2))
        out = {m: float('nan') for m in BENCHMARK_METRICS}
        out['MSE'] = mse_val
        out['RMSE'] = float(np.sqrt(mse_val))
        return out
    return calculate_metrics(obs_da, sim_da, metrics=BENCHMARK_METRICS, resolution='1D')


def compare_forcings(
    canonical_multimet_dir: Path | str,
    reconstructed_multimet_dir: Path | str,
    caravan_dir: Path | str,
    basin_file: Path | str,
    output_dir: Path | str,
    *,
    model_name: str = 'mean_embedding_forecast_lstm',
    head: str = 'regression',
    seq_length: int = 30,
    lead_time: int = 7,
    forecast_overlap: int | None = None,
    predict_last_n: int | None = None,
    hidden_size: int = 32,
    batch_size: int = 32,
    epochs: int = 2,
    seed: int = 42,
    train_start_date: str = '01/04/2018',
    train_end_date: str = '15/05/2018',
    test_start_date: str = '01/04/2018',
    test_end_date: str = '20/06/2018',
    hindcast_inputs: dict[str, list[str]] | list[str] | None = None,
    forecast_inputs: dict[str, list[str]] | list[str] | None = None,
    static_attributes: list[str] | None = None,
    target_variables: list[str] | None = None,
    trained_run_dir: Path | str | None = None,
    epoch: int | None = None,
) -> dict[str, Any]:
    """Compare model predictions and skill metrics across canonical vs. reconstructed forcings.

    Parameters
    ----------
    canonical_multimet_dir : Path | str
        Directory containing canonical MultiMet Zarr stores.
    reconstructed_multimet_dir : Path | str
        Directory containing locally reconstructed MultiMet Zarr stores.
    caravan_dir : Path | str
        Directory containing Caravan streamflow and static attributes.
    basin_file : Path | str
        Text file listing gauge IDs to evaluate.
    output_dir : Path | str
        Directory where CSV comparison tables will be written.

    Returns
    -------
    dict[str, Any]
        Dictionary containing ``summary``, ``per_lead_df``, ``per_basin_df``,
        and ``trained_run_dir``.
    """
    can_dir = Path(canonical_multimet_dir)
    rec_dir = Path(reconstructed_multimet_dir)
    car_dir = Path(caravan_dir)
    bas_file = Path(basin_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not can_dir.exists():
        raise FileNotFoundError(f'Canonical MultiMet directory not found: {can_dir}')
    if not rec_dir.exists():
        raise FileNotFoundError(f'Reconstructed MultiMet directory not found: {rec_dir}')

    configured_basins = load_basin_file(bas_file)
    total_basins = len(configured_basins)
    eval_epoch = epochs if epoch is None else epoch

    _open_zarr.cache_clear()

    if trained_run_dir is None:
        train_root = out_dir / f'train_{model_name}_{head}'
        if train_root.exists():
            shutil.rmtree(train_root)
        train_root.mkdir(parents=True, exist_ok=True)
        cfg_dict = _build_base_config_dict(
            experiment_name=f'bench_{model_name}_{head}',
            run_dir=train_root,
            dynamics_data_dir=can_dir,
            caravan_dir=car_dir,
            basin_file=bas_file,
            model_name=model_name,
            head=head,
            seq_length=seq_length,
            lead_time=lead_time,
            forecast_overlap=forecast_overlap,
            predict_last_n=predict_last_n,
            hidden_size=hidden_size,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            hindcast_inputs=hindcast_inputs,
            forecast_inputs=forecast_inputs,
            static_attributes=static_attributes,
            target_variables=target_variables,
        )
        train_cfg = Config(cfg_dict)
        _assert_no_fallback_or_imputation(train_cfg)
        start_training(train_cfg)
        actual_run_dir = sorted(train_root.glob('*'))[-1]
    else:
        actual_run_dir = Path(trained_run_dir)

    _open_zarr.cache_clear()
    cfg_can, eval_can_dir = _copy_trained_run_for_eval(
        actual_run_dir,
        out_dir / f'eval_canonical_{model_name}_{head}',
        can_dir,
        eval_epoch,
    )
    t0_can = time.perf_counter()
    start_evaluation(cfg=cfg_can, run_dir=eval_can_dir, epoch=eval_epoch, period='test')
    can_eval_sec = time.perf_counter() - t0_can

    _open_zarr.cache_clear()
    cfg_rec, eval_rec_dir = _copy_trained_run_for_eval(
        actual_run_dir,
        out_dir / f'eval_reconstructed_{model_name}_{head}',
        rec_dir,
        eval_epoch,
    )
    t0_rec = time.perf_counter()
    start_evaluation(cfg=cfg_rec, run_dir=eval_rec_dir, epoch=eval_epoch, period='test')
    rec_eval_sec = time.perf_counter() - t0_rec

    zarr_can_path = eval_can_dir / 'test' / f'model_epoch{eval_epoch:03d}' / 'test_results.zarr'
    zarr_rec_path = eval_rec_dir / 'test' / f'model_epoch{eval_epoch:03d}' / 'test_results.zarr'

    ds_can = xr.open_zarr(zarr_can_path, consolidated=False).load()
    ds_rec = xr.open_zarr(zarr_rec_path, consolidated=False).load()

    target_var = cfg_can.target_variables[0]
    obs_key = f'{target_var}_obs'
    sim_key = f'{target_var}_sim'

    can_basins = [str(b) for b in ds_can['basin'].values]
    rec_basins = [str(b) for b in ds_rec['basin'].values]
    evaluated_basins_set = set(can_basins) & set(rec_basins)
    evaluated_basins = len(evaluated_basins_set)
    missing_basins_count = total_basins - evaluated_basins

    time_steps = [int(ts) for ts in ds_can['time_step'].values if int(ts) >= 1]
    if not time_steps:
        time_steps = [int(ds_can['time_step'].values[-1])]

    per_basin_rows: list[dict[str, Any]] = []
    pred_nan_when_obs_valid_can = 0
    pred_nan_when_obs_valid_rec = 0
    basins_with_nan_metric: set[str] = set()

    for basin in configured_basins:
        if basin not in evaluated_basins_set:
            basins_with_nan_metric.add(basin)
            for ts in time_steps:
                per_basin_rows.append(
                    {
                        'model': model_name,
                        'head': head,
                        'basin': basin,
                        'lead_time': ts,
                        'status': 'MISSING_BASIN',
                        'n_valid_obs': 0,
                        'pred_nan_when_obs_valid_can': 0,
                        'pred_nan_when_obs_valid_rec': 0,
                        'pred_max_abs_diff': float('nan'),
                        'pred_mean_abs_diff': float('nan'),
                        'pred_rmse_diff': float('nan'),
                        'pred_pearson_r': float('nan'),
                        'NSE_canonical': float('nan'),
                        'NSE_reconstructed': float('nan'),
                        'delta_NSE': float('nan'),
                        'abs_delta_NSE': float('nan'),
                        'KGE_canonical': float('nan'),
                        'KGE_reconstructed': float('nan'),
                        'delta_KGE': float('nan'),
                        'abs_delta_KGE': float('nan'),
                        'RMSE_canonical': float('nan'),
                        'RMSE_reconstructed': float('nan'),
                        'delta_RMSE': float('nan'),
                        'FHV_canonical': float('nan'),
                        'FHV_reconstructed': float('nan'),
                        'delta_FHV': float('nan'),
                        'FLV_canonical': float('nan'),
                        'FLV_reconstructed': float('nan'),
                        'delta_FLV': float('nan'),
                        'FMS_canonical': float('nan'),
                        'FMS_reconstructed': float('nan'),
                        'delta_FMS': float('nan'),
                    }
                )
            continue

        for ts in time_steps:
            obs_c = ds_can[obs_key].sel(basin=basin, time_step=ts).drop_vars(
                ['basin', 'time_step'], errors='ignore'
            )
            sim_c = _reduce_sim_da(
                ds_can[sim_key].sel(basin=basin, time_step=ts)
            ).drop_vars(['basin', 'time_step'], errors='ignore')
            obs_r = ds_rec[obs_key].sel(basin=basin, time_step=ts).drop_vars(
                ['basin', 'time_step'], errors='ignore'
            )
            sim_r = _reduce_sim_da(
                ds_rec[sim_key].sel(basin=basin, time_step=ts)
            ).drop_vars(['basin', 'time_step'], errors='ignore')

            obs_valid_c = np.isfinite(obs_c.values)
            obs_valid_r = np.isfinite(obs_r.values)
            nan_c = int(np.sum(obs_valid_c & ~np.isfinite(sim_c.values)))
            nan_r = int(np.sum(obs_valid_r & ~np.isfinite(sim_r.values)))
            pred_nan_when_obs_valid_can += nan_c
            pred_nan_when_obs_valid_rec += nan_r

            m_can = _compute_safe_series_metrics(obs_c, sim_c)
            m_rec = _compute_safe_series_metrics(obs_r, sim_r)

            if (
                np.isnan(m_can['NSE'])
                or np.isnan(m_rec['NSE'])
                or np.isnan(m_can['KGE'])
                or np.isnan(m_rec['KGE'])
                or nan_c > 0
                or nan_r > 0
            ):
                basins_with_nan_metric.add(basin)

            diff_arr = sim_c.values - sim_r.values
            finite_pair = np.isfinite(sim_c.values) & np.isfinite(sim_r.values)
            if int(finite_pair.sum()) >= 2:
                d_sub = np.abs(diff_arr[finite_pair])
                pred_max_abs = float(np.max(d_sub))
                pred_mean_abs = float(np.mean(d_sub))
                pred_rmse = float(np.sqrt(np.mean(diff_arr[finite_pair] ** 2)))
                c_std = float(np.std(sim_c.values[finite_pair]))
                r_std = float(np.std(sim_r.values[finite_pair]))
                if c_std > 0.0 and r_std > 0.0:
                    pred_r = float(
                        np.corrcoef(
                            sim_c.values[finite_pair], sim_r.values[finite_pair]
                        )[0, 1]
                    )
                else:
                    pred_r = 1.0 if pred_max_abs == 0.0 else float('nan')
            else:
                pred_max_abs = float('nan')
                pred_mean_abs = float('nan')
                pred_rmse = float('nan')
                pred_r = float('nan')

            delta_nse = m_rec['NSE'] - m_can['NSE']
            delta_kge = m_rec['KGE'] - m_can['KGE']
            per_basin_rows.append(
                {
                    'model': model_name,
                    'head': head,
                    'basin': basin,
                    'lead_time': ts,
                    'status': 'OK' if (nan_c == 0 and nan_r == 0) else 'NAN_PRED',
                    'n_valid_obs': int(obs_valid_c.sum()),
                    'pred_nan_when_obs_valid_can': nan_c,
                    'pred_nan_when_obs_valid_rec': nan_r,
                    'pred_max_abs_diff': pred_max_abs,
                    'pred_mean_abs_diff': pred_mean_abs,
                    'pred_rmse_diff': pred_rmse,
                    'pred_pearson_r': pred_r,
                    'NSE_canonical': m_can['NSE'],
                    'NSE_reconstructed': m_rec['NSE'],
                    'delta_NSE': delta_nse,
                    'abs_delta_NSE': abs(delta_nse),
                    'KGE_canonical': m_can['KGE'],
                    'KGE_reconstructed': m_rec['KGE'],
                    'delta_KGE': delta_kge,
                    'abs_delta_KGE': abs(delta_kge),
                    'RMSE_canonical': m_can['RMSE'],
                    'RMSE_reconstructed': m_rec['RMSE'],
                    'delta_RMSE': m_rec['RMSE'] - m_can['RMSE'],
                    'Alpha_NSE_canonical': m_can['Alpha-NSE'],
                    'Alpha_NSE_reconstructed': m_rec['Alpha-NSE'],
                    'Beta_NSE_canonical': m_can['Beta-NSE'],
                    'Beta_NSE_reconstructed': m_rec['Beta-NSE'],
                    'FHV_canonical': m_can['FHV'],
                    'FHV_reconstructed': m_rec['FHV'],
                    'delta_FHV': m_rec['FHV'] - m_can['FHV'],
                    'FLV_canonical': m_can['FLV'],
                    'FLV_reconstructed': m_rec['FLV'],
                    'delta_FLV': m_rec['FLV'] - m_can['FLV'],
                    'FMS_canonical': m_can['FMS'],
                    'FMS_reconstructed': m_rec['FMS'],
                    'delta_FMS': m_rec['FMS'] - m_can['FMS'],
                }
            )

    per_basin_df = pd.DataFrame(per_basin_rows)
    per_lead_rows: list[dict[str, Any]] = []
    for ts, grp in per_basin_df.groupby('lead_time', as_index=False):
        per_lead_rows.append(
            {
                'model': model_name,
                'head': head,
                'lead_time': int(ts),
                'total_basins': total_basins,
                'evaluated_basins': evaluated_basins,
                'failed_or_nan_basins': int(
                    grp['NSE_canonical'].isna().sum()
                    + grp['NSE_reconstructed'].isna().sum()
                ),
                'pred_nan_when_obs_valid_can': int(
                    grp['pred_nan_when_obs_valid_can'].sum()
                ),
                'pred_nan_when_obs_valid_rec': int(
                    grp['pred_nan_when_obs_valid_rec'].sum()
                ),
                'median_NSE_canonical': float(grp['NSE_canonical'].median()),
                'median_NSE_reconstructed': float(
                    grp['NSE_reconstructed'].median()
                ),
                'median_abs_delta_NSE': float(grp['abs_delta_NSE'].median()),
                'max_abs_delta_NSE': float(grp['abs_delta_NSE'].max()),
                'median_KGE_canonical': float(grp['KGE_canonical'].median()),
                'median_KGE_reconstructed': float(
                    grp['KGE_reconstructed'].median()
                ),
                'median_abs_delta_KGE': float(grp['abs_delta_KGE'].median()),
                'max_abs_delta_KGE': float(grp['abs_delta_KGE'].max()),
                'median_RMSE_canonical': float(grp['RMSE_canonical'].median()),
                'median_RMSE_reconstructed': float(
                    grp['RMSE_reconstructed'].median()
                ),
                'pred_mean_abs_diff': float(grp['pred_mean_abs_diff'].mean()),
                'pred_max_abs_diff': float(grp['pred_max_abs_diff'].max()),
                'pred_rmse_diff': float(grp['pred_rmse_diff'].mean()),
                'pred_median_pearson_r': float(grp['pred_pearson_r'].median()),
            }
        )
    per_lead_df = pd.DataFrame(per_lead_rows)

    per_basin_csv = out_dir / f'forcings_comparison_per_basin_{model_name}_{head}.csv'
    per_lead_csv = out_dir / f'forcings_comparison_by_lead_{model_name}_{head}.csv'
    per_basin_df.to_csv(per_basin_csv, index=False)
    per_lead_df.to_csv(per_lead_csv, index=False)

    summary = {
        'model': model_name,
        'head': head,
        'seq_length': seq_length,
        'lead_time': lead_time,
        'total_basins': total_basins,
        'evaluated_basins': evaluated_basins,
        'missing_basins_count': missing_basins_count,
        'failed_or_nan_basins': len(basins_with_nan_metric),
        'pred_nan_when_obs_valid_can': pred_nan_when_obs_valid_can,
        'pred_nan_when_obs_valid_rec': pred_nan_when_obs_valid_rec,
        'canonical_eval_sec': can_eval_sec,
        'reconstructed_eval_sec': rec_eval_sec,
        'median_NSE_canonical': float(per_basin_df['NSE_canonical'].median()),
        'median_NSE_reconstructed': float(
            per_basin_df['NSE_reconstructed'].median()
        ),
        'median_abs_delta_NSE': float(per_basin_df['abs_delta_NSE'].median()),
        'max_abs_delta_NSE': float(per_basin_df['abs_delta_NSE'].max()),
        'median_KGE_canonical': float(per_basin_df['KGE_canonical'].median()),
        'median_KGE_reconstructed': float(
            per_basin_df['KGE_reconstructed'].median()
        ),
        'median_abs_delta_KGE': float(per_basin_df['abs_delta_KGE'].median()),
        'max_abs_delta_KGE': float(per_basin_df['abs_delta_KGE'].max()),
        'pred_mean_abs_diff': float(per_basin_df['pred_mean_abs_diff'].mean()),
        'pred_max_abs_diff': float(per_basin_df['pred_max_abs_diff'].max()),
        'pred_rmse_diff': float(per_basin_df['pred_rmse_diff'].mean()),
        'pred_median_pearson_r': float(per_basin_df['pred_pearson_r'].median()),
        'per_basin_csv': str(per_basin_csv),
        'per_lead_csv': str(per_lead_csv),
    }

    return {
        'summary': summary,
        'per_lead_df': per_lead_df,
        'per_basin_df': per_basin_df,
        'trained_run_dir': actual_run_dir,
    }


def benchmark_architectures(
    canonical_multimet_dir: Path | str,
    reconstructed_multimet_dir: Path | str,
    caravan_dir: Path | str,
    basin_file: Path | str,
    output_dir: Path | str,
    *,
    architectures: list[str] | None = None,
    seq_length: int = 30,
    lead_time: int = 7,
    forecast_overlap: int | None = None,
    predict_last_n: int | None = None,
    hidden_size: int = 32,
    batch_size: int = 32,
    epochs: int = 2,
    seed: int = 42,
    train_start_date: str = '01/04/2018',
    train_end_date: str = '15/05/2018',
    test_start_date: str = '01/04/2018',
    test_end_date: str = '20/06/2018',
    hindcast_inputs: dict[str, list[str]] | list[str] | None = None,
    forecast_inputs: dict[str, list[str]] | list[str] | None = None,
    static_attributes: list[str] | None = None,
    target_variables: list[str] | None = None,
) -> dict[str, Any]:
    """Benchmark model architectures for throughput, memory, accuracy, and forcing sensitivity."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    can_dir = Path(canonical_multimet_dir)
    rec_dir = Path(reconstructed_multimet_dir)
    car_dir = Path(caravan_dir)
    bas_file = Path(basin_file)

    arch_list = (
        [
            'mean_embedding_forecast_lstm',
            'handoff_forecast_lstm',
            'handoff_forecast_lstm_cmal',
            'cudalstm',
        ]
        if architectures is None
        else list(architectures)
    )

    arch_rows: list[dict[str, Any]] = []
    lead_dfs: list[pd.DataFrame] = []
    trained_runs: dict[str, Path] = {}

    for arch in arch_list:
        _open_zarr.cache_clear()
        gc.collect()
        rss_before = _get_current_rss_mb()
        tracemalloc.start()

        arch_out_dir = out_dir / f'arch_{arch}'
        if arch_out_dir.exists():
            shutil.rmtree(arch_out_dir)
        arch_out_dir.mkdir(parents=True, exist_ok=True)

        head = 'cmal' if arch.lower().endswith('_cmal') else 'regression'
        cfg_dict = _build_base_config_dict(
            experiment_name=f'arch_{arch}',
            run_dir=arch_out_dir / 'train',
            dynamics_data_dir=can_dir,
            caravan_dir=car_dir,
            basin_file=bas_file,
            model_name=arch,
            head=head,
            seq_length=seq_length,
            lead_time=lead_time,
            forecast_overlap=forecast_overlap,
            predict_last_n=predict_last_n,
            hidden_size=hidden_size,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            lazy_load=False,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            hindcast_inputs=hindcast_inputs,
            forecast_inputs=forecast_inputs,
            static_attributes=static_attributes,
            target_variables=target_variables,
        )
        cfg = Config(cfg_dict)
        _assert_no_fallback_or_imputation(cfg)

        t0_train = time.perf_counter()
        start_training(cfg)
        train_sec = time.perf_counter() - t0_train

        _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _get_current_rss_mb()
        max_rss_mb = _get_max_rss_mb()

        actual_run_dir = sorted((arch_out_dir / 'train').glob('*'))[-1]
        trained_runs[arch] = actual_run_dir

        ckpt_state = torch.load(
            actual_run_dir / f'model_epoch{epochs:03d}.pt',
            map_location='cpu',
            weights_only=True,
        )
        n_params = int(sum(v.numel() for v in ckpt_state.values()))

        train_ds = get_dataset(
            cfg= Config(actual_run_dir / 'config.yml'),
            is_train=False,
            period='train',
            compute_scaler=False,
        )
        n_train_samples = len(train_ds)
        n_train_batches = int(np.ceil(n_train_samples / batch_size))

        comp = compare_forcings(
            canonical_multimet_dir=can_dir,
            reconstructed_multimet_dir=rec_dir,
            caravan_dir=car_dir,
            basin_file=bas_file,
            output_dir=arch_out_dir / 'forcings_comp',
            model_name=arch,
            head=head,
            seq_length=cfg.seq_length,
            lead_time=cfg.lead_time,
            forecast_overlap=cfg.forecast_overlap,
            predict_last_n=cfg.predict_last_n,
            hidden_size=hidden_size,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            hindcast_inputs=hindcast_inputs,
            forecast_inputs=forecast_inputs,
            static_attributes=static_attributes,
            target_variables=target_variables,
            trained_run_dir=actual_run_dir,
            epoch=epochs,
        )
        comp_sum = comp['summary']
        lead_dfs.append(comp['per_lead_df'])

        eval_sec = comp_sum['canonical_eval_sec']
        total_basins = comp_sum['total_basins']

        arch_rows.append(
            {
                'architecture': arch,
                'model_class': cfg.model,
                'head': cfg.head,
                'seq_length': cfg.seq_length,
                'lead_time': cfg.lead_time,
                'forecast_overlap': cfg.forecast_overlap,
                'predict_last_n': cfg.predict_last_n,
                'hidden_size': hidden_size,
                'n_parameters': n_params,
                'epochs': epochs,
                'n_train_samples': n_train_samples,
                'train_total_sec': train_sec,
                'train_sec_per_epoch': train_sec / max(epochs, 1),
                'train_samples_per_sec': (n_train_samples * epochs)
                / max(train_sec, 1e-9),
                'train_batches_per_sec': (n_train_batches * epochs)
                / max(train_sec, 1e-9),
                'eval_total_sec': eval_sec,
                'eval_basins_per_sec': total_basins / max(eval_sec, 1e-9),
                'tracemalloc_peak_mb': tracemalloc_peak_bytes / (1024.0**2),
                'rss_delta_mb': max(0.0, rss_after - rss_before),
                'max_rss_mb': max_rss_mb,
                'total_basins': total_basins,
                'evaluated_basins': comp_sum['evaluated_basins'],
                'failed_or_nan_basins': comp_sum['failed_or_nan_basins'],
                'pred_nan_when_obs_valid_can': comp_sum[
                    'pred_nan_when_obs_valid_can'
                ],
                'pred_nan_when_obs_valid_rec': comp_sum[
                    'pred_nan_when_obs_valid_rec'
                ],
                'median_NSE_canonical': comp_sum['median_NSE_canonical'],
                'median_NSE_reconstructed': comp_sum['median_NSE_reconstructed'],
                'median_abs_delta_NSE': comp_sum['median_abs_delta_NSE'],
                'max_abs_delta_NSE': comp_sum['max_abs_delta_NSE'],
                'median_KGE_canonical': comp_sum['median_KGE_canonical'],
                'median_KGE_reconstructed': comp_sum['median_KGE_reconstructed'],
                'median_abs_delta_KGE': comp_sum['median_abs_delta_KGE'],
                'pred_mean_abs_diff': comp_sum['pred_mean_abs_diff'],
                'pred_max_abs_diff': comp_sum['pred_max_abs_diff'],
                'pred_rmse_diff': comp_sum['pred_rmse_diff'],
                'pred_median_pearson_r': comp_sum['pred_median_pearson_r'],
            }
        )

    # Benchmark memory & dataset loading modes (lazy_load=False vs lazy_load=True and basin window size)
    all_basins = load_basin_file(bas_file)
    half_n = min(len(all_basins), max(2, len(all_basins) // 2))
    mem_rows: list[dict[str, Any]] = []
    for mode_label, is_lazy, basin_subset in [
        ('eager_all_basins', False, all_basins),
        ('lazy_all_basins', True, all_basins),
        ('eager_limited_basins', False, all_basins[:half_n]),
        ('lazy_limited_basins', True, all_basins[:half_n]),
    ]:
        _open_zarr.cache_clear()
        gc.collect()
        sub_basin_file = out_dir / f'basins_{mode_label}.txt'
        sub_basin_file.write_text('\n'.join(basin_subset) + '\n')
        mem_run_dir = out_dir / f'mem_{mode_label}'
        if mem_run_dir.exists():
            shutil.rmtree(mem_run_dir)
        mem_run_dir.mkdir(parents=True, exist_ok=True)

        mem_cfg_dict = _build_base_config_dict(
            experiment_name=f'mem_{mode_label}',
            run_dir=mem_run_dir,
            dynamics_data_dir=can_dir,
            caravan_dir=car_dir,
            basin_file=sub_basin_file,
            model_name='mean_embedding_forecast_lstm',
            head='regression',
            seq_length=seq_length,
            lead_time=lead_time,
            forecast_overlap=forecast_overlap,
            predict_last_n=predict_last_n,
            hidden_size=hidden_size,
            batch_size=batch_size,
            epochs=1,
            seed=seed,
            lazy_load=is_lazy,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            hindcast_inputs=hindcast_inputs,
            forecast_inputs=forecast_inputs,
            static_attributes=static_attributes,
            target_variables=target_variables,
        )
        if hasattr(Config, 'limit_n_basins') and 'limited' in mode_label:
            mem_cfg_dict['limit_n_basins'] = half_n

        tracemalloc.start()
        t0_ds = time.perf_counter()
        ds_obj = get_dataset(
            cfg=Config(mem_cfg_dict),
            is_train=True,
            period='train',
            compute_scaler=True,
        )
        ds_init_sec = time.perf_counter() - t0_ds
        _, ds_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_rows.append(
            {
                'mode': mode_label,
                'lazy_load': is_lazy,
                'n_basins': len(basin_subset),
                'limit_n_basins_supported': hasattr(Config, 'limit_n_basins'),
                'n_samples': len(ds_obj),
                'dataset_init_sec': ds_init_sec,
                'tracemalloc_peak_mb': ds_peak_bytes / (1024.0**2),
            }
        )

    arch_df = pd.DataFrame(arch_rows)
    mem_df = pd.DataFrame(mem_rows)
    all_leads_df = (
        pd.concat(lead_dfs, ignore_index=True) if lead_dfs else pd.DataFrame()
    )

    arch_csv = out_dir / 'architecture_benchmark.csv'
    mem_csv = out_dir / 'memory_scaling_benchmark.csv'
    leads_csv = out_dir / 'architecture_by_lead_benchmark.csv'
    arch_df.to_csv(arch_csv, index=False)
    mem_df.to_csv(mem_csv, index=False)
    all_leads_df.to_csv(leads_csv, index=False)

    return {
        'architecture_df': arch_df,
        'memory_df': mem_df,
        'by_lead_df': all_leads_df,
        'trained_runs': trained_runs,
        'architecture_csv': str(arch_csv),
        'memory_csv': str(mem_csv),
    }


def benchmark_hot_start(
    canonical_multimet_dir: Path | str,
    caravan_dir: Path | str,
    basin_file: Path | str,
    output_dir: Path | str,
    *,
    seq_length: int = 30,
    lead_time: int = 7,
    forecast_overlap: int | None = None,
    predict_last_n: int | None = None,
    hidden_size: int = 32,
    batch_size: int = 32,
    epochs: int = 3,
    seed: int = 42,
    train_start_date: str = '01/04/2018',
    train_end_date: str = '15/05/2018',
    test_start_date: str = '01/04/2018',
    test_end_date: str = '20/06/2018',
    hindcast_inputs: dict[str, list[str]] | list[str] | None = None,
    forecast_inputs: dict[str, list[str]] | list[str] | None = None,
    static_attributes: list[str] | None = None,
    target_variables: list[str] | None = None,
) -> dict[str, Any]:
    """Benchmark (1) LSTM state-handoff hot-start vs cold-start inference and
    (2) checkpoint warm-start fine-tuning vs cold-start epoch convergence."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    can_dir = Path(canonical_multimet_dir)
    car_dir = Path(caravan_dir)
    bas_file = Path(basin_file)
    configured_basins = load_basin_file(bas_file)

    state_rows: list[dict[str, Any]] = []
    for model_name in ['handoff_forecast_lstm', 'mean_embedding_forecast_lstm']:
        _open_zarr.cache_clear()
        hs_root = out_dir / f'hot_start_state_{model_name}'
        if hs_root.exists():
            shutil.rmtree(hs_root)
        hs_root.mkdir(parents=True, exist_ok=True)

        eff_overlap = (
            seq_length
            if model_name == 'mean_embedding_forecast_lstm'
            else (
                min(10, max(1, seq_length // 2))
                if forecast_overlap is None
                else min(forecast_overlap, max(1, seq_length - 1))
            )
        )
        cfg_dict = _build_base_config_dict(
            experiment_name=f'hs_{model_name}',
            run_dir=hs_root,
            dynamics_data_dir=can_dir,
            caravan_dir=car_dir,
            basin_file=bas_file,
            model_name=model_name,
            head='regression',
            seq_length=seq_length,
            lead_time=lead_time,
            forecast_overlap=eff_overlap,
            predict_last_n=predict_last_n,
            hidden_size=hidden_size,
            batch_size=batch_size,
            epochs=1,
            seed=seed,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            hindcast_inputs=hindcast_inputs,
            forecast_inputs=forecast_inputs,
            static_attributes=static_attributes,
            target_variables=target_variables,
        )
        cfg = Config(cfg_dict)
        _assert_no_fallback_or_imputation(cfg)
        start_training(cfg)
        actual_run_dir = sorted(hs_root.glob('*'))[-1]

        eval_cfg = Config(actual_run_dir / 'config.yml')
        eval_cfg.update_config({'batch_size': 1, 'save_state': True})
        tester = RegressionTester(eval_cfg, run_dir=actual_run_dir, period='test')
        tester._load_weights(epoch=1)
        model = tester.model
        model.eval()

        batch_sampler = BasinBatchSampler(
            sample_index=tester.dataset._sample_index,
            batch_size=1,
            basins_indexes=get_samples_indexes(
                tester.basins, samples=list(tester.basins)
            ),
        )
        loader = MultimetDataLoader(
            tester.dataset,
            lazy_load=eval_cfg.lazy_load,
            logging_level=eval_cfg.logging_level,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=tester.dataset.collate_fn,
            pin_memory=False,
        )

        # Group samples by basin and take the final window per basin to test state save + hot reload
        basin_last_sample: dict[str, dict[str, Any]] = {}
        for batch in loader:
            b_idx = int(batch['basin_index'][0].item())
            b_name = tester.dataset._basins[b_idx]
            basin_last_sample[b_name] = batch

        state_dir = hs_root / 'saved_states'
        state_dir.mkdir(parents=True, exist_ok=True)

        cold_times_ms: list[float] = []
        hot_times_ms: list[float] = []
        save_times_ms: list[float] = []
        max_diffs: list[float] = []
        rmse_diffs: list[float] = []
        failed_basins = 0

        for basin in configured_basins:
            if basin not in basin_last_sample:
                failed_basins += 1
                continue
            data = basin_last_sample[basin]
            state_path = state_dir / f'state_{basin}.npz'

            model.reset_state()
            model.seq_length = seq_length
            eval_cfg.seq_length = seq_length

            with torch.inference_mode():
                t0_save = time.perf_counter()
                model.save_state(data, state_path)
                save_times_ms.append((time.perf_counter() - t0_save) * 1000.0)

                model.reset_state()
                t0_cold = time.perf_counter()
                cold_out = model(data)['y_hat'][:, -lead_time:, :]
                cold_times_ms.append((time.perf_counter() - t0_cold) * 1000.0)

                model.reset_state()
                model.load_state_from_disk(state_path)
                model.seq_length = 0
                eval_cfg.seq_length = 0
                hot_data = dict(data)
                hot_data['x_d_hindcast'] = {
                    k: v[:, :0, :] for k, v in data['x_d_hindcast'].items()
                }
                if model_name == 'mean_embedding_forecast_lstm':
                    hot_data['x_d_forecast'] = {
                        k: v[:, -lead_time:, :]
                        for k, v in data['x_d_forecast'].items()
                    }
                t0_hot = time.perf_counter()
                hot_out = model(hot_data)['y_hat'][:, -lead_time:, :]
                hot_times_ms.append((time.perf_counter() - t0_hot) * 1000.0)

                model.seq_length = seq_length
                eval_cfg.seq_length = seq_length

            diff = (cold_out - hot_out).abs().detach().cpu().numpy()
            if not np.all(np.isfinite(diff)):
                failed_basins += 1
            max_diffs.append(float(np.max(diff)))
            rmse_diffs.append(float(np.sqrt(np.mean(diff**2))))

        mean_cold_ms = float(np.mean(cold_times_ms))
        mean_hot_ms = float(np.mean(hot_times_ms))
        state_rows.append(
            {
                'model': model_name,
                'seq_length': seq_length,
                'lead_time': lead_time,
                'forecast_overlap': eff_overlap,
                'total_basins': len(configured_basins),
                'evaluated_basins': len(max_diffs),
                'failed_or_nan_basins': failed_basins,
                'mean_save_state_ms': float(np.mean(save_times_ms)),
                'mean_cold_forward_ms': mean_cold_ms,
                'mean_hot_forward_ms': mean_hot_ms,
                'hot_start_speedup_x': mean_cold_ms / max(mean_hot_ms, 1e-9),
                'max_abs_diff': float(np.max(max_diffs)),
                'median_max_abs_diff': float(np.median(max_diffs)),
                'mean_rmse_diff': float(np.mean(rmse_diffs)),
            }
        )

    # Part 2: Checkpoint warm-start (finetuning from pre-trained base run) vs cold-start training convergence
    _open_zarr.cache_clear()
    conv_root = out_dir / 'hot_start_convergence'
    if conv_root.exists():
        shutil.rmtree(conv_root)
    conv_root.mkdir(parents=True, exist_ok=True)

    pretrain_cfg_dict = _build_base_config_dict(
        experiment_name='pretrain_base',
        run_dir=conv_root / 'pretrain',
        dynamics_data_dir=can_dir,
        caravan_dir=car_dir,
        basin_file=bas_file,
        model_name='mean_embedding_forecast_lstm',
        head='regression',
        seq_length=seq_length,
        lead_time=lead_time,
        forecast_overlap=forecast_overlap,
        predict_last_n=predict_last_n,
        hidden_size=hidden_size,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        hindcast_inputs=hindcast_inputs,
        forecast_inputs=forecast_inputs,
        static_attributes=static_attributes,
        target_variables=target_variables,
    )
    start_training(Config(pretrain_cfg_dict))
    base_run_dir = sorted((conv_root / 'pretrain').glob('*'))[-1]

    # Cold-start run with different seed (seed + 1) vs Warm-start finetuning from base_run_dir
    cold_cfg_dict = _build_base_config_dict(
        experiment_name='cold_start_train',
        run_dir=conv_root / 'cold_start',
        dynamics_data_dir=can_dir,
        caravan_dir=car_dir,
        basin_file=bas_file,
        model_name='mean_embedding_forecast_lstm',
        head='regression',
        seq_length=seq_length,
        lead_time=lead_time,
        forecast_overlap=forecast_overlap,
        predict_last_n=predict_last_n,
        hidden_size=hidden_size,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed + 1,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        hindcast_inputs=hindcast_inputs,
        forecast_inputs=forecast_inputs,
        static_attributes=static_attributes,
        target_variables=target_variables,
    )
    t0_cold_tr = time.perf_counter()
    start_training(Config(cold_cfg_dict))
    cold_train_sec = time.perf_counter() - t0_cold_tr
    cold_run_dir = sorted((conv_root / 'cold_start').glob('*'))[-1]

    warm_cfg_dict = _build_base_config_dict(
        experiment_name='warm_start_finetune',
        run_dir=conv_root / 'warm_start',
        dynamics_data_dir=can_dir,
        caravan_dir=car_dir,
        basin_file=bas_file,
        model_name='mean_embedding_forecast_lstm',
        head='regression',
        seq_length=seq_length,
        lead_time=lead_time,
        forecast_overlap=forecast_overlap,
        predict_last_n=predict_last_n,
        hidden_size=hidden_size,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed + 1,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        hindcast_inputs=hindcast_inputs,
        forecast_inputs=forecast_inputs,
        static_attributes=static_attributes,
        target_variables=target_variables,
    )
    warm_cfg_dict.update(
        {
            'base_run_dir': str(base_run_dir),
            'is_finetuning': True,
            'finetune_modules': ['hindcast_lstm', 'forecast_lstm', 'head'],
        }
    )
    t0_warm_tr = time.perf_counter()
    start_training(Config(warm_cfg_dict))
    warm_train_sec = time.perf_counter() - t0_warm_tr
    warm_run_dir = sorted((conv_root / 'warm_start').glob('*'))[-1]

    conv_rows: list[dict[str, Any]] = []
    for mode_name, r_dir, tot_sec in [
        ('cold_start', cold_run_dir, cold_train_sec),
        ('warm_start_checkpoint', warm_run_dir, warm_train_sec),
    ]:
        r_cfg = Config(r_dir / 'config.yml')
        for ep in range(1, epochs + 1):
            start_evaluation(cfg=r_cfg, run_dir=r_dir, epoch=ep, period='test')
            metrics_csv = (
                r_dir / 'test' / f'model_epoch{ep:03d}' / 'test_metrics.csv'
            )
            df_m = pd.read_csv(metrics_csv)
            conv_rows.append(
                {
                    'init_mode': mode_name,
                    'epoch': ep,
                    'total_basins': len(configured_basins),
                    'evaluated_basins': len(df_m),
                    'failed_or_nan_basins': int(df_m['NSE'].isna().sum()),
                    'median_NSE': float(df_m['NSE'].median()),
                    'mean_NSE': float(df_m['NSE'].mean()),
                    'median_KGE': float(df_m['KGE'].median()),
                    'mean_KGE': float(df_m['KGE'].mean()),
                    'median_RMSE': float(df_m['RMSE'].median()),
                    'mean_RMSE': float(df_m['RMSE'].mean()),
                    'train_sec_per_epoch': tot_sec / max(epochs, 1),
                }
            )

    state_df = pd.DataFrame(state_rows)
    conv_df = pd.DataFrame(conv_rows)
    state_csv = out_dir / 'hot_start_state_benchmark.csv'
    conv_csv = out_dir / 'hot_start_convergence_benchmark.csv'
    state_df.to_csv(state_csv, index=False)
    conv_df.to_csv(conv_csv, index=False)

    return {
        'state_df': state_df,
        'convergence_df': conv_df,
        'state_csv': str(state_csv),
        'convergence_csv': str(conv_csv),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``benchmarks.model``."""
    parser = argparse.ArgumentParser(
        prog='benchmark-model',
        description='Canonical & Reconstructed MultiMet Benchmark Suite for OpenHydroNet.',
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'compare-forcings', 'architectures', 'hot-start'],
        default='all',
        help='Benchmark sub-suite to run.',
    )
    parser.add_argument(
        '--canonical-multimet-dir',
        type=Path,
        required=True,
        help='Path to canonical MultiMet Zarr directory.',
    )
    parser.add_argument(
        '--reconstructed-multimet-dir',
        type=Path,
        required=True,
        help='Path to locally reconstructed MultiMet Zarr directory.',
    )
    parser.add_argument(
        '--caravan-dir',
        type=Path,
        required=True,
        help='Path to Caravan directory (streamflow.zarr and attributes.zarr).',
    )
    parser.add_argument(
        '--basin-file',
        type=Path,
        required=True,
        help='Path to basin list file.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to write benchmark CSV/JSON outputs.',
    )
    parser.add_argument('--seq-length', type=int, default=30)
    parser.add_argument('--lead-time', type=int, default=7)
    parser.add_argument('--forecast-overlap', type=int, default=None)
    parser.add_argument('--predict-last-n', type=int, default=None)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-start-date', type=str, default='01/04/2018')
    parser.add_argument('--train-end-date', type=str, default='15/05/2018')
    parser.add_argument('--test-start-date', type=str, default='01/04/2018')
    parser.add_argument('--test-end-date', type=str, default='20/06/2018')
    parser.add_argument(
        '--architectures',
        nargs='+',
        default=None,
        help='Architectures to benchmark in architectures mode.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Entrypoint for ``benchmark-model`` CLI."""
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    if args.mode in ('all', 'compare-forcings'):
        comp_res = compare_forcings(
            canonical_multimet_dir=args.canonical_multimet_dir,
            reconstructed_multimet_dir=args.reconstructed_multimet_dir,
            caravan_dir=args.caravan_dir,
            basin_file=args.basin_file,
            output_dir=args.output_dir / 'compare_forcings',
            seq_length=args.seq_length,
            lead_time=args.lead_time,
            forecast_overlap=args.forecast_overlap,
            predict_last_n=args.predict_last_n,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            train_start_date=args.train_start_date,
            train_end_date=args.train_end_date,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
        )
        results['compare_forcings'] = comp_res['summary']

    if args.mode in ('all', 'architectures'):
        arch_res = benchmark_architectures(
            canonical_multimet_dir=args.canonical_multimet_dir,
            reconstructed_multimet_dir=args.reconstructed_multimet_dir,
            caravan_dir=args.caravan_dir,
            basin_file=args.basin_file,
            output_dir=args.output_dir / 'architectures',
            architectures=args.architectures,
            seq_length=args.seq_length,
            lead_time=args.lead_time,
            forecast_overlap=args.forecast_overlap,
            predict_last_n=args.predict_last_n,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            train_start_date=args.train_start_date,
            train_end_date=args.train_end_date,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
        )
        results['architectures'] = arch_res['architecture_df'].to_dict(
            orient='records'
        )
        results['memory_scaling'] = arch_res['memory_df'].to_dict(
            orient='records'
        )

    if args.mode in ('all', 'hot-start'):
        hs_res = benchmark_hot_start(
            canonical_multimet_dir=args.canonical_multimet_dir,
            caravan_dir=args.caravan_dir,
            basin_file=args.basin_file,
            output_dir=args.output_dir / 'hot_start',
            seq_length=args.seq_length,
            lead_time=args.lead_time,
            forecast_overlap=args.forecast_overlap,
            predict_last_n=args.predict_last_n,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            train_start_date=args.train_start_date,
            train_end_date=args.train_end_date,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
        )
        results['hot_start_state'] = hs_res['state_df'].to_dict(
            orient='records'
        )
        results['hot_start_convergence'] = hs_res['convergence_df'].to_dict(
            orient='records'
        )

    summary_json_path = args.output_dir / 'benchmark_summary.json'
    summary_json_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    main()
