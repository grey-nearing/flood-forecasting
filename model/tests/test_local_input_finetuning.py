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

"""Unit and integration tests for local input fine-tuning."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

from model.datasetzoo.multimet import Multimet, _find_product_zarr_path
from model.datautils.scaler import Scaler
from model.modelzoo import get_model, load_model_weights
from model.run import eval_run, finetune, start_run
from model.training.basetrainer import BaseTrainer
from model.utils.config import Config
from utils.file_paths import SCALER_ZARR_NAME


def _create_synthetic_datasets(tmp_path: Path) -> dict[str, Path]:
    basins = ['camels_01013500', 'camels_01022500']
    dates = pd.date_range('2000-01-01', '2000-06-30', freq='D')
    rng = np.random.default_rng(42)

    data_dir = tmp_path / 'data'
    global_dyn_dir = data_dir / 'global_dyn'
    local_dyn_dir = data_dir / 'local_dyn'
    global_dyn_dir.mkdir(parents=True)
    local_dyn_dir.mkdir(parents=True)

    # Static attributes
    attr_ds = xr.Dataset(
        {
            'area': ('basin', np.array([100.0, 250.0], dtype=np.float32)),
            'p_mean': ('basin', np.array([3.2, 4.1], dtype=np.float32)),
        },
        coords={'basin': basins},
    )
    attr_path = data_dir / 'attributes.zarr'
    attr_ds.to_zarr(attr_path, mode='w')

    # Target streamflow
    sf_ds = xr.Dataset(
        {
            'streamflow': (
                ('basin', 'date'),
                rng.uniform(0.5, 5.0, size=(len(basins), len(dates))).astype(
                    np.float32
                ),
            )
        },
        coords={'basin': basins, 'date': dates},
    )
    sf_path = data_dir / 'streamflow.zarr'
    sf_ds.to_zarr(sf_path, mode='w')

    lead_times = pd.timedelta_range('1D', '7D', freq='D')

    # Global hindcast dynamics: ERA5_LAND, CPC
    for prod, bands in [
        (
            'ERA5_LAND',
            ['era5land_total_precipitation', 'era5land_temperature_2m'],
        ),
        ('CPC', ['cpc_precipitation']),
    ]:
        pdir = global_dyn_dir / prod
        pdir.mkdir(parents=True)
        ds = xr.Dataset(
            {
                b: (
                    ('basin', 'date'),
                    rng.normal(1.0, 0.5, size=(len(basins), len(dates))).astype(
                        np.float32
                    ),
                )
                for b in bands
            },
            coords={'basin': basins, 'date': dates},
        )
        ds.to_zarr(pdir / 'timeseries.zarr', mode='w')

    # Global forecast dynamics: HRES (with lead_time dimension)
    hres_dir = global_dyn_dir / 'HRES'
    hres_dir.mkdir(parents=True)
    hres_ds = xr.Dataset(
        {
            b: (
                ('basin', 'date', 'lead_time'),
                rng.normal(
                    1.0,
                    0.5,
                    size=(len(basins), len(dates), len(lead_times)),
                ).astype(np.float32),
            )
            for b in ['hres_total_precipitation', 'hres_temperature_2m']
        },
        coords={'basin': basins, 'date': dates, 'lead_time': lead_times},
    )
    hres_ds.to_zarr(hres_dir / 'timeseries.zarr', mode='w')

    # Local hindcast dynamics: DAYMET (QPE)
    daymet_dir = local_dyn_dir / 'DAYMET'
    daymet_dir.mkdir(parents=True)
    daymet_ds = xr.Dataset(
        {
            b: (
                ('basin', 'date'),
                rng.normal(2.0, 0.8, size=(len(basins), len(dates))).astype(
                    np.float32
                ),
            )
            for b in ['daymet_prcp', 'daymet_tmax', 'daymet_tmin']
        },
        coords={'basin': basins, 'date': dates},
    )
    daymet_ds.to_zarr(daymet_dir / 'timeseries.zarr', mode='w')

    # Local forecast dynamics: GEFS_REFORECAST (QPF, with lead_time dimension)
    gefs_dir = local_dyn_dir / 'GEFS_REFORECAST'
    gefs_dir.mkdir(parents=True)
    gefs_ds = xr.Dataset(
        {
            b: (
                ('basin', 'date', 'lead_time'),
                rng.normal(
                    2.0,
                    0.8,
                    size=(len(basins), len(dates), len(lead_times)),
                ).astype(np.float32),
            )
            for b in [
                'gefs_reforecast_apcp_sfc',
                'gefs_reforecast_tmp_2m',
            ]
        },
        coords={'basin': basins, 'date': dates, 'lead_time': lead_times},
    )
    gefs_ds.to_zarr(gefs_dir / 'timeseries.zarr', mode='w')

    all_basins_file = tmp_path / 'all_basins.txt'
    all_basins_file.write_text('\n'.join(basins) + '\n')

    single_basin_file = tmp_path / 'single_basin.txt'
    single_basin_file.write_text(basins[0] + '\n')

    return {
        'attr_path': attr_path,
        'sf_path': sf_path,
        'global_dyn_dir': global_dyn_dir,
        'local_dyn_dir': local_dyn_dir,
        'all_basins_file': all_basins_file,
        'single_basin_file': single_basin_file,
    }


def _make_base_pretrain_cfg(
    tmp_path: Path, paths: dict[str, Path]
) -> dict[str, object]:
    embedding_spec = {
        'type': 'fc',
        'hiddens': [8],
        'activation': ['tanh'],
        'dropout': 0.0,
    }
    return {
        'experiment_name': 'pretrain_global',
        'run_dir': str(tmp_path / 'runs'),
        'dataset': 'multimet',
        'model': 'mean_embedding_forecast_lstm',
        'head': 'regression',
        'loss': 'mse',
        'device': 'cpu',
        'seed': 42,
        'epochs': 1,
        'batch_size': 16,
        'optimizer': 'Adam',
        'learning_rate_strategy': 'ReduceLROnPlateau',
        'initial_learning_rate': 0.001,
        'learning_rate_drop_factor': 0.5,
        'learning_rate_epochs_drop': 5,
        'seq_length': 10,
        'lead_time': 3,
        'forecast_overlap': 10,
        'predict_last_n': 3,
        'hidden_size': 16,
        'output_dropout': 0.0,
        'initial_forget_bias': 3,
        'weight_init_opts': [
            'lstm-ih-xavier',
            'lstm-hh-orthogonal',
            'fc-xavier',
        ],
        'statics_embedding': embedding_spec,
        'hindcast_embedding': embedding_spec,
        'forecast_embedding': embedding_spec,
        'target_variables': ['streamflow'],
        'static_attributes': ['area', 'p_mean'],
        'hindcast_inputs': {
            'era5_land': [
                'era5land_total_precipitation',
                'era5land_temperature_2m',
            ],
            'cpc': ['cpc_precipitation'],
            'hres': ['hres_total_precipitation', 'hres_temperature_2m'],
        },
        'forecast_inputs': {
            'hres': ['hres_total_precipitation', 'hres_temperature_2m'],
        },
        'nan_handling_method': 'masked_mean',
        'train_basin_file': str(paths['all_basins_file']),
        'validation_basin_file': str(paths['all_basins_file']),
        'test_basin_file': str(paths['all_basins_file']),
        'train_start_date': '01/01/2000',
        'train_end_date': '30/04/2000',
        'validation_start_date': '01/05/2000',
        'validation_end_date': '31/05/2000',
        'test_start_date': '01/05/2000',
        'test_end_date': '30/06/2000',
        'statics_data_dir': str(paths['attr_path']),
        'targets_data_dir': str(paths['sf_path']),
        'dynamics_data_dir': str(paths['global_dyn_dir']),
        'validate_every': 1,
        'validate_n_random_basins': 2,
        'save_weights_every': 1,
        'metrics': ['NSE'],
        'num_workers': 0,
    }


def _write_minimal_scaler(run_dir: Path) -> None:
    """Write a real minimal scaler.zarr to run_dir for unit tests."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            'streamflow': (
                ('basin', 'date'),
                np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
            ),
            'area': (('basin',), np.array([100.0, 200.0], dtype=np.float32)),
            'p_mean': (('basin',), np.array([3.0, 5.0], dtype=np.float32)),
        },
        coords={
            'basin': ['camels_01013500', 'camels_01022500'],
            'date': pd.date_range('2000-01-01', periods=2, freq='D'),
        },
    )
    scaler = Scaler(
        scaler_dir=run_dir,
        calculate_scaler=True,
        dataset=ds,
    )
    scaler.scaler = scaler.scaler.compute()
    scaler.save()


def _assert_extended_scaler_matches_raw_slice(
    ft_scaler: xr.Dataset,
    base_scaler: xr.Dataset,
    local_dyn_dir: Path,
) -> None:
    """Verify extended scaler preserves base stats and matches raw slice."""
    for base_var in base_scaler.data_vars:
        xr.testing.assert_equal(ft_scaler[base_var], base_scaler[base_var])

    raw_daymet = (
        xr.open_zarr(local_dyn_dir / 'DAYMET' / 'timeseries.zarr')
        .sel(
            basin=['camels_01013500'],
            date=slice('2000-01-01', '2000-05-03'),
        )
        .load()
    )
    for var in ('daymet_prcp', 'daymet_tmax', 'daymet_tmin'):
        np.testing.assert_allclose(
            float(ft_scaler[var].sel(parameter='center').values),
            float(np.nanmean(raw_daymet[var].values)),
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            float(ft_scaler[var].sel(parameter='scale').values),
            float(np.nanstd(raw_daymet[var].values)),
            rtol=1e-4,
        )

    raw_gefs = (
        xr.open_zarr(local_dyn_dir / 'GEFS_REFORECAST' / 'timeseries.zarr')
        .sel(
            basin=['camels_01013500'],
            date=slice('2000-01-01', '2000-05-03'),
            lead_time=slice(pd.Timedelta(days=1), pd.Timedelta(days=3)),
        )
        .load()
    )
    for var in ('gefs_reforecast_apcp_sfc', 'gefs_reforecast_tmp_2m'):
        np.testing.assert_allclose(
            float(ft_scaler[var].sel(parameter='center').values),
            float(np.nanmean(raw_gefs[var].values)),
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            float(ft_scaler[var].sel(parameter='scale').values),
            float(np.nanstd(raw_gefs[var].values)),
            rtol=1e-4,
        )


@pytest.mark.integration
def test_pretrain_and_local_input_finetune_end_to_end(tmp_path: Path) -> None:
    """Verify end-to-end pretraining, local input fine-tuning, and eval."""
    paths = _create_synthetic_datasets(tmp_path)
    pretrain_cfg = _make_base_pretrain_cfg(tmp_path, paths)
    pretrain_cfg_path = tmp_path / 'pretrain.yml'
    pretrain_cfg_path.write_text(yaml.safe_dump(pretrain_cfg))

    start_run(Config(pretrain_cfg_path), gpu=-1)
    base_run_dir = next((tmp_path / 'runs').glob('pretrain_global*'))
    base_ckpt = torch.load(
        base_run_dir / 'model_epoch001.pt', weights_only=True
    )
    base_scaler = xr.open_zarr(base_run_dir / SCALER_ZARR_NAME).load()

    # Fine-tune Mode A for 2 epochs: only the newly added local embeddings
    ft_cfg_a = {
        'experiment_name': 'ft_local_only',
        'base_run_dir': str(base_run_dir),
        'run_dir': str(tmp_path / 'runs'),
        'epochs': 2,
        'save_weights_every': 1,
        'initial_learning_rate': 0.01,
        'train_basin_file': str(paths['single_basin_file']),
        'validation_basin_file': str(paths['single_basin_file']),
        'test_basin_file': str(paths['single_basin_file']),
        'dynamics_data_dir': [
            str(paths['global_dyn_dir']),
            str(paths['local_dyn_dir']),
        ],
        'hindcast_inputs': {
            'era5_land': [
                'era5land_total_precipitation',
                'era5land_temperature_2m',
            ],
            'cpc': ['cpc_precipitation'],
            'hres': ['hres_total_precipitation', 'hres_temperature_2m'],
            'daymet': ['daymet_prcp', 'daymet_tmax', 'daymet_tmin'],
        },
        'forecast_inputs': {
            'hres': ['hres_total_precipitation', 'hres_temperature_2m'],
            'gefs_reforecast': [
                'gefs_reforecast_apcp_sfc',
                'gefs_reforecast_tmp_2m',
            ],
        },
        'finetune_modules': {
            'hindcast_embeddings_fc': ['daymet'],
            'forecast_embeddings_fc': ['gefs_reforecast'],
        },
    }
    ft_cfg_a_path = tmp_path / 'finetune_a.yml'
    ft_cfg_a_path.write_text(yaml.safe_dump(ft_cfg_a))

    finetune(ft_cfg_a_path, gpu=-1)
    ft_a_dir = next((tmp_path / 'runs').glob('ft_local_only*'))
    ft_a_ckpt_ep1 = torch.load(
        ft_a_dir / 'model_epoch001.pt', weights_only=True
    )
    ft_a_ckpt = torch.load(ft_a_dir / 'model_epoch002.pt', weights_only=True)

    ft_scaler = xr.open_zarr(ft_a_dir / SCALER_ZARR_NAME).load()
    _assert_extended_scaler_matches_raw_slice(
        ft_scaler, base_scaler, paths['local_dyn_dir']
    )

    # Verify frozen vs unfrozen parameters in Mode A, and verify that the
    # fine-tuned local embeddings numerically changed across epochs!
    for key, base_val in base_ckpt.items():
        assert torch.equal(base_val, ft_a_ckpt[key]), (
            f'Pretrained parameter {key} should be frozen in Mode A'
        )
    daymet_key = '_orig_mod.hindcast_embeddings_fc.daymet.net.0.weight'
    gefs_key = '_orig_mod.forecast_embeddings_fc.gefs_reforecast.net.0.weight'
    assert torch.isfinite(ft_a_ckpt[daymet_key]).all()
    assert torch.isfinite(ft_a_ckpt[gefs_key]).all()
    assert not torch.allclose(ft_a_ckpt_ep1[daymet_key], ft_a_ckpt[daymet_key])
    assert not torch.allclose(ft_a_ckpt_ep1[gefs_key], ft_a_ckpt[gefs_key])

    # Verify evaluation works on the fine-tuned run and writes finite metrics
    eval_run(
        Config(ft_a_dir / 'config.yml'),
        run_dir=ft_a_dir,
        period='test',
        epoch=2,
        gpu=-1,
    )
    metrics_file = ft_a_dir / 'test' / 'model_epoch002' / 'test_metrics.csv'
    assert metrics_file.exists()
    metrics_df = pd.read_csv(metrics_file)
    assert np.isfinite(metrics_df['NSE']).all()
    results_ds = xr.open_zarr(
        ft_a_dir / 'test' / 'model_epoch002' / 'test_results.zarr',
        consolidated=False,
    ).load()
    assert np.isfinite(results_ds['streamflow_sim'].values).all()

    # Fine-tune Mode C: local embeddings + static_embedding_fc + head
    ft_cfg_c = dict(ft_cfg_a)
    ft_cfg_c['experiment_name'] = 'ft_local_static_head'
    ft_cfg_c['epochs'] = 1
    ft_cfg_c['finetune_modules'] = {
        'hindcast_embeddings_fc': ['daymet'],
        'forecast_embeddings_fc': ['gefs_reforecast'],
        'static_embedding_fc': True,
        'head': True,
    }
    ft_cfg_c_path = tmp_path / 'finetune_c.yml'
    ft_cfg_c_path.write_text(yaml.safe_dump(ft_cfg_c))

    finetune(ft_cfg_c_path, gpu=-1)
    ft_c_dir = next((tmp_path / 'runs').glob('ft_local_static_head*'))
    ft_c_ckpt = torch.load(ft_c_dir / 'model_epoch001.pt', weights_only=True)

    # LSTMs and global embeddings remain frozen; head & static_embedding update
    assert torch.equal(
        base_ckpt['_orig_mod.hindcast_lstm.weight_ih_l0'],
        ft_c_ckpt['_orig_mod.hindcast_lstm.weight_ih_l0'],
    )
    assert torch.equal(
        base_ckpt['_orig_mod.forecast_lstm.weight_ih_l0'],
        ft_c_ckpt['_orig_mod.forecast_lstm.weight_ih_l0'],
    )
    assert torch.equal(
        base_ckpt['_orig_mod.hindcast_embeddings_fc.era5_land.net.0.weight'],
        ft_c_ckpt['_orig_mod.hindcast_embeddings_fc.era5_land.net.0.weight'],
    )
    assert torch.equal(
        base_ckpt['_orig_mod.shared_embeddings_fc.hres.net.0.weight'],
        ft_c_ckpt['_orig_mod.shared_embeddings_fc.hres.net.0.weight'],
    )
    assert not torch.equal(
        base_ckpt['_orig_mod.head.net.0.weight'],
        ft_c_ckpt['_orig_mod.head.net.0.weight'],
    )
    assert not torch.equal(
        base_ckpt['_orig_mod.static_embedding_fc.net.0.weight'],
        ft_c_ckpt['_orig_mod.static_embedding_fc.net.0.weight'],
    )


@pytest.mark.unit
@pytest.mark.parametrize('lazy_load', [False, True])
def test_multimet_masked_mean_allows_partial_local_product_record(
    tmp_path: Path,
    *,
    lazy_load: bool,
) -> None:
    """Retain samples under masked_mean when a local product has early NaNs."""
    paths = _create_synthetic_datasets(tmp_path)
    # Set first 30 days of local DAYMET to NaN while global ERA5_LAND is valid.
    daymet_zarr = paths['local_dyn_dir'] / 'DAYMET' / 'timeseries.zarr'
    daymet_ds = xr.open_zarr(daymet_zarr).load()
    daymet_ds['daymet_prcp'].loc[
        {'date': slice('2000-01-01', '2000-01-30')}
    ] = np.nan
    daymet_ds.to_zarr(daymet_zarr, mode='w')

    cfg_dict = _make_base_pretrain_cfg(tmp_path, paths)
    cfg_dict['lazy_load'] = lazy_load
    cfg_dict['run_dir'] = str(tmp_path / f'run_partial_{lazy_load}')
    Path(cfg_dict['run_dir']).mkdir(parents=True, exist_ok=True)
    cfg_dict['dynamics_data_dir'] = [
        str(paths['global_dyn_dir']),
        str(paths['local_dyn_dir']),
    ]
    cfg_dict['hindcast_inputs'] = {
        'era5_land': [
            'era5land_total_precipitation',
            'era5land_temperature_2m',
        ],
        'daymet': ['daymet_prcp', 'daymet_tmax', 'daymet_tmin'],
    }
    cfg = Config(cfg_dict)

    ds_masked = Multimet(cfg=cfg, is_train=True, period='train')
    cfg_global_dict = dict(cfg_dict)
    cfg_global_dict['run_dir'] = str(tmp_path / f'run_global_only_{lazy_load}')
    Path(cfg_global_dict['run_dir']).mkdir(parents=True, exist_ok=True)
    cfg_global_dict['hindcast_inputs'] = {
        'era5_land': [
            'era5land_total_precipitation',
            'era5land_temperature_2m',
        ],
    }
    ds_global = Multimet(
        cfg=Config(cfg_global_dict), is_train=True, period='train'
    )
    assert len(ds_masked) == len(ds_global)

    # Under nan_handling_method=None, NaN days drop touching samples
    cfg_strict_dict = dict(cfg_dict)
    cfg_strict_dict['run_dir'] = str(tmp_path / f'run_strict_{lazy_load}')
    Path(cfg_strict_dict['run_dir']).mkdir(parents=True, exist_ok=True)
    cfg_strict_dict['nan_handling_method'] = None
    ds_strict = Multimet(
        cfg=Config(cfg_strict_dict), is_train=True, period='train'
    )
    assert len(ds_strict) < len(ds_masked)


@pytest.mark.unit
def test_multimet_finetune_scaler_extension_rejects_all_nan_new_variable(
    tmp_path: Path,
) -> None:
    """Raise ValueError when a new fine-tuning variable is all-NaN."""
    paths = _create_synthetic_datasets(tmp_path)
    base_cfg_dict = _make_base_pretrain_cfg(tmp_path, paths)
    base_run_dir = tmp_path / 'base_run'
    base_cfg_dict['run_dir'] = str(base_run_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    # Build base dataset to persist base scaler.zarr
    Multimet(cfg=Config(base_cfg_dict), is_train=True, period='train')

    # Verify lazy_load=True works on the valid scaler-extension path
    ft_lazy_cfg = dict(base_cfg_dict)
    ft_lazy_cfg['is_finetuning'] = True
    ft_lazy_cfg['lazy_load'] = True
    ft_lazy_cfg['base_run_dir'] = str(base_run_dir)
    ft_lazy_cfg['run_dir'] = str(tmp_path / 'ft_run_lazy')
    Path(ft_lazy_cfg['run_dir']).mkdir(parents=True, exist_ok=True)
    ft_lazy_cfg['dynamics_data_dir'] = [
        str(paths['global_dyn_dir']),
        str(paths['local_dyn_dir']),
    ]
    ft_lazy_cfg['hindcast_inputs'] = {
        'era5_land': [
            'era5land_total_precipitation',
            'era5land_temperature_2m',
        ],
        'cpc': ['cpc_precipitation'],
        'hres': ['hres_total_precipitation', 'hres_temperature_2m'],
        'daymet': ['daymet_prcp', 'daymet_tmax', 'daymet_tmin'],
    }
    ds_lazy = Multimet(
        cfg=Config(ft_lazy_cfg),
        is_train=True,
        period='train',
        compute_scaler=False,
    )
    assert 'daymet_prcp' in ds_lazy.scaler.scaler.data_vars

    # Make local daymet_prcp all-NaN across the training split
    daymet_zarr = paths['local_dyn_dir'] / 'DAYMET' / 'timeseries.zarr'
    daymet_ds = xr.open_zarr(daymet_zarr).load()
    daymet_ds['daymet_prcp'][:] = np.nan
    daymet_ds.to_zarr(daymet_zarr, mode='w')

    ft_cfg_dict = dict(base_cfg_dict)
    ft_cfg_dict['is_finetuning'] = True
    ft_cfg_dict['base_run_dir'] = str(base_run_dir)
    ft_cfg_dict['run_dir'] = str(tmp_path / 'ft_run_nan')
    Path(ft_cfg_dict['run_dir']).mkdir(parents=True, exist_ok=True)
    ft_cfg_dict['dynamics_data_dir'] = [
        str(paths['global_dyn_dir']),
        str(paths['local_dyn_dir']),
    ]
    ft_cfg_dict['hindcast_inputs'] = {
        'era5_land': [
            'era5land_total_precipitation',
            'era5land_temperature_2m',
        ],
        'cpc': ['cpc_precipitation'],
        'hres': ['hres_total_precipitation', 'hres_temperature_2m'],
        'daymet': ['daymet_prcp', 'daymet_tmax', 'daymet_tmin'],
    }
    with pytest.raises(
        ValueError,
        match=r'Non-finite or zero normalization statistics.*daymet_prcp',
    ):
        Multimet(
            cfg=Config(ft_cfg_dict),
            is_train=True,
            period='train',
            compute_scaler=False,
        )

    # Verify evaluation fails if run_dir/scaler.zarr lacks new vars
    with pytest.raises(
        ValueError,
        match=(
            r'Fine-tuning evaluation dataset contains features'
            r'.*missing from scaler'
        ),
    ):
        Multimet(
            cfg=Config(ft_cfg_dict),
            is_train=False,
            period='test',
            compute_scaler=False,
        )


@pytest.mark.unit
def test_load_model_weights_validation_and_prefix_migration(
    tmp_path: Path,
) -> None:
    """Reject corrupted core weights, band mismatches, and prefix migration."""
    paths = _create_synthetic_datasets(tmp_path)
    cfg_dict = _make_base_pretrain_cfg(tmp_path, paths)
    run_dir = tmp_path / 'model_run'
    cfg_dict['run_dir'] = str(run_dir)
    cfg_dict['compile'] = False
    _write_minimal_scaler(run_dir)
    cfg = Config(cfg_dict)
    model = get_model(cfg)
    base_ckpt = tmp_path / 'base_model.pt'
    torch.save(model.state_dict(), base_ckpt)

    # 1. Missing non-embedding core weight must raise RuntimeError
    corrupted_state = dict(model.state_dict())
    del corrupted_state['hindcast_lstm.weight_ih_l0']
    ckpt_missing = tmp_path / 'corrupted_missing.pt'
    torch.save(corrupted_state, ckpt_missing)
    with pytest.raises(RuntimeError, match='Missing non-embedding keys'):
        load_model_weights(
            model, ckpt_missing, device='cpu', allow_new_embeddings=True
        )

    # 2. Unexpected weight key must raise RuntimeError
    unexpected_state = dict(model.state_dict())
    unexpected_state['unexpected_module.weight'] = torch.zeros(4, 4)
    ckpt_unexpected = tmp_path / 'corrupted_unexpected.pt'
    torch.save(unexpected_state, ckpt_unexpected)
    with pytest.raises(RuntimeError, match='Unexpected keys'):
        load_model_weights(
            model, ckpt_unexpected, device='cpu', allow_new_embeddings=True
        )

    # 3. Prefix migration ('hres' from shared to hindcast_embeddings_fc)
    migrated_cfg_dict = dict(cfg_dict)
    migrated_cfg_dict['forecast_inputs'] = {
        'gefs_reforecast': [
            'gefs_reforecast_apcp_sfc',
            'gefs_reforecast_tmp_2m',
        ],
    }
    migrated_model = get_model(Config(migrated_cfg_dict))
    with pytest.raises(
        ValueError,
        match=(
            "Product 'hres' moved from 'shared_embeddings_fc' "
            "in checkpoint to 'hindcast_embeddings_fc'"
        ),
    ):
        load_model_weights(
            migrated_model,
            base_ckpt,
            device='cpu',
            allow_new_embeddings=True,
        )

    # 4. Changing an existing product's band count raises RuntimeError
    band_mismatch_cfg_dict = dict(cfg_dict)
    band_mismatch_cfg_dict['hindcast_inputs'] = {
        'era5_land': ['era5land_total_precipitation'],  # 1 band instead of 2
        'cpc': ['cpc_precipitation'],
        'hres': ['hres_total_precipitation', 'hres_temperature_2m'],
    }
    band_mismatch_model = get_model(Config(band_mismatch_cfg_dict))
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_model_weights(
            band_mismatch_model,
            base_ckpt,
            device='cpu',
            allow_new_embeddings=True,
        )


@pytest.mark.unit
def test_freeze_model_parts_validation_and_frozen_new_embedding_guard(
    tmp_path: Path,
) -> None:
    """Validate finetune_modules dict values and frozen new embeddings."""
    paths = _create_synthetic_datasets(tmp_path)
    base_cfg_dict = _make_base_pretrain_cfg(tmp_path, paths)
    base_run_dir = tmp_path / 'base_run'
    base_cfg_dict['run_dir'] = str(base_run_dir)
    base_cfg_dict['compile'] = False
    _write_minimal_scaler(base_run_dir)
    base_model = get_model(Config(base_cfg_dict))
    base_ckpt = base_run_dir / 'model_epoch001.pt'
    torch.save(base_model.state_dict(), base_ckpt)

    run_idx = 0

    def _make_trainer(
        finetune_modules: object,
        *,
        add_daymet: bool = False,
    ) -> BaseTrainer:
        nonlocal run_idx
        run_idx += 1
        cfg_dict = dict(base_cfg_dict)
        cfg_dict['run_dir'] = str(tmp_path / f'trainer_runs_{run_idx}')
        cfg_dict['is_finetuning'] = True
        cfg_dict['base_run_dir'] = str(base_run_dir)
        cfg_dict['finetune_modules'] = finetune_modules
        if add_daymet:
            cfg_dict['hindcast_inputs'] = {
                **base_cfg_dict['hindcast_inputs'],  # type: ignore[dict-item]
                'daymet': ['daymet_prcp', 'daymet_tmax', 'daymet_tmin'],
            }
        cfg = Config(cfg_dict)
        trainer = BaseTrainer(cfg)
        trainer.model = trainer._get_model().to(trainer.device)  # noqa: SLF001
        trainer._load_model_weights(base_ckpt, allow_new_embeddings=True)  # noqa: SLF001
        return trainer

    # 1. Unknown submodule key in ModuleDict raises KeyError
    trainer_unknown = _make_trainer(
        {'hindcast_embeddings_fc': ['nonexistent_local_product']}
    )
    with pytest.raises(
        KeyError,
        match=(
            "Submodule 'nonexistent_local_product' not found in "
            "'hindcast_embeddings_fc'"
        ),
    ):
        trainer_unknown._freeze_model_parts()  # noqa: SLF001

    # 2. Invalid non-bool values on non-ModuleDict module raise ValueError (B4)
    for bad_spec in (
        {'head': ['nonexistent_part']},
        {'hindcast_lstm': 0},
        {'head': 'sub'},
    ):
        trainer_bad_non_dict = _make_trainer(bad_spec)
        with pytest.raises(
            ValueError,
            match=r'Invalid value .* for non-ModuleDict module',
        ):
            trainer_bad_non_dict._freeze_model_parts()  # noqa: SLF001

    # 3. Invalid value type on ModuleDict module raises ValueError (B4)
    for bad_dict_spec in (
        {'hindcast_embeddings_fc': 123},
        {'hindcast_embeddings_fc': None},
        {'hindcast_embeddings_fc': [123]},
    ):
        trainer_bad_dict = _make_trainer(bad_dict_spec)
        with pytest.raises(
            ValueError,
            match=r"Invalid value .* for ModuleDict 'hindcast_embeddings_fc'",
        ):
            trainer_bad_dict._freeze_model_parts()  # noqa: SLF001

    # 4. New product omitted from finetune_modules raises ValueError (B2)
    trainer_frozen_new = _make_trainer({'head': True}, add_daymet=True)
    with pytest.raises(
        ValueError,
        match=(
            r'Newly added product embedding '
            r"'hindcast_embeddings_fc\.daymet' is randomly initialized"
        ),
    ):
        trainer_frozen_new._freeze_model_parts()  # noqa: SLF001


@pytest.mark.unit
def test_find_product_zarr_path_multi_directory_and_cloud_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Resolve local dirs before cloud, warn on dupes, and fail if missing."""
    paths = _create_synthetic_datasets(tmp_path)
    global_dir = paths['global_dyn_dir']
    local_dir = paths['local_dyn_dir']

    # Local product is found even when a cloud path precedes local_dir
    resolved_daymet = _find_product_zarr_path(
        ['gs://caravan-multimet/v1.1', local_dir], 'DAYMET'
    )
    assert resolved_daymet == local_dir / 'DAYMET' / 'timeseries.zarr'

    # Product absent locally falls back to cloud path when configured
    resolved_cloud = _find_product_zarr_path(
        ['gs://caravan-multimet/v1.1', local_dir], 'ERA5_LAND'
    )
    assert resolved_cloud == Path(
        'gs://caravan-multimet/v1.1/ERA5_LAND/timeseries.zarr'
    )

    # Missing local dir in dynamics_data_dir raises FileNotFoundError (H1)
    missing_local_dir = tmp_path / 'nonexistent_local_dir'
    with pytest.raises(
        FileNotFoundError,
        match=f'Dynamics data path not found: {missing_local_dir}',
    ):
        _find_product_zarr_path(
            ['gs://caravan-multimet/v1.1', missing_local_dir], 'CPC'
        )

    # Duplicate product across local dirs logs WARNING and selects first
    second_local_dir = tmp_path / 'second_local_dir'
    (second_local_dir / 'DAYMET' / 'timeseries.zarr').mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        dup_resolved = _find_product_zarr_path(
            [local_dir, second_local_dir], 'DAYMET'
        )
    assert dup_resolved == local_dir / 'DAYMET' / 'timeseries.zarr'
    assert "Product 'DAYMET' found in multiple local directories" in caplog.text

    # Product absent from all local directories with no cloud path raises
    with pytest.raises(
        FileNotFoundError,
        match="Zarr store for product 'NONEXISTENT' not found in",
    ):
        _find_product_zarr_path([global_dir, local_dir], 'NONEXISTENT')
