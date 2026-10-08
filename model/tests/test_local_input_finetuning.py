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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

from model.datasetzoo.multimet import Multimet, _find_product_zarr_path
from model.modelzoo import get_model, load_model_weights
from model.run import eval_run, finetune, start_run
from model.training.basetrainer import BaseTrainer
from model.utils.config import Config


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
    base_scaler = xr.open_zarr(base_run_dir / 'scaler.zarr').load()

    # Fine-tune Mode A: only the newly added local embeddings (daymet + gefs)
    ft_cfg_a = {
        'experiment_name': 'ft_local_only',
        'base_run_dir': str(base_run_dir),
        'run_dir': str(tmp_path / 'runs'),
        'epochs': 1,
        'initial_learning_rate': 0.005,
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
    ft_a_ckpt = torch.load(ft_a_dir / 'model_epoch001.pt', weights_only=True)

    # Verify scaler extended with local features while keeping base stats exact
    ft_scaler = xr.open_zarr(ft_a_dir / 'scaler.zarr').load()
    for base_var in base_scaler.data_vars:
        xr.testing.assert_equal(ft_scaler[base_var], base_scaler[base_var])
    for new_var in (
        'daymet_prcp',
        'daymet_tmax',
        'daymet_tmin',
        'gefs_reforecast_apcp_sfc',
        'gefs_reforecast_tmp_2m',
    ):
        assert new_var in ft_scaler.data_vars
        scale_val = float(ft_scaler[new_var].sel(parameter='scale').values)
        assert np.isfinite(scale_val)
        assert scale_val > 0.0

    # Verify frozen vs unfrozen parameters in Mode A
    for key, base_val in base_ckpt.items():
        assert torch.equal(base_val, ft_a_ckpt[key]), (
            f'Pretrained parameter {key} should be frozen in Mode A'
        )
    daymet_key = '_orig_mod.hindcast_embeddings_fc.daymet.net.0.weight'
    gefs_key = '_orig_mod.forecast_embeddings_fc.gefs_reforecast.net.0.weight'
    assert daymet_key in ft_a_ckpt
    assert torch.isfinite(ft_a_ckpt[daymet_key]).all()
    assert gefs_key in ft_a_ckpt
    assert torch.isfinite(ft_a_ckpt[gefs_key]).all()

    # Verify evaluation works on the fine-tuned run and writes finite metrics
    eval_run(
        Config(ft_a_dir / 'config.yml'),
        run_dir=ft_a_dir,
        period='test',
        epoch=1,
        gpu=-1,
    )
    metrics_file = ft_a_dir / 'test' / 'model_epoch001' / 'test_metrics.csv'
    assert metrics_file.exists()
    metrics_df = pd.read_csv(metrics_file)
    assert np.isfinite(metrics_df['NSE']).all()
    results_ds = xr.open_zarr(
        ft_a_dir / 'test' / 'model_epoch001' / 'test_results.zarr',
        consolidated=False,
    ).load()
    assert np.isfinite(results_ds['streamflow_sim'].values).any()

    # Fine-tune Mode C: local embeddings + static_embedding_fc + head
    ft_cfg_c = dict(ft_cfg_a)
    ft_cfg_c['experiment_name'] = 'ft_local_static_head'
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
def test_multimet_masked_mean_allows_partial_local_product_record(
    tmp_path: Path,
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
    cfg_dict['run_dir'] = str(tmp_path / 'run_partial')
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
    # Compare against global-only dataset sample count: masked_mean must retain
    # all dates because the era5_land group is completely valid on every date.
    cfg_global_dict = dict(cfg_dict)
    cfg_global_dict['run_dir'] = str(tmp_path / 'run_global_only')
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


@pytest.mark.unit
def test_load_model_weights_allow_new_embeddings_rejects_corrupted_core_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject missing core or unexpected keys under allow_new_embeddings."""
    monkeypatch.setattr(
        'model.modelzoo.basemodel.Scaler', lambda **_kwargs: None
    )
    paths = _create_synthetic_datasets(tmp_path)
    cfg_dict = _make_base_pretrain_cfg(tmp_path, paths)
    cfg_dict['compile'] = False
    cfg = Config(cfg_dict)
    model = get_model(cfg)

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


@pytest.mark.unit
def test_freeze_model_parts_unknown_submodule_key_raises_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise KeyError when finetune_modules has an unknown product key."""
    monkeypatch.setattr(
        'model.modelzoo.basemodel.Scaler', lambda **_kwargs: None
    )
    paths = _create_synthetic_datasets(tmp_path)
    cfg_dict = _make_base_pretrain_cfg(tmp_path, paths)
    cfg_dict['compile'] = False
    cfg_dict['finetune_modules'] = {
        'hindcast_embeddings_fc': ['nonexistent_local_product'],
    }
    cfg = Config(cfg_dict)
    model = get_model(cfg)

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.cfg = cfg
    trainer.model = model
    expected_msg = (
        "Submodule 'nonexistent_local_product' not found in "
        "'hindcast_embeddings_fc'"
    )
    with pytest.raises(KeyError, match=expected_msg):
        trainer._freeze_model_parts()  # noqa: SLF001


@pytest.mark.unit
def test_find_product_zarr_path_multi_directory_and_cloud_fallback(
    tmp_path: Path,
) -> None:
    """Resolve local directories before cloud paths and fail if absent."""
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
    assert str(resolved_cloud).endswith('ERA5_LAND/timeseries.zarr')

    # Product absent from all local directories with no cloud path raises
    with pytest.raises(
        FileNotFoundError,
        match="Zarr store for product 'NONEXISTENT' not found in",
    ):
        _find_product_zarr_path([global_dir, local_dir], 'NONEXISTENT')
