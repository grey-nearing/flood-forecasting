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

"""Unit tests for model.training helpers and BaseTrainer."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from model.training import (
    get_loss_obj,
    get_optimizer,
    get_regularization_obj,
)
from model.training.basetrainer import BaseTrainer
from model.training.basin_scheduler import BasinWindowScheduler
from model.training.loss import (
    MaskedCMALLoss,
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
)

pytestmark = pytest.mark.unit


def test_get_optimizer_all_types(make_minimal_config):
    model = nn.Linear(5, 2)
    optimizers = [
        'adam',
        'adamw',
        'sgd',
        'asgd',
        'rmsprop',
        'adagrad',
        'adadelta',
        'adamax',
    ]
    for opt_name in optimizers:
        cfg = make_minimal_config(
            {'optimizer': opt_name, 'initial_learning_rate': 0.001}
        )
        opt = get_optimizer(model, cfg)
        assert isinstance(opt, torch.optim.Optimizer)

    # Unsupported optimizer
    cfg_invalid = make_minimal_config(
        {'optimizer': 'invalid_optimizer', 'initial_learning_rate': 0.001}
    )
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_optimizer(model, cfg_invalid)


def test_get_loss_obj(make_minimal_config):
    loss_types = {
        'mse': MaskedMSELoss,
        'rmse': MaskedRMSELoss,
        'nse': MaskedNSELoss,
        'cmalloss': MaskedCMALLoss,
        'cmal': MaskedCMALLoss,
    }
    for loss_name, expected_class in loss_types.items():
        cfg = make_minimal_config(
            {
                'loss': loss_name,
                'predict_last_n': 1,
                'target_variables': ['streamflow'],
                'target_loss_weights': None,
                'n_distributions': 3,
            }
        )
        loss_obj = get_loss_obj(cfg)
        assert isinstance(loss_obj, expected_class)

    # Unsupported loss
    cfg_invalid = make_minimal_config({'loss': 'invalid_loss'})
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_loss_obj(cfg_invalid)


def test_get_regularization_obj(make_minimal_config):
    cfg = make_minimal_config(
        {'regularization': ['forecast_overlap', ('forecast_overlap', 0.5)]}
    )
    reg_objs = get_regularization_obj(cfg)
    assert len(reg_objs) == 2
    assert reg_objs[0].name == 'forecast_overlap'
    assert reg_objs[0].weight == 1.0
    assert reg_objs[1].name == 'forecast_overlap'
    assert reg_objs[1].weight == 0.5

    # Unsupported regularization
    cfg_invalid = make_minimal_config({'regularization': ['invalid_reg']})
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_regularization_obj(cfg_invalid)


# --- max_basins_in_memory epoch rotation ---


def _make_rotating_trainer(n_basins: int, window: int, seed: int = 0):
    """Build a BaseTrainer stub with basin-rotation attributes wired up."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.basins = [f'basin_{i:03d}' for i in range(n_basins)]
    trainer._basin_scheduler = BasinWindowScheduler(
        trainer.basins, window=window, seed=seed
    )
    trainer._loaded_basin_epoch = None
    trainer.ds = MagicMock()
    trainer.loader = None
    trainer._get_data_loader = MagicMock(side_effect=lambda ds: MagicMock())
    return trainer


@pytest.mark.unit
def test_load_basins_for_epoch_is_noop_when_disabled():
    """With max_basins_in_memory=0, the trainer does not reload basins per epoch."""
    trainer = _make_rotating_trainer(n_basins=10, window=0)

    for epoch in range(1, 4):
        trainer._load_basins_for_epoch(epoch)

    trainer.ds.load_basins.assert_not_called()
    trainer._get_data_loader.assert_not_called()


@pytest.mark.unit
def test_load_basins_for_epoch_covers_every_basin_once_per_sweep():
    """Epochs 1..windows_per_sweep cover every basin once with the tail window last."""
    n_basins, window = 10, 3
    trainer = _make_rotating_trainer(n_basins=n_basins, window=window)
    sweep = trainer._basin_scheduler.windows_per_sweep
    assert sweep == 4  # ceil(10 / 3)

    for epoch in range(1, sweep + 1):
        trainer._load_basins_for_epoch(epoch)

    loaded = [call.args[0] for call in trainer.ds.load_basins.call_args_list]
    flat = [basin for window_basins in loaded for basin in window_basins]

    assert [len(w) for w in loaded] == [3, 3, 3, 1]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(trainer.basins)


@pytest.mark.unit
def test_load_basins_for_epoch_rotates_between_epochs():
    """Consecutive epochs load disjoint basin windows."""
    trainer = _make_rotating_trainer(n_basins=10, window=3)

    trainer._load_basins_for_epoch(1)
    trainer._load_basins_for_epoch(2)

    first, second = (
        call.args[0] for call in trainer.ds.load_basins.call_args_list
    )
    assert set(first).isdisjoint(second)


@pytest.mark.unit
def test_load_basins_for_epoch_is_idempotent():
    """Calling _load_basins_for_epoch twice for the same epoch only loads once."""
    trainer = _make_rotating_trainer(n_basins=10, window=3)

    trainer._load_basins_for_epoch(1)
    trainer._load_basins_for_epoch(1)

    assert trainer.ds.load_basins.call_count == 1
    assert trainer._get_data_loader.call_count == 1


@pytest.mark.unit
def test_load_basins_for_epoch_rebuilds_loader():
    """Each new epoch window rebuilds the DataLoader."""
    trainer = _make_rotating_trainer(n_basins=10, window=3)

    trainer._load_basins_for_epoch(1)
    first_loader = trainer.loader
    trainer._load_basins_for_epoch(2)

    assert first_loader is not None
    assert trainer.loader is not first_loader


@pytest.mark.unit
def test_basin_schedule_is_reproducible_across_restarts():
    """Two trainers with the same seed produce the same per-epoch basin schedule."""
    epochs = range(1, 9)

    first = _make_rotating_trainer(n_basins=10, window=3, seed=42)
    second = _make_rotating_trainer(n_basins=10, window=3, seed=42)

    assert [first._basin_scheduler.basins_for_epoch(e) for e in epochs] == [
        second._basin_scheduler.basins_for_epoch(e) for e in epochs
    ]


@pytest.mark.unit
def test_real_trainer_rotates_windows_and_unloads_validator(tmp_path):
    """End-to-end BaseTrainer with real Zarr stores rotates epoch windows and unloads validation basins."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from model.utils.config import Config

    basins = [f'basin_{i:02d}' for i in range(5)]
    dates = pd.date_range('1999-12-25', '2000-01-10', freq='D')
    lead_times = [np.timedelta64(1, 'D'), np.timedelta64(2, 'D')]
    n_basins, n_dates, n_leads = len(basins), len(dates), len(lead_times)

    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            'area': (
                ('basin',),
                np.linspace(10.0, 50.0, n_basins, dtype=np.float32),
            ),
            'era5land_precip': (
                ('basin', 'date'),
                rng.uniform(0.1, 5.0, (n_basins, n_dates)).astype(np.float32),
            ),
            'hres_precip': (
                ('basin', 'date', 'lead_time'),
                rng.uniform(0.1, 5.0, (n_basins, n_dates, n_leads)).astype(
                    np.float32
                ),
            ),
            'streamflow': (
                ('basin', 'date'),
                rng.uniform(1.0, 10.0, (n_basins, n_dates)).astype(np.float32),
            ),
        },
        coords={'basin': basins, 'date': dates, 'lead_time': lead_times},
    )

    statics_dir = tmp_path / 'statics'
    targets_dir = tmp_path / 'targets'
    dynamics_dir = tmp_path / 'dynamics'
    ds[['area']].to_zarr(statics_dir / 'attributes.zarr', mode='w')
    ds[['streamflow']].to_zarr(targets_dir / 'streamflow.zarr', mode='w')
    ds[['era5land_precip']].drop_vars('lead_time', errors='ignore').to_zarr(
        dynamics_dir / 'ERA5_LAND' / 'timeseries.zarr', mode='w'
    )
    ds[['hres_precip']].to_zarr(
        dynamics_dir / 'HRES' / 'timeseries.zarr', mode='w'
    )

    basin_file = tmp_path / 'basins.txt'
    basin_file.write_text('\n'.join(basins) + '\n')

    cfg = Config({
        'experiment_name': 'trainer_window_rotation',
        'run_dir': str(tmp_path / 'runs'),
        'dataset': 'multimet',
        'train_basin_file': str(basin_file),
        'validation_basin_file': str(basin_file),
        'test_basin_file': str(basin_file),
        'statics_data_dir': str(statics_dir),
        'targets_data_dir': str(targets_dir),
        'dynamics_data_dir': str(dynamics_dir),
        'train_start_date': '01/01/2000',
        'train_end_date': '05/01/2000',
        'validation_start_date': '01/01/2000',
        'validation_end_date': '05/01/2000',
        'test_start_date': '01/01/2000',
        'test_end_date': '05/01/2000',
        'hindcast_inputs': {
            'era5_land': ['era5land_precip'],
            'hres': ['hres_precip'],
        },
        'forecast_inputs': {'hres': ['hres_precip']},
        'static_attributes': ['area'],
        'target_variables': ['streamflow'],
        'model': 'mean_embedding_forecast_lstm',
        'hidden_size': 8,
        'head': 'regression',
        'output_activation': 'linear',
        'statics_embedding': {
            'type': 'fc',
            'hiddens': [8],
            'activation': 'tanh',
            'dropout': 0.0,
        },
        'hindcast_embedding': {
            'type': 'fc',
            'hiddens': [8],
            'activation': 'tanh',
            'dropout': 0.0,
        },
        'forecast_embedding': {
            'type': 'fc',
            'hiddens': [8],
            'activation': 'tanh',
            'dropout': 0.0,
        },
        'seq_length': 4,
        'lead_time': 2,
        'forecast_overlap': 4,
        'predict_last_n': 2,
        'timestep_counter': True,
        'output_dropout': 0.0,
        'compile': False,
        'device': 'cpu',
        'seed': 7,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'epochs': 3,
        'save_weights_every': 1,
        'batch_size': 8,
        'initial_learning_rate': 0.001,
        'metrics': ['NSE'],
        'num_workers': 0,
        'validate_every': 1,
        'validate_n_random_basins': 5,
        'max_basins_in_memory': 2,
        'cache': {'enabled': False},
    })

    trainer = BaseTrainer(cfg=cfg)
    trainer.initialize_training()

    assert trainer.ds.is_loaded
    assert len(trainer.ds.loaded_basins) == 2
    assert not trainer.validator.dataset.is_loaded

    epoch_windows = []
    orig_train_epoch = trainer._train_epoch

    def spy_train_epoch(epoch: int):
        epoch_windows.append(list(trainer.ds.loaded_basins))
        orig_train_epoch(epoch)

    trainer._train_epoch = spy_train_epoch
    trainer.train_and_validate()

    assert [len(w) for w in epoch_windows] == [2, 2, 1]
    seen_basins = [b for w in epoch_windows for b in w]
    assert len(seen_basins) == 5
    assert set(seen_basins) == set(basins)
    assert not trainer.validator.dataset.is_loaded

