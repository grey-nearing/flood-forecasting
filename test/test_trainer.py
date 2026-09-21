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

"""Unit tests for googlehydrology.training helpers and BaseTrainer."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from googlehydrology.training import (
    get_loss_obj,
    get_optimizer,
    get_regularization_obj,
)
from googlehydrology.training.basetrainer import BaseTrainer
from googlehydrology.training.basin_scheduler import BasinWindowScheduler
from googlehydrology.training.loss import (
    MaskedCMALLoss,
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
)


@pytest.mark.unit
def test_get_optimizer_all_types():
    model = nn.Linear(5, 2)
    optimizers = [
        'adam', 'adamw', 'sgd', 'asgd',
        'rmsprop', 'adagrad', 'adadelta', 'adamax',
    ]
    for opt_name in optimizers:
        cfg = MagicMock()
        cfg.optimizer = opt_name
        cfg.initial_learning_rate = 0.001
        opt = get_optimizer(model, cfg)
        assert isinstance(opt, torch.optim.Optimizer)

    # Unsupported optimizer
    cfg_invalid = MagicMock(
        optimizer='invalid_optimizer',
        initial_learning_rate=0.001,
    )
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_optimizer(model, cfg_invalid)


@pytest.mark.unit
def test_get_loss_obj():
    loss_types = {
        'mse': MaskedMSELoss,
        'rmse': MaskedRMSELoss,
        'nse': MaskedNSELoss,
        'cmalloss': MaskedCMALLoss,
        'cmal': MaskedCMALLoss,
    }
    for loss_name, expected_class in loss_types.items():
        cfg = MagicMock()
        cfg.loss = loss_name
        cfg.predict_last_n = 1
        cfg.no_loss_frequencies = []
        cfg.target_variables = ['flow']
        cfg.target_loss_weights = None
        cfg.n_distributions = 3

        loss_obj = get_loss_obj(cfg)
        assert isinstance(loss_obj, expected_class)

    # Unsupported loss
    cfg_invalid = MagicMock(loss='invalid_loss')
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_loss_obj(cfg_invalid)


@pytest.mark.unit
def test_get_regularization_obj():
    cfg = MagicMock()
    cfg.regularization = ['forecast_overlap', ('forecast_overlap', 0.5)]

    reg_objs = get_regularization_obj(cfg)
    assert len(reg_objs) == 2
    assert reg_objs[0].name == 'forecast_overlap'
    assert reg_objs[0].weight == 1.0
    assert reg_objs[1].name == 'forecast_overlap'
    assert reg_objs[1].weight == 0.5

    # Unsupported regularization
    cfg_invalid = MagicMock(regularization=['invalid_reg'])
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_regularization_obj(cfg_invalid)


# --- limit_n_basins epoch rotation ---


def _make_rotating_trainer(n_basins: int, window: int, seed: int = 0):
    """A BaseTrainer with only the basin-rotation machinery wired up.

    Constructing a real trainer would pull in a model, an optimizer and a
    dataset, none of which the rotation logic touches.
    """
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.basins = [f'basin_{i:03d}' for i in range(n_basins)]
    trainer._basin_scheduler = BasinWindowScheduler(
        trainer.basins, window=window, seed=seed
    )
    trainer._loaded_basin_epoch = None
    trainer.ds = MagicMock()
    trainer.loader = None
    # Instance attribute shadows the method; avoids building a real loader.
    trainer._get_data_loader = MagicMock(side_effect=lambda ds: MagicMock())
    return trainer


@pytest.mark.unit
def test_load_basins_for_epoch_is_noop_when_disabled():
    """With limit_n_basins unset, the trainer must not touch the dataset.

    This is the default path for every existing run.
    """
    trainer = _make_rotating_trainer(n_basins=10, window=0)

    for epoch in range(1, 4):
        trainer._load_basins_for_epoch(epoch)

    trainer.ds.load_basins.assert_not_called()
    trainer._get_data_loader.assert_not_called()


@pytest.mark.unit
def test_load_basins_for_epoch_covers_every_basin_once_per_sweep():
    """Exact coverage is the reason for permuting rather than resampling.

    Drawing a fresh random window each epoch samples with replacement and
    leaves a large fraction of basins untrained; walking disjoint windows
    over one permutation cannot.
    """
    n_basins, window = 10, 3
    trainer = _make_rotating_trainer(n_basins=n_basins, window=window)
    sweep = trainer._basin_scheduler.windows_per_sweep
    assert sweep == 4  # ceil(10 / 3): the last window is short, not wrapped.

    for epoch in range(1, sweep + 1):
        trainer._load_basins_for_epoch(epoch)

    loaded = [call.args[0] for call in trainer.ds.load_basins.call_args_list]
    flat = [basin for window_basins in loaded for basin in window_basins]

    assert all(len(w) <= window for w in loaded)
    assert len(flat) == len(set(flat)), 'a basin was trained on twice'
    assert set(flat) == set(trainer.basins), 'a basin was never trained on'


@pytest.mark.unit
def test_load_basins_for_epoch_rotates_between_epochs():
    """Consecutive epochs must see different basins."""
    trainer = _make_rotating_trainer(n_basins=10, window=3)

    trainer._load_basins_for_epoch(1)
    trainer._load_basins_for_epoch(2)

    first, second = (
        call.args[0] for call in trainer.ds.load_basins.call_args_list
    )
    assert set(first).isdisjoint(second)


@pytest.mark.unit
def test_load_basins_for_epoch_is_idempotent():
    """initialize_training loads epoch N, then the loop asks for it again.

    Reloading would double the work and the peak memory for no benefit.
    """
    trainer = _make_rotating_trainer(n_basins=10, window=3)

    trainer._load_basins_for_epoch(1)
    trainer._load_basins_for_epoch(1)

    assert trainer.ds.load_basins.call_count == 1
    assert trainer._get_data_loader.call_count == 1


@pytest.mark.unit
def test_load_basins_for_epoch_rebuilds_loader():
    """The loader samples over len(ds), which changes with the window."""
    trainer = _make_rotating_trainer(n_basins=10, window=3)

    trainer._load_basins_for_epoch(1)
    first_loader = trainer.loader
    trainer._load_basins_for_epoch(2)

    assert first_loader is not None
    assert trainer.loader is not first_loader


@pytest.mark.unit
def test_basin_schedule_is_reproducible_across_restarts():
    """A resumed run must reconstruct the schedule from the epoch number.

    Otherwise a restart re-randomizes the permutation and the exact-coverage
    guarantee is lost across the restart boundary.
    """
    epochs = range(1, 9)

    first = _make_rotating_trainer(n_basins=10, window=3, seed=42)
    second = _make_rotating_trainer(n_basins=10, window=3, seed=42)

    assert [first._basin_scheduler.basins_for_epoch(e) for e in epochs] == [
        second._basin_scheduler.basins_for_epoch(e) for e in epochs
    ]
