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

from pathlib import Path
from types import SimpleNamespace
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


class _TinyModel(nn.Module):
    """Model with a plain part, a per-product ModuleDict, and a head."""

    module_parts = ('encoder', 'embeddings', 'head')

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.embeddings = nn.ModuleDict(
            {'era5': nn.Linear(2, 2), 'cpc': nn.Linear(2, 2)}
        )
        self.head = nn.Linear(2, 1)


class _CompiledWrapper(nn.Module):
    """Stand-in for torch.compile's OptimizedModule (`_orig_mod` only)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner


def _make_freeze_trainer(
    model: nn.Module, finetune_modules: list | dict
) -> BaseTrainer:
    trainer = object.__new__(BaseTrainer)
    trainer.cfg = SimpleNamespace(finetune_modules=finetune_modules)
    trainer.model = model
    return trainer


def _trainable_names(model: nn.Module) -> set[str]:
    return {
        name for name, param in model.named_parameters() if param.requires_grad
    }


def test_freeze_model_parts_unwraps_compiled_model() -> None:
    """Parts are resolved on `_orig_mod`, so compiled models freeze right."""
    inner = _TinyModel()
    trainer = _make_freeze_trainer(
        _CompiledWrapper(inner), {'embeddings': ['cpc'], 'head': True}
    )

    trainer._freeze_model_parts()  # noqa: SLF001

    assert _trainable_names(inner) == {
        'embeddings.cpc.weight',
        'embeddings.cpc.bias',
        'head.weight',
        'head.bias',
    }


def test_freeze_model_parts_list_unfreezes_whole_parts() -> None:
    """A plain list unfreezes every parameter of each listed part."""
    model = _TinyModel()
    trainer = _make_freeze_trainer(model, ['embeddings'])

    trainer._freeze_model_parts()  # noqa: SLF001

    assert _trainable_names(model) == {
        'embeddings.era5.weight',
        'embeddings.era5.bias',
        'embeddings.cpc.weight',
        'embeddings.cpc.bias',
    }


@pytest.mark.parametrize(
    'finetune_modules',
    [['head', 'decoder'], {'head': True, 'decoder': True}],
)
def test_freeze_model_parts_raises_on_unknown_module(
    finetune_modules: list | dict,
) -> None:
    """Unknown names in `finetune_modules` raise instead of being skipped."""
    model = _TinyModel()
    trainer = _make_freeze_trainer(_CompiledWrapper(model), finetune_modules)

    with pytest.raises(
        ValueError,
        match=(
            r"Unresolved module name\(s\) in 'finetune_modules': "
            r"\['decoder'\]\. "
            r"Available modules: \['embeddings', 'encoder', 'head'\]"
        ),
    ):
        trainer._freeze_model_parts()  # noqa: SLF001
    # Nothing is unfrozen when the configuration is rejected.
    assert _trainable_names(model) == set()


@pytest.mark.parametrize('is_finetuning', [True, False])
def test_restore_training_state_forwards_finetuning_flag(
    tmp_path: Path, *, is_finetuning: bool
) -> None:
    """Resuming passes `allow_new_embeddings=cfg.is_finetuning` on."""
    base_run_dir = tmp_path / 'base_run'
    base_run_dir.mkdir()
    (base_run_dir / 'model_epoch002.pt').touch()
    optimizer = torch.optim.Adam(nn.Linear(2, 1).parameters())
    torch.save(
        optimizer.state_dict(), base_run_dir / 'optimizer_state_epoch002.pt'
    )

    trainer = object.__new__(BaseTrainer)
    trainer.cfg = SimpleNamespace(
        continue_from_epoch=None,
        base_run_dir=base_run_dir,
        is_finetuning=is_finetuning,
    )
    trainer.device = torch.device('cpu')
    trainer.optimizer = torch.optim.Adam(nn.Linear(2, 1).parameters())
    trainer._load_model_weights = MagicMock(return_value=(set(), set()))  # noqa: SLF001

    trainer._restore_training_state()  # noqa: SLF001

    trainer._load_model_weights.assert_called_once_with(  # noqa: SLF001
        base_run_dir / 'model_epoch002.pt',
        allow_new_embeddings=is_finetuning,
    )
