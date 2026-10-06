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
from model.utils.config import Config

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
                'no_loss_frequencies': [],
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


def test_basetrainer_gradient_clipping_and_checkpoints(make_minimal_config):
    cfg = make_minimal_config(
        {
            'epochs': 1,
            'max_updates_per_epoch': 2,
            'clip_gradient_norm': 0.01,
            'save_weights_every': 1,
        }
    )
    trainer = BaseTrainer(cfg)
    trainer.initialize_training()

    initial_head_weight = trainer.model.head.net[0].weight.detach().clone()
    trainer.train_and_validate()

    # Model weights updated and finite
    updated_head_weight = trainer.model.head.net[0].weight.detach()
    assert torch.all(torch.isfinite(updated_head_weight))
    assert not torch.equal(initial_head_weight, updated_head_weight)

    # Checkpoints saved
    assert (cfg.run_dir / 'model_epoch001.pt').exists()
    assert (cfg.run_dir / 'optimizer_state_epoch001.pt').exists()


@pytest.mark.parametrize(
    'strategy,expected_lr_after_epoch',
    [
        ('ConstantLR', 0.01),
        ('StepLR', 0.005),
        ('ReduceLROnPlateau', 0.01),
    ],
)
def test_basetrainer_lr_schedulers(
    make_minimal_config, strategy: str, expected_lr_after_epoch: float
):
    cfg = make_minimal_config(
        {
            'epochs': 1,
            'max_updates_per_epoch': 1,
            'initial_learning_rate': 0.01,
            'learning_rate_strategy': strategy,
            'learning_rate_epochs_drop': 1,
            'learning_rate_drop_factor': 0.5,
        }
    )
    trainer = BaseTrainer(cfg)
    trainer.initialize_training()
    trainer.train_and_validate()

    current_lr = trainer.optimizer.param_groups[0]['lr']
    assert current_lr == pytest.approx(expected_lr_after_epoch)


def test_basetrainer_unsupported_lr_scheduler(make_minimal_config):
    cfg = make_minimal_config(
        {'learning_rate_strategy': 'CosineAnnealingLR'}
    )
    trainer = BaseTrainer(cfg)
    trainer.initialize_training()
    with pytest.raises(
        ValueError, match='learning_rate_strategy unsupported'
    ):
        trainer._create_lr_scheduler()


def test_basetrainer_checkpoint_resume_continue_training(make_minimal_config):
    # 1. Run initial 1-epoch training
    base_cfg = make_minimal_config(
        {
            'epochs': 1,
            'max_updates_per_epoch': 2,
            'save_weights_every': 1,
        }
    )
    base_trainer = BaseTrainer(base_cfg)
    base_trainer.initialize_training()
    base_trainer.train_and_validate()

    base_run_dir = base_cfg.run_dir
    assert (base_run_dir / 'model_epoch001.pt').exists()
    assert (base_run_dir / 'optimizer_state_epoch001.pt').exists()

    # 2. Continue training from the saved run directory for 1 additional epoch
    resume_cfg = Config(base_run_dir / 'config.yml')
    resume_cfg.update_config(
        {
            'run_dir': base_run_dir,
            'is_continue_training': True,
            'epochs': 1,
            'max_updates_per_epoch': 2,
        }
    )
    resume_trainer = BaseTrainer(resume_cfg)
    assert resume_trainer._epoch == 1
    assert resume_cfg.base_run_dir == base_run_dir
    assert (
        resume_cfg.run_dir
        == base_run_dir / 'continue_training_from_epoch001'
    )

    resume_trainer.initialize_training()
    assert resume_trainer.experiment_logger.epoch == 1

    resume_trainer.train_and_validate()
    assert (resume_cfg.run_dir / 'model_epoch002.pt').exists()
    assert (resume_cfg.run_dir / 'optimizer_state_epoch002.pt').exists()


def test_basetrainer_finetuning_freezes_non_finetune_modules(
    make_minimal_config, single_basin_dataset, tmp_path
):
    # 1. Pre-train a base model for 1 epoch on the 5-basin dataset
    base_cfg = make_minimal_config(
        {
            'epochs': 1,
            'max_updates_per_epoch': 2,
            'save_weights_every': 1,
        }
    )
    base_trainer = BaseTrainer(base_cfg)
    base_trainer.initialize_training()
    base_trainer.train_and_validate()
    base_run_dir = base_cfg.run_dir

    # 2. Fine-tune only 'head' and 'static_embedding_fc' on the 1-basin dataset
    finetune_cfg = Config(base_run_dir / 'config.yml')
    finetune_cfg.update_config(
        {
            'base_run_dir': base_run_dir,
            'run_dir': tmp_path / 'finetune_runs',
            'experiment_name': 'finetune_test',
            'is_finetuning': True,
            'is_continue_training': False,
            'finetune_modules': ['head', 'static_embedding_fc'],
            'data_dir': single_basin_dataset.data_dir,
            'train_basin_file': single_basin_dataset.basin_file,
            'validation_basin_file': single_basin_dataset.basin_file,
            'test_basin_file': single_basin_dataset.basin_file,
            'epochs': 1,
            'max_updates_per_epoch': 2,
            'initial_learning_rate': 0.05,
        }
    )

    ft_trainer = BaseTrainer(finetune_cfg)
    ft_trainer.initialize_training()

    # Verify parameter freeze state
    for name, param in ft_trainer.model.named_parameters():
        if name.startswith(('head.', 'static_embedding_fc.')):
            assert param.requires_grad, f'Expected {name} to be trainable'
        else:
            assert not param.requires_grad, f'Expected {name} to be frozen'

    frozen_before = {
        name: param.detach().clone()
        for name, param in ft_trainer.model.named_parameters()
        if not param.requires_grad
    }
    trainable_before = {
        name: param.detach().clone()
        for name, param in ft_trainer.model.named_parameters()
        if param.requires_grad
    }

    ft_trainer.train_and_validate()

    # Frozen parameters must remain identical
    for name, param in ft_trainer.model.named_parameters():
        if not param.requires_grad:
            torch.testing.assert_close(
                param.detach(), frozen_before[name], rtol=0, atol=0
            )

    # At least one trainable parameter must have updated
    assert any(
        not torch.equal(param.detach(), trainable_before[name])
        for name, param in ft_trainer.model.named_parameters()
        if param.requires_grad
    )
