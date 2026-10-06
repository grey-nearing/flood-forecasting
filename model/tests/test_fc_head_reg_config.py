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

"""Unit tests for FC, Head, Regularization, and Config."""

import logging
from pathlib import Path

import pandas as pd
import pytest
import torch

from model.modelzoo.fc import FC
from model.modelzoo.head import CMAL, Regression, get_head
from model.training import get_regularization_obj
from model.training.regularization import (
    BaseRegularization,
    ForecastOverlapMSERegularization,
)
from model.utils import cmal_deterministic
from model.utils.config import Config

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# FC (model/modelzoo/fc.py)
# ---------------------------------------------------------------------------


def test_fc_single_and_multilayer_and_output_size():
    fc_single = FC(input_size=12, hidden_sizes=[16])
    assert fc_single.output_size == 16
    x = torch.randn(3, 5, 12)
    out_single = fc_single(x)
    assert out_single.shape == (3, 5, 16)

    fc_multi = FC(
        input_size=12,
        hidden_sizes=[16, 8, 4],
        activation=['relu', 'tanh'],
        dropout=0.2,
    )
    assert fc_multi.output_size == 4
    out_multi = fc_multi(x)
    assert out_multi.shape == (3, 5, 4)


@pytest.mark.parametrize('act', ['relu', 'tanh', 'sigmoid', 'linear'])
@pytest.mark.parametrize('dropout', [0.0, 0.2])
@pytest.mark.parametrize('xavier_init', [True, False])
def test_fc_activations_dropout_and_reset_parameters(
    act: str, dropout: float, xavier_init: bool
):
    fc = FC(
        input_size=8,
        hidden_sizes=[16, 4],
        activation=act,
        dropout=dropout,
        xavier_init=xavier_init,
    )
    assert fc.output_size == 4
    fc._reset_parameters()
    x = torch.randn(4, 8)
    out = fc(x)
    assert out.shape == (4, 4)
    assert torch.all(torch.isfinite(out))


def test_fc_error_cases():
    with pytest.raises(
        ValueError, match='hidden_sizes must at least have one entry'
    ):
        FC(input_size=8, hidden_sizes=[])

    # Mismatched activation list length (for 3 hidden sizes, expects >= 2 activations)
    with pytest.raises((ValueError, IndexError)):
        FC(
            input_size=8,
            hidden_sizes=[16, 8, 4],
            activation=['relu'],
        )

    # Exact-length activation list (len == len(hidden_sizes))
    fc_exact = FC(
        input_size=8,
        hidden_sizes=[16, 8, 4],
        activation=['relu', 'tanh', 'linear'],
    )
    assert fc_exact(torch.randn(2, 8)).shape == (2, 4)

    with pytest.raises(
        NotImplementedError, match='currently not supported as activation'
    ):
        FC(input_size=8, hidden_sizes=[16, 4], activation='gelu')


# ---------------------------------------------------------------------------
# Head (model/modelzoo/head.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('activation', ['linear', 'relu', 'softplus'])
def test_regression_head_activations(minimal_config, activation: str):
    minimal_config.update_config(
        {'head': 'regression', 'output_activation': activation}
    )
    head = get_head(minimal_config, n_in=10, n_out=2)
    assert isinstance(head, Regression)

    x = torch.tensor([[[-5.0] * 10, [5.0] * 10]])
    out = head(x)['y_hat']
    assert out.shape == (1, 2, 2)
    if activation == 'relu':
        assert torch.all(out >= 0.0)
    elif activation == 'softplus':
        assert torch.all(out > 0.0)


def test_regression_head_unsupported_activation_falls_back_to_linear(
    minimal_config, caplog
):
    minimal_config.update_config(
        {'head': 'regression', 'output_activation': 'invalid_act'}
    )
    with caplog.at_level(logging.WARNING):
        head = get_head(minimal_config, n_in=10, n_out=1)
    assert isinstance(head, Regression)
    assert len(head.net) == 1
    assert 'Ignored output activation invalid_act' in caplog.text


@pytest.mark.parametrize('head_name', ['cmal', 'cmal_deterministic'])
def test_cmal_head_constraints_and_deterministic_expectation(
    minimal_config, head_name: str
):
    minimal_config.update_config({'head': head_name})
    # 1 target * 4 params * 3 distributions = 12
    head = get_head(minimal_config, n_in=16, n_out=12, n_hidden=24)
    assert isinstance(head, CMAL)

    x = torch.randn(4, 6, 16)
    out = head(x)
    assert set(out.keys()) == {'mu', 'b', 'tau', 'pi'}
    for key in ('mu', 'b', 'tau', 'pi'):
        assert out[key].shape == (4, 6, 3)

    assert torch.all(out['b'] > 0)
    assert torch.all((out['tau'] > 0) & (out['tau'] < 1))
    torch.testing.assert_close(
        out['pi'].sum(dim=-1),
        torch.ones(4, 6),
        atol=1e-4,
        rtol=1e-4,
    )

    det = cmal_deterministic.generate_predictions.__wrapped__(
        out['mu'], out['b'], out['tau'], out['pi']
    )
    # 1 mean + 9 quantiles = 10 representative points along last dim
    assert det.shape == (4, 6, 10)

    # First element along last dim is the analytical mixture mean:
    # sum_k pi_k * (mu_k + b_k * (1 - 2 * tau_k) / (tau_k * (1 - tau_k)))
    mu, b, tau, pi = out['mu'], out['b'], out['tau'], out['pi']
    expected_mean = (
        pi * (mu + b * (1.0 - 2.0 * tau) / (tau * (1.0 - tau)))
    ).sum(dim=-1)
    torch.testing.assert_close(det[..., 0], expected_mean, atol=1e-5, rtol=1e-5)


def test_get_head_errors(make_minimal_config):
    cfg_none = make_minimal_config({'head': ''})
    with pytest.raises(ValueError, match="No 'head' specified"):
        get_head(cfg_none, n_in=8, n_out=1)

    cfg_unknown = make_minimal_config({'head': 'gmm'})
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_head(cfg_unknown, n_in=8, n_out=1)


# ---------------------------------------------------------------------------
# Regularization (model/training/regularization.py)
# ---------------------------------------------------------------------------


def test_base_regularization_raises_not_implemented(minimal_config):
    base_reg = BaseRegularization(
        cfg=minimal_config, name='base', weight=1.0
    )
    with pytest.raises(NotImplementedError):
        base_reg(prediction={}, ground_truth={}, other_model_data={})


def test_forecast_overlap_mse_regularization(minimal_config):
    reg = ForecastOverlapMSERegularization(
        cfg=minimal_config, weight=0.5
    )
    assert reg.name == 'forecast_overlap'
    assert reg.weight == 0.5

    y_forecast = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    y_hindcast = torch.tensor([[[1.0, 0.0], [5.0, 4.0]]])

    # Squared differences: 0, 4, 4, 0 -> mean = 2.0
    loss = reg(
        prediction={},
        ground_truth={},
        other_model_output={
            'y_forecast_overlap': y_forecast,
            'y_hindcast_overlap': y_hindcast,
        },
    )
    torch.testing.assert_close(loss, torch.tensor(2.0), atol=1e-6, rtol=1e-6)

    with pytest.raises(ValueError, match='y_hindcast_overlap is not present'):
        reg(
            prediction={},
            ground_truth={},
            other_model_output={'y_forecast_overlap': y_forecast},
        )

    with pytest.raises(ValueError, match='y_forecast_overlap is not present'):
        reg(
            prediction={},
            ground_truth={},
            other_model_output={'y_hindcast_overlap': y_hindcast},
        )


def test_get_regularization_obj_parsing_and_errors(make_minimal_config):
    cfg = make_minimal_config(
        {'regularization': ['forecast_overlap', ('forecast_overlap', 0.25)]}
    )
    regs = get_regularization_obj(cfg)
    assert len(regs) == 2
    assert isinstance(regs[0], ForecastOverlapMSERegularization)
    assert regs[0].weight == 1.0
    assert isinstance(regs[1], ForecastOverlapMSERegularization)
    assert regs[1].weight == 0.25

    cfg_bad = make_minimal_config({'regularization': ['l2_penalty']})
    with pytest.raises(
        NotImplementedError, match='not implemented'
    ):
        get_regularization_obj(cfg_bad)


# ---------------------------------------------------------------------------
# Config (model/utils/config.py)
# ---------------------------------------------------------------------------


def test_config_init_dump_and_reload(minimal_config, tmp_path):
    dump_dir = tmp_path / 'dumped'
    dump_dir.mkdir(parents=True, exist_ok=True)
    minimal_config.dump_config(folder=dump_dir, filename='test_cfg.yml')

    reloaded = Config(dump_dir / 'test_cfg.yml')
    assert reloaded.experiment_name == minimal_config.experiment_name
    assert reloaded.batch_size == minimal_config.batch_size
    assert reloaded.hidden_size == minimal_config.hidden_size
    assert reloaded.data_dir == minimal_config.data_dir
    assert reloaded.train_start_date == minimal_config.train_start_date

    with pytest.raises(FileNotFoundError):
        Config(dump_dir / 'does_not_exist.yml')


def test_config_update_and_unknown_key_check(minimal_config):
    minimal_config.update_config({'batch_size': 32, 'epochs': 5})
    assert minimal_config.batch_size == 32
    assert minimal_config.epochs == 5
    assert minimal_config.as_dict()['batch_size'] == 32

    with pytest.raises(ValueError, match='not recognized config keys'):
        minimal_config.update_config(
            {'completely_unknown_key': 123}, dev_mode=False
        )

    # dev_mode=True allows arbitrary keys
    minimal_config.update_config(
        {'completely_unknown_key': 123}, dev_mode=True
    )


def test_config_path_and_date_parsing_and_data_dir_propagation(tmp_path):
    raw = {
        'data_dir': str(tmp_path / 'data'),
        'train_basin_file': str(tmp_path / 'basins.txt'),
        'train_start_date': '01/01/2020',
        'train_end_date': ['31/01/2020', '28/02/2020'],
    }
    cfg = Config(raw, dev_mode=True)
    assert isinstance(cfg.data_dir, Path)
    assert isinstance(cfg.train_basin_file, Path)
    assert cfg.statics_data_dir == cfg.data_dir
    assert cfg.dynamics_data_dir == cfg.data_dir
    assert cfg.targets_data_dir == cfg.data_dir
    assert cfg.statics_data_path == cfg.data_dir
    assert cfg.dynamics_data_path == cfg.data_dir
    assert cfg.targets_data_path == cfg.data_dir
    assert cfg.train_start_date == [pd.Timestamp('2020-01-01')]
    assert cfg.train_end_date == [
        pd.Timestamp('2020-01-31'),
        pd.Timestamp('2020-02-28'),
    ]


def test_config_properties_setters_and_validation_errors(make_minimal_config):
    cfg = make_minimal_config()

    # Device setter validation
    cfg.device = 'cpu'
    assert cfg.device == 'cpu'
    cfg.device = 'cuda:0'
    assert cfg.device == 'cuda:0'
    cfg.device = 'mps'
    assert cfg.device == 'mps'
    with pytest.raises(ValueError, match="'device' must be either"):
        cfg.device = 'tpu:0'

    # Logging level validation
    for name, expected in [
        ('DEBUG', logging.DEBUG),
        ('INFO', logging.INFO),
        ('WARNING', logging.WARNING),
        ('ERROR', logging.ERROR),
        ('CRITICAL', logging.CRITICAL),
    ]:
        cfg.update_config({'logging_level': name})
        assert cfg.logging_level == expected

    cfg.update_config({'logging_level': 'INVALID_LEVEL'})
    with pytest.raises(ValueError, match='Invalid logging_level'):
        _ = cfg.logging_level

    # Seed setter can only be set when seed is None
    cfg_no_seed = make_minimal_config({'seed': None})
    assert cfg_no_seed.seed is None
    cfg_no_seed.seed = 123
    assert cfg_no_seed.seed == 123
    with pytest.raises(RuntimeError, match='Seed was already specified'):
        cfg_no_seed.seed = 456

    # Finetune modules property
    cfg.update_config({'finetune_modules': 'head'})
    assert cfg.finetune_modules == ['head']
    cfg.update_config({'finetune_modules': ['head', 'lstm']})
    assert cfg.finetune_modules == ['head', 'lstm']
    cfg.update_config({'finetune_modules': {'head': 'head'}})
    assert cfg.finetune_modules == {'head': 'head'}
    cfg.update_config({'finetune_modules': 42})
    with pytest.raises(ValueError, match='Unknown data type'):
        _ = cfg.finetune_modules

    # Embedding specs
    cfg.update_config(
        {
            'statics_embedding': {
                'type': 'fc',
                'hiddens': [16, 8],
                'activation': 'relu',
                'dropout': 0.1,
            },
            'dynamics_embedding': None,
        }
    )
    spec = cfg.statics_embedding
    assert spec is not None
    assert spec.type == 'fc'
    assert spec.hiddens == [16, 8]
    assert spec.activation == ['relu', 'relu']
    assert spec.dropout == 0.1
    assert cfg.dynamics_embedding is None

    # Setters for flags and metrics
    cfg.is_continue_training = True
    assert cfg.is_continue_training is True
    cfg.is_finetuning = True
    assert cfg.is_finetuning is True
    cfg.loss = 'NSE'
    assert cfg.loss == 'NSE'
    cfg.metrics = ['NSE', 'KGE']
    assert cfg.metrics == ['NSE', 'KGE']

    # Mandatory key missing / None raises ValueError
    cfg_missing = Config({}, dev_mode=True)
    with pytest.raises(ValueError, match='is not specified in the config'):
        _ = cfg_missing.batch_size
    cfg_none_val = Config({'batch_size': None}, dev_mode=True)
    with pytest.raises(ValueError, match="is mandatory but 'None'"):
        _ = cfg_none_val.batch_size
