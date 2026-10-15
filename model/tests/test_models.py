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

"""Unit tests for model.modelzoo architectures and layers."""

import pytest
import torch

from model.modelzoo import MODEL_REGISTRY, get_model, get_model_class
from model.modelzoo.basemodel import BaseModel
from model.modelzoo.fc import FC
from model.modelzoo.head import get_head
from model.modelzoo.positional_encoding import PositionalEncoding
from model.utils.config import Config


@pytest.mark.unit
def test_positional_encoding_concatenate():
    seq_len = 10
    batch_size = 4
    embedding_dim = 16

    pe = PositionalEncoding(
        embedding_dim=embedding_dim,
        position_type='concatenate',
        dropout=0.0,
        max_len=100,
    )
    x = torch.randn(seq_len, batch_size, embedding_dim)
    out = pe(x)
    # Concatenate mode doubles embedding dimension: [seq, batch, 2 * dim]
    assert out.shape == (seq_len, batch_size, embedding_dim * 2)


@pytest.mark.unit
def test_positional_encoding_sum():
    seq_len = 10
    batch_size = 4
    embedding_dim = 16

    pe = PositionalEncoding(
        embedding_dim=embedding_dim,
        position_type='sum',
        dropout=0.1,
        max_len=100,
    )
    x = torch.randn(seq_len, batch_size, embedding_dim)
    out = pe(x)
    # Sum mode keeps embedding dimension: [seq, batch, dim]
    assert out.shape == (seq_len, batch_size, embedding_dim)


@pytest.mark.unit
def test_positional_encoding_invalid_type():
    with pytest.raises(
        RuntimeError, match='Unrecognized positional encoding type'
    ):
        PositionalEncoding(
            embedding_dim=16,
            position_type='invalid_type',
            dropout=0.0,
        )


@pytest.mark.unit
def test_fc_empty_hidden_sizes():
    with pytest.raises(
        ValueError, match='hidden_sizes must at least have one entry'
    ):
        FC(input_size=10, hidden_sizes=[])


@pytest.mark.unit
def test_fc_activations_and_shapes():
    # Single layer linear (hidden_sizes=[output_size])
    fc_linear = FC(input_size=10, hidden_sizes=[5])
    x = torch.randn(4, 10)
    out = fc_linear(x)
    assert out.shape == (4, 5)

    # Multi-layer with activations and dropout
    activations = ['relu', 'tanh', 'sigmoid', 'linear']
    for act in activations:
        fc = FC(
            input_size=10,
            hidden_sizes=[20, 15, 3],
            activation=act,
            dropout=0.1,
            xavier_init=True,
        )
        out = fc(x)
        assert out.shape == (4, 3)

    # Multi-layer with list of activations
    fc_multi_act = FC(
        input_size=10,
        hidden_sizes=[20, 15, 3],
        activation=['relu', 'tanh'],
    )
    out = fc_multi_act(x)
    assert out.shape == (4, 3)

    # Unsupported activation
    with pytest.raises(
        NotImplementedError, match='currently not supported as activation'
    ):
        FC(input_size=10, hidden_sizes=[20, 3], activation='invalid_act')


@pytest.mark.unit
def test_head_regression(minimal_config):
    minimal_config.update_config(
        {'head': 'regression', 'output_activation': 'linear'}
    )
    head = get_head(cfg=minimal_config, n_in=32, n_out=1)

    x = torch.randn(4, 10, 32)  # [batch, seq, in_features]
    out = head(x)
    assert 'y_hat' in out
    assert out['y_hat'].shape == (4, 10, 1)


@pytest.mark.unit
def test_head_cmal(minimal_config):
    minimal_config.update_config({'head': 'cmal'})
    # n_out for CMAL should match n_targets * 4 * n_distributions
    head = get_head(cfg=minimal_config, n_in=32, n_out=12, n_hidden=50)

    x = torch.randn(4, 10, 32)
    out = head(x)
    assert 'mu' in out
    assert 'b' in out
    assert 'tau' in out
    assert 'pi' in out

    assert torch.all(out['b'] > 0)
    assert torch.all((out['tau'] > 0) & (out['tau'] < 1))


@pytest.mark.unit
def test_head_invalid_type(make_minimal_config):
    cfg_empty = make_minimal_config({'head': '', 'model': 'lstm'})
    with pytest.raises(ValueError, match="No 'head' specified"):
        get_head(cfg=cfg_empty, n_in=32, n_out=1)

    cfg_unsupported = make_minimal_config({'head': 'unknown_head'})
    with pytest.raises(NotImplementedError, match='not implemented'):
        get_head(cfg=cfg_unsupported, n_in=32, n_out=1)


@pytest.mark.unit
def test_base_model_methods(monkeypatch, minimal_config):
    sample_calls = []

    def fake_sample_pointpredictions(*args, **kwargs):
        sample_calls.append((args, kwargs))
        return {'y_hat': torch.zeros(1)}

    monkeypatch.setattr(
        'model.modelzoo.basemodel.sample_pointpredictions',
        fake_sample_pointpredictions,
    )
    monkeypatch.setattr(
        'model.modelzoo.basemodel.Scaler', lambda **_kwargs: None
    )

    minimal_config.update_config(
        {
            'target_variables': ['streamflow'],
            'head': 'regression',
            'base_run_dir': None,
            'run_dir': None,
            'is_finetuning': False,
        }
    )

    class SimpleModel(BaseModel):
        def forward(self, data):
            return data

    model = SimpleModel(minimal_config)
    data = {'x': torch.zeros(1)}

    # Test pre_model_hook
    hook_out = model.pre_model_hook(data, is_train=True)
    assert hook_out == data

    # Test sample method
    samples = model.sample(data, n_samples=5)
    assert 'y_hat' in samples
    assert len(sample_calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize('save_compiled', [True, False])
@pytest.mark.parametrize('eval_compiled', [True, False])
def test_load_weights_torch_compile_compatibility(
    tmp_path, save_compiled, eval_compiled
):
    from model.evaluation.tester import BaseTester
    from model.training.basetrainer import BaseTrainer

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    if save_compiled:
        model = torch.compile(model)

    weight_path = tmp_path / 'model_epoch001.pt'
    torch.save(model.state_dict(), weight_path)

    tester = BaseTester.__new__(BaseTester)
    tester.run_dir = tmp_path
    tester.device = 'cpu'
    eval_model = DummyModel()
    if eval_compiled:
        eval_model = torch.compile(eval_model)
    tester.model = eval_model
    tester._load_weights(epoch=1)

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.device = 'cpu'
    train_model = DummyModel()
    if eval_compiled:
        train_model = torch.compile(train_model)
    trainer.model = train_model
    trainer._load_model_weights(weight_path)


@pytest.mark.unit
def test_all_registered_models_expose_tunable_module_parts() -> None:
    """Verify every registered model class declares non-empty module_parts."""
    assert MODEL_REGISTRY, 'MODEL_REGISTRY must not be empty'
    for model_name, model_cls in MODEL_REGISTRY.items():
        resolved_cls = get_model_class(model_name)
        assert resolved_cls is model_cls
        assert isinstance(model_cls.module_parts, list)
        assert len(model_cls.module_parts) > 0, (
            f'Model {model_name} must declare non-empty module_parts'
        )
        assert model_cls.get_tunable_component_names() == list(
            model_cls.module_parts
        )

    with pytest.raises(NotImplementedError, match='not implemented'):
        get_model_class('non_existent_model_name')


@pytest.mark.unit
def test_describe_tunable_components_covers_all_parameters(
    monkeypatch: pytest.MonkeyPatch,
    minimal_config: Config,
) -> None:
    """Verify describe_tunable_components covers Handoff and MeanEmbedding."""
    monkeypatch.setattr(
        'model.modelzoo.basemodel.Scaler', lambda **_kwargs: None
    )
    minimal_config.update_config(
        {
            'model': 'handoff_forecast_lstm',
            'head': 'regression',
            'output_activation': 'linear',
            'hidden_size': 16,
            'forecast_hidden_size': 8,
            'state_handoff_network': {'hiddens': [12], 'activation': 'tanh'},
            'hindcast_inputs': [['era5_precip', 'era5_temp'], ['imerg_precip']],
            'forecast_inputs': [['hres_precip', 'hres_temp']],
            'static_attributes': ['area', 'p_mean'],
            'target_variables': ['streamflow'],
            'seq_length': 14,
            'lead_time': 5,
            'base_run_dir': None,
            'run_dir': None,
            'is_finetuning': False,
        }
    )
    handoff_model = get_model(minimal_config)
    assert isinstance(handoff_model, BaseModel)
    handoff_comps = handoff_model.describe_tunable_components()
    handoff_names = [c['name'] for c in handoff_comps]
    assert 'handoff_linear' in handoff_names
    for part in handoff_model.module_parts:
        assert part in handoff_names

    top_level_param_ids = {
        id(param)
        for part in handoff_model.module_parts
        for param in getattr(handoff_model, part).parameters()
    }
    all_model_param_ids = {id(p) for p in handoff_model.parameters()}
    assert top_level_param_ids == all_model_param_ids

    # Also verify MeanEmbeddingForecastLSTM exposes per-product subcomponents
    minimal_config.update_config(
        {
            'model': 'mean_embedding_forecast_lstm',
            'hindcast_inputs': {
                'era5': ['era5_precip', 'era5_temp'],
                'imerg': ['imerg_precip'],
            },
            'forecast_inputs': {
                'hres': ['hres_precip', 'hres_temp'],
                'imerg': ['imerg_precip'],
            },
            'statics_embedding': {'type': 'fc', 'hiddens': [8]},
            'dynamics_embedding': {'type': 'fc', 'hiddens': [8]},
        }
    )
    mean_model = get_model(minimal_config)
    assert isinstance(mean_model, BaseModel)
    mean_comps = mean_model.describe_tunable_components()
    sub_ids: set[object] = set()
    for comp in mean_comps:
        subcomponents = comp.get('subcomponents')
        if isinstance(subcomponents, list):
            for sub in subcomponents:
                if isinstance(sub, dict) and 'id' in sub:
                    sub_ids.add(sub['id'])
    assert 'hindcast_embeddings_fc.era5' in sub_ids
    assert 'forecast_embeddings_fc.hres' in sub_ids
    assert 'shared_embeddings_fc.imerg' in sub_ids
