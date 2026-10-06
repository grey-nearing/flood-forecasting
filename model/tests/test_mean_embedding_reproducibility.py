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

"""Keep model initialization independent of Python's string hash seed."""

import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import xarray as xr

import model as model_pkg
from model.datasetzoo.multimet import (
    _get_products_and_bands_from_feature_strings,
)
from model.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from model.utils.config import Config
from model.utils.configutils import flatten_feature_list

# A separate interpreter is required: changing os.environ in-process does not
# change the hash seed that Python selected at startup.
_PROCESS_CODE = """
import json
import sys
from pathlib import Path
import torch
from model.utils.config import Config
from model.utils.configutils import flatten_feature_list
from model.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from model.training import get_loss_obj

torch.set_num_threads(1)
config_paths = json.loads(sys.argv[1])
results = {}
for layout, config_path in config_paths.items():
    cfg = Config(Path(config_path))
    torch.manual_seed(cfg.seed)
    model = MeanEmbeddingForecastLSTM(cfg)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.manual_seed(99)
    data = {
        'x_s': torch.rand(2, len(cfg.static_attributes)),
        'x_d_hindcast': {
            name: torch.rand(2, cfg.seq_length, 1)
            for name in flatten_feature_list(cfg.hindcast_inputs)
        },
        'x_d_forecast': {
            name: torch.rand(2, cfg.seq_length + cfg.lead_time, 1)
            for name in flatten_feature_list(cfg.forecast_inputs)
        },
        'y': torch.rand(2, cfg.seq_length + cfg.lead_time, 1),
    }
    predictions = model(data)
    loss, _ = get_loss_obj(cfg)(predictions, data)
    loss.backward()
    gradients = {
        k: v.grad.detach().clone() for k, v in model.named_parameters()
    }
    results[layout] = {
        'weights': initial,
        'predictions': predictions,
        'gradients': gradients,
        'loss': loss,
    }
torch.save(results, sys.argv[2])
"""


def _config(tmp_path: Path, layout: str = 'dict') -> Config:
    """Build real, differently sized shared and period-specific embeddings."""
    hindcast = {
        'zulu': ['zulu_rain', 'zulu_temp'],
        'sharedtwo': ['sharedtwo_rain'],
        'alpha': ['alpha_rain'],
        'sharedone': ['sharedone_rain', 'sharedone_temp'],
        'mike': ['mike_rain'],
    }
    forecast = {
        'omega': ['omega_rain'],
        'sharedone': ['sharedone_rain', 'sharedone_temp'],
        'delta': ['delta_rain', 'delta_temp'],
        'sharedtwo': ['sharedtwo_rain'],
        'bravo': ['bravo_rain'],
    }
    if layout == 'nested':
        hindcast, forecast = list(hindcast.values()), list(forecast.values())
    elif layout == 'flat':
        hindcast = flatten_feature_list(hindcast)
        forecast = flatten_feature_list(forecast)
    embedding = {
        'type': 'fc',
        'hiddens': [4],
        'activation': ['tanh'],
        'dropout': 0.0,
    }
    cfg = Config(
        {
            'model': 'mean_embedding_forecast_lstm',
            'run_dir': str(tmp_path),
            'seed': 17,
            'seq_length': 4,
            'lead_time': 2,
            'forecast_overlap': 4,
            'predict_last_n': 2,
            'hidden_size': 4,
            'head': 'regression',
            'loss': 'mse',
            'target_variables': ['streamflow'],
            'static_attributes': ['area'],
            'hindcast_inputs': hindcast,
            'forecast_inputs': forecast,
            'statics_embedding': embedding.copy(),
            'hindcast_embedding': embedding.copy(),
            'forecast_embedding': embedding.copy(),
            'output_dropout': 0.0,
        }
    )
    xr.Dataset(
        {'streamflow': ('parameter', [0.0, 1.0, 0.0, 1.0])},
        coords={'parameter': ['center', 'scale', 'mean', 'std']},
    ).to_netcdf(tmp_path / 'scaler.nc', engine='scipy')
    return cfg


@pytest.mark.unit
@pytest.mark.parametrize('layout', ['dict', 'nested', 'flat'])
def test_embedding_groups_follow_config_order(
    tmp_path: Path, layout: str
) -> None:
    """Retain config order and instantiate shared groups only once."""
    cfg = _config(tmp_path, layout)
    with torch.random.fork_rng(devices=[]):
        model = MeanEmbeddingForecastLSTM(cfg)
    assert list(model.hindcast_embeddings_fc) == ['zulu', 'alpha', 'mike']
    assert list(model.forecast_embeddings_fc) == ['omega', 'delta', 'bravo']
    assert list(model.shared_embeddings_fc) == ['sharedtwo', 'sharedone']


@pytest.fixture(scope='module')
def hash_seed_snapshots(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, list[dict]]:
    """Run fresh interpreters for each PYTHONHASHSEED in parallel."""
    work_dir = tmp_path_factory.mktemp('hash_seed_repro')
    layouts = ('dict', 'nested', 'flat')
    config_paths: dict[str, str] = {}
    for layout in layouts:
        layout_dir = work_dir / layout
        layout_dir.mkdir(parents=True, exist_ok=True)
        cfg = _config(layout_dir, layout)
        options = cfg.as_dict()
        options['run_dir'] = str(cfg.run_dir)
        config_path = layout_dir / 'model.yml'
        config_path.write_text(json.dumps(options), encoding='utf-8')
        config_paths[layout] = str(config_path)

    config_arg = json.dumps(config_paths)
    pythonpath = str(Path(model_pkg.__file__).resolve().parents[1])
    procs: list[tuple[Path, subprocess.Popen]] = []
    for hash_seed in ('1', '7', '19'):
        output = work_dir / f'state-{hash_seed}.pt'
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = hash_seed
        # Use the active installation, also when testing an installed wheel.
        env['PYTHONPATH'] = pythonpath
        proc = subprocess.Popen(  # noqa: S603 - Fixed interpreter and test code.
            [
                sys.executable,
                '-c',
                _PROCESS_CODE,
                config_arg,
                str(output),
            ],
            env=env,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append((output, proc))

    by_layout: dict[str, list[dict]] = {layout: [] for layout in layouts}
    for output, proc in procs:
        stdout, stderr = proc.communicate(timeout=90)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                proc.args,
                output=stdout,
                stderr=stderr,
            )
        loaded = torch.load(output, weights_only=True)
        for layout in layouts:
            by_layout[layout].append(loaded[layout])
    return by_layout


@pytest.mark.integration
@pytest.mark.parametrize('layout', ['dict', 'nested', 'flat'])
def test_hash_seed_does_not_change_model_or_gradients(
    hash_seed_snapshots: dict[str, list[dict]],
    layout: str,
) -> None:
    """Compare weights, predictions, and gradients across interpreters."""
    snapshots = hash_seed_snapshots[layout]
    reference = snapshots[0]
    for actual in snapshots[1:]:
        for section in ('weights', 'predictions', 'gradients'):
            assert list(actual[section]) == list(reference[section])
            for name, value in actual[section].items():
                assert torch.isfinite(value).all(), (section, name)
                assert torch.equal(value, reference[section][name]), (
                    section,
                    name,
                )
        assert torch.equal(actual['loss'], reference['loss'])


@pytest.mark.unit
def test_seed_still_controls_initial_weights(tmp_path: Path) -> None:
    """Different tensor seeds must still select different initial weights."""
    cfg = _config(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = MeanEmbeddingForecastLSTM(cfg).state_dict()
        torch.manual_seed(2)
        second = MeanEmbeddingForecastLSTM(cfg).state_dict()
    assert list(first) == list(second)
    assert any(not torch.equal(first[name], second[name]) for name in first)


@pytest.mark.unit
def test_checkpoint_key_order_does_not_change_loading(tmp_path: Path) -> None:
    """Older checkpoints remain loadable by name, irrespective of key order."""
    cfg = _config(tmp_path)
    with torch.random.fork_rng(devices=[]):
        original = MeanEmbeddingForecastLSTM(cfg)
        restored = MeanEmbeddingForecastLSTM(cfg)
    checkpoint = tmp_path / 'reordered.pt'
    torch.save(
        OrderedDict(reversed(list(original.state_dict().items()))), checkpoint
    )
    restored.load_state_dict(
        torch.load(checkpoint, weights_only=True), strict=True
    )
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


@pytest.mark.unit
def test_all_shared_groups_remain_supported(tmp_path: Path) -> None:
    """Keep the existing shared-only construction and ordering."""
    cfg = _config(tmp_path)
    options = cfg.as_dict()
    options['forecast_inputs'] = options['hindcast_inputs']
    with torch.random.fork_rng(devices=[]):
        model = MeanEmbeddingForecastLSTM(Config(options))
    assert not model.hindcast_embeddings_fc
    assert not model.forecast_embeddings_fc
    assert list(model.shared_embeddings_fc) == list(options['hindcast_inputs'])


@pytest.mark.unit
@pytest.mark.parametrize('use_iterator', [False, True])
def test_product_features_keep_first_seen_order(*, use_iterator: bool) -> None:
    """Guard the already order-preserving feature parser mentioned in #305."""
    features = [
        'hres_temp',
        'era5land_rain',
        'hres_rain',
        'cpc_rain',
        'hres_temp',
    ]
    result = _get_products_and_bands_from_feature_strings(
        iter(features) if use_iterator else features,
    )
    assert list(result.items()) == [
        ('HRES', ['hres_temp', 'hres_rain']),
        ('ERA5_LAND', ['era5land_rain']),
        ('CPC', ['cpc_rain']),
    ]
