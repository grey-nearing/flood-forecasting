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

import gc
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import torch._dynamo
import xarray as xr

from model.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from model.utils.config import Config
from model.utils.configutils import group_features_list
from model.tests import Fixture
from model.tests.test_hot_start import get_base_cfg

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    """Seed Python, NumPy, and PyTorch RNGs before each test."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Move imported modules/types to the permanent GC generation.

    ``Multimet.__init__`` calls ``memory.release()`` (``gc.collect()``) four
    times per dataset initialization. Freezing permanent import-time objects
    avoids rescanning hundreds of thousands of library objects on every call
    while still collecting all objects allocated during tests.
    """
    del session
    gc.collect()
    gc.freeze()


def _cleanup_all_open_resources() -> None:
    """Close all logging FileHandlers and open matplotlib figures."""
    for handler in list(logging.root.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.root.removeHandler(handler)
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger) and logger.handlers:
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
    if plt.get_fignums():
        plt.close('all')


@pytest.fixture(autouse=True)
def cleanup_resources_after_test() -> None:
    """Close all logging FileHandlers and open figures after each test."""
    yield
    _cleanup_all_open_resources()


class SyntheticDatasetDir(type(Path())):
    """Path to a synthetic Multimet dataset directory with dict/attribute access."""

    def _init_metadata(
        self,
        basins: list[str],
        basin_file: Path,
        dates_config: dict[str, str],
    ) -> 'SyntheticDatasetDir':
        self.basins = basins
        self.data_dir = Path(self)
        self.statics_data_dir = Path(self)
        self.dynamics_data_dir = Path(self)
        self.targets_data_dir = Path(self)
        self.basin_file = basin_file
        self.train_basin_file = basin_file
        self.validation_basin_file = basin_file
        self.test_basin_file = basin_file
        self._meta: dict[str, Any] = {
            'data_dir': Path(self),
            'statics_data_dir': Path(self),
            'dynamics_data_dir': Path(self),
            'targets_data_dir': Path(self),
            'basin_file': basin_file,
            'train_basin_file': basin_file,
            'validation_basin_file': basin_file,
            'test_basin_file': basin_file,
            'basins': basins,
            **dates_config,
        }
        return self

    def __getitem__(self, key: str) -> Any:
        return self._meta[key]

    def keys(self):
        return self._meta.keys()


def _build_synthetic_multimet_dataset(
    root: Path, basins: list[str]
) -> SyntheticDatasetDir:
    """Write a deterministic synthetic Caravan-Multimet Zarr dataset to disk."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    dates = pd.date_range('2020-01-01', '2020-02-29', freq='D')
    lead_times = pd.to_timedelta(np.arange(1, 8), unit='D')
    n_basins = len(basins)
    n_dates = len(dates)
    n_leads = len(lead_times)

    day_idx = np.arange(n_dates, dtype=np.float32)[None, :]
    basin_idx = np.arange(n_basins, dtype=np.float32)[:, None]
    lead_idx = np.arange(1, n_leads + 1, dtype=np.float32)[None, None, :]

    # Static attributes (float32)
    area = np.array(
        [100.0 * (i + 1) for i in range(n_basins)], dtype=np.float32
    )
    p_mean = np.array(
        [2.5 + 0.5 * i for i in range(n_basins)], dtype=np.float32
    )
    ds_statics = xr.Dataset(
        {
            'area': (('basin',), area),
            'p_mean': (('basin',), p_mean),
        },
        coords={'basin': basins},
    )
    ds_statics.to_zarr(root / 'attributes.zarr', mode='w')

    # 2D Hindcast product: ERA5_LAND
    era5_precip = (
        2.0
        + 0.3 * basin_idx
        + np.sin(2.0 * np.pi * (day_idx + 3.0 * basin_idx) / 14.0)
        + 0.1 * rng.standard_normal((n_basins, n_dates)).astype(np.float32)
    ).astype(np.float32)
    ds_era5 = xr.Dataset(
        {
            'era5land_total_precipitation': (('basin', 'date'), era5_precip),
        },
        coords={'basin': basins, 'date': dates},
    )
    (root / 'ERA5_LAND').mkdir(parents=True, exist_ok=True)
    ds_era5.to_zarr(root / 'ERA5_LAND' / 'timeseries.zarr', mode='w')

    # 3D Forecast product: GRAPHCAST
    graphcast_precip = (
        2.0
        + 0.25 * basin_idx[:, :, None]
        + 0.05 * lead_idx
        + np.cos(
            2.0
            * np.pi
            * (day_idx[:, :, None] + lead_idx + 2.0 * basin_idx[:, :, None])
            / 14.0
        )
        + 0.1
        * rng.standard_normal((n_basins, n_dates, n_leads)).astype(np.float32)
    ).astype(np.float32)
    ds_graphcast = xr.Dataset(
        {
            'graphcast_total_precipitation': (
                ('basin', 'date', 'lead_time'),
                graphcast_precip,
            ),
        },
        coords={'basin': basins, 'date': dates, 'lead_time': lead_times},
    )
    (root / 'GRAPHCAST').mkdir(parents=True, exist_ok=True)
    ds_graphcast.to_zarr(root / 'GRAPHCAST' / 'timeseries.zarr', mode='w')

    # 2D Target timeseries: streamflow
    streamflow = (
        1.0
        + 0.4 * basin_idx
        + 0.5 * era5_precip
        + 0.25 * np.roll(era5_precip, 1, axis=1)
        + 0.05 * rng.standard_normal((n_basins, n_dates)).astype(np.float32)
    ).astype(np.float32)
    ds_targets = xr.Dataset(
        {
            'streamflow': (('basin', 'date'), streamflow),
        },
        coords={'basin': basins, 'date': dates},
    )
    ds_targets.to_zarr(root / 'streamflow.zarr', mode='w')

    basin_file = root / 'basins.txt'
    basin_file.write_text('\n'.join(basins) + '\n', encoding='utf-8')

    dates_config = {
        'train_start_date': '15/01/2020',
        'train_end_date': '24/01/2020',
        'validation_start_date': '25/01/2020',
        'validation_end_date': '03/02/2020',
        'test_start_date': '04/02/2020',
        'test_end_date': '13/02/2020',
    }
    return SyntheticDatasetDir(root)._init_metadata(
        basins=basins, basin_file=basin_file, dates_config=dates_config
    )


@pytest.fixture(scope='session')
def single_basin_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticDatasetDir:
    """Session-scoped 1-basin synthetic Multimet dataset (`basin_01`)."""
    root = tmp_path_factory.mktemp('single_basin_dataset')
    return _build_synthetic_multimet_dataset(root, ['basin_01'])


@pytest.fixture(scope='session')
def five_basin_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticDatasetDir:
    """Session-scoped 5-basin synthetic Multimet dataset (`basin_01`..`basin_05`)."""
    root = tmp_path_factory.mktemp('five_basin_dataset')
    basins = [f'basin_0{i}' for i in range(1, 6)]
    return _build_synthetic_multimet_dataset(root, basins)


@pytest.fixture
def make_minimal_config(
    tmp_path: Path, five_basin_dataset: SyntheticDatasetDir
) -> Callable[..., Config]:
    """Factory fixture returning a real Config instance with sensible defaults."""
    counter = 0

    def _make_config(
        overrides: dict[str, Any] | None = None, **kwargs: Any
    ) -> Config:
        nonlocal counter
        counter += 1
        run_dir = tmp_path / f'run_{counter}'
        img_log_dir = run_dir / 'img_log'
        run_dir.mkdir(parents=True, exist_ok=True)
        img_log_dir.mkdir(parents=True, exist_ok=True)

        embedding = {
            'type': 'fc',
            'hiddens': [8],
            'activation': ['tanh'],
            'dropout': 0.0,
        }
        base_dict: dict[str, Any] = {
            'experiment_name': 'test_experiment',
            'run_dir': run_dir,
            'base_run_dir': run_dir,
            'img_log_dir': img_log_dir,
            'dataset': 'multimet',
            'data_dir': Path(five_basin_dataset),
            'train_basin_file': five_basin_dataset.train_basin_file,
            'validation_basin_file': five_basin_dataset.validation_basin_file,
            'test_basin_file': five_basin_dataset.test_basin_file,
            'train_start_date': five_basin_dataset['train_start_date'],
            'train_end_date': five_basin_dataset['train_end_date'],
            'validation_start_date': five_basin_dataset[
                'validation_start_date'
            ],
            'validation_end_date': five_basin_dataset['validation_end_date'],
            'test_start_date': five_basin_dataset['test_start_date'],
            'test_end_date': five_basin_dataset['test_end_date'],
            'model': 'mean_embedding_forecast_lstm',
            'head': 'regression',
            'output_activation': 'linear',
            'output_dropout': 0.0,
            'hidden_size': 8,
            'seq_length': 10,
            'lead_time': 2,
            'forecast_overlap': 10,
            'predict_last_n': 2,
            'timestep_counter': False,
            'static_attributes': ['area', 'p_mean'],
            'hindcast_inputs': [
                'era5land_total_precipitation',
                'graphcast_total_precipitation',
            ],
            'forecast_inputs': ['graphcast_total_precipitation'],
            'target_variables': ['streamflow'],
            'statics_embedding': embedding.copy(),
            'hindcast_embedding': embedding.copy(),
            'forecast_embedding': embedding.copy(),
            'dynamics_embedding': embedding.copy(),
            'state_handoff_network': embedding.copy(),
            'optimizer': 'Adam',
            'loss': 'MSE',
            'initial_learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 1,
            'n_distributions': 3,
            'n_samples': 5,
            'negative_sample_handling': 'none',
            'negative_sample_max_retries': 1,
            'device': 'cpu',
            'seed': 42,
            'compile': False,
            'num_workers': 0,
            'verbose': 0,
            'log_interval': 1,
            'log_tensorboard': False,
            'log_n_figures': 0,
            'save_git_diff': False,
            'save_weights_every': 1,
            'save_validation_results': False,
            'validate_every': 0,
            'validate_n_random_basins': 0,
            'metrics': ['NSE', 'KGE'],
            'cache': {'enabled': False},
        }
        if overrides:
            base_dict.update(overrides)
        if kwargs:
            base_dict.update(kwargs)
        return Config(base_dict)

    return _make_config


@pytest.fixture
def minimal_config(make_minimal_config: Callable[..., Config]) -> Config:
    """Return a real Config instance initialized from a minimal valid dictionary."""
    return make_minimal_config()


def pytest_addoption(parser):
    parser.addoption(
        '--smoke-test',
        action='store_true',
        default=False,
        help=(
            'Skips some tests for faster execution. Out of single-timescale '
            'models/forcings, only test cudalstm on daymet.'
        ),
    )


@pytest.fixture
def get_config(tmp_path: Fixture[Path]) -> Fixture[Callable[[str], dict]]:
    """Provides a function to fetch a run config specified by name.

    The fetched run configuration will use a tmp folder as its run directory.

    Parameters
    ----------
    tmp_path : Fixture[Path]
        Tmp directory to use in the run configuration.

    Returns
    -------
    Fixture[Callable[[str], dict]]
        Function that returns a run configuration.
    """
    repo_root = Path(__file__).resolve().parents[2]

    def _get_config(name):
        config_file = (
            Path(__file__).parent / 'test_configs' / f'{name}.test.yml'
        )
        if not config_file.is_file():
            raise ValueError(f'Test config file not found at {config_file}.')
        config = Config(config_file)
        for key, val in list(config.as_dict().items()):
            if isinstance(val, Path) and not val.is_absolute():
                resolved = (repo_root / val).resolve()
                if resolved.exists():
                    config.as_dict()[key] = resolved
        config.run_dir = tmp_path
        return config

    return _get_config


@pytest.fixture
def forecast_config_updates() -> Fixture[Callable[[str], dict]]:
    """Provides a function to update forecast model configs.

    Returns
    -------
    Fixture[Callable[[str], dict]]
        Function that returns an update dict.
    """

    def _forecast_config_updates(forecast_model):
        update_dict = {'model': forecast_model.lower()}
        if forecast_model.lower() == 'handoff_forecast_lstm':
            update_dict['forecast_overlap'] = 10
            update_dict['regularization'] = ['forecast_overlap']
        if forecast_model.lower() == 'mean_embedding_forecast_lstm':
            update_dict['forecast_overlap'] = 30
            update_dict['hindcast_inputs'] = [
                'era5land_total_precipitation',
                'graphcast_total_precipitation',
            ]
        return update_dict

    return _forecast_config_updates


@pytest.fixture(
    params=['handoff_forecast_lstm', 'mean_embedding_forecast_lstm']
)
def forecast_model(request) -> str:
    """Fixture that provides single-timescale forecast models.

    Returns
    -------
    str
        Name of the single-timescale model.
    """
    if (
        request.config.getoption('--smoke-test')
        and request.param != 'handoff_forecast_lstm'
    ):
        pytest.skip('--smoke-test skips this test.')
    return request.param


@pytest.fixture(params=[True, False])
def lazy_load(request) -> bool:
    return request.param


@pytest.fixture(
    params=[
        ('daymet', ['prcp(mm/day)', 'tmax(C)']),
        ('nldas', ['PRCP(mm/day)', 'Tmax(C)']),
        ('maurer', ['PRCP(mm/day)', 'Tmax(C)']),
        ('maurer_extended', ['prcp(mm/day)', 'tmax(C)']),
        (
            ['daymet', 'nldas'],
            [
                'prcp(mm/day)_daymet',
                'tmax(C)_daymet',
                'PRCP(mm/day)_nldas',
                'Tmax(C)_nldas',
            ],
        ),
    ],
    ids=lambda param: str(param[0]),
)
def single_timescale_forcings(request) -> dict[str, str | list[str]]:
    """Fixture that provides daily forcings.

    Returns
    -------
    dict[str, str | list[str]]
        Dict ``{'forcings': <name>, 'variables': <list of variables>}``.
    """
    if (
        request.config.getoption('--smoke-test')
        and 'daymet' not in request.param[0]
    ):
        pytest.skip('--smoke-test skips this test.')
    return {'forcings': request.param[0], 'variables': request.param[1]}


@pytest.fixture(
    params=[('camels_us', ['QObs(mm/d)'])], ids=lambda param: param[0]
)
def daily_dataset(request) -> dict[str, list[str]]:
    """Fixture that provides daily datasets.

    Returns
    -------
    dict[str, list[str]]
        Dict ``{'dataset: <name>, 'target': <list of target variables>}``.
    """
    if (
        request.config.getoption('--smoke-test')
        and request.param[0] != 'camels_us'
    ):
        pytest.skip('--smoke-test skips this test.')
    return {'dataset': request.param[0], 'target': request.param[1]}


@pytest.fixture
def tiny_mean_embedding_model(
    tmp_path: Fixture[Path],
) -> Fixture[Callable[..., MeanEmbeddingForecastLSTM]]:
    """Return a factory for a tiny, deterministic MeanEmbeddingForecastLSTM.

    The default inputs give one shared group (``pr``, ``tmmn``), one
    hindcast-only group (``streamflow``) and one forecast-only group
    (``hres``). The model is seeded (without touching the global RNG state)
    and in eval mode.

    Parameters
    ----------
    tmp_path : Fixture[Path]
        Tmp directory used as run directory (a dummy scaler is written there).

    Returns
    -------
    Fixture[Callable[..., MeanEmbeddingForecastLSTM]]
        Factory ``(seq_length=6, lead_time=3, hindcast_inputs=...,
        forecast_inputs=..., head='regression', n_distributions=3)
        -> MeanEmbeddingForecastLSTM``. ``n_distributions`` is only used by
        the ``'cmal'`` head.
    """

    def _build(  # noqa: PLR0913
        *,
        seq_length: int = 6,
        lead_time: int = 3,
        hindcast_inputs: tuple[str, ...] = ('pr_a', 'tmmn_a', 'streamflow_lag'),
        forecast_inputs: tuple[str, ...] = ('pr_a', 'tmmn_a', 'hres_precip'),
        head: str = 'regression',
        n_distributions: int = 3,
    ) -> MeanEmbeddingForecastLSTM:
        options = get_base_cfg(tmp_path)
        embedding = {
            'type': 'fc',
            'hiddens': [8],
            'activation': ['tanh'],
            'dropout': 0.0,
        }
        options.update(
            {
                'model': 'MeanEmbeddingForecastLSTM',
                'seq_length': seq_length,
                'lead_time': lead_time,
                'forecast_overlap': seq_length,
                'hidden_size': 8,
                'output_dropout': 0.0,
                'head': head,
                'n_distributions': n_distributions,
                'hindcast_inputs': list(hindcast_inputs),
                'forecast_inputs': list(forecast_inputs),
                'hindcast_embedding': embedding,
                'forecast_embedding': embedding,
                'statics_embedding': {**embedding, 'hiddens': [4]},
            }
        )
        cfg = Config(options)
        xr.Dataset(
            {'streamflow': ('parameter', [0.0, 1.0, 0.0, 1.0])},
            coords={'parameter': ['center', 'scale', 'mean', 'std']},
        ).to_zarr(tmp_path / 'scaler.zarr', mode='w')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = MeanEmbeddingForecastLSTM(cfg)
        model.eval()
        return model

    return _build


@pytest.fixture
def tiny_mean_embedding_data() -> Fixture[Callable[..., dict]]:
    """Return a factory for random inputs of a tiny MeanEmbeddingForecastLSTM.

    Returns
    -------
    Fixture[Callable[..., dict]]
        Factory ``(cfg, batch_size=2, nan_at=None) -> data`` where hindcast
        inputs cover ``cfg.seq_length`` steps and forecast inputs the full
        ``cfg.seq_length + cfg.lead_time`` span. Values are seeded from a
        private generator. ``nan_at`` maps an input group (e.g. ``'pr'``) or
        a single feature (e.g. ``'pr_a'``) to the time indices that are set
        to NaN in every matching hindcast and forecast feature; indices
        beyond a feature's length are ignored.
    """

    def _make(
        cfg: Config,
        *,
        batch_size: int = 2,
        nan_at: dict[str, list[int]] | None = None,
    ) -> dict:
        generator = torch.Generator().manual_seed(1)
        data = {
            'x_d_hindcast': {
                name: torch.rand(
                    batch_size, cfg.seq_length, 1, generator=generator
                )
                for name in cfg.hindcast_inputs
            },
            'x_d_forecast': {
                name: torch.rand(
                    batch_size,
                    cfg.seq_length + cfg.lead_time,
                    1,
                    generator=generator,
                )
                for name in cfg.forecast_inputs
            },
            'x_s': torch.rand(
                batch_size, len(cfg.static_attributes), generator=generator
            ),
        }
        for key, steps in (nan_at or {}).items():
            matched = False
            for inputs, features in (
                (cfg.hindcast_inputs, data['x_d_hindcast']),
                (cfg.forecast_inputs, data['x_d_forecast']),
            ):
                names = group_features_list(inputs).get(key, [])
                names = names or [name for name in inputs if name == key]
                for name in names:
                    matched = True
                    tensor = features[name]
                    for step in steps:
                        if step < tensor.shape[1]:
                            tensor[:, step] = float('nan')
            if not matched:
                msg = f'nan_at: {key!r} is neither an input group nor feature'
                raise ValueError(msg)
        return data

    return _make


def assert_finite_grads(
    tensors_or_module: torch.nn.Module | Iterable[torch.Tensor],
) -> None:
    """Assert every (populated) gradient is finite and at least one exists."""
    if isinstance(tensors_or_module, torch.nn.Module):
        tensors = list(tensors_or_module.parameters())
    else:
        tensors = list(tensors_or_module)
    grads = [t.grad for t in tensors if t.grad is not None]
    assert grads, 'no gradients were populated'
    for grad in grads:
        assert torch.isfinite(grad).all()


def assert_grad_matches_finite_difference(
    loss_fn: Callable[[], torch.Tensor],
    tensor: torch.Tensor,
    index: tuple[int, ...],
    eps: float = 1e-6,
    atol: float = 1e-5,
) -> None:
    """Assert ``tensor.grad[index]`` equals a central finite difference.

    ``tensor.grad`` must already be populated by a backward pass of the same
    loss. Use double precision for the default tolerances.
    """
    assert tensor.grad is not None, 'tensor has no gradient'
    with torch.no_grad():
        original = tensor[index].item()
        tensor[index] = original + eps
        plus = loss_fn().item()
        tensor[index] = original - eps
        minus = loss_fn().item()
        tensor[index] = original
    expected = (plus - minus) / (2 * eps)
    assert tensor.grad[index].item() == pytest.approx(expected, abs=atol)
