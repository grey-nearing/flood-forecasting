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

"""Regression tests for the basin index space used during evaluation.

`BaseTester` keeps two basin lists that are easy to confuse:

* `self.basins` -- the basins we want metrics for, after dropping any whose
  observations are entirely NaN over the evaluation period.
* `dataset.loaded_basins` -- the basins the dataset actually materialized,
  which is the axis `_sample_index` numbers its basin column against.

Every sample carries a *positional* basin code into the second list. Resolving
those codes against the first list silently shifts every basin after the first
exclusion, which drops real basins from the results with no warning. These
tests pin the two lists apart and assert the results are still right.
"""

import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from googlehydrology.datasetzoo.caravan import load_caravan_timeseries_together
from googlehydrology.evaluation.evaluate import start_evaluation
from googlehydrology.training.train import start_training
from googlehydrology.utils.config import Config

# The NaN'd basin is deliberately *first* alphabetically so that excluding it
# shifts the position of every other basin. `load_basin_file` sorts, so this
# ordering is what the tester sees regardless of the file's line order.
TEST_BASINS = [
    'camels_07057500',  # excluded: streamflow is NaN'd below
    'camels_04115265',
    'camels_04216418',
    'camels_12115000',
    'camels_12377150',
    'camels_12451000',
    'camels_13235000',
    'camels_14236200',
]
NAN_BASIN = TEST_BASINS[0]
EXPECTED_EVALUATED = sorted(set(TEST_BASINS) - {NAN_BASIN})

# Training basins are disjoint from the NaN'd basin so training is unaffected.
TRAIN_BASINS = [
    'camels_12377150',
    'camels_12115000',
    'camels_12451000',
    'camels_04216418',
    'camels_14236200',
]


@pytest.fixture(scope='module')
def nan_basin_env(tmp_path_factory: pytest.TempPathFactory):
    """A tutorial-data environment with one basin's streamflow all NaN."""
    tmp_dir = tmp_path_factory.mktemp('tester_basin_index')
    base_path = Path(__file__).parent.parent

    # Copy rather than mutate the checked-in tutorial data.
    nc_dir = tmp_dir / 'Caravan-nc'
    shutil.copytree(base_path / 'tutorial' / 'Caravan-nc', nc_dir)

    target = nc_dir / 'timeseries' / 'netcdf' / 'camels' / f'{NAN_BASIN}.nc'
    with xr.open_dataset(target) as ds:
        ds = ds.load()
    ds['streamflow'] = xr.full_like(ds['streamflow'], np.nan)
    target.unlink()
    ds.to_netcdf(target)
    ds.close()

    raw_features = ['total_precipitation_sum', 'temperature_2m_mean']
    dynamics = load_caravan_timeseries_together(
        nc_dir, basins=TEST_BASINS, target_features=raw_features, csv=False
    )
    dynamics = dynamics.rename({
        'total_precipitation_sum': 'era5land_total_precipitation',
        'temperature_2m_mean': 'era5land_temperature_2m',
    })
    lead_times = pd.to_timedelta(np.arange(8), unit='D')
    forecast = dynamics.expand_dims(lead_time=lead_times).copy()

    dynamics_dir = tmp_dir / 'dynamics'
    zarr_path = dynamics_dir / 'ERA5_LAND' / 'timeseries.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_zarr(zarr_path, consolidated=True)
    dynamics.close()
    forecast.close()

    train_file = tmp_dir / 'train_basins.txt'
    test_file = tmp_dir / 'test_basins.txt'
    train_file.write_text('\n'.join(TRAIN_BASINS) + '\n')
    test_file.write_text('\n'.join(TEST_BASINS) + '\n')

    yield {
        'tmp_dir': tmp_dir,
        'nc_dir': str(nc_dir.resolve()),
        'dynamics_dir': str(dynamics_dir.resolve()),
        'train_basin_file': str(train_file.resolve()),
        'test_basin_file': str(test_file.resolve()),
    }

    try:
        from googlehydrology.datasetzoo.multimet import _open_zarr

        _open_zarr.cache_clear()
    except Exception:
        pass
    gc.collect()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _config(env: dict, run_dir: Path) -> Config:
    return Config({
        'experiment_name': 'basin_index',
        'run_dir': str(run_dir),
        'dataset': 'multimet',
        'train_basin_file': env['train_basin_file'],
        'validation_basin_file': env['train_basin_file'],
        'test_basin_file': env['test_basin_file'],
        'targets_data_dir': env['nc_dir'],
        'statics_data_dir': env['nc_dir'],
        'dynamics_data_dir': env['dynamics_dir'],
        'train_start_date': '01/01/2000',
        'train_end_date': '01/03/2000',
        'validation_start_date': '01/01/2001',
        'validation_end_date': '31/03/2001',
        'test_start_date': '01/01/2001',
        'test_end_date': '31/03/2001',
        'hindcast_inputs': {
            'era5land': [
                'era5land_total_precipitation',
                'era5land_temperature_2m',
            ]
        },
        'forecast_inputs': {
            'era5land': [
                'era5land_total_precipitation',
                'era5land_temperature_2m',
            ]
        },
        'static_attributes': ['area', 'p_mean'],
        'target_variables': ['streamflow'],
        'model': 'mean_embedding_forecast_lstm',
        'hidden_size': 16,
        'head': 'regression',
        'output_activation': 'linear',
        'statics_embedding': {
            'type': 'fc',
            'hiddens': [32, 16],
            'activation': ['tanh', 'linear'],
            'dropout': 0.0,
        },
        'hindcast_embedding': {
            'type': 'fc',
            'hiddens': [32, 16],
            'activation': ['tanh', 'linear'],
            'dropout': 0.0,
        },
        'forecast_embedding': {
            'type': 'fc',
            'hiddens': [32, 16],
            'activation': ['tanh', 'linear'],
            'dropout': 0.0,
        },
        'seq_length': 30,
        'lead_time': 7,
        'forecast_overlap': 10,
        'timestep_counter': True,
        'output_dropout': 0.0,
        'compile': False,
        'device': 'cpu',
        'seed': 42,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'epochs': 1,
        'save_weights_every': 1,
        'batch_size': 32,
        'initial_learning_rate': 0.001,
        'metrics': ['NSE'],
        'predict_last_n': 8,
        'num_workers': 0,
        'validate_every': None,
        'validate_n_random_basins': -1,
        'cache': {'enabled': False},
        'tester_skip_obs_all_nan': False,
    })


@pytest.fixture(scope='module')
def trained_run_dir(nan_basin_env) -> Path:
    """Train once; both evaluation tests reuse the weights."""
    runs_root = nan_basin_env['tmp_dir'] / 'runs'
    start_training(_config(nan_basin_env, runs_root))
    # `start_training` rewrites run_dir to a timestamped subdirectory.
    return next(iter(runs_root.glob('*')))


def _evaluate(run_dir: Path, *, skip_obs_all_nan: bool) -> list[str]:
    """Evaluate and return the basins that made it into the metrics file."""
    output_dir = run_dir / 'test'
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Load the run's own config, as `run.py` does: `start_training` moved
    # `run_dir` and wrote the scaler there, so a freshly built Config would
    # point at the wrong directory.
    cfg = Config(run_dir / 'config.yml')
    cfg.tester_skip_obs_all_nan = skip_obs_all_nan

    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=1, period='test')

    metrics = pd.read_csv(output_dir / 'model_epoch001' / 'test_metrics.csv')
    return sorted(metrics['basin'].astype(str))


@pytest.mark.slow
@pytest.mark.integration
def test_all_basins_evaluated_when_no_exclusions(trained_run_dir):
    """Baseline: with no exclusions, every requested basin gets metrics."""
    assert _evaluate(trained_run_dir, skip_obs_all_nan=False) == sorted(
        TEST_BASINS
    )


@pytest.mark.slow
@pytest.mark.integration
def test_exclusion_drops_only_the_excluded_basin(trained_run_dir):
    """Excluding an all-NaN basin must not take any other basin with it.

    `_calc_exclude_basins` removes the NaN'd basin from `self.basins` but not
    from the dataset, so the two lists differ in length by one. If evaluation
    resolves positional basin codes against `self.basins`, every basin after
    the exclusion shifts by one and the last one falls off the end -- it is
    simply never evaluated, and nothing reports that it went missing.
    """
    evaluated = _evaluate(trained_run_dir, skip_obs_all_nan=True)

    assert evaluated == EXPECTED_EVALUATED, (
        f'expected {len(EXPECTED_EVALUATED)} basins, got {len(evaluated)}; '
        f'missing {sorted(set(EXPECTED_EVALUATED) - set(evaluated))}'
    )
    assert NAN_BASIN not in evaluated
