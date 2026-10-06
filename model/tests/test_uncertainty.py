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

"""Integration tests that perform full runs on the uncertainty estimation code."""

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.evaluation.evaluate import start_evaluation
from model.tests import Fixture
from model.tests.test_config_runs import (
    get_basin_results,
    get_test_start_end_dates,
)
from model.training.train import start_training
from model.utils.config import Config


@pytest.fixture(scope='module')
def trained_uncertainty_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str, str, dict], Path]:
    """Train a model once per (forecast_model, head) pair and cache its run_dir.

    ``negative_sample_handling`` does not affect training weights or scaler
    computation; it only governs evaluation-time sampling.
    """
    cache: dict[tuple[str, str], Path] = {}
    config_file = Path(__file__).parent / 'test_configs' / 'forecast.test.yml'

    def _get_trained_run(
        forecast_model: str, head: str, model_updates: dict
    ) -> Path:
        key = (forecast_model, head)
        if key not in cache:
            run_root = tmp_path_factory.mktemp(
                f'uncertainty_{forecast_model}_{head}'
            )
            config = Config(config_file)
            config.run_dir = run_root
            updates = {
                'model': forecast_model,
                'head': head,
                'n_samples': 10,
                'negative_sample_max_retries': 1,
                'loss': 'CMALLoss',
                'n_distributions': 3,
            }
            updates.update(model_updates)
            config.update_config(updates)
            start_training(config)
            cache[key] = config.run_dir
        return cache[key]

    return _get_trained_run


@pytest.mark.parametrize(
    'negative_sample_handling', ['none', 'clip', 'truncate']
)
@pytest.mark.parametrize('head', ['cmal', 'cmal_deterministic'])
@pytest.mark.parametrize('forecast_model', ['mean_embedding_forecast_lstm'])
def test_daily_uncertainty(
    tmp_path: Fixture[Path],
    trained_uncertainty_runs: Callable[[str, str, dict], Path],
    forecast_config_updates: Fixture[Callable[[str], dict]],
    forecast_model: str,
    head: str,
    negative_sample_handling: str,
):
    """Test probabilistic output consistency across different heads and negative sample handling modes.

    This test verifies that training and evaluation produce valid uncertainty outputs
    for CMAL heads under various negative sample handling strategies
    ('none', 'clip', 'truncate').
    """
    base_run_dir = trained_uncertainty_runs(
        forecast_model, head, forecast_config_updates(forecast_model)
    )
    run_dir = tmp_path / base_run_dir.name
    shutil.copytree(base_run_dir, run_dir)

    config = Config(run_dir / 'config.yml')
    config.update_config(
        {
            'run_dir': run_dir,
            'negative_sample_handling': negative_sample_handling,
            'n_samples': 10,
            'negative_sample_max_retries': 1,
        }
    )

    basin = 'hysets_01075000'

    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')
    _check_uncertainty_output(config, basin, negative_sample_handling)


def _check_uncertainty_output(
    config: Config, basin: str, negative_sample_handling: str
):
    """Perform sanity checks on uncertainty prediction outputs for a given basin.

    This function verifies that:
        - The results file contains the expected simulated target variable.
        - The simulated results have a 'samples' dimension with the correct number of samples.
        - The simulated results fully cover the configured test date range.
        - No NaN or infinite values are present in the simulated samples.
        - The 'samples' dimension exists and has no NaN entries.
        - If negative sample handling was set to 'clip', all negative values are within floating-point tolerance of zero.
        - negative sample handling = 'truncate' testing is not yet implemented

    Parameters
    ----------
    config : Config
        The configuration object used for model training and evaluation.
    basin : str
        The ID of the basin for which predictions are being checked.
    negative_sample_handling : str
        Strategy used to handle negative samples during training ("truncate" or "clip"),
        which determines whether non-negativity is strictly enforced in the output.
    """
    results = (
        get_basin_results(config.run_dir, 1).sel(basin=basin).isel(time_step=-1)
    )

    sample_key = f'{config.target_variables[0]}_sim'
    assert sample_key in results.data_vars, (
        f'Expected {sample_key} in results, got {results.data_vars}'
    )
    # The model evaluation should produce a 'samples' dimension (probabilistic output)
    assert 'samples' in results[sample_key].dims, (
        f'"samples" dimension not found in {sample_key}'
    )

    # Assert the number of samples in the output matches the config
    assert results[sample_key].sizes['samples'] == config.n_samples, (
        f'Expected {config.n_samples} samples, got {results[sample_key].sizes["samples"]}'
    )

    # Check that the results file has the correct date range
    test_start_date, test_end_date = get_test_start_end_dates(config)
    assert pd.to_datetime(results['date'].values[0]) == test_start_date.floor(
        'D'
    )
    assert pd.to_datetime(results['date'].values[-1]) == test_end_date.floor(
        'D'
    )

    # Check that no NaN values are present in the generated samples
    test_dates = pd.date_range(test_start_date, test_end_date, freq='D')
    test_vals = results.sel(date=test_dates)
    # Assert all sample values are finite
    assert np.isfinite(test_vals[sample_key].values).all(), (
        f'Found non-finite values in {sample_key}'
    )

    negative_vals = test_vals[sample_key].values[
        test_vals[sample_key].values < 0
    ]
    if negative_sample_handling == 'clip':
        # For 'clip', we expect all non-negative values
        min_val = np.min(negative_vals) if len(negative_vals) > 0 else 0.0
        assert np.allclose(negative_vals, 0.0, atol=1e-6), (
            f'Found negative samples below tolerance. Smallest val: {min_val}'
        )
    elif negative_sample_handling == 'truncate':
        # TODO: Implement a more robust check for 'truncate' handling
        # where resampling is done to ensure non-negativity
        pass
