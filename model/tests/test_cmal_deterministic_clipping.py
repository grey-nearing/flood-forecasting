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

from unittest.mock import MagicMock

import pytest
import torch
import xarray as xr

from model.utils import samplingutils


def _model_with_handling(handling):
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.cfg.head = 'cmal_deterministic'
    model.cfg.target_variables = ['streamflow']
    model.cfg.predict_last_n = 2
    model.cfg.negative_sample_handling = handling
    return model


@pytest.mark.unit
@pytest.mark.parametrize(
    'handling, expected_min',
    [('clip', -2.5), ('none', -4.0), (None, -4.0), ('truncate', -4.0)],
)
def test_deterministic_cmal_negative_handling(monkeypatch, handling, expected_min):
    model = _model_with_handling(handling)

    scaler = MagicMock()
    scaler.scaler = {
        'streamflow': xr.DataArray(
            [5.0, 2.0],
            coords={'parameter': ['center', 'scale']},
            dims=['parameter'],
        )
    }

    generated = torch.tensor(
        [[[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [-4.0, -2.5, -1.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]]]
    )
    monkeypatch.setattr(
        samplingutils.cmal_deterministic,
        'generate_predictions',
        lambda *args: generated.clone(),
    )

    outputs = {
        'mu': torch.zeros(1, 2, 3),
        'b': torch.ones(1, 2, 3),
        'tau': torch.full((1, 2, 3), 0.5),
        'pi': torch.full((1, 2, 3), 1.0 / 3),
    }
    samples = samplingutils.sample_pointpredictions(
        model,
        {'y': torch.zeros(1, 2, 1)},
        n_samples=10,
        scaler=scaler,
        outputs=outputs,
    )

    assert samples['y_hat'].shape == (1, 2, 1, 10)
    assert samples['y_hat'].min().item() == expected_min
    if handling == 'clip':
        assert samples['y_hat'][0, 0, 0, 1].item() == -2.0


@pytest.mark.unit
def test_deterministic_cmal_rejects_unknown_negative_handling(monkeypatch):
    model = _model_with_handling('bogus')
    scaler = MagicMock()
    scaler.scaler = {
        'streamflow': xr.DataArray(
            [5.0, 2.0],
            coords={'parameter': ['center', 'scale']},
            dims=['parameter'],
        )
    }
    generated = torch.zeros(1, 2, 10)
    monkeypatch.setattr(
        samplingutils.cmal_deterministic,
        'generate_predictions',
        lambda *args: generated.clone(),
    )
    outputs = {
        'mu': torch.zeros(1, 2, 3),
        'b': torch.ones(1, 2, 3),
        'tau': torch.full((1, 2, 3), 0.5),
        'pi': torch.full((1, 2, 3), 1.0 / 3),
    }

    with pytest.raises(NotImplementedError, match='bogus'):
        samplingutils.sample_pointpredictions(
            model,
            {'y': torch.zeros(1, 2, 1)},
            n_samples=10,
            scaler=scaler,
            outputs=outputs,
        )



def _run_uncertainty_tester_evaluate(
    tmp_path, preds_tensor, obs_tensor, head, reduction
):
    import logging
    import numpy as np

    from model.datasetzoo.multimet import SampleIndexer
    from model.evaluation.tester import UncertaintyTester
    from model.utils.config import TesterSamplesReduction

    tester = object.__new__(UncertaintyTester)
    tester.period = 'test'
    tester.run_dir = tmp_path
    tester.init_model = False
    tester._disable_pbar = True
    tester.basins = ['basin_A']

    tester.cfg = MagicMock()
    tester.cfg.head = head
    tester.cfg.tester_sample_reduction = TesterSamplesReduction(reduction)
    tester.cfg.assimilate = False
    tester.cfg.validate_n_random_basins = 0
    tester.cfg.log_n_figures = 0
    tester.cfg.batch_size = 2
    tester.cfg.lazy_load = False
    tester.cfg.logging_level = logging.INFO
    tester.cfg.inference_mode = False
    tester.cfg.predict_last_n = 2
    tester.cfg.seq_length = 2
    tester.cfg.target_variables = ['streamflow']
    tester.cfg.clip_targets_to_zero = []
    tester.cfg.metrics = []

    dataset = MagicMock()
    dataset.lead_time = 0
    dataset._sample_index = SampleIndexer(
        (
            ('basin', np.array(['basin_A', 'basin_A'])),
            ('date', np.array([0, 1])),
        )
    )
    dataset.scaler.unscale.side_effect = lambda ds: ds
    tester.dataset = dataset

    dates = np.array(
        [['2020-01-01', '2020-01-02'], ['2020-01-02', '2020-01-03']],
        dtype='datetime64[ns]',
    )
    tester._evaluate = lambda model, loader, basins, **kwargs: iter(
        [
            {
                'basin': 'basin_A',
                'preds': preds_tensor,
                'obs': obs_tensor,
                'dates': dates,
                'losses': [{'loss': 0.1}],
                'mean_losses': {'loss': 0.1},
            }
        ]
    )
    tester._ensure_no_previous_results_saved = lambda epoch=None, **kwargs: None
    tester._save_incremental_results = lambda *args, **kwargs: None

    logger = MagicMock()
    model = MagicMock()
    tester.evaluate(
        epoch=1,
        save_results=False,
        metrics=['MSE'],
        model=model,
        experiment_logger=logger,
    )
    mse_calls = [
        call.kwargs['MSE']
        for call in logger.log_step.call_args_list
        if 'MSE' in call.kwargs
    ]
    assert len(mse_calls) == 1
    return mse_calls[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    'reduction, target_sample_idx',
    [('mean', 0), ('median', 5)],
)
def test_cmal_deterministic_evaluate_selects_mean_and_median_indices(
    tmp_path, reduction, target_sample_idx
):
    """BaseTester.evaluate with cmal_deterministic selects index 0 (mean) or 5 (q50)."""
    import numpy as np

    # Generate 10 summary statistics [mean, q10..q90] from an asymmetric CMAL distribution
    # (tau=0.2 -> right-skewed so mixture mean > q60).
    mu = torch.tensor([[[10.0]]], dtype=torch.float32)
    b = torch.tensor([[[2.0]]], dtype=torch.float32)
    tau = torch.tensor([[[0.2]]], dtype=torch.float32)
    pi = torch.tensor([[[1.0]]], dtype=torch.float32)
    summary_stats = samplingutils.cmal_deterministic._mixture_params_to_quantiles(
        mu, b, tau, pi
    )
    mean_val = mu + b * (1 - 2 * tau) / (tau * (1 - tau))
    summary_stats = torch.concat([mean_val, summary_stats], dim=-1).squeeze()

    # Verify asymmetry: index 0 != mean(samples) and index 5 (q50) != median(samples).
    sample_mean = float(np.mean(summary_stats.numpy()))
    sample_median = float(np.median(summary_stats.numpy()))
    assert abs(float(summary_stats[0]) - sample_mean) > 1.0
    assert abs(float(summary_stats[5]) - sample_median) > 0.1

    # Shape: [batch=2, time_step=2, targets=1, samples=10]
    preds = summary_stats.view(1, 1, 1, 10).repeat(2, 2, 1, 1)
    expected_slice = preds[:, :, :, target_sample_idx]

    # Set observations equal to the expected slice (index 0 for mean, index 5 for median).
    mse_det = _run_uncertainty_tester_evaluate(
        tmp_path,
        preds_tensor=preds,
        obs_tensor=expected_slice,
        head='cmal_deterministic',
        reduction=reduction,
    )
    assert mse_det == pytest.approx(0.0, abs=1e-6)

    # Verify that a Monte Carlo head ('cmal') reduces across all 10 samples instead,
    # resulting in non-zero MSE against the single slice and zero MSE against the
    # across-sample reduction.
    mse_mc_against_slice = _run_uncertainty_tester_evaluate(
        tmp_path,
        preds_tensor=preds,
        obs_tensor=expected_slice,
        head='cmal',
        reduction=reduction,
    )
    assert mse_mc_against_slice > 0.01

    mc_reduced_val = sample_mean if reduction == 'mean' else sample_median
    mc_reduced = torch.full_like(expected_slice, mc_reduced_val)
    mse_mc = _run_uncertainty_tester_evaluate(
        tmp_path,
        preds_tensor=preds,
        obs_tensor=mc_reduced,
        head='cmal',
        reduction=reduction,
    )
    assert mse_mc == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_critical_ruff_lint_rules():
    """Enforce zero critical ruff lint errors (F821, E9, B006, B023, PLE) in CI."""
    from pathlib import Path
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'ruff',
            'check',
            '--select',
            'F821,E9,B006,B023,PLE',
            'model',
            'multimet',
            'return_periods',
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f'Critical ruff lint errors found:\n{result.stdout}\n{result.stderr}'
    )
