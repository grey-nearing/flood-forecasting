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

"""Unit tests for return_periods.forecast_evaluation."""

import numpy as np
import pandas as pd
import pytest
from return_periods.forecast_evaluation import (
    compute_precision_recall_f1_single_rp,
    compute_series_return_period_thresholds,
    evaluate_forecast_return_periods,
    get_up_crossing_indices,
)


def test_get_up_crossing_indices():
    series = np.array([1.0, 5.0, 6.0, 2.0, 5.5, np.nan, 7.0])
    crossings = get_up_crossing_indices(series, 5.0)
    np.testing.assert_array_equal(crossings, np.array([1, 4, 6]))


def test_compute_precision_recall_f1_single_rp():
    obs = np.zeros(20)
    sim = np.zeros(20)
    obs[5] = 10.0
    obs[15] = 10.0
    # Event 1 detected within +1 day; Event 2 missed
    sim[6] = 8.0
    p, r, f1, n_obs, n_sim = compute_precision_recall_f1_single_rp(
        obs, sim, obs_thresh=5.0, sim_thresh=5.0, window_days=2
    )
    assert n_obs == 2
    assert n_sim == 1
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(2.0 / 3.0)


def test_evaluate_forecast_return_periods():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2011-12-31', freq='1D')
    obs_1d = rng.lognormal(mean=1.0, sigma=0.8, size=len(dates))
    sim_1d = obs_1d * rng.uniform(0.9, 1.1, size=len(dates))
    obs_2d = np.stack([obs_1d, obs_1d], axis=1)
    sim_2d = np.stack([sim_1d, sim_1d], axis=1)

    thresholds = compute_series_return_period_thresholds(obs_1d, dates)
    assert all(np.isfinite(v) for v in thresholds.values())

    metrics = evaluate_forecast_return_periods(
        obs_2d,
        sim_2d,
        dates,
        obs_thresholds_gema=thresholds,
        lead_indices=[0, 1],
        windows=[0, 2],
        compute_lp3_lead0=True,
    )
    assert 'RP_ExpRecallNorm' in metrics
    assert 'RP_ExpF1Norm_lead1' in metrics
    assert 'RP_ExpF1Norm_lead2' in metrics
    assert 0.0 <= metrics['RP_ExpRecallNorm'] <= 1.0
