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

"""Unit tests for `maas.thresholds` return period calculators and exceedance classification."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from maas.thresholds import (
    CANONICAL_RETURN_PERIODS,
    RETURN_PERIOD_YEARS,
    classify_exceedance,
    compute_empirical_weibull_return_periods,
    compute_gumbel_return_periods,
    compute_return_periods,
    estimate_return_period_years,
    ev1_fit_line,
    extract_annual_maxima,
    gumbel_frequency_factor,
    gumbel_quantile_from_return_periods,
    known_return_levels,
    scaled_index_flood_return_periods,
    thresholds_from_return_periods,
)
from return_periods import NotEnoughDataError


class TestReturnPeriodIntegration:
    """Tests direct integration with `return_periods.ReturnPeriodCalculator`."""

    def test_compute_return_periods_gema_monotonic_and_positive(self) -> None:
        rng = np.random.default_rng(42)
        years = pd.date_range('1990-01-01', '2023-12-31', freq='D')
        seasonal = 1.0 + 0.6 * np.sin(2.0 * np.pi * years.dayofyear / 365.25)
        values = (
            500.0
            * seasonal
            * rng.lognormal(mean=0.0, sigma=0.35, size=len(years))
        )
        series = pd.Series(values, index=years)

        rp = compute_return_periods(
            series,
            is_daily_hydrograph=True,
            water_year_start_month=10,
        )
        rp_keys = [f'return_period_{t}' for t in CANONICAL_RETURN_PERIODS]
        assert set(rp_keys).issubset(rp.keys())
        levels = [rp[key] for key in rp_keys]
        assert all(val is not None and val > 0.0 for val in levels)
        for earlier, later in zip(levels, levels[1:]):
            assert float(later) >= float(earlier)
        assert rp['years_of_record'] >= 30
        assert 'USGS Bulletin 17C EMA' in str(rp.get('method', ''))

    def test_compute_return_periods_insufficient_years_raises_not_enough_data(
        self,
    ) -> None:
        peaks = [120.0, 150.0]
        with pytest.raises(NotEnoughDataError):
            compute_return_periods(peaks)

    def test_compute_empirical_weibull_return_periods_matches_plotting_positions(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        dates = pd.date_range('1985-01-01', '2024-12-31', freq='D')
        series = pd.Series(
            rng.gamma(shape=3.0, scale=150.0, size=len(dates)),
            index=dates,
        )
        rp = compute_empirical_weibull_return_periods(
            series,
            is_daily_hydrograph=True,
        )
        rp_keys = [f'return_period_{t}' for t in CANONICAL_RETURN_PERIODS]
        levels = [rp[key] for key in rp_keys]
        assert all(val is not None and val > 0.0 for val in levels)
        for earlier, later in zip(levels, levels[1:]):
            assert float(later) >= float(earlier)
        assert rp['years_of_record'] >= 35
        assert len(rp['empirical_sorted_flows']) == rp['years_of_record']
        assert (
            len(rp['empirical_exceedance_probabilities'])
            == rp['years_of_record']
        )
        assert 'Weibull' in str(rp.get('method', ''))


class TestGumbelAndExceedanceMath:
    """Tests Gumbel EV1 frequency factors, log-Gumbel interpolation, and risk classification."""

    def test_gumbel_frequency_factor_analytical_formula(self) -> None:
        gamma = 0.5772156649015329
        for rp in RETURN_PERIOD_YEARS:
            expected = -(math.sqrt(6.0) / math.pi) * (
                gamma + math.log(math.log(float(rp) / (float(rp) - 1.0)))
            )
            assert math.isclose(
                gumbel_frequency_factor(rp), expected, rel_tol=1e-12
            )

    def test_extract_annual_maxima_and_gumbel_fit(self) -> None:
        times: list[str] = []
        values: list[float] = []
        for year in range(2000, 2020):
            times.append(f'{year}-05-01')
            values.append(100.0 + 15.0 * (year - 2000))
            times.append(f'{year}-08-01')
            values.append(50.0)
        maxima = extract_annual_maxima(times, values, min_valid_days=2)
        assert len(maxima) == 20
        assert maxima[0] == 100.0
        assert maxima[-1] == 100.0 + 15.0 * 19

        rp = compute_gumbel_return_periods(maxima)
        assert rp is not None
        levels = [rp[f'return_period_{t}'] for t in RETURN_PERIOD_YEARS]
        for earlier, later in zip(levels, levels[1:]):
            assert float(later) > float(earlier)

        fit = ev1_fit_line(rp)
        assert fit is not None
        intercept, slope = fit
        assert slope > 0.0
        assert intercept > 0.0

    def test_known_return_levels_and_ev1_interpolation(self) -> None:
        rp_dict = compute_gumbel_return_periods(
            [1000.0 + 100.0 * i for i in range(20)]
        )
        assert rp_dict is not None
        levels = known_return_levels(rp_dict)
        assert len(levels) == len(RETURN_PERIOD_YEARS)
        for rp, q_val in levels:
            interp_q = gumbel_quantile_from_return_periods(rp_dict, rp)
            assert interp_q is not None
            assert math.isclose(interp_q, q_val, abs_tol=0.5)
            est_rp = estimate_return_period_years(q_val, rp_dict)
            assert est_rp is not None
            assert math.isclose(est_rp, rp, rel_tol=0.05)

    def test_scaled_index_flood_and_thresholds(self) -> None:
        rp = scaled_index_flood_return_periods(500.0)
        assert (
            rp['return_period_100']
            > rp['return_period_50']
            > rp['return_period_20']
            > rp['return_period_5']
            > rp['return_period_2']
        )

        thresholds = thresholds_from_return_periods(rp, source='unit_test')
        assert thresholds['warning_2yr'] == rp['return_period_2']
        assert thresholds['danger_5yr'] == rp['return_period_5']
        assert thresholds['extreme_20yr'] == rp['return_period_20']
        assert thresholds['extreme_100yr'] == rp['return_period_100']
        assert thresholds['source'] == 'unit_test'

    def test_classify_exceedance_all_tiers(self) -> None:
        rp = {
            'return_period_2': 100.0,
            'return_period_5': 200.0,
            'return_period_20': 400.0,
            'return_period_100': 600.0,
        }
        assert classify_exceedance(50.0, rp)['rank'] == 0
        assert classify_exceedance(50.0, rp)['risk_level'] == 'NORMAL'
        assert classify_exceedance(120.0, rp)['rank'] == 1
        assert classify_exceedance(120.0, rp)['risk_level'] == 'WARNING'
        assert classify_exceedance(220.0, rp)['rank'] == 2
        assert classify_exceedance(220.0, rp)['risk_level'] == 'SEVERE'
        assert classify_exceedance(450.0, rp)['rank'] == 3
        assert classify_exceedance(450.0, rp)['risk_level'] == 'EXTREME'
        assert classify_exceedance(650.0, rp)['rank'] == 4
        assert classify_exceedance(650.0, rp)['return_period'] == '≥ 100-yr'
        assert classify_exceedance(None, rp)['rank'] == 0
