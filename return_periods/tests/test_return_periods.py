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

"""Unit and benchmark tests for the `return_periods` package.

Includes verification against the published USGS Bulletin 17C Appendix 10
case studies (Moose River at Victory, VT and Orestimba Creek near Newman, CA)
and the USGS `MGBT` R reference implementation (Cohn et al., 2013).
"""

import pathlib
from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from return_periods import (
    PLOTTING_POSITION_TYPES,
    BaseFitter,
    DistributionParameterError,
    DuplicateIndexError,
    GEMAFitter,
    GrubbsBeckTester,
    InvalidFlowValueError,
    LogLogTrendFitter,
    MultipleGrubbsBeckTester,
    NotEnoughDataError,
    ReturnPeriodCalculator,
    SimpleLogPearson3Fitter,
    bulletin17b_station_skew_mse,
    empirical_pdf,
    extract_annual_maximums,
    extract_n_highest_peaks,
    extract_peaks_by_separation_and_threshold,
    mgbt_order_statistic_pvalue,
    pearson3_cdf,
    pearson3_invcdf,
    pearson3_parameters_from_moments,
    pearson3_pmf,
    sample_moments,
    simple_empirical_plotting_position,
    threshold_exceedance_empirical_plotting_position,
)

mpl.use('Agg')

_TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / 'test_data'


def _load_example_data(
    filename: str,
    *,
    as_series: bool = False,
) -> pd.Series | np.ndarray:
    """Load an example annual peak flow CSV from test_data."""
    df = pd.read_csv(_TEST_DATA_DIR / filename)
    if as_series:
        return pd.Series(
            df['Peak Flow'].to_numpy(dtype=float),
            index=df['Year'].to_numpy(dtype=int),
            dtype=float,
        )
    return df['Peak Flow'].to_numpy(dtype=float)


class _DummyFitter(BaseFitter):
    """Concrete subclass of BaseFitter for testing base validation logic."""

    def exceedance_probabilities_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return np.asarray(flows, dtype=float)

    def flow_values_from_exceedance_probabilities(
        self,
        exceedance_probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self._check_exceedance_probabilities(exceedance_probabilities)


@pytest.mark.unit
class TestBaseFitter:
    """Tests for `BaseFitter` data validation and transformations."""

    @pytest.mark.parametrize(
        ('data', 'expected_exception'),
        [
            ([0, 1, 2, 3, 4], NotEnoughDataError),
            ([-1, 1, 2, 3, 4, 5], InvalidFlowValueError),
            ([np.nan, 1, 2, 3, 4, 5], InvalidFlowValueError),
            ([np.inf, 1, 2, 3, 4, 5], InvalidFlowValueError),
            ([-np.inf, 1, 2, 3, 4, 5], InvalidFlowValueError),
        ],
    )
    def test_invalid_or_insufficient_data(
        self,
        data: list[float],
        expected_exception: type[Exception],
    ) -> None:
        """Verify that negative, NaN/inf, or <5 positive flows raise errors."""
        with pytest.raises(expected_exception):
            _DummyFitter(data=data)

    @pytest.mark.parametrize('log_transform', [True, False])
    def test_transform_and_untransform(self, log_transform: bool) -> None:  # noqa: FBT001
        """Verify forward/inverse log10 transformations and zero counting."""
        data = [0, 1, 2, 3, 4, 5]
        positive_data = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        fitter = _DummyFitter(data=data, log_transform=log_transform)

        assert fitter.record_length == 5  # noqa: PLR2004
        assert fitter.total_record_length == 6  # noqa: PLR2004
        assert fitter.num_zero_flows == 1
        assert fitter.pilf_threshold == 0.0

        expected_transformed = (
            np.log10(positive_data) if log_transform else positive_data
        )
        np.testing.assert_allclose(
            fitter.transformed_sample, expected_transformed
        )
        np.testing.assert_allclose(
            fitter._untransform_data(fitter.transformed_sample),  # noqa: SLF001
            positive_data,
        )

    @pytest.mark.parametrize(
        'bad_probs',
        [
            [0.0, 0.5],
            [0.5, 1.0],
            [-0.1, 0.5],
            [0.5, 1.1],
            [np.nan, 0.5],
            [np.inf, 0.5],
        ],
    )
    def test_check_exceedance_probabilities_rejects_out_of_bounds(
        self,
        bad_probs: list[float],
    ) -> None:
        """Verify exceedance probabilities outside (0, 1) raise ValueError."""
        fitter = _DummyFitter(data=[1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match='exceedance probabilities'):
            fitter.flow_values_from_exceedance_probabilities(bad_probs)


@pytest.mark.unit
class TestTheoreticalDistributionUtilities:
    """Tests for Pearson Type III functions and `SimpleLogPearson3Fitter`."""

    @pytest.mark.parametrize(
        'moments',
        [
            [0.0, 0.0, 0.1],
            [0.0, -1.0, 0.1],
            [0.0, 1.0, 0.0],
            [np.nan, 1.0, 0.1],
            [0.0, np.nan, 0.1],
            [0.0, 1.0, np.nan],
            [np.inf, 1.0, 0.1],
            [0.0, np.inf, 0.1],
            [0.0, 1.0, -np.inf],
        ],
    )
    def test_invalid_pearson3_parameters(self, moments: list[float]) -> None:
        """Verify invalid standard deviation, zero skew, or inf raises error."""
        with pytest.raises(DistributionParameterError):
            pearson3_parameters_from_moments(moments=moments)

    def test_wrong_moments_length(self) -> None:
        """Verify moments tuple of length != 3 raises ValueError."""
        with pytest.raises(ValueError, match='3 moments'):
            pearson3_parameters_from_moments(moments=[1.0, 2.0])

    @pytest.mark.parametrize(
        ('moments', 'expected_params'),
        [
            ([0.0, 1.0, 1.0], (-2.0, 4.0, 0.5)),
            ([0.0, 1.0, -1.0], (2.0, 4.0, -0.5)),
        ],
    )
    def test_pearson3_parameters_from_moments(
        self,
        moments: list[float],
        expected_params: tuple[float, float, float],
    ) -> None:
        """Verify moment-to-parameter conversion for pos. & neg. skew."""
        params = pearson3_parameters_from_moments(moments=moments)
        np.testing.assert_allclose(params, expected_params)

    def test_pearson3_cdf_and_invcdf_roundtrip(self) -> None:
        """Verify pearson3_cdf and pearson3_invcdf are exact inverses."""
        percentiles = np.linspace(0.05, 0.95, 19)
        for skew in (0.6, -0.6):
            moments = (3.2, 0.4, skew)
            quantiles = pearson3_invcdf(
                percentiles=percentiles, moments=moments
            )
            recovered = pearson3_cdf(values=quantiles, moments=moments)
            np.testing.assert_allclose(recovered, percentiles, rtol=1e-6)

    def test_pearson3_cdf_clamps_outside_support(self) -> None:
        """Verify pearson3_cdf returns 0.0 / 1.0 outside support tau."""
        # Positive skew: tau = 0 - 2*1/1 = -2.0 (lower bound)
        cdf_pos = pearson3_cdf(
            values=[-5.0, -2.0, 0.0], moments=[0.0, 1.0, 1.0]
        )
        assert not np.any(np.isnan(cdf_pos))
        assert cdf_pos[0] == 0.0
        assert cdf_pos[1] == 0.0

        # Negative skew: tau = 0 - 2*1/(-1) = +2.0 (upper bound)
        cdf_neg = pearson3_cdf(values=[0.0, 2.0, 5.0], moments=[0.0, 1.0, -1.0])
        assert not np.any(np.isnan(cdf_neg))
        assert cdf_neg[1] == 1.0
        assert cdf_neg[2] == 1.0

    def test_pearson3_pmf(self) -> None:
        """Verify pearson3_pmf matches differences of pearson3_cdf."""
        bins = np.linspace(-1.99, 10.0, 25)
        pmf = pearson3_pmf(bin_edges=bins, moments=[0.0, 1.0, 1.0])
        assert len(pmf) == len(bins) - 1
        assert np.all(pmf >= 0.0)
        np.testing.assert_allclose(
            np.sum(pmf),
            pearson3_cdf([10.0], [0.0, 1.0, 1.0])[0]
            - pearson3_cdf([-1.99], [0.0, 1.0, 1.0])[0],
        )

    def test_sample_moments_matches_bulletin17c_formulas(self) -> None:
        """Verify sample_moments against Bulletin 17C Equations 5-7."""
        data = np.asarray([1.0, 2.0, 4.0, 7.0, 11.0])
        mean, std, skew = sample_moments(data)
        expected_mean = 5.0
        expected_std = np.sqrt(np.sum((data - 5.0) ** 2) / 4.0)
        expected_skew = (5.0 / (4.0 * 3.0)) * np.sum(
            ((data - expected_mean) / expected_std) ** 3
        )
        np.testing.assert_allclose(
            (mean, std, skew),
            (expected_mean, expected_std, expected_skew),
        )
        with pytest.raises(InvalidFlowValueError):
            sample_moments([1.0, 2.0, np.inf, 4.0])

    def test_simple_lp3_zero_flow_jennings_benson_adjustment(self) -> None:
        """Verify Jennings-Benson zero-flow probability adjustment."""
        orest = _load_example_data('orestimba_creek_example_data.csv')
        fitter = SimpleLogPearson3Fitter(data=orest)
        # Orestimba Creek has 70 positive years out of 82 total -> p_pos = 70/82
        p_pos = 70.0 / 82.0
        # For exceedance probabilities >= p_pos, flow quantile must be 0.0
        q_high_prob = fitter.flow_values_from_exceedance_probabilities(
            [p_pos, 0.90, 0.99]
        )
        np.testing.assert_allclose(q_high_prob, [0.0, 0.0, 0.0])

        # For exceedance probabilities < p_pos, roundtrip must be exact
        aeps = np.asarray([0.50, 0.20, 0.10, 0.04, 0.02, 0.01, 0.002])
        q_est = fitter.flow_values_from_exceedance_probabilities(aeps)
        assert np.all(q_est > 0.0)
        p_recovered = fitter.exceedance_probabilities_from_flow_values(q_est)
        np.testing.assert_allclose(p_recovered, aeps, rtol=1e-6)

        # Flow of 0.0 returns exceedance probability p_pos
        p_zero = fitter.exceedance_probabilities_from_flow_values([0.0])
        np.testing.assert_allclose(p_zero, [p_pos], rtol=1e-12)

    def test_simple_lp3_constant_or_mismatched_regional_skew_raises(
        self,
    ) -> None:
        """Verify constant data or mismatched regional_skew raises error."""
        with pytest.raises(DistributionParameterError):
            SimpleLogPearson3Fitter(data=[100.0, 100.0, 100.0, 100.0, 100.0])
        with pytest.raises(ValueError, match='regional_skew'):
            SimpleLogPearson3Fitter(
                data=[10.0, 20.0, 30.0, 40.0, 50.0],
                regional_skew=0.2,
            )


@pytest.mark.unit
class TestEmpiricalDistributionUtilities:
    """Tests for empirical plotting positions and `LogLogTrendFitter`."""

    def test_empirical_pdf(self) -> None:
        """Verify empirical_pdf normalizes bin counts to sum to 1."""
        data = np.linspace(0.0, 100.0, 101)
        hist, bin_edges = empirical_pdf(data, num_bins=10)
        assert len(hist) == 10  # noqa: PLR2004
        assert len(bin_edges) == 11  # noqa: PLR2004
        np.testing.assert_allclose(np.sum(hist), 1.0)
        with pytest.raises(ValueError, match='finite'):
            empirical_pdf([1.0, np.nan, 3.0])

    @pytest.mark.parametrize(
        'plotting_type',
        list(PLOTTING_POSITION_TYPES.keys()),
    )
    def test_simple_empirical_plotting_position_monotonicity(
        self,
        plotting_type: str,
    ) -> None:
        """Verify sorted data is ascending and exceedance probs descending."""
        # Include unsorted values and ties
        data = [100.0, 10.0, 50.0, 50.0, 200.0, 5.0, 50.0]
        sorted_data, probs = simple_empirical_plotting_position(
            data=data,
            plotting_position_type=plotting_type,
        )
        assert np.all(np.diff(sorted_data) >= 0)
        assert np.all(np.diff(probs) < 0)
        assert np.all((probs > 0.0) & (probs < 1.0))

    def test_threshold_exceedance_plotting_position(self) -> None:
        """Verify Hirsch-Stedinger plotting positions with a PILF threshold."""
        data = np.asarray([10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        # When threshold is 0 (all observations above threshold), matches simple
        s1, p1 = simple_empirical_plotting_position(data, 'weibull')
        s2, p2 = threshold_exceedance_empirical_plotting_position(
            data, thresholds=0.0, plotting_position_type='weibull'
        )
        np.testing.assert_allclose(s1, s2)
        np.testing.assert_allclose(p1, p2)

        # With PILF threshold = 100.0: 5 of 8 peaks >= 100 -> p_e = 5/8 = 0.625
        s_pilf, p_pilf = threshold_exceedance_empirical_plotting_position(
            data, thresholds=100.0, plotting_position_type='weibull'
        )
        np.testing.assert_allclose(s_pilf, np.sort(data))
        # The 5 peaks >= 100 have exceedance probabilities < 5/8 = 0.625
        assert np.all(p_pilf[3:] < 0.625)  # noqa: PLR2004
        # The 3 PILFs < 100 have exceedance probabilities > 5/8 = 0.625
        assert np.all(p_pilf[:3] > 0.625)  # noqa: PLR2004
        assert np.all(np.diff(p_pilf) < 0)

    def test_log_log_trend_fitter_roundtrip_and_zero_flows(self) -> None:
        """Verify LogLogTrendFitter roundtrip and zero-flow adjustment."""
        moose = _load_example_data('moose_river_example_data.csv')
        fitter = LogLogTrendFitter(data=moose)
        probs = np.linspace(0.05, 0.95, 10)
        flows = fitter.flow_values_from_exceedance_probabilities(probs)
        recovered = fitter.exceedance_probabilities_from_flow_values(flows)
        np.testing.assert_allclose(recovered, probs, rtol=1e-5)

        # With zero flows (Orestimba Creek: 70 positive / 82 total)
        orest = _load_example_data('orestimba_creek_example_data.csv')
        fitter_z = LogLogTrendFitter(data=orest)
        p_pos = 70.0 / 82.0
        np.testing.assert_allclose(
            fitter_z.flow_values_from_exceedance_probabilities([p_pos, 0.95]),
            [0.0, 0.0],
        )
        aeps = np.asarray([0.50, 0.20, 0.10, 0.01])
        q_z = fitter_z.flow_values_from_exceedance_probabilities(aeps)
        np.testing.assert_allclose(
            fitter_z.exceedance_probabilities_from_flow_values(q_z),
            aeps,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            fitter_z.exceedance_probabilities_from_flow_values([0.0]),
            [p_pos],
            rtol=1e-12,
        )

        # Constant data raises DistributionParameterError
        with pytest.raises(DistributionParameterError):
            LogLogTrendFitter(data=[50.0, 50.0, 50.0, 50.0, 50.0])


@pytest.mark.unit
class TestGrubbsBeckAndMGBT:
    """Tests for `MultipleGrubbsBeckTester` and `GrubbsBeckTester`."""

    def test_mgbt_single_order_statistic_pvalue(self) -> None:
        """Verify orthogonal-t p-value for Orestimba Creek order statistics."""
        # Orestimba Creek (n=82): r=30 has omega_30 = -2.04091670 -> p ~ 0.00067
        p30 = mgbt_order_statistic_pvalue(n=82, r=30, omega=-2.04091670)
        np.testing.assert_allclose(p30, 0.0006675, rtol=1e-3)

        # r=31 has omega_31 = -1.70491268 -> p ~ 0.0400
        p31 = mgbt_order_statistic_pvalue(n=82, r=31, omega=-1.70491268)
        np.testing.assert_allclose(p31, 0.0400, rtol=1e-2)

    def test_mgbt_moose_river_benchmark(self) -> None:
        """Verify MGBT finds 0 outliers and -inf threshold on Moose River."""
        moose = _load_example_data('moose_river_example_data.csv')
        # First 68 rows are the 1947-2014 record in Bulletin 17C Table 10-2
        tester_68 = MultipleGrubbsBeckTester(
            data=moose[:68], is_log_transformed=False
        )
        assert tester_68.klow == 0
        assert tester_68.threshold == -np.inf
        assert 10.0**tester_68.threshold == 0.0
        assert len(tester_68.out_of_population_sample) == 0
        assert len(tester_68.in_population_sample) == 68  # noqa: PLR2004

        # Full 75-year record also has 0 outliers
        tester_75 = MultipleGrubbsBeckTester(
            data=moose, is_log_transformed=False
        )
        assert tester_75.klow == 0
        assert tester_75.threshold == -np.inf

    def test_mgbt_orestimba_creek_benchmark(self) -> None:
        """Verify MGBT finds klow=30 and threshold=782 on Orestimba Creek."""
        orest = _load_example_data('orestimba_creek_example_data.csv')
        tester = MultipleGrubbsBeckTester(data=orest, is_log_transformed=False)
        # Bulletin 17C Appendix 10 (p. 113): MGBT identifies 30 PILFs (12 zeros
        # + 18 non-zero low outliers) with PILF threshold = 782 cfs.
        assert tester.klow == 30  # noqa: PLR2004
        assert tester.num_zero_flows == 12  # noqa: PLR2004
        assert len(tester.out_of_population_sample) == 18  # noqa: PLR2004
        assert len(tester.in_population_sample) == 52  # noqa: PLR2004
        np.testing.assert_allclose(10.0**tester.threshold, 782.0, rtol=1e-6)

    def test_legacy_bulletin17b_grubbs_beck_tester(self) -> None:
        """Verify Bulletin 17B GrubbsBeckTester for 1 and 0 low outliers."""
        # Construct 20 log10 peaks with 1 clear low outlier at index 0
        rng = np.random.default_rng(42)
        in_pop = np.sort(rng.normal(loc=3.5, scale=0.15, size=19))
        data = np.concatenate([[1.5], in_pop])
        tester = GrubbsBeckTester(data=data)
        assert len(tester.out_of_population_sample) == 1
        assert tester.out_of_population_sample[0] == 1.5  # noqa: PLR2004
        assert len(tester.in_population_sample) == 19  # noqa: PLR2004
        assert tester.threshold == float(in_pop[0])

        # With 0 low outliers, threshold is -np.inf (10**threshold == 0.0)
        tester_no_outlier = GrubbsBeckTester(data=in_pop)
        assert len(tester_no_outlier.out_of_population_sample) == 0
        assert tester_no_outlier.threshold == -np.inf
        assert 10.0**tester_no_outlier.threshold == 0.0


@pytest.mark.unit
class TestBulletin17CBenchmarksAndEMA:
    """End-to-end verification against USGS Bulletin 17C Appendix 10 tables."""

    def test_bulletin17c_example1_moose_river(self) -> None:
        """Verify Moose River station/weighted skew against B17C Table 10-5."""
        moose_all = _load_example_data('moose_river_example_data.csv')
        # Bulletin 17C Table 10-2 uses the 68 peaks from 1947 to 2014
        moose_1947_2014 = moose_all[:68]

        # 1. Station skew only (Bulletin 17C Figure 10-3, p. 110):
        # PeakFQ v7.1 reports station skew G = 0.397, MSE_G = 0.101
        fitter_station = GEMAFitter(data=moose_1947_2014)
        assert fitter_station.pilf_threshold == 0.0
        np.testing.assert_allclose(fitter_station.moments[2], 0.397, atol=1e-3)
        mse_g = bulletin17b_station_skew_mse(
            record_length=68, station_skew=fitter_station.moments[2]
        )
        np.testing.assert_allclose(mse_g, 0.101, atol=1e-3)

        # Because klow == 0, GEMAFitter and SimpleLogPearson3Fitter must match
        simple_fitter = SimpleLogPearson3Fitter(data=moose_1947_2014)
        np.testing.assert_allclose(
            fitter_station.moments, simple_fitter.moments, rtol=1e-10
        )

        # Compare station-skew quantiles against B17C Table 10-5 (p. 110)
        # AEP: 0.10, 0.04, 0.02, 0.01, 0.005, 0.002
        # Published PeakFQ EMA estimates: 3261, 3911, 4422, 4957, 5519, 6313
        aeps = [0.10, 0.04, 0.02, 0.01, 0.005, 0.002]
        published_q = [3261.0, 3911.0, 4422.0, 4957.0, 5519.0, 6313.0]
        estimated_q = fitter_station.flow_values_from_exceedance_probabilities(
            aeps
        )
        np.testing.assert_allclose(estimated_q, published_q, rtol=5e-4)

        # 2. Weighted skew (Bulletin 17C Figure 10-2, p. 109):
        # Regional skew = 0.44, Regional skew MSE = 0.078 -> G_w = 0.421
        fitter_weighted = GEMAFitter(
            data=moose_1947_2014,
            regional_skew=0.44,
            regional_skew_mse=0.078,
        )
        np.testing.assert_allclose(fitter_weighted.moments[2], 0.421, atol=1e-3)

    def test_bulletin17c_example2_orestimba_creek(self) -> None:
        """Verify Orestimba Creek PILF + EMA against Bulletin 17C Table 10-9."""
        orest = _load_example_data('orestimba_creek_example_data.csv')

        fitter = GEMAFitter(data=orest)
        np.testing.assert_allclose(fitter.pilf_threshold, 782.0, rtol=1e-6)

        # Published PeakFQ v7.1 EMA moments (Bulletin 17C Figure 10-5, p. 113):
        # Mean = 3.0227, Std = 0.6821, Skew = -0.9291
        np.testing.assert_allclose(
            fitter.moments,
            (3.0227, 0.6821, -0.9291),
            atol=5e-4,
        )

        # Published PeakFQ v7.1 EMA quantiles (Bulletin 17C Table 10-9, p. 115):
        aeps = [0.50, 0.20, 0.10, 0.04, 0.02, 0.01, 0.005, 0.002]
        published_q = [
            1339.0,
            4026.0,
            6328.0,
            9426.0,
            11690.0,
            13820.0,
            15800.0,
            18150.0,
        ]
        estimated_q = fitter.flow_values_from_exceedance_probabilities(aeps)
        np.testing.assert_allclose(estimated_q, published_q, rtol=5e-4)

        # Also verify legacy B17B single-outlier screening mode on Orestimba
        fitter_b17b = GEMAFitter(data=orest, use_multiple_grubbs_beck=False)
        assert fitter_b17b.pilf_threshold > 0.0
        assert np.all(
            np.isfinite(
                fitter_b17b.flow_values_from_exceedance_probabilities(aeps)
            )
        )

        # Constant data or mismatched regional_skew raises error
        with pytest.raises(DistributionParameterError):
            GEMAFitter(data=[100.0, 100.0, 100.0, 100.0, 100.0])
        with pytest.raises(ValueError, match='regional_skew'):
            GEMAFitter(data=orest, regional_skew=-0.5)

    def test_gema_near_zero_skew_does_not_overflow(self) -> None:
        """Verify _interval_gamma_moment_ratios handles tiny skew."""
        # Symmetric log-flows with a low outlier so skew crosses near 0
        symmetric_in_pop = np.linspace(2.5, 4.5, 41)
        data_log = np.concatenate([[0.5, 0.6], symmetric_in_pop])
        fitter = GEMAFitter(data=10.0**data_log)
        assert not np.any(np.isnan(fitter.moments))
        q = fitter.flow_values_from_exceedance_probabilities([0.5, 0.1, 0.01])
        assert np.all(np.isfinite(q))

    def test_zero_flow_scale_invariance_and_analytical_limit(self) -> None:
        """Verify MGBT and GEMAFitter are scale-invariant with zero flows."""
        orest_cfs = _load_example_data('orestimba_creek_example_data.csv')
        # Scale by an extreme unit factor (e.g. 1e-6 for mm/day on a tiny basin)
        scale = 1e-6
        orest_scaled = orest_cfs * scale

        fitter_cfs = GEMAFitter(data=orest_cfs)
        fitter_scaled = GEMAFitter(data=orest_scaled)

        assert fitter_cfs.outlier_tester.klow == 30  # noqa: PLR2004
        assert fitter_scaled.outlier_tester.klow == 30  # noqa: PLR2004
        np.testing.assert_allclose(
            fitter_cfs.outlier_tester.p_values,
            fitter_scaled.outlier_tester.p_values,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            fitter_cfs.moments[1], fitter_scaled.moments[1], rtol=1e-12
        )
        np.testing.assert_allclose(
            fitter_cfs.moments[2], fitter_scaled.moments[2], rtol=1e-12
        )
        aeps = [0.5, 0.1, 0.01, 0.002]
        q_cfs = fitter_cfs.flow_values_from_exceedance_probabilities(aeps)
        q_scaled = fitter_scaled.flow_values_from_exceedance_probabilities(aeps)
        np.testing.assert_allclose(q_scaled, q_cfs * scale, rtol=1e-11)


@pytest.mark.unit
class TestExtractPeaksUtilities:
    """Tests for `extract_peaks_utilities` functions."""

    @pytest.fixture
    def synthetic_hydrograph(self) -> pd.Series:
        """Create a 3-year daily hydrograph with known annual and POT peaks."""
        dates = pd.date_range('2018-01-01', '2020-12-31', freq='D')
        flows = np.full(len(dates), 10.0)
        series = pd.Series(flows, index=dates)
        series.loc['2018-03-15'] = 500.0
        series.loc['2018-11-20'] = 300.0
        series.loc['2019-04-10'] = 800.0
        series.loc['2020-02-05'] = 650.0
        return series

    def test_extract_annual_maximums_calendar_and_water_year(
        self,
        synthetic_hydrograph: pd.Series,
    ) -> None:
        """Verify calendar-year and Oct-1 water-year annual max extraction."""
        cal_peaks = extract_annual_maximums(synthetic_hydrograph)
        assert list(cal_peaks.index) == [2018, 2019, 2020]
        np.testing.assert_allclose(cal_peaks.to_numpy(), [500.0, 800.0, 650.0])

        # In water-year mode (start month 10), 2018-11-20 belongs to WY 2019
        wy_peaks = extract_annual_maximums(
            synthetic_hydrograph,
            max_missing_days_in_year=120,
            water_year_start_month=10,
        )
        assert 2019 in wy_peaks.index  # noqa: PLR2004
        assert wy_peaks.loc[2019] == 800.0  # noqa: PLR2004

    def test_extract_peaks_by_separation_and_threshold(
        self,
        synthetic_hydrograph: pd.Series,
    ) -> None:
        """Verify POT peak extraction respects separation and quantile."""
        pot = extract_peaks_by_separation_and_threshold(
            synthetic_hydrograph,
            min_peak_separation=pd.Timedelta('60D'),
            min_peak_quantile=0.95,
        )
        np.testing.assert_allclose(pot.to_numpy(), [800.0, 650.0, 500.0, 300.0])

    def test_extract_n_highest_peaks(
        self,
        synthetic_hydrograph: pd.Series,
    ) -> None:
        """Verify extract_n_highest_peaks returns top N independent peaks."""
        top2 = extract_n_highest_peaks(
            synthetic_hydrograph,
            num_peaks=2,
            min_peak_separation=pd.Timedelta('90D'),
        )
        assert len(top2) == 2  # noqa: PLR2004
        np.testing.assert_allclose(top2.to_numpy(), [800.0, 650.0])

    def test_extract_peaks_parameter_and_value_validation(
        self,
        synthetic_hydrograph: pd.Series,
    ) -> None:
        """Verify invalid parameters or negative/inf flows raise errors."""
        with pytest.raises(ValueError, match='max_missing_days_in_year'):
            extract_annual_maximums(
                synthetic_hydrograph, max_missing_days_in_year=365
            )
        with pytest.raises(ValueError, match='min_peak_separation'):
            extract_peaks_by_separation_and_threshold(
                synthetic_hydrograph, min_peak_separation=pd.Timedelta('0D')
            )
        with pytest.raises(ValueError, match='min_peak_quantile'):
            extract_peaks_by_separation_and_threshold(
                synthetic_hydrograph, min_peak_quantile=1.0
            )
        with pytest.raises(ValueError, match='num_peaks'):
            extract_n_highest_peaks(synthetic_hydrograph, num_peaks=0)

        bad_neg = synthetic_hydrograph.copy()
        bad_neg.iloc[0] = -5.0
        with pytest.raises(InvalidFlowValueError):
            extract_annual_maximums(bad_neg)

        bad_inf = synthetic_hydrograph.copy()
        bad_inf.iloc[0] = np.inf
        with pytest.raises(InvalidFlowValueError):
            extract_n_highest_peaks(bad_inf, num_peaks=2)


@pytest.mark.unit
class TestReturnPeriodCalculatorAndVisualizer:
    """Tests for `ReturnPeriodCalculator` and `ReturnPeriodVisualizer`."""

    def test_calculator_missing_or_invalid_fitter_raises(self) -> None:
        """Verify omitting or passing an unknown `fitter` raises ValueError."""
        peaks = _load_example_data(
            'moose_river_example_data.csv', as_series=True
        )
        assert isinstance(peaks, pd.Series)
        with pytest.raises(
            ValueError, match='Must explicitly provide `fitter`'
        ):
            ReturnPeriodCalculator(peaks_series=peaks)
        with pytest.raises(ValueError, match='Unknown fitter'):
            ReturnPeriodCalculator(peaks_series=peaks, fitter='unknown')
        with pytest.raises(ValueError, match='regional_skew'):
            ReturnPeriodCalculator(
                peaks_series=peaks, fitter='gema', regional_skew=0.2
            )

    def test_calculator_return_periods_and_percentiles_roundtrip(self) -> None:
        """Verify ReturnPeriodCalculator forward/inverse conversions."""
        peaks = _load_example_data(
            'moose_river_example_data.csv', as_series=True
        )
        assert isinstance(peaks, pd.Series)
        rpc = ReturnPeriodCalculator(peaks_series=peaks, fitter='gema')
        assert rpc.fitter_name == 'gema'

        return_periods = np.asarray([2.0, 5.0, 10.0, 25.0, 50.0, 100.0])
        flows = rpc.flow_values_from_return_periods(return_periods)
        recovered_rps = rpc.return_periods_from_flow_values(flows)
        np.testing.assert_allclose(recovered_rps, return_periods, rtol=1e-5)

        # Verify non-exceedance percentile roundtrip
        percentiles = np.linspace(0.1, 0.9, 9)
        p_flows = rpc.flow_values_from_percentiles(percentiles)
        recovered_non_exc = rpc.percentiles_from_flow_values(
            p_flows, non_exceedance=True
        )
        np.testing.assert_allclose(recovered_non_exc, percentiles, rtol=1e-5)

        # Default percentiles_from_flow_values returns exceedance (1 - F)
        recovered_exc = rpc.percentiles_from_flow_values(p_flows)
        np.testing.assert_allclose(recovered_exc, 1.0 - percentiles, rtol=1e-5)

        # Verify explicit class or alternate fitter names
        rpc_lp3 = ReturnPeriodCalculator(
            peaks_series=peaks, fitter='simple_lp3'
        )
        assert rpc_lp3.fitter_name == 'simple_lp3'
        rpc_ll = ReturnPeriodCalculator(
            peaks_series=peaks, fitter=LogLogTrendFitter
        )
        assert rpc_ll.fitter_name == 'log_linear'

    def test_calculator_duplicate_index_or_nan_raises(self) -> None:
        """Verify duplicate index or NaN in peaks_series raises error."""
        s_dup = pd.Series(
            [100.0, 200.0, 300.0, 400.0, 500.0], index=[1, 1, 2, 3, 4]
        )
        with pytest.raises(DuplicateIndexError):
            ReturnPeriodCalculator(peaks_series=s_dup, fitter='gema')

        s_nan = pd.Series(
            [100.0, np.nan, 300.0, 400.0, 500.0, 600.0],
            index=[1, 2, 3, 4, 5, 6],
        )
        with pytest.raises(InvalidFlowValueError):
            ReturnPeriodCalculator(peaks_series=s_nan, fitter='gema')

    def test_visualizer_plots_execute_cleanly(self) -> None:
        """Verify all visualizer plots return (Figure, Axes) without error."""
        dates = pd.date_range('2010-01-01', '2020-12-31', freq='D')
        rng = np.random.default_rng(123)
        hydro = pd.Series(
            rng.lognormal(mean=4.0, sigma=0.8, size=len(dates)), index=dates
        )

        rpc = ReturnPeriodCalculator(hydrograph_series=hydro, fitter='gema')
        for plot_fn in (
            rpc.plot_fitted_distribution,
            rpc.plot_return_periods,
            rpc.plot_exceedance_probabilities,
            rpc.plot_hydrograph,
        ):
            fig, ax = plot_fn(show=False)
            assert fig is not None
            assert ax is not None
            plt.close(fig)

        # Also verify plotting on Orestimba Creek (which has zero flows + PILFs)
        orest_series = _load_example_data(
            'orestimba_creek_example_data.csv', as_series=True
        )
        assert isinstance(orest_series, pd.Series)
        rpc_orest = ReturnPeriodCalculator(
            peaks_series=orest_series, fitter='gema'
        )
        for plot_fn in (
            rpc_orest.plot_fitted_distribution,
            rpc_orest.plot_return_periods,
            rpc_orest.plot_exceedance_probabilities,
        ):
            fig, ax = plot_fn(show=False)
            assert fig is not None
            assert ax is not None
            plt.close(fig)


@pytest.mark.unit
class TestCaravanUSGSBenchmarkCLI:
    """Tests for the Caravan USGS R + Fortran benchmark CLI."""

    def test_print_setup_instructions_exits_zero(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verify `--print-setup-instructions` prints setup instructions."""
        from return_periods.tools.run_caravan_usgs_benchmark import (  # noqa: PLC0415
            SETUP_INSTRUCTIONS,
            main,
        )

        rc = main(['--print-setup-instructions'])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'https://code.usgs.gov/water/peakfqr.git' in captured.out
        assert 'https://github.com/cran/MGBT.git' in captured.out
        assert 'peakfq.so' in SETUP_INSTRUCTIONS

    def test_parse_args_accepts_external_repo_paths(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify `parse_args` parses external R & Fortran repo paths."""
        from return_periods.tools.run_caravan_usgs_benchmark import (  # noqa: PLC0415
            parse_args,
        )

        args = parse_args(
            [
                '--caravan-dir',
                str(tmp_path / 'caravan'),
                '--peakfqr-repo',
                str(tmp_path / 'peakfqr'),
                '--mgbt-repo',
                str(tmp_path / 'MGBT'),
                '--output-dir',
                str(tmp_path / 'out'),
                '--workers',
                '8',
            ]
        )
        assert args.caravan_dir == tmp_path / 'caravan'
        assert args.peakfqr_repo == tmp_path / 'peakfqr'
        assert args.mgbt_repo == tmp_path / 'MGBT'
        assert args.output_dir == tmp_path / 'out'
        assert args.workers == 8  # noqa: PLR2004
