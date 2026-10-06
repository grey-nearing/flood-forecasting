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

"""USGS Bulletin 17C flood frequency and return period calculator package.

Provides:

- `ReturnPeriodCalculator`: High-level flood return period and quantile
  calculator with visualization support.
- `GEMAFitter`: Generalized Expected Moments Algorithm (EMA) Log-Pearson Type
  III fitter with Multiple Grubbs-Beck Test (MGBT) low-outlier screening.
- `SimpleLogPearson3Fitter`: Direct method-of-moments Log-Pearson Type III
  fitter.
- `LogLogTrendFitter`: Empirical log-log linear regression fallback fitter.
- `MultipleGrubbsBeckTester` & `GrubbsBeckTester`: Bulletin 17C MGBT and
  Bulletin 17B single-outlier PILF testers.
- Peak extraction utilities (`extract_annual_maximums`,
  `extract_peaks_by_separation_and_threshold`, `extract_n_highest_peaks`).
"""

from return_periods.base_fitter import BaseFitter
from return_periods.empirical_distribution_utilities import (
    PLOTTING_POSITION_TYPES,
    LogLogTrendFitter,
    empirical_pdf,
    simple_empirical_plotting_position,
    threshold_exceedance_empirical_plotting_position,
)
from return_periods.exceptions import (
    DataIntervalError,
    DistributionParameterError,
    DuplicateIndexError,
    InvalidFlowValueError,
    NotEnoughDataError,
)
from return_periods.extract_peaks_utilities import (
    extract_annual_maximums,
    extract_n_highest_peaks,
    extract_peaks_by_separation_and_threshold,
)
from return_periods.generalized_expected_moments_algorithm import GEMAFitter
from return_periods.grubbs_beck_tester import (
    GrubbsBeckTester,
    MultipleGrubbsBeckTester,
    mgbt_order_statistic_pvalue,
)
from return_periods.plotting_utilities import (
    exceedance_probability_plot,
    histogram_plot,
    hydrograph_plot,
    return_periods_plot,
)
from return_periods.return_period_calculator import ReturnPeriodCalculator
from return_periods.return_period_visualizer import ReturnPeriodVisualizer
from return_periods.theoretical_distribution_utilities import (
    SimpleLogPearson3Fitter,
    bulletin17b_station_skew_mse,
    pearson3_cdf,
    pearson3_invcdf,
    pearson3_parameters_from_moments,
    pearson3_pmf,
    sample_moments,
    weighted_skew,
)

__all__ = [
    'PLOTTING_POSITION_TYPES',
    'BaseFitter',
    'DataIntervalError',
    'DistributionParameterError',
    'DuplicateIndexError',
    'GEMAFitter',
    'GrubbsBeckTester',
    'InvalidFlowValueError',
    'LogLogTrendFitter',
    'MultipleGrubbsBeckTester',
    'NotEnoughDataError',
    'ReturnPeriodCalculator',
    'ReturnPeriodVisualizer',
    'SimpleLogPearson3Fitter',
    'bulletin17b_station_skew_mse',
    'empirical_pdf',
    'exceedance_probability_plot',
    'extract_annual_maximums',
    'extract_n_highest_peaks',
    'extract_peaks_by_separation_and_threshold',
    'histogram_plot',
    'hydrograph_plot',
    'mgbt_order_statistic_pvalue',
    'pearson3_cdf',
    'pearson3_invcdf',
    'pearson3_parameters_from_moments',
    'pearson3_pmf',
    'return_periods_plot',
    'sample_moments',
    'simple_empirical_plotting_position',
    'threshold_exceedance_empirical_plotting_position',
    'weighted_skew',
]
