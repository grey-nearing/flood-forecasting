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

"""Main entry point for calculating streamflow return periods and quantiles.

Implements flood frequency calculations following USGS Bulletin 17C (England
et al., 2019):
https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from return_periods import (
    base_fitter,
    empirical_distribution_utilities,
    exceptions,
    extract_peaks_utilities,
    generalized_expected_moments_algorithm,
    return_period_visualizer,
    theoretical_distribution_utilities,
)

# Supported distribution fitters:
# - 'gema': Bulletin 17C Expected Moments Algorithm with MGBT PILF screening
# - 'simple_lp3': Direct method-of-moments Log-Pearson Type III
# - 'log_linear': Empirical log-log linear regression
_DISTRIBUTION_FITTERS: dict[str, type[base_fitter.BaseFitter]] = {
    'gema': generalized_expected_moments_algorithm.GEMAFitter,
    'simple_lp3': theoretical_distribution_utilities.SimpleLogPearson3Fitter,
    'log_linear': empirical_distribution_utilities.LogLogTrendFitter,
}


class ReturnPeriodCalculator(return_period_visualizer.ReturnPeriodVisualizer):
    """Calculates flood return periods and discharge quantiles."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        peaks_series: pd.Series | None = None,
        hydrograph_series: pd.Series | None = None,
        fitter: str | type[base_fitter.BaseFitter] | None = None,
        max_missing_days_in_year: int = 100,
        water_year_start_month: int = 1,
        log_transform: bool = True,  # noqa: FBT001, FBT002
        use_multiple_grubbs_beck: bool = True,  # noqa: FBT001, FBT002
        regional_skew: float | None = None,
        regional_skew_mse: float | None = None,
    ):
        """Initialize and fit a flood return period calculator.

        Either `peaks_series` or `hydrograph_series` must be provided, and
        `fitter` must be explicitly specified (`'gema'`, `'simple_lp3'`,
        `'log_linear'`, or a `BaseFitter` subclass).

        Args:
            peaks_series: Optional `pd.Series` of annual peak flows.
            hydrograph_series: Optional `pd.Series` of daily streamflow indexed
                by a `pd.DatetimeIndex`.
            fitter: Explicit distribution fitter to use (`'gema'`,
                `'simple_lp3'`, `'log_linear'`, or a `BaseFitter` subclass).
            max_missing_days_in_year: Maximum missing days allowed per year
                when extracting annual peaks from `hydrograph_series`.
            water_year_start_month: Starting month (1-12) of the water year
                when extracting annual peaks from `hydrograph_series`.
            log_transform: Whether to apply a base-10 logarithmic transform.
            use_multiple_grubbs_beck: Whether `GEMAFitter` uses the Bulletin 17C
                Multiple Grubbs-Beck Test (`True`, default) or legacy Bulletin
                17B single-outlier test (`False`).
            regional_skew: Optional regional skew coefficient `G_bar`.
            regional_skew_mse: Optional MSE of the regional skew `MSE_G_bar`.

        Raises:
            ValueError: If `fitter` is not provided or invalid, or if neither
                `peaks_series` nor `hydrograph_series` is provided.
            DuplicateIndexError: If `peaks_series` has duplicate index labels.
        """
        if fitter is None:
            raise ValueError(
                'Must explicitly provide `fitter` (one of '
                f'{tuple(_DISTRIBUTION_FITTERS)} or a BaseFitter subclass).'
            )
        if (regional_skew is None) != (regional_skew_mse is None):
            raise ValueError(
                'Both `regional_skew` and `regional_skew_mse` must be '
                'provided together, or both must be None.'
            )

        self._log_transform = log_transform
        self._use_multiple_grubbs_beck = use_multiple_grubbs_beck
        self._regional_skew = regional_skew
        self._regional_skew_mse = regional_skew_mse

        if peaks_series is None and hydrograph_series is None:
            raise ValueError(
                'Must supply either peaks_series or hydrograph_series.'
            )

        self._hydrograph_series = hydrograph_series
        if peaks_series is None:
            assert hydrograph_series is not None
            self._peaks_series = (
                extract_peaks_utilities.extract_annual_maximums(
                    hydrograph_series=hydrograph_series,
                    max_missing_days_in_year=max_missing_days_in_year,
                    water_year_start_month=water_year_start_month,
                )
            )
        else:
            self._peaks_series = peaks_series

        self._fitter, self._fitter_name = self._fit_peak_distribution(fitter)

    @property
    def fitter(self) -> base_fitter.BaseFitter:
        """Return the active distribution fitter instance."""
        return self._fitter

    @property
    def fitter_name(self) -> str:
        """Return the identifier of the active fitter ('gema', etc.)."""
        return self._fitter_name

    @property
    def peaks_series(self) -> pd.Series:
        """Return the annual peak streamflow series used for fitting."""
        return self._peaks_series

    def flow_values_from_exceedance_probabilities(
        self,
        exceedance_probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate flow quantiles for annual exceedance probabilities `p`."""
        return self._fitter.flow_values_from_exceedance_probabilities(
            exceedance_probabilities=exceedance_probabilities
        )

    def exceedance_probabilities_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate annual exceedance probabilities `p` for given flows."""
        return self._fitter.exceedance_probabilities_from_flow_values(
            flows=flows
        )

    def flow_values_from_percentiles(
        self,
        percentiles: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate flow quantiles for cumulative non-exceedance percentiles.

        Args:
            percentiles: Non-exceedance probabilities `F(Q) = 1 - p` in
                `(0, 1)`.

        Returns:
            1D NumPy array of discharge values.
        """
        exceedance_probabilities = 1.0 - np.asarray(percentiles, dtype=float)
        return self._fitter.flow_values_from_exceedance_probabilities(
            exceedance_probabilities=exceedance_probabilities
        )

    def percentiles_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
        non_exceedance: bool = False,  # noqa: FBT001, FBT002
    ) -> np.ndarray:
        """Calculate probabilities associated with given flow values.

        Args:
            flows: Sequence of discharge values.
            non_exceedance: If `False` (default for backwards compatibility),
                returns annual exceedance probabilities `p = 1 - F(Q)`. If
                `True`, returns cumulative non-exceedance probabilities `F(Q)`
                (the exact inverse of `flow_values_from_percentiles`).

        Returns:
            1D NumPy array of probabilities in `(0, 1)`.
        """
        exceedance = self._fitter.exceedance_probabilities_from_flow_values(
            flows=flows
        )
        if non_exceedance:
            return 1.0 - exceedance
        return exceedance

    def flow_values_from_return_periods(
        self,
        return_periods: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate flow quantiles for given return periods in years.

        Args:
            return_periods: Return periods `T > 1` in years (`p = 1 / T`).

        Returns:
            1D NumPy array of discharge quantiles.
        """
        exceedance_probabilities = 1.0 / np.asarray(return_periods, dtype=float)
        return self._fitter.flow_values_from_exceedance_probabilities(
            exceedance_probabilities=exceedance_probabilities
        )

    def return_periods_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate return periods in years (`T = 1 / p`) for given flows."""
        exceedance_probabilities = (
            self._fitter.exceedance_probabilities_from_flow_values(flows=flows)
        )
        return 1.0 / exceedance_probabilities

    def _resolve_fitter_class(
        self,
        fitter: str | type[base_fitter.BaseFitter],
    ) -> tuple[type[base_fitter.BaseFitter], str]:
        """Resolve a fitter name or class to `(fitter_cls, fitter_name)`."""
        if isinstance(fitter, str):
            if fitter not in _DISTRIBUTION_FITTERS:
                raise ValueError(
                    f'Unknown fitter {fitter!r}. Expected one of '
                    f'{tuple(_DISTRIBUTION_FITTERS)}.'
                )
            return _DISTRIBUTION_FITTERS[fitter], fitter
        if isinstance(fitter, type) and issubclass(
            fitter, base_fitter.BaseFitter
        ):
            for name, cls in _DISTRIBUTION_FITTERS.items():
                if fitter is cls:
                    return fitter, name
            return fitter, fitter.__name__
        raise ValueError(
            f'Invalid fitter {fitter!r}. Expected one of '
            f'{tuple(_DISTRIBUTION_FITTERS)} or a BaseFitter subclass.'
        )

    def _instantiate_fitter(
        self,
        fitter_cls: type[base_fitter.BaseFitter],
        peak_values: np.ndarray,
    ) -> base_fitter.BaseFitter:
        """Instantiate the requested fitter class with configured options."""
        if fitter_cls is generalized_expected_moments_algorithm.GEMAFitter:
            return fitter_cls(
                data=peak_values,
                log_transform=self._log_transform,
                use_multiple_grubbs_beck=self._use_multiple_grubbs_beck,
                regional_skew=self._regional_skew,
                regional_skew_mse=self._regional_skew_mse,
            )
        if (
            fitter_cls
            is theoretical_distribution_utilities.SimpleLogPearson3Fitter
        ):
            return fitter_cls(
                data=peak_values,
                log_transform=self._log_transform,
                regional_skew=self._regional_skew,
                regional_skew_mse=self._regional_skew_mse,
            )
        return fitter_cls(
            data=peak_values,
            log_transform=self._log_transform,
        )

    def _fit_peak_distribution(
        self,
        fitter: str | type[base_fitter.BaseFitter],
    ) -> tuple[base_fitter.BaseFitter, str]:
        """Fit the requested peak flow distribution directly."""
        if self._peaks_series.index.duplicated().any():
            raise exceptions.DuplicateIndexError(
                'Duplicate index values found in peaks series.'
            )

        fitter_cls, fitter_name = self._resolve_fitter_class(fitter)
        peak_values = self._peaks_series.to_numpy(dtype=float)
        return self._instantiate_fitter(fitter_cls, peak_values), fitter_name
