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

"""Visualization mixins for `ReturnPeriodCalculator`."""

from collections.abc import Sequence

import matplotlib.axes
import matplotlib.figure
import pandas as pd

from return_periods import (
    base_fitter,
    empirical_distribution_utilities,
    generalized_expected_moments_algorithm,
    plotting_utilities,
    theoretical_distribution_utilities,
)

# Default return periods (in years) drawn on hydrograph plots.
_DEFAULT_PLOTTING_RETURN_PERIODS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


class ReturnPeriodVisualizer:
    """Provides diagnostic and flood-frequency plotting methods."""

    _fitter: base_fitter.BaseFitter
    _hydrograph_series: pd.Series | None
    _peaks_series: pd.Series

    def plot_fitted_distribution(
        self,
        num_bins: int = 10,
        plotting_position_type: str = 'weibull',
        ax: matplotlib.axes.Axes | None = None,
        show: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot the fitted distribution diagnostic chart."""
        if isinstance(
            self._fitter,
            (
                theoretical_distribution_utilities.SimpleLogPearson3Fitter,
                generalized_expected_moments_algorithm.GEMAFitter,
            ),
        ):
            return plotting_utilities.histogram_plot(
                fitter=self._fitter,
                moments=self._fitter.moments,
                num_bins=num_bins,
                ax=ax,
                show=show,
            )
        if isinstance(
            self._fitter,
            empirical_distribution_utilities.LogLogTrendFitter,
        ):
            return plotting_utilities.exceedance_probability_plot(
                fitter=self._fitter,
                plotting_position_type=plotting_position_type,
                ax=ax,
                show=show,
            )
        raise NotImplementedError(
            f'No plot implemented for fitter type: {type(self._fitter)}.'
        )

    def plot_return_periods(
        self,
        ax: matplotlib.axes.Axes | None = None,
        show: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot empirical and fitted discharge against return period."""
        return plotting_utilities.return_periods_plot(
            fitter=self._fitter,
            ax=ax,
            show=show,
        )

    def plot_exceedance_probabilities(
        self,
        plotting_position_type: str = 'weibull',
        ax: matplotlib.axes.Axes | None = None,
        show: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot empirical plotting positions and fitted exceedance curve."""
        return plotting_utilities.exceedance_probability_plot(
            fitter=self._fitter,
            plotting_position_type=plotting_position_type,
            ax=ax,
            show=show,
        )

    def plot_hydrograph(
        self,
        plot_return_periods: Sequence[float] = _DEFAULT_PLOTTING_RETURN_PERIODS,
        ax: matplotlib.axes.Axes | None = None,
        show: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot the daily hydrograph with horizontal return period lines."""
        if self._hydrograph_series is None:
            raise ValueError(
                'Hydrograph series is not available; initialize '
                'ReturnPeriodCalculator with hydrograph_series.'
            )

        probs = [1.0 / rp for rp in plot_return_periods]
        return_period_flows = (
            self._fitter.flow_values_from_exceedance_probabilities(
                exceedance_probabilities=probs
            )
        )
        return_period_values = {
            float(rp): float(flow)
            for rp, flow in zip(
                plot_return_periods, return_period_flows, strict=True
            )
        }

        return plotting_utilities.hydrograph_plot(
            hydrograph_series=self._hydrograph_series,
            return_period_values=return_period_values,
            peaks_series=self._peaks_series,
            ax=ax,
            show=show,
        )
