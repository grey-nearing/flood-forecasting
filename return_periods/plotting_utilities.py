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

"""Plotting utilities for flood frequency and return period visualization."""

from collections.abc import Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from return_periods import (
    base_fitter,
    empirical_distribution_utilities,
    theoretical_distribution_utilities,
)

# Number of points used to render smooth fitted curves.
_NUM_CURVE_POINTS = 100


def histogram_plot(
    fitter: base_fitter.BaseFitter,
    moments: Sequence[float] | np.ndarray,
    num_bins: int = 10,
    ax: matplotlib.axes.Axes | None = None,
    show: bool = False,  # noqa: FBT001, FBT002
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot an empirical histogram of log-flows vs. fitted Pearson III PMF.

    Args:
        fitter: Fitted distribution object.
        moments: Sequence of `(mean, std, skew)` in transformed space.
        num_bins: Number of histogram bins.
        ax: Optional Matplotlib `Axes` to draw on.
        show: Whether to call `plt.show()`.

    Returns:
        Tuple of `(fig, ax)`.
    """
    sample = fitter.transformed_sample

    empirical_pmf, bin_edges = empirical_distribution_utilities.empirical_pdf(
        data=sample,
        num_bins=num_bins,
    )
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    bin_widths = np.diff(bin_edges)

    pmf_bins = np.linspace(
        float(np.min(bin_edges)),
        float(np.max(bin_edges)),
        _NUM_CURVE_POINTS,
    )
    theoretical_pmf = theoretical_distribution_utilities.pearson3_pmf(
        bin_edges=pmf_bins,
        moments=moments,
    )
    # Scale fine-grid PMF to match histogram bin width.
    scale_factor = float(np.mean(bin_widths) / np.mean(np.diff(pmf_bins)))
    theoretical_pmf = theoretical_pmf * scale_factor
    pmf_bin_centers = pmf_bins[:-1] + np.diff(pmf_bins) / 2.0

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    ax.bar(
        bin_centers,
        empirical_pmf,
        width=bin_widths,
        alpha=0.5,
        label='Empirical Histogram',
    )
    ax.plot(
        pmf_bin_centers,
        theoretical_pmf,
        color='red',
        linewidth=2,
        label='Fitted Pearson III',
    )

    if fitter.pilf_threshold > 0:
        pilf_x = (
            np.log10(fitter.pilf_threshold)
            if fitter._log_transform  # noqa: SLF001
            else fitter.pilf_threshold
        )
        ax.axvline(
            pilf_x,
            color='black',
            linestyle='--',
            label='PILF Threshold',
        )

    ax.set_xlabel(
        r'$\log_{10}(\text{Peak Annual Discharge})$'
        if fitter._log_transform  # noqa: SLF001
        else 'Peak Annual Discharge'
    )
    ax.set_ylabel('Probability Mass')
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()
    return fig, ax


def exceedance_probability_plot(
    fitter: base_fitter.BaseFitter,
    plotting_position_type: str = 'weibull',
    ax: matplotlib.axes.Axes | None = None,
    show: bool = False,  # noqa: FBT001, FBT002
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot empirical plotting positions vs. fitted exceedance curve.

    Args:
        fitter: Fitted distribution object.
        plotting_position_type: Name of plotting position formula in
            `PLOTTING_POSITION_TYPES`.
        ax: Optional Matplotlib `Axes` to draw on.
        show: Whether to call `plt.show()`.

    Returns:
        Tuple of `(fig, ax)`.
    """
    if fitter.pilf_threshold > 0:
        sample, plotting_positions = (
            empirical_distribution_utilities.threshold_exceedance_empirical_plotting_position(
                data=fitter.raw_sample,
                thresholds=fitter.pilf_threshold,
                plotting_position_type=plotting_position_type,
            )
        )
    else:
        sample, plotting_positions = (
            empirical_distribution_utilities.simple_empirical_plotting_position(
                data=fitter.raw_sample,
                plotting_position_type=plotting_position_type,
            )
        )
    pos_mask = sample > 0.0

    exceedance_probabilities = np.linspace(
        base_fitter._EPSILON,  # noqa: SLF001
        1.0 - base_fitter._EPSILON,  # noqa: SLF001
        _NUM_CURVE_POINTS,
    )
    fitted_flows = fitter.flow_values_from_exceedance_probabilities(
        exceedance_probabilities
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    ax.plot(
        exceedance_probabilities,
        fitted_flows,
        'r-',
        linewidth=2,
        label='Fitted Frequency Curve',
    )
    ax.scatter(
        plotting_positions[pos_mask],
        sample[pos_mask],
        label='Empirical Plotting Positions',
    )
    if fitter.pilf_threshold > 0:
        ax.axhline(
            fitter.pilf_threshold,
            color='black',
            linestyle='--',
            label='PILF Threshold',
        )

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Peak Annual Discharge')
    ax.set_xlabel('Annual Exceedance Probability')
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()
    return fig, ax


def return_periods_plot(
    fitter: base_fitter.BaseFitter,
    ax: matplotlib.axes.Axes | None = None,
    show: bool = False,  # noqa: FBT001, FBT002
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot empirical and fitted peak discharges against return period (years).

    Args:
        fitter: Fitted distribution object.
        ax: Optional Matplotlib `Axes` to draw on.
        show: Whether to call `plt.show()`.

    Returns:
        Tuple of `(fig, ax)`.
    """
    if fitter.pilf_threshold > 0:
        sample, plotting_positions = (
            empirical_distribution_utilities.threshold_exceedance_empirical_plotting_position(
                data=fitter.raw_sample,
                thresholds=fitter.pilf_threshold,
            )
        )
    else:
        sample, plotting_positions = (
            empirical_distribution_utilities.simple_empirical_plotting_position(
                data=fitter.raw_sample,
            )
        )
    pos_mask = sample > 0.0
    empirical_return_periods = 1.0 / plotting_positions[pos_mask]

    return_periods = np.linspace(
        float(np.min(empirical_return_periods)),
        float(np.max(empirical_return_periods)),
        _NUM_CURVE_POINTS,
    )
    fitted_flows = fitter.flow_values_from_exceedance_probabilities(
        1.0 / return_periods
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    ax.plot(return_periods, fitted_flows, 'r-', label='Fitted Return Curve')
    ax.scatter(
        empirical_return_periods,
        sample[pos_mask],
        label='Observed Annual Peaks',
    )
    if fitter.pilf_threshold > 0:
        ax.axhline(
            fitter.pilf_threshold,
            color='black',
            linestyle='--',
            label='PILF Threshold',
        )

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Peak Annual Discharge')
    ax.set_xlabel('Return Period (Years)')
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()
    return fig, ax


def hydrograph_plot(
    hydrograph_series: pd.Series,
    return_period_values: dict[float, float],
    peaks_series: pd.Series | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show: bool = False,  # noqa: FBT001, FBT002
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a streamflow hydrograph overlaid with return period thresholds.

    Args:
        hydrograph_series: Time-indexed `pd.Series` of daily streamflow.
        return_period_values: Mapping from return period (years) to discharge.
        peaks_series: Optional `pd.Series` of annual maximums indexed by year.
        ax: Optional Matplotlib `Axes` to draw on.
        show: Whether to call `plt.show()`.

    Returns:
        Tuple of `(fig, ax)`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 6))
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    hydrograph_series.plot(ax=ax, label='Streamflow')

    if peaks_series is not None:
        peak_dates = [
            pd.to_datetime(f'{year}-06-15') for year in peaks_series.index
        ]
        ax.scatter(
            peak_dates,
            peaks_series.to_numpy(),
            color='red',
            zorder=5,
            label='Annual Peaks',
        )

    x_limits = ax.get_xlim()
    colors = ['cyan', 'orange', 'green', 'purple', 'brown', 'pink']
    for idx, (rp, flow) in enumerate(return_period_values.items()):
        color = colors[idx % len(colors)]
        ax.hlines(
            flow,
            xmin=x_limits[0],
            xmax=x_limits[1],
            color=color,
            linestyle='--',
            label=f'{rp}-yr ({flow:.2f})',
        )

    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    ax.set_ylabel('Streamflow')

    if show:
        plt.show()
    return fig, ax
