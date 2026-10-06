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

"""Utilities for working with empirical flood frequency distributions.

Implements empirical probability density histograms, standard plotting position
formulas (Weibull, Cunnane, Gringorten, Hazen, Blom, Median, APL), the
Hirsch-Stedinger (1987) threshold-exceedance plotting position estimator from
USGS Bulletin 17C Appendix 5, and a log-log linear trend fitter:
https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""

from collections.abc import Sequence

import numpy as np

from return_periods import base_fitter, exceptions

# Plotting position parameter `a` in p_i = (i - a) / (n + 1 - 2a) from
# USGS Bulletin 17C Table 5-1 (p. 76).
PLOTTING_POSITION_TYPES: dict[str, float] = {
    'weibull': 0.0,
    'median': 0.3175,
    'apl': 0.35,
    'blom': 0.375,
    'cunnane': 0.4,
    'gringorten': 0.44,
    'hazen': 0.5,
}


def empirical_pdf(
    data: Sequence[float] | np.ndarray,
    num_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate an empirical probability mass histogram over equal-width bins.

    Args:
        data: Sequence of observations.
        num_bins: Number of histogram bins (must be >= 1).

    Returns:
        Tuple of `(hist, bin_edges)` where `hist` sums to `1.0` and `bin_edges`
        has length `num_bins + 1`.

    Raises:
        ValueError: If `num_bins < 1` or `data` is empty or non-finite.
    """
    if num_bins < 1:
        raise ValueError(f'num_bins must be >= 1, got {num_bins}.')
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError('Cannot compute empirical_pdf on empty data.')
    if np.any(~np.isfinite(arr)):
        raise ValueError('data must contain only finite values.')

    hist, bin_edges = np.histogram(arr, bins=num_bins, density=False)
    hist = hist.astype(float) / float(np.sum(hist))
    return hist, bin_edges


def _get_plotting_alpha(plotting_position_type: str) -> float:
    """Validate and return the plotting position parameter `a`."""
    key = plotting_position_type.lower()
    if key not in PLOTTING_POSITION_TYPES:
        valid = ', '.join(sorted(PLOTTING_POSITION_TYPES))
        raise ValueError(
            f'Invalid plotting_position_type {plotting_position_type!r}. '
            f'Expected one of: {valid}.'
        )
    return PLOTTING_POSITION_TYPES[key]


def simple_empirical_plotting_position(
    data: Sequence[float] | np.ndarray,
    plotting_position_type: str = 'weibull',
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate empirical exceedance probabilities via Bulletin 17C Eq. 5-1.

    Sorts the observations in ascending order (smallest to largest flow) and
    assigns descending exceedance probabilities
    `p_i = (i - a) / (n + 1 - 2a)`, where `i = n` for the smallest flood and
    `i = 1` for the largest flood.

    Args:
        data: Sequence of peak flow observations.
        plotting_position_type: Name of plotting position formula in
            `PLOTTING_POSITION_TYPES`.

    Returns:
        Tuple of `(sorted_data, exceedance_probabilities)` where `sorted_data`
        is sorted ascending and `exceedance_probabilities` is sorted descending.

    Raises:
        ValueError: If `plotting_position_type` is unknown, `data` is empty,
            or `data` contains non-finite values.
    """
    alpha = _get_plotting_alpha(plotting_position_type)
    arr = np.asarray(data, dtype=float).ravel()
    record_length = len(arr)
    if record_length == 0:
        raise ValueError('Cannot compute plotting positions on empty data.')
    if np.any(~np.isfinite(arr)):
        raise ValueError('data must contain only finite values.')
    sorted_data = np.sort(arr)

    # Descending ranks: n for smallest flow (index 0), 1 for largest (index n-1)
    ranks = np.arange(record_length, 0, -1, dtype=float)
    exceedance_probs = (ranks - alpha) / (record_length + 1.0 - 2.0 * alpha)
    return sorted_data, exceedance_probs


def threshold_exceedance_empirical_plotting_position(
    data: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray | float,
    plotting_position_type: str = 'weibull',
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Hirsch-Stedinger threshold-exceedance plotting positions.

    Implements the Hirsch and Stedinger (1987) plotting position estimator
    described in USGS Bulletin 17C Appendix 5 (Equations 5-2 to 5-8, pp. 76-78)
    for records with Potentially Influential Low Flood (PILF) or historical
    perception thresholds.

    Args:
        data: Sequence of peak flow observations of length `n`.
        thresholds: Either a scalar perception/PILF threshold `T` applied across
            the record, or a sequence of per-year perception lower thresholds
            of length `n`.
        plotting_position_type: Name of plotting position formula in
            `PLOTTING_POSITION_TYPES`.

    Returns:
        Tuple of `(sorted_data, exceedance_probabilities)` where `sorted_data`
        is sorted ascending and `exceedance_probabilities` is the corresponding
        descending array of exceedance probabilities in `(0, 1)`.

    Raises:
        ValueError: If `data` is empty, contains non-finite values, or
            `thresholds` has an incompatible length or contains NaN.
    """
    alpha = _get_plotting_alpha(plotting_position_type)
    arr = np.asarray(data, dtype=float).ravel()
    n = len(arr)
    if n == 0:
        raise ValueError('Cannot compute plotting positions on empty data.')
    if np.any(~np.isfinite(arr)):
        raise ValueError('data must contain only finite values.')

    thresh_arr = np.asarray(thresholds, dtype=float)
    if np.any(np.isnan(thresh_arr)):
        raise ValueError('thresholds cannot contain NaN values.')
    if thresh_arr.ndim == 0 or thresh_arr.size == 1:
        thresh_per_obs = np.full(n, float(thresh_arr.item()), dtype=float)
    else:
        thresh_per_obs = thresh_arr.ravel()
        if len(thresh_per_obs) != n:
            raise ValueError(
                f'Length of thresholds ({len(thresh_per_obs)}) must match '
                f'length of data ({n}).'
            )

    order = np.argsort(arr, kind='mergesort')
    sorted_data = arr[order]
    inv_order = np.argsort(order)

    unique_thresholds = np.sort(np.unique(thresh_per_obs))[::-1]
    bounds = np.concatenate([unique_thresholds, [-np.inf]])

    exceedance_probs = np.empty(n, dtype=float)
    assigned = np.zeros(n, dtype=bool)
    p_cumulative = 0.0

    for idx in range(len(bounds)):
        t_low = bounds[idx]
        t_high = np.inf if idx == 0 else bounds[idx - 1]

        perceptible_mask = (
            thresh_per_obs <= t_low
            if np.isfinite(t_low)
            else np.ones(n, dtype=bool)
        )
        eligible_years = int(np.sum(perceptible_mask & (~assigned[inv_order])))
        in_band_mask = (
            (~assigned) & (sorted_data >= t_low) & (sorted_data < t_high)
        )
        k_band = int(np.sum(in_band_mask))

        if k_band == 0:
            continue

        if not np.isfinite(t_low) or eligible_years <= 0:
            p_next = 1.0
        else:
            n_denom = max(eligible_years, k_band)
            cond_prob = float(k_band) / float(n_denom)
            p_next = p_cumulative + (1.0 - p_cumulative) * cond_prob

        ranks_in_band = np.arange(k_band, 0, -1, dtype=float)
        rel_positions = (ranks_in_band - alpha) / (k_band + 1.0 - 2.0 * alpha)
        exceedance_probs[in_band_mask] = (
            p_cumulative + (p_next - p_cumulative) * rel_positions
        )

        assigned[in_band_mask] = True
        p_cumulative = p_next

    return sorted_data, exceedance_probs


class LogLogTrendFitter(base_fitter.BaseFitter):
    """Fits a log-log linear trend to empirical exceedance probabilities."""

    def __init__(
        self,
        data: Sequence[float] | np.ndarray,
        plotting_position_type: str = 'weibull',
        log_transform: bool = True,  # noqa: FBT001, FBT002
    ):
        """Fit a log-log linear regression to empirical plotting positions.

        Args:
            data: Sequence of peak flow values.
            plotting_position_type: Name of plotting position formula in
                `PLOTTING_POSITION_TYPES`.
            log_transform: Whether to log10-transform flow data.
        """
        super().__init__(data=data, log_transform=log_transform)

        sorted_sample, exceedance_probabilities = (
            simple_empirical_plotting_position(
                data=self.sample,
                plotting_position_type=plotting_position_type,
            )
        )
        transformed_sorted = self._transform_data(sorted_sample)
        if float(np.std(transformed_sorted)) <= 0.0:
            raise exceptions.DistributionParameterError(
                'Cannot fit LogLogTrendFitter on constant data with zero '
                'variance.'
            )
        y_data = np.log10(-np.log10(exceedance_probabilities))

        slope, intercept = np.polyfit(transformed_sorted, y_data, deg=1)
        self.slope: float = float(slope)
        self.intercept: float = float(intercept)

    def exceedance_probabilities_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate exceedance probabilities from the log-log linear fit.

        When the record contains zero flows (`num_zero_flows > 0`), applies the
        Jennings and Benson (1969) / USGS Bulletin 17B Appendix 5 conditional
        probability adjustment `p_uncond(q) = (N_pos / N_total) * p_cond(q)`.
        """
        arr = self._validate_flow_array(flows)
        p_pos = float(self.record_length) / float(self.total_record_length)
        pos_mask = (
            arr > 0.0 if self._log_transform else np.ones_like(arr, dtype=bool)
        )
        uncond_exceedance = np.full(arr.shape, p_pos, dtype=float)
        if np.any(pos_mask):
            transformed_flows = self._transform_data(arr[pos_mask])
            y_values = self.intercept + self.slope * transformed_flows
            cond_exceedance = np.power(10.0, -np.power(10.0, y_values))
            uncond_exceedance[pos_mask] = p_pos * cond_exceedance
        return np.clip(
            uncond_exceedance,
            base_fitter._EPSILON,  # noqa: SLF001
            1.0 - base_fitter._EPSILON,  # noqa: SLF001
        )

    def flow_values_from_exceedance_probabilities(
        self,
        exceedance_probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate flow quantiles from the log-log linear fit.

        When the record contains zero flows (`num_zero_flows > 0`), applies the
        Jennings and Benson (1969) / USGS Bulletin 17B Appendix 5 conditional
        probability adjustment (`p_cond = p_uncond / (N_pos / N_total)`),
        returning `0.0` for exceedance probabilities `>= N_pos / N_total`.
        """
        probs = self._check_exceedance_probabilities(exceedance_probabilities)
        p_pos = float(self.record_length) / float(self.total_record_length)
        cond_probs = probs / p_pos
        valid_mask = cond_probs < 1.0
        flows = np.zeros_like(probs, dtype=float)
        if np.any(valid_mask):
            y_values = np.log10(-np.log10(cond_probs[valid_mask]))
            transformed_flows = (y_values - self.intercept) / self.slope
            flows[valid_mask] = self._untransform_data(transformed_flows)
        return flows
