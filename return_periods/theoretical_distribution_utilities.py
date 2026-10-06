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

"""Utilities for working with the Pearson Type III distribution.

Implements moment-to-parameter conversions, cumulative distribution functions,
quantile (inverse CDF) functions, unbiased sample moment estimators, and the
SimpleLogPearson3Fitter following USGS Bulletin 17C (England et al., 2019):
https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""

from collections.abc import Sequence

import numpy as np
from scipy import special as scipy_special

from return_periods import base_fitter, exceptions

# Minimum absolute skew magnitude to avoid 4 / skew^2 singularity in Pearson III
_MIN_ABS_SKEW = 1e-4


def pearson3_parameters_from_moments(
    moments: Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert mean, standard deviation, and skew into Pearson III parameters.

    Follows the parameterization in USGS Bulletin 17C Appendix 7 (p. 79):
        alpha = 4 / skew^2
        beta = 0.5 * std * skew
        tau = mean - 2 * std / skew

    Args:
        moments: Sequence of three floats `(mean, std, skew)`.

    Returns:
        Tuple of `(tau, alpha, beta)` corresponding to the location bound,
        shape, and scale parameters of the Pearson Type III distribution.

    Raises:
        ValueError: If `moments` does not contain exactly 3 elements.
        DistributionParameterError: If `std <= 0`, `skew == 0`, or any moment
            is non-finite.
    """
    if len(moments) != 3:  # noqa: PLR2004
        raise ValueError(
            '3 moments (mean, standard deviation, skew) are required to '
            'parameterize a Pearson Type III distribution.'
        )

    mean, std, skew = (
        float(moments[0]),
        float(moments[1]),
        float(moments[2]),
    )

    if any(not np.isfinite(m) for m in (mean, std, skew)):
        raise exceptions.DistributionParameterError(
            'Non-finite value in moments.'
        )
    if std <= 0:
        raise exceptions.DistributionParameterError(
            f'Standard deviation must be > 0, got {std}.'
        )
    if skew == 0:
        raise exceptions.DistributionParameterError(
            f'Skew must be != 0 for a Pearson III distribution, got {skew}.'
        )

    alpha = 4.0 / (skew**2)
    beta = 0.5 * std * skew
    tau = mean - 2.0 * std / skew

    return tau, alpha, beta


def pearson3_cdf(
    values: Sequence[float] | np.ndarray,
    moments: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Evaluate the Pearson Type III cumulative distribution function.

    Values outside the semi-infinite support of the Pearson Type III
    distribution (`values <= tau` when `beta > 0`, or `values >= tau` when
    `beta < 0`) are clamped to `0.0` and `1.0` respectively instead of
    returning `NaN`.

    Args:
        values: Points at which to evaluate the CDF.
        moments: Sequence of `(mean, std, skew)`.

    Returns:
        1D NumPy array of cumulative non-exceedance probabilities in `[0, 1]`.
    """
    tau, alpha, beta = pearson3_parameters_from_moments(moments)
    arr = np.asarray(values, dtype=float)
    scaled = np.maximum(0.0, (arr - tau) / beta)
    if beta > 0:
        return np.asarray(scipy_special.gammainc(alpha, scaled), dtype=float)
    return np.asarray(scipy_special.gammaincc(alpha, scaled), dtype=float)


def pearson3_invcdf(
    percentiles: Sequence[float] | np.ndarray,
    moments: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Evaluate the inverse CDF (quantile function) of a Pearson III model.

    Args:
        percentiles: Cumulative non-exceedance probabilities in `(0, 1)`.
        moments: Sequence of `(mean, std, skew)`.

    Returns:
        1D NumPy array of quantile values corresponding to `percentiles`.

    Raises:
        ValueError: If any percentile is outside `(0, 1)`.
    """
    probs = np.asarray(percentiles, dtype=float)
    if np.any(probs <= 0) or np.any(probs >= 1) or np.any(~np.isfinite(probs)):
        raise ValueError('All percentiles must be strictly in (0, 1).')

    tau, alpha, beta = pearson3_parameters_from_moments(moments)

    if beta > 0:
        gamma_quantiles = scipy_special.gammaincinv(alpha, probs)
    else:
        gamma_quantiles = scipy_special.gammainccinv(alpha, probs)
    return np.asarray(gamma_quantiles * beta + tau, dtype=float)


def pearson3_pmf(
    bin_edges: Sequence[float] | np.ndarray,
    moments: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Calculate discrete probability mass within consecutive bin intervals.

    Args:
        bin_edges: Monotonically increasing sequence of bin boundary values.
        moments: Sequence of `(mean, std, skew)`.

    Returns:
        1D NumPy array of length `len(bin_edges) - 1` containing the
        probability mass within each bin interval.
    """
    cdf = pearson3_cdf(values=bin_edges, moments=moments)
    return np.diff(cdf)


def sample_moments(
    data: Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Calculate unbiased sample moments following USGS Bulletin 17C Eq. 5-7.

    Specifically:
        mean = (1 / n) * sum(X_i)
        std = sqrt((1 / (n - 1)) * sum((X_i - mean)^2))
        skew = (n / ((n - 1) * (n - 2))) * sum(((X_i - mean) / std)^3)

    Args:
        data: Sequence or 1D array of observations (at least 3 values).

    Returns:
        Tuple of `(mean, std, skew)`.

    Raises:
        InvalidFlowValueError: If any observation is NaN or infinite.
        NotEnoughDataError: If fewer than 3 observations are provided.
    """
    arr = np.asarray(data, dtype=float).ravel()
    if np.any(~np.isfinite(arr)):
        raise exceptions.InvalidFlowValueError(
            'Observations passed to sample_moments must be finite.'
        )
    n = len(arr)
    if n < 3:  # noqa: PLR2004
        raise exceptions.NotEnoughDataError(
            f'At least 3 observations are required to compute skew; got {n}.'
        )

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return mean, 0.0, 0.0
    skew = float((n / ((n - 1) * (n - 2))) * np.sum(((arr - mean) / std) ** 3))
    return mean, std, skew


def bulletin17b_station_skew_mse(
    record_length: int,
    station_skew: float,
) -> float:
    """Approximate the MSE of station skew using Bulletin 17B Equation 6.

    Args:
        record_length: Number of annual peak observations `n`.
        station_skew: Station skew coefficient `G`.

    Returns:
        Estimated mean square error of the station skew `MSE_G`.
    """
    abs_g = abs(float(station_skew))
    a = (
        -0.33 + 0.08 * abs_g
        if abs_g <= 0.90  # noqa: PLR2004
        else -0.52 + 0.30 * abs_g
    )
    b = 0.94 - 0.26 * abs_g if abs_g <= 1.50 else 0.55  # noqa: PLR2004

    return float(10.0 ** (a - b * np.log10(record_length / 10.0)))


def weighted_skew(
    station_skew: float,
    station_skew_mse: float,
    regional_skew: float,
    regional_skew_mse: float,
) -> float:
    """Compute generalized weighted skew per Bulletin 17C Equation 10 / 7-10.

    Args:
        station_skew: At-site skew estimate `G`.
        station_skew_mse: Mean square error of the at-site skew `MSE_G`.
        regional_skew: Regional skew estimate `G_bar`.
        regional_skew_mse: Mean square error of the regional skew `MSE_G_bar`.

    Returns:
        Variance-weighted skew coefficient `G_w`.

    Raises:
        ValueError: If either MSE is non-positive.
    """
    if station_skew_mse <= 0 or regional_skew_mse <= 0:
        raise ValueError('Skew MSE values must be strictly positive.')
    return float(
        (regional_skew_mse * station_skew + station_skew_mse * regional_skew)
        / (regional_skew_mse + station_skew_mse)
    )


class SimpleLogPearson3Fitter(base_fitter.BaseFitter):
    """Fits a Pearson Type III distribution directly from sample moments.

    When zero-flow years (`num_zero_flows > 0`) are present in the record,
    applies the Jennings and Benson (1969) / USGS Bulletin 17B Appendix 5
    conditional probability adjustment
    `p_uncond(q) = (N_pos / N_total) * p_cond(q)`.
    """

    def __init__(
        self,
        data: Sequence[float] | np.ndarray,
        log_transform: bool = True,  # noqa: FBT001, FBT002
        regional_skew: float | None = None,
        regional_skew_mse: float | None = None,
    ):
        """Fit a Log-Pearson Type III distribution using method of moments.

        Args:
            data: Sequence of annual peak flow observations.
            log_transform: Whether to log10-transform flow data before fitting.
            regional_skew: Optional regional skew coefficient `G_bar`.
            regional_skew_mse: Optional MSE of the regional skew `MSE_G_bar`.

        Raises:
            ValueError: If only one of `regional_skew` or `regional_skew_mse`
                is provided.
            DistributionParameterError: If the positive observations have zero
                variance.
        """
        if (regional_skew is None) != (regional_skew_mse is None):
            raise ValueError(
                'regional_skew and regional_skew_mse must both be provided or '
                'both be None.'
            )
        super().__init__(data=data, log_transform=log_transform)
        mean, std, skew = sample_moments(self.transformed_sample)
        if std <= 0.0:
            raise exceptions.DistributionParameterError(
                f'Standard deviation must be > 0, got {std}.'
            )
        if regional_skew is not None and regional_skew_mse is not None:
            mse_g = bulletin17b_station_skew_mse(
                record_length=self.record_length,
                station_skew=skew,
            )
            skew = weighted_skew(
                station_skew=skew,
                station_skew_mse=mse_g,
                regional_skew=regional_skew,
                regional_skew_mse=regional_skew_mse,
            )
        if abs(skew) < _MIN_ABS_SKEW:
            skew = _MIN_ABS_SKEW if skew >= 0 else -_MIN_ABS_SKEW
        self.moments: tuple[float, float, float] = (mean, std, skew)

    def exceedance_probabilities_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate annual exceedance probabilities for given flow values."""
        arr = self._validate_flow_array(flows)
        p_pos = self.record_length / self.total_record_length
        pos_mask = (
            arr > 0.0 if self._log_transform else np.ones_like(arr, dtype=bool)
        )
        exceedance_probs = np.full(arr.shape, p_pos, dtype=float)
        if np.any(pos_mask):
            transformed_flows = self._transform_data(arr[pos_mask])
            cdf = pearson3_cdf(
                values=transformed_flows,
                moments=self.moments,
            )
            exceedance_probs[pos_mask] = p_pos * (1.0 - cdf)
        # Clamp strictly inside (0, 1) to prevent division by zero when
        # converting exceedance probabilities to return periods.
        return np.clip(
            exceedance_probs,
            base_fitter._EPSILON,  # noqa: SLF001
            1.0 - base_fitter._EPSILON,  # noqa: SLF001
        )

    def flow_values_from_exceedance_probabilities(
        self,
        exceedance_probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate flow quantiles for given exceedance probabilities."""
        probs = self._check_exceedance_probabilities(exceedance_probabilities)
        p_pos = self.record_length / self.total_record_length
        cond_probs = probs / p_pos

        result = np.zeros_like(probs, dtype=float)
        positive_mask = cond_probs < 1.0
        if np.any(positive_mask):
            transformed_flows = pearson3_invcdf(
                percentiles=1.0 - cond_probs[positive_mask],
                moments=self.moments,
            )
            result[positive_mask] = self._untransform_data(transformed_flows)
        return result
