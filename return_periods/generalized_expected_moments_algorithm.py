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

"""Expected Moments Algorithm (EMA) for Log-Pearson Type III flood frequency.

Implements the Generalized Expected Moments Algorithm (Cohn et al., 1997) with
Multiple Grubbs-Beck Test (MGBT) low-outlier screening following USGS Bulletin
17C Appendix 7 (England et al., 2019, pp. 79-85):
https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""

from collections.abc import Sequence

import numpy as np
from scipy import special as scipy_special

from return_periods import (
    base_fitter,
    exceptions,
    grubbs_beck_tester,
    theoretical_distribution_utilities,
)

# Convergence tolerance on the L-infinity change in Pearson III moments.
_CONVERGENCE_TOLERANCE = 1e-10

# Maximum number of EMA outer iterations before stopping.
_MAX_ITERATIONS = 1000

# Minimum absolute skew magnitude to avoid 4 / skew^2 singularity in Pearson III
_MIN_ABS_SKEW = 1e-4

# Minimum negative skew bound (emafit.f line 1516 sk141 = -1.41d0) ensuring
# shape parameter alpha = 4 / skew^2 >= 2 and preventing EM divergence on
# heavily left-censored negatively skewed samples.
_MIN_SKEW = -1.41

# Small-skew transition bounds where the Wilson-Hilferty expansion prevents
# float64 catastrophic cancellation in tau^3 = (mu - 2*sigma/skew)^3.
_SMALL_SKEW_LOWER = 0.005
_SMALL_SKEW_UPPER = 0.010


def _wilson_hilferty_standardized_moments(
    z_low: np.ndarray,
    z_high: np.ndarray,
    skew: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute standardized Pearson III interval moments via Wilson-Hilferty.

    Evaluates E[Z^k | z_low <= Z <= z_high] for k = 1, 2, 3 where Z = (X - mu) /
    sigma using the exact Wilson-Hilferty cube-root transformation from USGS
    PeakFQ `mP3` and `whlp2z` (`emafit.f` lines 3067-3144), without the 0.0007
    transition typo that causes float64 catastrophic cancellation in `tau^3`.
    """
    g = float(skew)
    z_l_arr = np.asarray(z_low, dtype=float)
    z_h_arr = np.asarray(z_high, dtype=float)

    if abs(g) < 1e-12:  # noqa: PLR2004
        y_l, y_u = np.clip(z_l_arr, -1e20, 1e20), np.clip(z_h_arr, -1e20, 1e20)
    else:
        z_l_c = np.clip(z_l_arr, -1e20, 1e20)
        z_h_c = np.clip(z_h_arr, -1e20, 1e20)
        c_l = np.maximum(0.0, 1.0 + 0.5 * g * z_l_c) ** (1.0 / 3.0)
        c_h = np.maximum(0.0, 1.0 + 0.5 * g * z_h_c) ** (1.0 / 3.0)
        y_l = np.clip((6.0 / g) * ((g**2) / 36.0 - 1.0 + c_l), -1e20, 1e20)
        y_u = np.clip((6.0 / g) * ((g**2) / 36.0 - 1.0 + c_h), -1e20, 1e20)

    ylo, yhi = np.minimum(y_l, y_u), np.maximum(y_l, y_u)

    def _phi(y: np.ndarray) -> np.ndarray:
        res = np.zeros_like(y, dtype=float)
        mask = np.abs(y) < 40.0  # noqa: PLR2004
        res[mask] = np.exp(-0.5 * (y[mask] ** 2)) / np.sqrt(2.0 * np.pi)
        return res

    use_upper = ylo > 0.0
    p0 = np.maximum(
        np.where(
            use_upper,
            scipy_special.ndtr(-ylo) - scipy_special.ndtr(-yhi),
            scipy_special.ndtr(yhi) - scipy_special.ndtr(ylo),
        ),
        0.0,
    )
    phi_l, phi_u = _phi(ylo), _phi(yhi)

    valid = p0 > 1e-300  # noqa: PLR2004
    safe_p0 = np.where(valid, p0, 1.0)
    zl1 = np.where(scipy_special.ndtr(ylo) > 0.0, ylo, 0.0)
    zu1 = np.where(scipy_special.ndtr(yhi) < 1.0, yhi, 0.0)

    ey = np.zeros((10, len(ylo)), dtype=float)
    ey[0] = 1.0
    ey[1] = np.where(valid, (-phi_u + phi_l) / safe_p0, 0.0)
    for i in range(2, 10):
        boundary = -(zu1 ** (i - 1)) * phi_u + (zl1 ** (i - 1)) * phi_l
        ey[i] = np.where(valid, boundary / safe_p0 + (i - 1) * ey[i - 2], 0.0)

    # Wilson-Hilferty series expansion coefficients (emafit.f lines 3089-3092):
    a0 = -g * (3888.0 - 108.0 * (g**2) + (g**4)) / 23328.0
    a1 = (1.0 - (g**2) / 36.0) ** 2
    a2 = (g / 6.0) * (1.0 - (g**2) / 36.0)
    a3 = (g**2) / 108.0
    poly = np.array([a0, a1, a2, a3], dtype=float)
    p2 = np.convolve(poly, poly)
    p3 = np.convolve(p2, poly)

    z1 = sum(poly[i] * ey[i] for i in range(len(poly)))
    z2 = sum(p2[i] * ey[i] for i in range(len(p2)))
    z3 = sum(p3[i] * ey[i] for i in range(len(p3)))

    # If cdfu == cdfl (~valid), use closest endpoint (emafit.f line 3020)
    if np.any(~valid):
        t_closest = np.where(
            np.abs(z_l_arr) < np.abs(z_h_arr), z_l_arr, z_h_arr
        )
        z1 = np.where(valid, z1, t_closest)
        z2 = np.where(valid, z2, t_closest**2)
        z3 = np.where(valid, z3, t_closest**3)

    return z1, z2, z3


def _regularized_gamma_interval(
    alpha: float,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Compute P(alpha, high) - P(alpha, low) without upper-tail cancellation.

    Uses the lower regularized incomplete gamma `gammainc` when `low < alpha`
    and the upper complement `gammaincc` when `low >= alpha`.
    """
    use_upper = low >= alpha
    diff = np.empty_like(low, dtype=float)
    if np.any(~use_upper):
        m = ~use_upper
        diff[m] = scipy_special.gammainc(alpha, high[m]) - (
            scipy_special.gammainc(alpha, low[m])
        )
    if np.any(use_upper):
        m = use_upper
        diff[m] = scipy_special.gammaincc(alpha, low[m]) - (
            scipy_special.gammaincc(alpha, high[m])
        )
    return np.maximum(diff, 0.0)


def _interval_gamma_moment_ratios(
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute conditional Gamma moment ratios on [lower, upper] in log space.

    For `Y ~ Gamma(alpha, 1)` truncated to `[lower, upper]`, returns:
        m1 = E[Y | lower <= Y <= upper]
        m2 = E[Y^2 | lower <= Y <= upper]
        m3 = E[Y^3 | lower <= Y <= upper]
    using the analytical identity:
        Gamma(alpha + k) / Gamma(alpha) = prod_{j=0}^{k-1} (alpha + j)
    so that `Gamma(alpha)` never overflows even when `alpha = 4 / skew^2` is
    very large (small skew).
    """
    low = np.maximum(0.0, np.asarray(lower, dtype=float))
    high = np.maximum(low, np.asarray(upper, dtype=float))

    d0 = _regularized_gamma_interval(alpha, low, high)
    d1 = _regularized_gamma_interval(alpha + 1.0, low, high)
    d2 = _regularized_gamma_interval(alpha + 2.0, low, high)
    d3 = _regularized_gamma_interval(alpha + 3.0, low, high)

    # Safe division: if d0 underflows at extreme tails, fall back to the
    # finite endpoint closest to the distribution bulk (or midpoint if both
    # bounds are finite).
    valid = d0 > 1e-300  # noqa: PLR2004
    safe_d0 = np.where(valid, d0, 1.0)
    fallback_y = np.where(np.isfinite(high), 0.5 * (low + high), low)

    r1 = np.where(valid, alpha * (d1 / safe_d0), fallback_y)
    r2 = np.where(
        valid,
        alpha * (alpha + 1.0) * (d2 / safe_d0),
        r1**2,
    )
    r3 = np.where(
        valid,
        alpha * (alpha + 1.0) * (alpha + 2.0) * (d3 / safe_d0),
        r1**3,
    )
    return r1, r2, r3


class GEMAFitter(base_fitter.BaseFitter):
    """Fits a Log-Pearson Type III distribution via Bulletin 17C EMA + MGBT."""

    def __init__(
        self,
        data: Sequence[float] | np.ndarray,
        log_transform: bool = True,  # noqa: FBT001, FBT002
        use_multiple_grubbs_beck: bool = True,  # noqa: FBT001, FBT002
        regional_skew: float | None = None,
        regional_skew_mse: float | None = None,
    ):
        """Initialize and fit the Expected Moments Algorithm (EMA) model.

        Args:
            data: Sequence of annual peak streamflow observations (may include
                zero flows, which are screened and censored as PILFs).
            log_transform: Whether to apply a base-10 logarithmic transform.
            use_multiple_grubbs_beck: If `True` (default), screens for low
                outliers using the official Bulletin 17C Multiple Grubbs-Beck
                Test (`MultipleGrubbsBeckTester`). If `False`, uses the legacy
                Bulletin 17B single-outlier `GrubbsBeckTester`.
            regional_skew: Optional generalized regional skew `G_bar` for
                weighted skew estimation (Bulletin 17C Equation 7-10).
            regional_skew_mse: Optional mean square error `MSE_G_bar` of the
                regional skew estimate.
        """
        super().__init__(data=data, log_transform=log_transform)
        if (regional_skew is None) != (regional_skew_mse is None):
            raise ValueError(
                'Both `regional_skew` and `regional_skew_mse` must be '
                'provided together, or both must be None.'
            )
        self._use_multiple_grubbs_beck = use_multiple_grubbs_beck
        self._regional_skew = regional_skew
        self._regional_skew_mse = regional_skew_mse

        self._set_data_intervals()
        self.iterations: int = 0
        self.moments: tuple[float, float, float] = self._fit_pearson3_moments()

    def exceedance_probabilities_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate annual exceedance probabilities for given flow values."""
        transformed_flows = self._transform_data(flows)
        cdf = theoretical_distribution_utilities.pearson3_cdf(
            values=transformed_flows,
            moments=self.moments,
        )
        cdf = np.clip(
            cdf,
            base_fitter._EPSILON,  # noqa: SLF001
            1.0 - base_fitter._EPSILON,  # noqa: SLF001
        )
        return 1.0 - cdf

    def flow_values_from_exceedance_probabilities(
        self,
        exceedance_probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate flow quantiles for given exceedance probabilities."""
        probs = self._check_exceedance_probabilities(exceedance_probabilities)
        transformed_flows = theoretical_distribution_utilities.pearson3_invcdf(
            percentiles=1.0 - probs,
            moments=self.moments,
        )
        return self._untransform_data(transformed_flows)

    def _set_data_intervals(self) -> None:
        """Screen for PILFs and construct EMA lower/upper flow intervals."""
        if self._use_multiple_grubbs_beck:
            tester: (
                grubbs_beck_tester.MultipleGrubbsBeckTester
                | grubbs_beck_tester.GrubbsBeckTester
            ) = grubbs_beck_tester.MultipleGrubbsBeckTester(
                data=self.transformed_sample,
                num_zero_flows=self.num_zero_flows,
                is_log_transformed=True,
            )
            num_pilfs = tester.klow
        else:
            tester = grubbs_beck_tester.GrubbsBeckTester(
                data=self.transformed_sample
            )
            num_pilfs = (
                len(tester.out_of_population_sample) + self.num_zero_flows
            )

        self.outlier_tester = tester
        in_pop = tester.in_population_sample

        if num_pilfs > 0:
            pilf_thresh = (
                float(tester.threshold)
                if np.isfinite(tester.threshold)
                else float(in_pop[0])
            )
            self._pilf_threshold = pilf_thresh
            pilf_lows = np.full(num_pilfs, -np.inf, dtype=float)
            pilf_highs = np.full(num_pilfs, pilf_thresh, dtype=float)
            self._interval_low_values = np.concatenate([pilf_lows, in_pop])
            self._interval_high_values = np.concatenate([pilf_highs, in_pop])
        else:
            self._pilf_threshold = None
            self._interval_low_values = in_pop.copy()
            self._interval_high_values = in_pop.copy()

        if np.any(self._interval_low_values > self._interval_high_values):
            raise exceptions.DataIntervalError(
                'Lower interval bound cannot exceed upper interval bound.'
            )

    def _fit_pearson3_moments(self) -> tuple[float, float, float]:
        """Solve the Bulletin 17C EMA moment equations iteratively."""
        in_pop = self.outlier_tester.in_population_sample
        moments = theoretical_distribution_utilities.sample_moments(in_pop)
        if moments[1] <= 0.0:
            raise exceptions.DistributionParameterError(
                'Cannot fit Pearson Type III distribution via EMA when the '
                'retained uncensored sample has zero variance.'
            )

        # Ensure initial skew is non-zero so Pearson III parameters are finite.
        if abs(moments[2]) < _MIN_ABS_SKEW:
            moments = (
                moments[0],
                moments[1],
                _MIN_ABS_SKEW if moments[2] >= 0 else -_MIN_ABS_SKEW,
            )

        for iterations in range(1, _MAX_ITERATIONS + 1):
            new_moments = self._update_moments_from_intervals_and_moments(
                moments=moments
            )
            max_diff = max(
                abs(a - b) for a, b in zip(moments, new_moments, strict=True)
            )
            moments = new_moments
            if max_diff <= _CONVERGENCE_TOLERANCE:
                self.iterations = iterations
                return moments

        raise exceptions.DistributionParameterError(
            f'EMA failed to converge within {_MAX_ITERATIONS} iterations '
            f'(max moment change {max_diff:.3e} > '
            f'{_CONVERGENCE_TOLERANCE:.3e}).'
        )

    def _standardized_interval_moments(
        self,
        x_low: np.ndarray,
        x_high: np.ndarray,
        moments: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute standardized interval moments E[Z^k | z_low <= Z <= z_high]."""  # noqa: E501
        mu = float(moments[0])
        sigma = float(moments[1])
        skew = float(moments[2])
        z_low = (x_low - mu) / sigma
        z_high = (x_high - mu) / sigma
        abs_g = abs(skew)

        if abs_g <= _SMALL_SKEW_UPPER:
            z1_wh, z2_wh, z3_wh = _wilson_hilferty_standardized_moments(
                z_low=z_low,
                z_high=z_high,
                skew=skew,
            )

        if abs_g > _SMALL_SKEW_LOWER:
            tau, alpha, beta = (
                theoretical_distribution_utilities.pearson3_parameters_from_moments(
                    moments
                )
            )
            if beta > 0:
                lower = np.maximum(0.0, (x_low - tau) / beta)
                upper = np.maximum(lower, (x_high - tau) / beta)
            else:
                lower = np.maximum(0.0, (x_high - tau) / beta)
                upper = np.maximum(lower, (x_low - tau) / beta)

            d0 = _regularized_gamma_interval(alpha, lower, upper)
            g1, g2, g3 = _interval_gamma_moment_ratios(
                lower=lower,
                upper=upper,
                alpha=alpha,
            )
            s = beta / sigma
            dy1 = g1 - alpha
            dy2 = g2 - 2.0 * alpha * g1 + alpha**2
            dy3 = g3 - 3.0 * alpha * g2 + 3.0 * (alpha**2) * g1 - alpha**3

            # When an interval lies completely outside the Pearson III support
            # (d0 == 0), use the endpoint closest to the mean (emafit.f 3019).
            valid_gam = d0 > 1e-300  # noqa: PLR2004
            z_closest = np.where(np.abs(z_low) < np.abs(z_high), z_low, z_high)
            z1_gam = np.where(valid_gam, s * dy1, z_closest)
            z2_gam = np.where(valid_gam, (s**2) * dy2, z_closest**2)
            z3_gam = np.where(valid_gam, (s**3) * dy3, z_closest**3)

        if abs_g <= _SMALL_SKEW_LOWER:
            return z1_wh, z2_wh, z3_wh
        if abs_g >= _SMALL_SKEW_UPPER:
            return z1_gam, z2_gam, z3_gam

        t = (abs_g - _SMALL_SKEW_LOWER) / (
            _SMALL_SKEW_UPPER - _SMALL_SKEW_LOWER
        )
        w = 0.5 * (1.0 - float(np.cos(np.pi * t)))
        return (
            w * z1_gam + (1.0 - w) * z1_wh,
            w * z2_gam + (1.0 - w) * z2_wh,
            w * z3_gam + (1.0 - w) * z3_wh,
        )

    def _update_moments_from_intervals_and_moments(
        self,
        moments: Sequence[float] | np.ndarray,
    ) -> tuple[float, float, float]:
        """Execute one EMA expectation-maximization moment update step.

        Implements USGS Bulletin 17C Equations 7-1 to 7-10 (p. 80) and
        Equations 7-12 to 7-17 (p. 81-82).
        """
        mu = float(moments[0])
        sigma = float(moments[1])

        exact_mask = self._interval_low_values == self._interval_high_values
        interval_mask = ~exact_mask

        n_total = float(len(self._interval_low_values))
        n_exact = int(np.sum(exact_mask))

        x_exact = self._interval_low_values[exact_mask]

        if np.any(interval_mask):
            z1, z2, z3 = self._standardized_interval_moments(
                x_low=self._interval_low_values[interval_mask],
                x_high=self._interval_high_values[interval_mask],
                moments=moments,
            )
            ex1_int = mu + sigma * z1
            ex_mean = float((np.sum(x_exact) + np.sum(ex1_int)) / n_total)

            # Central moments of interval observations around ex_mean:
            # X - ex_mean = sigma * Z + (mu - ex_mean)
            dm = mu - ex_mean
            ex2_int = (sigma**2) * z2 + 2.0 * sigma * dm * z1 + dm**2
            ex3_int = (
                (sigma**3) * z3
                + 3.0 * (sigma**2) * dm * z2
                + 3.0 * sigma * (dm**2) * z1
                + dm**3
            )
        else:
            ex_mean = float(np.mean(x_exact))
            ex2_int = np.empty(0, dtype=float)
            ex3_int = np.empty(0, dtype=float)

        # Bulletin 17C Equations 7-2 to 7-5: small-sample bias correction
        # factors c2 and c3 apply to the uncensored systematic observations.
        c2 = float(n_exact) / float(n_exact - 1) if n_exact > 1 else 1.0
        c3 = (
            float(n_exact**2) / float((n_exact - 1) * (n_exact - 2))
            if n_exact > 2  # noqa: PLR2004
            else 1.0
        )

        exact_dev = x_exact - ex_mean
        mu2 = float((c2 * np.sum(exact_dev**2) + np.sum(ex2_int)) / n_total)
        mu3 = float((c3 * np.sum(exact_dev**3) + np.sum(ex3_int)) / n_total)

        std = float(np.sqrt(max(mu2, 1e-30)))
        skew = float(mu3 / (std**3))

        if (
            self._regional_skew is not None
            and self._regional_skew_mse is not None
        ):
            mse_g = (
                theoretical_distribution_utilities.bulletin17b_station_skew_mse(
                    record_length=int(n_total),
                    station_skew=skew,
                )
            )
            skew = theoretical_distribution_utilities.weighted_skew(
                station_skew=skew,
                station_skew_mse=mse_g,
                regional_skew=self._regional_skew,
                regional_skew_mse=self._regional_skew_mse,
            )

        skew = max(skew, _MIN_SKEW)
        if abs(skew) < _MIN_ABS_SKEW:
            skew = _MIN_ABS_SKEW if skew >= 0 else -_MIN_ABS_SKEW

        return ex_mean, std, skew
