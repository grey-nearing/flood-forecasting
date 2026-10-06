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

"""Single and Multiple Grubbs-Beck tests for low-outlier (PILF) screening.

Implements:

1. `MultipleGrubbsBeckTester`: The official USGS Bulletin 17C Appendix 6
   Multiple Grubbs-Beck Test (MGBT) of Cohn et al. (2013), Water Resources
   Research 49(8), 5047-5058, using orthogonal-t Gaussian quadrature over order
   statistics.
2. `GrubbsBeckTester`: The legacy USGS Bulletin 17B Appendix 4 Grubbs-Beck
   critical-value table test.

References:
    - England, J.F., Jr., et al. (2019), Guidelines for determining flood flow
      frequency—Bulletin 17C: U.S. Geological Survey Techniques and Methods,
      book 4, chap. B5, https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
    - Cohn, T.A., et al. (2013), A generalized Grubbs-Beck test statistic for
      detecting multiple potentially influential low outliers in flood series,
      Water Resources Research, 49(8), 5047-5058.
"""

import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import special as scipy_special
from scipy import stats as scipy_stats

from return_periods import exceptions

# Path to the Bulletin 17B K_n lookup table bundled in this package.
_KN_TABLE_PATH = (
    pathlib.Path(__file__).resolve().parent / 'bulletin17b_kn_table.csv'
)

# Sample size bounds for the Bulletin 17B K_n critical value table.
_MIN_B17B_SAMPLE_SIZE = 10
_MAX_B17B_SAMPLE_SIZE = 149

# Minimum positive sample size for MGBT order-statistic evaluation.
_MIN_MGBT_SAMPLE_SIZE = 5

# Number of Gauss-Legendre quadrature nodes for orthogonal-t p-value evaluation.
_MGBT_QUADRATURE_NODES = 128
_QUAD_EPS = float(np.sqrt(np.finfo(float).eps))
_GL_NODES, _GL_WEIGHTS = scipy_special.roots_legendre(_MGBT_QUADRATURE_NODES)
_GL_U = 0.5 * (1.0 - 2.0 * _QUAD_EPS) * _GL_NODES + 0.5
_GL_W = 0.5 * (1.0 - 2.0 * _QUAD_EPS) * _GL_WEIGHTS


def _load_kn_table() -> dict[int, float]:
    """Load the Bulletin 17B critical value table from disk."""
    kn_df = pd.read_csv(_KN_TABLE_PATH).set_index('Sample Size')
    return {int(idx): float(val) for idx, val in kn_df['KN Value'].items()}


def _mgbt_pvalues_vectorized(
    n: int,
    r_indices: np.ndarray,
    omega_values: np.ndarray,
) -> np.ndarray:
    """Evaluate MGBT p-values p(r; omega_r, n) via 2D Gaussian quadrature.

    Integrates the conditional noncentral Student's t distribution over the
    Beta(r, n + 1 - r) distribution of the r-th uniform order statistic
    following Cohn et al. (2013) Equations 11-18 and the USGS `MGBT` package
    (`peta.R`, `EMS.R`, `CondMoms.R`).

    Args:
        n: Total sample size.
        r_indices: 1D integer array of 1-based order-statistic indices `r`.
        omega_values: 1D float array of generalized Grubbs-Beck statistics
            `omega_r` corresponding to `r_indices`.

    Returns:
        1D float array of p-values in `[0, 1]`.
    """
    k_count = len(r_indices)
    if k_count == 0:
        return np.empty(0, dtype=float)

    r_col = r_indices[:, np.newaxis].astype(float)
    b_col = (n + 1 - r_indices)[:, np.newaxis].astype(float)
    w_col = omega_values[:, np.newaxis].astype(float)

    # Map Gauss-Legendre nodes u in [eps, 1 - eps] through inverse Beta CDF.
    br = scipy_special.betaincinv(r_col, b_col, _GL_U[np.newaxis, :])
    zr = scipy_special.ndtri(br)

    k_col = (n - r_indices)[:, np.newaxis].astype(float)

    # Evaluate conditional normal moments above zr on all (k_count * n_nodes).
    ratio = np.exp(scipy_stats.norm.logpdf(zr) - scipy_stats.norm.logsf(zr))
    e1 = ratio
    e2 = 1.0 + ratio * zr
    e3 = 2.0 * e1 + ratio * (zr**2)
    e4 = 3.0 * e2 + ratio * (zr**3)

    # Central moments above zr (EMS.R: V(n, r, qmin)).
    cm2 = e2 - e1**2
    cm3 = e3 - 3.0 * e2 * e1 + 2.0 * (e1**3)
    cm4 = e4 - 4.0 * e3 * e1 + 6.0 * e2 * (e1**2) - 3.0 * (e1**4)

    # Covariance matrix V of (M, S^2) for sample size k = n - r (probfun.f
    # line 1735 and EMS.R line 84):
    v11 = cm2 / k_col
    v12 = cm3 / np.sqrt(k_col * (k_col - 1.0))
    v22 = (cm4 - cm2**2) / k_col + (2.0 / (k_col * (k_col - 1.0))) * (cm2**2)

    # CondMomsChi2(n, r, zr) and expected standard deviation Es (EMS.R).
    moms2_1 = cm2
    moms2_2 = v22
    alpha = (moms2_1**2) / moms2_2
    beta = moms2_2 / moms2_1
    es = np.sqrt(beta) * np.exp(
        scipy_special.gammaln(alpha + 0.5) - scipy_special.gammaln(alpha)
    )

    # Covariance matrix VMS of (M, S) (EMS.R).
    cov11 = v11
    cov12 = v12 / (2.0 * es)
    cov22 = moms2_1 - es**2

    # Orthogonal-t parameters (peta.R).
    lamb = cov12 / cov22
    etap = w_col + lamb
    mu_mp = e1 - lamb * es
    sigma_sq = cov11 - (cov12**2) / cov22
    sigma_mp = np.sqrt(np.maximum(sigma_sq, 1e-30))

    q_stat = -(np.sqrt(moms2_1) / sigma_mp) * etap
    df = 2.0 * alpha
    ncp = (mu_mp - zr) / sigma_mp

    # Match USGS PeakFQ FP_TNC_CDF (probfun.f lines 1341-1345): use the
    # Abramowitz & Stegun (1964, p. 949) Eq. 26.7.10 normal approximation when
    # degrees of freedom nu = 2 * alpha > 20, and exact noncentral t otherwise.
    res = np.empty_like(q_stat, dtype=float)
    m_gt20 = df > 20.0  # noqa: PLR2004
    if np.any(m_gt20):
        q_m = q_stat[m_gt20]
        df_m = df[m_gt20]
        ncp_m = ncp[m_gt20]
        z_approx = (q_m * (1.0 - 1.0 / (4.0 * df_m)) - ncp_m) / np.sqrt(
            1.0 + (q_m**2) / (2.0 * df_m)
        )
        res[m_gt20] = scipy_special.ndtr(-z_approx)
    if np.any(~m_gt20):
        m_le20 = ~m_gt20
        res[m_le20] = scipy_stats.nct.sf(
            q_stat[m_le20], df=df[m_le20], nc=ncp[m_le20]
        )

    res = np.where(
        (sigma_sq <= 0.0) | (~np.isfinite(sigma_mp)) | np.isnan(res),
        1.0,
        res,
    )
    p_vals = res @ _GL_W
    return np.clip(p_vals, 0.0, 1.0)


def mgbt_order_statistic_pvalue(
    n: int,
    r: int,
    omega: float,
) -> float:
    """Compute the MGBT p-value for a single order statistic (r, omega, n)."""
    res = _mgbt_pvalues_vectorized(
        n=n,
        r_indices=np.asarray([r], dtype=int),
        omega_values=np.asarray([omega], dtype=float),
    )
    return float(res[0])


def _prepare_mgbt_samples(
    data: Sequence[float] | np.ndarray,
    num_zero_flows: int,
    is_log_transformed: bool,  # noqa: FBT001
) -> tuple[np.ndarray, int, int]:
    """Validate and sort MGBT positive log10 observations and zero count."""
    arr = np.asarray(data, dtype=float).ravel()
    if not is_log_transformed:
        if np.any(arr < 0) or np.any(~np.isfinite(arr)):
            raise exceptions.InvalidFlowValueError(
                'Raw flow values cannot be negative, NaN, or infinite.'
            )
        zero_count = int(np.sum(arr == 0)) + num_zero_flows
        sorted_positive = np.sort(np.log10(arr[arr > 0]))
    else:
        if np.any(~np.isfinite(arr)):
            raise exceptions.InvalidFlowValueError(
                'Log-transformed flow values cannot be NaN or infinite.'
            )
        zero_count = num_zero_flows
        sorted_positive = np.sort(arr)

    n_total = zero_count + len(sorted_positive)
    if len(sorted_positive) < _MIN_MGBT_SAMPLE_SIZE:
        raise exceptions.NotEnoughDataError(
            f'MGBT requires at least {_MIN_MGBT_SAMPLE_SIZE} positive '
            f'observations; got {len(sorted_positive)}.'
        )
    return sorted_positive, zero_count, n_total


def _compute_omega_statistics(
    sorted_positive: np.ndarray,
    zero_count: int,
    n_total: int,
) -> np.ndarray:
    """Compute generalized Grubbs-Beck statistics for r = 1..floor(n/2).

    Evaluates the exact analytical limit as the `Z = zero_count` zero flows
    approach `0+` (`log10(Q) = -M -> -inf`):
    - For `1 <= r < Z`, `Z - r` zeros remain in the upper sample alongside
      `n - Z` positive flows, so `omega_r` converges to the exact, scale-
      invariant limit `-sqrt((n - Z) * (n - r - 1) / ((n - r) * (Z - r)))`.
    - For `r = Z`, the upper sample contains only positive flows while
      `X_(Z) -> -inf`, so `omega_Z = -inf` (`p_Z = 0`).
    - For `r > Z`, `omega_r` is computed purely from the positive log10 flows.
    """
    n2 = n_total // 2
    omega_stats = np.empty(n2, dtype=float)
    for idx in range(n2):
        r = idx + 1
        if r < zero_count:
            omega_stats[idx] = -float(
                np.sqrt(
                    ((n_total - zero_count) * (n_total - r - 1))
                    / ((n_total - r) * (zero_count - r))
                )
            )
        elif r == zero_count:
            omega_stats[idx] = -np.inf
        else:
            pos_idx = idx - zero_count
            upper = sorted_positive[pos_idx + 1 :]
            mean_upper = float(np.mean(upper))
            std_upper = float(np.std(upper, ddof=1))
            # When std_upper == 0.0 (e.g., identical capped values at the top
            # of a record), censoring index r would leave the retained upper
            # sample with zero variance; setting omega_r = 0.0 prevents MGBT
            # from censoring observations into a zero-variance retained sample.
            omega_stats[idx] = (
                0.0
                if std_upper <= 0.0
                else (float(sorted_positive[pos_idx]) - mean_upper) / std_upper
            )
    return omega_stats


def _two_stage_mgbt_sweep(
    p_values: np.ndarray,
    zero_count: int,
    alpha_out: float,
    alpha_zero_in: float,
) -> int:
    """Execute the Bulletin 17C outward and inward significance sweeps."""
    n2 = len(p_values)
    # 1. Outward sweep from median (r = n2 down to 1) at level alpha_out (0.005)
    j1 = 0
    for idx in range(n2 - 1, -1, -1):
        if p_values[idx] < alpha_out:
            j1 = idx + 1
            break

    # 2. Inward sweep from smallest observation (r = 1 up to n2) at level
    # alpha_zero_in (0.10) per Cohn et al. (2013) / USGS MGBT17c.
    j3 = 0
    for idx in range(n2):
        if p_values[idx] < alpha_zero_in:
            j3 = idx + 1
        else:
            break

    return max(j1, j3, zero_count)


class MultipleGrubbsBeckTester:
    """Official USGS Bulletin 17C Multiple Grubbs-Beck Test (MGBT) for PILFs.

    Screens up to the lowest `floor(n / 2)` observations of an annual flood
    series for Potentially Influential Low Floods (PILFs) using the outward
    (`alpha_out = 0.005`) and inward (`alpha_in = 0.10`) sweeps defined in
    Cohn et al. (2013) and USGS Bulletin 17C Appendix 6.
    """

    def __init__(
        self,
        data: Sequence[float] | np.ndarray,
        alpha_out: float = 0.005,
        alpha_in: float = 0.10,
        num_zero_flows: int = 0,
        is_log_transformed: bool = True,  # noqa: FBT001, FBT002
    ):
        """Run the Multiple Grubbs-Beck Test on a flood series.

        Args:
            data: Sequence of flood observations. By default, assumed to be
                already log10-transformed positive flows (`is_log_transformed=
                True`), with `num_zero_flows` specifying any additional zero
                flows in the record. If `is_log_transformed=False`, `data` is
                treated as raw discharge values (which may include `0.0`).
            alpha_out: Significance level for the outward sweep from the median
                (Bulletin 17C default: `0.005`).
            alpha_in: Significance level for the inward sweep from the smallest
                observation (Bulletin 17C default: `0.10`).
            num_zero_flows: Number of additional zero-flow years in the record
                when `data` contains only positive log-transformed flows.
            is_log_transformed: Whether `data` is already in log10 space.

        Raises:
            NotEnoughDataError: If fewer than 5 positive observations are
                provided.
            ValueError: If `alpha_out` or `alpha_in` is not in `(0, 1)`.
        """
        if not (0.0 < alpha_out < 1.0) or not (0.0 < alpha_in < 1.0):
            raise ValueError(
                'Significance levels alpha_out and alpha_in must be in (0, 1).'
            )
        if num_zero_flows < 0:
            raise ValueError('num_zero_flows must be non-negative.')

        sorted_positive, zero_count, n_total = _prepare_mgbt_samples(
            data=data,
            num_zero_flows=num_zero_flows,
            is_log_transformed=is_log_transformed,
        )

        self._alpha_out = alpha_out
        self._alpha_in = alpha_in
        self._num_zero_flows = zero_count

        omega_stats = _compute_omega_statistics(
            sorted_positive=sorted_positive,
            zero_count=zero_count,
            n_total=n_total,
        )
        n2 = len(omega_stats)
        p_values = np.zeros(n2, dtype=float)
        finite_mask = np.isfinite(omega_stats)
        if np.any(finite_mask):
            r_finite = np.nonzero(finite_mask)[0] + 1
            p_values[finite_mask] = _mgbt_pvalues_vectorized(
                n=n_total,
                r_indices=r_finite,
                omega_values=omega_stats[finite_mask],
            )

        klow = _two_stage_mgbt_sweep(
            p_values=p_values,
            zero_count=zero_count,
            alpha_out=alpha_out,
            alpha_zero_in=alpha_in,
        )

        self._omega_statistics = omega_stats
        self._p_values = p_values

        positive_outliers = max(0, klow - zero_count)
        if klow > 0:
            self._threshold = float(sorted_positive[positive_outliers])
            strictly_below = sorted_positive < self._threshold
            self._out_of_pop_sample = sorted_positive[strictly_below]
            self._in_pop_sample = sorted_positive[~strictly_below]
            self._klow = zero_count + int(np.sum(strictly_below))
        else:
            self._out_of_pop_sample = sorted_positive[:0]
            self._in_pop_sample = sorted_positive
            self._threshold = -np.inf
            self._klow = 0

    @property
    def in_population_sample(self) -> np.ndarray:
        """Return retained observations at or above the PILF threshold."""
        return self._in_pop_sample

    @property
    def out_of_population_sample(self) -> np.ndarray:
        """Return positive low-outlier observations strictly below threshold."""
        return self._out_of_pop_sample

    @property
    def threshold(self) -> float:
        """Return the PILF threshold in the input (log10) space."""
        return self._threshold

    @property
    def klow(self) -> int:
        """Return the total number of low outliers (including zero flows)."""
        return self._klow

    @property
    def num_zero_flows(self) -> int:
        """Return the number of zero-flow observations in the MGBT record."""
        return self._num_zero_flows

    @property
    def p_values(self) -> np.ndarray:
        """Return MGBT p-values for order statistics `r = 1..floor(n / 2)`."""
        return self._p_values.copy()

    @property
    def omega_statistics(self) -> np.ndarray:
        """Return generalized Grubbs-Beck statistics `omega_r` (`r=1..n//2`)."""
        return self._omega_statistics.copy()


class GrubbsBeckTester:
    """Legacy Bulletin 17B single-outlier Grubbs-Beck critical-value test.

    Screens flood peaks using the 10% significance critical value table from
    USGS Bulletin 17B Appendix 4 (`bulletin17b_kn_table.csv`). For full
    Bulletin 17C compliance, prefer `MultipleGrubbsBeckTester`.
    """

    def __init__(self, data: Sequence[float] | np.ndarray):
        """Run the Bulletin 17B Grubbs-Beck test on `data`."""
        self._kn_table = _load_kn_table()
        arr = np.asarray(data, dtype=float).ravel()
        if np.any(~np.isfinite(arr)):
            raise exceptions.InvalidFlowValueError(
                'Log-transformed flow values cannot be NaN or infinite.'
            )
        sorted_data = np.sort(arr)
        if len(sorted_data) < _MIN_B17B_SAMPLE_SIZE:
            raise exceptions.NotEnoughDataError(
                'Need at least '
                f'{_MIN_B17B_SAMPLE_SIZE} data points for the Bulletin 17B '
                f'Grubbs-Beck test; got {len(sorted_data)}.'
            )
        self._grubbs_beck_test(sorted_data)

    @property
    def in_population_sample(self) -> np.ndarray:
        """Return observations retained as belonging to the flood population."""
        return self._in_pop_sample

    @property
    def out_of_population_sample(self) -> np.ndarray:
        """Return observations identified as low outliers (PILFs)."""
        return self._out_of_pop_sample

    @property
    def threshold(self) -> float:
        """Return the low-outlier threshold in the input (log10) space."""
        return self._threshold

    def _lookup_kn_value(self, sample_size: int) -> float:
        """Return the Bulletin 17B 10% critical value K_n for `sample_size`."""
        if sample_size < _MIN_B17B_SAMPLE_SIZE:
            raise exceptions.NotEnoughDataError(
                'Need at least '
                f'{_MIN_B17B_SAMPLE_SIZE} data points for the Grubbs-Beck test.'
            )
        clamped_size = min(sample_size, _MAX_B17B_SAMPLE_SIZE)
        return self._kn_table[clamped_size]

    def _grubbs_beck_test(self, sorted_data: np.ndarray) -> None:
        """Execute a sequential sweep using Bulletin 17B critical values."""
        sample_size = len(sorted_data)
        max_sample_position_to_test = sample_size - _MIN_B17B_SAMPLE_SIZE

        pilf_index = -1
        for s in range(max_sample_position_to_test + 1):
            sub_sample = sorted_data[s:]
            mean = float(sub_sample.mean())
            std = float(sub_sample.std(ddof=1))
            kn = self._lookup_kn_value(len(sub_sample))
            threshold = mean - kn * std
            if sorted_data[s] < threshold:
                pilf_index = s
            else:
                break

        if pilf_index >= 0:
            self._threshold = float(sorted_data[pilf_index + 1])
            strictly_below = sorted_data < self._threshold
            self._out_of_pop_sample = sorted_data[strictly_below]
            self._in_pop_sample = sorted_data[~strictly_below]
        else:
            self._in_pop_sample = sorted_data
            self._out_of_pop_sample = np.asarray([], dtype=float)
            self._threshold = -np.inf
