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

"""Return-period threshold estimation and exceedance classification for `maas`.

Integrates directly with `return_periods` (USGS Bulletin 17C Expected Moments
Algorithm + Multiple Grubbs-Beck Test and empirical Weibull plotting positions)
alongside EV1 (Gumbel) method-of-moments utilities for reanalysis records.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from maas.config import (
    CANONICAL_RETURN_PERIODS,
    RETURN_PERIOD_YEARS,
    parse_finite_float,
)
from return_periods import (
    GEMAFitter,
    ReturnPeriodCalculator,
    extract_annual_maximums,
    simple_empirical_plotting_position,
)

EULER_GAMMA = 0.5772156649015329
INDEX_FLOOD_MAF_RATIO = 3.3
INDEX_FLOOD_CV = 0.45

EXCEEDANCE_CLASSES: tuple[dict[str, Any], ...] = (
    {
        'rank': 0,
        'label': 'Normal',
        'risk_level': 'NORMAL',
        'return_period': '< 2-yr',
        'color': '#1e8e3e',
    },
    {
        'rank': 1,
        'label': '2-Yr Warning',
        'risk_level': 'WARNING',
        'return_period': '≥ 2-yr',
        'color': '#f9ab00',
    },
    {
        'rank': 2,
        'label': '5-Yr Severe',
        'risk_level': 'SEVERE',
        'return_period': '≥ 5-yr',
        'color': '#e8710a',
    },
    {
        'rank': 3,
        'label': '20-Yr+ Extreme',
        'risk_level': 'EXTREME',
        'return_period': '≥ 20-yr',
        'color': '#d93025',
    },
    {
        'rank': 4,
        'label': '100-Yr+ Extreme',
        'risk_level': 'EXTREME',
        'return_period': '≥ 100-yr',
        'color': '#a50e0e',
    },
)

UNASSESSED_LABEL = 'Not assessed (offline fallback)'
UNASSESSED_COLOR = '#80868b'
RISK_RANK: dict[str, int] = {
    'UNKNOWN': -1,
    'NORMAL': 0,
    'WARNING': 1,
    'SEVERE': 2,
    'EXTREME': 3,
}


def _to_peaks_series(
    series: pd.Series | Sequence[float] | np.ndarray,
    *,
    is_daily_hydrograph: bool = False,
    max_missing_days_in_year: int = 65,
    water_year_start_month: int = 1,
) -> pd.Series:
    """Convert an annual-maxima sequence or daily hydrograph to a `pd.Series`."""
    if is_daily_hydrograph:
        if not isinstance(series, pd.Series) or not isinstance(
            series.index, pd.DatetimeIndex
        ):
            raise TypeError(
                'When `is_daily_hydrograph=True`, `series` must be a '
                '`pd.Series` indexed by a `pd.DatetimeIndex`.'
            )
        return extract_annual_maximums(
            hydrograph_series=series,
            max_missing_days_in_year=max_missing_days_in_year,
            water_year_start_month=water_year_start_month,
        )
    if isinstance(series, pd.Series):
        if series.index.duplicated().any():
            return pd.Series(
                series.to_numpy(dtype=float),
                index=range(len(series)),
                dtype=float,
            )
        return series.astype(float)
    arr = np.asarray(series, dtype=float).ravel()
    return pd.Series(arr, index=range(len(arr)), dtype=float)


def _format_rp_key(period: float) -> str:
    fval = float(period)
    if fval.is_integer():
        return f'return_period_{int(fval)}'
    return f'return_period_{fval}'


def compute_return_periods(
    series: pd.Series | Sequence[float] | np.ndarray,
    return_periods: Sequence[float] = CANONICAL_RETURN_PERIODS,
    *,
    is_daily_hydrograph: bool = False,
    max_missing_days_in_year: int = 65,
    water_year_start_month: int = 1,
    regional_skew: float | None = None,
    regional_skew_mse: float | None = None,
) -> dict[str, Any]:
    """Compute return-period discharge thresholds via USGS Bulletin 17C EMA + MGBT.

    Args:
        series: Either an annual maximum series (1D sequence/array or `pd.Series`)
            or a daily streamflow `pd.Series` with `pd.DatetimeIndex` when
            `is_daily_hydrograph=True`.
        return_periods: Return periods in years (`T > 1`) to evaluate.
        is_daily_hydrograph: Whether `series` is a daily streamflow hydrograph
            requiring annual peak extraction first.
        max_missing_days_in_year: Maximum missing days allowed per year when
            extracting annual peaks from a daily hydrograph.
        water_year_start_month: Starting month (`1..12`) of the water year.
        regional_skew: Optional regional skew coefficient.
        regional_skew_mse: Optional MSE of the regional skew coefficient.

    Returns:
        Dictionary containing `return_period_<T>` discharge thresholds in
        $\text{m}^3/\text{s}$, `years_of_record`, `pilf_count`, `pilf_threshold`,
        `method`, and `unit`.
    """
    peaks = _to_peaks_series(
        series,
        is_daily_hydrograph=is_daily_hydrograph,
        max_missing_days_in_year=max_missing_days_in_year,
        water_year_start_month=water_year_start_month,
    )
    rpc = ReturnPeriodCalculator(
        peaks_series=peaks,
        fitter='gema',
        regional_skew=regional_skew,
        regional_skew_mse=regional_skew_mse,
    )
    quantiles = rpc.flow_values_from_return_periods(return_periods)
    out: dict[str, Any] = {
        _format_rp_key(t): round(float(q), 2)
        for t, q in zip(return_periods, quantiles, strict=True)
    }
    fitter = rpc.fitter
    pilf_count = (
        int(fitter.outlier_tester.klow)
        if isinstance(fitter, GEMAFitter)
        and hasattr(fitter.outlier_tester, 'klow')
        else 0
    )
    pilf_thresh = (
        round(float(10.0**fitter.outlier_tester.threshold), 3)
        if isinstance(fitter, GEMAFitter)
        and math.isfinite(fitter.outlier_tester.threshold)
        else 0.0
    )
    out.update(
        {
            'method': 'USGS Bulletin 17C EMA + MGBT (Log-Pearson Type III)',
            'years_of_record': len(peaks),
            'pilf_count': pilf_count,
            'pilf_threshold': pilf_thresh,
            'unit': 'm³/s',
        }
    )
    return out


def compute_empirical_weibull_return_periods(
    series: pd.Series | Sequence[float] | np.ndarray,
    return_periods: Sequence[float] = CANONICAL_RETURN_PERIODS,
    *,
    is_daily_hydrograph: bool = False,
    max_missing_days_in_year: int = 65,
    water_year_start_month: int = 1,
) -> dict[str, Any]:
    """Compute empirical Weibull plotting positions and log-linear return periods.

    Args:
        series: Either an annual maximum series or a daily streamflow `pd.Series`
            when `is_daily_hydrograph=True`.
        return_periods: Return periods in years (`T > 1`) to evaluate.
        is_daily_hydrograph: Whether `series` is a daily streamflow hydrograph.
        max_missing_days_in_year: Maximum missing days allowed per year.
        water_year_start_month: Starting month (`1..12`) of the water year.

    Returns:
        Dictionary containing `return_period_<T>` discharge thresholds,
        `empirical_sorted_flows`, `empirical_exceedance_probabilities`,
        `empirical_return_periods`, `years_of_record`, `method`, and `unit`.
    """
    peaks = _to_peaks_series(
        series,
        is_daily_hydrograph=is_daily_hydrograph,
        max_missing_days_in_year=max_missing_days_in_year,
        water_year_start_month=water_year_start_month,
    )
    peak_vals = peaks.dropna().to_numpy(dtype=float)
    pos_vals = peak_vals[peak_vals > 0.0]
    if len(pos_vals) == 0:
        raise ValueError(
            'Cannot compute Weibull return periods without positive peak flows.'
        )
    sorted_flows, exceedance_probs = simple_empirical_plotting_position(
        pos_vals,
        plotting_position_type='weibull',
    )
    empirical_rps = 1.0 / exceedance_probs
    rpc = ReturnPeriodCalculator(peaks_series=peaks, fitter='log_linear')
    quantiles = rpc.flow_values_from_return_periods(return_periods)
    out: dict[str, Any] = {
        _format_rp_key(t): round(float(q), 2)
        for t, q in zip(return_periods, quantiles, strict=True)
    }
    out.update(
        {
            'empirical_sorted_flows': [
                round(float(x), 3) for x in sorted_flows
            ],
            'empirical_exceedance_probabilities': [
                round(float(p), 6) for p in exceedance_probs
            ],
            'empirical_return_periods': [
                round(float(t), 3) for t in empirical_rps
            ],
            'years_of_record': len(peaks),
            'method': 'Empirical Weibull plotting positions (log-linear fit)',
            'unit': 'm³/s',
        }
    )
    return out


def gumbel_frequency_factor(return_period_yrs: float) -> float:
    """EV1 (Gumbel) frequency factor $K_T$ for the method of moments (Chow, 1951)."""
    t = float(return_period_yrs)
    if t <= 1.0:
        raise ValueError(f'return_period_yrs must be > 1.0, got {t}.')
    return -(math.sqrt(6.0) / math.pi) * (
        EULER_GAMMA + math.log(math.log(t / (t - 1.0)))
    )


def extract_annual_maxima(
    times: Sequence[Any],
    values: Sequence[Any],
    min_valid_days: int = 300,
) -> list[float]:
    """Calendar-year maxima of a daily series (years with >= `min_valid_days` valid values)."""
    by_year: dict[str, list[float]] = {}
    for t, v in zip(times, values, strict=False):
        fv = parse_finite_float(v)
        if fv is None or fv < 0.0:
            continue
        year = str(t)[:4]
        if year.isdigit():
            by_year.setdefault(year, []).append(fv)
    return [
        max(vals)
        for _, vals in sorted(by_year.items())
        if len(vals) >= min_valid_days
    ]


def compute_gumbel_return_periods(
    annual_maxima: Sequence[float],
    return_periods: Sequence[int] = RETURN_PERIOD_YEARS,
) -> dict[str, float] | None:
    """Fit EV1 by moments to annual maxima; returns `{'return_period_T': Q_T}` in m³/s."""
    n = len(annual_maxima)
    if n < 8:
        return None
    mean = sum(annual_maxima) / n
    if mean <= 1e-3:
        return None
    std = math.sqrt(
        max(sum((x - mean) ** 2 for x in annual_maxima) / (n - 1), 0.0)
    )
    return {
        f'return_period_{t}': round(
            max(mean + gumbel_frequency_factor(t) * std, 0.0), 2
        )
        for t in return_periods
    }


def ev1_fit_line(rps: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """Least-squares EV1 line $Q = a + b K_T$ through known `return_period_T` levels."""
    pts: list[tuple[float, float]] = []
    for key, val in (rps or {}).items():
        skey = str(key)
        if not skey.startswith('return_period_'):
            continue
        suffix = skey.rsplit('_', 1)[1]
        t = parse_finite_float(suffix)
        fv = parse_finite_float(val)
        if t is not None and fv is not None and t > 1.0:
            pts.append((gumbel_frequency_factor(t), fv))
    if len(pts) < 2:
        return None
    n = len(pts)
    mean_k = sum(p[0] for p in pts) / n
    mean_q = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mean_k) ** 2 for p in pts)
    if sxx <= 1e-12:
        return None
    slope = sum((p[0] - mean_k) * (p[1] - mean_q) for p in pts) / sxx
    return mean_q - slope * mean_k, slope


def gumbel_quantile_from_return_periods(
    rps: Mapping[str, Any] | None,
    return_period_yrs: float,
) -> float | None:
    """Interpolate $Q_T$ from known return levels via the EV1 line $Q = a + b K_T$."""
    line = ev1_fit_line(rps)
    if line is None:
        return None
    return round(
        max(
            line[0] + line[1] * gumbel_frequency_factor(return_period_yrs), 0.0
        ),
        2,
    )


def known_return_levels(
    rps: Mapping[str, Any] | None,
) -> list[tuple[float, float]]:
    """Sorted, strictly increasing `(T, Q_T)` pairs (`T >= 2 yr`, `Q_T > 0`)."""
    pts: list[tuple[float, float]] = []
    for key, val in (rps or {}).items():
        skey = str(key)
        if not skey.startswith('return_period_'):
            continue
        suffix = skey.rsplit('_', 1)[1]
        t = parse_finite_float(suffix)
        fv = parse_finite_float(val)
        if t is not None and fv is not None and fv > 0.0 and t >= 2.0:
            pts.append((t, fv))
    levels: list[tuple[float, float]] = []
    for t, q in sorted(pts):
        if not levels or q > levels[-1][1]:
            levels.append((t, q))
    return levels


def estimate_return_period_years(
    value: float | None,
    rps: Mapping[str, Any] | None,
) -> float | None:
    """Estimate the return period (years) of a flow value from known return levels.

    Interpolates the EV1 reduced variate $K_T$ piecewise-linearly between
    consecutive known levels (exact at each level, extrapolated above the top
    level up to 1000 yr, and `None` below the 2-yr threshold).
    """
    fv = parse_finite_float(value)
    if fv is None:
        return None
    levels = known_return_levels(rps)
    if len(levels) < 2 or fv < levels[0][1]:
        return None
    ks = [(gumbel_frequency_factor(t), q) for t, q in levels]
    (k0, q0), (k1, q1) = ks[-2], ks[-1]
    for (ka, qa), (kb, qb) in zip(ks, ks[1:], strict=False):
        if fv <= qb:
            (k0, q0), (k1, q1) = (ka, qa), (kb, qb)
            break
    k = k0 + (fv - q0) * (k1 - k0) / (q1 - q0)
    u = -(math.pi / math.sqrt(6.0)) * k - EULER_GAMMA
    if u > 30.0:
        return 1.0
    if u < -30.0:
        return 1000.0
    denom = 1.0 - math.exp(-math.exp(u))
    if denom <= 1e-12:
        return 1000.0
    return round(min(max(1.0 / denom, 1.0), 1000.0), 1)


def scaled_index_flood_return_periods(
    reference_flow: float,
    return_periods: Sequence[int] = RETURN_PERIOD_YEARS,
) -> dict[str, float]:
    """Deterministic index-flood return periods scaled from a reference flow (m³/s)."""
    maf = max(float(reference_flow or 0.0), 0.1) * INDEX_FLOOD_MAF_RATIO
    return {
        f'return_period_{t}': round(
            max(maf * (1.0 + gumbel_frequency_factor(t) * INDEX_FLOOD_CV), 0.0),
            2,
        )
        for t in return_periods
    }


def thresholds_from_return_periods(
    rps: Mapping[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    """Map return-period levels onto the MaaS 2/5/20/100-yr threshold contract."""
    rps_map = rps or {}
    return {
        'warning_2yr': rps_map.get('return_period_2'),
        'danger_5yr': rps_map.get('return_period_5'),
        'extreme_20yr': rps_map.get('return_period_20'),
        'extreme_100yr': rps_map.get('return_period_100'),
        'source': source,
        'unit': 'm³/s',
    }


def classify_exceedance(
    peak: float | None,
    rps: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return-period exceedance class of a peak flow against model-specific levels."""
    rank = 0
    fpeak = parse_finite_float(peak)
    if fpeak is not None:
        for r, key in (
            (4, 'return_period_100'),
            (3, 'return_period_20'),
            (2, 'return_period_5'),
            (1, 'return_period_2'),
        ):
            thr = parse_finite_float((rps or {}).get(key))
            if thr is not None and thr > 0.0 and fpeak >= thr:
                rank = r
                break
    return dict(EXCEEDANCE_CLASSES[rank])
