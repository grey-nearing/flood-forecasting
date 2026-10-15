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

"""Return-period flood event detection evaluation utilities.

Provides upward threshold-crossing event extraction and precision, recall, F1,
and exceedance-weighted expected score evaluation across forecast lead times
using USGS Bulletin 17C (GEMA) and Log-Pearson Type III return-period discharge
quantiles.
"""

import logging
from collections.abc import Sequence
from typing import TypedDict, Unpack

import numpy as np
import pandas as pd

from return_periods import extract_peaks_utilities
from return_periods.return_period_calculator import ReturnPeriodCalculator

LOGGER = logging.getLogger(__name__)

RETURN_PERIODS: list[float] = [1.01, 2.0, 5.0, 10.0, 20.0]
EXCEEDANCE_WEIGHTS: dict[float, float] = {
    1.01: 1.0,
    2.0: 0.5,
    5.0: 0.2,
    10.0: 0.1,
    20.0: 0.05,
}
EXCEEDANCE_WEIGHT_SUM: float = sum(EXCEEDANCE_WEIGHTS.values())
PRIMARY_WINDOW_DAYS: int = 2
_MIN_FIT_POINTS: int = 5
_EXPECTED_2D_NDIM: int = 2
_DEFAULT_MAX_LEADS: int = 7
_SUPPORTED_FITTERS: frozenset[str] = frozenset(
    {
        'gema',
        'simple_lp3',
        'log_linear',
    }
)


class ThresholdFitOptions(TypedDict, total=False):
    """Optional annual-maximum extraction settings for threshold fitting."""

    max_missing_days_in_year: int
    water_year_start_month: int


class ForecastReturnPeriodOptions(TypedDict, total=False):
    """Optional configuration for `evaluate_forecast_return_periods`."""

    obs_thresholds_gema: dict[float, float] | None
    obs_thresholds_lp3: dict[float, float] | None
    lead_indices: Sequence[int] | None
    windows: Sequence[int] | None
    compute_lp3_lead0: bool
    max_missing_days_in_year: int
    water_year_start_month: int


def get_up_crossing_indices(
    series: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return indices where `series` crosses upward across `threshold`.

    Index `0` is counted as an upward crossing if `series[0] >= threshold`. For
    any index `i >= 1`, both `series[i]` and `series[i - 1]` must be finite with
    `series[i - 1] < threshold` and `series[i] >= threshold`. Interior `NaN`
    gaps do not act as below-threshold values and therefore do not split a
    single flood exceedance into multiple crossings.

    Args:
        series: 1D array of daily streamflow values.
        threshold: Return-period discharge threshold.

    Returns:
        1D integer NumPy array of upward-crossing event indices.
    """
    if not np.isfinite(threshold):
        return np.array([], dtype=int)
    arr = np.asarray(series, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=int)
    valid = ~np.isnan(arr)
    exceed = np.zeros(len(arr), dtype=bool)
    exceed[valid] = arr[valid] >= threshold
    below = np.zeros(len(arr), dtype=bool)
    below[valid] = arr[valid] < threshold
    up_cross = np.zeros(len(arr), dtype=bool)
    up_cross[0] = exceed[0]
    up_cross[1:] = exceed[1:] & below[:-1]
    return np.where(up_cross)[0]


def _check_events_in_window(
    event_indices: np.ndarray,
    ref_series: np.ndarray,
    ref_threshold: float,
    window_days: int,
    *,
    discard_nans: bool,
) -> list[bool]:
    """Evaluate whether each event in `event_indices` matches `ref_series`.

    Args:
        event_indices: 1D integer indices of candidate crossing events.
        ref_series: Counterpart 1D streamflow series checked within
            `[idx - window_days, idx + window_days]`.
        ref_threshold: Discharge threshold for `ref_series`.
        window_days: Half-width of the matching window in days (`>= 0`).
        discard_nans: When `True` (used for precision against `obs`), drops any
            event whose reference window contains a `NaN` so a simulated alarm
            during an observed data gap is not counted as a false alarm. When
            `False` (used for recall against `sim`), drops an event only when
            its reference window is all-`NaN`.

    Returns:
        List of boolean hit indicators for all evaluable events.
    """
    results: list[bool] = []
    num_steps = len(ref_series)
    for event_idx in event_indices:
        start_idx = max(0, int(event_idx) - window_days)
        end_idx = min(num_steps, int(event_idx) + window_days + 1)
        window_slice = ref_series[start_idx:end_idx]
        if discard_nans and np.isnan(window_slice).any():
            continue
        if np.isnan(window_slice).all():
            continue
        results.append(bool(np.nanmax(window_slice) >= ref_threshold))
    return results


def compute_precision_recall_f1_single_rp(
    obs: np.ndarray,
    sim: np.ndarray,
    obs_thresh: float,
    sim_thresh: float,
    window_days: int = PRIMARY_WINDOW_DAYS,
) -> tuple[float, float, float, int, int]:
    """Compute precision, recall, F1, `n_obs`, and `n_sim` for one RP.

    Precision and recall are `np.nan` whenever their respective evaluable event
    denominators (`n_sim` or `n_obs`) are zero or either threshold is
    non-finite. When `n_obs > 0` and the simulation predicts zero crossings on a
    valid series (`len(raw_sim_events) == 0`), `recall = 0.0`, `f1 = 0.0`, and
    `precision = np.nan`.

    Args:
        obs: 1D array of observed daily streamflow.
        sim: 1D array of simulated daily streamflow aligned with `obs`.
        obs_thresh: Return-period discharge threshold for `obs`.
        sim_thresh: Return-period discharge threshold for `sim`.
        window_days: Matching tolerance window (`+/- window_days` days).

    Returns:
        Tuple `(precision, recall, f1, n_obs, n_sim)` where `n_obs` and `n_sim`
        are the counts of evaluable observed and simulated flood events.

    Raises:
        ValueError: If `obs` and `sim` do not have the same 1D length or
            `window_days < 0`.
    """
    obs_arr = np.asarray(obs, dtype=np.float64)
    sim_arr = np.asarray(sim, dtype=np.float64)
    if obs_arr.ndim != 1 or sim_arr.ndim != 1 or len(obs_arr) != len(sim_arr):
        raise ValueError(
            'obs and sim must be 1D arrays of equal length; got shapes '
            f'{obs_arr.shape} and {sim_arr.shape}.'
        )
    if window_days < 0:
        raise ValueError(f'window_days must be >= 0, got {window_days}.')

    if not np.isfinite(obs_thresh):
        return np.nan, np.nan, np.nan, 0, 0

    raw_obs_events = get_up_crossing_indices(obs_arr, obs_thresh)
    if not np.isfinite(sim_thresh):
        return np.nan, np.nan, np.nan, len(raw_obs_events), 0

    raw_sim_events = get_up_crossing_indices(sim_arr, sim_thresh)
    prec_hits = _check_events_in_window(
        raw_sim_events,
        obs_arr,
        obs_thresh,
        window_days,
        discard_nans=True,
    )
    rec_hits = _check_events_in_window(
        raw_obs_events,
        sim_arr,
        sim_thresh,
        window_days,
        discard_nans=False,
    )
    n_obs = len(rec_hits)
    n_sim = len(prec_hits)

    precision = float(np.mean(prec_hits)) if n_sim > 0 else np.nan
    recall = float(np.mean(rec_hits)) if n_obs > 0 else np.nan

    if n_obs == 0:
        f1 = np.nan
    elif n_sim == 0:
        f1 = 0.0 if len(raw_sim_events) == 0 else np.nan
    elif precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1, n_obs, n_sim


def compute_series_return_period_thresholds(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    fitter: str = 'gema',
    return_periods: Sequence[float] | None = None,
    **options: Unpack[ThresholdFitOptions],
) -> dict[float, float]:
    """Fit return-period discharge thresholds for a daily streamflow series.

    Validates input shapes, finiteness, non-negativity, and `fitter` explicitly.
    When the extracted annual maximum series has fewer than 5 positive peaks or
    zero variance among positive peaks, logs a warning and returns `np.nan` for
    each requested return period.

    Args:
        values: 1D array of daily streamflow values (`>= 0`).
        dates: `pd.DatetimeIndex` of equal length corresponding to `values`.
        fitter: Distribution fitter name (`'gema'`, `'simple_lp3'`, or
            `'log_linear'`).
        return_periods: Sequence of return periods in years (`> 1`). Defaults to
            `RETURN_PERIODS`.
        **options: Optional `max_missing_days_in_year` (default `100`) and
            `water_year_start_month` (default `1`).

    Returns:
        Dictionary mapping each return period `rp` to its fitted discharge
        quantile (`float`, or `np.nan` when insufficient positive peaks exist).

    Raises:
        TypeError: If `dates` is not a `pd.DatetimeIndex`.
        ValueError: If `values` and `dates` differ in length, `fitter` is
            unknown, or year extraction options are out of range.
        InvalidFlowValueError: If `values` contains negative or infinite flows.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError('dates must be a pandas.DatetimeIndex.')
    if fitter not in _SUPPORTED_FITTERS:
        raise ValueError(
            f'Unknown fitter {fitter!r}. Expected one of '
            f'{sorted(_SUPPORTED_FITTERS)}.'
        )
    target_rps = (
        list(RETURN_PERIODS)
        if return_periods is None
        else [float(rp) for rp in return_periods]
    )
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or len(arr) != len(dates):
        raise ValueError(
            f'values must be a 1D array matching len(dates)={len(dates)}; '
            f'got shape {arr.shape}.'
        )

    max_missing_days = options.get('max_missing_days_in_year', 100)
    water_year_month = options.get('water_year_start_month', 1)

    hydrograph = pd.Series(arr, index=dates, dtype=np.float64)
    peaks_series = extract_peaks_utilities.extract_annual_maximums(
        hydrograph_series=hydrograph,
        max_missing_days_in_year=max_missing_days,
        water_year_start_month=water_year_month,
    )
    peak_vals = peaks_series.to_numpy(dtype=np.float64)
    positive_peaks = peak_vals[peak_vals > 0.0]
    if len(positive_peaks) < _MIN_FIT_POINTS:
        LOGGER.warning(
            'Insufficient positive annual peaks (%d < %d) for fitter %r; '
            'returning NaN thresholds.',
            len(positive_peaks),
            _MIN_FIT_POINTS,
            fitter,
        )
        return dict.fromkeys(target_rps, np.nan)
    if float(np.ptp(positive_peaks)) <= 0.0:
        LOGGER.warning(
            'Positive annual peaks have zero variance for fitter %r; '
            'returning NaN thresholds.',
            fitter,
        )
        return dict.fromkeys(target_rps, np.nan)

    calc = ReturnPeriodCalculator(
        peaks_series=peaks_series,
        fitter=fitter,
    )
    q_vals = calc.flow_values_from_return_periods(target_rps)
    return {
        rp: float(q_val) for rp, q_val in zip(target_rps, q_vals, strict=True)
    }


def _evaluate_single_lead_window(
    obs_lead: np.ndarray,
    sim_lead: np.ndarray,
    obs_thresholds: dict[float, float],
    sim_thresholds: dict[float, float],
    lead_window: tuple[int, int],
) -> dict[str, float | int]:
    """Compute per-RP and exceedance-weighted metrics for `(lead, window)`."""
    out: dict[str, float | int] = {}
    lead_idx, window_days = lead_window
    lead_day = lead_idx + 1
    win_sfx = '' if window_days == PRIMARY_WINDOW_DAYS else f'_w{window_days}'
    is_primary_lead0 = lead_idx == 0 and window_days == PRIMARY_WINDOW_DAYS

    exp_rec = 0.0
    exp_prec = 0.0
    exp_f1 = 0.0
    rec_wt_sum = 0.0
    prec_wt_sum = 0.0
    f1_wt_sum = 0.0

    for rp in RETURN_PERIODS:
        prec, rec, f1_val, n_obs, n_sim = compute_precision_recall_f1_single_rp(
            obs_lead,
            sim_lead,
            obs_thresholds[rp],
            sim_thresholds[rp],
            window_days=window_days,
        )
        if is_primary_lead0:
            out[f'RP{rp:g}_Precision'] = prec
            out[f'RP{rp:g}_Recall'] = rec
            out[f'RP{rp:g}_F1'] = f1_val
            out[f'RP{rp:g}_NObs'] = n_obs
            out[f'RP{rp:g}_NSim'] = n_sim
            out[f'SimThresh_RP{rp:g}'] = sim_thresholds[rp]
        out[f'RP{rp:g}_Precision{win_sfx}_lead{lead_day}'] = prec
        out[f'RP{rp:g}_Recall{win_sfx}_lead{lead_day}'] = rec
        out[f'RP{rp:g}_F1{win_sfx}_lead{lead_day}'] = f1_val

        weight = EXCEEDANCE_WEIGHTS[rp]
        if np.isfinite(rec):
            exp_rec += weight * rec
            rec_wt_sum += weight
        if np.isfinite(prec):
            exp_prec += weight * prec
            prec_wt_sum += weight
        if np.isfinite(f1_val):
            exp_f1 += weight * f1_val
            f1_wt_sum += weight

    exp_rec_val = exp_rec if rec_wt_sum > 0.0 else np.nan
    exp_rec_norm = exp_rec / rec_wt_sum if rec_wt_sum > 0.0 else np.nan
    exp_prec_norm = exp_prec / prec_wt_sum if prec_wt_sum > 0.0 else np.nan
    exp_f1_norm = exp_f1 / f1_wt_sum if f1_wt_sum > 0.0 else np.nan

    out[f'RP_ExpRecall{win_sfx}_lead{lead_day}'] = exp_rec_val
    out[f'RP_ExpRecallNorm{win_sfx}_lead{lead_day}'] = exp_rec_norm
    out[f'RP_ExpPrecisionNorm{win_sfx}_lead{lead_day}'] = exp_prec_norm
    out[f'RP_ExpF1Norm{win_sfx}_lead{lead_day}'] = exp_f1_norm

    if is_primary_lead0:
        out['RP_ExpRecall'] = exp_rec_val
        out['RP_ExpRecallNorm'] = exp_rec_norm
        out['RP_ExpPrecisionNorm'] = exp_prec_norm
        out['RP_ExpF1Norm'] = exp_f1_norm

    return out


def _evaluate_lp3_lead0(
    obs_lead0: np.ndarray,
    sim_lead0: np.ndarray,
    dates: pd.DatetimeIndex,
    obs_thresholds_lp3: dict[float, float] | None,
    fit_options: ThresholdFitOptions,
) -> dict[str, float]:
    """Evaluate `simple_lp3` return-period recall and F1 on `lead_idx == 0`."""
    if obs_thresholds_lp3 is None:
        obs_thresholds_lp3 = compute_series_return_period_thresholds(
            obs_lead0,
            dates,
            fitter='simple_lp3',
            **fit_options,
        )
    sim_thresholds_lp3 = compute_series_return_period_thresholds(
        sim_lead0,
        dates,
        fitter='simple_lp3',
        **fit_options,
    )
    out: dict[str, float] = {}
    exp_rec_lp3 = 0.0
    exp_f1_lp3 = 0.0
    rec_wt_sum = 0.0
    f1_wt_sum = 0.0
    for rp in RETURN_PERIODS:
        _, rec, f1_val, _, _ = compute_precision_recall_f1_single_rp(
            obs_lead0,
            sim_lead0,
            obs_thresholds_lp3[rp],
            sim_thresholds_lp3[rp],
            window_days=PRIMARY_WINDOW_DAYS,
        )
        out[f'RP{rp:g}_Recall_lp3'] = rec
        out[f'RP{rp:g}_F1_lp3'] = f1_val
        weight = EXCEEDANCE_WEIGHTS[rp]
        if np.isfinite(rec):
            exp_rec_lp3 += weight * rec
            rec_wt_sum += weight
        if np.isfinite(f1_val):
            exp_f1_lp3 += weight * f1_val
            f1_wt_sum += weight

    out['RP_ExpRecall_lp3'] = exp_rec_lp3 if rec_wt_sum > 0.0 else np.nan
    out['RP_ExpRecallNorm_lp3'] = (
        exp_rec_lp3 / rec_wt_sum if rec_wt_sum > 0.0 else np.nan
    )
    out['RP_ExpF1Norm_lp3'] = (
        exp_f1_lp3 / f1_wt_sum if f1_wt_sum > 0.0 else np.nan
    )
    return out


def evaluate_forecast_return_periods(
    obs_2d: np.ndarray,
    sim_2d: np.ndarray,
    dates: pd.DatetimeIndex,
    **options: Unpack[ForecastReturnPeriodOptions],
) -> dict[str, float | int]:
    """Evaluate return-period flood event detection across forecast lead times.

    Both `obs_2d` and `sim_2d` must be 2D arrays of shape `(N_dates, N_leads)`
    whose rows correspond to `dates` (`len(dates) == N_dates`) and whose column
    `k` holds the 1D daily observed and simulated streamflow series for
    0-indexed lead column `k` (reported under `*_lead{k + 1}`). Exceedance-
    weighted summary scores (`RP_ExpRecallNorm*`, `RP_ExpPrecisionNorm*`,
    `RP_ExpF1Norm*`) are normalized over the return periods where the
    corresponding metric is defined (finite).

    Args:
        obs_2d: 2D array of observed daily streamflow with shape
            `(N_dates, N_leads)`.
        sim_2d: 2D array of simulated daily streamflow with shape
            `(N_dates, N_leads)`.
        dates: `pd.DatetimeIndex` of length `N_dates` indexing axis 0.
        **options: Optional keyword parameters defined in
            `ForecastReturnPeriodOptions` (`obs_thresholds_gema`,
            `obs_thresholds_lp3`, `lead_indices`, `windows`,
            `compute_lp3_lead0`, `max_missing_days_in_year`,
            `water_year_start_month`).

    Returns:
        Dictionary mapping metric and count names to their computed `float` or
        `int` values.

    Raises:
        TypeError: If `dates` is not a `pd.DatetimeIndex`.
        ValueError: If `obs_2d` and `sim_2d` are not 2D arrays of identical
            shape matching `len(dates)`, or if `lead_indices` is out of bounds.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError('dates must be a pandas.DatetimeIndex.')
    obs_arr = np.asarray(obs_2d, dtype=np.float64)
    sim_arr = np.asarray(sim_2d, dtype=np.float64)
    if (
        obs_arr.ndim != _EXPECTED_2D_NDIM
        or sim_arr.ndim != _EXPECTED_2D_NDIM
        or obs_arr.shape != sim_arr.shape
        or obs_arr.shape[0] != len(dates)
    ):
        raise ValueError(
            'obs_2d and sim_2d must be 2D arrays of identical shape with '
            f'rows == len(dates)={len(dates)}; got {obs_arr.shape} and '
            f'{sim_arr.shape}.'
        )

    num_leads = obs_arr.shape[1]
    raw_leads = options.get('lead_indices')
    lead_indices = (
        list(range(min(_DEFAULT_MAX_LEADS, num_leads)))
        if raw_leads is None
        else [int(idx) for idx in raw_leads]
    )
    if any(idx < 0 or idx >= num_leads for idx in lead_indices):
        raise ValueError(
            f'lead_indices {lead_indices} out of bounds for {num_leads} '
            'columns.'
        )

    raw_windows = options.get('windows')
    windows = (
        [0, PRIMARY_WINDOW_DAYS]
        if raw_windows is None
        else [int(win) for win in raw_windows]
    )
    compute_lp3_lead0 = options.get('compute_lp3_lead0', True)
    fit_options: ThresholdFitOptions = {}
    if 'max_missing_days_in_year' in options:
        fit_options['max_missing_days_in_year'] = options[
            'max_missing_days_in_year'
        ]
    if 'water_year_start_month' in options:
        fit_options['water_year_start_month'] = options[
            'water_year_start_month'
        ]

    obs_thresholds_gema = options.get('obs_thresholds_gema')
    if obs_thresholds_gema is None:
        obs_thresholds_gema = compute_series_return_period_thresholds(
            obs_arr[:, 0],
            dates,
            fitter='gema',
            **fit_options,
        )

    out: dict[str, float | int] = {
        f'ObsThresh_RP{rp:g}': obs_thresholds_gema[rp] for rp in RETURN_PERIODS
    }

    for lead_idx in lead_indices:
        obs_lead = obs_arr[:, lead_idx]
        sim_lead = sim_arr[:, lead_idx]
        sim_thresholds_gema = compute_series_return_period_thresholds(
            sim_lead,
            dates,
            fitter='gema',
            **fit_options,
        )
        for window_days in windows:
            out.update(
                _evaluate_single_lead_window(
                    obs_lead,
                    sim_lead,
                    obs_thresholds_gema,
                    sim_thresholds_gema,
                    (lead_idx, window_days),
                )
            )
        if lead_idx == 0 and compute_lp3_lead0:
            out.update(
                _evaluate_lp3_lead0(
                    obs_lead,
                    sim_lead,
                    dates,
                    options.get('obs_thresholds_lp3'),
                    fit_options,
                )
            )

    return out
