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

Provides threshold-crossing event extraction and precision/recall/F1 evaluation
for multi-lead hydrological forecasts using USGS Bulletin 17C (GEMA) and
Log-Pearson Type III return-period quantiles.
"""

import numpy as np
import pandas as pd
from return_periods import exceptions
from return_periods.return_period_calculator import ReturnPeriodCalculator

RETURN_PERIODS = [1.01, 2.0, 5.0, 10.0, 20.0]
EXCEEDANCE_WEIGHTS = {1.01: 1.0, 2.0: 0.5, 5.0: 0.2, 10.0: 0.1, 20.0: 0.05}
EXCEEDANCE_WEIGHT_SUM = 1.85  # sum(EXCEEDANCE_WEIGHTS.values())


def get_up_crossing_indices(series: np.ndarray, threshold: float) -> np.ndarray:
    """Return indices where `series` crosses from below `threshold` to >= `threshold`."""
    if not np.isfinite(threshold):
        return np.array([], dtype=int)
    valid = ~np.isnan(series)
    exceed = np.zeros(len(series), dtype=bool)
    exceed[valid] = series[valid] >= threshold
    up_cross = exceed.copy()
    up_cross[1:] = exceed[1:] & (~exceed[:-1])
    return np.where(up_cross)[0]


def _check_events_in_window(
    event_indices: np.ndarray,
    ref_series: np.ndarray,
    ref_threshold: float,
    window_days: int,
    discard_nans: bool,
) -> list[bool]:
    results = []
    n = len(ref_series)
    for idx in event_indices:
        s = max(0, idx - window_days)
        e = min(n, idx + window_days + 1)
        w = ref_series[s:e]
        if discard_nans and np.isnan(w).any():
            continue
        if np.isnan(w).all():
            continue
        results.append(bool(np.nanmax(w) >= ref_threshold))
    return results


def compute_precision_recall_f1_single_rp(
    obs: np.ndarray,
    sim: np.ndarray,
    obs_thresh: float,
    sim_thresh: float,
    window_days: int = 2,
) -> tuple[float, float, float, int, int]:
    """Compute precision, recall, F1, n_obs, and n_sim for a single return period."""
    obs_events = (
        get_up_crossing_indices(obs, obs_thresh)
        if np.isfinite(obs_thresh)
        else np.array([], dtype=int)
    )
    sim_events = (
        get_up_crossing_indices(sim, sim_thresh)
        if np.isfinite(sim_thresh)
        else np.array([], dtype=int)
    )
    n_obs = len(obs_events)
    n_sim = len(sim_events)

    if not np.isfinite(obs_thresh):
        return np.nan, np.nan, np.nan, 0, n_sim

    if n_obs == 0 and n_sim == 0:
        return 1.0, 1.0, 1.0, 0, 0
    if n_obs > 0 and (n_sim == 0 or not np.isfinite(sim_thresh)):
        return 0.0, 0.0, 0.0, n_obs, 0
    if n_obs == 0 and n_sim > 0:
        return 0.0, 0.0, 0.0, 0, n_sim

    prec_hits = _check_events_in_window(
        sim_events, obs, obs_thresh, window_days, discard_nans=True
    )
    rec_hits = _check_events_in_window(
        obs_events, sim, sim_thresh, window_days, discard_nans=False
    )

    precision = float(np.mean(prec_hits)) if len(prec_hits) > 0 else 0.0
    recall = float(np.mean(rec_hits)) if len(rec_hits) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1, n_obs, n_sim


def compute_series_return_period_thresholds(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    fitter: str = 'gema',
    return_periods: list[float] | None = None,
) -> dict[float, float]:
    """Fit return-period discharge thresholds for a daily streamflow series."""
    if return_periods is None:
        return_periods = RETURN_PERIODS
    arr = np.asarray(values, dtype=np.float64)
    clean = np.where(np.isnan(arr), np.nan, np.maximum(arr, 0.0))
    s = pd.Series(clean, index=dates, dtype=np.float64)
    try:
        calc = ReturnPeriodCalculator(
            hydrograph_series=s,
            fitter=fitter,
            max_missing_days_in_year=182,
            water_year_start_month=10,
        )
        q_vals = calc.flow_values_from_return_periods(return_periods)
        return {rp: float(q) for rp, q in zip(return_periods, q_vals)}
    except (
        exceptions.NotEnoughDataError,
        exceptions.InvalidFlowValueError,
        exceptions.DistributionParameterError,
        exceptions.DataIntervalError,
        ValueError,
        RuntimeError,
    ):
        return {rp: np.nan for rp in return_periods}


def evaluate_forecast_return_periods(
    obs_2d: np.ndarray,
    sim_2d: np.ndarray,
    dates: pd.DatetimeIndex,
    obs_thresholds_gema: dict[float, float] | None = None,
    obs_thresholds_lp3: dict[float, float] | None = None,
    lead_indices: list[int] | None = None,
    windows: list[int] | None = None,
    compute_lp3_lead0: bool = True,
) -> dict[str, float]:
    """Evaluate return-period flood precision, recall, F1, and expected recall across lead times."""
    if lead_indices is None:
        lead_indices = [0, 1, 2, 3, 4, 5, 6]
    if windows is None:
        windows = [0, 2]
    if obs_thresholds_gema is None:
        obs_thresholds_gema = compute_series_return_period_thresholds(
            obs_2d[:, 0], dates, fitter='gema'
        )

    out: dict[str, float] = {}
    for rp in RETURN_PERIODS:
        out[f'ObsThresh_RP{rp:g}'] = obs_thresholds_gema[rp]

    for k in lead_indices:
        lead_day = k + 1
        obs_k = obs_2d[:, k]
        sim_k = sim_2d[:, k]
        sim_thresholds_gema = compute_series_return_period_thresholds(
            sim_k, dates, fitter='gema'
        )

        for w in windows:
            w_sfx = '' if w == 2 else f'_w{w}'
            exp_rec = 0.0
            exp_prec = 0.0
            exp_f1 = 0.0
            valid_rp = True
            for rp in RETURN_PERIODS:
                p, r, f1, n_obs, n_sim = compute_precision_recall_f1_single_rp(
                    obs_k,
                    sim_k,
                    obs_thresholds_gema[rp],
                    sim_thresholds_gema[rp],
                    window_days=w,
                )
                if k == 0 and w == 2:
                    out[f'RP{rp:g}_Precision'] = p
                    out[f'RP{rp:g}_Recall'] = r
                    out[f'RP{rp:g}_F1'] = f1
                    out[f'RP{rp:g}_NObs'] = n_obs
                    out[f'RP{rp:g}_NSim'] = n_sim
                    out[f'SimThresh_RP{rp:g}'] = sim_thresholds_gema[rp]
                out[f'RP{rp:g}_Precision{w_sfx}_lead{lead_day}'] = p
                out[f'RP{rp:g}_Recall{w_sfx}_lead{lead_day}'] = r
                out[f'RP{rp:g}_F1{w_sfx}_lead{lead_day}'] = f1

                if np.isfinite(r):
                    wt = EXCEEDANCE_WEIGHTS[rp]
                    exp_rec += wt * r
                    exp_prec += wt * p
                    exp_f1 += wt * f1
                else:
                    valid_rp = False

            if valid_rp:
                out[f'RP_ExpRecall{w_sfx}_lead{lead_day}'] = exp_rec
                out[f'RP_ExpRecallNorm{w_sfx}_lead{lead_day}'] = (
                    exp_rec / EXCEEDANCE_WEIGHT_SUM
                )
                out[f'RP_ExpPrecisionNorm{w_sfx}_lead{lead_day}'] = (
                    exp_prec / EXCEEDANCE_WEIGHT_SUM
                )
                out[f'RP_ExpF1Norm{w_sfx}_lead{lead_day}'] = (
                    exp_f1 / EXCEEDANCE_WEIGHT_SUM
                )
                if k == 0 and w == 2:
                    out['RP_ExpRecall'] = exp_rec
                    out['RP_ExpRecallNorm'] = exp_rec / EXCEEDANCE_WEIGHT_SUM
                    out['RP_ExpPrecisionNorm'] = (
                        exp_prec / EXCEEDANCE_WEIGHT_SUM
                    )
                    out['RP_ExpF1Norm'] = exp_f1 / EXCEEDANCE_WEIGHT_SUM
            else:
                out[f'RP_ExpRecall{w_sfx}_lead{lead_day}'] = np.nan
                out[f'RP_ExpRecallNorm{w_sfx}_lead{lead_day}'] = np.nan
                out[f'RP_ExpPrecisionNorm{w_sfx}_lead{lead_day}'] = np.nan
                out[f'RP_ExpF1Norm{w_sfx}_lead{lead_day}'] = np.nan
                if k == 0 and w == 2:
                    out['RP_ExpRecall'] = np.nan
                    out['RP_ExpRecallNorm'] = np.nan
                    out['RP_ExpPrecisionNorm'] = np.nan
                    out['RP_ExpF1Norm'] = np.nan

        if k == 0 and compute_lp3_lead0:
            if obs_thresholds_lp3 is None:
                obs_thresholds_lp3 = compute_series_return_period_thresholds(
                    obs_k, dates, fitter='simple_lp3'
                )
            sim_thresholds_lp3 = compute_series_return_period_thresholds(
                sim_k, dates, fitter='simple_lp3'
            )
            exp_rec_lp3 = 0.0
            exp_f1_lp3 = 0.0
            for rp in RETURN_PERIODS:
                p, r, f1, _, _ = compute_precision_recall_f1_single_rp(
                    obs_k,
                    sim_k,
                    obs_thresholds_lp3[rp],
                    sim_thresholds_lp3[rp],
                    window_days=2,
                )
                out[f'RP{rp:g}_Recall_lp3'] = r
                out[f'RP{rp:g}_F1_lp3'] = f1
                exp_rec_lp3 += EXCEEDANCE_WEIGHTS[rp] * r
                exp_f1_lp3 += EXCEEDANCE_WEIGHTS[rp] * f1
            out['RP_ExpRecall_lp3'] = exp_rec_lp3
            out['RP_ExpRecallNorm_lp3'] = exp_rec_lp3 / EXCEEDANCE_WEIGHT_SUM
            out['RP_ExpF1Norm_lp3'] = exp_f1_lp3 / EXCEEDANCE_WEIGHT_SUM

    return out
