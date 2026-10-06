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

"""Utilities for extracting flood peaks from daily streamflow hydrographs."""

import numpy as np
import pandas as pd

from return_periods import exceptions

_DAYS_PER_STANDARD_YEAR = 365
_DEFAULT_PEAK_SEPARATION = pd.Timedelta('90D')


def _validate_hydrograph_series(hydrograph_series: pd.Series) -> None:
    """Validate that `hydrograph_series` has a DatetimeIndex and valid flows."""
    if not isinstance(hydrograph_series.index, pd.DatetimeIndex):
        raise TypeError(
            'hydrograph_series must be indexed by a pandas.DatetimeIndex.'
        )
    valid_vals = hydrograph_series.dropna().to_numpy(dtype=float)
    if np.any(~np.isfinite(valid_vals)):
        raise exceptions.InvalidFlowValueError(
            'hydrograph_series cannot contain infinite flow values.'
        )
    if np.any(valid_vals < 0.0):
        raise exceptions.InvalidFlowValueError(
            'hydrograph_series cannot contain negative flow values.'
        )


def extract_annual_maximums(
    hydrograph_series: pd.Series,
    max_missing_days_in_year: int = 100,
    water_year_start_month: int = 1,
) -> pd.Series:
    """Extract the Annual Maximum Series (AMS) from a daily hydrograph.

    Args:
        hydrograph_series: Time-indexed `pd.Series` of streamflow values.
        max_missing_days_in_year: Maximum number of missing (or NaN) days
            permitted in a year before that year is excluded.
        water_year_start_month: Starting month (1-12) of the hydrological year.
            Defaults to `1` (calendar year Jan 1 - Dec 31). Set to `10` for the
            standard USGS water year (Oct 1 - Sep 30).

    Returns:
        A `pd.Series` of annual maximum peak flows indexed by integer year.

    Raises:
        TypeError: If `hydrograph_series.index` is not a `pd.DatetimeIndex`.
        ValueError: If `water_year_start_month` is not in `1..12` or
            `max_missing_days_in_year` is not in `0..364`.
        InvalidFlowValueError: If `hydrograph_series` contains negative or
            infinite values.
    """
    _validate_hydrograph_series(hydrograph_series)
    if not (0 <= max_missing_days_in_year < _DAYS_PER_STANDARD_YEAR):
        raise ValueError(
            f'max_missing_days_in_year must be in '
            f'0..{_DAYS_PER_STANDARD_YEAR - 1}, got '
            f'{max_missing_days_in_year}.'
        )
    if not (1 <= water_year_start_month <= 12):  # noqa: PLR2004
        raise ValueError(
            f'water_year_start_month must be in 1..12, got '
            f'{water_year_start_month}.'
        )

    dt_index = hydrograph_series.index
    if water_year_start_month == 1:
        year_labels = dt_index.year
    else:
        year_labels = np.where(
            dt_index.month >= water_year_start_month,
            dt_index.year + 1,
            dt_index.year,
        )

    min_valid_days = _DAYS_PER_STANDARD_YEAR - max_missing_days_in_year
    years: list[int] = []
    peaks: list[float] = []
    for year, group in hydrograph_series.groupby(year_labels):
        valid_group = group.dropna()
        if len(valid_group) >= min_valid_days:
            years.append(int(year))
            peaks.append(float(valid_group.max()))

    return pd.Series(peaks, index=years, dtype=float)


def extract_peaks_by_separation_and_threshold(
    hydrograph_series: pd.Series,
    min_peak_separation: pd.Timedelta = _DEFAULT_PEAK_SEPARATION,
    min_peak_quantile: float = 0.9,
) -> pd.Series:
    """Extract Peaks-Over-Threshold (POT) separated by a minimum time window.

    Args:
        hydrograph_series: Time-indexed `pd.Series` of streamflow values.
        min_peak_separation: Minimum time separation required between any two
            extracted peaks.
        min_peak_quantile: Quantile in `(0, 1)` used to set the minimum peak
            discharge threshold.

    Returns:
        A `pd.Series` of independent flood peaks indexed by timestamp.

    Raises:
        TypeError: If `hydrograph_series.index` is not a `pd.DatetimeIndex`.
        ValueError: If `min_peak_separation <= 0` or `min_peak_quantile` is not
            in `(0, 1)`.
        InvalidFlowValueError: If `hydrograph_series` contains negative or
            infinite values.
    """
    _validate_hydrograph_series(hydrograph_series)
    if min_peak_separation <= pd.Timedelta(0):
        raise ValueError(
            f'min_peak_separation must be > 0, got {min_peak_separation}.'
        )
    if not (0.0 < min_peak_quantile < 1.0):
        raise ValueError(
            f'min_peak_quantile must be in (0, 1), got {min_peak_quantile}.'
        )

    threshold = float(hydrograph_series.quantile(min_peak_quantile))
    working_series = hydrograph_series.sort_index().copy(deep=True)

    peak_times: list[pd.Timestamp] = []
    peak_flows: list[float] = []
    while True:
        current_max = working_series.max()
        if pd.isna(current_max) or current_max <= threshold:
            break
        peak_time = working_series.idxmax()
        peak_flows.append(float(current_max))
        peak_times.append(peak_time)
        working_series.loc[
            peak_time - min_peak_separation : peak_time + min_peak_separation
        ] = np.nan

    return pd.Series(peak_flows, index=peak_times, dtype=float)


def extract_n_highest_peaks(
    hydrograph_series: pd.Series,
    num_peaks: int,
    min_peak_separation: pd.Timedelta = _DEFAULT_PEAK_SEPARATION,
) -> pd.Series:
    """Extract the `num_peaks` largest independent flood peaks.

    Args:
        hydrograph_series: Time-indexed `pd.Series` of streamflow values.
        num_peaks: Maximum number of independent peaks to extract.
        min_peak_separation: Minimum time separation required between any two
            extracted peaks.

    Returns:
        A `pd.Series` of up to `num_peaks` independent flood peaks indexed by
        timestamp.

    Raises:
        TypeError: If `hydrograph_series.index` is not a `pd.DatetimeIndex`.
        ValueError: If `num_peaks < 1` or `min_peak_separation <= 0`.
        InvalidFlowValueError: If `hydrograph_series` contains negative or
            infinite values.
    """
    _validate_hydrograph_series(hydrograph_series)
    if num_peaks < 1:
        raise ValueError(f'num_peaks must be >= 1, got {num_peaks}.')
    if min_peak_separation <= pd.Timedelta(0):
        raise ValueError(
            f'min_peak_separation must be > 0, got {min_peak_separation}.'
        )

    working_series = hydrograph_series.sort_index().copy(deep=True)

    peak_times: list[pd.Timestamp] = []
    peak_flows: list[float] = []
    while len(peak_flows) < num_peaks and not working_series.isna().all():
        current_max = float(working_series.max())
        peak_time = working_series.idxmax()
        peak_flows.append(current_max)
        peak_times.append(peak_time)
        working_series.loc[
            peak_time - min_peak_separation : peak_time + min_peak_separation
        ] = np.nan

    return pd.Series(peak_flows, index=peak_times, dtype=float)
