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

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Pandas switched from "Y" to "YE" and similar identifiers in 2.2.0. This snippet checks which one is correct for the
# current pandas installation.
_YE_FREQ = 'YE'
_ME_FREQ = 'ME'
_QE_FREQ = 'QE'
try:
    to_offset(_YE_FREQ)
except ValueError:
    _YE_FREQ = 'Y'
    _ME_FREQ = 'M'
    _QE_FREQ = 'Q'


def load_basin_file(basin_file: Path) -> list[str]:
    """Load list of basins from text file.

    Note: Basins names are not allowed to end with '_period*'

    Parameters
    ----------
    basin_file : Path
        Path to a basin txt file. File has to contain one basin id per row, while empty rows are ignored.

    Returns
    -------
    list[str]
        List of basin ids as strings.

    Raises
    ------
    ValueError
        In case of invalid basin names that would cause problems internally.
    """
    with basin_file.open('r') as fp:
        basins = sorted(basin.strip() for basin in fp if basin.strip())

    # sanity check basin names
    problematic_basins = [
        basin for basin in basins if basin.split('_')[-1].startswith('period')
    ]
    if problematic_basins:
        msg = [
            f'The following basin names are invalid {problematic_basins}. Check documentation of the ',
            "'load_basin_file()' functions for details.",
        ]
        raise ValueError(' '.join(msg))

    return basins


def infer_datetime_coord(xr: DataArray | Dataset) -> str:
    """Checks for coordinate with 'date' in its name and returns the name.

    Parameters
    ----------
    xr : DataArray | Dataset
        Array to infer coordinate name of.

    Returns
    -------
    str
        Name of datetime coordinate name.

    Raises
    ------
    RuntimeError
        If none or multiple coordinates with 'date' in its name are found.
    """
    candidates = [c for c in list(xr.coords) if 'date' in c]
    if len(candidates) > 1:
        raise RuntimeError(
            "Found multiple coordinates with 'date' in its name."
        )
    if not candidates:
        raise RuntimeError(
            "Did not find any coordinate with 'date' in its name"
        )

    return candidates[0]


def get_frequency_factor(freq_one: str, freq_two: str) -> float:
    """Get relative factor between the two frequencies.

    Parameters
    ----------
    freq_one : str
        String representation of the first frequency.
    freq_two : str
        String representation of the second frequency.

    Returns
    -------
    float
        Ratio of `freq_one` to `freq_two`.

    Raises
    ------
    ValueError
        If the frequency factor cannot be determined. This can be the case if the frequencies do not represent a fixed
        time delta and are not directly comparable (e.g., because they have the same unit)
        E.g., a month does not represent a fixed time delta. Thus, 1D and 1M are not comparable. However, 1M and 2M are
        comparable since they have the same unit.
    """
    if freq_one == freq_two:
        return 1

    offset_one = to_offset(freq_one)
    offset_two = to_offset(freq_two)
    if offset_one.n < 0 or offset_two.n < 0:
        # Would be possible to implement, but we should never need negative frequencies, so it seems reasonable to
        # fail gracefully rather than to open ourselves to potential unexpected corner cases.
        raise NotImplementedError('Cannot compare negative frequencies.')
    # avoid division by zero errors
    if offset_one.n == offset_two.n == 0:
        return 1
    if offset_two.n == 0:
        return np.inf
    if offset_one.name == offset_two.name:
        return offset_one.n / offset_two.n

    # some simple hard-coded cases
    factor = None
    regex_month_or_day = '-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|MON|TUE|WED|THU|FRI|SAT|SUN)$'
    for i, (one, two) in enumerate(
        [(offset_one, offset_two), (offset_two, offset_one)]
    ):
        # the offset anchor is irrelevant for the ratio between the frequencies, so we remove it from the string
        name_one = re.sub(regex_month_or_day, '', one.name)
        name_two = re.sub(regex_month_or_day, '', two.name)
        if (name_one in ['A', _YE_FREQ] and name_two == _ME_FREQ) or (
            name_one in ['AS', 'YS'] and name_two == 'MS'
        ):
            factor = 12 * one.n / two.n
        if (name_one in ['A', _YE_FREQ] and name_two == _QE_FREQ) or (
            name_one in ['AS', 'YS'] and name_two == 'QS'
        ):
            factor = 4 * one.n / two.n
        if (name_one == _QE_FREQ and name_two == _ME_FREQ) or (
            name_one == 'QS' and name_two == 'MS'
        ):
            factor = 3 * one.n / two.n
        if name_one == 'W' and name_two == 'D':
            factor = 7 * one.n / two.n

        if factor is not None:
            if i == 1:
                return (
                    1 / factor
                )  # `one` was `offset_two`, `two` was `offset_one`
            return factor

    # If all other checks didn't match, we try to convert the frequencies to timedeltas. However, we first need to avoid
    # two cases: (1) pd.to_timedelta currently interprets 'M' as minutes, while it means months in to_offset.
    # (2) Using 'M', 'Y', and 'y' in pd.to_timedelta is deprecated and won't work in the future, so we don't allow it.
    if any(
        re.sub(regex_month_or_day, '', offset.name)
        in ['M', 'Y', 'A', 'y', 'ME', 'YE']
        for offset in [offset_one, offset_two]
    ):
        raise ValueError(
            f'Frequencies {freq_one} and/or {freq_two} are not comparable.'
        )
    try:
        factor = pd.to_timedelta(freq_one) / pd.to_timedelta(freq_two)
    except ValueError as err:
        raise ValueError(
            f'Frequencies {freq_one} and/or {freq_two} are not comparable.'
        ) from err
    return factor
