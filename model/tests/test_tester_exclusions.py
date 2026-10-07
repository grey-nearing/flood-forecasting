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

"""Tests for `BaseTester._calc_exclude_basins`."""

from types import SimpleNamespace

import dask.array as dask_array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from model.evaluation.tester import BaseTester

RECORD_START = '2000-01-01'
RECORD_DAYS = 60
BASINS = [f'b{i:02d}' for i in range(6)]


def _reference_calc_exclude_basins(tester):
    """Per-basin run-endpoint reference calculation for differential checks."""
    if not tester.cfg.tester_skip_obs_all_nan:
        return

    period_start, period_end = (
        tester.cfg.test_start_date,
        tester.cfg.test_end_date,
    )
    if tester.period == 'validation':
        period_start, period_end = (
            tester.cfg.validation_start_date,
            tester.cfg.validation_end_date,
        )
    elif tester.period == 'train':
        period_start, period_end = (
            tester.cfg.train_start_date,
            tester.cfg.train_end_date,
        )

    for basin in tester.basins:
        basin_ds = tester.dataset.full_dataset.sel(basin=basin)
        diffs = np.diff(basin_ds.streamflow.isnull(), prepend=[0], append=[0])
        (starts,), (ends,) = np.where(diffs == 1), np.where(diffs == -1)

        nan_date_starts = basin_ds.date.data[starts]
        nan_date_ends = basin_ds.date.data[ends - 1]
        for start, end in zip(period_start, period_end, strict=False):
            if np.any((nan_date_starts <= start) & (nan_date_ends >= end)):
                yield basin


def _make_tester(values, windows, *, record_start=RECORD_START, days=None):
    """Build a lightweight stub for `_calc_exclude_basins`."""
    days = values.shape[1] if days is None else days
    dates = pd.date_range(record_start, periods=days, freq='D')
    dataset = xr.Dataset(
        {'streamflow': (('basin', 'date'), values)},
        coords={'basin': list(BASINS[: values.shape[0]]), 'date': dates},
    )
    starts = [pd.Timestamp(s) for s, _ in windows]
    ends = [pd.Timestamp(e) for _, e in windows]
    cfg = SimpleNamespace(
        tester_skip_obs_all_nan=True,
        lazy_load=False,
        train_start_date=starts,
        train_end_date=ends,
        test_start_date=starts,
        test_end_date=ends,
        validation_start_date=starts,
        validation_end_date=ends,
    )
    return SimpleNamespace(
        cfg=cfg,
        period='test',
        basins=list(dataset.basin.values),
        dataset=SimpleNamespace(full_dataset=dataset),
    )


def _both(tester):
    return (
        set(_reference_calc_exclude_basins(tester)),
        set(BaseTester._calc_exclude_basins(tester)),
    )


def test_disabled_flag_excludes_nothing():
    values = np.full((3, RECORD_DAYS), np.nan)
    tester = _make_tester(values, [('2000-01-10', '2000-01-20')])
    tester.cfg.tester_skip_obs_all_nan = False

    reference, vectorized = _both(tester)
    assert reference == vectorized == set()


def test_agrees_on_hand_built_patterns():
    """Each basin exercises a different relationship to the evaluation window."""
    window = ('2000-01-11', '2000-01-20')  # positions 10..19
    values = np.ones((6, RECORD_DAYS))
    # b00: no NaNs at all                                  -> keep
    # b01: NaN exactly over the window                     -> exclude
    values[1, 10:20] = np.nan
    # b02: NaN over a strict superset of the window        -> exclude
    values[2, 5:40] = np.nan
    # b03: NaN over all but the last day of the window     -> keep
    values[3, 10:19] = np.nan
    # b04: two NaN runs that straddle but do not cover it  -> keep
    values[4, 5:15] = np.nan
    values[4, 16:30] = np.nan
    # b05: entire record NaN                               -> exclude
    values[5, :] = np.nan

    reference, vectorized = _both(_make_tester(values, [window]))

    assert reference == vectorized
    assert vectorized == {'b01', 'b02', 'b05'}


def test_agrees_across_multiple_windows():
    """A basin is excluded if any configured window is fully missing."""
    windows = [('2000-01-05', '2000-01-08'), ('2000-02-01', '2000-02-05')]
    values = np.ones((3, RECORD_DAYS))
    values[1, 4:8] = np.nan  # covers only the first window
    values[2, 31:36] = np.nan  # covers only the second window

    reference, vectorized = _both(_make_tester(values, windows))

    assert reference == vectorized
    assert vectorized == {'b01', 'b02'}


def test_respects_train_validation_and_test_period_dates():
    """Each period uses its own configured start and end date windows."""
    values = np.ones((3, RECORD_DAYS))
    values[0, 0:5] = np.nan  # NaN during train window
    values[1, 10:15] = np.nan  # NaN during validation window
    values[2, 20:25] = np.nan  # NaN during test window

    tester = _make_tester(values, [('2000-01-21', '2000-01-25')])
    tester.cfg.train_start_date = [pd.Timestamp('2000-01-01')]
    tester.cfg.train_end_date = [pd.Timestamp('2000-01-05')]
    tester.cfg.validation_start_date = [pd.Timestamp('2000-01-11')]
    tester.cfg.validation_end_date = [pd.Timestamp('2000-01-15')]
    tester.cfg.test_start_date = [pd.Timestamp('2000-01-21')]
    tester.cfg.test_end_date = [pd.Timestamp('2000-01-25')]

    tester.period = 'train'
    assert set(BaseTester._calc_exclude_basins(tester)) == {'b00'}

    tester.period = 'validation'
    assert set(BaseTester._calc_exclude_basins(tester)) == {'b01'}

    tester.period = 'test'
    assert set(BaseTester._calc_exclude_basins(tester)) == {'b02'}


@pytest.mark.parametrize(
    'window',
    [
        ('1999-01-01', '2000-01-20'),  # starts before the record
        ('2000-02-20', '2001-01-01'),  # ends after the record
        ('1998-01-01', '1999-01-01'),  # entirely before the record
    ],
)
def test_short_record_is_not_excluded_by_either(window):
    """Windows extending outside the dataset record do not exclude basins."""
    values = np.full((3, RECORD_DAYS), np.nan)

    reference, vectorized = _both(_make_tester(values, [window]))

    assert reference == vectorized == set()


def test_window_exactly_spanning_the_record_is_excluded():
    """A window equal to the full record excludes all-NaN basins."""
    values = np.full((2, RECORD_DAYS), np.nan)
    last = pd.Timestamp(RECORD_START) + pd.Timedelta(days=RECORD_DAYS - 1)
    window = (RECORD_START, last.strftime('%Y-%m-%d'))

    reference, vectorized = _both(_make_tester(values, [window]))

    assert reference == vectorized == {'b00', 'b01'}


def test_agrees_on_randomized_nan_patterns():
    """Compare reference and vectorized implementations on random NaN runs."""
    rng = np.random.default_rng(20250922)
    dates = pd.date_range(RECORD_START, periods=RECORD_DAYS, freq='D')

    for _ in range(300):
        values = rng.random((len(BASINS), RECORD_DAYS))
        for basin_row in range(len(BASINS)):
            for _ in range(rng.integers(0, 4)):
                lo = int(rng.integers(0, RECORD_DAYS))
                hi = min(RECORD_DAYS, lo + int(rng.integers(1, 30)))
                values[basin_row, lo:hi] = np.nan

        i = int(rng.integers(0, RECORD_DAYS))
        j = int(rng.integers(i, RECORD_DAYS))
        window = (dates[i].strftime('%Y-%m-%d'), dates[j].strftime('%Y-%m-%d'))

        reference, vectorized = _both(_make_tester(values, [window]))
        assert reference == vectorized, (
            f'disagreement on window {window}: '
            f'reference={sorted(reference)} vectorized={sorted(vectorized)}'
        )


def test_agrees_when_data_is_dask_backed():
    """Eager numpy and chunked Dask arrays produce identical exclusions."""
    window = ('2000-01-11', '2000-01-20')
    values = np.ones((6, RECORD_DAYS))
    values[1, 10:20] = np.nan
    values[2, 5:40] = np.nan
    values[4, 10:19] = np.nan
    values[5, :] = np.nan

    eager = _make_tester(values, [window])
    lazy = _make_tester(values, [window])
    lazy.dataset.full_dataset['streamflow'] = (
        ('basin', 'date'),
        dask_array.from_array(values, chunks=(2, 16)),
    )
    lazy.cfg.lazy_load = True

    reference, vectorized_eager = _both(eager)
    vectorized_lazy = set(BaseTester._calc_exclude_basins(lazy))

    assert reference == vectorized_eager == vectorized_lazy
