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

"""Differential tests for `BaseTester._calc_exclude_basins`.

The method was rewritten from a per-basin Python loop into a single
vectorized reduction. These tests keep a transcription of the original
implementation and assert the new one agrees with it, including on randomly
generated NaN patterns, so the rewrite is pinned to observed behaviour
rather than to my reading of it.

The one deliberate difference is documented in
`test_short_record_is_not_excluded_by_either`: both implementations must
decline to exclude a basin whose record does not span the evaluation window.
That case is the whole reason the new code carries an explicit guard -- a
plain "is everything in the window NaN?" reduction answers `True` for a
window that is empty or truncated, which the old code never did.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from googlehydrology.evaluation.tester import BaseTester

RECORD_START = '2000-01-01'
RECORD_DAYS = 60
BASINS = [f'b{i:02d}' for i in range(6)]


def _legacy_calc_exclude_basins(tester):
    """Transcription of the pre-vectorization implementation.

    Kept verbatim (modulo the `self.` prefixes) so the differential tests
    compare against what actually shipped, not a paraphrase of it.
    """
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

    for basin in tester.basins:
        basin_ds = tester.dataset._dataset.sel(basin=basin)
        diffs = np.diff(basin_ds.streamflow.isnull(), prepend=[0], append=[0])
        (starts,), (ends,) = np.where(diffs == 1), np.where(diffs == -1)

        nan_date_starts = basin_ds.date.data[starts]
        nan_date_ends = basin_ds.date.data[ends - 1]
        for start, end in zip(period_start, period_end):
            if np.any((nan_date_starts <= start) & (nan_date_ends >= end)):
                yield basin


def _make_tester(values, windows, *, record_start=RECORD_START, days=None):
    """A stub exposing only what `_calc_exclude_basins` actually reads."""
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
        test_start_date=starts,
        test_end_date=ends,
        validation_start_date=starts,
        validation_end_date=ends,
    )
    return SimpleNamespace(
        cfg=cfg,
        period='test',
        basins=list(dataset.basin.values),
        dataset=SimpleNamespace(_dataset=dataset),
    )


def _both(tester):
    return (
        set(_legacy_calc_exclude_basins(tester)),
        set(BaseTester._calc_exclude_basins(tester)),
    )


def test_disabled_flag_excludes_nothing():
    values = np.full((3, RECORD_DAYS), np.nan)
    tester = _make_tester(values, [('2000-01-10', '2000-01-20')])
    tester.cfg.tester_skip_obs_all_nan = False

    legacy, vectorized = _both(tester)
    assert legacy == vectorized == set()


def test_agrees_on_hand_built_patterns():
    """Each basin exercises a different relationship to the window."""
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

    legacy, vectorized = _both(_make_tester(values, [window]))

    assert legacy == vectorized
    assert vectorized == {'b01', 'b02', 'b05'}


def test_agrees_across_multiple_windows():
    """A basin is excluded if *any* configured window is fully missing."""
    windows = [('2000-01-05', '2000-01-08'), ('2000-02-01', '2000-02-05')]
    values = np.ones((3, RECORD_DAYS))
    values[1, 4:8] = np.nan  # covers only the first window
    values[2, 31:36] = np.nan  # covers only the second window

    legacy, vectorized = _both(_make_tester(values, windows))

    assert legacy == vectorized
    assert vectorized == {'b01', 'b02'}


@pytest.mark.parametrize(
    'window',
    [
        ('1999-01-01', '2000-01-20'),  # starts before the record
        ('2000-02-20', '2001-01-01'),  # ends after the record
        ('1998-01-01', '1999-01-01'),  # entirely before the record
    ],
)
def test_short_record_is_not_excluded_by_either(window):
    """The case that forced an explicit guard in the vectorized version.

    Every basin here is entirely NaN, so a naive "is the window all NaN?"
    reduction would exclude all of them -- for an empty window it would even
    reduce over nothing and answer True. The original never did that,
    because a run of NaNs cannot extend past the end of the record.
    """
    values = np.full((3, RECORD_DAYS), np.nan)

    legacy, vectorized = _both(_make_tester(values, [window]))

    assert legacy == vectorized == set()


def test_window_exactly_spanning_the_record_is_excluded():
    """The boundary that must still fire: window == record, all NaN."""
    values = np.full((2, RECORD_DAYS), np.nan)
    last = pd.Timestamp(RECORD_START) + pd.Timedelta(days=RECORD_DAYS - 1)
    window = (RECORD_START, last.strftime('%Y-%m-%d'))

    legacy, vectorized = _both(_make_tester(values, [window]))

    assert legacy == vectorized == {'b00', 'b01'}


def test_agrees_on_randomized_nan_patterns():
    """Fuzz the two implementations against each other.

    Random run lengths and window placements, with the window kept inside
    the record so both implementations are in their agreed domain.
    """
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

        legacy, vectorized = _both(_make_tester(values, [window]))
        assert legacy == vectorized, (
            f'disagreement on window {window}: '
            f'legacy={sorted(legacy)} vectorized={sorted(vectorized)}'
        )


def test_agrees_when_data_is_dask_backed():
    """Both loading modes must give the same answer.

    With `lazy_load: true` the observations are a chunked dask array rather
    than a numpy one, so the reduction builds a graph that has to be
    computed before the basin names can be indexed out of it. Chunking the
    basin axis is what makes this a real test: the reduction has to combine
    partial results across chunks.
    """
    dask_array = pytest.importorskip('dask.array')

    window = ('2000-01-11', '2000-01-20')
    values = np.ones((6, RECORD_DAYS))
    values[1, 10:20] = np.nan  # exactly the window
    values[2, 5:40] = np.nan  # superset of the window
    values[4, 10:19] = np.nan  # one day short
    values[5, :] = np.nan  # everything

    eager = _make_tester(values, [window])
    lazy = _make_tester(values, [window])
    # Chunk across both axes so no single chunk holds a whole basin.
    lazy.dataset._dataset['streamflow'] = (
        ('basin', 'date'),
        dask_array.from_array(values, chunks=(2, 16)),
    )
    lazy.cfg.lazy_load = True

    legacy, vectorized_eager = _both(eager)
    vectorized_lazy = set(BaseTester._calc_exclude_basins(lazy))

    assert legacy == vectorized_eager == vectorized_lazy
    assert vectorized_lazy == {'b01', 'b02', 'b05'}

