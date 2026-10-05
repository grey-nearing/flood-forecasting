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

"""Tests for BasinWindowScheduler.

The coverage guarantee is the whole reason this class exists, so it is
asserted directly rather than being left implicit.
"""

import numpy as np
import pytest

from googlehydrology.training.basin_scheduler import BasinWindowScheduler

BASINS = [f'B{i:05d}' for i in range(1000)]


def test_disabled_returns_none_so_caller_uses_the_cheap_path():
    sched = BasinWindowScheduler(BASINS, window=0, seed=None)
    assert not sched.enabled
    assert sched.basins_for_epoch(0) is None
    assert sched.basins_for_epoch(37) is None


def test_negative_window_is_treated_as_disabled():
    assert not BasinWindowScheduler(BASINS, window=-1, seed=None).enabled


def test_empty_basins_rejected():
    with pytest.raises(ValueError, match='must not be empty'):
        BasinWindowScheduler([], window=10, seed=0)


def test_seed_is_required_when_enabled():
    """An unseeded permutation looks fine until the run is resumed.

    Coverage only breaks across a restart, which is far too quiet a failure
    to leave to chance.
    """
    with pytest.raises(ValueError, match='seed is required'):
        BasinWindowScheduler(BASINS, window=100, seed=None)


def test_seed_is_not_required_when_disabled():
    """The default path must not gain a new way to fail."""
    assert not BasinWindowScheduler(BASINS, window=0, seed=None).enabled


def test_window_size_and_sweep_length():
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    assert sched.windows_per_sweep == 10
    assert len(sched.basins_for_epoch(0)) == 100


def test_every_basin_exactly_once_per_sweep():
    """The core guarantee: exact coverage, no repeats, no omissions."""
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    seen = []
    for epoch in range(sched.windows_per_sweep):
        seen.extend(sched.basins_for_epoch(epoch))

    assert len(seen) == len(BASINS), 'sweep length mismatch'
    assert len(set(seen)) == len(BASINS), 'a basin repeated within one sweep'
    assert set(seen) == set(BASINS), 'a basin was omitted'


def test_ragged_window_covers_without_duplicates():
    """B not divisible by W: last window is short, not wrapped."""
    basins = [f'B{i}' for i in range(105)]
    sched = BasinWindowScheduler(basins, window=25, seed=0)
    assert sched.windows_per_sweep == 5  # ceil(105/25)

    seen = []
    for epoch in range(sched.windows_per_sweep):
        seen.extend(sched.basins_for_epoch(epoch))

    assert len(seen) == 105
    assert set(seen) == set(basins)
    assert len(sched.basins_for_epoch(4)) == 5  # the short tail


def test_sweeps_repeat_cyclically():
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    for epoch in range(sched.windows_per_sweep):
        assert (
            sched.basins_for_epoch(epoch)
            == sched.basins_for_epoch(epoch + sched.windows_per_sweep)
        )


def test_seed_is_reproducible_and_resume_safe():
    """A resumed run must rebuild the identical schedule from the epoch."""
    a = BasinWindowScheduler(BASINS, window=100, seed=7)
    b = BasinWindowScheduler(BASINS, window=100, seed=7)
    for epoch in range(25):
        assert a.basins_for_epoch(epoch) == b.basins_for_epoch(epoch)


def test_different_seeds_give_different_schedules():
    a = BasinWindowScheduler(BASINS, window=100, seed=1)
    b = BasinWindowScheduler(BASINS, window=100, seed=2)
    assert a.basins_for_epoch(0) != b.basins_for_epoch(0)


def test_windows_are_shuffled_not_file_ordered():
    """Guards the regional-bias fix.

    A file-ordered contiguous window has neighbouring basins with adjacent
    indices. After permutation the indices in a window should be spread over
    the whole range, so the window is geographically diverse.
    """
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    window = sched.basins_for_epoch(0)
    indices = np.array([BASINS.index(b) for b in window])

    # A file-ordered window of 100 spans ~100. A permuted one should span
    # most of the 1000-wide range.
    assert indices.max() - indices.min() > 500

    # And should not be a monotonic run.
    assert not np.all(np.diff(indices) > 0)


def test_coverage_beats_random_contiguous_starts():
    """Directly contrast against PR #246's scheme at realistic scale."""
    n_basins, window, epochs = 16000, 100, 160
    basins = [f'B{i:05d}' for i in range(n_basins)]

    sched = BasinWindowScheduler(basins, window=window, seed=0)
    covered = set()
    for epoch in range(epochs):
        covered.update(sched.basins_for_epoch(epoch))
    assert len(covered) == n_basins, 'permuted schedule must cover everything'

    # PR #246: fresh random start each epoch, contiguous wrap window.
    rng = np.random.default_rng(0)
    seen = np.zeros(n_basins, dtype=bool)
    for _ in range(epochs):
        start = int(rng.integers(n_basins))
        seen[np.arange(start, start + window) % n_basins] = True
    assert seen.mean() < 0.75, 'sanity: random starts should miss a lot'
