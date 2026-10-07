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

"""Tests for BasinWindowScheduler."""

import numpy as np
import pytest

from model.training.basin_scheduler import BasinWindowScheduler

BASINS = [f'B{i:05d}' for i in range(1000)]


def test_disabled_returns_none():
    sched = BasinWindowScheduler(BASINS, window=0, seed=None)
    assert not sched.enabled
    assert sched.windows_per_sweep == 1
    assert sched.basins_for_epoch(1) is None
    assert sched.basins_for_epoch(37) is None


@pytest.mark.parametrize('invalid_window', [-1, -10, True, False, 1.5, '10'])
def test_invalid_window_rejected(invalid_window):
    with pytest.raises(ValueError, match='window must be a non-negative integer'):
        BasinWindowScheduler(BASINS, window=invalid_window, seed=0)


@pytest.mark.parametrize('invalid_epoch', [0, -1, True, False, 1.5, '1'])
def test_invalid_epoch_rejected(invalid_epoch):
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    with pytest.raises(ValueError, match='epoch must be a positive integer'):
        sched.basins_for_epoch(invalid_epoch)


def test_empty_basins_rejected():
    with pytest.raises(ValueError, match='must not be empty'):
        BasinWindowScheduler([], window=10, seed=0)


def test_seed_is_required_when_enabled():
    with pytest.raises(ValueError, match='seed is required'):
        BasinWindowScheduler(BASINS, window=100, seed=None)


def test_seed_is_not_required_when_disabled():
    assert not BasinWindowScheduler(BASINS, window=0, seed=None).enabled


def test_window_at_least_pool_size_uses_single_window_sweep():
    sched = BasinWindowScheduler(BASINS[:10], window=10, seed=0)
    assert sched.enabled
    assert sched.windows_per_sweep == 1
    assert set(sched.basins_for_epoch(1)) == set(BASINS[:10])


def test_window_size_and_sweep_length():
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    assert sched.windows_per_sweep == 10
    assert len(sched.basins_for_epoch(1)) == 100


def test_every_basin_visited_once_per_1_indexed_sweep():
    """Epochs 1..windows_per_sweep visit every basin exactly once."""
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    seen = []
    for epoch in range(1, sched.windows_per_sweep + 1):
        seen.extend(sched.basins_for_epoch(epoch))

    assert len(seen) == len(BASINS)
    assert len(set(seen)) == len(BASINS)
    assert set(seen) == set(BASINS)


def test_ragged_window_visits_full_windows_first_and_short_tail_last():
    """When B is not divisible by W, epochs 1..K-1 are full and epoch K is the tail."""
    basins = [f'B{i}' for i in range(105)]
    sched = BasinWindowScheduler(basins, window=25, seed=0)
    assert sched.windows_per_sweep == 5  # ceil(105 / 25)

    lengths = [
        len(sched.basins_for_epoch(epoch))
        for epoch in range(1, sched.windows_per_sweep + 1)
    ]
    assert lengths == [25, 25, 25, 25, 5]

    seen = [
        b
        for epoch in range(1, sched.windows_per_sweep + 1)
        for b in sched.basins_for_epoch(epoch)
    ]
    assert len(seen) == 105
    assert set(seen) == set(basins)


def test_sweeps_repeat_cyclically():
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    for epoch in range(1, sched.windows_per_sweep + 1):
        assert sched.basins_for_epoch(epoch) == sched.basins_for_epoch(
            epoch + sched.windows_per_sweep
        )


def test_seed_is_reproducible_and_resume_safe():
    a = BasinWindowScheduler(BASINS, window=100, seed=7)
    b = BasinWindowScheduler(BASINS, window=100, seed=7)
    for epoch in range(1, 26):
        assert a.basins_for_epoch(epoch) == b.basins_for_epoch(epoch)


def test_different_seeds_give_different_schedules():
    a = BasinWindowScheduler(BASINS, window=100, seed=1)
    b = BasinWindowScheduler(BASINS, window=100, seed=2)
    assert a.basins_for_epoch(1) != b.basins_for_epoch(1)


def test_windows_are_shuffled_not_file_ordered():
    sched = BasinWindowScheduler(BASINS, window=100, seed=0)
    window = sched.basins_for_epoch(1)
    indices = np.array([BASINS.index(b) for b in window])

    assert indices.max() - indices.min() > 500
    assert not np.all(np.diff(indices) > 0)
