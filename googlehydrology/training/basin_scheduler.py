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

"""Chooses which basins are resident in memory for each training epoch."""

import numpy as np


class BasinWindowScheduler:
    """Yields the basins to materialize for a given epoch.

    ``limit_n_basins: W`` trains on W basins at a time, swapping the set each
    epoch so that peak memory is bounded by W rather than by the size of the
    full dataset.

    Which W basins to pick matters more than it first appears:

    * **Coverage.** Choosing a fresh random window each epoch samples *with
      replacement*, so some basins are never trained on. For B=16000 and
      W=100 the fraction never seen after E epochs is ``(1 - W/B)**E`` --
      about 37% after 160 epochs, and still ~4% after 500.
    * **Correlation.** Basin files are ordered by gauge ID, which correlates
      with agency and geography, so a window that is contiguous *in file
      order* is roughly one region. Every gradient in that epoch then comes
      from a spatially correlated sample.
    * **Extraction cost.** A contiguous ``.sel(basin=...)`` against a chunked
      dask array is much cheaper than a scattered one, so windows do want to
      be contiguous in storage order.

    This class satisfies all three by permuting the basin list **once** and
    then walking **disjoint** consecutive windows over that permutation.
    Windows stay contiguous (cheap to extract), coverage becomes exact (every
    basin exactly once per sweep, zero variance), and each window is a
    geographically arbitrary sample.

    The permutation is seeded, so a resumed run reconstructs the identical
    schedule from the epoch number alone. `seed` is therefore required
    whenever rotation is enabled, and has no default.
    """

    def __init__(
        self, basins: list[str], window: int, seed: int | None
    ) -> None:
        if not basins:
            raise ValueError('basins must not be empty.')

        self._basins = list(basins)
        self._window = window
        self._enabled = window > 0

        if self._enabled:
            if seed is None:
                # The schedule is reconstructed from the epoch number alone,
                # which only works if the permutation is identical every
                # time the process starts. An unseeded permutation would
                # still *look* correct within a single run and only lose
                # coverage across a resume, so refuse it outright.
                raise ValueError(
                    'A seed is required when basin rotation is enabled, so '
                    'that a resumed run reproduces the same schedule.'
                )
            permutation = np.random.default_rng(seed).permutation(
                len(self._basins)
            )
            self._order = [self._basins[i] for i in permutation]
            # Ceiling division: the final window of a sweep is short rather
            # than wrapping, so no basin is visited twice within one sweep.
            self._windows_per_sweep = -(-len(self._basins) // window)
        else:
            self._order = self._basins
            self._windows_per_sweep = 1

    @property
    def enabled(self) -> bool:
        """Whether basin rotation is active."""
        return self._enabled

    @property
    def windows_per_sweep(self) -> int:
        """Epochs needed for every basin to be trained on exactly once."""
        return self._windows_per_sweep

    def basins_for_epoch(self, epoch: int) -> list[str] | None:
        """Basins to load for ``epoch``, or ``None`` meaning "all of them".

        ``None`` rather than the full list is returned when rotation is off,
        so callers can hand it straight to ``load_basins()`` and take the
        cheaper no-subsetting path.
        """
        if not self._enabled:
            return None

        index = epoch % self._windows_per_sweep
        start = index * self._window
        return self._order[start : start + self._window]
