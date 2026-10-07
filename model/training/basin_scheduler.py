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

"""Selects which basins are loaded into memory for each training epoch."""

import numpy as np


class BasinWindowScheduler:
    """Selects a subset of basins to load into memory for each training epoch.

    When ``max_basins_in_memory`` is set to a positive integer ``window``,
    this scheduler shuffles the full basin list once using ``seed`` and splits
    it into non-overlapping groups of at most ``window`` basins. Each training
    epoch (1-indexed) loads the next group in order, and the cycle repeats once
    every basin has been visited.

    Args:
        basins: Full list of training basin IDs. Must not be empty.
        window: Maximum number of basins to load per epoch. ``0`` disables
            windowing and loads all basins at once.
        seed: Random seed used to shuffle the basin list. Required when
            ``window > 0`` so resumed runs use the same basin order.

    Raises:
        ValueError: If ``basins`` is empty, ``window`` is negative or not an
            integer, or ``window > 0`` and ``seed`` is ``None``.
    """

    def __init__(
        self, basins: list[str], window: int, seed: int | None
    ) -> None:
        if not basins:
            raise ValueError('basins must not be empty.')
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError(
                f'window must be a non-negative integer (>= 0), got {window!r}.'
            )

        self._basins = list(basins)
        self._window = window
        self._enabled = window > 0

        if self._enabled:
            if seed is None:
                raise ValueError(
                    'A seed is required when basin rotation is enabled, so '
                    'that a resumed run reproduces the same schedule.'
                )
            permutation = np.random.default_rng(seed).permutation(
                len(self._basins)
            )
            self._order = [self._basins[i] for i in permutation]
            self._windows_per_sweep = -(-len(self._basins) // window)
        else:
            self._order = self._basins
            self._windows_per_sweep = 1

    @property
    def enabled(self) -> bool:
        """Whether per-epoch basin rotation is active."""
        return self._enabled

    @property
    def windows_per_sweep(self) -> int:
        """Number of epochs needed to visit every basin once."""
        return self._windows_per_sweep

    def basins_for_epoch(self, epoch: int) -> list[str] | None:
        """Return the basin list for 1-indexed ``epoch``, or ``None`` for all basins.

        Args:
            epoch: 1-indexed training epoch number (``>= 1``).

        Returns:
            List of basin IDs for the given epoch when rotation is enabled, or
            ``None`` when rotation is disabled (``window == 0``).

        Raises:
            ValueError: If ``epoch`` is not an integer ``>= 1``.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError(
                f'epoch must be a positive integer (>= 1), got {epoch!r}.'
            )
        if not self._enabled:
            return None

        index = (epoch - 1) % self._windows_per_sweep
        start = index * self._window
        return self._order[start : start + self._window]
