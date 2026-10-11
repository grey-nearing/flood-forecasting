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

"""Elevation grid loading and vectorized coordinate sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from multimet.catchment_delineation.config import (
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
)
from multimet.catchment_delineation.delineator import CatchmentCoverageError
from multimet.catchment_delineation.tiles import tile_key_to_filename

if TYPE_CHECKING:
    from pathlib import Path
else:
    from pathlib import Path

_MIN_VALID_ELEVATION_M: float = -9000.0
_MAX_VALID_ELEVATION_M: float = 9000.0


class ElevationTiles:
    """Loads and samples 5x5 degree 3-arc-second DEM elevation tiles."""

    def __init__(self, tiles_dir: str | Path) -> None:
        """Initialize the tile loader with a directory of .npy tiles."""
        self.tiles_dir = Path(tiles_dir)
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def tile_path(self, lat_top: int, lon_left: int) -> Path:
        """Return the filesystem path for the (lat_top, lon_left) tile."""
        return self.tiles_dir / tile_key_to_filename(lat_top, lon_left)

    def has_tile(self, lat_top: int, lon_left: int) -> bool:
        """Return True if the requested tile is cached or exists on disk."""
        key = (int(lat_top), int(lon_left))
        return key in self._cache or self.tile_path(*key).exists()

    def get_tile(self, lat_top: int, lon_left: int) -> np.ndarray:
        """Return a memory-mapped (6000, 6000) elevation tile array.

        Raises:
            CatchmentCoverageError: If the tile file does not exist on disk.
            ValueError: If the tile array does not have shape (6000, 6000).
        """
        key = (int(lat_top), int(lon_left))
        if key in self._cache:
            return self._cache[key]

        path = self.tile_path(*key)
        if not path.exists():
            raise CatchmentCoverageError(
                f'Elevation tile {tile_key_to_filename(*key)} not found in'
                f' {self.tiles_dir}.'
            )

        arr = np.load(path, mmap_mode='r')
        if arr.shape != (TILE_CELLS, TILE_CELLS):
            raise ValueError(
                f'Invalid elevation tile shape {arr.shape} for {path.name}; '
                f'expected ({TILE_CELLS}, {TILE_CELLS}).'
            )
        self._cache[key] = arr
        return arr

    def sample(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        *,
        strict: bool = True,
    ) -> np.ndarray:
        """Sample elevation in meters at arbitrary coordinate arrays.

        Args:
            lats: Array of latitudes in decimal degrees.
            lons: Array of longitudes in decimal degrees.
            strict: If True, raises `CatchmentCoverageError` when a requested
                tile file is missing. If False, leaves missing tiles as
                `np.nan`.

        Returns:
            `float32` array of elevations (meters) with the same shape as
            `lats`, with nodata pixels (`< -9000` or `> 9000`) set to `np.nan`.
        """
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = (
            (np.asarray(lons, dtype=np.float64) + 180.0) % 360.0
        ) - 180.0
        if lats_arr.shape != lons_arr.shape:
            raise ValueError(
                f'lats shape {lats_arr.shape} must match lons shape '
                f'{lons_arr.shape}.'
            )

        elev = np.full(lats_arr.shape, np.nan, dtype=np.float32)
        if lats_arr.size == 0:
            return elev

        tile_lat_tops = (np.ceil(lats_arr / TILE_DEG) * TILE_DEG).astype(int)
        tile_lon_lefts = (np.floor(lons_arr / TILE_DEG) * TILE_DEG).astype(int)

        unique_tiles = set(
            zip(
                tile_lat_tops.flatten().tolist(),
                tile_lon_lefts.flatten().tolist(),
                strict=False,
            )
        )

        for t_lat, t_lon in unique_tiles:
            mask = (tile_lat_tops == t_lat) & (tile_lon_lefts == t_lon)
            if not np.any(mask):
                continue

            if not strict and not self.has_tile(t_lat, t_lon):
                continue

            grid = self.get_tile(t_lat, t_lon)

            r_idx = np.round((t_lat - lats_arr[mask]) / RES_DEG).astype(int)
            c_idx = np.round((lons_arr[mask] - t_lon) / RES_DEG).astype(int)
            r_idx = np.clip(r_idx, 0, TILE_CELLS - 1)
            c_idx = np.clip(c_idx, 0, TILE_CELLS - 1)

            sampled = grid[r_idx, c_idx].astype(np.float32)
            sampled[
                (sampled < _MIN_VALID_ELEVATION_M)
                | (sampled > _MAX_VALID_ELEVATION_M)
            ] = np.nan
            elev[mask] = sampled

        return elev


class GlobalElevationGrid:
    """Samples elevation from a memory-mapped global overview DEM `.npy`."""

    def __init__(
        self,
        path: str | Path,
        *,
        res_deg: float,
        lat_bounds: tuple[float, float] = (-56.0, 84.0),
        lon_bounds: tuple[float, float] = (-180.0, 180.0),
        **kwargs: float,
    ) -> None:
        """Initialize the global overview elevation grid."""
        self.path = Path(path)
        self.res_deg = float(res_deg)
        self.top_lat = float(kwargs.get('top_lat', lat_bounds[1]))
        self.bottom_lat = float(kwargs.get('bottom_lat', lat_bounds[0]))
        self.left_lon = float(kwargs.get('left_lon', lon_bounds[0]))
        self.right_lon = float(kwargs.get('right_lon', lon_bounds[1]))
        self._grid: np.ndarray | None = None

    def _load_grid(self) -> np.ndarray:
        if self._grid is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    f'Global elevation grid not found at {self.path}.'
                )
            self._grid = np.load(self.path, mmap_mode='r')
        return self._grid

    def sample(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Sample elevation in meters from the global overview grid."""
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = (
            (np.asarray(lons, dtype=np.float64) + 180.0) % 360.0
        ) - 180.0
        if lats_arr.shape != lons_arr.shape:
            raise ValueError(
                f'lats shape {lats_arr.shape} must match lons shape '
                f'{lons_arr.shape}.'
            )

        out = np.full(lats_arr.shape, np.nan, dtype=np.float32)
        in_bounds = (
            (lats_arr >= self.bottom_lat)
            & (lats_arr <= self.top_lat)
            & (lons_arr >= self.left_lon)
            & (lons_arr <= self.right_lon)
        )
        if not np.any(in_bounds):
            return out

        grid = self._load_grid()
        nrows, ncols = grid.shape
        r_idx = np.round(
            (self.top_lat - lats_arr[in_bounds]) / self.res_deg
        ).astype(int)
        c_idx = np.round(
            (lons_arr[in_bounds] - self.left_lon) / self.res_deg
        ).astype(int)
        r_idx = np.clip(r_idx, 0, nrows - 1)
        c_idx = np.clip(c_idx, 0, ncols - 1)

        sampled = grid[r_idx, c_idx].astype(np.float32)
        sampled[
            (sampled < _MIN_VALID_ELEVATION_M)
            | (sampled > _MAX_VALID_ELEVATION_M)
        ] = np.nan
        out[in_bounds] = sampled
        return out
