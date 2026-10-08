"""Elevation grid loading and vectorized coordinate sampling."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from multimet.catchment_delineation.config import (
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
)
from multimet.catchment_delineation.delineator import CatchmentCoverageError
from multimet.catchment_delineation.tiles import tile_key_to_filename
_MAGIC_NEG_9000_0 = -9000.0
_MAGIC_9000_0 = 9000.0


class ElevationTiles:
    """Loads and samples 5x5 degree 3-arc-second 
        (6000x6000) DEM elevation tiles."""

    def __init__(self, tiles_dir: str | Path) -> None:
        """Docstring."""
        self.tiles_dir = Path(tiles_dir)
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def tile_path(self, lat_top: int, lon_left: int) -> Path:
        """Docstring."""
        return self.tiles_dir / tile_key_to_filename(lat_top, lon_left)

    def has_tile(self, lat_top: int, lon_left: int) -> bool:
        """Docstring."""
        key = (int(lat_top), int(lon_left))
        return key in self._cache or self.tile_path(*key).exists()

    def get_tile(self, lat_top: int, lon_left: int) -> np.ndarray:
        """Returns a memory-mapped (6000, 6000) elevation tile array.

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
        """Samples elevation in meters at arbitrary coordinate arrays.

        Args:
            lats: Array of latitudes in decimal degrees.
            lons: Array of longitudes in decimal degrees.
            strict: If True, raises `CatchmentCoverageError` when a requested 
                tile
              file is missing. If False, leaves missing tiles as `np.nan`.

        Returns:
            `float32` array of elevations 
                (meters) with the same shape as `lats`,
            with nodata pixels (`< -9000` or `> 9000`) set to `np.nan`.
        """
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = (
            (np.asarray(lons, dtype=np.float64) + 180.0) % 360.0
        ) - 180.0
        if lats_arr.shape != lons_arr.shape:
            raise ValueError(
                f'lats shape {lats_arr.shape} must match lons shape 
                    {lons_arr.shape}.'
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
            sampled[(sampled < _MAGIC_NEG_9000_0) | (sampled > _MAGIC_9000_0)] = np.nan
            elev[mask] = sampled

        return elev


class GlobalElevationGrid:
    """Samples elevation from a memory-mapped global overview DEM `.npy` 
        grid."""

    def __init__(
        self,
        path: str | Path,
        *,
        res_deg: float,
        top_lat: float = 84.0,
        bottom_lat: float = -56.0,
        left_lon: float = -180.0,
        right_lon: float = 180.0,
    ) -> None:
        """Docstring."""
        self.path = Path(path)
        self.res_deg = float(res_deg)
        self.top_lat = float(top_lat)
        self.bottom_lat = float(bottom_lat)
        self.left_lon = float(left_lon)
        self.right_lon = float(right_lon)
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
        """Samples elevation in meters from the global overview grid."""
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = (
            (np.asarray(lons, dtype=np.float64) + 180.0) % 360.0
        ) - 180.0
        if lats_arr.shape != lons_arr.shape:
            raise ValueError(
                f'lats shape {lats_arr.shape} must match lons shape 
                    {lons_arr.shape}.'
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
        sampled[(sampled < _MAGIC_NEG_9000_0) | (sampled > _MAGIC_9000_0)] = np.nan
        out[in_bounds] = sampled
        return out
