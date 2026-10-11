# Copyright 2026 Google LLC
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

"""Slippy Map XYZ Tile Server for HydroSHEDS 90m DEM.

Delegates elevation grid loading and sampling to
`multimet.catchment_delineation.elevation` and renders 3D shaded relief
(hillshade) combined with hypsometric topographic tinting as 256x256 PNG tiles.

Multi-Resolution Pyramid Architecture:
- `zoom <= 6`: Samples the global 30-arc-second (`~1 km`) overview grid
  (`hyd_glo_dem_30s.npy`) so world/continental tiles render without reading
  hundreds of 5x5-degree native tiles.
- `7 <= zoom <= 9`: Samples the global 15-arc-second (`~500 m`) overview grid
  (`hyd_glo_dem_15s.npy`), falling back to 30-arc-second if 15s is absent.
- `zoom >= 10`: Samples native 3-arc-second (`~90 m`) 5x5-degree tiles via
  `ElevationTiles`, backfilling any missing coastal or uncached pixels from
  the 15s/30s overview grids.
"""

from __future__ import annotations

import io
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from frontend.config import CACHE_DIR, DEM_DIR
from multimet.catchment_delineation.elevation import (
    ElevationTiles,
    GlobalElevationGrid,
)

HYDROSHEDS_DEM_BASE = DEM_DIR / 'hydrosheds_90m'
if not HYDROSHEDS_DEM_BASE.exists():
    HYDROSHEDS_DEM_BASE = CACHE_DIR / 'hydrosheds_dem'

ELEVATION_TILES_DIR = HYDROSHEDS_DEM_BASE / 'elevation_tiles_5deg'
if not ELEVATION_TILES_DIR.exists():
    ELEVATION_TILES_DIR = CACHE_DIR / 'hydrosheds_dem' / 'elevation_tiles_5deg'

GLOBAL_DEM_30S_NPY = HYDROSHEDS_DEM_BASE / 'hyd_glo_dem_30s.npy'
if not GLOBAL_DEM_30S_NPY.exists():
    GLOBAL_DEM_30S_NPY = CACHE_DIR / 'hydrosheds_dem' / 'hyd_glo_dem_30s.npy'

GLOBAL_DEM_15S_NPY = HYDROSHEDS_DEM_BASE / 'hyd_glo_dem_15s.npy'
if not GLOBAL_DEM_15S_NPY.exists():
    GLOBAL_DEM_15S_NPY = CACHE_DIR / 'hydrosheds_dem' / 'hyd_glo_dem_15s.npy'

GLOBAL_TOP_LAT: float = 84.0
GLOBAL_BOTTOM_LAT: float = -56.0
GLOBAL_LEFT_LON: float = -180.0
GLOBAL_RIGHT_LON: float = 180.0
RES_30S_DEG: float = 1.0 / 120.0
RES_15S_DEG: float = 1.0 / 240.0

_COARSE_OVERVIEW_MAX_ZOOM: int = 6
_MEDIUM_OVERVIEW_MAX_ZOOM: int = 9
_TILE_SIZE_PX: int = 256
_BUFFERED_TILE_SIZE_PX: int = 258
_OCEAN_RGBA: tuple[int, int, int, int] = (15, 23, 42, 255)
_OCEAN_RGB: tuple[int, int, int] = (15, 23, 42)

_BLANK_PNG: bytes | None = None


def get_blank_tile() -> bytes:
    """Return a cached 256x256 dark-ocean RGBA PNG tile."""
    global _BLANK_PNG  # noqa: PLW0603
    if _BLANK_PNG is None:
        img = Image.new('RGBA', (_TILE_SIZE_PX, _TILE_SIZE_PX), _OCEAN_RGBA)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        _BLANK_PNG = buf.getvalue()
    return _BLANK_PNG


def tile_to_latlon_bounds(
    z: int,
    x: int,
    y: int,
) -> tuple[float, float, float, float]:
    """Compute `(min_lat, min_lon, max_lat, max_lon)` for Web Mercator tile."""
    n = 2.0**z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    lat_min = math.degrees(
        math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))
    )
    return min(lat_min, lat_max), lon_min, max(lat_min, lat_max), lon_max


class DemTileServer:
    """Render 256x256 XYZ hillshade tiles from HydroSHEDS DEMs."""

    def __init__(
        self,
        tiles_dir: Path | None = None,
        max_cache_size: int = 2048,
    ) -> None:
        """Initialize the tile server with 3s tiles and overview pyramids."""
        self.tiles_dir = Path(tiles_dir) if tiles_dir else ELEVATION_TILES_DIR
        self._elevation_tiles = ElevationTiles(self.tiles_dir)
        self._global_30s: GlobalElevationGrid | None = (
            GlobalElevationGrid(
                GLOBAL_DEM_30S_NPY,
                res_deg=RES_30S_DEG,
                top_lat=GLOBAL_TOP_LAT,
                bottom_lat=GLOBAL_BOTTOM_LAT,
                left_lon=GLOBAL_LEFT_LON,
                right_lon=GLOBAL_RIGHT_LON,
            )
            if GLOBAL_DEM_30S_NPY.is_file()
            else None
        )
        self._global_15s: GlobalElevationGrid | None = (
            GlobalElevationGrid(
                GLOBAL_DEM_15S_NPY,
                res_deg=RES_15S_DEG,
                top_lat=GLOBAL_TOP_LAT,
                bottom_lat=GLOBAL_BOTTOM_LAT,
                left_lon=GLOBAL_LEFT_LON,
                right_lon=GLOBAL_RIGHT_LON,
            )
            if GLOBAL_DEM_15S_NPY.is_file()
            else None
        )
        self._lru_cache: OrderedDict[tuple[int, int, int], bytes] = (
            OrderedDict()
        )
        self.max_cache_size = max_cache_size

    def get_elevation_tile(
        self,
        lat_top: int,
        lon_left: int,
    ) -> np.ndarray | None:
        """Load a 5x5-degree 3-arc-second elevation array if present on disk."""
        if not self._elevation_tiles.has_tile(lat_top, lon_left):
            return None
        return self._elevation_tiles.get_tile(lat_top, lon_left)

    def sample_elevation(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        zoom: int = 10,
    ) -> np.ndarray:
        """Sample elevation across 30s (`z<=6`), 15s (`z<=9`), and 3s grids."""
        if zoom <= _COARSE_OVERVIEW_MAX_ZOOM and self._global_30s is not None:
            return self._global_30s.sample(lats, lons)

        if zoom <= _MEDIUM_OVERVIEW_MAX_ZOOM:
            if self._global_15s is not None:
                return self._global_15s.sample(lats, lons)
            if self._global_30s is not None:
                return self._global_30s.sample(lats, lons)

        elev = self._elevation_tiles.sample(lats, lons, strict=False)

        missing = np.isnan(elev)
        if np.any(missing):
            if self._global_15s is not None:
                fallback = self._global_15s.sample(lats, lons)
                elev = np.where(missing, fallback, elev)
            elif self._global_30s is not None:
                fallback = self._global_30s.sample(lats, lons)
                elev = np.where(missing, fallback, elev)

        return elev

    def render_tile(self, z: int, x: int, y: int) -> bytes:
        """Render a single 256x256 PNG hillshade map tile for `(z, x, y)`.

        Samples a 1-pixel buffered 258x258 grid so `np.gradient` computes
        seamless finite-difference slopes across tile boundaries. Coastal
        `NaN` ocean pixels are recorded in `ocean_mask` before replacing `NaN`
        with `0.0 m` sea-level elevation for finite-difference stability, and
        then restored to `_OCEAN_RGB` in the final composited image.
        """
        n_tiles = max(1, 1 << max(0, z))
        if y < 0 or y >= n_tiles:
            return get_blank_tile()
        x = x % n_tiles

        cache_key = (z, x, y)
        if cache_key in self._lru_cache:
            self._lru_cache.move_to_end(cache_key)
            return self._lru_cache[cache_key]

        min_lat, min_lon, max_lat, max_lon = tile_to_latlon_bounds(z, x, y)

        if max_lat < GLOBAL_BOTTOM_LAT or min_lat > GLOBAL_TOP_LAT:
            return get_blank_tile()

        px_size_lat = (max_lat - min_lat) / float(_TILE_SIZE_PX)
        px_size_lon = (max_lon - min_lon) / float(_TILE_SIZE_PX)

        grid_lats = np.linspace(
            max_lat + px_size_lat,
            min_lat - px_size_lat,
            _BUFFERED_TILE_SIZE_PX,
        )
        grid_lons = np.linspace(
            min_lon - px_size_lon,
            max_lon + px_size_lon,
            _BUFFERED_TILE_SIZE_PX,
        )
        mesh_lons, mesh_lats = np.meshgrid(grid_lons, grid_lats)

        elev_buffered = self.sample_elevation(mesh_lats, mesh_lons, zoom=z)

        if np.isnan(elev_buffered).all():
            return get_blank_tile()

        ocean_mask = np.isnan(elev_buffered)[1:257, 1:257]
        valid_mask = ~np.isnan(elev_buffered)
        if not np.all(valid_mask):
            elev_buffered = np.where(valid_mask, elev_buffered, 0.0)

        z_factor = max(2.0, min(9.0, 1.8 * math.pow(1.25, max(0, 12 - z))))
        dy, dx = np.gradient(elev_buffered * z_factor)

        slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
        aspect = np.arctan2(-dx, dy)

        azimuth = 315.0 * np.pi / 180.0
        altitude = 45.0 * np.pi / 180.0
        shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(
            slope
        ) * np.cos(azimuth - aspect)
        shaded = (255.0 * (shaded + 1.0) / 2.0).clip(0, 255).astype(np.float32)

        elev_256 = elev_buffered[1:257, 1:257]
        shaded_256 = shaded[1:257, 1:257]

        stops = [0.0, 150.0, 350.0, 750.0, 1500.0, 2500.0, 4000.0]
        colors = np.array(
            [
                [45, 75, 52],
                [85, 118, 72],
                [155, 150, 98],
                [195, 162, 105],
                [158, 102, 68],
                [112, 78, 68],
                [230, 228, 235],
            ],
            dtype=np.float32,
        )

        norm_elev = np.clip(elev_256, stops[0], stops[-1])
        r_grid = np.interp(norm_elev, stops, colors[:, 0])
        g_grid = np.interp(norm_elev, stops, colors[:, 1])
        b_grid = np.interp(norm_elev, stops, colors[:, 2])

        rgb = np.stack([r_grid, g_grid, b_grid], axis=-1)

        intensity = (shaded_256 / 128.0) - 1.0
        blend = np.clip(rgb + intensity[:, :, None] * 70.0, 0, 255).astype(
            np.uint8
        )

        if np.any(ocean_mask):
            blend[ocean_mask] = _OCEAN_RGB

        img = Image.fromarray(blend, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        tile_bytes = buf.getvalue()

        if len(self._lru_cache) >= self.max_cache_size:
            self._lru_cache.popitem(last=False)
        self._lru_cache[cache_key] = tile_bytes

        return tile_bytes


__all__ = [
    'DemTileServer',
    'get_blank_tile',
    'tile_to_latlon_bounds',
]
