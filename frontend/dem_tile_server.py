"""High-Performance Slippy Map XYZ Tile Server for HydroSHEDS 90m DEM.

Renders authentic 3D shaded relief (hillshade) combined with hypsometric
topographic
tinting directly from the official HydroSHEDS 3 arc-second (90m) conditioned
elevation model.
"""

from collections import OrderedDict
import io
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image

from frontend.config import DEM_DIR

HYDROSHEDS_DEM_BASE = DEM_DIR / "hydrosheds_90m"
if not HYDROSHEDS_DEM_BASE.exists():
  HYDROSHEDS_DEM_BASE = (
      Path.home() / ".cache" / "earthkit_hydro" / "data" / "hydrosheds_dem"
  )

ELEVATION_TILES_DIR = HYDROSHEDS_DEM_BASE / "elevation_tiles_5deg"
if not ELEVATION_TILES_DIR.exists():
  ELEVATION_TILES_DIR = (
      Path.home()
      / ".cache"
      / "earthkit_hydro"
      / "data"
      / "hydrosheds_dem"
      / "elevation_tiles_5deg"
  )

GLOBAL_DEM_30S_NPY = HYDROSHEDS_DEM_BASE / "hyd_glo_dem_30s.npy"
if not GLOBAL_DEM_30S_NPY.exists():
  GLOBAL_DEM_30S_NPY = (
      Path.home()
      / ".cache"
      / "earthkit_hydro"
      / "data"
      / "hydrosheds_dem"
      / "hyd_glo_dem_30s.npy"
  )

GLOBAL_DEM_15S_NPY = HYDROSHEDS_DEM_BASE / "hyd_glo_dem_15s.npy"
if not GLOBAL_DEM_15S_NPY.exists():
  GLOBAL_DEM_15S_NPY = (
      Path.home()
      / ".cache"
      / "earthkit_hydro"
      / "data"
      / "hydrosheds_dem"
      / "hyd_glo_dem_15s.npy"
  )

RES_DEG = 1.0 / 1200.0  # 3 arc-seconds (~90m)
TILE_CELLS = 6000  # 5 deg * 1200 cells/deg
TILE_DEG = 5.0

GLOBAL_TOP_LAT = 84.0
GLOBAL_BOTTOM_LAT = -56.0
GLOBAL_LEFT_LON = -180.0
GLOBAL_RIGHT_LON = 180.0
RES_30S_DEG = 1.0 / 120.0  # 30 arc-seconds (~900m)
RES_15S_DEG = 1.0 / 240.0  # 15 arc-seconds (~450m)

# Blank dark ocean 256x256 PNG
_BLANK_PNG: Optional[bytes] = None


def get_blank_tile() -> bytes:
  global _BLANK_PNG
  if _BLANK_PNG is None:
    img = Image.new("RGBA", (256, 256), (15, 23, 42, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _BLANK_PNG = buf.getvalue()
  return _BLANK_PNG


def tile_to_latlon_bounds(
    z: int, x: int, y: int
) -> Tuple[float, float, float, float]:
  """Computes (min_lat, min_lon, max_lat, max_lon) for standard Web Mercator tile."""
  n = 2.0**z
  lon_min = x / n * 360.0 - 180.0
  lon_max = (x + 1) / n * 360.0 - 180.0
  lat_max = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
  lat_min = math.degrees(
      math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))
  )
  return min(lat_min, lat_max), lon_min, max(lat_min, lat_max), lon_max


class DemTileServer:
  """Renders 256x256 XYZ raster tiles from the global HydroSHEDS elevation dataset."""

  def __init__(
      self, tiles_dir: Optional[Path] = None, max_cache_size: int = 2048
  ):
    self.tiles_dir = Path(tiles_dir) if tiles_dir else ELEVATION_TILES_DIR
    self._tile_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    self._global_30s: Optional[np.ndarray] = None
    self._global_15s: Optional[np.ndarray] = None
    self._lru_cache: OrderedDict[Tuple[int, int, int], bytes] = OrderedDict()
    self.max_cache_size = max_cache_size

  def _get_global_30s(self) -> Optional[np.ndarray]:
    if self._global_30s is None and GLOBAL_DEM_30S_NPY.exists():
      try:
        self._global_30s = np.load(GLOBAL_DEM_30S_NPY, mmap_mode="r")
      except Exception:
        self._global_30s = None
    return self._global_30s

  def _get_global_15s(self) -> Optional[np.ndarray]:
    if self._global_15s is None and GLOBAL_DEM_15S_NPY.exists():
      try:
        self._global_15s = np.load(GLOBAL_DEM_15S_NPY, mmap_mode="r")
      except Exception:
        self._global_15s = None
    return self._global_15s

  def get_elevation_tile(
      self, lat_top: int, lon_left: int
  ) -> Optional[np.ndarray]:
    """Loads a 5x5 degree 3-arc-second elevation array with memory-mapping."""
    key = (lat_top, lon_left)
    if key in self._tile_cache:
      return self._tile_cache[key]

    lat_str = f"n{key[0]:02d}" if key[0] >= 0 else f"s{abs(key[0]):02d}"
    lon_str = f"w{abs(key[1]):03d}" if key[1] < 0 else f"e{key[1]:03d}"
    path = self.tiles_dir / f"{lat_str}{lon_str}.npy"

    if not path.exists():
      self._tile_cache[key] = None
      return None

    try:
      arr = np.load(path, mmap_mode="r")
      self._tile_cache[key] = arr
      return arr
    except Exception:
      self._tile_cache[key] = None
      return None

  def _sample_from_global_grid(
      self,
      lats: np.ndarray,
      lons: np.ndarray,
      grid: np.ndarray,
      res_deg: float,
  ) -> np.ndarray:
    """Samples elevation from a global memory-mapped HydroSHEDS grid (-180..180, -56..84)."""
    out = np.full(lats.shape, np.nan, dtype=np.float32)
    in_bounds = (
        (lats >= GLOBAL_BOTTOM_LAT)
        & (lats <= GLOBAL_TOP_LAT)
        & (lons >= GLOBAL_LEFT_LON)
        & (lons <= GLOBAL_RIGHT_LON)
    )
    if not np.any(in_bounds):
      return out

    nrows, ncols = grid.shape
    r_idx = np.floor((GLOBAL_TOP_LAT - lats[in_bounds]) / res_deg).astype(int)
    c_idx = np.floor((lons[in_bounds] - GLOBAL_LEFT_LON) / res_deg).astype(int)
    r_idx = np.clip(r_idx, 0, nrows - 1)
    c_idx = np.clip(c_idx, 0, ncols - 1)

    sampled = grid[r_idx, c_idx].astype(np.float32)
    sampled[(sampled < -9000) | (sampled > 9000)] = np.nan
    out[in_bounds] = sampled
    return out

  def sample_elevation(
      self, lats: np.ndarray, lons: np.ndarray, zoom: int = 10
  ) -> np.ndarray:
    """Vectorized extraction of elevations across global pyramids and 5x5 degree 3s tiles."""
    # Wrap longitudes into [-180, 180]
    lons = ((lons + 180.0) % 360.0) - 180.0

    # At global/continental zooms (z <= 6), 30s (~900m) global pyramid is finer than pixel resolution
    if zoom <= 6:
      g30 = self._get_global_30s()
      if g30 is not None:
        return self._sample_from_global_grid(lats, lons, g30, RES_30S_DEG)

    # At regional zooms (7 <= z <= 9), 15s (~450m) global pyramid is finer than pixel resolution
    if zoom <= 9:
      g15 = self._get_global_15s()
      if g15 is not None:
        return self._sample_from_global_grid(lats, lons, g15, RES_15S_DEG)
      g30 = self._get_global_30s()
      if g30 is not None:
        return self._sample_from_global_grid(lats, lons, g30, RES_30S_DEG)

    h, w = lats.shape
    elev = np.full((h, w), np.nan, dtype=np.float32)

    # Identify unique 5x5 degree 3s tiles needed
    tile_lat_tops = np.ceil(lats / TILE_DEG) * TILE_DEG
    tile_lon_lefts = np.floor(lons / TILE_DEG) * TILE_DEG

    unique_tiles = set(
        zip(
            tile_lat_tops.flatten().astype(int),
            tile_lon_lefts.flatten().astype(int),
        )
    )

    for t_lat, t_lon in unique_tiles:
      mask = (tile_lat_tops == t_lat) & (tile_lon_lefts == t_lon)
      if not np.any(mask):
        continue

      grid = self.get_elevation_tile(t_lat, t_lon)
      if grid is None:
        continue

      # Row and col indices in this tile
      r_idx = np.round((t_lat - lats[mask]) / RES_DEG).astype(int)
      c_idx = np.round((lons[mask] - t_lon) / RES_DEG).astype(int)

      r_idx = np.clip(r_idx, 0, TILE_CELLS - 1)
      c_idx = np.clip(c_idx, 0, TILE_CELLS - 1)

      sampled = grid[r_idx, c_idx].astype(np.float32)
      # Replace invalid nodata values (< -9000 or > 9000) with NaN
      sampled[(sampled < -9000) | (sampled > 9000)] = np.nan
      elev[mask] = sampled

    # Fill any unpopulated regions (continents without local 3s tiles) from the global 15s/30s DEM
    missing = np.isnan(elev)
    if np.any(missing):
      g15 = self._get_global_15s()
      if g15 is not None:
        fallback = self._sample_from_global_grid(lats, lons, g15, RES_15S_DEG)
        elev = np.where(missing, fallback, elev)
      else:
        g30 = self._get_global_30s()
        if g30 is not None:
          fallback = self._sample_from_global_grid(lats, lons, g30, RES_30S_DEG)
          elev = np.where(missing, fallback, elev)

    return elev

  def render_tile(self, z: int, x: int, y: int) -> bytes:
    """Renders a single 256x256 PNG map tile for (z, x, y)."""
    n_tiles = max(1, 1 << max(0, z))
    if y < 0 or y >= n_tiles:
      return get_blank_tile()
    x = x % n_tiles

    cache_key = (z, x, y)
    if cache_key in self._lru_cache:
      self._lru_cache.move_to_end(cache_key)
      return self._lru_cache[cache_key]

    min_lat, min_lon, max_lat, max_lon = tile_to_latlon_bounds(z, x, y)

    # Skip only polar tiles completely outside global HydroSHEDS coverage (-56..84 deg latitude)
    if max_lat < GLOBAL_BOTTOM_LAT or min_lat > GLOBAL_TOP_LAT:
      return get_blank_tile()

    # Create coordinate grid with 1-pixel buffer for seamless gradient computation (258x258)
    px_size_lat = (max_lat - min_lat) / 256.0
    px_size_lon = (max_lon - min_lon) / 256.0

    grid_lats = np.linspace(max_lat + px_size_lat, min_lat - px_size_lat, 258)
    grid_lons = np.linspace(min_lon - px_size_lon, max_lon + px_size_lon, 258)
    mesh_lons, mesh_lats = np.meshgrid(grid_lons, grid_lats)

    elev_buffered = self.sample_elevation(mesh_lats, mesh_lons, zoom=z)

    if np.isnan(elev_buffered).all():
      return get_blank_tile()

    ocean_mask = np.isnan(elev_buffered)[1:257, 1:257]
    valid_mask = ~np.isnan(elev_buffered)
    if not np.all(valid_mask):
      elev_buffered = np.where(valid_mask, elev_buffered, 0.0)

    # 1. 3D Hillshade Calculation
    # Dynamic vertical exaggeration scaled for natural terrain relief
    z_factor = max(2.0, min(9.0, 1.8 * math.pow(1.25, max(0, 12 - z))))
    dy, dx = np.gradient(elev_buffered * z_factor)

    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    # Solar illumination from Northwest (315 deg azimuth, 45 deg altitude)
    azimuth = 315.0 * np.pi / 180.0
    altitude = 45.0 * np.pi / 180.0
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(
        slope
    ) * np.cos(azimuth - aspect)
    shaded = (255.0 * (shaded + 1.0) / 2.0).clip(0, 255).astype(np.float32)

    # Crop back to 256x256
    elev_256 = elev_buffered[1:257, 1:257]
    shaded_256 = shaded[1:257, 1:257]

    # 2. Hypsometric Topographic Color Tinting (Meters -> RGB)
    stops = [0.0, 150.0, 350.0, 750.0, 1500.0, 2500.0, 4000.0]
    colors = np.array(
        [
            [45, 75, 52],  # Lowland lush green
            [85, 118, 72],  # Valley soft olive
            [155, 150, 98],  # Low hills warm tan
            [195, 162, 105],  # Plateau golden
            [158, 102, 68],  # Mountain ridge terracotta
            [112, 78, 68],  # High rocky peak
            [230, 228, 235],  # Summit alpine rock/snow
        ],
        dtype=np.float32,
    )

    # Normalize and interpolate colors
    norm_elev = np.clip(elev_256, stops[0], stops[-1])
    r_grid = np.interp(norm_elev, stops, colors[:, 0])
    g_grid = np.interp(norm_elev, stops, colors[:, 1])
    b_grid = np.interp(norm_elev, stops, colors[:, 2])

    rgb = np.stack([r_grid, g_grid, b_grid], axis=-1)

    # 3. Soft-light lighting blend: RGB + intensity * contrast
    intensity = (shaded_256 / 128.0) - 1.0  # -1.0 (shadow) to +1.0 (sunlight)
    blend = np.clip(rgb + intensity[:, :, None] * 70.0, 0, 255).astype(np.uint8)

    # Render ocean / nodata pixels with dark slate ocean background
    if np.any(ocean_mask):
      blend[ocean_mask] = (15, 23, 42)

    # Generate PNG bytes
    img = Image.fromarray(blend, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    tile_bytes = buf.getvalue()

    # Update LRU cache
    if len(self._lru_cache) >= self.max_cache_size:
      self._lru_cache.popitem(last=False)
    self._lru_cache[cache_key] = tile_bytes

    return tile_bytes
