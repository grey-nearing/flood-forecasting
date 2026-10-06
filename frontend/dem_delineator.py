"""Pure DEM Flow-Direction Watershed Delineator adapter for OpenHydroNet.

Wraps the core `multimet.catchment_delineation` package (`DemDelineator`) on `main`
and adds multi-DEM dataset resolution (`hydrosheds_90m` vs `merit_hydro_90m`)
plus local HydroSHEDS / MERIT-Hydro 5x5 tile directory discovery.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from frontend.config import (
    CACHE_DIR,
    DEM_DIR,
    ensure_flood_forecasting_on_sys_path,
)

ensure_flood_forecasting_on_sys_path()

from multimet.catchment_delineation import (  # pylint: disable=g-import-not-at-top
    CatchmentAreaMismatchError,
    CatchmentCoverageError,
    DEM_MAX_LAT,
    DEM_MAX_LON,
    DEM_MIN_LAT,
    DEM_MIN_LON,
    DemDelineator as FloodForecastingDemDelineator,
    INFLOW_MAP,
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
    download_tile_from_gcs,
    download_tiles_for_bbox,
    is_tile_available,
    latlon_to_tile_key,
    list_available_tiles,
    tile_key_to_filename,
)
from multimet.catchment_delineation.tiles import is_tile_in_coverage  # pylint: disable=g-import-not-at-top

logger = logging.getLogger(__name__)

# Primary & Fallback DEM Tile Directories
HYDROSHEDS_DEM_TILES_DIR = DEM_DIR / "hydrosheds_90m" / "tiles_5deg"
MERIT_DEM_TILES_DIR = DEM_DIR / "merit_hydro_90m" / "tiles_5deg"
CACHE_TILES_DIR = CACHE_DIR / "hydrosheds_dem" / "tiles_5deg"
CACHE_MERIT_TILES_DIR = CACHE_DIR / "merit_dem" / "tiles_5deg"
CACHE_MERIT_BASINS_TILES_DIR = CACHE_DIR / "merit_basins" / "tiles_5deg"

DEFAULT_HYDROSHEDS_GCS_URI = os.environ.get(
    "EARTHKIT_HYDRO_DEM_GCS_URI",
    "gs://Google_hydrology_bucket/dem_tiles/hydrosheds_90m/tiles_5deg",
)
DEFAULT_MERIT_GCS_URI = os.environ.get(
    "EARTHKIT_MERIT_DEM_GCS_URI",
    "",
)

DEM_DATASETS: Dict[str, Dict[str, Any]] = {
    "hydrosheds_90m": {
        "id": "hydrosheds_90m",
        "name": "HydroSHEDS 90m Conditioned DEM (3 arc-sec)",
        "resolution": "90m (3 arc-second)",
        "tiles_dir": HYDROSHEDS_DEM_TILES_DIR,
        "fallback_dirs": [
            CACHE_TILES_DIR,
            Path.home() / ".cache" / "googlehydrology" / "dem_tiles",
        ],
        "gcs_uri": DEFAULT_HYDROSHEDS_GCS_URI,
        "river_network_id": "hydroatlas",
        "citation": "Lehner et al. (2008), HydroSHEDS Technical Documentation",
    },
    "merit_hydro_90m": {
        "id": "merit_hydro_90m",
        "name": "MERIT-Hydro 90m DEM (3 arc-sec)",
        "resolution": "90m (3 arc-second)",
        "tiles_dir": MERIT_DEM_TILES_DIR,
        "fallback_dirs": [
            CACHE_MERIT_TILES_DIR,
            CACHE_MERIT_BASINS_TILES_DIR,
        ],
        "gcs_uri": DEFAULT_MERIT_GCS_URI,
        "river_network_id": "merit-hydro",
        "citation": "Yamazaki et al. (2019), Water Resources Research",
    },
}

DEM_ALIASES: Dict[str, str] = {
    "hydrosheds_90m": "hydrosheds_90m",
    "hydrosheds": "hydrosheds_90m",
    "hydroatlas": "hydrosheds_90m",
    "merit_hydro_90m": "merit_hydro_90m",
    "merit-hydro": "merit_hydro_90m",
    "merit_hydro": "merit_hydro_90m",
}

DEFAULT_DEM_ID = "hydrosheds_90m"


def resolve_dem_id(dem_or_dataset_id: Optional[str]) -> str:
  """Resolves a DEM or dataset identifier to a canonical DEM_DATASETS key."""
  if not dem_or_dataset_id:
    return DEFAULT_DEM_ID
  key = str(dem_or_dataset_id).strip().lower()
  if key in DEM_DATASETS:
    return key
  if key in DEM_ALIASES:
    return DEM_ALIASES[key]
  raise ValueError(
      f"Unknown DEM / hydrography dataset: {dem_or_dataset_id!r}. "
      f"Supported datasets: {sorted(DEM_DATASETS.keys())}"
  )


def resolve_dem_tiles_dir(dem_id: str = DEFAULT_DEM_ID) -> Path:
  """Resolves the active 5x5 degree D8 flow-direction tiles directory for a DEM."""
  canonical = resolve_dem_id(dem_id)
  info = DEM_DATASETS[canonical]
  primary: Path = info["tiles_dir"]
  if primary.exists() and any(primary.glob("*.npy")):
    return primary
  for fb in info.get("fallback_dirs", []):
    if fb.exists() and any(fb.glob("*.npy")):
      return fb
  # Prefer existing fallback cache directory if present so on-demand downloads land in user cache
  for fb in info.get("fallback_dirs", []):
    if fb.parent.exists():
      try:
        fb.mkdir(parents=True, exist_ok=True)
        return fb
      except OSError:
        pass
  return primary


class DemDelineator(FloodForecastingDemDelineator):
  """Multi-tile DEM watershed delineator extending multimet.catchment_delineation.DemDelineator."""

  def __init__(
      self,
      tiles_dir: Optional[Union[str, Path]] = None,
      cache_tiles: bool = True,
      auto_download_gcs: bool = True,
      gcs_uri: Optional[str] = None,
      dem_id: str = DEFAULT_DEM_ID,
  ):
    self.dem_id = resolve_dem_id(dem_id)
    dem_info = DEM_DATASETS[self.dem_id]
    resolved_tiles_dir = (
        Path(tiles_dir)
        if tiles_dir is not None
        else resolve_dem_tiles_dir(self.dem_id)
    )
    resolved_gcs_uri = (
        gcs_uri if gcs_uri is not None else dem_info.get("gcs_uri") or None
    )
    try:
      resolved_tiles_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
      pass
    super().__init__(
        tiles_dir=resolved_tiles_dir,
        cache_tiles=cache_tiles,
    )
    self.auto_download_gcs = auto_download_gcs
    self.gcs_uri = resolved_gcs_uri

  def get_tile(self, lat_top: float, lon_left: float) -> np.ndarray:
    """Routes parent get_tile() calls through _load_tile() for multi-DEM support."""
    return self._load_tile(lat_top, lon_left)

  def _load_tile(self, lat_top: float, lon_left: float) -> np.ndarray:
    """Loads a 5x5 degree (6000, 6000) uint8 D8 tile for HydroSHEDS or MERIT-Hydro."""
    tile_lat = int(round(lat_top))
    tile_lon = int(round(lon_left))
    key = (tile_lat, tile_lon)
    if self.cache_tiles and key in self._tile_cache:
      return self._tile_cache[key]

    tile_name = tile_key_to_filename(tile_lat, tile_lon)
    if not is_tile_in_coverage(tile_lat, tile_lon):
      raise CatchmentCoverageError(
          f"DEM tile {tile_name} is outside the global DEM coverage "
          f"domain ({DEM_MIN_LAT}° to {DEM_MAX_LAT}° latitude, "
          f"{DEM_MIN_LON}° to {DEM_MAX_LON}° longitude)."
      )

    assert self.tiles_dir is not None
    local_npy = self.tiles_dir / tile_name
    if not local_npy.exists():
      if self.auto_download_gcs:
        if self.dem_id == "merit_hydro_90m" and not self.gcs_uri:
          from frontend.tools.download_merit_d8_tiles import download_merit_d8_tile  # pylint: disable=g-import-not-at-top

          download_merit_d8_tile(tile_lat, tile_lon, target_dir=self.tiles_dir)
        elif self.gcs_uri:
          download_tile_from_gcs(
              tile_lat,
              tile_lon,
              target_dir=self.tiles_dir,
              source_uri=self.gcs_uri,
          )
      if not local_npy.exists():
        raise FileNotFoundError(
            f"Missing flow direction tile for ({tile_lat}, {tile_lon}) in "
            f"{self.tiles_dir} (expected {local_npy.name})."
        )

    arr = np.load(local_npy, mmap_mode="r")
    if self.cache_tiles:
      self._tile_cache[key] = arr
    return arr

  def delineate(
      self,
      lat: float,
      lon: float,
      snap_window_cells: int = 12,
      max_cells: Optional[int] = None,
      simplify_tolerance: Optional[float] = None,
      catchment_id: Optional[str] = None,
      expected_area_km2: Optional[float] = None,
      area_tolerance: float = 0.50,
  ) -> Dict[str, Any]:
    """Delineates catchment and attaches active DEM metadata."""
    feature = super().delineate(
        lat=lat,
        lon=lon,
        snap_window_cells=snap_window_cells,
        max_cells=max_cells,
        simplify_tolerance=simplify_tolerance,
        catchment_id=catchment_id,
        expected_area_km2=expected_area_km2,
        area_tolerance=area_tolerance,
    )
    dem_info = DEM_DATASETS.get(self.dem_id, DEM_DATASETS[DEFAULT_DEM_ID])
    props = feature.get("properties", {})
    props["dem_id"] = self.dem_id
    props["dem_name"] = dem_info["name"]
    if self.dem_id == "merit_hydro_90m":
      props["delineation_method"] = (
          "DEM Digital Elevation Flow-Routing "
          "(90m MERIT-Hydro Multi-Tile Seamless Grid)"
      )
    return feature


_DELINEATORS: Dict[str, DemDelineator] = {}


def get_dem_delineator(dem_id: str = DEFAULT_DEM_ID) -> DemDelineator:
  """Returns a singleton DemDelineator instance for the requested DEM."""
  canonical = resolve_dem_id(dem_id)
  if canonical not in _DELINEATORS:
    _DELINEATORS[canonical] = DemDelineator(dem_id=canonical)
  return _DELINEATORS[canonical]


__all__ = [
    "CatchmentAreaMismatchError",
    "CatchmentCoverageError",
    "DEFAULT_DEM_ID",
    "DEM_DATASETS",
    "DEM_MAX_LAT",
    "DEM_MAX_LON",
    "DEM_MIN_LAT",
    "DEM_MIN_LON",
    "DemDelineator",
    "FloodForecastingDemDelineator",
    "HYDROSHEDS_DEM_TILES_DIR",
    "INFLOW_MAP",
    "MERIT_DEM_TILES_DIR",
    "RES_DEG",
    "TILE_CELLS",
    "TILE_DEG",
    "download_tile_from_gcs",
    "download_tiles_for_bbox",
    "get_dem_delineator",
    "is_tile_available",
    "latlon_to_tile_key",
    "list_available_tiles",
    "resolve_dem_id",
    "resolve_dem_tiles_dir",
    "tile_key_to_filename",
]
