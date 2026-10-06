"""Configuration module for the OpenHydroNet Interactive Web Platform."""

import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Repository root for OpenHydroNet (flood-forecasting)
FLOOD_FORECASTING_REPO_DIR = Path(
    os.environ.get(
        "FLOOD_FORECASTING_REPO_DIR",
        str(BASE_DIR.parent),
    )
)

if os.environ.get("TEST_TMPDIR"):
  tmp_root = Path(os.environ["TEST_TMPDIR"])
  DATA_DIR = tmp_root / "data"
else:
  DATA_DIR = BASE_DIR / "data"

# Shared Read-Only (or Admin-Synced) Global Reference Datasets
SHARED_DIR = DATA_DIR / "shared"
SHARED_DEMS_DIR = SHARED_DIR / "dems"
SHARED_HYDROATLAS_DIR = SHARED_DIR / "hydroatlas"
SHARED_HYDRORIVERS_DIR = SHARED_DIR / "hydrorivers"
SHARED_HYDROFABRICS_DIR = SHARED_DIR / "hydrofabrics"
SHARED_GRIDDED_ARCHIVES_DIR = SHARED_DIR / "gridded_archives"
SHARED_WEATHER_TILES_CACHE_DIR = SHARED_DIR / "weather_tiles_cache"
SHARED_MODELS_DIR = SHARED_DIR / "models"
SHARED_MAAS_CACHE_DIR = SHARED_DIR / "maas_cache"

# Isolated Per-User Workspaces
USERS_DIR = DATA_DIR / "users"
PROFILES_DIR = DATA_DIR / "profiles"

# Organized User & Profile Storage Directories (Legacy Backwards-Compatibility)
POLYGONS_DIR = DATA_DIR / "polygons"
ATTRIBUTES_DIR = DATA_DIR / "attributes"
HISTORICAL_DIR = DATA_DIR / "historical"
FORECAST_DIR = DATA_DIR / "forecast"
REALTIME_DIR = FORECAST_DIR  # Maintained for backwards-compatibility
ARCHIVES_DIR = DATA_DIR / "archives"  # Maintained for backwards-compatibility

# Raw Project Baseline & Reference Data Stores
BASE_LAYERS_DIR = DATA_DIR / "base_layers"
DEM_DIR = BASE_LAYERS_DIR / "dem"
RIVER_NETWORKS_DIR = BASE_LAYERS_DIR / "river_networks"
HYDRO_BASINS_DIR = BASE_LAYERS_DIR / "hydro_basins"
CLIMATOLOGY_DIR = BASE_LAYERS_DIR / "climatology"

# External Permanent Storage & Cache Locations
HOME_DIR = Path.home()
PERM_DATA_DIR = HOME_DIR / "data"
PERM_POLYGONS_DIR = PERM_DATA_DIR / "input" / "polygons"
CACHE_DIR = HOME_DIR / ".cache" / "earthkit_hydro" / "data"

STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist safely
for p in (
    DATA_DIR,
    SHARED_DIR,
    SHARED_DEMS_DIR,
    SHARED_HYDROATLAS_DIR,
    SHARED_HYDRORIVERS_DIR,
    SHARED_HYDROFABRICS_DIR,
    SHARED_GRIDDED_ARCHIVES_DIR,
    SHARED_WEATHER_TILES_CACHE_DIR,
    SHARED_MODELS_DIR,
    SHARED_MAAS_CACHE_DIR,
    USERS_DIR,
    PROFILES_DIR,
    POLYGONS_DIR,
    ATTRIBUTES_DIR,
    HISTORICAL_DIR,
    FORECAST_DIR,
    ARCHIVES_DIR,
    BASE_LAYERS_DIR,
    DEM_DIR,
    RIVER_NETWORKS_DIR,
    HYDRO_BASINS_DIR,
    CLIMATOLOGY_DIR,
    STATIC_DIR,
):
  try:
    p.mkdir(parents=True, exist_ok=True)
  except (PermissionError, OSError):
    pass


def ensure_flood_forecasting_on_sys_path(
    repo_dir: Optional[Path] = None,
) -> bool:
  """Ensures FLOOD_FORECASTING_REPO_DIR (and optional src/ dir) is on sys.path."""
  target_repo = (
      Path(repo_dir)
      if repo_dir is not None
      else Path(FLOOD_FORECASTING_REPO_DIR)
  )
  if not target_repo.exists():
    return False
  src_dir = target_repo / "src"
  if src_dir.is_dir():
    src_str = str(src_dir)
    if src_str not in sys.path:
      sys.path.insert(0, src_str)
  repo_str = str(target_repo)
  if repo_str not in sys.path:
    sys.path.insert(0, repo_str)
  return True


def extend_multimet_package_path(repo_dir: Optional[Path] = None) -> bool:
  """Ensures the unified `multimet` package and any sibling checkouts are on `multimet.__path__`."""
  target_repo = (
      Path(repo_dir)
      if repo_dir is not None
      else Path(FLOOD_FORECASTING_REPO_DIR)
  )
  ensure_flood_forecasting_on_sys_path(repo_dir=target_repo)
  candidates = [
      target_repo / "multimet",
      target_repo / "src" / "multimet",
      target_repo.parent / "flood-forecasting-multimet" / "multimet",
      target_repo.parent / "flood-forecasting-multimet" / "src" / "multimet",
      target_repo.parent / "flood-forecasting-static-extractor" / "multimet",
      target_repo.parent
      / "flood-forecasting-static-extractor"
      / "src"
      / "multimet",
  ]
  existing_candidates = [c for c in candidates if c.is_dir()]
  if not existing_candidates:
    return False

  for c in existing_candidates:
    parent_str = str(c.parent)
    if parent_str not in sys.path:
      sys.path.insert(0, parent_str)

  try:
    import multimet  # pylint: disable=g-import-not-at-top
  except ImportError:
    return False

  for c in existing_candidates:
    c_str = str(c)
    if c_str not in multimet.__path__:
      multimet.__path__.append(c_str)
  return True


DEFAULT_HYDRO_DATASET = "hydroatlas"

# Supported Hydrography Datasets (1:1 DEM <-> River Network Pairing)
HYDRO_DATASETS: Dict[str, Dict[str, Any]] = {
    "hydroatlas": {
        "id": "hydroatlas",
        "name": "HydroATLAS / HydroSHEDS",
        "resolution": "15 arc-second (~500m) / 30 arc-second",
        "dem_id": "hydrosheds_90m",
        "dem_name": "HydroSHEDS 90m Conditioned DEM (3 arc-sec)",
        "dem_resolution": "3 arc-second (~90m)",
        "dem_tiles_dir": str(DEM_DIR / "hydrosheds_90m" / "tiles_5deg"),
        "river_network_id": "hydroatlas",
        "river_network_name": "HydroRIVERS / HydroATLAS River Network",
        "description": "Global comprehensive hydro-environmental database linking river networks, sub-basins, and hydro-ecological attributes.",
        "default_snap_radius_km": 5.0,
        "citation": "Linke et al. (2019), Scientific Data",
    },
    "merit-hydro": {
        "id": "merit-hydro",
        "name": "MERIT Hydro",
        "resolution": "3 arc-second (~90m)",
        "dem_id": "merit_hydro_90m",
        "dem_name": "MERIT-Hydro 90m DEM (3 arc-sec)",
        "dem_resolution": "3 arc-second (~90m)",
        "dem_tiles_dir": str(DEM_DIR / "merit_hydro_90m" / "tiles_5deg"),
        "river_network_id": "merit-hydro",
        "river_network_name": "MERIT-Basins River Network",
        "description": "Multi-Error-Removed Improved-Terrain Hydrography dataset with high-precision flow direction and river networks.",
        "default_snap_radius_km": 2.0,
        "citation": "Yamazaki et al. (2019), Water Resources Research",
    },
}

HYDRO_DATASET_ALIASES: Dict[str, str] = {
    "hydroatlas": "hydroatlas",
    "hydrosheds": "hydroatlas",
    "hydrosheds_90m": "hydroatlas",
    "hydrorivers": "hydroatlas",
    "merit-hydro": "merit-hydro",
    "merit_hydro": "merit-hydro",
    "merit_hydro_90m": "merit-hydro",
    "merit-basins": "merit-hydro",
    "merit_basins": "merit-hydro",
}


def resolve_hydro_dataset_id(dataset_or_dem_id: Optional[str]) -> str:
  """Resolves a dataset or DEM alias (e.g. 'hydrosheds_90m', 'merit_hydro_90m') to a canonical HYDRO_DATASETS key."""
  if not dataset_or_dem_id:
    return DEFAULT_HYDRO_DATASET
  key = str(dataset_or_dem_id).strip().lower()
  if key in HYDRO_DATASETS:
    return key
  return HYDRO_DATASET_ALIASES.get(key, key)


# Weather Data Settings
WEATHER_CONFIG = {
    "historical": {
        "source": "cds",  # or 'arco-era5', 'mars'
        "dataset_name": "reanalysis-era5-single-levels",
        "default_variables": ["total_precipitation", "2m_temperature"],
        "default_chunking": {"time": 744, "lat": 20, "lon": 20},
    },
    "forecast": {
        "source": "ecmwf-open-data",
        "model": "ifs",  # 'ifs' (deterministic), 'aifs', 'aifs-ens'
        "stream": "oper",
        "default_variables": ["tp", "2t"],
        "forecast_horizon_hours": 240,  # 10 days
    },
}
