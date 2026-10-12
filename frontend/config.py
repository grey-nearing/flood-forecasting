# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration module for the OpenHydroNet Interactive Web Platform."""

import os
import sys
from pathlib import Path
from typing import Any

import pyproj.datadir

from multimet.catchment_delineation.datasets import resolve_dem_dataset
from utils.file_paths import (
    FLOOD_FORECASTING_REPO_DIR_ENV,
    HYDRORIVERS_SHP_NAME,
    OPENHYDRONET_CACHE_ROOT,
    OPENHYDRONET_DATA_CACHE_DIR,
    OPENHYDRONET_DATA_DIR_ENV,
    OPENHYDRONET_HYDROBASINS_CACHE_DIR,
    OPENHYDRONET_HYDRORIVERS_SHP_CACHE_PATH,
    OPENHYDRONET_HYDROSHEDS_DEM_CACHE_DIR,
    OPENHYDRONET_HYDROSHEDS_DIR_CACHE_DIR,
    OPENHYDRONET_MERIT_BASINS_CACHE_DIR,
    OPENHYDRONET_MERIT_DEM_CACHE_DIR,
    OPENHYDRONET_MERIT_DIR_CACHE_DIR,
    OPENHYDRONET_PROFILES_CACHE_DIR,
    OPENHYDRONET_PROFILES_DIR_ENV,
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Repository root for OpenHydroNet (flood-forecasting)
FLOOD_FORECASTING_REPO_DIR = Path(
    os.environ.get(
        FLOOD_FORECASTING_REPO_DIR_ENV,
        str(BASE_DIR.parent),
    )
)

if os.environ.get(OPENHYDRONET_DATA_DIR_ENV):
    DATA_DIR = Path(os.environ[OPENHYDRONET_DATA_DIR_ENV])
elif os.environ.get('TEST_TMPDIR'):
    DATA_DIR = Path(os.environ['TEST_TMPDIR']) / 'data'
else:
    DATA_DIR = BASE_DIR / 'data'

# Shared Read-Only (or Admin-Synced) Global Reference Datasets
SHARED_DIR = DATA_DIR / 'shared'
SHARED_DEMS_DIR = SHARED_DIR / 'dems'
SHARED_HYDROATLAS_DIR = SHARED_DIR / 'hydroatlas'
SHARED_HYDRORIVERS_DIR = SHARED_DIR / 'hydrorivers'
SHARED_HYDROFABRICS_DIR = SHARED_DIR / 'hydrofabrics'
SHARED_GRIDDED_ARCHIVES_DIR = SHARED_DIR / 'gridded_archives'
SHARED_WEATHER_TILES_CACHE_DIR = SHARED_DIR / 'weather_tiles_cache'
SHARED_MODELS_DIR = SHARED_DIR / 'models'
SHARED_MAAS_CACHE_DIR = SHARED_DIR / 'maas_cache'

# Isolated Per-User Workspaces
USERS_DIR = DATA_DIR / 'users'
PROFILES_DIR = DATA_DIR / 'profiles'

# Organized User & Profile Storage Directories (Legacy Backwards-Compatibility)
POLYGONS_DIR = DATA_DIR / 'polygons'
ATTRIBUTES_DIR = DATA_DIR / 'attributes'
HISTORICAL_DIR = DATA_DIR / 'historical'
FORECAST_DIR = DATA_DIR / 'forecast'
REALTIME_DIR = FORECAST_DIR
ARCHIVES_DIR = DATA_DIR / 'archives'

# Raw Project Baseline & Reference Data Stores
BASE_LAYERS_DIR = DATA_DIR / 'base_layers'
DEM_DIR = BASE_LAYERS_DIR / 'dem'
RIVER_NETWORKS_DIR = BASE_LAYERS_DIR / 'river_networks'
HYDRO_BASINS_DIR = BASE_LAYERS_DIR / 'hydro_basins'
CLIMATOLOGY_DIR = BASE_LAYERS_DIR / 'climatology'

# Centralized Cache & Static Assets Locations
CACHE_DIR = OPENHYDRONET_DATA_CACHE_DIR
STATIC_DIR = BASE_DIR / 'static'

DATA_DIRECTORIES: tuple[Path, ...] = (
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
)

_proj_dir = pyproj.datadir.get_data_dir()
if _proj_dir and Path(_proj_dir).is_dir():
    os.environ.setdefault('PROJ_DATA', _proj_dir)
    os.environ.setdefault('PROJ_LIB', _proj_dir)


def ensure_data_dirs() -> None:
    """Create all local workspace and reference data directories if absent."""
    for directory in DATA_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_flood_forecasting_on_sys_path(
    repo_dir: Path | None = None,
) -> bool:
    """Ensure `FLOOD_FORECASTING_REPO_DIR` (and `src/`) is on `sys.path`.

    Args:
        repo_dir: Optional repository root directory override.

    Returns:
        `True` if the repository directory exists and is present on `sys.path`,
        `False` if the target directory does not exist.
    """
    target_repo = (
        Path(repo_dir)
        if repo_dir is not None
        else Path(FLOOD_FORECASTING_REPO_DIR)
    )
    if not target_repo.exists():
        return False
    src_dir = target_repo / 'src'
    if src_dir.is_dir():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    repo_str = str(target_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return True


def extend_multimet_package_path(repo_dir: Path | None = None) -> bool:
    """Ensure the repository root containing `multimet` is on `sys.path`.

    Args:
        repo_dir: Optional repository root directory override.

    Returns:
        `True` if the repository root exists and contains a `multimet` package
        directory, `False` otherwise.
    """
    target_repo = (
        Path(repo_dir)
        if repo_dir is not None
        else Path(FLOOD_FORECASTING_REPO_DIR)
    )
    if not ensure_flood_forecasting_on_sys_path(repo_dir=target_repo):
        return False
    candidates = (
        target_repo / 'multimet',
        target_repo / 'src' / 'multimet',
    )
    return any(candidate.is_dir() for candidate in candidates)


DEFAULT_HYDRO_DATASET = 'hydroatlas'

# Supported Hydrography Datasets (1:1 DEM <-> River Network Pairing)
HYDRO_DATASETS: dict[str, dict[str, Any]] = {
    'hydroatlas': {
        'id': 'hydroatlas',
        'name': 'HydroATLAS / HydroSHEDS',
        'resolution': '15 arc-second (~500m) / 30 arc-second',
        'dem_id': 'hydrosheds_90m',
        'dem_name': 'HydroSHEDS 90m Conditioned DEM (3 arc-sec)',
        'dem_resolution': '3 arc-second (~90m)',
        'dem_tiles_dir': str(DEM_DIR / 'hydrosheds_90m' / 'tiles_5deg'),
        'river_network_id': 'hydroatlas',
        'river_network_name': 'HydroRIVERS',
        'description': (
            'Global comprehensive hydro-environmental database linking river '
            'networks, sub-basins, and hydro-ecological attributes.'
        ),
        'default_snap_radius_km': 5.0,
        'citation': 'Linke et al. (2019), Scientific Data',
    },
    'merit-hydro': {
        'id': 'merit-hydro',
        'name': 'MERIT Hydro',
        'resolution': '3 arc-second (~90m)',
        'dem_id': 'merit_hydro_90m',
        'dem_name': 'MERIT-Hydro 90m DEM (3 arc-sec)',
        'dem_resolution': '3 arc-second (~90m)',
        'dem_tiles_dir': str(DEM_DIR / 'merit_hydro_90m' / 'tiles_5deg'),
        'river_network_id': 'merit-hydro',
        'river_network_name': 'MERIT-Basins',
        'description': (
            'Multi-Error-Removed Improved-Terrain Hydrography dataset with '
            'high-precision flow direction and river networks.'
        ),
        'default_snap_radius_km': 2.0,
        'citation': 'Yamazaki et al. (2019), Water Resources Research',
    },
}

HYDRO_DATASET_ALIASES: dict[str, str] = {
    'hydroatlas': 'hydroatlas',
    'hydrosheds': 'hydroatlas',
    'hydrosheds_90m': 'hydroatlas',
    'hydrorivers': 'hydroatlas',
    'merit-hydro': 'merit-hydro',
    'merit_hydro': 'merit-hydro',
    'merit_hydro_90m': 'merit-hydro',
    'merit-basins': 'merit-hydro',
    'merit_basins': 'merit-hydro',
}


def resolve_hydro_dataset_id(dataset_or_dem_id: str | None) -> str:
    """Resolve a dataset or DEM alias to a canonical `HYDRO_DATASETS` key.

    Args:
        dataset_or_dem_id: Dataset or DEM identifier string (for example,
            `'hydrosheds_90m'` or `'merit_hydro_90m'`).

    Returns:
        Canonical dataset key (`'hydroatlas'` or `'merit-hydro'`).
    """
    if not dataset_or_dem_id:
        return DEFAULT_HYDRO_DATASET
    key = str(dataset_or_dem_id).strip().lower()
    if key in HYDRO_DATASETS:
        return key
    return HYDRO_DATASET_ALIASES.get(key, key)


DEFAULT_DEM_ID = 'hydrosheds_90m'

# Candidate local and cache directories for 5x5 degree DEM flow-direction tiles
DEM_TILE_LOCATIONS: dict[str, tuple[Path, ...]] = {
    'hydrosheds_90m': (
        DEM_DIR / 'hydrosheds_90m' / 'tiles_5deg',
        OPENHYDRONET_HYDROSHEDS_DEM_CACHE_DIR,
        OPENHYDRONET_HYDROSHEDS_DIR_CACHE_DIR,
    ),
    'merit_hydro_90m': (
        DEM_DIR / 'merit_hydro_90m' / 'tiles_5deg',
        OPENHYDRONET_MERIT_DEM_CACHE_DIR,
        OPENHYDRONET_MERIT_DIR_CACHE_DIR,
    ),
}


def resolve_dem_tiles_dir(dem_id: str = DEFAULT_DEM_ID) -> Path:
    """Resolve the local directory containing 5x5 degree D8 `.npy` tiles.

    Args:
        dem_id: DEM dataset identifier (defaults to `'hydrosheds_90m'`).

    Returns:
        `Path` to the resolved 5x5 degree D8 tile directory.
    """
    extend_multimet_package_path()
    dataset = resolve_dem_dataset(dem_id)
    candidates = DEM_TILE_LOCATIONS.get(dataset.id, ())
    for path in candidates:
        if path.exists() and any(path.glob('*.npy')):
            return path
    for path in candidates:
        if path.exists():
            return path
    return (
        candidates[0] if candidates else (DEM_DIR / dataset.id / 'tiles_5deg')
    )


def resolve_hydrography_paths() -> dict[str, Path]:
    """Resolve local or cache paths for HydroRIVERS and MERIT-Basins.

    Returns:
        Mapping of hydrography dataset keys to resolved local `Path` objects.
    """
    hydrorivers_shp = RIVER_NETWORKS_DIR / 'hydrorivers' / HYDRORIVERS_SHP_NAME
    if not hydrorivers_shp.is_file():
        hydrorivers_shp = OPENHYDRONET_HYDRORIVERS_SHP_CACHE_PATH

    hydrobasins_dir = HYDRO_BASINS_DIR / 'hydrobasins_units'
    if not (hydrobasins_dir.is_dir() and any(hydrobasins_dir.glob('*.shp'))):
        hydrobasins_dir = OPENHYDRONET_HYDROBASINS_CACHE_DIR

    merit_dir = RIVER_NETWORKS_DIR / 'merit_rivers'
    if not (merit_dir.is_dir() and any(merit_dir.glob('*.shp'))):
        merit_dir = OPENHYDRONET_MERIT_BASINS_CACHE_DIR

    return {
        'hydrorivers_shp': hydrorivers_shp,
        'hydrobasins_dir': hydrobasins_dir,
        'merit_rivers_dir': merit_dir,
        'merit_catchments_dir': merit_dir,
    }


# Weather Data Settings
WEATHER_CONFIG: dict[str, dict[str, Any]] = {
    'historical': {
        'source': 'open-multimet',
        'default_variables': ['total_precipitation', '2m_temperature'],
        'default_chunking': {'time': 365, 'lat': 20, 'lon': 20},
    },
    'forecast': {
        'source': 'ecmwf-open-data',
        'model': 'ifs',
        'stream': 'oper',
        'default_variables': ['tp', '2t'],
        'forecast_horizon_hours': 240,
    },
}

__all__ = [
    'ARCHIVES_DIR',
    'ATTRIBUTES_DIR',
    'BASE_DIR',
    'BASE_LAYERS_DIR',
    'CACHE_DIR',
    'CLIMATOLOGY_DIR',
    'DATA_DIR',
    'DATA_DIRECTORIES',
    'DEFAULT_DEM_ID',
    'DEFAULT_HYDRO_DATASET',
    'DEM_DIR',
    'DEM_TILE_LOCATIONS',
    'FLOOD_FORECASTING_REPO_DIR',
    'FORECAST_DIR',
    'HISTORICAL_DIR',
    'HYDRO_BASINS_DIR',
    'HYDRO_DATASETS',
    'HYDRO_DATASET_ALIASES',
    'OPENHYDRONET_CACHE_ROOT',
    'OPENHYDRONET_PROFILES_CACHE_DIR',
    'OPENHYDRONET_PROFILES_DIR_ENV',
    'POLYGONS_DIR',
    'PROFILES_DIR',
    'REALTIME_DIR',
    'RIVER_NETWORKS_DIR',
    'SHARED_DEMS_DIR',
    'SHARED_DIR',
    'SHARED_GRIDDED_ARCHIVES_DIR',
    'SHARED_HYDROATLAS_DIR',
    'SHARED_HYDROFABRICS_DIR',
    'SHARED_HYDRORIVERS_DIR',
    'SHARED_MAAS_CACHE_DIR',
    'SHARED_MODELS_DIR',
    'SHARED_WEATHER_TILES_CACHE_DIR',
    'STATIC_DIR',
    'USERS_DIR',
    'WEATHER_CONFIG',
    'ensure_data_dirs',
    'ensure_flood_forecasting_on_sys_path',
    'extend_multimet_package_path',
    'resolve_dem_tiles_dir',
    'resolve_hydro_dataset_id',
    'resolve_hydrography_paths',
]
