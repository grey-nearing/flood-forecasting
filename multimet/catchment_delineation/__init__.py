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

"""Catchment Delineation Package.

Scientific watershed delineation and hydrography package supporting:
- Multi-DEM 90m D8 flow-direction rasters (HydroSHEDS 90m and MERIT-Hydro 90m
  up to 90°N) with seamless cross-tile boundary routing.
- Vector river network querying and pour-point snapping (HydroRIVERS v1.0 and
  MERIT-Basins).
- Official unit-catchment ridgeline and exact pour-point delineation
  (HydroBASINS Level 12 and MERIT-Basins).
- Hybrid vector-guided + 90m D8 raster delineation.
- 3-arc-second and global overview elevation sampling.
"""

from multimet.catchment_delineation.config import (
    DEM_MAX_LAT,
    DEM_MAX_LON,
    DEM_MIN_LAT,
    DEM_MIN_LON,
    INFLOW_MAP,
    RES_DEG,
    TILE_CELLS,
    TILE_DEG,
)
from multimet.catchment_delineation.datasets import (
    DEM_ALIASES,
    DEM_DATASETS,
    HYDROSHEDS_90M,
    MERIT_HYDRO_90M,
    DemDataset,
    resolve_dem_dataset,
)
from multimet.catchment_delineation.delineator import (
    CatchmentAreaMismatchError,
    CatchmentCoverageError,
    DemDelineator,
    delineate_catchment,
    delineate_coordinates,
    delineate_dem,
)
from multimet.catchment_delineation.elevation import (
    ElevationTiles,
    GlobalElevationGrid,
)
from multimet.catchment_delineation.gcs import (
    download_tile_from_gcs,
    download_tiles_for_bbox,
)
from multimet.catchment_delineation.hybrid import delineate_hybrid
from multimet.catchment_delineation.hydrography import (
    HydroBasinsLayer,
    MeritBasinsLayer,
    ReachSnap,
    RiverNetwork,
    RiverSnapError,
    UnitCatchmentLayer,
)
from multimet.catchment_delineation.merit import (
    MERIT_HYDRO_EE_ASSET,
    download_merit_d8_tile,
    fetch_merit_d8_half_tile,
)
from multimet.catchment_delineation.tiles import (
    filename_to_tile_key,
    is_coord_in_coverage,
    is_tile_available,
    is_tile_in_coverage,
    latlon_to_tile_key,
    list_available_tiles,
    tile_key_to_filename,
)
from multimet.catchment_delineation.vector_delineator import (
    UnitCatchmentDelineator,
    VectorCatchment,
    clip_unit_catchment_to_pour_point,
)

__all__ = [
    'DEM_ALIASES',
    'DEM_DATASETS',
    'DEM_MAX_LAT',
    'DEM_MAX_LON',
    'DEM_MIN_LAT',
    'DEM_MIN_LON',
    'HYDROSHEDS_90M',
    'INFLOW_MAP',
    'MERIT_HYDRO_90M',
    'MERIT_HYDRO_EE_ASSET',
    'RES_DEG',
    'TILE_CELLS',
    'TILE_DEG',
    'CatchmentAreaMismatchError',
    'CatchmentCoverageError',
    'DemDataset',
    'DemDelineator',
    'ElevationTiles',
    'GlobalElevationGrid',
    'HydroBasinsLayer',
    'MeritBasinsLayer',
    'ReachSnap',
    'RiverNetwork',
    'RiverSnapError',
    'UnitCatchmentDelineator',
    'UnitCatchmentLayer',
    'VectorCatchment',
    'clip_unit_catchment_to_pour_point',
    'delineate_catchment',
    'delineate_coordinates',
    'delineate_dem',
    'delineate_hybrid',
    'download_merit_d8_tile',
    'download_tile_from_gcs',
    'download_tiles_for_bbox',
    'fetch_merit_d8_half_tile',
    'filename_to_tile_key',
    'is_coord_in_coverage',
    'is_tile_available',
    'is_tile_in_coverage',
    'latlon_to_tile_key',
    'list_available_tiles',
    'resolve_dem_dataset',
    'tile_key_to_filename',
]
