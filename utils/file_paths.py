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
"""Centralized file names, GCS URIs, upstream URLs, and workspace paths."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository & Documentation URLs
# ---------------------------------------------------------------------------
GITHUB_REPO_URL = 'https://github.com/google-research/flood-forecasting'
GITHUB_MULTIMET_PACKAGE_URL = (
    'https://github.com/google-research/flood-forecasting/tree/main/multimet'
)
GITHUB_FLOODHUB_CONFIG_URL = (
    'https://github.com/google-research/flood-forecasting/blob/main/'
    'example-configs/floodhub-settings-config.yml'
)
DOCS_URL = 'https://openhydronet.readthedocs.io'

# ---------------------------------------------------------------------------
# Canonical Google Cloud Storage (GCS) Bucket & Dataset URIs
# ---------------------------------------------------------------------------
OPEN_MULTIMET_BUCKET_URI = 'gs://open-multimet'

DEFAULT_GCS_HYDROATLAS_URI = (
    f'{OPEN_MULTIMET_BUCKET_URI}/ancillary-data/hydroatlas'
)
DEFAULT_GCS_ERA5_CLIMATE_URI = (
    f'{OPEN_MULTIMET_BUCKET_URI}/ancillary-data/hydroatlas/era5_climate'
)
DEFAULT_GCS_GRIDDED_ERA5_URI = (
    f'{OPEN_MULTIMET_BUCKET_URI}/data/era5_land/daily_surface.zarr'
)
DEFAULT_GCS_DEM_TILES_URI = (
    f'{OPEN_MULTIMET_BUCKET_URI}/ancillary-data/dems/tiles_5deg'
)
HYDROSHEDS_TILES_5DEG_GCS_URI = DEFAULT_GCS_DEM_TILES_URI
DEFAULT_GCS_CPC_ARCHIVE_URI = (
    f'{OPEN_MULTIMET_BUCKET_URI}/gridded-data-archives/CPC/daily_surface.zarr'
)
DEFAULT_GCS_IMERG_ARCHIVE_URI = (
    f'{OPEN_MULTIMET_BUCKET_URI}/gridded-data-archives/IMERG/daily_surface.zarr'
)
DEFAULT_GCS_HRES_ARCHIVE_URI = (
    f'{OPEN_MULTIMET_BUCKET_URI}/gridded-data-archives/HRES/daily_surface.zarr'
)

ECMWF_OPEN_DATA_BUCKET = 'ecmwf-open-data'
WEATHERBENCH2_HRES_ZARR_URI = (
    'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
)

# ---------------------------------------------------------------------------
# Standard Zarr Store, Parquet, GeoParquet & GDB Filenames
# ---------------------------------------------------------------------------
ATTRIBUTES_ZARR_NAME = 'attributes.zarr'
STREAMFLOW_ZARR_NAME = 'streamflow.zarr'
STREAMFLOW_REALTIME_ZARR_NAME = 'streamflow_realtime.zarr'
TARGETS_ZARR_NAME = 'targets.zarr'
TIMESERIES_ZARR_NAME = 'timeseries.zarr'
SCALER_ZARR_NAME = 'scaler.zarr'
TEST_RESULTS_ZARR_NAME = 'test_results.zarr'
HISTORICAL_MASTER_ZARR_NAME = 'historical_training_master.zarr'
FORECAST_LATEST_ZARR_NAME = 'forecast_latest.zarr'
DAILY_SURFACE_ZARR_NAME = 'daily_surface.zarr'

BASIN_ATLAS_GDB_NAME = 'BasinATLAS_v10.gdb'
HYDRO_ATLAS_LEV12_PARQUET_NAME = 'hydro_atlas_lev12.parquet'
HYBAS_LEV12_GLOB_PATTERN = 'hybas_*_lev12_v1c.geoparquet'
HYBAS_LEV12_SUFFIX = '_lev12_v1c.geoparquet'
ATTRIBUTES_COMBINED_PARQUET_NAME = 'attributes_combined.parquet'
PER_BASIN_METRICS_PARQUET_NAME = 'per_basin_metrics.parquet'
BASIN_SHAPES_GEOPARQUET_SUFFIX = '_basin_shapes.geoparquet'

# ---------------------------------------------------------------------------
# Upstream External HTTP / API Endpoints
# ---------------------------------------------------------------------------
DEFAULT_CMR_GRANULES_URL = 'https://cmr.earthdata.nasa.gov/search/granules.json'
NASA_CMR_GRANULES_URL = DEFAULT_CMR_GRANULES_URL
DEFAULT_GESDISC_URL = (
    'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07'
)
NASA_GESDISC_IMERG_DAILY_URL = DEFAULT_GESDISC_URL
DEFAULT_GESDISC_SLASH_URL = f'{DEFAULT_GESDISC_URL}/'

DEFAULT_CPC_PSL_BASE_URL = (
    'https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/'
)
DEFAULT_CPC_PSL_URL_TEMPLATE = f'{DEFAULT_CPC_PSL_BASE_URL}precip.{{year}}.nc'
NOAA_PSL_CPC_URL_TEMPLATE = DEFAULT_CPC_PSL_URL_TEMPLATE
DEFAULT_CPC_LANDING_PAGE_URL = (
    'https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html'
)

DEFAULT_CHIRPS_NETCDF_URL = (
    'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/'
)
DEFAULT_CHIRPS_TIFS_URL = (
    'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/'
)
DEFAULT_CHIRPS_GEFS_V2_URL = (
    'https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v2/daily/global/'
)
DEFAULT_CHIRPS_GEFS_V3_URL = (
    'https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/daily/global/'
)
DEFAULT_ECMWF_FORECASTS_URL = 'https://data.ecmwf.int/forecasts/'

DYNAMICAL_STAC_CATALOG_URL = 'https://stac.dynamical.org/catalog.json'
MERIT_HYDRO_EE_ASSET = 'MERIT/Hydro/v1_0_1'
GEE_HIGHVOLUME_ASSETS_URL = (
    'https://earthengine-highvolume.googleapis.com/v1/'
    'projects/earthengine-public/assets/'
)
EE_MERIT_GET_PIXELS_URL = (
    f'{GEE_HIGHVOLUME_ASSETS_URL}{MERIT_HYDRO_EE_ASSET}:getPixels'
)

FLOODHUB_BASE_URL = 'https://floodforecasting.googleapis.com/v1'
GEOGLOWS_BASE_URL = 'https://geoglows.ecmwf.int/api/v2'
GEOGLOWS_CLOUDFRONT_URL = 'https://d14ritg1bypdp7.cloudfront.net'
GEOGLOWS_ARCGIS_LIVEFEEDS_URL = (
    'https://livefeeds3.arcgis.com/arcgis/rest/services/'
    'GEOGLOWS/GlobalWaterModel_Medium/MapServer/0/query'
)
GLOFAS_BASE_URL = 'https://flood-api.open-meteo.com/v1/flood'
GLOFAS_OWS_URL = 'https://globalfloods-ows.ecmwf.int/glofas-ows/ows.py'
JAXA_STAC_CATALOG_URL = 'https://data.earth.jaxa.jp/stac/cog/v1/catalog.json'
OPEN_METEO_ELEVATION_URL = 'https://api.open-meteo.com/v1/elevation'
RAINVIEWER_MAPS_URL = 'https://api.rainviewer.com/public/weather-maps.json'

# ---------------------------------------------------------------------------
# Local System & Interactive Frontend Workspace Paths (~/.cache/openhydronet)
# ---------------------------------------------------------------------------
DEFAULT_NETRC_PATH = '~/.netrc'
PROC_SELF_STATUS_PATH = '/proc/self/status'
DEFAULT_LOCAL_SERVER_URL = 'http://localhost:8080'

OPENHYDRONET_CACHE_ROOT = Path.home() / '.cache' / 'openhydronet'
OPENHYDRONET_DATA_CACHE_DIR = OPENHYDRONET_CACHE_ROOT / 'data'
OPENHYDRONET_WEATHER_CACHE_DIR = OPENHYDRONET_CACHE_ROOT / 'weather'
OPENHYDRONET_HYDROATLAS_CACHE_DIR = OPENHYDRONET_DATA_CACHE_DIR / 'hydroatlas'
OPENHYDRONET_ERA5_CLIMATE_CACHE_DIR = (
    OPENHYDRONET_DATA_CACHE_DIR / 'era5_climate'
)
FLOODHUB_API_KEY_FILE = OPENHYDRONET_CACHE_ROOT / 'floodhub_api_key'

__all__ = [
    'ATTRIBUTES_COMBINED_PARQUET_NAME',
    'ATTRIBUTES_ZARR_NAME',
    'BASIN_ATLAS_GDB_NAME',
    'BASIN_SHAPES_GEOPARQUET_SUFFIX',
    'DAILY_SURFACE_ZARR_NAME',
    'DEFAULT_CHIRPS_GEFS_V2_URL',
    'DEFAULT_CHIRPS_GEFS_V3_URL',
    'DEFAULT_CHIRPS_NETCDF_URL',
    'DEFAULT_CHIRPS_TIFS_URL',
    'DEFAULT_CMR_GRANULES_URL',
    'DEFAULT_CPC_LANDING_PAGE_URL',
    'DEFAULT_CPC_PSL_BASE_URL',
    'DEFAULT_CPC_PSL_URL_TEMPLATE',
    'DEFAULT_ECMWF_FORECASTS_URL',
    'DEFAULT_GCS_CPC_ARCHIVE_URI',
    'DEFAULT_GCS_DEM_TILES_URI',
    'DEFAULT_GCS_ERA5_CLIMATE_URI',
    'DEFAULT_GCS_GRIDDED_ERA5_URI',
    'DEFAULT_GCS_HRES_ARCHIVE_URI',
    'DEFAULT_GCS_HYDROATLAS_URI',
    'DEFAULT_GCS_IMERG_ARCHIVE_URI',
    'DEFAULT_GESDISC_SLASH_URL',
    'DEFAULT_GESDISC_URL',
    'DEFAULT_LOCAL_SERVER_URL',
    'DEFAULT_NETRC_PATH',
    'DOCS_URL',
    'DYNAMICAL_STAC_CATALOG_URL',
    'ECMWF_OPEN_DATA_BUCKET',
    'EE_MERIT_GET_PIXELS_URL',
    'FLOODHUB_API_KEY_FILE',
    'FLOODHUB_BASE_URL',
    'FORECAST_LATEST_ZARR_NAME',
    'GEE_HIGHVOLUME_ASSETS_URL',
    'GEOGLOWS_ARCGIS_LIVEFEEDS_URL',
    'GEOGLOWS_BASE_URL',
    'GEOGLOWS_CLOUDFRONT_URL',
    'GITHUB_FLOODHUB_CONFIG_URL',
    'GITHUB_MULTIMET_PACKAGE_URL',
    'GITHUB_REPO_URL',
    'GLOFAS_BASE_URL',
    'GLOFAS_OWS_URL',
    'HISTORICAL_MASTER_ZARR_NAME',
    'HYBAS_LEV12_GLOB_PATTERN',
    'HYBAS_LEV12_SUFFIX',
    'HYDROSHEDS_TILES_5DEG_GCS_URI',
    'HYDRO_ATLAS_LEV12_PARQUET_NAME',
    'JAXA_STAC_CATALOG_URL',
    'MERIT_HYDRO_EE_ASSET',
    'NASA_CMR_GRANULES_URL',
    'NASA_GESDISC_IMERG_DAILY_URL',
    'NOAA_PSL_CPC_URL_TEMPLATE',
    'OPENHYDRONET_CACHE_ROOT',
    'OPENHYDRONET_DATA_CACHE_DIR',
    'OPENHYDRONET_ERA5_CLIMATE_CACHE_DIR',
    'OPENHYDRONET_HYDROATLAS_CACHE_DIR',
    'OPENHYDRONET_WEATHER_CACHE_DIR',
    'OPEN_METEO_ELEVATION_URL',
    'OPEN_MULTIMET_BUCKET_URI',
    'PER_BASIN_METRICS_PARQUET_NAME',
    'PROC_SELF_STATUS_PATH',
    'RAINVIEWER_MAPS_URL',
    'SCALER_ZARR_NAME',
    'STREAMFLOW_REALTIME_ZARR_NAME',
    'STREAMFLOW_ZARR_NAME',
    'TARGETS_ZARR_NAME',
    'TEST_RESULTS_ZARR_NAME',
    'TIMESERIES_ZARR_NAME',
    'WEATHERBENCH2_HRES_ZARR_URI',
]

