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

"""Open-MultiMet catchment timeseries extractors.

Reduces gridded meteorology to per-basin daily timeseries in the Caravan
MultiMet schema. Every extractor reads either from a gridded Zarr archive
(:mod:`multimet.timeseries_extractors.gridded_archive`) or from an upstream
provider / Icechunk catalog (:mod:`multimet.timeseries_extractors.dynamical`).
"""

from __future__ import annotations

import importlib
from typing import Any

# Submodules reachable as ``multimet.timeseries_extractors.<name>``.
_SUBMODULES = frozenset({
    "base",
    "config",
    "cpc",
    "dask_runner",
    "dynamical",
    "era5_land",
    "gridded_archive",
    "hres",
    "imerg",
    "realtime",
    "runner",
    "zarr_writer",
})

# Public symbols re-exported from their defining submodule.
_LAZY_SYMBOLS = {
    "Product": "multimet.timeseries_extractors.config",
    "ProductType": "multimet.timeseries_extractors.config",
    "BaseExtractor": "multimet.timeseries_extractors.base",
    "CPCExtractor": "multimet.timeseries_extractors.cpc",
    "ERA5LandExtractor": "multimet.timeseries_extractors.era5_land",
    "HRESExtractor": "multimet.timeseries_extractors.hres",
    "find_latest_hres_open_data_date": "multimet.timeseries_extractors.hres",
    "IMERGExtractor": "multimet.timeseries_extractors.imerg",
    "get_bounding_box": "multimet.utils.geometry",
    "load_basin_geometries": "multimet.utils.geometry",
    "MultiMetZarrWriter": "multimet.timeseries_extractors.zarr_writer",
    "ZonalWeightCalculator": "multimet.utils.zonal",
    "ZonalWeightMatrix": "multimet.utils.zonal",
    "calculate_fao56_penman_monteith_pet": "multimet.utils.climate",
    "extract_multimet_serial": "multimet.timeseries_extractors.runner",
    "extract_multimet_dask": "multimet.timeseries_extractors.dask_runner",
    "extract_product_dask": "multimet.timeseries_extractors.dask_runner",
    "RealtimeFetchResult": "multimet.timeseries_extractors.realtime",
    "RealtimeForcingFetcher": "multimet.timeseries_extractors.realtime",
    "fetch_realtime_multimet": "multimet.timeseries_extractors.realtime",
    "inspect_store_last_valid_date": "multimet.timeseries_extractors.realtime",
    "read_hot_start_state_date": "multimet.timeseries_extractors.realtime",
    "DynamicalDataLoader": "multimet.timeseries_extractors.dynamical",
    "DynamicalExtractor": "multimet.timeseries_extractors.dynamical",
    "DynamicalIMERGExtractor": "multimet.timeseries_extractors.dynamical",
    "AIFSExtractor": "multimet.timeseries_extractors.dynamical",
    "load_dynamical": "multimet.timeseries_extractors.dynamical",
    "BoundingBox": "multimet.utils.spatial",
    "find_lat_lon_dims": "multimet.utils.spatial",
    "slice_coordinates_by_bounds": "multimet.utils.spatial",
    "slice_dataset_by_bounds": "multimet.utils.spatial",
    # Gridded archive source.
    "ArchiveBandSpec": "multimet.timeseries_extractors.gridded_archive",
    "GriddedArchiveSpec": "multimet.timeseries_extractors.gridded_archive",
    "GriddedArchiveError": "multimet.timeseries_extractors.gridded_archive",
    "extract_from_archive": "multimet.timeseries_extractors.gridded_archive",
    "extract_forecast_from_archive": (
        "multimet.timeseries_extractors.gridded_archive"
    ),
    "extract_nowcast_from_archive": (
        "multimet.timeseries_extractors.gridded_archive"
    ),
    "get_archive_spec": "multimet.timeseries_extractors.gridded_archive",
    "open_gridded_archive": "multimet.timeseries_extractors.gridded_archive",
}

__all__ = sorted(_SUBMODULES | set(_LAZY_SYMBOLS))


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module objects are untyped.
  if name in _SUBMODULES:
    return importlib.import_module(f"{__name__}.{name}")
  if name in _LAZY_SYMBOLS:
    module = importlib.import_module(_LAZY_SYMBOLS[name])
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return sorted(set(globals()) | set(__all__))
