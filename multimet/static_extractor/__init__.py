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

"""Caravan Static Attributes Extractor.

Computes Caravan-compatible static watershed attributes from HydroATLAS Level 12
and ERA5-Land datasets for user-supplied watershed polygons.

The public surface is deliberately a pure Python library: typed results
(:class:`CatchmentAttributes`), plain geospatial queries
(:meth:`StaticAttributesExtractor.read_subbasins`), an explicit attribute
registry (:data:`ATTRIBUTE_REGISTRY`) and unit conversion
(:func:`to_physical_units`). Presentation concerns (JSON sanitising, map
rendering, UI grouping, application cache directories) live in the consuming
application, not here.
"""

from multimet.static_extractor.climate import (
    CARAVAN_CLIMATE_ALIASES,
    ERA5ClimateLoader,
    ERA5GriddedExtractor,
    calculate_fao_pm_pet,
    calculate_knoben_moisture_and_seasonality,
    compute_caravan_climate_metrics,
    expand_caravan_climate_aliases,
)
from multimet.static_extractor.config import (
    ADDITIONAL_PROPERTIES,
    CARAVAN_CLIMATE_COLUMNS,
    CARAVAN_CLIMATE_INDICES,
    CONTINENT_BBOXES,
    CONTINENT_MAP,
    DEFAULT_GCS_ERA5_CLIMATE_URI,
    DEFAULT_GCS_GRIDDED_ERA5_URI,
    DEFAULT_GCS_HYDROATLAS_URI,
    IGNORE_PROPERTIES,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
    UPSTREAM_PROPERTIES,
)
from multimet.static_extractor.extractor import (
    CatchmentAttributes,
    StaticAttributesExtractor,
    UnsupportedHydroATLASLevelError,
    compute_pour_point_properties,
)
from multimet.static_extractor.schema import (
    ATTRIBUTE_CATEGORIES,
    ATTRIBUTE_DEFINITIONS,
    ATTRIBUTE_REGISTRY,
    AttributeDefinition,
    UnknownAttributeWarning,
    get_attribute_definition,
    to_physical_units,
    unknown_attribute_keys,
)
from multimet.utils.gcs import (
    download_hydroatlas_from_gcs,
)

__all__ = [
    "StaticAttributesExtractor",
    "CatchmentAttributes",
    "UnsupportedHydroATLASLevelError",
    "discover_datasets",
    "run_batch_extraction",
    "ERA5ClimateLoader",
    "ERA5GriddedExtractor",
    "compute_caravan_climate_metrics",
    "calculate_fao_pm_pet",
    "calculate_knoben_moisture_and_seasonality",
    "compute_pour_point_properties",
    "download_hydroatlas_from_gcs",
    "expand_caravan_climate_aliases",
    "AttributeDefinition",
    "UnknownAttributeWarning",
    "get_attribute_definition",
    "to_physical_units",
    "unknown_attribute_keys",
    "ATTRIBUTE_CATEGORIES",
    "ATTRIBUTE_DEFINITIONS",
    "ATTRIBUTE_REGISTRY",
    "CARAVAN_CLIMATE_ALIASES",
    "CARAVAN_CLIMATE_COLUMNS",
    "CARAVAN_CLIMATE_INDICES",
    "DEFAULT_GCS_HYDROATLAS_URI",
    "DEFAULT_GCS_ERA5_CLIMATE_URI",
    "DEFAULT_GCS_GRIDDED_ERA5_URI",
    "MAJORITY_PROPERTIES",
    "POUR_POINT_PROPERTIES",
    "IGNORE_PROPERTIES",
    "ADDITIONAL_PROPERTIES",
    "UPSTREAM_PROPERTIES",
    "CONTINENT_MAP",
    "CONTINENT_BBOXES",
]


def __getattr__(name: str):
  if name in ("discover_datasets", "run_batch_extraction"):
    from multimet.static_extractor import batch_runner
    return getattr(batch_runner, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
