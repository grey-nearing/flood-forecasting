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

"""MultiMet data workflows for static watershed attributes and gridded archives.

Exposes:

* Static attribute extraction (:class:`StaticAttributesExtractor`,
  :class:`ERA5ClimateLoader`, :class:`ERA5GriddedExtractor`,
  :func:`compute_caravan_climate_metrics`).
* Gridded meteorological archive builders
  (:mod:`multimet.gridded_archive_builders`).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "StaticAttributesExtractor",
    "ERA5ClimateLoader",
    "ERA5GriddedExtractor",
    "compute_caravan_climate_metrics",
    "gridded_archive_builders",
]

_STATIC_EXPORTS = frozenset({
    "StaticAttributesExtractor",
    "ERA5ClimateLoader",
    "ERA5GriddedExtractor",
    "compute_caravan_climate_metrics",
})

_SUBMODULES = frozenset({
    "gridded_archive_builders",
})


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module objects are untyped.
  if name in _SUBMODULES:
    return importlib.import_module(f"{__name__}.{name}")
  if name in _STATIC_EXPORTS:
    mod = importlib.import_module(f"{__name__}.static_extractor")
    return getattr(mod, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return sorted(set(globals()) | _SUBMODULES | _STATIC_EXPORTS)
