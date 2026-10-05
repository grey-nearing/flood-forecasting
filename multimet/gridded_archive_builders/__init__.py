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

"""Open-MultiMet gridded precipitation archive builders.

Contains the ETL pipelines that assemble the unified, analysis-ready daily
gridded precipitation archives that back Open-MultiMet:

* :mod:`multimet.gridded_archive_builders.build_cpc_archive` - NOAA CPC Global
  Unified daily gauge-based precipitation (0.5 degree, 1979 to present).
* :mod:`multimet.gridded_archive_builders.build_imerg_archive` - NASA GPM IMERG
  Early V07 daily precipitation (0.1 degree, 2000 to present).
* :mod:`multimet.gridded_archive_builders.storage` - Zarr target resolution,
  batch write, and strict resume primitives.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "build_cpc_archive",
    "build_imerg_archive",
    "storage",
]

_SUBMODULES = frozenset(__all__)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module objects are untyped.
  if name in _SUBMODULES:
    return importlib.import_module(f"{__name__}.{name}")
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return sorted(set(globals()) | _SUBMODULES)
