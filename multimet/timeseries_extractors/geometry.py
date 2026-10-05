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

"""Catchment geometry loader and normalizer (re-exported from multimet.utils.geometry)."""

from multimet.utils.geometry import (  # noqa: F401
    CANDIDATE_ID_COLUMNS,
    SUPPORTED_GEOMETRY_EXTENSIONS,
    _discover_geometry_files,
    _load_single_basin_geometry,
    _resolve_geometry_sources,
    find_all_dataset_dirs,
    find_vector_file_in_dir,
    get_bounding_box,
    load_basin_geometries,
)
