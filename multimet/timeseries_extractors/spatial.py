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

"""Re-exports spatial bounding box utilities from multimet.utils.spatial."""

from multimet.utils.spatial import (
    BoundingBox,
    coerce_bounding_box,
    find_lat_lon_dims,
    slice_coordinates_by_bounds,
    slice_dataset_by_bounds,
)

__all__ = [
    "BoundingBox",
    "coerce_bounding_box",
    "find_lat_lon_dims",
    "slice_coordinates_by_bounds",
    "slice_dataset_by_bounds",
]
