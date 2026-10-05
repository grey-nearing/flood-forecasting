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

"""Re-exports zonal weight calculators and sparse matrix operators from multimet.utils.zonal."""

from multimet.utils.zonal import (
    MIN_VALID_COVERAGE_FRACTION,
    ZonalWeightCalculator,
    ZonalWeightMatrix,
    _compute_basin_weights_worker,
    weighted_mean_valid,
    weighted_mean_valid_with_coverage,
)

__all__ = [
    "MIN_VALID_COVERAGE_FRACTION",
    "ZonalWeightCalculator",
    "ZonalWeightMatrix",
    "_compute_basin_weights_worker",
    "weighted_mean_valid",
    "weighted_mean_valid_with_coverage",
]
