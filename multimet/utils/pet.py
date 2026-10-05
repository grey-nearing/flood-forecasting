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

"""FAO-56 Penman-Monteith potential evapotranspiration (PET) computation.

Re-exports shared PET implementations from `multimet.utils.climate`.
"""

from __future__ import annotations

from multimet.utils.climate import (
    calculate_fao56_penman_monteith_pet,
    calculate_fao_pm_pet,
)

__all__ = [
    "calculate_fao56_penman_monteith_pet",
    "calculate_fao_pm_pet",
]
