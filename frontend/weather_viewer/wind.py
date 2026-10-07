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

"""Viewport wind-particle JSON formatter calling multimet.weather_fetcher."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from multimet.weather_fetcher.fetcher import (
    compute_wind_speed_and_direction,
    fetch_wind_grid,
)


def extract_wind_vectors(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str = "ecmwf_ifs",
    step_idx: int = 0,
    subsample: int = 2,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    bilinear: bool = False,
) -> Dict[str, Any]:
  """Formats subsampled 10m U/V wind vectors for client-side Canvas streamlines."""
  return fetch_wind_grid(
      arrays,
      stream_info,
      model_key=model_key,
      step_idx=step_idx,
      subsample=subsample,
      bbox=bbox,
      bilinear=bilinear,
  )


format_wind_vectors = extract_wind_vectors

__all__ = [
    "compute_wind_speed_and_direction",
    "extract_wind_vectors",
    "format_wind_vectors",
]
