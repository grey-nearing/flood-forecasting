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

"""U/V wind vector field extraction and viewport subsampling."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from multimet.weather_viewer.config import STEP_HOURS, SUPPORTED_MODELS
from multimet.weather_viewer.tiles import (
    file_step_for_lead,
    get_model_data_info_from_streams,
    sample_stream_grid,
)


def compute_wind_speed_and_direction(
    u: np.ndarray, v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes wind speed (m/s) and meteorological direction (degrees [0, 360))."""
  u_arr = np.asarray(u, dtype=np.float64)
  v_arr = np.asarray(v, dtype=np.float64)
  speed = np.sqrt(u_arr**2 + v_arr**2)
  direction = np.mod(np.degrees(np.arctan2(-u_arr, -v_arr)) + 360.0, 360.0)
  return speed, direction


def _resolve_base_time(
    stream_info: Mapping[str, Mapping[str, Any]], model_key: str
) -> datetime.datetime:
  """Resolves the forecast issue timestamp for a model."""
  init_time = get_model_data_info_from_streams(stream_info, model_key).get(
      "init_time"
  )
  if init_time:
    return datetime.datetime.fromisoformat(str(init_time).replace("Z", "+00:00"))
  return datetime.datetime.now(datetime.timezone.utc).replace(
      minute=0, second=0, microsecond=0
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
  """Extracts subsampled U/V 10m wind vectors from synced model streams.

  Args:
    arrays: Memory-mapped model stream arrays keyed by `<model>_<stream>`.
    stream_info: Stream metadata dictionary keyed by `<model>_<stream>`.
    model_key: Model identifier (e.g., `'ecmwf_ifs'`, `'ecmwf_aifs'`, `'noaa_gfs'`).
    step_idx: 3-hourly viewer step index (`0..80`).
    subsample: Subsampling factor (`1..4`), producing `step_deg = 1.0 * subsample`.
    bbox: Optional viewport bounding box `(min_lon, min_lat, max_lon, max_lat)`
      in EPSG:4326 degrees. Defaults to global coverage.
    bilinear: Whether to use bilinear interpolation when sampling.

  Returns:
    Dictionary containing `header`, `u`, and `v` lists for streamline rendering.

  Raises:
    FileNotFoundError: If U10 or V10 wind streams are not available in `arrays`.
    ValueError: If `model_key` is unsupported or `step_idx` exceeds the run horizon.
  """
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")

  u_stream = f"{model_key}_u10"
  v_stream = f"{model_key}_v10"
  u_info = stream_info.get(u_stream)
  v_info = stream_info.get(v_stream)
  if (
      not u_info
      or not v_info
      or u_stream not in arrays
      or v_stream not in arrays
  ):
    raise FileNotFoundError(
        f"Wind streams '{u_stream}' and '{v_stream}' not found in synced data."
    )

  subsample_clamped = max(1, min(4, int(subsample)))
  step_idx_clamped = max(0, int(step_idx))
  step_deg = 1.0 * subsample_clamped
  lead_h = step_idx_clamped * STEP_HOURS

  fu = file_step_for_lead(u_info, lead_h, is_rate=False)
  fv = file_step_for_lead(v_info, lead_h, is_rate=False)
  if fu is None or fv is None:
    raise ValueError(
        f"Requested lead hour {lead_h}h exceeds synced wind forecast horizon."
    )

  if bbox is not None:
    min_lon, min_lat, max_lon, max_lat = bbox
    la1 = float(min(90.0, max(-90.0, max_lat)))
    la2 = float(min(90.0, max(-90.0, min_lat)))
    lo1 = float(min(180.0, max(-180.0, min_lon)))
    lo2 = float(min(180.0, max(-180.0, max_lon)))
    ny = max(1, int(round((la1 - la2) / step_deg)) + 1)
    nx = max(1, int(round((lo2 - lo1) / step_deg)) + 1)
    lats = np.array([la1 - i * step_deg for i in range(ny)], dtype=np.float64)
    lons = np.array([lo1 + j * step_deg for j in range(nx)], dtype=np.float64)
  else:
    la1 = 90.0
    lo1 = -180.0
    lats = np.array(
        [90.0 - i * step_deg for i in range(int(180 / step_deg) + 1)],
        dtype=np.float64,
    )
    lons = np.array(
        [-180.0 + j * step_deg for j in range(int(360 / step_deg))],
        dtype=np.float64,
    )

  u = sample_stream_grid(
      arrays, stream_info, u_stream, fu, lats, lons, bilinear=bilinear
  )
  v = sample_stream_grid(
      arrays, stream_info, v_stream, fv, lats, lons, bilinear=bilinear
  )

  valid_dt = _resolve_base_time(stream_info, model_key) + datetime.timedelta(
      hours=lead_h
  )
  source = (
      "archived_run"
      if u_info.get("archived_run") and v_info.get("archived_run")
      else "local_binary"
  )
  return {
      "header": {
          "model": model_key,
          "step_hours": lead_h,
          "valid_time": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
          "nx": len(lons),
          "ny": len(lats),
          "lo1": lo1,
          "la1": la1,
          "dx": step_deg,
          "dy": step_deg,
          "data_source": source,
      },
      "u": np.round(np.nan_to_num(u).astype(np.float64), 2).ravel().tolist(),
      "v": np.round(np.nan_to_num(v).astype(np.float64), 2).ravel().tolist(),
  }
