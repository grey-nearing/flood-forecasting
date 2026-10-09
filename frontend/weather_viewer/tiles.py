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

"""WebMercator XYZ PNG raster tile and full-world animation frame renderer."""

from __future__ import annotations

import collections
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from frontend.weather_viewer.colormaps import (
    colorize_indexed,
    colorize_rgba,
    encode_indexed_png,
    encode_rgba_png,
)
from multimet.weather_fetcher.config import (
    NUM_STEPS,
    STEP_HOURS,
    STREAM_SUFFIX,
    SUPPORTED_MODELS,
)
from multimet.weather_fetcher.fetcher import (
    fetch_forecast_grid,
    file_step_for_lead,
    get_model_data_info_from_streams,
    rate_file_steps,
)

TILE_VERSION: str = "3"
FRAME_SIZE: int = 1440
FRAME_VERSION: str = "1"
MERCATOR_MAX_LAT: float = 85.0511287798
FRAME_VARIABLES: Tuple[str, ...] = (
    "precipitation",
    "accumulated_precip",
    "temperature",
    "pressure",
)

_TRANSPARENT_TILE: Optional[bytes] = None
_EMPTY_FRAME: Optional[bytes] = None
_FRAME_COORDS: Optional[Tuple[np.ndarray, np.ndarray]] = None

_FRAME_CACHE: "collections.OrderedDict[Tuple[Any, ...], bytes]" = (
    collections.OrderedDict()
)
_FRAME_CACHE_LIMIT_BYTES: int = 384 << 20
_FRAME_CACHE_BYTES: int = 0
_FRAME_LOCK = threading.Lock()


def clear_frame_cache() -> None:
  """Clears the in-memory rendered animation frame cache."""
  global _FRAME_CACHE_BYTES
  with _FRAME_LOCK:
    _FRAME_CACHE.clear()
    _FRAME_CACHE_BYTES = 0


def tile_coordinates(
    z: int, x: int, y: int, tile_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns latitudes (rows) and longitudes (columns) of a Web Mercator tile."""
  n = 2.0**z
  frac = (np.arange(tile_size, dtype=np.float64) + 0.5) / float(tile_size)
  lons = (x + frac) / n * 360.0 - 180.0
  lats = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + frac) / n))))
  return lats, lons


def frame_coordinates() -> Tuple[np.ndarray, np.ndarray]:
  """Returns latitudes (rows) and longitudes (columns) of a full-world frame."""
  global _FRAME_COORDS
  if _FRAME_COORDS is None:
    frac = (np.arange(FRAME_SIZE, dtype=np.float64) + 0.5) / FRAME_SIZE
    lons = frac * 360.0 - 180.0
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * frac))))
    _FRAME_COORDS = (lats, lons)
  return _FRAME_COORDS


def transparent_tile() -> bytes:
  """Returns a cached 256x256 fully transparent RGBA PNG tile."""
  global _TRANSPARENT_TILE
  if _TRANSPARENT_TILE is None:
    _TRANSPARENT_TILE = encode_rgba_png(np.zeros((256, 256, 4), np.uint8))
  return _TRANSPARENT_TILE


def empty_frame() -> bytes:
  """Returns a cached 1x1 transparent indexed PNG frame."""
  global _EMPTY_FRAME
  if _EMPTY_FRAME is None:
    _EMPTY_FRAME = encode_indexed_png(
        np.zeros((1, 1), np.uint8), np.zeros((1, 4), np.uint8)
    )
  return _EMPTY_FRAME


def _frame_signature_from_streams(
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step: int,
    strict: bool = True,
) -> Optional[Tuple[Tuple[Any, ...], float]]:
  """Returns `(signature_tuple, natural_lead_hours)` for a viewer step."""
  lead_h = step * STEP_HOURS
  if var_key == "accumulated_precip":
    info = stream_info.get(f"{model_key}_precip")
    if not info:
      if strict:
        raise FileNotFoundError(
            f"Precipitation stream '{model_key}_precip' not found in synced data."
        )
      return None
    leads = info["lead_hours"]
    if lead_h > leads[-1]:
      return None
    k = max(i for i, lead in enumerate(leads) if lead <= lead_h)
    return ("accumulated", k), float(leads[k])

  suffix = STREAM_SUFFIX.get(var_key)
  if suffix is None:
    return None
  stream_id = f"{model_key}_{suffix}"
  info = stream_info.get(stream_id)
  if not info:
    if strict:
      raise FileNotFoundError(
          f"Weather stream '{stream_id}' not found in synced data."
      )
    return None
  if suffix == "precip":
    rate_steps = rate_file_steps(info, lead_h)
    if not rate_steps:
      return None
    return ("rate",) + tuple(rate_steps), float(
        info["lead_hours"][rate_steps[-1]]
    )
  file_step = file_step_for_lead(info, lead_h, is_rate=False)
  if file_step is None:
    return None
  return ("plane", file_step), float(info["lead_hours"][file_step])


def compute_frame_index(
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    strict: bool = True,
) -> Dict[str, Any]:
  """Computes the frame deduplication index for `/api/weather/frames/.../index.json`."""
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  if var_key not in FRAME_VARIABLES:
    raise ValueError(f"Layer '{var_key}' has no animation frames")

  data_info = get_model_data_info_from_streams(stream_info, model_key)
  if not data_info["real_variables"]:
    if strict:
      raise FileNotFoundError(
          f"No synced forecast run found for model '{model_key}'."
      )
    return {
        "model": model_key,
        "variable": var_key,
        "init_time": None,
        "data_source": "unavailable",
        "frame_version": FRAME_VERSION,
        "tile_version": TILE_VERSION,
        "projection": "EPSG:3857",
        "width": FRAME_SIZE,
        "height": FRAME_SIZE,
        "bounds": [[-MERCATOR_MAX_LAT, -180.0], [MERCATOR_MAX_LAT, 180.0]],
        "step_hours": STEP_HOURS,
        "max_step": 0,
        "step_frames": [None] * NUM_STEPS,
        "frame_steps": [],
    }

  max_step = min(NUM_STEPS - 1, int(data_info["max_lead_hours"] // STEP_HOURS))
  sigs = [
      _frame_signature_from_streams(
          stream_info, model_key, var_key, s, strict=strict
      )
      if s <= max_step
      else None
      for s in range(NUM_STEPS)
  ]
  groups: Dict[Tuple[Any, ...], List[int]] = {}
  natural: Dict[Tuple[Any, ...], float] = {}
  for s, sig in enumerate(sigs):
    if sig is not None:
      groups.setdefault(sig[0], []).append(s)
      natural[sig[0]] = sig[1]
  rep_of: Dict[Tuple[Any, ...], int] = {}
  for key, steps in groups.items():
    exact = [s for s in steps if s * STEP_HOURS == natural[key]]
    rep_of[key] = exact[0] if exact else steps[0]
  step_frames = [None if sig is None else rep_of[sig[0]] for sig in sigs]
  return {
      "model": model_key,
      "variable": var_key,
      "init_time": data_info["init_time"],
      "data_source": (
          "archived_run"
          if var_key in data_info["real_variables"]
          else "unavailable"
      ),
      "frame_version": FRAME_VERSION,
      "tile_version": TILE_VERSION,
      "projection": "EPSG:3857",
      "width": FRAME_SIZE,
      "height": FRAME_SIZE,
      "bounds": [[-MERCATOR_MAX_LAT, -180.0], [MERCATOR_MAX_LAT, 180.0]],
      "step_hours": STEP_HOURS,
      "max_step": max_step,
      "step_frames": step_frames,
      "frame_steps": sorted(set(rep_of.values())),
  }


def evaluate_tile_field(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step_idx: int,
    lats: np.ndarray,
    lons: np.ndarray,
    bilinear: bool = False,
    strict: bool = True,
) -> Optional[np.ndarray]:
  """Evaluates physical field values on `(lats, lons)` from synced streams."""
  suffix = STREAM_SUFFIX.get(var_key)
  if suffix is None:
    if strict:
      raise ValueError(f"Unsupported scalar forecast grid variable '{var_key}'.")
    return None
  stream_id = f"{model_key}_{suffix}"
  if not strict and (stream_id not in stream_info or stream_id not in arrays):
    return None
  return fetch_forecast_grid(
      arrays,
      stream_info,
      model_key=model_key,
      var_key=var_key,
      step_idx=step_idx,
      lats=lats,
      lons=lons,
      bilinear=bilinear,
  )


def render_raster_tile(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step_idx: int,
    z: int,
    x: int,
    y: int,
    bilinear: bool = True,
    strict: bool = True,
) -> bytes:
  """Renders a 256x256 Web Mercator PNG tile from synced model streams."""
  step_idx = max(0, int(step_idx))
  lats, lons = tile_coordinates(z, x, y)
  values = evaluate_tile_field(
      arrays,
      stream_info,
      model_key,
      var_key,
      step_idx,
      lats,
      lons,
      bilinear=bilinear,
      strict=strict,
  )
  if values is None:
    return transparent_tile()
  rgba = colorize_rgba(var_key, values)
  rgba[np.abs(lats) > 85.0511] = 0
  return encode_rgba_png(rgba)


def render_weather_frame(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step_idx: int,
    strict: bool = True,
) -> bytes:
  """Renders a whole-world indexed PNG animation frame for one viewer step."""
  global _FRAME_CACHE_BYTES
  index = compute_frame_index(stream_info, model_key, var_key, strict=strict)
  step = max(0, min(NUM_STEPS - 1, int(step_idx)))
  rep = index["step_frames"][step]
  if rep is None:
    return empty_frame()

  suffix = STREAM_SUFFIX.get(var_key, "")
  stream_id = f"{model_key}_{suffix}"
  stream_file = str((stream_info.get(stream_id) or {}).get("file") or "")
  arr_id = id(arrays.get(stream_id)) if stream_id in arrays else 0
  key = (arr_id, stream_file, model_key, var_key, rep, index["init_time"])

  with _FRAME_LOCK:
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
      _FRAME_CACHE.move_to_end(key)
      return cached

  lats, lons = frame_coordinates()
  values = evaluate_tile_field(
      arrays,
      stream_info,
      model_key,
      var_key,
      rep,
      lats,
      lons,
      bilinear=False,
      strict=strict,
  )
  png = (
      empty_frame()
      if values is None
      else encode_indexed_png(*colorize_indexed(var_key, values))
  )
  with _FRAME_LOCK:
    if key not in _FRAME_CACHE:
      _FRAME_CACHE[key] = png
      _FRAME_CACHE_BYTES += len(png)
      while (
          _FRAME_CACHE_BYTES > _FRAME_CACHE_LIMIT_BYTES
          and len(_FRAME_CACHE) > 1
      ):
        _, old = _FRAME_CACHE.popitem(last=False)
        _FRAME_CACHE_BYTES -= len(old)
  return png
