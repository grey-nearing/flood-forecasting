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

"""WebMercator XYZ PNG raster tile & colorbar LUT renderer with bilinear interpolation."""

from __future__ import annotations

import mmap
from pathlib import Path
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import zlib

import numpy as np

from multimet.weather_viewer.config import (
    DEFAULT_MSLP_OFFSET_HPA,
    FRAME_SIZE,
    FRAME_VARIABLES,
    FRAME_VERSION,
    GRID_DEG,
    MAX_LEAD_HOURS,
    MERCATOR_MAX_LAT,
    N_LAT,
    N_LON,
    NUM_STEPS,
    PRESSURE_LEVELS,
    RAIN_ACCUM_CLASSES,
    RAIN_RATE_CLASSES,
    run_lead_hours,
    STEP_HOURS,
    STREAM_FILES,
    STREAM_SUFFIX,
    SUPPORTED_MODELS,
    SUPPORTED_VARIABLES,
    TEMP_LEVELS,
    TILE_VERSION,
)
from multimet.weather_viewer.sync import load_run_metadata, require_data_dir

_TRANSPARENT_TILE: Optional[bytes] = None
_EMPTY_FRAME: Optional[bytes] = None
_FRAME_COORDS: Optional[Tuple[np.ndarray, np.ndarray]] = None


def scan_streams(
    target_dir: Union[str, Path],
) -> Tuple[
    Dict[str, Tuple[mmap.mmap, int, int, int, bool]],
    Dict[str, Dict[str, Any]],
    Dict[str, np.ndarray],
]:
  """Memory-maps every forecast binary stream in target_dir.

  Args:
    target_dir: Directory containing `<model>_<stream>.bin` files and optional
      `latest_dynamical_meta.json`.

  Returns:
    Tuple of `(handles, stream_info, arrays)`.
  """
  resolved_dir = require_data_dir(target_dir)
  if not resolved_dir.exists() or not resolved_dir.is_dir():
    return {}, {}, {}

  runs = load_run_metadata(resolved_dir)
  plane_bytes = N_LAT * N_LON * 2
  handles: Dict[str, Tuple[mmap.mmap, int, int, int, bool]] = {}
  infos: Dict[str, Dict[str, Any]] = {}
  arrays: Dict[str, np.ndarray] = {}

  for stream_id, fname, is_precip in STREAM_FILES:
    fpath = resolved_dir / fname
    if not fpath.exists() or not fpath.is_file():
      continue
    file_size = fpath.stat().st_size
    n_steps = file_size // plane_bytes
    if n_steps < 1:
      continue
    with open(fpath, "rb") as f_handle:
      mm = mmap.mmap(f_handle.fileno(), 0, access=mmap.ACCESS_READ)

    model_key, suffix = stream_id.rsplit("_", 1)
    run = runs.get(model_key)
    run_leads = (run or {}).get("lead_hours")
    if run_leads:
      archived = len(run_leads) == n_steps
    else:
      archived = bool(run) and run["lead_steps"] == n_steps

    handles[stream_id] = (mm, n_steps, N_LAT, N_LON, is_precip)
    arrays[stream_id] = np.frombuffer(
        mm, dtype=np.float16, count=n_steps * N_LAT * N_LON
    ).reshape(n_steps, N_LAT, N_LON)
    if archived:
      lead_hours = run_leads or run_lead_hours(model_key, n_steps)
    else:
      lead_hours = [STEP_HOURS * i for i in range(n_steps)]

    infos[stream_id] = {
        "model": model_key,
        "file": str(fpath),
        "n_steps": n_steps,
        "lead_hours": lead_hours,
        "archived_run": archived,
        "init_time": run["init_time"] if archived and run else None,
        "downloaded_utc": run.get("downloaded_utc") if archived and run else None,
        "title": run.get("title") if archived and run else None,
        "offset": (
            (run["mslp_offset_hpa"] if run else DEFAULT_MSLP_OFFSET_HPA)
            if suffix == "mslp"
            else 0.0
        ),
    }

  return handles, infos, arrays


def get_model_data_info_from_streams(
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
) -> Dict[str, Any]:
  """Inspects stream metadata to report available variables and run info."""
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")

  real_variables: List[str] = []
  init_time: Optional[str] = None
  downloaded_utc: Optional[str] = None
  title: Optional[str] = None
  max_lead: int = MAX_LEAD_HOURS

  for var_key, suffix in (("precipitation", "precip"), ("temperature", "temp")):
    info = stream_info.get(f"{model_key}_{suffix}")
    if info and info.get("archived_run"):
      real_variables.append(var_key)
      init_time = info.get("init_time")
      downloaded_utc = info.get("downloaded_utc")
      title = info.get("title")
      max_lead = min(max_lead, int(info["lead_hours"][-1]))

  if "precipitation" in real_variables:
    real_variables.insert(1, "accumulated_precip")
  if real_variables:
    mslp = stream_info.get(f"{model_key}_mslp")
    if mslp and mslp.get("archived_run"):
      real_variables.append("pressure")
    if all(
        (stream_info.get(f"{model_key}_{c}") or {}).get("archived_run")
        for c in ("u10", "v10")
    ):
      real_variables.append("wind")

  all_variables = (
      "precipitation",
      "accumulated_precip",
      "temperature",
      "pressure",
      "wind",
  )
  return {
      "data_source": "archived_run" if real_variables else "unavailable",
      "init_time": init_time,
      "max_lead_hours": max_lead if real_variables else 0,
      "real_variables": real_variables,
      "missing_variables": [
          v for v in all_variables if v not in real_variables
      ],
      "tile_version": TILE_VERSION,
      "downloaded_utc": downloaded_utc,
      "dataset_title": title,
      "frame_version": FRAME_VERSION,
  }


def file_step_for_lead(
    info: Mapping[str, Any], lead_h: float, is_rate: bool
) -> Optional[int]:
  """Returns index of stored plane for a lead time, or None beyond the run."""
  leads = info["lead_hours"]
  if not leads or lead_h < 0 or lead_h > leads[-1]:
    return None
  if is_rate and info.get("archived_run"):
    target = lead_h if lead_h > 0 else 1e-6
    for i, lead in enumerate(leads):
      if lead >= target:
        return i
    return None
  best_i, best_d = 0, None
  for i, lead in enumerate(leads):
    d = abs(lead - lead_h)
    if best_d is None or d <= best_d:
      best_i, best_d = i, d
    elif lead > lead_h:
      break
  return best_i


def rate_file_steps(info: Mapping[str, Any], lead_h: float) -> List[int]:
  """Returns stored rain-rate plane indices to average for the step ending at lead_h."""
  first = file_step_for_lead(info, lead_h, is_rate=True)
  if first is None:
    return []
  if not info.get("archived_run") or lead_h <= 0:
    return [first]
  steps = [
      i
      for i, lead in enumerate(info["lead_hours"])
      if 0 < lead <= lead_h and lead > lead_h - STEP_HOURS
  ]
  return steps or [first]


def grid_indices(
    lats: np.ndarray,
    lons: np.ndarray,
    n_lat: int = N_LAT,
    n_lon: int = N_LON,
    grid_deg: float = GRID_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes nearest-neighbor row and column indices on a global regular grid."""
  rows = np.clip(np.rint((90.0 - lats) / grid_deg), 0, n_lat - 1).astype(
      np.intp
  )
  cols = (
      np.rint(((lons + 180.0) % 360.0) / grid_deg).astype(np.intp) % n_lon
  )
  return rows, cols


def bilinear_sample_grid(
    plane: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    grid_deg: float = GRID_DEG,
) -> np.ndarray:
  """Samples a 2D global grid at target `(lats, lons)` using bilinear interpolation.

  Args:
    plane: 2D array of shape `(n_lat, n_lon)` covering `+90..-90` and `-180..+180`.
    lats: 1D array of target latitudes in degrees.
    lons: 1D array of target longitudes in degrees.
    grid_deg: Grid spacing in degrees (default 0.25).

  Returns:
    2D float32 array of shape `(len(lats), len(lons))`.
  """
  n_lat, n_lon = plane.shape
  lat_arr = np.asarray(lats, dtype=np.float64)
  lon_arr = np.asarray(lons, dtype=np.float64)

  row_f = np.clip((90.0 - lat_arr) / grid_deg, 0.0, float(n_lat - 1))
  col_f = ((lon_arr + 180.0) % 360.0) / grid_deg

  r0 = np.floor(row_f).astype(np.intp)
  r1 = np.minimum(r0 + 1, n_lat - 1)
  dr = (row_f - r0).astype(np.float32)[:, None]

  c0 = np.floor(col_f).astype(np.intp) % n_lon
  c1 = (c0 + 1) % n_lon
  dc = (col_f - np.floor(col_f)).astype(np.float32)[None, :]

  plane_f32 = plane.astype(np.float32)
  v00 = plane_f32[np.ix_(r0, c0)]
  v01 = plane_f32[np.ix_(r0, c1)]
  v10 = plane_f32[np.ix_(r1, c0)]
  v11 = plane_f32[np.ix_(r1, c1)]

  interp = (
      (1.0 - dr) * ((1.0 - dc) * v00 + dc * v01)
      + dr * ((1.0 - dc) * v10 + dc * v11)
  )
  return interp.astype(np.float32)


def sample_stream_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    stream_id: str,
    file_step: int,
    lats: np.ndarray,
    lons: np.ndarray,
    bilinear: bool = False,
) -> np.ndarray:
  """Samples one stored plane on a `(lats, lons)` grid."""
  if stream_id not in arrays or stream_id not in stream_info:
    raise FileNotFoundError(f"Weather stream {stream_id!r} is not loaded.")
  plane = arrays[stream_id][file_step]
  if bilinear:
    values = bilinear_sample_grid(plane, lats, lons, grid_deg=GRID_DEG)
  else:
    rows, cols = grid_indices(lats, lons)
    values = plane[np.ix_(rows, cols)].astype(np.float32)
  offset = float(stream_info[stream_id].get("offset", 0.0))
  return values + offset if offset else values


def mean_rate_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    stream_id: str,
    file_steps: Sequence[int],
    lats: np.ndarray,
    lons: np.ndarray,
    bilinear: bool = False,
) -> np.ndarray:
  """Computes mean rain rate over one or more stored planes on a `(lats, lons)` grid."""
  if not file_steps:
    raise ValueError("file_steps must be non-empty.")
  if len(file_steps) == 1:
    return sample_stream_grid(
        arrays, stream_info, stream_id, file_steps[0], lats, lons, bilinear=bilinear
    )
  planes = np.nan_to_num(
      arrays[stream_id][list(file_steps)].astype(np.float32), nan=0.0
  ).mean(axis=0)
  if bilinear:
    return bilinear_sample_grid(planes, lats, lons, grid_deg=GRID_DEG)
  rows, cols = grid_indices(lats, lons)
  return planes[np.ix_(rows, cols)].astype(np.float32)


def compute_accumulated_precip_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    lead_h: float,
) -> Optional[Tuple[np.ndarray, float]]:
  """Computes rain (mm) accumulated from forecast start to lead_h on the global grid."""
  stream_id = f"{model_key}_precip"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Precipitation stream {stream_id!r} not found in synced data directory."
    )
  leads = info["lead_hours"]
  if lead_h > leads[-1]:
    return None
  k = max(i for i, lead in enumerate(leads) if lead <= lead_h)
  total = np.zeros((N_LAT, N_LON), dtype=np.float32)
  for i in range(1, k + 1):
    rate = arrays[stream_id][i].astype(np.float32)
    total += np.nan_to_num(rate, nan=0.0) * float(leads[i] - leads[i - 1])
  return total, GRID_DEG


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


def _png_chunk(tag: bytes, data: bytes) -> bytes:
  return (
      struct.pack("!I", len(data))
      + tag
      + data
      + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
  )


def make_png_bytes(
    width: int, height: int, raw_pixels: Union[bytes, bytearray]
) -> bytes:
  """Encodes raw RGBA scanlines (with filter byte per row) into PNG bytes."""
  header = b"\x89PNG\r\n\x1a\n"
  ihdr = _png_chunk(
      b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 6, 0, 0, 0)
  )
  idat = _png_chunk(b"IDAT", zlib.compress(bytes(raw_pixels), 1))
  iend = _png_chunk(b"IEND", b"")
  return header + ihdr + idat + iend


def encode_rgba_png(rgba: np.ndarray) -> bytes:
  """Encodes an `(H, W, 4)` uint8 RGBA array as a PNG byte stream."""
  height, width, _ = rgba.shape
  raw = np.zeros((height, 1 + width * 4), dtype=np.uint8)
  raw[:, 1:] = rgba.reshape(height, width * 4)
  return make_png_bytes(width, height, raw.tobytes())


def encode_indexed_png(idx: np.ndarray, palette: np.ndarray) -> bytes:
  """Encodes an 8-bit palette PNG with per-entry alpha (tRNS)."""
  height, width = idx.shape
  raw = np.zeros((height, width + 1), dtype=np.uint8)
  raw[:, 1:] = idx
  return (
      b"\x89PNG\r\n\x1a\n"
      + _png_chunk(
          b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 3, 0, 0, 0)
      )
      + _png_chunk(b"PLTE", palette[:, :3].astype(np.uint8).tobytes())
      + _png_chunk(b"tRNS", palette[:, 3].astype(np.uint8).tobytes())
      + _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
      + _png_chunk(b"IEND", b"")
  )


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


def classify_values(
    values: np.ndarray,
    classes: Sequence[Tuple[float, Tuple[int, int, int, int]]],
) -> np.ndarray:
  """Maps 2D scalar values into discrete RGBA color classes."""
  rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
  for lower, color in classes:
    rgba[values >= lower] = color
  return rgba


def colorize_rgba(var_key: str, values: np.ndarray) -> np.ndarray:
  """Maps 2D physical field values to an `(H, W, 4)` RGBA uint8 array."""
  finite = np.isfinite(values)
  v = np.where(finite, values, 0.0)
  if var_key == "precipitation":
    rgba = classify_values(v, RAIN_RATE_CLASSES)
  elif var_key == "accumulated_precip":
    rgba = classify_values(v, RAIN_ACCUM_CLASSES)
  elif var_key == "temperature":
    norm = np.clip((v + 35.0) / 80.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    for channel, centre in ((0, 3), (1, 2), (2, 1)):
      rgba[..., channel] = (
          255 * np.clip(1.5 - np.abs(norm * 4 - centre), 0.0, 1.0)
      ).astype(np.uint8)
    rgba[..., 3] = 180
  elif var_key == "pressure":
    rem = np.abs(np.mod(v, 4.0))
    isobar = (rem < 0.3) | (rem > 3.7)
    norm = np.clip((v - 980.0) / 50.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 14, 165, 233
    rgba[..., 3] = (30 + norm * 50).astype(np.uint8)
    rgba[isobar] = (255, 255, 255, 220)
  elif var_key == "wind":
    norm = np.clip(v / 40.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    for channel, centre in ((0, 3), (1, 2), (2, 1)):
      rgba[..., channel] = (
          255 * np.clip(1.5 - np.abs(norm * 4 - centre), 0.0, 1.0)
      ).astype(np.uint8)
    rgba[..., 3] = 180
  else:
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
  rgba[~finite] = 0
  return rgba


def _box_smooth(values: np.ndarray) -> np.ndarray:
  padded = np.pad(values, 1, mode="edge")
  h, w = values.shape
  total = np.zeros_like(values, dtype=np.float32)
  for dy in range(3):
    for dx in range(3):
      total += padded[dy : dy + h, dx : dx + w]
  return total / 9.0


def colorize_indexed(
    var_key: str, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns palette indices (uint8) and RGBA palette (index 0 is transparent)."""
  finite = np.isfinite(values)
  v = np.where(finite, values, 0.0).astype(np.float32)
  if var_key in ("precipitation", "accumulated_precip"):
    classes = (
        RAIN_RATE_CLASSES if var_key == "precipitation" else RAIN_ACCUM_CLASSES
    )
    bounds = np.array([lower for lower, _ in classes], dtype=np.float32)
    idx = np.searchsorted(bounds, v, side="right").astype(np.uint8)
    palette = np.array(
        [(0, 0, 0, 0)] + [color for _, color in classes], dtype=np.uint8
    )
  elif var_key == "temperature":
    n = TEMP_LEVELS
    norm = np.clip((v + 35.0) / 80.0, 0.0, 1.0)
    idx = (1 + np.rint(norm * (n - 1))).astype(np.uint8)
    levels = np.linspace(0.0, 1.0, n)
    palette = np.zeros((n + 1, 4), dtype=np.uint8)
    for channel, centre in ((0, 3), (1, 2), (2, 1)):
      palette[1:, channel] = (
          255 * np.clip(1.5 - np.abs(levels * 4 - centre), 0.0, 1.0)
      ).astype(np.uint8)
    palette[1:, 3] = 180
  elif var_key == "pressure":
    n = PRESSURE_LEVELS
    smooth = _box_smooth(v)
    norm = np.clip((smooth - 980.0) / 50.0, 0.0, 1.0)
    idx = (1 + np.rint(norm * (n - 1))).astype(np.uint8)
    band = np.floor(smooth / 4.0)
    edge = np.zeros(band.shape, dtype=bool)
    edge[:-1, :] |= band[:-1, :] != band[1:, :]
    edge[:, :-1] |= band[:, :-1] != band[:, 1:]
    idx[edge] = n + 1
    levels = np.linspace(0.0, 1.0, n)
    palette = np.zeros((n + 2, 4), dtype=np.uint8)
    palette[1 : n + 1, 0] = 14
    palette[1 : n + 1, 1] = 165
    palette[1 : n + 1, 2] = 233
    palette[1 : n + 1, 3] = (30 + levels * 50).astype(np.uint8)
    palette[n + 1] = (255, 255, 255, 220)
  else:
    idx = np.zeros(v.shape, dtype=np.uint8)
    palette = np.zeros((1, 4), dtype=np.uint8)
  idx[~finite] = 0
  return idx, palette


def render_colorbar_lut_png(
    var_key: str, width: int = 256, height: int = 16
) -> bytes:
  """Renders a horizontal colorbar LUT PNG for a supported weather variable."""
  if var_key not in SUPPORTED_VARIABLES:
    raise ValueError(
        f"Unsupported variable {var_key!r}. Supported: {list(SUPPORTED_VARIABLES)}"
    )
  var_cfg = SUPPORTED_VARIABLES[var_key]
  vmin = float(var_cfg["min"])
  vmax = float(var_cfg["max"])
  ramp = np.linspace(vmin, vmax, width, dtype=np.float32)[None, :]
  grid = np.repeat(ramp, height, axis=0)
  rgba = colorize_rgba(var_key, grid)
  return encode_rgba_png(rgba)


def _frame_signature_from_streams(
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step: int,
) -> Optional[Tuple[Tuple[Any, ...], float]]:
  """Returns `(signature_tuple, natural_lead_hours)` for a viewer step."""
  lead_h = step * STEP_HOURS
  if var_key == "accumulated_precip":
    info = stream_info.get(f"{model_key}_precip")
    if not info:
      raise FileNotFoundError(
          f"Precipitation stream '{model_key}_precip' not found in synced data."
      )
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
    raise FileNotFoundError(
        f"Weather stream '{stream_id}' not found in synced data."
    )
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
) -> Dict[str, Any]:
  """Computes the frame deduplication index for `/api/weather/frames/.../index.json`."""
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  if var_key not in FRAME_VARIABLES:
    raise ValueError(f"Layer '{var_key}' has no animation frames")

  data_info = get_model_data_info_from_streams(stream_info, model_key)
  if not data_info["real_variables"]:
    raise FileNotFoundError(
        f"No synced forecast run found for model '{model_key}'."
    )
  max_step = min(NUM_STEPS - 1, int(data_info["max_lead_hours"] // STEP_HOURS))
  sigs = [
      _frame_signature_from_streams(stream_info, model_key, var_key, s)
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
      "data_source": "archived_run",
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
) -> Optional[np.ndarray]:
  """Evaluates physical field values on `(lats, lons)` from synced streams."""
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  lead_h = step_idx * STEP_HOURS
  if var_key == "accumulated_precip":
    accumulated = compute_accumulated_precip_grid(
        arrays, stream_info, model_key, lead_h
    )
    if accumulated is None:
      return None
    grid, res = accumulated
    if bilinear:
      return bilinear_sample_grid(grid, lats, lons, grid_deg=res)
    rows, cols = grid_indices(
        lats, lons, n_lat=grid.shape[0], n_lon=grid.shape[1], grid_deg=res
    )
    return grid[np.ix_(rows, cols)]

  suffix = STREAM_SUFFIX.get(var_key)
  if suffix is None:
    raise ValueError(f"Variable '{var_key}' is not a raster tile variable.")
  stream_id = f"{model_key}_{suffix}"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Weather stream '{stream_id}' not found in synced data directory."
    )
  if suffix == "precip":
    rate_steps = rate_file_steps(info, lead_h)
    if not rate_steps:
      return None
    return mean_rate_grid(
        arrays, stream_info, stream_id, rate_steps, lats, lons, bilinear=bilinear
    )
  file_step = file_step_for_lead(info, lead_h, is_rate=False)
  if file_step is None:
    return None
  return sample_stream_grid(
      arrays, stream_info, stream_id, file_step, lats, lons, bilinear=bilinear
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
) -> bytes:
  """Renders a whole-world indexed PNG animation frame for one viewer step."""
  index = compute_frame_index(stream_info, model_key, var_key)
  step = max(0, min(NUM_STEPS - 1, int(step_idx)))
  rep = index["step_frames"][step]
  if rep is None:
    return empty_frame()
  lats, lons = frame_coordinates()
  values = evaluate_tile_field(
      arrays, stream_info, model_key, var_key, rep, lats, lons, bilinear=False
  )
  if values is None:
    return empty_frame()
  idx, palette = colorize_indexed(var_key, values)
  return encode_indexed_png(idx, palette)
