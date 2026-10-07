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

"""Pure weather data fetching engine for gridded NWP forecasts, points, and basins."""

from __future__ import annotations

import datetime
import math
import mmap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from multimet.weather_fetcher.config import (
    DEFAULT_MSLP_OFFSET_HPA,
    GRID_DEG,
    MAX_LEAD_HOURS,
    N_LAT,
    N_LON,
    NUM_STEPS,
    run_lead_hours,
    RUN_METADATA_FILE,
    STEP_HOURS,
    STREAM_FILES,
    STREAM_SUFFIX,
    SUPPORTED_MODELS,
)
from multimet.weather_fetcher.sync import (
    current_run_dir,
    list_available_runs,
    load_run_metadata,
    read_sync_status,
    require_data_dir,
)

_PROBE_STREAM_SUFFIX: Dict[str, str] = {
    **STREAM_SUFFIX,
    "wind_u": "u10",
    "wind_v": "v10",
}


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
        "downloaded_utc": (
            run.get("downloaded_utc") if archived and run else None
        ),
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
      "downloaded_utc": downloaded_utc,
      "dataset_title": title,
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
  """Samples a 2D global grid at target `(lats, lons)` using bilinear interpolation."""
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
        arrays,
        stream_info,
        stream_id,
        file_steps[0],
        lats,
        lons,
        bilinear=bilinear,
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


def fetch_forecast_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step_idx: int,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    bilinear: bool = False,
) -> Optional[np.ndarray]:
  """Fetches a 2D physical forecast array for `(model_key, var_key, step_idx)`.

  If `lats` and `lons` are omitted, returns the native `721 x 1440` (`0.25 deg`)
  global grid. Returns `None` if `step_idx` is beyond the run's max lead time.

  Raises:
    FileNotFoundError: If the requested model/variable stream is not synced.
    ValueError: If `model_key` or `var_key` is unsupported.
  """
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  if lats is None:
    lats = np.linspace(90.0, -90.0, N_LAT, dtype=np.float64)
  if lons is None:
    lons = np.linspace(-180.0, 180.0, N_LON, endpoint=False, dtype=np.float64)

  lead_h = max(0, int(step_idx)) * STEP_HOURS
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
    raise ValueError(f"Unsupported scalar forecast grid variable '{var_key}'.")
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
        arrays,
        stream_info,
        stream_id,
        rate_steps,
        lats,
        lons,
        bilinear=bilinear,
    )
  file_step = file_step_for_lead(info, lead_h, is_rate=False)
  if file_step is None:
    return None
  return sample_stream_grid(
      arrays, stream_info, stream_id, file_step, lats, lons, bilinear=bilinear
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
    return datetime.datetime.fromisoformat(
        str(init_time).replace("Z", "+00:00")
    )
  return datetime.datetime.now(datetime.timezone.utc).replace(
      minute=0, second=0, microsecond=0
  )


def fetch_wind_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str = "ecmwf_ifs",
    step_idx: int = 0,
    subsample: int = 2,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    bilinear: bool = False,
) -> Dict[str, Any]:
  """Fetches subsampled 10m U/V wind vector arrays and grid metadata.

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


def round_or_none(value: Optional[float], digits: int) -> Optional[float]:
  """Rounds finite floats to `digits` decimals, returning None for non-finite."""
  if value is None or not math.isfinite(value):
    return None
  return round(float(value), digits)


def extract_point_value(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    lead_h: float,
    lat: float,
    lon: float,
) -> Optional[float]:
  """Extracts a single scalar forecast value at `(lat, lon)` from synced streams."""
  suffix = _PROBE_STREAM_SUFFIX.get(var_key)
  if suffix is None:
    raise ValueError(f"Unsupported probe variable '{var_key}'")
  stream_id = f"{model_key}_{suffix}"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Weather stream '{stream_id}' not found in synced data directory."
    )

  rows, cols = grid_indices(
      np.array([lat], dtype=np.float64), np.array([lon], dtype=np.float64)
  )
  if suffix == "precip":
    rate_steps = rate_file_steps(info, lead_h)
    if not rate_steps:
      return None
    if len(rate_steps) > 1:
      values = arrays[stream_id][rate_steps, rows[0], cols[0]].astype(
          np.float64
      )
      return float(np.nan_to_num(values, nan=0.0).mean())
    file_step = rate_steps[0]
  else:
    file_step = file_step_for_lead(info, lead_h, is_rate=False)
    if file_step is None:
      return None

  value = float(arrays[stream_id][file_step, rows[0], cols[0]]) + float(
      info.get("offset", 0.0)
  )
  return value if math.isfinite(value) else None


def extract_accumulation_series(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    lat: float,
    lon: float,
    lead_hours: Sequence[float],
) -> List[Optional[float]]:
  """Computes cumulative precipitation (mm) since forecast start at `(lat, lon)`."""
  stream_id = f"{model_key}_precip"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Precipitation stream '{stream_id}' not found in synced data directory."
    )
  rows, cols = grid_indices(
      np.array([lat], dtype=np.float64), np.array([lon], dtype=np.float64)
  )
  rates = np.nan_to_num(
      arrays[stream_id][:, rows[0], cols[0]].astype(np.float64), nan=0.0
  )
  leads = info["lead_hours"]
  totals: List[float] = [0.0]
  for i in range(1, len(leads)):
    totals.append(
        totals[-1] + float(rates[i]) * float(leads[i] - leads[i - 1])
    )
  out: List[Optional[float]] = []
  for lead_h in lead_hours:
    if lead_h > leads[-1]:
      out.append(None)
      continue
    idx = max(i for i, lead in enumerate(leads) if lead <= lead_h)
    out.append(totals[idx])
  return out


def fetch_point_timeseries(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    lat: float,
    lon: float,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
  """Fetches comparative 10-day multi-model meteorological soundings at `(lat, lon)`."""
  lead_hours = [i * STEP_HOURS for i in range(NUM_STEPS)]
  now_utc = datetime.datetime.now(datetime.timezone.utc).replace(
      minute=0, second=0, microsecond=0
  )

  if models is not None:
    target_models = list(models)
  else:
    target_models = [
        m
        for m in SUPPORTED_MODELS
        if f"{m}_precip" in stream_info and f"{m}_temp" in stream_info
    ]

  if not target_models:
    raise FileNotFoundError(
        "No synced forecast models available in data directory for point probe."
    )

  results: Dict[str, Any] = {
      "latitude": round(float(lat), 4),
      "longitude": round(float(lon), 4),
      "query_time_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "lead_hours": lead_hours,
      "models": {},
  }

  for m_key in target_models:
    if m_key not in SUPPORTED_MODELS:
      raise ValueError(f"Unknown weather model '{m_key}'")
    m_info = SUPPORTED_MODELS[m_key]
    data_info = get_model_data_info_from_streams(stream_info, m_key)
    accum = extract_accumulation_series(
        arrays, stream_info, m_key, lat, lon, lead_hours
    )

    has_wind = (
        f"{m_key}_u10" in stream_info
        and f"{m_key}_v10" in stream_info
        and f"{m_key}_u10" in arrays
        and f"{m_key}_v10" in arrays
    )
    has_mslp = f"{m_key}_mslp" in stream_info and f"{m_key}_mslp" in arrays

    precip_curve: List[Optional[float]] = []
    temp_curve: List[Optional[float]] = []
    wind_spd_curve: List[Optional[float]] = []
    wind_dir_curve: List[Optional[int]] = []
    pressure_curve: List[Optional[float]] = []

    for lead_h in lead_hours:
      precip_curve.append(
          round_or_none(
              extract_point_value(
                  arrays, stream_info, m_key, "precipitation", lead_h, lat, lon
              ),
              2,
          )
      )
      temp_curve.append(
          round_or_none(
              extract_point_value(
                  arrays, stream_info, m_key, "temperature", lead_h, lat, lon
              ),
              1,
          )
      )
      if has_wind:
        u_w = extract_point_value(
            arrays, stream_info, m_key, "wind_u", lead_h, lat, lon
        )
        v_w = extract_point_value(
            arrays, stream_info, m_key, "wind_v", lead_h, lat, lon
        )
        if u_w is not None and v_w is not None:
          wind_spd_curve.append(round(math.sqrt(u_w**2 + v_w**2), 1))
          wind_dir_curve.append(
              int((math.degrees(math.atan2(-u_w, -v_w)) + 360) % 360)
          )
        else:
          wind_spd_curve.append(None)
          wind_dir_curve.append(None)
      else:
        wind_spd_curve.append(None)
        wind_dir_curve.append(None)

      if has_mslp:
        p_val = extract_point_value(
            arrays, stream_info, m_key, "pressure", lead_h, lat, lon
        )
        pressure_curve.append(round_or_none(p_val, 1))
      else:
        pressure_curve.append(None)

    results["models"][m_key] = {
        "name": m_info["name"],
        "badge": m_info["badge"],
        "precip_rate_mmh": precip_curve,
        "accum_precip_mm": [round_or_none(a, 1) for a in accum],
        "temp_c": temp_curve,
        "wind_speed_mps": wind_spd_curve,
        "wind_direction_deg": wind_dir_curve,
        "pressure_hpa": pressure_curve,
        **data_info,
    }

  return results


extract_point_probe = fetch_point_timeseries


def geometry_points(geom: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
  """Extracts exterior ring `(lats, lons)` lists from a GeoJSON geometry."""
  lats: List[float] = []
  lons: List[float] = []
  coords = geom.get("coordinates", [])
  rings = []
  if geom.get("type") == "Polygon" and coords:
    rings = [coords[0]]
  elif geom.get("type") == "MultiPolygon" and coords:
    rings = [poly[0] for poly in coords if poly]
  for ring in rings:
    for pt in ring:
      if len(pt) >= 2:
        lons.append(float(pt[0]))
        lats.append(float(pt[1]))
  return lats, lons


def fetch_catchment_summary(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    geojson_feature: Mapping[str, Any],
    step_idx: int = 0,
    model_key: str = "ecmwf_ifs",
) -> Dict[str, Any]:
  """Calculates basin-averaged precipitation and temperature for a catchment."""
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  if (
      f"{model_key}_precip" not in stream_info
      or f"{model_key}_temp" not in stream_info
  ):
    raise FileNotFoundError(
        f"Synced precipitation/temperature streams for '{model_key}' not found."
    )

  step_idx_clamped = max(0, int(step_idx))
  lead_h = step_idx_clamped * STEP_HOURS

  props = geojson_feature.get("properties") or {}
  catchment_id = (
      props.get("catchment_id") or geojson_feature.get("id") or "basin"
  )
  area_km2 = float(props.get("area_km2", 1250.0))

  lats, lons = geometry_points(geojson_feature.get("geometry") or {})
  if lats and lons:
    c_lat = sum(lats) / len(lats)
    c_lon = sum(lons) / len(lons)
  else:
    c_lat = float(props.get("outlet_latitude", 40.0))
    c_lon = float(props.get("outlet_longitude", -86.0))

  sample_points = list(zip(lats, lons)) if lats and lons else [(c_lat, c_lon)]
  stride = max(1, len(sample_points) // 10)
  rates = [
      r
      for r in (
          extract_point_value(
              arrays,
              stream_info,
              model_key,
              "precipitation",
              lead_h,
              p_lat,
              p_lon,
          )
          for p_lat, p_lon in sample_points[::stride]
      )
      if r is not None
  ]
  temp_c = extract_point_value(
      arrays, stream_info, model_key, "temperature", lead_h, c_lat, c_lon
  )

  data_info = get_model_data_info_from_streams(stream_info, model_key)
  window_h = int(data_info["max_lead_hours"] or 0)
  accum_list = extract_accumulation_series(
      arrays, stream_info, model_key, c_lat, c_lon, [window_h]
  )
  accum = accum_list[0] if accum_list else None

  init_time = data_info.get("init_time")
  if init_time:
    base_dt = datetime.datetime.fromisoformat(
        str(init_time).replace("Z", "+00:00")
    )
  else:
    base_dt = datetime.datetime.now(datetime.timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
  valid_dt = base_dt + datetime.timedelta(hours=lead_h)

  return {
      "catchment_id": catchment_id,
      "area_km2": round(area_km2, 1),
      "step_hours": lead_h,
      "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "basin_mean_precip_mmh": (
          round(sum(rates) / len(rates), 2) if rates else None
      ),
      "basin_max_precip_mmh": round(max(rates), 2) if rates else None,
      "basin_accumulated_10d_mm": round_or_none(accum, 1),
      "basin_mean_temp_c": round_or_none(temp_c, 1),
      "centroid": {"latitude": round(c_lat, 4), "longitude": round(c_lon, 4)},
      "model": model_key,
      "data_source": data_info["data_source"],
      "accumulation_hours": window_h,
  }


extract_catchment_weather_summary = fetch_catchment_summary


class WeatherDataFetcher:
  """Pure backend weather data fetcher for gridded NWP forecasts.

  Requires an explicit `data_dir` path and never fabricates synthetic weather
  grids or renders visual PNG tiles/colormaps. If a requested model run is
  missing from `data_dir`, methods raise `FileNotFoundError`.
  """

  def __init__(self, data_dir: Union[str, Path]):
    self.data_dir: Path = require_data_dir(data_dir)
    self.handles: Dict[str, Tuple[mmap.mmap, int, int, int, bool]] = {}
    self.stream_info: Dict[str, Dict[str, Any]] = {}
    self.arrays: Dict[str, np.ndarray] = {}
    self._loaded_signature: Optional[Tuple[str, int]] = None
    self._loaded_dir: Optional[Path] = None
    self._load_streams()

  def _resolve_active_dir(self) -> Path:
    active = current_run_dir(self.data_dir)
    return active if active is not None else self.data_dir

  def _compute_signature(
      self, target_dir: Path
  ) -> Optional[Tuple[str, int]]:
    if not target_dir.exists():
      return None
    resolved = target_dir.resolve()
    meta_file = resolved / RUN_METADATA_FILE
    mtime = meta_file.stat().st_mtime_ns if meta_file.exists() else 0
    return (str(resolved), mtime)

  def _load_streams(self) -> None:
    target_dir = self._resolve_active_dir()
    handles, infos, arrays = scan_streams(target_dir)
    self.handles = handles
    self.stream_info = infos
    self.arrays = arrays
    self._loaded_signature = self._compute_signature(target_dir)
    self._loaded_dir = target_dir.resolve() if target_dir.exists() else None

  def reload_if_changed(self) -> bool:
    """Reloads memory-mapped streams if a newer run was swapped into data_dir."""
    target_dir = self._resolve_active_dir()
    sig = self._compute_signature(target_dir)
    if sig == self._loaded_signature:
      return False
    self._load_streams()
    return True

  def list_available_runs(self) -> List[Dict[str, Any]]:
    """Lists all available synced forecast runs in `data_dir`."""
    return list_available_runs(self.data_dir)

  def get_model_info(self, model_key: str) -> Dict[str, Any]:
    """Returns metadata and availability status for `model_key`."""
    return get_model_data_info_from_streams(self.stream_info, model_key)

  def get_all_models_info(self) -> List[Dict[str, Any]]:
    """Returns metadata and availability status for all supported models."""
    return [
        {**info, **self.get_model_info(key)}
        for key, info in SUPPORTED_MODELS.items()
    ]

  def get_sync_status(self) -> Dict[str, Any]:
    """Returns synchronization status from `<data_dir>/sync_status.json`."""
    status = read_sync_status(self.data_dir)
    if not status:
      status = {
          "last_result": "never",
          "message": "No automatic update has run yet.",
      }
    return {
        "last_check_utc": status.get("last_check_utc"),
        "last_success_utc": status.get("last_success_utc"),
        "last_result": status.get("last_result"),
        "message": status.get("message"),
        "check_interval_minutes": status.get("check_interval_minutes", 60),
        "auto_update": self._loaded_dir is not None,
        "data_dir": str(self._loaded_dir) if self._loaded_dir else None,
    }

  def fetch_forecast_grid(
      self,
      model_key: str,
      var_key: str,
      step_idx: int,
      lats: Optional[np.ndarray] = None,
      lons: Optional[np.ndarray] = None,
      bilinear: bool = False,
  ) -> Optional[np.ndarray]:
    """Fetches a 2D physical forecast grid for `(model_key, var_key, step_idx)`."""
    return fetch_forecast_grid(
        self.arrays,
        self.stream_info,
        model_key=model_key,
        var_key=var_key,
        step_idx=step_idx,
        lats=lats,
        lons=lons,
        bilinear=bilinear,
    )

  def fetch_wind_grid(
      self,
      model_key: str = "ecmwf_ifs",
      step_idx: int = 0,
      subsample: int = 2,
      bbox: Optional[Tuple[float, float, float, float]] = None,
      bilinear: bool = False,
  ) -> Dict[str, Any]:
    """Fetches subsampled 10m U/V wind vector arrays and grid metadata."""
    return fetch_wind_grid(
        self.arrays,
        self.stream_info,
        model_key=model_key,
        step_idx=step_idx,
        subsample=subsample,
        bbox=bbox,
        bilinear=bilinear,
    )

  get_wind_vectors = fetch_wind_grid

  def fetch_point_timeseries(
      self,
      lat: float,
      lon: float,
      models: Optional[Sequence[str]] = None,
  ) -> Dict[str, Any]:
    """Fetches 10-day multi-model meteogram time series at `(lat, lon)`."""
    return fetch_point_timeseries(
        self.arrays, self.stream_info, lat=lat, lon=lon, models=models
    )

  probe_point = fetch_point_timeseries

  def fetch_catchment_summary(
      self,
      geojson_feature: Mapping[str, Any],
      step_idx: int = 0,
      model_key: str = "ecmwf_ifs",
  ) -> Dict[str, Any]:
    """Computes catchment-averaged precipitation and temperature summary."""
    return fetch_catchment_summary(
        self.arrays,
        self.stream_info,
        geojson_feature=geojson_feature,
        step_idx=step_idx,
        model_key=model_key,
    )

  summarize_catchment = fetch_catchment_summary
