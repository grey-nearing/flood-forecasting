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

"""Weather data fetching engine for gridded NWP forecasts, points, and basins.

All functions read the float16 binary planes written by
`multimet.weather_fetcher.sync`. Missing data stays missing: grid cells that
are `NaN` on disk (for example outside the NOAA HRRR CONUS domain or over the
ocean for NOAA CPC) are returned as `NaN` (JSON `null`), never as `0.0`.
Forecast leads that a model did not store are reported as `None`, never as the
nearest stored lead.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import math
import mmap
from pathlib import Path
import sys
import threading
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import shapely.geometry
import shapely.validation

from multimet.utils.geometry import geodesic_area_km2
from multimet.utils.zonal import (
    MIN_VALID_COVERAGE_FRACTION,
    weighted_mean_valid_with_coverage,
    ZonalWeightCalculator,
)
from multimet.weather_fetcher.config import (
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

StreamHandle = Tuple[mmap.mmap, int, int, int, bool]

# Grid cell centres of the stored 0.25 deg global planes.
GRID_LATS: np.ndarray = np.linspace(90.0, -90.0, N_LAT, dtype=np.float64)
GRID_LONS: np.ndarray = np.linspace(
    -180.0, 180.0, N_LON, endpoint=False, dtype=np.float64
)

# Wind vectors are returned on a 1 deg grid at subsample=1 (4x the native grid).
WIND_BASE_STEP_DEG: float = 1.0
WIND_MAX_SUBSAMPLE: int = 4

_PROBE_STREAM_SUFFIX: Dict[str, str] = {
    **STREAM_SUFFIX,
    "wind_u": "u10",
    "wind_v": "v10",
}

# Accumulated-precipitation grid cache. Keys identify the exact binary file
# (path, run init time, file mtime, plane count) plus the last plane index
# included in the total, so a re-synced file can never serve a stale total.
_AccumKey = Tuple[str, Optional[str], int, int, int]
_ACCUM_GRID_CACHE: "collections.OrderedDict[_AccumKey, np.ndarray]" = (
    collections.OrderedDict()
)
_ACCUM_GRID_CACHE_SIZE: int = 32
_ACCUM_GRID_LOCK = threading.Lock()

# `sys.getrefcount(mm)` inside `_mmap_has_external_references` counts: the
# caller's local variable, the function parameter, and getrefcount's own
# argument. Any NumPy array created from the mapping adds one more reference.
_MMAP_BASELINE_REFCOUNT: int = 3


def clear_accum_grid_cache() -> None:
  """Clears the incremental accumulated precipitation grid cache."""
  with _ACCUM_GRID_LOCK:
    _ACCUM_GRID_CACHE.clear()


def _evict_accum_grid_cache(files: Iterable[str]) -> None:
  """Drops cached accumulation grids computed from the given binary files."""
  retired = set(files)
  if not retired:
    return
  with _ACCUM_GRID_LOCK:
    for key in [k for k in _ACCUM_GRID_CACHE if k[0] in retired]:
      del _ACCUM_GRID_CACHE[key]


def _validate_lat_lon(lat_arr: np.ndarray, lon_arr: np.ndarray) -> None:
  """Raises ValueError for non-finite or out-of-range coordinates."""
  if lat_arr.size and (
      not np.all(np.isfinite(lat_arr))
      or float(lat_arr.min()) < -90.0
      or float(lat_arr.max()) > 90.0
  ):
    raise ValueError("Latitudes must be finite and within [-90, 90] degrees.")
  if lon_arr.size and (
      not np.all(np.isfinite(lon_arr))
      or float(lon_arr.min()) < -180.0
      or float(lon_arr.max()) > 360.0
  ):
    raise ValueError(
        "Longitudes must be finite and within [-180, 360] degrees."
    )


def _validate_step_idx(step_idx: int) -> int:
  """Returns `step_idx` as a non-negative integer or raises ValueError."""
  if isinstance(step_idx, bool) or int(step_idx) != step_idx or step_idx < 0:
    raise ValueError(
        f"step_idx must be a non-negative integer, got {step_idx!r}."
    )
  return int(step_idx)


def _parse_init_time(init_time: Any) -> datetime.datetime:
  """Parses an ISO-8601 run issue time into an aware UTC datetime."""
  parsed = datetime.datetime.fromisoformat(
      str(init_time).replace("Z", "+00:00")
  )
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=datetime.timezone.utc)
  return parsed.astimezone(datetime.timezone.utc)


def scan_streams(
    target_dir: Union[str, Path],
) -> Tuple[
    Dict[str, StreamHandle],
    Dict[str, Dict[str, Any]],
    Dict[str, np.ndarray],
]:
  """Memory-maps every forecast binary stream in target_dir.

  Every `<model>_<stream>.bin` file must be described by an entry for `<model>`
  in `latest_dynamical_meta.json` and must hold exactly
  `len(lead_hours)` complete `721 x 1440` float16 planes.

  Args:
    target_dir: Directory containing `<model>_<stream>.bin` files and
      `latest_dynamical_meta.json`.

  Returns:
    Tuple of `(handles, stream_info, arrays)`. All three are empty when the
    directory does not exist or holds no stream files.

  Raises:
    FileNotFoundError: If stream files exist without readable run metadata.
    ValueError: If a stream file is truncated, corrupt, or does not match the
      lead hours recorded in the run metadata.
  """
  resolved_dir = require_data_dir(target_dir)
  if not resolved_dir.exists() or not resolved_dir.is_dir():
    return {}, {}, {}

  present = [
      (stream_id, resolved_dir / fname, is_precip)
      for stream_id, fname, is_precip in STREAM_FILES
      if (resolved_dir / fname).is_file()
  ]
  if not present:
    return {}, {}, {}

  runs = load_run_metadata(resolved_dir)
  if not runs:
    raise FileNotFoundError(
        f"{resolved_dir} holds forecast stream files but no readable"
        f" {RUN_METADATA_FILE}; the run init time and lead hours are unknown."
    )

  plane_bytes = N_LAT * N_LON * 2
  specs: List[Tuple[str, Path, bool, int, List[int], Dict[str, Any], int]] = []
  for stream_id, fpath, is_precip in present:
    model_key, suffix = stream_id.rsplit("_", 1)
    run = runs.get(model_key)
    if run is None:
      raise ValueError(
          f"{fpath.name} is not described by {RUN_METADATA_FILE} (no entry"
          f" for model {model_key!r})."
      )
    stat = fpath.stat()
    file_size = int(stat.st_size)
    if file_size < plane_bytes or file_size % plane_bytes != 0:
      raise ValueError(
          f"{fpath} is truncated or corrupt: {file_size} bytes is not a"
          f" positive multiple of the {plane_bytes}-byte {N_LAT}x{N_LON}"
          " float16 plane size."
      )
    n_steps = file_size // plane_bytes
    run_leads = run.get("lead_hours")
    if run_leads:
      lead_hours = [int(h) for h in run_leads]
      if len(lead_hours) != n_steps:
        raise ValueError(
            f"{fpath.name} holds {n_steps} planes but {RUN_METADATA_FILE}"
            f" lists {len(lead_hours)} lead hours for {model_key!r}."
        )
    else:
      if int(run.get("lead_steps") or 0) != n_steps:
        raise ValueError(
            f"{fpath.name} holds {n_steps} planes but {RUN_METADATA_FILE}"
            f" records lead_steps={run.get('lead_steps')} for {model_key!r}."
        )
      lead_hours = run_lead_hours(model_key, n_steps)
    if lead_hours != sorted(set(lead_hours)) or lead_hours[0] < 0:
      raise ValueError(
          f"Lead hours for {model_key!r} must be strictly increasing and"
          f" non-negative, got {lead_hours}."
      )
    specs.append((
        stream_id,
        fpath,
        is_precip,
        n_steps,
        lead_hours,
        run,
        stat.st_mtime_ns,
    ))

  handles: Dict[str, StreamHandle] = {}
  infos: Dict[str, Dict[str, Any]] = {}
  arrays: Dict[str, np.ndarray] = {}
  for stream_id, fpath, is_precip, n_steps, lead_hours, run, mtime_ns in specs:
    model_key, suffix = stream_id.rsplit("_", 1)
    with open(fpath, "rb") as f_handle:
      mm = mmap.mmap(f_handle.fileno(), 0, access=mmap.ACCESS_READ)
    handles[stream_id] = (mm, n_steps, N_LAT, N_LON, is_precip)
    arrays[stream_id] = np.frombuffer(
        mm, dtype=np.float16, count=n_steps * N_LAT * N_LON
    ).reshape(n_steps, N_LAT, N_LON)
    infos[stream_id] = {
        "model": model_key,
        "stream": suffix,
        "file": str(fpath),
        "n_steps": n_steps,
        "lead_hours": lead_hours,
        "archived_run": True,
        "init_time": run["init_time"],
        "downloaded_utc": run.get("downloaded_utc"),
        "title": run.get("title"),
        "offset": float(run["mslp_offset_hpa"]) if suffix == "mslp" else 0.0,
        "mtime_ns": int(mtime_ns),
        "size_bytes": n_steps * plane_bytes,
    }

  return handles, infos, arrays


def _mmap_has_external_references(mm: mmap.mmap) -> bool:
  """True when a NumPy array (or anything else) still references `mm`."""
  return sys.getrefcount(mm) > _MMAP_BASELINE_REFCOUNT


def close_unreferenced_mmaps(handles: Dict[str, StreamHandle]) -> int:
  """Closes the mappings in `handles` that no array references any more.

  Mappings still referenced by live NumPy views are left open; CPython unmaps
  them as soon as the last view is garbage collected. `handles` is emptied.
  The caller must not hold any other reference to the retired mappings (for
  example a dict of arrays built from them) while calling this.

  Returns:
    Number of mappings closed synchronously.
  """
  closed = 0
  for stream_id in list(handles):
    mm = handles.pop(stream_id)[0]
    if mm.closed:
      continue
    if _mmap_has_external_references(mm):
      continue
    mm.close()
    closed += 1
  return closed


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
  stored_leads: Optional[List[int]] = None

  for var_key, suffix in (("precipitation", "precip"), ("temperature", "temp")):
    info = stream_info.get(f"{model_key}_{suffix}")
    if info and info.get("archived_run"):
      real_variables.append(var_key)
      init_time = info.get("init_time")
      downloaded_utc = info.get("downloaded_utc")
      title = info.get("title")
      max_lead = min(max_lead, int(info["lead_hours"][-1]))
      if stored_leads is None:
        stored_leads = [int(h) for h in info["lead_hours"]]

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
      "stored_lead_hours": stored_leads,
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
  """Returns the index of the stored plane to use for `lead_h`, or None.

  Rain-rate planes (`is_rate=True`) hold the mean rate over the model interval
  that ends at their lead hour, so the plane for `lead_h` is the first stored
  lead `>= lead_h` (lead 0 maps to the first interval). State variables
  (`is_rate=False`) are only returned when the model stored a plane exactly at
  `lead_h`; a neighbouring lead is never substituted.

  Returns None for negative leads, leads beyond the run, and (for state
  variables) leads the model did not store.
  """
  leads = info["lead_hours"]
  if not leads or lead_h < 0 or lead_h > leads[-1]:
    return None
  if is_rate:
    target = lead_h if lead_h > 0 else 1e-6
    for i, lead in enumerate(leads):
      if lead >= target:
        return i
    return None
  for i, lead in enumerate(leads):
    if abs(float(lead) - float(lead_h)) < 1e-6:
      return i
  return None


def rate_file_steps(info: Mapping[str, Any], lead_h: float) -> List[int]:
  """Returns the rain-rate plane indices averaged for the step at `lead_h`."""
  first = file_step_for_lead(info, lead_h, is_rate=True)
  if first is None:
    return []
  if lead_h <= 0:
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
  """Computes nearest-cell row and column indices on a global regular grid.

  Exact half-cell positions round towards the next cell (`floor(x + 0.5)`),
  never with banker's rounding.

  Raises:
    ValueError: If latitudes are outside `[-90, 90]` or longitudes outside
      `[-180, 360]`.
  """
  lat_arr = np.asarray(lats, dtype=np.float64)
  lon_arr = np.asarray(lons, dtype=np.float64)
  _validate_lat_lon(lat_arr, lon_arr)
  rows = np.clip(
      np.floor((90.0 - lat_arr) / grid_deg + 0.5), 0, n_lat - 1
  ).astype(np.intp)
  cols = (
      np.floor(((lon_arr + 180.0) % 360.0) / grid_deg + 0.5).astype(np.intp)
      % n_lon
  )
  return rows, cols


def bilinear_sample_grid(
    plane: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    grid_deg: float = GRID_DEG,
    min_valid_weight: float = MIN_VALID_COVERAGE_FRACTION,
) -> np.ndarray:
  """Bilinearly samples a global `(lat, lon)` plane on the `lats` x `lons` grid.

  Returns an array of shape `(len(lats), len(lons))`. Longitudes wrap across
  the antimeridian and rows are clamped at the poles. `NaN` neighbours are
  excluded from the weighted mean; a sample is `NaN` unless at least
  `min_valid_weight` (default 80%) of its bilinear weight comes from finite
  cells, so a valid grid node next to a masked cell keeps its value while
  points inside a masked region stay `NaN`.

  Raises:
    ValueError: If latitudes are outside `[-90, 90]` or longitudes outside
      `[-180, 360]`.
  """
  n_lat, n_lon = plane.shape
  lat_arr = np.asarray(lats, dtype=np.float64)
  lon_arr = np.asarray(lons, dtype=np.float64)
  _validate_lat_lon(lat_arr, lon_arr)

  row_f = np.clip((90.0 - lat_arr) / grid_deg, 0.0, float(n_lat - 1))
  col_f = ((lon_arr + 180.0) % 360.0) / grid_deg

  r0 = np.floor(row_f).astype(np.intp)
  r1 = np.minimum(r0 + 1, n_lat - 1)
  dr = (row_f - r0).astype(np.float32)[:, None]

  c0 = np.floor(col_f).astype(np.intp) % n_lon
  c1 = (c0 + 1) % n_lon
  dc = (col_f - np.floor(col_f)).astype(np.float32)[None, :]

  corners = (
      (plane[np.ix_(r0, c0)].astype(np.float32), (1.0 - dr) * (1.0 - dc)),
      (plane[np.ix_(r0, c1)].astype(np.float32), (1.0 - dr) * dc),
      (plane[np.ix_(r1, c0)].astype(np.float32), dr * (1.0 - dc)),
      (plane[np.ix_(r1, c1)].astype(np.float32), dr * dc),
  )
  out_shape = (lat_arr.size, lon_arr.size)
  numerator = np.zeros(out_shape, dtype=np.float32)
  valid_weight = np.zeros(out_shape, dtype=np.float32)
  for values, weight in corners:
    finite = np.isfinite(values)
    full_weight = np.broadcast_to(weight, out_shape)
    numerator += np.where(
        finite, full_weight * np.where(finite, values, 0.0), 0.0
    )
    valid_weight += np.where(finite, full_weight, 0.0)

  enough = valid_weight >= (min_valid_weight - 1e-6)
  return np.where(
      enough, numerator / np.maximum(valid_weight, 1e-12), np.nan
  ).astype(np.float32)


def sample_stream_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    stream_id: str,
    file_step: int,
    lats: np.ndarray,
    lons: np.ndarray,
    bilinear: bool = False,
) -> np.ndarray:
  """Samples one stored plane on a `(lats, lons)` grid in physical units."""
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
  """Computes the mean rain rate over stored planes on a `(lats, lons)` grid.

  A cell that is `NaN` in any of the planes is `NaN` in the mean.
  """
  if not file_steps:
    raise ValueError("file_steps must be non-empty.")
  sampled = [
      sample_stream_grid(
          arrays, stream_info, stream_id, step, lats, lons, bilinear=bilinear
      )
      for step in file_steps
  ]
  if len(sampled) == 1:
    return sampled[0]
  return np.mean(np.stack(sampled, axis=0), axis=0).astype(np.float32)


def _accum_cache_base_key(
    info: Mapping[str, Any], n_steps: int
) -> Tuple[str, Optional[str], int, int]:
  return (
      str(info.get("file") or ""),
      info.get("init_time"),
      int(info.get("mtime_ns") or 0),
      int(n_steps),
  )


def compute_accumulated_precip_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    lead_h: float,
) -> Optional[Tuple[np.ndarray, float]]:
  """Computes rain (mm) accumulated from forecast start to lead_h, globally.

  Returns `(grid, grid_deg)` where `grid` is a read-only float32 array, or
  `None` when `lead_h` lies beyond the run. Cells that are `NaN` in any stored
  rate plane up to `lead_h` are `NaN` (including at lead 0 for cells outside a
  regional model's domain).

  Raises:
    FileNotFoundError: If the model's precipitation stream is not loaded.
  """
  stream_id = f"{model_key}_precip"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Precipitation stream {stream_id!r} not found in synced data"
        " directory."
    )
  leads = info["lead_hours"]
  if lead_h < 0 or lead_h > leads[-1]:
    return None
  k = max(i for i, lead in enumerate(leads) if lead <= lead_h)
  arr_obj = arrays[stream_id]
  n_steps = int(arr_obj.shape[0])
  base_key = _accum_cache_base_key(info, n_steps)
  cache_key = base_key + (k,)

  start_k = 0
  start_total: Optional[np.ndarray] = None
  with _ACCUM_GRID_LOCK:
    cached = _ACCUM_GRID_CACHE.get(cache_key)
    if cached is not None:
      _ACCUM_GRID_CACHE.move_to_end(cache_key)
      return cached, GRID_DEG
    for key, c_total in _ACCUM_GRID_CACHE.items():
      if key[:4] == base_key and start_k <= key[4] < k:
        start_k = key[4]
        start_total = c_total

  if start_total is None:
    domain_plane = arr_obj[1] if n_steps > 1 else arr_obj[0]
    total = np.where(
        np.isfinite(domain_plane.astype(np.float32)), 0.0, np.nan
    ).astype(np.float32)
  else:
    total = start_total.copy()
  for i in range(start_k + 1, k + 1):
    total += arr_obj[i].astype(np.float32) * np.float32(leads[i] - leads[i - 1])
  total.flags.writeable = False

  with _ACCUM_GRID_LOCK:
    _ACCUM_GRID_CACHE[cache_key] = total
    _ACCUM_GRID_CACHE.move_to_end(cache_key)
    while len(_ACCUM_GRID_CACHE) > _ACCUM_GRID_CACHE_SIZE:
      _ACCUM_GRID_CACHE.popitem(last=False)
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

  `step_idx` counts 3-hour viewer steps (`lead = 3 * step_idx` hours). If
  `lats` and `lons` are omitted, returns the native `721 x 1440` (`0.25 deg`)
  global grid. Returns `None` when the lead is beyond the run or, for
  temperature and pressure, when the model did not store that exact lead.
  Masked cells are `NaN`.

  Raises:
    FileNotFoundError: If the requested model/variable stream is not synced.
    ValueError: If `model_key`, `var_key`, `step_idx`, or the coordinates are
      invalid.
  """
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  lead_h = _validate_step_idx(step_idx) * STEP_HOURS
  if lats is None:
    lats = GRID_LATS
  if lons is None:
    lons = GRID_LONS

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
  """Computes wind speed (m/s) and meteorological direction in `[0, 360)`."""
  u_arr = np.asarray(u, dtype=np.float64)
  v_arr = np.asarray(v, dtype=np.float64)
  speed = np.sqrt(u_arr**2 + v_arr**2)
  direction = np.mod(np.degrees(np.arctan2(-u_arr, -v_arr)) + 360.0, 360.0)
  return speed, direction


def _resolve_base_time(
    stream_info: Mapping[str, Mapping[str, Any]], model_key: str
) -> datetime.datetime:
  """Returns the forecast issue time recorded in the run metadata.

  Raises:
    ValueError: If no synced stream of `model_key` records an init time.
  """
  init_time = get_model_data_info_from_streams(stream_info, model_key).get(
      "init_time"
  )
  if not init_time:
    raise ValueError(
        f"No run init time is recorded for '{model_key}'; valid times cannot"
        " be computed."
    )
  return _parse_init_time(init_time)


def fetch_wind_grid(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    step_idx: int = 0,
    subsample: int = 2,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    bilinear: bool = False,
) -> Dict[str, Any]:
  """Fetches 10 m U/V wind components on a coarse grid plus grid metadata.

  The grid spacing is `1 deg * subsample` (`subsample` in `1..4`). `bbox` is
  `(min_lon, min_lat, max_lon, max_lat)`; a box whose `min_lon` is greater than
  `max_lon` crosses the antimeridian and is unwrapped eastwards, so `lo1` in the
  header may exceed 180. Masked cells are returned as `None`.

  Raises:
    FileNotFoundError: If U10 or V10 wind streams are not available in `arrays`.
    ValueError: If `model_key`, `step_idx`, `subsample`, or `bbox` is invalid,
      or if the model did not store wind at the requested lead.
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

  if (
      isinstance(subsample, bool)
      or int(subsample) != subsample
      or not 1 <= int(subsample) <= WIND_MAX_SUBSAMPLE
  ):
    raise ValueError(
        f"subsample must be an integer in [1, {WIND_MAX_SUBSAMPLE}], got"
        f" {subsample!r}."
    )
  step_deg = WIND_BASE_STEP_DEG * int(subsample)
  lead_h = _validate_step_idx(step_idx) * STEP_HOURS

  fu = file_step_for_lead(u_info, lead_h, is_rate=False)
  fv = file_step_for_lead(v_info, lead_h, is_rate=False)
  if fu is None or fv is None:
    raise ValueError(
        f"Lead hour {lead_h}h is not stored for '{model_key}' wind (stored"
        f" leads: {u_info['lead_hours']})."
    )

  if bbox is not None:
    min_lon, min_lat, max_lon, max_lat = (float(b) for b in bbox)
    if not all(math.isfinite(b) for b in (min_lon, min_lat, max_lon, max_lat)):
      raise ValueError(f"bbox must be finite, got {bbox!r}.")
    if min_lat > max_lat:
      raise ValueError(f"bbox min_lat exceeds max_lat: {bbox!r}.")
    la1 = min(90.0, max(-90.0, max_lat))
    la2 = min(90.0, max(-90.0, min_lat))
    lo1, lo2 = min_lon, max_lon
    if lo2 < lo1:
      lo2 += 360.0
    if lo2 - lo1 > 360.0:
      raise ValueError(
          f"bbox spans more than 360 degrees of longitude: {bbox!r}."
      )
    ny = max(1, int(round((la1 - la2) / step_deg)) + 1)
    nx = max(1, int(round((lo2 - lo1) / step_deg)) + 1)
    lats = np.clip(
        la1 - step_deg * np.arange(ny, dtype=np.float64), -90.0, 90.0
    )
    lons_header = lo1 + step_deg * np.arange(nx, dtype=np.float64)
  else:
    la1 = 90.0
    lo1 = -180.0
    lats = 90.0 - step_deg * np.arange(
        int(180 / step_deg) + 1, dtype=np.float64
    )
    lons_header = -180.0 + step_deg * np.arange(
        int(360 / step_deg), dtype=np.float64
    )
  lons_sample = ((lons_header + 180.0) % 360.0) - 180.0

  u = sample_stream_grid(
      arrays, stream_info, u_stream, fu, lats, lons_sample, bilinear=bilinear
  )
  v = sample_stream_grid(
      arrays, stream_info, v_stream, fv, lats, lons_sample, bilinear=bilinear
  )
  valid_dt = _resolve_base_time(stream_info, model_key) + datetime.timedelta(
      hours=lead_h
  )

  def _to_json_list(values: np.ndarray) -> List[Optional[float]]:
    return [
        round(float(x), 2) if math.isfinite(x) else None
        for x in values.astype(np.float64).ravel().tolist()
    ]

  u_list = _to_json_list(u)
  return {
      "header": {
          "model": model_key,
          "step_hours": lead_h,
          "valid_time": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
          "nx": int(len(lons_header)),
          "ny": int(len(lats)),
          "lo1": float(lo1),
          "la1": float(la1),
          "dx": step_deg,
          "dy": step_deg,
          "data_source": "archived_run",
          "missing_count": sum(1 for x in u_list if x is None),
      },
      "u": u_list,
      "v": _to_json_list(v),
  }


def round_or_none(value: Optional[float], digits: int) -> Optional[float]:
  """Rounds finite floats to `digits` decimals; None for None or non-finite."""
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
  """Extracts one nearest-cell forecast value at `(lat, lon)`, or None.

  Returns None when the lead is beyond the run, when a state variable was not
  stored at exactly `lead_h`, or when the cell is masked (`NaN`).
  """
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
    values = arrays[stream_id][rate_steps, rows[0], cols[0]].astype(np.float64)
    value = float(values.mean())
  else:
    file_step = file_step_for_lead(info, lead_h, is_rate=False)
    if file_step is None:
      return None
    value = float(arrays[stream_id][file_step, rows[0], cols[0]]) + float(
        info.get("offset", 0.0)
    )
  return value if math.isfinite(value) else None


def _cumulative_totals(rates: np.ndarray, leads: Sequence[int]) -> np.ndarray:
  """Cumulative rain (mm) per stored lead from interval-mean rates (mm/h).

  `rates[i]` is the mean rate over `(leads[i-1], leads[i]]`. A `NaN` rate makes
  every later total `NaN`; a cell that is `NaN` in its first interval is
  treated as outside the model domain, so even its lead-0 total is `NaN`.
  """
  n = len(leads)
  totals = np.empty(n, dtype=np.float64)
  domain_rate = rates[1] if n > 1 else rates[0]
  totals[0] = 0.0 if np.isfinite(domain_rate) else np.nan
  for i in range(1, n):
    totals[i] = totals[i - 1] + float(rates[i]) * float(leads[i] - leads[i - 1])
  return totals


def extract_accumulation_series(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    lat: float,
    lon: float,
    lead_hours: Sequence[float],
) -> List[Optional[float]]:
  """Computes cumulative rain (mm) since forecast start at `(lat, lon)`.

  Returns one value per requested lead: None beyond the run or where the cell
  is masked; otherwise the total up to the last stored lead `<= lead_h`.
  """
  stream_id = f"{model_key}_precip"
  info = stream_info.get(stream_id)
  if not info or stream_id not in arrays:
    raise FileNotFoundError(
        f"Precipitation stream '{stream_id}' not found in synced data"
        " directory."
    )
  rows, cols = grid_indices(
      np.array([lat], dtype=np.float64), np.array([lon], dtype=np.float64)
  )
  rates = arrays[stream_id][:, rows[0], cols[0]].astype(np.float64)
  leads = info["lead_hours"]
  totals = _cumulative_totals(rates, leads)
  out: List[Optional[float]] = []
  for lead_h in lead_hours:
    if lead_h < 0 or lead_h > leads[-1]:
      out.append(None)
      continue
    idx = max(i for i, lead in enumerate(leads) if lead <= lead_h)
    out.append(float(totals[idx]) if math.isfinite(totals[idx]) else None)
  return out


def fetch_point_timeseries(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    lat: float,
    lon: float,
    models: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
  """Fetches multi-model meteogram series at `(lat, lon)` on 3-hourly leads.

  Each model entry reports `stored_lead_hours`; curves hold None at leads the
  model did not store, beyond its horizon, or where the cell is masked.

  Args:
    arrays: Loaded stream arrays.
    stream_info: Loaded stream metadata.
    lat: Latitude in degrees `[-90, 90]`.
    lon: Longitude in degrees `[-180, 360]`.
    models: Model keys to include; defaults to every model with a synced
      precipitation stream.
    strict: If True, every requested model must have a synced precipitation
      stream (`FileNotFoundError` otherwise). If False, unsynced requested
      models are reported with `data_source="unavailable"` and None curves.

  Raises:
    FileNotFoundError: If no requested model is synced (always), or if a
      requested model lacks precipitation (strict mode).
    ValueError: If a model key or coordinate is invalid.
  """
  _validate_lat_lon(
      np.array([lat], dtype=np.float64), np.array([lon], dtype=np.float64)
  )
  lead_hours = [i * STEP_HOURS for i in range(NUM_STEPS)]
  now_utc = datetime.datetime.now(datetime.timezone.utc).replace(
      minute=0, second=0, microsecond=0
  )

  if models is not None:
    target_models = list(models)
  else:
    target_models = [
        m for m in SUPPORTED_MODELS if f"{m}_precip" in stream_info
    ]
  for m_key in target_models:
    if m_key not in SUPPORTED_MODELS:
      raise ValueError(f"Unknown weather model '{m_key}'")
  if not target_models or not any(
      f"{m}_precip" in stream_info for m in target_models
  ):
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
    m_info = SUPPORTED_MODELS[m_key]
    data_info = get_model_data_info_from_streams(stream_info, m_key)

    has_precip = (
        f"{m_key}_precip" in stream_info and f"{m_key}_precip" in arrays
    )
    has_temp = f"{m_key}_temp" in stream_info and f"{m_key}_temp" in arrays
    if strict and not has_precip:
      raise FileNotFoundError(
          f"Precipitation stream for '{m_key}' not found in synced data"
          " directory."
      )
    has_wind = all(
        f"{m_key}_{c}" in stream_info and f"{m_key}_{c}" in arrays
        for c in ("u10", "v10")
    )
    has_mslp = f"{m_key}_mslp" in stream_info and f"{m_key}_mslp" in arrays

    if has_precip:
      accum = extract_accumulation_series(
          arrays, stream_info, m_key, lat, lon, lead_hours
      )
    else:
      accum = [None] * len(lead_hours)

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
          if has_precip
          else None
      )
      temp_curve.append(
          round_or_none(
              extract_point_value(
                  arrays, stream_info, m_key, "temperature", lead_h, lat, lon
              ),
              1,
          )
          if has_temp
          else None
      )
      u_w = v_w = None
      if has_wind:
        u_w = extract_point_value(
            arrays, stream_info, m_key, "wind_u", lead_h, lat, lon
        )
        v_w = extract_point_value(
            arrays, stream_info, m_key, "wind_v", lead_h, lat, lon
        )
      if u_w is not None and v_w is not None:
        speed, direction = compute_wind_speed_and_direction(
            np.array([u_w]), np.array([v_w])
        )
        wind_spd_curve.append(round(float(speed[0]), 1))
        wind_dir_curve.append(int(direction[0]) % 360)
      else:
        wind_spd_curve.append(None)
        wind_dir_curve.append(None)
      pressure_curve.append(
          round_or_none(
              extract_point_value(
                  arrays, stream_info, m_key, "pressure", lead_h, lat, lon
              ),
              1,
          )
          if has_mslp
          else None
      )

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


def geometry_points(geom: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
  """Extracts exterior-ring `(lats, lons)` vertices from a GeoJSON geometry."""
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


def geojson_polygon_to_shapely(
    geometry: Mapping[str, Any],
) -> shapely.geometry.base.BaseGeometry:
  """Converts a GeoJSON Polygon/MultiPolygon mapping to a valid Shapely shape.

  Raises:
    ValueError: If the geometry is missing, not a Polygon/MultiPolygon, empty,
      invalid, outside `[-180, 180] x [-90, 90]`, or crosses the antimeridian
      (longitude span above 180 degrees; split such polygons first).
  """
  if not geometry or geometry.get("type") not in ("Polygon", "MultiPolygon"):
    raise ValueError(
        "Catchment geometry must be a GeoJSON Polygon or MultiPolygon, got"
        f" {geometry.get('type') if geometry else None!r}."
    )
  if not geometry.get("coordinates"):
    raise ValueError("Catchment geometry has no coordinates.")
  polygon = shapely.geometry.shape(dict(geometry))
  if polygon.is_empty:
    raise ValueError("Catchment geometry is empty.")
  if not polygon.is_valid:
    raise ValueError(
        "Catchment geometry is not a valid polygon:"
        f" {shapely.validation.explain_validity(polygon)}."
    )
  minx, miny, maxx, maxy = polygon.bounds
  if minx < -180.0 or maxx > 180.0 or miny < -90.0 or maxy > 90.0:
    raise ValueError(
        "Catchment geometry must use EPSG:4326 degrees within"
        f" [-180, 180] x [-90, 90]; bounds are {polygon.bounds}."
    )
  if maxx - minx > 180.0:
    raise ValueError(
        "Catchment geometry spans more than 180 degrees of longitude; polygons"
        " crossing the antimeridian must be split at +/-180 first."
    )
  return polygon


def _basin_cell_values(
    arrays: Mapping[str, np.ndarray],
    stream_id: str,
    file_steps: Sequence[int],
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    offset: float,
) -> np.ndarray:
  """Returns per-cell values (mean over `file_steps`) at the basin cells."""
  stacked = np.stack(
      [
          arrays[stream_id][step][lat_idx, lon_idx].astype(np.float64)
          for step in file_steps
      ],
      axis=0,
  )
  return stacked.mean(axis=0) + offset


def fetch_catchment_summary(
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    geojson_feature: Mapping[str, Any],
    step_idx: int,
    model_key: str,
) -> Dict[str, Any]:
  """Computes exact area-weighted basin statistics for a catchment polygon.

  Grid cells are weighted by their intersection area with the polygon (holes
  excluded) times `cos(latitude)`. A statistic is None when less than 80% of
  the basin area has finite data, when the lead is not stored, or when the
  lead lies beyond the run.

  Args:
    arrays: Loaded stream arrays.
    stream_info: Loaded stream metadata.
    geojson_feature: GeoJSON Feature with a Polygon/MultiPolygon `geometry`,
      an `id` or `properties.catchment_id`, and optionally
      `properties.area_km2` (otherwise the WGS84 geodesic area is reported).
    step_idx: 3-hour viewer step (`lead = 3 * step_idx` hours).
    model_key: Supported model key.

  Raises:
    FileNotFoundError: If the model's precipitation stream is not synced.
    KeyError: If the feature has neither `id` nor `properties.catchment_id`.
    ValueError: If `model_key`, `step_idx`, the geometry, or `area_km2` is
      invalid.
  """
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  precip_stream = f"{model_key}_precip"
  if precip_stream not in stream_info or precip_stream not in arrays:
    raise FileNotFoundError(
        f"Synced precipitation stream for '{model_key}' not found."
    )
  lead_h = _validate_step_idx(step_idx) * STEP_HOURS

  props = geojson_feature.get("properties") or {}
  catchment_id = props.get("catchment_id") or geojson_feature.get("id")
  if catchment_id is None or str(catchment_id).strip() == "":
    raise KeyError(
        "Catchment feature needs an 'id' or 'properties.catchment_id'."
    )
  polygon = geojson_polygon_to_shapely(geojson_feature.get("geometry") or {})
  if "area_km2" in props:
    area_km2 = float(props["area_km2"])
    if not math.isfinite(area_km2) or area_km2 <= 0.0:
      raise ValueError(
          "properties.area_km2 must be a positive number, got"
          f" {props['area_km2']!r}."
      )
    area_source = "properties"
  else:
    area_km2 = geodesic_area_km2(polygon)
    area_source = "geometry"

  calculator = ZonalWeightCalculator(
      GRID_LATS, GRID_LONS, cell_res_lat=GRID_DEG, cell_res_lon=GRID_DEG
  )
  lat_idx, lon_idx, weights = calculator.compute_weights(
      str(catchment_id), polygon
  )
  precip_info = stream_info[precip_stream]
  data_info = get_model_data_info_from_streams(stream_info, model_key)
  base_dt = _resolve_base_time(stream_info, model_key)
  valid_dt = base_dt + datetime.timedelta(hours=lead_h)

  mean_rate: float = float("nan")
  max_rate: float = float("nan")
  rate_missing = 1.0
  rate_steps = rate_file_steps(precip_info, lead_h)
  if rate_steps and len(weights):
    cell_rates = _basin_cell_values(
        arrays, precip_stream, rate_steps, lat_idx, lon_idx, 0.0
    )
    mean_rate, rate_missing = weighted_mean_valid_with_coverage(
        cell_rates, weights
    )
    if math.isfinite(mean_rate):
      max_rate = float(np.nanmax(cell_rates))

  mean_temp: float = float("nan")
  temp_missing = 1.0
  temp_stream = f"{model_key}_temp"
  if temp_stream in stream_info and temp_stream in arrays and len(weights):
    temp_info = stream_info[temp_stream]
    temp_step = file_step_for_lead(temp_info, lead_h, is_rate=False)
    if temp_step is not None:
      cell_temps = _basin_cell_values(
          arrays,
          temp_stream,
          [temp_step],
          lat_idx,
          lon_idx,
          float(temp_info.get("offset", 0.0)),
      )
      mean_temp, temp_missing = weighted_mean_valid_with_coverage(
          cell_temps, weights
      )

  window_h = int(data_info["max_lead_hours"] or 0)
  accum_mean: float = float("nan")
  accum_missing = 1.0
  accumulated = compute_accumulated_precip_grid(
      arrays, stream_info, model_key, window_h
  )
  if accumulated is not None and len(weights):
    accum_grid, _ = accumulated
    accum_mean, accum_missing = weighted_mean_valid_with_coverage(
        accum_grid[lat_idx, lon_idx].astype(np.float64), weights
    )

  centroid = polygon.centroid
  return {
      "catchment_id": catchment_id,
      "area_km2": round(area_km2, 1),
      "area_km2_source": area_source,
      "step_hours": lead_h,
      "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "basin_mean_precip_mmh": round_or_none(mean_rate, 2),
      "basin_max_precip_mmh": round_or_none(max_rate, 2),
      "basin_accumulated_10d_mm": round_or_none(accum_mean, 1),
      "basin_mean_temp_c": round_or_none(mean_temp, 1),
      "missing_area_fraction": {
          "precipitation": round(float(rate_missing), 3),
          "accumulated_precip": round(float(accum_missing), 3),
          "temperature": round(float(temp_missing), 3),
      },
      "grid_cells": int(len(weights)),
      "centroid": {
          "latitude": round(float(centroid.y), 4),
          "longitude": round(float(centroid.x), 4),
      },
      "model": model_key,
      "data_source": data_info["data_source"],
      "accumulation_hours": window_h,
  }


@dataclasses.dataclass(frozen=True)
class _StreamState:
  """Immutable snapshot of one loaded run (swapped atomically on reload)."""

  handles: Dict[str, StreamHandle]
  stream_info: Dict[str, Dict[str, Any]]
  arrays: Dict[str, np.ndarray]
  signature: Optional[Tuple[str, int]]
  directory: Optional[Path]


class WeatherDataFetcher:
  """Backend weather data fetcher for synced gridded NWP forecast runs.

  Requires an explicit `data_dir` and reads only the binary planes that
  `multimet.weather_fetcher.sync` wrote there. `reload_if_changed()` swaps in a
  newer run atomically: requests that are still reading the previous run keep
  their arrays, and the old memory maps are released once those requests
  finish. Call `close()` (or use the instance as a context manager) when done.
  """

  def __init__(self, data_dir: Union[str, Path]):
    self.data_dir: Path = require_data_dir(data_dir)
    self._lock = threading.RLock()
    self._state = _StreamState({}, {}, {}, None, None)
    self._load_streams()

  def __enter__(self) -> "WeatherDataFetcher":
    return self

  def __exit__(self, *exc_info: Any) -> None:
    self.close()

  @property
  def handles(self) -> Dict[str, StreamHandle]:
    """Memory-map handles of the currently loaded run."""
    return self._state.handles

  @property
  def stream_info(self) -> Dict[str, Dict[str, Any]]:
    """Stream metadata of the currently loaded run."""
    return self._state.stream_info

  @property
  def arrays(self) -> Dict[str, np.ndarray]:
    """Read-only `(step, lat, lon)` float16 arrays of the loaded run."""
    return self._state.arrays

  def snapshot(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Returns a consistent `(arrays, stream_info)` pair from one loaded run."""
    state = self._state
    return state.arrays, state.stream_info

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

  def _swap_state(self, new_state: _StreamState) -> _StreamState:
    """Installs `new_state` atomically and returns the state it replaced."""
    with self._lock:
      old = self._state
      self._state = new_state
      return old

  def _retire_state(self, state: _StreamState) -> int:
    """Releases a replaced run: evicts its cached totals, closes idle maps.

    `state` must be the only remaining reference to the retired run (callers
    pass the result of `_swap_state` directly) so that dropping it releases the
    run's NumPy views before the mappings are checked for external references.
    """
    files = [info["file"] for info in state.stream_info.values()]
    handles = state.handles
    del state
    _evict_accum_grid_cache(files)
    return close_unreferenced_mmaps(handles)

  def close(self) -> None:
    """Releases the loaded run.

    Memory maps that no NumPy view references any more are closed now; maps
    still referenced by in-flight readers are closed by the garbage collector
    when those readers finish.
    """
    self._retire_state(self._swap_state(_StreamState({}, {}, {}, None, None)))

  def _load_streams(self) -> None:
    with self._lock:
      target_dir = self._resolve_active_dir()
      handles, infos, arrays = scan_streams(target_dir)
      new_state = _StreamState(
          handles=handles,
          stream_info=infos,
          arrays=arrays,
          signature=self._compute_signature(target_dir),
          directory=target_dir.resolve() if target_dir.exists() else None,
      )
      del handles, infos, arrays
      self._retire_state(self._swap_state(new_state))

  def reload_if_changed(self) -> bool:
    """Reloads the streams if a newer run was swapped into `data_dir`."""
    with self._lock:
      target_dir = self._resolve_active_dir()
      sig = self._compute_signature(target_dir)
      if sig == self._state.signature:
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
    _, stream_info = self.snapshot()
    return [
        {**info, **get_model_data_info_from_streams(stream_info, key)}
        for key, info in SUPPORTED_MODELS.items()
    ]

  def get_sync_status(self) -> Dict[str, Any]:
    """Returns synchronization status from `<data_dir>/sync_status.json`.

    `sync_status_found` is False when no synchronizer has written a status
    file yet; the other fields are then None/"never".
    """
    status = read_sync_status(self.data_dir)
    found = bool(status)
    if not found:
      status = {
          "last_result": "never",
          "message": "No automatic update has run yet.",
      }
    loaded_dir = self._state.directory
    return {
        "last_check_utc": status.get("last_check_utc"),
        "last_success_utc": status.get("last_success_utc"),
        "last_result": status.get("last_result"),
        "message": status.get("message"),
        "check_interval_minutes": status.get("check_interval_minutes"),
        "sync_status_found": found,
        "auto_update": found,
        "data_dir": str(loaded_dir) if loaded_dir else None,
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
    """Fetches a 2D forecast grid for `(model_key, var_key, step_idx)`."""
    arrays, stream_info = self.snapshot()
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

  def fetch_wind_grid(
      self,
      model_key: str,
      step_idx: int = 0,
      subsample: int = 2,
      bbox: Optional[Tuple[float, float, float, float]] = None,
      bilinear: bool = False,
  ) -> Dict[str, Any]:
    """Fetches coarse 10 m U/V wind component arrays and grid metadata."""
    arrays, stream_info = self.snapshot()
    return fetch_wind_grid(
        arrays,
        stream_info,
        model_key=model_key,
        step_idx=step_idx,
        subsample=subsample,
        bbox=bbox,
        bilinear=bilinear,
    )

  def fetch_point_timeseries(
      self,
      lat: float,
      lon: float,
      models: Optional[Sequence[str]] = None,
      strict: bool = True,
  ) -> Dict[str, Any]:
    """Fetches multi-model meteogram time series at `(lat, lon)`."""
    arrays, stream_info = self.snapshot()
    return fetch_point_timeseries(
        arrays,
        stream_info,
        lat=lat,
        lon=lon,
        models=models,
        strict=strict,
    )

  def fetch_catchment_summary(
      self,
      geojson_feature: Mapping[str, Any],
      step_idx: int,
      model_key: str,
  ) -> Dict[str, Any]:
    """Computes area-weighted catchment precipitation/temperature statistics."""
    arrays, stream_info = self.snapshot()
    return fetch_catchment_summary(
        arrays,
        stream_info,
        geojson_feature=geojson_feature,
        step_idx=step_idx,
        model_key=model_key,
    )
