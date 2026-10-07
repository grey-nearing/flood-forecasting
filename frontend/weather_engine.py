"""Earthkit Hydro weather visualization UI adapter for the Weather Viewer tab.

Delegates core catalog definitions, binary stream scanning, and point/catchment
meteogram queries to `multimet.weather_fetcher`, and Web Mercator tile rendering,
indexed PNG animation frame encoding, colormaps, and wind vector extraction to
`frontend.weather_viewer`.
"""

from __future__ import annotations

import collections
import datetime
import json
import logging
import math
import mmap
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ws_root = str(Path(__file__).resolve().parents[1])
if _ws_root not in sys.path:
  sys.path.insert(0, _ws_root)

from frontend.weather_viewer.colormaps import (
    classify_values as _classify,
    colorize_indexed as _colorize_indexed,
    colorize_rgba as _colorize,
    encode_indexed_png as _encode_indexed_png,
    encode_rgba_png as _encode_rgba_png,
    make_png_bytes,
    PRESSURE_LEVELS as _PRESSURE_LEVELS,
    RAIN_ACCUM_CLASSES,
    RAIN_RATE_CLASSES,
    SUPPORTED_VARIABLES,
    TEMP_LEVELS as _TEMP_LEVELS,
)
from frontend.weather_viewer.tiles import (
    empty_frame as _empty_frame,
    frame_coordinates as _frame_coordinates,
    FRAME_SIZE,
    FRAME_VARIABLES,
    FRAME_VERSION,
    MERCATOR_MAX_LAT,
    tile_coordinates as _tile_coordinates,
    TILE_VERSION,
    transparent_tile as _transparent_tile,
)
from frontend.weather_viewer.wind import extract_wind_vectors
from multimet.weather_fetcher.cli import resolve_default_weather_data_dir
from multimet.weather_fetcher.config import (
    DEFAULT_MSLP_OFFSET_HPA,
    GRID_DEG,
    MAX_LEAD_HOURS,
    N_LAT,
    N_LON,
    NUM_STEPS,
    RUN_DATASET_TO_MODEL as _RUN_DATASET_TO_MODEL,
    run_lead_hours as _run_lead_hours,
    RUN_METADATA_FILE,
    STEP_HOURS,
    STREAM_FILES as _STREAM_FILES,
    STREAM_SUFFIX as _STREAM_SUFFIX,
    SUPPORTED_MODELS,
    SYNC_STATUS_FILE,
)
from multimet.weather_fetcher.fetcher import (
    extract_accumulation_series,
    extract_point_value,
    file_step_for_lead as _file_step_for_lead,
    geometry_points as _geometry_points,
    grid_indices as _grid_indices,
    rate_file_steps as _rate_file_steps,
    round_or_none as _round_or_none,
    scan_streams as _scan_streams,
)
from multimet.weather_fetcher.sync import load_run_metadata as _load_run_metadata

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "forecasts"


def _weather_data_root() -> Path:
  """Local-disk root written by weather_sync.sync_latest."""
  return resolve_default_weather_data_dir()


WEATHER_DATA_ROOT = _weather_data_root()
LOCAL_CURRENT_DIR = WEATHER_DATA_ROOT / "current"

CANDIDATE_DATA_DIRS = [
    LOCAL_CURRENT_DIR,
    DATA_DIR,
    Path(
        "/tmp/openhydronet"
        "experimental/users/hydro_user/alertnexus_frontend/data/forecasts"
    ),
]

MMAP_HANDLES: Dict[str, Tuple[mmap.mmap, int, int, int, bool]] = {}
STREAM_INFO: Dict[str, Dict[str, Any]] = {}
_ARRAYS: Dict[str, np.ndarray] = {}
_INITIALIZED = False
_INIT_LOCK = threading.Lock()
_LOADED_SIGNATURE: Optional[Tuple[str, int]] = None
_LOADED_DIR: Optional[Path] = None

_ACCUM_CACHE: "collections.OrderedDict[Tuple[str, int], np.ndarray]" = (
    collections.OrderedDict()
)
_ACCUM_CACHE_SIZE = 24
_ACCUM_LOCK = threading.Lock()
_SYNTHETIC_ACCUM_DEG = 1.0

_FRAME_CACHE: "collections.OrderedDict[Tuple[Any, ...], bytes]" = (
    collections.OrderedDict()
)
_FRAME_CACHE_LIMIT_BYTES = 384 << 20
_FRAME_CACHE_BYTES = 0
_FRAME_LOCK = threading.Lock()


def _find_data_dir() -> Optional[Path]:
  """First candidate directory that holds .bin streams."""
  for c_dir in CANDIDATE_DATA_DIRS:
    try:
      if c_dir.exists() and any(c_dir.glob("*.bin")):
        return c_dir
    except OSError:
      continue
  return None


def _dir_signature(target_dir: Optional[Path]) -> Optional[Tuple[str, int]]:
  if target_dir is None:
    return None
  resolved = target_dir.resolve()
  try:
    mtime = (resolved / RUN_METADATA_FILE).stat().st_mtime_ns
  except OSError:
    mtime = 0
  return (str(resolved), mtime)


def _clear_frame_cache() -> None:
  global _FRAME_CACHE_BYTES
  with _FRAME_LOCK:
    _FRAME_CACHE.clear()
    _FRAME_CACHE_BYTES = 0


def _install_streams(target_dir: Path) -> None:
  """Swaps in the streams of target_dir (caller holds _INIT_LOCK)."""
  global MMAP_HANDLES, STREAM_INFO, _ARRAYS, _LOADED_SIGNATURE, _LOADED_DIR
  handles, infos, arrays = _scan_streams(target_dir)
  MMAP_HANDLES, STREAM_INFO, _ARRAYS = handles, infos, arrays
  _LOADED_SIGNATURE = _dir_signature(target_dir)
  _LOADED_DIR = target_dir.resolve()
  with _ACCUM_LOCK:
    _ACCUM_CACHE.clear()
  _clear_frame_cache()
  if infos and os.environ.get("EARTHKIT_WEATHER_WARM_CACHE", "1") == "1":
    threading.Thread(
        target=_warm_caches,
        args=([info["file"] for info in infos.values()],),
        name="weather-cache-warmup",
        daemon=True,
    ).start()


def init_weather_streams() -> None:
  """Memory-maps the forecast streams found in the candidate directories."""
  global _INITIALIZED
  if _INITIALIZED:
    return
  with _INIT_LOCK:
    if _INITIALIZED:
      return
    target_dir = _find_data_dir()
    if not target_dir:
      target_dir = DATA_DIR
      target_dir.mkdir(parents=True, exist_ok=True)
    _install_streams(target_dir)
    _INITIALIZED = True


def reload_if_changed() -> bool:
  """Switches to a newer downloaded run if there is one; True if it did."""
  if not _INITIALIZED:
    init_weather_streams()
    return True
  target_dir = _find_data_dir()
  if target_dir is None or _dir_signature(target_dir) == _LOADED_SIGNATURE:
    return False
  with _INIT_LOCK:
    if _dir_signature(target_dir) == _LOADED_SIGNATURE:
      return False
    logger.info(
        "[WeatherEngine] Loading new forecast run from %s", target_dir.resolve()
    )
    _install_streams(target_dir)
  return True


def get_sync_status() -> Dict[str, Any]:
  """Last automatic check for new runs (written by weather_sync), plus what is loaded."""
  status: Dict[str, Any] = {}
  try:
    status = json.loads(
        (WEATHER_DATA_ROOT / SYNC_STATUS_FILE).read_text(encoding="utf-8")
    )
  except (OSError, ValueError):
    status = {
        "last_result": "never",
        "message": "No automatic update has run yet.",
    }
  loaded = _LOADED_DIR.resolve() if _LOADED_DIR else None
  return {
      "last_check_utc": status.get("last_check_utc"),
      "last_success_utc": status.get("last_success_utc"),
      "last_result": status.get("last_result"),
      "message": status.get("message"),
      "check_interval_minutes": status.get("check_interval_minutes", 60),
      "auto_update": loaded is not None
      and str(loaded).startswith(str(WEATHER_DATA_ROOT.resolve())),
      "data_dir": str(loaded) if loaded else None,
  }


def _warm_caches(paths: Sequence[str]) -> None:
  _warm_page_cache(paths)
  _prewarm_frames()


def _warm_page_cache(paths: Sequence[str]) -> None:
  """Reads each file once so later random access hits the OS page cache."""
  for path in paths:
    try:
      with open(path, "rb") as f_handle:
        while f_handle.read(8 << 20):
          pass
    except OSError as e:
      logger.debug("Page-cache warm-up skipped %s: %s", path, e)


def _mean_rate_grid(
    stream_id: str,
    file_steps: Sequence[int],
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
  """Mean rain rate over one or more stored planes on a lat x lon grid."""
  if len(file_steps) == 1:
    return _grid_values(stream_id, file_steps[0], lats, lons)
  rows, cols = _grid_indices(lats, lons)
  values = _ARRAYS[stream_id][np.ix_(list(file_steps), rows, cols)]
  return np.nan_to_num(values.astype(np.float32), nan=0.0).mean(axis=0)


def _grid_values(
    stream_id: str, file_step: int, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
  """Nearest-neighbour values of one stored plane on a lat x lon grid."""
  rows, cols = _grid_indices(lats, lons)
  plane = _ARRAYS[stream_id][file_step]
  values = plane[np.ix_(rows, cols)].astype(np.float32)
  offset = STREAM_INFO.get(stream_id, {}).get("offset", 0.0)
  return values + offset if offset else values


def _accumulated_global(
    model_key: str, lead_h: float
) -> Optional[Tuple[np.ndarray, float]]:
  """Rain (mm) accumulated from the forecast start to lead_h on a global grid."""
  stream_id = f"{model_key}_precip"
  info = STREAM_INFO.get(stream_id)
  if info:
    leads = info["lead_hours"]
    if lead_h > leads[-1]:
      return None
    k = max(i for i, lead in enumerate(leads) if lead <= lead_h)
    key = (stream_id, k)
    res = GRID_DEG
    shape = (N_LAT, N_LON)

    def increment(i: int) -> np.ndarray:
      rate = _ARRAYS[stream_id][i].astype(np.float32)
      return np.nan_to_num(rate, nan=0.0) * float(leads[i] - leads[i - 1])

  else:
    k = int(lead_h // STEP_HOURS)
    key = (f"{model_key}_synthetic", k)
    res = _SYNTHETIC_ACCUM_DEG
    grid_lats = 90.0 - np.arange(int(180 / res) + 1) * res
    grid_lons = -180.0 + np.arange(int(360 / res)) * res
    shape = (len(grid_lats), len(grid_lons))

    def increment(i: int) -> np.ndarray:
      return (
          _procedural_field("precipitation", model_key, i, grid_lats, grid_lons)
          * STEP_HOURS
      )

  with _ACCUM_LOCK:
    cached = _ACCUM_CACHE.get(key)
    if cached is not None:
      _ACCUM_CACHE.move_to_end(key)
      return cached, res
    start_k, start_total = 0, None
    for (cache_id, cache_k), total in _ACCUM_CACHE.items():
      if cache_id == key[0] and start_k <= cache_k < k:
        start_k, start_total = cache_k, total
    total = (
        np.zeros(shape, np.float32)
        if start_total is None
        else start_total.copy()
    )
    for i in range(start_k + 1, k + 1):
      total += increment(i)
    _ACCUM_CACHE[key] = total
    while len(_ACCUM_CACHE) > _ACCUM_CACHE_SIZE:
      _ACCUM_CACHE.popitem(last=False)
    return total, res


def _sample_global(
    grid: np.ndarray, res: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
  n_lat, n_lon = grid.shape
  rows = np.clip(np.rint((90.0 - lats) / res), 0, n_lat - 1).astype(np.intp)
  cols = np.rint(((lons + 180.0) % 360.0) / res).astype(np.intp) % n_lon
  return grid[np.ix_(rows, cols)]


def _procedural_sample(
    var_key: str, model_key: str, step_idx: int, lat_deg: float, lon_deg: float
) -> float:
  """Fallback UI visualization sample when local binaries are not synced."""
  lead_h = step_idx * 3
  lat_r = math.radians(lat_deg)
  lon_r = math.radians(lon_deg)
  s_lat = math.sin(lat_r)
  c_lat = math.cos(lat_r)
  wave = (
      math.sin(3 * lon_r + lead_h * 0.05) * 4.5 * math.cos(lat_r * 2.0)
      + math.sin(5 * lon_r - lead_h * 0.08) * 2.8 * s_lat
  )

  if var_key in ["precipitation", "accumulated_precip"]:
    itcz = abs(lat_deg - (4.0 * math.sin(lon_r * 2.0 + lead_h * 0.02)))
    storm = abs(
        abs(lat_deg) - 45.0 + 5.0 * math.sin(4 * lon_r + lead_h * 0.04)
    )
    p = 0.0
    if itcz < 8.0:
      p += (8.0 - itcz) * 0.95 * max(0.0, math.sin(lon_r * 4.0 + lead_h * 0.1))
    if storm < 10.0:
      p += (
          (10.0 - storm)
          * 0.75
          * max(0.0, math.cos(lon_r * 3.0 + lead_h * 0.06))
      )
    if "aifs" in model_key:
      p = max(0.0, p * 1.04 - 0.04)
    elif "graphcast" in model_key:
      p = max(0.0, p * 0.98 + (0.1 if p > 0.5 else 0.0))
    elif "gfs" in model_key:
      p = max(0.0, p * 1.08)
    if var_key == "accumulated_precip":
      return p * min(lead_h + 1, 24.0) * 0.4
    return p

  if var_key == "temperature":
    b_t = 28.0 * c_lat - 36.0 * (s_lat**2)
    if lat_deg > 70:
      b_t -= 16.0
    elif lat_deg < -60:
      b_t -= 26.0
    solar_hour = (12 + lead_h + int(lon_deg / 15.0)) % 24
    diurnal = math.sin((solar_hour - 8.0) * math.pi / 12.0) * 5.0 * c_lat
    val = b_t + wave + diurnal
    if "aifs" in model_key:
      val += 0.2 * math.sin(lon_r * 4.0)
    elif "graphcast" in model_key:
      val -= 0.15 * math.cos(lat_r * 3.0)
    return val

  if var_key == "wind_u":
    u_base = -7.0 * math.cos(lat_r * 3.0) + 12.0 * math.exp(
        -(((abs(lat_deg) - 48.0) / 14.0) ** 2)
    )
    return u_base + wave * 0.8

  if var_key == "wind_v":
    v_base = 2.0 * math.sin(lat_r * 2.0)
    return v_base + math.cos(3 * lon_r + lead_h * 0.05) * 4.2 * s_lat

  if var_key == "pressure":
    p_base = 1013.25 + 6.0 * math.cos(lat_r * 4.0) - 8.0 * math.exp(
        -(((abs(lat_deg) - 60.0) / 15.0) ** 2)
    )
    return p_base - wave * 1.5

  return 0.0


def _procedural_field(
    var_key: str,
    model_key: str,
    step_idx: int,
    lats: Sequence[float],
    lons: Sequence[float],
) -> np.ndarray:
  """Vectorized fallback field on a lat x lon grid for offline UI previews."""
  lead_h = step_idx * 3
  lat = np.asarray(lats, dtype=np.float64)[:, None]
  lon = np.asarray(lons, dtype=np.float64)[None, :]
  shape = (lat.shape[0], lon.shape[1])
  lat_r = np.radians(lat)
  lon_r = np.radians(lon)
  s_lat = np.sin(lat_r)
  c_lat = np.cos(lat_r)
  wave = (
      np.sin(3 * lon_r + lead_h * 0.05) * 4.5 * np.cos(lat_r * 2.0)
      + np.sin(5 * lon_r - lead_h * 0.08) * 2.8 * s_lat
  )

  if var_key in ("precipitation", "accumulated_precip"):
    itcz = np.abs(lat - 4.0 * np.sin(lon_r * 2.0 + lead_h * 0.02))
    storm = np.abs(
        np.abs(lat) - 45.0 + 5.0 * np.sin(4 * lon_r + lead_h * 0.04)
    )
    p = np.where(
        itcz < 8.0,
        (8.0 - itcz) * 0.95 * np.maximum(0.0, np.sin(lon_r * 4.0 + lead_h * 0.1)),
        0.0,
    ) + np.where(
        storm < 10.0,
        (10.0 - storm)
        * 0.75
        * np.maximum(0.0, np.cos(lon_r * 3.0 + lead_h * 0.06)),
        0.0,
    )
    if "aifs" in model_key:
      p = np.maximum(0.0, p * 1.04 - 0.04)
    elif "graphcast" in model_key:
      p = np.maximum(0.0, p * 0.98 + np.where(p > 0.5, 0.1, 0.0))
    elif "gfs" in model_key:
      p = np.maximum(0.0, p * 1.08)
    result = p
  elif var_key == "temperature":
    b_t = 28.0 * c_lat - 36.0 * s_lat**2
    b_t = np.where(lat > 70, b_t - 16.0, np.where(lat < -60, b_t - 26.0, b_t))
    solar_hour = np.mod(12 + lead_h + np.trunc(lon / 15.0), 24)
    diurnal = np.sin((solar_hour - 8.0) * np.pi / 12.0) * 5.0 * c_lat
    result = b_t + wave + diurnal
    if "aifs" in model_key:
      result = result + 0.2 * np.sin(lon_r * 4.0)
    elif "graphcast" in model_key:
      result = result - 0.15 * np.cos(lat_r * 3.0)
  elif var_key == "wind_u":
    u_base = -7.0 * np.cos(lat_r * 3.0) + 12.0 * np.exp(
        -(((np.abs(lat) - 48.0) / 14.0) ** 2)
    )
    result = u_base + wave * 0.8
  elif var_key == "wind_v":
    result = 2.0 * np.sin(lat_r * 2.0) + np.cos(
        3 * lon_r + lead_h * 0.05
    ) * 4.2 * s_lat
  elif var_key == "pressure":
    p_base = 1013.25 + 6.0 * np.cos(lat_r * 4.0) - 8.0 * np.exp(
        -(((np.abs(lat) - 60.0) / 15.0) ** 2)
    )
    result = p_base - wave * 1.5
  else:
    result = np.zeros(shape)
  return np.broadcast_to(result, shape).astype(np.float32)


def get_model_data_info(model_key: str) -> Dict[str, Any]:
  """Says whether a model's fields come from an archived run or are synthetic."""
  init_weather_streams()
  streams = STREAM_INFO
  real_variables: List[str] = []
  init_time = None
  downloaded_utc = None
  title = None
  max_lead = MAX_LEAD_HOURS
  for var_key, suffix in (("precipitation", "precip"), ("temperature", "temp")):
    info = streams.get(f"{model_key}_{suffix}")
    if info and info["archived_run"]:
      real_variables.append(var_key)
      init_time = info["init_time"]
      downloaded_utc = info.get("downloaded_utc")
      title = info.get("title")
      max_lead = min(max_lead, info["lead_hours"][-1])
  if "precipitation" in real_variables:
    real_variables.insert(1, "accumulated_precip")
  if real_variables:
    mslp = streams.get(f"{model_key}_mslp")
    if mslp and mslp["archived_run"]:
      real_variables.append("pressure")
    if all(
        (streams.get(f"{model_key}_{c}") or {}).get("archived_run")
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
      "data_source": "archived_run" if real_variables else "synthetic",
      "init_time": init_time,
      "max_lead_hours": max_lead if real_variables else MAX_LEAD_HOURS,
      "real_variables": real_variables,
      "synthetic_variables": [
          v for v in all_variables if v not in real_variables
      ],
      "tile_version": TILE_VERSION,
      "downloaded_utc": downloaded_utc,
      "dataset_title": title,
      "frame_version": FRAME_VERSION,
  }


def get_weather_models_info() -> List[Dict[str, Any]]:
  """SUPPORTED_MODELS entries plus where each model's data comes from."""
  return [
      {**info, **get_model_data_info(key)}
      for key, info in SUPPORTED_MODELS.items()
  ]


def _base_time(model_key: str) -> datetime.datetime:
  """Time of lead 0: the archived run's init time, else the current hour."""
  init_time = get_model_data_info(model_key)["init_time"]
  if init_time:
    return datetime.datetime.fromisoformat(init_time.replace("Z", "+00:00"))
  return datetime.datetime.now(datetime.timezone.utc).replace(
      minute=0, second=0, microsecond=0
  )


def _tile_field(
    model_key: str,
    var_key: str,
    step_idx: int,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Optional[np.ndarray]:
  """Field values for a tile, or None when there is nothing to draw."""
  lead_h = step_idx * STEP_HOURS
  if var_key == "accumulated_precip":
    accumulated = _accumulated_global(model_key, lead_h)
    if accumulated is None:
      return None
    return _sample_global(accumulated[0], accumulated[1], lats, lons)
  suffix = _STREAM_SUFFIX.get(var_key)
  if suffix is None and var_key != "pressure":
    return None
  stream_id = f"{model_key}_{suffix}" if suffix else None
  info = STREAM_INFO.get(stream_id) if stream_id else None
  if info:
    if suffix == "precip":
      rate_steps = _rate_file_steps(info, lead_h)
      if not rate_steps:
        return None
      return _mean_rate_grid(stream_id, rate_steps, lats, lons)
    file_step = _file_step_for_lead(info, lead_h, is_rate=False)
    if file_step is None:
      return None
    return _grid_values(stream_id, file_step, lats, lons)
  return _procedural_field(var_key, model_key, step_idx, lats, lons)


def generate_raster_tile(
    model_key: str, var_key: str, step_idx: int, z: int, x: int, y: int
) -> bytes:
  """Renders a 256x256 Web Mercator PNG tile for the given atmospheric field."""
  init_weather_streams()
  step_idx = max(0, int(step_idx))
  lats, lons = _tile_coordinates(z, x, y)
  values = _tile_field(model_key, var_key, step_idx, lats, lons)
  if values is None:
    return _transparent_tile()
  rgba = _colorize(var_key, values)
  rgba[np.abs(lats) > 85.0511] = 0
  return _encode_rgba_png(rgba)


def _frame_signature(
    model_key: str, var_key: str, step: int
) -> Optional[Tuple[Tuple[Any, ...], float]]:
  lead_h = step * STEP_HOURS
  if var_key == "accumulated_precip":
    info = STREAM_INFO.get(f"{model_key}_precip")
    if not info:
      return ("synthetic", step), lead_h
    leads = info["lead_hours"]
    if lead_h > leads[-1]:
      return None
    k = max(i for i, lead in enumerate(leads) if lead <= lead_h)
    return ("accumulated", k), leads[k]
  suffix = _STREAM_SUFFIX.get(var_key)
  if suffix is None:
    return None
  info = STREAM_INFO.get(f"{model_key}_{suffix}")
  if not info:
    return ("synthetic", step), lead_h
  if suffix == "precip":
    rate_steps = _rate_file_steps(info, lead_h)
    if not rate_steps:
      return None
    return ("rate",) + tuple(rate_steps), info["lead_hours"][rate_steps[-1]]
  file_step = _file_step_for_lead(info, lead_h, is_rate=False)
  if file_step is None:
    return None
  return ("plane", file_step), info["lead_hours"][file_step]


def get_frame_index(model_key: str, var_key: str) -> Dict[str, Any]:
  """Which frame each 3-hourly viewer step shows, for /api/weather/frames/.../index.json."""
  if model_key not in SUPPORTED_MODELS:
    raise ValueError(f"Unknown weather model '{model_key}'")
  if var_key not in FRAME_VARIABLES:
    raise ValueError(f"Layer '{var_key}' has no animation frames")
  init_weather_streams()
  data_info = get_model_data_info(model_key)
  max_step = min(NUM_STEPS - 1, int(data_info["max_lead_hours"] // STEP_HOURS))
  sigs = [
      _frame_signature(model_key, var_key, s) if s <= max_step else None
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
          else "synthetic"
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


def generate_weather_frame(
    model_key: str, var_key: str, step_idx: int
) -> bytes:
  """Whole-world PNG frame for one viewer step (see get_frame_index)."""
  global _FRAME_CACHE_BYTES
  index = get_frame_index(model_key, var_key)
  step = max(0, min(NUM_STEPS - 1, int(step_idx)))
  rep = index["step_frames"][step]
  if rep is None:
    return _empty_frame()
  key = (model_key, var_key, rep, index["init_time"])
  with _FRAME_LOCK:
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
      _FRAME_CACHE.move_to_end(key)
      return cached
  lats, lons = _frame_coordinates()
  values = _tile_field(model_key, var_key, rep, lats, lons)
  png = (
      _empty_frame()
      if values is None
      else _encode_indexed_png(*_colorize_indexed(var_key, values))
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


def _prewarm_frames() -> None:
  """Renders the rain frames of every model with real data, so Play starts fast."""
  if os.environ.get("EARTHKIT_WEATHER_PREWARM_FRAMES", "1") != "1":
    return
  streams = STREAM_INFO
  for model_key in SUPPORTED_MODELS:
    info = streams.get(f"{model_key}_precip")
    if not info or not info["archived_run"]:
      continue
    try:
      for step in get_frame_index(model_key, "precipitation")["frame_steps"]:
        if STREAM_INFO is not streams:
          return
        generate_weather_frame(model_key, "precipitation", step)
    except Exception as e:  # pylint: disable=broad-except
      logger.warning(
          "[WeatherEngine] Frame pre-render failed for %s: %s", model_key, e
      )


def get_wind_vectors(
    model_key: str = "ecmwf_ifs", step_idx: int = 0, subsample: int = 2
) -> Dict[str, Any]:
  """Returns downsampled global U/V vector matrices for client Canvas streamlines."""
  init_weather_streams()
  u_stream = f"{model_key}_u10"
  v_stream = f"{model_key}_v10"
  if u_stream in STREAM_INFO and v_stream in STREAM_INFO:
    lead_h = max(0, int(step_idx)) * STEP_HOURS
    fu = _file_step_for_lead(STREAM_INFO[u_stream], lead_h, is_rate=False)
    fv = _file_step_for_lead(STREAM_INFO[v_stream], lead_h, is_rate=False)
    if fu is not None and fv is not None:
      res = extract_wind_vectors(
          _ARRAYS,
          STREAM_INFO,
          model_key=model_key,
          step_idx=step_idx,
          subsample=subsample,
      )
      if not (
          STREAM_INFO[u_stream]["archived_run"]
          and STREAM_INFO[v_stream]["archived_run"]
      ):
        res["header"]["data_source"] = "synthetic"
      return res

  subsample = max(1, min(4, subsample))
  step_idx = max(0, int(step_idx))
  step_deg = 1.0 * subsample
  lats = np.array([90.0 - i * step_deg for i in range(int(180 / step_deg) + 1)])
  lons = np.array([-180.0 + j * step_deg for j in range(int(360 / step_deg))])
  lead_h = step_idx * STEP_HOURS
  u = _procedural_field("wind_u", model_key, step_idx, lats, lons)
  v = _procedural_field("wind_v", model_key, step_idx, lats, lons)
  valid_dt = _base_time(model_key) + datetime.timedelta(hours=lead_h)
  return {
      "header": {
          "model": model_key,
          "step_hours": lead_h,
          "valid_time": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
          "nx": len(lons),
          "ny": len(lats),
          "lo1": -180.0,
          "la1": 90.0,
          "dx": step_deg,
          "dy": step_deg,
          "data_source": "synthetic",
      },
      "u": np.round(np.nan_to_num(u).astype(np.float64), 2).ravel().tolist(),
      "v": np.round(np.nan_to_num(v).astype(np.float64), 2).ravel().tolist(),
  }


def _point_value(
    model_key: str, var_key: str, lead_h: float, lat: float, lon: float
) -> Optional[float]:
  """One value at a point: archived/placeholder stream first, else synthetic."""
  suffix = _STREAM_SUFFIX.get(var_key)
  stream_id = f"{model_key}_{suffix}" if suffix else None
  if stream_id and stream_id in STREAM_INFO and stream_id in _ARRAYS:
    return extract_point_value(
        _ARRAYS, STREAM_INFO, model_key, var_key, lead_h, lat, lon
    )
  return _procedural_sample(
      var_key, model_key, int(lead_h // STEP_HOURS), lat, lon
  )


def _accumulation_series(
    model_key: str, lat: float, lon: float, lead_hours: Sequence[float]
) -> List[Optional[float]]:
  """Rain accumulated since the forecast start at each lead (None beyond run)."""
  stream_id = f"{model_key}_precip"
  if stream_id in STREAM_INFO and stream_id in _ARRAYS:
    return extract_accumulation_series(
        _ARRAYS, STREAM_INFO, model_key, lat, lon, lead_hours
    )
  accum, out = 0.0, []
  for lead_h in lead_hours:
    accum += (
        _procedural_sample(
            "precipitation", model_key, int(lead_h // STEP_HOURS), lat, lon
        )
        * STEP_HOURS
    )
    out.append(accum)
  return out


def get_weather_probe(lat: float, lon: float) -> Dict[str, Any]:
  """Returns comparative 10-day multi-model meteorological soundings."""
  init_weather_streams()
  lead_hours = [i * STEP_HOURS for i in range(NUM_STEPS)]
  now_utc = datetime.datetime.now(datetime.timezone.utc).replace(
      minute=0, second=0, microsecond=0
  )

  results: Dict[str, Any] = {
      "latitude": round(lat, 4),
      "longitude": round(lon, 4),
      "query_time_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "lead_hours": lead_hours,
      "models": {},
  }

  for m_key, m_info in SUPPORTED_MODELS.items():
    data_info = get_model_data_info(m_key)
    accum = _accumulation_series(m_key, lat, lon, lead_hours)
    precip_curve: List[Optional[float]] = []
    temp_curve: List[Optional[float]] = []
    wind_spd_curve: List[float] = []
    wind_dir_curve: List[int] = []
    pressure_curve: List[float] = []

    for step, lead_h in enumerate(lead_hours):
      precip_curve.append(
          _round_or_none(
              _point_value(m_key, "precipitation", lead_h, lat, lon), 2
          )
      )
      temp_curve.append(
          _round_or_none(
              _point_value(m_key, "temperature", lead_h, lat, lon), 1
          )
      )
      u_w = _procedural_sample("wind_u", m_key, step, lat, lon)
      v_w = _procedural_sample("wind_v", m_key, step, lat, lon)
      wind_spd_curve.append(round(math.sqrt(u_w**2 + v_w**2), 1))
      wind_dir_curve.append(
          int((math.degrees(math.atan2(-u_w, -v_w)) + 360) % 360)
      )
      pressure_curve.append(
          round(_procedural_sample("pressure", m_key, step, lat, lon), 1)
      )

    results["models"][m_key] = {
        "name": m_info["name"],
        "badge": m_info["badge"],
        "precip_rate_mmh": precip_curve,
        "accum_precip_mm": [_round_or_none(a, 1) for a in accum],
        "temp_c": temp_curve,
        "wind_speed_mps": wind_spd_curve,
        "wind_direction_deg": wind_dir_curve,
        "pressure_hpa": pressure_curve,
        **data_info,
    }

  return results


def get_catchment_weather_summary(
    geojson_feature: Dict[str, Any],
    step_idx: int = 0,
    model_key: str = "ecmwf_ifs",
) -> Dict[str, Any]:
  """Calculates basin-averaged precipitation and temperature for an active catchment."""
  init_weather_streams()
  if model_key not in SUPPORTED_MODELS:
    model_key = "ecmwf_ifs"
  step_idx = max(0, int(step_idx))
  lead_h = step_idx * STEP_HOURS

  props = geojson_feature.get("properties", {})
  catchment_id = props.get("catchment_id") or geojson_feature.get("id", "basin")
  area_km2 = props.get("area_km2", 1250.0)

  lats, lons = _geometry_points(geojson_feature.get("geometry", {}))
  if lats and lons:
    c_lat = sum(lats) / len(lats)
    c_lon = sum(lons) / len(lons)
  else:
    c_lat = props.get("outlet_latitude", 40.0)
    c_lon = props.get("outlet_longitude", -86.0)

  sample_points = list(zip(lats, lons)) if lats and lons else [(c_lat, c_lon)]
  stride = max(1, len(sample_points) // 10)
  rates = [
      r
      for r in (
          _point_value(model_key, "precipitation", lead_h, p_lat, p_lon)
          for p_lat, p_lon in sample_points[::stride]
      )
      if r is not None
  ]
  temp_c = _point_value(model_key, "temperature", lead_h, c_lat, c_lon)

  data_info = get_model_data_info(model_key)
  window_h = data_info["max_lead_hours"]
  accum = _accumulation_series(model_key, c_lat, c_lon, [window_h])[0]

  valid_dt = _base_time(model_key) + datetime.timedelta(hours=lead_h)
  return {
      "catchment_id": catchment_id,
      "area_km2": round(area_km2, 1),
      "step_hours": lead_h,
      "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "basin_mean_precip_mmh": (
          round(sum(rates) / len(rates), 2) if rates else None
      ),
      "basin_max_precip_mmh": round(max(rates), 2) if rates else None,
      "basin_accumulated_10d_mm": _round_or_none(accum, 1),
      "basin_mean_temp_c": _round_or_none(temp_c, 1),
      "centroid": {"latitude": round(c_lat, 4), "longitude": round(c_lon, 4)},
      "model": model_key,
      "data_source": data_info["data_source"],
      "accumulation_hours": window_h,
  }
