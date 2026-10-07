"""Earthkit Hydro weather visualization UI adapter for the Weather Viewer tab.

Delegates all backend forecast stream loading, grid slicing, wind vector
extraction, point meteograms, and catchment weather summaries to
`multimet.weather_fetcher`, and all Web Mercator tile rendering, indexed PNG
animation frame encoding, and colormaps to `frontend.weather_viewer`.
"""

from __future__ import annotations

import datetime
import json
import logging
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
    clear_frame_cache as _clear_frame_cache,
    compute_frame_index,
    empty_frame as _empty_frame,
    evaluate_tile_field,
    frame_coordinates as _frame_coordinates,
    FRAME_SIZE,
    FRAME_VARIABLES,
    FRAME_VERSION,
    MERCATOR_MAX_LAT,
    render_raster_tile,
    render_weather_frame,
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
    clear_accum_grid_cache as _clear_accum_grid_cache,
    extract_accumulation_series,
    extract_point_value,
    fetch_catchment_summary,
    fetch_point_timeseries,
    file_step_for_lead as _file_step_for_lead,
    geometry_points as _geometry_points,
    get_model_data_info_from_streams,
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


def _close_handles() -> None:
  """Closes open memory-mapped stream handles before reloading."""
  _ARRAYS.clear()
  for mm, _, _, _, _ in MMAP_HANDLES.values():
    if not mm.closed:
      mm.close()
  MMAP_HANDLES.clear()


def _install_streams(target_dir: Path) -> None:
  """Swaps in the streams of target_dir via multimet.weather_fetcher (caller holds _INIT_LOCK)."""
  global MMAP_HANDLES, STREAM_INFO, _ARRAYS, _LOADED_SIGNATURE, _LOADED_DIR
  _close_handles()
  handles, infos, arrays = _scan_streams(target_dir)
  MMAP_HANDLES, STREAM_INFO, _ARRAYS = handles, infos, arrays
  _LOADED_SIGNATURE = _dir_signature(target_dir)
  _LOADED_DIR = target_dir.resolve()
  _clear_accum_grid_cache()
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


def get_model_data_info(model_key: str) -> Dict[str, Any]:
  """Returns metadata and availability status for `model_key` via multimet.weather_fetcher."""
  init_weather_streams()
  info = get_model_data_info_from_streams(STREAM_INFO, model_key)
  return {
      "data_source": info["data_source"],
      "init_time": info["init_time"],
      "max_lead_hours": (
          info["max_lead_hours"] if info["real_variables"] else MAX_LEAD_HOURS
      ),
      "real_variables": info["real_variables"],
      "missing_variables": info["missing_variables"],
      "synthetic_variables": info["missing_variables"],
      "tile_version": TILE_VERSION,
      "downloaded_utc": info["downloaded_utc"],
      "dataset_title": info["dataset_title"],
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
  """Evaluates physical field values on `(lats, lons)` via frontend.weather_viewer."""
  init_weather_streams()
  return evaluate_tile_field(
      _ARRAYS,
      STREAM_INFO,
      model_key=model_key,
      var_key=var_key,
      step_idx=step_idx,
      lats=lats,
      lons=lons,
      bilinear=False,
      strict=False,
  )


def generate_raster_tile(
    model_key: str, var_key: str, step_idx: int, z: int, x: int, y: int
) -> bytes:
  """Renders a 256x256 Web Mercator PNG tile via frontend.weather_viewer."""
  init_weather_streams()
  return render_raster_tile(
      _ARRAYS,
      STREAM_INFO,
      model_key=model_key,
      var_key=var_key,
      step_idx=step_idx,
      z=z,
      x=x,
      y=y,
      bilinear=False,
      strict=False,
  )


def get_frame_index(model_key: str, var_key: str) -> Dict[str, Any]:
  """Which frame each 3-hourly viewer step shows, for /api/weather/frames/.../index.json."""
  init_weather_streams()
  return compute_frame_index(
      STREAM_INFO, model_key=model_key, var_key=var_key, strict=False
  )


def generate_weather_frame(
    model_key: str, var_key: str, step_idx: int
) -> bytes:
  """Whole-world PNG frame for one viewer step via frontend.weather_viewer."""
  init_weather_streams()
  return render_weather_frame(
      _ARRAYS,
      STREAM_INFO,
      model_key=model_key,
      var_key=var_key,
      step_idx=step_idx,
      strict=False,
  )


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
  """Returns downsampled global U/V vector matrices via multimet.weather_fetcher."""
  init_weather_streams()
  u_stream = f"{model_key}_u10"
  v_stream = f"{model_key}_v10"
  if (
      u_stream in STREAM_INFO
      and v_stream in STREAM_INFO
      and u_stream in _ARRAYS
      and v_stream in _ARRAYS
  ):
    lead_h = max(0, int(step_idx)) * STEP_HOURS
    fu = _file_step_for_lead(STREAM_INFO[u_stream], lead_h, is_rate=False)
    fv = _file_step_for_lead(STREAM_INFO[v_stream], lead_h, is_rate=False)
    if fu is not None and fv is not None:
      return extract_wind_vectors(
          _ARRAYS,
          STREAM_INFO,
          model_key=model_key,
          step_idx=step_idx,
          subsample=subsample,
      )

  subsample = max(1, min(4, int(subsample)))
  step_idx = max(0, int(step_idx))
  step_deg = 1.0 * subsample
  nx = int(360 / step_deg)
  ny = int(180 / step_deg) + 1
  lead_h = step_idx * STEP_HOURS
  valid_dt = _base_time(model_key) + datetime.timedelta(hours=lead_h)
  return {
      "header": {
          "model": model_key,
          "step_hours": lead_h,
          "valid_time": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
          "nx": nx,
          "ny": ny,
          "lo1": -180.0,
          "la1": 90.0,
          "dx": step_deg,
          "dy": step_deg,
          "data_source": "unavailable",
      },
      "u": [0.0] * (nx * ny),
      "v": [0.0] * (nx * ny),
  }


def _point_value(
    model_key: str, var_key: str, lead_h: float, lat: float, lon: float
) -> Optional[float]:
  """Extracts one scalar value at `(lat, lon)` from synced streams, or None."""
  init_weather_streams()
  suffix = _STREAM_SUFFIX.get(var_key)
  if var_key == "wind_u":
    suffix = "u10"
  elif var_key == "wind_v":
    suffix = "v10"
  stream_id = f"{model_key}_{suffix}" if suffix else None
  if stream_id and stream_id in STREAM_INFO and stream_id in _ARRAYS:
    return extract_point_value(
        _ARRAYS, STREAM_INFO, model_key, var_key, lead_h, lat, lon
    )
  return None


def _accumulation_series(
    model_key: str, lat: float, lon: float, lead_hours: Sequence[float]
) -> List[Optional[float]]:
  """Rain accumulated since the forecast start at each lead (None if unsynced)."""
  init_weather_streams()
  stream_id = f"{model_key}_precip"
  if stream_id in STREAM_INFO and stream_id in _ARRAYS:
    return extract_accumulation_series(
        _ARRAYS, STREAM_INFO, model_key, lat, lon, lead_hours
    )
  return [None] * len(lead_hours)


def get_weather_probe(lat: float, lon: float) -> Dict[str, Any]:
  """Returns comparative 10-day multi-model meteorological soundings via multimet.weather_fetcher."""
  init_weather_streams()
  probe = fetch_point_timeseries(
      _ARRAYS,
      STREAM_INFO,
      lat=lat,
      lon=lon,
      models=list(SUPPORTED_MODELS.keys()),
      strict=False,
  )
  for m_data in probe["models"].values():
    m_data["tile_version"] = TILE_VERSION
    m_data["frame_version"] = FRAME_VERSION
    m_data["synthetic_variables"] = m_data.get("missing_variables", [])
  return probe


def get_catchment_weather_summary(
    geojson_feature: Dict[str, Any],
    step_idx: int = 0,
    model_key: str = "ecmwf_ifs",
) -> Dict[str, Any]:
  """Calculates basin-averaged precipitation and temperature via multimet.weather_fetcher."""
  init_weather_streams()
  if model_key not in SUPPORTED_MODELS:
    model_key = "ecmwf_ifs"
  if (
      f"{model_key}_precip" in STREAM_INFO
      and f"{model_key}_temp" in STREAM_INFO
  ):
    return fetch_catchment_summary(
        _ARRAYS,
        STREAM_INFO,
        geojson_feature=geojson_feature,
        step_idx=step_idx,
        model_key=model_key,
    )

  step_idx_clamped = max(0, int(step_idx))
  lead_h = step_idx_clamped * STEP_HOURS
  props = geojson_feature.get("properties") or {}
  catchment_id = (
      props.get("catchment_id") or geojson_feature.get("id") or "basin"
  )
  area_km2 = float(props.get("area_km2", 1250.0))
  lats, lons = _geometry_points(geojson_feature.get("geometry") or {})
  if lats and lons:
    c_lat = sum(lats) / len(lats)
    c_lon = sum(lons) / len(lons)
  else:
    c_lat = float(props.get("outlet_latitude", 40.0))
    c_lon = float(props.get("outlet_longitude", -86.0))
  valid_dt = _base_time(model_key) + datetime.timedelta(hours=lead_h)
  return {
      "catchment_id": catchment_id,
      "area_km2": round(area_km2, 1),
      "step_hours": lead_h,
      "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "basin_mean_precip_mmh": None,
      "basin_max_precip_mmh": None,
      "basin_accumulated_10d_mm": None,
      "basin_mean_temp_c": None,
      "centroid": {"latitude": round(c_lat, 4), "longitude": round(c_lon, 4)},
      "model": model_key,
      "data_source": "unavailable",
      "accumulation_hours": 0,
  }
