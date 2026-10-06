"""Earthkit Hydro weather visualization engine for the Weather Viewer tab.

Provides:
  1. Web Mercator PNG tiles (rain rate, rain accumulation, 2 m temperature and
     mean sea level pressure).
  2. A coarse global U/V grid for the client-side wind particle animation.
  3. Multi-model point forecasts for the meteogram of a clicked location.
  4. Basin-averaged rain for the active catchment.

Data sources, in order of preference:
  * Archived model runs stored as float16 ``<model>_<variable>.bin`` grids
    (721 x 1440 at 0.25 deg, one plane per lead time) in the first directory of
    ``CANDIDATE_DATA_DIRS`` that holds ``.bin`` files. A stream only counts as a
    real archived run when ``latest_dynamical_meta.json`` in the same directory
    lists that run (init time and number of lead steps). Other ``.bin`` files
    are placeholders written by ``weather_sync.py`` and are reported as
    synthetic.
  * A deterministic synthetic field (``_procedural_sample``) for everything
    else, so the viewer keeps working offline.

Every model-level response carries a ``data_source`` of ``"archived_run"`` or
``"synthetic"`` so the UI can say which one the user is looking at.
"""

import collections
import datetime
import json
import logging
import math
import mmap
import os
from pathlib import Path
import struct
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zlib

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "forecasts"


def _weather_data_root() -> Path:
  """Local-disk root written by weather_sync.sync_latest (same rule as there)."""
  env = os.environ.get("EARTHKIT_WEATHER_DATA_DIR")
  if env:
    return Path(env).expanduser().resolve()
  return Path.home() / ".cache" / "openhydronet" / "weather"


# Runs downloaded by weather_sync.py. "current" is a symlink that is repointed
# atomically when a newer run has been downloaded (checked hourly).
WEATHER_DATA_ROOT = _weather_data_root()
LOCAL_CURRENT_DIR = WEATHER_DATA_ROOT / "current"
SYNC_STATUS_FILE = "sync_status.json"

# Candidate search directories for binary weather tensors, in order: the
# auto-updated local runs, this project's data dir, and the one-off alertnexus
# download (2026-08-13) as a last resort.
CANDIDATE_DATA_DIRS = [
    LOCAL_CURRENT_DIR,
    DATA_DIR,
    Path("/tmp/openhydronet"),
]

# Resolution specifications
N_LAT = 721    # +90.0 to -90.0 at 0.25 deg
N_LON = 1440   # -180.0 to +179.75 at 0.25 deg
GRID_DEG = 0.25
NUM_STEPS = 81  # Viewer timeline: 10 days at 3-hour steps (0, 3, ..., 240 h).
STEP_HOURS = 3
MAX_LEAD_HOURS = (NUM_STEPS - 1) * STEP_HOURS
# Reported by /api/weather/models and put into tile URLs by the viewer. Tiles
# are cached by browsers for a day, so bump this whenever tile values change.
TILE_VERSION = "3"

# Run metadata written next to archived dynamical.org extracts.
RUN_METADATA_FILE = "latest_dynamical_meta.json"
_RUN_DATASET_TO_MODEL = {
    "ecmwf_aifs_single_forecast": "ecmwf_aifs",
    "noaa_gfs_forecast": "noaa_gfs",
    "ecmwf_ifs_ens_forecast_15_day_0_25_degree": "ecmwf_ifs",
}
# weather_sync stores MSLP as hPa - 1000 (float16 precision); added back here.
DEFAULT_MSLP_OFFSET_HPA = 1000.0

# Supported models and variables
SUPPORTED_MODELS = {
    "ecmwf_ifs": {
        "id": "ecmwf_ifs",
        "name": "ECMWF IFS (ensemble control run)",
        "badge": "0.25° Physics 10-Day",
        "type": "physics",
        "resolution": "0.25° Global (9km native)",
        "organization": "ECMWF",
    },
    "ecmwf_aifs": {
        "id": "ecmwf_aifs",
        "name": "ECMWF AIFS",
        "badge": "0.25° Global AI 10-Day",
        "type": "ai",
        "resolution": "0.25° Global",
        "organization": "ECMWF",
    },
    "graphcast": {
        "id": "graphcast",
        "name": "Google DeepMind GraphCast",
        "badge": "0.25° AI 10-Day",
        "type": "ai",
        "resolution": "0.25° Global",
        "organization": "Google DeepMind",
    },
    "noaa_gfs": {
        "id": "noaa_gfs",
        "name": "NOAA GFS",
        "badge": "0.25° Physics 10-Day",
        "type": "physics",
        "resolution": "0.25° Global",
        "organization": "NOAA / NWS",
    },
}

SUPPORTED_VARIABLES = {
    "precipitation": {
        "id": "precipitation",
        "name": "Total Precipitation Rate",
        "unit": "mm/h",
        "min": 0.0,
        "max": 25.0,
    },
    "accumulated_precip": {
        "id": "accumulated_precip",
        "name": "Precipitation Accumulated Since Forecast Start",
        "unit": "mm",
        "min": 0.0,
        "max": 250.0,
    },
    "temperature": {
        "id": "temperature",
        "name": "2m Ambient Temperature",
        "unit": "°C",
        "min": -40.0,
        "max": 45.0,
    },
    "wind": {
        "id": "wind",
        "name": "10m Surface Wind Velocity",
        "unit": "m/s",
        "min": 0.0,
        "max": 40.0,
    },
    "pressure": {
        "id": "pressure",
        "name": "Mean Sea Level Pressure",
        "unit": "hPa",
        "min": 960.0,
        "max": 1040.0,
    },
}

# Tile colour classes: (lower bound, RGBA). Values below the first bound are
# transparent. The Weather Viewer legend in static/index.html mirrors these.
RAIN_RATE_CLASSES = (
    (0.1, (125, 211, 252, 150)),
    (0.5, (59, 130, 246, 185)),
    (2.0, (34, 197, 94, 205)),
    (5.0, (234, 179, 8, 220)),
    (10.0, (239, 68, 68, 235)),
    (20.0, (192, 38, 211, 245)),
)
RAIN_ACCUM_CLASSES = (
    (1.0, (125, 211, 252, 140)),
    (5.0, (59, 130, 246, 175)),
    (10.0, (34, 197, 94, 195)),
    (25.0, (234, 179, 8, 215)),
    (50.0, (239, 68, 68, 230)),
    (100.0, (192, 38, 211, 245)),
)

# (stream id, file name, is precipitation)
_STREAM_FILES = (
    ("ecmwf_ifs_precip", "ecmwf_ifs_precip.bin", True),
    ("ecmwf_ifs_temp", "ecmwf_ifs_temp.bin", False),
    ("ecmwf_ifs_mslp", "ecmwf_ifs_mslp.bin", False),
    ("ecmwf_ifs_u10", "ecmwf_ifs_u10.bin", False),
    ("ecmwf_ifs_v10", "ecmwf_ifs_v10.bin", False),
    ("ecmwf_aifs_precip", "ecmwf_aifs_precip.bin", True),
    ("ecmwf_aifs_temp", "ecmwf_aifs_temp.bin", False),
    ("ecmwf_aifs_mslp", "ecmwf_aifs_mslp.bin", False),
    ("ecmwf_aifs_u10", "ecmwf_aifs_u10.bin", False),
    ("ecmwf_aifs_v10", "ecmwf_aifs_v10.bin", False),
    ("graphcast_precip", "graphcast_precip.bin", True),
    ("graphcast_temp", "graphcast_temp.bin", False),
    ("noaa_gfs_precip", "noaa_gfs_precip.bin", True),
    ("noaa_gfs_temp", "noaa_gfs_temp.bin", False),
    ("noaa_gfs_mslp", "noaa_gfs_mslp.bin", False),
    ("noaa_gfs_u10", "noaa_gfs_u10.bin", False),
    ("noaa_gfs_v10", "noaa_gfs_v10.bin", False),
)

# Viewer variable -> suffix of the stream that holds it.
_STREAM_SUFFIX = {
    "precipitation": "precip",
    "accumulated_precip": "precip",
    "temperature": "temp",
    "pressure": "mslp",
}

MMAP_HANDLES: Dict[str, Tuple[mmap.mmap, int, int, int, bool]] = {}
# Per-stream metadata: model, n_steps, lead_hours, archived_run, init_time.
STREAM_INFO: Dict[str, Dict[str, Any]] = {}
_ARRAYS: Dict[str, np.ndarray] = {}
_INITIALIZED = False
_INIT_LOCK = threading.Lock()
_TRANSPARENT_TILE: Optional[bytes] = None
# (resolved data dir, metadata mtime) of the loaded run; see reload_if_changed.
_LOADED_SIGNATURE: Optional[Tuple[str, int]] = None
_LOADED_DIR: Optional[Path] = None

# Running rain totals keyed by (stream id, lead index); ~4 MB each at 0.25 deg.
_ACCUM_CACHE: "collections.OrderedDict[Tuple[str, int], np.ndarray]" = (
    collections.OrderedDict()
)
_ACCUM_CACHE_SIZE = 24
_ACCUM_LOCK = threading.Lock()
_SYNTHETIC_ACCUM_DEG = 1.0  # Synthetic totals are only a fallback; keep cheap.


def _load_run_metadata(target_dir: Path) -> Dict[str, Dict[str, Any]]:
  """Reads archived-run metadata (init time, lead steps) keyed by model."""
  path = target_dir / RUN_METADATA_FILE
  if not path.exists():
    return {}
  try:
    meta = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, ValueError) as e:
    logger.warning("Could not read %s: %s", path, e)
    return {}
  runs = {}
  for dataset_key, dataset in (meta.get("datasets") or {}).items():
    model_key = dataset.get("model") or _RUN_DATASET_TO_MODEL.get(dataset_key)
    if model_key not in SUPPORTED_MODELS or not dataset.get("init_time"):
      continue
    init_time = str(dataset["init_time"])
    if not init_time.endswith("Z"):
      init_time += "Z"
    lead_hours = dataset.get("lead_hours")
    runs[model_key] = {
        "init_time": init_time,
        "lead_steps": int(dataset.get("lead_steps") or 0),
        # Written by weather_sync.sync_latest; older extracts only have lead_steps.
        "lead_hours": [int(h) for h in lead_hours] if lead_hours else None,
        "downloaded_utc": dataset.get("downloaded_utc") or meta.get("last_updated_utc"),
        "title": dataset.get("title"),
        "mslp_offset_hpa": float(
            dataset.get("mslp_offset_hpa", DEFAULT_MSLP_OFFSET_HPA)
        ),
    }
  return runs


def _run_lead_hours(model_key: str, n_steps: int) -> List[int]:
  """Lead hours of each plane in an archived run file."""
  if model_key == "ecmwf_aifs":
    return [6 * i for i in range(n_steps)]  # 6-hourly
  if model_key == "noaa_gfs":
    hourly = list(range(min(n_steps, 121)))  # hourly to +120 h ...
    return hourly + [120 + 3 * (i + 1) for i in range(n_steps - len(hourly))]
  return [STEP_HOURS * i for i in range(n_steps)]


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


def _scan_streams(target_dir: Path):
  """Memory-maps every stream in target_dir; returns (handles, info, arrays)."""
  target_dir = target_dir.resolve()
  runs = _load_run_metadata(target_dir)
  plane_bytes = N_LAT * N_LON * 2
  handles, infos, arrays = {}, {}, {}
  for stream_id, fname, is_precip in _STREAM_FILES:
    fpath = target_dir / fname
    if not fpath.exists():
      continue
    try:
      n_steps = fpath.stat().st_size // plane_bytes
      if n_steps < 1:
        continue
      with open(fpath, "rb") as f_handle:
        mm = mmap.mmap(f_handle.fileno(), 0, access=mmap.ACCESS_READ)
    except (OSError, ValueError) as e:
      logger.warning("[WeatherEngine] Could not map %s: %s", fname, e)
      continue

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
      lead_hours = run_leads or _run_lead_hours(model_key, n_steps)
    else:
      lead_hours = [STEP_HOURS * i for i in range(n_steps)]
    infos[stream_id] = {
        "model": model_key,
        "file": str(fpath),
        "n_steps": n_steps,
        "lead_hours": lead_hours,
        "archived_run": archived,
        "init_time": run["init_time"] if archived else None,
        "downloaded_utc": run.get("downloaded_utc") if archived else None,
        "title": run.get("title") if archived else None,
        # Value stored = true value - offset (MSLP only).
        "offset": (
            (run["mslp_offset_hpa"] if run else DEFAULT_MSLP_OFFSET_HPA)
            if suffix == "mslp"
            else 0.0
        ),
    }
  return handles, infos, arrays


def _install_streams(target_dir: Path):
  """Swaps in the streams of target_dir (caller holds _INIT_LOCK)."""
  global MMAP_HANDLES, STREAM_INFO, _ARRAYS, _LOADED_SIGNATURE, _LOADED_DIR
  handles, infos, arrays = _scan_streams(target_dir)
  # Rebind (not mutate) so a request that already holds the old dicts keeps a
  # consistent view. Old maps are closed by the GC once nothing uses them;
  # their files stay readable even after weather_sync deletes old runs.
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


def init_weather_streams():
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
  """Switches to a newer downloaded run if there is one; True if it did.

  Cheap (a few stat calls). Called by the hourly weather sync after each check,
  so the server shows a new run without a restart.
  """
  if not _INITIALIZED:
    init_weather_streams()
    return True
  target_dir = _find_data_dir()
  if target_dir is None or _dir_signature(target_dir) == _LOADED_SIGNATURE:
    return False
  with _INIT_LOCK:
    if _dir_signature(target_dir) == _LOADED_SIGNATURE:
      return False
    logger.info("[WeatherEngine] Loading new forecast run from %s", target_dir.resolve())
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
    status = {"last_result": "never", "message": "No automatic update has run yet."}
  loaded = _LOADED_DIR.resolve() if _LOADED_DIR else None
  return {
      "last_check_utc": status.get("last_check_utc"),
      "last_success_utc": status.get("last_success_utc"),
      "last_result": status.get("last_result"),
      "message": status.get("message"),
      "check_interval_minutes": status.get("check_interval_minutes", 60),
      "auto_update": loaded is not None and str(loaded).startswith(
          str(WEATHER_DATA_ROOT.resolve())
      ),
      "data_dir": str(loaded) if loaded else None,
  }


def _warm_caches(paths: Sequence[str]):
  _warm_page_cache(paths)
  _prewarm_frames()


def _warm_page_cache(paths: Sequence[str]):
  """Reads each file once so later random access hits the OS page cache."""
  for path in paths:
    try:
      with open(path, "rb") as f_handle:
        while f_handle.read(8 << 20):
          pass
    except OSError as e:
      logger.debug("Page-cache warm-up skipped %s: %s", path, e)


def _file_step_for_lead(
    info: Dict[str, Any], lead_h: float, is_rate: bool
) -> Optional[int]:
  """Index of the stored plane for a lead time, or None beyond the run."""
  leads = info["lead_hours"]
  if lead_h < 0 or lead_h > leads[-1]:
    return None
  if is_rate and info["archived_run"]:
    # Archived rain rates are averages over the interval that ends at each
    # lead, so lead 0 is empty; use the first interval that covers lead_h.
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


def _rate_file_steps(info: Dict[str, Any], lead_h: float) -> List[int]:
  """Stored rain-rate planes to average for the viewer step ending at lead_h.

  Archived rates are averages over the interval ending at each stored lead.
  A viewer step shows the mean rate over the STEP_HOURS ending at lead_h, so
  hourly output (GFS to +120 h) is averaged over three planes instead of
  showing one hour in three. Coarser output (AIFS, 6-hourly) uses the single
  interval that covers lead_h, and lead 0 uses the first stored interval.
  Returns [] beyond the end of the run.
  """
  first = _file_step_for_lead(info, lead_h, is_rate=True)
  if first is None:
    return []
  if not info["archived_run"] or lead_h <= 0:
    return [first]
  steps = [
      i
      for i, lead in enumerate(info["lead_hours"])
      if 0 < lead <= lead_h and lead > lead_h - STEP_HOURS
  ]
  return steps or [first]


def _mean_rate_grid(
    stream_id: str, file_steps: Sequence[int], lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
  """Mean rain rate over one or more stored planes on a lat x lon grid."""
  if len(file_steps) == 1:
    return _grid_values(stream_id, file_steps[0], lats, lons)
  rows, cols = _grid_indices(lats, lons)
  values = _ARRAYS[stream_id][np.ix_(list(file_steps), rows, cols)]
  return np.nan_to_num(values.astype(np.float32), nan=0.0).mean(axis=0)


def _grid_indices(
    lats: np.ndarray, lons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  rows = np.clip(np.rint((90.0 - lats) / GRID_DEG), 0, N_LAT - 1).astype(np.intp)
  cols = np.rint(((lons + 180.0) % 360.0) / GRID_DEG).astype(np.intp) % N_LON
  return rows, cols


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
  """Rain (mm) accumulated from the forecast start to lead_h on a global grid.

  Returns (grid, grid spacing in degrees), or None beyond the end of the run.
  Totals are cached and built from the closest earlier cached total, so
  stepping the timeline forward only adds the new forecast hours.
  """
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

    def increment(i):
      rate = _ARRAYS[stream_id][i].astype(np.float32)
      return np.nan_to_num(rate, nan=0.0) * float(leads[i] - leads[i - 1])
  else:
    k = int(lead_h // STEP_HOURS)
    key = (f"{model_key}_synthetic", k)
    res = _SYNTHETIC_ACCUM_DEG
    grid_lats = 90.0 - np.arange(int(180 / res) + 1) * res
    grid_lons = -180.0 + np.arange(int(360 / res)) * res
    shape = (len(grid_lats), len(grid_lons))

    def increment(i):
      return _procedural_field(
          "precipitation", model_key, i, grid_lats, grid_lons
      ) * STEP_HOURS

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
        np.zeros(shape, np.float32) if start_total is None else start_total.copy()
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


def make_png_bytes(width: int, height: int, raw_pixels: bytearray) -> bytes:
  """Pure-Python standard library fast PNG encoder with zlib compression."""
  def chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + tag
        + data
        + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )

  header = b"\x89PNG\r\n\x1a\n"
  ihdr = chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 6, 0, 0, 0))
  idat = chunk(b"IDAT", zlib.compress(bytes(raw_pixels), 1))
  iend = chunk(b"IEND", b"")
  return header + ihdr + idat + iend


def _encode_rgba_png(rgba: np.ndarray) -> bytes:
  height, width, _ = rgba.shape
  raw = np.zeros((height, 1 + width * 4), dtype=np.uint8)  # filter byte 0
  raw[:, 1:] = rgba.reshape(height, width * 4)
  return make_png_bytes(width, height, raw.tobytes())


def _transparent_tile() -> bytes:
  global _TRANSPARENT_TILE
  if _TRANSPARENT_TILE is None:
    _TRANSPARENT_TILE = _encode_rgba_png(np.zeros((256, 256, 4), np.uint8))
  return _TRANSPARENT_TILE


def _procedural_sample(
    var_key: str, model_key: str, step_idx: int, lat_deg: float, lon_deg: float
) -> float:
  """Computes high-fidelity mathematical atmospheric circulation fields on the fly.

  Ensures gorgeous, realistic meteorological maps everywhere on Earth even if
  the multi-gigabyte binary files have not yet been pre-downloaded to disk.
  """
  lead_h = step_idx * 3
  lat_r = math.radians(lat_deg)
  lon_r = math.radians(lon_deg)
  s_lat = math.sin(lat_r)
  c_lat = math.cos(lat_r)

  # Planetary Rossby Waves
  wave = (
      math.sin(3 * lon_r + lead_h * 0.05) * 4.5 * math.cos(lat_r * 2.0)
      + math.sin(5 * lon_r - lead_h * 0.08) * 2.8 * s_lat
  )

  if var_key in ["precipitation", "accumulated_precip"]:
    # ITCZ equatorial convergence belt & Mid-latitude baroclinic storm tracks
    itcz = abs(lat_deg - (4.0 * math.sin(lon_r * 2.0 + lead_h * 0.02)))
    storm = abs(abs(lat_deg) - 45.0 + 5.0 * math.sin(4 * lon_r + lead_h * 0.04))
    p = 0.0
    if itcz < 8.0:
      p += (8.0 - itcz) * 0.95 * max(0.0, math.sin(lon_r * 4.0 + lead_h * 0.1))
    if storm < 10.0:
      p += (10.0 - storm) * 0.75 * max(0.0, math.cos(lon_r * 3.0 + lead_h * 0.06))

    # Model nuances
    if "aifs" in model_key:
      p = max(0.0, p * 1.04 - 0.04)
    elif "graphcast" in model_key:
      p = max(0.0, p * 0.98 + (0.1 if p > 0.5 else 0.0))
    elif "gfs" in model_key:
      p = max(0.0, p * 1.08)

    if var_key == "accumulated_precip":
      # Approximate cumulative rainfall up to lead_h
      return p * min(lead_h + 1, 24.0) * 0.4
    return p

  elif var_key == "temperature":
    # Thermal gradient from equator to poles + diurnal solar cycle
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

  elif var_key == "wind_u":
    # Trade easterlies (<30), Mid-lat westerlies (30-60), polar easterlies
    u_base = -7.0 * math.cos(lat_r * 3.0) + 12.0 * math.exp(
        -(((abs(lat_deg) - 48.0) / 14.0) ** 2)
    )
    return u_base + wave * 0.8

  elif var_key == "wind_v":
    v_base = 2.0 * math.sin(lat_r * 2.0)
    return v_base + math.cos(3 * lon_r + lead_h * 0.05) * 4.2 * s_lat

  elif var_key == "pressure":
    # Surface pressure centered around 1013.25 hPa
    # Subtropical highs at ~30 deg, subpolar lows at ~60 deg
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
  """Vectorized ``_procedural_sample`` on a lat x lon grid (rates, not sums)."""
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
    storm = np.abs(np.abs(lat) - 45.0 + 5.0 * np.sin(4 * lon_r + lead_h * 0.04))
    p = np.where(
        itcz < 8.0,
        (8.0 - itcz) * 0.95 * np.maximum(0.0, np.sin(lon_r * 4.0 + lead_h * 0.1)),
        0.0,
    ) + np.where(
        storm < 10.0,
        (10.0 - storm) * 0.75 * np.maximum(0.0, np.cos(lon_r * 3.0 + lead_h * 0.06)),
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
        (streams.get(f"{model_key}_{c}") or {}).get("archived_run") for c in ("u10", "v10")
    ):
      real_variables.append("wind")
  all_variables = ("precipitation", "accumulated_precip", "temperature", "pressure", "wind")
  return {
      "data_source": "archived_run" if real_variables else "synthetic",
      "init_time": init_time,
      "max_lead_hours": max_lead if real_variables else MAX_LEAD_HOURS,
      "real_variables": real_variables,
      "synthetic_variables": [v for v in all_variables if v not in real_variables],
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


def _tile_coordinates(z: int, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
  """Latitudes (rows) and longitudes (columns) of a tile's pixel centres."""
  n = 2.0**z
  frac = (np.arange(256, dtype=np.float64) + 0.5) / 256.0
  lons = (x + frac) / n * 360.0 - 180.0
  lats = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + frac) / n))))
  return lats, lons


def _tile_field(
    model_key: str, var_key: str, step_idx: int, lats: np.ndarray, lons: np.ndarray
) -> Optional[np.ndarray]:
  """Field values for a tile, or None when there is nothing to draw."""
  lead_h = step_idx * STEP_HOURS
  if var_key == "accumulated_precip":
    accumulated = _accumulated_global(model_key, lead_h)
    if accumulated is None:
      return None  # beyond the end of this run
    return _sample_global(accumulated[0], accumulated[1], lats, lons)
  suffix = _STREAM_SUFFIX.get(var_key)
  if suffix is None and var_key != "pressure":
    return None  # e.g. "wind" (drawn client-side) or "radar"
  stream_id = f"{model_key}_{suffix}" if suffix else None
  info = STREAM_INFO.get(stream_id) if stream_id else None
  if info:
    if suffix == "precip":
      rate_steps = _rate_file_steps(info, lead_h)
      if not rate_steps:
        return None  # beyond the end of this run
      return _mean_rate_grid(stream_id, rate_steps, lats, lons)
    file_step = _file_step_for_lead(info, lead_h, is_rate=False)
    if file_step is None:
      return None  # beyond the end of this run
    return _grid_values(stream_id, file_step, lats, lons)
  return _procedural_field(var_key, model_key, step_idx, lats, lons)


def _classify(values: np.ndarray, classes) -> np.ndarray:
  rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
  for lower, color in classes:
    rgba[values >= lower] = color
  return rgba


def _colorize(var_key: str, values: np.ndarray) -> np.ndarray:
  finite = np.isfinite(values)
  v = np.where(finite, values, 0.0)
  if var_key == "precipitation":
    rgba = _classify(v, RAIN_RATE_CLASSES)
  elif var_key == "accumulated_precip":
    rgba = _classify(v, RAIN_ACCUM_CLASSES)
  elif var_key == "temperature":
    # Turbo-like ramp from -35 °C (blue) to +45 °C (red).
    norm = np.clip((v + 35.0) / 80.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    for channel, centre in ((0, 3), (1, 2), (2, 1)):
      rgba[..., channel] = (
          255 * np.clip(1.5 - np.abs(norm * 4 - centre), 0.0, 1.0)
      ).astype(np.uint8)
    rgba[..., 3] = 180
  elif var_key == "pressure":
    # White isobars every 4 hPa over light blue shading (darker = higher).
    rem = np.abs(np.mod(v, 4.0))
    isobar = (rem < 0.3) | (rem > 3.7)
    norm = np.clip((v - 980.0) / 50.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 14, 165, 233
    rgba[..., 3] = (30 + norm * 50).astype(np.uint8)
    rgba[isobar] = (255, 255, 255, 220)
  else:
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
  rgba[~finite] = 0
  return rgba


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


# ---------------------------------------------------------------------------
# Animation frames: one whole-world Web Mercator image per forecast step.
#
# The viewer downloads every frame of the chosen model and layer once, then
# plays them like a video (see the Weather Viewer script in static/index.html).
# A frame is FRAME_SIZE x FRAME_SIZE pixels covering +-85.05 deg latitude and
# 360 deg longitude, i.e. 0.25 deg per pixel along the equator -- the data's own
# resolution, so nothing is lost compared with map tiles. Frames are palette
# PNGs (one byte per pixel), which keeps them small.
# ---------------------------------------------------------------------------
FRAME_SIZE = 1440
FRAME_VERSION = "1"  # Bump when frame colours or layout change (frames are cached by browsers).
MERCATOR_MAX_LAT = 85.0511287798
FRAME_VARIABLES = ("precipitation", "accumulated_precip", "temperature", "pressure")
_TEMP_LEVELS = 128
_PRESSURE_LEVELS = 51
_FRAME_CACHE: "collections.OrderedDict[Tuple[Any, ...], bytes]" = collections.OrderedDict()
_FRAME_CACHE_LIMIT_BYTES = 384 << 20
_FRAME_CACHE_BYTES = 0
_FRAME_LOCK = threading.Lock()
_FRAME_COORDS: Optional[Tuple[np.ndarray, np.ndarray]] = None
_EMPTY_FRAME: Optional[bytes] = None


def _clear_frame_cache():
  global _FRAME_CACHE_BYTES
  with _FRAME_LOCK:
    _FRAME_CACHE.clear()
    _FRAME_CACHE_BYTES = 0


def _frame_coordinates() -> Tuple[np.ndarray, np.ndarray]:
  """Latitudes (rows) and longitudes (columns) of the frame's pixel centres."""
  global _FRAME_COORDS
  if _FRAME_COORDS is None:
    frac = (np.arange(FRAME_SIZE, dtype=np.float64) + 0.5) / FRAME_SIZE
    lons = frac * 360.0 - 180.0
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * frac))))
    _FRAME_COORDS = (lats, lons)
  return _FRAME_COORDS


def _frame_signature(
    model_key: str, var_key: str, step: int
) -> Optional[Tuple[Tuple[Any, ...], float]]:
  """(what step `step` draws, the lead that data belongs to), or None if nothing.

  Steps with the same signature draw identical images, e.g. 3 h and 6 h for a
  6-hourly model. The lead is where that output naturally sits on the timeline
  (end of a rain interval, time of an instantaneous field).
  """
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
  """Which frame each 3-hourly viewer step shows, for /api/weather/frames/.../index.json.

  ``step_frames[s]`` is the step whose frame is drawn at step s (null = nothing
  to draw, e.g. beyond the end of the run). ``frame_steps`` lists the distinct
  frames in time order; the viewer plays only those, so a 6-hourly model is not
  shown as pairs of identical frames.
  """
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
          "archived_run" if var_key in data_info["real_variables"] else "synthetic"
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


def _box_smooth(values: np.ndarray) -> np.ndarray:
  padded = np.pad(values, 1, mode="edge")
  h, w = values.shape
  total = np.zeros_like(values, dtype=np.float32)
  for dy in range(3):
    for dx in range(3):
      total += padded[dy:dy + h, dx:dx + w]
  return total / 9.0


def _colorize_indexed(var_key: str, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Palette indices (uint8) and RGBA palette; index 0 is transparent.

  Same colours as _colorize (tiles) and the legend in static/index.html.
  """
  finite = np.isfinite(values)
  v = np.where(finite, values, 0.0).astype(np.float32)
  if var_key in ("precipitation", "accumulated_precip"):
    classes = RAIN_RATE_CLASSES if var_key == "precipitation" else RAIN_ACCUM_CLASSES
    bounds = np.array([lower for lower, _ in classes], dtype=np.float32)
    idx = np.searchsorted(bounds, v, side="right").astype(np.uint8)
    palette = np.array([(0, 0, 0, 0)] + [color for _, color in classes], dtype=np.uint8)
  elif var_key == "temperature":
    n = _TEMP_LEVELS
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
    n = _PRESSURE_LEVELS
    smooth = _box_smooth(v)
    norm = np.clip((smooth - 980.0) / 50.0, 0.0, 1.0)
    idx = (1 + np.rint(norm * (n - 1))).astype(np.uint8)
    # One-pixel white isobars every 4 hPa where the 4-hPa band changes.
    band = np.floor(smooth / 4.0)
    edge = np.zeros(band.shape, dtype=bool)
    edge[:-1, :] |= band[:-1, :] != band[1:, :]
    edge[:, :-1] |= band[:, :-1] != band[:, 1:]
    idx[edge] = n + 1
    levels = np.linspace(0.0, 1.0, n)
    palette = np.zeros((n + 2, 4), dtype=np.uint8)
    palette[1:n + 1, 0], palette[1:n + 1, 1], palette[1:n + 1, 2] = 14, 165, 233
    palette[1:n + 1, 3] = (30 + levels * 50).astype(np.uint8)
    palette[n + 1] = (255, 255, 255, 220)
  else:
    idx = np.zeros(v.shape, dtype=np.uint8)
    palette = np.zeros((1, 4), dtype=np.uint8)
  idx[~finite] = 0
  return idx, palette


def _png_chunk(tag: bytes, data: bytes) -> bytes:
  return (
      struct.pack("!I", len(data))
      + tag
      + data
      + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
  )


def _encode_indexed_png(idx: np.ndarray, palette: np.ndarray) -> bytes:
  """8-bit palette PNG with per-entry alpha (tRNS)."""
  height, width = idx.shape
  raw = np.zeros((height, width + 1), dtype=np.uint8)  # filter byte 0 per row
  raw[:, 1:] = idx
  return (
      b"\x89PNG\r\n\x1a\n"
      + _png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 3, 0, 0, 0))
      + _png_chunk(b"PLTE", palette[:, :3].astype(np.uint8).tobytes())
      + _png_chunk(b"tRNS", palette[:, 3].astype(np.uint8).tobytes())
      + _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
      + _png_chunk(b"IEND", b"")
  )


def _empty_frame() -> bytes:
  global _EMPTY_FRAME
  if _EMPTY_FRAME is None:
    _EMPTY_FRAME = _encode_indexed_png(
        np.zeros((1, 1), np.uint8), np.zeros((1, 4), np.uint8)
    )
  return _EMPTY_FRAME


def generate_weather_frame(model_key: str, var_key: str, step_idx: int) -> bytes:
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
  png = _empty_frame() if values is None else _encode_indexed_png(
      *_colorize_indexed(var_key, values)
  )
  with _FRAME_LOCK:
    if key not in _FRAME_CACHE:
      _FRAME_CACHE[key] = png
      _FRAME_CACHE_BYTES += len(png)
      while _FRAME_CACHE_BYTES > _FRAME_CACHE_LIMIT_BYTES and len(_FRAME_CACHE) > 1:
        _, old = _FRAME_CACHE.popitem(last=False)
        _FRAME_CACHE_BYTES -= len(old)
  return png


def _prewarm_frames():
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
          return  # a newer run was loaded meanwhile
        generate_weather_frame(model_key, "precipitation", step)
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("[WeatherEngine] Frame pre-render failed for %s: %s", model_key, e)


def get_wind_vectors(
    model_key: str = "ecmwf_ifs", step_idx: int = 0, subsample: int = 2
) -> Dict[str, Any]:
  """Returns downsampled global U/V vector matrices for client Canvas streamlines."""
  init_weather_streams()
  subsample = max(1, min(4, subsample))
  step_idx = max(0, int(step_idx))

  # Default subsample=2: ny=91, nx=180 (2 deg) for fast transfer.
  step_deg = 1.0 * subsample
  lats = np.array([90.0 - i * step_deg for i in range(int(180 / step_deg) + 1)])
  lons = np.array([-180.0 + j * step_deg for j in range(int(360 / step_deg))])
  lead_h = step_idx * STEP_HOURS

  u = v = None
  source = "synthetic"
  u_info = STREAM_INFO.get(f"{model_key}_u10")
  v_info = STREAM_INFO.get(f"{model_key}_v10")
  if u_info and v_info:
    fu = _file_step_for_lead(u_info, lead_h, is_rate=False)
    fv = _file_step_for_lead(v_info, lead_h, is_rate=False)
    if fu is not None and fv is not None:
      u = _grid_values(f"{model_key}_u10", fu, lats, lons)
      v = _grid_values(f"{model_key}_v10", fv, lats, lons)
      if u_info["archived_run"] and v_info["archived_run"]:
        source = "archived_run"
  if u is None:
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
          "data_source": source,
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
  info = STREAM_INFO.get(stream_id) if stream_id else None
  if info:
    rows, cols = _grid_indices(np.array([lat]), np.array([lon]))
    if suffix == "precip":
      rate_steps = _rate_file_steps(info, lead_h)
      if not rate_steps:
        return None
      if len(rate_steps) > 1:
        values = _ARRAYS[stream_id][rate_steps, rows[0], cols[0]].astype(np.float64)
        return float(np.nan_to_num(values, nan=0.0).mean())
      file_step = rate_steps[0]
    else:
      file_step = _file_step_for_lead(info, lead_h, is_rate=False)
      if file_step is None:
        return None
    value = float(_ARRAYS[stream_id][file_step, rows[0], cols[0]]) + info.get("offset", 0.0)
    return value if math.isfinite(value) else None
  return _procedural_sample(var_key, model_key, int(lead_h // STEP_HOURS), lat, lon)


def _accumulation_series(
    model_key: str, lat: float, lon: float, lead_hours: Sequence[float]
) -> List[Optional[float]]:
  """Rain accumulated since the forecast start at each lead (None beyond run)."""
  info = STREAM_INFO.get(f"{model_key}_precip")
  if not info:
    accum, out = 0.0, []
    for lead_h in lead_hours:
      accum += _procedural_sample(
          "precipitation", model_key, int(lead_h // STEP_HOURS), lat, lon
      ) * STEP_HOURS
      out.append(accum)
    return out
  rows, cols = _grid_indices(np.array([lat]), np.array([lon]))
  rates = np.nan_to_num(
      _ARRAYS[f"{model_key}_precip"][:, rows[0], cols[0]].astype(np.float64)
  )
  leads = info["lead_hours"]
  totals = [0.0]
  for i in range(1, len(leads)):
    totals.append(totals[-1] + rates[i] * (leads[i] - leads[i - 1]))
  out = []
  for lead_h in lead_hours:
    if lead_h > leads[-1]:
      out.append(None)
      continue
    idx = max(i for i, lead in enumerate(leads) if lead <= lead_h)
    out.append(totals[idx])
  return out


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
  if value is None or not math.isfinite(value):
    return None
  return round(value, digits)


def get_weather_probe(lat: float, lon: float) -> Dict[str, Any]:
  """Returns comparative 10-day multi-model meteorological soundings.

  Values beyond the end of an archived run are None (JSON null).
  """
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
          _round_or_none(_point_value(m_key, "precipitation", lead_h, lat, lon), 2)
      )
      temp_curve.append(
          _round_or_none(_point_value(m_key, "temperature", lead_h, lat, lon), 1)
      )
      u_w = _procedural_sample("wind_u", m_key, step, lat, lon)
      v_w = _procedural_sample("wind_v", m_key, step, lat, lon)
      wind_spd_curve.append(round(math.sqrt(u_w**2 + v_w**2), 1))
      wind_dir_curve.append(int((math.degrees(math.atan2(-u_w, -v_w)) + 360) % 360))
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


def _geometry_points(geom: Dict[str, Any]) -> Tuple[List[float], List[float]]:
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
        lons.append(pt[0])
        lats.append(pt[1])
  return lats, lons


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

  # Sample up to ~10 boundary points for speed.
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
      "basin_mean_precip_mmh": round(sum(rates) / len(rates), 2) if rates else None,
      "basin_max_precip_mmh": round(max(rates), 2) if rates else None,
      "basin_accumulated_10d_mm": _round_or_none(accum, 1),
      "basin_mean_temp_c": _round_or_none(temp_c, 1),
      "centroid": {"latitude": round(c_lat, 4), "longitude": round(c_lon, 4)},
      "model": model_key,
      "data_source": data_info["data_source"],
      "accumulation_hours": window_h,
  }
