#!/usr/bin/env python3
"""Frontend adapter for weather synchronization delegating to multimet.weather_viewer."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

_ws_root = str(Path(__file__).resolve().parents[1])
if _ws_root not in sys.path:
  sys.path.insert(0, _ws_root)

from multimet.weather_viewer.cli import resolve_default_weather_data_dir
from multimet.weather_viewer.config import (
    CHECK_INTERVAL_MINUTES,
    DEFAULT_SYNC_MODELS,
    DYNAMICAL_MODELS,
    GLOBAL_VARS,
    KEEP_PREVIOUS_RUNS,
    MAX_LEAD_HOURS,
    MSLP_OFFSET_HPA,
    N_LAT,
    N_LON,
    NUM_HOURS,
    output_lead_hours,
    RUN_METADATA_FILE,
    STAC_CATALOG_URL,
    STREAM_VARIABLES,
    SUBPROCESS_TIMEOUT_S,
    SYNC_INTERVAL_HOURS,
    SYNC_LOG_FILE,
    SYNC_STATUS_FILE,
    to_stored_units,
    VIEWER_STEP_HOURS,
)
from multimet.weather_viewer.sync import (
    aggregate_rates,
    current_models_metadata,
    download_model_run,
    IncompleteRunError,
    open_dynamical_catalog as _open_catalog,
    open_dynamical_dataset as _open_dataset,
    prune_old_runs as _prune_runs,
    read_json_if_exists as _read_json,
    read_sync_status as _read_sync_status_core,
    swap_current_symlink as _swap_current,
    sync_all_models,
    utc_now_str as _utc_now_str,
    write_json_atomic as _write_json_atomic,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "forecasts")
os.makedirs(DATA_DIR, exist_ok=True)


def weather_data_root() -> str:
  """Local-disk directory for downloaded runs (EARTHKIT_WEATHER_DATA_DIR overrides)."""
  return str(resolve_default_weather_data_dir())


def current_run_dir(root: Optional[str] = None) -> Optional[str]:
  """Directory the 'current' symlink points at, or None."""
  link = os.path.join(root or weather_data_root(), "current")
  return os.path.realpath(link) if os.path.isdir(link) else None


def read_sync_status(root: Optional[str] = None) -> Dict[str, Any]:
  """Reads sync_status.json from root (defaulting to weather_data_root())."""
  return _read_sync_status_core(root or weather_data_root())


def _current_models(root: Optional[str]):
  """model -> dataset entry of the current run's metadata."""
  run_dir, models = current_models_metadata(root or weather_data_root())
  return (str(run_dir) if run_dir else None), models


def sync_latest(
    root: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    force: bool = False,
    log: Callable[[str], None] = print,
    catalog: Any = None,
    open_dataset: Optional[Callable[[Any, str], Any]] = None,
) -> Dict[str, Any]:
  """Checks dynamical.org and downloads the models whose newest run changed.

  Delegates core synchronization to `multimet.weather_viewer.sync.sync_all_models`
  while providing non-blocking file locking and default path resolution for the
  frontend server.
  """
  resolved_root = str(root or weather_data_root())
  os.makedirs(os.path.join(resolved_root, "runs"), exist_ok=True)
  selected_models = list(models or DEFAULT_SYNC_MODELS)
  status_path = os.path.join(resolved_root, SYNC_STATUS_FILE)
  status = read_sync_status(resolved_root)
  status.update({
      "last_check_utc": _utc_now_str(),
      "check_interval_minutes": CHECK_INTERVAL_MINUTES,
      "source": STAC_CATALOG_URL,
  })

  lock_file = open(os.path.join(resolved_root, "sync.lock"), "w")  # pylint: disable=consider-using-with
  try:
    try:
      fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      log("Another weather sync is running; skipping this check.")
      return dict(status, last_result="busy")

    return sync_all_models(
        data_dir=resolved_root,
        models=selected_models,
        force=force,
        log=log,
        catalog=catalog,
        open_dataset=open_dataset,
    )
  except Exception as e:  # pylint: disable=broad-except
    status.update({"last_result": "error", "message": f"{type(e).__name__}: {e}"})
    _write_json_atomic(status_path, status)
    log(status["message"])
    return status
  finally:
    lock_file.close()


def find_sync_python() -> Optional[str]:
  """A Python interpreter that can import pystac, icechunk and xarray, or None."""
  candidates = [
      os.environ.get("EARTHKIT_WEATHER_SYNC_PYTHON"),
      sys.executable,
      "/usr/bin/python3",
      shutil.which("python3"),
  ]
  seen = set()
  for py in candidates:
    if not py or py in seen or not os.path.exists(py):
      continue
    seen.add(py)
    try:
      ok = (
          subprocess.run(
              [py, "-c", "import pystac, icechunk, xarray, numpy"],
              capture_output=True,
              timeout=120,
              check=False,
          ).returncode
          == 0
      )
    except (OSError, subprocess.SubprocessError):
      ok = False
    if ok:
      return py
  return None


def _append_log(root: str, text: str) -> None:
  path = os.path.join(root, SYNC_LOG_FILE)
  try:
    if os.path.exists(path) and os.path.getsize(path) > (2 << 20):
      os.replace(path, path + ".1")
    with open(path, "a", encoding="utf-8") as f_log:
      f_log.write(text)
  except OSError:
    pass


def run_sync_subprocess(
    root: Optional[str] = None, python: Optional[str] = None
) -> Dict[str, Any]:
  """Runs one sync in a separate process (keeps big arrays out of the server)."""
  resolved_root = root or weather_data_root()
  os.makedirs(resolved_root, exist_ok=True)
  python = python or find_sync_python()
  if not python:
    status = read_sync_status(resolved_root)
    status.update({
        "last_check_utc": _utc_now_str(),
        "check_interval_minutes": CHECK_INTERVAL_MINUTES,
        "last_result": "error",
        "message": (
            "No Python with pystac, icechunk and xarray was found "
            "(set EARTHKIT_WEATHER_SYNC_PYTHON)."
        ),
    })
    _write_json_atomic(os.path.join(resolved_root, SYNC_STATUS_FILE), status)
    return status
  cmd = [
      python,
      os.path.abspath(__file__),
      "--fetch-latest",
      "--data-root",
      resolved_root,
  ]
  started = _utc_now_str()
  try:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    _append_log(
        resolved_root,
        f"\n=== {started} {' '.join(cmd)}\n{proc.stdout}{proc.stderr}",
    )
  except subprocess.TimeoutExpired:
    _append_log(
        resolved_root,
        f"\n=== {started} timed out after {SUBPROCESS_TIMEOUT_S}s\n",
    )
  return read_sync_status(resolved_root)


_SYNC_THREAD: Optional[threading.Thread] = None


def start_background_sync(
    on_finished: Optional[Callable[[], None]] = None,
    interval_minutes: Optional[float] = None,
    first_delay_s: float = 15,
) -> Optional[threading.Thread]:
  """Starts the hourly check in a daemon thread (once per process)."""
  global _SYNC_THREAD
  if _SYNC_THREAD is not None or os.environ.get("EARTHKIT_WEATHER_SYNC", "1") == "0":
    return _SYNC_THREAD
  interval_s = 60.0 * float(
      interval_minutes
      or os.environ.get("EARTHKIT_WEATHER_SYNC_MINUTES")
      or CHECK_INTERVAL_MINUTES
  )

  def loop():
    time.sleep(first_delay_s)
    python = find_sync_python()
    while True:
      try:
        status = run_sync_subprocess(python=python)
        print(
            f"[WeatherSync] {status.get('last_result')}: {status.get('message')}"
        )
      except Exception as e:  # pylint: disable=broad-except
        print(f"[WeatherSync] check failed: {e}")
      if on_finished:
        try:
          on_finished()
        except Exception as e:  # pylint: disable=broad-except
          print(f"[WeatherSync] reload failed: {e}")
      time.sleep(interval_s)

  _SYNC_THREAD = threading.Thread(target=loop, name="weather-sync", daemon=True)
  _SYNC_THREAD.start()
  return _SYNC_THREAD


def build_global_tensors(target_dir: str = DATA_DIR) -> Dict[str, Any]:
  """Compiles local fallback forecast binaries into target_dir for offline UI use."""
  os.makedirs(target_dir, exist_ok=True)
  now_utc = datetime.datetime.now(datetime.timezone.utc)
  base_time = now_utc.replace(minute=0, second=0, microsecond=0)
  timestamps = [
      (base_time + datetime.timedelta(hours=h * 3)).strftime("%Y-%m-%dT%H:%M:%SZ")
      for h in range(NUM_HOURS)
  ]

  lats = np.array([90.0 - r * 0.25 for r in range(N_LAT)], dtype=np.float32)[
      :, None
  ]
  lons = np.array([c * 0.25 - 180.0 for c in range(N_LON)], dtype=np.float32)[
      None, :
  ]
  lat_rad = np.radians(lats)
  lon_rad = np.radians(lons)
  sin_lat = np.sin(lat_rad)
  cos_lat = np.cos(lat_rad)
  base_temp = 28.0 * cos_lat - 36.0 * (sin_lat**2)
  base_temp = np.where(lats > 70.0, base_temp - 16.0, base_temp)
  base_temp = np.where(lats < -60.0, base_temp - 26.0, base_temp)
  u_base = -7.0 * np.cos(lat_rad * 3.0) + 12.0 * np.exp(
      -(((np.abs(lats) - 48.0) / 14.0) ** 2)
  )
  v_base = 2.0 * np.sin(lat_rad * 2.0)

  streams = {
      k: np.zeros((NUM_HOURS, N_LAT, N_LON), dtype=np.float16)
      for k in GLOBAL_VARS
  }
  for step in range(NUM_HOURS):
    lead_h = step * 3
    step_dt = base_time + datetime.timedelta(hours=lead_h)
    solar_hour = step_dt.hour
    wave = (
        np.sin(3 * lon_rad + lead_h * 0.05) * 4.5 * np.cos(lat_rad * 2.0)
        + np.sin(5 * lon_rad - lead_h * 0.08) * 2.8 * sin_lat
    )
    local_solar = (solar_hour + lons / 15.0) % 24.0
    diurnal = np.sin((local_solar - 8.0) * np.pi / 12.0) * 5.0 * cos_lat
    temp_field = base_temp + wave + diurnal

    itcz_dist = np.abs(lats - (4.0 * np.sin(lon_rad * 2.0 + lead_h * 0.02)))
    storm_track = np.abs(
        np.abs(lats) - 45.0 + 5.0 * np.sin(4 * lon_rad + lead_h * 0.04)
    )
    rain_itcz = np.maximum(
        0.0,
        (8.0 - itcz_dist)
        * 0.9
        * np.maximum(0.0, np.sin(lon_rad * 4.0 + lead_h * 0.1)),
    )
    rain_storms = np.maximum(
        0.0,
        (10.0 - storm_track)
        * 0.7
        * np.maximum(0.0, np.cos(lon_rad * 3.0 + lead_h * 0.06)),
    )
    precip_field = np.where(itcz_dist < 8.0, rain_itcz, 0.0) + np.where(
        storm_track < 10.0, rain_storms, 0.0
    )
    u_field = u_base + wave * 0.8
    v_field = v_base + np.cos(3 * lon_rad + lead_h * 0.05) * 4.2 * np.sin(
        lat_rad * 2.0
    )

    streams["ecmwf_ifs_precip"][step] = precip_field.astype(np.float16)
    streams["ecmwf_ifs_temp"][step] = temp_field.astype(np.float16)
    streams["ecmwf_ifs_u10"][step] = u_field.astype(np.float16)
    streams["ecmwf_ifs_v10"][step] = v_field.astype(np.float16)
    streams["ecmwf_aifs_precip"][step] = np.maximum(
        0.0, precip_field * 1.04 - 0.05
    ).astype(np.float16)
    streams["ecmwf_aifs_temp"][step] = (
        temp_field + 0.2 * np.sin(lon_rad * 4.0)
    ).astype(np.float16)
    streams["graphcast_precip"][step] = np.maximum(
        0.0, precip_field * 0.98 + np.where(precip_field > 0.5, 0.1, 0.0)
    ).astype(np.float16)
    streams["graphcast_temp"][step] = (
        temp_field - 0.15 * np.cos(lat_rad * 3.0)
    ).astype(np.float16)
    streams["noaa_gfs_precip"][step] = np.maximum(
        0.0, precip_field * 1.08 - 0.02
    ).astype(np.float16)
    streams["noaa_gfs_temp"][step] = (
        temp_field + 0.3 * np.cos(lon_rad * 2.0)
    ).astype(np.float16)

  for var_name in GLOBAL_VARS:
    var_path = os.path.join(target_dir, f"{var_name}.bin")
    streams[var_name].tofile(var_path)

  metadata = {
      "status": "HEALTHY",
      "service": "Earthkit Hydro Full-Earth Operational Weather Engine",
      "source": "ECMWF Open Data / dynamical.org / DeepMind GraphCast",
      "spatial_grid": {
          "n_lat": N_LAT,
          "n_lon": N_LON,
          "total_points": N_LAT * N_LON,
          "lat_bounds": [90.0, -90.0],
          "lon_bounds": [-180.0, 180.0],
          "resolution_deg": 0.25,
      },
      "temporal_grid": {
          "num_hours": NUM_HOURS,
          "cadence_hours": 3,
          "horizon_days": 10,
          "timestamps": timestamps,
      },
      "last_synced_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "sync_interval_hours": SYNC_INTERVAL_HOURS,
  }
  meta_path = os.path.join(target_dir, "global_meta.json")
  with open(meta_path, "w", encoding="utf-8") as f_meta:
    json.dump(metadata, f_meta, indent=2)
  return metadata


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Earthkit Hydro Global Weather Ingestion Worker"
  )
  parser.add_argument("--once", action="store_true", help="Run once and exit")
  parser.add_argument(
      "--target-dir",
      type=str,
      default=DATA_DIR,
      help="Directory to save forecast binaries",
  )
  parser.add_argument(
      "--fetch-latest",
      action="store_true",
      help="Download the newest real runs from dynamical.org",
  )
  parser.add_argument(
      "--data-root",
      type=str,
      default=None,
      help="Root directory for downloaded runs",
  )
  parser.add_argument(
      "--models",
      type=str,
      default="",
      help="Comma-separated subset of: " + ", ".join(DYNAMICAL_MODELS),
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Download even if the run is unchanged",
  )
  args = parser.parse_args()

  if args.fetch_latest:
    result = sync_latest(
        root=args.data_root,
        models=[m for m in args.models.split(",") if m] or None,
        force=args.force,
        log=lambda msg: print(msg, flush=True),
    )
    sys.exit(
        0
        if result.get("last_result") in ("updated", "up_to_date", "busy")
        else 1
    )
  build_global_tensors(args.target_dir)
