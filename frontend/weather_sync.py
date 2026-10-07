#!/usr/bin/env python3
"""Frontend adapter for weather synchronization delegating to multimet.weather_fetcher."""

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

from multimet.weather_fetcher.cli import resolve_default_weather_data_dir
from multimet.weather_fetcher.config import (
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
from multimet.weather_fetcher.sync import (
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

  Delegates core synchronization to `multimet.weather_fetcher.sync.sync_all_models`
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
