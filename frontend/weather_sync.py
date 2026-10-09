#!/usr/bin/env python3
"""Server-side driver around ``multimet.weather_fetcher.sync``.

The library never picks directories, never swallows exceptions and never
fabricates data. This module adds what the long-running OpenHydroNet server
needs on top of it:

* the frontend data-directory convention (``$OPENHYDRONET_WEATHER_DATA_DIR`` or
  ``~/.cache/openhydronet/weather``);
* an advisory ``sync.lock`` so two server instances on one host never run the
  same synchronisation twice;
* failure isolation between upstream providers (dynamical.org, ECMWF Open
  Data, NOAA PSL): an unreachable provider is recorded per model in
  ``sync_status.json`` and does not stop the other providers from refreshing;
* a subprocess runner plus a daemon thread so the periodic check keeps large
  arrays out of the HTTP server process.

Run ``python frontend/weather_sync.py --data-root <dir>`` for one manual sync.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

_ws_root = str(Path(__file__).resolve().parents[1])
if _ws_root not in sys.path:
  sys.path.insert(0, _ws_root)

# pylint: disable=g-import-not-at-top,wrong-import-position
from multimet.weather_fetcher.config import (
    DEFAULT_SYNC_MODELS,
    DYNAMICAL_MODELS,
    MSLP_OFFSET_HPA,
    output_lead_hours,
    RUN_METADATA_FILE,
    SOURCE_LABELS,
    SYNC_STATUS_FILE,
    to_stored_units,
)
from multimet.weather_fetcher.sync import (
    aggregate_rates,
    current_models_metadata,
    current_run_dir as _current_run_dir_core,
    IncompleteRunError,
    read_sync_status as _read_sync_status_core,
    sync_all_models,
    utc_now_str as _utc_now_str,
    write_json_atomic as _write_json_atomic,
)

CHECK_INTERVAL_MINUTES = 60
SYNC_INTERVAL_HOURS = 6
SUBPROCESS_TIMEOUT_S = 45 * 60
SYNC_LOG_FILE = "sync.log"
# pylint: enable=g-import-not-at-top,wrong-import-position

__all__ = [
    "CHECK_INTERVAL_MINUTES",
    "IncompleteRunError",
    "MSLP_OFFSET_HPA",
    "RUN_METADATA_FILE",
    "aggregate_rates",
    "current_run_dir",
    "find_sync_python",
    "output_lead_hours",
    "read_sync_status",
    "run_sync_subprocess",
    "start_background_sync",
    "sync_latest",
    "to_stored_units",
    "weather_data_root",
]

logger = logging.getLogger(__name__)

_LOCK_FILE = "sync.lock"
_LOG_ROTATE_BYTES = 2 << 20
_SYNC_PYTHON_PROBE = "import icechunk, numpy, pystac, scipy, xarray"
_SYNC_PYTHON_PROBE_TIMEOUT_S = 120


from utils.file_paths import OPENHYDRONET_WEATHER_CACHE_DIR


def weather_data_root() -> str:
  """Local-disk root of downloaded runs used by the frontend server.

  ``$OPENHYDRONET_WEATHER_DATA_DIR`` overrides the default of
  ``~/.cache/openhydronet/weather``. The library itself takes the
  directory explicitly; this convention exists only for the server process.
  """
  env = os.environ.get("OPENHYDRONET_WEATHER_DATA_DIR", "").strip()
  if env:
    return str(Path(env).expanduser().resolve())
  return str(OPENHYDRONET_WEATHER_CACHE_DIR)


def current_run_dir(root: Optional[str] = None) -> Optional[str]:
  """Resolved directory the ``current`` symlink points at, or ``None``."""
  resolved = _current_run_dir_core(root or weather_data_root())
  return str(resolved) if resolved is not None else None


def read_sync_status(root: Optional[str] = None) -> Dict[str, Any]:
  """Reads ``sync_status.json`` from ``root`` (default: weather_data_root)."""
  return _read_sync_status_core(root or weather_data_root())


def _provider_of(model_key: str) -> str:
  """Upstream provider label of a model (raises for unsupported keys)."""
  if model_key not in DYNAMICAL_MODELS:
    raise ValueError(
        f"Unsupported weather model {model_key!r}. Supported:"
        f" {list(DYNAMICAL_MODELS)}"
    )
  return SOURCE_LABELS[DYNAMICAL_MODELS[model_key]["source"]]


def _provider_groups(models: Sequence[str]) -> List[List[str]]:
  """Splits models into one ordered group per upstream provider."""
  groups: Dict[str, List[str]] = {}
  for model_key in models:
    groups.setdefault(_provider_of(model_key), []).append(model_key)
  return list(groups.values())


def _remove_own_partial_dirs(root: str, log: Callable[[str], None]) -> None:
  """Deletes staging directories left by a failed sync of this process.

  ``sync_all_models`` names its staging directory ``<run>.<pid>.partial`` and
  cannot clean it up when an extraction raises (the library contains no
  exception handling by design). Removing our own leftovers here keeps later
  checks from treating them as a live synchronisation of this process.
  """
  suffix = f".{os.getpid()}.partial"
  runs_dir = Path(root) / "runs"
  if not runs_dir.is_dir():
    return
  for item in runs_dir.iterdir():
    if item.is_dir() and item.name.endswith(suffix):
      log(f"Removing staging directory {item.name} after a failed sync")
      shutil.rmtree(item, ignore_errors=True)


def _models_summary(root: str) -> Dict[str, Dict[str, Any]]:
  """Per-model summary of the published run (same shape as the library)."""
  _, current = current_models_metadata(root)
  return {
      key: {
          "init_time": entry.get("init_time"),
          "downloaded_utc": entry.get("downloaded_utc"),
          "source": entry.get("source"),
      }
      for key, entry in current.items()
  }


def _merge_outcome(
    status: Dict[str, Any],
    updated: List[str],
    errors: Dict[str, str],
    models: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
  """Sets ``last_result`` / ``message`` from the merged per-provider outcome."""
  downloaded = ", ".join(
      f"{key} {(models.get(key) or {}).get('init_time')}" for key in updated
  )
  failed = "; ".join(f"{key}: {msg}" for key, msg in errors.items())
  if updated and not errors:
    status["last_result"] = "updated"
    status["message"] = f"Downloaded new run: {downloaded}"
  elif updated:
    status["last_result"] = "partial"
    status["message"] = f"Downloaded new run: {downloaded}; failed: {failed}"
  elif errors:
    status["last_result"] = "error"
    status["message"] = failed
  else:
    status["last_result"] = "up_to_date"
    status["message"] = "All models already have the newest published run."
  return status


def sync_latest(
    root: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    force: bool = False,
    log: Callable[[str], None] = print,
    catalog: Any = None,
    open_dataset: Optional[Callable[[Any, str], Any]] = None,
    cpc_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
  """Refreshes every selected model whose newest upstream run changed.

  Models are synchronised provider by provider through
  ``multimet.weather_fetcher.sync.sync_all_models``. An exception raised while
  talking to one provider (catalog unreachable, GRIB download failed, ...) is
  recorded for that provider's models in ``errors`` and the remaining
  providers are still processed; nothing is retried with substitute data.

  Args:
    root: Data directory (default ``weather_data_root()``).
    models: Model keys (default ``DEFAULT_SYNC_MODELS``).
    force: Re-download even when the newest run is already published.
    log: Line logger.
    catalog: Optional pre-opened STAC catalog (or test stub).
    open_dataset: Optional ``(catalog, dataset_id) -> xr.Dataset`` opener.
    cpc_cache_dir: NOAA PSL CPC NetCDF cache (default ``<root>/cpc_cache``).

  Returns:
    The merged status written to ``<root>/sync_status.json``; ``last_result``
    is ``updated``, ``partial``, ``error``, ``up_to_date`` or ``busy``.

  Raises:
    ValueError: If ``models`` contains an unsupported key.
  """
  resolved_root = str(root or weather_data_root())
  os.makedirs(os.path.join(resolved_root, "runs"), exist_ok=True)
  selected = list(models) if models is not None else list(DEFAULT_SYNC_MODELS)
  groups = _provider_groups(selected)
  status_path = os.path.join(resolved_root, SYNC_STATUS_FILE)

  with open(os.path.join(resolved_root, _LOCK_FILE), "w") as lock_file:
    try:
      fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      status = read_sync_status(resolved_root)
      status.update({
          "last_check_utc": _utc_now_str(),
          "last_result": "busy",
          "message": (
              "Another weather sync holds"
              f" {os.path.join(resolved_root, _LOCK_FILE)}; skipping."
          ),
      })
      log(status["message"])
      return status

    updated: List[str] = []
    errors: Dict[str, str] = {}
    for group in groups:
      try:
        result = sync_all_models(
            data_dir=resolved_root,
            models=group,
            force=force,
            log=log,
            catalog=catalog,
            open_dataset=open_dataset,
            cpc_cache_dir=cpc_cache_dir,
        )
      except Exception as exc:  # pylint: disable=broad-except
        # Provider isolation: the failure is recorded per model below and
        # surfaces in sync_status.json; it is never replaced by other data.
        _remove_own_partial_dirs(resolved_root, log)
        message = f"{type(exc).__name__}: {exc}"
        log(f"[{', '.join(group)}] sync failed: {message}")
        for model_key in group:
          errors[model_key] = message
        continue
      if result.get("last_result") == "busy":
        return result
      updated.extend(result.get("updated_models", []))
      errors.update(result.get("errors", {}))

    status = read_sync_status(resolved_root)
    sources = {key: _provider_of(key) for key in selected}
    status.update({
        "last_check_utc": _utc_now_str(),
        "check_interval_minutes": CHECK_INTERVAL_MINUTES,
        "source": ", ".join(dict.fromkeys(sources.values())),
        "sources": sources,
        "updated_models": updated,
        "errors": errors,
        "models": _models_summary(resolved_root),
    })
    _merge_outcome(status, updated, errors, status["models"])
    _write_json_atomic(status_path, status)
    log(f"{status['last_result']}: {status['message']}")
    return status


def find_sync_python() -> Optional[str]:
  """Interpreter able to import the sync dependencies, or ``None``.

  Candidates, in order: ``$EARTHKIT_WEATHER_SYNC_PYTHON``, the running
  interpreter, ``/usr/bin/python3`` and ``python3`` on ``PATH``.
  """
  candidates = [
      os.environ.get("EARTHKIT_WEATHER_SYNC_PYTHON"),
      sys.executable,
      "/usr/bin/python3",
      shutil.which("python3"),
  ]
  seen = set()
  for python in candidates:
    if not python or python in seen or not os.path.exists(python):
      continue
    seen.add(python)
    try:
      probe = subprocess.run(
          [python, "-c", _SYNC_PYTHON_PROBE],
          capture_output=True,
          timeout=_SYNC_PYTHON_PROBE_TIMEOUT_S,
          check=False,
      )
    except (OSError, subprocess.SubprocessError) as exc:
      logger.warning("Probing %s failed: %s", python, exc)
      continue
    if probe.returncode == 0:
      return python
  return None


def _append_log(root: str, text: str) -> None:
  """Appends to ``<root>/<SYNC_LOG_FILE>`` (rotated once past 2 MiB)."""
  path = os.path.join(root, SYNC_LOG_FILE)
  try:
    if os.path.exists(path) and os.path.getsize(path) > _LOG_ROTATE_BYTES:
      os.replace(path, path + ".1")
    with open(path, "a", encoding="utf-8") as f_log:
      f_log.write(text)
  except OSError as exc:
    logger.warning("Could not append to weather sync log %s: %s", path, exc)


def run_sync_subprocess(
    root: Optional[str] = None,
    python: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
  """Runs one sync in a separate process and returns the resulting status."""
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
            "No Python interpreter able to import icechunk, pystac, scipy and"
            " xarray was found (set EARTHKIT_WEATHER_SYNC_PYTHON)."
        ),
    })
    _write_json_atomic(os.path.join(resolved_root, SYNC_STATUS_FILE), status)
    return status
  cmd = [python, os.path.abspath(__file__), "--data-root", resolved_root]
  if models is not None:
    cmd.extend(["--models", ",".join(models)])
  if force:
    cmd.append("--force")
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
        f"\n=== {started} {' '.join(cmd)} (exit {proc.returncode})\n"
        f"{proc.stdout}{proc.stderr}",
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
  """Starts the periodic check in a daemon thread (once per process).

  ``EARTHKIT_WEATHER_SYNC=0`` disables it; ``EARTHKIT_WEATHER_SYNC_MINUTES``
  or ``interval_minutes`` override ``CHECK_INTERVAL_MINUTES``.
  """
  global _SYNC_THREAD  # pylint: disable=global-statement
  if _SYNC_THREAD is not None:
    return _SYNC_THREAD
  if os.environ.get("EARTHKIT_WEATHER_SYNC", "1") == "0":
    return None
  interval_s = 60.0 * float(
      interval_minutes
      or os.environ.get("EARTHKIT_WEATHER_SYNC_MINUTES")
      or CHECK_INTERVAL_MINUTES
  )

  def loop() -> None:
    time.sleep(first_delay_s)
    python = find_sync_python()
    while True:
      try:
        status = run_sync_subprocess(python=python)
        print(
            f"[WeatherSync] {status.get('last_result')}:"
            f" {status.get('message')}",
            flush=True,
        )
      except Exception as exc:  # pylint: disable=broad-except
        # The daemon thread must survive to try again next interval.
        print(f"[WeatherSync] check failed: {exc!r}", flush=True)
      if on_finished:
        try:
          on_finished()
        except Exception as exc:  # pylint: disable=broad-except
          print(f"[WeatherSync] reload failed: {exc!r}", flush=True)
      time.sleep(interval_s)

  _SYNC_THREAD = threading.Thread(
      target=loop, name="weather-sync", daemon=True
  )
  _SYNC_THREAD.start()
  return _SYNC_THREAD


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description=(
          "Synchronise the newest weather model runs into the frontend data"
          " directory (one pass, then exit)."
      )
  )
  parser.add_argument(
      "--data-root",
      type=str,
      default=None,
      help=(
          "Root directory for downloaded runs (default:"
          " $OPENHYDRONET_WEATHER_DATA_DIR or ~/.cache/openhydronet/weather)."
      ),
  )
  parser.add_argument(
      "--cpc-cache-dir",
      type=str,
      default=None,
      help="NOAA PSL CPC NetCDF cache (default: <data-root>/cpc_cache).",
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
      help="Download even if the newest run is already published.",
  )
  parser.add_argument(
      "--fetch-latest",
      action="store_true",
      help="Accepted for backwards compatibility; a sync is always performed.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Runs one synchronisation; exit code 0 unless a model failed."""
  args = _build_parser().parse_args(argv)
  result = sync_latest(
      root=args.data_root,
      models=[m.strip() for m in args.models.split(",") if m.strip()] or None,
      force=args.force,
      log=lambda msg: print(msg, flush=True),
      cpc_cache_dir=args.cpc_cache_dir,
  )
  ok_results = ("updated", "up_to_date", "busy")
  return 0 if result.get("last_result") in ok_results else 1


if __name__ == "__main__":
  sys.exit(main())
