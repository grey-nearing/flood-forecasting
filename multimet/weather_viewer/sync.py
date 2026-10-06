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

"""Scheduled and on-demand forecast run downloader with atomic directory swap."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from multimet.weather_viewer.config import (
    CHECK_INTERVAL_MINUTES,
    DEFAULT_MSLP_OFFSET_HPA,
    DEFAULT_SYNC_MODELS,
    DYNAMICAL_MODELS,
    KEEP_PREVIOUS_RUNS,
    MAX_LEAD_HOURS,
    MSLP_OFFSET_HPA,
    output_lead_hours,
    RUN_DATASET_TO_MODEL,
    RUN_METADATA_FILE,
    STAC_CATALOG_URL,
    STREAM_VARIABLES,
    SUPPORTED_MODELS,
    SYNC_STATUS_FILE,
    to_stored_units,
)


class IncompleteRunError(RuntimeError):
  """Raised when the newest published forecast run has incomplete lead steps."""


def utc_now_str() -> str:
  """Returns current UTC timestamp in ISO-8601 format."""
  return datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%dT%H:%M:%SZ"
  )


def require_data_dir(data_dir: Union[str, Path, None]) -> Path:
  """Validates that an explicit data_dir path was supplied."""
  if data_dir is None or str(data_dir).strip() == "":
    raise ValueError(
        "An explicit data_dir path is required for multimet.weather_viewer."
    )
  return Path(data_dir).expanduser().resolve()


def aggregate_rates(
    rates: Any,
    in_leads: Sequence[int],
    out_leads: Sequence[int],
) -> np.ndarray:
  """Computes mean rate over each output interval (previous output lead, lead].

  rates[i] is the model's mean rate over (in_leads[i-1], in_leads[i]].
  Hourly GFS rain is averaged over each 3-hour viewer step instead of sampling
  one hour in three. Lead 0 has no preceding interval and is all zeros.

  Args:
    rates: Input rate array of shape (n_in_leads, ...).
    in_leads: Input lead hours corresponding to axis 0 of ``rates``.
    out_leads: Output viewer lead hours.

  Returns:
    Float32 array of shape (len(out_leads), ...).
  """
  rates_arr = np.asarray(rates, dtype=np.float32)
  in_leads_list = [int(h) for h in in_leads]
  out = np.zeros((len(out_leads),) + rates_arr.shape[1:], dtype=np.float32)
  prev: Optional[int] = None
  for j, lead in enumerate(out_leads):
    lead_int = int(lead)
    if prev is not None and lead_int > prev:
      total = np.zeros(rates_arr.shape[1:], dtype=np.float32)
      for i in range(1, len(in_leads_list)):
        if prev < in_leads_list[i] <= lead_int:
          dt = in_leads_list[i] - in_leads_list[i - 1]
          total += np.nan_to_num(rates_arr[i], nan=0.0) * dt
      out[j] = total / float(lead_int - prev)
    prev = lead_int
  return out


def write_json_atomic(path: Union[str, Path], data: Mapping[str, Any]) -> None:
  """Atomically writes a JSON dictionary to disk."""
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  tmp = Path(f"{target}.tmp{os.getpid()}")
  tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
  os.replace(tmp, target)


def read_json_if_exists(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
  """Reads a JSON file if it exists and is non-empty."""
  target = Path(path)
  if not target.exists() or not target.is_file() or target.stat().st_size == 0:
    return None
  return json.loads(target.read_text(encoding="utf-8"))


def current_run_dir(data_dir: Union[str, Path]) -> Optional[Path]:
  """Returns the resolved directory that '<data_dir>/current' points to, or None."""
  root = require_data_dir(data_dir)
  link = root / "current"
  if link.exists() and link.is_dir():
    return link.resolve()
  if root.exists() and any(root.glob("*.bin")):
    return root.resolve()
  return None


def read_sync_status(data_dir: Union[str, Path]) -> Dict[str, Any]:
  """Reads sync_status.json from data_dir if present."""
  root = require_data_dir(data_dir)
  return read_json_if_exists(root / SYNC_STATUS_FILE) or {}


def load_run_metadata(target_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
  """Reads archived-run metadata (init time, lead steps) keyed by model."""
  folder = require_data_dir(target_dir)
  meta = read_json_if_exists(folder / RUN_METADATA_FILE)
  if not meta:
    return {}
  runs: Dict[str, Dict[str, Any]] = {}
  for dataset_key, dataset in (meta.get("datasets") or {}).items():
    model_key = dataset.get("model") or RUN_DATASET_TO_MODEL.get(dataset_key)
    if model_key not in SUPPORTED_MODELS or not dataset.get("init_time"):
      continue
    init_time = str(dataset["init_time"])
    if not init_time.endswith("Z"):
      init_time += "Z"
    lead_hours = dataset.get("lead_hours")
    runs[model_key] = {
        "init_time": init_time,
        "lead_steps": int(dataset.get("lead_steps") or 0),
        "lead_hours": [int(h) for h in lead_hours] if lead_hours else None,
        "downloaded_utc": (
            dataset.get("downloaded_utc") or meta.get("last_updated_utc")
        ),
        "title": dataset.get("title"),
        "mslp_offset_hpa": float(
            dataset.get("mslp_offset_hpa", DEFAULT_MSLP_OFFSET_HPA)
        ),
    }
  return runs


def current_models_metadata(
    data_dir: Union[str, Path],
) -> Tuple[Optional[Path], Dict[str, Dict[str, Any]]]:
  """Returns (active_run_dir, model_key -> dataset_entry) for the current run."""
  root = require_data_dir(data_dir)
  run_dir = current_run_dir(root)
  meta = read_json_if_exists(run_dir / RUN_METADATA_FILE) if run_dir else None
  models: Dict[str, Dict[str, Any]] = {}
  for entry in ((meta or {}).get("datasets") or {}).values():
    if entry.get("model"):
      models[entry["model"]] = entry
  return run_dir, models


def open_dynamical_catalog(catalog_url: str = STAC_CATALOG_URL) -> Any:
  """Opens the dynamical.org STAC catalog."""
  import pystac

  return pystac.Catalog.from_file(catalog_url)


def open_dynamical_dataset(catalog: Any, dataset_id: str) -> xr.Dataset:
  """Opens an Icechunk Zarr store for a dynamical.org dataset."""
  import icechunk

  collection = catalog.get_child(dataset_id)
  if collection is None:
    raise ValueError(f"{dataset_id} is not in {STAC_CATALOG_URL}")
  repo = icechunk.Repository.open(
      icechunk.http_storage(collection.assets["icechunk-https"].href)
  )
  return xr.open_zarr(repo.readonly_session("main").store, chunks=None)


def latest_init_time_str(ds: xr.Dataset) -> str:
  """Returns the latest init_time coordinate as an ISO-8601 string (unit='s')."""
  return str(np.datetime_as_string(ds["init_time"].values[-1], unit="s"))


def is_plane_complete(planes: np.ndarray, stream: str) -> bool:
  """Checks whether the final forecast lead plane has been populated."""
  if planes.shape[0] == 0:
    return False
  last_plane = planes[-1]
  if float(np.isnan(last_plane).mean()) > 0.5:
    return False
  if stream != "precip" and bool(np.isnan(last_plane).all()):
    return False
  return True


def _extract_model_streams(
    ds: xr.Dataset,
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Extracts binary streams for a model run, returning (entry, error_msg)."""
  init_val = ds["init_time"].values[-1]
  init_str = str(np.datetime_as_string(init_val, unit="s"))
  all_leads = (ds["lead_time"].values / np.timedelta64(1, "h")).astype(int)
  in_idx = [i for i, h in enumerate(all_leads) if 0 <= h <= MAX_LEAD_HOURS]
  in_leads = [int(all_leads[i]) for i in in_idx]
  out_leads = output_lead_hours(in_leads)
  out_pos = [in_leads.index(h) for h in out_leads]
  sel: Dict[str, Any] = {"init_time": init_val}
  if "ensemble_member" in ds.dims:
    sel["ensemble_member"] = cfg.get("ensemble_member", 0)

  streams: List[str] = []
  for stream in cfg["streams"]:
    var = STREAM_VARIABLES[stream]
    if var not in ds:
      log(f"   [!] {cfg['dataset']} has no {var}; skipped")
      continue
    t0 = time.time()
    raw = ds[var].sel(**sel).isel(lead_time=in_idx).values.astype(np.float32)
    if stream == "precip":
      planes = aggregate_rates(raw, in_leads, out_leads)
    else:
      planes = raw[out_pos]
    del raw
    if not is_plane_complete(planes, stream):
      return None, f"{cfg['dataset']} {init_str}: {var} not complete yet"
    out_file = out_dir / f"{model_key}_{stream}.bin"
    to_stored_units(stream, planes).tofile(str(out_file))
    streams.append(stream)
    log(
        f"   -> {model_key}_{stream}.bin: {len(out_leads)} leads in"
        f" {time.time() - t0:.1f}s"
    )
  if not {"precip", "temp"} <= set(streams):
    return None, f"{cfg['dataset']}: rain or temperature missing"
  entry = {
      "id": cfg["dataset"],
      "type": "forecast",
      "model": model_key,
      "title": cfg["title"],
      "init_time": init_str,
      "lead_steps": len(out_leads),
      "lead_hours": out_leads,
      "streams": streams,
      "variables": [STREAM_VARIABLES[s] for s in streams],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "downloaded_utc": utc_now_str(),
  }
  return entry, None


def download_model_run(
    ds: xr.Dataset,
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Union[str, Path],
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
  """Writes <model>_<stream>.bin files for the newest run; returns metadata."""
  target_dir = require_data_dir(out_dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  entry, err = _extract_model_streams(ds, model_key, cfg, target_dir, log=log)
  if err is not None:
    if "not complete yet" in err:
      raise IncompleteRunError(err)
    raise RuntimeError(err)
  assert entry is not None
  return entry


def swap_current_symlink(data_dir: Union[str, Path], run_name: str) -> Path:
  """Atomically repoints <data_dir>/current to runs/<run_name>."""
  root = require_data_dir(data_dir)
  tmp = root / f"current.tmp{os.getpid()}"
  if os.path.lexists(tmp):
    tmp.unlink()
  os.symlink(Path("runs") / run_name, tmp)
  current_link = root / "current"
  os.replace(tmp, current_link)
  return current_link


def prune_old_runs(
    data_dir: Union[str, Path],
    keep_previous: int = KEEP_PREVIOUS_RUNS,
) -> None:
  """Deletes superseded run directories while preserving current and recent runs."""
  root = require_data_dir(data_dir)
  runs_dir = root / "runs"
  if not runs_dir.is_dir():
    return
  current = current_run_dir(root)
  names = sorted(p.name for p in runs_dir.iterdir())
  finished = [n for n in names if not n.endswith(".partial")]
  keep = set(finished[-(keep_previous + 1) :])
  for name in names:
    path = runs_dir / name
    if name in keep or (current is not None and path.resolve() == current):
      continue
    if path.is_dir():
      shutil.rmtree(path, ignore_errors=True)


def sync_all_models(
    data_dir: Union[str, Path],
    models: Optional[Sequence[str]] = None,
    force: bool = False,
    log: Callable[[str], None] = print,
    catalog: Any = None,
    open_dataset: Optional[Callable[[Any, str], xr.Dataset]] = None,
) -> Dict[str, Any]:
  """Checks dynamical.org and downloads models whose newest run changed.

  Args:
    data_dir: Explicit local root directory for downloaded runs.
    models: Sequence of model keys to synchronize. Defaults to
      ``DEFAULT_SYNC_MODELS``.
    force: If True, re-downloads runs even when init_time matches.
    log: Logging callback.
    catalog: Optional pre-opened STAC catalog (or test stub).
    open_dataset: Optional dataset opener callback `(catalog, dataset_id) -> xr.Dataset`.

  Returns:
    Synchronization status dictionary written to `<data_dir>/sync_status.json`.
  """
  root = require_data_dir(data_dir)
  runs_root = root / "runs"
  runs_root.mkdir(parents=True, exist_ok=True)
  selected_models = list(models) if models is not None else list(DEFAULT_SYNC_MODELS)
  for m in selected_models:
    if m not in DYNAMICAL_MODELS:
      raise ValueError(
          f"Unsupported dynamical model {m!r}. Supported: {list(DYNAMICAL_MODELS)}"
      )

  status_path = root / SYNC_STATUS_FILE
  status = read_sync_status(root)
  status.update({
      "last_check_utc": utc_now_str(),
      "check_interval_minutes": CHECK_INTERVAL_MINUTES,
      "source": STAC_CATALOG_URL,
  })

  run_dir, current = current_models_metadata(root)
  active_catalog = catalog if catalog is not None else open_dynamical_catalog()
  dataset_opener = open_dataset if open_dataset is not None else open_dynamical_dataset

  plan: Dict[str, str] = {}
  datasets: Dict[str, xr.Dataset] = {}
  errors: Dict[str, str] = {}

  for model_key in selected_models:
    cfg = DYNAMICAL_MODELS[model_key]
    ds = dataset_opener(active_catalog, cfg["dataset"])
    datasets[model_key] = ds
    latest = latest_init_time_str(ds)
    have = current.get(model_key)
    files_ok = bool(
        have
        and run_dir is not None
        and all(
            (run_dir / f"{model_key}_{s}.bin").exists()
            for s in have.get("streams", [])
        )
    )
    if not force and files_ok and have.get("init_time") == latest:
      log(f"[{model_key}] up to date (run {latest})")
    else:
      plan[model_key] = latest
      prev_init = have.get("init_time") if have else "none"
      log(f"[{model_key}] new run {latest} (have {prev_init})")

  updated: List[str] = []
  if plan:
    for item in runs_root.iterdir():
      if item.name.endswith(".partial") and item.is_dir():
        shutil.rmtree(item, ignore_errors=True)

    run_name = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    base_name, n = run_name, 1
    while (runs_root / run_name).exists():
      run_name = f"{base_name}-{n}"
      n += 1

    new_dir = runs_root / f"{run_name}.partial"
    new_dir.mkdir(parents=True, exist_ok=True)
    entries: Dict[str, Dict[str, Any]] = {}

    # Preserve existing non-queried models as well as queried models
    all_candidate_models = list(
        dict.fromkeys(list(current.keys()) + selected_models)
    )
    for model_key in all_candidate_models:
      entry: Optional[Dict[str, Any]] = None
      if model_key in plan:
        entry, err = _extract_model_streams(
            datasets[model_key],
            model_key,
            DYNAMICAL_MODELS[model_key],
            new_dir,
            log=log,
        )
        if err is not None:
          errors[model_key] = err
          log(f"[{model_key}] keeping previous run: {err}")
          for stream in DYNAMICAL_MODELS[model_key]["streams"]:
            partial = new_dir / f"{model_key}_{stream}.bin"
            if partial.exists():
              partial.unlink()
        else:
          updated.append(model_key)

      if entry is None and model_key in current and run_dir is not None:
        entry = current[model_key]
        for stream in entry.get("streams", []):
          src = run_dir / f"{model_key}_{stream}.bin"
          if src.exists():
            shutil.copy2(src, new_dir / f"{model_key}_{stream}.bin")
      if entry is not None:
        entries[model_key] = entry

    if updated:
      meta = {
          "status": "HEALTHY",
          "source": "dynamical.org",
          "last_updated_utc": utc_now_str(),
          "datasets": {
              e["id"].replace("-", "_"): e for e in entries.values()
          },
      }
      write_json_atomic(new_dir / RUN_METADATA_FILE, meta)
      final_dir = runs_root / run_name
      os.replace(new_dir, final_dir)
      swap_current_symlink(root, run_name)
      prune_old_runs(root)
      status["last_success_utc"] = utc_now_str()
    else:
      shutil.rmtree(new_dir, ignore_errors=True)
  elif not errors:
    status["last_success_utc"] = utc_now_str()

  _, now_current = current_models_metadata(root)
  status["models"] = {
      k: {
          "init_time": v.get("init_time"),
          "downloaded_utc": v.get("downloaded_utc"),
      }
      for k, v in now_current.items()
  }
  status["errors"] = errors
  status["updated_models"] = updated
  if updated:
    status["last_result"] = "updated"
    status["message"] = "Downloaded new run: " + ", ".join(
        f"{k} {plan[k]}" for k in updated
    )
  elif errors:
    status["last_result"] = (
        "error" if len(errors) == len(selected_models) else "partial"
    )
    status["message"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
  else:
    status["last_result"] = "up_to_date"
    status["message"] = "All models already have the newest published run."

  write_json_atomic(status_path, status)
  log(status["message"])
  return status


def sync_model(
    data_dir: Union[str, Path],
    model_key: str,
    force: bool = False,
    log: Callable[[str], None] = print,
    catalog: Any = None,
    open_dataset: Optional[Callable[[Any, str], xr.Dataset]] = None,
) -> Dict[str, Any]:
  """Synchronizes a single model into data_dir."""
  return sync_all_models(
      data_dir=data_dir,
      models=[model_key],
      force=force,
      log=log,
      catalog=catalog,
      open_dataset=open_dataset,
  )


class WeatherSynchronizer:
  """Manages scheduled and on-demand forecast synchronization in a target data_dir."""

  def __init__(
      self,
      data_dir: Union[str, Path],
      models: Optional[Sequence[str]] = None,
  ):
    self.data_dir = require_data_dir(data_dir)
    self.models = (
        list(models) if models is not None else list(DEFAULT_SYNC_MODELS)
    )

  def current_run_dir(self) -> Optional[Path]:
    """Returns the resolved current run directory, or None if not synced."""
    return current_run_dir(self.data_dir)

  def get_status(self) -> Dict[str, Any]:
    """Returns the synchronization status dictionary."""
    return read_sync_status(self.data_dir)

  def sync_all(
      self,
      force: bool = False,
      log: Callable[[str], None] = print,
      catalog: Any = None,
      open_dataset: Optional[Callable[[Any, str], xr.Dataset]] = None,
  ) -> Dict[str, Any]:
    """Synchronizes all configured models."""
    return sync_all_models(
        data_dir=self.data_dir,
        models=self.models,
        force=force,
        log=log,
        catalog=catalog,
        open_dataset=open_dataset,
    )

  def sync_model(
      self,
      model_key: str,
      force: bool = False,
      log: Callable[[str], None] = print,
      catalog: Any = None,
      open_dataset: Optional[Callable[[Any, str], xr.Dataset]] = None,
  ) -> Dict[str, Any]:
    """Synchronizes a single model."""
    return sync_model(
        data_dir=self.data_dir,
        model_key=model_key,
        force=force,
        log=log,
        catalog=catalog,
        open_dataset=open_dataset,
    )
