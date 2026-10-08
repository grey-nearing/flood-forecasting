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

from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
import xarray as xr

from multimet.weather_fetcher.config import (
    CHECK_INTERVAL_MINUTES,
    DEFAULT_MSLP_OFFSET_HPA,
    DEFAULT_SYNC_MODELS,
    DYNAMICAL_MODELS,
    GRID_DEG,
    KEEP_PREVIOUS_RUNS,
    MAX_LEAD_HOURS,
    MSLP_OFFSET_HPA,
    N_LAT,
    N_LON,
    output_lead_hours,
    RUN_DATASET_TO_MODEL,
    RUN_METADATA_FILE,
    STAC_CATALOG_URL,
    STEP_HOURS,
    STREAM_VARIABLES,
    SUPPORTED_MODELS,
    SYNC_STATUS_FILE,
    to_stored_units,
)

_HRES_LEAD_HOURS: Tuple[int, ...] = tuple(
    list(range(0, 145, 3)) + list(range(150, 241, 6))
)
_HRES_STREAM_TO_PARAM: Dict[str, str] = {
    "precip": "tp",
    "temp": "2t",
    "mslp": "msl",
    "u10": "10u",
    "v10": "10v",
}


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
        "An explicit data_dir path is required for multimet.weather_fetcher."
    )
  return Path(data_dir).expanduser().resolve()


def aggregate_rates(
    rates: Any,
    in_leads: Sequence[int],
    out_leads: Sequence[int],
) -> np.ndarray:
  """Computes mean rate over each output interval (previous output lead, lead].

  rates[i] is the model's mean rate over (in_leads[i-1], in_leads[i]].
  Hourly GFS rain is averaged over each 3-hour step instead of sampling
  one hour in three. Lead 0 has no preceding interval and is all zeros.

  Args:
    rates: Input rate array of shape (n_in_leads, ...).
    in_leads: Input lead hours corresponding to axis 0 of ``rates``.
    out_leads: Output lead hours.

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


def list_available_runs(data_dir: Union[str, Path]) -> List[Dict[str, Any]]:
  """Lists all completed forecast run directories stored under `<data_dir>/runs`."""
  root = require_data_dir(data_dir)
  runs_dir = root / "runs"
  active = current_run_dir(root)
  results: List[Dict[str, Any]] = []
  if not runs_dir.is_dir():
    if active is not None:
      runs_meta = load_run_metadata(active)
      if runs_meta:
        results.append({
            "run_name": active.name,
            "path": str(active),
            "is_current": True,
            "models": runs_meta,
        })
    return results

  for item in sorted(runs_dir.iterdir()):
    if not item.is_dir() or item.name.endswith(".partial"):
      continue
    resolved = item.resolve()
    runs_meta = load_run_metadata(resolved)
    results.append({
        "run_name": item.name,
        "path": str(resolved),
        "is_current": active is not None and resolved == active,
        "models": runs_meta,
    })
  return results


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
  """Returns the latest init_time (or analysis window start) as ISO-8601 string."""
  if "init_time" in ds:
    return str(np.datetime_as_string(ds["init_time"].values[-1], unit="s"))
  times = ds["time"].values
  end_time = times[-1].astype("datetime64[h]")
  first_time = times[0].astype("datetime64[h]")
  candidate = end_time - np.timedelta64(MAX_LEAD_HOURS, "h")
  start_time = candidate if candidate > first_time else first_time
  return str(np.datetime_as_string(start_time, unit="s"))


def is_plane_complete(
    planes: np.ndarray, stream: str, max_nan_fraction: float = 0.5
) -> bool:
  """Checks whether the final forecast lead plane has been populated."""
  if planes.shape[0] == 0:
    return False
  last_plane = planes[-1]
  if float(np.isnan(last_plane).mean()) > max_nan_fraction:
    return False
  if stream != "precip" and bool(np.isnan(last_plane).all()):
    return False
  return True


def _build_2d_to_global_reprojector(
    lat2d: np.ndarray, lon2d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Builds nearest-neighbor index mapping from a 2D projected grid to 721x1440."""
  tree = cKDTree(
      np.column_stack([
          np.asarray(lat2d, dtype=np.float64).ravel(),
          np.asarray(lon2d, dtype=np.float64).ravel(),
      ])
  )
  glats = np.linspace(90.0, -90.0, N_LAT, dtype=np.float64)
  glons = np.linspace(-180.0, 180.0, N_LON, endpoint=False, dtype=np.float64)
  g_lat2d, g_lon2d = np.meshgrid(glats, glons, indexing="ij")
  dist, flat_idx = tree.query(
      np.column_stack([g_lat2d.ravel(), g_lon2d.ravel()]),
      distance_upper_bound=GRID_DEG,
  )
  valid_mask = np.isfinite(dist).reshape(N_LAT, N_LON)
  safe_idx = np.where(np.isfinite(dist), flat_idx, 0).astype(np.intp).reshape(
      N_LAT, N_LON
  )
  return valid_mask, safe_idx


def _resample_1d_rectilinear_to_global(
    planes: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray
) -> np.ndarray:
  """Resamples `(T, H_src, W_src)` 1D rectilinear grid onto `(T, N_LAT, N_LON)`."""
  if planes.shape[1] == N_LAT and planes.shape[2] == N_LON:
    return planes
  glats = np.linspace(90.0, -90.0, N_LAT, dtype=np.float64)
  glons = np.linspace(-180.0, 180.0, N_LON, endpoint=False, dtype=np.float64)
  lat_order = np.argsort(src_lats)
  lon_order = np.argsort(src_lons)
  sorted_lats = np.asarray(src_lats, dtype=np.float64)[lat_order]
  sorted_lons = np.asarray(src_lons, dtype=np.float64)[lon_order]
  lat_pos = np.clip(
      np.searchsorted(sorted_lats, glats), 0, len(sorted_lats) - 1
  )
  lon_pos = np.clip(
      np.searchsorted(sorted_lons, glons), 0, len(sorted_lons) - 1
  )
  row_idx = lat_order[lat_pos]
  col_idx = lon_order[lon_pos]
  return planes[:, row_idx[:, None], col_idx[None, :]]


def _extract_dynamical_analysis_streams(
    ds: xr.Dataset,
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Extracts trailing 10-day 3-hourly binary streams from a dynamical.org analysis dataset."""
  times = ds["time"].values
  if len(times) == 0:
    return None, f"{cfg['dataset']}: empty time coordinate"
  end_time = times[-1].astype("datetime64[h]")
  first_time = times[0].astype("datetime64[h]")
  candidate = end_time - np.timedelta64(MAX_LEAD_HOURS, "h")
  start_time = candidate if candidate > first_time else first_time
  init_str = str(np.datetime_as_string(start_time, unit="s"))

  ds_window = ds.sel(time=slice(start_time, end_time))
  win_times = ds_window["time"].values
  on_hour_mask = win_times.astype("datetime64[m]") == win_times.astype(
      "datetime64[h]"
  ).astype("datetime64[m]")
  if bool(np.any(on_hour_mask)):
    ds_window = ds_window.isel(time=np.flatnonzero(on_hour_mask))
    win_times = ds_window["time"].values
  in_leads = [
      int((t.astype("datetime64[h]") - start_time) / np.timedelta64(1, "h"))
      for t in win_times
  ]
  out_leads = output_lead_hours(in_leads)
  if not out_leads:
    return None, f"{cfg['dataset']}: no valid 3-hourly steps in window"

  src_lats = ds_window["latitude"].values
  src_lons = ds_window["longitude"].values
  streams: List[str] = []
  for stream in cfg["streams"]:
    var = STREAM_VARIABLES[stream]
    if var not in ds_window:
      log(f"   [!] {cfg['dataset']} has no {var}; skipped")
      continue
    t0 = time.time()
    raw = ds_window[var].values.astype(np.float32)
    if not is_plane_complete(raw, stream):
      return None, f"{cfg['dataset']} {init_str}: {var} not complete yet"
    resampled = _resample_1d_rectilinear_to_global(raw, src_lats, src_lons)
    del raw
    if stream == "precip":
      planes = aggregate_rates(resampled, in_leads, out_leads)
    else:
      out_pos = [in_leads.index(h) for h in out_leads]
      planes = resampled[out_pos]
    del resampled
    out_file = out_dir / f"{model_key}_{stream}.bin"
    to_stored_units(stream, planes).tofile(str(out_file))
    streams.append(stream)
    log(
        f"   -> {model_key}_{stream}.bin: {len(out_leads)} leads in"
        f" {time.time() - t0:.1f}s"
    )

  if not set(cfg["streams"]) <= set(streams):
    return None, f"{cfg['dataset']}: required streams missing"
  return {
      "id": cfg["dataset"],
      "type": "analysis",
      "model": model_key,
      "title": cfg["title"],
      "init_time": init_str,
      "lead_steps": len(out_leads),
      "lead_hours": out_leads,
      "streams": streams,
      "variables": [STREAM_VARIABLES[s] for s in streams],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "downloaded_utc": utc_now_str(),
  }, None


def _latest_hres_run_info(
    fs: Any = None,
) -> Optional[Tuple[str, str, str]]:
  """Finds the newest complete 240h ECMWF IFS HRES run on gs://ecmwf-open-data."""
  import gcsfs

  gcs = fs if fs is not None else gcsfs.GCSFileSystem(token="anon")
  now = datetime.datetime.now(datetime.timezone.utc)
  for days_back in range(0, 5):
    dt = now - datetime.timedelta(days=days_back)
    d_str = dt.strftime("%Y%m%d")
    for cycle in ("12z", "00z"):
      hh = cycle[:2]
      idx_path = (
          f"ecmwf-open-data/{d_str}/{cycle}/ifs/0p25/oper/"
          f"{d_str}{hh}0000-240h-oper-fc.index"
      )
      if gcs.exists(idx_path):
        init_str = f"{dt.strftime('%Y-%m-%d')}T{hh}:00:00"
        return d_str, cycle, init_str
  return None


def _fetch_single_hres_step(
    gcs: Any, date_str: str, cycle: str, lead_h: int
) -> Dict[str, np.ndarray]:
  """Downloads and decodes surface GRIB2 parameters for one HRES lead step."""
  from multimet.timeseries_extractors.hres import decode_grib2_message

  hh = cycle[:2]
  prefix = (
      f"ecmwf-open-data/{date_str}/{cycle}/ifs/0p25/oper/"
      f"{date_str}{hh}0000-{lead_h}h-oper-fc"
  )
  idx_text = gcs.cat(f"{prefix}.index").decode("utf-8")
  wanted = {"tp", "2t", "msl", "10u", "10v"}
  byte_ranges: Dict[str, Tuple[int, int]] = {}
  for line in idx_text.splitlines():
    line_s = line.strip()
    if not line_s:
      continue
    entry = json.loads(line_s)
    param = entry.get("param")
    if param in wanted and entry.get("levtype") == "sfc":
      offset = int(entry["_offset"])
      length = int(entry["_length"])
      byte_ranges[param] = (offset, offset + length)

  decoded: Dict[str, np.ndarray] = {}
  grib_path = f"{prefix}.grib2"
  for param, (start_b, end_b) in byte_ranges.items():
    msg_bytes = gcs.cat_file(grib_path, start=start_b, end=end_b)
    decoded[param] = decode_grib2_message(
        msg_bytes, (N_LAT, N_LON)
    ).astype(np.float32)
  return decoded


def _extract_hres_streams(
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] = print,
    fs: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Downloads ECMWF IFS HRES 0.25 deg GRIB2 streams from gs://ecmwf-open-data."""
  import gcsfs

  gcs = fs if fs is not None else gcsfs.GCSFileSystem(token="anon")
  run_info = _latest_hres_run_info(gcs)
  if run_info is None:
    return None, "ecmwf_open_data: no complete 240h HRES run found"
  date_str, cycle, init_str = run_info
  leads = list(_HRES_LEAD_HOURS)
  t0 = time.time()
  log(f"   -> Fetching {len(leads)} HRES GRIB2 steps for {date_str} {cycle}...")

  with ThreadPoolExecutor(max_workers=16) as pool:
    step_dicts = list(
        pool.map(
            lambda h: _fetch_single_hres_step(gcs, date_str, cycle, h), leads
        )
    )

  streams: List[str] = []
  for stream in cfg["streams"]:
    param = _HRES_STREAM_TO_PARAM[stream]
    if any(param not in d for d in step_dicts):
      return None, f"ecmwf_hres {init_str}: missing GRIB2 parameter {param}"
    raw = np.stack([d[param] for d in step_dicts], axis=0)
    if stream == "precip":
      rates_mms = np.zeros_like(raw, dtype=np.float32)
      for i in range(1, len(leads)):
        dt_seconds = float(leads[i] - leads[i - 1]) * 3600.0
        diff_mm = (raw[i] - raw[i - 1]) * 1000.0
        rates_mms[i] = np.clip(diff_mm / dt_seconds, 0.0, None)
      planes = rates_mms
    else:
      planes = raw
    del raw
    out_file = out_dir / f"{model_key}_{stream}.bin"
    to_stored_units(stream, planes).tofile(str(out_file))
    streams.append(stream)

  log(
      f"   -> {model_key}: wrote {len(streams)} streams ({len(leads)} leads) in"
      f" {time.time() - t0:.1f}s"
  )
  return {
      "id": cfg["dataset"],
      "type": "forecast",
      "model": model_key,
      "title": cfg["title"],
      "init_time": init_str,
      "lead_steps": len(leads),
      "lead_hours": leads,
      "streams": streams,
      "variables": [STREAM_VARIABLES[s] for s in streams],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "downloaded_utc": utc_now_str(),
  }, None


def _open_cpc_window_dataset(cache_dir: Path) -> Tuple[xr.Dataset, str]:
  """Opens NOAA PSL CPC global precipitation NetCDF and returns trailing 10-day window."""
  from multimet.utils.cpc import ensure_psl_cpc_netcdf

  cpc_cache = cache_dir / "cpc_nc"
  cpc_cache.mkdir(parents=True, exist_ok=True)
  current_year = datetime.datetime.now(datetime.timezone.utc).year
  nc_path = ensure_psl_cpc_netcdf(current_year, str(cpc_cache))
  ds = xr.open_dataset(nc_path, engine="h5netcdf")
  precip = ds["precip"].values
  times = ds["time"].values
  valid_days = [
      i for i in range(len(times)) if not bool(np.isnan(precip[i]).all())
  ]
  if not valid_days:
    ds.close()
    nc_prev = ensure_psl_cpc_netcdf(current_year - 1, str(cpc_cache))
    ds = xr.open_dataset(nc_prev, engine="h5netcdf")
    precip = ds["precip"].values
    times = ds["time"].values
    valid_days = [
        i for i in range(len(times)) if not bool(np.isnan(precip[i]).all())
    ]

  last_idx = valid_days[-1]
  start_idx = max(0, last_idx - 10)
  ds_win = ds.isel(time=slice(start_idx, last_idx + 1)).load()
  ds.close()
  start_time = ds_win["time"].values[0].astype("datetime64[s]")
  init_str = str(np.datetime_as_string(start_time, unit="s"))
  return ds_win, init_str


def _extract_cpc_streams(
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    cache_dir: Path,
    log: Callable[[str], None] = print,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Extracts NOAA CPC Unified global gauge precipitation into 0.25 deg binary stream."""
  t0 = time.time()
  ds_win, init_str = _open_cpc_window_dataset(cache_dir)
  raw_mm_day = ds_win["precip"].values.astype(np.float32)
  n_days = raw_mm_day.shape[0]
  if n_days < 2:
    return None, "noaa_cpc: fewer than 2 valid days in CPC NetCDF"

  # CPC longitude is [0.25 .. 359.75] (720 cols); roll by 360 to [-179.75 .. 179.75]
  rolled = np.roll(raw_mm_day, shift=raw_mm_day.shape[2] // 2, axis=2)
  src_lats = ds_win["lat"].values.astype(np.float64)
  src_lons = (
      (ds_win["lon"].values.astype(np.float64) + 180.0) % 360.0
  ) - 180.0
  rolled_lons = np.roll(src_lons, shift=len(src_lons) // 2)

  resampled_mm_day = _resample_1d_rectilinear_to_global(
      rolled, src_lats, rolled_lons
  )
  # Convert mm/day -> mm/s so to_stored_units("precip", ...) yields mm/h (mm/day / 24)
  rates_mms = np.zeros_like(resampled_mm_day, dtype=np.float32)
  rates_mms[1:] = np.clip(
      np.nan_to_num(resampled_mm_day[1:], nan=0.0) / 86400.0, 0.0, None
  )
  lead_hours = [24 * i for i in range(n_days)]
  out_file = out_dir / f"{model_key}_precip.bin"
  to_stored_units("precip", rates_mms).tofile(str(out_file))
  log(
      f"   -> {model_key}_precip.bin: {n_days} daily steps in"
      f" {time.time() - t0:.1f}s"
  )
  return {
      "id": cfg["dataset"],
      "type": "analysis",
      "model": model_key,
      "title": cfg["title"],
      "init_time": init_str,
      "lead_steps": len(lead_hours),
      "lead_hours": lead_hours,
      "streams": ["precip"],
      "variables": [STREAM_VARIABLES["precip"]],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "downloaded_utc": utc_now_str(),
  }, None


def _extract_model_streams(
    ds: xr.Dataset,
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Extracts binary streams for a dynamical.org model run, returning (entry, error_msg)."""
  if cfg.get("source") == "dynamical_analysis" or (
      "time" in ds.dims and "init_time" not in ds.dims
  ):
    return _extract_dynamical_analysis_streams(
        ds, model_key, cfg, out_dir, log=log
    )

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

  is_2d_proj = "latitude" in ds and ds["latitude"].ndim == 2
  reproj_mask: Optional[np.ndarray] = None
  reproj_idx: Optional[np.ndarray] = None
  if is_2d_proj:
    reproj_mask, reproj_idx = _build_2d_to_global_reprojector(
        ds["latitude"].values, ds["longitude"].values
    )

  def _fetch_one_stream(stream: str) -> Tuple[str, Optional[str]]:
    var = STREAM_VARIABLES[stream]
    if var not in ds:
      log(f"   [!] {cfg['dataset']} has no {var}; skipped")
      return stream, "skipped"
    t0 = time.time()
    raw = ds[var].sel(**sel).isel(lead_time=in_idx).values.astype(np.float32)
    if not is_plane_complete(raw, stream):
      return stream, f"{cfg['dataset']} {init_str}: {var} not complete yet"
    if is_2d_proj and reproj_mask is not None and reproj_idx is not None:
      flat = raw.reshape(raw.shape[0], -1)
      mapped = flat[:, reproj_idx]
      mapped[:, ~reproj_mask] = np.nan
      raw = mapped
    if stream == "precip":
      planes = aggregate_rates(raw, in_leads, out_leads)
      if is_2d_proj and reproj_mask is not None:
        planes[:, ~reproj_mask] = np.nan
    else:
      planes = raw[out_pos]
    del raw
    out_file = out_dir / f"{model_key}_{stream}.bin"
    stored = to_stored_units(stream, planes)
    if is_2d_proj and reproj_mask is not None and stream == "precip":
      stored[:, ~reproj_mask] = np.float16(np.nan)
    stored.tofile(str(out_file))
    log(
        f"   -> {model_key}_{stream}.bin: {len(out_leads)} leads in"
        f" {time.time() - t0:.1f}s"
    )
    return stream, None

  with ThreadPoolExecutor(max_workers=max(1, len(cfg["streams"]))) as pool:
    stream_results = list(pool.map(_fetch_one_stream, cfg["streams"]))

  streams: List[str] = []
  for stream, err_msg in stream_results:
    if err_msg == "skipped":
      continue
    if err_msg is not None:
      return None, err_msg
    streams.append(stream)
  required = (
      {"precip"}
      if tuple(cfg["streams"]) == ("precip",)
      else {"precip", "temp"}
  )
  if not required <= set(streams):
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
  os.symlink(Path("runs") / run_name, tmp, target_is_directory=True)
  current_link = root / "current"
  if os.name == "nt" and os.path.lexists(current_link):
    current_link.unlink()
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
  """Checks upstream sources and downloads models whose newest run changed.

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
  selected_models = (
      list(models) if models is not None else list(DEFAULT_SYNC_MODELS)
  )
  for m in selected_models:
    if m not in DYNAMICAL_MODELS:
      raise ValueError(
          f"Unsupported weather model {m!r}. Supported: {list(DYNAMICAL_MODELS)}"
      )

  status_path = root / SYNC_STATUS_FILE
  status = read_sync_status(root)
  status.update({
      "last_check_utc": utc_now_str(),
      "check_interval_minutes": CHECK_INTERVAL_MINUTES,
      "source": STAC_CATALOG_URL,
  })

  run_dir, current = current_models_metadata(root)
  needs_dynamical = any(
      DYNAMICAL_MODELS[m].get("source", "dynamical")
      in ("dynamical", "dynamical_analysis")
      or open_dataset is not None
      for m in selected_models
  )
  active_catalog = (
      (catalog if catalog is not None else open_dynamical_catalog())
      if needs_dynamical
      else catalog
  )
  dataset_opener = (
      open_dataset if open_dataset is not None else open_dynamical_dataset
  )

  plan: Dict[str, str] = {}
  datasets: Dict[str, xr.Dataset] = {}
  errors: Dict[str, str] = {}

  for model_key in selected_models:
    cfg = DYNAMICAL_MODELS[model_key]
    src_type = cfg.get("source", "dynamical")
    if open_dataset is not None or src_type in (
        "dynamical",
        "dynamical_analysis",
    ):
      ds = dataset_opener(active_catalog, cfg["dataset"])
      datasets[model_key] = ds
      latest = latest_init_time_str(ds)
    elif src_type == "ecmwf_open_data":
      hres_info = _latest_hres_run_info()
      if hres_info is None:
        errors[model_key] = "ecmwf_open_data: no complete 240h HRES run found"
        continue
      latest = hres_info[2]
    elif src_type == "noaa_psl_cpc":
      ds_cpc, latest = _open_cpc_window_dataset(root)
      ds_cpc.close()
    else:
      raise ValueError(f"Unknown source type {src_type!r} for {model_key!r}")

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
    now_epoch = time.time()
    for item in runs_root.iterdir():
      if (
          item.name.endswith(".partial")
          and item.is_dir()
          and (now_epoch - item.stat().st_mtime) > 7200.0
      ):
        shutil.rmtree(item, ignore_errors=True)

    run_name = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    base_name, n = run_name, 1
    while (runs_root / run_name).exists():
      run_name = f"{base_name}-{n}"
      n += 1

    new_dir = runs_root / f"{run_name}.{os.getpid()}.partial"
    new_dir.mkdir(parents=True, exist_ok=True)
    entries: Dict[str, Dict[str, Any]] = {}

    for model_key in selected_models:
      if model_key not in plan:
        continue
      cfg = DYNAMICAL_MODELS[model_key]
      src_type = cfg.get("source", "dynamical")
      if model_key in datasets:
        entry, err = _extract_model_streams(
            datasets[model_key],
            model_key,
            cfg,
            new_dir,
            log=log,
        )
      elif src_type == "ecmwf_open_data":
        entry, err = _extract_hres_streams(
            model_key, cfg, new_dir, log=log
        )
      elif src_type == "noaa_psl_cpc":
        entry, err = _extract_cpc_streams(
            model_key, cfg, new_dir, root, log=log
        )
      else:
        entry, err = None, f"No dataset opened for {model_key}"
      if err is not None:
        errors[model_key] = err
        log(f"[{model_key}] keeping previous run: {err}")
        for stream in DYNAMICAL_MODELS[model_key]["streams"]:
          partial = new_dir / f"{model_key}_{stream}.bin"
          if partial.exists():
            partial.unlink()
      elif entry is not None:
        updated.append(model_key)
        entries[model_key] = entry

    latest_run_dir, latest_current = current_models_metadata(root)
    for model_key, prev_entry in latest_current.items():
      if model_key in entries or latest_run_dir is None:
        continue
      for stream in prev_entry.get("streams", []):
        src = latest_run_dir / f"{model_key}_{stream}.bin"
        if src.exists():
          shutil.copy2(src, new_dir / f"{model_key}_{stream}.bin")
      entries[model_key] = prev_entry

    if updated:
      while (runs_root / run_name).exists():
        run_name = f"{base_name}-{n}"
        n += 1
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

  def list_runs(self) -> List[Dict[str, Any]]:
    """Lists all available forecast runs in data_dir."""
    return list_available_runs(self.data_dir)

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
