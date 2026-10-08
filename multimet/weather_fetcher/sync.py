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

"""Scheduled and on-demand forecast run downloader with atomic directory swap.

Every model is reduced to float16 planes on the 0.25 degree viewer grid
(``N_LAT`` x ``N_LON``; +90 to -90 latitude, -180 to +179.75 longitude) inside
a staging directory ``<data_dir>/runs/<run>.<pid>.partial``. The directory is
renamed to ``<data_dir>/runs/<run>`` and exposed through the
``<data_dir>/current`` symlink only after every selected model has either been
extracted or carried over from the previous run.

Missing upstream values stay ``NaN`` end to end: no plane is filled with
zeros, no lead step is interpolated, and no target cell is sampled from a
neighbouring source cell. Coarser source grids (NOAA CPC 0.5 deg) are mapped
by picking the source cell that contains each target cell centre; finer grids
(NASA IMERG 0.1 deg, NOAA HRRR 3 km) are averaged over the source cells whose
centres fall inside each target cell and become ``NaN`` when fewer than
``MIN_VALID_RESAMPLE_FRACTION`` of those cells are finite.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import datetime
import json
import os
from pathlib import Path
import shutil
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import sparse
import xarray as xr

from multimet.utils.cpc import (
    ensure_psl_cpc_netcdf,
    EXPECTED_PSL_LATS,
    EXPECTED_PSL_LONS,
    NOAA_PSL_URL_TEMPLATE,
)
from multimet.utils.http import check_http_url_exists
from multimet.weather_fetcher.config import (
    CHECK_INTERVAL_MINUTES,
    CPC_CACHE_REFRESH_HOURS,
    CPC_MAX_PUBLICATION_LAG_DAYS,
    DEFAULT_MSLP_OFFSET_HPA,
    DEFAULT_SYNC_MODELS,
    DYNAMICAL_MODELS,
    GRID_DEG,
    KEEP_PREVIOUS_RUNS,
    MAX_LEAD_HOURS,
    MIN_VALID_RESAMPLE_FRACTION,
    MODEL_NATIVE_LEAD_HOURS,
    MSLP_OFFSET_HPA,
    N_LAT,
    N_LON,
    output_lead_hours,
    RUN_DATASET_TO_MODEL,
    RUN_METADATA_FILE,
    SOURCE_LABELS,
    STAC_CATALOG_URL,
    STEP_HOURS,
    STREAM_VARIABLES,
    SUPPORTED_MODELS,
    SYNC_STATUS_FILE,
    to_stored_units,
)

# ECMWF Open Data (gs://ecmwf-open-data) IFS HRES 0.25 deg surface GRIB2.
_HRES_BUCKET: str = "ecmwf-open-data"
_HRES_LEAD_HOURS: Tuple[int, ...] = MODEL_NATIVE_LEAD_HOURS["ecmwf_hres"]
_HRES_STREAM_TO_PARAM: Dict[str, str] = {
    "precip": "tp",
    "temp": "2t",
    "mslp": "msl",
    "u10": "10u",
    "v10": "10v",
}
# Raw GRIB2 units of each parameter ("tp" is metres accumulated since init).
_HRES_PARAM_UNITS: Dict[str, str] = {
    "tp": "m",
    "2t": "K",
    "msl": "Pa",
    "10u": "m s-1",
    "10v": "m s-1",
}
# Only the 00z and 12z cycles extend to 240 h.
_HRES_CYCLES: Tuple[str, ...] = ("12z", "00z")
_HRES_LOOKBACK_DAYS: int = 5

# A lead plane counts as published when it holds at least this fraction of the
# finite cells of the reference plane (the first lead after lead 0).
_MAX_PLANE_DEFICIT_FRACTION: float = 0.01

# NOAA PSL CPC Unified gauge analysis (daily, 0.5 deg, land only).
_CPC_WINDOW_DAYS: int = MAX_LEAD_HOURS // 24
_CPC_MAX_SCAN_DAYS: int = 60
_CPC_DAY_END_HOUR_UTC: int = 12
_CPC_HTTP_HEADERS: Dict[str, str] = {
    "User-Agent": "OpenMultiMet/1.1 (Google Research)"
}

# Latitude rows read per request from half-hourly analysis stores (IMERG is
# chunked in 30-day blocks, so one request per band keeps memory bounded).
_ANALYSIS_BAND_ROWS: int = 50

# Staging directories older than this are never considered live.
_STALE_PARTIAL_MAX_AGE_S: float = 24.0 * 3600.0


class IncompleteRunError(RuntimeError):
  """Raised when the newest upstream run has not been fully published yet."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _resolve_now(now: Optional[datetime.datetime]) -> datetime.datetime:
  """Returns ``now`` as an aware UTC datetime (current time when omitted)."""
  if now is None:
    return datetime.datetime.now(datetime.timezone.utc)
  if now.tzinfo is None or now.utcoffset() is None:
    raise ValueError("now must be a timezone-aware datetime")
  return now.astimezone(datetime.timezone.utc)


def utc_now_str(now: Optional[datetime.datetime] = None) -> str:
  """Returns ``now`` (default: current UTC time) as an ISO-8601 'Z' string."""
  return _resolve_now(now).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_seconds(value: Any) -> str:
  """Formats a numpy/pandas datetime as ``YYYY-MM-DDTHH:MM:SS``."""
  return str(np.datetime_as_string(np.datetime64(value), unit="s"))


def require_data_dir(data_dir: Union[str, Path, None]) -> Path:
  """Validates that an explicit data_dir path was supplied."""
  if data_dir is None or str(data_dir).strip() == "":
    raise ValueError(
        "An explicit data_dir path is required for multimet.weather_fetcher."
    )
  return Path(data_dir).expanduser().resolve()


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
  if not target.is_file() or target.stat().st_size == 0:
    return None
  return json.loads(target.read_text(encoding="utf-8"))


def _global_grid_axes() -> Tuple[np.ndarray, np.ndarray]:
  """Returns the viewer grid cell-centre latitudes (90..-90) and longitudes."""
  glats = np.linspace(90.0, -90.0, N_LAT, dtype=np.float64)
  glons = np.linspace(-180.0, 180.0, N_LON, endpoint=False, dtype=np.float64)
  return glats, glons


def _wrap_longitudes(lons: Any) -> np.ndarray:
  """Maps longitudes onto ``[-180, 180)``."""
  return ((np.asarray(lons, dtype=np.float64) + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------


def aggregate_rates(
    rates: Any,
    in_leads: Sequence[int],
    out_leads: Sequence[int],
) -> np.ndarray:
  """Computes mean rate over each output interval (previous output lead, lead].

  ``rates[i]`` is the model's mean rate over ``(in_leads[i-1], in_leads[i]]``.
  Hourly GFS rain is averaged over each 3-hour step instead of sampling one
  hour in three. An output interval is only computed when the input steps
  cover it exactly (contiguous steps from the previous output lead to the
  lead); otherwise the plane is ``NaN``. ``NaN`` input cells propagate to the
  output through plain arithmetic. The first output lead has no preceding
  interval when it coincides with the first input lead: that plane is ``0.0``
  wherever the first informative input plane is finite and ``NaN`` elsewhere.

  Args:
    rates: Input rate array of shape ``(len(in_leads), ...)``.
    in_leads: Strictly increasing input lead hours (axis 0 of ``rates``).
    out_leads: Strictly increasing output lead hours.

  Returns:
    Float32 array of shape ``(len(out_leads),) + rates.shape[1:]``.

  Raises:
    ValueError: If the lead lists are inconsistent with ``rates`` or not
      strictly increasing, or an output lead precedes the first input lead.
  """
  rates_arr = np.asarray(rates, dtype=np.float32)
  in_list = [int(h) for h in in_leads]
  out_list = [int(h) for h in out_leads]
  if rates_arr.ndim < 1 or rates_arr.shape[0] != len(in_list):
    raise ValueError(
        f"rates has {rates_arr.shape[0] if rates_arr.ndim else 0} planes but"
        f" {len(in_list)} input leads were given."
    )
  if len(in_list) > 1 and np.any(np.diff(in_list) <= 0):
    raise ValueError(f"in_leads must be strictly increasing, got {in_list}")
  if len(out_list) > 1 and np.any(np.diff(out_list) <= 0):
    raise ValueError(f"out_leads must be strictly increasing, got {out_list}")
  plane_shape = rates_arr.shape[1:]
  out = np.full((len(out_list),) + plane_shape, np.nan, dtype=np.float32)
  if not out_list or not in_list:
    return out
  if out_list[0] < in_list[0]:
    raise ValueError(
        f"Output lead {out_list[0]} h precedes first input lead {in_list[0]} h"
    )

  # Domain mask for the no-interval plane: the first input plane after lead 0
  # with any finite cell (upstream stores keep lead 0 of rate fields NaN).
  reference = rates_arr[0]
  for k in range(1, len(in_list)):
    if np.isfinite(rates_arr[k]).any():
      reference = rates_arr[k]
      break

  prev = in_list[0]
  for j, lead in enumerate(out_list):
    if j == 0 and lead == prev:
      out[j] = np.where(
          np.isfinite(reference), np.float32(0.0), np.float32(np.nan)
      )
      continue
    steps = [i for i in range(1, len(in_list)) if prev < in_list[i] <= lead]
    covered = (
        bool(steps)
        and in_list[steps[0] - 1] == prev
        and in_list[steps[-1]] == lead
    )
    if covered:
      total = np.zeros(plane_shape, dtype=np.float32)
      for i in steps:
        total = total + rates_arr[i] * np.float32(in_list[i] - in_list[i - 1])
      out[j] = total / np.float32(lead - prev)
    prev = lead
  return out


# ---------------------------------------------------------------------------
# Spatial regridding onto the 0.25 deg viewer grid
# ---------------------------------------------------------------------------


def _axis_bin_matrix(
    src: Any, targets: np.ndarray, periodic: bool
) -> sparse.csr_matrix:
  """Builds the ``(n_target, n_src)`` 0/1 map of one coordinate axis.

  When the source spacing is not coarser than the target spacing every source
  centre is assigned to the target cell that contains it (area binning). When
  the source is coarser, each target centre picks the single source cell that
  contains it; target centres outside the source domain get no entry.

  Args:
    src: Source cell-centre coordinates (1-D).
    targets: Equally spaced target cell centres (ascending or descending).
    periodic: Treat the axis as longitude (wrapped onto ``[-180, 180)``).

  Returns:
    CSR matrix with at most one nonzero per source column (fine case) or per
    target row (coarse case).
  """
  src_arr = np.asarray(src, dtype=np.float64).ravel()
  if src_arr.size == 0:
    raise ValueError("Source coordinate axis is empty.")
  if periodic:
    src_arr = _wrap_longitudes(src_arr)
  n_target = len(targets)
  t0 = float(targets[0])
  dt = float(targets[1] - targets[0])
  step = abs(dt)
  tol = 1e-6 * max(1.0, step)
  if src_arr.size < 2:
    spacing = step
  else:
    spacing = float(np.median(np.abs(np.diff(np.sort(src_arr)))))

  if spacing <= step + tol:
    pos = (src_arr - t0) / dt
    idx = np.rint(pos)
    if periodic:
      idx = np.mod(idx, n_target)
    ok = np.isfinite(pos) & (idx >= 0) & (idx < n_target)
    rows = idx[ok].astype(np.intp)
    cols = np.flatnonzero(ok)
  else:
    order = np.argsort(src_arr)
    sorted_src = src_arr[order]
    if periodic:
      ext = np.concatenate(
          [sorted_src[-1:] - 360.0, sorted_src, sorted_src[:1] + 360.0]
      )
      ext_order = np.concatenate([order[-1:], order, order[:1]])
    else:
      ext, ext_order = sorted_src, order
    k = np.searchsorted(ext, targets)
    lo = np.clip(k - 1, 0, len(ext) - 1)
    hi = np.clip(k, 0, len(ext) - 1)
    d_lo = np.abs(targets - ext[lo])
    d_hi = np.abs(targets - ext[hi])
    best = np.where(d_hi < d_lo, hi, lo)
    ok = np.minimum(d_lo, d_hi) <= spacing / 2.0 + tol
    rows = np.flatnonzero(ok)
    cols = ext_order[best[ok]]
  return sparse.csr_matrix(
      (np.ones(len(rows), dtype=np.float32), (rows, cols)),
      shape=(n_target, src_arr.size),
  )


def _axis_totals(matrix: sparse.csr_matrix) -> np.ndarray:
  """Number of source entries mapped into each target row of ``matrix``."""
  return np.asarray(matrix.sum(axis=1), dtype=np.float32).ravel()


def _binned_sums_counts(
    a_rows: sparse.csr_matrix,
    a_cols_t: sparse.csr_matrix,
    plane: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Sums and finite counts of ``plane`` binned by row and column matrices."""
  finite = np.isfinite(plane)
  vals = np.where(finite, plane, np.float32(0.0)).astype(np.float32)
  sums = np.asarray((a_rows @ vals) @ a_cols_t, dtype=np.float32)
  counts = np.asarray(
      (a_rows @ finite.astype(np.float32)) @ a_cols_t, dtype=np.float32
  )
  return sums, counts


def _finalize_binned(
    sums: np.ndarray,
    counts: np.ndarray,
    totals: np.ndarray,
    min_valid_fraction: float,
) -> np.ndarray:
  """Mean of binned values where enough source cells are finite, else NaN."""
  enough = (totals > 0) & (counts > 0)
  enough &= counts >= (min_valid_fraction * totals - 1e-6)
  safe_counts = np.where(counts > 0, counts, np.float32(1.0))
  return np.where(enough, sums / safe_counts, np.float32(np.nan)).astype(
      np.float32
  )


def _resample_1d_rectilinear_to_global(
    planes: Any,
    src_lats: Any,
    src_lons: Any,
    min_valid_fraction: float = MIN_VALID_RESAMPLE_FRACTION,
) -> np.ndarray:
  """Maps ``(T, H_src, W_src)`` rectilinear planes onto ``(T, N_LAT, N_LON)``.

  Source longitudes may be given on ``[0, 360)`` or ``[-180, 180)`` and either
  axis may be ascending or descending. Finer source axes are averaged over the
  source centres falling inside each target cell; coarser axes pick the source
  cell containing the target centre. Target cells outside the source domain,
  or with fewer than ``min_valid_fraction`` finite contributing cells, are
  ``NaN``. Planes already on the viewer grid are returned unchanged.

  Args:
    planes: Array of shape ``(T, H_src, W_src)`` or ``(H_src, W_src)``.
    src_lats: Source latitudes (``H_src``).
    src_lons: Source longitudes (``W_src``).
    min_valid_fraction: Minimum finite fraction of contributing source cells.

  Returns:
    Float32 array of shape ``(T, N_LAT, N_LON)``.

  Raises:
    ValueError: If the shapes are inconsistent or the source grid does not
      overlap the global grid at all.
  """
  arr = np.asarray(planes, dtype=np.float32)
  if arr.ndim == 2:
    arr = arr[None]
  lats = np.asarray(src_lats, dtype=np.float64).ravel()
  lons = np.asarray(src_lons, dtype=np.float64).ravel()
  if arr.ndim != 3 or arr.shape[1] != lats.size or arr.shape[2] != lons.size:
    raise ValueError(
        f"planes shape {arr.shape} does not match {lats.size} latitudes x"
        f" {lons.size} longitudes."
    )
  glats, glons = _global_grid_axes()
  if (
      arr.shape[1:] == (N_LAT, N_LON)
      and np.allclose(lats, glats, atol=1e-6)
      and np.allclose(_wrap_longitudes(lons), glons, atol=1e-6)
  ):
    return arr
  a_lat = _axis_bin_matrix(lats, glats, periodic=False)
  a_lon = _axis_bin_matrix(lons, glons, periodic=True)
  totals = np.outer(_axis_totals(a_lat), _axis_totals(a_lon)).astype(
      np.float32
  )
  if not totals.any():
    raise ValueError("Source grid does not overlap the 0.25 deg global grid.")
  a_lon_t = a_lon.T.tocsr()
  out = np.empty((arr.shape[0], N_LAT, N_LON), dtype=np.float32)
  for t in range(arr.shape[0]):
    sums, counts = _binned_sums_counts(a_lat, a_lon_t, arr[t])
    out[t] = _finalize_binned(sums, counts, totals, min_valid_fraction)
  return out


class _GridReprojector:
  """Area-binning map from a 2-D (projected) grid onto the viewer grid.

  Every source cell is assigned to the 0.25 deg target cell containing its
  centre; a target cell holds the mean of its finite source cells, or ``NaN``
  when fewer than ``min_valid_fraction`` of them are finite or when no source
  cell maps into it (outside the regional domain).
  """

  def __init__(
      self,
      lat2d: Any,
      lon2d: Any,
      min_valid_fraction: float = MIN_VALID_RESAMPLE_FRACTION,
  ):
    lat = np.asarray(lat2d, dtype=np.float64)
    lon = np.asarray(lon2d, dtype=np.float64)
    if lat.ndim != 2 or lat.shape != lon.shape:
      raise ValueError(
          f"latitude {lat.shape} and longitude {lon.shape} must be equal 2-D"
          " arrays."
      )
    self.source_shape: Tuple[int, int] = (int(lat.shape[0]), int(lat.shape[1]))
    flat_lat = lat.ravel()
    flat_lon = _wrap_longitudes(lon.ravel())
    row = np.rint((90.0 - flat_lat) / GRID_DEG)
    col = np.mod(np.rint((flat_lon + 180.0) / GRID_DEG), N_LON)
    ok = np.isfinite(flat_lat) & np.isfinite(flat_lon)
    ok &= (row >= 0) & (row < N_LAT)
    cells = (row[ok] * N_LON + col[ok]).astype(np.intp)
    self._matrix = sparse.csr_matrix(
        (np.ones(cells.size, dtype=np.float32), (cells, np.flatnonzero(ok))),
        shape=(N_LAT * N_LON, flat_lat.size),
    )
    self._totals = _axis_totals(self._matrix)
    if not self._totals.any():
      raise ValueError(
          "No source cell of the projected grid maps onto the global grid."
      )
    self.valid_mask: np.ndarray = (self._totals > 0).reshape(N_LAT, N_LON)
    self._min_valid_fraction = float(min_valid_fraction)

  def apply(self, planes: Any) -> np.ndarray:
    """Reprojects ``(..., H_src, W_src)`` planes to ``(..., N_LAT, N_LON)``."""
    arr = np.asarray(planes, dtype=np.float32)
    if arr.ndim < 2 or tuple(arr.shape[-2:]) != self.source_shape:
      raise ValueError(
          f"planes shape {arr.shape} does not end with {self.source_shape}."
      )
    flat = arr.reshape(-1, self.source_shape[0] * self.source_shape[1])
    out = np.empty((flat.shape[0], N_LAT * N_LON), dtype=np.float32)
    for t in range(flat.shape[0]):
      finite = np.isfinite(flat[t])
      vals = np.where(finite, flat[t], np.float32(0.0)).astype(np.float32)
      sums = np.asarray(self._matrix @ vals, dtype=np.float32)
      counts = np.asarray(
          self._matrix @ finite.astype(np.float32), dtype=np.float32
      )
      out[t] = _finalize_binned(
          sums, counts, self._totals, self._min_valid_fraction
      )
    return out.reshape(arr.shape[:-2] + (N_LAT, N_LON))


def _build_2d_to_global_reprojector(
    lat2d: Any, lon2d: Any
) -> _GridReprojector:
  """Builds the area-binning reprojector for a 2-D lat/lon source grid."""
  return _GridReprojector(lat2d, lon2d)


# ---------------------------------------------------------------------------
# Completeness checks
# ---------------------------------------------------------------------------


def _finite_counts(planes: np.ndarray) -> np.ndarray:
  """Number of finite cells in every plane of ``planes``."""
  return np.isfinite(planes).reshape(planes.shape[0], -1).sum(axis=1)


def is_plane_complete(
    planes: Any,
    stream: str,
    max_deficit_fraction: float = _MAX_PLANE_DEFICIT_FRACTION,
) -> bool:
  """Checks that every lead plane of a stream has been populated upstream.

  The reference plane is the first plane for non-precipitation streams and the
  first plane after lead 0 for precipitation (rate fields are ``NaN`` at lead
  0 upstream). Every later plane must hold at least
  ``(1 - max_deficit_fraction)`` times the finite cells of the reference.

  Args:
    planes: Array of shape ``(n_leads, ...)``.
    stream: Stream suffix ('precip', 'temp', 'mslp', 'u10', 'v10').
    max_deficit_fraction: Tolerated relative loss of finite cells per plane.

  Returns:
    ``True`` when all planes are populated, ``False`` otherwise.
  """
  arr = np.asarray(planes)
  if arr.ndim < 2 or arr.shape[0] == 0:
    return False
  counts = _finite_counts(arr)
  start = 1 if (stream == "precip" and arr.shape[0] > 1) else 0
  reference = counts[start]
  if reference == 0:
    return False
  return bool(
      np.all(counts[start:] >= (1.0 - max_deficit_fraction) * reference)
  )


def _first_deficient_lead(
    planes: np.ndarray,
    leads: Sequence[int],
    reference_count: int,
    start: int,
) -> Optional[int]:
  """Returns the first lead whose plane has too few finite cells, or None."""
  counts = _finite_counts(planes)
  threshold = (1.0 - _MAX_PLANE_DEFICIT_FRACTION) * reference_count
  for i in range(start, len(leads)):
    if counts[i] < threshold:
      return int(leads[i])
  return None


# ---------------------------------------------------------------------------
# Run directory layout and metadata
# ---------------------------------------------------------------------------


def current_run_dir(data_dir: Union[str, Path]) -> Optional[Path]:
  """Returns the resolved directory '<data_dir>/current' points to, or None."""
  root = require_data_dir(data_dir)
  link = root / "current"
  if link.is_dir():
    return link.resolve()
  return None


def load_run_metadata(
    target_dir: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
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
        "source": dataset.get("source"),
        "mslp_offset_hpa": float(
            dataset.get("mslp_offset_hpa", DEFAULT_MSLP_OFFSET_HPA)
        ),
    }
  return runs


def list_available_runs(data_dir: Union[str, Path]) -> List[Dict[str, Any]]:
  """Lists completed forecast run directories stored under '<data_dir>/runs'."""
  root = require_data_dir(data_dir)
  runs_dir = root / "runs"
  active = current_run_dir(root)
  results: List[Dict[str, Any]] = []
  if not runs_dir.is_dir():
    return results
  for item in sorted(runs_dir.iterdir()):
    if not item.is_dir() or item.name.endswith(".partial"):
      continue
    resolved = item.resolve()
    results.append({
        "run_name": item.name,
        "path": str(resolved),
        "is_current": active is not None and resolved == active,
        "models": load_run_metadata(resolved),
    })
  return results


def read_sync_status(data_dir: Union[str, Path]) -> Dict[str, Any]:
  """Reads sync_status.json from data_dir if present."""
  root = require_data_dir(data_dir)
  return read_json_if_exists(root / SYNC_STATUS_FILE) or {}


def current_models_metadata(
    data_dir: Union[str, Path],
) -> Tuple[Optional[Path], Dict[str, Dict[str, Any]]]:
  """Returns (active_run_dir, model_key -> dataset entry) of the current run."""
  root = require_data_dir(data_dir)
  run_dir = current_run_dir(root)
  meta = read_json_if_exists(run_dir / RUN_METADATA_FILE) if run_dir else None
  models: Dict[str, Dict[str, Any]] = {}
  for entry in ((meta or {}).get("datasets") or {}).values():
    if entry.get("model"):
      models[entry["model"]] = entry
  return run_dir, models


def swap_current_symlink(data_dir: Union[str, Path], run_name: str) -> Path:
  """Atomically repoints '<data_dir>/current' to 'runs/<run_name>'.

  Args:
    data_dir: Explicit local root directory of downloaded runs.
    run_name: Name of an existing directory under '<data_dir>/runs'.

  Returns:
    Path of the 'current' symlink.

  Raises:
    FileNotFoundError: If 'runs/<run_name>' is not an existing directory.
  """
  root = require_data_dir(data_dir)
  target = root / "runs" / run_name
  if not target.is_dir():
    raise FileNotFoundError(f"Run directory does not exist: {target}")
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
  """Deletes superseded completed runs, keeping current and recent ones.

  Staging directories (``*.partial``) are never touched; they belong to a
  running synchronisation or are cleaned up by ``sync_all_models`` once their
  owning process is gone.
  """
  root = require_data_dir(data_dir)
  runs_dir = root / "runs"
  if not runs_dir.is_dir():
    return
  current = current_run_dir(root)
  finished = sorted(
      p.name
      for p in runs_dir.iterdir()
      if p.is_dir() and not p.name.endswith(".partial")
  )
  keep = set(finished[-(max(0, keep_previous) + 1) :])
  for name in finished:
    path = runs_dir / name
    if name in keep or (current is not None and path.resolve() == current):
      continue
    shutil.rmtree(path, ignore_errors=True)


def _partial_dir_pid(path: Path) -> Optional[int]:
  """Extracts the owning pid from a '<run>.<pid>.partial' directory name."""
  parts = path.name.split(".")
  if len(parts) < 3 or parts[-1] != "partial" or not parts[-2].isdigit():
    return None
  return int(parts[-2])


def _pid_is_alive(pid: int) -> Optional[bool]:
  """Returns whether ``pid`` is running, or None when it cannot be known."""
  if pid == os.getpid():
    return True
  proc_root = Path("/proc")
  if not proc_root.is_dir():
    return None
  return (proc_root / str(pid)).exists()


def _classify_partial_dirs(
    runs_root: Path, now_epoch: float
) -> Tuple[List[Path], List[Path]]:
  """Splits staging directories into (live, stale) by owner pid and age."""
  live: List[Path] = []
  stale: List[Path] = []
  for item in runs_root.iterdir():
    if not item.is_dir() or not item.name.endswith(".partial"):
      continue
    age_s = now_epoch - item.stat().st_mtime
    pid = _partial_dir_pid(item)
    alive = _pid_is_alive(pid) if pid is not None else None
    if alive is None:
      is_live = age_s < _STALE_PARTIAL_MAX_AGE_S
    else:
      is_live = alive and age_s < _STALE_PARTIAL_MAX_AGE_S
    (live if is_live else stale).append(item)
  return live, stale


# ---------------------------------------------------------------------------
# dynamical.org forecast and analysis datasets
# ---------------------------------------------------------------------------


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


@dataclasses.dataclass(frozen=True)
class _AnalysisWindow:
  """Trailing 10-day window of a sub-hourly analysis store in 3 h bins.

  Attributes:
    init_time: Window start (lead 0) as ``datetime64[m]``.
    lead_hours: Output leads ``0, 3, ..., MAX_LEAD_HOURS``.
    frame_minutes: Native time step of the store.
    frame_indices: ``(len(lead_hours) - 1, frames_per_step)`` indices into the
      store's time axis of the frames making up the bin ending at each lead
      after 0; ``-1`` marks frames missing from the store.
  """

  init_time: np.datetime64
  lead_hours: Tuple[int, ...]
  frame_minutes: int
  frame_indices: np.ndarray


def _analysis_window(times: Any) -> _AnalysisWindow:
  """Derives UTC-aligned 3 h bins over the trailing 10 days of ``times``.

  Timestamps mark the start of each frame; the bin ending at lead ``L``
  averages the frames starting at ``L - 3 h, ..., L - frame``. The last bin
  ends at the latest multiple of 3 h (since the epoch) that is fully covered
  by published frames.
  """
  t = np.asarray(times).astype("datetime64[m]")
  if t.size < 2:
    raise ValueError("Analysis dataset needs at least two time steps.")
  diffs = np.diff(t[-min(t.size, 49) :]) / np.timedelta64(1, "m")
  frame_minutes = int(np.median(diffs))
  step_minutes = STEP_HOURS * 60
  if frame_minutes <= 0 or step_minutes % frame_minutes != 0:
    raise ValueError(
        f"Analysis time step of {frame_minutes} min does not divide"
        f" {STEP_HOURS} h."
    )
  frames_per_step = step_minutes // frame_minutes
  step = np.timedelta64(STEP_HOURS, "h")
  frame = np.timedelta64(frame_minutes, "m")
  epoch = np.datetime64("1970-01-01T00:00", "m")
  last_end = t[-1] + frame
  last_bin_end = epoch + ((last_end - epoch) // step) * step
  init_time = last_bin_end - np.timedelta64(MAX_LEAD_HOURS, "h")
  lead_hours = tuple(range(0, MAX_LEAD_HOURS + 1, STEP_HOURS))
  n_bins = len(lead_hours) - 1
  bin_starts = init_time + np.arange(n_bins) * step
  expected = bin_starts[:, None] + np.arange(frames_per_step) * frame
  flat = expected.ravel()
  pos = np.clip(np.searchsorted(t, flat), 0, t.size - 1)
  found = t[pos] == flat
  indices = np.where(found, pos, -1).reshape(n_bins, frames_per_step)
  return _AnalysisWindow(
      init_time=init_time,
      lead_hours=lead_hours,
      frame_minutes=frame_minutes,
      frame_indices=indices.astype(np.int64),
  )


def latest_init_time_str(ds: xr.Dataset) -> str:
  """Returns the latest init_time (or analysis window start) as ISO-8601."""
  if "init_time" in ds:
    return _iso_seconds(ds["init_time"].values[-1])
  return _iso_seconds(_analysis_window(ds["time"].values).init_time)


def _stack_expected_frames(
    block: np.ndarray, rel_indices: np.ndarray, present: np.ndarray
) -> np.ndarray:
  """Arranges a contiguous time block as ``(n_bins, frames, rows, cols)``."""
  n_bins, frames = rel_indices.shape
  flat_idx = rel_indices.ravel()
  flat_present = present.ravel()
  if flat_present.all() and np.array_equal(
      flat_idx, np.arange(n_bins * frames)
  ):
    return block.reshape((n_bins, frames) + block.shape[1:])
  stack = np.full((n_bins * frames,) + block.shape[1:], np.nan, np.float32)
  stack[flat_present] = block[flat_idx[flat_present]]
  return stack.reshape((n_bins, frames) + block.shape[1:])


def _extract_dynamical_analysis_streams(
    ds: xr.Dataset,
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] = print,
    band_rows: int = _ANALYSIS_BAND_ROWS,
) -> Tuple[Optional[Dict[str, Any]], Optional[IncompleteRunError]]:
  """Extracts trailing 10-day 3-hourly precipitation from an analysis store.

  Each stored plane is the mean rate over the 3 h ending at its lead,
  averaged over all native frames of the bin (six half-hour frames for
  IMERG); a bin with a missing frame, or a cell that is ``NaN`` in any frame,
  is ``NaN``. Frames are read in latitude bands so that memory stays bounded
  regardless of the store's time chunking.

  Returns:
    ``(entry, None)`` on success or ``(None, IncompleteRunError)`` when no
    complete bin is available.
  """
  dataset_id = cfg["dataset"]
  if tuple(cfg["streams"]) != ("precip",):
    raise ValueError(
        f"{dataset_id}: analysis datasets only provide the precip stream."
    )
  var = STREAM_VARIABLES["precip"]
  if var not in ds:
    raise KeyError(f"{dataset_id} has no variable {var!r}.")
  da = ds[var]
  if tuple(da.dims) != ("time", "latitude", "longitude"):
    raise ValueError(
        f"{dataset_id}: expected dims (time, latitude, longitude), got"
        f" {tuple(da.dims)}."
    )
  window = _analysis_window(ds["time"].values)
  init_str = _iso_seconds(window.init_time)
  frame_idx = window.frame_indices
  present = frame_idx >= 0
  bins_complete = present.all(axis=1)
  n_bins = frame_idx.shape[0]
  if not bins_complete.any():
    return None, IncompleteRunError(
        f"{dataset_id}: no complete {STEP_HOURS} h bin in the window starting"
        f" {init_str}."
    )
  for k in np.flatnonzero(~bins_complete):
    log(
        f"   [!] {dataset_id}: bin ending lead {window.lead_hours[k + 1]} h"
        " has missing frames; stored as NaN"
    )

  src_lats = np.asarray(ds["latitude"].values, dtype=np.float64)
  src_lons = np.asarray(ds["longitude"].values, dtype=np.float64)
  glats, glons = _global_grid_axes()
  a_lat = _axis_bin_matrix(src_lats, glats, periodic=False)
  a_lon = _axis_bin_matrix(src_lons, glons, periodic=True)
  totals = np.outer(_axis_totals(a_lat), _axis_totals(a_lon)).astype(
      np.float32
  )
  if not totals.any():
    raise ValueError(
        f"{dataset_id}: source grid does not overlap the global grid."
    )
  a_lon_t = a_lon.T.tocsr()
  lo = int(frame_idx[present].min())
  hi = int(frame_idx[present].max()) + 1
  rel_idx = frame_idx - lo

  t0 = time.time()
  complete_bins = [int(k) for k in np.flatnonzero(bins_complete)]
  sums = np.zeros((len(complete_bins), N_LAT, N_LON), dtype=np.float32)
  counts = np.zeros((len(complete_bins), N_LAT, N_LON), dtype=np.uint16)
  n_rows = src_lats.size
  for r0 in range(0, n_rows, max(1, band_rows)):
    r1 = min(n_rows, r0 + max(1, band_rows))
    band_matrix = a_lat[:, r0:r1]
    touched = np.flatnonzero(band_matrix.getnnz(axis=1) > 0)
    if touched.size == 0:
      continue
    i0, i1 = int(touched[0]), int(touched[-1]) + 1
    a_sub = band_matrix[i0:i1]
    block = np.asarray(
        da.isel(time=slice(lo, hi), latitude=slice(r0, r1)).values,
        dtype=np.float32,
    )
    stack = _stack_expected_frames(block, rel_idx, present)
    means = stack.mean(axis=1)
    del block, stack
    for slot, k in enumerate(complete_bins):
      band_sums, band_counts = _binned_sums_counts(a_sub, a_lon_t, means[k])
      sums[slot, i0:i1] += band_sums
      counts[slot, i0:i1] += np.rint(band_counts).astype(np.uint16)
    del means

  rates = np.full((n_bins + 1, N_LAT, N_LON), np.nan, dtype=np.float32)
  for slot, k in enumerate(complete_bins):
    rates[k + 1] = _finalize_binned(
        sums[slot],
        counts[slot].astype(np.float32),
        totals,
        MIN_VALID_RESAMPLE_FRACTION,
    )
  del sums, counts
  leads = list(window.lead_hours)
  planes = aggregate_rates(rates, leads, leads)
  del rates
  stored = to_stored_units("precip", planes, source_units=da.attrs.get("units"))
  stored.tofile(str(out_dir / f"{model_key}_precip.bin"))
  log(
      f"   -> {model_key}_precip.bin: {len(leads)} leads in"
      f" {time.time() - t0:.1f}s"
  )
  return {
      "id": dataset_id,
      "type": "analysis",
      "model": model_key,
      "title": cfg["title"],
      "source": SOURCE_LABELS[cfg["source"]],
      "init_time": init_str,
      "lead_steps": len(leads),
      "lead_hours": leads,
      "streams": ["precip"],
      "variables": [var],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "accumulation_window": (
          f"mean rate over the {STEP_HOURS} h ending at each lead"
          f" ({window.frame_minutes} min frames)"
      ),
      "downloaded_utc": utc_now_str(),
  }, None


def _extract_model_streams(
    ds: xr.Dataset,
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] = print,
    max_workers: int = 2,
) -> Tuple[Optional[Dict[str, Any]], Optional[IncompleteRunError]]:
  """Extracts binary streams for the newest run of a dynamical.org dataset.

  Non-precipitation streams load only the stored 3-hourly leads; precipitation
  loads every native lead within 0..240 h and averages the rates over each
  stored interval. 2-D projected grids (HRRR) are area-binned onto the viewer
  grid; rectilinear grids are resampled with
  ``_resample_1d_rectilinear_to_global``.

  Returns:
    ``(entry, None)`` on success or ``(None, IncompleteRunError)`` when a lead
    plane of the newest run has not been published yet.

  Raises:
    KeyError: If a configured stream variable is missing from the dataset.
    ValueError: If the lead axis or the ensemble member is unusable.
  """
  if cfg.get("source") == "dynamical_analysis" or "init_time" not in ds.dims:
    return _extract_dynamical_analysis_streams(
        ds, model_key, cfg, out_dir, log=log
    )
  dataset_id = cfg["dataset"]
  init_val = ds["init_time"].values[-1]
  init_str = _iso_seconds(init_val)
  lead_hours_f = ds["lead_time"].values / np.timedelta64(1, "h")
  all_leads = np.rint(lead_hours_f).astype(int)
  if not np.allclose(lead_hours_f, all_leads):
    raise ValueError(f"{dataset_id}: lead_time axis is not in whole hours.")
  in_idx = [i for i, h in enumerate(all_leads) if 0 <= h <= MAX_LEAD_HOURS]
  in_leads = [int(all_leads[i]) for i in in_idx]
  out_leads = output_lead_hours(in_leads)
  if len(out_leads) < 2:
    raise ValueError(
        f"{dataset_id}: fewer than two {STEP_HOURS}-hourly leads within"
        f" 0..{MAX_LEAD_HOURS} h (native leads: {in_leads})."
    )
  out_pos = [in_leads.index(h) for h in out_leads]
  ref_pos = out_pos[1]
  sel: Dict[str, Any] = {"init_time": init_val}
  member: Optional[int] = None
  if "ensemble_member" in ds.dims:
    member = int(cfg.get("ensemble_member", 0))
    members = [int(m) for m in ds["ensemble_member"].values]
    if member not in members:
      raise ValueError(
          f"{dataset_id}: ensemble member {member} not in {members}."
      )
    sel["ensemble_member"] = member

  regrid: Callable[[np.ndarray], np.ndarray]
  if ds["latitude"].ndim == 2:
    reprojector = _build_2d_to_global_reprojector(
        ds["latitude"].values, ds["longitude"].values
    )
    regrid = reprojector.apply
  else:
    src_lats = np.asarray(ds["latitude"].values, dtype=np.float64)
    src_lons = np.asarray(ds["longitude"].values, dtype=np.float64)

    def regrid(planes: np.ndarray) -> np.ndarray:
      return _resample_1d_rectilinear_to_global(planes, src_lats, src_lons)

  streams = tuple(cfg["streams"])
  for stream in streams:
    if STREAM_VARIABLES[stream] not in ds:
      raise KeyError(
          f"{dataset_id} has no variable {STREAM_VARIABLES[stream]!r}."
      )

  def _one_stream(stream: str) -> Optional[IncompleteRunError]:
    var = STREAM_VARIABLES[stream]
    da = ds[var].sel(**sel)
    is_precip = stream == "precip"
    units = da.attrs.get("units")
    t0 = time.time()
    ref_plane = da.isel(lead_time=in_idx[ref_pos]).values
    ref_count = int(np.isfinite(ref_plane).sum())
    del ref_plane
    if ref_count == 0:
      return IncompleteRunError(
          f"{dataset_id} {init_str}: {var} has no data at lead"
          f" {in_leads[ref_pos]} h yet."
      )
    last_count = int(np.isfinite(da.isel(lead_time=in_idx[-1]).values).sum())
    if last_count < (1.0 - _MAX_PLANE_DEFICIT_FRACTION) * ref_count:
      return IncompleteRunError(
          f"{dataset_id} {init_str}: {var} lead {in_leads[-1]} h not"
          " published yet."
      )
    if is_precip:
      load_idx, loaded_leads, start = in_idx, in_leads, 1
    else:
      load_idx = [in_idx[p] for p in out_pos]
      loaded_leads, start = out_leads, 0
    raw = np.asarray(da.isel(lead_time=load_idx).values, dtype=np.float32)
    bad_lead = _first_deficient_lead(raw, loaded_leads, ref_count, start)
    if bad_lead is not None:
      return IncompleteRunError(
          f"{dataset_id} {init_str}: {var} lead {bad_lead} h not published"
          " yet."
      )
    grid = regrid(raw)
    del raw
    planes = aggregate_rates(grid, in_leads, out_leads) if is_precip else grid
    del grid
    stored = to_stored_units(stream, planes, source_units=units)
    stored.tofile(str(out_dir / f"{model_key}_{stream}.bin"))
    log(
        f"   -> {model_key}_{stream}.bin: {len(out_leads)} leads in"
        f" {time.time() - t0:.1f}s"
    )
    return None

  workers = max(1, min(int(max_workers), len(streams)))
  with ThreadPoolExecutor(max_workers=workers) as pool:
    results = list(pool.map(_one_stream, streams))
  for err in results:
    if err is not None:
      return None, err

  entry: Dict[str, Any] = {
      "id": dataset_id,
      "type": "forecast",
      "model": model_key,
      "title": cfg["title"],
      "source": SOURCE_LABELS[cfg["source"]],
      "init_time": init_str,
      "lead_steps": len(out_leads),
      "lead_hours": out_leads,
      "streams": list(streams),
      "variables": [STREAM_VARIABLES[s] for s in streams],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "downloaded_utc": utc_now_str(),
  }
  if member is not None:
    entry["ensemble_member"] = member
  return entry, None


def download_model_run(
    ds: xr.Dataset,
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Union[str, Path],
    log: Callable[[str], None] = print,
    max_workers: int = 2,
) -> Dict[str, Any]:
  """Writes <model>_<stream>.bin files for the newest run; returns metadata.

  Raises:
    IncompleteRunError: If the newest run is not fully published yet.
  """
  target_dir = require_data_dir(out_dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  entry, err = _extract_model_streams(
      ds, model_key, cfg, target_dir, log=log, max_workers=max_workers
  )
  if err is not None:
    raise err
  assert entry is not None
  return entry


# ---------------------------------------------------------------------------
# ECMWF Open Data IFS HRES (gs://ecmwf-open-data)
# ---------------------------------------------------------------------------


def _anonymous_gcs() -> Any:
  """Returns an anonymous gcsfs file system."""
  import gcsfs

  return gcsfs.GCSFileSystem(token="anon")


def _hres_prefix(date_str: str, cycle: str, lead_h: int) -> str:
  """Object prefix (without extension) of one HRES surface step."""
  hh = cycle[:2]
  return (
      f"{_HRES_BUCKET}/{date_str}/{cycle}/ifs/0p25/oper/"
      f"{date_str}{hh}0000-{lead_h}h-oper-fc"
  )


def _latest_hres_run_info(
    fs: Any = None,
    now: Optional[datetime.datetime] = None,
    lookback_days: int = _HRES_LOOKBACK_DAYS,
) -> Optional[Tuple[str, str, str]]:
  """Finds the newest 00z/12z HRES run whose 240 h step is published.

  Returns:
    ``(YYYYMMDD, cycle, init_iso)`` or ``None`` if no complete run exists
    within ``lookback_days``.
  """
  gcs = fs if fs is not None else _anonymous_gcs()
  moment = _resolve_now(now)
  for days_back in range(lookback_days + 1):
    day = moment - datetime.timedelta(days=days_back)
    d_str = day.strftime("%Y%m%d")
    for cycle in _HRES_CYCLES:
      if gcs.exists(f"{_hres_prefix(d_str, cycle, MAX_LEAD_HOURS)}.index"):
        return d_str, cycle, f"{day.strftime('%Y-%m-%d')}T{cycle[:2]}:00:00"
  return None


def _fetch_single_hres_step(
    gcs: Any, date_str: str, cycle: str, lead_h: int
) -> Dict[str, np.ndarray]:
  """Downloads and decodes the surface GRIB2 parameters of one HRES step.

  Raises:
    KeyError: If the step's index lacks one of the required parameters.
  """
  from multimet.timeseries_extractors.hres import decode_grib2_message

  prefix = _hres_prefix(date_str, cycle, lead_h)
  idx_text = gcs.cat(f"{prefix}.index").decode("utf-8")
  wanted = set(_HRES_STREAM_TO_PARAM.values())
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
  missing = sorted(wanted - set(byte_ranges))
  if missing:
    raise KeyError(f"{prefix}.index lacks surface parameters {missing}.")

  decoded: Dict[str, np.ndarray] = {}
  grib_path = f"{prefix}.grib2"
  for param, (start_b, end_b) in byte_ranges.items():
    msg_bytes = gcs.cat_file(grib_path, start=start_b, end=end_b)
    decoded[param] = decode_grib2_message(
        msg_bytes, (N_LAT, N_LON), param=param, context=f"{prefix}.grib2"
    ).astype(np.float32)
  return decoded


def _extract_hres_streams(
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] = print,
    fs: Any = None,
    leads: Optional[Sequence[int]] = None,
    max_workers: int = 16,
    now: Optional[datetime.datetime] = None,
    run_info: Optional[Tuple[str, str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[IncompleteRunError]]:
  """Downloads ECMWF IFS HRES 0.25 deg GRIB2 streams from gs://ecmwf-open-data.

  Total precipitation (metres accumulated since init) is de-accumulated into
  mean rates over each native step; negative packing noise is clamped to zero
  while ``NaN`` cells propagate.

  Args:
    model_key: Model key ('ecmwf_hres').
    cfg: ``DYNAMICAL_MODELS[model_key]``.
    out_dir: Staging directory receiving ``<model>_<stream>.bin`` files.
    log: Logging callback.
    fs: Optional gcsfs-like file system (``exists``, ``cat``, ``cat_file``).
    leads: Lead hours to fetch (default: all native HRES steps to 240 h).
    max_workers: Concurrent GRIB2 step downloads.
    now: Aware reference time used to search for the newest run.
    run_info: Pre-resolved ``_latest_hres_run_info`` result.

  Returns:
    ``(entry, None)`` on success or ``(None, IncompleteRunError)`` when no
    complete run is available or a plane is not fully populated.
  """
  gcs = fs if fs is not None else _anonymous_gcs()
  info = run_info if run_info is not None else _latest_hres_run_info(gcs, now)
  if info is None:
    return None, IncompleteRunError(
        f"ecmwf_open_data: no complete {MAX_LEAD_HOURS} h HRES run (00z/12z)"
        f" found in the last {_HRES_LOOKBACK_DAYS} days."
    )
  date_str, cycle, init_str = info
  native = leads if leads is not None else _HRES_LEAD_HOURS
  lead_list = [int(h) for h in native]
  out_leads = output_lead_hours(lead_list)
  out_pos = [lead_list.index(h) for h in out_leads]
  t0 = time.time()
  log(
      f"   -> Fetching {len(lead_list)} HRES GRIB2 steps for {date_str}"
      f" {cycle}..."
  )
  with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
    step_dicts = list(
        pool.map(
            lambda h: _fetch_single_hres_step(gcs, date_str, cycle, h),
            lead_list,
        )
    )

  streams = tuple(cfg["streams"])
  for stream in streams:
    param = _HRES_STREAM_TO_PARAM[stream]
    raw = np.stack([d[param] for d in step_dicts], axis=0)
    if stream == "precip":
      tp_mm = raw * np.float32(1000.0)
      rates = np.empty_like(tp_mm)
      rates[0] = np.where(
          np.isfinite(tp_mm[0]), np.float32(0.0), np.float32(np.nan)
      )
      for i in range(1, len(lead_list)):
        dt_seconds = np.float32((lead_list[i] - lead_list[i - 1]) * 3600.0)
        rates[i] = np.maximum(tp_mm[i] - tp_mm[i - 1], 0.0) / dt_seconds
      del tp_mm
      if not is_plane_complete(rates, stream):
        return None, IncompleteRunError(
            f"ecmwf_hres {init_str}: {param} planes are not fully populated."
        )
      planes = aggregate_rates(rates, lead_list, out_leads)
      units = "mm/s"
    else:
      if not is_plane_complete(raw, stream):
        return None, IncompleteRunError(
            f"ecmwf_hres {init_str}: {param} planes are not fully populated."
        )
      planes = raw[out_pos]
      units = _HRES_PARAM_UNITS[param]
    del raw
    stored = to_stored_units(stream, planes, source_units=units)
    stored.tofile(str(out_dir / f"{model_key}_{stream}.bin"))

  log(
      f"   -> {model_key}: wrote {len(streams)} streams ({len(out_leads)}"
      f" leads) in {time.time() - t0:.1f}s"
  )
  return {
      "id": cfg["dataset"],
      "type": "forecast",
      "model": model_key,
      "title": cfg["title"],
      "source": SOURCE_LABELS[cfg["source"]],
      "init_time": init_str,
      "lead_steps": len(out_leads),
      "lead_hours": out_leads,
      "streams": list(streams),
      "variables": [STREAM_VARIABLES[s] for s in streams],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "downloaded_utc": utc_now_str(),
  }, None


# ---------------------------------------------------------------------------
# NOAA PSL CPC Unified gauge analysis
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _CpcWindow:
  """Trailing ``_CPC_WINDOW_DAYS`` daily CPC planes in PSL orientation.

  Attributes:
    init_time: ISO time of lead 0 (12 UTC of the day before the first day).
    day_dates: ISO dates of the daily planes (leads 24, 48, ...).
    lead_hours: ``0, 24, ..., 24 * len(day_dates)``.
    mm_per_day: ``(len(day_dates), 360, 720)`` daily totals, ``NaN`` where the
      analysis has no value (oceans, unpublished days).
    lats: PSL latitudes (89.75 .. -89.75).
    lons: PSL longitudes (0.25 .. 359.75).
    files: Local NetCDF files the window was read from.
  """

  init_time: str
  day_dates: Tuple[str, ...]
  lead_hours: Tuple[int, ...]
  mm_per_day: np.ndarray
  lats: np.ndarray
  lons: np.ndarray
  files: Tuple[str, ...]


def _cpc_year_file(
    year: int,
    cache_dir: Path,
    now: datetime.datetime,
    log: Callable[[str], None],
    required: bool,
) -> Optional[str]:
  """Returns the cached NOAA PSL ``precip.<year>.nc``, downloading if needed.

  A cached file older than ``CPC_CACHE_REFRESH_HOURS`` is downloaded again so
  that newly published days become visible. When ``required`` is False and no
  cached copy exists, the upstream URL is probed first and ``None`` is
  returned if PSL has not published the year yet (early January).
  """
  cache_dir.mkdir(parents=True, exist_ok=True)
  local = cache_dir / f"precip.{year}.nc"
  url = NOAA_PSL_URL_TEMPLATE.format(year=year)
  if not local.exists() and not required:
    if not check_http_url_exists(
        url,
        headers=_CPC_HTTP_HEADERS,
        resource_label=f"NOAA PSL CPC NetCDF for year {year}",
    ):
      log(f"   [noaa_cpc] {url} is not published yet")
      return None
  stale = (
      local.exists()
      and (now.timestamp() - local.stat().st_mtime)
      > CPC_CACHE_REFRESH_HOURS * 3600.0
  )
  if stale:
    log(f"   [noaa_cpc] refreshing {local.name} (cache older than"
        f" {CPC_CACHE_REFRESH_HOURS:g} h)")
  return ensure_psl_cpc_netcdf(year, str(cache_dir), force_download=stale)


def _validate_cpc_dataset(ds: xr.Dataset, nc_path: str) -> pd.DatetimeIndex:
  """Checks PSL CPC NetCDF structure and returns its normalized dates."""
  for key in ("precip", "lat", "lon", "time"):
    if key not in ds:
      raise KeyError(f"Required key {key!r} not found in CPC NetCDF {nc_path}.")
  if tuple(ds["precip"].dims) != ("time", "lat", "lon"):
    raise ValueError(
        f"Expected 'precip' dimensions ('time', 'lat', 'lon') in {nc_path},"
        f" got {tuple(ds['precip'].dims)}."
    )
  lats = np.asarray(ds["lat"].values, dtype=np.float32)
  lons = np.asarray(ds["lon"].values, dtype=np.float32)
  if lats.shape != EXPECTED_PSL_LATS.shape or not np.allclose(
      lats, EXPECTED_PSL_LATS, atol=1e-3
  ):
    raise ValueError(
        f"Unexpected latitude coordinates in {nc_path}: expected 360 points"
        " descending from 89.75 to -89.75."
    )
  if lons.shape != EXPECTED_PSL_LONS.shape or not np.allclose(
      lons, EXPECTED_PSL_LONS, atol=1e-3
  ):
    raise ValueError(
        f"Unexpected longitude coordinates in {nc_path}: expected 720 points"
        " ascending from 0.25 to 359.75."
    )
  return pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).normalize()


def _cpc_day_plane(ds: xr.Dataset, index: int) -> np.ndarray:
  """Reads one daily CPC plane with fill values (< 0) mapped to NaN."""
  plane = np.asarray(ds["precip"].isel(time=index).values, dtype=np.float32)
  return np.where(plane < 0, np.float32(np.nan), plane).astype(np.float32)


def _cpc_last_valid_date(
    nc_path: str, today: datetime.date, max_scan_days: int
) -> Optional[datetime.date]:
  """Newest date up to ``today`` with any finite value, scanning backwards."""
  with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
    dates = _validate_cpc_dataset(ds, nc_path)
    candidates = [i for i in range(len(dates)) if dates[i].date() <= today]
    for i in reversed(candidates[-max_scan_days:]):
      if np.isfinite(_cpc_day_plane(ds, i)).any():
        return dates[i].date()
  return None


def _cpc_read_days(
    nc_path: str, wanted: Sequence[datetime.date]
) -> Dict[datetime.date, np.ndarray]:
  """Reads the daily planes of ``wanted`` dates that carry any finite value."""
  planes: Dict[datetime.date, np.ndarray] = {}
  with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
    dates = _validate_cpc_dataset(ds, nc_path)
    lookup = {d.date(): i for i, d in enumerate(dates)}
    for day in wanted:
      index = lookup.get(day)
      if index is None:
        continue
      plane = _cpc_day_plane(ds, index)
      if np.isfinite(plane).any():
        planes[day] = plane
  return planes


def _load_cpc_window(
    cache_dir: Path,
    now: datetime.datetime,
    log: Callable[[str], None] = print,
) -> _CpcWindow:
  """Loads the newest ``_CPC_WINDOW_DAYS`` CPC days, stitching year files.

  The window ends on the newest published day (searched backwards from
  ``now`` over at most ``_CPC_MAX_SCAN_DAYS`` days) and may span the previous
  calendar year's file. Days absent from the files stay ``NaN``.

  Raises:
    RuntimeError: If no published CPC day exists within the scan range.
  """
  today = now.date()
  files: Dict[int, str] = {}
  last_valid: Optional[datetime.date] = None
  current = _cpc_year_file(today.year, cache_dir, now, log, required=False)
  if current is not None:
    files[today.year] = current
    last_valid = _cpc_last_valid_date(current, today, _CPC_MAX_SCAN_DAYS)
  if last_valid is None:
    previous = _cpc_year_file(
        today.year - 1, cache_dir, now, log, required=True
    )
    files[today.year - 1] = previous
    last_valid = _cpc_last_valid_date(previous, today, _CPC_MAX_SCAN_DAYS)
  if last_valid is None:
    raise RuntimeError(
        f"noaa_cpc: no CPC day with data within {_CPC_MAX_SCAN_DAYS} days"
        f" before {today.isoformat()}."
    )
  lag_days = (today - last_valid).days
  if lag_days > CPC_MAX_PUBLICATION_LAG_DAYS:
    log(
        f"   [!] noaa_cpc: newest published day {last_valid.isoformat()} lags"
        f" {lag_days} days behind {today.isoformat()}"
        f" (> {CPC_MAX_PUBLICATION_LAG_DAYS})"
    )
  days = [
      last_valid - datetime.timedelta(days=_CPC_WINDOW_DAYS - 1 - k)
      for k in range(_CPC_WINDOW_DAYS)
  ]
  for year in sorted({d.year for d in days}):
    if year not in files:
      files[year] = _cpc_year_file(year, cache_dir, now, log, required=True)
  planes = np.full(
      (len(days),) + EXPECTED_PSL_LATS.shape + EXPECTED_PSL_LONS.shape,
      np.nan,
      dtype=np.float32,
  )
  for year, path in files.items():
    wanted = [d for d in days if d.year == year]
    for day, plane in _cpc_read_days(path, wanted).items():
      planes[days.index(day)] = plane
  for k, day in enumerate(days):
    if not np.isfinite(planes[k]).any():
      log(f"   [!] noaa_cpc: {day.isoformat()} has no data; stored as NaN")
  init_dt = datetime.datetime.combine(
      days[0] - datetime.timedelta(days=1),
      datetime.time(hour=_CPC_DAY_END_HOUR_UTC),
  )
  return _CpcWindow(
      init_time=init_dt.strftime("%Y-%m-%dT%H:%M:%S"),
      day_dates=tuple(d.isoformat() for d in days),
      lead_hours=tuple(24 * k for k in range(len(days) + 1)),
      mm_per_day=planes,
      lats=EXPECTED_PSL_LATS.astype(np.float64),
      lons=EXPECTED_PSL_LONS.astype(np.float64),
      files=tuple(files[y] for y in sorted(files)),
  )


def _extract_cpc_streams(
    model_key: str,
    cfg: Mapping[str, Any],
    out_dir: Path,
    window: _CpcWindow,
    log: Callable[[str], None] = print,
) -> Tuple[Optional[Dict[str, Any]], Optional[IncompleteRunError]]:
  """Writes the NOAA CPC daily gauge analysis as a 0.25 deg precip stream.

  Lead ``24 k`` holds the mean rate (mm/h) of CPC day ``k``, whose nominal
  gauge day ends at 12 UTC; lead 0 is the start of the first day. Oceans and
  unpublished days stay ``NaN``.
  """
  t0 = time.time()
  mm_day = _resample_1d_rectilinear_to_global(
      window.mm_per_day, window.lats, window.lons
  )
  rates = np.concatenate(
      [np.full((1, N_LAT, N_LON), np.nan, dtype=np.float32), mm_day], axis=0
  )
  del mm_day
  leads = list(window.lead_hours)
  planes = aggregate_rates(rates, leads, leads)
  del rates
  stored = to_stored_units("precip", planes, source_units="mm/day")
  stored.tofile(str(out_dir / f"{model_key}_precip.bin"))
  log(
      f"   -> {model_key}_precip.bin: {len(leads)} daily leads in"
      f" {time.time() - t0:.1f}s"
  )
  return {
      "id": cfg["dataset"],
      "type": "analysis",
      "model": model_key,
      "title": cfg["title"],
      "source": SOURCE_LABELS[cfg["source"]],
      "init_time": window.init_time,
      "lead_steps": len(leads),
      "lead_hours": leads,
      "day_dates": list(window.day_dates),
      "accumulation_window": (
          "each lead ends a 24 h CPC gauge day (nominally 12 UTC to 12 UTC);"
          " the stored rate is the daily total divided by 24 h"
      ),
      "streams": ["precip"],
      "variables": [STREAM_VARIABLES["precip"]],
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
      "downloaded_utc": utc_now_str(),
  }, None


# ---------------------------------------------------------------------------
# Synchronisation driver
# ---------------------------------------------------------------------------


def _carry_forward_entries(
    root: Path,
    new_dir: Path,
    fresh: Mapping[str, Dict[str, Any]],
    log: Callable[[str], None],
) -> Dict[str, Dict[str, Any]]:
  """Links or copies the current run's streams of models not refreshed now."""
  carried: Dict[str, Dict[str, Any]] = {}
  run_dir, current = current_models_metadata(root)
  if run_dir is None:
    return carried
  same_device = run_dir.stat().st_dev == new_dir.stat().st_dev
  for model_key, prev_entry in current.items():
    if model_key in fresh:
      continue
    sources = [
        run_dir / f"{model_key}_{stream}.bin"
        for stream in prev_entry.get("streams", [])
    ]
    if not sources or not all(src.is_file() for src in sources):
      log(f"[{model_key}] previous run is missing stream files; dropped")
      continue
    for src in sources:
      dst = new_dir / src.name
      if same_device:
        os.link(src, dst)
      else:
        shutil.copy2(src, dst)
    carried[model_key] = dict(prev_entry)
  return carried


def _write_status(
    status_path: Path,
    status: Dict[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
  """Persists and logs the synchronisation status."""
  write_json_atomic(status_path, status)
  log(str(status.get("message", "")))
  return status


def sync_all_models(
    data_dir: Union[str, Path],
    models: Optional[Sequence[str]] = None,
    force: bool = False,
    log: Callable[[str], None] = print,
    catalog: Any = None,
    open_dataset: Optional[Callable[[Any, str], xr.Dataset]] = None,
    cpc_cache_dir: Union[str, Path, None] = None,
    hres_fs: Any = None,
    now: Optional[datetime.datetime] = None,
    max_workers: int = 2,
) -> Dict[str, Any]:
  """Checks upstream sources and downloads models whose newest run changed.

  Models are planned and extracted one after another into a single staging
  directory; models that are up to date (or failed) are carried over from the
  current run so the published run always holds every available model.

  Args:
    data_dir: Explicit local root directory for downloaded runs.
    models: Model keys to synchronize (default ``DEFAULT_SYNC_MODELS``).
    force: Re-download runs even when the init_time is already synced.
    log: Logging callback.
    catalog: Optional pre-opened STAC catalog (or test stub). Opened lazily
      from ``STAC_CATALOG_URL`` when a dynamical.org model is selected and
      ``open_dataset`` is not supplied.
    open_dataset: Optional ``(catalog, dataset_id) -> xr.Dataset`` opener.
    cpc_cache_dir: Directory caching NOAA PSL CPC annual NetCDF files
      (default ``<data_dir>/cpc_cache``).
    hres_fs: Optional gcsfs-like file system for ECMWF Open Data.
    now: Aware reference time (default: current UTC time).
    max_workers: Concurrent stream extractions per dynamical.org model.

  Returns:
    Status dictionary written to ``<data_dir>/sync_status.json``. Its
    ``last_result`` is ``"busy"`` when another process is synchronising the
    same directory, ``"updated"`` when every planned model was refreshed,
    ``"partial"`` when some were, ``"error"`` when none could be refreshed and
    ``"up_to_date"`` when nothing needed downloading.

  Raises:
    ValueError: If an unsupported model key is requested.
  """
  root = require_data_dir(data_dir)
  runs_root = root / "runs"
  runs_root.mkdir(parents=True, exist_ok=True)
  moment = _resolve_now(now)
  selected_models = (
      list(models) if models is not None else list(DEFAULT_SYNC_MODELS)
  )
  for m in selected_models:
    if m not in DYNAMICAL_MODELS:
      raise ValueError(
          f"Unsupported weather model {m!r}. Supported:"
          f" {list(DYNAMICAL_MODELS)}"
      )
  cpc_dir = (
      Path(cpc_cache_dir).expanduser().resolve()
      if cpc_cache_dir is not None
      else root / "cpc_cache"
  )
  sources = {
      m: SOURCE_LABELS[DYNAMICAL_MODELS[m]["source"]] for m in selected_models
  }

  status_path = root / SYNC_STATUS_FILE
  status = read_sync_status(root)
  status.update({
      "last_check_utc": utc_now_str(moment),
      "check_interval_minutes": CHECK_INTERVAL_MINUTES,
      "source": ", ".join(dict.fromkeys(sources.values())),
      "sources": sources,
  })

  live_partials, stale_partials = _classify_partial_dirs(
      runs_root, moment.timestamp()
  )
  for stale in stale_partials:
    log(f"Removing abandoned staging directory {stale.name}")
    shutil.rmtree(stale, ignore_errors=True)
  if live_partials:
    status["last_result"] = "busy"
    status["message"] = (
        "Another synchronisation is in progress: "
        + ", ".join(p.name for p in live_partials)
    )
    return _write_status(status_path, status, log)

  run_dir, current = current_models_metadata(root)
  dataset_opener = (
      open_dataset if open_dataset is not None else open_dynamical_dataset
  )
  active_catalog = catalog
  plan: Dict[str, str] = {}
  datasets: Dict[str, xr.Dataset] = {}
  hres_runs: Dict[str, Tuple[str, str, str]] = {}
  cpc_windows: Dict[str, _CpcWindow] = {}
  errors: Dict[str, str] = {}

  for model_key in selected_models:
    cfg = DYNAMICAL_MODELS[model_key]
    src_type = cfg["source"]
    if src_type in ("dynamical", "dynamical_analysis"):
      if active_catalog is None and open_dataset is None:
        active_catalog = open_dynamical_catalog()
      ds = dataset_opener(active_catalog, cfg["dataset"])
      datasets[model_key] = ds
      latest = latest_init_time_str(ds)
    elif src_type == "ecmwf_open_data":
      info = _latest_hres_run_info(hres_fs, now=moment)
      if info is None:
        errors[model_key] = (
            f"ecmwf_open_data: no complete {MAX_LEAD_HOURS} h HRES run"
            f" (00z/12z) found in the last {_HRES_LOOKBACK_DAYS} days."
        )
        log(f"[{model_key}] {errors[model_key]}")
        continue
      hres_runs[model_key] = info
      latest = info[2]
    elif src_type == "noaa_psl_cpc":
      window = _load_cpc_window(cpc_dir, moment, log=log)
      cpc_windows[model_key] = window
      latest = window.init_time
    else:
      raise ValueError(f"Unknown source type {src_type!r} for {model_key!r}")

    have = current.get(model_key)
    files_ok = bool(
        have
        and run_dir is not None
        and all(
            (run_dir / f"{model_key}_{s}.bin").is_file()
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
    run_name = moment.strftime("%Y%m%dT%H%M%SZ")
    base_name, n = run_name, 1
    while (runs_root / run_name).exists():
      run_name = f"{base_name}-{n}"
      n += 1
    new_dir = runs_root / f"{run_name}.{os.getpid()}.partial"
    new_dir.mkdir(parents=True, exist_ok=False)
    entries: Dict[str, Dict[str, Any]] = {}

    for model_key in selected_models:
      if model_key not in plan:
        continue
      cfg = DYNAMICAL_MODELS[model_key]
      if model_key in datasets:
        entry, err = _extract_model_streams(
            datasets[model_key],
            model_key,
            cfg,
            new_dir,
            log=log,
            max_workers=max_workers,
        )
      elif model_key in hres_runs:
        entry, err = _extract_hres_streams(
            model_key,
            cfg,
            new_dir,
            log=log,
            fs=hres_fs,
            now=moment,
            run_info=hres_runs[model_key],
        )
      else:
        entry, err = _extract_cpc_streams(
            model_key, cfg, new_dir, cpc_windows[model_key], log=log
        )
      if err is not None:
        errors[model_key] = str(err)
        log(f"[{model_key}] keeping previous run: {err}")
        for stream in cfg["streams"]:
          partial = new_dir / f"{model_key}_{stream}.bin"
          if partial.exists():
            partial.unlink()
      else:
        assert entry is not None
        updated.append(model_key)
        entries[model_key] = entry

    if updated:
      entries.update(_carry_forward_entries(root, new_dir, entries, log))
      meta = {
          "status": "HEALTHY",
          "source": ", ".join(
              dict.fromkeys(e["source"] for e in entries.values())
          ),
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
          "source": v.get("source"),
      }
      for k, v in now_current.items()
  }
  status["errors"] = errors
  status["updated_models"] = updated
  if updated and not errors:
    status["last_result"] = "updated"
    status["message"] = "Downloaded new run: " + ", ".join(
        f"{k} {plan[k]}" for k in updated
    )
  elif updated:
    status["last_result"] = "partial"
    status["message"] = (
        "Downloaded new run: "
        + ", ".join(f"{k} {plan[k]}" for k in updated)
        + "; failed: "
        + "; ".join(f"{k}: {v}" for k, v in errors.items())
    )
  elif errors:
    status["last_result"] = "error"
    status["message"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
  else:
    status["last_result"] = "up_to_date"
    status["message"] = "All models already have the newest published run."
  return _write_status(status_path, status, log)


def sync_model(
    data_dir: Union[str, Path],
    model_key: str,
    force: bool = False,
    log: Callable[[str], None] = print,
    catalog: Any = None,
    open_dataset: Optional[Callable[[Any, str], xr.Dataset]] = None,
    cpc_cache_dir: Union[str, Path, None] = None,
    hres_fs: Any = None,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
  """Synchronizes a single model into data_dir."""
  return sync_all_models(
      data_dir=data_dir,
      models=[model_key],
      force=force,
      log=log,
      catalog=catalog,
      open_dataset=open_dataset,
      cpc_cache_dir=cpc_cache_dir,
      hres_fs=hres_fs,
      now=now,
  )


class WeatherSynchronizer:
  """Manages scheduled and on-demand forecast synchronization in a data_dir."""

  def __init__(
      self,
      data_dir: Union[str, Path],
      models: Optional[Sequence[str]] = None,
      cpc_cache_dir: Union[str, Path, None] = None,
  ):
    self.data_dir = require_data_dir(data_dir)
    self.models = (
        list(models) if models is not None else list(DEFAULT_SYNC_MODELS)
    )
    self.cpc_cache_dir = (
        Path(cpc_cache_dir).expanduser().resolve()
        if cpc_cache_dir is not None
        else self.data_dir / "cpc_cache"
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
      hres_fs: Any = None,
      now: Optional[datetime.datetime] = None,
  ) -> Dict[str, Any]:
    """Synchronizes all configured models."""
    return sync_all_models(
        data_dir=self.data_dir,
        models=self.models,
        force=force,
        log=log,
        catalog=catalog,
        open_dataset=open_dataset,
        cpc_cache_dir=self.cpc_cache_dir,
        hres_fs=hres_fs,
        now=now,
    )

  def sync_model(
      self,
      model_key: str,
      force: bool = False,
      log: Callable[[str], None] = print,
      catalog: Any = None,
      open_dataset: Optional[Callable[[Any, str], xr.Dataset]] = None,
      hres_fs: Any = None,
      now: Optional[datetime.datetime] = None,
  ) -> Dict[str, Any]:
    """Synchronizes a single model."""
    return sync_model(
        data_dir=self.data_dir,
        model_key=model_key,
        force=force,
        log=log,
        catalog=catalog,
        open_dataset=open_dataset,
        cpc_cache_dir=self.cpc_cache_dir,
        hres_fs=hres_fs,
        now=now,
    )
