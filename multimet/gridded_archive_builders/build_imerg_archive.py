# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ETL pipeline to build the unified Open-MultiMet daily IMERG surface archive.

Ingests NASA GPM IMERG Early V07 daily precipitation at its native 0.1 degree
resolution:

- Spatial resolution: 0.1 degree x 0.1 degree global grid
  - ``latitude``: 1,800 points from -89.95 to 89.95 (ascending)
  - ``longitude``: 3,600 points from -179.95 to 179.95 (ascending)
- Temporal resolution: Daily aggregation (00:00:00 UTC to 24:00:00 UTC)
- Variable: ``imerg_precipitation`` (mm/day, ``float32``)

Supported upstream sources:

#. ``gesdisc``: Discovers the exact published Level 3 Daily NetCDF-4 granule
   (``3B-DAY-E.MS.MRG.3IMERG.*.V07*.nc4``) via NASA CMR and downloads it from
   NASA GES DISC using Earthdata Login credentials (via ``~/.netrc``,
   environment variables, or CLI flags).
#. ``local``: Ingests either pre-downloaded daily NetCDF-4 files
   (``--local_format=nc4``) or 48 half-hourly HDF5 granules
   (``--local_format=h5``) from a local directory. In ``h5`` mode, all 48
   unique half-hourly intervals for the day must be present, and all 48
   half-hourly observations at a grid cell must be valid for the daily total at
   that cell to be finite; any cell with missing half-hour observations is
   masked to ``NaN``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import concurrent.futures
import datetime
import io
import logging
import os
import re
import threading

import h5py  # type: ignore[import-untyped]
from multimet.utils import storage
from multimet.utils.http import (
    DEFAULT_CMR_GRANULES_URL,
    EarthdataSession,
    download_http_file,
    get_earthdata_credentials_from_netrc,
    query_cmr_granules,
)
import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr

DEFAULT_GESDISC_URL = (
    "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07"
)
IMERG_HHR_SHORT_NAME = "GPM_3IMERGHHE"
IMERG_DAILY_SHORT_NAME = "GPM_3IMERGDE"
DEFAULT_START_DATE = "2000-06-01"

LAT_COUNT = 1800
LON_COUNT = 3600
IMERG_LATS = np.linspace(-89.95, 89.95, LAT_COUNT, dtype=np.float32)
IMERG_LONS = np.linspace(-179.95, 179.95, LON_COUNT, dtype=np.float32)
IMERG_VARIABLE = "imerg_precipitation"

# All 48 half-hour start tokens (HHMMSS) in a complete UTC day.
EXPECTED_HHR_START_TOKENS: frozenset[str] = frozenset(
    f"{hour:02d}{minute:02d}00"
    for hour in range(24)
    for minute in (0, 30)
)
_HHR_START_TOKEN_RE = re.compile(r"-S(?P<token>\d{6})")

IMERG_ATTRS = {
    "title": "Open-MultiMet NASA GPM IMERG Early V07 Daily Surface Archive",
    "spatial_resolution": "0.1 degree x 0.1 degree",
    "temporal_resolution": (
        "Daily (00:00:00 UTC to 24:00:00 UTC, left-labeled)"
    ),
    "units": "precipitation [mm]",
    "citation": (
        "Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, Jackson"
        " Tan (2024), GPM IMERG Early Precipitation L3 Half Hourly 0.1"
        " degree x 0.1 degree V07, Greenbelt, MD, GES DISC,"
        " 10.5067/GPM/IMERG/3B-HH-E/07"
    ),
    "license": "CC-BY-4.0",
    "institution": "NASA GSFC / Open-MultiMet",
}

_netcdf_lock = threading.Lock()

__all__ = [
    "DEFAULT_CMR_GRANULES_URL",
    "DEFAULT_GESDISC_URL",
    "DEFAULT_START_DATE",
    "EXPECTED_HHR_START_TOKENS",
    "EarthdataSession",
    "GESDISCImergSource",
    "IMERG_ATTRS",
    "IMERG_DAILY_SHORT_NAME",
    "IMERG_HHR_SHORT_NAME",
    "IMERG_LATS",
    "IMERG_LONS",
    "IMERG_VARIABLE",
    "LAT_COUNT",
    "LON_COUNT",
    "LocalImergSource",
    "build_arg_parser",
    "build_batch_dataset",
    "build_imerg_archive",
    "download_daily_imerg",
    "get_earthdata_credentials_from_netrc",
    "main",
    "parse_imerg_netcdf_to_grid",
    "query_cmr_granules",
    "write_batch_in_place",
    "write_batch_to_zarr",
]


def _date_token_regex(date_str: str) -> re.Pattern[str]:
  """Builds a regex matching a delimited YYYYMMDD date token in filenames."""
  return re.compile(rf"(?:^|[._-]){re.escape(date_str)}(?:-S\d{{6}}|[._-])")


def download_daily_imerg(
    url: str,
    dest_path: str,
    session: requests.Session | None = None,
) -> str:
  """Downloads a daily IMERG NetCDF-4 file from NASA GES DISC."""
  if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
    return dest_path

  active_session = session if session is not None else EarthdataSession()
  return download_http_file(
      url,
      dest_path,
      session=active_session,
      timeout=120,
      min_bytes=1024,
      resource_label="NASA GES DISC",
  )


def parse_imerg_netcdf_to_grid(
    nc_path: str,
    expected_date: pd.Timestamp | None = None,
) -> np.ndarray:
  """Reads a NASA IMERG V07 daily NetCDF-4 file into a (lat, lon) float32 grid.

  Strictly validates:
  - Required V07 precipitation variable (``precipitation``; rejects legacy V06
    ``precipitationCal``)
  - Required ``lat`` and ``lon`` coordinates matching ``IMERG_LATS`` /
    ``IMERG_LONS``
  - Internal ``time`` coordinate matching ``expected_date`` when provided
  - 2D shape ``(1800, 3600)`` after squeezing ``time`` and transposing
    ``(lon, lat)`` -> ``(lat, lon)``
  - At least one finite value in the daily grid

  Args:
    nc_path: Path to the local daily NetCDF-4 file.
    expected_date: Optional expected UTC date for internal timestamp
      verification.

  Returns:
    2D ``float32`` array of shape ``(LAT_COUNT, LON_COUNT)`` with negative/fill
    values masked to ``np.nan``.
  """
  with _netcdf_lock:
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
      if "precipitation" not in ds:
        raise KeyError(
            f"Required V07 variable 'precipitation' not found in {nc_path}: "
            f"{list(ds.data_vars)}"
        )

      for coord_name in ("lat", "lon"):
        if coord_name not in ds.coords and coord_name not in ds:
          raise KeyError(
              f"Required coordinate {coord_name!r} not found in {nc_path}."
          )

      if "time" in ds.coords or "time" in ds:
        time_vals = pd.to_datetime(ds["time"].values)
        if len(time_vals) != 1:
          raise ValueError(
              f"Expected single daily timestamp in {nc_path}, got "
              f"{len(time_vals)}."
          )
        if expected_date is not None:
          file_date = pd.Timestamp(time_vals[0]).normalize()
          exp_date = pd.Timestamp(expected_date).normalize()
          if file_date != exp_date:
            raise ValueError(
                f"IMERG NetCDF {nc_path} internal time "
                f"{file_date.strftime('%Y-%m-%d')} does not match expected "
                f"date {exp_date.strftime('%Y-%m-%d')}."
            )
      elif expected_date is not None:
        raise KeyError(
            f"Required 'time' coordinate not found in IMERG NetCDF {nc_path}."
        )

      file_lats = np.asarray(ds["lat"].values, dtype=np.float32)
      file_lons = np.asarray(ds["lon"].values, dtype=np.float32)
      if file_lats.shape != IMERG_LATS.shape or not np.allclose(
          file_lats, IMERG_LATS, atol=1e-2
      ):
        raise ValueError(
            f"Latitude coordinate in {nc_path} does not match expected "
            f"IMERG_LATS (shape={file_lats.shape})."
        )
      if file_lons.shape != IMERG_LONS.shape or not np.allclose(
          file_lons, IMERG_LONS, atol=1e-2
      ):
        raise ValueError(
            f"Longitude coordinate in {nc_path} does not match expected "
            f"IMERG_LONS (shape={file_lons.shape})."
        )

      da = ds["precipitation"]
      if "time" in da.dims:
        da = da.squeeze("time")

      if da.dims == ("lon", "lat"):
        da = da.transpose("lat", "lon")
      elif da.dims != ("lat", "lon"):
        raise ValueError(
            f"Unexpected dimensions {da.dims} for 'precipitation' in "
            f"{nc_path}; expected ('lat', 'lon') or ('lon', 'lat')."
        )

      grid = np.asarray(da.values, dtype=np.float32)

  if grid.shape != (LAT_COUNT, LON_COUNT):
    raise ValueError(
        f"Unexpected IMERG grid shape {grid.shape} in {nc_path}; "
        f"expected ({LAT_COUNT}, {LON_COUNT})."
    )

  grid = np.where((grid >= 0.0) & (~np.isnan(grid)), grid, np.nan).astype(
      np.float32
  )
  if not np.isfinite(grid).any():
    raise ValueError(
        f"IMERG NetCDF {nc_path} contains no finite precipitation values "
        "(all-NaN)."
    )
  return grid


def _parse_h5_granule_bytes(content: bytes) -> np.ndarray:
  """Parses a single half-hourly IMERG HDF5 granule into a (lat, lon) array."""
  with h5py.File(io.BytesIO(content), "r") as f:
    if "Grid" not in f:
      raise KeyError("HDF5 granule is missing required 'Grid' group.")
    grid_group = f["Grid"]
    if "precipitation" not in grid_group:
      raise KeyError(
          "Required V07 'Grid/precipitation' dataset not found in HDF5 granule."
      )
    precip_ds = grid_group["precipitation"]

    if "lat" in grid_group and "lon" in grid_group:
      h5_lats = np.asarray(grid_group["lat"][:], dtype=np.float32)
      h5_lons = np.asarray(grid_group["lon"][:], dtype=np.float32)
      if h5_lats.shape != IMERG_LATS.shape or not np.allclose(
          h5_lats, IMERG_LATS, atol=1e-2
      ):
        raise ValueError(
            f"Unexpected 'Grid/lat' in HDF5 granule: shape={h5_lats.shape}"
        )
      if h5_lons.shape != IMERG_LONS.shape or not np.allclose(
          h5_lons, IMERG_LONS, atol=1e-2
      ):
        raise ValueError(
            f"Unexpected 'Grid/lon' in HDF5 granule: shape={h5_lons.shape}"
        )

    data = precip_ds[:]
    if data.ndim == 3:
      if data.shape[0] != 1:
        raise ValueError(
            f"Expected single time slice in HDF5 granule, got shape "
            f"{data.shape}."
        )
      data = data[0]
    if data.shape != (LON_COUNT, LAT_COUNT):
      raise ValueError(
          f"Unexpected HDF5 precipitation array shape {data.shape}; "
          f"expected ({LON_COUNT}, {LAT_COUNT})."
      )
    return np.asarray(data.T, dtype=np.float32)


def _read_and_parse_h5_granule(fpath: str) -> np.ndarray:
  """Reads a local HDF5 granule from disk and parses its precipitation grid."""
  with open(fpath, "rb") as f:
    content = f.read()
  return _parse_h5_granule_bytes(content)


class GESDISCImergSource:
  """Fetches NASA GPM IMERG V07 Daily NetCDF-4 files from NASA GES DISC."""

  def __init__(
      self,
      cache_dir: str,
      base_url: str = DEFAULT_GESDISC_URL,
      username: str | None = None,
      password: str | None = None,
      token: str | None = None,
      netrc_path: str | None = None,
      cleanup_cache: bool = False,
  ):
    self.cache_dir = cache_dir
    self.base_url = base_url.rstrip("/")
    self.cleanup_cache = cleanup_cache
    self.session = EarthdataSession(
        username=username,
        password=password,
        token=token,
        netrc_path=netrc_path,
    )
    tail = self.base_url.rsplit("/", 1)[-1]
    if "." in tail:
      short_part, ver_part = tail.split(".", 1)
      self.collection_short_name = short_part or IMERG_DAILY_SHORT_NAME
      self.version: str | None = ver_part
    else:
      self.collection_short_name = tail or IMERG_DAILY_SHORT_NAME
      self.version = None

  def _resolve_daily_granule_url(self, date: pd.Timestamp) -> str:
    """Discovers the exact daily NetCDF-4 URL for ``date`` via NASA CMR."""
    cmr_kwargs: dict[str, object] = {}
    if self.version is not None:
      cmr_kwargs["version"] = self.version
    cmr_urls = query_cmr_granules(
        self.collection_short_name, date, **cmr_kwargs  # type: ignore[arg-type]
    )
    nc_urls = [u for u in cmr_urls if u.endswith((".nc4", ".nc"))]
    if not nc_urls:
      raise FileNotFoundError(
          f"No published IMERG V07 daily NetCDF-4 granule found in NASA CMR "
          f"for {date.strftime('%Y-%m-%d')} "
          f"(collection={self.collection_short_name}, version={self.version})."
      )
    if len(nc_urls) > 1:
      raise ValueError(
          f"Multiple daily IMERG NetCDF-4 granules returned by NASA CMR for "
          f"{date.strftime('%Y-%m-%d')}: {nc_urls}"
      )
    return nc_urls[0]

  def extract_date(self, date: pd.Timestamp) -> np.ndarray:
    """Downloads and parses the daily IMERG NetCDF-4 file for ``date``."""
    date_str = date.strftime("%Y%m%d")
    token_re = _date_token_regex(date_str)
    os.makedirs(self.cache_dir, exist_ok=True)

    cached_matches = [
        os.path.join(self.cache_dir, fname)
        for fname in sorted(os.listdir(self.cache_dir))
        if fname.endswith((".nc4", ".nc")) and token_re.search(fname)
    ]
    if len(cached_matches) > 1:
      raise ValueError(
          f"Multiple conflicting cached IMERG files found for "
          f"{date.strftime('%Y-%m-%d')}: {cached_matches}"
      )
    if len(cached_matches) == 1:
      nc_path = cached_matches[0]
    else:
      target_url = self._resolve_daily_granule_url(date)
      dest_path = os.path.join(self.cache_dir, os.path.basename(target_url))
      nc_path = download_daily_imerg(
          target_url, dest_path, session=self.session
      )

    grid = parse_imerg_netcdf_to_grid(nc_path, expected_date=date)
    if self.cleanup_cache and os.path.exists(nc_path):
      os.remove(nc_path)
    return grid


class LocalImergSource:
  """Ingests daily IMERG grids from local NetCDF-4 or 48 half-hourly HDF5s."""

  def __init__(
      self,
      local_dir: str,
      local_format: str = "nc4",
      granule_workers: int = 8,
  ):
    if local_format not in ("nc4", "h5"):
      raise ValueError(
          f"Invalid local_format={local_format!r}. Must be 'nc4' or 'h5'."
      )
    if not os.path.isdir(local_dir):
      raise FileNotFoundError(
          f"Local IMERG directory does not exist: {local_dir}"
      )
    self.local_dir = local_dir
    self.local_format = local_format
    self.granule_workers = granule_workers

  def _candidate_dirs(self, date: pd.Timestamp) -> list[str]:
    """Returns candidate directories to search for ``date``."""
    month_str = date.strftime("%Y%m")
    year_str = date.strftime("%Y")
    mm_str = date.strftime("%m")
    candidates = [
        self.local_dir,
        os.path.join(self.local_dir, month_str),
        os.path.join(self.local_dir, year_str, mm_str),
        os.path.join(self.local_dir, year_str),
    ]
    seen: list[str] = []
    for d in candidates:
      if os.path.isdir(d) and d not in seen:
        seen.append(d)
    return seen

  def extract_date(self, date: pd.Timestamp) -> np.ndarray:
    """Extracts the daily IMERG grid for ``date`` in ``self.local_format``."""
    date_str = date.strftime("%Y%m%d")
    token_re = _date_token_regex(date_str)
    candidate_dirs = self._candidate_dirs(date)

    if self.local_format == "nc4":
      nc_files: list[str] = []
      for d in candidate_dirs:
        for fname in sorted(os.listdir(d)):
          if fname.endswith((".nc4", ".nc")) and token_re.search(fname):
            nc_files.append(os.path.join(d, fname))
      if not nc_files:
        raise FileNotFoundError(
            f"No local IMERG NetCDF-4 file found for "
            f"{date.strftime('%Y-%m-%d')} in {self.local_dir}."
        )
      if len(nc_files) > 1:
        raise ValueError(
            f"Multiple conflicting local IMERG NetCDF-4 files found for "
            f"{date.strftime('%Y-%m-%d')} in {self.local_dir}: {nc_files}"
        )
      return parse_imerg_netcdf_to_grid(nc_files[0], expected_date=date)

    # self.local_format == "h5"
    h5_files: list[str] = []
    for d in candidate_dirs:
      for fname in sorted(os.listdir(d)):
        if fname.endswith((".RT-H5", ".HDF5", ".h5")) and token_re.search(
            fname
        ):
          h5_files.append(os.path.join(d, fname))
      if h5_files:
        break

    if len(h5_files) != 48:
      raise ValueError(
          f"Expected 48 half-hourly IMERG HDF5 granules for "
          f"{date.strftime('%Y-%m-%d')}, found {len(h5_files)} in "
          f"{self.local_dir}."
      )

    observed_tokens: set[str] = set()
    for fpath in h5_files:
      fname = os.path.basename(fpath)
      match = _HHR_START_TOKEN_RE.search(fname)
      if match is None:
        raise ValueError(
            f"Cannot parse half-hour start token '-S<HHMMSS>' from HDF5 "
            f"granule filename: {fname}"
        )
      observed_tokens.add(match.group("token"))

    if observed_tokens != EXPECTED_HHR_START_TOKENS:
      missing_tokens = sorted(EXPECTED_HHR_START_TOKENS - observed_tokens)
      extra_tokens = sorted(observed_tokens - EXPECTED_HHR_START_TOKENS)
      raise ValueError(
          f"Half-hourly HDF5 granules for {date.strftime('%Y-%m-%d')} do not "
          f"cover all 48 unique 30-minute intervals (missing={missing_tokens}, "
          f"unexpected={extra_tokens})."
      )

    if self.granule_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(48, self.granule_workers)
      ) as pool:
        granule_grids = list(pool.map(_read_and_parse_h5_granule, h5_files))
    else:
      granule_grids = [_read_and_parse_h5_granule(fpath) for fpath in h5_files]

    daily_sum = np.zeros((LAT_COUNT, LON_COUNT), dtype=np.float64)
    valid_counts = np.zeros((LAT_COUNT, LON_COUNT), dtype=np.int32)

    for grid_rate in granule_grids:
      valid_mask = (grid_rate >= 0.0) & (~np.isnan(grid_rate))
      daily_sum[valid_mask] += grid_rate[valid_mask] * 0.5
      valid_counts[valid_mask] += 1

    # Missing data in ALWAYS means missing data out: require all 48 half-hour
    # observations at a grid cell to be valid for its daily sum to be finite.
    complete_mask = valid_counts == 48
    result = np.full((LAT_COUNT, LON_COUNT), np.nan, dtype=np.float32)
    result[complete_mask] = daily_sum[complete_mask].astype(np.float32)
    if not np.isfinite(result).any():
      raise ValueError(
          f"Accumulated daily IMERG grid for {date.strftime('%Y-%m-%d')} "
          "contains no finite values (all-NaN)."
      )
    return result


def build_batch_dataset(
    batch_dates: Sequence[pd.Timestamp],
    batch_grids: Sequence[np.ndarray],
) -> xr.Dataset:
  """Assembles a batch of daily IMERG grids into the canonical Zarr schema."""
  return xr.Dataset(
      data_vars={
          IMERG_VARIABLE: (
              ["time", "latitude", "longitude"],
              np.stack(batch_grids, axis=0).astype(np.float32),
          ),
      },
      coords={
          "time": list(batch_dates),
          "latitude": IMERG_LATS,
          "longitude": IMERG_LONS,
      },
      attrs=dict(IMERG_ATTRS),
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    is_initial_write: bool = False,
    consolidated: bool = True,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store."""
  storage.write_dataset_batch_to_zarr(
      ds_batch,
      target_zarr_url,
      project=project,
      is_initial_write=is_initial_write,
      consolidated=consolidated,
      time_chunk_size=1,
  )


def write_batch_in_place(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    date_to_idx: dict[str, int] | None = None,
) -> None:
  """Writes a batch of dates directly in-place into existing Zarr slices."""
  storage.write_dataset_batch_in_place(
      ds_batch,
      target_zarr_url,
      project=project,
      date_to_idx=date_to_idx,
  )


def build_imerg_archive(
    target_zarr: str,
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    source_type: str = "gesdisc",
    local_format: str = "nc4",
    project: str | None = None,
    gesdisc_url: str = DEFAULT_GESDISC_URL,
    batch_size: int = 30,
    num_workers: int = 4,
    granule_workers: int = 8,
    cache_dir: str | None = None,
    cleanup_cache: bool = False,
    overwrite: bool = False,
    in_place: bool = False,
    local_dir: str | None = None,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    earthdata_token: str | None = None,
    netrc_path: str | None = None,
) -> None:
  """Builds or updates the unified IMERG daily native-resolution Zarr archive."""
  if source_type not in ("gesdisc", "local"):
    raise ValueError(
        f"Invalid source_type={source_type!r}. Must be 'gesdisc' or 'local'."
    )
  if local_dir is not None:
    source_type = "local"
  if source_type == "local" and not local_dir:
    raise ValueError("--local_dir must be provided when --source=local.")

  full_target_url, _, store_exists, has_consolidated, mapper = (
      storage.inspect_zarr_store(
          target_zarr, project=project, overwrite=overwrite
      )
  )

  if end_date is None:
    end_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

  if pd.Timestamp(end_date) < pd.Timestamp(start_date):
    raise ValueError(
        f"end_date ({end_date}) must be >= start_date ({start_date})."
    )

  requested_dates = pd.date_range(start_date, end_date, freq="1D")
  date_to_idx: dict[str, int] = {}
  in_place_dates = pd.DatetimeIndex([])
  append_dates = requested_dates

  if in_place:
    if not store_exists:
      raise ValueError(
          f"Cannot run --in_place: target store {full_target_url} does not "
          "exist."
      )
    time_pd = storage.decode_zarr_time_index(mapper)
    date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)}
    in_place_dates = requested_dates
    append_dates = pd.DatetimeIndex([])
  elif store_exists:
    append_dates, date_to_idx = storage.plan_archive_resume(
        mapper, requested_dates, has_consolidated=has_consolidated
    )
    if len(append_dates) == 0:
      logging.info(
          "Store already contains all valid dates up to %s. Nothing to do!",
          end_date,
      )
      return

  with storage.managed_cache_dir(
      cache_dir, cleanup_cache, prefix="imerg_cache_"
  ) as effective_cache_dir:
    if source_type == "local":
      logging.info(
          "Using local directory archive from %s (format=%s)",
          local_dir,
          local_format,
      )
      source: LocalImergSource | GESDISCImergSource = LocalImergSource(
          local_dir=str(local_dir),
          local_format=local_format,
          granule_workers=granule_workers,
      )
    else:
      logging.info("Using NASA GES DISC public daily NetCDF archive")
      source = GESDISCImergSource(
          cache_dir=effective_cache_dir,
          base_url=gesdisc_url,
          username=earthdata_username,
          password=earthdata_password,
          token=earthdata_token,
          netrc_path=netrc_path,
          cleanup_cache=cleanup_cache,
      )

    is_first_write = not store_exists

    def _extract_chunk(
        chunk_dates: pd.DatetimeIndex,
    ) -> list[tuple[pd.Timestamp, np.ndarray]]:
      if num_workers > 1 and len(chunk_dates) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(chunk_dates), num_workers)
        ) as executor:
          futures = {
              executor.submit(source.extract_date, dt): dt
              for dt in chunk_dates
          }
          results_map: dict[pd.Timestamp, np.ndarray] = {}
          for future in concurrent.futures.as_completed(futures):
            dt_item = futures[future]
            results_map[dt_item] = future.result()
          return [(dt, results_map[dt]) for dt in chunk_dates]
      return [(dt, source.extract_date(dt)) for dt in chunk_dates]

    if len(in_place_dates) > 0:
      for i in tqdm.trange(
          0, len(in_place_dates), batch_size, desc="IMERG In-Place Update"
      ):
        chunk = in_place_dates[i : i + batch_size]
        extracted = _extract_chunk(chunk)
        ip_dates = [dt for dt, _ in extracted]
        ip_grids = [grid for _, grid in extracted]
        ds_batch = build_batch_dataset(ip_dates, ip_grids)
        write_batch_in_place(
            ds_batch, full_target_url, project=project, date_to_idx=date_to_idx
        )

    if len(append_dates) > 0:
      for i in tqdm.trange(
          0, len(append_dates), batch_size, desc="Processing IMERG Batches"
      ):
        chunk = append_dates[i : i + batch_size]
        extracted = _extract_chunk(chunk)
        batch_dates = [dt for dt, _ in extracted]
        batch_grids = [grid for _, grid in extracted]
        ds_batch = build_batch_dataset(batch_dates, batch_grids)
        write_batch_to_zarr(
            ds_batch,
            full_target_url,
            project=project,
            is_initial_write=is_first_write,
            consolidated=has_consolidated or is_first_write,
        )
        is_first_write = False

    logging.info(
        "IMERG archive build complete for %s to %s!", start_date, end_date
    )


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the IMERG archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-imerg-archive",
      description="Build the unified IMERG daily surface precipitation archive.",
  )
  parser.add_argument(
      "--target_zarr",
      type=str,
      required=True,
      help="Destination Zarr store (local path or explicit gs:// URI).",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=DEFAULT_START_DATE,
      help="First date to ingest (YYYY-MM-DD, default 2000-06-01).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=None,
      help="Last date to ingest (YYYY-MM-DD, default yesterday UTC).",
  )
  parser.add_argument(
      "--source",
      type=str,
      default="gesdisc",
      choices=["gesdisc", "local"],
      help="Upstream source type: 'gesdisc' (NASA GES DISC) or 'local'.",
  )
  parser.add_argument(
      "--local_format",
      type=str,
      default="nc4",
      choices=["nc4", "h5"],
      help="Local file format when --source=local ('nc4' or 'h5').",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="Optional GCP project used for billing/auth of gs:// requests.",
  )
  parser.add_argument(
      "--gesdisc_url",
      type=str,
      default=DEFAULT_GESDISC_URL,
      help="Base URL for NASA GES DISC IMERG V07 daily archive.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=30,
      help="Number of daily slices accumulated before each Zarr write.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=4,
      help="Number of concurrent date download/extraction workers.",
  )
  parser.add_argument(
      "--granule_workers",
      type=int,
      default=8,
      help="Number of concurrent HDF5 granule workers per date (local mode).",
  )
  parser.add_argument(
      "--cache_dir",
      "--local_cache",
      dest="cache_dir",
      type=str,
      default=None,
      help="Local directory used to stage downloaded NASA GES DISC files.",
  )
  parser.add_argument(
      "--cleanup_cache",
      action="store_true",
      help="Delete downloaded NetCDF files after processing and on exit.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Delete and rebuild the target store instead of resuming it.",
  )
  parser.add_argument(
      "--in_place",
      action="store_true",
      help="Rewrite dates that already exist in the target store in place.",
  )
  parser.add_argument(
      "--local_dir",
      type=str,
      default=None,
      help="Local directory containing pre-downloaded NetCDF-4 or HDF5 files.",
  )
  parser.add_argument(
      "--earthdata_username",
      type=str,
      default=None,
      help="NASA Earthdata Login username.",
  )
  parser.add_argument(
      "--earthdata_password",
      type=str,
      default=None,
      help="NASA Earthdata Login password.",
  )
  parser.add_argument(
      "--earthdata_token",
      type=str,
      default=None,
      help="NASA Earthdata Bearer token.",
  )
  parser.add_argument(
      "--netrc_path",
      type=str,
      default=None,
      help="Custom path to a .netrc file containing Earthdata credentials.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-imerg-archive``."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  build_imerg_archive(
      target_zarr=args.target_zarr,
      start_date=args.start_date,
      end_date=args.end_date,
      source_type=args.source,
      local_format=args.local_format,
      project=args.project,
      gesdisc_url=args.gesdisc_url,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      granule_workers=args.granule_workers,
      cache_dir=args.cache_dir,
      cleanup_cache=args.cleanup_cache,
      overwrite=args.overwrite,
      in_place=args.in_place,
      local_dir=args.local_dir,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
  )


if __name__ == "__main__":
  main()
