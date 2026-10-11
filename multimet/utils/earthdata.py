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

"""Shared NASA Earthdata Login session, CMR granule discovery, and IMERG helpers."""

from __future__ import annotations

import os
import threading

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
from utils.file_paths import NASA_GESDISC_IMERG_DAILY_URL
import xarray as xr

DEFAULT_GESDISC_URL = NASA_GESDISC_IMERG_DAILY_URL
IMERG_HHR_SHORT_NAME = "GPM_3IMERGHHE"
IMERG_DAILY_SHORT_NAME = "GPM_3IMERGDE"

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

_netcdf_lock = threading.Lock()


def build_gesdisc_daily_url(
    dt: pd.Timestamp,
    base_url: str = DEFAULT_GESDISC_URL,
    version_suffix: str = "V07B",
) -> str:
  """Constructs the NASA GES DISC HTTPS URL for a daily IMERG Early NetCDF-4 file."""
  dt = pd.to_datetime(dt)
  yyyy = dt.strftime("%Y")
  mm = dt.strftime("%m")
  yyyymmdd = dt.strftime("%Y%m%d")
  filename = (
      f"3B-DAY-E.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.{version_suffix}.nc4"
  )
  return f"{base_url.rstrip('/')}/{yyyy}/{mm}/{filename}"


def download_daily_imerg(
    url: str,
    dest_path: str,
    session: requests.Session | None = None,
    max_retries: int = 3,
) -> str:
  """Downloads a daily IMERG NetCDF-4 file from NASA GES DISC."""
  if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
    return dest_path
  if session is None:
    session = EarthdataSession()

  import logging
  from pathlib import Path
  import time

  os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
  temp_path = f"{dest_path}.tmp.{os.getpid()}.{time.time_ns()}"

  for attempt in range(max_retries):
    with session.get(url, stream=True, timeout=120) as resp:
      if resp.status_code in (401, 403):
        Path(temp_path).unlink(missing_ok=True)
        raise PermissionError(
            f"NASA GES DISC returned HTTP {resp.status_code} Unauthorized "
            f"for URL:\n  {url}\nAccess to NASA IMERG data requires NASA "
            "Earthdata Login authentication."
        )
      if resp.status_code == 404:
        Path(temp_path).unlink(missing_ok=True)
        raise FileNotFoundError(
            f"NASA GES DISC returned HTTP 404 Not Found for URL: {url}"
        )
      if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
        logging.warning(
            "Transient HTTP %d from %s (attempt %d/%d); retrying...",
            resp.status_code,
            url,
            attempt + 1,
            max_retries,
        )
        time.sleep(2**attempt)
        continue
      resp.raise_for_status()
      with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
          if chunk:
            f.write(chunk)

    if os.path.getsize(temp_path) <= 1024:
      Path(temp_path).unlink(missing_ok=True)
      raise ValueError(
          f"Downloaded IMERG granule is unexpectedly small "
          f"({os.path.getsize(temp_path)} bytes): {url}"
      )
    os.replace(temp_path, dest_path)
    return dest_path

  Path(temp_path).unlink(missing_ok=True)
  raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


def parse_imerg_netcdf_to_grid(nc_path: str) -> np.ndarray:
  """Reads a NASA IMERG V07 daily NetCDF-4 file into a (lat, lon) float32 grid."""
  with _netcdf_lock, xr.open_dataset(nc_path) as ds:
    if "precipitation" not in ds:
      raise KeyError(
          f"IMERG V07 variable 'precipitation' not found in {nc_path} "
          f"(found variables: {list(ds.data_vars)}). Legacy V06 "
          "'precipitationCal' files are not supported."
      )
    if "lat" not in ds or "lon" not in ds:
      raise KeyError(
          f"Required coordinates 'lat' and 'lon' not found in {nc_path}."
      )

    file_lats = np.asarray(ds["lat"].values, dtype=np.float32)
    file_lons = np.asarray(ds["lon"].values, dtype=np.float32)
    if file_lats.shape != (LAT_COUNT,) or not np.allclose(
        file_lats, IMERG_LATS, atol=1e-3
    ):
      raise ValueError(
          f"Latitude coordinate in {nc_path} does not match expected IMERG "
          f"grid (shape={file_lats.shape}, expected=({LAT_COUNT},))."
      )
    if file_lons.shape != (LON_COUNT,) or not np.allclose(
        file_lons, IMERG_LONS, atol=1e-3
    ):
      raise ValueError(
          f"Longitude coordinate in {nc_path} does not match expected IMERG "
          f"grid (shape={file_lons.shape}, expected=({LON_COUNT},))."
      )

    da = ds["precipitation"]
    if "time" in da.dims:
      if da.sizes["time"] != 1:
        raise ValueError(
            f"Expected single daily time step in {nc_path}, got "
            f"time={da.sizes['time']}."
        )
      da = da.squeeze("time")
    if da.dims == ("lon", "lat"):
      da = da.transpose("lat", "lon")
    elif da.dims != ("lat", "lon"):
      raise ValueError(
          f"Unexpected spatial dimensions {da.dims} in {nc_path}; expected "
          "('lat', 'lon') or ('lon', 'lat')."
      )
    arr = np.asarray(da.values, dtype=np.float32)

  if arr.shape != (LAT_COUNT, LON_COUNT):
    raise ValueError(
        f"Unexpected IMERG grid shape {arr.shape} in {nc_path}; expected "
        f"({LAT_COUNT}, {LON_COUNT})."
    )
  arr = np.where((arr < 0.0) | ~np.isfinite(arr), np.nan, arr)
  if not np.isfinite(arr).any():
    raise ValueError(
        f"IMERG daily NetCDF {nc_path} contains an all-NaN precipitation grid."
    )
  return arr


__all__ = [
    "DEFAULT_CMR_GRANULES_URL",
    "DEFAULT_GESDISC_URL",
    "EXPECTED_HHR_START_TOKENS",
    "EarthdataSession",
    "IMERG_DAILY_SHORT_NAME",
    "IMERG_HHR_SHORT_NAME",
    "IMERG_LATS",
    "IMERG_LONS",
    "IMERG_VARIABLE",
    "LAT_COUNT",
    "LON_COUNT",
    "build_gesdisc_daily_url",
    "download_daily_imerg",
    "get_earthdata_credentials_from_netrc",
    "parse_imerg_netcdf_to_grid",
    "query_cmr_granules",
]
