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

"""Shared NOAA PSL CPC Global Unified Gauge-Based Precipitation NetCDF download and standardization."""

from __future__ import annotations

import logging
import os

from multimet.utils.http import download_http_file
import numpy as np
import pandas as pd
import xarray as xr

CPC_LATS = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
CPC_LONS = np.linspace(-179.75, 179.75, 720, dtype=np.float32)
EXPECTED_PSL_LATS = np.linspace(89.75, -89.75, 360, dtype=np.float32)
EXPECTED_PSL_LONS = np.linspace(0.25, 359.75, 720, dtype=np.float32)
CPC_VARIABLE = "cpc_precipitation"
NOAA_PSL_URL_TEMPLATE = (
    "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc"
)


def _is_cached_cpc_netcdf_usable(
    nc_path: str,
    *,
    required_end_date: pd.Timestamp | str | None = None,
) -> bool:
  """Returns True if ``nc_path`` exists and covers ``required_end_date``."""
  if not os.path.exists(nc_path) or os.path.getsize(nc_path) <= 1024:
    return False
  if required_end_date is None:
    return True
  with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
    if "precip" not in ds or "time" not in ds:
      return False
    time_raw = pd.to_datetime(ds["time"].values)
    dates = pd.DatetimeIndex(time_raw.strftime("%Y-%m-%d"))
    if len(dates) == 0:
      return False
    precip_raw = np.asarray(ds["precip"].values, dtype=np.float32)
    if precip_raw.ndim != 3:
      return False
    finite_per_day = np.isfinite(
        np.where(precip_raw < 0, np.nan, precip_raw)
    ).any(axis=(1, 2))
    if not finite_per_day.any():
      return False
    last_valid_idx = int(np.where(finite_per_day)[0][-1])
    last_valid_date = pd.Timestamp(dates[last_valid_idx]).normalize()
    return bool(last_valid_date >= pd.Timestamp(required_end_date).normalize())


def ensure_psl_cpc_netcdf(
    year: int,
    cache_dir: str,
    url_template: str = NOAA_PSL_URL_TEMPLATE,
    *,
    force_download: bool = False,
    required_end_date: pd.Timestamp | str | None = None,
    required_date: pd.Timestamp | str | None = None,
) -> str:
  """Downloads and caches a yearly NOAA PSL CPC NetCDF file if needed."""
  os.makedirs(cache_dir, exist_ok=True)
  local_path = os.path.join(cache_dir, f"precip.{year}.nc")
  req_dt = required_end_date if required_end_date is not None else required_date
  if not force_download and _is_cached_cpc_netcdf_usable(
      local_path, required_end_date=req_dt
  ):
    return local_path

  url = url_template.format(year=year)
  logging.info("Downloading NOAA PSL CPC NetCDF for %d from %s...", year, url)
  downloaded = download_http_file(
      url,
      local_path,
      headers={"User-Agent": "OpenMultiMet/1.1 (Google Research)"},
      timeout=180,
      min_bytes=1024,
      resource_label=f"NOAA PSL CPC NetCDF for year {year}",
  )
  logging.info(
      "Cached %s (%.1f MB)", downloaded, os.path.getsize(downloaded) / 1e6
  )
  return downloaded


def process_cpc_netcdf_to_dataset(
    nc_path: str,
    target_start_date: pd.Timestamp | None = None,
    target_end_date: pd.Timestamp | None = None,
    *,
    trim_trailing_unpublished: bool = False,
) -> xr.Dataset | None:
  """Reads a yearly NOAA PSL NetCDF file and standardizes it to MultiMet schema."""
  with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
    for required_key in ("precip", "lat", "lon", "time"):
      if required_key not in ds:
        raise KeyError(
            f"Required key {required_key!r} not found in CPC NetCDF {nc_path}."
        )
    da = ds["precip"]
    if da.dims != ("time", "lat", "lon"):
      raise ValueError(
          f"Expected 'precip' dimensions ('time', 'lat', 'lon') in {nc_path}, "
          f"got {da.dims}."
      )
    raw_lats = np.asarray(ds["lat"].values, dtype=np.float32)
    raw_lons = np.asarray(ds["lon"].values, dtype=np.float32)
    if raw_lats.shape != EXPECTED_PSL_LATS.shape or not np.allclose(
        raw_lats, EXPECTED_PSL_LATS, atol=1e-3
    ):
      raise ValueError(
          f"Unexpected latitude coordinates in {nc_path}: expected 360 points "
          "descending from 89.75 to -89.75."
      )
    if raw_lons.shape != EXPECTED_PSL_LONS.shape or not np.allclose(
        raw_lons, EXPECTED_PSL_LONS, atol=1e-3
    ):
      raise ValueError(
          f"Unexpected longitude coordinates in {nc_path}: expected 720 points "
          "ascending from 0.25 to 359.75."
      )
    precip_raw = da.values
    time_raw = pd.to_datetime(ds["time"].values)

  if precip_raw.ndim != 3 or precip_raw.shape[1:] != (
      len(CPC_LATS),
      len(CPC_LONS),
  ):
    raise ValueError(
        f"Unexpected 'precip' array shape {precip_raw.shape} in {nc_path}."
    )

  dates = pd.DatetimeIndex(time_raw.strftime("%Y-%m-%d"))
  if len(dates) == 0:
    raise ValueError(f"CPC NetCDF {nc_path} has an empty time coordinate.")
  expected_full_dates = pd.date_range(dates[0], dates[-1], freq="1D")
  if len(dates) != len(expected_full_dates) or not (
      dates == expected_full_dates
  ).all():
    raise ValueError(
        f"Time coordinate in {nc_path} is not strictly contiguous daily."
    )

  precip_lat_inv = precip_raw[:, ::-1, :]
  precip_shifted = np.concatenate(
      [precip_lat_inv[:, :, 360:], precip_lat_inv[:, :, :360]], axis=2
  )
  precip_clean = np.where(precip_shifted < 0, np.nan, precip_shifted).astype(
      np.float32
  )

  finite_per_day = np.isfinite(precip_clean).any(axis=(1, 2))
  if not finite_per_day.any():
    raise ValueError(
        f"CPC NetCDF {nc_path} contains no finite precipitation values on any "
        "day."
    )

  if trim_trailing_unpublished:
    last_valid_idx = int(np.where(finite_per_day)[0][-1])
    dates = dates[: last_valid_idx + 1]
    precip_clean = precip_clean[: last_valid_idx + 1]
    finite_per_day = finite_per_day[: last_valid_idx + 1]

  if not finite_per_day.all():
    bad_dates = [
        dates[i].strftime("%Y-%m-%d")
        for i in np.where(~finite_per_day)[0]
    ]
    raise ValueError(
        f"CPC NetCDF {nc_path} contains all-NaN daily slices on dates: "
        f"{bad_dates}"
    )

  if target_start_date is not None or target_end_date is not None:
    mask = np.ones(len(dates), dtype=bool)
    if target_start_date is not None:
      mask &= dates >= target_start_date
    if target_end_date is not None:
      mask &= dates <= target_end_date

    dates = dates[mask]
    precip_clean = precip_clean[mask]

  if len(dates) == 0:
    return None

  return xr.Dataset(
      data_vars={
          CPC_VARIABLE: (
              ["time", "latitude", "longitude"],
              precip_clean,
              {
                  "units": "mm/day",
                  "long_name": (
                      "CPC Global Unified Gauge-Based Daily Precipitation"
                  ),
                  "standard_name": "precipitation_amount",
              },
          ),
      },
      coords={
          "time": dates.values,
          "latitude": CPC_LATS,
          "longitude": CPC_LONS,
      },
      attrs={
          "title": (
              "Open-MultiMet NOAA CPC Global Unified Daily Precipitation"
              " Archive"
          ),
          "spatial_resolution": "0.50 degree",
          "description": (
              "Daily gauge-based global precipitation analysis from NOAA PSL"
              " standardized to Caravan MultiMet v1.1"
          ),
          "license": (
              "Usage Restrictions: None (Public Domain / NOAA PSL: "
              "https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html)"
          ),
          "institution": "NOAA PSL / CPC / Open-MultiMet",
          "citation": (
              "Chen et al. (2008) J. Geophys. Res. 113, D04110; Xie et al."
              " (2007) J. Hydrometeorol. 8, 607-626."
          ),
          "product": "CPC",
          "version": "1.1",
      },
  )
