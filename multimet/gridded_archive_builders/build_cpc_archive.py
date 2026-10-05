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

"""ETL pipeline to build the unified Open-MultiMet daily CPC surface archive.

Ingests NOAA CPC Global Unified Daily Precipitation from the NOAA Physical
Sciences Laboratory (PSL) yearly NetCDF archives published at
``https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc``.

Standardizes spatial dimensions to the Caravan MultiMet specification:

- Dimensions: ``(time, latitude, longitude)``
- Latitude: 360 points from -89.75 to 89.75 (0.5 deg resolution, ascending)
- Longitude: 720 points from -179.75 to 179.75 (0.5 deg, shifted from
  ``[0, 360)``)
- Variable: ``cpc_precipitation`` (float32, mm/day, NaN over missing values)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import datetime
import functools
import logging
import multiprocessing as mp
import os
import time

from multimet.utils import storage
from multimet.utils.http import check_http_url_exists, download_http_file
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

# NOAA PSL publishes CPC Global Unified Precipitation from 1979 onwards.
DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = datetime.date.today().year

# Maximum allowed upstream publication lag (in days) when extending to present
# without an explicit --end_date.
MAX_PUBLICATION_LAG_DAYS = 7

# Standard CPC 0.5 deg coordinates (target Caravan MultiMet layout).
CPC_LATS = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
CPC_LONS = np.linspace(-179.75, 179.75, 720, dtype=np.float32)

# Expected raw NOAA PSL NetCDF coordinates prior to standardization.
EXPECTED_PSL_LATS = np.linspace(89.75, -89.75, 360, dtype=np.float32)
EXPECTED_PSL_LONS = np.linspace(0.25, 359.75, 720, dtype=np.float32)

# Name of the single data variable written to the archive.
CPC_VARIABLE = "cpc_precipitation"

NOAA_PSL_URL_TEMPLATE = (
    "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc"
)


def _is_cached_cpc_netcdf_usable(
    nc_path: str,
    *,
    required_end_date: pd.Timestamp | None = None,
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
    required_end_date: pd.Timestamp | None = None,
) -> str:
  """Downloads and caches a yearly NOAA PSL CPC NetCDF file if needed.

  Args:
    year: Four-digit calendar year to fetch.
    cache_dir: Local directory where downloaded NetCDF files are cached.
    url_template: URL template accepting ``{year}``.
    force_download: If ``True``, always downloads a fresh copy from ``url_template``
      even if a cached file exists in ``cache_dir``.
    required_end_date: Optional date that a cached file must contain finite
      precipitation data through in order to be reused without re-downloading.

  Returns:
    Path to the local ``precip.{year}.nc`` file.

  Raises:
    FileNotFoundError: If the upstream server returns HTTP 404.
    requests.HTTPError: If the upstream server returns another HTTP error.
    ValueError: If the downloaded file is unexpectedly small.
  """
  os.makedirs(cache_dir, exist_ok=True)
  local_path = os.path.join(cache_dir, f"precip.{year}.nc")
  if not force_download and _is_cached_cpc_netcdf_usable(
      local_path, required_end_date=required_end_date
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
    expected_year: int | None = None,
    trim_trailing_unpublished: bool = False,
    reference_date: pd.Timestamp | None = None,
    max_lag_days: int = MAX_PUBLICATION_LAG_DAYS,
) -> xr.Dataset | None:
  """Reads a yearly NOAA PSL NetCDF file and standardizes it to MultiMet schema.

  Validates input coordinates, dimensions, calendar year bounds, and daily data
  completeness before performing spatial transformation:
  1. Verifies ``lat`` is descending ``[89.75 .. -89.75]`` and ``lon`` is
     ascending ``[0.25 .. 359.75]``.
  2. Inverts latitude axis (PSL: ``[89.75 .. -89.75]`` -> ``[-89.75 .. 89.75]``).
  3. Shifts longitude axis (PSL: ``[0.25 .. 359.75]`` -> ``[-179.75 .. 179.75]``).
  4. Masks missing values (``< 0`` or fill values) to ``np.nan``.
  5. Verifies that every retained day contains finite precipitation values and
     that the file covers the required calendar range for ``expected_year``.

  Args:
    nc_path: Path to local ``precip.{year}.nc`` file.
    target_start_date: Optional inclusive lower bound filter on dates.
    target_end_date: Optional inclusive upper bound filter on dates.
    expected_year: Optional calendar year that all timestamps in ``nc_path``
      must belong to and cover.
    trim_trailing_unpublished: If ``True`` (used only within the active
      publication window) and the dataset contains valid dates followed by
      trailing all-NaN slices (future pre-allocated dates in NOAA PSL's active
      yearly file), strips the trailing all-NaN dates prior to checking for
      interior all-NaN days.
    reference_date: Optional reference "today" date used to enforce
      ``max_lag_days`` when ``trim_trailing_unpublished`` is ``True`` and
      ``target_end_date`` is ``None``.
    max_lag_days: Maximum allowed lag (in days) between ``reference_date`` and
      the last finite date in an auto-trimmed active-year file.

  Returns:
    Standardized ``xarray.Dataset`` with dimensions
    ``(time, latitude, longitude)``, or ``None`` if no dates fall within the
    ``target_start_date`` / ``target_end_date`` filter.

  Raises:
    KeyError: If required variables or coordinates are missing.
    ValueError: If coordinate values, dimensions, date monotonicity, calendar
      year coverage, publication lag, or daily finite data checks fail.
  """
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

  file_years = set(int(y) for y in dates.year)
  if len(file_years) != 1:
    raise ValueError(
        f"CPC NetCDF {nc_path} spans multiple calendar years: "
        f"{sorted(file_years)}."
    )
  actual_year = int(dates[0].year)
  if expected_year is not None and actual_year != expected_year:
    raise ValueError(
        f"CPC NetCDF {nc_path} contains timestamps for year {actual_year}, "
        f"expected {expected_year}."
    )

  # Invert latitude: north->south [89.75 .. -89.75] to [-89.75 .. 89.75].
  precip_lat_inv = precip_raw[:, ::-1, :]

  # Shift longitude: split at 180 deg (index 360) and concatenate [360:] + [:360].
  precip_shifted = np.concatenate(
      [precip_lat_inv[:, :, 360:], precip_lat_inv[:, :, :360]], axis=2
  )

  # Mask missing values (< 0) with NaN.
  precip_clean = np.where(precip_shifted < 0, np.nan, precip_shifted).astype(
      np.float32
  )

  finite_per_day = np.isfinite(precip_clean).any(axis=(1, 2))
  if not finite_per_day.any():
    if (
        trim_trailing_unpublished
        and target_end_date is None
        and reference_date is not None
        and expected_year is not None
    ):
      ref_ts = pd.Timestamp(reference_date).normalize()
      prev_year_end = pd.Timestamp(f"{expected_year - 1}-12-31")
      if 0 <= int((ref_ts - prev_year_end).days) <= max_lag_days:
        return None
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

  if expected_year is not None:
    req_start = pd.Timestamp(f"{expected_year}-01-01")
    if target_start_date is not None:
      req_start = max(req_start, pd.Timestamp(target_start_date).normalize())
    if dates[0] > req_start:
      raise ValueError(
          f"CPC NetCDF {nc_path} starts at {dates[0].strftime('%Y-%m-%d')}, "
          f"after required start date {req_start.strftime('%Y-%m-%d')}."
      )
    if (
        target_end_date is not None
        and pd.Timestamp(target_end_date).year == expected_year
    ):
      req_end = pd.Timestamp(target_end_date).normalize()
      if dates[-1] < req_end:
        raise ValueError(
            f"CPC NetCDF {nc_path} ends at {dates[-1].strftime('%Y-%m-%d')}, "
            f"before required end date {req_end.strftime('%Y-%m-%d')}."
        )
    elif not trim_trailing_unpublished:
      req_end = pd.Timestamp(f"{expected_year}-12-31")
      if dates[-1] < req_end:
        raise ValueError(
            f"CPC NetCDF {nc_path} ends at {dates[-1].strftime('%Y-%m-%d')}, "
            f"before end of year {req_end.strftime('%Y-%m-%d')}."
        )

  if (
      trim_trailing_unpublished
      and target_end_date is None
      and reference_date is not None
  ):
    ref_ts = pd.Timestamp(reference_date).normalize()
    inferred_year = expected_year if expected_year is not None else int(dates[-1].year)
    year_end = pd.Timestamp(f"{inferred_year}-12-31")
    if inferred_year == ref_ts.year or dates[-1] < year_end:
      lag_days = int((ref_ts - dates[-1]).days)
      if lag_days > max_lag_days:
        raise ValueError(
            f"CPC NetCDF {nc_path} latest valid date is "
            f"{dates[-1].strftime('%Y-%m-%d')}, which lags reference date "
            f"{ref_ts.strftime('%Y-%m-%d')} by {lag_days} days "
            f"(max allowed lag: {max_lag_days} days)."
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


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str | None = None,
    is_initial_write: bool = False,
    consolidated: bool = False,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store."""
  chunk_days = min(30, len(ds_batch["time"]))
  storage.write_dataset_batch_to_zarr(
      ds_batch,
      target_zarr_url,
      project=project,
      is_initial_write=is_initial_write,
      consolidated=consolidated,
      time_chunk_size=chunk_days,
  )


def _extract_single_year_task(
    year: int,
    *,
    cache_dir: str,
    url_template: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    cleanup_cache: bool,
    year_starts: dict[int, pd.Timestamp],
    reference_date: pd.Timestamp,
    max_lag_days: int,
    force_download: bool,
    store_exists: bool,
) -> tuple[int, xr.Dataset | None]:
  """Worker function to download and transform a single year of CPC data."""
  t0 = time.time()
  logging.info(
      "Worker [%d] downloading CPC PSL NetCDF for year %d...",
      os.getpid(),
      year,
  )
  y_start = year_starts.get(year, start_date)
  required_cache_end: pd.Timestamp | None = None
  if end_date is not None and end_date.year == year:
    required_cache_end = end_date
  elif year >= reference_date.year and end_date is None:
    required_cache_end = reference_date
  elif store_exists and y_start is not None:
    required_cache_end = y_start

  nc_path = ensure_psl_cpc_netcdf(
      year,
      cache_dir=cache_dir,
      url_template=url_template,
      force_download=force_download,
      required_end_date=required_cache_end,
  )

  year_end = pd.Timestamp(f"{year}-12-31")
  can_trim_tail = year == reference_date.year or (
      end_date is None
      and year == reference_date.year - 1
      and 0 <= int((reference_date - year_end).days) <= max_lag_days
  )
  ds_year = process_cpc_netcdf_to_dataset(
      nc_path,
      target_start_date=y_start,
      target_end_date=end_date,
      expected_year=year,
      trim_trailing_unpublished=can_trim_tail,
      reference_date=reference_date,
      max_lag_days=max_lag_days,
  )

  if cleanup_cache and os.path.exists(nc_path):
    os.remove(nc_path)
    logging.info("Cleaned up cached file: %s", nc_path)

  logging.info(
      "Worker [%d] finished year %d in %.2fs",
      os.getpid(),
      year,
      time.time() - t0,
  )
  return year, ds_year


def build_cpc_archive(
    target_zarr: str,
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    project: str | None = None,
    cache_dir: str | None = None,
    source_url_template: str = NOAA_PSL_URL_TEMPLATE,
    cleanup_cache: bool = False,
    overwrite: bool = False,
    extend_archive: bool = False,
    num_workers: int | None = None,
    reference_date: str | pd.Timestamp | None = None,
    max_lag_days: int = MAX_PUBLICATION_LAG_DAYS,
) -> None:
  """Executes the NOAA CPC daily gridded archive build.

  Args:
    target_zarr: Destination Zarr store URI or local directory path.
    start_year: First NOAA PSL yearly file to ingest.
    end_year: Last NOAA PSL yearly file to ingest (inclusive). Defaults to the
      year of ``reference_date`` (current calendar year).
    start_date: Optional inclusive lower bound date filter (YYYY-MM-DD).
    end_date: Optional inclusive upper bound date filter (YYYY-MM-DD).
    project: Optional GCP project for GCS billing/authentication.
    cache_dir: Local directory for staging downloaded NetCDF files. If ``None``,
      an isolated temporary directory is created and cleaned up on exit.
    source_url_template: Upstream URL template accepting ``{year}``.
    cleanup_cache: Whether to delete cached NetCDF files after ingestion.
    overwrite: Whether to delete and rebuild an existing target store.
    extend_archive: If ``True``, requires the target archive to exist and forces
      fresh downloads of all required yearly files without using cached files.
    num_workers: Number of parallel worker processes.
    reference_date: Optional reference "today" date (defaults to current local
      date) used for active-year and early-January rollover lag bounds.
    max_lag_days: Maximum allowed publication lag (in days) when ``end_date`` is
      omitted.
  """
  if extend_archive and overwrite:
    raise ValueError("--extend_archive and --overwrite cannot be used together.")

  ref_ts = (
      pd.Timestamp(reference_date).normalize()
      if reference_date is not None
      else pd.Timestamp(datetime.date.today())
  )
  if end_year is None:
    end_year = int(ref_ts.year)

  if end_year < start_year:
    raise ValueError(
        f"end_year ({end_year}) must be >= start_year ({start_year})."
    )

  full_target_url, _, store_exists, has_consolidated, mapper = (
      storage.inspect_zarr_store(
          target_zarr, project=project, overwrite=overwrite
      )
  )
  if extend_archive and not store_exists:
    raise FileNotFoundError(
        f"Cannot run --extend_archive: target store {full_target_url} does not "
        "exist."
    )

  if num_workers is None or num_workers <= 0:
    num_workers = min(32, os.cpu_count() or 4)

  t_start_filter = pd.Timestamp(start_date).normalize() if start_date else None
  t_end_filter = pd.Timestamp(end_date).normalize() if end_date else None
  if (
      t_start_filter is not None
      and t_end_filter is not None
      and t_end_filter < t_start_filter
  ):
    raise ValueError(
        f"end_date ({end_date}) must be >= start_date ({start_date})."
    )

  if t_start_filter:
    start_year = max(start_year, t_start_filter.year)
  if t_end_filter:
    end_year = min(end_year, t_end_filter.year)

  years = list(range(start_year, end_year + 1))
  logging.info(
      "Building CPC archive for %d years: %d to %d (Workers: %d)",
      len(years),
      start_year,
      end_year,
      num_workers,
  )
  logging.info("Target: %s", full_target_url)

  is_first_write = not store_exists
  year_starts: dict[int, pd.Timestamp] = {}
  last_written_date: pd.Timestamp | None = None

  if store_exists:
    effective_start = t_start_filter or pd.Timestamp(f"{start_year}-01-01")
    effective_end = t_end_filter or pd.Timestamp(f"{end_year}-12-31")
    requested_dates = pd.date_range(effective_start, effective_end, freq="1D")
    append_dates, date_to_idx = storage.plan_archive_resume(
        mapper, requested_dates, has_consolidated=has_consolidated
    )
    if len(append_dates) == 0:
      logging.info(
          "Store already contains all requested dates up to %s. Done!",
          effective_end.strftime("%Y-%m-%d"),
      )
      return

    last_written_date = pd.Timestamp(max(date_to_idx.keys()))
    resume_start = pd.Timestamp(append_dates[0])
    years = [y for y in years if y >= resume_start.year]
    if not years:
      logging.info("Store already contains all requested years. Done!")
      return
    year_starts[resume_start.year] = resume_start
    is_first_write = False

  total_days_processed = 0
  t0_total = time.time()

  def _is_in_rollover_window(y: int) -> bool:
    prev_dec31 = pd.Timestamp(f"{y - 1}-12-31")
    return (
        t_end_filter is None
        and y == ref_ts.year
        and 0 <= int((ref_ts - prev_dec31).days) <= max_lag_days
    )

  def _handle_year_result(y: int, ds_year: xr.Dataset | None) -> int:
    nonlocal is_first_write, last_written_date
    if ds_year is None:
      if (
          y == years[-1]
          and last_written_date is not None
          and t_end_filter is None
          and int((ref_ts - last_written_date).days) <= max_lag_days
          and (y == last_written_date.year or _is_in_rollover_window(y))
      ):
        logging.info(
            "Year %d already up to date through %s.",
            y,
            last_written_date.strftime("%Y-%m-%d"),
        )
        return 0
      raise ValueError(
          f"No valid CPC data found for year {y} within requested date bounds."
      )

    year_times = pd.DatetimeIndex(pd.to_datetime(ds_year["time"].values))
    if last_written_date is not None:
      expected_next = last_written_date + pd.Timedelta(days=1)
      if pd.Timestamp(year_times[0]) != expected_next:
        raise ValueError(
            "Non-contiguous CPC archive dates between "
            f"{last_written_date.strftime('%Y-%m-%d')} and "
            f"{pd.Timestamp(year_times[0]).strftime('%Y-%m-%d')}."
        )

    write_batch_to_zarr(
        ds_year,
        full_target_url,
        project=project,
        is_initial_write=is_first_write,
        consolidated=has_consolidated,
    )
    is_first_write = False
    last_written_date = pd.Timestamp(year_times[-1])
    return len(year_times)

  with storage.managed_cache_dir(
      cache_dir, cleanup_cache, prefix="cpc_cache_"
  ) as effective_cache_dir:
    if (
        years
        and _is_in_rollover_window(years[-1])
        and (len(years) > 1 or last_written_date is not None)
    ):
      rollover_year = years[-1]
      local_rollover = os.path.join(
          effective_cache_dir, f"precip.{rollover_year}.nc"
      )
      has_local_rollover = (
          not extend_archive
          and _is_cached_cpc_netcdf_usable(local_rollover)
      )
      if not has_local_rollover:
        rollover_url = source_url_template.format(year=rollover_year)
        if not check_http_url_exists(
            rollover_url,
            headers={"User-Agent": "OpenMultiMet/1.1 (Google Research)"},
            resource_label=f"NOAA PSL CPC NetCDF for year {rollover_year}",
        ):
          logging.info(
              "NOAA PSL file for new year %d is not published yet (within "
              "%d-day rollover window of %s).",
              rollover_year,
              max_lag_days,
              ref_ts.strftime("%Y-%m-%d"),
          )
          years = years[:-1]

    if not years:
      if (
          last_written_date is not None
          and int((ref_ts - last_written_date).days) <= max_lag_days
      ):
        logging.info(
            "Store already up to date through %s within %d-day rollover "
            "window.",
            last_written_date.strftime("%Y-%m-%d"),
            max_lag_days,
        )
        return
      raise FileNotFoundError(
          f"NOAA PSL CPC NetCDF for year {ref_ts.year} is not published and "
          f"existing archive date {last_written_date} lags "
          f"{ref_ts.strftime('%Y-%m-%d')} by more than {max_lag_days} days."
      )

    worker_fn = functools.partial(
        _extract_single_year_task,
        cache_dir=effective_cache_dir,
        url_template=source_url_template,
        start_date=t_start_filter,
        end_date=t_end_filter,
        cleanup_cache=cleanup_cache,
        year_starts=year_starts,
        reference_date=ref_ts,
        max_lag_days=max_lag_days,
        force_download=extend_archive,
        store_exists=store_exists,
    )
    if num_workers > 1 and len(years) > 1:
      mp_ctx = mp.get_context("spawn")
      with mp_ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap(worker_fn, years, chunksize=1)
        for y, ds_year in tqdm.tqdm(
            iterator,
            total=len(years),
            desc=f"Processing CPC ({num_workers} workers)",
        ):
          total_days_processed += _handle_year_result(y, ds_year)
    else:
      for y in tqdm.tqdm(years, desc="Processing CPC (sequential)"):
        _, ds_year = worker_fn(y)
        total_days_processed += _handle_year_result(y, ds_year)

    logging.info(
        "CPC Archive build complete! Processed %d total days across %d years "
        "in %.1fs.",
        total_days_processed,
        len(years),
        time.time() - t0_total,
    )


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the CPC archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-cpc-archive",
      description="Build the unified CPC daily precipitation archive.",
  )
  parser.add_argument(
      "--target_zarr",
      type=str,
      required=True,
      help="Destination Zarr store (local path or explicit gs:// URI).",
  )
  parser.add_argument(
      "--start_year",
      type=int,
      default=DEFAULT_START_YEAR,
      help="First NOAA PSL yearly file to ingest (e.g. 1979).",
  )
  parser.add_argument(
      "--end_year",
      type=int,
      default=DEFAULT_END_YEAR,
      help="Last NOAA PSL yearly file to ingest (inclusive).",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=None,
      help="Optional lower bound date filter within the year range.",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=None,
      help="Optional upper bound date filter within the year range.",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="Optional GCP project used for billing/auth of gs:// requests.",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      default=None,
      help="Local directory used to cache downloaded NOAA PSL NetCDF files.",
  )
  parser.add_argument(
      "--source_url_template",
      type=str,
      default=NOAA_PSL_URL_TEMPLATE,
      help="Upstream NOAA PSL URL template containing {year}.",
  )
  parser.add_argument(
      "--cleanup_cache",
      action="store_true",
      help="Delete downloaded NetCDF files once they have been written.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Delete and rebuild the target store instead of resuming it.",
  )
  parser.add_argument(
      "--extend_archive",
      "--extend-archive",
      dest="extend_archive",
      action="store_true",
      help=(
          "Extend an existing archive from its last date to the latest "
          "published data, ignoring any pre-cached files."
      ),
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=min(32, os.cpu_count() or 4),
      help="Number of parallel extraction worker processes.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-cpc-archive``."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  build_cpc_archive(
      target_zarr=args.target_zarr,
      start_year=args.start_year,
      end_year=args.end_year,
      start_date=args.start_date,
      end_date=args.end_date,
      project=args.project,
      cache_dir=args.cache_dir,
      source_url_template=args.source_url_template,
      cleanup_cache=args.cleanup_cache,
      overwrite=args.overwrite,
      extend_archive=args.extend_archive,
      num_workers=args.num_workers,
  )


if __name__ == "__main__":
  main()
