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

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime
import http.client
import io
import logging
import os
import random
import re
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import urllib.parse

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr

from pathlib import Path

from multimet.timeseries_extractors.base import BaseExtractor
from multimet.timeseries_extractors.config import DEFAULT_STORAGE_PATHS, Product
from multimet.utils.zonal import ZonalWeightCalculator, ZonalWeightMatrix
from multimet.utils.earthdata import (
    EarthdataSession,
    download_daily_imerg,
    get_earthdata_credentials_from_netrc,
)
from multimet.utils.zonal import (
    weighted_mean_valid as _weighted_mean_valid,
    weighted_mean_valid_with_coverage as _weighted_mean_valid_with_coverage,
)

logger = logging.getLogger(__name__)


def resolve_date_to_imerg_files(
    storage_dir: str, dt: pd.Timestamp
) -> List[str]:
  """Resolves a date to its corresponding IMERG HDF5 file(s)."""
  dt = pd.to_datetime(dt)
  month_str = dt.strftime("%Y%m")
  day_str = dt.strftime("%Y%m%d")
  m_dir = os.path.join(storage_dir, month_str)

  files = []
  if os.path.exists(m_dir):
    for fname in os.listdir(m_dir):
      if day_str in fname and fname.endswith((".RT-H5", ".HDF5")):
        files.append(os.path.join(m_dir, fname))

  return sorted(files)


class IMERGExtractor(BaseExtractor):
  """Extractor for NASA GPM IMERG Early Precipitation (0.1 deg, [-60, 60] latitude)."""

  def __init__(
      self,
      data_dir: Optional[str] = None,
      source: str = "auto",
      username: Optional[str] = None,
      password: Optional[str] = None,
      token: Optional[str] = None,
      netrc_path: Optional[str] = None,
  ):
    super().__init__(Product.IMERG, data_dir)
    default_url = DEFAULT_STORAGE_PATHS[Product.IMERG].get(
        "gesdisc_url",
        "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07/",
    )

    source_lower = source.lower().strip()
    if source_lower in ("archive", "gridded_archive", "zarr", "zarr_archive"):
      self.source = "archive"
      if data_dir is None or not str(data_dir).strip():
        raise ValueError(
            "IMERGExtractor with source='archive' requires an explicit Zarr "
            "store URI or path via data_dir."
        )
      self.data_dir = str(data_dir)
    elif source_lower in ("auto", "default"):
      if data_dir is not None and (
          str(data_dir).startswith(("gs://", "gcs://"))
          or str(data_dir).rstrip("/").endswith(".zarr")
      ):
        self.source = "archive"
        self.data_dir = str(data_dir)
      else:
        self.source = "gesdisc"
        self.data_dir = str(data_dir) if data_dir is not None else default_url
    elif source_lower in ("gesdisc", "public", "nasa", "upstream"):
      self.source = "gesdisc"
      self.data_dir = str(data_dir) if data_dir is not None else default_url
    elif source_lower == "dynamical":
      self.source = "dynamical"
      self.data_dir = str(data_dir) if data_dir is not None else ""
    elif source_lower in ("h5", "local"):
      self.source = "h5"
      if data_dir is None or not str(data_dir).strip():
        raise ValueError(
            "IMERGExtractor with source='h5' requires an explicit local "
            "directory via data_dir."
        )
      self.data_dir = str(data_dir)
    else:
      self.source = source_lower
      self.data_dir = str(data_dir) if data_dir is not None else default_url

    self.session = EarthdataSession(
        username=username, password=password, token=token, netrc_path=netrc_path
    )

    # Standard IMERG grid: 1800 lats x 3600 lons (0.1 deg)
    # Latitudes: -89.95 to 89.95 (south to north)
    self.lats = np.linspace(-89.95, 89.95, 1800, dtype=np.float64)
    # Longitudes: -179.95 to 179.95
    self.lons = np.linspace(-179.95, 179.95, 3600, dtype=np.float64)
    self.zonal_calc = ZonalWeightCalculator(
        self.lats, self.lons, cell_res_lat=0.1, cell_res_lon=0.1
    )

  def get_daily_file(self, dt: pd.Timestamp) -> str:
    """Finds or downloads the daily IMERG NetCDF4 file for a given date."""
    dt = pd.to_datetime(dt)
    date_str = dt.strftime("%Y%m%d")
    year = dt.year
    month = dt.month

    # 1. Check if data_dir is a local directory containing matching daily files
    if os.path.isdir(self.data_dir):
      for fname in os.listdir(self.data_dir):
        if date_str in fname and fname.endswith((".nc4", ".nc", ".HDF5", ".h5")):
          return os.path.join(self.data_dir, fname)
      ym_dir = os.path.join(self.data_dir, str(year), f"{month:02d}")
      if os.path.isdir(ym_dir):
        for fname in os.listdir(ym_dir):
          if date_str in fname and fname.endswith((".nc4", ".nc", ".HDF5", ".h5")):
            return os.path.join(ym_dir, fname)

    # 2. Check local cache directory
    cache_dir = os.environ.get("MULTIMET_IMERG_CACHE", "/tmp/multimet_imerg_cache")
    filename = f"3B-DAY-E.MS.MRG.3IMERG.{date_str}-S000000-E235959.V07B.nc4"
    cached_path = os.path.join(cache_dir, filename)
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1000:
      return cached_path

    if os.path.isdir(cache_dir):
      for fname in os.listdir(cache_dir):
        if date_str in fname and fname.endswith((".nc4", ".nc")):
          f_path = os.path.join(cache_dir, fname)
          if os.path.getsize(f_path) > 1000:
            return f_path

    # 3. If not cached locally, download from NASA GES DISC
    base_url = (
        self.data_dir
        if (
            self.data_dir.startswith("http://")
            or self.data_dir.startswith("https://")
        )
        else "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07"
    )
    url = f"{base_url.rstrip('/')}/{year}/{month:02d}/{filename}"

    return download_daily_imerg(url, cached_path, session=self.session)

  def extract_day_from_nc4(
      self,
      nc_path: str,
      basins_gdf: gpd.GeoDataFrame,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of IMERG precipitation from a daily NetCDF4 file."""
    with xr.open_dataset(nc_path) as ds:
      if "precipitation" not in ds:
        raise KeyError(
            f"IMERG V07 variable 'precipitation' not found in {nc_path}. "
            f"Variables: {list(ds.data_vars.keys())}"
        )
      da = ds["precipitation"]

      if "time" in da.dims:
        da = da.squeeze("time")
      if da.dims == ("lon", "lat"):
        da = da.transpose("lat", "lon")

      if use_bounding_box:
        min_lon, min_lat, max_lon, max_lat = basins_gdf.total_bounds
        pad = 0.2
        lat_vals = da.lat.values
        lon_vals = da.lon.values

        if lat_vals[0] > lat_vals[-1]:
          lat_slice = slice(float(max_lat + pad), float(min_lat - pad))
        else:
          lat_slice = slice(float(min_lat - pad), float(max_lat + pad))

        if lon_vals[0] > lon_vals[-1]:
          lon_slice = slice(float(max_lon + pad), float(min_lon - pad))
        else:
          lon_slice = slice(float(min_lon - pad), float(max_lon + pad))

        sub_da = da.sel(lat=lat_slice, lon=lon_slice).load()
      else:
        sub_da = da.load()
      sub_lats = sub_da.lat.values
      sub_lons = sub_da.lon.values

      raw_vals = sub_da.values.astype(np.float32)
      raw_vals = np.where(raw_vals < 0, np.nan, raw_vals)

      if weights_matrix is not None and (
          weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
          and np.allclose(weights_matrix.lats, sub_lats)
          and np.allclose(weights_matrix.lons, sub_lons)
      ):
        matrix = weights_matrix
      else:
        matrix = ZonalWeightMatrix.from_geodataframe(
            basins_gdf, sub_lats, sub_lons, cell_res_lat=0.1, cell_res_lon=0.1
        )

      res, missing = matrix.reduce_2d_with_coverage(raw_vals)
      return {
          "imerg_precipitation": res,
          "imerg_missing_fraction": missing,
      }

  @staticmethod
  def parse_imerg_h5_bytes(content: bytes) -> np.ndarray:
    """Parses an IMERG HDF5 byte stream and returns a 2D (1800, 3600) array in mm/hr."""
    if h5py is None:
      raise ImportError("h5py is required to parse raw HDF5 files.")
    with h5py.File(io.BytesIO(content), "r") as h5:
      if "Grid/precipitation" not in h5:
        raise KeyError("Could not find 'Grid/precipitation' band in IMERG HDF5 file.")
      ds = h5["Grid/precipitation"]

      raw = np.squeeze(ds[()])
      transposed = np.transpose(raw).astype(np.float32)
      transposed[transposed < 0] = np.nan
      return transposed

  def extract_day_from_imerg_files(
      self,
      imerg_files: List[str],
      basin_ids: List[str],
      weights_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of IMERG precipitation across half-hourly HDF5 files."""
    num_basins = len(basin_ids)
    res = np.full(num_basins, np.nan, dtype=np.float32)
    missing_out = np.ones(num_basins, dtype=np.float32)

    if len(imerg_files) != 48:
      return {
          "imerg_precipitation": res,
          "imerg_missing_fraction": missing_out,
      }

    import h5py

    daily_accum = np.zeros((1800, 3600), dtype=np.float32)
    valid_count = np.zeros((1800, 3600), dtype=np.int32)

    for fpath in imerg_files:
      if not os.path.exists(fpath):
        return {
            "imerg_precipitation": res,
            "imerg_missing_fraction": missing_out,
        }
      with h5py.File(fpath, "r") as hf:
        if "Grid/precipitation" not in hf:
          return {
              "imerg_precipitation": res,
              "imerg_missing_fraction": missing_out,
          }
        arr = hf["Grid/precipitation"][:]

        if arr.ndim == 3:
          arr = arr[0]
        if arr.shape == (3600, 1800):
          arr = arr.T

        valid = arr >= 0.0
        # Rate (mm/hr) * 0.5 hr = mm per 30-minute step
        daily_accum[valid] += arr[valid] * 0.5
        valid_count[valid] += 1

    daily_grid = np.where(valid_count == 48, daily_accum, np.nan)

    for b_idx, b_id in enumerate(basin_ids):
      if b_id not in weights_dict:
        continue
      lat_idx, lon_idx, w = weights_dict[b_id]
      if len(w) == 0:
        continue
      vals = daily_grid[lat_idx, lon_idx]
      val_mean, m_frac = _weighted_mean_valid_with_coverage(vals, w)
      res[b_idx] = val_mean
      missing_out[b_idx] = m_frac

    return {
        "imerg_precipitation": res,
        "imerg_missing_fraction": missing_out,
    }

  def extract_for_basins_dynamical(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_dt: pd.Timestamp,
      end_dt: pd.Timestamp,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Delegates to DynamicalIMERGExtractor for dynamical.org Icechunk catalog."""
    from multimet.timeseries_extractors.dynamical import DynamicalIMERGExtractor

    dyn_ext = DynamicalIMERGExtractor(data_dir=self.data_dir)
    return dyn_ext.extract_for_basins(
        basins_gdf=basins_gdf,
        start_date=start_dt,
        end_date=end_dt,
        weights_matrix=weights_matrix,
        use_bounding_box=use_bounding_box,
    )

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      matrix: Optional[ZonalWeightMatrix] = None,
      weights_dict: Optional[
          Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
      ] = None,
      use_bounding_box: bool = True,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of IMERG precipitation across basins."""
    dt = pd.to_datetime(dt)
    if self.source == "archive":
      from multimet.timeseries_extractors.gridded_archive import extract_nowcast_from_archive

      ds_day = extract_nowcast_from_archive(
          Product.IMERG,
          self.data_dir,
          basins_gdf,
          start_date=dt,
          end_date=dt,
          weights_matrix=matrix,
          use_bounding_box=use_bounding_box,
      )
      return {
          var: ds_day[var].values[:, 0].astype(np.float32)
          for var in ds_day.data_vars
      }
    if self.source in ("gesdisc", "auto", "default", "public", "nasa"):
      nc_path = self.get_daily_file(dt)
      res = self.extract_day_from_nc4(
          nc_path,
          basins_gdf,
          weights_matrix=matrix,
          use_bounding_box=use_bounding_box,
      )
      cache_dir = os.path.realpath(
          os.environ.get("MULTIMET_IMERG_CACHE", "/tmp/multimet_imerg_cache")
      )
      real_nc_path = os.path.realpath(nc_path)
      if (
          real_nc_path.startswith(cache_dir)
          and real_nc_path != cache_dir
          and os.path.exists(real_nc_path)
      ):
        Path(real_nc_path).unlink(missing_ok=True)
      return res
    elif self.source in ("dynamical", "cloud"):
      ds = self.extract_for_basins_dynamical(
          basins_gdf,
          start_dt=dt,
          end_dt=dt,
          weights_matrix=matrix,
          use_bounding_box=use_bounding_box,
      )
      out = {
          "imerg_precipitation": ds["imerg_precipitation"].values[:, 0].astype(
              np.float32
          )
      }
      if "imerg_missing_fraction" in ds.data_vars:
        out["imerg_missing_fraction"] = ds["imerg_missing_fraction"].values[
            :, 0
        ].astype(np.float32)
      return out
    else:
      imerg_files = resolve_date_to_imerg_files(self.data_dir, dt)
      basin_ids = list(basins_gdf.index)
      if weights_dict is None:
        weights_dict = {}
        for b_id in basin_ids:
          geom = basins_gdf.loc[b_id].geometry
          w = self.zonal_calc.compute_weights(b_id, geom)
          if w is not None:
            weights_dict[b_id] = w
      return self.extract_day_from_imerg_files(
          imerg_files, basin_ids, weights_dict
      )

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts daily accumulated IMERG precipitation for given basins."""
    del kwargs
    if start_date is None or end_date is None:
      raise ValueError(
          "IMERGExtractor.extract_for_basins requires both start_date and "
          "end_date to be explicitly provided."
      )
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
      raise ValueError(
          f"end_date ({end_dt}) must be >= start_date ({start_dt})."
      )

    if self.source == "archive":
      from multimet.timeseries_extractors.gridded_archive import extract_nowcast_from_archive

      return extract_nowcast_from_archive(
          Product.IMERG,
          self.data_dir,
          basins_gdf,
          start_date=start_dt,
          end_date=end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=use_bounding_box,
      )

    if self.source == "dynamical":
      return self.extract_for_basins_dynamical(
          basins_gdf,
          start_dt,
          end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=use_bounding_box,
      )

    basin_ids = list(basins_gdf.index)
    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    precip_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )
    missing_matrix = np.ones(
        (len(basin_ids), len(date_idx)), dtype=np.float32
    )

    if self.source in ("gesdisc", "auto", "default", "public", "nasa"):
      for d_idx, dt in enumerate(
          tqdm.tqdm(
              date_idx,
              desc=(
                  f"IMERG GES-DISC [{start_dt.strftime('%Y-%m-%d')} to"
                  f" {end_dt.strftime('%Y-%m-%d')}]"
              ),
              unit="day",
              leave=True,
          )
      ):
        day_res = self.extract_day(
            dt,
            basins_gdf,
            matrix=weights_matrix,
            use_bounding_box=use_bounding_box,
        )
        precip_matrix[:, d_idx] = day_res["imerg_precipitation"]
        if "imerg_missing_fraction" in day_res:
          missing_matrix[:, d_idx] = day_res["imerg_missing_fraction"]
    else:
      weights_dict = {}
      for b_id in basin_ids:
        geom = basins_gdf.loc[b_id].geometry
        w = self.zonal_calc.compute_weights(b_id, geom)
        if w is not None:
          weights_dict[b_id] = w

      for d_idx, dt in enumerate(
          tqdm.tqdm(
              date_idx,
              desc=(
                  f"IMERG [{start_dt.strftime('%Y-%m-%d')} to"
                  f" {end_dt.strftime('%Y-%m-%d')}]"
              ),
              unit="day",
              leave=True,
          )
      ):
        sub_files = resolve_date_to_imerg_files(self.data_dir, dt)
        if sub_files:
          day_res = self.extract_day_from_imerg_files(
              sub_files, basin_ids, weights_dict
          )
          precip_matrix[:, d_idx] = day_res["imerg_precipitation"]
          missing_matrix[:, d_idx] = day_res["imerg_missing_fraction"]

    ds = xr.Dataset(
        data_vars={
            "imerg_precipitation": (["basin", "date"], precip_matrix),
            "imerg_missing_fraction": (["basin", "date"], missing_matrix),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    return ds
