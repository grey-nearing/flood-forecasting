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

"""Real-time meteorological forcing fetcher for Cold-Start and Hot-Start forecasting.

Fetches the newest operational forecasts and recent spin-up forcing series directly
from live public upstream feeds and writes/appends them into MultiMet Caravan-schema
Zarr stores (``<output_dir>/<PRODUCT>/timeseries.zarr``) ready for execution by
``openhydronet``'s ``Multimet`` dataset class.

Supported operational modes
---------------------------
1. **Cold-Start (``mode="coldstart"``)**:
   Fetches 365 days of historical spin-up (``[t0 - 365d, t0]``) plus the 10-day
   operational forecast issued on ``t0`` across ``HRES``, ``IMERG``, and ``CPC``.
   By default, historical spin-up dates (``d < t0``) use the 1-day lead-time
   optimization for ``HRES`` (downloading only ``step=24h``, which is the only
   lead step consumed by ``Multimet._extract_hindcasts`` / ``forecast_overlap``),
   reducing Cold-Start HRES download volume by 10x while fetching all 10 lead
   days (``24h..240h``) on the forecast issue window.

2. **Hot-Start (``mode="hotstart"``)**:
   Inspects the existing per-product Zarr stores (and/or a saved ``openhydronet``
   hot-start state ``.npz`` file/directory) to determine the last valid date per
   product, automatically re-fetching any trailing ``NaN`` dates caused by upstream
   publication latency alongside newly elapsed days up to ``t0`` and updating the
   Zarr stores in-place without clobbering previously valid slices.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import glob
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from multimet.timeseries_extractors.config import (
    MISSING_FRACTION_VAR,
    PRODUCT_BANDS,
    PRODUCT_TYPES,
    Product,
    ProductType,
)
from multimet.timeseries_extractors.cpc import CPCExtractor
from multimet.timeseries_extractors.dynamical import (
    DYNAMICAL_FORECAST_DATASETS,
    AIFSEnsExtractor,
    AIFSExtractor,
    DynamicalDataLoader,
    DynamicalForecastExtractor,
    DynamicalIMERGExtractor,
    GEFSExtractor,
    GFSExtractor,
    IFSEnsExtractor,
    find_latest_dynamical_forecast_date,
)
from multimet.timeseries_extractors.hres import (
    ECMWF_OPEN_DATA_BUCKET,
    HRESExtractor,
    find_latest_hres_open_data_date,
)
from multimet.timeseries_extractors.imerg import IMERGExtractor
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter
from multimet.utils.gcs import configure_gcp_project
from multimet.utils.geometry import load_basin_geometries
from multimet.utils.zonal import ZonalWeightMatrix

logger = logging.getLogger(__name__)

DEFAULT_REALTIME_PRODUCTS: Tuple[str, ...] = ("HRES", "IMERG", "CPC")
SUPPORTED_REALTIME_PRODUCTS: Tuple[str, ...] = (
    "HRES",
    "IMERG",
    "CPC",
    "AIFS",
    "AIFS_ENS",
    "GFS",
    "GEFS",
    "IFS_ENS",
    "DYNAMICAL_IMERG",
)
DYNAMICAL_FORECAST_EXTRACTOR_CLASSES: Mapping[
    str, type[DynamicalForecastExtractor]
] = {
    "AIFS": AIFSExtractor,
    "AIFS_ENS": AIFSEnsExtractor,
    "GFS": GFSExtractor,
    "GEFS": GEFSExtractor,
    "IFS_ENS": IFSEnsExtractor,
}
DEFAULT_COLDSTART_LOOKBACK_DAYS: int = 365
DEFAULT_HOTSTART_LOOKBACK_DAYS: int = 7
DEFAULT_FULL_FORECAST_DAYS: int = 2
SUPPORTED_IMERG_SOURCES: Tuple[str, ...] = ("gesdisc",)


@dataclasses.dataclass
class RealtimeFetchResult(Mapping[str, str]):
  """Result of a real-time forcing fetch operation.

  Behaves as a ``Mapping[str, str]`` from product name (e.g. ``"HRES"``) to its
  output Zarr store path for drop-in compatibility with ``extract_multimet_serial``,
  while exposing operational metadata about the resolved reference date and
  per-product extraction windows.
  """

  stores: Dict[str, str]
  mode: str
  reference_date: pd.Timestamp
  start_date: pd.Timestamp
  end_date: pd.Timestamp
  product_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]
  basins: List[str]
  elapsed_seconds: float = 0.0

  def __getitem__(self, key: str) -> str:
    return self.stores[key.upper()]

  def __iter__(self) -> Iterator[str]:
    return iter(self.stores)

  def __len__(self) -> int:
    return len(self.stores)

  def summary(self) -> Dict[str, Any]:
    """Returns a JSON-serializable summary of the fetch run."""
    return {
        "mode": self.mode,
        "reference_date": self.reference_date.strftime("%Y-%m-%d"),
        "start_date": self.start_date.strftime("%Y-%m-%d"),
        "end_date": self.end_date.strftime("%Y-%m-%d"),
        "basins": list(self.basins),
        "stores": dict(self.stores),
        "product_windows": {
            prod: (
                w[0].strftime("%Y-%m-%d"),
                w[1].strftime("%Y-%m-%d"),
            )
            for prod, w in self.product_windows.items()
        },
        "elapsed_seconds": round(float(self.elapsed_seconds), 3),
    }


def read_hot_start_state_date(
    hot_start_state_path: Union[str, os.PathLike],
) -> pd.Timestamp:
  """Extracts the saved state timestamp from an ``openhydronet`` hot-start ``.npz`` file or directory.

  Supports either a single ``state_<basin>.npz`` file or a directory containing
  one or more ``*.npz`` files. When multiple basin state files are present in a
  directory, returns the earliest date across all basins so every basin's catch-up
  window is covered.
  """
  path_str = os.fspath(hot_start_state_path)
  if not os.path.exists(path_str):
    raise FileNotFoundError(
        f"Hot-start state path does not exist: {path_str!r}"
    )

  npz_files: List[str] = []
  if os.path.isdir(path_str):
    npz_files = sorted(glob.glob(os.path.join(path_str, "*.npz")))
    if not npz_files:
      raise FileNotFoundError(
          f"No .npz hot-start state files found in directory: {path_str!r}"
      )
  elif path_str.endswith(".npz"):
    npz_files = [path_str]
  else:
    raise ValueError(
        "Expected a .npz hot-start state file or directory containing .npz"
        f" files, got: {path_str!r}"
    )

  dates: List[pd.Timestamp] = []
  date_keys = (
      "date",
      "last_date",
      "timestamp",
      "time",
      "end_date",
      "forecast_date",
      "spinup_end_date",
  )
  for fpath in npz_files:
    with np.load(fpath, allow_pickle=False) as data:
      matched_key: Optional[str] = None
      for key in date_keys:
        if key in data.files:
          matched_key = key
          val = data[key]
          scalar = val.item() if getattr(val, "ndim", 0) == 0 else val[0]
          if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8")
          dates.append(pd.to_datetime(scalar).floor("D"))
          break
      if matched_key is None:
        raise ValueError(
            f"Hot-start state file {fpath!r} does not contain any recognized"
            f" date key {date_keys}; found keys: {list(data.files)}"
        )

  return min(dates)


REQUIRED_STORE_BANDS: Mapping[Product, Tuple[str, ...]] = {
    Product.CPC: ("cpc_precipitation",),
}


def inspect_store_last_valid_date(
    writer: MultiMetZarrWriter,
    product: Product,
    basin_ids: Optional[Sequence[str]] = None,
    require_all_leads_on_last_date: bool = False,
    require_all_leads_from: Optional[pd.Timestamp] = None,
) -> Optional[pd.Timestamp]:
  """Finds the latest date in a product's Zarr store that contains valid upstream data.

  Walks backwards from the end of the store's ``date`` coordinate so that any
  trailing ``NaN`` slices or latency-bridged slices (where
  ``missing_fraction >= 1.0`` because the upstream feed had not yet published
  the latest day) are excluded, ensuring subsequent ``hotstart`` runs re-fetch
  and heal those dates automatically.
  """
  if not writer.store_exists(product):
    return None

  store_path = writer.get_store_path(product)
  with xr.open_zarr(store_path) as ds:
    if "date" not in ds.coords or len(ds["date"]) == 0:
      return None
    store_basins = [str(b).rstrip("\x00") for b in ds["basin"].values]
    if basin_ids is not None:
      missing_basins = [b for b in basin_ids if str(b) not in set(store_basins)]
      if missing_basins:
        return None
      ds_sub = ds.sel(basin=[str(b) for b in basin_ids])
    else:
      ds_sub = ds

    required_bands = REQUIRED_STORE_BANDS.get(product, PRODUCT_BANDS[product])
    if any(b not in ds_sub.data_vars for b in required_bands):
      return None

    band_arrays = [
        np.asarray(ds_sub[b].values, dtype=np.float32) for b in required_bands
    ]
    for b in PRODUCT_BANDS[product]:
      if b not in required_bands and b in ds_sub.data_vars:
        arr_opt = np.asarray(ds_sub[b].values, dtype=np.float32)
        if not np.all(np.isnan(arr_opt)):
          band_arrays.append(arr_opt)
    missing_var = MISSING_FRACTION_VAR.get(product)
    miss_vals = (
        np.asarray(ds_sub[missing_var].values, dtype=np.float32)
        if missing_var and missing_var in ds_sub.data_vars
        else None
    )
    dates = pd.DatetimeIndex(pd.to_datetime(ds_sub["date"].values).floor("D"))
    is_forecast = PRODUCT_TYPES[product] == ProductType.FORECAST

    for d_idx in range(len(dates) - 1, -1, -1):
      dt = dates[d_idx]
      if is_forecast:
        need_all_leads = require_all_leads_on_last_date or (
            require_all_leads_from is not None and dt >= require_all_leads_from
        )
        if need_all_leads:
          valid = all(
              bool(np.all(~np.isnan(arr[:, d_idx, :]))) for arr in band_arrays
          )
          if valid and miss_vals is not None:
            valid = bool(np.all(miss_vals[:, d_idx, :] < 1.0))
        else:
          valid = all(
              bool(np.all(~np.isnan(arr[:, d_idx, 0]))) for arr in band_arrays
          )
          if valid and miss_vals is not None:
            valid = bool(np.all(miss_vals[:, d_idx, 0] < 1.0))
      else:
        valid = all(
            bool(np.all(~np.isnan(arr[:, d_idx]))) for arr in band_arrays
        )
        if valid and miss_vals is not None:
          valid = bool(np.all(miss_vals[:, d_idx] < 1.0))

      if valid:
        return dt
  return None


class RealtimeForcingFetcher:
  """Orchestrates Cold-Start and Hot-Start real-time meteorological forcing extraction."""

  def __init__(
      self,
      output_dir: Union[str, os.PathLike],
      *,
      hres_bucket: str = ECMWF_OPEN_DATA_BUCKET,
      imerg_source: str = "gesdisc",
      cpc_cache_dir: Optional[str] = None,
      earthdata_username: Optional[str] = None,
      earthdata_password: Optional[str] = None,
      earthdata_token: Optional[str] = None,
      netrc_path: Optional[str] = None,
      gcp_project: Optional[str] = None,
      hres_fs: Optional[Any] = None,
      dynamical_loaders: Optional[Mapping[str, DynamicalDataLoader]] = None,
      include_ensemble_members: bool = False,
  ):
    self.output_dir = os.fspath(output_dir)
    self.hres_bucket = hres_bucket
    self.imerg_source = imerg_source.lower().strip()
    if self.imerg_source not in SUPPORTED_IMERG_SOURCES:
      raise ValueError(
          f"Invalid imerg_source {imerg_source!r}; expected one of"
          f" {SUPPORTED_IMERG_SOURCES}."
      )
    self.cpc_cache_dir = (
        os.path.join(tempfile.gettempdir(), "multimet_cpc_cache")
        if cpc_cache_dir is None
        else cpc_cache_dir
    )
    self.earthdata_username = earthdata_username
    self.earthdata_password = earthdata_password
    self.earthdata_token = earthdata_token
    self.netrc_path = netrc_path
    self.gcp_project = gcp_project
    self.hres_fs = hres_fs
    self.dynamical_loaders: Dict[str, DynamicalDataLoader] = (
        {str(k).strip().upper(): v for k, v in dynamical_loaders.items()}
        if dynamical_loaders is not None
        else {}
    )
    self.include_ensemble_members = bool(include_ensemble_members)

    if self.output_dir.startswith(("gs://", "gcs://")) or gcp_project:
      self.gcp_project = configure_gcp_project(gcp_project)
    else:
      os.makedirs(self.output_dir, exist_ok=True)

    self.writer = MultiMetZarrWriter(self.output_dir)

  def resolve_reference_date(
      self,
      reference_date: Optional[Union[str, pd.Timestamp]] = None,
      require_full_10d: bool = True,
      products: Optional[Sequence[Union[str, Product]]] = None,
  ) -> pd.Timestamp:
    """Resolves the forecast issue date ``t0`` across all requested forecast products."""
    if reference_date is None or str(reference_date).strip().lower() == "latest":
      norm_prods = (
          [
              p.value if isinstance(p, Product) else str(p).strip().upper()
              for p in products
          ]
          if products is not None
          else list(DEFAULT_REALTIME_PRODUCTS)
      )
      dyn_forecast_prods = [
          p for p in norm_prods if p in DYNAMICAL_FORECAST_EXTRACTOR_CLASSES
      ]
      discovered_dates: List[pd.Timestamp] = []

      if "HRES" in norm_prods:
        t_hres = find_latest_hres_open_data_date(
            bucket=self.hres_bucket,
            require_full_10d=require_full_10d,
            fs=self.hres_fs,
        )
        logger.info(
            "Auto-discovered latest published ECMWF HRES initialization date: %s",
            t_hres.strftime("%Y-%m-%d"),
        )
        discovered_dates.append(t_hres)

      for prod_name in dyn_forecast_prods:
        ext_cls = DYNAMICAL_FORECAST_EXTRACTOR_CLASSES[prod_name]
        loader = self.dynamical_loaders.get(prod_name)
        if loader is None:
          loader = DynamicalDataLoader(ext_cls.DEFAULT_DATASET_ID)
          self.dynamical_loaders[prod_name] = loader
        t_prod = find_latest_dynamical_forecast_date(
            ext_cls.DEFAULT_DATASET_ID,
            require_full_10d=require_full_10d,
            loader=loader,
        )
        discovered_dates.append(t_prod)

      if discovered_dates:
        t0 = min(discovered_dates)
        if dyn_forecast_prods:
          logger.info(
              "Auto-discovered latest published forecast initialization date"
              " across %s: %s",
              norm_prods,
              t0.strftime("%Y-%m-%d"),
          )
        return t0

      if "DYNAMICAL_IMERG" in norm_prods:
        loader = self.dynamical_loaders.get("DYNAMICAL_IMERG")
        if loader is None:
          loader = DynamicalDataLoader(
              DynamicalIMERGExtractor.DEFAULT_DATASET_ID
          )
          self.dynamical_loaders["DYNAMICAL_IMERG"] = loader
        raw_times = pd.DatetimeIndex(
            pd.to_datetime(loader.ds[loader.time_dim].values).tz_localize(None)
        )
        last_ts = raw_times[-1]
        last_day = last_ts.floor("D")
        if int(np.sum(raw_times.floor("D") == last_day)) >= 48:
          return pd.Timestamp(last_day)
        return pd.Timestamp(last_day - pd.Timedelta(days=1))

      return (
          pd.Timestamp.now("UTC").tz_localize(None).floor("D")
          - pd.Timedelta(days=1)
      )
    return pd.to_datetime(reference_date).floor("D")

  def plan_product_window(
      self,
      product: Product,
      basin_ids: Sequence[str],
      reference_date: pd.Timestamp,
      mode: str,
      lookback_days: Optional[int] = None,
      hot_start_state_date: Optional[pd.Timestamp] = None,
      overwrite: bool = False,
      spinup_only_lead_1d: bool = True,
      full_forecast_days: int = DEFAULT_FULL_FORECAST_DAYS,
  ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Determines the ``(start_date, end_date)`` extraction window for a product.

    - In ``coldstart`` mode: defaults to ``lookback_days=365`` ->
      ``[reference_date - 365d, reference_date]``. If ``overwrite=False`` and
      the store already contains a contiguous valid prefix covering the start
      of the spin-up window for all ``basin_ids``, resumes from the last valid
      date in the store so interrupted Cold-Start runs resume fast.
    - In ``hotstart`` mode: inspects ``hot_start_state_date`` and the product's
      existing Zarr store to find the last valid date, then fetches from
      ``last_valid_date`` (or ``reference_date - lookback_days`` if no store
      exists) through ``reference_date``.
    """
    mode_norm = mode.lower().strip()
    if mode_norm not in ("coldstart", "cold_start", "hotstart", "hot_start"):
      raise ValueError(
          f"Invalid mode {mode!r}; expected 'coldstart' or 'hotstart'."
      )
    is_coldstart = mode_norm in ("coldstart", "cold_start")
    require_all_leads_from = (
        reference_date
        - pd.Timedelta(days=max(0, int(full_forecast_days) - 1))
        if spinup_only_lead_1d
        else pd.Timestamp.min
    )

    if is_coldstart:
      lb = (
          int(lookback_days)
          if lookback_days is not None
          else DEFAULT_COLDSTART_LOOKBACK_DAYS
      )
      target_start = reference_date - pd.Timedelta(days=lb)
      if not overwrite and self.writer.store_exists(product):
        info = self.writer.get_store_info(product)
        if info is not None and len(info["dates"]) > 0:
          store_basins = set(info["basins"])
          if all(str(b) in store_basins for b in basin_ids):
            min_store_dt = pd.to_datetime(info["dates"].min()).floor("D")
            last_valid = inspect_store_last_valid_date(
                self.writer,
                product,
                basin_ids,
                require_all_leads_from=require_all_leads_from,
            )
            if (
                last_valid is not None
                and min_store_dt <= target_start
                and last_valid >= target_start
            ):
              resume_start = min(last_valid, reference_date)
              return (resume_start, reference_date)
      return (target_start, reference_date)

    # Hot-start mode
    if overwrite:
      lb = (
          int(lookback_days)
          if lookback_days is not None
          else DEFAULT_HOTSTART_LOOKBACK_DAYS
      )
      return (reference_date - pd.Timedelta(days=lb), reference_date)

    last_valid = inspect_store_last_valid_date(
        self.writer,
        product,
        basin_ids,
        require_all_leads_from=require_all_leads_from,
    )
    candidates: List[pd.Timestamp] = []
    if last_valid is not None:
      candidates.append(last_valid)
    if hot_start_state_date is not None:
      candidates.append(hot_start_state_date)

    if candidates:
      start_dt = min(candidates)
      if lookback_days is not None:
        explicit_start = reference_date - pd.Timedelta(days=int(lookback_days))
        start_dt = min(start_dt, explicit_start)
      start_dt = min(start_dt, reference_date)
      return (start_dt, reference_date)

    lb = (
        int(lookback_days)
        if lookback_days is not None
        else DEFAULT_HOTSTART_LOOKBACK_DAYS
    )
    return (reference_date - pd.Timedelta(days=lb), reference_date)

  def _extract_product_dataset(
      self,
      prod_name: str,
      basins_gdf: gpd.GeoDataFrame,
      start_dt: pd.Timestamp,
      end_dt: pd.Timestamp,
      reference_dt: pd.Timestamp,
      *,
      spinup_only_lead_1d: bool = True,
      full_forecast_days: int = 1,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
  ) -> xr.Dataset:
    """Runs the live upstream extractor for a single product over ``[start_dt, end_dt]``."""
    spinup_cutoff: Optional[pd.Timestamp] = None
    if spinup_only_lead_1d:
      spinup_cutoff = reference_dt - pd.Timedelta(
          days=max(0, int(full_forecast_days) - 1)
      )

    if prod_name == "HRES":
      extractor = HRESExtractor(
          data_dir=self.hres_bucket,
          source="open_data",
          fs=self.hres_fs,
      )
      if (
          weights_matrix is not None
          and extractor.lats is not None
          and extractor.lons is not None
          and weights_matrix.grid_shape
          != (len(extractor.lats), len(extractor.lons))
      ):
        weights_matrix = None
      return extractor.extract_for_basins_open_data(
          basins_gdf,
          start_date=start_dt,
          end_date=end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=True,
          spinup_only_before=spinup_cutoff,
      )

    if prod_name in DYNAMICAL_FORECAST_EXTRACTOR_CLASSES:
      extractor_cls = DYNAMICAL_FORECAST_EXTRACTOR_CLASSES[prod_name]
      loader = self.dynamical_loaders.get(prod_name)
      if extractor_cls.DEFAULT_IS_ENSEMBLE:
        extractor = extractor_cls(
            loader=loader,
            include_ensemble_members=self.include_ensemble_members,
        )
      else:
        extractor = extractor_cls(loader=loader)
      self.dynamical_loaders[prod_name] = extractor.loader
      return extractor.extract_for_basins(
          basins_gdf,
          start_date=start_dt,
          end_date=end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=True,
          spinup_only_before=spinup_cutoff,
      )

    if prod_name == "DYNAMICAL_IMERG":
      extractor = DynamicalIMERGExtractor(
          loader=self.dynamical_loaders.get("DYNAMICAL_IMERG")
      )
      self.dynamical_loaders["DYNAMICAL_IMERG"] = extractor.loader
      return extractor.extract_for_basins(
          basins_gdf,
          start_date=start_dt,
          end_date=end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=True,
      )

    if prod_name == "IMERG":
      extractor = IMERGExtractor(
          source="gesdisc",
          username=self.earthdata_username,
          password=self.earthdata_password,
          token=self.earthdata_token,
          netrc_path=self.netrc_path,
      )
      if (
          weights_matrix is not None
          and extractor.lats is not None
          and extractor.lons is not None
          and weights_matrix.grid_shape
          != (len(extractor.lats), len(extractor.lons))
      ):
        weights_matrix = None
      return extractor.extract_for_basins(
          basins_gdf,
          start_date=start_dt,
          end_date=end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=True,
      )

    if prod_name == "CPC":
      extractor = CPCExtractor(
          source="psl",
          cache_dir=self.cpc_cache_dir,
      )
      if (
          weights_matrix is not None
          and extractor.lats is not None
          and extractor.lons is not None
          and weights_matrix.grid_shape
          != (len(extractor.lats), len(extractor.lons))
      ):
        weights_matrix = None
      return extractor.extract_for_basins(
          basins_gdf,
          start_date=start_dt,
          end_date=end_dt,
          weights_matrix=weights_matrix,
          use_bounding_box=True,
      )

    raise ValueError(
        f"Unsupported real-time product {prod_name!r}. "
        f"Supported real-time products: {list(SUPPORTED_REALTIME_PRODUCTS)}"
    )

  def fetch(
      self,
      basins: Union[
          str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any], Sequence[Any]
      ],
      *,
      mode: str = "hotstart",
      reference_date: Optional[Union[str, pd.Timestamp]] = None,
      lookback_days: Optional[int] = None,
      products: Optional[Sequence[Union[str, Product]]] = None,
      hot_start_state_path: Optional[Union[str, os.PathLike]] = None,
      spinup_only_lead_1d: bool = True,
      full_forecast_days: int = DEFAULT_FULL_FORECAST_DAYS,
      id_column: Optional[str] = None,
      overwrite: bool = False,
      weights_cache: Optional[str] = None,
  ) -> RealtimeFetchResult:
    """Fetches real-time meteorological forcings and writes/appends to Zarr stores.

    Args:
      basins: Catchment geometries (file path, GeoDataFrame, or GeoJSON dict).
      mode: ``"coldstart"`` (default 365-day spin-up + 10-day forecast) or
        ``"hotstart"`` (incremental catch-up from existing Zarr store or state
        file + 10-day forecast).
      reference_date: Forecast issue date ``t0`` (``"YYYY-MM-DD"``, Timestamp,
        or ``"latest"`` / ``None`` to auto-discover the newest published HRES
        or dynamical.org 00z forecast).
      lookback_days: Optional explicit lookback window in days before
        ``reference_date`` (defaults to ``365`` for ``coldstart`` and ``7`` for
        ``hotstart`` when no existing Zarr store is present).
      products: Sequence of products to fetch (defaults to
        ``("HRES", "IMERG", "CPC")``; also supports ``"AIFS"``, ``"AIFS_ENS"``,
        ``"GFS"``, ``"GEFS"``, ``"IFS_ENS"``, and ``"DYNAMICAL_IMERG"``).
      hot_start_state_path: Optional path to a saved ``openhydronet``
        hot-start ``.npz`` state file or directory of ``*.npz`` files.
      spinup_only_lead_1d: When ``True`` (default), historical forecast dates
        before the forecast issue window only download ``lead_time=1D``
        (``0h..24h``), cutting Cold-Start download volume by 10x. Set ``False``
        to fetch all 10 lead days for every historical spin-up date.
      full_forecast_days: Number of trailing initialization dates up to and
        including ``reference_date`` for which all 10 forecast lead days are
        fetched (default ``2``, i.e. ``reference_date - 1d`` and ``reference_date``).
      id_column: Optional basin ID column name in ``basins``.
      overwrite: Whether to rebuild the destination Zarr stores from scratch.
      weights_cache: Optional path to precomputed ``.npz`` weights matrix.

    Returns:
      :class:`RealtimeFetchResult` mapping product names to Zarr store paths.
    """
    t_start = time.time()
    basins_gdf = load_basin_geometries(basins, id_column=id_column)
    basin_ids = [str(b) for b in basins_gdf.index]

    if products is None:
      target_prods = list(DEFAULT_REALTIME_PRODUCTS)
    else:
      target_prods = [
          p.value if isinstance(p, Product) else str(p).strip().upper()
          for p in products
      ]

    for prod_name in target_prods:
      if prod_name not in SUPPORTED_REALTIME_PRODUCTS:
        raise ValueError(
            f"Unsupported product {prod_name!r}. Supported products: "
            f"{list(SUPPORTED_REALTIME_PRODUCTS)}"
        )

    ref_dt = self.resolve_reference_date(reference_date, products=target_prods)
    state_dt: Optional[pd.Timestamp] = None
    if hot_start_state_path is not None:
      state_dt = read_hot_start_state_date(hot_start_state_path)
      logger.info(
          "Detected hot-start state timestamp: %s",
          state_dt.strftime("%Y-%m-%d"),
      )

    loaded_weights: Optional[ZonalWeightMatrix] = None
    if weights_cache is not None:
      if not os.path.exists(weights_cache):
        raise FileNotFoundError(
            f"Weights cache path does not exist: {weights_cache!r}"
        )
      loaded_weights = ZonalWeightMatrix.load(weights_cache)

    stores: Dict[str, str] = {}
    product_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    overall_start = ref_dt
    overall_end = ref_dt

    for prod_name in target_prods:
      prod_enum = Product[prod_name]

      start_dt, end_dt = self.plan_product_window(
          prod_enum,
          basin_ids=basin_ids,
          reference_date=ref_dt,
          mode=mode,
          lookback_days=lookback_days,
          hot_start_state_date=state_dt,
          overwrite=overwrite,
          spinup_only_lead_1d=spinup_only_lead_1d,
          full_forecast_days=full_forecast_days,
      )
      product_windows[prod_name] = (start_dt, end_dt)
      overall_start = min(overall_start, start_dt)
      overall_end = max(overall_end, end_dt)

      logger.info(
          "[%s] Fetching %s from %s to %s (reference_date=%s)...",
          mode.upper(),
          prod_name,
          start_dt.strftime("%Y-%m-%d"),
          end_dt.strftime("%Y-%m-%d"),
          ref_dt.strftime("%Y-%m-%d"),
      )

      ds = self._extract_product_dataset(
          prod_name,
          basins_gdf=basins_gdf,
          start_dt=start_dt,
          end_dt=end_dt,
          reference_dt=ref_dt,
          spinup_only_lead_1d=spinup_only_lead_1d,
          full_forecast_days=full_forecast_days,
          weights_matrix=loaded_weights,
      )

      store_path = self.writer.write_or_append(
          ds,
          prod_enum,
          overwrite_existing_basins=overwrite,
          preserve_existing_valid=not overwrite,
      )
      self.writer.consolidate_metadata(prod_enum)
      stores[prod_name] = store_path
      logger.info("Updated %s Zarr store at %s", prod_name, store_path)

    return RealtimeFetchResult(
        stores=stores,
        mode=mode.lower().strip(),
        reference_date=ref_dt,
        start_date=overall_start,
        end_date=overall_end,
        product_windows=product_windows,
        basins=basin_ids,
        elapsed_seconds=time.time() - t_start,
    )


def fetch_realtime_multimet(
    basins: Union[
        str, os.PathLike, gpd.GeoDataFrame, Dict[str, Any], Sequence[Any]
    ],
    output_dir: Union[str, os.PathLike],
    *,
    mode: str = "hotstart",
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    lookback_days: Optional[int] = None,
    products: Optional[Sequence[Union[str, Product]]] = None,
    hot_start_state_path: Optional[Union[str, os.PathLike]] = None,
    imerg_source: str = "gesdisc",
    hres_bucket: str = ECMWF_OPEN_DATA_BUCKET,
    cpc_cache_dir: Optional[str] = None,
    spinup_only_lead_1d: bool = True,
    full_forecast_days: int = DEFAULT_FULL_FORECAST_DAYS,
    id_column: Optional[str] = None,
    overwrite: bool = False,
    weights_cache: Optional[str] = None,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
    gcp_project: Optional[str] = None,
    hres_fs: Optional[Any] = None,
    dynamical_loaders: Optional[Mapping[str, DynamicalDataLoader]] = None,
    include_ensemble_members: bool = False,
) -> RealtimeFetchResult:
  """Convenience entry point to fetch Cold-Start or Hot-Start real-time forcings.

  Args:
    basins: Catchment geometries (GeoJSON/Shapefile path, GeoDataFrame, or dict).
    output_dir: Destination directory for ``<PRODUCT>/timeseries.zarr`` stores.
    mode: ``"coldstart"`` (365-day spin-up + 10-day forecast) or ``"hotstart"``
      (incremental catch-up + 10-day forecast).
    reference_date: Forecast issue date ``t0`` (``"YYYY-MM-DD"`` or ``"latest"``).
    lookback_days: Optional override for spin-up / catch-up lookback days.
    products: Products to fetch (defaults to ``("HRES", "IMERG", "CPC")``; also
      supports ``"AIFS"``, ``"AIFS_ENS"``, ``"GFS"``, ``"GEFS"``, ``"IFS_ENS"``,
      and ``"DYNAMICAL_IMERG"``).
    hot_start_state_path: Optional path to ``openhydronet`` hot-start state
      ``.npz`` file or directory.
    imerg_source: ``"gesdisc"`` (NASA GES DISC HTTP with Earthdata Login).
    hres_bucket: GCS bucket name for ECMWF Open Data (default ``"ecmwf-open-data"``).
    cpc_cache_dir: Local cache directory for yearly NOAA PSL CPC NetCDF files.
    spinup_only_lead_1d: Whether to fetch only ``lead_time=1D`` (``step=24h``)
      for historical forecast spin-up dates prior to the forecast issue window.
    full_forecast_days: Number of trailing initialization dates up to ``t0``
      with full 10-day forecast lead extraction (default ``2``).
    id_column: Optional column name for basin identifiers in ``basins``.
    overwrite: Whether to overwrite existing Zarr stores from scratch.
    weights_cache: Optional path to cached ``.npz`` zonal weight matrix.
    earthdata_username: Optional NASA Earthdata Login username.
    earthdata_password: Optional NASA Earthdata Login password.
    earthdata_token: Optional NASA Earthdata Bearer token.
    netrc_path: Optional path to custom ``.netrc`` file.
    gcp_project: Optional GCP project ID for GCS operations.
    hres_fs: Optional custom filesystem object for testing HRES Open Data reads.
    dynamical_loaders: Optional mapping of dynamical.org product names to
      pre-initialized :class:`DynamicalDataLoader` instances.
    include_ensemble_members: Whether to include raw 4D
      ``(basin, date, ensemble_member, lead_time)`` bands alongside ensemble
      summary statistics for ``IFS_ENS``, ``GEFS``, and ``AIFS_ENS``.

  Returns:
    :class:`RealtimeFetchResult` mapping product names to Zarr store paths.
  """
  fetcher = RealtimeForcingFetcher(
      output_dir=output_dir,
      hres_bucket=hres_bucket,
      imerg_source=imerg_source,
      cpc_cache_dir=cpc_cache_dir,
      earthdata_username=earthdata_username,
      earthdata_password=earthdata_password,
      earthdata_token=earthdata_token,
      netrc_path=netrc_path,
      gcp_project=gcp_project,
      hres_fs=hres_fs,
      dynamical_loaders=dynamical_loaders,
      include_ensemble_members=include_ensemble_members,
  )
  return fetcher.fetch(
      basins=basins,
      mode=mode,
      reference_date=reference_date,
      lookback_days=lookback_days,
      products=products,
      hot_start_state_path=hot_start_state_path,
      spinup_only_lead_1d=spinup_only_lead_1d,
      full_forecast_days=full_forecast_days,
      id_column=id_column,
      overwrite=overwrite,
      weights_cache=weights_cache,
  )


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the CLI argument parser for ``python -m multimet.timeseries_extractors.realtime``."""
  parser = argparse.ArgumentParser(
      prog="multimet-realtime",
      description=(
          "Fetch operational real-time meteorological forcings (Cold-Start or "
          "Hot-Start) into MultiMet Zarr stores."
      ),
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--basins_path",
      nargs="+",
      required=True,
      help="Path(s) to GeoJSON or Shapefile catchment boundaries.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Destination directory for <PRODUCT>/timeseries.zarr stores.",
  )
  parser.add_argument(
      "--mode",
      type=str,
      choices=("coldstart", "hotstart"),
      default="hotstart",
      help=(
          "Execution mode: 'coldstart' fetches 365-day spin-up + 10-day forecast; "
          "'hotstart' incrementally updates existing stores up to reference_date."
      ),
  )
  parser.add_argument(
      "--reference_date",
      type=str,
      default="latest",
      help=(
          "Forecast initialization date (YYYY-MM-DD) or 'latest' to auto-discover "
          "the newest published 00z forecast run."
      ),
  )
  parser.add_argument(
      "--lookback_days",
      type=int,
      default=None,
      help=(
          "Optional override for spin-up / catch-up lookback days before "
          "reference_date (defaults to 365 for coldstart, 7 for hotstart)."
      ),
  )
  parser.add_argument(
      "--products",
      type=str,
      default=",".join(DEFAULT_REALTIME_PRODUCTS),
      help=(
          "Comma-separated list of real-time products to fetch "
          f"({','.join(SUPPORTED_REALTIME_PRODUCTS)})."
      ),
  )
  parser.add_argument(
      "--include_ensemble_members",
      action="store_true",
      help=(
          "Also emit raw 4D (basin, date, ensemble_member, lead_time) variables "
          "for ensemble products (IFS_ENS, GEFS, AIFS_ENS)."
      ),
  )
  parser.add_argument(
      "--hot_start_state",
      type=str,
      default=None,
      help="Optional path to openhydronet hot-start .npz state file or directory.",
  )
  parser.add_argument(
      "--imerg_source",
      type=str,
      choices=SUPPORTED_IMERG_SOURCES,
      default="gesdisc",
      help="Upstream source for IMERG daily precipitation.",
  )
  parser.add_argument(
      "--hres_bucket",
      type=str,
      default=ECMWF_OPEN_DATA_BUCKET,
      help="GCS bucket name or URI for ECMWF Open Data HRES GRIB2 files.",
  )
  parser.add_argument(
      "--cpc_cache_dir",
      type=str,
      default=None,
      help="Local cache directory for NOAA PSL CPC yearly NetCDF files.",
  )
  parser.add_argument(
      "--full_hindcast_leads",
      action="store_true",
      help=(
          "Fetch all 10 lead days (24h..240h) for every historical spin-up date "
          "instead of only lead_time=1D (24h) on dates prior to reference_date."
      ),
  )
  parser.add_argument(
      "--full_forecast_days",
      type=int,
      default=DEFAULT_FULL_FORECAST_DAYS,
      help=(
          "Number of trailing initialization dates up to reference_date for "
          "which all 10 forecast lead days are fetched."
      ),
  )
  parser.add_argument(
      "--id_column",
      type=str,
      default=None,
      help="Optional column name for basin ID in the input geometry file.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite existing destination Zarr stores from scratch.",
  )
  parser.add_argument(
      "--weights_cache",
      type=str,
      default=None,
      help="Optional path to precomputed .npz zonal weight matrix.",
  )
  parser.add_argument(
      "--earthdata_username",
      type=str,
      default=None,
      help="Optional NASA Earthdata Login username (for --imerg_source=gesdisc).",
  )
  parser.add_argument(
      "--earthdata_password",
      type=str,
      default=None,
      help="Optional NASA Earthdata Login password (for --imerg_source=gesdisc).",
  )
  parser.add_argument(
      "--earthdata_token",
      type=str,
      default=None,
      help="Optional NASA Earthdata Bearer token (for --imerg_source=gesdisc).",
  )
  parser.add_argument(
      "--netrc_path",
      type=str,
      default=None,
      help="Optional path to .netrc file for NASA Earthdata credentials.",
  )
  parser.add_argument(
      "--gcp_project",
      type=str,
      default=None,
      help="Optional GCP project ID for GCS operations.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> RealtimeFetchResult:
  """CLI entry point for ``python -m multimet.timeseries_extractors.realtime``."""
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  prods = [p.strip().upper() for p in args.products.split(",") if p.strip()]

  print(
      f"▶ Starting MultiMet real-time fetch (mode={args.mode}, "
      f"reference_date={args.reference_date}, products={prods})"
  )
  result = fetch_realtime_multimet(
      basins=args.basins_path,
      output_dir=args.output_dir,
      mode=args.mode,
      reference_date=args.reference_date,
      lookback_days=args.lookback_days,
      products=prods,
      hot_start_state_path=args.hot_start_state,
      imerg_source=args.imerg_source,
      hres_bucket=args.hres_bucket,
      cpc_cache_dir=args.cpc_cache_dir,
      spinup_only_lead_1d=not args.full_hindcast_leads,
      full_forecast_days=args.full_forecast_days,
      id_column=args.id_column,
      overwrite=args.overwrite,
      weights_cache=args.weights_cache,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
      gcp_project=args.gcp_project,
      include_ensemble_members=args.include_ensemble_members,
  )
  print(
      f"\n✓ Completed real-time {result.mode} fetch for reference_date="
      f"{result.reference_date.strftime('%Y-%m-%d')} in "
      f"{result.elapsed_seconds:.2f}s:"
  )
  for prod, store_path in result.stores.items():
    w_start, w_end = result.product_windows[prod]
    print(
        f"  • {prod:10s} [{w_start.strftime('%Y-%m-%d')} .. "
        f"{w_end.strftime('%Y-%m-%d')}] -> {store_path}"
    )
  return result


if __name__ == "__main__":
  main()
