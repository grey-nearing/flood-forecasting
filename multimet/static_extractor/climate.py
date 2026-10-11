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

"""Climate metrics and ERA5 indices computation following Caravan methodology.

Implements FAO-56 Penman-Monteith potential evapotranspiration, Knoben et al. (2018)
climate indices, Addor et al. (2017) extreme precipitation indices, pre-extracted
basin ERA5-Land Zarr timeseries loading, area-weighted Level 12 continental ERA5
precomputed climate indices loading, and gridded ERA5-Land Zarr extraction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import dask
import numpy as np
import pandas as pd
import shapely.geometry
import xarray as xr
import zarr

from multimet.static_extractor.config import CARAVAN_CLIMATE_COLUMNS, CONTINENT_MAP
from multimet.timeseries_extractors.gridded_archive import open_gridded_archive
from multimet.utils.climate import (
    calculate_fao_pm_pet,
    calculate_knoben_moisture_and_seasonality,
    compute_caravan_climate_metrics,
    depth_to_mm,
    normalize_era5_pet_sign,
    pressure_to_kpa,
    temp_to_celsius,
)
from multimet.utils.zonal import (
    MIN_VALID_COVERAGE_FRACTION,
    ZonalWeightCalculator,
)

__all__ = [
    "CARAVAN_CLIMATE_ALIASES",
    "CARAVAN_CLIMATE_STORED_KEYS",
    "ERA5ClimateLoader",
    "ERA5GriddedExtractor",
    "ERA5TimeseriesZarrLoader",
    "NAN_CLIMATE_METRICS",
    "calculate_fao_pm_pet",
    "calculate_knoben_moisture_and_seasonality",
    "compute_caravan_climate_metrics",
    "compute_climate_metrics_from_dataframe",
    "depth_to_mm",
    "expand_caravan_climate_aliases",
    "normalize_era5_pet_sign",
    "pressure_to_kpa",
    "temp_to_celsius",
]

logger = logging.getLogger(__name__)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Caravan (>= v1.5) convention: the unsuffixed PET-derived indices are computed
# with FAO-56 Penman-Monteith PET, so the ``*_FAO_PM`` columns are exact aliases
# of the unsuffixed columns. The ``*_ERA5_LAND`` columns (ERA5-Land native
# potential evaporation) are distinct quantities and are never aliased. This
# table is the single place where that relationship is defined.
CARAVAN_CLIMATE_ALIASES: Mapping[str, str] = MappingProxyType({
    "pet_mean_FAO_PM": "pet_mean",
    "aridity_FAO_PM": "aridity",
    "moisture_index_FAO_PM": "moisture_index",
    "seasonality_FAO_PM": "seasonality",
})

# Keys physically stored in the Level 12 continental ``*_climate_indices.txt`` tables.
CARAVAN_CLIMATE_STORED_KEYS: Tuple[str, ...] = tuple(
    k for k in CARAVAN_CLIMATE_COLUMNS if k not in CARAVAN_CLIMATE_ALIASES
)


def expand_caravan_climate_aliases(values: Mapping[str, Any]) -> Dict[str, Any]:
  """Returns all 18 Caravan climate columns from the stored keys, filling aliases and NaNs."""
  out: Dict[str, Any] = {}
  for key in CARAVAN_CLIMATE_COLUMNS:
    source_key = CARAVAN_CLIMATE_ALIASES.get(key, key)
    out[key] = values.get(source_key, np.nan)
  return out


NAN_CLIMATE_METRICS: Dict[str, float] = {
    "p_mean": np.nan,
    "pet_mean": np.nan,
    "pet_mean_FAO_PM": np.nan,
    "pet_mean_ERA5_LAND": np.nan,
    "aridity": np.nan,
    "aridity_FAO_PM": np.nan,
    "aridity_ERA5_LAND": np.nan,
    "frac_snow": np.nan,
    "moisture_index": np.nan,
    "moisture_index_FAO_PM": np.nan,
    "moisture_index_ERA5_LAND": np.nan,
    "seasonality": np.nan,
    "seasonality_FAO_PM": np.nan,
    "seasonality_ERA5_LAND": np.nan,
    "high_prec_freq": np.nan,
    "high_prec_dur": np.nan,
    "low_prec_freq": np.nan,
    "low_prec_dur": np.nan,
}

_PRECIP_CANDIDATES: Tuple[str, ...] = (
    "era5land_total_precipitation",
    "total_precipitation_sum",
    "total_precipitation_24hr",
    "total_precipitation",
    "prcp",
    "precip",
    "tp",
)
_TEMP_CANDIDATES: Tuple[str, ...] = (
    "era5land_temperature_2m",
    "temperature_2m_mean",
    "2m_temperature",
    "temperature_2m",
    "temperature",
    "temp",
    "t2m",
)
_PET_ERA5_CANDIDATES: Tuple[str, ...] = (
    "era5land_potential_evaporation_DEPRECATED",
    "potential_evaporation_sum_ERA5_LAND",
    "potential_evaporation_sum",
    "potential_evaporation",
    "pet_era5",
    "pev",
    "pet",
)
_PET_FAO_CANDIDATES: Tuple[str, ...] = (
    "era5land_potential_evaporation_FAO_PENMAN_MONTEITH",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    "pet_fao",
    "fao_pet",
)
_D2M_CANDIDATES: Tuple[str, ...] = (
    "era5land_dewpoint_temperature_2m",
    "dewpoint_temperature_2m_mean",
    "dewpoint_temperature_2m",
    "2m_dewpoint_temperature",
    "d2m",
)
_SP_CANDIDATES: Tuple[str, ...] = (
    "era5land_surface_pressure",
    "surface_pressure_mean",
    "surface_pressure",
    "sp",
)
_SSR_CANDIDATES: Tuple[str, ...] = (
    "era5land_surface_net_solar_radiation",
    "surface_net_solar_radiation_mean",
    "surface_net_solar_radiation",
    "ssr",
)
_STR_CANDIDATES: Tuple[str, ...] = (
    "era5land_surface_net_thermal_radiation",
    "surface_net_thermal_radiation_mean",
    "surface_net_thermal_radiation",
    "str",
)
_U10_CANDIDATES: Tuple[str, ...] = (
    "era5land_u_component_of_wind_10m",
    "u_component_of_wind_10m_mean",
    "u_component_of_wind_10m",
    "10m_u_component_of_wind",
    "u10",
)
_V10_CANDIDATES: Tuple[str, ...] = (
    "era5land_v_component_of_wind_10m",
    "v_component_of_wind_10m_mean",
    "v_component_of_wind_10m",
    "10m_v_component_of_wind",
    "v10",
)


def compute_climate_metrics_from_dataframe(
    timeseries_df: pd.DataFrame,
    baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
    units_map: Optional[Mapping[str, Optional[str]]] = None,
    include_fao_pm: bool = True,
) -> Dict[str, float]:
  """Computes the 18 Caravan climate indices from a daily basin DataFrame.

  Args:
    timeseries_df: Daily DataFrame indexed by ``pd.DatetimeIndex``.
    baseline_years: Optional ``(start_year, end_year)`` filter. Restricted to
      that inclusive window; if no rows fall within ``baseline_years``, all
      18 climate metrics are returned as ``NaN``.
    units_map: Optional mapping from column name to CF unit string (e.g. ``'m'``,
      ``'K'``, ``'Pa'``).
    include_fao_pm: Whether to compute FAO-56 Penman-Monteith PET indices.

  Returns:
    Dictionary of 18 Caravan climate metrics.
  """
  if timeseries_df is None or timeseries_df.empty:
    raise ValueError("timeseries_df must be a non-empty DataFrame.")

  cols = timeseries_df.columns
  p_col = next((c for c in _PRECIP_CANDIDATES if c in cols), None)
  t_col = next((c for c in _TEMP_CANDIDATES if c in cols), None)
  if not p_col or not t_col:
    raise ValueError(
        "timeseries_df is missing required precipitation/temperature columns "
        f"(found {list(cols)})."
    )

  ts_df = timeseries_df
  if not isinstance(ts_df.index, pd.DatetimeIndex):
    ts_df = ts_df.copy()
    ts_df.index = pd.to_datetime(ts_df.index)

  if baseline_years is not None:
    sliced = ts_df.loc[
        (ts_df.index.year >= baseline_years[0])
        & (ts_df.index.year <= baseline_years[1])
    ]
    if sliced.empty:
      return dict(NAN_CLIMATE_METRICS)
    ts_df = sliced

  u_map = units_map or {}
  pet_era5_col = next((c for c in _PET_ERA5_CANDIDATES if c in cols), None)
  pet_fao_col = (
      next((c for c in _PET_FAO_CANDIDATES if c in cols), None)
      if include_fao_pm
      else None
  )

  if ts_df[p_col].isna().all() or ts_df[t_col].isna().all():
    return dict(NAN_CLIMATE_METRICS)

  p_series = pd.Series(
      depth_to_mm(ts_df[p_col].to_numpy(dtype=float), u_map.get(p_col), p_col),
      index=ts_df.index,
  )
  t_series = pd.Series(
      temp_to_celsius(
          ts_df[t_col].to_numpy(dtype=float), u_map.get(t_col), t_col
      ),
      index=ts_df.index,
  )

  pet_era5_series: Optional[pd.Series] = None
  if pet_era5_col and not ts_df[pet_era5_col].isna().all():
    era5_mm = depth_to_mm(
        ts_df[pet_era5_col].to_numpy(dtype=float),
        u_map.get(pet_era5_col),
        pet_era5_col,
    )
    pet_era5_series = pd.Series(
        normalize_era5_pet_sign(era5_mm),
        index=ts_df.index,
    )

  pet_fao_series: Optional[pd.Series] = None
  if include_fao_pm:
    if pet_fao_col and not ts_df[pet_fao_col].isna().all():
      fao_mm = depth_to_mm(
          ts_df[pet_fao_col].to_numpy(dtype=float),
          u_map.get(pet_fao_col),
          pet_fao_col,
      )
      fao_clipped = np.where(np.isfinite(fao_mm), np.maximum(fao_mm, 0.0), np.nan)
      pet_fao_series = pd.Series(fao_clipped, index=ts_df.index)
    else:
      d2m_col = next((c for c in _D2M_CANDIDATES if c in cols), None)
      sp_col = next((c for c in _SP_CANDIDATES if c in cols), None)
      ssr_col = next((c for c in _SSR_CANDIDATES if c in cols), None)
      str_col = next((c for c in _STR_CANDIDATES if c in cols), None)
      u10_col = next((c for c in _U10_CANDIDATES if c in cols), None)
      v10_col = next((c for c in _V10_CANDIDATES if c in cols), None)
      if all((d2m_col, sp_col, ssr_col, str_col, u10_col, v10_col)):
        aux_cols = [d2m_col, sp_col, ssr_col, str_col, u10_col, v10_col]
        if not any(ts_df[c].isna().all() for c in aux_cols):
          ssr_u = u_map.get(ssr_col)
          str_u = u_map.get(str_col)
          if (
              ssr_u is not None
              and str_u is not None
              and str(ssr_u).strip().lower() != str(str_u).strip().lower()
          ):
            raise ValueError(
                f"Mismatched radiation units between {ssr_col!r} ({ssr_u!r}) "
                f"and {str_col!r} ({str_u!r})."
            )
          rad_units = ssr_u or str_u or "W/m^2"
          pet_fao_series = calculate_fao_pm_pet(
              surface_pressure_kpa=pd.Series(
                  pressure_to_kpa(
                      ts_df[sp_col].to_numpy(dtype=float),
                      u_map.get(sp_col),
                      sp_col,
                  ),
                  index=ts_df.index,
              ),
              temperature_2m_c=t_series,
              dewpoint_temperature_2m_c=pd.Series(
                  temp_to_celsius(
                      ts_df[d2m_col].to_numpy(dtype=float),
                      u_map.get(d2m_col),
                      d2m_col,
                  ),
                  index=ts_df.index,
              ),
              u_component_of_wind_10m=ts_df[u10_col],
              v_component_of_wind_10m=ts_df[v10_col],
              surface_net_solar_radiation_mean=ts_df[ssr_col],
              surface_net_thermal_radiation_mean=ts_df[str_col],
              radiation_units=rad_units,
          )

  return compute_caravan_climate_metrics(
      precipitation=p_series,
      temperature=t_series,
      pet_era5=pet_era5_series,
      pet_fao=pet_fao_series,
  )


class ERA5TimeseriesZarrLoader:
  """Loads pre-extracted basin-level ERA5-Land daily time series from Zarr store(s)."""

  def __init__(
      self,
      zarr_uris: Union[str, Path, Sequence[Union[str, Path]]],
  ):
    if isinstance(zarr_uris, (str, Path)):
      raw_list = [
          s.strip() for s in str(zarr_uris).split(",") if s.strip()
      ]
    else:
      raw_list = [str(u).strip() for u in zarr_uris if str(u).strip()]

    if not raw_list:
      raise ValueError("At least one ERA5 timeseries Zarr URI or path must be provided.")

    self.zarr_uris: List[str] = raw_list
    self._stores: List[xr.Dataset] = []
    self._store_basin_dims: List[str] = []
    self._store_time_dims: List[str] = []
    self._store_dates: List[pd.DatetimeIndex] = []
    self._basin_lookup: Dict[str, Tuple[int, int]] = {}
    self._metrics_cache: Dict[Tuple[str, Optional[Tuple[int, int]]], Dict[str, float]] = {}

    for store_idx, uri in enumerate(self.zarr_uris):
      ds = open_gridded_archive(uri)
      basin_dim = next(
          (d for d in ("basin", "gauge_id", "catchment_id") if d in ds.coords or d in ds.dims),
          None,
      )
      time_dim = next(
          (d for d in ("date", "time") if d in ds.coords or d in ds.dims),
          None,
      )
      if basin_dim is None or time_dim is None:
        raise ValueError(
            f"ERA5 timeseries Zarr store at {uri!r} must contain a basin coordinate "
            "('basin' or 'gauge_id') and a time coordinate ('date' or 'time'); "
            f"found dims={list(ds.dims)}, coords={list(ds.coords)}."
        )
      basin_vals = [str(b) for b in ds[basin_dim].values]
      dates = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values).floor("D"))
      self._stores.append(ds)
      self._store_basin_dims.append(basin_dim)
      self._store_time_dims.append(time_dim)
      self._store_dates.append(dates)
      for row_idx, b_id in enumerate(basin_vals):
        if b_id in self._basin_lookup:
          prev_store_idx, _ = self._basin_lookup[b_id]
          raise ValueError(
              f"Duplicate basin ID {b_id!r} found across ERA5 timeseries Zarr stores "
              f"({self.zarr_uris[prev_store_idx]!r} and {uri!r})."
          )
        self._basin_lookup[b_id] = (store_idx, row_idx)

  def has_basin(self, basin_id: str) -> bool:
    """Returns True if ``basin_id`` is present in any configured timeseries Zarr store."""
    return str(basin_id) in self._basin_lookup

  def get_metrics_for_basin(
      self,
      basin_id: str,
      baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
  ) -> Dict[str, float]:
    """Computes the 18 Caravan climate metrics for a single basin from pre-extracted timeseries."""
    b_str = str(basin_id)
    cache_key = (b_str, baseline_years)
    if cache_key in self._metrics_cache:
      return dict(self._metrics_cache[cache_key])
    batch = self.get_metrics_for_basins_batch([b_str], baseline_years=baseline_years)
    return dict(batch[b_str])

  def get_metrics_for_basins_batch(
      self,
      basin_ids: Sequence[str],
      baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
      chunk_basins: int = 512,
  ) -> Dict[str, Dict[str, float]]:
    """Computes the 18 Caravan climate metrics for a batch of basins via bulk Zarr reads."""
    results: Dict[str, Dict[str, float]] = {}
    by_store: Dict[int, List[Tuple[str, int]]] = {}

    for raw_id in basin_ids:
      b_id = str(raw_id)
      cache_key = (b_id, baseline_years)
      if cache_key in self._metrics_cache:
        results[b_id] = dict(self._metrics_cache[cache_key])
        continue
      if b_id not in self._basin_lookup:
        raise KeyError(
            f"Basin {b_id!r} not found in any configured ERA5 timeseries Zarr store "
            f"({self.zarr_uris})."
        )
      s_idx, r_idx = self._basin_lookup[b_id]
      by_store.setdefault(s_idx, []).append((b_id, r_idx))

    for s_idx, items in by_store.items():
      ds = self._stores[s_idx]
      basin_dim = self._store_basin_dims[s_idx]
      time_dim = self._store_time_dims[s_idx]
      full_dates = self._store_dates[s_idx]
      uri = self.zarr_uris[s_idx]

      if baseline_years is not None:
        start_y, end_y = baseline_years
        t_indices = np.where(
            (full_dates.year >= start_y) & (full_dates.year <= end_y)
        )[0]
        if len(t_indices) == 0:
          logger.warning(
              "ERA5 timeseries Zarr at %s has no dates in baseline_years=%s; returning NaN.",
              uri,
              baseline_years,
          )
          for b_id, _ in items:
            results[b_id] = dict(NAN_CLIMATE_METRICS)
            self._metrics_cache[(b_id, baseline_years)] = dict(NAN_CLIMATE_METRICS)
          continue
        t_start = int(t_indices[0])
        t_end = int(t_indices[-1]) + 1
        rel_t = t_indices - t_start
        ds_time = ds.isel({time_dim: slice(t_start, t_end)})
        date_index = full_dates[t_indices]
      else:
        rel_t = np.arange(len(full_dates), dtype=int)
        ds_time = ds
        date_index = full_dates

      var_names = list(ds_time.data_vars)
      p_col = next((c for c in _PRECIP_CANDIDATES if c in var_names), None)
      t_col = next((c for c in _TEMP_CANDIDATES if c in var_names), None)
      if not p_col or not t_col:
        raise ValueError(
            f"Required ERA5 climate variables (precip/temp) not found in {uri}. "
            f"Available variables: {var_names}"
        )
      pet_era5_col = next((c for c in _PET_ERA5_CANDIDATES if c in var_names), None)
      pet_fao_col = next((c for c in _PET_FAO_CANDIDATES if c in var_names), None)

      cols_to_load = [p_col, t_col]
      if pet_era5_col:
        cols_to_load.append(pet_era5_col)
      if pet_fao_col:
        cols_to_load.append(pet_fao_col)

      aux_cols = [
          next((c for c in cand if c in var_names), None)
          for cand in (
              _D2M_CANDIDATES,
              _SP_CANDIDATES,
              _SSR_CANDIDATES,
              _STR_CANDIDATES,
              _U10_CANDIDATES,
              _V10_CANDIDATES,
          )
      ]
      has_all_aux = all(c is not None for c in aux_cols)
      if pet_fao_col is None and has_all_aux:
        cols_to_load.extend([c for c in aux_cols if c is not None])

      units_map = {
          c: dict(ds_time[c].attrs).get("units") for c in cols_to_load
      }

      items_sorted = sorted(items, key=lambda x: x[1])
      step = max(1, int(chunk_basins))
      for offset in range(0, len(items_sorted), step):
        batch_items = items_sorted[offset : offset + step]
        row_indices = [r_idx for _, r_idx in batch_items]
        sub_ds = ds_time[cols_to_load].isel({basin_dim: row_indices})
        with dask.config.set(scheduler="threads"):
          computed = sub_ds.compute()

        # If pet_fao_col is present but all-NaN for some basins in this chunk, load aux cols if available
        loaded_arrays: Dict[str, np.ndarray] = {}
        for c in cols_to_load:
          da = computed[c].transpose(basin_dim, time_dim)
          loaded_arrays[c] = np.asarray(da.values, dtype=np.float64)[:, rel_t]

        if (
            pet_fao_col is not None
            and has_all_aux
            and np.any(np.all(np.isnan(loaded_arrays[pet_fao_col]), axis=1))
        ):
          aux_only = [c for c in aux_cols if c is not None and c not in loaded_arrays]
          if aux_only:
            sub_aux = ds_time[aux_only].isel({basin_dim: row_indices})
            with dask.config.set(scheduler="threads"):
              comp_aux = sub_aux.compute()
            for c in aux_only:
              units_map[c] = dict(ds_time[c].attrs).get("units")
              da = comp_aux[c].transpose(basin_dim, time_dim)
              loaded_arrays[c] = np.asarray(da.values, dtype=np.float64)[:, rel_t]

        for i, (b_id, _) in enumerate(batch_items):
          df_basin = pd.DataFrame(
              {c: loaded_arrays[c][i] for c in loaded_arrays},
              index=date_index,
          )
          metrics = compute_climate_metrics_from_dataframe(
              df_basin,
              baseline_years=None,
              units_map=units_map,
              include_fao_pm=True,
          )
          results[b_id] = metrics
          self._metrics_cache[(b_id, baseline_years)] = dict(metrics)

    return results


class ERA5ClimateLoader:
  """Loads and aggregates Level 12 precomputed ERA5 climate indices from a specified directory or GCS."""

  def __init__(
      self,
      cache_dir: Optional[Union[str, Path]] = None,
      gcs_source_uri: Optional[str] = None,
      no_download: bool = False,
  ):
    self.no_download = bool(no_download)
    if cache_dir is not None and str(cache_dir).startswith(("gs://", "gcs://")):
      if not gcs_source_uri:
        gcs_source_uri = str(cache_dir)
      cache_dir = None

    self.gcs_source_uri = gcs_source_uri.rstrip("/") if gcs_source_uri else None

    if self.no_download or (cache_dir is None and self.gcs_source_uri is not None):
      self.no_download = True
      self.cache_dir = Path(cache_dir) if cache_dir else None
      if self.gcs_source_uri is None and (self.cache_dir is None or not self.cache_dir.exists()):
        raise ValueError(
            "gcs_source_uri (or an existing cache_dir) must be explicitly provided when no_download=True."
        )
    else:
      if not cache_dir:
        raise ValueError("cache_dir must be explicitly provided.")
      self.cache_dir = Path(cache_dir)
      self.cache_dir.mkdir(parents=True, exist_ok=True)

    self.loaded_continents: Set[str] = set()
    self.records: Dict[int, Dict[str, Any]] = {}
    # Guards ``loaded_continents``/``records`` when one loader is shared across threads.
    self._load_lock = threading.RLock()

  def _download_from_gcs(self, continent_code: str, target_file: Path) -> None:
    """Downloads a continental climate indices file from GCS to target_file."""
    if not self.gcs_source_uri:
      raise FileNotFoundError(
          f"Climate indices file not found at {target_file} and no gcs_source_uri was provided."
      )
    from multimet.utils.gcs import download_file_from_gcs

    gcs_src = f"{self.gcs_source_uri}/{continent_code}_climate_indices.txt"
    download_file_from_gcs(source_uri=gcs_src, dest_path=target_file, timeout=120)

  def _ensure_file_on_disk(self, continent_code: str) -> Path:
    if self.cache_dir is None:
      raise ValueError("cache_dir is not configured for local disk storage.")
    txt_path = self.cache_dir / f"{continent_code}_climate_indices.txt"
    if txt_path.exists() and txt_path.stat().st_size > 0:
      return txt_path

    if not self.gcs_source_uri:
      raise FileNotFoundError(
          f"Continental climate indices file not found at {txt_path}. "
          "Provide a directory containing the climate index files or specify --gcs-era5-climate-uri."
      )

    logger.info(
        "Downloading ERA5 climate indices for '%s' from %s...",
        continent_code,
        self.gcs_source_uri,
    )
    self._download_from_gcs(continent_code, txt_path)
    return txt_path

  def _parse_climate_lines(self, lines, continent_code: str) -> None:
    count = 0
    for line in lines:
      line = line.strip()
      if not line:
        continue
      item = json.loads(line)
      gid = str(item.get("gauge_id", ""))
      if gid.startswith("hybas_"):
        hid = int(gid.split("_")[1])
        self.records[hid] = item
        count += 1
      elif "HYBAS_ID" in item and item["HYBAS_ID"] is not None:
        hid = int(item["HYBAS_ID"])
        self.records[hid] = item
        count += 1
    self.loaded_continents.add(continent_code)
    logger.debug(
        "Loaded %d Level 12 climate records for continent '%s'", count, continent_code
    )

  def _stream_continent_from_gcs(self, continent_code: str) -> None:
    """Streams a continental climate indices file directly from GCS into memory."""
    if not self.gcs_source_uri:
      raise FileNotFoundError(
          f"Cannot stream climate indices for '{continent_code}': no gcs_source_uri was provided."
      )
    from multimet.utils.gcs import read_bytes_from_gcs

    gcs_src = f"{self.gcs_source_uri}/{continent_code}_climate_indices.txt"
    logger.info(
        "Streaming ERA5 climate indices for '%s' in memory from %s...",
        continent_code,
        gcs_src,
    )
    raw_text = read_bytes_from_gcs(gcs_src).decode("utf-8")
    self._parse_climate_lines(raw_text.splitlines(), continent_code)

  def ensure_continent(self, continent_code: str) -> None:
    """Ensures continental climate index records are loaded in memory (thread-safe)."""
    if continent_code in self.loaded_continents:
      return

    with self._load_lock:
      if continent_code in self.loaded_continents:
        return

      if self.no_download:
        if self.cache_dir is not None:
          local_txt = self.cache_dir / f"{continent_code}_climate_indices.txt"
          if local_txt.exists() and local_txt.stat().st_size > 0:
            with open(local_txt, "r", encoding="utf-8") as f:
              self._parse_climate_lines(f, continent_code)
            return
        self._stream_continent_from_gcs(continent_code)
        return

      txt_path = self._ensure_file_on_disk(continent_code)
      with open(txt_path, "r", encoding="utf-8") as f:
        self._parse_climate_lines(f, continent_code)

  @staticmethod
  def _continents_for(hybas_ids: Sequence[int]) -> Set[str]:
    """Maps HYBAS_IDs to HydroBASINS continent codes via their leading digit."""
    needed_continents: Set[str] = set()
    for hid in hybas_ids:
      first_digit = int(str(int(hid))[0])
      if first_digit not in CONTINENT_MAP:
        raise ValueError(
            f"Unrecognized continent prefix {first_digit} in HYBAS_ID {hid}."
        )
      needed_continents.add(CONTINENT_MAP[first_digit])
    return needed_continents

  def get_indices_table(self, hybas_ids: Sequence[int]) -> pd.DataFrame:
    """Returns the Level 12 Caravan climate indices of the given sub-basins as a table.

    Args:
      hybas_ids: HydroBASINS Level 12 identifiers.

    Returns:
      DataFrame indexed by ``HYBAS_ID`` (int64, in the requested order) with one
      float column per entry of ``CARAVAN_CLIMATE_COLUMNS``. Alias columns
      (``*_FAO_PM``) are filled from their canonical counterparts per
      :data:`CARAVAN_CLIMATE_ALIASES`; sub-basins or columns absent from the
      precomputed tables are NaN. No values are rescaled or rounded.
    """
    ids = [int(h) for h in hybas_ids]
    for continent in sorted(self._continents_for(ids)):
      self.ensure_continent(continent)

    rows = [expand_caravan_climate_aliases(self.records.get(hid, {})) for hid in ids]
    table = pd.DataFrame(rows, columns=list(CARAVAN_CLIMATE_COLUMNS), dtype=float)
    table.index = pd.Index(ids, name="HYBAS_ID", dtype="int64")
    return table

  def get_indices_for_subbasins(
      self, hybas_ids: List[int], weights: List[float]
  ) -> Dict[str, float]:
    """Calculates area-weighted average ERA5 climate indices for a set of Level 12 sub-basins."""
    for c in sorted(self._continents_for(hybas_ids)):
      self.ensure_continent(c)

    total_requested_w = sum(float(w) for w in weights)
    valid_weights = []
    valid_records = []
    for hid, w in zip(hybas_ids, weights, strict=True):
      if hid in self.records:
        valid_records.append(self.records[hid])
        valid_weights.append(float(w))

    if not valid_records or total_requested_w <= 0 or sum(valid_weights) == 0:
      return expand_caravan_climate_aliases({})

    w_arr = np.array(valid_weights, dtype=float)
    norm_w = w_arr / total_requested_w

    raw_res: Dict[str, float] = {}
    for k in CARAVAN_CLIMATE_STORED_KEYS:
      vals = np.array([r.get(k, np.nan) for r in valid_records], dtype=float)
      finite = np.isfinite(vals)
      valid_cov = float(np.sum(norm_w[finite])) if np.any(finite) else 0.0
      if valid_cov >= MIN_VALID_COVERAGE_FRACTION:
        raw_res[k] = float(
            np.sum(vals[finite] * norm_w[finite]) / valid_cov
        )
      else:
        raw_res[k] = np.nan

    return expand_caravan_climate_aliases(raw_res)


class ERA5GriddedExtractor:
  """Recalculates Caravan climate metrics directly from a gridded ERA5 Zarr dataset.

  Uses :class:`multimet.utils.zonal.ZonalWeightCalculator` for exact spherical
  polygon-to-grid intersection weights and :func:`compute_climate_metrics_from_dataframe`
  for the Caravan climate index equations.
  """

  def __init__(
      self,
      zarr_uri: Union[str, Path],
  ):
    """Initializes the ERA5GriddedExtractor.

    Args:
      zarr_uri: GCS URI or local path to the gridded daily surface ERA5 Zarr store.
    """
    if not zarr_uri:
      raise ValueError("zarr_uri must be explicitly provided.")
    self.zarr_uri = str(zarr_uri)
    self._ds = None
    self._lats: Optional[np.ndarray] = None
    self._lons: Optional[np.ndarray] = None
    self._dlat: float = 0.1
    self._dlon: float = 0.1

  def _open_dataset(self):
    if self._ds is not None:
      return self._ds

    logger.info("Opening gridded ERA5 Zarr store at: %s", self.zarr_uri)
    self._ds = zarr.open(self.zarr_uri, mode="r")
    lat_keys = [k for k in ["latitude", "lat"] if k in self._ds]
    lon_keys = [k for k in ["longitude", "lon"] if k in self._ds]
    if not lat_keys or not lon_keys:
      raise KeyError(f"Latitude/Longitude coordinates not found in {self.zarr_uri}")

    self._lats = np.asarray(self._ds[lat_keys[0]][:], dtype=np.float64)
    self._lons = np.asarray(self._ds[lon_keys[0]][:], dtype=np.float64)
    self._dlat = abs(float(self._lats[1] - self._lats[0])) if len(self._lats) > 1 else 0.1
    self._dlon = abs(float(self._lons[1] - self._lons[0])) if len(self._lons) > 1 else 0.1
    return self._ds

  def compute_zonal_weights(
      self, polygon: Any
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes area-intersection weights between a polygon and grid cells via ZonalWeightCalculator."""
    self._open_dataset()
    if not hasattr(polygon, "bounds"):
      polygon = shapely.geometry.shape(polygon)
    calc = ZonalWeightCalculator(
        self._lats, self._lons, cell_res_lat=self._dlat, cell_res_lon=self._dlon
    )
    lat_idx, lon_idx, weights = calc.compute_weights("_poly", polygon)
    return (
        lat_idx.astype(int),
        lon_idx.astype(int),
        weights.astype(np.float32),
    )

  @staticmethod
  def _parse_time_coordinate(time_arr: np.ndarray, time_attrs: Dict[str, Any], zarr_uri: str) -> pd.DatetimeIndex:
    """Parses CF-compliant time coordinates without guessing units or epochs."""
    from multimet.utils.storage import parse_cf_time_coordinate

    return parse_cf_time_coordinate(time_arr, time_attrs, store_label=zarr_uri)

  def extract_climate_metrics_for_polygon(
      self,
      polygon: Any,
      baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
      include_fao_pm: bool = True,
  ) -> Dict[str, float]:
    """Extracts daily gridded series and calculates the Caravan climate metrics for a single polygon."""
    batch_res = self.extract_climate_metrics_for_polygons_batch(
        [(polygon, "_single_catchment")],
        baseline_years=baseline_years,
        include_fao_pm=include_fao_pm,
    )
    return batch_res["_single_catchment"]

  def extract_climate_metrics_for_polygons_batch(
      self,
      polygons: List[Tuple[Any, str]],
      baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
      chunk_days: int = 365,
      include_fao_pm: bool = True,
  ) -> Dict[str, Dict[str, float]]:
    """Extracts Caravan climate metrics for a batch of polygons in a single pass over Zarr time chunks.

    Computes spatial grid-cell weights (lat_idx, lon_idx, weights) for all polygons
    upfront using :class:`ZonalWeightCalculator` and streams each daily Zarr slice
    once across all polygons in the batch.

    Args:
      polygons: List of (polygon, catchment_id) tuples.
      baseline_years: Optional (start_year, end_year) climate baseline.
      chunk_days: Number of daily time steps to read per Zarr slice.
      include_fao_pm: If False, skips reading/computing FAO-56 Penman-Monteith
        PET and only extracts precipitation, temperature, and native ERA5-Land PEV.

    Returns:
      Dictionary mapping each catchment_id to its 18 Caravan climate metrics.
    """
    if not polygons:
      return {}

    ds = self._open_dataset()

    p_name = next((v for v in _PRECIP_CANDIDATES if v in ds), None)
    t_name = next((v for v in _TEMP_CANDIDATES if v in ds), None)
    pet_fao_name = (
        next((v for v in _PET_FAO_CANDIDATES if v in ds), None)
        if include_fao_pm
        else None
    )
    pet_era5_name = next((v for v in _PET_ERA5_CANDIDATES if v in ds), None)

    if not p_name or not t_name:
      raise ValueError(
          f"Required climate variables (precip/temp) not found in {self.zarr_uri}. "
          f"Available keys: {list(ds.keys())[:10]}"
      )

    d2m_name = None
    sp_name = None
    ssr_name = None
    str_name = None
    u10_name = None
    v10_name = None
    can_compute_fao_pet = False
    if include_fao_pm:
      d2m_name = next((v for v in _D2M_CANDIDATES if v in ds), None)
      sp_name = next((v for v in _SP_CANDIDATES if v in ds), None)
      ssr_name = next((v for v in _SSR_CANDIDATES if v in ds), None)
      str_name = next((v for v in _STR_CANDIDATES if v in ds), None)
      u10_name = next((v for v in _U10_CANDIDATES if v in ds), None)
      v10_name = next((v for v in _V10_CANDIDATES if v in ds), None)
      can_compute_fao_pet = all(
          (d2m_name, sp_name, ssr_name, str_name, u10_name, v10_name)
      )

    time_keys = [k for k in ["time", "date"] if k in ds]
    if not time_keys:
      raise KeyError(f"Time coordinate ('time' or 'date') not found in {self.zarr_uri}")
    time_arr = np.asarray(ds[time_keys[0]][:])
    time_attrs = dict(ds[time_keys[0]].attrs)
    full_date_index = self._parse_time_coordinate(time_arr, time_attrs, self.zarr_uri)

    if baseline_years is not None:
      start_y, end_y = baseline_years
      t_indices = np.where(
          (full_date_index.year >= start_y) & (full_date_index.year <= end_y)
      )[0]
      if len(t_indices) == 0:
        logger.warning(
            "Gridded ERA5 archive at %s contains no records in baseline_years=%s; returning NaN.",
            self.zarr_uri,
            baseline_years,
        )
        return {cid: dict(NAN_CLIMATE_METRICS) for _, cid in polygons}
    else:
      t_indices = np.arange(len(full_date_index), dtype=int)

    date_index = full_date_index[t_indices]
    t_start = int(t_indices[0])
    t_end = int(t_indices[-1]) + 1
    rel_t_indices = t_indices - t_start

    # 1. Compute spatial grid-cell weights for all polygons upfront
    results: Dict[str, Dict[str, float]] = {}
    valid_specs = []
    for poly, cid in polygons:
      lat_idx, lon_idx, weights = self.compute_zonal_weights(poly)
      if len(weights) == 0:
        logger.warning(
            "Catchment polygon '%s' does not intersect gridded ERA5 coordinate domain at %s; returning NaN.",
            cid,
            self.zarr_uri,
        )
        results[cid] = dict(NAN_CLIMATE_METRICS)
      else:
        valid_specs.append((cid, lat_idx, lon_idx, weights.reshape(1, -1)))

    if not valid_specs:
      return results

    min_lat_i = min(int(np.min(lat_idx)) for _, lat_idx, _, _ in valid_specs)
    max_lat_i = max(int(np.max(lat_idx)) for _, lat_idx, _, _ in valid_specs) + 1
    min_lon_i = min(int(np.min(lon_idx)) for _, _, lon_idx, _ in valid_specs)
    max_lon_i = max(int(np.max(lon_idx)) for _, _, lon_idx, _ in valid_specs) + 1

    num_valid = len(valid_specs)
    total_span = t_end - t_start
    p_daily_raw = np.full((num_valid, total_span), np.nan, dtype=np.float64)
    t_daily_raw = np.full((num_valid, total_span), np.nan, dtype=np.float64)
    pet_fao_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if pet_fao_name
        else None
    )
    pet_era5_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if pet_era5_name
        else None
    )
    d2m_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if can_compute_fao_pet
        else None
    )
    sp_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if can_compute_fao_pet
        else None
    )
    ssr_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if can_compute_fao_pet
        else None
    )
    str_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if can_compute_fao_pet
        else None
    )
    u10_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if can_compute_fao_pet
        else None
    )
    v10_daily_raw = (
        np.full((num_valid, total_span), np.nan, dtype=np.float64)
        if can_compute_fao_pet
        else None
    )

    def _weighted_nanmean(cells: np.ndarray, w_matrix: np.ndarray) -> np.ndarray:
      total_w = float(np.sum(w_matrix))
      out = np.full((cells.shape[0],), np.nan, dtype=np.float64)
      if total_w <= 0.0:
        return out
      valid_w = np.where(np.isnan(cells), 0.0, w_matrix)
      w_sum = np.sum(valid_w, axis=1)
      num = np.nansum(cells * w_matrix, axis=1)
      valid_enough = (w_sum > 0.0) & (
          (w_sum / total_w) >= MIN_VALID_COVERAGE_FRACTION
      )
      return np.divide(num, w_sum, out=out, where=valid_enough)

    # 2. Stream through time chunks once across all polygons in the batch
    step = max(1, int(chunk_days))
    for offset in range(0, total_span, step):
      b_start = t_start + offset
      b_end = min(t_end, b_start + step)
      rel_slice = slice(offset, offset + (b_end - b_start))

      p_block = ds[p_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
      t_block = ds[t_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
      pet_fao_block = (
          ds[pet_fao_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if pet_fao_name
          else None
      )
      pet_era5_block = (
          ds[pet_era5_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if pet_era5_name
          else None
      )

      need_fao_aux_block = can_compute_fao_pet and (
          pet_fao_block is None
          or any(
              bool(
                  np.all(
                      np.isnan(
                          pet_fao_block[
                              :, lat_idx - min_lat_i, lon_idx - min_lon_i
                          ]
                      )
                  )
              )
              for _, lat_idx, lon_idx, _ in valid_specs
          )
      )
      d2m_block = (
          ds[d2m_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if need_fao_aux_block and d2m_name
          else None
      )
      sp_block = (
          ds[sp_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if need_fao_aux_block and sp_name
          else None
      )
      ssr_block = (
          ds[ssr_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if need_fao_aux_block and ssr_name
          else None
      )
      str_block = (
          ds[str_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if need_fao_aux_block and str_name
          else None
      )
      u10_block = (
          ds[u10_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if need_fao_aux_block and u10_name
          else None
      )
      v10_block = (
          ds[v10_name][b_start:b_end, min_lat_i:max_lat_i, min_lon_i:max_lon_i]
          if need_fao_aux_block and v10_name
          else None
      )

      for i, (_, lat_idx, lon_idx, w_matrix) in enumerate(valid_specs):
        r_lat = lat_idx - min_lat_i
        r_lon = lon_idx - min_lon_i
        p_daily_raw[i, rel_slice] = _weighted_nanmean(p_block[:, r_lat, r_lon], w_matrix)
        t_daily_raw[i, rel_slice] = _weighted_nanmean(t_block[:, r_lat, r_lon], w_matrix)
        if pet_fao_block is not None and pet_fao_daily_raw is not None:
          pet_fao_daily_raw[i, rel_slice] = _weighted_nanmean(
              pet_fao_block[:, r_lat, r_lon], w_matrix
          )
        if pet_era5_block is not None and pet_era5_daily_raw is not None:
          pet_era5_daily_raw[i, rel_slice] = _weighted_nanmean(
              pet_era5_block[:, r_lat, r_lon], w_matrix
          )
        if need_fao_aux_block:
          d2m_daily_raw[i, rel_slice] = _weighted_nanmean(
              d2m_block[:, r_lat, r_lon], w_matrix
          )
          sp_daily_raw[i, rel_slice] = _weighted_nanmean(
              sp_block[:, r_lat, r_lon], w_matrix
          )
          ssr_daily_raw[i, rel_slice] = _weighted_nanmean(
              ssr_block[:, r_lat, r_lon], w_matrix
          )
          str_daily_raw[i, rel_slice] = _weighted_nanmean(
              str_block[:, r_lat, r_lon], w_matrix
          )
          u10_daily_raw[i, rel_slice] = _weighted_nanmean(
              u10_block[:, r_lat, r_lon], w_matrix
          )
          v10_daily_raw[i, rel_slice] = _weighted_nanmean(
              v10_block[:, r_lat, r_lon], w_matrix
          )

    var_list = [
        v
        for v in (
            p_name,
            t_name,
            pet_fao_name,
            pet_era5_name,
            d2m_name,
            sp_name,
            ssr_name,
            str_name,
            u10_name,
            v10_name,
        )
        if v is not None
    ]
    units_map = {v: dict(ds[v].attrs).get("units") for v in var_list}

    if include_fao_pm and pet_fao_name is None and not can_compute_fao_pet:
      logger.warning(
          "No FAO-56 Penman-Monteith PET variable or full meteorological bands found in %s; "
          "unsuffixed and *_FAO_PM PET attributes will be NaN.",
          self.zarr_uri,
      )
    if pet_era5_name is None:
      logger.warning(
          "No native ERA5-Land potential evaporation variable found in %s; "
          "*_ERA5_LAND PET attributes will be NaN.",
          self.zarr_uri,
      )

    # 3. Compute Caravan climate indices for each polygon via shared DataFrame helper
    for i, (cid, _, _, _) in enumerate(valid_specs):
      p_vals = p_daily_raw[i, rel_t_indices]
      t_vals = t_daily_raw[i, rel_t_indices]
      if np.all(np.isnan(p_vals)) or np.all(np.isnan(t_vals)):
        logger.warning(
            "Gridded ERA5 archive at %s returned all NaNs for catchment '%s'.",
            self.zarr_uri,
            cid,
        )
        results[cid] = dict(NAN_CLIMATE_METRICS)
        continue

      cols_dict: Dict[str, np.ndarray] = {
          p_name: p_vals,
          t_name: t_vals,
      }
      if pet_fao_daily_raw is not None and pet_fao_name is not None:
        cols_dict[pet_fao_name] = pet_fao_daily_raw[i, rel_t_indices]
      if pet_era5_daily_raw is not None and pet_era5_name is not None:
        cols_dict[pet_era5_name] = pet_era5_daily_raw[i, rel_t_indices]
      if can_compute_fao_pet:
        cols_dict[d2m_name] = d2m_daily_raw[i, rel_t_indices]
        cols_dict[sp_name] = sp_daily_raw[i, rel_t_indices]
        cols_dict[ssr_name] = ssr_daily_raw[i, rel_t_indices]
        cols_dict[str_name] = str_daily_raw[i, rel_t_indices]
        cols_dict[u10_name] = u10_daily_raw[i, rel_t_indices]
        cols_dict[v10_name] = v10_daily_raw[i, rel_t_indices]

      df_poly = pd.DataFrame(cols_dict, index=date_index)
      results[cid] = compute_climate_metrics_from_dataframe(
          df_poly,
          baseline_years=None,
          units_map=units_map,
          include_fao_pm=include_fao_pm,
      )

    return results
