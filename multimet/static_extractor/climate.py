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
climate indices, Addor et al. (2017) extreme precipitation indices, and
area-weighted Level 12 continental ERA5 precomputed climate indices loading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import shapely.geometry
import zarr

from multimet.static_extractor.config import CARAVAN_CLIMATE_COLUMNS, CONTINENT_MAP
from multimet.utils.climate import (
    calculate_fao_pm_pet,
    calculate_knoben_moisture_and_seasonality,
    compute_caravan_climate_metrics,
    depth_to_mm,
    normalize_era5_pet_sign,
    pressure_to_kpa,
    temp_to_celsius,
)

__all__ = [
    "ERA5ClimateLoader",
    "ERA5GriddedExtractor",
    "calculate_fao_pm_pet",
    "calculate_knoben_moisture_and_seasonality",
    "compute_caravan_climate_metrics",
    "depth_to_mm",
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
      gid = item.get("gauge_id", "")
      if gid.startswith("hybas_"):
        hid = int(gid.split("_")[1])
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

    valid_weights = []
    valid_records = []
    for hid, w in zip(hybas_ids, weights, strict=True):
      if hid in self.records:
        valid_records.append(self.records[hid])
        valid_weights.append(float(w))

    if not valid_records or sum(valid_weights) == 0:
      return expand_caravan_climate_aliases({})

    tot_w = sum(valid_weights)
    norm_w = np.array(valid_weights) / tot_w

    raw_res: Dict[str, float] = {}
    for k in CARAVAN_CLIMATE_STORED_KEYS:
      vals = np.array([r.get(k, np.nan) for r in valid_records], dtype=float)
      raw_res[k] = float(np.sum(vals * norm_w))

    return expand_caravan_climate_aliases(raw_res)


class ERA5GriddedExtractor:
  """Recalculates Caravan climate metrics directly from a gridded ERA5 Zarr dataset."""

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
    """Computes area-intersection weights between a polygon and grid cells."""
    self._open_dataset()
    if not hasattr(polygon, "bounds"):
      polygon = shapely.geometry.shape(polygon)

    minx, miny, maxx, maxy = polygon.bounds

    lat_mask = (self._lats >= miny - self._dlat) & (self._lats <= maxy + self._dlat)
    lon_mask = (self._lons >= minx - self._dlon) & (self._lons <= maxx + self._dlon)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
      return (
          np.array([], dtype=int),
          np.array([], dtype=int),
          np.array([], dtype=np.float32),
      )

    lat_list = []
    lon_list = []
    w_list = []

    half_lat = self._dlat / 2.0
    half_lon = self._dlon / 2.0

    for li in lat_indices:
      lat_val = self._lats[li]
      cell_miny = lat_val - half_lat
      cell_maxy = lat_val + half_lat
      lat_cos = max(0.01, np.cos(np.radians(lat_val)))

      for lj in lon_indices:
        lon_val = self._lons[lj]
        cell_minx = lon_val - half_lon
        cell_maxx = lon_val + half_lon

        cell_box = shapely.geometry.box(cell_minx, cell_miny, cell_maxx, cell_maxy)
        if polygon.intersects(cell_box):
          inter = polygon.intersection(cell_box)
          inter_area = inter.area * lat_cos
          if inter_area > 0.0:
            lat_list.append(li)
            lon_list.append(lj)
            w_list.append(inter_area)

    if not w_list:
      return (
          np.array([], dtype=int),
          np.array([], dtype=int),
          np.array([], dtype=np.float32),
      )

    weights = np.array(w_list, dtype=np.float32)
    tot_w = np.sum(weights)
    if tot_w > 0:
      weights = weights / tot_w
    return np.array(lat_list, dtype=int), np.array(lon_list, dtype=int), weights

  @staticmethod
  def _depth_to_mm(series: np.ndarray, units: Optional[str], var_name: str) -> np.ndarray:
    """Converts a precipitation or PET series to mm/day using declared array units."""
    return depth_to_mm(series, units, var_name)

  @staticmethod
  def _temp_to_celsius(series: np.ndarray, units: Optional[str], var_name: str) -> np.ndarray:
    """Converts a temperature series to degrees Celsius using declared array units."""
    return temp_to_celsius(series, units, var_name)

  @staticmethod
  def _pressure_to_kpa(series: np.ndarray, units: Optional[str], var_name: str) -> np.ndarray:
    """Converts a surface pressure series to kPa using declared array units."""
    return pressure_to_kpa(series, units, var_name)

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
    upfront and streams each daily Zarr slice once across all polygons in the batch.

    Args:
      polygons: List of (polygon, catchment_id) tuples.
      baseline_years: Optional (start_year, end_year) climate baseline.
      chunk_days: Number of daily time steps to read per Zarr slice.
      include_fao_pm: If False, skips reading/computing FAO-56 Penman-Monteith
        PET and only extracts precipitation, temperature, and native ERA5-Land PEV.

    Returns:
      Dictionary mapping each catchment_id to its 18 Caravan climate metrics.
    """
    nan_result = {
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

    if not polygons:
      return {}

    ds = self._open_dataset()

    p_name = next(
        (
            v
            for v in [
                "era5land_total_precipitation",
                "total_precipitation_24hr",
                "total_precipitation",
                "tp",
            ]
            if v in ds
        ),
        None,
    )
    t_name = next(
        (
            v
            for v in [
                "era5land_temperature_2m",
                "2m_temperature",
                "temperature_2m",
                "temp",
                "t2m",
            ]
            if v in ds
        ),
        None,
    )
    pet_fao_name = (
        next(
            (
                v
                for v in [
                    "era5land_potential_evaporation_FAO_PENMAN_MONTEITH",
                    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
                ]
                if v in ds
            ),
            None,
        )
        if include_fao_pm
        else None
    )
    pet_era5_name = next(
        (
            v
            for v in [
                "era5land_potential_evaporation_DEPRECATED",
                "potential_evaporation_sum_ERA5_LAND",
                "potential_evaporation",
                "pev",
                "pet",
            ]
            if v in ds
        ),
        None,
    )

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
      d2m_name = next(
          (
              v
              for v in [
                  "era5land_dewpoint_temperature_2m",
                  "dewpoint_temperature_2m_mean",
                  "dewpoint_temperature_2m",
                  "2m_dewpoint_temperature",
                  "d2m",
              ]
              if v in ds
          ),
          None,
      )
      sp_name = next(
          (
              v
              for v in [
                  "era5land_surface_pressure",
                  "surface_pressure_mean",
                  "surface_pressure",
                  "sp",
              ]
              if v in ds
          ),
          None,
      )
      ssr_name = next(
          (
              v
              for v in [
                  "era5land_surface_net_solar_radiation",
                  "surface_net_solar_radiation_mean",
                  "surface_net_solar_radiation",
                  "ssr",
              ]
              if v in ds
          ),
          None,
      )
      str_name = next(
          (
              v
              for v in [
                  "era5land_surface_net_thermal_radiation",
                  "surface_net_thermal_radiation_mean",
                  "surface_net_thermal_radiation",
                  "str",
              ]
              if v in ds
          ),
          None,
      )
      u10_name = next(
          (
              v
              for v in [
                  "era5land_u_component_of_wind_10m",
                  "u_component_of_wind_10m_mean",
                  "u_component_of_wind_10m",
                  "10m_u_component_of_wind",
                  "u10",
              ]
              if v in ds
          ),
          None,
      )
      v10_name = next(
          (
              v
              for v in [
                  "era5land_v_component_of_wind_10m",
                  "v_component_of_wind_10m_mean",
                  "v_component_of_wind_10m",
                  "10m_v_component_of_wind",
                  "v10",
              ]
              if v in ds
          ),
          None,
      )
      can_compute_fao_pet = all(
          (d2m_name, sp_name, ssr_name, str_name, u10_name, v10_name)
      )

    # Parse time coordinate and restrict to baseline_years before reading any spatial chunks
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
        return {cid: dict(nan_result) for _, cid in polygons}
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
        results[cid] = dict(nan_result)
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
      valid_w = np.where(np.isnan(cells), 0.0, w_matrix)
      w_sum = np.sum(valid_w, axis=1)
      num = np.nansum(cells * w_matrix, axis=1)
      out = np.full(num.shape, np.nan, dtype=np.float64)
      return np.divide(num, w_sum, out=out, where=w_sum > 0)

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

    p_units = dict(ds[p_name].attrs).get("units")
    t_units = dict(ds[t_name].attrs).get("units")
    pet_fao_units = dict(ds[pet_fao_name].attrs).get("units") if pet_fao_name else None
    pet_era5_units = dict(ds[pet_era5_name].attrs).get("units") if pet_era5_name else None
    d2m_units = dict(ds[d2m_name].attrs).get("units") if d2m_name else None
    sp_units = dict(ds[sp_name].attrs).get("units") if sp_name else None
    ssr_units = dict(ds[ssr_name].attrs).get("units") if ssr_name else None
    str_units = dict(ds[str_name].attrs).get("units") if str_name else None

    if (
        ssr_units is not None
        and str_units is not None
        and str(ssr_units).strip().lower() != str(str_units).strip().lower()
    ):
      raise ValueError(
          f"Mismatched radiation units between {ssr_name!r} ({ssr_units!r}) "
          f"and {str_name!r} ({str_units!r}) in {self.zarr_uri}."
      )
    rad_units = ssr_units or str_units or "W/m^2"

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

    # 3. Compute Caravan climate indices for each polygon
    for i, (cid, _, _, _) in enumerate(valid_specs):
      p_vals = p_daily_raw[i, rel_t_indices]
      t_vals = t_daily_raw[i, rel_t_indices]
      if np.all(np.isnan(p_vals)) or np.all(np.isnan(t_vals)):
        logger.warning(
            "Gridded ERA5 archive at %s returned all NaNs for catchment '%s'.",
            self.zarr_uri,
            cid,
        )
        results[cid] = dict(nan_result)
        continue

      p_series = self._depth_to_mm(p_vals, p_units, p_name)
      t_series = self._temp_to_celsius(t_vals, t_units, t_name)

      pet_fao_s = None
      if include_fao_pm:
        if pet_fao_daily_raw is not None and pet_fao_name is not None:
          fao_vals = pet_fao_daily_raw[i, rel_t_indices]
          if not np.all(np.isnan(fao_vals)):
            pet_fao_s = pd.Series(
                self._depth_to_mm(fao_vals, pet_fao_units, pet_fao_name),
                index=date_index,
            )
        if pet_fao_s is None and can_compute_fao_pet:
          d2m_c = self._temp_to_celsius(
              d2m_daily_raw[i, rel_t_indices], d2m_units, d2m_name
          )
          sp_kpa = self._pressure_to_kpa(
              sp_daily_raw[i, rel_t_indices], sp_units, sp_name
          )
          ssr_vals = ssr_daily_raw[i, rel_t_indices]
          str_vals = str_daily_raw[i, rel_t_indices]
          u10_vals = u10_daily_raw[i, rel_t_indices]
          v10_vals = v10_daily_raw[i, rel_t_indices]
          if not any(
              np.all(np.isnan(arr))
              for arr in (d2m_c, sp_kpa, ssr_vals, str_vals, u10_vals, v10_vals)
          ):
            pet_fao_s = calculate_fao_pm_pet(
                surface_pressure_kpa=pd.Series(sp_kpa, index=date_index),
                temperature_2m_c=pd.Series(t_series, index=date_index),
                dewpoint_temperature_2m_c=pd.Series(d2m_c, index=date_index),
                u_component_of_wind_10m=pd.Series(u10_vals, index=date_index),
                v_component_of_wind_10m=pd.Series(v10_vals, index=date_index),
                surface_net_solar_radiation_mean=pd.Series(ssr_vals, index=date_index),
                surface_net_thermal_radiation_mean=pd.Series(str_vals, index=date_index),
                radiation_units=rad_units,
            )

      pet_era5_s = None
      if pet_era5_daily_raw is not None and pet_era5_name is not None:
        era5_vals = pet_era5_daily_raw[i, rel_t_indices]
        if not np.all(np.isnan(era5_vals)):
          era5_mm = self._depth_to_mm(era5_vals, pet_era5_units, pet_era5_name)
          pet_era5_s = pd.Series(
              normalize_era5_pet_sign(era5_mm),
              index=date_index,
          )

      results[cid] = compute_caravan_climate_metrics(
          pd.Series(p_series, index=date_index),
          pd.Series(t_series, index=date_index),
          pet_era5=pet_era5_s,
          pet_fao=pet_fao_s,
      )

    return results
