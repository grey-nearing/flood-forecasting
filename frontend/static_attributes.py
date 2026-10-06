"""HydroATLAS & Caravan Static Catchment Attributes Extractor for OpenHydroNet.

Wraps the core `multimet.static_extractor` package on `main`, adding:
1. Automatic local path resolution across `~/.cache/googlehydrology/...` and
   `~/.cache/openhydronet/data/...` so both caches work out-of-the-box.
2. Fast viewport-level raw BasinATLAS polygon + attribute querying
   (`extract_raw_subbasins_in_bbox`) for the Geographical Features Map Viewer tab.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyogrio
import shapely.geometry

from frontend.config import (
    HYDRO_BASINS_DIR,
    ensure_flood_forecasting_on_sys_path,
)

ensure_flood_forecasting_on_sys_path()

from multimet.static_extractor import climate as _gh_climate  # pylint: disable=g-import-not-at-top
from multimet.static_extractor import config as _gh_config  # pylint: disable=g-import-not-at-top
from multimet.static_extractor import extractor as _gh_extractor  # pylint: disable=g-import-not-at-top

logger = logging.getLogger(__name__)

# Re-export canonical constants and helpers from multimet.static_extractor
ATTRIBUTE_DEFINITIONS = _gh_config.ATTRIBUTE_DEFINITIONS
for _k, _v in ATTRIBUTE_DEFINITIONS.items():
  if "label" not in _v and "name" in _v:
    _v["label"] = _v["name"]

CONTINENT_MAP = _gh_config.CONTINENT_MAP

CARAVAN_CLIMATE_COLUMNS: List[str] = [
    "p_mean",
    "pet_mean",
    "pet_mean_FAO_PM",
    "pet_mean_ERA5_LAND",
    "aridity",
    "aridity_FAO_PM",
    "aridity_ERA5_LAND",
    "frac_snow",
    "moisture_index",
    "moisture_index_FAO_PM",
    "moisture_index_ERA5_LAND",
    "seasonality",
    "seasonality_FAO_PM",
    "seasonality_ERA5_LAND",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
]
CARAVAN_HYDROATLAS_COLUMNS: List[str] = [
    k for k in ATTRIBUTE_DEFINITIONS if k not in CARAVAN_CLIMATE_COLUMNS
]
CARAVAN_ALL_COLUMNS: List[str] = (
    CARAVAN_HYDROATLAS_COLUMNS + CARAVAN_CLIMATE_COLUMNS
)

calculate_fao_pm_pet = _gh_climate.calculate_fao_pm_pet
calculate_knoben_moisture_and_seasonality = (
    _gh_climate.calculate_knoben_moisture_and_seasonality
)
_split_list = _gh_climate._split_list  # pylint: disable=protected-access
compute_caravan_climate_metrics = _gh_climate.compute_caravan_climate_metrics
_BaseERA5ClimateLoader = _gh_climate.ERA5ClimateLoader
_BaseERA5GriddedExtractor = _gh_climate.ERA5GriddedExtractor
_BaseStaticAttributesExtractor = _gh_extractor.StaticAttributesExtractor


def _resolve_best_gdb_path(
    gdb_path: Optional[Union[str, Path]] = None,
) -> Path:
  """Resolves the BasinATLAS v10 Geodatabase path across standard cache locations."""
  if gdb_path is not None:
    return Path(gdb_path)
  env_gdb = os.environ.get("HYDROATLAS_GDB_PATH")
  if env_gdb:
    return Path(env_gdb)
  candidates = [
      Path.home()
      / ".cache"
      / "googlehydrology"
      / "hydroatlas"
      / "BasinATLAS_v10.gdb",
      HYDRO_BASINS_DIR / "basin_atlas" / "BasinATLAS_v10.gdb",
      Path.home()
      / ".cache"
      / "openhydronet"
      / "data"
      / "basin_atlas"
      / "BasinATLAS_v10.gdb",
  ]
  for cand in candidates:
    if cand.exists() and any(cand.glob("*.gdbtable")):
      return cand
  return candidates[0]


def _resolve_best_era5_cache_dir(
    era5_cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
  """Resolves the Level 12 precomputed ERA5 climate indices directory."""
  if era5_cache_dir is not None:
    return Path(era5_cache_dir)
  env_dir = os.environ.get("ERA5_CLIMATE_CACHE_DIR")
  if env_dir:
    return Path(env_dir)
  candidates = [
      Path.home() / ".cache" / "openhydronet" / "data" / "era5_climate",
      Path.home() / ".cache" / "googlehydrology" / "era5_climate",
  ]
  for cand in candidates:
    if cand.exists() and any(cand.glob("*_climate_indices.txt")):
      return cand
  return candidates[0]


class ERA5ClimateLoader(_BaseERA5ClimateLoader):
  """ERA5 Level 12 precomputed climate loader with automatic local cache discovery."""

  def __init__(
      self,
      cache_dir: Optional[Union[str, Path]] = None,
      no_download: bool = False,
  ):
    super().__init__(
        cache_dir=_resolve_best_era5_cache_dir(cache_dir),
        no_download=no_download,
    )

  def get_indices_for_subbasins(
      self, hybas_ids: List[int], weights: List[float]
  ) -> Dict[str, float]:
    """Calculates area-weighted average ERA5 climate indices with fallback for legacy cache keys."""
    res = super().get_indices_for_subbasins(hybas_ids, weights)
    fallback_pairs = (
        ("pet_mean_ERA5_LAND", "pet_mean"),
        ("aridity_ERA5_LAND", "aridity"),
        ("moisture_index_ERA5_LAND", "moisture_index"),
        ("seasonality_ERA5_LAND", "seasonality"),
    )
    for era5_key, base_key in fallback_pairs:
      val = res.get(era5_key, np.nan)
      base_val = res.get(base_key, np.nan)
      if (val is None or np.isnan(val)) and base_val is not None and not np.isnan(base_val):
        res[era5_key] = float(base_val)
    return res


class ERA5RawGriddedExtractor(_BaseERA5GriddedExtractor):
  """Backwards-compatible alias for ERA5GriddedExtractor supporting extract_climate_indices_for_polygon."""

  def __init__(self, zarr_uri: Optional[str] = None):
    if zarr_uri:
      super().__init__(zarr_uri=zarr_uri)
    else:
      self.zarr_uri = None
      self._ds = None
      self._lats = None
      self._lons = None
      self._dlat = 0.1
      self._dlon = 0.1

  def extract_climate_indices_for_polygon(
      self,
      polygon: Any,
      baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
  ) -> Dict[str, float]:
    """Alias for extract_climate_metrics_for_polygon."""
    if not self.zarr_uri:
      return {}
    try:
      return self.extract_climate_metrics_for_polygon(
          polygon, baseline_years=baseline_years
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("ERA5RawGriddedExtractor failed: %s", e)
      return {}


class StaticAttributesExtractor(_BaseStaticAttributesExtractor):
  """HydroATLAS + ERA5-Land Caravan static extractor with raw BasinATLAS viewport querying."""

  def __init__(
      self,
      gdb_path: Optional[Union[str, Path]] = None,
      era5_source: str = "hybas",
      era5_cache_dir: Optional[Union[str, Path]] = None,
      gridded_era5_uri: Optional[str] = None,
      auto_download: bool = True,
  ):
    norm_source = (
        "hybas"
        if (not era5_source or era5_source.strip().lower() == "precomputed")
        else era5_source.strip().lower()
    )
    resolved_gdb = _resolve_best_gdb_path(gdb_path)
    resolved_era5 = _resolve_best_era5_cache_dir(era5_cache_dir)
    super().__init__(
        gdb_path=resolved_gdb,
        era5_source=norm_source,
        era5_cache_dir=resolved_era5,
        gridded_era5_uri=gridded_era5_uri,
        no_download=not auto_download,
    )
    self.era5_loader = ERA5ClimateLoader(
        cache_dir=resolved_era5, no_download=not auto_download
    )

  def _ensure_gdb(self) -> None:
    """No-op compatibility hook (super().__init__ initializes self.gdb_path)."""

  def append_attributes_to_zarr(
      self,
      master_zarr_path: Union[str, Path],
      catchment_id: str,
      attributes_result: Dict[str, Any],
  ) -> bool:
    """Appends static attributes to master_zarr_path if catchment_id is present in the store."""
    try:
      return super().append_attributes_to_zarr(
          master_zarr_path, catchment_id, attributes_result
      )
    except KeyError:
      logger.debug(
          "Catchment %s not in master Zarr %s; skipping Zarr attribute append.",
          catchment_id,
          master_zarr_path,
      )
      return False
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning(
          "Could not append attributes for %s to %s: %s",
          catchment_id,
          master_zarr_path,
          e,
      )
      return False

  def extract_attributes(
      self,
      polygon_geojson: Dict[str, Any],
      catchment_id: str = "custom_catchment",
      baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
  ) -> Dict[str, Any]:
    """Extracts all Caravan static attributes and attaches frontend status/flat_attributes keys."""
    res = super().extract_attributes_for_polygon(
        polygon_geojson=polygon_geojson,
        catchment_id=catchment_id,
        baseline_years=baseline_years or (1981, 2020),
        era5_source=self.era5_source,
    )
    res.setdefault("status", "success")
    res.setdefault("flat_attributes", res.get("caravan_attributes", {}))
    return res

  def extract_attributes_for_polygon(
      self,
      polygon_geojson: Any,
      catchment_id: Optional[str] = "custom_catchment",
      min_overlap_threshold: float = 0.0,
      baseline_years: Optional[Tuple[int, int]] = (1981, 2020),
      timeseries_df: Any = None,
      era5_source: Optional[str] = None,
      use_raw_gridded_era5: bool = False,
      _batch_mode: bool = False,
      _skip_climate: bool = False,
  ) -> Dict[str, Any]:
    """Extracts all Caravan static attributes for a catchment polygon."""
    target_source = "gridded" if use_raw_gridded_era5 else (era5_source or self.era5_source)
    if target_source == "precomputed":
      target_source = "hybas"
    res = super().extract_attributes_for_polygon(
        polygon_geojson=polygon_geojson,
        catchment_id=catchment_id,
        min_overlap_threshold=min_overlap_threshold,
        baseline_years=baseline_years or (1981, 2020),
        timeseries_df=timeseries_df,
        era5_source=target_source,
        _batch_mode=_batch_mode,
        _skip_climate=_skip_climate,
    )
    res.setdefault("status", "success")
    res.setdefault("flat_attributes", res.get("caravan_attributes", {}))
    return res

  def get_raw_hydroatlas_map_layer(
      self,
      bbox: Tuple[float, float, float, float],
      zoom: int = 7,
      attribute: str = "ele_mt_sav",
      level: Optional[int] = None,
      max_features: int = 800,
  ) -> Dict[str, Any]:
    """Alias for extract_raw_subbasins_in_bbox used by GET /api/attributes/map-layer."""
    return self.extract_raw_subbasins_in_bbox(
        bbox=bbox,
        attribute_key=attribute,
        zoom=zoom,
        level=level,
        max_features=max_features,
    )

  def extract_raw_subbasins_in_bbox(
      self,
      bbox: Tuple[float, float, float, float],
      attribute_key: str = "ele_mt_sav",
      zoom: int = 7,
      level: Optional[int] = None,
      max_features: int = 800,
  ) -> Dict[str, Any]:
    """Queries raw HydroATLAS BasinATLAS sub-basin polygons in a bounding box for map visualization."""
    self._ensure_gdb()
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon = max(-180.0, float(min_lon))
    min_lat = max(-90.0, float(min_lat))
    max_lon = min(180.0, float(max_lon))
    max_lat = min(90.0, float(max_lat))

    attr_info = ATTRIBUTE_DEFINITIONS.get(
        attribute_key,
        {
            "label": attribute_key,
            "name": attribute_key,
            "unit": "",
            "category": "HydroATLAS",
            "desc": attribute_key,
        },
    )
    attr_label = attr_info.get("label") or attr_info.get("name") or attribute_key

    is_era5_climate = attribute_key in CARAVAN_CLIMATE_COLUMNS

    if is_era5_climate:
      chosen_level = 12
    elif level is not None and 1 <= int(level) <= 12:
      chosen_level = int(level)
    else:
      if zoom <= 3:
        chosen_level = 4
      elif zoom <= 4:
        chosen_level = 5
      elif zoom <= 5:
        chosen_level = 6
      elif zoom <= 6:
        chosen_level = 7
      elif zoom <= 7:
        chosen_level = 8
      elif zoom <= 8:
        chosen_level = 10
      else:
        chosen_level = 12

    layer_name = f"BasinATLAS_v10_lev{chosen_level:02d}"

    read_cols = ["HYBAS_ID", "SUB_AREA", "UP_AREA"]
    if not is_era5_climate and attribute_key not in (
        "basin_area",
        "HYBAS_ID",
        "SUB_AREA",
        "UP_AREA",
    ):
      read_cols.append(attribute_key)

    try:
      gdf = pyogrio.read_dataframe(
          self.gdb_path,
          layer=layer_name,
          bbox=(min_lon, min_lat, max_lon, max_lat),
          columns=read_cols,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning(
          "Failed to read layer %s from %s: %s", layer_name, self.gdb_path, e
      )
      return {
          "type": "FeatureCollection",
          "features": [],
          "properties": {
              "attribute": attribute_key,
              "label": attr_label,
              "unit": attr_info["unit"],
              "category": attr_info["category"],
              "level": chosen_level,
              "count": 0,
              "min_value": None,
              "max_value": None,
          },
      }

    if gdf.empty:
      return {
          "type": "FeatureCollection",
          "features": [],
          "properties": {
              "attribute": attribute_key,
              "label": attr_label,
              "unit": attr_info["unit"],
              "category": attr_info["category"],
              "level": chosen_level,
              "count": 0,
              "min_value": None,
              "max_value": None,
          },
      }

    if len(gdf) > max_features:
      gdf = gdf.iloc[:max_features].copy()

    # Pre-load ERA5 climate records if needed
    if is_era5_climate and self.era5_loader is not None:
      needed_continents = set()
      for hid in gdf["HYBAS_ID"].tolist():
        first_digit = int(str(int(hid))[0])
        if first_digit in CONTINENT_MAP:
          needed_continents.add(CONTINENT_MAP[first_digit])
      for cont in sorted(needed_continents):
        try:
          self.era5_loader.ensure_continent(cont)
        except Exception as e:  # pylint: disable=broad-exception-caught
          logger.warning("Could not load ERA5 continent %s: %s", cont, e)

    # Simplify geometries slightly at low zooms for snappy map rendering
    simplify_tol = 0.0
    if chosen_level <= 5:
      simplify_tol = 0.015
    elif chosen_level <= 7:
      simplify_tol = 0.006
    elif chosen_level <= 9:
      simplify_tol = 0.002

    features: List[Dict[str, Any]] = []
    valid_values: List[float] = []

    raw_climate_key = attribute_key
    if attribute_key == "pet_mean_FAO_PM":
      raw_climate_key = "pet_mean"
    elif attribute_key == "aridity_FAO_PM":
      raw_climate_key = "aridity"
    elif attribute_key == "moisture_index_FAO_PM":
      raw_climate_key = "moisture_index"
    elif attribute_key == "seasonality_FAO_PM":
      raw_climate_key = "seasonality"

    for _, row in gdf.iterrows():
      geom = row.geometry
      if geom is None or geom.is_empty:
        continue
      if simplify_tol > 0.0:
        geom = geom.simplify(simplify_tol, preserve_topology=True)

      hid = int(row["HYBAS_ID"])
      sub_area = float(row.get("SUB_AREA", 0.0) or 0.0)
      up_area = float(row.get("UP_AREA", 0.0) or 0.0)

      val: Optional[float] = None
      if attribute_key == "basin_area":
        val = sub_area
      elif is_era5_climate and self.era5_loader is not None:
        rec = self.era5_loader.records.get(hid)
        if rec is not None and raw_climate_key in rec:
          raw_v = rec[raw_climate_key]
          if raw_v is not None and not (
              isinstance(raw_v, float) and math.isnan(raw_v)
          ):
            val = float(raw_v)
      else:
        raw_v = row.get(attribute_key)
        if raw_v is not None and not (
            isinstance(raw_v, float) and math.isnan(raw_v)
        ):
          fval = float(raw_v)
          if fval > -990.0 and fval != -9999.0:
            if attribute_key in ("tmp_dc_syr", "tmp_dc_smn", "tmp_dc_smx"):
              fval = fval / 10.0
            elif attribute_key in ("ari_ix_sav", "SLOPE_GRAD"):
              fval = fval / 100.0
            elif attribute_key in ("slp_dg_sav",):
              fval = fval / 10.0
            val = fval

      if val is not None and math.isfinite(val):
        val = round(val, 4)
        valid_values.append(val)
      else:
        val = None

      features.append({
          "type": "Feature",
          "properties": {
              "HYBAS_ID": hid,
              "sub_area_km2": round(sub_area, 2),
              "up_area_km2": round(up_area, 2),
              "attribute": attribute_key,
              "value": val,
              "unit": attr_info["unit"],
          },
          "geometry": shapely.geometry.mapping(geom),
      })

    min_val = round(min(valid_values), 4) if valid_values else None
    max_val = round(max(valid_values), 4) if valid_values else None

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "attribute": attribute_key,
            "label": attr_label,
            "unit": attr_info["unit"],
            "category": attr_info["category"],
            "description": attr_info.get("desc", ""),
            "level": chosen_level,
            "layer": layer_name,
            "count": len(features),
            "min_value": min_val,
            "max_value": max_val,
        },
    }


_extractor_instance: Optional[StaticAttributesExtractor] = None


def get_attributes_extractor() -> StaticAttributesExtractor:
  global _extractor_instance
  if _extractor_instance is None:
    _extractor_instance = StaticAttributesExtractor()
  return _extractor_instance
