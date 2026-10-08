"""Frontend adapter for multimet.static_extractor."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Tuple, Sequence
import json

from frontend.config import (
    ensure_flood_forecasting_on_sys_path,
)

ensure_flood_forecasting_on_sys_path()

import numpy as np
import pandas as pd
import shapely.geometry

from multimet.static_extractor import (
    StaticAttributesExtractor,
    CatchmentAttributes,
    ATTRIBUTE_REGISTRY,
    get_attribute_definition,
    to_physical_units,
    UnsupportedHydroATLASLevelError,
    CARAVAN_CLIMATE_ALIASES,
)
from multimet.static_extractor.config import (
    DEFAULT_GCS_HYDROATLAS_URI,
    DEFAULT_GCS_ERA5_CLIMATE_URI,
    DEFAULT_GCS_GRIDDED_ERA5_URI,
)

logger = logging.getLogger(__name__)

MAX_MAP_FEATURES = 800

ZOOM_TO_LEVEL: Dict[int, int] = {
    0: 4, 1: 4, 2: 4, 3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 10,
    9: 12, 10: 12, 11: 12, 12: 12, 13: 12, 14: 12, 15: 12, 16: 12, 17: 12, 18: 12, 19: 12, 20: 12
}

LEVEL_SIMPLIFY_TOL: Dict[int, float] = {
    1: 0.015, 2: 0.015, 3: 0.015, 4: 0.015, 5: 0.015,
    6: 0.006, 7: 0.006,
    8: 0.002, 9: 0.002,
    10: 0.0005, 11: 0.0005, 12: 0.0005
}

def resolve_app_static_extractor_paths() -> Dict[str, Any]:
  # Use frontend.config logic
  repo_root = Path(__file__).resolve().parents[1]
  
  resolved_gdb = None
  if os.environ.get("HYDROATLAS_GDB_PATH"):
    resolved_gdb = Path(os.environ["HYDROATLAS_GDB_PATH"])
  else:
    gdb_candidates = [
      Path.home() / ".cache" / "googlehydrology" / "hydroatlas" / "BasinATLAS_v10.gdb",
      Path.home() / ".cache" / "googlehydrology" / "hydroatlas",
      repo_root / "frontend" / "data" / "shared" / "hydroatlas" / "BasinATLAS_v10.gdb",
      repo_root / "frontend" / "data" / "shared" / "hydroatlas",
      repo_root / "frontend" / "data" / "base_layers" / "hydro_basins" / "basin_atlas" / "BasinATLAS_v10.gdb",
      Path.home() / ".cache" / "earthkit_hydro" / "data" / "basin_atlas" / "BasinATLAS_v10.gdb",
      Path.home() / ".cache" / "earthkit_hydro" / "hydroatlas" / "BasinATLAS_v10.gdb",
    ]
    for cand in gdb_candidates:
      if not cand.exists():
        continue
      if cand.name.endswith(".gdb") and any(cand.glob("*.gdbtable")):
        resolved_gdb = cand
        break
      if (cand / "hydro_atlas_lev12.parquet").exists() and (cand / "subpolygons").is_dir():
        resolved_gdb = cand
        break
    if resolved_gdb is None:
      resolved_gdb = gdb_candidates[0]

  resolved_era5 = None
  if os.environ.get("ERA5_CLIMATE_CACHE_DIR"):
    resolved_era5 = Path(os.environ["ERA5_CLIMATE_CACHE_DIR"])
  else:
    era5_candidates = [
      Path.home() / ".cache" / "googlehydrology" / "era5_climate",
      repo_root / "frontend" / "data" / "shared" / "hydroatlas" / "era5_climate",
      Path.home() / ".cache" / "earthkit_hydro" / "data" / "era5_climate",
      Path.home() / ".cache" / "earthkit_hydro" / "era5_climate",
    ]
    for cand in era5_candidates:
      if cand.exists() and any(cand.glob("*_climate_indices.txt")):
        resolved_era5 = cand
        break
    if resolved_era5 is None:
      resolved_era5 = era5_candidates[0]

  resolved_gridded = os.environ.get("ERA5_GRIDDED_ZARR_URI") or None
  
  return {
    "gdb_path": resolved_gdb,
    "era5_cache_dir": resolved_era5,
    "gridded_era5_uri": resolved_gridded,
    "gcs_gdb_uri": DEFAULT_GCS_HYDROATLAS_URI,
    "gcs_era5_climate_uri": DEFAULT_GCS_ERA5_CLIMATE_URI,
  }


_extractor_instance: Optional[StaticAttributesExtractor] = None

def get_attributes_extractor() -> StaticAttributesExtractor:
  global _extractor_instance
  if _extractor_instance is None:
    paths = resolve_app_static_extractor_paths()
    _extractor_instance = StaticAttributesExtractor(
      gdb_path=paths["gdb_path"],
      era5_cache_dir=paths["era5_cache_dir"],
      gridded_era5_uri=paths["gridded_era5_uri"],
      gcs_gdb_uri=paths["gcs_gdb_uri"],
      gcs_era5_climate_uri=paths["gcs_era5_climate_uri"],
    )
  return _extractor_instance

def zoom_to_hydroatlas_level(zoom: int, available: Sequence[int]) -> int:
  target = ZOOM_TO_LEVEL.get(zoom, 12)
  valid = [l for l in available if l <= target]
  if valid:
    return max(valid)
  return min(available) if available else 12

def sanitize_for_json(obj: Any) -> Any:
  """Recursively converts NaN/Inf floats and numpy scalars to JSON-safe Python types."""
  if obj is None:
    return None
  if isinstance(obj, bool):
    return obj
  if isinstance(obj, (np.integer, int)):
    return int(obj)
  if isinstance(obj, (np.floating, float)):
    fv = float(obj)
    if math.isnan(fv) or math.isinf(fv):
      return None
    return fv
  if isinstance(obj, dict):
    return {str(k): sanitize_for_json(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [sanitize_for_json(item) for item in obj]
  return obj


def card_payload_from_cache_entry(catchment_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
  """Constructs the attribute card payload from a cached entry."""
  if "attributes" in entry:
    attrs = entry["attributes"]
  else:
    attrs = entry.get("raw_attributes", entry)
  
  # For area, try to pull it from the summary if available
  summary = entry.get("summary", {})
  area_km2 = entry.get("area_km2") or summary.get("total_area_km2") or attrs.get("basin_area") or 0.0
  
  subbasin_ids = tuple(entry.get("subbasin_ids") or summary.get("subbasin_ids") or [])
  subbasin_weights_km2 = tuple(entry.get("subbasin_weights_km2") or [0.0] * len(subbasin_ids))
  res = CatchmentAttributes(
    catchment_id=catchment_id,
    attributes=attrs,
    subbasin_ids=subbasin_ids,
    subbasin_weights_km2=subbasin_weights_km2,
    area_km2=area_km2,
    area_fraction_used_for_aggregation=entry.get(
      "area_fraction_used_for_aggregation",
      attrs.get("area_fraction_used_for_aggregation", 1.0),
    ),
  )
  return build_attribute_card_payload(res)


def build_attribute_card_payload(res: CatchmentAttributes) -> Dict[str, Any]:
  physical = res.physical_units(strict=False)
  
  processed_attributes = {}
  categories_dict: Dict[str, List[Dict[str, Any]]] = {
    "Topography": [],
    "Climate": [],
    "Soils": [],
    "Land Cover": [],
    "Hydrology": [],
    "Anthropogenic": [],
  }
  
  for key, val in physical.items():
    defn = get_attribute_definition(key)
    if not defn:
      continue
    
    unit = defn.physical_unit
    if math.isnan(val) or math.isinf(val):
      scaled_val = val
    elif unit in ["m", "mm/yr", "people", "M m³"]:
      scaled_val = round(val, 1) if unit != "people" else int(round(val))
    elif unit.startswith("class"):
      scaled_val = int(round(val))
    else:
      scaled_val = round(val, 3)
      
    processed_attributes[key] = scaled_val
    
    categories_dict.setdefault(defn.category, []).append({
      "key": key,
      "name": defn.name,
      "value": scaled_val,
      "unit": defn.physical_unit,
      "description": defn.description,
      "category": defn.category,
    })
    
  summary = {
    "catchment_id": res.catchment_id,
    "elevation_mean_m": processed_attributes.get("ele_mt_sav", np.nan),
    "slope_mean_deg": processed_attributes.get("slp_dg_sav", np.nan),
    "annual_precip_mm": processed_attributes.get("pre_mm_syr", np.nan),
    "annual_temp_c": processed_attributes.get("tmp_dc_syr", np.nan),
    "aridity_index": processed_attributes.get("ari_ix_sav", np.nan),
    "era5_p_mean_mm_day": processed_attributes.get("p_mean", np.nan),
    "era5_pet_mean_mm_day": processed_attributes.get("pet_mean_ERA5_LAND", np.nan),
    "era5_fao_pet_mean_mm_day": processed_attributes.get("pet_mean_FAO_PM", np.nan),
    "era5_aridity": processed_attributes.get("aridity_ERA5_LAND", np.nan),
    "era5_fao_aridity": processed_attributes.get("aridity_FAO_PM", np.nan),
    "era5_frac_snow_pc": processed_attributes.get("frac_snow", np.nan),
    "forest_fraction_pc": processed_attributes.get("for_pc_sse", np.nan),
    "cropland_fraction_pc": processed_attributes.get("crp_pc_sse", np.nan),
    "urban_fraction_pc": processed_attributes.get("urb_pc_sse", np.nan),
    "dominant_land_cover_class": physical.get("glc_cl_smj", np.nan),
    "soil_clay_pc": processed_attributes.get("cly_pc_sav", np.nan),
    "soil_sand_pc": processed_attributes.get("snd_pc_sav", np.nan),
    "soil_silt_pc": processed_attributes.get("slt_pc_sav", np.nan),
    "groundwater_table_depth_cm": processed_attributes.get("gwt_cm_sav", np.nan),
    "soil_water_content_pc": processed_attributes.get("swc_pc_syr", np.nan),
    "inundation_max_pc": processed_attributes.get("inu_pc_smx", np.nan),
    "total_area_km2": round(res.area_km2, 2),
    "intersected_subbasins": len(res.subbasin_ids),
    "subbasin_ids": list(res.subbasin_ids),
  }
  
  return sanitize_for_json({
    "status": "success",
    "catchment_id": res.catchment_id,
    "attributes": dict(res.attributes),
    "raw_attributes": dict(res.attributes),
    "flat_attributes": processed_attributes,
    "summary": summary,
    "categories": categories_dict,
    "processed_attributes": processed_attributes,
    "intersected_subbasins_count": len(res.subbasin_ids),
    "total_area_km2": round(res.area_km2, 2),
  })

def build_map_layer_geojson(extractor: StaticAttributesExtractor, bbox: Tuple[float,float,float,float], *, zoom: int, attribute: str, level: Optional[int] = None) -> Dict[str, Any]:
  avail = extractor.available_levels
  target_level = level if level is not None else zoom_to_hydroatlas_level(zoom, avail)
  
  defn = get_attribute_definition(attribute)
  if defn and defn.source == "era5_climate":
    target_level = 12
      
  label = defn.name if defn else attribute
  unit = defn.physical_unit if defn else ""
  category = defn.category if defn else ""
  desc = defn.description if defn else ""
  
  cols = ["HYBAS_ID", "SUB_AREA", "UP_AREA"]
  if attribute in ATTRIBUTE_REGISTRY and attribute not in cols:
    if defn and defn.source == "hydroatlas":
      cols.append(attribute)
      
  gdf = extractor.read_subbasins(bbox, level=target_level, columns=cols)
      
  truncated = False
  if len(gdf) > MAX_MAP_FEATURES:
    gdf = gdf.head(MAX_MAP_FEATURES).copy()
    truncated = True

  if target_level == 12 and defn and defn.source == "era5_climate" and len(gdf) > 0:
    hybas_ids = [int(h) for h in gdf["HYBAS_ID"].tolist()]
    try:
      clim_df = extractor.read_subbasin_climate_indices(hybas_ids)
      gdf = gdf.merge(clim_df, on="HYBAS_ID", how="left")
    except Exception as e:
      logger.warning("Could not load climate indices: %s", e)
            
  features = []
  valid_vals = []
  simplify_tol = LEVEL_SIMPLIFY_TOL.get(target_level, 0.0)
  
  for _, row in gdf.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
      continue
    geom_simplified = geom.simplify(simplify_tol, preserve_topology=True) if simplify_tol > 0 else geom
    
    hybas_id = int(row.get("HYBAS_ID", 0))
    sub_area = round(float(row.get("SUB_AREA", 0.0)), 2)
    up_area = round(float(row.get("UP_AREA", 0.0)), 2)
    
    row_attrs = {attribute: row.get(attribute)} if attribute in row else {}
    phys = to_physical_units(row_attrs, strict=False)
    scaled_val = phys.get(attribute)
    
    if scaled_val is not None and not math.isnan(scaled_val) and not math.isinf(scaled_val):
      scaled_val = round(float(scaled_val), 4)
      valid_vals.append(scaled_val)
    else:
      scaled_val = None
        
    features.append({
      "type": "Feature",
      "geometry": shapely.geometry.mapping(geom_simplified),
      "properties": {
        "HYBAS_ID": hybas_id,
        "SUB_AREA": sub_area,
        "UP_AREA": up_area,
        "sub_area_km2": sub_area,
        "up_area_km2": up_area,
        "attribute": attribute,
        attribute: scaled_val,
        "value": scaled_val,
        "unit": unit,
        "level": target_level,
      },
    })
      
  min_v = round(float(min(valid_vals)), 4) if valid_vals else None
  max_v = round(float(max(valid_vals)), 4) if valid_vals else None
  mean_v = round(float(np.mean(valid_vals)), 4) if valid_vals else None
  
  props = {
    "attribute": attribute,
    "attribute_key": attribute,
    "label": label,
    "unit": unit,
    "category": category,
    "description": desc,
    "level": target_level,
    "layer": f"BasinATLAS_v10_lev{target_level:02d}",
    "count": len(features),
    "min_value": min_v,
    "mean_value": mean_v,
    "max_value": max_v,
    "truncated": truncated
  }
      
  return sanitize_for_json({
    "status": "success",
    "type": "FeatureCollection",
    "features": features,
    "properties": props,
  })

def build_schema_payload() -> Dict[str, Any]:
  attr_list = []
  for k, v in ATTRIBUTE_REGISTRY.items():
    attr_list.append({
      "key": v.key,
      "name": v.name,
      "label": v.name,
      "category": v.category,
      "unit": v.physical_unit,
      "desc": v.description,
      "description": v.description,
    })
      
  attr_dict = {a["key"]: a for a in attr_list}
  return {
    "attributes": attr_dict,
    "attribute_list": attr_list,
    "extractor_source": "multimet.static_extractor"
  }
