"""Hydrography river network indexing supporting both authentic 90m MERIT Hydro (MERIT-Basins) and HydroATLAS (HydroSHEDS)."""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, box, mapping

from frontend.config import (
    DATA_DIR,
    HYDRO_DATASETS,
    RIVER_NETWORKS_DIR,
    resolve_hydro_dataset_id,
)

CACHE_DIR = Path.home() / ".cache" / "earthkit_hydro" / "data"

# HydroATLAS / HydroSHEDS (15/30 arc-sec)
HYDRORIVERS_SHP = RIVER_NETWORKS_DIR / "hydrorivers" / "HydroRIVERS_v10.shp"
if not HYDRORIVERS_SHP.exists():
  HYDRORIVERS_SHP = CACHE_DIR / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10.shp"

HYDROATLAS_Z2 = RIVER_NETWORKS_DIR / "global_pyramids" / "global_rivers_fast_z2.geojson"
if not HYDROATLAS_Z2.exists():
  HYDROATLAS_Z2 = CACHE_DIR / "global_rivers_fast_z2.geojson"

HYDROATLAS_Z4 = RIVER_NETWORKS_DIR / "global_pyramids" / "global_rivers_fast_z4.geojson"
if not HYDROATLAS_Z4.exists():
  HYDROATLAS_Z4 = CACHE_DIR / "global_rivers_fast_z4.geojson"

# MERIT Hydro / MERIT-Basins (3 arc-sec, ~90m)
MERIT_DIR = RIVER_NETWORKS_DIR / "merit_rivers"
if not MERIT_DIR.exists():
  MERIT_DIR = CACHE_DIR / "merit_basins" / "ms_riv_network"

MERIT_INDEX_FILE = RIVER_NETWORKS_DIR / "merit_partitions_index.json"
if not MERIT_INDEX_FILE.exists():
  MERIT_INDEX_FILE = CACHE_DIR / "merit_basins" / "merit_partitions_index.json"

MERIT_Z2 = RIVER_NETWORKS_DIR / "global_pyramids" / "merit_rivers_fast_z2.geojson"
if not MERIT_Z2.exists():
  MERIT_Z2 = RIVER_NETWORKS_DIR / "merit_rivers_fast_z2.geojson"
if not MERIT_Z2.exists():
  MERIT_Z2 = CACHE_DIR / "merit_basins" / "merit_rivers_fast_z2.geojson"

MERIT_Z4 = RIVER_NETWORKS_DIR / "global_pyramids" / "merit_rivers_fast_z4.geojson"
if not MERIT_Z4.exists():
  MERIT_Z4 = RIVER_NETWORKS_DIR / "merit_rivers_fast_z4.geojson"
if not MERIT_Z4.exists():
  MERIT_Z4 = CACHE_DIR / "merit_basins" / "merit_rivers_fast_z4.geojson"

# In-memory caches for ultra-fast initial world loads
_CACHES: Dict[str, Dict[str, Any]] = {
    "hydroatlas_z2": None,
    "hydroatlas_z4": None,
    "merit_z2": None,
    "merit_z4": None,
}
_MERIT_PARTITIONS: Optional[List[Dict[str, Any]]] = None


def _resolve_merit_partition_path(part: Dict[str, Any]) -> Optional[Path]:
  """Resolves a MERIT-Basins partition shapefile path across local data and cache directories."""
  raw_path = Path(part.get("path", ""))
  if raw_path.exists():
    return raw_path
  name = part.get("name") or raw_path.name
  for subdir in ("pfaf_level_02", "pfaf_level_01"):
    candidate = RIVER_NETWORKS_DIR / subdir / name
    if candidate.exists():
      return candidate
  return None


def _preload_all():
  global _CACHES, _MERIT_PARTITIONS

  if HYDROATLAS_Z2.exists() and _CACHES["hydroatlas_z2"] is None:
    try:
      with open(HYDROATLAS_Z2, "r") as f:
        _CACHES["hydroatlas_z2"] = json.load(f)
    except Exception as e:
      print(f"Error loading HydroATLAS Z2: {e}")

  if HYDROATLAS_Z4.exists() and _CACHES["hydroatlas_z4"] is None:
    try:
      with open(HYDROATLAS_Z4, "r") as f:
        _CACHES["hydroatlas_z4"] = json.load(f)
    except Exception as e:
      print(f"Error loading HydroATLAS Z4: {e}")

  if MERIT_Z2.exists() and _CACHES["merit_z2"] is None:
    try:
      with open(MERIT_Z2, "r") as f:
        _CACHES["merit_z2"] = json.load(f)
    except Exception as e:
      print(f"Error loading MERIT Z2: {e}")

  if MERIT_Z4.exists() and _CACHES["merit_z4"] is None:
    try:
      with open(MERIT_Z4, "r") as f:
        _CACHES["merit_z4"] = json.load(f)
    except Exception as e:
      print(f"Error loading MERIT Z4: {e}")

  if MERIT_INDEX_FILE.exists() and _MERIT_PARTITIONS is None:
    try:
      with open(MERIT_INDEX_FILE, "r") as f:
        raw_parts = json.load(f)
      # Prefer pfaf_level_02 (61 sub-continental partitions) over pfaf_level_01 (9 continental
      # partitions) so reaches are never duplicated and spatial bounding boxes are tighter.
      l2_parts = []
      l1_parts = []
      for p in raw_parts:
        resolved = _resolve_merit_partition_path(p)
        if resolved is None:
          continue
        entry = dict(p)
        entry["path"] = str(resolved)
        if "pfaf_level_02" in str(resolved) or p.get("name", "").startswith(
            tuple(f"riv_pfaf_{d}{s}" for d in range(1, 10) for s in range(1, 10))
        ):
          l2_parts.append(entry)
        else:
          l1_parts.append(entry)
      _MERIT_PARTITIONS = l2_parts if l2_parts else l1_parts
    except Exception as e:
      print(f"Error loading MERIT partitions index: {e}")


# Initialize in-memory caches
_preload_all()

# The HydroATLAS z4 pyramid holds every HydroRIVERS reach with Strahler order >= 7 (checked against the
# shapefile) with simplified geometry. Wide views that only need those orders can be answered from memory
# in milliseconds instead of scanning the 1.2 GB shapefile (2-30 s per request at continental extents).
_Z4_COMPLETE_MIN_ORDER = 7
_MERIT_Z4_COMPLETE_MIN_ORDER = 6
_Z4_INDEX: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (per-feature [minx, miny, maxx, maxy], orders)
_MERIT_Z4_INDEX: Optional[Tuple[np.ndarray, np.ndarray]] = None
_Z4_INDEX_LOCK = threading.Lock()


def _build_pyramid_index(feats: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
  boxes = np.full((len(feats), 4), np.nan)
  orders = np.zeros(len(feats), dtype=np.int16)
  for i, f in enumerate(feats):
    geom = f.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if geom.get("type") == "MultiLineString":
      coords = [pt for part in coords for pt in part]
    if coords:
      xy = np.asarray(coords, dtype=float)[:, :2]
      boxes[i] = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    props = f.get("properties") or {}
    orders[i] = int(props.get("stream_order") or props.get("river_class") or 0)
  return boxes, orders


def _hydroatlas_z4_index() -> Optional[Tuple[np.ndarray, np.ndarray]]:
  """Lazily builds per-feature bounding boxes and stream orders for the HydroATLAS z4 pyramid."""
  global _Z4_INDEX
  if _Z4_INDEX is None and _CACHES.get("hydroatlas_z4") is not None:
    with _Z4_INDEX_LOCK:
      if _Z4_INDEX is None:
        _Z4_INDEX = _build_pyramid_index(_CACHES["hydroatlas_z4"]["features"])
  return _Z4_INDEX


def _merit_z4_index() -> Optional[Tuple[np.ndarray, np.ndarray]]:
  """Lazily builds per-feature bounding boxes and stream orders for the MERIT-Hydro z4 pyramid."""
  global _MERIT_Z4_INDEX
  if _MERIT_Z4_INDEX is None and _CACHES.get("merit_z4") is not None:
    with _Z4_INDEX_LOCK:
      if _MERIT_Z4_INDEX is None:
        _MERIT_Z4_INDEX = _build_pyramid_index(_CACHES["merit_z4"]["features"])
  return _MERIT_Z4_INDEX


class HydroRiverNetwork:
  """Manages real multi-dataset river network spatial queries for MERIT-Hydro and HydroATLAS."""

  def __init__(self, dataset_id: str = "hydroatlas"):
    resolved_id = resolve_hydro_dataset_id(dataset_id)
    if resolved_id not in HYDRO_DATASETS:
      raise ValueError(f"Unknown dataset '{dataset_id}'")
    self.dataset_id = resolved_id
    self.dataset_meta = HYDRO_DATASETS[resolved_id]
    self.is_merit = resolved_id == "merit-hydro"

  def get_rivers_in_bbox(
      self,
      min_lon: float,
      min_lat: float,
      max_lon: float,
      max_lat: float,
      zoom: int = 2,
      min_stream_order: Optional[int] = None,
      prefer_cache: bool = False,
  ) -> Dict[str, Any]:
    """Retrieves authentic river reaches intersecting a bounding box filtered by zoom level LOD.

    Args:
      prefer_cache: Opt-in (the MaaS tab sends `lod=fast`). When `min_stream_order` is >= 7, answer from
        the in-memory HydroATLAS z4 pyramid (same reaches, simplified geometry, fewer properties).
    """
    global _CACHES, _MERIT_PARTITIONS
    _preload_all()

    # Clamp coordinates to valid world ranges
    min_lon = max(-180.0, min_lon)
    max_lon = min(180.0, max_lon)
    min_lat = max(-85.0, min_lat)
    max_lat = min(85.0, max_lat)

    if min_lon >= max_lon:
      min_lon, max_lon = -180.0, 180.0
    if min_lat >= max_lat:
      min_lat, max_lat = -85.0, 85.0

    # -------------------------------------------------------------
    # 0. Opt-in fast path: wide views that only need order >= 7 (or >= 6 for MERIT)
    # -------------------------------------------------------------
    if prefer_cache and min_stream_order is not None:
      min_complete = _MERIT_Z4_COMPLETE_MIN_ORDER if self.is_merit else _Z4_COMPLETE_MIN_ORDER
      if min_stream_order >= min_complete:
        index = _merit_z4_index() if self.is_merit else _hydroatlas_z4_index()
        cache_key = "merit_z4" if self.is_merit else "hydroatlas_z4"
        if index is not None and _CACHES.get(cache_key) is not None:
          boxes, orders = index
          in_view = (
              (orders >= min_stream_order)
              & (boxes[:, 2] >= min_lon)
              & (boxes[:, 0] <= max_lon)
              & (boxes[:, 3] >= min_lat)
              & (boxes[:, 1] <= max_lat)
          )
          pyramid = _CACHES[cache_key]["features"]
          features = [pyramid[i] for i in np.flatnonzero(in_view)]
          return {
              "type": "FeatureCollection",
              "properties": {
                  "dataset": self.dataset_id,
                  "dataset_name": self.dataset_meta["name"],
                  "resolution": self.dataset_meta["resolution"],
                  "zoom": zoom,
                  "min_stream_order_applied": min_stream_order,
                  "feature_count": len(features),
                  "source": (
                      "MERIT-Basins z4 pyramid (in-memory, simplified geometry)"
                      if self.is_merit
                      else "HydroATLAS z4 pyramid (in-memory, simplified geometry)"
                  ),
                  "theme": "hydro-dynamics" if self.is_merit else "eco-hydrology",
              },
              "features": features,
          }

    # -------------------------------------------------------------
    # 1. Global View (Zoom 1 - 3): Instant return of world stems
    # -------------------------------------------------------------
    if zoom <= 3:
      cache_key = "merit_z2" if self.is_merit else "hydroatlas_z2"
      cached = _CACHES.get(cache_key)
      if cached is not None:
        if min_stream_order is not None:
          target_order = min_stream_order
        elif self.is_merit:
          # MERIT Strahler orders span 1-9 (Order 8-9: ~6.1k reaches, Order 7-9: ~21.2k reaches)
          target_order = 8 if zoom <= 2 else 7
        else:
          # HydroATLAS classes span 1-10 (Class 9-10: ~7.6k reaches, Class 8-10: ~22.8k reaches)
          target_order = 9 if zoom <= 2 else 8
        filtered_feats = [
            f
            for f in cached["features"]
            if (
                f.get("properties", {}).get("stream_order")
                or f.get("properties", {}).get("river_class")
                or 1
            )
            >= target_order
        ]
        if not filtered_feats:
          filtered_feats = cached["features"]
        return {
            "type": "FeatureCollection",
            "properties": {
                "dataset": self.dataset_id,
                "dataset_name": self.dataset_meta["name"],
                "resolution": self.dataset_meta["resolution"],
                "zoom": zoom,
                "feature_count": len(filtered_feats),
                "theme": "hydro-dynamics" if self.is_merit else "eco-hydrology",
            },
            "features": filtered_feats,
        }

    # -------------------------------------------------------------
    # 2. Continental View (Zoom 4)
    # -------------------------------------------------------------
    if zoom == 4 and min_stream_order is None:
      cache_key = "merit_z4" if self.is_merit else "hydroatlas_z4"
      target_order = 6 if self.is_merit else 7
      index = _merit_z4_index() if self.is_merit else _hydroatlas_z4_index()
      if index is not None and _CACHES.get(cache_key) is not None:
        boxes, orders = index
        in_view = (
            (orders >= target_order)
            & (boxes[:, 2] >= min_lon)
            & (boxes[:, 0] <= max_lon)
            & (boxes[:, 3] >= min_lat)
            & (boxes[:, 1] <= max_lat)
        )
        pyramid = _CACHES[cache_key]["features"]
        filtered_feats = [pyramid[i] for i in np.flatnonzero(in_view)]
        return {
            "type": "FeatureCollection",
            "properties": {
                "dataset": self.dataset_id,
                "dataset_name": self.dataset_meta["name"],
                "resolution": self.dataset_meta["resolution"],
                "zoom": zoom,
                "feature_count": len(filtered_feats),
                "theme": "hydro-dynamics" if self.is_merit else "eco-hydrology",
            },
            "features": filtered_feats,
        }
      cached = _CACHES.get(cache_key) or _CACHES.get(
          "merit_z2" if self.is_merit else "hydroatlas_z2"
      )
      if cached is not None:
        filtered_feats = [
            f
            for f in cached["features"]
            if (
                f.get("properties", {}).get("stream_order")
                or f.get("properties", {}).get("river_class")
                or 1
            )
            >= target_order
        ]
        if not filtered_feats:
          filtered_feats = cached["features"]
        return {
            "type": "FeatureCollection",
            "properties": {
                "dataset": self.dataset_id,
                "dataset_name": self.dataset_meta["name"],
                "resolution": self.dataset_meta["resolution"],
                "zoom": zoom,
                "feature_count": len(filtered_feats),
                "theme": "hydro-dynamics" if self.is_merit else "eco-hydrology",
            },
            "features": filtered_feats,
        }

    # -------------------------------------------------------------
    # 3. Regional / Local Multi-Scale View (Zoom >= 5)
    # -------------------------------------------------------------
    if min_stream_order is not None:
      order_threshold = min_stream_order
    else:
      if zoom <= 5:
        order_threshold = 5  # Major regional rivers
      elif zoom <= 7:
        order_threshold = 4  # Medium rivers
      elif zoom <= 9:
        order_threshold = 3  # Streams & tributaries
      elif zoom <= 11:
        order_threshold = 2  # Small tributaries
      else:
        order_threshold = 1  # All reaches & headwaters

    # -------------------------------------------------------------
    # CASE A: MERIT Hydro (90m MERIT-Basins Shapefiles)
    # -------------------------------------------------------------
    if self.is_merit and _MERIT_PARTITIONS:
      try:
        features = []
        search_bbox = (min_lon, min_lat, max_lon, max_lat)

        # Find intersecting Pfafstetter partition shapefiles
        for part in _MERIT_PARTITIONS:
          p_bbox = part.get("bbox", [])
          if len(p_bbox) != 4 or any(np.isnan(p_bbox)):
            continue
          p_minx, p_miny, p_maxx, p_maxy = p_bbox
          # Check bounding box overlap
          if not (
              max_lon < p_minx
              or min_lon > p_maxx
              or max_lat < p_miny
              or min_lat > p_maxy
          ):
            shp_path = part["path"]
            if not Path(shp_path).exists():
              continue

            gdf = gpd.read_file(
                shp_path,
                bbox=search_bbox,
                where=f'"order" >= {int(order_threshold)}',
                engine="pyogrio",
            )
            if gdf.empty:
              continue

            for row in gdf.itertuples(index=False):
              geom = getattr(row, "geometry", None)
              if geom is None or geom.is_empty:
                continue

              comid = int(getattr(row, "COMID", 0))
              reach_id = f"MERIT_{comid}"
              order = int(getattr(row, "order", 1))
              up_area = round(float(getattr(row, "uparea", 0.0)), 1)
              sinuosity = round(float(getattr(row, "sinuosity", 1.0)), 2)
              slope = round(float(getattr(row, "slope", 0.0)), 4)
              length_km = round(float(getattr(row, "lengthkm", 0.0)), 2)
              next_down = int(getattr(row, "NextDownID", 0))

              if up_area > 50000:
                name = f"MERIT Main Stem ({reach_id})"
              elif up_area > 5000:
                name = f"MERIT Regional River ({reach_id})"
              else:
                name = f"MERIT Tributary Reach ({reach_id})"

              features.append({
                  "type": "Feature",
                  "id": reach_id,
                  "geometry": mapping(geom),
                  "properties": {
                      "reach_id": reach_id,
                      "dataset": "merit-hydro",
                      "river_name": name,
                      "stream_order": order,
                      "upstream_area_km2": up_area,
                      "sinuosity": sinuosity,
                      "slope": slope,
                      "length_km": length_km,
                      "next_down": next_down,
                      "theme": "hydro-dynamics",
                  },
              })

        return {
            "type": "FeatureCollection",
            "properties": {
                "dataset": "merit-hydro",
                "dataset_name": self.dataset_meta["name"],
                "resolution": "~90m (3 arc-sec)",
                "zoom": zoom,
                "min_stream_order_applied": order_threshold,
                "feature_count": len(features),
                "source": "Authentic MERIT-Basins 90m Hydrography",
                "theme": "hydro-dynamics",
            },
            "features": features,
        }
      except Exception as e:
        print(f"Error querying MERIT-Basins shapefiles: {e}")

    # -------------------------------------------------------------
    # CASE B: HydroATLAS (HydroSHEDS / HydroRIVERS Shapefile)
    # -------------------------------------------------------------
    if HYDRORIVERS_SHP.exists():
      try:
        search_bbox = (min_lon, min_lat, max_lon, max_lat)
        gdf = gpd.read_file(
            str(HYDRORIVERS_SHP),
            bbox=search_bbox,
            where=f"ORD_STRA >= {order_threshold}",
            engine="pyogrio",
        )

        features = []
        for _, row in gdf.iterrows():
          geom = row["geometry"]
          if geom is None or geom.is_empty:
            continue

          hyriv_id = int(row.get("HYRIV_ID", 0))
          reach_id = f"HYRIV_{hyriv_id}"
          order = int(row.get("ORD_STRA", 1))
          river_class = int(row.get("ORD_CLAS", order))
          up_area = round(float(row.get("UPLAND_SKM", 0.0)), 1)
          discharge = round(float(row.get("DIS_AV_CMS", 0.0)), 2)
          length_km = round(float(row.get("LENGTH_KM", 0.0)), 2)
          catch_area = round(float(row.get("CATCH_SKM", 0.0)), 2)
          dist_ocean = round(float(row.get("DIST_DN_KM", 0.0)), 1)
          hybas_id = int(row.get("HYBAS_L12", 0))

          name = f"HydroATLAS Sub-Basin {hybas_id} (Class {river_class})"
          features.append({
              "type": "Feature",
              "id": reach_id,
              "geometry": mapping(geom),
              "properties": {
                  "reach_id": reach_id,
                  "dataset": "hydroatlas",
                  "river_name": name,
                  "stream_order": order,
                  "river_class": river_class,
                  "hydrobasins_unit": hybas_id,
                  "local_catchment_km2": catch_area,
                  "upstream_area_km2": up_area,
                  "mean_discharge_m3s": discharge,
                  "length_km": length_km,
                  "dist_to_ocean_km": dist_ocean,
                  "next_down": int(row.get("NEXT_DOWN", 0)),
                  "main_river": int(row.get("MAIN_RIV", 0)),
                  "theme": "eco-hydrology",
              },
          })

        return {
            "type": "FeatureCollection",
            "properties": {
                "dataset": "hydroatlas",
                "dataset_name": self.dataset_meta["name"],
                "resolution": "15/30 arc-sec (~500m/1km)",
                "zoom": zoom,
                "min_stream_order_applied": order_threshold,
                "feature_count": len(features),
                "source": "Authentic HydroSHEDS / HydroATLAS Database",
                "theme": "eco-hydrology",
            },
            "features": features,
        }
      except Exception as e:
        print(f"Error querying HydroRIVERS shapefile: {e}")

    return {
        "type": "FeatureCollection",
        "properties": {
            "dataset": self.dataset_id,
            "zoom": zoom,
            "feature_count": 0,
        },
        "features": [],
    }
