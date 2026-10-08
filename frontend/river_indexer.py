"""Frontend river network map layer indexer.

Uses in-memory `z2`/`z4` simplified pyramids for global/continental slippy-map
views (`zoom <= 4`) and delegates regional/local spatial queries (`zoom >= 5`)
to `multimet.catchment_delineation.RiverNetwork`.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import mapping

from frontend.config import (
    CACHE_DIR,
    HYDRO_DATASETS,
    RIVER_NETWORKS_DIR,
    ensure_flood_forecasting_on_sys_path,
    resolve_hydro_dataset_id,
)
from frontend.delineator import _get_river_network

ensure_flood_forecasting_on_sys_path()

from multimet.utils.hydrography import Reach  # pylint: disable=g-import-not-at-top

HYDROATLAS_Z2 = (
    RIVER_NETWORKS_DIR / "global_pyramids" / "global_rivers_fast_z2.geojson"
)
if not HYDROATLAS_Z2.exists():
  HYDROATLAS_Z2 = CACHE_DIR / "global_rivers_fast_z2.geojson"

HYDROATLAS_Z4 = (
    RIVER_NETWORKS_DIR / "global_pyramids" / "global_rivers_fast_z4.geojson"
)
if not HYDROATLAS_Z4.exists():
  HYDROATLAS_Z4 = CACHE_DIR / "global_rivers_fast_z4.geojson"

MERIT_Z2 = (
    RIVER_NETWORKS_DIR / "global_pyramids" / "merit_rivers_fast_z2.geojson"
)
if not MERIT_Z2.exists():
  MERIT_Z2 = RIVER_NETWORKS_DIR / "merit_rivers_fast_z2.geojson"
if not MERIT_Z2.exists():
  MERIT_Z2 = CACHE_DIR / "merit_basins" / "merit_rivers_fast_z2.geojson"

MERIT_Z4 = (
    RIVER_NETWORKS_DIR / "global_pyramids" / "merit_rivers_fast_z4.geojson"
)
if not MERIT_Z4.exists():
  MERIT_Z4 = RIVER_NETWORKS_DIR / "merit_rivers_fast_z4.geojson"
if not MERIT_Z4.exists():
  MERIT_Z4 = CACHE_DIR / "merit_basins" / "merit_rivers_fast_z4.geojson"

_CACHES: Dict[str, Optional[Dict[str, Any]]] = {
    "hydroatlas_z2": None,
    "hydroatlas_z4": None,
    "merit_z2": None,
    "merit_z4": None,
}

_Z4_COMPLETE_MIN_ORDER = 7
_MERIT_Z4_COMPLETE_MIN_ORDER = 6
_Z4_INDEX: Optional[Tuple[np.ndarray, np.ndarray]] = None
_MERIT_Z4_INDEX: Optional[Tuple[np.ndarray, np.ndarray]] = None
_Z4_INDEX_LOCK = threading.Lock()


def _preload_all() -> None:
  for key, path in (
      ("hydroatlas_z2", HYDROATLAS_Z2),
      ("hydroatlas_z4", HYDROATLAS_Z4),
      ("merit_z2", MERIT_Z2),
      ("merit_z4", MERIT_Z4),
  ):
    if _CACHES[key] is None and path.exists():
      try:
        with open(path, "r", encoding="utf-8") as f:
          _CACHES[key] = json.load(f)
      except Exception:
        _CACHES[key] = None


_preload_all()


def _build_pyramid_index(
    feats: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
  boxes = np.full((len(feats), 4), np.nan)
  orders = np.zeros(len(feats), dtype=np.int16)
  for i, f in enumerate(feats):
    geom = f.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if geom.get("type") == "MultiLineString":
      coords = [pt for part in coords for pt in part]
    if coords:
      xy = np.asarray(coords, dtype=float)[:, :2]
      boxes[i] = (
          xy[:, 0].min(),
          xy[:, 1].min(),
          xy[:, 0].max(),
          xy[:, 1].max(),
      )
    props = f.get("properties") or {}
    orders[i] = int(props.get("stream_order") or props.get("river_class") or 0)
  return boxes, orders


def _hydroatlas_z4_index() -> Optional[Tuple[np.ndarray, np.ndarray]]:
  global _Z4_INDEX
  if _Z4_INDEX is None and _CACHES.get("hydroatlas_z4") is not None:
    with _Z4_INDEX_LOCK:
      if _Z4_INDEX is None:
        _Z4_INDEX = _build_pyramid_index(_CACHES["hydroatlas_z4"]["features"])
  return _Z4_INDEX


def _merit_z4_index() -> Optional[Tuple[np.ndarray, np.ndarray]]:
  global _MERIT_Z4_INDEX
  if _MERIT_Z4_INDEX is None and _CACHES.get("merit_z4") is not None:
    with _Z4_INDEX_LOCK:
      if _MERIT_Z4_INDEX is None:
        _MERIT_Z4_INDEX = _build_pyramid_index(_CACHES["merit_z4"]["features"])
  return _MERIT_Z4_INDEX


def _reach_to_ui_feature(reach: Reach) -> Dict[str, Any]:
  """Formats a backend `Reach` object into a GeoJSON Feature for the map UI."""
  if reach.dataset == "merit-hydro":
    reach_id = f"MERIT_{reach.reach_id}"
    up_area = round(float(reach.upstream_area_km2), 1)
    if up_area > 50000:
      name = f"MERIT Main Stem ({reach_id})"
    elif up_area > 5000:
      name = f"MERIT Regional River ({reach_id})"
    else:
      name = f"MERIT Tributary Reach ({reach_id})"
    return {
        "type": "Feature",
        "id": reach_id,
        "geometry": mapping(reach.geometry),
        "properties": {
            "reach_id": reach_id,
            "dataset": "merit-hydro",
            "river_name": name,
            "stream_order": reach.stream_order,
            "upstream_area_km2": up_area,
            "sinuosity": round(
                float(reach.extra.get("sinuosity", 1.0)), 2
            ),
            "slope": round(float(reach.extra.get("slope", 0.0)), 4),
            "length_km": round(float(reach.length_km), 2),
            "next_down": reach.next_down,
            "theme": "hydro-dynamics",
        },
    }

  reach_id = f"HYRIV_{reach.reach_id}"
  river_class = int(reach.extra.get("river_class", reach.stream_order))
  hybas_id = int(reach.extra.get("hydrobasins_unit", 0))
  name = f"HydroATLAS Sub-Basin {hybas_id} (Class {river_class})"
  return {
      "type": "Feature",
      "id": reach_id,
      "geometry": mapping(reach.geometry),
      "properties": {
          "reach_id": reach_id,
          "dataset": "hydroatlas",
          "river_name": name,
          "stream_order": reach.stream_order,
          "river_class": river_class,
          "hydrobasins_unit": hybas_id,
          "local_catchment_km2": round(
              float(reach.extra.get("local_catchment_km2", 0.0)), 2
          ),
          "upstream_area_km2": round(float(reach.upstream_area_km2), 1),
          "mean_discharge_m3s": round(
              float(reach.extra.get("mean_discharge_m3s", 0.0)), 2
          ),
          "length_km": round(float(reach.length_km), 2),
          "dist_to_ocean_km": round(
              float(reach.extra.get("dist_to_ocean_km", 0.0)), 1
          ),
          "next_down": reach.next_down,
          "main_river": int(reach.extra.get("main_river", 0)),
          "theme": "eco-hydrology",
      },
  }


def zoom_to_min_stream_order(zoom: int) -> int:
  """Maps slippy-map zoom level (`zoom >= 5`) to minimum Strahler stream order."""
  if zoom <= 5:
    return 5
  if zoom <= 7:
    return 4
  if zoom <= 9:
    return 3
  if zoom <= 11:
    return 2
  return 1


class HydroRiverNetwork:
  """Frontend map layer provider for MERIT-Hydro and HydroATLAS river networks."""

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
    """Retrieves river reaches intersecting a bounding box filtered by zoom LOD."""
    _preload_all()

    min_lon = max(-180.0, min_lon)
    max_lon = min(180.0, max_lon)
    min_lat = max(-85.0, min_lat)
    max_lat = min(85.0, max_lat)

    if min_lon >= max_lon:
      min_lon, max_lon = -180.0, 180.0
    if min_lat >= max_lat:
      min_lat, max_lat = -85.0, 85.0

    theme = "hydro-dynamics" if self.is_merit else "eco-hydrology"

    # 0. Opt-in fast path for wide views requesting high stream orders
    if prefer_cache and min_stream_order is not None:
      min_complete = (
          _MERIT_Z4_COMPLETE_MIN_ORDER
          if self.is_merit
          else _Z4_COMPLETE_MIN_ORDER
      )
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
                      else (
                          "HydroATLAS z4 pyramid (in-memory, simplified"
                          " geometry)"
                      )
                  ),
                  "theme": theme,
              },
              "features": features,
          }

    # 1. Global View (Zoom 1 - 3)
    if zoom <= 3:
      cache_key = "merit_z2" if self.is_merit else "hydroatlas_z2"
      cached = _CACHES.get(cache_key)
      if cached is not None:
        if min_stream_order is not None:
          target_order = min_stream_order
        elif self.is_merit:
          target_order = 8 if zoom <= 2 else 7
        else:
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
                "theme": theme,
            },
            "features": filtered_feats,
        }

    # 2. Continental View (Zoom 4)
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
                "theme": theme,
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
                "theme": theme,
            },
            "features": filtered_feats,
        }

    # 3. Regional / Local View (Zoom >= 5): Delegate to backend RiverNetwork
    order_threshold = (
        min_stream_order
        if min_stream_order is not None
        else zoom_to_min_stream_order(zoom)
    )
    network = _get_river_network(self.dataset_id)
    if network is not None:
      try:
        reaches = network.query_reaches(
            bbox=(min_lon, min_lat, max_lon, max_lat),
            min_stream_order=int(order_threshold),
        )
        features = [_reach_to_ui_feature(r) for r in reaches]
        return {
            "type": "FeatureCollection",
            "properties": {
                "dataset": self.dataset_id,
                "dataset_name": self.dataset_meta["name"],
                "resolution": (
                    "~90m (3 arc-sec)"
                    if self.is_merit
                    else "15/30 arc-sec (~500m/1km)"
                ),
                "zoom": zoom,
                "min_stream_order_applied": order_threshold,
                "feature_count": len(features),
                "source": (
                    "Authentic MERIT-Basins 90m Hydrography"
                    if self.is_merit
                    else "Authentic HydroSHEDS / HydroATLAS Database"
                ),
                "theme": theme,
            },
            "features": features,
        }
      except Exception as e:
        print(f"Error querying RiverNetwork ({self.dataset_id}): {e}")

    return {
        "type": "FeatureCollection",
        "properties": {
            "dataset": self.dataset_id,
            "zoom": zoom,
            "feature_count": 0,
        },
        "features": [],
    }
