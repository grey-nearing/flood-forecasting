"""Per-model river networks for the MaaS map, delegating core spatial indexing to `maas.networks`.

Binds `frontend.config.CACHE_DIR` and `frontend.config.RIVER_NETWORKS_DIR` while delegating
pyramid construction, level-of-detail extraction, and upstream-area snapping to `maas.networks`.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frontend.config import CACHE_DIR, RIVER_NETWORKS_DIR
from maas.config import (
    GEOGLOWS_MIN_AREA_KM2,
    GLOFAS_MIN_AREA_KM2,
    GLOFAS_NLAT,
    GLOFAS_NLON,
    GLOFAS_RES_DEG as GLOFAS_RES,
    NETWORK_LABELS,
    TE_BLOCK,
    TE_MIN_AREA_KM2,
)
from maas.networks import (
    CACHE_VERSION as _CACHE_VERSION,
    DISTANCE_WEIGHT as _DISTANCE_WEIGHT,
    FLOODHUB_LOD,
    GEOGLOWS_LOD,
    GLOFAS_LOD,
    LDD_DCOL as _LDD_DCOL,
    LDD_DROW as _LDD_DROW,
    MODELS,
    RANGE_SLACK as _RANGE_SLACK,
    TE_LOD,
    _choose,
    _gather_ranges,
    _index_of,
    _pack_level,
    _point_levels,
    _split_chains,
    as_linkno as _as_linkno,
    build_geoglows_pyramid,
    build_glofas_and_te_pyramids,
    extract_level_features as _level_features,
    load_geoglows_lookup,
    load_network_pyramid as _load_levels,
    lod_for_zoom as _lod,
    pyramid_signature as _signature,
    query_geoglows_reaches,
    resolve_cross_network_click,
    save_network_pyramid as _save_levels,
    snap_geoglows_reach_from_gpkg,
    snap_glofas_cell_from_network,
)

_LOG = logging.getLogger(__name__)


def _data_dir(name: str) -> Path:
  path = RIVER_NETWORKS_DIR / name
  return path if path.exists() else CACHE_DIR / name


GLOFAS_DIR = _data_dir("glofas_v4")
GEOGLOWS_DIR = _data_dir("geoglows_v2")
GEOGLOWS_GPKG = GEOGLOWS_DIR / "global_streams_simplified.gpkg"
GEOGLOWS_ATTRS = GEOGLOWS_DIR / "geoglows_attrs_v1.npz"

_LOCK = threading.Lock()
_NETWORKS: Dict[str, Dict[str, Any]] = {}
_GEOGLOWS_LOOKUP: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _build_glofas_and_te() -> Tuple[Dict[str, Any], Dict[str, Any]]:
  return build_glofas_and_te_pyramids(GLOFAS_DIR)


def _geoglows_lookup() -> Tuple[np.ndarray, np.ndarray]:
  """(sorted LINKNO, DSContArea km²) for reaches >= GEOGLOWS_MIN_AREA_KM2."""
  global _GEOGLOWS_LOOKUP
  if _GEOGLOWS_LOOKUP is None:
    _GEOGLOWS_LOOKUP = load_geoglows_lookup(GEOGLOWS_ATTRS, min_area_km2=GEOGLOWS_MIN_AREA_KM2)
  return _GEOGLOWS_LOOKUP


def _build_geoglows() -> Dict[str, Any]:
  return build_geoglows_pyramid(GEOGLOWS_DIR)


def _geoglows_reaches(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                      min_area: float) -> List[Dict[str, Any]]:
  return query_geoglows_reaches(
      GEOGLOWS_GPKG,
      _geoglows_lookup(),
      min_lon,
      min_lat,
      max_lon,
      max_lat,
      min_area,
  )


def _cache_path(model: str) -> Path:
  return (GEOGLOWS_DIR if model == "geoglows" else GLOFAS_DIR) / f"{model}_network_v{_CACHE_VERSION}.npz"


def _network(model: str) -> Optional[Dict[str, Any]]:
  """Loads (building and caching on first use) the gridded/merged network for glofas, todays_earth or geoglows."""
  if model in _NETWORKS:
    return _NETWORKS[model]
  table = {"glofas": GLOFAS_LOD, "todays_earth": TE_LOD, "geoglows": GEOGLOWS_LOD}[model]
  with _LOCK:
    if model in _NETWORKS:
      return _NETWORKS[model]
    net = _load_levels(_cache_path(model), _signature(table))
    if net is None:
      try:
        if model == "geoglows":
          net = _build_geoglows()
          _save_levels(_cache_path(model), net["levels"], _signature(table))
        else:
          glofas, te = _build_glofas_and_te()
          _save_levels(_cache_path("glofas"), glofas["levels"], _signature(GLOFAS_LOD),
                       cell_lin=glofas["cell_lin"], cell_area=glofas["cell_area"])
          _save_levels(_cache_path("todays_earth"), te["levels"], _signature(TE_LOD))
          _NETWORKS["glofas"], _NETWORKS["todays_earth"] = glofas, te
          net = _NETWORKS[model]
      except (OSError, ValueError, KeyError, ImportError) as e:
        _LOG.warning("River network for %s unavailable: %s", model, e)
        return None
    _NETWORKS[model] = net
    return net


def _floodhub_features(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                       zoom: int) -> Tuple[List[Dict[str, Any]], int]:
  from frontend import river_indexer  # pylint: disable=g-import-not-at-top

  min_order = _lod(FLOODHUB_LOD, zoom)[1]
  gj = river_indexer.HydroRiverNetwork("hydroatlas").get_rivers_in_bbox(
      min_lon, min_lat, max_lon, max_lat, zoom=zoom, min_stream_order=min_order, prefer_cache=True)
  feats = []
  for f in gj.get("features") or []:
    p = f.get("properties") or {}
    reach_id = p.get("reach_id")
    feats.append({"type": "Feature", "geometry": f.get("geometry"),
                  "properties": {"river_id": reach_id, "upstream_area_km2": p.get("upstream_area_km2"),
                                 "stream_order": p.get("stream_order")}})
  return feats, min_order


def get_model_network(model: str, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                      zoom: int) -> Dict[str, Any]:
  """GeoJSON FeatureCollection of `model`'s river network in a bounding box at a zoom-dependent level of detail."""
  model = model if model in MODELS else "floodhub"
  min_lon, max_lon = max(-180.0, min_lon), min(180.0, max_lon)
  min_lat, max_lat = max(-85.0, min_lat), min(85.0, max_lat)
  if min_lon >= max_lon:
    min_lon, max_lon = -180.0, 180.0
  if min_lat >= max_lat:
    min_lat, max_lat = -85.0, 85.0
  props: Dict[str, Any] = {"model": model, "network": NETWORK_LABELS[model], "zoom": zoom}
  feats: List[Dict[str, Any]] = []
  if model == "floodhub":
    feats, props["min_stream_order"] = _floodhub_features(min_lon, min_lat, max_lon, max_lat, zoom)
  else:
    table = {"glofas": GLOFAS_LOD, "todays_earth": TE_LOD, "geoglows": GEOGLOWS_LOD}[model]
    row = _lod(table, zoom)
    props["min_area_km2"] = row[1]
    try:
      if row[2] is None:
        feats = _geoglows_reaches(min_lon, min_lat, max_lon, max_lat, row[1])
      else:
        net = _network(model)
        if net is None:
          props["available"] = False
        else:
          level = net["levels"][[r for r in table if r[2] is not None].index(row)]
          feats = _level_features(level, min_lon, min_lat, max_lon, max_lat)
    except (OSError, ValueError, ImportError) as e:
      _LOG.warning("River network query failed for %s: %s", model, e)
      props["available"] = False
  props["feature_count"] = len(feats)
  return {"type": "FeatureCollection", "properties": props, "features": feats}


def snap_glofas_cell(lat: float, lon: float, target_area_km2: float,
                     area_range: Optional[Tuple[float, float]] = None,
                     radius_cells: int = 3) -> Optional[Dict[str, Any]]:
  """GloFAS river cell near (lat, lon) matching an upstream area (radius 3 cells ≈ 16 km)."""
  if not target_area_km2 or target_area_km2 <= 0:
    return None
  net = _network("glofas")
  return snap_glofas_cell_from_network(
      net,
      lat,
      lon,
      target_area_km2=target_area_km2,
      area_range=area_range,
      radius_cells=radius_cells,
  )


def snap_geoglows_reach(lat: float, lon: float, target_area_km2: float,
                        area_range: Optional[Tuple[float, float]] = None,
                        radius_deg: float = 0.15) -> Optional[Dict[str, Any]]:
  """GEOGLOWS reach near (lat, lon) matching an upstream area."""
  if not target_area_km2 or target_area_km2 <= 0 or not GEOGLOWS_GPKG.exists():
    return None
  return snap_geoglows_reach_from_gpkg(
      GEOGLOWS_GPKG,
      _geoglows_lookup(),
      lat,
      lon,
      target_area_km2=target_area_km2,
      area_range=area_range,
      radius_deg=radius_deg,
  )


def resolve_click(lat: float, lon: float, upstream_area_km2: Any, area_min_km2: Any = None,
                  network: Optional[str] = None, river_id: Any = None) -> Optional[Dict[str, Any]]:
  """Maps a click on one model's river line to every model's forecast element on the same river."""
  return resolve_cross_network_click(
      lat,
      lon,
      upstream_area_km2,
      snap_glofas_fn=snap_glofas_cell,
      snap_geoglows_fn=snap_geoglows_reach,
      area_min_km2=area_min_km2,
      network=network,
      river_id=river_id,
  )


def build_all() -> None:
  """Builds every derived network cache (run once after downloading the data)."""
  for model in ("glofas", "todays_earth", "geoglows"):
    t = time.time()
    net = _network(model)
    sizes = [len(lvl["offsets"]) - 1 for lvl in net["levels"]] if net else None
    print(f"{model}: lines per level {sizes} ({time.time() - t:.1f} s)", flush=True)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  if "--build" in sys.argv:
    build_all()
