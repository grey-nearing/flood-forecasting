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
    FLOODHUB_PYRAMID_LOD,
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
    build_floodhub_pyramid,
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
    snap_geoglows_reach_from_network,
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


def _find_hydrorivers_shp() -> Optional[Path]:
  for base in (RIVER_NETWORKS_DIR, CACHE_DIR, CACHE_DIR.parent):
    cand = base / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10.shp"
    if cand.exists():
      return cand
  return None


def _find_floodhub_lut_path() -> Optional[Path]:
  for base in (RIVER_NETWORKS_DIR, CACHE_DIR, CACHE_DIR.parent):
    cand = base / "hydrorivers_floodhub_v1.npz"
    if cand.exists():
      return cand
  return None


def _build_floodhub() -> Dict[str, Any]:
  shp_path = _find_hydrorivers_shp()
  if shp_path is None:
    raise FileNotFoundError("HydroRIVERS_v10.shp not found")
  return build_floodhub_pyramid(shp_path, _find_floodhub_lut_path())


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
  """Loads (building and caching on first use) the gridded/merged network for floodhub, glofas, todays_earth or geoglows."""
  if model in _NETWORKS:
    return _NETWORKS[model]
  table = {
      "floodhub": FLOODHUB_PYRAMID_LOD,
      "glofas": GLOFAS_LOD,
      "todays_earth": TE_LOD,
      "geoglows": GEOGLOWS_LOD,
  }[model]
  with _LOCK:
    if model in _NETWORKS:
      return _NETWORKS[model]
    net = _load_levels(_cache_path(model), _signature(table))
    if net is None:
      try:
        if model == "geoglows":
          net = _build_geoglows()
          extra = {k: v for k, v in net.items() if k != "levels"}
          _save_levels(_cache_path(model), net["levels"], _signature(table), **extra)
        elif model == "floodhub":
          net = _build_floodhub()
          extra = {k: v for k, v in net.items() if k != "levels"}
          _save_levels(_cache_path(model), net["levels"], _signature(table), **extra)
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


_FLOODHUB_LUT: Optional[Dict[str, Any]] = None


def _get_floodhub_lut() -> Optional[Dict[str, Any]]:
  """Lazily loads the pre-cached HydroRIVERS-to-FloodHub lookup tables into memory."""
  global _FLOODHUB_LUT
  if _FLOODHUB_LUT is not None:
    return _FLOODHUB_LUT
  with _LOCK:
    if _FLOODHUB_LUT is not None:
      return _FLOODHUB_LUT
    hr_path = _find_floodhub_lut_path()
    fh_path = None
    for base in (RIVER_NETWORKS_DIR, CACHE_DIR, CACHE_DIR.parent):
      cand_fh = base / "floodhub_gauges_v1.npz"
      if fh_path is None and cand_fh.exists():
        fh_path = cand_fh
    if hr_path is None:
      return None
    try:
      with np.load(hr_path, allow_pickle=False) as z:
        hyriv_id = z["hyriv_id"].astype(np.int32)
        hybas_l12 = z["hybas_l12"].astype(np.int64)
        has_fc = z["has_forecast"].astype(np.bool_)
      hybas_to_sev: Dict[int, int] = {}
      phys_lats = np.empty(0, dtype=np.float32)
      phys_lons = np.empty(0, dtype=np.float32)
      phys_gids = np.empty(0, dtype="U32")
      phys_sevs = np.empty(0, dtype=np.int8)
      if fh_path is not None:
        with np.load(fh_path, allow_pickle=False) as fz:
          fh_hb = fz["hybas_id"].astype(np.int64)
          fh_sev = fz["severity_rank"].astype(np.int8)
          fh_fc = fz["has_forecast"].astype(np.bool_)
          for hb_val, s_val in zip(fh_hb[fh_fc & (fh_hb > 0) & (fh_sev > 0)],
                                   fh_sev[fh_fc & (fh_hb > 0) & (fh_sev > 0)]):
            hybas_to_sev[int(hb_val)] = int(s_val)
          phys_mask = fh_fc & (fh_hb == 0)
          phys_lats = fz["lat"][phys_mask].astype(np.float32)
          phys_lons = fz["lon"][phys_mask].astype(np.float32)
          phys_gids = fz["gauge_id"][phys_mask]
          phys_sevs = fh_sev[phys_mask]
      _FLOODHUB_LUT = {
          "hyriv_id": hyriv_id,
          "hybas_l12": hybas_l12,
          "has_forecast": has_fc,
          "hybas_to_sev": hybas_to_sev,
          "phys_lats": phys_lats,
          "phys_lons": phys_lons,
          "phys_gids": phys_gids,
          "phys_sevs": phys_sevs,
      }
      return _FLOODHUB_LUT
    except Exception as e:  # pylint: disable=broad-except
      _LOG.warning("Failed to load FloodHub HydroRIVERS lookup cache: %s", e)
      return None


def _floodhub_features(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                       zoom: int) -> Tuple[List[Dict[str, Any]], int, int]:
  min_order = _lod(FLOODHUB_LOD, zoom)[1]
  lut = _get_floodhub_lut()
  hybas_to_sev = lut["hybas_to_sev"] if lut is not None else {}

  net = _network("floodhub")
  if net is not None:
    row = _lod(FLOODHUB_PYRAMID_LOD, zoom)
    level_idx = [r for r in FLOODHUB_PYRAMID_LOD if r[2] is not None].index(row)
    feats = _level_features(
        net["levels"][level_idx],
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        hybas_to_sev=hybas_to_sev,
    )
    fc_count = sum(1 for f in feats if (f.get("properties") or {}).get("has_forecast"))
    return feats, min_order, fc_count

  from frontend import river_indexer  # pylint: disable=g-import-not-at-top

  gj = river_indexer.HydroRiverNetwork("hydroatlas").get_rivers_in_bbox(
      min_lon, min_lat, max_lon, max_lat, zoom=zoom, min_stream_order=min_order, prefer_cache=True)
  raw_feats = gj.get("features") or []

  rids = np.zeros(len(raw_feats), dtype=np.int32)
  if lut is not None and raw_feats:
    for i, f in enumerate(raw_feats):
      rid_str = str((f.get("properties") or {}).get("reach_id") or "").replace("HYRIV_", "")
      if rid_str.isdigit():
        rids[i] = int(rid_str)
    hyriv_arr = lut["hyriv_id"]
    pos = np.searchsorted(hyriv_arr, rids)
    safe_pos = np.minimum(pos, len(hyriv_arr) - 1)
    matched = (rids > 0) & (pos < len(hyriv_arr)) & (hyriv_arr[safe_pos] == rids)
    has_fc_arr = matched & lut["has_forecast"][safe_pos]
    hybas_arr = np.where(matched, lut["hybas_l12"][safe_pos], 0)
  else:
    has_fc_arr = np.ones(len(raw_feats), dtype=np.bool_)
    hybas_arr = np.zeros(len(raw_feats), dtype=np.int64)

  feats = []
  fc_count = 0
  for i, f in enumerate(raw_feats):
    p = f.get("properties") or {}
    reach_id = p.get("reach_id")
    has_fc = bool(has_fc_arr[i])
    hb_id = int(hybas_arr[i])
    gid = f"hybas_{hb_id}" if (has_fc and hb_id > 0) else None
    sev_rank = int(hybas_to_sev.get(hb_id, 0)) if has_fc else 0
    if has_fc:
      fc_count += 1
    feats.append({
        "type": "Feature",
        "geometry": f.get("geometry"),
        "properties": {
            "river_id": reach_id,
            "upstream_area_km2": p.get("upstream_area_km2"),
            "stream_order": p.get("stream_order"),
            "has_forecast": has_fc,
            "gauge_id": gid,
            "severity_rank": sev_rank,
        },
    })
  return feats, min_order, fc_count


def _promote_active_flood_features(
    model: str,
    feats: List[Dict[str, Any]],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int,
    active_keys: set[str],
    ows_grid: Optional[np.ndarray] = None,
) -> None:
  """Promote active flood reaches (`exceedance_rank >= 1`) from finer pyramid levels into coarse zoom views."""
  if zoom >= 8:
    return
  if model in ("floodhub", "geoglows") and not active_keys:
    return
  if model in ("glofas", "todays_earth") and ows_grid is None and not active_keys:
    return
  net = _network(model)
  if net is None or not net.get("levels"):
    return
  levels = net["levels"]
  # Pick a finer pyramid level (down to ~1,000-2,500 km²) so regional floods pop on the world map
  fine_lvl = levels[min(len(levels) - 1, 2)]
  b = fine_lvl["bbox"]
  in_box = (
      (b[:, 2] >= min_lon)
      & (b[:, 0] <= max_lon)
      & (b[:, 3] >= min_lat)
      & (b[:, 1] <= max_lat)
  )
  offsets = fine_lvl["offsets"]
  coords = fine_lvl["coords"]
  amax = fine_lvl["amax"]
  amin = fine_lvl["amin"]
  rids = fine_lvl.get("river_id")
  hbs = fine_lvl.get("hybas_l12")

  if model == "floodhub":
    existing_gids = {
        str((f.get("properties") or {}).get("gauge_id") or "")
        for f in feats
    }
    needed_hb = [
        int(k[6:])
        for k in active_keys
        if k.startswith("hybas_") and k[6:].isdigit() and k not in existing_gids
    ]
    if not needed_hb or hbs is None:
      return
    hit = np.flatnonzero(
        in_box & np.isin(hbs, np.asarray(needed_hb, dtype=np.int64))
    )
  elif model == "geoglows":
    existing_rids = {
        int((f.get("properties") or {}).get("river_id") or 0)
        for f in feats
        if (f.get("properties") or {}).get("river_id") is not None
    }
    needed_rids = [
        int(k)
        for k in active_keys
        if k.isdigit() and int(k) not in existing_rids
    ]
    if not needed_rids or rids is None:
      return
    hit = np.flatnonzero(
        in_box & np.isin(rids, np.asarray(needed_rids, dtype=np.int32))
    )
  else:
    if ows_grid is None:
      return
    starts = offsets[:-1]
    ends = np.maximum(starts, offsets[1:] - 1)
    mids = (starts + ends) // 2
    ranks = np.zeros(len(starts), dtype=np.uint8)
    for idx_arr in (starts, mids, ends):
      r_i = np.clip(((75.0 - coords[idx_arr, 1]) / 0.05).astype(np.int32), 0, 2699)
      c_i = np.clip(((coords[idx_arr, 0] + 180.0) / 0.05).astype(np.int32), 0, 7199)
      ranks = np.maximum(ranks, ows_grid[r_i, c_i])
    hit = np.flatnonzero(in_box & (ranks >= 1))

  if not hit.size:
    return
  hit = hit[:250]

  existing_end_keys: set[tuple[int, int]] = set()
  if model in ("glofas", "todays_earth"):
    for f in feats:
      fc = (f.get("geometry") or {}).get("coordinates") or []
      if fc:
        existing_end_keys.add((int(round(fc[-1][0] * 100)), int(round(fc[-1][1] * 100))))

  for idx_val in hit.tolist():
    s = int(offsets[idx_val])
    e = int(offsets[idx_val + 1])
    if e - s < 2:
      continue
    line_coords = np.round(coords[s:e].astype(np.float64), 4).tolist()
    if model == "floodhub":
      hb_id = int(hbs[idx_val]) if hbs is not None else 0
      rid_val = int(rids[idx_val]) if rids is not None else 0
      feats.append({
          "type": "Feature",
          "geometry": {"type": "LineString", "coordinates": line_coords},
          "properties": {
              "upstream_area_km2": round(float(amax[idx_val]), 1),
              "area_min_km2": round(float(amin[idx_val]), 1),
              "river_id": f"HYRIV_{rid_val}" if rid_val > 0 else None,
              "has_forecast": True,
              "gauge_id": f"hybas_{hb_id}" if hb_id > 0 else None,
              "promoted_flood": True,
          },
      })
    elif model == "geoglows":
      rid_val = int(rids[idx_val]) if rids is not None else 0
      feats.append({
          "type": "Feature",
          "geometry": {"type": "LineString", "coordinates": line_coords},
          "properties": {
              "upstream_area_km2": round(float(amax[idx_val]), 1),
              "area_min_km2": round(float(amin[idx_val]), 1),
              "river_id": rid_val,
              "promoted_flood": True,
          },
      })
    else:
      end_key = (int(round(line_coords[-1][0] * 100)), int(round(line_coords[-1][1] * 100)))
      if end_key in existing_end_keys:
        continue
      existing_end_keys.add(end_key)
      feats.append({
          "type": "Feature",
          "geometry": {"type": "LineString", "coordinates": line_coords},
          "properties": {
              "upstream_area_km2": round(float(amax[idx_val]), 1),
              "area_min_km2": round(float(amin[idx_val]), 1),
              "promoted_flood": True,
          },
      })


def get_model_network(model: str, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                      zoom: int, *, refresh: bool = False) -> Dict[str, Any]:
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
    props["min_area_km2"] = _lod(FLOODHUB_PYRAMID_LOD, zoom)[1]
    feats, props["min_stream_order"], props["forecast_feature_count"] = _floodhub_features(
        min_lon, min_lat, max_lon, max_lat, zoom
    )
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
  if feats:
    try:
      from frontend.maas_live_status import (  # pylint: disable=g-import-not-at-top
          clear_live_status_cache,
          enrich_network_features_live,
          get_active_flood_keys,
          get_glofas_ows_grid,
          get_last_refresh_iso,
      )
      if refresh:
        clear_live_status_cache(model)
      elif zoom < 8:
        active_keys = get_active_flood_keys(model)
        ows_grid = get_glofas_ows_grid() if model in ("glofas", "todays_earth") else None
        if active_keys or ows_grid is not None:
          _promote_active_flood_features(
              model, feats, min_lon, min_lat, max_lon, max_lat, zoom, active_keys, ows_grid=ows_grid
          )
      colored_cnt, pending_cnt = enrich_network_features_live(model, feats)
      props["live_colored_count"] = colored_cnt
      props["live_pending_count"] = pending_cnt
      props["active_flood_count"] = sum(
          1 for f in feats if int((f.get("properties") or {}).get("exceedance_rank") or 0) >= 1
      )
      props["last_refreshed_utc"] = get_last_refresh_iso(model)
    except Exception as e:  # pylint: disable=broad-except
      _LOG.debug("Live forecast status enrichment skipped for %s: %s", model, e)
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
  if not target_area_km2 or target_area_km2 <= 0:
    return None
  net = _network("geoglows")
  snapped = snap_geoglows_reach_from_network(
      net,
      lat,
      lon,
      target_area_km2=target_area_km2,
      area_range=area_range,
      radius_deg=radius_deg,
  )
  if snapped is not None:
    return snapped
  if not GEOGLOWS_GPKG.exists():
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
  for model in ("glofas", "todays_earth", "geoglows", "floodhub"):
    t = time.time()
    net = _network(model)
    sizes = [len(lvl["offsets"]) - 1 for lvl in net["levels"]] if net else None
    print(f"{model}: lines per level {sizes} ({time.time() - t:.1f} s)", flush=True)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  if "--build" in sys.argv:
    build_all()
