"""River snapping and catchment delineation supporting official terrain ridgeline polygons (MERIT-Basins & HydroBASINS) and channel network corridor envelopes."""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping
from shapely.ops import unary_union

from frontend.config import (
    DATA_DIR,
    HYDRO_BASINS_DIR,
    HYDRO_DATASETS,
    RIVER_NETWORKS_DIR,
    ensure_flood_forecasting_on_sys_path,
    resolve_hydro_dataset_id,
)

ensure_flood_forecasting_on_sys_path()

CACHE_DIR = Path.home() / ".cache" / "openhydronet" / "data"
PERM_POLYGONS_DIR = Path.home() / "data" / "input" / "polygons"

# HydroSHEDS / HydroATLAS
HYDRORIVERS_SHP = PERM_POLYGONS_DIR / "hydrorivers" / "HydroRIVERS_v10.shp"
if not HYDRORIVERS_SHP.exists():
  HYDRORIVERS_SHP = RIVER_NETWORKS_DIR / "hydrorivers" / "HydroRIVERS_v10.shp"
if not HYDRORIVERS_SHP.exists():
  HYDRORIVERS_SHP = CACHE_DIR / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10.shp"

HYDROBASINS_DIR = PERM_POLYGONS_DIR / "hydrobasins"
if not HYDROBASINS_DIR.exists():
  HYDROBASINS_DIR = HYDRO_BASINS_DIR / "hydrobasins_units"
if not HYDROBASINS_DIR.exists():
  HYDROBASINS_DIR = CACHE_DIR / "hydrobasins"

HYDROBASINS_INDEX_FILE = HYDROBASINS_DIR / "hydrobasins_partitions_index.json"

# MERIT Hydro / MERIT-Basins
MERIT_DIR = PERM_POLYGONS_DIR / "merit_basins"
if not MERIT_DIR.exists():
  MERIT_DIR = RIVER_NETWORKS_DIR / "merit_rivers"
if not MERIT_DIR.exists():
  MERIT_DIR = CACHE_DIR / "merit_basins" / "ms_riv_network"

MERIT_INDEX_FILE = PERM_POLYGONS_DIR / "merit_basins" / "merit_partitions_index.json"
if not MERIT_INDEX_FILE.exists():
  MERIT_INDEX_FILE = RIVER_NETWORKS_DIR / "merit_partitions_index.json"
if not MERIT_INDEX_FILE.exists():
  MERIT_INDEX_FILE = CACHE_DIR / "merit_basins" / "merit_partitions_index.json"

_MERIT_PARTITIONS: Optional[List[Dict[str, Any]]] = None
_HYDROBASINS_PARTITIONS: Optional[List[Dict[str, Any]]] = None


def _get_merit_partitions() -> List[Dict[str, Any]]:
  global _MERIT_PARTITIONS
  if _MERIT_PARTITIONS is None and MERIT_INDEX_FILE.exists():
    try:
      with open(MERIT_INDEX_FILE, "r") as f:
        _MERIT_PARTITIONS = json.load(f)
    except Exception as e:
      print(f"Error loading MERIT partitions: {e}")
  return _MERIT_PARTITIONS or []


def _find_merit_shp(comid: int, prefix: str = "cat") -> Optional[Path]:
  """Locates the Pfafstetter shapefile containing the given COMID."""
  comid_str = str(comid)
  pfaf_2 = comid_str[:2]
  pfaf_1 = comid_str[:1]
  for base_dir in [
      PERM_POLYGONS_DIR / "merit_basins",
      CACHE_DIR / "merit_basins",
  ]:
    if not base_dir.exists():
      continue
    matches = list(base_dir.glob(f"**/{prefix}_pfaf_{pfaf_2}_*.shp"))
    if matches:
      return matches[0]
    matches = list(base_dir.glob(f"**/{prefix}_pfaf_{pfaf_1}_*.shp"))
    if matches:
      return matches[0]
  return None


def _get_hydrobasins_partitions() -> List[Dict[str, Any]]:
  global _HYDROBASINS_PARTITIONS
  if _HYDROBASINS_PARTITIONS is None and HYDROBASINS_INDEX_FILE.exists():
    try:
      with open(HYDROBASINS_INDEX_FILE, "r") as f:
        _HYDROBASINS_PARTITIONS = json.load(f)
    except Exception as e:
      print(f"Error loading HydroBASINS partitions: {e}")
  return _HYDROBASINS_PARTITIONS or []


class HydroDelineator:
  """Manages authentic river snapping and official catchment delineation across hydrography datasets."""

  def __init__(self, dataset_id: str = "hydroatlas"):
    resolved_id = resolve_hydro_dataset_id(dataset_id)
    if resolved_id not in HYDRO_DATASETS:
      raise ValueError(
          f"Unknown dataset '{dataset_id}'. Choose from:"
          f" {list(HYDRO_DATASETS.keys())}"
      )
    self.dataset_id = resolved_id
    self.dataset_meta = HYDRO_DATASETS[resolved_id]
    self.dem_id = self.dataset_meta.get("dem_id", "hydrosheds_90m")
    self.is_merit = resolved_id == "merit-hydro"

  def snap_to_river(
      self, lat: float, lon: float, snap_radius_km: Optional[float] = None
  ) -> Tuple[float, float, float, Dict[str, Any]]:
    """Snaps click coordinates to the nearest authentic river reach."""
    radius_km = snap_radius_km or self.dataset_meta["default_snap_radius_km"]
    deg_radius = radius_km / 111.0
    click_pt = Point(lon, lat)
    search_bbox = (
        lon - deg_radius,
        lat - deg_radius,
        lon + deg_radius,
        lat + deg_radius,
    )

    # CASE A: MERIT Hydro (90m MERIT-Basins)
    if self.is_merit:
      partitions = _get_merit_partitions()
      best_row = None
      best_shp_path = None
      min_score = float("inf")
      best_proj_pt = None

      for part in partitions:
        p_bbox = part.get("bbox", [])
        if len(p_bbox) != 4 or any(np.isnan(p_bbox)):
          continue
        p_minx, p_miny, p_maxx, p_maxy = p_bbox
        if not (
            search_bbox[2] < p_minx
            or search_bbox[0] > p_maxx
            or search_bbox[3] < p_miny
            or search_bbox[1] > p_maxy
        ):
          shp_path = part["path"]
          try:
            gdf = gpd.read_file(shp_path, bbox=search_bbox, engine="pyogrio")
            for _, row in gdf.iterrows():
              geom = row["geometry"]
              if geom is None or geom.is_empty:
                continue
              dist_deg = geom.distance(click_pt)
              order = int(row.get("order", 1))
              up_area = float(row.get("uparea", 1.0))
              score = dist_deg / (1.0 + math.log10(max(1.0, up_area)) * 0.25)
              if score < min_score:
                min_score = score
                best_row = row
                best_shp_path = shp_path
                proj_dist = geom.project(click_pt)
                best_proj_pt = geom.interpolate(proj_dist)
          except Exception:
            pass

      if best_row is not None and best_proj_pt is not None:
        dist_m = float(best_proj_pt.distance(click_pt) * 111000.0)
        max_snap_m = (
            (snap_radius_km * 1000.0) if snap_radius_km is not None else 250.0
        )
        if dist_m <= max_snap_m:
          comid = int(best_row.get("COMID", 0))
          attrs = {
              "reach_id": f"MERIT_{comid}",
              "dataset": "merit-hydro",
              "river_name": f"MERIT River Reach (COMID {comid})",
              "stream_order": int(best_row.get("order", 1)),
              "upstream_area_km2": round(float(best_row.get("uparea", 0.0)), 1),
              "sinuosity": round(float(best_row.get("sinuosity", 1.0)), 2),
              "slope": round(float(best_row.get("slope", 0.0)), 4),
              "length_km": round(float(best_row.get("lengthkm", 0.0)), 2),
              "next_down": int(best_row.get("NextDownID", 0)),
              "shp_path": best_shp_path,
              "reach_geometry": (
                  mapping(best_row["geometry"])
                  if best_row["geometry"] is not None
                  else None
              ),
          }
          return float(best_proj_pt.y), float(best_proj_pt.x), dist_m, attrs

    # CASE B: HydroATLAS (HydroSHEDS)
    elif HYDRORIVERS_SHP.exists():
      try:
        gdf = gpd.read_file(
            str(HYDRORIVERS_SHP), bbox=search_bbox, engine="pyogrio"
        )
        if not gdf.empty:
          best_row = None
          min_score = float("inf")
          best_proj_pt = None

          for _, row in gdf.iterrows():
            geom = row["geometry"]
            if geom is None or geom.is_empty:
              continue
            dist_deg = geom.distance(click_pt)
            order = int(row.get("ORD_STRA", 1))
            up_area = float(row.get("UPLAND_SKM", 1.0))
            score = dist_deg / (1.0 + math.log10(max(1.0, up_area)) * 0.25)
            if score < min_score:
              min_score = score
              best_row = row
              proj_dist = geom.project(click_pt)
              best_proj_pt = geom.interpolate(proj_dist)

          if best_row is not None and best_proj_pt is not None:
            dist_m = float(best_proj_pt.distance(click_pt) * 111000.0)
            max_snap_m = (
                (snap_radius_km * 1000.0)
                if snap_radius_km is not None
                else 250.0
            )
            if dist_m <= max_snap_m:
              hyriv_id = int(best_row.get("HYRIV_ID", 0))
              river_class = int(
                  best_row.get("ORD_CLAS", best_row.get("ORD_STRA", 1))
              )
              hybas_id = int(best_row.get("HYBAS_L12", 0))
              attrs = {
                  "reach_id": f"HYRIV_{hyriv_id}",
                  "dataset": "hydroatlas",
                  "river_name": (
                      f"HydroATLAS Sub-Basin {hybas_id} (Class {river_class})"
                  ),
                  "stream_order": int(best_row.get("ORD_STRA", 1)),
                  "river_class": river_class,
                  "hydrobasins_unit": hybas_id,
                  "upstream_area_km2": round(
                      float(best_row.get("UPLAND_SKM", 0.0)), 1
                  ),
                  "mean_discharge_m3s": round(
                      float(best_row.get("DIS_AV_CMS", 0.0)), 2
                  ),
                  "length_km": round(float(best_row.get("LENGTH_KM", 0.0)), 2),
                  "dist_to_ocean_km": round(
                      float(best_row.get("DIST_DN_KM", 0.0)), 1
                  ),
                  "next_down": int(best_row.get("NEXT_DOWN", 0)),
                  "main_river": int(best_row.get("MAIN_RIV", 0)),
                  "reach_geometry": (
                      mapping(best_row["geometry"])
                      if best_row["geometry"] is not None
                      else None
                  ),
              }
              return float(best_proj_pt.y), float(best_proj_pt.x), dist_m, attrs
      except Exception as e:
        print(f"Error snapping in HydroRIVERS: {e}")

    # Fallback for arbitrary clicked coordinates
    return (
        lat,
        lon,
        0.0,
        {
            "reach_id": f"POINT_{abs(int(lat * 1000))}_{abs(int(lon * 1000))}",
            "dataset": self.dataset_id,
            "river_name": f"Selected Location ({lat:.4f}°N, {lon:.4f}°E)",
            "stream_order": 1,
            "upstream_area_km2": 50.0,
        },
    )

  def delineate_catchment(
      self,
      lat: float,
      lon: float,
      snap_radius_km: Optional[float] = None,
      mode: str = "official_ridgeline",
  ) -> Dict[str, Any]:
    """Delineates the contributing drainage polygon for an outlet reach.

    Args:
        lat: Outlet latitude.
        lon: Outlet longitude.
        snap_radius_km: Snap tolerance in km.
        mode: 'official_ridgeline' (dissolved official unit catchment polygons)
          or 'channel_corridor' (stream network buffer envelope).

    Returns:
        GeoJSON Feature containing the delineated watershed polygon.
    """
    snapped_lat, snapped_lon, snap_dist_m, reach_attrs = self.snap_to_river(
        lat, lon, snap_radius_km
    )

    reach_id = reach_attrs.get("reach_id", "REACH_0")
    upstream_area = reach_attrs.get("upstream_area_km2", 250.0)
    stream_order = reach_attrs.get("stream_order", 3)

    polygon = None
    upstream_count = 0
    method_name = "Official Terrain Ridgeline Catchment"

    # -------------------------------------------------------------
    # CASE 0: Pure DEM Elevation Flow-Routing (90m Terrain Grid)
    # -------------------------------------------------------------
    if mode == "dem_flow_direction":
      from frontend.dem_delineator import DemDelineator

      dem_delin = DemDelineator(dem_id=self.dem_id)
      dem_feature = dem_delin.delineate(
          lat=snapped_lat,
          lon=snapped_lon,
          snap_window_cells=4,
      )
      dem_props = dem_feature.setdefault("properties", {})
      outlet_info = dem_props.setdefault("outlet", {})
      outlet_lat = float(outlet_info.get("latitude", snapped_lat))
      outlet_lon = float(outlet_info.get("longitude", snapped_lon))
      clean_dataset = self.dataset_id.replace("-", "_")
      catchment_id = (
          f"catchment_{clean_dataset}_{reach_id}_{abs(int(outlet_lat * 1000))}_{abs(int(outlet_lon * 1000))}"
      )
      dem_cell_id = outlet_info.get("reach_id", "DEM_0_0")
      outlet_info.update({
          "input_latitude": lat,
          "input_longitude": lon,
          "latitude": round(outlet_lat, 5),
          "longitude": round(outlet_lon, 5),
          "reach_id": reach_id,
          "dem_cell_id": dem_cell_id,
          "snap_distance_m": round(
              float(snap_dist_m or outlet_info.get("snap_distance_m", 0.0)), 1
          ),
      })
      merged_reach_attrs = dict(dem_props.get("reach_attributes") or {})
      merged_reach_attrs.update(reach_attrs)
      merged_reach_attrs["dem_cell_id"] = dem_cell_id
      merged_reach_attrs["upstream_area_km2"] = dem_props.get(
          "area_km2", upstream_area
      )
      dem_props.update({
          "catchment_id": catchment_id,
          "gauge_id": catchment_id,
          "dataset": self.dataset_id,
          "dataset_name": self.dataset_meta["name"],
          "dem_id": self.dem_id,
          "dem_name": self.dataset_meta.get(
              "dem_name", "HydroSHEDS 90m Conditioned DEM (3 arc-sec)"
          ),
          "river_network": self.dataset_meta.get(
              "river_network_name", self.dataset_meta["name"]
          ),
          "stream_order": stream_order,
          "reach_attributes": merged_reach_attrs,
          "upstream_reaches_count": max(
              1, int(dem_props.get("upstream_reaches_count", 1))
          ),
      })
      return dem_feature

    # -------------------------------------------------------------
    # CASE A: MERIT-Hydro Official Unit Catchment Delineation
    # -------------------------------------------------------------
    if self.is_merit:
      comid = None
      if reach_id.startswith("MERIT_"):
        try:
          comid = int(reach_id.replace("MERIT_", ""))
        except ValueError:
          pass

      # Fallback: Point-in-polygon containment lookup in MERIT catchments for arbitrary click points
      if comid is None:
        pt = Point(lon, lat)
        search_deg = 0.5
        s_bbox = (
            lon - search_deg,
            lat - search_deg,
            lon + search_deg,
            lat + search_deg,
        )
        for part in _get_merit_partitions():
          p_bbox = part.get("bbox", [])
          if len(p_bbox) == 4 and not any(np.isnan(p_bbox)):
            if not (
                s_bbox[2] < p_bbox[0]
                or s_bbox[0] > p_bbox[2]
                or s_bbox[3] < p_bbox[1]
                or s_bbox[1] > p_bbox[3]
            ):
              cat_shp = _find_merit_shp(
                  int(
                      part["name"]
                      .split("_")[2]
                      .replace("MERIT", "")
                      .replace(".", "")
                  ),
                  "cat",
              )
              if cat_shp and cat_shp.exists():
                try:
                  gdf_c = gpd.read_file(
                      str(cat_shp), bbox=s_bbox, engine="pyogrio"
                  )
                  cont = gdf_c[gdf_c.contains(pt)]
                  if not cont.empty:
                    comid = int(cont.iloc[0]["COMID"])
                    break
                except Exception:
                  pass

      if comid is not None:
        try:
          riv_shp_path = _find_merit_shp(comid, "riv")
          cat_shp_path = _find_merit_shp(comid, "cat")

          if riv_shp_path and riv_shp_path.exists():
            gdf_net = gpd.read_file(str(riv_shp_path), engine="pyogrio")
            down_to_up: Dict[int, List[int]] = {}
            for _, r in gdf_net.iterrows():
              cid = int(r["COMID"])
              nid = int(r["NextDownID"])
              down_to_up.setdefault(nid, []).append(cid)

            visited_comids: Set[int] = set()
            queue = [comid]
            while queue and len(visited_comids) < 3000:
              cur = queue.pop(0)
              if cur in visited_comids:
                continue
              visited_comids.add(cur)
              for up_id in down_to_up.get(cur, []):
                if up_id not in visited_comids:
                  queue.append(up_id)

            upstream_count = len(visited_comids)

            if cat_shp_path and cat_shp_path.exists():
              gdf_cat = gpd.read_file(str(cat_shp_path), engine="pyogrio")
              match_cats = gdf_cat[gdf_cat["COMID"].isin(visited_comids)]
              if not match_cats.empty:
                if mode == "exact_pour_point":
                  trib_ids = visited_comids - {comid}
                  trib_geoms = []
                  if trib_ids:
                    sub_trib = match_cats[match_cats["COMID"].isin(trib_ids)]
                    trib_geoms = [
                        g
                        for g in sub_trib["geometry"].values
                        if g is not None and not g.is_empty
                    ]

                  local_row = match_cats[match_cats["COMID"] == comid]
                  if not local_row.empty:
                    local_geom = local_row["geometry"].values[0]
                    reach_geom = reach_attrs.get("reach_geometry")
                    clipped_local = self._clip_local_catchment_to_pour_point(
                        local_geom, reach_geom, snapped_lat, snapped_lon
                    )
                    dissolved = unary_union(trib_geoms + [clipped_local])
                  else:
                    dissolved = unary_union([
                        g
                        for g in match_cats["geometry"].values
                        if g is not None and not g.is_empty
                    ])

                  polygon = dissolved
                  method_name = "MERIT-Basins Exact Pour-Point Drainage Basin"

                  lat_scale = 111.0
                  lon_scale = 111.0 * math.cos(math.radians(snapped_lat))
                  upstream_area = round(
                      float(polygon.area * lat_scale * lon_scale), 1
                  )
                elif mode == "unit_catchment":
                  local_row = match_cats[match_cats["COMID"] == comid]
                  if not local_row.empty:
                    polygon = local_row["geometry"].values[0]
                    method_name = "MERIT-Basins Official Unit Catchment Polygon"
                    upstream_area = round(float(local_row.get("unitarea", local_row.get("uparea", [1.0])).values[0]), 1)
                    upstream_count = 1
                else:
                  cat_geoms = [
                      g
                      for g in match_cats["geometry"].values
                      if g is not None and not g.is_empty
                  ]
                  if cat_geoms:
                    dissolved = unary_union(cat_geoms)
                    polygon = dissolved
                    method_name = (
                        "MERIT-Basins Official Unit Ridgeline Watershed"
                    )

        except Exception as e:
          print(f"MERIT catchment delineation error: {e}")

    # -------------------------------------------------------------
    # CASE B: HydroATLAS / HydroBASINS Sub-Basin Delineation (Reaches & Arbitrary Lat/Lon Points)
    # -------------------------------------------------------------
    elif HYDRORIVERS_SHP.exists():
      hyriv_str = reach_id.replace("HYRIV_", "")
      hybas_id = reach_attrs.get("hydrobasins_unit")

      try:
        # Find matching HydroBASINS continental shapefile
        hb_partitions = _get_hydrobasins_partitions()
        target_hb_shp = None
        for part in hb_partitions:
          p_bbox = part.get("bbox", [])
          if len(p_bbox) == 4 and not any(np.isnan(p_bbox)):
            if (p_bbox[0] <= snapped_lon <= p_bbox[2]) and (
                p_bbox[1] <= snapped_lat <= p_bbox[3]
            ):
              target_hb_shp = part["path"]
              break

        if target_hb_shp and Path(target_hb_shp).exists():
          # Fast spatial bounding box query using shapefile spatial index (.sbn)
          search_bbox = (snapped_lon - 0.08, snapped_lat - 0.08, snapped_lon + 0.08, snapped_lat + 0.08)
          try:
            gdf_candidate = gpd.read_file(str(target_hb_shp), bbox=search_bbox, engine="pyogrio")
          except Exception:
            gdf_candidate = gpd.GeoDataFrame()

          if gdf_candidate.empty:
            search_bbox_large = (snapped_lon - 0.5, snapped_lat - 0.5, snapped_lon + 0.5, snapped_lat + 0.5)
            try:
              gdf_candidate = gpd.read_file(str(target_hb_shp), bbox=search_bbox_large, engine="pyogrio")
            except Exception:
              gdf_candidate = gpd.GeoDataFrame()

          # If hybas_id not known from river reach, locate containing unit polygon directly by lat/lon
          if not hybas_id:
            click_pt = Point(snapped_lon, snapped_lat)
            containing = gdf_candidate[gdf_candidate.contains(click_pt)] if not gdf_candidate.empty else gdf_candidate
            if not containing.empty:
              hybas_id = int(containing.iloc[0]["HYBAS_ID"])
              match_row = containing.iloc[:1]
            elif not gdf_candidate.empty:
              # Nearest polygon fallback
              gdf_candidate["dist_deg"] = gdf_candidate.distance(click_pt)
              match_row = gdf_candidate.sort_values("dist_deg").iloc[:1]
              hybas_id = int(match_row["HYBAS_ID"].values[0])
            else:
              match_row = gpd.GeoDataFrame()
          else:
            match_row = gdf_candidate[gdf_candidate["HYBAS_ID"] == hybas_id] if not gdf_candidate.empty else gpd.GeoDataFrame()
            if match_row.empty:
              try:
                match_row = gpd.read_file(str(target_hb_shp), where=f"HYBAS_ID = {hybas_id}", engine="pyogrio")
              except Exception:
                match_row = gpd.GeoDataFrame()

          if not match_row.empty:
            main_bas = int(match_row["MAIN_BAS"].values[0])

            if mode == "unit_catchment":
              polygon = match_row["geometry"].values[0]
              method_name = "HydroBASINS Level 12 Unit Catchment Polygon"
              upstream_area = round(float(match_row["SUB_AREA"].values[0]), 1)
              upstream_count = 1
            else:
              # Read all sub-basins within the same major river basin using pyogrio where filter
              try:
                gdf_main = gpd.read_file(str(target_hb_shp), where=f"MAIN_BAS = {main_bas}", engine="pyogrio")
              except Exception:
                gdf_main = match_row

              hb_down_to_up: Dict[int, List[int]] = {}
              for _, r in gdf_main.iterrows():
                hid = int(r["HYBAS_ID"])
                nd = int(r["NEXT_DOWN"])
                hb_down_to_up.setdefault(nd, []).append(hid)

              visited_hybas: Set[int] = set()
              queue = [int(hybas_id)]
              while queue and len(visited_hybas) < 3000:
                cur = queue.pop(0)
                if cur in visited_hybas:
                  continue
                visited_hybas.add(cur)
                for up_id in hb_down_to_up.get(cur, []):
                  if up_id not in visited_hybas:
                    queue.append(up_id)

              upstream_count = len(visited_hybas)

              if mode == "exact_pour_point":
                # 1. Gather all upstream tributary unit sub-basins
                trib_ids = visited_hybas - {int(hybas_id)}
                trib_geoms = []
                if trib_ids:
                  sub_trib = gdf_main[gdf_main["HYBAS_ID"].isin(trib_ids)]
                  trib_geoms = [
                      g
                      for g in sub_trib["geometry"].values
                      if g is not None and not g.is_empty
                  ]

                # 2. Slice the local clicked unit catchment at the pour-point cross section
                local_geom = match_row["geometry"].values[0]
                reach_geom = reach_attrs.get("reach_geometry")
                clipped_local = self._clip_local_catchment_to_pour_point(
                    local_geom, reach_geom, snapped_lat, snapped_lon
                )

                all_pieces = trib_geoms + [clipped_local]
                dissolved = unary_union(all_pieces)
                polygon = dissolved
                method_name = (
                    "Exact Pour-Point Drainage Basin (On-The-Fly Delineation)"
                )

                # Calculate accurate delineated area
                lat_scale = 111.0
                lon_scale = 111.0 * math.cos(math.radians(snapped_lat))
                upstream_area = round(
                    float(polygon.area * lat_scale * lon_scale), 1
                )
              else:
                sub_hb = gdf_main[gdf_main["HYBAS_ID"].isin(visited_hybas)]
                if not sub_hb.empty:
                  hb_geoms = [
                      g
                      for g in sub_hb["geometry"].values
                      if g is not None and not g.is_empty
                  ]
                  if hb_geoms:
                    dissolved = unary_union(hb_geoms)
                    polygon = dissolved
                    method_name = (
                        "HydroBASINS Level 12 Official Ridgeline Polygon"
                    )
      except Exception as e:
        print(f"HydroSHEDS catchment delineation error: {e}")

    if polygon is None:
      raise ValueError(
          f"Failed to delineate catchment at ({lat:.4f}, {lon:.4f}) "
          f"for dataset '{self.dataset_id}' (mode='{mode}'): no watershed "
          "geometry found at coordinates."
      )

    bounds = polygon.bounds
    bbox_dict = {
        "min_lon": round(bounds[0], 5),
        "min_lat": round(bounds[1], 5),
        "max_lon": round(bounds[2], 5),
        "max_lat": round(bounds[3], 5),
    }

    clean_dataset = self.dataset_id.replace("-", "_")
    catchment_id = (
        f"catchment_{clean_dataset}_{reach_id}_{abs(int(snapped_lat * 1000))}_{abs(int(snapped_lon * 1000))}"
    )

    geojson_feature = {
        "type": "Feature",
        "properties": {
            "catchment_id": catchment_id,
            "dataset": self.dataset_id,
            "dataset_name": self.dataset_meta["name"],
            "dem_id": self.dem_id,
            "dem_name": self.dataset_meta.get(
                "dem_name", "HydroSHEDS 90m Conditioned DEM (3 arc-sec)"
            ),
            "river_network": self.dataset_meta.get(
                "river_network_name", self.dataset_meta["name"]
            ),
            "outlet": {
                "input_latitude": lat,
                "input_longitude": lon,
                "latitude": snapped_lat,
                "longitude": snapped_lon,
                "reach_id": reach_id,
                "snap_distance_m": round(snap_dist_m, 1),
            },
            "area_km2": upstream_area,
            "stream_order": stream_order,
            "reach_attributes": reach_attrs,
            "bbox": bbox_dict,
            "upstream_reaches_count": upstream_count,
            "delineation_method": method_name,
            "delineation_mode": mode,
        },
        "geometry": mapping(polygon),
    }

    return geojson_feature

  def _clip_local_catchment_to_pour_point(
      self,
      local_poly: Polygon,
      reach_geom: Optional[Union[LineString, Dict[str, Any]]],
      snapped_lat: float,
      snapped_lon: float,
  ) -> Polygon:
    """Clips a local unit catchment polygon to retain only the terrain area upstream of the clicked pour point."""
    from shapely.geometry import shape

    snapped_pt = Point(snapped_lon, snapped_lat)
    if reach_geom is None:
      b = local_poly.bounds
      diag = max(0.2, math.hypot(b[2] - b[0], b[3] - b[1])) * 4.0
      p0 = np.array([snapped_pt.x, snapped_pt.y])
      half_plane = Polygon([
          (p0[0] - diag, p0[1]),
          (p0[0] + diag, p0[1]),
          (p0[0] + diag, p0[1] + diag),
          (p0[0] - diag, p0[1] + diag),
          (p0[0] - diag, p0[1]),
      ])
      clipped = local_poly.intersection(half_plane)
      if clipped and not clipped.is_empty and clipped.area > 0:
        return clipped
      return local_poly
    if isinstance(reach_geom, dict):
      reach_geom = shape(reach_geom)
    if reach_geom.is_empty:
      return local_poly

    try:
      proj_dist = reach_geom.project(snapped_pt)
      coords = list(reach_geom.coords)
      if len(coords) < 2:
        return local_poly

      # Find line segment containing proj_dist
      cum_len = 0.0
      seg_idx = 0
      for i in range(len(coords) - 1):
        p1 = Point(coords[i])
        p2 = Point(coords[i + 1])
        seg_len = p1.distance(p2)
        if cum_len + seg_len >= proj_dist - 1e-6:
          seg_idx = i
          break
        cum_len += seg_len

      p1 = np.array(coords[seg_idx])
      p2 = np.array(coords[seg_idx + 1])
      v = p2 - p1
      v_norm_len = np.linalg.norm(v)
      if v_norm_len < 1e-9:
        return local_poly

      v_norm = v / v_norm_len  # Downstream flow direction unit vector
      perp = np.array(
          [-v_norm[1], v_norm[0]]
      )  # Perpendicular cross-section unit vector

      # Construct upstream clipping half-plane box
      b = local_poly.bounds
      diag = max(0.2, math.hypot(b[2] - b[0], b[3] - b[1])) * 4.0
      p0 = np.array([snapped_pt.x, snapped_pt.y])

      half_plane_pts = [
          p0 - diag * perp,
          p0 + diag * perp,
          p0 + diag * perp - diag * v_norm,
          p0 - diag * perp - diag * v_norm,
          p0 - diag * perp,
      ]
      half_plane = Polygon(half_plane_pts)

      clipped = local_poly.intersection(half_plane)
      if clipped and not clipped.is_empty and clipped.area > 0:
        return clipped
      return local_poly
    except Exception as e:
      print(f"Error in pour-point clipping: {e}")
      return local_poly
