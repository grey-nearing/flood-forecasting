"""Frontend UI facade for river snapping and catchment delineation.

Delegates all scientific D8 raster routing, vector reach snapping, and
unit-catchment topology traversal to `multimet.catchment_delineation`, adding
only process-level singleton caching, thread locking, local/cache path
resolution, and UI presentation strings.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple

from shapely.geometry import mapping

from frontend.config import (
    DEFAULT_DEM_ID,
    HYDRO_DATASETS,
    ensure_flood_forecasting_on_sys_path,
    extend_multimet_package_path,
    resolve_dem_tiles_dir,
    resolve_hydro_dataset_id,
    resolve_hydrography_paths,
)

ensure_flood_forecasting_on_sys_path()
extend_multimet_package_path()

from multimet.catchment_delineation import (  # pylint: disable=g-import-not-at-top
    CatchmentAreaMismatchError,
    CatchmentCoverageError,
    DemDelineator,
    HydroBasinsLayer,
    MeritBasinsLayer,
    RiverNetwork,
    RiverSnapError,
    UnitCatchmentDelineator,
    delineate_hybrid,
    download_merit_d8_tile,
    resolve_dem_dataset,
)
from multimet.utils.hydrography import (  # pylint: disable=g-import-not-at-top
    discover_partitions,
    merit_partition_for_comid,
)

_REGISTRY_LOCK = threading.Lock()
_DEM_DELINEATORS: Dict[str, DemDelineator] = {}
_DEM_LOCKS: Dict[str, threading.Lock] = {}
_RIVER_NETWORKS: Dict[str, Optional[RiverNetwork]] = {}
_VECTOR_DELINEATORS: Dict[str, Optional[UnitCatchmentDelineator]] = {}


def get_dem_lock(dem_id: str = DEFAULT_DEM_ID) -> threading.Lock:
  """Returns the per-dataset thread lock protecting `DemDelineator` tile caches."""
  dataset = resolve_dem_dataset(dem_id)
  with _REGISTRY_LOCK:
    if dataset.id not in _DEM_LOCKS:
      _DEM_LOCKS[dataset.id] = threading.Lock()
    return _DEM_LOCKS[dataset.id]


def get_dem_delineator(dem_id: str = DEFAULT_DEM_ID) -> DemDelineator:
  """Returns a cached `DemDelineator` instance for the requested DEM dataset."""
  dataset = resolve_dem_dataset(dem_id)
  with _REGISTRY_LOCK:
    if dataset.id not in _DEM_DELINEATORS:
      tiles_dir = resolve_dem_tiles_dir(dataset.id)
      tile_fetcher = None
      if dataset.id == "merit_hydro_90m":
        ee_project = (
            os.environ.get("OPENHYDRONET_EE_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or ""
        ).strip()
        if ee_project:

          def _fetch(
              lat_top: int, lon_left: int, target_dir: Path
          ) -> Path:
            return download_merit_d8_tile(
                lat_top=lat_top,
                lon_left=lon_left,
                target_dir=target_dir,
                ee_project=ee_project,
            )

          tile_fetcher = _fetch

      _DEM_DELINEATORS[dataset.id] = DemDelineator(
          tiles_dir=tiles_dir,
          dataset=dataset,
          tile_fetcher=tile_fetcher,
      )
    return _DEM_DELINEATORS[dataset.id]


def _get_river_network(dataset_id: str) -> Optional[RiverNetwork]:
  """Lazily initializes and caches `RiverNetwork` for `dataset_id`."""
  with _REGISTRY_LOCK:
    if dataset_id in _RIVER_NETWORKS:
      return _RIVER_NETWORKS[dataset_id]
    paths = resolve_hydrography_paths()
    try:
      if dataset_id == "merit-hydro":
        merit_dir = paths["merit_rivers_dir"]
        net = (
            RiverNetwork.from_merit_basins(merit_dir)
            if merit_dir.exists()
            else None
        )
      else:
        shp = paths["hydrorivers_shp"]
        net = RiverNetwork.from_hydrorivers(shp) if shp.exists() else None
    except Exception:
      net = None
    _RIVER_NETWORKS[dataset_id] = net
    return net


def _get_vector_delineator(
    dataset_id: str,
) -> Optional[UnitCatchmentDelineator]:
  """Lazily initializes and caches `UnitCatchmentDelineator` for `dataset_id`."""
  with _REGISTRY_LOCK:
    if dataset_id in _VECTOR_DELINEATORS:
      return _VECTOR_DELINEATORS[dataset_id]
    paths = resolve_hydrography_paths()
    try:
      if dataset_id == "merit-hydro":
        cat_dir = paths["merit_catchments_dir"]
        layer = MeritBasinsLayer(cat_dir) if cat_dir.exists() else None
      else:
        hb_dir = paths["hydrobasins_dir"]
        layer = HydroBasinsLayer(hb_dir) if hb_dir.exists() else None
      delin = UnitCatchmentDelineator(layer) if layer is not None else None
    except Exception:
      delin = None
    _VECTOR_DELINEATORS[dataset_id] = delin
    return delin


def _find_merit_shp(comid: int, prefix: str = "cat") -> Optional[Path]:
  """Locates the Pfafstetter shapefile containing the given COMID."""
  merit_dir = resolve_hydrography_paths()["merit_catchments_dir"]
  if not merit_dir.exists():
    return None
  try:
    parts = discover_partitions(merit_dir, f"{prefix}_pfaf_*.shp")
    return merit_partition_for_comid(parts, comid).path
  except Exception:
    return None


def _format_reach_attrs_for_ui(snap: Any) -> Dict[str, Any]:
  """Formats a backend `ReachSnap` into the UI reach_attributes dictionary."""
  reach = snap.reach
  geom_mapping = mapping(reach.geometry) if reach.geometry is not None else None
  if reach.dataset == "merit-hydro":
    comid = reach.reach_id
    return {
        "reach_id": f"MERIT_{comid}",
        "dataset": "merit-hydro",
        "river_name": f"MERIT River Reach (COMID {comid})",
        "stream_order": reach.stream_order,
        "upstream_area_km2": round(float(reach.upstream_area_km2), 1),
        "sinuosity": round(float(reach.extra.get("sinuosity", 1.0)), 2),
        "slope": round(float(reach.extra.get("slope", 0.0)), 4),
        "length_km": round(float(reach.length_km), 2),
        "next_down": reach.next_down,
        "reach_geometry": geom_mapping,
    }

  hyriv_id = reach.reach_id
  river_class = int(reach.extra.get("river_class", reach.stream_order))
  hybas_id = int(reach.extra.get("hydrobasins_unit", 0))
  return {
      "reach_id": f"HYRIV_{hyriv_id}",
      "dataset": "hydroatlas",
      "river_name": f"HydroATLAS Sub-Basin {hybas_id} (Class {river_class})",
      "stream_order": reach.stream_order,
      "river_class": river_class,
      "hydrobasins_unit": hybas_id,
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
      "reach_geometry": geom_mapping,
  }


class HydroDelineator:
  """Frontend facade for river snapping and catchment delineation."""

  def __init__(self, dataset_id: str = "hydroatlas"):
    resolved_id = resolve_hydro_dataset_id(dataset_id)
    if resolved_id not in HYDRO_DATASETS:
      raise ValueError(
          f"Unknown dataset '{dataset_id}'. Choose from:"
          f" {list(HYDRO_DATASETS.keys())}"
      )
    self.dataset_id = resolved_id
    self.dataset_meta = HYDRO_DATASETS[resolved_id]
    self.dem_id = self.dataset_meta.get("dem_id", DEFAULT_DEM_ID)
    self.is_merit = resolved_id == "merit-hydro"

  def snap_to_river(
      self, lat: float, lon: float, snap_radius_km: Optional[float] = None
  ) -> Tuple[float, float, float, Dict[str, Any]]:
    """Snaps click coordinates to the nearest river reach or returns a point label."""
    radius_km = snap_radius_km or float(
        self.dataset_meta["default_snap_radius_km"]
    )
    max_snap_m = (
        (snap_radius_km * 1000.0) if snap_radius_km is not None else 250.0
    )
    network = _get_river_network(self.dataset_id)
    if network is not None:
      try:
        snap = network.snap_to_reach(
            lat,
            lon,
            search_radius_km=radius_km,
            max_distance_m=max_snap_m,
        )
        return (
            snap.lat,
            snap.lon,
            snap.distance_m,
            _format_reach_attrs_for_ui(snap),
        )
      except (RiverSnapError, Exception):
        pass

    return (
        float(lat),
        float(lon),
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
      mode: str = "dem_flow_direction",
  ) -> Dict[str, Any]:
    """Delineates the contributing drainage polygon and enriches with UI metadata."""
    radius_km = snap_radius_km or float(
        self.dataset_meta["default_snap_radius_km"]
    )
    max_snap_m = (
        (snap_radius_km * 1000.0) if snap_radius_km is not None else 250.0
    )
    snapped_lat, snapped_lon, snap_dist_m, reach_attrs = self.snap_to_river(
        lat, lon, snap_radius_km
    )
    reach_id = reach_attrs.get("reach_id", "REACH_0")
    stream_order = int(reach_attrs.get("stream_order", 1))
    clean_dataset = self.dataset_id.replace("-", "_")

    # 1. Hybrid / 90m D8 Flow-Direction Mode
    if mode == "dem_flow_direction":
      dem_delin = get_dem_delineator(self.dem_id)
      dem_lock = get_dem_lock(self.dem_id)
      network = _get_river_network(self.dataset_id)

      with dem_lock:
        if network is not None:
          dem_feature = delineate_hybrid(
              dem=dem_delin,
              network=network,
              lat=lat,
              lon=lon,
              snap_radius_km=radius_km,
              max_snap_distance_m=max_snap_m,
          )
        else:
          dem_feature = dem_delin.delineate(
              lat=snapped_lat,
              lon=snapped_lon,
              snap_window_cells=4,
          )

      dem_props = dem_feature.setdefault("properties", {})
      outlet_info = dem_props.setdefault("outlet", {})
      outlet_lat = float(outlet_info.get("latitude", snapped_lat))
      outlet_lon = float(outlet_info.get("longitude", snapped_lon))
      catchment_id = (
          f"catchment_{clean_dataset}_{reach_id}_"
          f"{abs(int(outlet_lat * 1000))}_{abs(int(outlet_lon * 1000))}"
      )
      dem_cell_id = outlet_info.get(
          "dem_cell_id", outlet_info.get("reach_id", "DEM_0_0")
      )
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
          "area_km2", reach_attrs.get("upstream_area_km2", 50.0)
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

    # 2. Vector Unit-Catchment Modes (official_ridgeline, exact_pour_point, unit_catchment)
    vec_delin = _get_vector_delineator(self.dataset_id)
    if vec_delin is None:
      raise ValueError(
          f"Vector unit-catchment data for '{self.dataset_id}' is not"
          " available."
      )

    unit_id: Optional[int] = None
    if self.is_merit and str(reach_id).startswith("MERIT_"):
      try:
        unit_id = int(str(reach_id).replace("MERIT_", ""))
      except ValueError:
        unit_id = None
    elif not self.is_merit and reach_attrs.get("hydrobasins_unit"):
      unit_id = int(reach_attrs["hydrobasins_unit"])

    if unit_id is None:
      try:
        unit_id = vec_delin.layer.locate_unit(snapped_lat, snapped_lon)
      except LookupError:
        unit_id = None

    if unit_id is None:
      raise ValueError(
          f"Failed to delineate catchment at ({lat:.4f}, {lon:.4f}) "
          f"for dataset '{self.dataset_id}' (mode='{mode}'): no watershed "
          "geometry found at coordinates."
      )

    if mode == "unit_catchment":
      vc = vec_delin.delineate_unit_catchment(unit_id)
    elif mode == "exact_pour_point":
      reach_geom = reach_attrs.get("reach_geometry")
      if reach_geom is not None:
        try:
          vc = vec_delin.delineate_exact_pour_point(
              unit_id, snapped_lat, snapped_lon, reach_geom
          )
        except ValueError:
          vc = vec_delin.delineate_ridgeline(unit_id)
      else:
        vc = vec_delin.delineate_ridgeline(unit_id)
    else:
      vc = vec_delin.delineate_ridgeline(unit_id)

    catchment_id = (
        f"catchment_{clean_dataset}_{reach_id}_"
        f"{abs(int(snapped_lat * 1000))}_{abs(int(snapped_lon * 1000))}"
    )
    outlet = {
        "input_latitude": lat,
        "input_longitude": lon,
        "latitude": snapped_lat,
        "longitude": snapped_lon,
        "reach_id": reach_id,
        "snap_distance_m": round(snap_dist_m, 1),
    }
    extra_props = {
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
        "reach_attributes": reach_attrs,
        "delineation_mode": mode,
    }
    return vc.to_feature(
        catchment_id=catchment_id,
        outlet=outlet,
        extra_properties=extra_props,
    )
