"""Hybrid vector-guided + 90m D8 raster catchment delineation."""

from __future__ import annotations

from typing import Any

from multimet.catchment_delineation.delineator import (
    CatchmentAreaMismatchError,
    DemDelineator,
)
from multimet.catchment_delineation.hydrography import (
    ReachSnap,
    RiverNetwork,
    RiverSnapError,
)


def delineate_hybrid(
    dem: DemDelineator,
    network: RiverNetwork,
    lat: float,
    lon: float,
    *,
    snap_radius_km: float = 5.0,
    max_snap_distance_m: float = 250.0,
    coarse_snap_window_cells: int = 4,
    fine_snap_window_cells: int = 12,
    area_underestimate_ratio: float = 0.25,
    hint_area_range_km2: tuple[float, float] = (10.0, 25000.0),
    area_tolerance: float = 0.55,
    max_cells: int | None = None,
    catchment_id: str | None = None,
) -> dict[str, Any]:
    """Delineates a catchment using vector river snapping + 90m D8 flow routing.

    1. Attempts to snap ``(lat, lon)`` to the nearest reach in ``network`` 
        within
       ``max_snap_distance_m`` (searching within ``snap_radius_km``). If no 
           reach
       is within ``max_snap_distance_m``, falls back to pure D8 raster
       delineation at ``(lat, lon)``.
    2. Runs ``dem.delineate`` at the pour point with 
        ``coarse_snap_window_cells``.
    3. If a vector reach was snapped whose ``upstream_area_km2`` lies within
       ``hint_area_range_km2`` and the initial D8 polygon underestimates that 
           area
       by more than ``area_underestimate_ratio`` (indicating the coarse window
       landed on a tiny bank tributary rather than the mainstem accumulation
       channel), retries ``dem.delineate`` with ``fine_snap_window_cells`` and
       ``expected_area_km2=snap.reach.upstream_area_km2``.
    """
    snap: ReachSnap | None = None
    try:
        snap = network.snap_to_reach(
            lat,
            lon,
            search_radius_km=snap_radius_km,
            max_distance_m=max_snap_distance_m,
        )
        pour_lat = snap.lat
        pour_lon = snap.lon
    except RiverSnapError:
        pour_lat = float(lat)
        pour_lon = float(lon)

    dem_feature = dem.delineate(
        lat=pour_lat,
        lon=pour_lon,
        snap_window_cells=coarse_snap_window_cells,
        max_cells=max_cells,
        catchment_id=catchment_id,
    )

    if snap is not None:
        dem_area = float(
            dem_feature.get('properties', {}).get('area_km2') or 0.0
        )
        hint_area = float(snap.reach.upstream_area_km2)
        min_hint, max_hint = hint_area_range_km2
        if (
            min_hint <= hint_area <= max_hint
            and dem_area < area_underestimate_ratio * hint_area
        ):
            try:
                dem_feature = dem.delineate(
                    lat=pour_lat,
                    lon=pour_lon,
                    snap_window_cells=fine_snap_window_cells,
                    expected_area_km2=hint_area,
                    area_tolerance=area_tolerance,
                    max_cells=max_cells,
                    catchment_id=catchment_id,
                )
            except CatchmentAreaMismatchError:
                pass

    dem_props = dem_feature.setdefault('properties', {})
    outlet_info = dem_props.setdefault('outlet', {})
    dem_cell_id = outlet_info.get('reach_id', 'DEM_0_0')
    outlet_info['input_latitude'] = float(lat)
    outlet_info['input_longitude'] = float(lon)
    outlet_info['dem_cell_id'] = dem_cell_id

    if snap is not None:
        reach_prefix = (
            'MERIT' if snap.reach.dataset == 'merit-hydro' else 'HYRIV'
        )
        reach_code = f'{reach_prefix}_{snap.reach.reach_id}'
        outlet_info['reach_id'] = reach_code
        outlet_info['vector_snapped_latitude'] = round(snap.lat, 5)
        outlet_info['vector_snapped_longitude'] = round(snap.lon, 5)
        outlet_info['vector_snap_distance_m'] = round(snap.distance_m, 1)
        dem_props['stream_order'] = snap.reach.stream_order

        merged_attrs = dict(dem_props.get('reach_attributes') or {})
        merged_attrs.update(snap.reach.extra)
        merged_attrs.update(
            {
                'reach_id': reach_code,
                'dataset': snap.reach.dataset,
                'stream_order': snap.reach.stream_order,
                'vector_upstream_area_km2': round(
                    snap.reach.upstream_area_km2, 1
                ),
                'upstream_area_km2': dem_props.get(
                    'area_km2', round(snap.reach.upstream_area_km2, 1)
                ),
                'length_km': round(snap.reach.length_km, 2),
                'next_down': snap.reach.next_down,
                'dem_cell_id': dem_cell_id,
            }
        )
        dem_props['reach_attributes'] = merged_attrs

    return dem_feature
