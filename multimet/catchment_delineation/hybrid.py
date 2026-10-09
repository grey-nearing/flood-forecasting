# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid vector-guided + 90m D8 raster catchment delineation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multimet.catchment_delineation.delineator import DemDelineator
    from multimet.catchment_delineation.hydrography import (
        ReachSnap,
        RiverNetwork,
    )


def delineate_hybrid(
    dem: DemDelineator,
    network: RiverNetwork,
    lat: float,
    lon: float,
    **kwargs: object,
) -> dict[str, object]:
    """Delineate a catchment using vector river snapping + 90m D8 flow routing.

    1. Attempts to snap `(lat, lon)` to the nearest reach in `network` within
       `max_snap_distance_m` (searching within `snap_radius_km`). If no reach
       is within `max_snap_distance_m`, falls back to pure D8 raster
       delineation at `(lat, lon)`.
    2. Runs `dem.delineate` at the pour point with `coarse_snap_window_cells`.
    3. If a vector reach was snapped whose `upstream_area_km2` lies within
       `hint_area_range_km2` and the initial D8 polygon underestimates that area
       by more than `area_underestimate_ratio` (indicating the coarse window
       landed on a tiny bank tributary rather than the mainstem accumulation
       channel), retries `dem.delineate` with `fine_snap_window_cells` and
       `expected_area_km2=snap.reach.upstream_area_km2`.
    """
    snap_radius_km = float(kwargs.get('snap_radius_km', 5.0))  # type: ignore[arg-type]
    max_snap_distance_m = float(kwargs.get('max_snap_distance_m', 250.0))  # type: ignore[arg-type]
    coarse_snap_window_cells = int(kwargs.get('coarse_snap_window_cells', 4))  # type: ignore[call-overload]
    fine_snap_window_cells = int(kwargs.get('fine_snap_window_cells', 12))  # type: ignore[call-overload]
    area_underestimate_ratio = float(
        kwargs.get('area_underestimate_ratio', 0.25)  # type: ignore[arg-type]
    )
    hint_area_range_km2: tuple[float, float] = kwargs.get(  # type: ignore[assignment]
        'hint_area_range_km2', (10.0, 25000.0)
    )
    area_tolerance = float(kwargs.get('area_tolerance', 0.55))  # type: ignore[arg-type]
    max_cells_raw = kwargs.get('max_cells')
    max_cells = int(max_cells_raw) if max_cells_raw is not None else None  # type: ignore[call-overload]
    catchment_id_raw = kwargs.get('catchment_id')
    catchment_id = (
        str(catchment_id_raw) if catchment_id_raw is not None else None
    )

    snap: ReachSnap | None = network.try_snap_to_reach(
        lat,
        lon,
        search_radius_km=snap_radius_km,
        max_distance_m=max_snap_distance_m,
    )
    if snap is not None:
        pour_lat = snap.lat
        pour_lon = snap.lon
    else:
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
            dem_feature = dem.delineate(
                lat=pour_lat,
                lon=pour_lon,
                snap_window_cells=fine_snap_window_cells,
                expected_area_km2=hint_area,
                area_tolerance=area_tolerance,
                max_cells=max_cells,
                catchment_id=catchment_id,
            )

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
        merged_attrs.update({
            'reach_id': reach_code,
            'dataset': snap.reach.dataset,
            'stream_order': snap.reach.stream_order,
            'vector_upstream_area_km2': round(snap.reach.upstream_area_km2, 1),
            'upstream_area_km2': dem_props.get(
                'area_km2', round(snap.reach.upstream_area_km2, 1)
            ),
            'length_km': round(snap.reach.length_km, 2),
            'next_down': snap.reach.next_down,
            'dem_cell_id': dem_cell_id,
        })
        dem_props['reach_attributes'] = merged_attrs

    return dem_feature
