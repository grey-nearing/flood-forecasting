# Copyright 2026 Google LLC
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

"""Frontend UI facade for river snapping and catchment delineation.

Delegates all scientific D8 raster routing, vector reach snapping, and
unit-catchment topology traversal to `multimet.catchment_delineation`, adding
only process-level singleton caching, thread locking, local/cache path
resolution, and UI presentation strings.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

from shapely.geometry import mapping

from frontend.config import (
    DEFAULT_DEM_ID,
    HYDRO_DATASETS,
    resolve_dem_tiles_dir,
    resolve_hydro_dataset_id,
    resolve_hydrography_paths,
)
from multimet.catchment_delineation import (
    CatchmentAreaMismatchError,
    CatchmentCoverageError,
    DemDelineator,
    HydroBasinsLayer,
    MeritBasinsLayer,
    ReachSnap,
    RiverNetwork,
    RiverSnapError,
    UnitCatchmentDelineator,
    delineate_hybrid,
    download_merit_d8_tile,
    resolve_dem_dataset,
)
from multimet.utils.hydrography import discover_partitions

if TYPE_CHECKING:
    from pathlib import Path

_MIN_COMID_LEN: int = 2
_DEFAULT_MAX_SNAP_M: float = 250.0
_M_PER_KM: float = 1000.0
_COORD_ID_SCALE: int = 1000
_SNAP_WINDOW_CELLS: int = 4

_REGISTRY_LOCK = threading.Lock()
_DEM_DELINEATORS: dict[str, DemDelineator] = {}
_DEM_LOCKS: dict[str, threading.Lock] = {}
_RIVER_NETWORKS: dict[str, RiverNetwork | None] = {}
_VECTOR_DELINEATORS: dict[str, UnitCatchmentDelineator | None] = {}


def reset_delineator_caches() -> None:
    """Clear process-level singleton caches for DEM and vector delineators."""
    with _REGISTRY_LOCK:
        _DEM_DELINEATORS.clear()
        _DEM_LOCKS.clear()
        _RIVER_NETWORKS.clear()
        _VECTOR_DELINEATORS.clear()


def get_dem_lock(dem_id: str = DEFAULT_DEM_ID) -> threading.Lock:
    """Return the per-dataset thread lock protecting `DemDelineator` caches."""
    dataset = resolve_dem_dataset(dem_id)
    with _REGISTRY_LOCK:
        if dataset.id not in _DEM_LOCKS:
            _DEM_LOCKS[dataset.id] = threading.Lock()
        return _DEM_LOCKS[dataset.id]


def get_dem_delineator(dem_id: str = DEFAULT_DEM_ID) -> DemDelineator:
    """Return a cached `DemDelineator` for the requested DEM dataset."""
    dataset = resolve_dem_dataset(dem_id)
    with _REGISTRY_LOCK:
        tiles_dir = resolve_dem_tiles_dir(dataset.id)
        resolved_tiles = tiles_dir.expanduser().resolve()
        cached = _DEM_DELINEATORS.get(dataset.id)
        if cached is not None and cached.tiles_dir == resolved_tiles:
            return cached
        tile_fetcher = None
        if dataset.id == 'merit_hydro_90m':
            ee_project = (
                os.environ.get('OPENHYDRONET_EE_PROJECT')
                or os.environ.get('GOOGLE_CLOUD_PROJECT')
                or ''
            ).strip()
            if ee_project:

                def _fetch(
                    lat_top: int,
                    lon_left: int,
                    target_dir: Path,
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


def get_river_network(dataset_id: str) -> RiverNetwork | None:
    """Lazily initialize and cache `RiverNetwork` for `dataset_id`."""
    with _REGISTRY_LOCK:
        cached = _RIVER_NETWORKS.get(dataset_id)
        if cached is not None:
            return cached
        paths = resolve_hydrography_paths()
        net: RiverNetwork | None = None
        if dataset_id == 'merit-hydro':
            merit_dir = paths['merit_rivers_dir']
            if merit_dir.is_dir() and bool(
                discover_partitions(merit_dir, 'riv_pfaf_*.shp')
            ):
                net = RiverNetwork.from_merit_basins(merit_dir)
        else:
            shp = paths['hydrorivers_shp']
            if shp.is_file():
                net = RiverNetwork.from_hydrorivers(shp)
        if net is not None:
            _RIVER_NETWORKS[dataset_id] = net
        return net


_get_river_network = get_river_network


def get_vector_delineator(
    dataset_id: str,
) -> UnitCatchmentDelineator | None:
    """Lazily initialize and cache `UnitCatchmentDelineator`."""
    with _REGISTRY_LOCK:
        cached = _VECTOR_DELINEATORS.get(dataset_id)
        if cached is not None:
            return cached
        paths = resolve_hydrography_paths()
        layer: MeritBasinsLayer | HydroBasinsLayer | None = None
        if dataset_id == 'merit-hydro':
            cat_dir = paths['merit_catchments_dir']
            if (
                cat_dir.is_dir()
                and bool(discover_partitions(cat_dir, 'riv_pfaf_*.shp'))
                and bool(discover_partitions(cat_dir, 'cat_pfaf_*.shp'))
            ):
                layer = MeritBasinsLayer(cat_dir)
        else:
            hb_dir = paths['hydrobasins_dir']
            if hb_dir.is_dir() and bool(discover_partitions(hb_dir, '*.shp')):
                layer = HydroBasinsLayer(hb_dir)
        if layer is None:
            return None
        delin = UnitCatchmentDelineator(layer)
        _VECTOR_DELINEATORS[dataset_id] = delin
        return delin


_get_vector_delineator = get_vector_delineator


def find_merit_shp(comid: int, prefix: str = 'cat') -> Path | None:
    """Locate the Pfafstetter shapefile containing the given COMID."""
    merit_dir = resolve_hydrography_paths()['merit_catchments_dir']
    if not merit_dir.is_dir():
        return None
    comid_str = str(int(comid))
    if len(comid_str) < _MIN_COMID_LEN:
        return None
    parts = discover_partitions(merit_dir, f'{prefix}_pfaf_*.shp')
    pfaf2 = comid_str[:2]
    pfaf1 = comid_str[:1]
    for part in parts:
        if part.pfaf_code == pfaf2:
            return part.path
    for part in parts:
        if part.pfaf_code == pfaf1:
            return part.path
    return None


_find_merit_shp = find_merit_shp


def _format_reach_attrs_for_ui(snap: ReachSnap) -> dict[str, Any]:
    """Format a backend `ReachSnap` into the UI `reach_attributes` dict."""
    reach = snap.reach
    geom_mapping = (
        mapping(reach.geometry) if reach.geometry is not None else None
    )
    if reach.dataset == 'merit-hydro':
        comid = reach.reach_id
        return {
            'reach_id': f'MERIT_{comid}',
            'dataset': 'merit-hydro',
            'river_name': f'MERIT River Reach (COMID {comid})',
            'stream_order': reach.stream_order,
            'upstream_area_km2': round(float(reach.upstream_area_km2), 1),
            'sinuosity': round(float(reach.extra.get('sinuosity', 1.0)), 2),
            'slope': round(float(reach.extra.get('slope', 0.0)), 4),
            'length_km': round(float(reach.length_km), 2),
            'next_down': reach.next_down,
            'reach_geometry': geom_mapping,
        }

    hyriv_id = reach.reach_id
    river_class = int(reach.extra.get('river_class', reach.stream_order))
    hybas_id = int(reach.extra.get('hydrobasins_unit', 0))
    return {
        'reach_id': f'HYRIV_{hyriv_id}',
        'dataset': 'hydroatlas',
        'river_name': f'HydroATLAS Sub-Basin {hybas_id} (Class {river_class})',
        'stream_order': reach.stream_order,
        'river_class': river_class,
        'hydrobasins_unit': hybas_id,
        'upstream_area_km2': round(float(reach.upstream_area_km2), 1),
        'mean_discharge_m3s': round(
            float(reach.extra.get('mean_discharge_m3s', 0.0)), 2
        ),
        'length_km': round(float(reach.length_km), 2),
        'dist_to_ocean_km': round(
            float(reach.extra.get('dist_to_ocean_km', 0.0)), 1
        ),
        'next_down': reach.next_down,
        'main_river': int(reach.extra.get('main_river', 0)),
        'reach_geometry': geom_mapping,
    }


def _resolve_vector_unit_id(  # noqa: PLR0913
    vec_delin: UnitCatchmentDelineator,
    *,
    is_merit: bool,
    reach_id: str,
    reach_attrs: dict[str, Any],
    snapped_lat: float,
    snapped_lon: float,
) -> int | None:
    """Resolve the unit catchment ID from reach metadata or spatial query."""
    unit_id: int | None = None
    if is_merit and reach_id.startswith('MERIT_'):
        suffix = reach_id.removeprefix('MERIT_')
        if suffix.isdigit():
            unit_id = int(suffix)
    elif not is_merit and reach_attrs.get('hydrobasins_unit'):
        unit_id = int(reach_attrs['hydrobasins_unit'])

    if unit_id is None:
        unit_id = vec_delin.layer.try_locate_unit(snapped_lat, snapped_lon)
    return unit_id


class HydroDelineator:
    """Frontend facade for river snapping and catchment delineation."""

    def __init__(self, dataset_id: str = 'hydroatlas') -> None:
        """Initialize the delineator facade for the requested hydro dataset."""
        resolved_id = resolve_hydro_dataset_id(dataset_id)
        if resolved_id not in HYDRO_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_id}'. Choose from:"
                f' {list(HYDRO_DATASETS.keys())}'
            )
        self.dataset_id = resolved_id
        self.dataset_meta = HYDRO_DATASETS[resolved_id]
        self.dem_id = str(self.dataset_meta.get('dem_id', DEFAULT_DEM_ID))
        self.is_merit = resolved_id == 'merit-hydro'

    def snap_to_river(
        self,
        lat: float,
        lon: float,
        snap_radius_km: float | None = None,
    ) -> tuple[float, float, float, dict[str, Any]]:
        """Snap coordinates to the nearest reach or return unsnapped point."""
        radius_km = snap_radius_km or float(
            self.dataset_meta['default_snap_radius_km']
        )
        max_snap_m = (
            (snap_radius_km * _M_PER_KM)
            if snap_radius_km is not None
            else _DEFAULT_MAX_SNAP_M
        )
        network = get_river_network(self.dataset_id)
        if network is not None:
            snap = network.try_snap_to_reach(
                lat,
                lon,
                search_radius_km=radius_km,
                max_distance_m=max_snap_m,
            )
            if snap is not None:
                return (
                    snap.lat,
                    snap.lon,
                    snap.distance_m,
                    _format_reach_attrs_for_ui(snap),
                )

        lat_key = abs(int(lat * _COORD_ID_SCALE))
        lon_key = abs(int(lon * _COORD_ID_SCALE))
        return (
            float(lat),
            float(lon),
            0.0,
            {
                'reach_id': f'POINT_{lat_key}_{lon_key}',
                'dataset': self.dataset_id,
                'river_name': f'Selected Location ({lat:.4f}°N, {lon:.4f}°E)',
                'stream_order': None,
                'upstream_area_km2': None,
            },
        )

    def delineate_catchment(
        self,
        lat: float,
        lon: float,
        snap_radius_km: float | None = None,
        mode: str = 'dem_flow_direction',
    ) -> dict[str, Any]:
        """Delineate the contributing drainage polygon with UI metadata."""
        radius_km = snap_radius_km or float(
            self.dataset_meta['default_snap_radius_km']
        )
        max_snap_m = (
            (snap_radius_km * _M_PER_KM)
            if snap_radius_km is not None
            else _DEFAULT_MAX_SNAP_M
        )
        snapped_lat, snapped_lon, snap_dist_m, reach_attrs = self.snap_to_river(
            lat, lon, snap_radius_km
        )
        reach_id = str(reach_attrs.get('reach_id') or 'REACH_0')
        raw_order = reach_attrs.get('stream_order')
        stream_order = int(raw_order) if raw_order is not None else None
        clean_dataset = self.dataset_id.replace('-', '_')

        if mode == 'dem_flow_direction':
            return self._delineate_dem_mode(
                lat=lat,
                lon=lon,
                snapped_lat=snapped_lat,
                snapped_lon=snapped_lon,
                snap_dist_m=snap_dist_m,
                radius_km=radius_km,
                max_snap_m=max_snap_m,
                reach_id=reach_id,
                stream_order=stream_order,
                clean_dataset=clean_dataset,
                reach_attrs=reach_attrs,
            )

        return self._delineate_vector_mode(
            lat=lat,
            lon=lon,
            snapped_lat=snapped_lat,
            snapped_lon=snapped_lon,
            snap_dist_m=snap_dist_m,
            mode=mode,
            reach_id=reach_id,
            stream_order=stream_order,
            clean_dataset=clean_dataset,
            reach_attrs=reach_attrs,
        )

    def _delineate_dem_mode(  # noqa: PLR0913
        self,
        *,
        lat: float,
        lon: float,
        snapped_lat: float,
        snapped_lon: float,
        snap_dist_m: float,
        radius_km: float,
        max_snap_m: float,
        reach_id: str,
        stream_order: int | None,
        clean_dataset: str,
        reach_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run hybrid or D8 flow-direction raster delineation."""
        dem_delin = get_dem_delineator(self.dem_id)
        dem_lock = get_dem_lock(self.dem_id)
        network = get_river_network(self.dataset_id)

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
                    snap_window_cells=_SNAP_WINDOW_CELLS,
                )

        dem_props = dem_feature.setdefault('properties', {})
        if 'area_km2' not in dem_props or dem_props['area_km2'] is None:
            raise ValueError('DEM delineation result is missing area_km2.')
        dem_area_km2 = float(dem_props['area_km2'])

        outlet_info = dem_props.setdefault('outlet', {})
        outlet_lat = float(outlet_info.get('latitude', snapped_lat))
        outlet_lon = float(outlet_info.get('longitude', snapped_lon))
        lat_key = abs(int(outlet_lat * _COORD_ID_SCALE))
        lon_key = abs(int(outlet_lon * _COORD_ID_SCALE))
        catchment_id = (
            f'catchment_{clean_dataset}_{reach_id}_{lat_key}_{lon_key}'
        )
        dem_cell_id = outlet_info.get(
            'dem_cell_id', outlet_info.get('reach_id', 'DEM_0_0')
        )
        outlet_info.update(
            {
                'input_latitude': lat,
                'input_longitude': lon,
                'latitude': round(outlet_lat, 5),
                'longitude': round(outlet_lon, 5),
                'reach_id': reach_id,
                'dem_cell_id': dem_cell_id,
                'snap_distance_m': round(
                    float(
                        snap_dist_m or outlet_info.get('snap_distance_m', 0.0)
                    ),
                    1,
                ),
            }
        )
        merged_reach_attrs = dict(dem_props.get('reach_attributes') or {})
        merged_reach_attrs.update(reach_attrs)
        merged_reach_attrs['dem_cell_id'] = dem_cell_id
        merged_reach_attrs['upstream_area_km2'] = dem_area_km2
        dem_props.update(
            {
                'catchment_id': catchment_id,
                'gauge_id': catchment_id,
                'dataset': self.dataset_id,
                'dataset_name': self.dataset_meta['name'],
                'dem_id': self.dem_id,
                'dem_name': self.dataset_meta.get(
                    'dem_name', 'HydroSHEDS 90m Conditioned DEM (3 arc-sec)'
                ),
                'river_network': self.dataset_meta.get(
                    'river_network_name', self.dataset_meta['name']
                ),
                'stream_order': stream_order,
                'reach_attributes': merged_reach_attrs,
                'upstream_reaches_count': max(
                    1, int(dem_props.get('upstream_reaches_count', 1))
                ),
            }
        )
        return dem_feature

    def _delineate_vector_mode(  # noqa: PLR0913
        self,
        *,
        lat: float,
        lon: float,
        snapped_lat: float,
        snapped_lon: float,
        snap_dist_m: float,
        mode: str,
        reach_id: str,
        stream_order: int | None,
        clean_dataset: str,
        reach_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run vector unit-catchment delineation."""
        vec_delin = get_vector_delineator(self.dataset_id)
        if vec_delin is None:
            raise ValueError(
                f"Vector unit-catchment data for '{self.dataset_id}' is not"
                ' available.'
            )

        unit_id = _resolve_vector_unit_id(
            vec_delin,
            is_merit=self.is_merit,
            reach_id=reach_id,
            reach_attrs=reach_attrs,
            snapped_lat=snapped_lat,
            snapped_lon=snapped_lon,
        )
        if unit_id is None:
            raise ValueError(
                f'Failed to delineate catchment at ({lat:.4f}, {lon:.4f}) '
                f"for dataset '{self.dataset_id}' (mode='{mode}'): no "
                'watershed geometry found at coordinates.'
            )

        reach_geom = reach_attrs.get('reach_geometry')
        if mode == 'unit_catchment':
            vc = vec_delin.delineate_unit_catchment(unit_id)
        elif mode == 'exact_pour_point' and reach_geom is not None:
            vc = vec_delin.delineate_exact_pour_point(
                unit_id, snapped_lat, snapped_lon, reach_geom
            )
        else:
            vc = vec_delin.delineate_ridgeline(unit_id)

        lat_key = abs(int(snapped_lat * _COORD_ID_SCALE))
        lon_key = abs(int(snapped_lon * _COORD_ID_SCALE))
        catchment_id = (
            f'catchment_{clean_dataset}_{reach_id}_{lat_key}_{lon_key}'
        )
        outlet = {
            'input_latitude': lat,
            'input_longitude': lon,
            'latitude': snapped_lat,
            'longitude': snapped_lon,
            'reach_id': reach_id,
            'snap_distance_m': round(snap_dist_m, 1),
        }
        updated_reach_attrs = dict(reach_attrs)
        if updated_reach_attrs.get('upstream_area_km2') is None:
            updated_reach_attrs['upstream_area_km2'] = vc.area_km2
        extra_props = {
            'dataset': self.dataset_id,
            'dataset_name': self.dataset_meta['name'],
            'dem_id': self.dem_id,
            'dem_name': self.dataset_meta.get(
                'dem_name', 'HydroSHEDS 90m Conditioned DEM (3 arc-sec)'
            ),
            'river_network': self.dataset_meta.get(
                'river_network_name', self.dataset_meta['name']
            ),
            'stream_order': stream_order,
            'reach_attributes': updated_reach_attrs,
            'delineation_mode': mode,
        }
        return vc.to_feature(
            catchment_id=catchment_id,
            outlet=outlet,
            extra_properties=extra_props,
        )


__all__ = [
    'CatchmentAreaMismatchError',
    'CatchmentCoverageError',
    'HydroDelineator',
    'RiverSnapError',
    'find_merit_shp',
    'get_dem_delineator',
    'get_dem_lock',
    'get_river_network',
    'get_vector_delineator',
    'reset_delineator_caches',
]
