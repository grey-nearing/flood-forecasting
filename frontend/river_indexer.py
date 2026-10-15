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

"""Frontend river network map layer indexer.

Uses in-memory `z2`/`z4` simplified pyramids for global/continental slippy-map
views (`zoom <= 4`) and delegates regional/local spatial queries (`zoom >= 5`)
to `multimet.catchment_delineation.RiverNetwork`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import mapping

import frontend.config as _frontend_config
from frontend.config import (
    HYDRO_DATASETS,
    resolve_hydro_dataset_id,
)
from frontend.delineator import get_river_network

if TYPE_CHECKING:
    from multimet.utils.hydrography import Reach

_Z4_COMPLETE_MIN_ORDER: int = 7
_MERIT_Z4_COMPLETE_MIN_ORDER: int = 6
_MAIN_STEM_AREA_KM2: float = 50000.0
_REGIONAL_RIVER_AREA_KM2: float = 5000.0
_Z2_MAX_ZOOM: int = 2
_GLOBAL_MAX_ZOOM: int = 3
_CONTINENTAL_ZOOM: int = 4
_ZOOM_ORDER_5: int = 5
_ZOOM_ORDER_7: int = 7
_ZOOM_ORDER_9: int = 9
_ZOOM_ORDER_11: int = 11

_CACHES: dict[str, dict[str, Any] | None] = {
    'hydroatlas_z2': None,
    'hydroatlas_z4': None,
    'merit_z2': None,
    'merit_z4': None,
}
_Z4_INDEX: tuple[np.ndarray, np.ndarray] | None = None
_MERIT_Z4_INDEX: tuple[np.ndarray, np.ndarray] | None = None
_Z4_INDEX_LOCK = threading.Lock()
_INDEXERS: dict[str, HydroRiverNetwork] = {}


def reset_river_indexer_caches() -> None:
    """Clear in-memory pyramid caches and cached `HydroRiverNetwork` objects."""
    global _Z4_INDEX, _MERIT_Z4_INDEX  # noqa: PLW0603
    with _Z4_INDEX_LOCK:
        for key in _CACHES:
            _CACHES[key] = None
        _Z4_INDEX = None
        _MERIT_Z4_INDEX = None
        _INDEXERS.clear()


def _resolve_pyramid_path(key: str) -> Path | None:
    """Return the first existing GeoJSON pyramid file for `key`, or `None`."""
    river_dir = Path(_frontend_config.RIVER_NETWORKS_DIR)
    cache_dir = Path(_frontend_config.CACHE_DIR)
    candidates_by_key: dict[str, tuple[Path, ...]] = {
        'hydroatlas_z2': (
            river_dir / 'global_pyramids' / 'global_rivers_fast_z2.geojson',
            cache_dir / 'global_rivers_fast_z2.geojson',
        ),
        'hydroatlas_z4': (
            river_dir / 'global_pyramids' / 'global_rivers_fast_z4.geojson',
            cache_dir / 'global_rivers_fast_z4.geojson',
        ),
        'merit_z2': (
            river_dir / 'global_pyramids' / 'merit_rivers_fast_z2.geojson',
            river_dir / 'merit_rivers_fast_z2.geojson',
            cache_dir / 'merit_basins' / 'merit_rivers_fast_z2.geojson',
        ),
        'merit_z4': (
            river_dir / 'global_pyramids' / 'merit_rivers_fast_z4.geojson',
            river_dir / 'merit_rivers_fast_z4.geojson',
            cache_dir / 'merit_basins' / 'merit_rivers_fast_z4.geojson',
        ),
    }
    for cand in candidates_by_key.get(key, ()):
        if cand.is_file():
            return cand
    return None


def _ensure_pyramids_loaded() -> None:
    """Lazily load available simplified pyramid GeoJSON files into memory."""
    for key in ('hydroatlas_z2', 'hydroatlas_z4', 'merit_z2', 'merit_z4'):
        if _CACHES[key] is None:
            path = _resolve_pyramid_path(key)
            if path is not None:
                with path.open('r', encoding='utf-8') as f:
                    _CACHES[key] = json.load(f)


def _build_pyramid_index(
    feats: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build bounding-box and stream-order arrays for viewport filtering."""
    boxes = np.full((len(feats), 4), np.nan)
    orders = np.zeros(len(feats), dtype=np.int16)
    for i, feat in enumerate(feats):
        geom = feat.get('geometry') or {}
        coords = geom.get('coordinates') or []
        if geom.get('type') == 'MultiLineString':
            coords = [pt for part in coords for pt in part]
        if coords:
            xy = np.asarray(coords, dtype=float)[:, :2]
            boxes[i] = (
                xy[:, 0].min(),
                xy[:, 1].min(),
                xy[:, 0].max(),
                xy[:, 1].max(),
            )
        props = feat.get('properties') or {}
        orders[i] = int(
            props.get('stream_order') or props.get('river_class') or 0
        )
    return boxes, orders


def _hydroatlas_z4_index() -> tuple[np.ndarray, np.ndarray] | None:
    """Return or build the spatial index for the HydroATLAS z4 pyramid."""
    global _Z4_INDEX  # noqa: PLW0603
    cached = _CACHES.get('hydroatlas_z4')
    if _Z4_INDEX is None and cached is not None:
        with _Z4_INDEX_LOCK:
            if _Z4_INDEX is None:
                _Z4_INDEX = _build_pyramid_index(cached['features'])
    return _Z4_INDEX


def _merit_z4_index() -> tuple[np.ndarray, np.ndarray] | None:
    """Return or build the spatial index for the MERIT-Basins z4 pyramid."""
    global _MERIT_Z4_INDEX  # noqa: PLW0603
    cached = _CACHES.get('merit_z4')
    if _MERIT_Z4_INDEX is None and cached is not None:
        with _Z4_INDEX_LOCK:
            if _MERIT_Z4_INDEX is None:
                _MERIT_Z4_INDEX = _build_pyramid_index(cached['features'])
    return _MERIT_Z4_INDEX


def reach_to_ui_feature(reach: Reach) -> dict[str, Any]:
    """Format a backend `Reach` object into a GeoJSON Feature for the map UI."""
    if reach.dataset == 'merit-hydro':
        reach_id = f'MERIT_{reach.reach_id}'
        up_area = round(float(reach.upstream_area_km2), 1)
        if up_area > _MAIN_STEM_AREA_KM2:
            name = f'MERIT Main Stem ({reach_id})'
        elif up_area > _REGIONAL_RIVER_AREA_KM2:
            name = f'MERIT Regional River ({reach_id})'
        else:
            name = f'MERIT Tributary Reach ({reach_id})'
        return {
            'type': 'Feature',
            'id': reach_id,
            'geometry': mapping(reach.geometry),
            'properties': {
                'reach_id': reach_id,
                'dataset': 'merit-hydro',
                'river_name': name,
                'stream_order': reach.stream_order,
                'upstream_area_km2': up_area,
                'sinuosity': round(float(reach.extra.get('sinuosity', 1.0)), 2),
                'slope': round(float(reach.extra.get('slope', 0.0)), 4),
                'length_km': round(float(reach.length_km), 2),
                'next_down': reach.next_down,
                'theme': 'hydro-dynamics',
            },
        }

    reach_id = f'HYRIV_{reach.reach_id}'
    river_class = int(reach.extra.get('river_class', reach.stream_order))
    hybas_id = int(reach.extra.get('hydrobasins_unit', 0))
    name = f'HydroATLAS Sub-Basin {hybas_id} (Class {river_class})'
    return {
        'type': 'Feature',
        'id': reach_id,
        'geometry': mapping(reach.geometry),
        'properties': {
            'reach_id': reach_id,
            'dataset': 'hydroatlas',
            'river_name': name,
            'stream_order': reach.stream_order,
            'river_class': river_class,
            'hydrobasins_unit': hybas_id,
            'local_catchment_km2': round(
                float(reach.extra.get('local_catchment_km2', 0.0)), 2
            ),
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
            'theme': 'eco-hydrology',
        },
    }


_reach_to_ui_feature = reach_to_ui_feature


def zoom_to_min_stream_order(zoom: int) -> int:
    """Map slippy-map zoom level (`zoom >= 5`) to minimum Strahler order."""
    if zoom <= _ZOOM_ORDER_5:
        return 5
    if zoom <= _ZOOM_ORDER_7:
        return 4
    if zoom <= _ZOOM_ORDER_9:
        return 3
    if zoom <= _ZOOM_ORDER_11:
        return 2
    return 1


class HydroRiverNetwork:
    """Frontend map layer provider for MERIT-Hydro and HydroATLAS networks."""

    def __init__(self, dataset_id: str = 'hydroatlas') -> None:
        """Initialize the river network indexer for `dataset_id`."""
        resolved_id = resolve_hydro_dataset_id(dataset_id)
        if resolved_id not in HYDRO_DATASETS:
            raise ValueError(f"Unknown dataset '{dataset_id}'")
        self.dataset_id = resolved_id
        self.dataset_meta = HYDRO_DATASETS[resolved_id]
        self.is_merit = resolved_id == 'merit-hydro'
        self._theme = 'hydro-dynamics' if self.is_merit else 'eco-hydrology'

    def _build_collection(
        self,
        features: list[dict[str, Any]],
        zoom: int,
        **extra_props: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Build a standardized GeoJSON `FeatureCollection` response."""
        props: dict[str, Any] = {
            'dataset': self.dataset_id,
            'dataset_name': self.dataset_meta['name'],
            'resolution': self.dataset_meta['resolution'],
            'zoom': zoom,
            'feature_count': len(features),
            'theme': self._theme,
        }
        props.update(extra_props)
        return {
            'type': 'FeatureCollection',
            'properties': props,
            'features': features,
        }

    def _filter_cached_by_order(
        self,
        cached_features: list[dict[str, Any]],
        target_order: int,
    ) -> list[dict[str, Any]]:
        """Filter cached pyramid features by minimum stream order."""
        filtered = [
            f
            for f in cached_features
            if (
                f.get('properties', {}).get('stream_order')
                or f.get('properties', {}).get('river_class')
                or 1
            )
            >= target_order
        ]
        return filtered or cached_features

    def _query_z4_pyramid(
        self,
        bbox: tuple[float, float, float, float],
        min_order: int,
    ) -> list[dict[str, Any]] | None:
        """Query the z4 in-memory pyramid if available."""
        min_lon, min_lat, max_lon, max_lat = bbox
        index = _merit_z4_index() if self.is_merit else _hydroatlas_z4_index()
        cache_key = 'merit_z4' if self.is_merit else 'hydroatlas_z4'
        cached = _CACHES.get(cache_key)
        if index is None or cached is None:
            return None
        boxes, orders = index
        in_view = (
            (orders >= min_order)
            & (boxes[:, 2] >= min_lon)
            & (boxes[:, 0] <= max_lon)
            & (boxes[:, 3] >= min_lat)
            & (boxes[:, 1] <= max_lat)
        )
        pyramid = cached['features']
        return [pyramid[i] for i in np.flatnonzero(in_view)]

    def _try_prefer_cache_collection(
        self,
        bbox: tuple[float, float, float, float],
        zoom: int,
        min_stream_order: int | None,
    ) -> dict[str, Any] | None:
        """Return z4 pyramid collection when `prefer_cache=True` qualifies."""
        if min_stream_order is None:
            return None
        min_complete = (
            _MERIT_Z4_COMPLETE_MIN_ORDER
            if self.is_merit
            else _Z4_COMPLETE_MIN_ORDER
        )
        if min_stream_order < min_complete:
            return None
        z4_feats = self._query_z4_pyramid(bbox, min_stream_order)
        if z4_feats is None:
            return None
        source_label = (
            'MERIT-Basins z4 pyramid (in-memory, simplified geometry)'
            if self.is_merit
            else 'HydroATLAS z4 pyramid (in-memory, simplified geometry)'
        )
        return self._build_collection(
            z4_feats,
            zoom,
            min_stream_order_applied=min_stream_order,
            source=source_label,
        )

    def _try_coarse_zoom_collection(
        self,
        bbox: tuple[float, float, float, float],
        zoom: int,
        min_stream_order: int | None,
    ) -> dict[str, Any] | None:
        """Return z2 or z4 pyramid collection for `zoom <= 4` if cached."""
        if zoom <= _GLOBAL_MAX_ZOOM:
            cache_key = 'merit_z2' if self.is_merit else 'hydroatlas_z2'
            cached = _CACHES.get(cache_key)
            if cached is None:
                return None
            if min_stream_order is not None:
                target_order = min_stream_order
            elif self.is_merit:
                target_order = 8 if zoom <= _Z2_MAX_ZOOM else 7
            else:
                target_order = 9 if zoom <= _Z2_MAX_ZOOM else 8
            filtered = self._filter_cached_by_order(
                cached['features'], target_order
            )
            return self._build_collection(filtered, zoom)

        if zoom == _CONTINENTAL_ZOOM and min_stream_order is None:
            target_order = 6 if self.is_merit else 7
            z4_feats = self._query_z4_pyramid(bbox, target_order)
            if z4_feats is not None:
                return self._build_collection(z4_feats, zoom)
            cache_key = 'merit_z4' if self.is_merit else 'hydroatlas_z4'
            fallback_key = 'merit_z2' if self.is_merit else 'hydroatlas_z2'
            cached = _CACHES.get(cache_key) or _CACHES.get(fallback_key)
            if cached is not None:
                filtered = self._filter_cached_by_order(
                    cached['features'], target_order
                )
                return self._build_collection(filtered, zoom)
        return None

    def get_rivers_in_bbox(  # noqa: PLR0913, PLR0917
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        zoom: int = 2,
        min_stream_order: int | None = None,
        prefer_cache: bool = False,  # noqa: FBT001, FBT002
    ) -> dict[str, Any]:
        """Retrieve river reaches in a bounding box filtered by zoom LOD."""
        _ensure_pyramids_loaded()

        min_lon, max_lon = max(-180.0, min_lon), min(180.0, max_lon)
        min_lat, max_lat = max(-85.0, min_lat), min(85.0, max_lat)
        if min_lon >= max_lon:
            min_lon, max_lon = -180.0, 180.0
        if min_lat >= max_lat:
            min_lat, max_lat = -85.0, 85.0
        bbox = (min_lon, min_lat, max_lon, max_lat)

        if prefer_cache:
            pref_res = self._try_prefer_cache_collection(
                bbox, zoom, min_stream_order
            )
            if pref_res is not None:
                return pref_res

        coarse_res = self._try_coarse_zoom_collection(
            bbox, zoom, min_stream_order
        )
        if coarse_res is not None:
            return coarse_res

        order_threshold = (
            min_stream_order
            if min_stream_order is not None
            else zoom_to_min_stream_order(zoom)
        )
        network = get_river_network(self.dataset_id)
        if network is not None:
            reaches = network.query_reaches(
                bbox=bbox,
                min_stream_order=int(order_threshold),
            )
            features = [reach_to_ui_feature(r) for r in reaches]
            res_str = (
                '~90m (3 arc-sec)'
                if self.is_merit
                else '15/30 arc-sec (~500m/1km)'
            )
            src_str = (
                'Authentic MERIT-Basins 90m Hydrography'
                if self.is_merit
                else 'Authentic HydroSHEDS / HydroATLAS Database'
            )
            return self._build_collection(
                features,
                zoom,
                resolution=res_str,
                min_stream_order_applied=order_threshold,
                source=src_str,
            )

        z4_feats = self._query_z4_pyramid(bbox, int(order_threshold))
        if z4_feats is not None:
            source_label = (
                'MERIT-Basins z4 pyramid (in-memory, simplified geometry)'
                if self.is_merit
                else 'HydroATLAS z4 pyramid (in-memory, simplified geometry)'
            )
            return self._build_collection(
                z4_feats,
                zoom,
                min_stream_order_applied=order_threshold,
                source=source_label,
            )

        return {
            'type': 'FeatureCollection',
            'properties': {
                'dataset': self.dataset_id,
                'zoom': zoom,
                'feature_count': 0,
            },
            'features': [],
        }


def get_river_indexer(dataset_id: str = 'hydroatlas') -> HydroRiverNetwork:
    """Lazily instantiate and return a cached `HydroRiverNetwork` indexer."""
    resolved_id = resolve_hydro_dataset_id(dataset_id)
    with _Z4_INDEX_LOCK:
        if resolved_id not in _INDEXERS:
            _INDEXERS[resolved_id] = HydroRiverNetwork(resolved_id)
        return _INDEXERS[resolved_id]


__all__ = [
    'HydroRiverNetwork',
    'get_river_indexer',
    'reach_to_ui_feature',
    'reset_river_indexer_caches',
    'zoom_to_min_stream_order',
]
