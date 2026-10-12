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

"""Per-model river networks for the MaaS map, delegating to `maas.networks`.

Binds `frontend.config.CACHE_DIR` and `frontend.config.RIVER_NETWORKS_DIR`
while delegating pyramid construction, level-of-detail extraction, and
upstream-area snapping to `maas.networks`.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import frontend.config as _frontend_config
from frontend.maas_live_status import (
    clear_live_status_cache,
    enrich_network_features_live,
    get_active_flood_keys,
    get_glofas_ows_grid,
    get_last_refresh_iso,
)
from frontend.river_indexer import HydroRiverNetwork
from maas.config import (
    GEOGLOWS_MIN_AREA_KM2,
    NETWORK_LABELS,
)
from maas.networks import (
    CACHE_VERSION,
    FLOODHUB_LOD,
    FLOODHUB_PYRAMID_LOD,
    GEOGLOWS_LOD,
    GLOFAS_LOD,
    MODELS,
    build_floodhub_pyramid,
    build_geoglows_pyramid,
    build_glofas_pyramid,
    extract_level_features,
    load_geoglows_lookup,
    load_network_pyramid,
    lod_for_zoom,
    pyramid_signature,
    query_geoglows_reaches,
    resolve_cross_network_click,
    save_network_pyramid,
    snap_geoglows_reach_from_gpkg,
    snap_geoglows_reach_from_network,
    snap_glofas_cell_from_network,
)
from maas.tools.build_network_pyramids import (
    find_floodhub_lut,
    find_hydrorivers_shp,
)

_LOG = logging.getLogger(__name__)

_PROMOTE_MAX_ZOOM: int = 8
_MAX_PROMOTED_HITS: int = 250
_MIN_LINE_VERTICES: int = 2
_GLOFAS_OWS_TOP_LAT: float = 75.0
_GLOFAS_OWS_LEFT_LON: float = -180.0
_GLOFAS_OWS_RES_DEG: float = 0.05
_GLOFAS_OWS_MAX_ROW: int = 2699
_GLOFAS_OWS_MAX_COL: int = 7199


def _data_dir(name: str) -> Path:
    """Resolve active data directory for `name` (`glofas_v4`, `geoglows_v2`)."""
    local_dir = Path(_frontend_config.RIVER_NETWORKS_DIR) / name
    cache_dir = Path(_frontend_config.CACHE_DIR) / name
    if local_dir.is_dir() and any(local_dir.iterdir()):
        return local_dir
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        return cache_dir
    return local_dir if local_dir.exists() else cache_dir


def _glofas_dir() -> Path:
    """Return the active `glofas_v4` directory."""
    return _data_dir('glofas_v4')


def _geoglows_dir() -> Path:
    """Return the active `geoglows_v2` directory."""
    return _data_dir('geoglows_v2')


def _geoglows_gpkg() -> Path:
    """Return the path to `global_streams_simplified.gpkg`."""
    return _geoglows_dir() / 'global_streams_simplified.gpkg'


def _geoglows_attrs() -> Path:
    """Return the path to `geoglows_attrs_v1.npz`."""
    return _geoglows_dir() / 'geoglows_attrs_v1.npz'


_LOCK = threading.Lock()
_NETWORKS: dict[str, dict[str, Any]] = {}
_GEOGLOWS_LOOKUP: tuple[np.ndarray, np.ndarray] | None = None
_FLOODHUB_LUT: dict[str, Any] | None = None


def reset_maas_network_caches() -> None:
    """Clear in-memory cached network pyramids and lookup tables."""
    global _GEOGLOWS_LOOKUP, _FLOODHUB_LUT  # noqa: PLW0603
    with _LOCK:
        _NETWORKS.clear()
        _GEOGLOWS_LOOKUP = None
        _FLOODHUB_LUT = None


def _geoglows_lookup() -> tuple[np.ndarray, np.ndarray]:
    """Return `(sorted LINKNO, DSContArea km²)` for GEOGLOWS reaches."""
    global _GEOGLOWS_LOOKUP  # noqa: PLW0603
    if _GEOGLOWS_LOOKUP is None:
        _GEOGLOWS_LOOKUP = load_geoglows_lookup(
            _geoglows_attrs(), min_area_km2=GEOGLOWS_MIN_AREA_KM2
        )
    return _GEOGLOWS_LOOKUP


def _find_hydrorivers_shp() -> Path | None:
    """Locate `HydroRIVERS_v10.shp` across configured data/cache directories."""
    river_dir = Path(_frontend_config.RIVER_NETWORKS_DIR)
    cache_dir = Path(_frontend_config.CACHE_DIR)
    return find_hydrorivers_shp((river_dir, cache_dir, cache_dir.parent))


def _find_floodhub_lut_path() -> Path | None:
    """Locate `hydrorivers_floodhub_v1.npz` if present on disk."""
    river_dir = Path(_frontend_config.RIVER_NETWORKS_DIR)
    cache_dir = Path(_frontend_config.CACHE_DIR)
    return find_floodhub_lut((river_dir, cache_dir, cache_dir.parent))


def _can_build_network(model: str) -> bool:
    """Return True if all source files required to build `model` exist."""
    if model == 'geoglows':
        return _geoglows_gpkg().is_file() and _geoglows_attrs().is_file()
    if model == 'floodhub':
        return _find_hydrorivers_shp() is not None
    glofas_dir = _glofas_dir()
    return (glofas_dir / 'upArea_repaired.nc').is_file() and (
        glofas_dir / 'ldd_repaired.nc'
    ).is_file()


def _cache_path(model: str) -> Path:
    """Return the `.npz` pyramid cache file path for `model`."""
    base = _geoglows_dir() if model == 'geoglows' else _glofas_dir()
    return base / f'{model}_network_v{CACHE_VERSION}.npz'


def _network(model: str) -> dict[str, Any] | None:
    """Load or build pyramid network for `floodhub`, `glofas`, or `geoglows`."""
    if model in _NETWORKS:
        return _NETWORKS[model]
    table = {
        'floodhub': FLOODHUB_PYRAMID_LOD,
        'glofas': GLOFAS_LOD,
        'geoglows': GEOGLOWS_LOD,
    }[model]
    with _LOCK:
        if model in _NETWORKS:
            return _NETWORKS[model]
        c_path = _cache_path(model)
        sig = pyramid_signature(table)
        net = load_network_pyramid(c_path, sig)
        if net is None:
            if not _can_build_network(model):
                return None
            if model == 'geoglows':
                net = build_geoglows_pyramid(_geoglows_dir())
                extra = {k: v for k, v in net.items() if k != 'levels'}
                save_network_pyramid(c_path, net['levels'], sig, **extra)
            elif model == 'floodhub':
                shp_path = _find_hydrorivers_shp()
                if shp_path is None:
                    return None
                net = build_floodhub_pyramid(
                    shp_path, _find_floodhub_lut_path()
                )
                extra = {k: v for k, v in net.items() if k != 'levels'}
                save_network_pyramid(c_path, net['levels'], sig, **extra)
            else:
                glofas = build_glofas_pyramid(_glofas_dir())
                save_network_pyramid(
                    c_path,
                    glofas['levels'],
                    pyramid_signature(GLOFAS_LOD),
                    cell_lin=glofas['cell_lin'],
                    cell_area=glofas['cell_area'],
                )
                net = glofas
        _NETWORKS[model] = net
        return net


def get_cached_network(model: str) -> dict[str, Any] | None:
    """Return the loaded or built pyramid network for `model`."""
    return _network(model)


def _get_floodhub_lut() -> dict[str, Any] | None:
    """Lazily load pre-cached HydroRIVERS-to-FloodHub lookup tables."""
    global _FLOODHUB_LUT  # noqa: PLW0603
    if _FLOODHUB_LUT is not None:
        return _FLOODHUB_LUT
    with _LOCK:
        if _FLOODHUB_LUT is not None:
            return _FLOODHUB_LUT
        hr_path = _find_floodhub_lut_path()
        if hr_path is None or not hr_path.is_file():
            return None
        river_dir = Path(_frontend_config.RIVER_NETWORKS_DIR)
        cache_dir = Path(_frontend_config.CACHE_DIR)
        fh_path: Path | None = None
        for base in (river_dir, cache_dir, cache_dir.parent):
            cand_fh = base / 'floodhub_gauges_v1.npz'
            if fh_path is None and cand_fh.is_file():
                fh_path = cand_fh
        with np.load(hr_path, allow_pickle=False) as z:
            hyriv_id = z['hyriv_id'].astype(np.int32)
            hybas_l12 = z['hybas_l12'].astype(np.int64)
            has_fc = z['has_forecast'].astype(np.bool_)
        hybas_to_sev: dict[int, int] = {}
        phys_lats = np.empty(0, dtype=np.float32)
        phys_lons = np.empty(0, dtype=np.float32)
        phys_gids = np.empty(0, dtype='U32')
        phys_sevs = np.empty(0, dtype=np.int8)
        if fh_path is not None:
            with np.load(fh_path, allow_pickle=False) as fz:
                fh_hb = fz['hybas_id'].astype(np.int64)
                fh_sev = fz['severity_rank'].astype(np.int8)
                fh_fc = fz['has_forecast'].astype(np.bool_)
                active_mask = fh_fc & (fh_hb > 0) & (fh_sev > 0)
                for hb_val, s_val in zip(
                    fh_hb[active_mask],
                    fh_sev[active_mask],
                    strict=False,
                ):
                    hybas_to_sev[int(hb_val)] = int(s_val)
                phys_mask = fh_fc & (fh_hb == 0)
                phys_lats = fz['lat'][phys_mask].astype(np.float32)
                phys_lons = fz['lon'][phys_mask].astype(np.float32)
                phys_gids = fz['gauge_id'][phys_mask]
                phys_sevs = fh_sev[phys_mask]
        _FLOODHUB_LUT = {
            'hyriv_id': hyriv_id,
            'hybas_l12': hybas_l12,
            'has_forecast': has_fc,
            'hybas_to_sev': hybas_to_sev,
            'phys_lats': phys_lats,
            'phys_lons': phys_lons,
            'phys_gids': phys_gids,
            'phys_sevs': phys_sevs,
        }
        return _FLOODHUB_LUT


def _floodhub_features(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Extract FloodHub river reaches for the viewport and zoom level."""
    min_order = lod_for_zoom(FLOODHUB_LOD, zoom)[1]
    lut = _get_floodhub_lut()
    hybas_to_sev = lut['hybas_to_sev'] if lut is not None else {}

    net = _network('floodhub')
    if net is not None:
        row = lod_for_zoom(FLOODHUB_PYRAMID_LOD, zoom)
        level_idx = [r for r in FLOODHUB_PYRAMID_LOD if r[2] is not None].index(
            row
        )
        feats = extract_level_features(
            net['levels'][level_idx],
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            hybas_to_sev=hybas_to_sev,
        )
        fc_count = sum(
            1 for f in feats if (f.get('properties') or {}).get('has_forecast')
        )
        return feats, min_order, fc_count

    gj = HydroRiverNetwork('hydroatlas').get_rivers_in_bbox(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        zoom=zoom,
        min_stream_order=min_order,
        prefer_cache=True,
    )
    raw_feats = gj.get('features') or []

    rids = np.zeros(len(raw_feats), dtype=np.int32)
    if lut is not None and raw_feats:
        for i, f in enumerate(raw_feats):
            rid_str = str(
                (f.get('properties') or {}).get('reach_id') or ''
            ).removeprefix('HYRIV_')
            if rid_str.isdigit():
                rids[i] = int(rid_str)
        hyriv_arr = lut['hyriv_id']
        pos = np.searchsorted(hyriv_arr, rids)
        safe_pos = np.minimum(pos, len(hyriv_arr) - 1)
        matched = (
            (rids > 0) & (pos < len(hyriv_arr)) & (hyriv_arr[safe_pos] == rids)
        )
        has_fc_arr = matched & lut['has_forecast'][safe_pos]
        hybas_arr = np.where(matched, lut['hybas_l12'][safe_pos], 0)
    else:
        has_fc_arr = np.ones(len(raw_feats), dtype=np.bool_)
        hybas_arr = np.zeros(len(raw_feats), dtype=np.int64)

    feats = []
    fc_count = 0
    for i, f in enumerate(raw_feats):
        p = f.get('properties') or {}
        reach_id = p.get('reach_id')
        has_fc = bool(has_fc_arr[i])
        hb_id = int(hybas_arr[i]) or int(p.get('hydrobasins_unit') or 0)
        gid = f'hybas_{hb_id}' if (has_fc and hb_id > 0) else None
        sev_rank = int(hybas_to_sev.get(hb_id, 0)) if has_fc else 0
        if has_fc:
            fc_count += 1
        feats.append(
            {
                'type': 'Feature',
                'geometry': f.get('geometry'),
                'properties': {
                    'river_id': reach_id,
                    'upstream_area_km2': p.get('upstream_area_km2'),
                    'stream_order': p.get('stream_order'),
                    'has_forecast': has_fc,
                    'gauge_id': gid,
                    'severity_rank': sev_rank,
                },
            }
        )
    return feats, min_order, fc_count


def _find_promoted_indices(  # noqa: PLR0913, PLR0917
    model: str,
    feats: list[dict[str, Any]],
    fine_lvl: dict[str, Any],
    in_box: np.ndarray,
    active_keys: set[str],
    ows_grid: np.ndarray | None,
) -> np.ndarray | None:
    """Return index array of promoted flood reaches in `fine_lvl`."""
    rids = fine_lvl.get('river_id')
    hbs = fine_lvl.get('hybas_l12')
    if model == 'floodhub':
        existing_gids = {
            str((f.get('properties') or {}).get('gauge_id') or '')
            for f in feats
        }
        needed_hb = [
            int(k[6:])
            for k in active_keys
            if k.startswith('hybas_')
            and k[6:].isdigit()
            and k not in existing_gids
        ]
        if not needed_hb or hbs is None:
            return None
        return np.flatnonzero(
            in_box & np.isin(hbs, np.asarray(needed_hb, dtype=np.int64))
        )
    if model == 'geoglows':
        existing_rids = {
            int((f.get('properties') or {}).get('river_id') or 0)
            for f in feats
            if (f.get('properties') or {}).get('river_id') is not None
        }
        needed_rids = [
            int(k)
            for k in active_keys
            if k.isdigit() and int(k) not in existing_rids
        ]
        if not needed_rids or rids is None:
            return None
        return np.flatnonzero(
            in_box & np.isin(rids, np.asarray(needed_rids, dtype=np.int32))
        )
    if ows_grid is None:
        return None
    offsets = fine_lvl['offsets']
    coords = fine_lvl['coords']
    starts = offsets[:-1]
    ends = np.maximum(starts, offsets[1:] - 1)
    mids = (starts + ends) // 2
    ranks = np.zeros(len(starts), dtype=np.uint8)
    for idx_arr in (starts, mids, ends):
        r_i = np.clip(
            (
                (_GLOFAS_OWS_TOP_LAT - coords[idx_arr, 1]) / _GLOFAS_OWS_RES_DEG
            ).astype(np.int32),
            0,
            _GLOFAS_OWS_MAX_ROW,
        )
        c_i = np.clip(
            (
                (coords[idx_arr, 0] - _GLOFAS_OWS_LEFT_LON)
                / _GLOFAS_OWS_RES_DEG
            ).astype(np.int32),
            0,
            _GLOFAS_OWS_MAX_COL,
        )
        ranks = np.maximum(ranks, ows_grid[r_i, c_i])
    return np.flatnonzero(in_box & (ranks >= 1))


def _append_promoted_feature(
    model: str,
    feats: list[dict[str, Any]],
    fine_lvl: dict[str, Any],
    idx_val: int,
    existing_end_keys: set[tuple[int, int]],
) -> None:
    """Append a single promoted flood feature from `fine_lvl` at `idx_val`."""
    offsets = fine_lvl['offsets']
    s = int(offsets[idx_val])
    e = int(offsets[idx_val + 1])
    if e - s < _MIN_LINE_VERTICES:
        return
    coords = fine_lvl['coords']
    amax = fine_lvl['amax']
    amin = fine_lvl['amin']
    rids = fine_lvl.get('river_id')
    hbs = fine_lvl.get('hybas_l12')
    line_coords = np.round(coords[s:e].astype(np.float64), 4).tolist()
    geom = {'type': 'LineString', 'coordinates': line_coords}
    base_props: dict[str, Any] = {
        'upstream_area_km2': round(float(amax[idx_val]), 1),
        'area_min_km2': round(float(amin[idx_val]), 1),
        'promoted_flood': True,
    }
    if model == 'floodhub':
        hb_id = int(hbs[idx_val]) if hbs is not None else 0
        rid_val = int(rids[idx_val]) if rids is not None else 0
        base_props.update(
            river_id=f'HYRIV_{rid_val}' if rid_val > 0 else None,
            has_forecast=True,
            gauge_id=f'hybas_{hb_id}' if hb_id > 0 else None,
        )
        feats.append(
            {'type': 'Feature', 'geometry': geom, 'properties': base_props}
        )
        return
    if model == 'geoglows':
        base_props['river_id'] = int(rids[idx_val]) if rids is not None else 0
        feats.append(
            {'type': 'Feature', 'geometry': geom, 'properties': base_props}
        )
        return
    end_key = (
        round(line_coords[-1][0] * 100),
        round(line_coords[-1][1] * 100),
    )
    if end_key in existing_end_keys:
        return
    existing_end_keys.add(end_key)
    feats.append(
        {'type': 'Feature', 'geometry': geom, 'properties': base_props}
    )


def _promote_active_flood_features(  # noqa: PLR0913, PLR0917
    model: str,
    feats: list[dict[str, Any]],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int,
    active_keys: set[str],
    ows_grid: np.ndarray | None = None,
) -> None:
    """Promote active flood reaches from finer levels into coarse zoom views."""
    if zoom >= _PROMOTE_MAX_ZOOM:
        return
    if model in ('floodhub', 'geoglows') and not active_keys:
        return
    if model == 'glofas' and ows_grid is None and not active_keys:
        return
    net = _network(model)
    if net is None or not net.get('levels'):
        return
    levels = net['levels']
    fine_lvl = levels[min(len(levels) - 1, 2)]
    b = fine_lvl['bbox']
    in_box = (
        (b[:, 2] >= min_lon)
        & (b[:, 0] <= max_lon)
        & (b[:, 3] >= min_lat)
        & (b[:, 1] <= max_lat)
    )
    hit = _find_promoted_indices(
        model, feats, fine_lvl, in_box, active_keys, ows_grid
    )
    if hit is None or not hit.size:
        return
    existing_end_keys: set[tuple[int, int]] = set()
    if model == 'glofas':
        for f in feats:
            fc = (f.get('geometry') or {}).get('coordinates') or []
            if fc:
                existing_end_keys.add(
                    (round(fc[-1][0] * 100), round(fc[-1][1] * 100))
                )
    for idx_val in hit[:_MAX_PROMOTED_HITS].tolist():
        _append_promoted_feature(
            model, feats, fine_lvl, idx_val, existing_end_keys
        )


def _extract_non_floodhub_features(  # noqa: PLR0913, PLR0917
    model: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int,
    props: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract GloFAS or GEOGLOWS features for `zoom` and `bbox`."""
    table = {'glofas': GLOFAS_LOD, 'geoglows': GEOGLOWS_LOD}[model]
    row = lod_for_zoom(table, zoom)
    props['min_area_km2'] = row[1]
    if row[2] is None:
        gpkg_path = _geoglows_gpkg()
        attrs_path = _geoglows_attrs()
        if gpkg_path.is_file() and attrs_path.is_file():
            return query_geoglows_reaches(
                gpkg_path,
                _geoglows_lookup(),
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                row[1],
            )
        net = _network(model)
        if net is None or not net.get('levels'):
            props['available'] = False
            return []
        return extract_level_features(
            net['levels'][-1], min_lon, min_lat, max_lon, max_lat
        )
    net = _network(model)
    if net is None:
        props['available'] = False
        return []
    level_idx = [r for r in table if r[2] is not None].index(row)
    return extract_level_features(
        net['levels'][level_idx], min_lon, min_lat, max_lon, max_lat
    )


def get_model_network(  # noqa: PLR0913, PLR0917
    model: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return GeoJSON `FeatureCollection` of `model`'s river network in bbox."""
    model = model if model in MODELS else 'floodhub'
    min_lon, max_lon = max(-180.0, min_lon), min(180.0, max_lon)
    min_lat, max_lat = max(-85.0, min_lat), min(85.0, max_lat)
    if min_lon >= max_lon:
        min_lon, max_lon = -180.0, 180.0
    if min_lat >= max_lat:
        min_lat, max_lat = -85.0, 85.0

    props: dict[str, Any] = {
        'model': model,
        'network': NETWORK_LABELS[model],
        'zoom': zoom,
    }
    if model == 'floodhub':
        props['min_area_km2'] = lod_for_zoom(FLOODHUB_PYRAMID_LOD, zoom)[1]
        (
            feats,
            props['min_stream_order'],
            props['forecast_feature_count'],
        ) = _floodhub_features(min_lon, min_lat, max_lon, max_lat, zoom)
    else:
        feats = _extract_non_floodhub_features(
            model, min_lon, min_lat, max_lon, max_lat, zoom, props
        )

    if feats:
        if refresh:
            clear_live_status_cache(model)
        elif zoom < _PROMOTE_MAX_ZOOM:
            active_keys = get_active_flood_keys(model)
            ows_grid = get_glofas_ows_grid() if model == 'glofas' else None
            if active_keys or ows_grid is not None:
                _promote_active_flood_features(
                    model,
                    feats,
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    zoom,
                    active_keys,
                    ows_grid=ows_grid,
                )
        colored_cnt, pending_cnt = enrich_network_features_live(model, feats)
        props['live_colored_count'] = colored_cnt
        props['live_pending_count'] = pending_cnt
        props['active_flood_count'] = sum(
            1
            for f in feats
            if int((f.get('properties') or {}).get('exceedance_rank') or 0) >= 1
        )
        props['last_refreshed_utc'] = get_last_refresh_iso(model)

    props['feature_count'] = len(feats)
    return {'type': 'FeatureCollection', 'properties': props, 'features': feats}


def snap_glofas_cell(
    lat: float,
    lon: float,
    target_area_km2: float,
    area_range: tuple[float, float] | None = None,
    radius_cells: int = 3,
) -> dict[str, Any] | None:
    """Snap to a GloFAS river cell near `(lat, lon)` matching upstream area."""
    if not target_area_km2 or target_area_km2 <= 0:
        return None
    net = _network('glofas')
    return snap_glofas_cell_from_network(
        net,
        lat,
        lon,
        target_area_km2=target_area_km2,
        area_range=area_range,
        radius_cells=radius_cells,
    )


def snap_geoglows_reach(
    lat: float,
    lon: float,
    target_area_km2: float,
    area_range: tuple[float, float] | None = None,
    radius_deg: float = 0.15,
) -> dict[str, Any] | None:
    """Snap to a GEOGLOWS reach near `(lat, lon)` matching upstream area."""
    if not target_area_km2 or target_area_km2 <= 0:
        return None
    net = _network('geoglows')
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
    gpkg_path = _geoglows_gpkg()
    attrs_path = _geoglows_attrs()
    if not (gpkg_path.is_file() and attrs_path.is_file()):
        return None
    return snap_geoglows_reach_from_gpkg(
        gpkg_path,
        _geoglows_lookup(),
        lat,
        lon,
        target_area_km2=target_area_km2,
        area_range=area_range,
        radius_deg=radius_deg,
    )


def resolve_click(  # noqa: PLR0913, PLR0917
    lat: float,
    lon: float,
    upstream_area_km2: float | str | None,
    area_min_km2: float | str | None = None,
    network: str | None = None,
    river_id: int | str | None = None,
) -> dict[str, Any] | None:
    """Map a click on one model's river line to all models on the same river."""
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
    """Build every derived network cache."""
    for model in ('glofas', 'geoglows', 'floodhub'):
        t = time.time()
        net = _network(model)
        sizes = (
            [len(lvl['offsets']) - 1 for lvl in net['levels']] if net else None
        )
        _LOG.info(
            '%s: lines per level %s (%.1f s)', model, sizes, time.time() - t
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if '--build' in sys.argv:
        build_all()
