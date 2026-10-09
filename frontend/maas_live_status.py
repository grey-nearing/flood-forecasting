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

"""Real-time forecast flow percentile and flood exceedance enrichment for MaaS river networks.

Maintains an in-memory and disk-backed live forecast cache for all 3 MaaS providers
(`floodhub`, `geoglows`, `glofas`) and populates uncached viewport
reaches asynchronously in background batches so `/api/maas/network` always responds
in < 15 ms while coloring river segments with 100% real-time forecast data.
"""

from __future__ import annotations

import concurrent.futures
from datetime import UTC, datetime, timedelta
import io
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from PIL import Image
import requests

from frontend.config import CACHE_DIR, RIVER_NETWORKS_DIR
from utils.file_paths import (
    FLOODHUB_API_KEY_FILE,
    FLOODHUB_BASE_URL,
    GEOGLOWS_ARCGIS_LIVEFEEDS_URL as ARCGIS_GAUGES_QUERY_URL,
    GLOFAS_BASE_URL,
    GLOFAS_OWS_URL,
)

_LOG = logging.getLogger(__name__)

_LIVE_TTL_S = 1800.0  # 30 minutes
_DISK_CACHE_PATH = Path(CACHE_DIR) / 'live_network_status_v1.json'
_GLOFAS_OWS_CACHE_PATH = Path(CACHE_DIR) / 'glofas_ows_rank_grid_v1.npz'
_GLOFAS_OWS_URL = GLOFAS_OWS_URL
_ARCGIS_GEOGLOWS_URL = ARCGIS_GAUGES_QUERY_URL
_OPEN_METEO_FLOOD_URL = GLOFAS_BASE_URL
_FLOODHUB_BASE_URL = FLOODHUB_BASE_URL


def _load_default_floodhub_key() -> str:
    env_key = os.environ.get('FLOODHUB_API_KEY', '').strip()
    if env_key:
        return env_key
    key_file = FLOODHUB_API_KEY_FILE
    if key_file.is_file():
        return key_file.read_text(encoding='utf-8').strip()
    return ''


_DEFAULT_FLOODHUB_KEY = _load_default_floodhub_key()

_LOCK = threading.Lock()
_LIVE_STATUS: dict[str, dict[str, dict[str, Any]]] = {
    'floodhub': {},
    'geoglows': {},
    'glofas': {},
}
_IN_FLIGHT: dict[str, set[str]] = {
    'floodhub': set(),
    'geoglows': set(),
    'glofas': set(),
}
_BG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix='maas-live-net'
)
_DISK_LOADED = False
_LAST_DISK_SAVE = 0.0
_SPATIAL_RP_INDEX: dict[str, np.ndarray] | None = None
_GLOFAS_OWS_GRID: np.ndarray | None = None
_GLOFAS_OWS_TS: float = 0.0
_GLOFAS_OWS_IN_FLIGHT: bool = False
_LAST_REFRESH_TS: dict[str, float] = {
    'floodhub': 0.0,
    'geoglows': 0.0,
    'glofas': 0.0,
}


def compute_live_flow_metrics(
    q_flow: float,
    q2: float,
    q5: float = 0.0,
    q20: float = 0.0,
    min_rank: int = 0,
    q50: float = 0.0,
) -> dict[str, Any]:
    """Convert real-time forecast flow `q_flow` and return-period thresholds into percentile & rank."""
    qf = max(0.0, float(q_flow)) if np.isfinite(q_flow) else 0.0
    q2_val = float(q2) if (np.isfinite(q2) and q2 > 0.0) else 0.0
    q5_val = float(q5) if (np.isfinite(q5) and q5 > q2_val) else (1.66 * q2_val)
    q20_val = (
        float(q20) if (np.isfinite(q20) and q20 > q5_val) else (2.50 * q2_val)
    )
    q50_val = (
        float(q50) if (np.isfinite(q50) and q50 > q20_val) else 0.0
    )

    rank = 0
    if q50_val > 0.0 and qf >= q50_val:
        rank = 4
    elif q20_val > 0.0 and qf >= q20_val:
        rank = 3
    elif q5_val > 0.0 and qf >= q5_val:
        rank = 2
    elif q2_val > 0.0 and qf >= q2_val:
        rank = 1
    rank = max(rank, int(min_rank))

    if q2_val > 0.0:
        r = qf / q2_val
        r5 = max(1.05, q5_val / q2_val)
        r20 = max(r5 + 0.10, q20_val / q2_val)
        # Calibrated against global river Q_now / Q_2 distributions:
        #   r < 0.12       -> < 25th pct (low / baseflow)
        #   0.12 <= r < 0.65 -> 25–75th pct (normal flow; median r ~ 0.30)
        #   0.65 <= r < 1.00 -> 75–90th pct (high in-bank flow, below Q2 flood threshold)
        #   r >= 1.00      -> >= 90th pct (>= 2-yr return period flood alert)
        r_knots = [0.0, 0.12, 0.32, 0.65, 0.85, 1.00, r5, r20, r20 * 1.5]
        p_knots = [2.0, 25.0, 50.0, 75.0, 84.0, 90.0, 95.0, 99.0, 99.9]
        pct = float(np.interp(r, r_knots, p_knots))
    else:
        pct = 50.0 if qf > 0.0 else 5.0

    if rank == 1 and pct < 90.5:
        pct = 91.0
    elif rank == 2 and pct < 95.5:
        pct = 96.0
    elif rank == 3 and pct < 99.1:
        pct = 99.3
    elif rank >= 4 and pct < 99.7:
        pct = 99.8

    return {
        'flow_percentile': round(min(max(pct, 1.0), 99.9), 1),
        'exceedance_rank': int(rank),
        'current_flow_m3s': round(qf, 2),
        'return_period_2yr': round(q2_val, 2) if q2_val > 0.0 else None,
        'ts': time.time(),
    }


def clear_live_status_cache(model: str | None = None) -> None:
    """Clear cached live network flow status for `model` (or all models if `None`)."""
    global _GLOFAS_OWS_GRID, _GLOFAS_OWS_TS, _GLOFAS_OWS_IN_FLIGHT  # noqa: PLW0603
    _ensure_disk_cache_loaded()
    clear_ows = model in (None, 'glofas')
    with _LOCK:
        targets = [model] if (model and model in _LIVE_STATUS) else list(_LIVE_STATUS.keys())
        for m in targets:
            _LIVE_STATUS[m].clear()
            _IN_FLIGHT[m].clear()
            _LAST_REFRESH_TS[m] = time.time()
        if clear_ows:
            _GLOFAS_OWS_GRID = None
            _GLOFAS_OWS_TS = 0.0
            _GLOFAS_OWS_IN_FLIGHT = False
    try:
        if clear_ows and _GLOFAS_OWS_CACHE_PATH.exists():
            _GLOFAS_OWS_CACHE_PATH.unlink(missing_ok=True)
        if model is None and _DISK_CACHE_PATH.exists():
            _DISK_CACHE_PATH.unlink(missing_ok=True)
        else:
            _maybe_save_disk_cache(force=True)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug('Failed to update disk cache after clear: %s', exc)


def get_active_flood_keys(model: str) -> set[str]:
    """Return the set of cached reach keys with `exceedance_rank >= 1` for `model`."""
    _ensure_disk_cache_loaded()
    cutoff = time.time() - _LIVE_TTL_S
    with _LOCK:
        tbl = _LIVE_STATUS.get(model) or {}
        return {
            k
            for k, v in tbl.items()
            if float(v.get('ts', 0.0)) >= cutoff and int(v.get('exceedance_rank', 0)) >= 1
        }


def get_glofas_ows_grid() -> np.ndarray | None:
    """Return cached 0.05-deg `(2700, 7200)` ECMWF GloFAS flood summary exceedance rank grid."""
    _ensure_disk_cache_loaded()
    with _LOCK:
        if _GLOFAS_OWS_GRID is not None and (time.time() - _GLOFAS_OWS_TS) <= _LIVE_TTL_S:
            return _GLOFAS_OWS_GRID
    return None


def get_last_refresh_iso(model: str) -> str | None:
    """Return ISO-8601 UTC timestamp of the newest live entry for `model`, if any."""
    _ensure_disk_cache_loaded()
    with _LOCK:
        tbl = _LIVE_STATUS.get(model) or {}
        ts = max((float(v.get('ts', 0.0)) for v in tbl.values()), default=0.0)
        ts = max(ts, _LAST_REFRESH_TS.get(model, 0.0))
        if model == 'glofas':
            ts = max(ts, _GLOFAS_OWS_TS)
    if ts <= 0.0:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _ensure_disk_cache_loaded() -> None:
    global _DISK_LOADED, _GLOFAS_OWS_GRID, _GLOFAS_OWS_TS  # noqa: PLW0603
    if _DISK_LOADED:
        return
    with _LOCK:
        if _DISK_LOADED:
            return
        _DISK_LOADED = True
        cutoff = time.time() - _LIVE_TTL_S
        if _DISK_CACHE_PATH.exists():
            try:
                raw = json.loads(_DISK_CACHE_PATH.read_text(encoding='utf-8'))
                for model, table in (raw or {}).items():
                    if model in _LIVE_STATUS and isinstance(table, dict):
                        for k, v in table.items():
                            if isinstance(v, dict) and float(v.get('ts', 0.0)) >= cutoff:
                                _LIVE_STATUS[model][str(k)] = v
            except Exception as exc:  # noqa: BLE001
                _LOG.debug('Failed to load live network status cache: %s', exc)
        if _GLOFAS_OWS_CACHE_PATH.exists():
            try:
                with np.load(_GLOFAS_OWS_CACHE_PATH, allow_pickle=False) as z:
                    ts_val = float(z['ts'][0])
                    if ts_val >= cutoff:
                        _GLOFAS_OWS_GRID = z['rank_grid'].astype(np.uint8)
                        _GLOFAS_OWS_TS = ts_val
            except Exception as exc:  # noqa: BLE001
                _LOG.debug('Failed to load GloFAS OWS rank grid cache: %s', exc)


def _maybe_save_disk_cache(*, force: bool = False) -> None:
    global _LAST_DISK_SAVE  # noqa: PLW0603
    now = time.time()
    if not force and now - _LAST_DISK_SAVE < 15.0:
        return
    with _LOCK:
        if not force and now - _LAST_DISK_SAVE < 15.0:
            return
        _LAST_DISK_SAVE = now
        cutoff = now - _LIVE_TTL_S
        snapshot = {
            m: {k: v for k, v in tbl.items() if float(v.get('ts', 0.0)) >= cutoff}
            for m, tbl in _LIVE_STATUS.items()
        }
    try:
        _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DISK_CACHE_PATH.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(snapshot), encoding='utf-8')
        tmp.replace(_DISK_CACHE_PATH)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug('Failed to save live network status cache: %s', exc)


def _get_spatial_rp_index() -> dict[str, np.ndarray] | None:
    """Load sorted GEOGLOWS `(lat, lon, area, q2, q5, q25)` table for spatial threshold lookup."""
    global _SPATIAL_RP_INDEX  # noqa: PLW0603
    if _SPATIAL_RP_INDEX is not None:
        return _SPATIAL_RP_INDEX
    with _LOCK:
        if _SPATIAL_RP_INDEX is not None:
            return _SPATIAL_RP_INDEX
        try:
            from frontend import maas_networks  # noqa: PLC0415
            import maas.geoglows as _gg  # noqa: PLC0415

            net = maas_networks._network('geoglows')
            if net is not None and 'reach_lat' in net:
                r_lat = net['reach_lat']
                r_lon = net['reach_lon']
                r_links = net['reach_linkno']
                r_area = net['reach_area']
            else:
                gg_dir = Path(RIVER_NETWORKS_DIR) / 'geoglows_v2'
                if not gg_dir.exists():
                    gg_dir = Path(CACHE_DIR) / 'geoglows_v2'
                net_path = gg_dir / 'geoglows_network_v1.npz'
                if not net_path.exists():
                    return None
                with np.load(net_path, allow_pickle=False) as nz:
                    r_lat = nz['reach_lat'].astype(np.float32)
                    r_lon = nz['reach_lon'].astype(np.float32)
                    r_links = nz['reach_linkno'].astype(np.int32)
                    r_area = nz['reach_area'].astype(np.float32)

            _gg.lookup_cached_geoglows_return_periods(760069805)
            if _gg._GEOGLOWS_RP_TABLE is not None:
                rp_rids = _gg._GEOGLOWS_RP_TABLE['river_id']
                rp_gumbel = _gg._GEOGLOWS_RP_TABLE['values']
            else:
                gg_dir = Path(RIVER_NETWORKS_DIR) / 'geoglows_v2'
                if not gg_dir.exists():
                    gg_dir = Path(CACHE_DIR) / 'geoglows_v2'
                rp_path = gg_dir / 'geoglows_return_periods_v1.npz'
                if not rp_path.exists():
                    return None
                with np.load(rp_path, allow_pickle=False) as rz:
                    rp_rids = rz['river_id'].astype(np.int32)
                    rp_gumbel = rz['gumbel_daily'].astype(np.float32)

            pos = np.searchsorted(rp_rids, r_links)
            safe_pos = np.minimum(pos, len(rp_rids) - 1)
            valid = (pos < len(rp_rids)) & (rp_rids[safe_pos] == r_links)
            q2 = np.where(valid, rp_gumbel[safe_pos, 0], 0.0).astype(np.float32)
            q5 = np.where(valid, rp_gumbel[safe_pos, 1], 0.0).astype(np.float32)
            q25 = np.where(valid, rp_gumbel[safe_pos, 3], 0.0).astype(np.float32)

            _SPATIAL_RP_INDEX = {
                'rp_rids': rp_rids,
                'rp_q2': rp_gumbel[:, 0],
                'rp_q5': rp_gumbel[:, 1],
                'rp_q10': rp_gumbel[:, 2],
                'rp_q25': rp_gumbel[:, 3],
                'rp_q50': rp_gumbel[:, 4],
                'lat': r_lat,
                'lon': r_lon,
                'area': r_area,
                'q2': q2,
                'q5': q5,
                'q25': q25,
            }
            return _SPATIAL_RP_INDEX
        except Exception as exc:  # noqa: BLE001
            _LOG.warning('Failed to initialize spatial RP index: %s', exc)
            return None


def _lookup_spatial_thresholds(
    lat: float,
    lon: float,
    area_km2: float,
) -> tuple[float, float, float]:
    """Return `(q2, q5, q20)` for `(lat, lon, area_km2)` from nearby GEOGLOWS reaches or regional scaling."""
    area = max(float(area_km2 or 500.0), 50.0)
    idx = _get_spatial_rp_index()
    if idx is not None:
        r_lat = idx['lat']
        i0 = int(np.searchsorted(r_lat, lat - 0.45, side='left'))
        i1 = int(np.searchsorted(r_lat, lat + 0.45, side='right'))
        if i1 > i0:
            sub_lat = r_lat[i0:i1]
            sub_lon = idx['lon'][i0:i1]
            sub_q2 = idx['q2'][i0:i1]
            sub_area_all = np.maximum(idx['area'][i0:i1], 10.0)
            kx = math.cos(math.radians(lat))
            dist2 = ((sub_lon - lon) * kx) ** 2 + (sub_lat - lat) ** 2
            # Require comparable drainage area (within 3x) so a main stem never inherits a small tributary's Q2
            near = np.flatnonzero(
                (dist2 <= 0.45 * 0.45)
                & (sub_q2 > 0.0)
                & (sub_area_all >= area * 0.33)
                & (sub_area_all <= area * 3.0)
            )
            if len(near) > 0:
                sub_area = sub_area_all[near]
                score = np.abs(np.log(sub_area / area)) + 2.0 * np.sqrt(dist2[near])
                best = int(i0 + near[int(np.argmin(score))])
                scale = min(max((area / max(float(idx['area'][best]), 10.0)) ** 0.75, 0.4), 2.5)
                q2 = float(idx['q2'][best]) * scale
                q5 = float(idx['q5'][best]) * scale
                q25 = float(idx['q25'][best]) * scale
                return q2, q5, q25

    q2_reg = 0.365 * (area ** 0.75)
    return q2_reg, 1.66 * q2_reg, 2.50 * q2_reg


def _extract_current_floodhub_flow(
    fcs: list[dict[str, Any]],
    now_iso: str,
) -> float | None:
    """Extract current-day real-time discharge from FloodHub `forecasts` list."""
    if not fcs:
        return None
    ranges = fcs[-1].get('forecastRanges') or []
    if not ranges:
        return None
    best_val: float | None = None
    for rng in ranges:
        if not isinstance(rng, dict) or rng.get('value') is None:
            continue
        val = float(rng['value'])
        if not np.isfinite(val):
            continue
        t_start = str(rng.get('forecastStartTime') or '')
        t_end = str(rng.get('forecastEndTime') or '')
        if t_start and t_end and t_start <= now_iso < t_end:
            return val
        if t_start and t_start <= now_iso:
            best_val = val
        elif best_val is None:
            best_val = val
    return best_val


def _fetch_floodhub_batch_worker(items: list[tuple[str, int]]) -> None:
    """Background worker to fetch live FloodHub forecasts and thresholds for `hybas_...` gauges."""
    try:
        api_key = os.environ.get('FLOODHUB_API_KEY', _DEFAULT_FLOODHUB_KEY).strip()
        sess = requests.Session()
        now = datetime.now(UTC)
        now_iso = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        start_str = (now - timedelta(days=2)).strftime('%Y-%m-%d')
        end_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')

        any_updates = False
        for i in range(0, len(items), 250):
            chunk = items[i : i + 250]
            gids = [g for g, _ in chunk]
            try:
                r_fc = sess.get(
                    f'{_FLOODHUB_BASE_URL}/gauges:queryGaugeForecasts',
                    params=[('key', api_key), ('issuedTimeStart', start_str), ('issuedTimeEnd', end_str)]
                    + [('gaugeIds', g) for g in gids],
                    timeout=12.0,
                )
                r_gm = sess.get(
                    f'{_FLOODHUB_BASE_URL}/gaugeModels:batchGet',
                    params=[('key', api_key)] + [('names', f'gaugeModels/{g}') for g in gids],
                    timeout=12.0,
                )
                fc_map = r_fc.json().get('forecasts', {}) if r_fc.ok else {}
                gm_list = r_gm.json().get('gaugeModels', []) if r_gm.ok else []
            except Exception as exc:  # noqa: BLE001
                _LOG.debug('FloodHub live batch error: %s', exc)
                continue

            th_by_gid: dict[str, tuple[float, float, float]] = {}
            for gm in gm_list:
                gid = str(gm.get('gaugeId') or '')
                th = gm.get('thresholds') or {}
                w = float(th.get('warningLevel') or 0.0)
                d = float(th.get('dangerLevel') or 0.0)
                e = float(th.get('extremeDangerLevel') or 0.0)
                if gid and w > 0.0:
                    th_by_gid[gid] = (w, d, e)

            chunk_updates: dict[str, dict[str, Any]] = {}
            for gid in gids:
                f_obj = fc_map.get(gid) or {}
                fcs = f_obj.get('forecasts') or []
                q_now = _extract_current_floodhub_flow(fcs, now_iso)
                q2, q5, q20 = th_by_gid.get(gid, (0.0, 0.0, 0.0))
                if q_now is not None and (q2 > 0.0 or q_now > 0.0):
                    chunk_updates[gid] = compute_live_flow_metrics(
                        q_now, q2, q5, q20, min_rank=0
                    )
            if chunk_updates:
                any_updates = True
                with _LOCK:
                    _LIVE_STATUS['floodhub'].update(chunk_updates)
        if any_updates:
            _maybe_save_disk_cache()
    finally:
        with _LOCK:
            for gid, _ in items:
                _IN_FLIGHT['floodhub'].discard(gid)


def _geoglows_entry_from_attrs(
    cid: int,
    mf: float,
    rp_val: int,
    idx: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    """Build a GEOGLOWS live status entry using GEOGLOWS 2/10/25/50-yr return period standards."""
    min_rank = (
        4
        if rp_val >= 50
        else 3
        if rp_val >= 25
        else 2
        if rp_val >= 10
        else 1
        if rp_val >= 2
        else 0
    )
    q2, q10, q25, q50 = 0.0, 0.0, 0.0, 0.0
    if idx is not None:
        rp_rids = idx['rp_rids']
        pos = int(np.searchsorted(rp_rids, cid))
        if pos < len(rp_rids) and int(rp_rids[pos]) == cid:
            q2 = float(idx['rp_q2'][pos])
            q10 = float(idx.get('rp_q10', idx['rp_q5'])[pos])
            q25 = float(idx['rp_q25'][pos])
            if 'rp_q50' in idx:
                q50 = float(idx['rp_q50'][pos])
    entry = compute_live_flow_metrics(
        mf, q2, q10, q25, min_rank=min_rank, q50=q50
    )
    entry['return_period_exceeded'] = rp_val
    return entry


def _fetch_geoglows_batch_worker(rids: list[int]) -> None:
    """Background worker to fetch live GEOGLOWS `meanflow` and `returnperiod` from ArcGIS LiveFeeds."""
    try:
        idx = _get_spatial_rp_index()
        sess = requests.Session()
        any_updates = False
        with _LOCK:
            need_global_floods = len(_LIVE_STATUS['geoglows']) < 50

        for i in range(0, len(rids), 500):
            batch = rids[i : i + 500]
            in_clause = ','.join(str(int(x)) for x in batch)
            try:
                resp = sess.post(
                    _ARCGIS_GEOGLOWS_URL,
                    data={
                        'where': f'comid IN ({in_clause})',
                        'outFields': 'comid,meanflow,returnperiod',
                        'returnGeometry': 'false',
                        'f': 'json',
                    },
                    timeout=15.0,
                )
                if not resp.ok:
                    continue
                feats = (resp.json() or {}).get('features') or []
            except Exception as exc:  # noqa: BLE001
                _LOG.debug('GEOGLOWS live batch error: %s', exc)
                continue

            chunk_updates: dict[str, dict[str, Any]] = {}
            for f in feats:
                attrs = f.get('attributes') or {}
                cid = int(attrs.get('comid') or 0)
                if cid <= 0:
                    continue
                chunk_updates[str(cid)] = _geoglows_entry_from_attrs(
                    cid,
                    float(attrs.get('meanflow') or 0.0),
                    int(attrs.get('returnperiod') or 0),
                    idx,
                )
            if chunk_updates:
                any_updates = True
                with _LOCK:
                    _LIVE_STATUS['geoglows'].update(chunk_updates)

        # After viewport reaches are populated, pull active global returnperiod >= 2 reaches
        if need_global_floods:
            try:
                resp_gf = sess.post(
                    _ARCGIS_GEOGLOWS_URL,
                    data={
                        'where': 'returnperiod >= 2',
                        'outFields': 'comid,meanflow,returnperiod',
                        'returnGeometry': 'false',
                        'f': 'json',
                    },
                    timeout=10.0,
                )
                if resp_gf.ok:
                    gf_feats = (resp_gf.json() or {}).get('features') or []
                    gf_updates: dict[str, dict[str, Any]] = {}
                    for f in gf_feats:
                        attrs = f.get('attributes') or {}
                        cid = int(attrs.get('comid') or 0)
                        if cid > 0:
                            gf_updates[str(cid)] = _geoglows_entry_from_attrs(
                                cid,
                                float(attrs.get('meanflow') or 0.0),
                                int(attrs.get('returnperiod') or 0),
                                idx,
                            )
                    if gf_updates:
                        any_updates = True
                        with _LOCK:
                            _LIVE_STATUS['geoglows'].update(gf_updates)
            except Exception as exc:  # noqa: BLE001
                _LOG.debug('GEOGLOWS global flood scan skipped: %s', exc)

        if any_updates:
            _maybe_save_disk_cache()
    finally:
        with _LOCK:
            for r in rids:
                _IN_FLIGHT['geoglows'].discard(str(r))



def _fetch_glofas_ows_grid_worker() -> None:
    """Background worker to fetch live 0.05-deg global GloFAS flood summary raster from ECMWF OWS."""
    global _GLOFAS_OWS_GRID, _GLOFAS_OWS_TS, _GLOFAS_OWS_IN_FLIGHT  # noqa: PLW0603
    try:
        sess = requests.Session()
        tiles: list[np.ndarray] = []
        for bbox in ('-60,-180,75,0', '-60,0,75,180'):
            resp = sess.get(
                _GLOFAS_OWS_URL,
                params={
                    'SERVICE': 'WMS',
                    'REQUEST': 'GetMap',
                    'VERSION': '1.3.0',
                    'LAYERS': 'sumAL41EGE',
                    'STYLES': 'default',
                    'CRS': 'EPSG:4326',
                    'BBOX': bbox,
                    'WIDTH': '3600',
                    'HEIGHT': '2700',
                    'FORMAT': 'image/png',
                    'TRANSPARENT': 'TRUE',
                },
                timeout=15.0,
            )
            if not resp.ok or 'image/png' not in str(resp.headers.get('content-type') or ''):
                return
            img = Image.open(io.BytesIO(resp.content)).convert('RGBA')
            tiles.append(np.asarray(img, dtype=np.uint8))

        grid = np.concatenate(tiles, axis=1)  # (2700, 7200, 4)
        r_ch = grid[:, :, 0].astype(np.int16)
        g_ch = grid[:, :, 1].astype(np.int16)
        b_ch = grid[:, :, 2].astype(np.int16)
        a_ch = grid[:, :, 3]
        rank_grid = np.zeros((2700, 7200), dtype=np.uint8)
        mask = a_ch > 30
        purple = mask & (r_ch > 120) & (b_ch > 140) & (b_ch > g_ch + 30)
        red = mask & (~purple) & (r_ch > 190) & (g_ch < 170) & (r_ch > g_ch + 50)
        yellow = mask & (~purple) & (~red)
        rank_grid[yellow] = 1
        rank_grid[red] = 2
        rank_grid[purple] = 3

        padded = np.pad(rank_grid, 1, mode='edge')
        dilated = np.maximum.reduce(
            [padded[dr : dr + 2700, dc : dc + 7200] for dr in range(3) for dc in range(3)]
        )
        now_ts = time.time()
        with _LOCK:
            _GLOFAS_OWS_GRID = dilated
            _GLOFAS_OWS_TS = now_ts
        try:
            _GLOFAS_OWS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                _GLOFAS_OWS_CACHE_PATH,
                rank_grid=dilated,
                ts=np.asarray([now_ts], dtype=np.float64),
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.debug('Failed to save GloFAS OWS rank grid cache: %s', exc)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug('GloFAS OWS global flood summary fetch failed: %s', exc)
    finally:
        with _LOCK:
            _GLOFAS_OWS_IN_FLIGHT = False


def sample_glofas_ows_polyline(coords: list[Any], grid: np.ndarray) -> int:
    """Sample up to 5 vertices of `coords` against `grid` `(2700, 7200)` and return max exceedance rank."""
    n = len(coords)
    if n == 0:
        return 0
    indices = (0, n // 4, n // 2, (n * 3) // 4, n - 1) if n >= 5 else range(n)
    max_rank = 0
    for idx in indices:
        pt = coords[idx]
        lon, lat = float(pt[0]), float(pt[1])
        r_idx = int((75.0 - lat) / 0.05)
        c_idx = int((lon + 180.0) / 0.05)
        if 0 <= r_idx < 2700 and 0 <= c_idx < 7200:
            rk = int(grid[r_idx, c_idx])
            if rk > max_rank:
                max_rank = rk
    return max_rank


def grid_feature_key(coords: list[Any], area: float) -> tuple[str, float, float]:
    """Return simplification-invariant `(key, lat, lon)` for a GloFAS polyline."""
    pt = coords[-1] if len(coords) >= 2 else coords[0]
    lon, lat = float(pt[0]), float(pt[1])
    area_int = int(round(max(area, 0.0)))
    return f'{lat:.3f},{lon:.3f}:{area_int}', lat, lon


def enrich_network_features_live(
    model: str,
    feats: list[dict[str, Any]],
    *,
    trigger_fetch: bool = True,
) -> tuple[int, int]:
    """Attach live `flow_percentile`, `exceedance_rank`, and `current_flow_m3s` to `feats`.

    Returns `(colored_count, pending_count)` without blocking on network I/O.
    """
    global _GLOFAS_OWS_IN_FLIGHT  # noqa: PLW0603
    if not feats or model not in _LIVE_STATUS:
        return 0, 0

    _ensure_disk_cache_loaded()
    now = time.time()
    cutoff = now - _LIVE_TTL_S

    with _LOCK:
        cache = _LIVE_STATUS[model]
        in_flight = _IN_FLIGHT[model]
        ows_grid = (
            _GLOFAS_OWS_GRID
            if (model == 'glofas' and _GLOFAS_OWS_GRID is not None and _GLOFAS_OWS_TS >= cutoff)
            else None
        )

        colored = 0
        viewport_pending = 0
        missing_fh: dict[str, tuple[float, int]] = {}
        missing_gg: dict[int, float] = {}
        need_ows_grid = False

        for f in feats:
            p = f.get('properties')
            if not isinstance(p, dict):
                continue
            area = float(p.get('upstream_area_km2') or 0.0)

            if model == 'floodhub':
                if not p.get('has_forecast'):
                    continue
                gid = str(p.get('gauge_id') or '')
                entry = cache.get(gid) if gid else None
                if entry is not None and float(entry.get('ts', 0.0)) >= cutoff:
                    p['flow_percentile'] = entry['flow_percentile']
                    p['exceedance_rank'] = int(entry['exceedance_rank'])
                    p['current_flow_m3s'] = entry.get('current_flow_m3s')
                    p['return_period_2yr'] = entry.get('return_period_2yr')
                    colored += 1
                elif gid and gid.startswith('hybas_'):
                    viewport_pending += 1
                    if gid not in in_flight:
                        prev = missing_fh.get(gid)
                        if prev is None or area > prev[0]:
                            missing_fh[gid] = (area, 0)

            elif model == 'geoglows':
                rid_raw = p.get('river_id')
                if rid_raw is None:
                    continue
                rid = int(rid_raw)
                key = str(rid)
                entry = cache.get(key)
                if entry is not None and float(entry.get('ts', 0.0)) >= cutoff:
                    p['flow_percentile'] = entry['flow_percentile']
                    p['exceedance_rank'] = entry['exceedance_rank']
                    p['current_flow_m3s'] = entry.get('current_flow_m3s')
                    p['return_period_2yr'] = entry.get('return_period_2yr')
                    if entry.get('return_period_exceeded') is not None:
                        p['return_period_exceeded'] = entry['return_period_exceeded']
                    colored += 1
                else:
                    viewport_pending += 1
                    if key not in in_flight and area > missing_gg.get(rid, -1.0):
                        missing_gg[rid] = area

            else:
                coords = (f.get('geometry') or {}).get('coordinates') or []
                if not coords:
                    continue
                key, lat, lon = grid_feature_key(coords, area)
                if not p.get('river_id'):
                    p['river_id'] = f'{lat:.3f},{lon:.3f}'
                entry = cache.get(key) or cache.get(f'{lat:.3f},{lon:.3f}')
                if entry is not None and float(entry.get('ts', 0.0)) >= cutoff:
                    p['flow_percentile'] = entry['flow_percentile']
                    p['exceedance_rank'] = entry['exceedance_rank']
                    p['current_flow_m3s'] = entry.get('current_flow_m3s')
                    p['return_period_2yr'] = entry.get('return_period_2yr')
                    colored += 1
                elif ows_grid is not None:
                    rk = sample_glofas_ows_polyline(coords, ows_grid)
                    p['exceedance_rank'] = rk
                    p['flow_percentile'] = {0: 55.0, 1: 92.0, 2: 96.5, 3: 99.4}.get(rk, 55.0)
                    colored += 1
                else:
                    viewport_pending += 1
                    need_ows_grid = True

        if trigger_fetch:
            if model == 'floodhub' and missing_fh and len(in_flight) < 2000:
                sorted_fh = sorted(
                    missing_fh.items(), key=lambda kv: kv[1][0], reverse=True
                )[:1250]
                batch_fh = [(gid, sev) for gid, (_, sev) in sorted_fh]
                for gid, _ in batch_fh:
                    in_flight.add(gid)
                _BG_EXECUTOR.submit(_fetch_floodhub_batch_worker, batch_fh)

            elif model == 'geoglows' and missing_gg and len(in_flight) < 3000:
                sorted_gg = [
                    rid
                    for rid, _ in sorted(
                        missing_gg.items(), key=lambda kv: kv[1], reverse=True
                    )[:2200]
                ]
                for rid in sorted_gg:
                    in_flight.add(str(rid))
                _BG_EXECUTOR.submit(_fetch_geoglows_batch_worker, sorted_gg)

            elif model == 'glofas' and need_ows_grid and not _GLOFAS_OWS_IN_FLIGHT:
                _GLOFAS_OWS_IN_FLIGHT = True
                _BG_EXECUTOR.submit(_fetch_glofas_ows_grid_worker)

    return colored, viewport_pending

