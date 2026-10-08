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

"""Google FloodHub REST client and gauge/forecast normalizer."""

import concurrent.futures
import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests

from maas.config import FLOODHUB_BASE_URL, parse_finite_float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two `(lat, lon)` points."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))

FH_SEVERITY_MAP: dict[str, str] = {
    'NO_FLOODING': 'NO_FLOODING',
    'ABOVE_NORMAL': 'WARNING',
    'SEVERE': 'DANGER',
    'EXTREME': 'EXTREME_DANGER',
    'WARNING': 'WARNING',
    'DANGER': 'DANGER',
    'EXTREME_DANGER': 'EXTREME_DANGER',
}

FH_TREND_MAP: dict[str, str] = {
    'RISE': 'RISING',
    'RISING': 'RISING',
    'FALL': 'FALLING',
    'FALLING': 'FALLING',
    'NO_CHANGE': 'STEADY',
    'STEADY': 'STEADY',
}

FH_SEVERITY_RANK: dict[str, int] = {
    'UNKNOWN': 0,
    'NO_FLOODING': 0,
    'WARNING': 1,
    'DANGER': 2,
    'EXTREME_DANGER': 3,
}

FH_SEVERITY_LABELS: dict[str, str] = {
    'NO_FLOODING': 'Normal (FloodHub official)',
    'WARNING': 'Warning level (FloodHub official)',
    'DANGER': 'Danger level (FloodHub official)',
    'EXTREME_DANGER': 'Extreme danger (FloodHub official)',
}

FH_SEVERITY_TO_RISK: dict[str, str] = {
    'NO_FLOODING': 'NORMAL',
    'WARNING': 'WARNING',
    'DANGER': 'SEVERE',
    'EXTREME_DANGER': 'EXTREME',
}

FLOODHUB_GAUGE_SEARCH_RADIUS_KM = 30.0


def normalize_floodhub_severity(raw: Any) -> str:
    """Normalize raw FloodHub severity strings to canonical levels."""
    return FH_SEVERITY_MAP.get(str(raw or '').strip().upper(), 'UNKNOWN')


def normalize_floodhub_trend(raw: Any) -> str:
    """Normalize raw FloodHub forecast trend strings to canonical levels."""
    return FH_TREND_MAP.get(str(raw or '').strip().upper(), 'UNKNOWN')


def parse_floodhub_gauges_response(
    payload: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Parse FloodHub `searchLatestFloodStatusByArea` JSON payload into gauge records."""
    gauges: list[dict[str, Any]] = []
    if not isinstance(payload, Mapping) or 'floodStatuses' not in payload:
        return gauges
    for status in payload.get('floodStatuses') or ():
        if not isinstance(status, Mapping):
            continue
        loc = status.get('gaugeLocation') or {}
        lat = parse_finite_float(loc.get('latitude'))
        lon = parse_finite_float(loc.get('longitude'))
        gauges.append(
            {
                'gauge_id': str(status.get('gaugeId') or ''),
                'lat': lat if lat is not None else 0.0,
                'lon': lon if lon is not None else 0.0,
                'severity': str(status.get('severity') or 'UNKNOWN'),
                'forecast_trend': str(
                    status.get('forecastTrend') or 'NO_CHANGE'
                ),
                'issued_time': str(status.get('issuedTime') or ''),
                'quality_verified': bool(status.get('qualityVerified', False)),
                'source': str(status.get('source') or 'HYBAS'),
            }
        )
    return gauges


def parse_floodhub_forecast_response(
    forecast_payload: Mapping[str, Any] | None,
    metadata_payload: Mapping[str, Any] | None,
    gauge_id: str,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Parse FloodHub `queryGaugeForecasts` and `gaugeModels:batchGet` JSON payloads.

    Scans issues from newest to oldest, skipping `"NaN"` strings via
    `parse_finite_float`, and selects the most recent issue with finite values
    covering the last 24 hours. Does not fabricate synthetic data when no valid
    forecast exists.
    """
    ref_now = now_utc if now_utc is not None else datetime.now(UTC)
    forecast_series: list[dict[str, Any]] = []
    latest_issue_time = ''
    fallback_reason: str | None = None
    nan_issues_skipped = 0

    forecasts_map = (
        forecast_payload.get('forecasts')
        if isinstance(forecast_payload, Mapping)
        else None
    )
    if isinstance(forecasts_map, Mapping) and gauge_id in forecasts_map:
        g_data = forecasts_map[gauge_id]
        fcasts = (
            g_data.get('forecasts', []) if isinstance(g_data, Mapping) else []
        )
        fresh_cutoff = (ref_now - timedelta(hours=24)).strftime(
            '%Y-%m-%dT%H:%M:%SZ'
        )
        for issue in reversed(fcasts):
            if not isinstance(issue, Mapping):
                continue
            points: list[dict[str, Any]] = []
            for rng in issue.get('forecastRanges') or ():
                if not isinstance(rng, Mapping):
                    continue
                start_t = str(rng.get('forecastStartTime') or '')
                val = parse_finite_float(rng.get('value'))
                if start_t and val is not None:
                    points.append({'time': start_t, 'discharge': round(val, 2)})
            if not points:
                nan_issues_skipped += 1
                continue
            if max(p['time'] for p in points) >= fresh_cutoff:
                forecast_series = points
                latest_issue_time = str(issue.get('issuedTime') or '')
            else:
                issued_str = str(issue.get('issuedTime') or '')
                fallback_reason = (
                    f'latest finite FloodHub forecast ({issued_str}) is stale'
                )
            break
        if not forecast_series and not fallback_reason:
            fallback_reason = (
                'FloodHub forecast values are all NaN'
                if fcasts
                else 'no FloodHub forecasts issued'
            )
    else:
        fallback_reason = 'FloodHub forecast API unavailable'

    thresholds: dict[str, float | None] = {
        'warning_2yr': None,
        'danger_5yr': None,
        'extreme_20yr': None,
    }
    unit = 'CUBIC_METERS_PER_SECOND'
    models_list = (
        metadata_payload.get('gaugeModels')
        if isinstance(metadata_payload, Mapping)
        else None
    )
    if isinstance(models_list, Sequence) and len(models_list) > 0:
        m_info = models_list[0]
        if isinstance(m_info, Mapping):
            unit = str(
                m_info.get('gaugeValueUnit') or 'CUBIC_METERS_PER_SECOND'
            )
            th = m_info.get('thresholds') or {}
            if isinstance(th, Mapping):
                for out_key, api_key in (
                    ('warning_2yr', 'warningLevel'),
                    ('danger_5yr', 'dangerLevel'),
                    ('extreme_20yr', 'extremeDangerLevel'),
                ):
                    val = parse_finite_float(th.get(api_key))
                    thresholds[out_key] = (
                        round(val, 2) if val is not None else None
                    )

    has_live = len(forecast_series) > 0
    return {
        'model': 'google_floodhub',
        'available': has_live,
        'status': 'live' if has_live else 'unavailable',
        'gauge_id': gauge_id,
        'issued_time': latest_issue_time,
        'unit': unit,
        'thresholds': thresholds,
        'data': forecast_series,
        'fallback_reason': None if has_live else fallback_reason,
        'nan_issues_skipped': nan_issues_skipped,
    }


def _future_values(
    records: Sequence[Mapping[str, Any]],
    key: str,
    now_utc: datetime | None = None,
) -> list[float]:
    """Values of `key` at or after today's UTC date (all values if none are)."""
    ref_now = now_utc if now_utc is not None else datetime.now(UTC)
    today = ref_now.strftime('%Y-%m-%d')
    pairs = [
        (str(r.get('time', ''))[:10], parse_finite_float(r.get(key)))
        for r in records or ()
    ]
    future = [v for t, v in pairs if v is not None and t >= today]
    return future or [v for _, v in pairs if v is not None]


def derive_floodhub_severity_from_forecast(
    forecast: Mapping[str, Any],
    now_utc: datetime | None = None,
) -> tuple[str, str]:
    """Derive `(severity, trend)` from a FloodHub forecast versus its thresholds."""
    vals = _future_values(
        forecast.get('data') or (), 'discharge', now_utc=now_utc
    )
    th = forecast.get('thresholds') or {}
    warn = parse_finite_float(th.get('warning_2yr'))
    danger = parse_finite_float(th.get('danger_5yr'))
    extreme = parse_finite_float(th.get('extreme_20yr'))
    if not vals or warn is None:
        return 'UNKNOWN', 'UNKNOWN'
    peak = max(vals)
    if extreme is not None and peak >= extreme:
        severity = 'EXTREME_DANGER'
    elif danger is not None and peak >= danger:
        severity = 'DANGER'
    elif peak >= warn:
        severity = 'WARNING'
    else:
        severity = 'NO_FLOODING'
    third = max(len(vals) // 3, 1)
    head = sum(vals[:third]) / third
    tail = sum(vals[-third:]) / third
    if tail > head * 1.05:
        trend = 'RISING'
    elif tail < head * 0.95:
        trend = 'FALLING'
    else:
        trend = 'STEADY'
    return severity, trend


_CATALOG_MEM_CACHE: dict[str, dict[str, Any]] = {}
_RANK_TO_SEVERITY: tuple[str, ...] = (
    'NO_FLOODING',
    'WARNING',
    'DANGER',
    'EXTREME_DANGER',
)


def _resolve_catalog_path(cache_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if cache_dir is not None:
        candidates.append(cache_dir / 'floodhub_gauges_v1.npz')
        candidates.append(cache_dir.parent / 'floodhub_gauges_v1.npz')
    for p in candidates:
        if p.exists():
            return p
    return candidates[0] if candidates else None


def load_or_build_floodhub_catalog(
    cache_dir: Path | None,
    *,
    api_key: str = '',
    base_url: str = FLOODHUB_BASE_URL,
    build_if_missing: bool = False,
) -> dict[str, Any] | None:
    """Load (or optionally download and cache) the global FloodHub gauge catalog."""
    cat_path = _resolve_catalog_path(cache_dir)
    if cat_path is None:
        return None
    key_str = str(cat_path.resolve())
    cached = _CATALOG_MEM_CACHE.get(key_str)
    if cached is not None:
        return cached

    if not cat_path.exists():
        if not build_if_missing or not api_key.strip():
            return None
        _download_and_save_global_catalog(cat_path, api_key=api_key, base_url=base_url)
        if not cat_path.exists():
            return None

    if cat_path.stat().st_size == 0:
        return None
    with np.load(cat_path, allow_pickle=False) as z:
        gids = z['gauge_id']
        lats = z['lat'].astype(np.float32)
        lons = z['lon'].astype(np.float32)
        hybas = z['hybas_id'].astype(np.int64)
        has_fc = z['has_forecast'].astype(np.bool_)
        q_ver = z['quality_verified'].astype(np.bool_)
        sev_rank = z['severity_rank'].astype(np.int8)
        area_km2 = (
            z['upstream_area_km2'].astype(np.float32)
            if 'upstream_area_km2' in z.files
            else np.zeros(len(gids), dtype=np.float32)
        )
        fetched_at = (
            int(z['fetched_at'][0])
            if 'fetched_at' in z.files and len(z['fetched_at']) > 0
            else 0
        )
    active_mask = has_fc
    hybas_pos_idx = np.flatnonzero(hybas > 0)
    hybas_order = np.argsort(hybas[hybas_pos_idx])
    hybas_sorted_idx = hybas_pos_idx[hybas_order]
    hybas_sorted_ids = hybas[hybas_sorted_idx]
    cat = {
        'path': str(cat_path),
        'gauge_id': gids,
        'lat': lats,
        'lon': lons,
        'hybas_id': hybas,
        'has_forecast': has_fc,
        'quality_verified': q_ver,
        'severity_rank': sev_rank,
        'upstream_area_km2': area_km2,
        'active_idx': np.flatnonzero(active_mask),
        'hybas_sorted_idx': hybas_sorted_idx,
        'hybas_sorted_ids': hybas_sorted_ids,
        'fetched_at': fetched_at,
    }
    _CATALOG_MEM_CACHE[key_str] = cat
    return cat


def _download_and_save_global_catalog(
    target_path: Path,
    *,
    api_key: str,
    base_url: str = FLOODHUB_BASE_URL,
) -> None:
    """Download all global FloodHub status tiles in parallel and save as compressed `.npz`."""
    tiles: list[tuple[float, float, float, float]] = []
    for lon0 in range(-180, 180, 60):
        lon1 = lon0 + 60 - 0.0001
        for lat0, lat1 in ((-60.0, 0.0), (0.0, 75.0)):
            tiles.append((lat0, lat1, float(lon0), float(lon1)))

    url = f"{base_url.rstrip('/')}/floodStatus:searchLatestFloodStatusByArea?key={api_key}"

    def _fetch_tile(tile: tuple[float, float, float, float]) -> list[dict[str, Any]]:
        lat0, lat1, lon0, lon1 = tile
        sess = requests.Session()
        out: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            body: dict[str, Any] = {
                'loop': {
                    'vertices': [
                        {'latitude': lat0, 'longitude': lon0},
                        {'latitude': lat1, 'longitude': lon0},
                        {'latitude': lat1, 'longitude': lon1},
                        {'latitude': lat0, 'longitude': lon1},
                    ]
                },
                'pageSize': 50000,
                'includeNonQualityVerified': True,
            }
            if page_token:
                body['pageToken'] = page_token
            resp = sess.post(url, json=body, timeout=25.0)
            resp.raise_for_status()
            payload = resp.json()
            out.extend(payload.get('floodStatuses') or [])
            page_token = payload.get('nextPageToken')
            if not page_token:
                break
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(_fetch_tile, tiles))

    by_id: dict[str, Mapping[str, Any]] = {}
    for sub in results:
        for st in sub:
            gid = st.get('gaugeId')
            loc = st.get('gaugeLocation') or {}
            if gid and loc.get('latitude') is not None and loc.get('longitude') is not None:
                by_id[str(gid)] = st

    gids = sorted(by_id.keys())
    n = len(gids)
    lats = np.empty(n, dtype=np.float32)
    lons = np.empty(n, dtype=np.float32)
    hybas = np.zeros(n, dtype=np.int64)
    has_fc = np.zeros(n, dtype=np.bool_)
    q_ver = np.zeros(n, dtype=np.bool_)
    sev_rank = np.zeros(n, dtype=np.int8)

    for i, gid in enumerate(gids):
        st = by_id[gid]
        loc = st['gaugeLocation']
        lats[i] = float(loc['latitude'])
        lons[i] = float(loc['longitude'])
        if gid.startswith('hybas_') and gid[6:].isdigit():
            hybas[i] = int(gid[6:])
        has_fc[i] = ('forecastTimeRange' in st) or ('forecastTrend' in st)
        q_ver[i] = bool(st.get('qualityVerified', False))
        sev_norm = normalize_floodhub_severity(st.get('severity'))
        sev_rank[i] = FH_SEVERITY_RANK.get(sev_norm, 0)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix('.tmp.npz')
    np.savez_compressed(
        tmp_path,
        gauge_id=np.array(gids, dtype='U32'),
        lat=lats,
        lon=lons,
        hybas_id=hybas,
        has_forecast=has_fc,
        quality_verified=q_ver,
        severity_rank=sev_rank,
        fetched_at=np.array([int(time.time())], dtype=np.int64),
    )
    tmp_path.replace(target_path)


class FloodHubClient:
    """Client for Google FloodHub FloodForecasting v1 REST endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str = FLOODHUB_BASE_URL,
        timeout_s: float = 12.0,
        session: requests.Session | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout_s = timeout_s
        self.session = session if session is not None else requests.Session()
        self.cache_dir = cache_dir

    def search_gauges_bbox(  # noqa: PLR0913, PLR0917
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        page_size: int = 100,
        include_non_verified: bool = True,  # noqa: FBT001, FBT002
    ) -> list[dict[str, Any]]:
        """Query FloodHub gauges and latest flood status inside a bounding box."""
        url = (
            f'{self.base_url}/floodStatus:searchLatestFloodStatusByArea'
            f'?key={self.api_key}'
        )
        payload = {
            'loop': {
                'vertices': [
                    {'latitude': min_lat, 'longitude': min_lon},
                    {'latitude': max_lat, 'longitude': min_lon},
                    {'latitude': max_lat, 'longitude': max_lon},
                    {'latitude': min_lat, 'longitude': max_lon},
                ]
            },
            'pageSize': page_size,
            'includeNonQualityVerified': include_non_verified,
        }
        resp = self.session.post(url, json=payload, timeout=self.timeout_s)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return parse_floodhub_gauges_response(resp.json())

    def fetch_forecast(
        self,
        gauge_id: str,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """Fetch 7-day forecast and return-period thresholds for a FloodHub gauge."""
        if not gauge_id or not gauge_id.strip():
            raise ValueError('`gauge_id` must be a non-empty string.')
        ref_now = now_utc if now_utc is not None else datetime.now(UTC)
        start_str = (ref_now - timedelta(days=7)).strftime('%Y-%m-%d')
        end_str = (ref_now + timedelta(days=1)).strftime('%Y-%m-%d')

        f_url = f'{self.base_url}/gauges:queryGaugeForecasts'
        f_params = {
            'key': self.api_key,
            'gaugeIds': gauge_id,
            'issuedTimeStart': start_str,
            'issuedTimeEnd': end_str,
        }
        f_resp = self.session.get(
            f_url, params=f_params, timeout=self.timeout_s
        )
        f_payload: Mapping[str, Any] | None = None
        if f_resp.status_code != 404:
            f_resp.raise_for_status()
            f_payload = f_resp.json()

        m_url = (
            f'{self.base_url}/gaugeModels:batchGet'
            f'?key={self.api_key}&names=gaugeModels/{gauge_id}'
        )
        m_resp = self.session.get(m_url, timeout=self.timeout_s)
        m_payload: Mapping[str, Any] | None = None
        if m_resp.status_code != 404:
            m_resp.raise_for_status()
            m_payload = m_resp.json()

        return parse_floodhub_forecast_response(
            f_payload, m_payload, gauge_id, now_utc=ref_now
        )

    def fetch_flood_status(self, gauge_id: str) -> dict[str, Any] | None:
        """Fetch latest FloodHub flood status for `gauge_id`."""
        if not gauge_id or not gauge_id.strip():
            raise ValueError('`gauge_id` must be a non-empty string.')
        url = f'{self.base_url}/floodStatus:queryLatestFloodStatusByGaugeIds'
        resp = self.session.get(
            url,
            params={'key': self.api_key, 'gaugeIds': gauge_id},
            timeout=self.timeout_s,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        for st in (payload or {}).get('floodStatuses') or ():
            if isinstance(st, Mapping) and st.get('gaugeId') == gauge_id:
                return dict(st)
        return None

    def lookup_cached_gauge(self, gauge_id: str) -> dict[str, Any] | None:
        r"""Look up a single FloodHub gauge by `gauge_id` in the local catalog in $O(\log N)$."""
        if not gauge_id:
            return None
        cat = load_or_build_floodhub_catalog(
            self.cache_dir, api_key=self.api_key, base_url=self.base_url, build_if_missing=False
        )
        if cat is None:
            return None
        gids = cat['gauge_id']
        pos = int(np.searchsorted(gids, gauge_id))
        if pos >= len(gids) or str(gids[pos]) != gauge_id:
            return None
        s_rank = int(cat['severity_rank'][pos])
        sev = _RANK_TO_SEVERITY[min(max(s_rank, 0), 3)]
        area = float(cat['upstream_area_km2'][pos])
        return {
            'gauge_id': gauge_id,
            'lat': round(float(cat['lat'][pos]), 5),
            'lon': round(float(cat['lon'][pos]), 5),
            'source': 'HYBAS' if gauge_id.startswith('hybas_') else gauge_id.split('_')[0],
            'quality_verified': bool(cat['quality_verified'][pos]),
            'severity': sev,
            'severity_rank': s_rank,
            'has_forecast': bool(cat['has_forecast'][pos]),
            'upstream_area_km2': round(area, 1) if area > 0 else None,
        }

    def query_cached_gauges_bbox(  # noqa: PLR0913
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        *,
        only_with_forecast: bool = True,
        min_area_km2: float = 0.0,
        limit: int = 5000,
    ) -> list[dict[str, Any]] | None:
        """Return pre-cached FloodHub gauges inside a bounding box if the local catalog exists."""
        cat = load_or_build_floodhub_catalog(
            self.cache_dir, api_key=self.api_key, base_url=self.base_url, build_if_missing=False
        )
        if cat is None:
            return None
        lats = cat['lat']
        lons = cat['lon']
        mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
        if only_with_forecast:
            mask &= cat['has_forecast']
        if min_area_km2 > 0:
            mask &= (cat['upstream_area_km2'] >= min_area_km2) | (cat['hybas_id'] == 0)
        idxs = np.flatnonzero(mask)
        if len(idxs) > limit:
            areas = cat['upstream_area_km2'][idxs]
            top = np.argsort(-areas)[:limit]
            idxs = idxs[top]
        gids = cat['gauge_id']
        sevs = cat['severity_rank']
        qvers = cat['quality_verified']
        areas = cat['upstream_area_km2']
        has_fcs = cat['has_forecast']
        out: list[dict[str, Any]] = []
        for idx in idxs:
            s_rank = int(sevs[idx])
            sev = _RANK_TO_SEVERITY[min(max(s_rank, 0), 3)]
            gid = str(gids[idx])
            out.append(
                {
                    'gauge_id': gid,
                    'lat': round(float(lats[idx]), 5),
                    'lon': round(float(lons[idx]), 5),
                    'source': 'HYBAS' if gid.startswith('hybas_') else gid.split('_')[0],
                    'quality_verified': bool(qvers[idx]),
                    'severity': sev,
                    'severity_rank': s_rank,
                    'has_forecast': bool(has_fcs[idx]),
                    'upstream_area_km2': round(float(areas[idx]), 1) if areas[idx] > 0 else None,
                }
            )
        return out

    def find_nearest_gauge(
        self,
        lat: float,
        lon: float,
        radius_km: float = FLOODHUB_GAUGE_SEARCH_RADIUS_KM,
        *,
        target_area_km2: float | None = None,
    ) -> tuple[str | None, dict[str, Any] | None, float | None]:
        """Find the closest FloodHub gauge with an active forecast within `radius_km` of `(lat, lon)`."""
        dlat = radius_km / 111.0
        dlon = radius_km / max(111.0 * math.cos(math.radians(lat)), 1.0)

        cat = load_or_build_floodhub_catalog(
            self.cache_dir, api_key=self.api_key, base_url=self.base_url, build_if_missing=False
        )
        if cat is not None:
            lats = cat['lat']
            lons = cat['lon']
            mask = (
                cat['has_forecast']
                & (lats >= lat - dlat)
                & (lats <= lat + dlat)
                & (lons >= lon - dlon)
                & (lons <= lon + dlon)
            )
            cand_idx = np.flatnonzero(mask)
            if len(cand_idx) > 0:
                best_idx: int | None = None
                best_score = float('inf')
                best_dist = float('inf')
                t_area = (
                    float(target_area_km2)
                    if target_area_km2 is not None and float(target_area_km2) > 0
                    else None
                )
                log_target = math.log10(max(t_area, 10.0)) if t_area else 0.0
                areas = cat['upstream_area_km2']
                for idx in cand_idx:
                    d_km = haversine_km(lat, lon, float(lats[idx]), float(lons[idx]))
                    if d_km > radius_km:
                        continue
                    g_area = float(areas[idx])
                    if t_area is not None and g_area > 0:
                        log_err = abs(math.log10(max(g_area, 10.0)) - log_target)
                        score = log_err + 0.04 * d_km
                    else:
                        score = d_km
                    if score < best_score:
                        best_score = score
                        best_dist = d_km
                        best_idx = int(idx)
                if best_idx is not None:
                    gid = str(cat['gauge_id'][best_idx])
                    s_rank = int(cat['severity_rank'][best_idx])
                    sev = _RANK_TO_SEVERITY[min(max(s_rank, 0), 3)]
                    return (
                        gid,
                        {
                            'lat': round(float(lats[best_idx]), 5),
                            'lon': round(float(lons[best_idx]), 5),
                            'source': 'HYBAS' if gid.startswith('hybas_') else gid.split('_')[0],
                            'quality_verified': bool(cat['quality_verified'][best_idx]),
                            'severity': sev,
                            'upstream_area_km2': round(float(areas[best_idx]), 1)
                            if areas[best_idx] > 0
                            else None,
                        },
                        round(best_dist, 2),
                    )
            return None, None, None

        if not self.api_key.strip():
            return None, None, None
        gauges = self.search_gauges_bbox(
            min_lat=lat - dlat,
            min_lon=lon - dlon,
            max_lat=lat + dlat,
            max_lon=lon + dlon,
            page_size=100,
            include_non_verified=True,
        )
        best: tuple[float, dict[str, Any]] | None = None
        for g in gauges:
            gid = g.get('gauge_id')
            glat = parse_finite_float(g.get('lat'))
            glon = parse_finite_float(g.get('lon'))
            if not gid or glat is None or glon is None:
                continue
            d = haversine_km(lat, lon, glat, glon)
            if d <= radius_km and (best is None or d < best[0]):
                best = (d, g)
        if best is None:
            return None, None, None
        dist_km, g = best
        return (
            str(g['gauge_id']),
            {
                'lat': g.get('lat'),
                'lon': g.get('lon'),
                'source': g.get('source'),
                'quality_verified': g.get('quality_verified'),
                'severity': g.get('severity'),
                'forecast_trend': g.get('forecast_trend'),
                'issued_time': g.get('issued_time'),
            },
            round(dist_km, 2),
        )

    def enrich_forecast_status(  # noqa: PLR0913
        self,
        gauge_id: str,
        forecast: Mapping[str, Any],
        *,
        lat: float | None = None,
        lon: float | None = None,
        gauge_meta: dict[str, Any] | None = None,
        dist_km: float | None = None,
    ) -> dict[str, Any]:
        """Enrich a FloodHub forecast payload with severity, trend, and gauge location metadata."""
        meta = gauge_meta or self.lookup_cached_gauge(gauge_id) or {}
        if (
            dist_km is None
            and lat is not None
            and lon is not None
            and meta.get('lat') is not None
            and meta.get('lon') is not None
        ):
            dist_km = round(
                haversine_km(
                    float(lat),
                    float(lon),
                    float(meta['lat']),
                    float(meta['lon']),
                ),
                2,
            )

        severity, trend = derive_floodhub_severity_from_forecast(forecast)
        if severity != 'UNKNOWN':
            sev_source = 'derived_from_forecast_and_thresholds'
        elif meta.get('severity') and normalize_floodhub_severity(meta.get('severity')) != 'UNKNOWN':
            severity = normalize_floodhub_severity(meta.get('severity'))
            trend = normalize_floodhub_trend(meta.get('forecast_trend'))
            sev_source = 'floodhub_catalog'
        else:
            status_obj = self.fetch_flood_status(gauge_id)
            severity = normalize_floodhub_severity(
                (status_obj or {}).get('severity')
                if isinstance(status_obj, Mapping)
                else None
            )
            trend = normalize_floodhub_trend(
                (status_obj or {}).get('forecastTrend')
                if isinstance(status_obj, Mapping)
                else None
            )
            sev_source = 'floodhub_flood_status' if severity != 'UNKNOWN' else 'unavailable'
            loc = (
                (status_obj or {}).get('gaugeLocation')
                if isinstance(status_obj, Mapping)
                else None
            ) or {}
            if not meta and 'latitude' in loc and 'longitude' in loc:
                meta = {
                    'lat': loc.get('latitude'),
                    'lon': loc.get('longitude'),
                    'quality_verified': (status_obj or {}).get('qualityVerified'),
                }

        return {
            **forecast,
            'severity': severity,
            'severity_rank': FH_SEVERITY_RANK.get(severity, 0),
            'trend': trend,
            'severity_source': sev_source,
            'gauge_location': (
                {'lat': meta.get('lat'), 'lon': meta.get('lon')}
                if meta.get('lat') is not None and meta.get('lon') is not None
                else None
            ),
            'distance_km': dist_km,
            'quality_verified': meta.get('quality_verified'),
        }


__all__ = [
    'FH_SEVERITY_LABELS',
    'FH_SEVERITY_MAP',
    'FH_SEVERITY_RANK',
    'FH_SEVERITY_TO_RISK',
    'FH_TREND_MAP',
    'FLOODHUB_GAUGE_SEARCH_RADIUS_KM',
    'FloodHubClient',
    'derive_floodhub_severity_from_forecast',
    'load_or_build_floodhub_catalog',
    'normalize_floodhub_severity',
    'normalize_floodhub_trend',
    'parse_floodhub_forecast_response',
    'parse_floodhub_gauges_response',
]
