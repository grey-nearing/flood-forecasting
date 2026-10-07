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

"""Google FloodHub REST client and gauge/forecast/inundation normalizer."""

import math
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from maas.config import FLOODHUB_BASE_URL, parse_finite_float

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

FH_LEVEL_ORDER: dict[str, int] = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
FH_LEVEL_LABELS: dict[str, str] = {
    'HIGH': 'High likelihood',
    'MEDIUM': 'Medium likelihood',
    'LOW': 'Low likelihood',
}
FH_LEVEL_COLORS: dict[str, str] = {
    'HIGH': '#0e7490',
    'MEDIUM': '#06b6d4',
    'LOW': '#67e8f9',
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


def round_geojson_coords(obj: Any, ndigits: int = 5) -> Any:
    """Recursively round nested GeoJSON coordinate sequences."""
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), ndigits) for v in obj]
        return [round_geojson_coords(o, ndigits) for o in obj]
    return obj


def kml_rings(elem: Any) -> list[list[tuple[float, float]]]:
    """Extract all coordinate rings under a KML element (namespace-agnostic)."""
    rings: list[list[tuple[float, float]]] = []
    for node in elem.iter():
        if node.tag.split('}')[-1] != 'coordinates' or not node.text:
            continue
        ring: list[tuple[float, float]] = []
        for tok in node.text.split():
            parts = tok.split(',')
            if len(parts) >= 2:
                lon = parse_finite_float(parts[0])
                lat = parse_finite_float(parts[1])
                if lon is not None and lat is not None:
                    ring.append((lon, lat))
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def kml_to_geometry(
    kml_text: str,
    tolerance_deg: float = 0.0005,
) -> Any | None:
    """Parse FloodHub KML polygons into a simplified shapely `(Multi)Polygon`."""
    if not kml_text or not kml_text.strip():
        return None
    root = ET.fromstring(kml_text)  # noqa: S314
    polys = []
    for el in root.iter():
        if el.tag.split('}')[-1] != 'Polygon':
            continue
        outer: list[tuple[float, float]] = []
        inners: list[list[tuple[float, float]]] = []
        for child in el:
            tag = child.tag.split('}')[-1]
            if tag == 'outerBoundaryIs':
                rings = kml_rings(child)
                outer = rings[0] if rings else []
            elif tag == 'innerBoundaryIs':
                inners.extend(kml_rings(child))
        if len(outer) < 4:
            continue
        poly = Polygon(outer, inners)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polys.append(poly)
    if not polys:
        return None
    geom = unary_union(polys)
    geom = geom.simplify(tolerance_deg, preserve_topology=True)
    if geom.geom_type == 'GeometryCollection':
        parts = [
            g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')
        ]
        geom = unary_union(parts) if parts else None
    if (
        geom is None
        or geom.is_empty
        or geom.geom_type not in ('Polygon', 'MultiPolygon')
    ):
        return None
    return geom


def geom_area_km2(geom: Any) -> float:
    """Approximate area (km²) of a lon/lat geometry via local equirectangular scaling."""
    if geom is None or geom.is_empty:
        return 0.0
    lat0 = float(geom.centroid.y)
    return round(
        float(geom.area)
        * 111.32
        * 111.32
        * max(math.cos(math.radians(lat0)), 0.01),
        2,
    )


class FloodHubClient:
    """Client for Google FloodHub FloodForecasting v1 REST endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str = FLOODHUB_BASE_URL,
        timeout_s: float = 12.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout_s = timeout_s
        self.session = session if session is not None else requests.Session()

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
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
        for st in (payload or {}).get('floodStatuses') or ():
            if isinstance(st, Mapping) and st.get('gaugeId') == gauge_id:
                return dict(st)
        return None

    def fetch_polygon_geometry(
        self,
        polygon_id: str,
    ) -> dict[str, Any] | None:
        """Fetch and simplify a FloodHub serialized KML inundation polygon."""
        if not polygon_id or not polygon_id.strip():
            raise ValueError('`polygon_id` must be a non-empty string.')
        encoded_id = urllib.parse.quote(str(polygon_id), safe='')
        url = f'{self.base_url}/serializedPolygons/{encoded_id}'
        resp = self.session.get(
            url,
            params={'key': self.api_key},
            timeout=self.timeout_s,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
        kml = payload.get('kml') if isinstance(payload, Mapping) else None
        if not isinstance(kml, str) or not kml.strip():
            return None
        geom = kml_to_geometry(kml)
        if geom is None:
            return None
        return {
            'geometry': {
                'type': geom.geom_type,
                'coordinates': round_geojson_coords(
                    mapping(geom)['coordinates']
                ),
            },
            'area_km2': geom_area_km2(geom),
        }


__all__ = [
    'FH_LEVEL_COLORS',
    'FH_LEVEL_LABELS',
    'FH_LEVEL_ORDER',
    'FH_SEVERITY_LABELS',
    'FH_SEVERITY_MAP',
    'FH_SEVERITY_RANK',
    'FH_SEVERITY_TO_RISK',
    'FH_TREND_MAP',
    'FLOODHUB_GAUGE_SEARCH_RADIUS_KM',
    'FloodHubClient',
    'MultiPolygon',
    'derive_floodhub_severity_from_forecast',
    'geom_area_km2',
    'kml_rings',
    'kml_to_geometry',
    'normalize_floodhub_severity',
    'normalize_floodhub_trend',
    'parse_floodhub_forecast_response',
    'parse_floodhub_gauges_response',
    'round_geojson_coords',
]
