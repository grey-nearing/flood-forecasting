# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HTTP server for OpenHydroNet using standard library `http.server`."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import logging
import math
import mimetypes
import os
import re
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from frontend.config import (
    HYDRO_DATASETS,
    STATIC_DIR,
    resolve_hydro_dataset_id,
)
from frontend.delineator import HydroDelineator
from frontend.dem_tile_server import DemTileServer
from frontend.finetune_service import (
    clear_finetune_data,
    get_finetune_workspace_status,
    list_finetune_models,
    reset_finetune_session,
    run_finetune_epochs,
    upload_finetune_csv,
)
from frontend.forecast_zarr import ForecastZarrExtractor
from frontend.historical_zarr import HistoricalZarrExtractor
from frontend.jobs import get_job_manager
from frontend.maas_engine import (
    fetch_floodhub_gauges_bbox,
    get_maas_fetcher,
    get_maas_watershed_polygon,
    get_unified_maas_forecast,
)
from frontend.maas_networks import get_model_network
from frontend.profile_manager import (
    find_zarr_dirs,
    get_profile_manager,
    is_valid_watershed_name,
)
from frontend.realtime_forecast_service import (
    clear_realtime_streamflow_csv,
    fetch_realtime_for_basin,
    inspect_realtime_stores_for_basin,
    is_valid_issue_date_str,
    run_hydrological_model_for_basin,
    upload_realtime_streamflow_csv,
    validate_streamflow_csv_request,
)
from frontend.river_indexer import HydroRiverNetwork
from frontend.static_attributes import (
    ATTRIBUTE_REGISTRY,
    build_attribute_card_payload,
    build_map_layer_geojson,
    build_schema_payload,
    card_payload_from_cache_entry,
    get_attributes_extractor,
    sanitize_for_json,
)
from frontend.weather_engine import (
    SUPPORTED_MODELS,
    SUPPORTED_VARIABLES,
    generate_raster_tile,
    generate_weather_frame,
    get_catchment_weather_summary,
    get_frame_index,
    get_sync_status,
    get_weather_models_info,
    get_weather_probe,
    get_wind_vectors,
)
from frontend.weather_engine import (
    reload_if_changed as reload_weather_if_changed,
)
from frontend.weather_sources import list_weather_sources
from frontend.weather_sync import run_sync_subprocess
from maas.glofas import DEFAULT_GLOFAS_REFERENCE, GLOFAS_REFERENCE_FIELDS
from utils.file_paths import (
    DEFAULT_LOCAL_SERVER_URL,
    HISTORICAL_MASTER_ZARR_NAME,
    RAINVIEWER_MAPS_URL,
)

if TYPE_CHECKING:
    from pathlib import Path

    from frontend.profile_manager import ProfileManager
    from multimet.static_extractor import StaticAttributesExtractor

logger = logging.getLogger(__name__)

_INT_RE = re.compile(r'^[+-]?\d+$')
_FLOAT_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
_GZIP_MIN_BYTES = 1024
_DEFAULT_PORT = 8000
_DEFAULT_HOST = '127.0.0.1'
_GUEST_USERNAME = 'guest'
_LOCAL_CORS_PREFIXES = (
    'http://localhost',
    'http://127.0.0.1',
    'https://localhost',
    'https://127.0.0.1',
)

_river_indexers: dict[str, HydroRiverNetwork] = {}
_delineators: dict[str, HydroDelineator] = {}
_dem_tile_server: DemTileServer | None = None


def _get_river_indexer(dataset: str) -> HydroRiverNetwork | None:
    """Return a lazily initialized `HydroRiverNetwork` for `dataset`."""
    resolved = resolve_hydro_dataset_id(dataset)
    if resolved not in HYDRO_DATASETS:
        return None
    if resolved not in _river_indexers:
        _river_indexers[resolved] = HydroRiverNetwork(resolved)
    return _river_indexers[resolved]


def _get_delineator(dataset: str) -> HydroDelineator | None:
    """Return a lazily initialized `HydroDelineator` for `dataset`."""
    resolved = resolve_hydro_dataset_id(dataset)
    if resolved not in HYDRO_DATASETS:
        return None
    if resolved not in _delineators:
        _delineators[resolved] = HydroDelineator(resolved)
    return _delineators[resolved]


def get_dem_tile_server() -> DemTileServer:
    """Return the singleton `DemTileServer` instance."""
    global _dem_tile_server  # noqa: PLW0603
    if _dem_tile_server is None:
        _dem_tile_server = DemTileServer()
    return _dem_tile_server


def _parse_int(value: object, default: int | None = None) -> int | None:
    """Parse an integer from a scalar or string without raising."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if _INT_RE.fullmatch(s):
        return int(s)
    if _FLOAT_RE.fullmatch(s):
        fval = float(s)
        if math.isfinite(fval):
            return int(fval)
    return default


def _parse_float(value: object, default: float | None = None) -> float | None:
    """Parse a finite float from a scalar or string without raising."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        fval = float(value)
        return fval if math.isfinite(fval) else default
    s = str(value).strip()
    if _FLOAT_RE.fullmatch(s):
        fval = float(s)
        if math.isfinite(fval):
            return fval
    return default


def _parse_bbox(
    bbox_param: str | None,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Parse a 4-tuple bounding box string `'a,b,c,d'` or return `default`."""
    if not bbox_param:
        return default
    parts = [p.strip() for p in bbox_param.split(',')]
    if len(parts) != 4:  # noqa: PLR2004
        return default
    vals = [_parse_float(p) for p in parts]
    if any(v is None for v in vals):
        return default
    return (
        float(vals[0]),  # type: ignore[arg-type]
        float(vals[1]),  # type: ignore[arg-type]
        float(vals[2]),  # type: ignore[arg-type]
        float(vals[3]),  # type: ignore[arg-type]
    )


def _sanitize_username_token(username: str) -> str:
    """Sanitize a raw username string using the profile manager's rules."""
    raw = username.strip().lower()
    return ''.join(c for c in raw if c.isalnum() or c in ('_', '-', '.')).strip(
        '.'
    )


def _inspect_zarr_archive(zarr_path: Path) -> tuple[list[str], int, int]:
    """Read basin IDs, timestep count, and lead-time count from a Zarr store."""
    has_zarr_meta = (
        (zarr_path / '.zmetadata').exists()
        or (zarr_path / '.zgroup').exists()
        or (zarr_path / 'zarr.json').exists()
    )
    if not has_zarr_meta:
        return [], 0, 0
    ds = xr.open_zarr(str(zarr_path), decode_timedelta=False)
    basins = (
        [str(b) for b in ds.basin.values]
        if ('basin' in ds.coords or 'basin' in ds.dims)
        else []
    )
    time_coord = (
        ds.coords.get('date')
        if 'date' in ds.coords
        else (ds.coords.get('time') if 'time' in ds.coords else None)
    )
    n_times = len(time_coord.values) if time_coord is not None else 0
    n_leads = len(ds.lead_time.values) if 'lead_time' in ds.coords else 0
    return basins, n_times, n_leads


def _normalize_uploaded_feature(
    feat: dict[str, Any],
    idx: int,
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    """Normalize a user-uploaded GeoJSON feature into a Polygon feature."""
    geom_dict = feat.get('geometry', feat)
    if not isinstance(geom_dict, dict) or 'type' not in geom_dict:
        return None
    geom = shape(geom_dict)
    if geom.is_empty:
        return None
    if geom.geom_type not in ('Polygon', 'MultiPolygon'):
        if geom.geom_type != 'GeometryCollection':
            return None
        polys = [
            g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')
        ]
        if not polys:
            return None
        geom = unary_union(polys)

    cent = geom.centroid
    props = dict(feat.get('properties') or {})
    cid = (
        props.get('catchment_id')
        or props.get('gauge_id')
        or props.get('station_id')
        or props.get('basin_id')
        or props.get('id')
        or props.get('site_no')
        or props.get('huc8')
        or props.get('HUC8')
        or props.get('huc10')
        or props.get('huc12')
        or f'user_basin_{idx + 1}'
    )
    name = (
        props.get('name')
        or props.get('station_name')
        or props.get('basin_name')
        or props.get('SITE_NAME')
        or f'Watershed {idx + 1} ({cid})'
    )
    lat_scale = 111.0
    lon_scale = 111.0 * float(np.cos(np.radians(cent.y)))
    calc_area = float(geom.area * lat_scale * lon_scale)
    area_km2 = round(
        float(props.get('area_km2') or props.get('area') or calc_area),
        2,
    )
    bbox = [round(float(b), 5) for b in geom.bounds]
    norm_f = {
        'type': 'Feature',
        'geometry': mapping(geom),
        'properties': {
            **props,
            'catchment_id': str(cid),
            'name': str(name),
            'area_km2': area_km2,
            'dataset_name': 'Custom User-Uploaded Watershed',
            'delineation_method': 'User GeoJSON Polygon',
            'upstream_reaches_count': 1,
            'stream_order': props.get('stream_order', 3),
            'outlet': {
                'latitude': round(float(cent.y), 4),
                'longitude': round(float(cent.x), 4),
                'snap_distance_m': 0,
            },
            'bbox': bbox,
        },
    }
    summary = {
        'catchment_id': str(cid),
        'name': str(name),
        'area_km2': area_km2,
        'latitude': round(float(cent.y), 4),
        'longitude': round(float(cent.x), 4),
        'bbox': bbox,
    }
    return norm_f, summary, area_km2


class OpenHydroNetHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenHydroNet JSON API and static assets."""

    def _resolve_allowed_origin(self) -> str:
        """Resolve the allowed CORS origin header for local requests."""
        origin = self.headers.get('Origin', '')
        if origin.startswith(_LOCAL_CORS_PREFIXES):
            return origin
        return DEFAULT_LOCAL_SERVER_URL

    def _send_cors_headers(self, *, include_methods: bool = True) -> None:
        """Write standard CORS headers onto the current HTTP response."""
        self.send_header(
            'Access-Control-Allow-Origin',
            self._resolve_allowed_origin(),
        )
        if include_methods:
            self.send_header(
                'Access-Control-Allow-Methods',
                'GET, POST, OPTIONS',
            )
            self.send_header(
                'Access-Control-Allow-Headers',
                'Content-Type, Authorization',
            )

    def _set_cors_headers(
        self,
        status: int = 200,
        content_type: str = 'application/json',
    ) -> None:
        """Send response status, content type, and CORS headers."""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self._send_cors_headers()
        self.end_headers()

    def _send_tile_png(
        self,
        tile_bytes: bytes,
        status: int = 200,
        max_age: int = 300,
    ) -> None:
        """Send a PNG raster tile or weather frame response with caching."""
        self.send_response(status)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(tile_bytes)))
        self._send_cors_headers(include_methods=False)
        self.send_header('Cache-Control', f'public, max-age={max_age}')
        self.end_headers()
        self.wfile.write(tile_bytes)

    def _send_json(self, data: object, status: int = 200) -> None:
        """Serialize `data` as JSON (with optional gzip) and write response."""
        body = json.dumps(sanitize_for_json(data)).encode('utf-8')
        accept_enc = self.headers.get('Accept-Encoding', '')
        if 'gzip' in accept_enc and len(body) > _GZIP_MIN_BYTES:
            payload = gzip.compress(body)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Encoding', 'gzip')
        else:
            payload = body
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        req_path = urllib.parse.urlparse(self.path).path
        if req_path.startswith(
            (
                '/api/profile/',
                '/api/account/',
                '/api/forecast/',
                '/api/watersheds',
                '/api/archives',
            )
        ):
            self.send_header(
                'Cache-Control',
                'no-store, no-cache, must-revalidate',
            )
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(
        self,
        message: str,
        status: int = 400,
        code: str | None = None,
    ) -> None:
        """Send a structured JSON error payload."""
        payload: dict[str, Any] = {
            'error': message,
            'message': message,
            'status': 'error',
        }
        if code:
            payload['code'] = code
        self._send_json(payload, status=status)

    def _get_request_username(
        self,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Resolve active username from payload, header, query, or session."""
        pm = get_profile_manager()
        if data and isinstance(data, dict) and data.get('username'):
            u = str(data['username']).strip().lower()
            if u:
                return u
        header_user = self.headers.get('X-User-Profile')
        if header_user and header_user.strip():
            return header_user.strip().lower()
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        if 'username' in q and q['username'] and q['username'][0].strip():
            return q['username'][0].strip().lower()
        return pm.active_username

    def _get_server_url(self) -> str:
        """Resolve the server base URL from the `Host` header or default."""
        host = self.headers.get('Host')
        if host and host.strip():
            return f'http://{host.strip()}'
        return DEFAULT_LOCAL_SERVER_URL

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self._set_cors_headers(200, 'text/plain')

    def do_GET(self) -> None:
        """Dispatch HTTP GET requests across JSON API and static routes."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if self._dispatch_simple_get(path, query):
            return
        if self._dispatch_prefix_get(path, query):
            return
        self._send_error('Not Found', status=404)

    def _dispatch_simple_get(
        self,
        path: str,
        query: dict[str, list[str]],
    ) -> bool:
        """Dispatch exact-path GET endpoints; return True if handled."""
        routes = {
            '/api/datasets': self._get_datasets,
            '/api/weather/sources': self._get_weather_sources,
            '/api/attributes/schema': self._get_attributes_schema,
            '/api/attributes/map-layer': self._get_attributes_map_layer,
            '/api/attributes/extracted': self._get_attributes_extracted,
            '/api/hydro/rivers': self._get_hydro_rivers,
            '/api/watersheds': self._get_watersheds,
            '/api/profile/current': self._get_profile_current,
            '/api/profile/list': self._get_profile_list,
            '/api/account/summary': self._get_account_summary,
            '/api/profile/account': self._get_account_summary,
            '/api/weather/forecast/info': self._get_forecast_info,
            '/api/weather/forecast/latest_info': self._get_forecast_info,
            '/api/weather/models': self._get_weather_models,
            '/api/weather/wind-vectors': self._get_weather_wind_vectors,
            '/api/weather/probe': self._get_weather_probe,
            '/api/weather/catchment-summary': (
                self._get_weather_catchment_summary
            ),
            '/api/weather/radar-times': self._get_weather_radar_times,
            '/api/maas/models': self._get_maas_models,
            '/api/maas/gauges': self._get_maas_gauges,
            '/api/maas/network': self._get_maas_network,
            '/api/maas/forecast': self._get_maas_forecast,
            '/api/maas/watershed': self._get_maas_watershed,
            '/api/maas/reaches': self._get_maas_reaches,
            '/api/maas/return-periods': self._get_maas_return_periods,
            '/api/forecast/status': self._get_forecast_status,
            '/api/finetune/models': self._get_finetune_models,
            '/api/finetune/status': self._get_finetune_status,
            '/api/archives': self._get_archives,
            '/api/attributes/csv': self._get_attributes_csv,
            '/api/archives/caravan_hydroatlas_attributes.csv': (
                self._get_attributes_csv
            ),
            '/': self._get_index_html,
            '/index.html': self._get_index_html,
        }
        handler = routes.get(path)
        if handler is None:
            return False
        handler(query)
        return True

    def _dispatch_prefix_get(
        self,
        path: str,
        query: dict[str, list[str]],  # noqa: ARG002
    ) -> bool:
        """Dispatch prefix-matched GET endpoints; return True if handled."""
        if path.startswith('/api/jobs'):
            self._get_jobs(path)
            return True
        if path.startswith('/api/weather/frames/'):
            self._get_weather_frame(path)
            return True
        if path.startswith('/api/weather/tiles/'):
            self._get_weather_tile(path)
            return True
        if path.startswith('/api/tiles/dem/'):
            self._get_dem_tile(path)
            return True
        if path.startswith('/static/'):
            self._get_static_asset(path)
            return True
        return False

    def _get_datasets(self, _query: dict[str, list[str]]) -> None:
        self._send_json({'datasets': list(HYDRO_DATASETS.values())})

    def _get_weather_sources(self, _query: dict[str, list[str]]) -> None:
        self._send_json({'sources': list_weather_sources()})

    def _get_attributes_schema(self, _query: dict[str, list[str]]) -> None:
        self._send_json(build_schema_payload())

    def _get_attributes_map_layer(self, query: dict[str, list[str]]) -> None:
        attribute = query.get('attribute', ['ele_mt_sav'])[0].strip()
        if attribute not in ATTRIBUTE_REGISTRY:
            self._send_error(f"Unknown attribute '{attribute}'", status=400)
            return
        zoom = _parse_int(query.get('zoom', ['7'])[0], default=7) or 7
        level_param = query.get('level', [''])[0]
        level = _parse_int(level_param) if level_param else None
        bbox = _parse_bbox(
            query.get('bbox', [None])[0],  # type: ignore[list-item]
            (-90.0, 35.0, -80.0, 45.0),
        )
        extractor = get_attributes_extractor()
        layer_geojson = build_map_layer_geojson(
            extractor,
            bbox=bbox,
            zoom=zoom,
            attribute=attribute,
            level=level,
        )
        self._send_json(layer_geojson)

    def _get_attributes_extracted(self, _query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        attr_dir = pm.get_attributes_dir(user)
        json_cache_path = attr_dir / 'extracted_attributes.json'
        results_by_id: dict[str, Any] = {}
        if json_cache_path.exists():
            cached = json.loads(json_cache_path.read_text(encoding='utf-8'))
            if isinstance(cached, dict):
                raw_by_id = cached.get('results_by_id', cached)
                if isinstance(raw_by_id, dict):
                    for cid, payload in raw_by_id.items():
                        if isinstance(payload, dict):
                            results_by_id[cid] = card_payload_from_cache_entry(
                                cid,
                                payload,
                            )
        results_list = list(results_by_id.values())
        self._send_json(
            {
                'status': 'success',
                'username': user,
                'count': len(results_list),
                'results': results_list,
                'results_by_id': results_by_id,
            }
        )

    def _get_hydro_rivers(self, query: dict[str, list[str]]) -> None:
        raw_dataset = query.get('dataset', query.get('dem', ['hydroatlas']))[0]
        network = _get_river_indexer(raw_dataset)
        if network is None:
            self._send_error(
                f'Unknown hydrography dataset: {raw_dataset}',
                status=400,
            )
            return
        zoom = _parse_int(query.get('zoom', ['7'])[0], default=7) or 7
        min_order_param = query.get('min_order', [''])[0]
        min_order = _parse_int(min_order_param) if min_order_param else None
        min_lon, min_lat, max_lon, max_lat = _parse_bbox(
            query.get('bbox', [None])[0],  # type: ignore[list-item]
            (-180.0, -90.0, 180.0, 90.0),
        )
        geojson = network.get_rivers_in_bbox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            zoom=zoom,
            min_stream_order=min_order,
            prefer_cache=query.get('lod', [''])[0] == 'fast',
        )
        self._send_json(geojson)

    def _get_watersheds(self, _query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        features = pm.load_watersheds(username=user)
        self._send_json(
            {
                'type': 'FeatureCollection',
                'count': len(features),
                'features': features,
            }
        )

    def _get_jobs(self, path: str) -> None:
        parts = [p for p in path.split('/') if p]
        job_mgr = get_job_manager()
        if len(parts) >= 3 and parts[1] == 'jobs':  # noqa: PLR2004
            job_id = parts[2]
            job = job_mgr.get_job(job_id)
            if job is None:
                self._send_error(f"Job '{job_id}' not found", status=404)
                return
            self._send_json(job)
            return
        if path == '/api/jobs':
            self._send_json({'jobs': job_mgr.list_jobs(limit=20)})
            return
        self._send_error('Not Found', status=404)

    def _get_profile_current(self, _query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        self._send_json(
            {
                'active_username': user,
                'profile': pm.get_profile_info(user),
                'available_profiles': pm.list_profiles(),
            }
        )

    def _get_profile_list(self, _query: dict[str, list[str]]) -> None:
        pm = get_profile_manager()
        self._send_json(
            {
                'active_username': pm.active_username,
                'profiles': pm.list_profiles(),
            }
        )

    def _get_account_summary(self, _query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        self._send_json(pm.get_account_summary(user))

    def _get_forecast_info(self, _query: dict[str, list[str]]) -> None:
        extractor = ForecastZarrExtractor()
        self._send_json(extractor.get_latest_issue_info())

    def _get_weather_models(self, _query: dict[str, list[str]]) -> None:
        self._send_json(
            {
                'models': get_weather_models_info(),
                'variables': list(SUPPORTED_VARIABLES.values()),
                'sync': get_sync_status(),
            }
        )

    def _get_weather_frame(self, path: str) -> None:
        parts = path.strip('/').split('/')
        if len(parts) != 6:  # noqa: PLR2004
            self._send_error(
                'Expected /api/weather/frames/{model}/{variable}/{step}.png',
                status=400,
            )
            return
        model, variable, leaf = parts[3], parts[4], parts[5]
        if model not in SUPPORTED_MODELS or variable not in SUPPORTED_VARIABLES:
            self._send_error(
                f'Unsupported weather frame request: {model}/{variable}',
                status=400,
            )
            return
        if leaf == 'index.json':
            self._send_json(get_frame_index(model, variable))
            return
        step = _parse_int(leaf.split('.')[0])
        if step is None:
            self._send_error('Invalid frame step', status=400)
            return
        png = generate_weather_frame(model, variable, step)
        self._send_tile_png(png, max_age=86400)

    def _get_weather_tile(self, path: str) -> None:
        parts = path.strip('/').split('/')
        if len(parts) < 9:  # noqa: PLR2004
            self._send_error('Invalid weather tile path', status=400)
            return
        model, variable = parts[3], parts[4]
        step = _parse_int(parts[5])
        z = _parse_int(parts[6])
        x = _parse_int(parts[7])
        y = _parse_int(parts[8].split('.')[0])
        if (
            model not in SUPPORTED_MODELS
            or variable not in SUPPORTED_VARIABLES
            or step is None
            or z is None
            or x is None
            or y is None
        ):
            self._send_error('Invalid weather tile parameters', status=400)
            return
        tile_bytes = generate_raster_tile(model, variable, step, z, x, y)
        self._send_tile_png(tile_bytes)

    def _get_weather_wind_vectors(self, query: dict[str, list[str]]) -> None:
        model = query.get('model', ['ecmwf_ifs'])[0]
        step = _parse_int(query.get('step', ['0'])[0], default=0) or 0
        subsample = _parse_int(query.get('subsample', ['2'])[0], default=2) or 2
        self._send_json(get_wind_vectors(model, step, subsample))

    def _get_weather_probe(self, query: dict[str, list[str]]) -> None:
        lat = _parse_float(query.get('lat', ['40.0'])[0], default=40.0)
        lon = _parse_float(query.get('lon', ['-86.0'])[0], default=-86.0)
        if lat is None or lon is None:
            self._send_error('Invalid lat/lon coordinates', status=400)
            return
        self._send_json(get_weather_probe(lat, lon))

    def _get_weather_catchment_summary(
        self,
        query: dict[str, list[str]],
    ) -> None:
        step = _parse_int(query.get('step', ['0'])[0], default=0) or 0
        watershed_id = query.get('catchment_id', [''])[0] or None
        user = self._get_request_username()
        pm = get_profile_manager()
        user_watersheds = pm.load_watersheds(user)
        matched_ws: dict[str, Any] | None = None
        if watershed_id:
            for w in user_watersheds:
                wid = str(
                    w.get('properties', {}).get('catchment_id') or w.get('id')
                )
                if wid == str(watershed_id):
                    matched_ws = w
                    break
        if matched_ws is None and user_watersheds:
            matched_ws = user_watersheds[0]
        if matched_ws is None:
            self._send_error(
                'No active catchment found for weather summary.',
                status=404,
            )
            return
        model = query.get('model', ['ecmwf_ifs'])[0]
        self._send_json(get_catchment_weather_summary(matched_ws, step, model))

    def _get_weather_radar_times(self, _query: dict[str, list[str]]) -> None:
        req = urllib.request.Request(  # noqa: S310
            RAINVIEWER_MAPS_URL,
            headers={'User-Agent': 'OpenHydroNet/1.0'},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
            data = json.loads(resp.read().decode('utf-8'))
        self._send_json(data)

    def _get_maas_models(self, _query: dict[str, list[str]]) -> None:
        self._send_json(
            {
                'models': [
                    {
                        'id': 'floodhub',
                        'name': 'Google FloodHub',
                        'type': 'AI / Physics',
                        'horizon_days': 7,
                        'units': 'm³/s',
                        'status': 'operational',
                    },
                    {
                        'id': 'glofas',
                        'name': 'Copernicus GloFAS',
                        'type': '30-Day Ensemble (CEMS)',
                        'horizon_days': 15,
                        'units': 'm³/s',
                        'status': 'operational',
                    },
                    {
                        'id': 'geoglows',
                        'name': 'GEOGLOWS ECMWF',
                        'type': '15-Day 51-Member Ensemble',
                        'horizon_days': 15,
                        'units': 'm³/s',
                        'status': 'operational',
                    },
                ]
            }
        )

    def _get_maas_gauges(self, query: dict[str, list[str]]) -> None:
        bbox_str = query.get('bbox', [''])[0]
        if bbox_str:
            min_lat, min_lon, max_lat, max_lon = _parse_bbox(
                bbox_str,
                (32.0, -118.0, 34.0, -116.0),
            )
        else:
            min_lat = (
                _parse_float(query.get('min_lat', ['32.0'])[0], default=32.0)
                or 32.0
            )
            min_lon = (
                _parse_float(
                    query.get('min_lon', ['-118.0'])[0], default=-118.0
                )
                or -118.0
            )
            max_lat = (
                _parse_float(query.get('max_lat', ['34.0'])[0], default=34.0)
                or 34.0
            )
            max_lon = (
                _parse_float(
                    query.get('max_lon', ['-116.0'])[0], default=-116.0
                )
                or -116.0
            )
        gauges = fetch_floodhub_gauges_bbox(min_lat, min_lon, max_lat, max_lon)
        self._send_json({'gauges': gauges, 'count': len(gauges)})

    def _get_maas_network(self, query: dict[str, list[str]]) -> None:
        model = (query.get('model', ['glofas'])[0] or 'glofas').strip().lower()
        zoom = _parse_int(query.get('zoom', ['6'])[0], default=6) or 6
        bbox_str = query.get('bbox', [''])[0]
        if bbox_str:
            min_lon, min_lat, max_lon, max_lat = _parse_bbox(
                bbox_str,
                (-180.0, -60.0, 180.0, 75.0),
            )
        else:
            min_lon = (
                _parse_float(
                    query.get('min_lon', ['-180.0'])[0],
                    default=-180.0,
                )
                or -180.0
            )
            min_lat = (
                _parse_float(query.get('min_lat', ['-60.0'])[0], default=-60.0)
                or -60.0
            )
            max_lon = (
                _parse_float(query.get('max_lon', ['180.0'])[0], default=180.0)
                or 180.0
            )
            max_lat = (
                _parse_float(query.get('max_lat', ['75.0'])[0], default=75.0)
                or 75.0
            )
        refresh = (query.get('refresh', ['0'])[0] or '0').strip().lower() in (
            '1',
            'true',
            'yes',
        )
        self._send_json(
            get_model_network(
                model,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                zoom,
                refresh=refresh,
            )
        )

    def _get_maas_forecast(self, query: dict[str, list[str]]) -> None:
        lat = _parse_float(query.get('lat', ['32.756'])[0], default=32.756)
        lon = _parse_float(query.get('lon', ['-117.252'])[0], default=-117.252)
        if lat is None or lon is None:
            self._send_error('Invalid lat/lon coordinates', status=400)
            return
        gauge_id = query.get('gauge_id', [''])[0] or None
        river_id_str = query.get('river_id', [''])[0] or None
        river_id = (
            int(river_id_str)
            if river_id_str and river_id_str.isdigit()
            else None
        )
        reach_id = query.get('reach_id', [''])[0] or None
        if (
            not reach_id
            and river_id_str
            and river_id_str.upper().startswith('HYRIV_')
        ):
            reach_id = river_id_str
        upstream_area_km2 = _parse_float(
            query.get('upstream_area_km2', [''])[0] or None
        )
        area_min_km2 = _parse_float(query.get('area_min_km2', [''])[0] or None)
        network = query.get('network', [''])[0] or None
        models_str = query.get('models', ['floodhub,glofas,geoglows'])[0]
        requested_models = [
            m.strip().lower() for m in models_str.split(',') if m.strip()
        ]
        raw_reference = query.get('glofas_reference', [''])[0]
        glofas_reference = (
            raw_reference.strip().lower()
            if raw_reference.strip()
            else DEFAULT_GLOFAS_REFERENCE
        )
        if glofas_reference not in GLOFAS_REFERENCE_FIELDS:
            self._send_error(
                'Invalid glofas_reference; expected one of '
                f'{sorted(GLOFAS_REFERENCE_FIELDS)}',
                status=400,
            )
            return
        data = get_unified_maas_forecast(
            lat,
            lon,
            gauge_id=gauge_id,
            river_id=river_id,
            requested_models=requested_models,
            reach_id=reach_id,
            upstream_area_km2=upstream_area_km2,
            area_min_km2=area_min_km2,
            network=network,
            glofas_reference=glofas_reference,
        )
        self._send_json(data)

    def _get_maas_watershed(self, query: dict[str, list[str]]) -> None:
        lat = _parse_float(query.get('lat', ['32.756'])[0], default=32.756)
        lon = _parse_float(query.get('lon', ['-117.252'])[0], default=-117.252)
        if lat is None or lon is None:
            self._send_error('Invalid lat/lon coordinates', status=400)
            return
        fabric = query.get(
            'geofabric',
            query.get('fabric', ['hydroatlas_full']),
        )[0]
        gauge_id = query.get('gauge_id', [''])[0] or None
        river_id_str = query.get('river_id', [''])[0] or None
        river_id = (
            int(river_id_str)
            if river_id_str and river_id_str.isdigit()
            else None
        )
        data = get_maas_watershed_polygon(
            lat,
            lon,
            fabric=fabric,
            gauge_id=gauge_id,
            river_id=river_id,
        )
        self._send_json(data)

    def _get_maas_reaches(self, query: dict[str, list[str]]) -> None:
        lat = _parse_float(query.get('lat', ['32.756'])[0], default=32.756)
        lon = _parse_float(query.get('lon', ['-117.252'])[0], default=-117.252)
        if lat is None or lon is None:
            self._send_error('Invalid lat/lon coordinates', status=400)
            return
        gauge_id = query.get('gauge_id', [''])[0] or None
        river_id = query.get('river_id', [''])[0] or None
        upstream_area_km2 = _parse_float(
            query.get('upstream_area_km2', [''])[0] or None
        )
        area_min_km2 = _parse_float(query.get('area_min_km2', [''])[0] or None)
        network = query.get('network', [''])[0] or None
        reaches = get_maas_fetcher().resolve_reaches(
            lat,
            lon,
            upstream_area_km2=upstream_area_km2,
            area_min_km2=area_min_km2,
            network=network,
            river_id=river_id,
            gauge_id=gauge_id,
        )
        self._send_json(reaches)

    def _get_maas_return_periods(self, query: dict[str, list[str]]) -> None:
        provider = (
            (query.get('provider', ['glofas'])[0] or 'glofas').strip().lower()
        )
        method = (query.get('method', ['gema'])[0] or 'gema').strip().lower()
        lat = _parse_float(query.get('lat', [''])[0] or None)
        lon = _parse_float(query.get('lon', [''])[0] or None)
        reach_id = query.get('reach_id', query.get('river_id', ['']))[0] or None
        rps = get_maas_fetcher().fetch_return_periods(
            provider,
            reach_id=reach_id,
            lat=lat,
            lon=lon,
            method=method,
        )
        self._send_json(
            {'provider': provider, 'method': method, 'return_periods': rps}
        )

    def _get_finetune_models(self, _query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        self._send_json({'models': list_finetune_models(user)})

    def _get_finetune_status(self, query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        basin_id = (
            query.get('basin_id', [''])[0]
            or query.get('catchment_id', [''])[0]
            or ''
        ).strip()
        if not basin_id:
            ws = pm.load_watersheds(user)
            if ws:
                basin_id = str(
                    ws[0].get('properties', {}).get('catchment_id')
                    or ws[0].get('id')
                    or ''
                )
        if not basin_id:
            basin_id = 'custom_basin_01'
        model_id = (
            query.get('model_id', ['google-floodhub-settings-110-epochs'])[0]
            or 'google-floodhub-settings-110-epochs'
        ).strip()
        split_ratio = _parse_float(query.get('split_ratio', [''])[0] or None)
        payload = get_finetune_workspace_status(
            username=user,
            basin_id=basin_id,
            model_id=model_id,
            split_ratio=split_ratio,
        )
        self._send_json(payload)

    def _get_forecast_status(self, query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        catchment_id = (
            query.get('catchment_id', [''])[0]
            or query.get('basin_id', [''])[0]
            or None
        )
        raw_cids = query.get('catchment_ids', []) + query.get('basin_ids', [])
        catchment_ids = [
            part.strip()
            for item in raw_cids
            if item
            for part in str(item).split(',')
            if part.strip()
        ]
        lookback_days = (
            _parse_int(
                query.get('nowcast_lookback_days', ['14'])[0],
                default=14,
            )
            or 14
        )
        query_upstream = query.get('query_upstream', ['1'])[0] not in (
            '0',
            'false',
            'False',
        )
        requested_issue_date = (
            query.get('issue_date', [''])[0].strip()
            or query.get('requested_issue_date', [''])[0].strip()
            or None
        )
        if not is_valid_issue_date_str(requested_issue_date):
            self._send_error(
                f'Invalid issue_date {requested_issue_date!r}; expected '
                'YYYY-MM-DD.',
                status=400,
            )
            return

        if not catchment_id and catchment_ids:
            catchment_id = catchment_ids[0]
        if not catchment_id:
            ws = pm.load_watersheds(user)
            if ws:
                catchment_id = str(
                    ws[0].get('properties', {}).get('catchment_id')
                    or ws[0].get('id')
                    or ''
                )
        if not catchment_id:
            self._send_json(
                {
                    'status': 'no_catchment',
                    'username': user,
                    'basin_id': None,
                    'selected_catchment_ids': [],
                    'catchments_status': {},
                    'issue_date': None,
                    'expected_state_date': None,
                    'saved_state': {'exists': False, 'state_date': None},
                    'is_state_from_previous_day': False,
                    'can_fetch_hotstart': False,
                    'can_run_coldstart': False,
                    'can_run_hotstart': False,
                    'coldstart_status_message': (
                        'Select or delineate a catchment first.'
                    ),
                    'hotstart_status_message': (
                        'Select or delineate a catchment first.'
                    ),
                    'nowcast_products': {},
                    'forecast_products': {},
                    'input_products': {},
                }
            )
            return

        payload = inspect_realtime_stores_for_basin(
            username=user,
            basin_id=str(catchment_id),
            nowcast_lookback_days=lookback_days,
            query_upstream_if_empty=query_upstream,
            basin_ids=catchment_ids or None,
            requested_issue_date=requested_issue_date,
        )
        self._send_json(payload)

    def _get_archives(self, _query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        archives: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        self._collect_historical_archives(user, pm, archives, seen_paths)
        self._collect_forecast_archives(user, pm, archives, seen_paths)
        self._collect_attribute_archives(user, pm, archives, seen_paths)
        self._collect_polygon_archives(user, pm, archives, seen_paths)
        self._send_json({'archives': archives})

    def _collect_historical_archives(
        self,
        user: str,
        pm: ProfileManager,
        archives: list[dict[str, Any]],
        seen_paths: set[str],
    ) -> None:
        p_dir = pm.get_profile_dir(user)
        hist_dirs = (
            pm.get_historical_dir(user),
            pm.get_dynamics_dir(user),
            pm.get_targets_dir(user),
            p_dir / 'archives',
        )
        for base_dir in hist_dirs:
            if not base_dir.exists():
                continue
            for item in find_zarr_dirs(base_dir):
                if 'forecast' in item.name or str(item) in seen_paths:
                    continue
                seen_paths.add(str(item))
                meta = pm.inspect_zarr_store(
                    item,
                    p_dir,
                    default_category='Historical Data',
                )
                basin_list = list(meta.get('basins') or [])
                n_times = int(meta.get('n_timesteps') or 0)
                size_bytes = int(meta.get('size_bytes') or 0)
                is_master = (
                    item.name == HISTORICAL_MASTER_ZARR_NAME
                    or len(basin_list) > 1
                )
                cid = (
                    f'HISTORICAL_{len(basin_list)}_BASINS'
                    if is_master
                    else (
                        basin_list[0]
                        if basin_list
                        else (
                            item.parent.name
                            if item.parent != base_dir
                            else item.stem
                        )
                    )
                )
                archives.append(
                    {
                        'catchment_id': cid,
                        'name': item.name,
                        'filename': item.name,
                        'type': (
                            'master_historical' if is_master else 'historical'
                        ),
                        'category': 'Historical Data',
                        'size_bytes': size_bytes,
                        'size_mb': round(size_bytes / (1024 * 1024), 3),
                        'total_basins': len(basin_list),
                        'basins': basin_list,
                        'n_timesteps': n_times,
                        'path': str(item),
                    }
                )

    def _collect_forecast_archives(
        self,
        user: str,
        pm: ProfileManager,
        archives: list[dict[str, Any]],
        seen_paths: set[str],
    ) -> None:
        p_dir = pm.get_profile_dir(user)
        fc_dirs = [pm.get_forecast_dir(user)]
        rt_dir = p_dir / 'realtime'
        if rt_dir.exists():
            fc_dirs.append(rt_dir)
        for base_dir in fc_dirs:
            if not base_dir.exists():
                continue
            for item in find_zarr_dirs(base_dir):
                if str(item) in seen_paths:
                    continue
                seen_paths.add(str(item))
                meta = pm.inspect_zarr_store(
                    item,
                    p_dir,
                    default_category='Forecast Weather',
                )
                mfc_basins = list(meta.get('basins') or [])
                mfc_lead = int(
                    meta.get('lead_time_days') or meta.get('n_timesteps') or 0
                )
                size_bytes = int(meta.get('size_bytes') or 0)
                cid = (
                    f'FORECAST_{len(mfc_basins)}_BASINS'
                    if len(mfc_basins) > 1
                    else (mfc_basins[0] if mfc_basins else item.stem)
                )
                archives.append(
                    {
                        'catchment_id': cid,
                        'name': item.name,
                        'filename': item.name,
                        'type': 'forecast',
                        'category': 'Forecast Weather',
                        'size_bytes': size_bytes,
                        'size_mb': round(size_bytes / (1024 * 1024), 3),
                        'total_basins': len(mfc_basins),
                        'basins': mfc_basins,
                        'lead_time_days': mfc_lead,
                        'path': str(item),
                    }
                )

    def _collect_attribute_archives(
        self,
        user: str,
        pm: ProfileManager,
        archives: list[dict[str, Any]],
        seen_paths: set[str],
    ) -> None:
        for base_dir in (pm.get_attributes_dir(user), pm.get_statics_dir(user)):
            if not base_dir.exists():
                continue
            for ext in ('*.csv', '*.json', '*.parquet', '*.nc'):
                for attr_file in base_dir.rglob(ext):
                    if str(attr_file) in seen_paths or not attr_file.is_file():
                        continue
                    seen_paths.add(str(attr_file))
                    size_bytes = attr_file.stat().st_size
                    archives.append(
                        {
                            'catchment_id': attr_file.stem,
                            'name': attr_file.name,
                            'filename': attr_file.name,
                            'type': 'attributes',
                            'category': 'Static Attributes',
                            'size_bytes': size_bytes,
                            'size_mb': round(size_bytes / (1024 * 1024), 3),
                            'download_url': (
                                f'/api/attributes/csv?username={user}'
                                if attr_file.suffix == '.csv'
                                else None
                            ),
                            'path': str(attr_file),
                        }
                    )

    def _collect_polygon_archives(
        self,
        user: str,
        pm: ProfileManager,
        archives: list[dict[str, Any]],
        seen_paths: set[str],
    ) -> None:
        has_watersheds = bool(pm.load_watersheds(user))
        base_dir = pm.get_polygons_dir(user)
        if not base_dir.exists():
            return
        for ext in ('*.geojson', '*.json', '*.shp', '*.gpkg'):
            for poly_file in base_dir.glob(ext):
                if poly_file.name == 'watersheds.json' and not has_watersheds:
                    continue
                if str(poly_file) in seen_paths or not poly_file.is_file():
                    continue
                seen_paths.add(str(poly_file))
                size_bytes = poly_file.stat().st_size
                archives.append(
                    {
                        'catchment_id': poly_file.stem,
                        'name': poly_file.name,
                        'filename': poly_file.name,
                        'type': 'polygons',
                        'category': 'Polygons & Boundaries',
                        'size_bytes': size_bytes,
                        'size_mb': round(size_bytes / (1024 * 1024), 3),
                        'path': str(poly_file),
                    }
                )

    def _get_attributes_csv(self, _query: dict[str, list[str]]) -> None:
        user = self._get_request_username()
        pm = get_profile_manager()
        csv_candidates = (
            pm.get_attributes_dir(user) / 'caravan_hydroatlas_attributes.csv',
            pm.get_profile_dir(user)
            / 'archives'
            / 'caravan_hydroatlas_attributes.csv',
        )
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path is None:
            self._send_error(
                'Caravan attributes CSV has not been extracted yet.',
                status=404,
            )
            return
        content = csv_path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv; charset=utf-8')
        self.send_header(
            'Content-Disposition',
            'attachment; filename="attributes.csv"',
        )
        self.send_header('Content-Length', str(len(content)))
        self._send_cors_headers(include_methods=False)
        self.end_headers()
        self.wfile.write(content)

    def _get_dem_tile(self, path: str) -> None:
        parts = path.removeprefix('/api/tiles/dem/').split('/')
        if len(parts) != 3 or not parts[2].endswith('.png'):  # noqa: PLR2004
            self._send_error('Invalid tile coordinates', status=400)
            return
        z = _parse_int(parts[0])
        x = _parse_int(parts[1])
        y = _parse_int(parts[2].removesuffix('.png'))
        if z is None or x is None or y is None:
            self._send_error('Invalid tile coordinates', status=400)
            return
        tile_bytes = get_dem_tile_server().render_tile(z, x, y)
        self._send_tile_png(tile_bytes)

    def _get_index_html(self, _query: dict[str, list[str]]) -> None:
        index_path = STATIC_DIR / 'index.html'
        if not index_path.exists():
            self._send_error('Not Found', status=404)
            return
        content = index_path.read_bytes()
        self._set_cors_headers(200, 'text/html; charset=utf-8')
        self.wfile.write(content)

    def _get_static_asset(self, path: str) -> None:
        rel_path = path.removeprefix('/static/')
        file_path = (STATIC_DIR / rel_path).resolve()
        if (
            file_path.exists()
            and file_path.is_file()
            and file_path.is_relative_to(STATIC_DIR.resolve())
        ):
            mime, _ = mimetypes.guess_type(str(file_path))
            self._set_cors_headers(200, mime or 'application/octet-stream')
            self.wfile.write(file_path.read_bytes())
            return
        self._send_error('Not Found', status=404)

    def do_POST(self) -> None:
        """Dispatch HTTP POST requests across JSON API endpoints."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = (
            _parse_int(self.headers.get('Content-Length'), default=0) or 0
        )
        if content_length > 0:
            body_text = self.rfile.read(content_length).decode('utf-8').strip()
            data = json.loads(body_text) if body_text else {}
        else:
            data = {}

        routes = {
            '/api/delineate': self._post_delineate,
            '/api/watersheds/delete': self._post_watersheds_delete,
            '/api/watersheds/rename': self._post_watersheds_rename,
            '/api/watersheds/clear': self._post_watersheds_clear,
            '/api/profile/login': self._post_profile_login,
            '/api/profile/create': self._post_profile_create,
            '/api/profile/delete': self._post_profile_delete,
            '/api/profile/logout': self._post_profile_logout,
            '/api/profile/settings': self._post_profile_settings,
            '/api/profile/save_watersheds': self._post_profile_save_watersheds,
            '/api/polygons/upload': self._post_polygons_upload,
            '/api/weather/historical': self._post_weather_historical,
            '/api/weather/forecast': self._post_weather_forecast,
            '/api/attributes/extract': self._post_attributes_extract,
            '/api/forecast/fetch-realtime': self._post_forecast_fetch_realtime,
            '/api/forecast/run-model': self._post_forecast_run_model,
            '/api/forecast/upload-streamflow': (
                self._post_forecast_upload_streamflow
            ),
            '/api/forecast/clear-streamflow': (
                self._post_forecast_clear_streamflow
            ),
            '/api/finetune/upload-csv': self._post_finetune_upload_csv,
            '/api/finetune/clear-data': self._post_finetune_clear_data,
            '/api/finetune/run': self._post_finetune_run,
            '/api/finetune/reset': self._post_finetune_reset,
            '/api/weather/refresh': self._post_weather_refresh,
            '/api/weather/sync': self._post_weather_refresh,
        }
        handler = routes.get(path)
        if handler is None:
            self._send_error('Not Found', status=404)
            return
        handler(data)

    def _post_delineate(self, data: dict[str, Any]) -> None:
        raw_dataset = (
            data.get('dataset')
            or data.get('dem')
            or data.get('dem_id')
            or 'hydroatlas'
        )
        lat = _parse_float(data.get('latitude'))
        lon = _parse_float(data.get('longitude'))
        snap_radius = _parse_float(data.get('snap_radius_km'))
        mode = str(data.get('mode') or 'dem_flow_direction')

        if lat is None or lon is None:
            self._send_error('Missing latitude or longitude', status=400)
            return

        delineator = _get_delineator(str(raw_dataset))
        if delineator is None:
            self._send_error(
                f'Unknown hydrography dataset: {raw_dataset}',
                status=400,
            )
            return

        user = self._get_request_username(data)
        pm = get_profile_manager()
        geojson = delineator.delineate_catchment(
            lat=lat,
            lon=lon,
            snap_radius_km=snap_radius,
            mode=mode,
        )
        catchment_id = geojson['properties']['catchment_id']
        geojson['properties']['source'] = 'delineated'

        existing_feats = pm.load_watersheds(user)
        updated = False
        for i, ef in enumerate(existing_feats):
            ef_id = str(
                ef.get('properties', {}).get('catchment_id') or ef.get('id')
            )
            if ef_id == str(catchment_id):
                existing_name = (ef.get('properties') or {}).get('name')
                if isinstance(existing_name, str) and existing_name.strip():
                    geojson['properties']['name'] = existing_name.strip()
                existing_feats[i] = geojson
                updated = True
                break
        if not updated:
            existing_feats.append(geojson)
        pm.save_watersheds(existing_feats, username=user)
        self._send_json(geojson)

    def _post_watersheds_delete(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        pm = get_profile_manager()
        catchment_id = data.get('catchment_id')
        if not catchment_id:
            self._send_error('Missing catchment_id to delete', status=400)
            return
        existing_feats = pm.load_watersheds(user)
        new_feats = [
            f
            for f in existing_feats
            if str(f.get('properties', {}).get('catchment_id') or f.get('id'))
            != str(catchment_id)
        ]
        pm.save_watersheds(new_feats, username=user)

        attr_dir = pm.get_attributes_dir(user)
        json_cache_path = attr_dir / 'extracted_attributes.json'
        csv_path = attr_dir / 'caravan_hydroatlas_attributes.csv'
        if json_cache_path.exists():
            cached = json.loads(json_cache_path.read_text(encoding='utf-8'))
            r_map = (
                dict(cached.get('results_by_id', cached))
                if isinstance(cached, dict)
                else {}
            )
            if str(catchment_id) in r_map:
                del r_map[str(catchment_id)]
                json_cache_path.write_text(
                    json.dumps(
                        sanitize_for_json({'results_by_id': r_map}),
                        indent=2,
                    ),
                    encoding='utf-8',
                )
                if r_map:
                    get_attributes_extractor().export_caravan_csv(
                        list(r_map.values()),
                        csv_path,
                    )
                elif csv_path.exists():
                    csv_path.unlink()

        self._send_json(
            {
                'status': (
                    'deleted'
                    if len(new_feats) < len(existing_feats)
                    else 'not_found'
                ),
                'catchment_id': str(catchment_id),
                'remaining_count': len(new_feats),
            }
        )

    def _post_watersheds_rename(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        pm = get_profile_manager()
        catchment_id = data.get('catchment_id')
        new_name = data.get('name')

        if (
            not isinstance(catchment_id, str)
            or not catchment_id.strip()
            or not is_valid_watershed_name(new_name)
        ):
            self._send_error(
                'Missing catchment_id or valid name (1-120 printable chars)',
                status=400,
            )
            return

        clean_id = catchment_id.strip()
        clean_name = str(new_name).strip()

        existing_feats = pm.load_watersheds(user)
        has_match = any(
            str(
                f.get('properties', {}).get('catchment_id')
                or f.get('id')
                or ''
            )
            == clean_id
            for f in existing_feats
        )
        if not has_match:
            self._send_error('Catchment not found', status=404)
            return

        updated_feature = pm.rename_watershed(
            clean_id,
            clean_name,
            username=user,
        )
        self._send_json(
            {
                'status': 'renamed',
                'feature': sanitize_for_json(updated_feature),
            }
        )

    def _post_watersheds_clear(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        pm = get_profile_manager()
        pm.save_watersheds([], username=user)
        attr_dir = pm.get_attributes_dir(user)
        for fname in (
            'extracted_attributes.json',
            'caravan_hydroatlas_attributes.csv',
        ):
            fpath = attr_dir / fname
            if fpath.exists():
                fpath.unlink()
        self._send_json({'status': 'cleared', 'remaining_count': 0})

    def _post_profile_login(self, data: dict[str, Any]) -> None:
        username = str(data.get('username') or 'guest')
        create_if_missing = bool(data.get('create_if_missing', True))
        pm = get_profile_manager()
        clean_user = _sanitize_username_token(username) or _GUEST_USERNAME
        if (
            not create_if_missing
            and clean_user != _GUEST_USERNAME
            and not pm.account_exists(clean_user)
        ):
            self._send_error(
                f"Account '{clean_user}' does not exist. Click "
                "'Create Account' to make a new account.",
                status=404,
            )
            return

        prof = pm.login_profile(
            username=username,
            email=data.get('email'),
            display_name=data.get('display_name'),
            settings=data.get('settings'),
            create_if_missing=create_if_missing,
        )
        active_user = str(prof.get('username') or 'guest')
        loaded_feats = pm.load_watersheds(active_user)
        self._send_json(
            {
                'status': 'success',
                'message': f"Logged in as '{active_user}'",
                'profile': prof,
                'watersheds': loaded_feats,
                'count': len(loaded_feats),
                'account_summary': pm.get_account_summary(active_user),
            }
        )

    def _post_profile_create(self, data: dict[str, Any]) -> None:
        raw_username = str(data.get('username') or '')
        pm = get_profile_manager()
        clean_user = _sanitize_username_token(raw_username)
        if not raw_username.strip() or not clean_user:
            self._send_error(
                'Please enter a username (letters, numbers, _, -, .).',
                status=400,
            )
            return
        if clean_user == _GUEST_USERNAME:
            self._send_error(
                "'guest' is reserved for the guest session. "
                'Choose another username.',
                status=400,
            )
            return
        if pm.account_exists(clean_user):
            self._send_error(
                f"Account '{clean_user}' already exists. Sign in instead.",
                status=409,
            )
            return

        prof = pm.create_account(
            username=raw_username,
            email=data.get('email'),
            display_name=data.get('display_name'),
        )
        created_user = str(prof.get('username') or clean_user)
        self._send_json(
            {
                'status': 'success',
                'message': f"Created account '{created_user}'",
                'profile': prof,
                'watersheds': [],
                'count': 0,
                'account_summary': pm.get_account_summary(created_user),
            }
        )

    def _post_profile_delete(self, data: dict[str, Any]) -> None:
        raw_username = str(data.get('username') or '')
        pm = get_profile_manager()
        clean_user = _sanitize_username_token(raw_username)
        if not raw_username.strip() or not clean_user:
            self._send_error(
                'Specify an account username to remove.',
                status=400,
                code='invalid_account',
            )
            return
        if clean_user == _GUEST_USERNAME:
            self._send_error(
                'The guest session cannot be removed.',
                status=400,
                code='invalid_account',
            )
            return
        if not pm.account_exists(clean_user):
            self._send_error(
                f"Account '{clean_user}' does not exist.",
                status=404,
                code='account_not_found',
            )
            return

        removed = pm.delete_account(raw_username)
        prof = pm.get_profile_info(pm.active_username)
        loaded_feats = pm.load_watersheds(pm.active_username)
        self._send_json(
            {
                'status': 'success',
                'message': f"Removed account '{removed}'",
                'removed': removed,
                'profile': prof,
                'watersheds': loaded_feats,
                'count': len(loaded_feats),
                'account_summary': pm.get_account_summary(pm.active_username),
            }
        )

    def _post_profile_logout(self, _data: dict[str, Any]) -> None:
        pm = get_profile_manager()
        prof = pm.logout_profile()
        loaded_feats = pm.load_watersheds('guest')
        self._send_json(
            {
                'status': 'success',
                'message': 'Switched to Guest profile',
                'profile': prof,
                'watersheds': loaded_feats,
                'count': len(loaded_feats),
                'account_summary': pm.get_account_summary('guest'),
            }
        )

    def _post_profile_settings(self, data: dict[str, Any]) -> None:
        """Merge ``settings`` into the requesting user's ``profile.json``.

        The target profile is resolved from the request (payload ``username``,
        ``X-User-Profile`` header, or query string) so that one browser session
        cannot rewrite another account's preferences through the process-global
        active profile. Only the guest session or an existing account can be
        updated; unknown usernames are rejected instead of being created.
        """
        user = self._get_request_username(data)
        settings = data.get('settings')
        if not isinstance(settings, dict):
            self._send_error(
                "Expected a 'settings' object mapping preference names to"
                ' values.',
                status=400,
            )
            return
        pm = get_profile_manager()
        if user != _GUEST_USERNAME and not pm.account_exists(user):
            self._send_error(
                f"Account '{user}' does not exist.",
                status=404,
                code='account_not_found',
            )
            return
        prof = pm.save_settings(settings, username=user)
        self._send_json(
            {
                'status': 'saved',
                'username': prof['username'],
                'profile': prof,
            }
        )

    @staticmethod
    def _parse_watersheds_payload(
        payload: object,
    ) -> list[dict[str, Any]] | str:
        """Validate a ``watersheds`` payload into a list of GeoJSON Features.

        Args:
            payload: A list of Feature dicts or a FeatureCollection dict.

        Returns:
            The list of Feature dicts, or an error message string describing
            why the payload is not a valid watersheds list.
        """
        if isinstance(payload, dict):
            if payload.get('type') != 'FeatureCollection':
                return (
                    "'watersheds' object must be a GeoJSON FeatureCollection."
                )
            payload = payload.get('features')
        if not isinstance(payload, list):
            return (
                "'watersheds' must be a list of GeoJSON Features or a"
                ' FeatureCollection.'
            )
        for idx, feat in enumerate(payload):
            if not isinstance(feat, dict) or feat.get('type') != 'Feature':
                return f'watersheds[{idx}] is not a GeoJSON Feature object.'
            if 'geometry' not in feat:
                return f"watersheds[{idx}] is missing its 'geometry' member."
        return payload

    def _post_profile_save_watersheds(self, data: dict[str, Any]) -> None:
        """Persist the profile's watersheds and rebuild catchment artifacts.

        When the payload carries a ``watersheds`` list (or FeatureCollection),
        it replaces the profile's stored watersheds. Without a payload, the
        watersheds already stored for the profile are re-written so that
        ``polygons/watersheds.json`` and the ``catchments/`` GeoJSONs and basin
        split lists are synchronized from a single source of truth.
        """
        user = self._get_request_username(data)
        pm = get_profile_manager()
        if 'watersheds' in data:
            parsed = self._parse_watersheds_payload(data['watersheds'])
            if isinstance(parsed, str):
                self._send_error(parsed, status=400)
                return
            features = parsed
        else:
            features = pm.load_watersheds(user)
        pm.save_watersheds(features, username=user)
        basin_ids = [
            str(f.get('properties', {}).get('catchment_id') or f.get('id'))
            for f in pm.load_watersheds(user)
        ]
        self._send_json(
            {
                'status': 'saved',
                'count': len(basin_ids),
                'username': user,
                'catchment_ids': basin_ids,
            }
        )

    @staticmethod
    def _extract_raw_geojson_features(polygon_data: object) -> list[Any]:
        if isinstance(polygon_data, list):
            return polygon_data
        if not isinstance(polygon_data, dict):
            return []
        gtype = polygon_data.get('type')
        if gtype == 'FeatureCollection':
            return list(polygon_data.get('features', []))
        if gtype == 'Feature':
            return [polygon_data]
        return [
            {
                'type': 'Feature',
                'geometry': polygon_data.get('geometry', polygon_data),
                'properties': polygon_data.get('properties', {}),
            }
        ]

    @staticmethod
    def _merge_watershed_features(
        existing_feats: list[dict[str, Any]],
        new_feats: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        index_by_id = {
            str(f.get('properties', {}).get('catchment_id') or f.get('id')): i
            for i, f in enumerate(existing_feats)
        }
        for nf in new_feats:
            cid = str(
                nf.get('properties', {}).get('catchment_id') or nf.get('id')
            )
            if cid in index_by_id:
                existing_feats[index_by_id[cid]] = nf
            else:
                index_by_id[cid] = len(existing_feats)
                existing_feats.append(nf)
        return existing_feats

    def _post_polygons_upload(self, data: dict[str, Any]) -> None:
        polygon_data = data.get('polygon') or data.get('geojson') or data
        if not polygon_data:
            self._send_error(
                'No polygon or GeoJSON payload provided',
                status=400,
            )
            return

        raw_features = self._extract_raw_geojson_features(polygon_data)
        if not raw_features:
            self._send_error('No features found in GeoJSON', status=400)
            return

        normalized_features: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        total_area = 0.0
        for idx, feat in enumerate(raw_features):
            if not isinstance(feat, dict):
                continue
            parsed_feat = _normalize_uploaded_feature(feat, idx)
            if parsed_feat is None:
                continue
            norm_f, summary, area_km2 = parsed_feat
            normalized_features.append(norm_f)
            summaries.append(summary)
            total_area += area_km2

        if not normalized_features:
            self._send_error(
                'No valid Polygon / MultiPolygon features could be parsed.',
                status=400,
            )
            return

        user = self._get_request_username(data)
        pm = get_profile_manager()
        merged = self._merge_watershed_features(
            pm.load_watersheds(user),
            normalized_features,
        )
        pm.save_watersheds(merged, username=user)

        all_minx = min(f['properties']['bbox'][0] for f in normalized_features)
        all_miny = min(f['properties']['bbox'][1] for f in normalized_features)
        all_maxx = max(f['properties']['bbox'][2] for f in normalized_features)
        all_maxy = max(f['properties']['bbox'][3] for f in normalized_features)
        self._send_json(
            {
                'type': 'FeatureCollection',
                'count': len(normalized_features),
                'total_area_km2': round(total_area, 2),
                'bbox': [all_minx, all_miny, all_maxx, all_maxy],
                'summary': summaries,
                'features': normalized_features,
            }
        )

    def _post_weather_historical(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        catchment_id = data.get('catchment_id')
        catchment_ids = data.get('catchment_ids') or data.get('basin_ids')
        extract_all = bool(data.get('all_active', False))
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        weather_source = str(
            data.get('weather_source')
            or data.get('source')
            or data.get('dataset')
            or 'era5'
        )
        frequency = str(data.get('frequency', '1D'))
        variables = data.get('variables')
        notify_email = str(data.get('notify_email') or '').strip()
        run_async = bool(data.get('async') or 'notify_email' in data)

        pm = get_profile_manager()
        extractor = HistoricalZarrExtractor(
            output_dir=pm.get_historical_dir(user)
        )
        job_mgr = get_job_manager()
        server_url = self._get_server_url()

        user_watersheds = pm.load_watersheds(user)
        ws_by_id = {
            str(f.get('properties', {}).get('catchment_id') or f.get('id')): f
            for f in user_watersheds
        }

        if (
            extract_all
            or catchment_ids
            or (not catchment_id and data.get('features'))
        ):
            features_to_extract = self._resolve_batch_features(
                data,
                catchment_ids,
                extract_all=extract_all,
                ws_by_id=ws_by_id,
                user_watersheds=user_watersheds,
            )
            if not features_to_extract:
                self._send_error(
                    'No basins found to extract in batch',
                    status=400,
                )
                return
            if run_async:
                job_id = job_mgr.submit_job(
                    job_type='historical_weather_batch',
                    task_fn=extractor.extract_and_archive_batch,
                    features=features_to_extract,
                    start_date=start_date,
                    end_date=end_date,
                    weather_source=weather_source,
                    freq=frequency,
                    variables=variables,
                    notify_email=notify_email,
                    server_url=server_url,
                    metadata={
                        'weather_source': weather_source,
                        'basin_count': len(features_to_extract),
                        'start_date': start_date,
                        'end_date': end_date,
                    },
                )
                self._send_json(
                    {
                        'status': 'queued',
                        'job_id': job_id,
                        'message': (
                            'Historical extraction job submitted for '
                            f'{len(features_to_extract)} basins.'
                        ),
                        'batch': True,
                        'basin_count': len(features_to_extract),
                        'notify_email': notify_email,
                        'weather_source': weather_source,
                    }
                )
                return
            res = extractor.extract_and_archive_batch(
                features=features_to_extract,
                start_date=start_date,
                end_date=end_date,
                weather_source=weather_source,
                freq=frequency,
                variables=variables,
            )
            self._send_json(res)
            return

        polygon_input = data.get('polygon') or data.get('geometry')
        if not polygon_input and str(catchment_id) in ws_by_id:
            polygon_input = ws_by_id[str(catchment_id)]
        if not polygon_input and not catchment_id:
            self._send_error(
                'catchment_id, catchment_ids, or polygon geometry is required',
                status=400,
            )
            return
        if not polygon_input:
            self._send_error(
                f"Catchment '{catchment_id}' not found. Please delineate a "
                'basin or supply polygon first.',
                status=404,
            )
            return
        if run_async:
            job_id = job_mgr.submit_job(
                job_type='historical_weather_single',
                task_fn=extractor.extract_and_archive,
                polygon_input=polygon_input,
                basin_id=catchment_id,
                start_date=start_date,
                end_date=end_date,
                weather_source=weather_source,
                freq=frequency,
                variables=variables,
                notify_email=notify_email,
                server_url=server_url,
                metadata={
                    'weather_source': weather_source,
                    'basin_id': catchment_id,
                    'start_date': start_date,
                    'end_date': end_date,
                },
            )
            self._send_json(
                {
                    'status': 'queued',
                    'job_id': job_id,
                    'message': (
                        'Historical extraction job submitted for basin '
                        f'{catchment_id}.'
                    ),
                    'batch': False,
                    'basin_id': catchment_id,
                    'notify_email': notify_email,
                    'weather_source': weather_source,
                }
            )
            return
        res = extractor.extract_and_archive(
            polygon_input=polygon_input,
            basin_id=catchment_id,
            start_date=start_date,
            end_date=end_date,
            weather_source=weather_source,
            freq=frequency,
            variables=variables,
        )
        self._send_json(res)

    def _resolve_batch_features(
        self,
        data: dict[str, Any],
        catchment_ids: list[str] | None,
        *,
        extract_all: bool,
        ws_by_id: dict[str, dict[str, Any]],
        user_watersheds: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if data.get('features'):
            return list(data['features'])
        if catchment_ids:
            return [
                ws_by_id[str(cid)]
                for cid in catchment_ids
                if str(cid) in ws_by_id
            ]
        if extract_all:
            return user_watersheds
        return []

    def _post_weather_forecast(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        pm = get_profile_manager()
        job_mgr = get_job_manager()

        catchment_id = data.get('catchment_id')
        catchment_ids = data.get('catchment_ids') or []
        extract_all = bool(
            data.get('all_active')
            or data.get('extract_all')
            or data.get('batch')
        )
        notify_email = str(data.get('notify_email') or '').strip()
        run_async = bool(data.get('async') or notify_email)

        raw_horizon = (
            data.get('horizon_days') or data.get('horizon_hours') or 10
        )
        raw_val = _parse_int(raw_horizon, default=10) or 10
        horizon_days = raw_val // 24 if raw_val > 15 else raw_val  # noqa: PLR2004
        horizon_days = max(1, min(horizon_days, 15))

        extractor = ForecastZarrExtractor(output_dir=pm.get_forecast_dir(user))
        user_watersheds = pm.load_watersheds(username=user)
        ws_by_id = {
            str(f['properties']['catchment_id']): f
            for f in user_watersheds
            if 'catchment_id' in f.get('properties', {})
        }
        features_to_extract = self._resolve_batch_features(
            data,
            catchment_ids,
            extract_all=extract_all,
            ws_by_id=ws_by_id,
            user_watersheds=list(ws_by_id.values()),
        )

        if extract_all or len(features_to_extract) > 1:
            if not features_to_extract:
                self._send_error(
                    'No watershed polygons found in user profile to extract '
                    'forecast for.',
                    status=400,
                )
                return
            if run_async:
                job_id = job_mgr.submit_job(
                    job_type='realtime_forecast_batch',
                    task_fn=extractor.fetch_and_archive_batch,
                    features=features_to_extract,
                    horizon_days=horizon_days,
                    notify_email=notify_email,
                    server_url=self._get_server_url(),
                    metadata={
                        'source': 'dynamical.org',
                        'basin_count': len(features_to_extract),
                        'horizon_days': horizon_days,
                        'username': user,
                    },
                )
                self._send_json(
                    {
                        'status': 'queued',
                        'job_id': job_id,
                        'message': (
                            'Real-time forecast extraction job queued for '
                            f'{len(features_to_extract)} basins.'
                        ),
                        'batch': True,
                        'basin_count': len(features_to_extract),
                        'notify_email': notify_email,
                    }
                )
                return
            self._send_json(
                extractor.fetch_and_archive_batch(
                    features=features_to_extract,
                    horizon_days=horizon_days,
                )
            )
            return

        polygon_input = data.get('polygon') or data.get('geometry')
        if not polygon_input and features_to_extract:
            polygon_input = features_to_extract[0]
        elif (
            not polygon_input and catchment_id and str(catchment_id) in ws_by_id
        ):
            polygon_input = ws_by_id[str(catchment_id)]

        if not polygon_input and not catchment_id:
            self._send_error(
                'catchment_id, polygon, or features list is required',
                status=400,
            )
            return
        if not polygon_input:
            self._send_error(
                f"Catchment '{catchment_id}' not found in active profile.",
                status=404,
            )
            return
        if run_async:
            job_id = job_mgr.submit_job(
                job_type='realtime_forecast_single',
                task_fn=extractor.fetch_and_archive,
                catchment_feature=polygon_input,
                basin_id=catchment_id,
                horizon_days=horizon_days,
                notify_email=notify_email,
                server_url=self._get_server_url(),
                metadata={
                    'source': 'dynamical.org',
                    'catchment_id': catchment_id,
                    'horizon_days': horizon_days,
                    'username': user,
                },
            )
            self._send_json(
                {
                    'status': 'queued',
                    'job_id': job_id,
                    'message': (
                        'Real-time forecast extraction queued for basin '
                        f'{catchment_id}.'
                    ),
                    'batch': False,
                    'catchment_id': catchment_id,
                    'notify_email': notify_email,
                }
            )
            return
        self._send_json(
            extractor.fetch_and_archive(
                catchment_feature=polygon_input,
                basin_id=catchment_id,
                horizon_days=horizon_days,
            )
        )

    def _post_attributes_extract(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        pm = get_profile_manager()
        is_batch = bool(
            data.get('all_active', False) or data.get('scope') == 'all'
        )
        catchment_ids_input = (
            data.get('catchment_ids') or data.get('basin_ids') or []
        )
        features_input = data.get('features', [])
        catchment_id = data.get('catchment_id')
        polygon_input = data.get('polygon') or data.get('geometry')
        era5_source = str(data.get('era5_source') or 'hybas')

        extractor = get_attributes_extractor()
        user_watersheds = pm.load_watersheds(user)
        ws_by_id = {
            str(f.get('properties', {}).get('catchment_id') or f.get('id')): f
            for f in user_watersheds
        }
        attr_dir = pm.get_attributes_dir(user)
        json_cache_path = attr_dir / 'extracted_attributes.json'
        existing_by_id: dict[str, Any] = {}
        if json_cache_path.exists():
            cached = json.loads(json_cache_path.read_text(encoding='utf-8'))
            if isinstance(cached, dict):
                existing_by_id = dict(cached.get('results_by_id', cached))

        if (
            is_batch
            or catchment_ids_input
            or features_input
            or (not polygon_input and not catchment_id and user_watersheds)
        ):
            target_features = self._resolve_batch_features(
                {'features': features_input},
                catchment_ids_input,
                extract_all=True,
                ws_by_id=ws_by_id,
                user_watersheds=user_watersheds,
            )
            if not target_features:
                self._send_error(
                    'No matching active or uploaded catchments to extract '
                    'attributes for.',
                    status=400,
                )
                return
            normalized_batch = self._normalize_attribute_batch(target_features)
            results = extractor.extract_attributes_batch(
                normalized_batch,
                era5_source=era5_source,
            )
            csv_path = self._persist_extracted_attributes(
                user,
                pm,
                extractor,
                existing_by_id,
                results,
            )
            ui_results = [build_attribute_card_payload(r) for r in results]
            ui_existing = {
                cid: card_payload_from_cache_entry(cid, p)
                for cid, p in existing_by_id.items()
            }
            first = ui_results[0] if ui_results else {}
            self._send_json(
                {
                    'status': 'success',
                    'batch': True,
                    'count': len(ui_results),
                    'csv_path': str(csv_path),
                    'results': ui_results,
                    'results_by_id': ui_existing,
                    'catchment_id': first.get('catchment_id'),
                    'summary': first.get('summary', {}),
                    'categories': first.get('categories', {}),
                    'attributes': first.get('attributes', {}),
                    'processed_attributes': first.get(
                        'processed_attributes',
                        {},
                    ),
                    'flat_attributes': first.get('flat_attributes', {}),
                    'intersected_subbasins_count': sum(
                        int(r.get('intersected_subbasins_count', 0))
                        for r in ui_results
                    ),
                }
            )
            return

        if not polygon_input and str(catchment_id) in ws_by_id:
            polygon_input = ws_by_id[str(catchment_id)]
        if not polygon_input and not catchment_id:
            self._send_error(
                'catchment_id, catchment_ids, features, or polygon geometry '
                'is required',
                status=400,
            )
            return
        if not polygon_input:
            self._send_error(
                f"Catchment '{catchment_id}' not found. Please delineate a "
                'basin or upload a polygon first.',
                status=404,
            )
            return
        result = extractor.extract_attributes_for_polygon(
            polygon_input,
            catchment_id=catchment_id,
            era5_source=era5_source,
        )
        csv_path = self._persist_extracted_attributes(
            user,
            pm,
            extractor,
            existing_by_id,
            [result],
        )
        ui_payload = build_attribute_card_payload(result)
        ui_existing = {
            cid: card_payload_from_cache_entry(cid, p)
            for cid, p in existing_by_id.items()
        }
        self._send_json(
            {
                **ui_payload,
                'status': 'success',
                'batch': False,
                'count': 1,
                'csv_path': str(csv_path),
                'results': [ui_payload],
                'results_by_id': ui_existing,
            }
        )

    def _normalize_attribute_batch(
        self,
        target_features: list[Any],
    ) -> list[Any]:
        normalized: list[Any] = []
        for idx, feat in enumerate(target_features):
            if isinstance(feat, dict):
                f_copy = dict(feat)
                props = dict(f_copy.get('properties') or {})
                if not (props.get('gauge_id') or props.get('catchment_id')):
                    props['catchment_id'] = str(
                        f_copy.get('id') or f'basin_{idx + 1}'
                    )
                f_copy['properties'] = props
                normalized.append(f_copy)
            else:
                normalized.append(feat)
        return normalized

    def _persist_extracted_attributes(
        self,
        user: str,
        pm: ProfileManager,
        extractor: StaticAttributesExtractor,
        existing_by_id: dict[str, Any],
        new_results: list[Any],
    ) -> Path:
        attr_dir = pm.get_attributes_dir(user)
        json_cache_path = attr_dir / 'extracted_attributes.json'
        for r in new_results:
            cid = str(
                r.catchment_id
                if hasattr(r, 'catchment_id')
                else r.get('catchment_id', 'basin')
            )
            existing_by_id[cid] = (
                r.to_dict() if hasattr(r, 'to_dict') else dataclasses.asdict(r)
            )
        json_cache_path.parent.mkdir(parents=True, exist_ok=True)
        json_cache_path.write_text(
            json.dumps(
                sanitize_for_json({'results_by_id': existing_by_id}),
                indent=2,
            ),
            encoding='utf-8',
        )
        csv_path = attr_dir / 'caravan_hydroatlas_attributes.csv'
        all_results = list(existing_by_id.values()) or new_results
        extractor.export_caravan_csv(all_results, csv_path)

        master_zarr = pm.get_historical_dir(user) / HISTORICAL_MASTER_ZARR_NAME
        if master_zarr.exists():
            basins, _, _ = _inspect_zarr_archive(master_zarr)
            basin_set = set(basins)
            for r in new_results:
                cid = str(
                    r.catchment_id
                    if hasattr(r, 'catchment_id')
                    else r.get('catchment_id', 'basin')
                )
                if cid in basin_set:
                    extractor.append_attributes_to_zarr(master_zarr, cid, r)
        return csv_path

    def _post_forecast_fetch_realtime(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        catchment_id = data.get('catchment_id') or data.get('basin_id')
        catchment_ids = data.get('catchment_ids') or data.get('basin_ids')
        mode = str(data.get('mode', 'coldstart'))
        reference_date = data.get('reference_date', 'latest')
        products = data.get('products')
        lookback_days = data.get('lookback_days')
        overwrite = bool(data.get('overwrite', False))

        norm_mode = mode.lower().strip().replace('-', '').replace('_', '')
        if norm_mode not in ('coldstart', 'hotstart'):
            self._send_error(
                f"Invalid mode {mode!r}; expected 'coldstart' or 'hotstart'.",
                status=400,
            )
            return
        if norm_mode == 'hotstart':
            status_info = inspect_realtime_stores_for_basin(
                username=user,
                basin_id=str(catchment_id or ''),
                basin_ids=catchment_ids,
            )
            if not status_info.get('can_fetch_hotstart', False):
                msg = status_info.get('hotstart_status_message', '')
                self._send_error(
                    f'Cannot fetch in Hot-Start mode: {msg}',
                    status=400,
                )
                return

        payload = fetch_realtime_for_basin(
            username=user,
            basin_id=catchment_id,
            basin_ids=catchment_ids,
            mode=mode,
            reference_date=reference_date,
            products=products,
            lookback_days=lookback_days,
            overwrite=overwrite,
        )
        self._send_json(payload)

    def _post_forecast_run_model(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        catchment_id = data.get('catchment_id') or data.get('basin_id')
        catchment_ids = data.get('catchment_ids') or data.get('basin_ids')
        mode = str(data.get('mode', 'coldstart'))
        model_run_dir = data.get('model_run_dir')
        requested_issue_date = data.get('requested_issue_date') or data.get(
            'issue_date'
        )
        precip_overrides = data.get('precip_overrides')
        temp_overrides = data.get('temp_overrides')
        product_overrides = data.get('product_overrides')
        is_what_if = bool(data.get('is_what_if', False))

        norm_mode = mode.lower().strip().replace('-', '').replace('_', '')
        if norm_mode not in ('coldstart', 'hotstart'):
            self._send_error(
                f"Invalid mode {mode!r}; expected 'coldstart' or 'hotstart'.",
                status=400,
            )
            return
        if norm_mode == 'hotstart':
            status_info = inspect_realtime_stores_for_basin(
                username=user,
                basin_id=str(catchment_id or ''),
                basin_ids=catchment_ids,
            )
            if not status_info.get('can_run_hotstart', False):
                msg = status_info.get('hotstart_status_message', '')
                self._send_error(
                    f'Cannot run model in Hot-Start mode: {msg}',
                    status=400,
                )
                return

        payload = run_hydrological_model_for_basin(
            username=user,
            basin_id=catchment_id,
            basin_ids=catchment_ids,
            mode=mode,
            model_run_dir=model_run_dir,
            requested_issue_date=requested_issue_date,
            precip_overrides=precip_overrides,
            temp_overrides=temp_overrides,
            product_overrides=product_overrides,
            is_what_if=is_what_if,
        )
        self._send_json(payload)

    def _post_forecast_upload_streamflow(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        catchment_id = data.get('catchment_id') or data.get('basin_id')
        cid_str = str(catchment_id) if catchment_id else None
        csv_content = str(data.get('csv_content') or data.get('csv') or '')
        filename = str(data.get('filename') or 'streamflow.csv')
        err_msg = validate_streamflow_csv_request(csv_content, cid_str)
        if err_msg is not None:
            self._send_error(err_msg, status=400)
            return
        payload = upload_realtime_streamflow_csv(
            username=user,
            catchment_id=cid_str,
            csv_content=csv_content,
            filename=filename,
        )
        self._send_json(payload)

    def _post_forecast_clear_streamflow(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        catchment_id = data.get('catchment_id') or data.get('basin_id')
        cid_str = str(catchment_id or '').strip()
        if not cid_str:
            self._send_error('catchment_id is required', status=400)
            return
        payload = clear_realtime_streamflow_csv(
            username=user,
            catchment_id=cid_str,
        )
        self._send_json(payload)

    def _post_finetune_upload_csv(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        basin_id = str(
            data.get('basin_id')
            or data.get('catchment_id')
            or 'custom_basin_01'
        ).strip()
        data_type = str(data.get('data_type') or '').strip()
        csv_content = str(data.get('csv_content') or data.get('csv') or '')
        if not data_type or not csv_content.strip():
            self._send_error(
                'Both data_type and non-empty csv_content are required.',
                status=400,
            )
            return
        filename = str(data.get('filename') or f'{data_type}.csv')
        raw_units = data.get('units')
        units = str(raw_units) if raw_units is not None else None
        area_km2 = _parse_float(data.get('area_km2'))
        lead_time = _parse_int(data.get('lead_time'), default=7) or 7
        model_id = str(
            data.get('model_id') or 'google-floodhub-settings-110-epochs'
        )
        upload_info = upload_finetune_csv(
            username=user,
            basin_id=basin_id,
            data_type=data_type,
            csv_content=csv_content,
            filename=filename,
            units=units,
            area_km2=area_km2,
            target_lead_time=lead_time,
        )
        status_payload = get_finetune_workspace_status(
            username=user,
            basin_id=basin_id,
            model_id=model_id,
        )
        self._send_json(
            {
                'status': 'uploaded',
                'upload': upload_info,
                'workspace': status_payload,
            }
        )

    def _post_finetune_clear_data(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        basin_id = str(
            data.get('basin_id')
            or data.get('catchment_id')
            or 'custom_basin_01'
        ).strip()
        model_id = str(
            data.get('model_id') or 'google-floodhub-settings-110-epochs'
        )
        clear_finetune_data(username=user, basin_id=basin_id)
        status_payload = get_finetune_workspace_status(
            username=user,
            basin_id=basin_id,
            model_id=model_id,
        )
        self._send_json(
            {
                'status': 'cleared',
                'workspace': status_payload,
            }
        )

    def _post_finetune_run(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        basin_id = str(
            data.get('basin_id')
            or data.get('catchment_id')
            or 'custom_basin_01'
        ).strip()
        model_id = str(
            data.get('model_id') or 'google-floodhub-settings-110-epochs'
        ).strip()
        epochs = _parse_int(data.get('epochs'), default=5) or 5
        learning_rate = _parse_float(data.get('learning_rate'))
        loss = data.get('loss')
        lr_strategy = data.get('learning_rate_strategy')
        batch_size = _parse_int(data.get('batch_size'))
        finetune_modules = data.get('finetune_modules')
        split_ratio = _parse_float(data.get('split_ratio'), default=0.7) or 0.7
        payload = run_finetune_epochs(
            username=user,
            basin_id=basin_id,
            model_id=model_id,
            epochs=epochs,
            learning_rate=learning_rate,
            loss=str(loss) if loss else None,
            learning_rate_strategy=(str(lr_strategy) if lr_strategy else None),
            batch_size=batch_size,
            finetune_modules=finetune_modules,
            split_ratio=split_ratio,
        )
        self._send_json(payload)

    def _post_finetune_reset(self, data: dict[str, Any]) -> None:
        user = self._get_request_username(data)
        basin_id = str(
            data.get('basin_id')
            or data.get('catchment_id')
            or 'custom_basin_01'
        ).strip()
        model_id = str(
            data.get('model_id') or 'google-floodhub-settings-110-epochs'
        ).strip()
        payload = reset_finetune_session(
            username=user,
            basin_id=basin_id,
            model_id=model_id,
        )
        self._send_json(payload)

    def _post_weather_refresh(self, data: dict[str, Any]) -> None:
        req_model = data.get('model')
        req_models = data.get('models')
        force = bool(data.get('force', False))
        async_mode = bool(data.get('async', True))

        if req_models:
            target_models = list(req_models)
        elif req_model and req_model != 'all':
            target_models = [str(req_model)]
        else:
            target_models = list(SUPPORTED_MODELS.keys())

        for m in target_models:
            if m not in SUPPORTED_MODELS:
                self._send_error(f"Unsupported model '{m}'", status=400)
                return

        def _sync_task() -> dict[str, Any]:
            status = run_sync_subprocess(models=target_models, force=force)
            reload_weather_if_changed()
            return {
                'sync': get_sync_status(),
                'models': get_weather_models_info(),
                'requested_models': target_models,
                'result': status,
            }

        if async_mode:
            job_id = get_job_manager().submit_job(
                job_type='weather_viewer_refresh',
                task_fn=_sync_task,
                metadata={'models': target_models, 'force': force},
            )
            self._send_json(
                {
                    'status': 'queued',
                    'job_id': job_id,
                    'requested_models': target_models,
                    'message': 'Weather sync queued in background',
                }
            )
            return
        self._send_json(_sync_task())

    def log_message(self, fmt: str, *args: object) -> None:
        """Log HTTP access events via standard logger."""
        logger.info('[%s] %s', self.log_date_time_string(), fmt % args)


EarthkitHydroHandler = OpenHydroNetHandler


def _start_maas_warmup() -> None:
    """Preload MaaS network pyramids in a background daemon thread."""

    def _warmup() -> None:
        for model_name in ('glofas', 'geoglows', 'floodhub'):
            get_model_network(model_name, -180.0, -85.0, 180.0, 85.0, 3)

    threading.Thread(
        target=_warmup,
        name='MaaSPyramidWarmup',
        daemon=True,
    ).start()


def run_server(port: int = _DEFAULT_PORT, host: str = _DEFAULT_HOST) -> None:
    """Start the multi-threaded OpenHydroNet HTTP server."""
    get_profile_manager().clear_guest_data()
    _start_maas_warmup()

    ThreadingHTTPServer.allow_reuse_address = True
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, OpenHydroNetHandler)
    logger.info('OpenHydroNet Web Server listening on http://%s:%d', host, port)
    httpd.serve_forever()


def main() -> None:
    """CLI entry point for the OpenHydroNet Web Server."""
    parser = argparse.ArgumentParser(description='OpenHydroNet Web Server')
    default_port = (
        _parse_int(os.environ.get('PORT'), default=_DEFAULT_PORT)
        or _DEFAULT_PORT
    )
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        default=default_port,
        help='Port to run the server on',
    )
    parser.add_argument(
        '--host',
        type=str,
        default=_DEFAULT_HOST,
        help='Host address to bind to',
    )
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == '__main__':
    main()

__all__ = [
    'EarthkitHydroHandler',
    'HTTPServer',
    'OpenHydroNetHandler',
    'ThreadingHTTPServer',
    'get_dem_tile_server',
    'main',
    'run_server',
]
