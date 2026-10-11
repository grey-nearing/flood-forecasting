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

"""Frontend adapter for the Models-as-a-Service (MaaS) Hub tab.

Delegates all backend provider HTTP queries, Zarr extraction, reach snapping,
and return-period calculations to `maas` (`MaaSConfig`, `MaaSDataFetcher`,
`maas.floodhub`, `maas.glofas`, `maas.geoglows`, `maas.networks`,
`maas.thresholds`), and all UI presentation, multi-model consensus rows,
aligned chart timelines, and watershed polygons to `frontend.maas_viewer`
(`MaaSViewer`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import frontend.config as _frontend_config
from frontend.maas_viewer import MaaSViewer
from maas.config import (
    FLOODHUB_BASE_URL,
    GEOGLOWS_BASE_URL,
    GLOFAS_BASE_URL,
    MaaSConfig,
    parse_int,
)
from maas.fetcher import MaaSDataFetcher
from maas.networks import is_geoglows_river_id
from utils.file_paths import FLOODHUB_API_KEY_FILE


def _load_default_floodhub_key() -> str:
    """Load the default FloodHub API key from environment or local key file."""
    env_key = os.environ.get('FLOODHUB_API_KEY', '').strip()
    if env_key:
        return env_key
    key_file = FLOODHUB_API_KEY_FILE
    if key_file.is_file():
        return key_file.read_text(encoding='utf-8').strip()
    return ''


DEFAULT_FLOODHUB_KEY: str = _load_default_floodhub_key()


def get_maas_config() -> MaaSConfig:
    """Build an explicit `MaaSConfig` from `frontend.config` and environment."""
    fh_key = os.environ.get('FLOODHUB_API_KEY', DEFAULT_FLOODHUB_KEY).strip()
    return MaaSConfig(
        river_networks_dir=Path(_frontend_config.RIVER_NETWORKS_DIR),
        cache_dir=Path(_frontend_config.CACHE_DIR),
        floodhub_api_key=fh_key,
        floodhub_base_url=FLOODHUB_BASE_URL,
        glofas_base_url=GLOFAS_BASE_URL,
        geoglows_base_url=GEOGLOWS_BASE_URL,
    )


def get_maas_fetcher() -> MaaSDataFetcher:
    """Return a `MaaSDataFetcher` bound to the current `MaaSConfig`."""
    return MaaSDataFetcher(get_maas_config())


def get_maas_viewer() -> MaaSViewer:
    """Return a `MaaSViewer` bound to the current `MaaSConfig`."""
    return MaaSViewer(get_maas_config())


def fetch_floodhub_gauges_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> list[dict[str, Any]]:
    """Fetch FloodHub gauges inside `(min_lat, min_lon, max_lat, max_lon)`."""
    return get_maas_fetcher().fetch_gauges(
        (float(min_lat), float(min_lon), float(max_lat), float(max_lon))
    )


def fetch_floodhub_forecast(gauge_id: str) -> dict[str, Any]:
    """Fetch 7-day forecast and return-period thresholds for FloodHub gauge."""
    fetcher = get_maas_fetcher()
    if not fetcher.config.floodhub_api_key.strip():
        return {
            'model': 'google_floodhub',
            'available': False,
            'status': 'unavailable',
            'gauge_id': gauge_id,
            'data': [],
        }
    return fetcher.floodhub.fetch_forecast(gauge_id)


def fetch_glofas_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 15,
) -> dict[str, Any]:
    """Fetch GloFAS v4 ensemble forecast via `maas`."""
    return get_maas_fetcher().glofas.fetch_forecast(
        float(lat), float(lon), forecast_days=int(forecast_days)
    )


def fetch_glofas_return_periods(
    lat: float,
    lon: float,
    method: str = 'gema',
) -> dict[str, Any] | None:
    """Fetch or compute GloFAS v4 reanalysis return periods via `maas`."""
    return get_maas_fetcher().fetch_return_periods(
        'glofas', lat=float(lat), lon=float(lon), method=method
    )


def fetch_geoglows_river_id(lat: float, lon: float) -> int | None:
    """Resolve the 9-digit GEOGLOWS `LINKNO` reach ID at `(lat, lon)`."""
    return get_maas_fetcher().geoglows.fetch_river_id(float(lat), float(lon))


def fetch_geoglows_forecast(
    lat: float,
    lon: float,
    river_id: int | str | None = None,
) -> dict[str, Any]:
    """Fetch GEOGLOWS v2 ensemble forecast via `maas`."""
    fetcher = get_maas_fetcher()
    rid = (
        parse_int(river_id)
        if is_geoglows_river_id(river_id)
        else fetcher.geoglows.fetch_river_id(float(lat), float(lon))
    )
    if rid is None:
        return {
            'model': 'geoglows',
            'available': False,
            'status': 'unavailable',
            'river_id': None,
            'data': [],
        }
    return fetcher.geoglows.fetch_forecast(rid)


def fetch_geoglows_return_periods(
    river_id: int | str,
    mean_flow: float | None = None,  # noqa: ARG001
    method: str = 'gema',
) -> dict[str, Any] | None:
    """Fetch or compute GEOGLOWS v2 return periods via `maas`."""
    return get_maas_fetcher().fetch_return_periods(
        'geoglows', reach_id=river_id, method=method
    )


def aggregate_maas_forecast(
    lat: float,
    lon: float,
    gauge_id: str | None = None,
    river_id: int | None = None,
) -> dict[str, Any]:
    """Fetch multi-model forecasts via `MaaSDataFetcher.fetch_forecasts`."""
    return get_maas_fetcher().fetch_forecasts(
        float(lat),
        float(lon),
        gauge_id=gauge_id,
        river_id=river_id,
    )


def get_unified_maas_forecast(  # noqa: PLR0913, PLR0917
    lat: float,
    lon: float,
    gauge_id: str | None = None,
    river_id: int | str | None = None,
    requested_models: list[str] | None = None,
    reach_id: str | None = None,
    upstream_area_km2: float | str | None = None,
    area_min_km2: float | str | None = None,
    network: str | None = None,
) -> dict[str, Any]:
    """Render unified 3-provider MaaS forecast view via `MaaSViewer`."""
    return get_maas_viewer().render_forecast_view(
        float(lat),
        float(lon),
        gauge_id=gauge_id,
        river_id=(
            parse_int(river_id) if is_geoglows_river_id(river_id) else None
        ),
        reach_id=reach_id,
        requested_models=requested_models,
        upstream_area_km2=upstream_area_km2,
        area_min_km2=area_min_km2,
        network=network,
    )


def get_maas_watershed_polygon(  # noqa: PLR0913, PLR0917
    lat: float,
    lon: float,
    fabric: str = 'hydroatlas_full',
    gauge_id: str | None = None,
    river_id: int | None = None,
    geofabric: str | None = None,
) -> dict[str, Any]:
    """Render watershed polygon for selected hydrofabric via `MaaSViewer`."""
    return get_maas_viewer().render_watershed_polygon(
        float(lat),
        float(lon),
        fabric=fabric,
        gauge_id=gauge_id,
        river_id=river_id,
        geofabric=geofabric,
    )


__all__ = [
    'DEFAULT_FLOODHUB_KEY',
    'aggregate_maas_forecast',
    'fetch_floodhub_forecast',
    'fetch_floodhub_gauges_bbox',
    'fetch_geoglows_forecast',
    'fetch_geoglows_return_periods',
    'fetch_geoglows_river_id',
    'fetch_glofas_forecast',
    'fetch_glofas_return_periods',
    'get_maas_config',
    'get_maas_fetcher',
    'get_maas_viewer',
    'get_maas_watershed_polygon',
    'get_unified_maas_forecast',
]
