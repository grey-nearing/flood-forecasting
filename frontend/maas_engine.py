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

"""Earthkit Hydro frontend adapter for the Models-as-a-Service (MaaS) Hub tab.

Delegates all backend provider HTTP queries, Zarr extraction, reach snapping,
and return-period calculations to `maas` (`MaaSConfig`, `MaaSDataFetcher`,
`maas.floodhub`, `maas.glofas`, `maas.geoglows`,
`maas.networks`, `maas.thresholds`), and all UI presentation, multi-model
consensus rows, aligned chart timelines, and watershed polygons to
`frontend.maas_viewer` (`MaaSViewer`).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

_ws_root = str(Path(__file__).resolve().parents[1])
if _ws_root not in sys.path:
    sys.path.insert(0, _ws_root)

try:
    from frontend.config import (
        CACHE_DIR as _DEFAULT_CACHE_DIR,
        RIVER_NETWORKS_DIR as _DEFAULT_RIVER_DIR,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        CACHE_DIR as _DEFAULT_CACHE_DIR,
        RIVER_NETWORKS_DIR as _DEFAULT_RIVER_DIR,
    )

from frontend.maas_viewer import (
    MaaSViewer,
    build_aligned_timeline,
    build_consensus_row,
    build_flood_summary,
    reach_exceedance_summary,
    spread_confidence,
)
from maas.config import (
    DEFAULT_MAAS_MODELS,
    FLOODHUB_BASE_URL,
    GEOGLOWS_BASE_URL,
    GLOFAS_BASE_URL,
    MAAS_MODEL_NAMES,
    RETURN_PERIOD_YEARS,
    MaaSConfig,
    convert_discharge_units,
    normalize_requested_models,
    parse_finite_float,
    parse_float_or_default,
    parse_int,
)
from maas.fetcher import (
    MaaSDataFetcher,
    SQLiteCache,
    align_daily_series,
    daily_series,
    fetch_forecasts,
    fetch_gauges,
    fetch_historical,
    fetch_return_periods,
    resolve_reaches,
    window_peak,
)
from maas.floodhub import haversine_km
from maas.networks import (
    glofas_cell_center,
    glofas_cell_polygon,
    is_geoglows_river_id,
)
from maas.thresholds import (
    CANONICAL_RETURN_PERIODS,
    EXCEEDANCE_CLASSES,
    RISK_RANK,
    UNASSESSED_COLOR,
    UNASSESSED_LABEL,
    classify_exceedance,
    compute_empirical_weibull_return_periods,
    compute_gumbel_return_periods,
    compute_return_periods,
    estimate_return_period_years,
    extract_annual_maxima,
    gumbel_quantile_from_return_periods,
    thresholds_from_return_periods,
)
from utils.file_paths import FLOODHUB_API_KEY_FILE


def _load_default_floodhub_key() -> str:
    env_key = os.environ.get('FLOODHUB_API_KEY', '').strip()
    if env_key:
        return env_key
    key_file = FLOODHUB_API_KEY_FILE
    if key_file.is_file():
        return key_file.read_text(encoding='utf-8').strip()
    return ''


DEFAULT_FLOODHUB_KEY = _load_default_floodhub_key()


def get_maas_config() -> MaaSConfig:
    """Build an explicit `MaaSConfig` from `frontend.config` and environment overrides."""
    fh_key = os.environ.get('FLOODHUB_API_KEY', DEFAULT_FLOODHUB_KEY).strip()
    return MaaSConfig(
        river_networks_dir=Path(_DEFAULT_RIVER_DIR),
        cache_dir=Path(_DEFAULT_CACHE_DIR),
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
    """Fetch FloodHub gauges inside `(min_lat, min_lon, max_lat, max_lon)` via `maas`."""
    return get_maas_fetcher().fetch_gauges(
        (float(min_lat), float(min_lon), float(max_lat), float(max_lon))
    )


def fetch_floodhub_forecast(gauge_id: str) -> dict[str, Any]:
    """Fetch 7-day forecast and return-period thresholds for a FloodHub gauge via `maas`."""
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
    """Resolve the 9-digit GEOGLOWS `LINKNO` reach ID at `(lat, lon)` via `maas`."""
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


def get_unified_maas_forecast(  # noqa: PLR0913
    lat: float,
    lon: float,
    gauge_id: str | None = None,
    river_id: Any = None,
    requested_models: list[str] | None = None,
    reach_id: str | None = None,
    upstream_area_km2: Any = None,
    area_min_km2: Any = None,
    network: str | None = None,
) -> dict[str, Any]:
    """Render unified 3-provider MaaS forecast view via `MaaSViewer.render_forecast_view`."""
    return get_maas_viewer().render_forecast_view(
        float(lat),
        float(lon),
        gauge_id=gauge_id,
        river_id=parse_int(river_id) if is_geoglows_river_id(river_id) else None,
        reach_id=reach_id,
        requested_models=requested_models,
        upstream_area_km2=upstream_area_km2,
        area_min_km2=area_min_km2,
        network=network,
    )


def get_maas_watershed_polygon(  # noqa: PLR0913
    lat: float,
    lon: float,
    fabric: str = 'hydroatlas_full',
    gauge_id: str | None = None,
    river_id: int | None = None,
    geofabric: str | None = None,
) -> dict[str, Any]:
    """Render watershed polygon for the selected hydrofabric via `MaaSViewer`."""
    return get_maas_viewer().render_watershed_polygon(
        float(lat),
        float(lon),
        fabric=fabric,
        gauge_id=gauge_id,
        river_id=river_id,
        geofabric=geofabric,
    )


__all__ = [
    'CANONICAL_RETURN_PERIODS',
    'DEFAULT_MAAS_MODELS',
    'EXCEEDANCE_CLASSES',
    'MAAS_MODEL_NAMES',
    'MaaSConfig',
    'MaaSDataFetcher',
    'MaaSViewer',
    'RETURN_PERIOD_YEARS',
    'RISK_RANK',
    'SQLiteCache',
    'UNASSESSED_COLOR',
    'UNASSESSED_LABEL',
    'aggregate_maas_forecast',
    'align_daily_series',
    'build_aligned_timeline',
    'build_consensus_row',
    'build_flood_summary',
    'classify_exceedance',
    'compute_empirical_weibull_return_periods',
    'compute_gumbel_return_periods',
    'compute_return_periods',
    'convert_discharge_units',
    'daily_series',
    'estimate_return_period_years',
    'extract_annual_maxima',
    'fetch_floodhub_forecast',
    'fetch_floodhub_gauges_bbox',
    'fetch_forecasts',
    'fetch_gauges',
    'fetch_geoglows_forecast',
    'fetch_geoglows_return_periods',
    'fetch_geoglows_river_id',
    'fetch_glofas_forecast',
    'fetch_glofas_return_periods',
    'fetch_historical',
    'fetch_return_periods',
    'get_maas_config',
    'get_maas_fetcher',
    'get_maas_viewer',
    'get_maas_watershed_polygon',
    'get_unified_maas_forecast',
    'glofas_cell_center',
    'glofas_cell_polygon',
    'gumbel_quantile_from_return_periods',
    'haversine_km',
    'is_geoglows_river_id',
    'normalize_requested_models',
    'parse_finite_float',
    'parse_float_or_default',
    'parse_int',
    'reach_exceedance_summary',
    'resolve_reaches',
    'spread_confidence',
    'thresholds_from_return_periods',
    'window_peak',
]
