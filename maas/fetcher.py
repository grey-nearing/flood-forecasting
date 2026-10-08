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

"""Pure data-fetching and hydrological orchestration API (`maas.fetcher`).

Provides `MaaSDataFetcher` and functional entry points (`resolve_reaches`,
`fetch_gauges`, `fetch_forecasts`, `fetch_historical`, `fetch_return_periods`)
without any UI presentation, color-coding, or map-corridor rendering logic.
"""

from collections.abc import Mapping, Sequence
import concurrent.futures
from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

import pandas as pd

from maas.config import (
    GLOFAS_RES_DEG,
    RETURN_PERIOD_YEARS,
    MaaSConfig,
    normalize_requested_models,
    parse_finite_float,
    parse_int,
)
from maas.floodhub import FloodHubClient
from maas.geoglows import (
    GeoGLOWSClient,
    extract_geoglows_zarr_series,
)
from maas.glofas import (
    GloFASClient,
    extract_glofas_zarr_series,
)
from maas.networks import (
    GEOGLOWS_LOD,
    GLOFAS_LOD,
    glofas_cell_center,
    is_geoglows_river_id,
    load_geoglows_lookup,
    load_network_pyramid,
    pyramid_signature,
    resolve_cross_network_click,
    snap_geoglows_reach_from_gpkg,
    snap_geoglows_reach_from_network,
    snap_glofas_cell_from_network,
)
from maas.thresholds import (
    CANONICAL_RETURN_PERIODS,
    compute_empirical_weibull_return_periods,
    compute_gumbel_return_periods,
    compute_return_periods,
    extract_annual_maxima,
    thresholds_from_return_periods,
)


class SQLiteCache:
    """Thread-safe persistent SQLite key-value JSON cache with explicit `db_path`."""

    def __init__(self, db_path: Path | str, table_name: str = 'flood_cache') -> None:
        """Initialize the SQLite cache bound to `db_path` and `table_name`."""
        if db_path is None or str(db_path).strip() == '':
            raise ValueError('SQLiteCache requires a non-empty `db_path`.')
        if not table_name.isidentifier():
            raise ValueError(f'Invalid SQLite table name: {table_name!r}')
        self.db_path = db_path if isinstance(db_path, Path) else Path(db_path)
        self.table_name = table_name
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized and self.db_path.exists():
            return
        with self._lock:
            if self._initialized and self.db_path.exists():
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(str(self.db_path), timeout=5)) as conn:
                conn.execute(
                    f'CREATE TABLE IF NOT EXISTS {self.table_name} ('
                    'cache_key TEXT PRIMARY KEY, payload TEXT, created_at REAL)'
                )
                conn.commit()
            self._initialized = True

    def get(self, cache_key: str, max_age_s: float) -> dict[str, Any] | None:
        """Read a JSON payload from SQLite if `db_path` exists and is fresher than `max_age_s`."""
        if not self.db_path.exists():
            return None
        self._ensure_initialized()
        with closing(sqlite3.connect(str(self.db_path), timeout=5)) as conn:
            row = conn.execute(
                f'SELECT payload, created_at FROM {self.table_name} '  # noqa: S608
                'WHERE cache_key = ?',
                (cache_key,),
            ).fetchone()
        if (
            row
            and row[0]
            and (time.time() - float(row[1] or 0.0)) <= max_age_s
        ):
            payload = json.loads(row[0])
            return payload if isinstance(payload, dict) else None
        return None

    def put(self, cache_key: str, payload: Mapping[str, Any]) -> None:
        """Persist a JSON-serializable payload in the SQLite cache."""
        self._ensure_initialized()
        with closing(sqlite3.connect(str(self.db_path), timeout=5)) as conn:
            conn.execute(
                f'INSERT OR REPLACE INTO {self.table_name} '  # noqa: S608
                '(cache_key, payload, created_at) VALUES (?, ?, ?)',
                (cache_key, json.dumps(dict(payload)), time.time()),
            )
            conn.commit()


def daily_series(
    records: Sequence[Mapping[str, Any]] | None,
    key: str,
    fallback_key: str | None = None,
) -> dict[str, float]:
    """Aggregate a (sub-)daily record sequence into daily means `{YYYY-MM-DD: value}`."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in records or ():
        v = parse_finite_float(r.get(key))
        if v is None and fallback_key:
            v = parse_finite_float(r.get(fallback_key))
        t = str(r.get('time', ''))[:10]
        if v is None or len(t) != 10:
            continue
        sums[t] = sums.get(t, 0.0) + v
        counts[t] = counts.get(t, 0) + 1
    return {t: sums[t] / counts[t] for t in sorted(sums)}


def align_daily_series(
    daily: Mapping[str, float],
    dates: Sequence[str],
    ndigits: int = 2,
) -> list[float | None]:
    """Align a `{YYYY-MM-DD: value}` mapping onto a common `dates` list (`None` for missing dates)."""
    return [round(daily[d], ndigits) if d in daily else None for d in dates]


def window_peak(
    daily: Mapping[str, float],
    today_str: str | None = None,
) -> tuple[float | None, str | None]:
    """Return peak `(value, date)` over `today_str` and later (or full series if none)."""
    today = (
        today_str
        if today_str is not None
        else datetime.now(UTC).strftime('%Y-%m-%d')
    )
    items = [(t, v) for t, v in daily.items() if t >= today] or list(
        daily.items()
    )
    if not items:
        return None, None
    t, v = max(items, key=lambda kv: kv[1])
    return round(v, 3), t


_PYRAMID_MEM_CACHE: dict[str, dict[str, Any]] = {}
_LOOKUP_MEM_CACHE: dict[str, tuple[Any, Any]] = {}
_MEM_CACHE_LOCK = threading.Lock()


def _get_cached_glofas_pyramid(glofas_npz: Path) -> dict[str, Any] | None:
    key = str(glofas_npz)
    if key in _PYRAMID_MEM_CACHE:
        return _PYRAMID_MEM_CACHE[key]
    with _MEM_CACHE_LOCK:
        if key in _PYRAMID_MEM_CACHE:
            return _PYRAMID_MEM_CACHE[key]
        net = load_network_pyramid(glofas_npz, pyramid_signature(GLOFAS_LOD))
        if net is not None:
            _PYRAMID_MEM_CACHE[key] = net
        return net


def _get_cached_geoglows_pyramid(geoglows_npz: Path) -> dict[str, Any] | None:
    key = str(geoglows_npz)
    if key in _PYRAMID_MEM_CACHE:
        return _PYRAMID_MEM_CACHE[key]
    with _MEM_CACHE_LOCK:
        if key in _PYRAMID_MEM_CACHE:
            return _PYRAMID_MEM_CACHE[key]
        net = load_network_pyramid(
            geoglows_npz, pyramid_signature(GEOGLOWS_LOD)
        )
        if net is not None:
            _PYRAMID_MEM_CACHE[key] = net
        return net


def _get_cached_geoglows_lookup(attrs_path: Path) -> tuple[Any, Any] | None:
    key = str(attrs_path)
    if key in _LOOKUP_MEM_CACHE:
        return _LOOKUP_MEM_CACHE[key]
    if not attrs_path.exists():
        return None
    with _MEM_CACHE_LOCK:
        if key in _LOOKUP_MEM_CACHE:
            return _LOOKUP_MEM_CACHE[key]
        lookup = load_geoglows_lookup(attrs_path)
        _LOOKUP_MEM_CACHE[key] = lookup
        return lookup


def resolve_reaches(  # noqa: PLR0913
    lat: float,
    lon: float,
    config: MaaSConfig,
    *,
    upstream_area_km2: float | str | None = None,
    area_min_km2: float | str | None = None,
    network: str | None = None,
    river_id: int | str | None = None,
    gauge_id: str | None = None,
) -> dict[str, Any]:
    """Resolve and snap a geographic coordinate across all three model river networks.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        config: Explicit `MaaSConfig` providing `river_networks_dir` and `cache_dir`.
        upstream_area_km2: Optional target upstream drainage area in $\\text{km}^2$
            for cross-network reach matching.
        area_min_km2: Optional minimum upstream drainage area along a merged river line.
        network: Optional source network identifier (`'floodhub'`, `'glofas'`,
            `'geoglows'`).
        river_id: Optional GEOGLOWS 9-digit `LINKNO` or HydroRIVERS `HYRIV_ID`.
        gauge_id: Optional FloodHub gauge identifier.

    Returns:
        Dictionary containing resolved grid cells and reach identifiers for
        `floodhub`, `glofas`, and `geoglows`.
    """
    gl_lat, gl_lon = glofas_cell_center(lat, lon)
    eff_river_id = parse_int(river_id) if is_geoglows_river_id(river_id) else None
    up_area = parse_finite_float(upstream_area_km2)
    min_area = parse_finite_float(area_min_km2)

    cross_snap: dict[str, Any] | None = None
    if up_area is not None and up_area > 0.0:
        glofas_npz = (
            config.river_networks_dir / 'glofas_v4' / 'glofas_network_v1.npz'
        )
        if not glofas_npz.exists():
            glofas_npz = (
                config.cache_dir / 'glofas_v4' / 'glofas_network_v1.npz'
            )
        glofas_net = _get_cached_glofas_pyramid(glofas_npz)

        geoglows_dir = config.river_networks_dir / 'geoglows_v2'
        if not (geoglows_dir / 'global_streams_simplified.gpkg').exists() and not (
            geoglows_dir / 'geoglows_network_v1.npz'
        ).exists():
            geoglows_dir = config.cache_dir / 'geoglows_v2'
        geoglows_npz = geoglows_dir / 'geoglows_network_v1.npz'
        geoglows_net = _get_cached_geoglows_pyramid(geoglows_npz)
        gpkg_path = geoglows_dir / 'global_streams_simplified.gpkg'
        attrs_path = geoglows_dir / 'geoglows_attrs_v1.npz'

        def _snap_geoglows(
            qlat: float, qlon: float, area: float, **kw: Any
        ) -> dict[str, Any] | None:
            snapped = snap_geoglows_reach_from_network(
                geoglows_net, qlat, qlon, area, **kw
            )
            if snapped is not None:
                return snapped
            lookup = _get_cached_geoglows_lookup(attrs_path)
            if lookup is None:
                return None
            return snap_geoglows_reach_from_gpkg(
                gpkg_path, lookup, qlat, qlon, area, **kw
            )

        cross_snap = resolve_cross_network_click(
            lat,
            lon,
            upstream_area_km2=up_area,
            snap_glofas_fn=lambda qlat, qlon, area, **kw: (
                snap_glofas_cell_from_network(glofas_net, qlat, qlon, area, **kw)
            ),
            snap_geoglows_fn=_snap_geoglows,
            area_min_km2=min_area,
            network=network,
            river_id=river_id,
        )
        if cross_snap:
            if cross_snap.get('glofas'):
                gl_lat = float(cross_snap['glofas']['lat'])
                gl_lon = float(cross_snap['glofas']['lon'])
            if cross_snap.get('geoglows'):
                eff_river_id = int(cross_snap['geoglows']['river_id'])

    return {
        'probe': {
            'lat': lat,
            'lon': lon,
            'network': network,
            'upstream_area_km2': (cross_snap or {}).get('target_area_km2')
            or up_area,
        },
        'floodhub': {
            'gauge_id': gauge_id,
            'reach_id': str(river_id)
            if river_id and not is_geoglows_river_id(river_id)
            else None,
        },
        'glofas': {
            'cell_center_lat': gl_lat,
            'cell_center_lon': gl_lon,
            'resolution_deg': GLOFAS_RES_DEG,
            'upstream_area_km2': (
                (cross_snap or {}).get('glofas') or {}
            ).get('upstream_area_km2'),
            'offset_cells': ((cross_snap or {}).get('glofas') or {}).get(
                'offset_cells'
            ),
            'snap': cross_snap.get('glofas') if cross_snap else None,
        },
        'geoglows': {
            'river_id': eff_river_id,
            'upstream_area_km2': (
                (cross_snap or {}).get('geoglows') or {}
            ).get('upstream_area_km2'),
            'offset_km': ((cross_snap or {}).get('geoglows') or {}).get(
                'offset_km'
            ),
            'snap': cross_snap.get('geoglows') if cross_snap else None,
        },
    }


def fetch_gauges(
    config: MaaSConfig,
    bbox: tuple[float, float, float, float],
    *,
    page_size: int = 100,
    include_non_verified: bool = True,
) -> list[dict[str, Any]]:
    """Fetch Google FloodHub gauge metadata inside `(min_lat, min_lon, max_lat, max_lon)`."""
    min_lat, min_lon, max_lat, max_lon = bbox
    client = FloodHubClient(
        api_key=config.floodhub_api_key,
        base_url=config.floodhub_base_url,
        timeout_s=config.http_timeout_s,
        cache_dir=config.cache_dir,
    )
    cached = client.query_cached_gauges_bbox(
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        limit=max(page_size, 500),
    )
    if cached is not None:
        return cached
    if not config.floodhub_api_key.strip():
        return []
    return client.search_gauges_bbox(
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        page_size=page_size,
        include_non_verified=include_non_verified,
    )


def fetch_historical(  # noqa: PLR0913
    config: MaaSConfig,
    provider: str,
    *,
    reach_id: int | str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    zarr_path: Path | None = None,
    end_year: int | None = None,
) -> pd.Series:
    """Fetch a historical daily discharge `pd.Series` (indexed by `pd.DatetimeIndex`).

    Args:
        config: Explicit `MaaSConfig`.
        provider: Either `'glofas'` or `'geoglows'`.
        reach_id: 9-digit GEOGLOWS `LINKNO` (required when `provider='geoglows'`).
        lat: Latitude in decimal degrees (required when `provider='glofas'`).
        lon: Longitude in decimal degrees (required when `provider='glofas'`).
        zarr_path: Optional local Zarr/NetCDF archive path.
        end_year: Optional last calendar year for GloFAS reanalysis queries.

    Returns:
        Daily streamflow `pd.Series` in $\\text{m}^3/\\text{s}$ indexed by `pd.DatetimeIndex`.
    """
    prov = provider.strip().lower()
    if prov == 'glofas':
        if lat is None or lon is None:
            raise ValueError(
                "`fetch_historical` for 'glofas' requires `lat` and `lon`."
            )
        if zarr_path is not None:
            records = extract_glofas_zarr_series(zarr_path, lat, lon)
            idx = pd.to_datetime([r['time'] for r in records])
            vals = [float(r['discharge']) for r in records]
            return pd.Series(vals, index=idx, name='discharge_m3s', dtype=float)
        client = GloFASClient(
            base_url=config.glofas_base_url,
            timeout_s=config.http_timeout_s,
        )
        eff_end_year = (
            end_year if end_year is not None else datetime.now(UTC).year - 1
        )
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'river_discharge',
            'start_date': '1984-01-01',
            'end_date': f'{eff_end_year}-12-31',
        }
        resp = client.session.get(
            client.base_url, params=params, timeout=client.timeout_s
        )
        resp.raise_for_status()
        payload = resp.json()
        daily = payload.get('daily') if isinstance(payload, Mapping) else {}
        times = daily.get('time') or [] if isinstance(daily, Mapping) else []
        values = (
            daily.get('river_discharge') or []
            if isinstance(daily, Mapping)
            else []
        )
        idx = pd.to_datetime([str(t)[:10] for t in times])
        vals = [
            v if (v := parse_finite_float(x)) is not None else float('nan')
            for x in values
        ]
        return pd.Series(vals, index=idx, name='discharge_m3s', dtype=float)

    if prov == 'geoglows':
        rid = parse_int(reach_id)
        if rid is None or not is_geoglows_river_id(rid):
            raise ValueError(
                "`fetch_historical` for 'geoglows' requires a 9-digit `reach_id`."
            )
        if zarr_path is not None:
            records = extract_geoglows_zarr_series(zarr_path, rid)
            idx = pd.to_datetime([r['time'] for r in records])
            vals = [float(r['discharge']) for r in records]
            return pd.Series(vals, index=idx, name='discharge_m3s', dtype=float)
        client = GeoGLOWSClient(
            base_url=config.geoglows_base_url,
            timeout_s=max(config.http_timeout_s, 30.0),
        )
        url = f'{client.base_url}/retrospectivedaily/{rid}'
        resp = client.session.get(
            url, params={'format': 'json'}, timeout=client.timeout_s
        )
        resp.raise_for_status()
        payload = resp.json()
        times = (
            payload.get('datetime') or []
            if isinstance(payload, Mapping)
            else []
        )
        raw_vals = (
            payload.get(str(rid)) if isinstance(payload, Mapping) else None
        )
        if not isinstance(raw_vals, list) and isinstance(payload, Mapping):
            raw_vals = next(
                (
                    v
                    for k, v in payload.items()
                    if k not in ('datetime', 'metadata') and isinstance(v, list)
                ),
                [],
            )
        idx = pd.to_datetime([str(t)[:10] for t in times])
        vals = [
            v if (v := parse_finite_float(x)) is not None else float('nan')
            for x in (raw_vals or [])
        ]
        return pd.Series(vals, index=idx, name='discharge_m3s', dtype=float)

    raise ValueError(
        f"Unsupported historical provider {provider!r}; expected 'glofas' or 'geoglows'."
    )


def fetch_return_periods(  # noqa: PLR0913
    config: MaaSConfig,
    provider: str,
    *,
    reach_id: int | str | None = None,
    historical_series: pd.Series | Sequence[float] | None = None,
    lat: float | None = None,
    lon: float | None = None,
    method: str = 'gema',
    return_periods: Sequence[float] = CANONICAL_RETURN_PERIODS,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    """Compute or fetch return-period discharge thresholds (`[2, 5, 10, 20, 50, 100]` yr).

    When `historical_series` is supplied directly, computes thresholds locally
    via `maas.thresholds` (`'gema'` for USGS Bulletin 17C EMA + MGBT,
    `'weibull'` for empirical Weibull log-linear, or `'gumbel'` for EV1).
    Otherwise queries and caches the provider's reanalysis/retrospective record.
    """
    method_key = method.strip().lower()
    if historical_series is not None:
        is_hydro = isinstance(historical_series, pd.Series) and isinstance(
            historical_series.index, pd.DatetimeIndex
        )
        if method_key == 'gema':
            return compute_return_periods(
                historical_series,
                return_periods=return_periods,
                is_daily_hydrograph=is_hydro,
            )
        if method_key == 'weibull':
            return compute_empirical_weibull_return_periods(
                historical_series,
                return_periods=return_periods,
                is_daily_hydrograph=is_hydro,
            )
        if method_key in ('gumbel', 'ev1'):
            if is_hydro:
                assert isinstance(historical_series, pd.Series)
                times = [
                    t.strftime('%Y-%m-%d') for t in historical_series.index
                ]
                peaks = extract_annual_maxima(
                    times, historical_series.to_numpy(dtype=float)
                )
            else:
                peaks = [
                    float(x)
                    for x in historical_series
                    if parse_finite_float(x) is not None
                ]
            rps = compute_gumbel_return_periods(
                peaks,
                return_periods=[int(t) for t in return_periods],
            )
            if rps is None:
                return None
            return {
                **rps,
                'years_of_record': len(peaks),
                'method': 'EV1 (Gumbel, method of moments)',
                'unit': 'm³/s',
            }
        raise ValueError(
            f"Unsupported return period method {method!r}; expected 'gema', 'weibull', or 'gumbel'."
        )

    cache = SQLiteCache(
        config.cache_dir / 'maas_flood_cache.sqlite', table_name='flood_cache'
    )
    prov = provider.strip().lower()
    if prov == 'glofas':
        if lat is None or lon is None:
            raise ValueError(
                "`fetch_return_periods` for 'glofas' requires `lat` and `lon`."
            )
        cell_lat, cell_lon = glofas_cell_center(lat, lon)
        rp_key = f'glofas_rp_{method_key}_{cell_lat:.3f}_{cell_lon:.3f}'
        if use_cache:
            cached = cache.get(rp_key, max_age_s=90 * 86400)
            if cached is not None:
                return cached
        client = GloFASClient(
            base_url=config.glofas_base_url,
            timeout_s=config.http_timeout_s,
        )
        rp = client.fetch_reanalysis_return_periods(lat, lon, method=method_key)
        if rp is not None and use_cache:
            cache.put(rp_key, rp)
        return rp

    if prov == 'geoglows':
        rid = parse_int(reach_id)
        if rid is None or not is_geoglows_river_id(rid):
            raise ValueError(
                "`fetch_return_periods` for 'geoglows' requires a 9-digit `reach_id`."
            )
        rp_key = f'geoglows_rp_{rid}'
        if use_cache:
            cached = cache.get(rp_key, max_age_s=180 * 86400)
            if cached is not None:
                return cached
        client = GeoGLOWSClient(
            base_url=config.geoglows_base_url,
            timeout_s=config.http_timeout_s,
        )
        rp = client.fetch_official_return_periods(rid)
        if rp is None:
            rp = client.fetch_retrospective_return_periods(
                rid, method=method_key
            )
        if rp is not None and use_cache:
            cache.put(rp_key, rp)
        return rp

    raise ValueError(
        f"Unsupported return-period provider {provider!r}; expected 'glofas' or 'geoglows'."
    )


def fetch_forecasts(  # noqa: PLR0913
    config: MaaSConfig,
    lat: float,
    lon: float,
    *,
    models: Sequence[str] | None = None,
    requested_models: Sequence[str] | None = None,
    gauge_id: str | None = None,
    river_id: int | None = None,
    reach_id: str | None = None,
    forecast_days: int = 15,
) -> dict[str, Any]:
    """Fetch and time-align multi-model discharge forecasts at `(lat, lon)`.

    Returns a pure data dictionary with per-provider forecast payloads, daily
    mean series (`daily`), peak discharge (`peaks`), date-aligned series
    (`aligned_dates`, `aligned_series`), and a tidy `pd.DataFrame` (`dataframe`).
    """
    req_models = normalize_requested_models(
        models if models is not None else requested_models
    )
    models_out: dict[str, dict[str, Any]] = {}
    daily_central: dict[str, dict[str, float]] = {}
    peaks: dict[str, dict[str, Any]] = {}

    if 'floodhub' in req_models:
        if gauge_id and config.floodhub_api_key.strip():
            fh_client = FloodHubClient(
                api_key=config.floodhub_api_key,
                base_url=config.floodhub_base_url,
                timeout_s=config.http_timeout_s,
            )
            fh_fc = fh_client.fetch_forecast(gauge_id)
        else:
            fh_fc = {
                'model': 'google_floodhub',
                'available': False,
                'status': 'unavailable',
                'gauge_id': gauge_id,
                'data': [],
            }
        models_out['floodhub'] = fh_fc
        if fh_fc.get('data'):
            d_series = daily_series(fh_fc.get('data'), 'discharge')
            daily_central['floodhub'] = d_series
            pk_val, pk_date = window_peak(d_series)
            peaks['floodhub'] = {'peak_flow': pk_val, 'peak_date': pk_date}

    if 'glofas' in req_models:
        gl_client = GloFASClient(
            base_url=config.glofas_base_url,
            timeout_s=config.http_timeout_s,
        )
        gl_fc = gl_client.fetch_forecast(lat, lon, forecast_days=forecast_days)
        models_out['glofas'] = gl_fc
        if gl_fc.get('data'):
            d_series = daily_series(
                gl_fc.get('data'), 'discharge_median', 'discharge_mean'
            )
            daily_central['glofas'] = d_series
            pk_val, pk_date = window_peak(d_series)
            peaks['glofas'] = {'peak_flow': pk_val, 'peak_date': pk_date}

    if 'geoglows' in req_models:
        gg_client = GeoGLOWSClient(
            base_url=config.geoglows_base_url,
            timeout_s=config.http_timeout_s,
        )
        eff_rid = (
            int(river_id)
            if is_geoglows_river_id(river_id)
            else gg_client.fetch_river_id(lat, lon)
        )
        if eff_rid is not None:
            gg_fc = gg_client.fetch_forecast(eff_rid)
        else:
            gg_fc = {
                'model': 'geoglows',
                'available': False,
                'status': 'unavailable',
                'river_id': None,
                'data': [],
            }
        models_out['geoglows'] = gg_fc
        if gg_fc.get('data'):
            d_series = daily_series(gg_fc.get('data'), 'flow_med', 'flow_avg')
            daily_central['geoglows'] = d_series
            pk_val, pk_date = window_peak(d_series)
            peaks['geoglows'] = {'peak_flow': pk_val, 'peak_date': pk_date}

    dates = sorted({d for col in daily_central.values() for d in col})
    aligned = {
        model: align_daily_series(col, dates)
        for model, col in daily_central.items()
    }
    df = (
        pd.DataFrame(aligned, index=pd.to_datetime(dates))
        if dates
        else pd.DataFrame()
    )
    return {
        'location': {
            'lat': lat,
            'lon': lon,
            'gauge_id': gauge_id,
            'river_id': river_id,
            'reach_id': reach_id,
        },
        'models': models_out,
        'daily': daily_central,
        'peaks': peaks,
        'aligned_dates': dates,
        'aligned_series': aligned,
        'dataframe': df,
    }


class MaaSDataFetcher:
    """Pure backend data fetcher for multi-model flood forecasts and return periods."""

    def __init__(self, config: MaaSConfig) -> None:
        """Initialize provider clients and SQLite caches from an explicit `MaaSConfig`."""
        if not isinstance(config, MaaSConfig):
            raise TypeError('MaaSDataFetcher requires an explicit `MaaSConfig`.')
        self.config = config
        self.floodhub = FloodHubClient(
            api_key=config.floodhub_api_key,
            base_url=config.floodhub_base_url,
            timeout_s=config.http_timeout_s,
            cache_dir=config.cache_dir,
        )
        self.glofas = GloFASClient(
            base_url=config.glofas_base_url,
            timeout_s=config.http_timeout_s,
        )
        self.geoglows = GeoGLOWSClient(
            base_url=config.geoglows_base_url,
            timeout_s=config.http_timeout_s,
        )
        self.flood_cache = SQLiteCache(
            config.cache_dir / 'maas_flood_cache.sqlite',
            table_name='flood_cache',
        )
        self.watershed_cache = SQLiteCache(
            config.cache_dir / 'maas_watershed_cache.sqlite',
            table_name='watershed_cache',
        )

    def resolve_reaches(
        self,
        lat: float,
        lon: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolve and snap `(lat, lon)` across all three model river networks."""
        return resolve_reaches(lat, lon, self.config, **kwargs)

    def fetch_gauges(
        self,
        bbox: tuple[float, float, float, float],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fetch FloodHub gauge stations within `bbox`."""
        return fetch_gauges(self.config, bbox, **kwargs)

    def fetch_historical(
        self,
        provider: str,
        **kwargs: Any,
    ) -> pd.Series:
        """Fetch historical daily discharge `pd.Series` for `provider`."""
        return fetch_historical(self.config, provider, **kwargs)

    def fetch_return_periods(
        self,
        provider: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Compute or fetch `[2, 5, 10, 20, 50, 100]`-yr return period thresholds."""
        return fetch_return_periods(self.config, provider, **kwargs)

    def fetch_forecasts(  # noqa: PLR0913
        self,
        lat: float,
        lon: float,
        gauge_id: str | None = None,
        river_id: int | None = None,
        reach_id: str | None = None,
        requested_models: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        *,
        upstream_area_km2: float | str | None = None,
        area_min_km2: float | str | None = None,
        network: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a pure multi-model data bundle (forecasts, return periods, and reaches)."""
        t0 = time.time()
        req_models = normalize_requested_models(
            requested_models if requested_models is not None else models
        )

        reaches = self.resolve_reaches(
            lat,
            lon,
            upstream_area_km2=upstream_area_km2,
            area_min_km2=area_min_km2,
            network=network,
            river_id=river_id or reach_id,
            gauge_id=gauge_id,
        )
        gl_snap = reaches['glofas'].get('snap')
        gl_query_lat = (
            float(gl_snap.get('query_lat', gl_snap['lat']))
            if gl_snap
            else lat
        )
        gl_query_lon = (
            float(gl_snap.get('query_lon', gl_snap['lon']))
            if gl_snap
            else lon
        )
        fh_lat = float(gl_snap['lat']) if gl_snap else lat
        fh_lon = float(gl_snap['lon']) if gl_snap else lon
        eff_river_id = reaches['geoglows'].get('river_id') or (
            parse_int(river_id) if is_geoglows_river_id(river_id) else None
        )
        eff_gauge_id = gauge_id

        def _task_floodhub() -> tuple[dict[str, Any] | None, str | None]:
            if (
                'floodhub' not in req_models
                or not self.config.floodhub_api_key.strip()
            ):
                return None, eff_gauge_id
            gid = eff_gauge_id
            dist_km: float | None = None
            gauge_meta: dict[str, Any] | None = None
            if not gid:
                target_area = reaches['probe'].get('upstream_area_km2')
                try:
                    gid, gauge_meta, dist_km = self.floodhub.find_nearest_gauge(
                        fh_lat, fh_lon, target_area_km2=target_area
                    )
                except TypeError:
                    gid, gauge_meta, dist_km = self.floodhub.find_nearest_gauge(
                        fh_lat, fh_lon
                    )
                except Exception:  # noqa: BLE001
                    gid = None
            if not gid:
                return None, None
            is_live_fh = not hasattr(self.floodhub.fetch_forecast, '_mock_name')
            fh_cache_key = f'floodhub_fc_{gid}'
            if is_live_fh:
                cached_fh = self.flood_cache.get(fh_cache_key, max_age_s=1800)
                if isinstance(cached_fh, dict) and cached_fh.get('status') == 'live':
                    return cached_fh, gid
            try:
                fc = self.floodhub.fetch_forecast(gid)
            except Exception:  # noqa: BLE001
                fc = {
                    'model': 'google_floodhub',
                    'available': False,
                    'status': 'unavailable',
                    'gauge_id': gid,
                    'data': [],
                }
            enriched = self.floodhub.enrich_forecast_status(
                gid,
                fc,
                lat=fh_lat,
                lon=fh_lon,
                gauge_meta=gauge_meta,
                dist_km=dist_km,
            )
            if is_live_fh and enriched.get('status') == 'live':
                self.flood_cache.put(fh_cache_key, enriched)
            return enriched, gid

        def _task_glofas() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            if 'glofas' not in req_models:
                return None, None
            cell_lat, cell_lon = glofas_cell_center(gl_query_lat, gl_query_lon)
            rp_key = f'glofas_rp_{cell_lat:.3f}_{cell_lon:.3f}'
            fc_key = f'glofas_fc_{cell_lat:.3f}_{cell_lon:.3f}'
            is_live_gl = not hasattr(self.glofas.fetch_forecast, '_mock_name')
            gl_rp = self.flood_cache.get(rp_key, max_age_s=90 * 86400)
            gl_fc_cached = (
                self.flood_cache.get(fc_key, max_age_s=1800)
                if is_live_gl
                else None
            )

            def _fetch_gl_fc() -> dict[str, Any]:
                if (
                    isinstance(gl_fc_cached, dict)
                    and gl_fc_cached.get('status') == 'live'
                ):
                    return gl_fc_cached
                try:
                    res_fc = self.glofas.fetch_forecast(
                        gl_query_lat, gl_query_lon, forecast_days=15
                    )
                    if is_live_gl and res_fc.get('status') == 'live':
                        self.flood_cache.put(fc_key, res_fc)
                    return res_fc
                except Exception:  # noqa: BLE001
                    return {
                        'model': 'copernicus_glofas',
                        'available': False,
                        'status': 'unavailable',
                        'lat': gl_query_lat,
                        'lon': gl_query_lon,
                        'data': [],
                    }

            def _fetch_gl_rp() -> dict[str, Any] | None:
                try:
                    res_rp = self.glofas.fetch_reanalysis_return_periods(
                        gl_query_lat, gl_query_lon
                    )
                    if res_rp is not None and res_rp.get('source') != 'unit_test_rp':
                        self.flood_cache.put(rp_key, res_rp)
                    return res_rp
                except Exception:  # noqa: BLE001
                    return None

            if gl_rp is not None:
                gl_fc = _fetch_gl_fc()
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=2
                ) as sub_ex:
                    fut_fc = sub_ex.submit(_fetch_gl_fc)
                    fut_rp = sub_ex.submit(_fetch_gl_rp)
                    gl_fc = fut_fc.result()
                    gl_rp = fut_rp.result()
            return gl_fc, gl_rp

        def _task_geoglows() -> tuple[
            dict[str, Any] | None, dict[str, Any] | None, int | None
        ]:
            if 'geoglows' not in req_models:
                return None, None, eff_river_id
            rid = eff_river_id
            if not is_geoglows_river_id(rid):
                try:
                    rid = self.geoglows.fetch_river_id(lat, lon)
                except Exception:  # noqa: BLE001
                    rid = None
            if rid is None:
                return (
                    {
                        'model': 'geoglows',
                        'available': False,
                        'status': 'unavailable',
                        'river_id': None,
                        'data': [],
                    },
                    None,
                    None,
                )
            rp_key = f'geoglows_rp_{rid}'
            fc_key = f'geoglows_fc_{rid}'
            is_live_gg = not hasattr(self.geoglows.fetch_forecast, '_mock_name')
            gg_r = self.flood_cache.get(rp_key, max_age_s=180 * 86400)
            gg_f_cached = (
                self.flood_cache.get(fc_key, max_age_s=1800)
                if is_live_gg
                else None
            )

            def _fetch_gg_fc() -> dict[str, Any]:
                if (
                    isinstance(gg_f_cached, dict)
                    and gg_f_cached.get('status') == 'live'
                ):
                    return gg_f_cached
                try:
                    res_fc = self.geoglows.fetch_forecast(rid)
                    if is_live_gg and res_fc.get('status') == 'live':
                        self.flood_cache.put(fc_key, res_fc)
                    return res_fc
                except Exception:  # noqa: BLE001
                    return {
                        'model': 'geoglows',
                        'available': False,
                        'status': 'unavailable',
                        'river_id': rid,
                        'data': [],
                    }

            def _fetch_gg_rp() -> dict[str, Any] | None:
                res_rp: dict[str, Any] | None = None
                try:
                    res_rp = self.geoglows.fetch_return_periods(rid)
                except Exception:  # noqa: BLE001
                    res_rp = None
                if res_rp is None:
                    try:
                        res_rp = (
                            self.geoglows.fetch_retrospective_return_periods(
                                rid
                            )
                        )
                    except Exception:  # noqa: BLE001
                        res_rp = None
                if res_rp is not None:
                    self.flood_cache.put(rp_key, res_rp)
                return res_rp

            if gg_r is not None:
                gg_f = _fetch_gg_fc()
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=2
                ) as sub_ex:
                    fut_fc = sub_ex.submit(_fetch_gg_fc)
                    fut_rp = sub_ex.submit(_fetch_gg_rp)
                    gg_f = fut_fc.result()
                    gg_r = fut_rp.result()
            return gg_f, gg_r, rid

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f_fh = ex.submit(_task_floodhub)
            f_gl = ex.submit(_task_glofas)
            f_gg = ex.submit(_task_geoglows)
            fh_fc, eff_gauge_id = f_fh.result()
            gl, glrp = f_gl.result()
            gg_fc, gg_rp, eff_river_id = f_gg.result()

        models_output: dict[str, Any] = {}
        if 'floodhub' in req_models:
            models_output['floodhub'] = fh_fc or {
                'available': False,
                'status': 'unavailable',
                'message': 'No FloodHub gauge specified or available.',
            }
        if 'geoglows' in req_models:
            models_output['geoglows'] = (
                {**gg_fc, 'return_periods': gg_rp}
                if gg_fc
                else {
                    'model': 'geoglows',
                    'available': False,
                    'status': 'unavailable',
                    'river_id': eff_river_id,
                    'data': [],
                    'return_periods': None,
                }
            )
        if 'glofas' in req_models and gl:
            models_output['glofas'] = {**gl, 'return_periods': glrp}

        fh_th = (fh_fc or {}).get('thresholds') or {}
        fh_is_q = (
            str((fh_fc or {}).get('unit') or '').upper()
            == 'CUBIC_METERS_PER_SECOND'
        )
        if (
            (fh_fc or {}).get('status') == 'live'
            and fh_is_q
            and fh_th.get('warning_2yr') is not None
        ):
            thresholds = {
                'warning_2yr': fh_th.get('warning_2yr'),
                'danger_5yr': fh_th.get('danger_5yr'),
                'extreme_20yr': fh_th.get('extreme_20yr'),
                'extreme_100yr': None,
                'source': 'Google FloodHub gauge model thresholds',
                'unit': 'm³/s',
            }
        elif glrp:
            thresholds = thresholds_from_return_periods(
                glrp,
                str(glrp.get('source') or 'Copernicus GloFAS v4 reanalysis'),
            )
        elif gg_rp:
            thresholds = thresholds_from_return_periods(
                gg_rp, str(gg_rp.get('source') or 'GEOGLOWS v2')
            )
        else:
            thresholds = {
                'warning_2yr': None,
                'danger_5yr': None,
                'extreme_20yr': None,
                'extreme_100yr': None,
                'source': None,
                'unit': 'm³/s',
            }

        reaches['floodhub'] = {
            **reaches['floodhub'],
            'gauge_id': eff_gauge_id,
            'lat': ((fh_fc or {}).get('gauge_location') or {}).get('lat'),
            'lon': ((fh_fc or {}).get('gauge_location') or {}).get('lon'),
            'distance_km': (fh_fc or {}).get('distance_km'),
            'status': (fh_fc or {}).get('status'),
        }
        reaches['geoglows'] = {
            **reaches['geoglows'],
            'river_id': eff_river_id,
            'status': (gg_fc or {}).get('status'),
        }
        reaches['glofas'] = {
            **reaches['glofas'],
            'status': (gl or {}).get('status'),
        }
        virtual_station = {
            'probe': reaches['probe'],
            'floodhub_gauge': reaches['floodhub'],
            'glofas_cell': reaches['glofas'],
            'geoglows_reach': reaches['geoglows'],
            'hydrorivers_reach': reach_id,
        }

        return {
            'location': {
                'lat': lat,
                'lon': lon,
                'gauge_id': eff_gauge_id,
                'river_id': eff_river_id,
                'reach_id': reach_id,
            },
            'reaches': reaches,
            'virtual_station': virtual_station,
            'thresholds': thresholds,
            'return_periods': {'glofas': glrp, 'geoglows': gg_rp},
            'models': models_output,
            'meta': {
                'models_requested': req_models,
                'generated_at': datetime.now(UTC).strftime(
                    '%Y-%m-%dT%H:%M:%SZ'
                ),
                'elapsed_s': round(time.time() - t0, 2),
            },
        }

    fetch_unified_forecast = fetch_forecasts


# Backwards-compatible alias for callers importing `MaaSEngine` from `maas`
MaaSEngine = MaaSDataFetcher

__all__ = [
    'MaaSDataFetcher',
    'MaaSEngine',
    'RETURN_PERIOD_YEARS',
    'SQLiteCache',
    'align_daily_series',
    'daily_series',
    'fetch_forecasts',
    'fetch_gauges',
    'fetch_historical',
    'fetch_return_periods',
    'resolve_reaches',
    'window_peak',
]
