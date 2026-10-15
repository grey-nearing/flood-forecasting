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

import concurrent.futures
import json
import math
import re
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, date, datetime
from numbers import Real
from pathlib import Path
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
    DEFAULT_GLOFAS_REFERENCE,
    GloFASClient,
    describe_glofas_reference,
    extract_glofas_zarr_series,
    glofas_reference_field,
)
from maas.networks import (
    FLOODHUB_PYRAMID_LOD,
    GEOGLOWS_LOD,
    GLOFAS_LOD,
    glofas_cell_center,
    is_geoglows_river_id,
    load_geoglows_lookup,
    load_network_pyramid,
    pyramid_signature,
    resolve_cross_network_click,
    snap_floodhub_reach_from_network,
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

_ISO_DATE_LEN: int = 10
_ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_SQLITE_TIMEOUT_S: float = 5.0


class SQLiteCache:
    """Thread-safe persistent SQLite key-value JSON cache (explicit `db_path`).

    `table_name` is validated as a Python identifier in `__init__`, so the
    SQL statements below can only ever interpolate a plain identifier.
    """

    def __init__(
        self, db_path: Path | str, table_name: str = 'flood_cache'
    ) -> None:
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
            with closing(
                sqlite3.connect(str(self.db_path), timeout=_SQLITE_TIMEOUT_S)
            ) as conn:
                conn.execute(
                    f'CREATE TABLE IF NOT EXISTS {self.table_name} ('
                    'cache_key TEXT PRIMARY KEY, payload TEXT, created_at REAL)'
                )
                conn.commit()
            self._initialized = True

    def get(self, cache_key: str, max_age_s: float) -> dict[str, Any] | None:
        """Read a JSON payload if the cache file exists and is fresh enough.

        Returns `None` when `db_path` does not exist, the key is missing, or
        the entry is older than `max_age_s` seconds.
        """
        if not self.db_path.exists():
            return None
        self._ensure_initialized()
        with closing(
            sqlite3.connect(str(self.db_path), timeout=_SQLITE_TIMEOUT_S)
        ) as conn:
            row = conn.execute(
                f'SELECT payload, created_at FROM {self.table_name} '  # noqa: S608
                'WHERE cache_key = ?',
                (cache_key,),
            ).fetchone()
        if row and row[0] and (time.time() - float(row[1] or 0.0)) <= max_age_s:
            payload = json.loads(row[0])
            return payload if isinstance(payload, dict) else None
        return None

    def put(self, cache_key: str, payload: Mapping[str, Any]) -> None:
        """Persist a JSON-serializable payload in the SQLite cache."""
        self._ensure_initialized()
        with closing(
            sqlite3.connect(str(self.db_path), timeout=_SQLITE_TIMEOUT_S)
        ) as conn:
            conn.execute(
                f'INSERT OR REPLACE INTO {self.table_name} '  # noqa: S608
                '(cache_key, payload, created_at) VALUES (?, ?, ?)',
                (cache_key, json.dumps(dict(payload)), time.time()),
            )
            conn.commit()


def _record_day(record: Mapping[str, Any]) -> str:
    """Return the `YYYY-MM-DD` day of a record's `time` field, validated."""
    if 'time' not in record:
        raise KeyError(f'Forecast record is missing a `time` field: {record!r}')
    raw_time = record['time']
    if not isinstance(raw_time, str):
        raise TypeError(
            f'Forecast record `time` must be an ISO-8601 string, '
            f'got {raw_time!r}'
        )
    text = raw_time.strip()
    day = text[:_ISO_DATE_LEN]
    if not _ISO_DATE_RE.match(day) or (
        len(text) > _ISO_DATE_LEN and text[_ISO_DATE_LEN] not in ('T', ' ')
    ):
        raise ValueError(
            f'Forecast record `time` is not an ISO-8601 date/timestamp: '
            f'{raw_time!r}'
        )
    # Rejects impossible calendar dates (e.g. month 13) with a ValueError.
    date.fromisoformat(day)
    return day


def _record_value(record: Mapping[str, Any], key: str) -> float | None:
    """Return the finite float at `record[key]`, or `None` when it is missing.

    `None`, `NaN`, and `±Inf` all mean "no value" and yield `None`. Any
    non-numeric type (including `bool` and numeric strings) raises
    `TypeError`, because provider parsers always emit numbers or `None`.
    """
    if key not in record:
        raise KeyError(f'Forecast record is missing field {key!r}: {record!r}')
    raw = record[key]
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise TypeError(
            f'Forecast record field {key!r} must be numeric or None, '
            f'got {raw!r}'
        )
    value = float(raw)
    return value if math.isfinite(value) else None


def daily_series(
    records: Sequence[Mapping[str, Any]] | None,
    key: str,
) -> dict[str, float]:
    """Aggregate (sub-)daily records into daily means `{YYYY-MM-DD: value}`.

    Each record must carry an ISO-8601 `time` (`YYYY-MM-DD` or a UTC
    timestamp starting with that date) and a numeric-or-`None` field `key`.
    Missing values (`None`/`NaN`) are skipped, never imputed; days with no
    finite value are absent from the result.

    Raises:
        KeyError: If a record lacks `time` or `key`.
        TypeError: If `time` is not a string or the value is non-numeric.
        ValueError: If `time` is not a valid ISO-8601 date/timestamp.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records or ():
        if not isinstance(record, Mapping):
            raise TypeError(
                f'Expected a mapping record in forecast data, got {record!r}'
            )
        day = _record_day(record)
        value = _record_value(record, key)
        if value is None:
            continue
        sums[day] = sums.get(day, 0.0) + value
        counts[day] = counts.get(day, 0) + 1
    return {t: sums[t] / counts[t] for t in sorted(sums)}


def align_daily_series(
    daily: Mapping[str, float],
    dates: Sequence[str],
    ndigits: int = 2,
) -> list[float | None]:
    """Align a `{timestamp_or_date: value}` mapping onto a common `dates` list.

    Entries missing from `daily` become `None` (never filled).
    """
    return [round(daily[d], ndigits) if d in daily else None for d in dates]


def window_peak(
    daily: Mapping[str, float],
    today_str: str | None = None,
) -> tuple[float | None, str | None]:
    """Return the peak `(value, date)` on or after `today_str`.

    Falls back to the full series only when no entry is on or after
    `today_str` (defaults to today, UTC). Returns `(None, None)` for an
    empty series.
    """
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


def _get_cached_floodhub_pyramid(floodhub_npz: Path) -> dict[str, Any] | None:
    key = str(floodhub_npz)
    if key in _PYRAMID_MEM_CACHE:
        return _PYRAMID_MEM_CACHE[key]
    with _MEM_CACHE_LOCK:
        if key in _PYRAMID_MEM_CACHE:
            return _PYRAMID_MEM_CACHE[key]
        net = load_network_pyramid(
            floodhub_npz, pyramid_signature(FLOODHUB_PYRAMID_LOD)
        )
        if net is not None:
            _PYRAMID_MEM_CACHE[key] = net
        return net


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


def resolve_reaches(  # noqa: C901, PLR0913
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
    r"""Resolve and snap a coordinate across all three model river networks.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        config: Explicit `MaaSConfig` providing `river_networks_dir` and
            `cache_dir`.
        upstream_area_km2: Optional target upstream drainage area in
            $\text{km}^2$ for cross-network reach matching.
        area_min_km2: Optional minimum upstream drainage area along a merged
            river line.
        network: Optional source network identifier (`'floodhub'`,
            `'glofas'`, `'geoglows'`).
        river_id: Optional GEOGLOWS 9-digit `LINKNO` or HydroRIVERS
            `HYRIV_ID`.
        gauge_id: Optional FloodHub gauge identifier.

    Returns:
        Dictionary containing resolved grid cells and reach identifiers for
        `floodhub`, `glofas`, and `geoglows`.
    """
    gl_lat, gl_lon = glofas_cell_center(lat, lon)
    eff_river_id = (
        parse_int(river_id) if is_geoglows_river_id(river_id) else None
    )
    up_area = parse_finite_float(upstream_area_km2)
    min_area = parse_finite_float(area_min_km2)

    glofas_npz = (
        config.river_networks_dir / 'glofas_v4' / 'glofas_network_v1.npz'
    )
    if not glofas_npz.exists():
        glofas_npz = config.cache_dir / 'glofas_v4' / 'glofas_network_v1.npz'
    glofas_net = _get_cached_glofas_pyramid(glofas_npz)

    floodhub_npz = (
        config.river_networks_dir / 'glofas_v4' / 'floodhub_network_v1.npz'
    )
    if not floodhub_npz.exists():
        floodhub_npz = (
            config.cache_dir / 'glofas_v4' / 'floodhub_network_v1.npz'
        )
    floodhub_net = _get_cached_floodhub_pyramid(floodhub_npz)

    geoglows_dir = config.river_networks_dir / 'geoglows_v2'
    if (
        not (geoglows_dir / 'global_streams_simplified.gpkg').exists()
        and not (geoglows_dir / 'geoglows_network_v1.npz').exists()
    ):
        geoglows_dir = config.cache_dir / 'geoglows_v2'
    geoglows_npz = geoglows_dir / 'geoglows_network_v1.npz'
    geoglows_net = _get_cached_geoglows_pyramid(geoglows_npz)
    gpkg_path = geoglows_dir / 'global_streams_simplified.gpkg'
    attrs_path = geoglows_dir / 'geoglows_attrs_v1.npz'

    if up_area is None or up_area <= 0.0:
        # Infer the upstream area from the clicked network first, then from
        # the remaining networks in a fixed order.
        area_snappers = {
            'floodhub': lambda: snap_floodhub_reach_from_network(
                floodhub_net, lat, lon, None
            ),
            'geoglows': lambda: snap_geoglows_reach_from_network(
                geoglows_net, lat, lon, None
            ),
            'glofas': lambda: snap_glofas_cell_from_network(
                glofas_net, lat, lon, None
            ),
        }
        first = network if network in area_snappers else 'glofas'
        order = [first] + [k for k in area_snappers if k != first]
        for net_key in order:
            inferred = area_snappers[net_key]()
            if inferred and parse_finite_float(
                inferred.get('upstream_area_km2')
            ):
                up_area = float(inferred['upstream_area_km2'])
                break

    cross_snap: dict[str, Any] | None = None
    if up_area is not None and up_area > 0.0:

        def _snap_geoglows(
            qlat: float,
            qlon: float,
            area: float,
            **kw: Any,  # noqa: ANN401
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
                snap_glofas_cell_from_network(
                    glofas_net, qlat, qlon, area, **kw
                )
            ),
            snap_geoglows_fn=_snap_geoglows,
            area_min_km2=min_area,
            network=network,
            river_id=river_id,
            snap_floodhub_fn=lambda qlat, qlon, area, **kw: (
                snap_floodhub_reach_from_network(
                    floodhub_net, qlat, qlon, area, **kw
                )
            ),
            gauge_id=gauge_id,
        )
        if cross_snap:
            if cross_snap.get('glofas'):
                gl_lat = float(cross_snap['glofas']['lat'])
                gl_lon = float(cross_snap['glofas']['lon'])
            if cross_snap.get('geoglows'):
                eff_river_id = int(cross_snap['geoglows']['river_id'])

    fh_snap = (cross_snap or {}).get('floodhub') or {}
    is_merged_fh_line = (
        network == 'floodhub'
        and min_area is not None
        and up_area is not None
        and min_area < up_area
    )
    resolved_gauge_id = (
        fh_snap.get('gauge_id')
        if (not gauge_id or is_merged_fh_line) and fh_snap.get('gauge_id')
        else gauge_id
    )
    resolved_fh_reach_id = (
        fh_snap.get('river_id')
        if (not river_id or is_geoglows_river_id(river_id) or is_merged_fh_line)
        and fh_snap.get('river_id')
        else (
            str(river_id)
            if river_id and not is_geoglows_river_id(river_id)
            else None
        )
    )
    gg_snap = (cross_snap or {}).get('geoglows') or {}

    return {
        'probe': {
            'lat': lat,
            'lon': lon,
            'network': network,
            'upstream_area_km2': (cross_snap or {}).get('target_area_km2')
            or up_area,
        },
        'floodhub': {
            'gauge_id': resolved_gauge_id,
            'reach_id': resolved_fh_reach_id,
            'lat': fh_snap.get('lat'),
            'lon': fh_snap.get('lon'),
            'upstream_area_km2': fh_snap.get('upstream_area_km2'),
            'offset_km': fh_snap.get('offset_km'),
            'snap': cross_snap.get('floodhub') if cross_snap else None,
        },
        'glofas': {
            'cell_center_lat': gl_lat,
            'cell_center_lon': gl_lon,
            'resolution_deg': GLOFAS_RES_DEG,
            'upstream_area_km2': ((cross_snap or {}).get('glofas') or {}).get(
                'upstream_area_km2'
            ),
            'offset_cells': ((cross_snap or {}).get('glofas') or {}).get(
                'offset_cells'
            ),
            'snap': cross_snap.get('glofas') if cross_snap else None,
        },
        'geoglows': {
            'river_id': eff_river_id,
            'lat': gg_snap.get('lat'),
            'lon': gg_snap.get('lon'),
            'upstream_area_km2': gg_snap.get('upstream_area_km2'),
            'offset_km': gg_snap.get('offset_km'),
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
    """Fetch FloodHub gauge metadata inside a bounding box.

    `bbox` is `(min_lat, min_lon, max_lat, max_lon)` in decimal degrees.
    """
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


def _records_to_series(records: Sequence[Mapping[str, Any]]) -> pd.Series:
    """Build a daily `discharge_m3s` series from `time`/`discharge` rows."""
    idx = pd.to_datetime([r['time'] for r in records])
    vals = [float(r['discharge']) for r in records]
    return pd.Series(vals, index=idx, name='discharge_m3s', dtype=float)


def _glofas_historical_series(  # noqa: PLR0913
    config: MaaSConfig,
    lat: float,
    lon: float,
    *,
    zarr_path: Path | None,
    end_year: int | None,
    step_index: int | None,
    member_index: int | None,
) -> pd.Series:
    """GloFAS v4 reanalysis series from a local archive or Open-Meteo."""
    if zarr_path is not None:
        return _records_to_series(
            extract_glofas_zarr_series(
                zarr_path,
                lat,
                lon,
                step_index=step_index,
                member_index=member_index,
            )
        )
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
        daily.get('river_discharge') or [] if isinstance(daily, Mapping) else []
    )
    idx = pd.to_datetime([str(t)[:_ISO_DATE_LEN] for t in times])
    vals = [
        v if (v := parse_finite_float(x)) is not None else float('nan')
        for x in values
    ]
    return pd.Series(vals, index=idx, name='discharge_m3s', dtype=float)


def _geoglows_historical_series(
    config: MaaSConfig,
    rid: int,
    *,
    zarr_path: Path | None,
) -> pd.Series:
    """GEOGLOWS v2 retrospective series from a local archive or the REST API."""
    if zarr_path is not None:
        return _records_to_series(extract_geoglows_zarr_series(zarr_path, rid))
    client = GeoGLOWSClient(
        base_url=config.geoglows_base_url,
        timeout_s=max(config.http_timeout_s, 30.0),
        cache_dir=config.cache_dir,
    )
    url = f'{client.base_url}/retrospectivedaily/{rid}'
    resp = client.session.get(
        url, params={'format': 'json'}, timeout=client.timeout_s
    )
    resp.raise_for_status()
    payload = resp.json()
    times = (
        payload.get('datetime') or [] if isinstance(payload, Mapping) else []
    )
    raw_vals = payload.get(str(rid)) if isinstance(payload, Mapping) else None
    if not isinstance(raw_vals, list) and isinstance(payload, Mapping):
        raw_vals = next(
            (
                v
                for k, v in payload.items()
                if k not in ('datetime', 'metadata') and isinstance(v, list)
            ),
            [],
        )
    idx = pd.to_datetime([str(t)[:_ISO_DATE_LEN] for t in times])
    vals = [
        v if (v := parse_finite_float(x)) is not None else float('nan')
        for x in (raw_vals or [])
    ]
    return pd.Series(vals, index=idx, name='discharge_m3s', dtype=float)


def fetch_historical(  # noqa: PLR0913
    config: MaaSConfig,
    provider: str,
    *,
    reach_id: int | str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    zarr_path: Path | None = None,
    end_year: int | None = None,
    step_index: int | None = None,
    member_index: int | None = None,
) -> pd.Series:
    r"""Fetch a historical daily discharge `pd.Series` (`pd.DatetimeIndex`).

    Args:
        config: Explicit `MaaSConfig`.
        provider: Either `'glofas'` or `'geoglows'`.
        reach_id: 9-digit GEOGLOWS `LINKNO` (required when
            `provider='geoglows'`).
        lat: Latitude in decimal degrees (required when `provider='glofas'`).
        lon: Longitude in decimal degrees (required when `provider='glofas'`).
        zarr_path: Optional local Zarr/NetCDF archive path.
        end_year: Optional last calendar year for GloFAS reanalysis queries.
        step_index: Positional lead-time index when a local GloFAS archive
            has a `step` dimension (see `extract_glofas_zarr_series`).
        member_index: Positional ensemble-member index when a local GloFAS
            archive has a `number` dimension.

    Returns:
        Daily streamflow `pd.Series` in $\text{m}^3/\text{s}$ indexed by
        `pd.DatetimeIndex`.
    """
    prov = provider.strip().lower()
    if prov == 'glofas':
        if lat is None or lon is None:
            raise ValueError(
                "`fetch_historical` for 'glofas' requires `lat` and `lon`."
            )
        return _glofas_historical_series(
            config,
            lat,
            lon,
            zarr_path=zarr_path,
            end_year=end_year,
            step_index=step_index,
            member_index=member_index,
        )

    if prov == 'geoglows':
        rid = parse_int(reach_id)
        if rid is None or not is_geoglows_river_id(rid):
            raise ValueError(
                "`fetch_historical` for 'geoglows' requires a 9-digit "
                '`reach_id`.'
            )
        return _geoglows_historical_series(config, rid, zarr_path=zarr_path)

    raise ValueError(
        f'Unsupported historical provider {provider!r}; '
        "expected 'glofas' or 'geoglows'."
    )


_GLOFAS_RP_CACHE_MAX_AGE_S: float = 90.0 * 86400.0
_GEOGLOWS_RP_CACHE_MAX_AGE_S: float = 180.0 * 86400.0


def _local_return_periods(
    historical_series: pd.Series | Sequence[float],
    method: str,
    return_periods: Sequence[float],
) -> dict[str, Any] | None:
    """Compute return periods locally from a supplied historical record."""
    method_key = method.strip().lower()
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
            times = [t.strftime('%Y-%m-%d') for t in historical_series.index]
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
        f'Unsupported return period method {method!r}; '
        "expected 'gema', 'weibull', or 'gumbel'."
    )


def fetch_return_periods(  # noqa: C901, PLR0913
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
    """Compute or fetch return-period discharge thresholds.

    Canonical return periods are `[2, 5, 10, 20, 50, 100]` years. When
    `historical_series` is supplied directly, computes thresholds locally via
    `maas.thresholds` (`'gema'` for USGS Bulletin 17C EMA + MGBT,
    `'weibull'` for empirical Weibull log-linear, or `'gumbel'` for EV1).
    Otherwise queries and caches the provider's reanalysis/retrospective
    record.
    """
    method_key = method.strip().lower()
    if historical_series is not None:
        return _local_return_periods(historical_series, method, return_periods)

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
            cached = cache.get(rp_key, max_age_s=_GLOFAS_RP_CACHE_MAX_AGE_S)
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
                "`fetch_return_periods` for 'geoglows' requires a 9-digit "
                '`reach_id`.'
            )
        rp_key = f'geoglows_rp_{rid}'
        if use_cache:
            cached = cache.get(rp_key, max_age_s=_GEOGLOWS_RP_CACHE_MAX_AGE_S)
            if cached is not None:
                return cached
        client = GeoGLOWSClient(
            base_url=config.geoglows_base_url,
            timeout_s=config.http_timeout_s,
            cache_dir=config.cache_dir,
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
        f'Unsupported return-period provider {provider!r}; '
        "expected 'glofas' or 'geoglows'."
    )


def _central_series_and_peak(
    forecast: Mapping[str, Any],
    key: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Daily-mean series of `forecast['data'][key]` and its forecast peak."""
    series = daily_series(forecast.get('data'), key)
    peak_value, peak_date = window_peak(series)
    return series, {'peak_flow': peak_value, 'peak_date': peak_date}


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
    glofas_reference: str = DEFAULT_GLOFAS_REFERENCE,
) -> dict[str, Any]:
    """Fetch and time-align multi-model discharge forecasts at `(lat, lon)`.

    Returns a pure data dictionary with per-provider forecast payloads, daily
    mean series (`daily`), peak discharge (`peaks`), date-aligned series
    (`aligned_dates`, `aligned_series`), a tidy `pd.DataFrame`
    (`dataframe`), and the GloFAS reference series descriptor
    (`glofas_reference`).

    Args:
        config: Explicit `MaaSConfig`.
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        models: Provider identifiers to query (alias of `requested_models`).
        requested_models: Provider identifiers to query.
        gauge_id: Optional FloodHub gauge identifier.
        river_id: Optional 9-digit GEOGLOWS `LINKNO`.
        reach_id: Optional HydroRIVERS reach identifier (echoed back).
        forecast_days: GloFAS forecast horizon in days.
        glofas_reference: Native GloFAS series used for the GloFAS daily
            series and peak: `'control'` (unperturbed control run),
            `'median'` (ensemble median), or `'mean'` (ensemble mean).
            GEOGLOWS always uses `flow_med`; FloodHub uses `discharge`.
    """
    req_models = normalize_requested_models(
        models if models is not None else requested_models
    )
    gl_reference = describe_glofas_reference(glofas_reference)
    gl_field = glofas_reference_field(glofas_reference)
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
            daily_central['floodhub'], peaks['floodhub'] = (
                _central_series_and_peak(fh_fc, 'discharge')
            )

    if 'glofas' in req_models:
        gl_client = GloFASClient(
            base_url=config.glofas_base_url,
            timeout_s=config.http_timeout_s,
        )
        gl_fc = gl_client.fetch_forecast(lat, lon, forecast_days=forecast_days)
        models_out['glofas'] = gl_fc
        if gl_fc.get('data'):
            daily_central['glofas'], peaks['glofas'] = _central_series_and_peak(
                gl_fc, gl_field
            )
            peaks['glofas']['reference'] = gl_reference['key']

    if 'geoglows' in req_models:
        gg_client = GeoGLOWSClient(
            base_url=config.geoglows_base_url,
            timeout_s=config.http_timeout_s,
            cache_dir=config.cache_dir,
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
            daily_central['geoglows'], peaks['geoglows'] = (
                _central_series_and_peak(gg_fc, 'flow_med')
            )

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
        'glofas_reference': gl_reference,
        'daily': daily_central,
        'peaks': peaks,
        'aligned_dates': dates,
        'aligned_series': aligned,
        'dataframe': df,
    }


class MaaSDataFetcher:
    """Pure backend fetcher for multi-model flood forecasts and return periods.

    The multi-model bundle returned by `fetch_forecasts` carries the raw
    provider payloads; the GloFAS payload includes all three native reference
    series per record (`discharge_deterministic`, `discharge_median`,
    `discharge_mean`) so callers choose the reference explicitly.
    """

    def __init__(self, config: MaaSConfig) -> None:
        """Initialize provider clients and SQLite caches from a `MaaSConfig`."""
        if not isinstance(config, MaaSConfig):
            raise TypeError(
                'MaaSDataFetcher requires an explicit `MaaSConfig`.'
            )
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
            cache_dir=config.cache_dir,
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
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Resolve and snap `(lat, lon)` across all three river networks."""
        return resolve_reaches(lat, lon, self.config, **kwargs)

    def fetch_gauges(
        self,
        bbox: tuple[float, float, float, float],
        **kwargs: Any,  # noqa: ANN401
    ) -> list[dict[str, Any]]:
        """Fetch FloodHub gauge stations within `bbox`."""
        return fetch_gauges(self.config, bbox, **kwargs)

    def fetch_historical(
        self,
        provider: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> pd.Series:
        """Fetch historical daily discharge `pd.Series` for `provider`."""
        return fetch_historical(self.config, provider, **kwargs)

    def fetch_return_periods(
        self,
        provider: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any] | None:
        """Compute or fetch `[2, 5, 10, 20, 50, 100]`-yr return periods."""
        return fetch_return_periods(self.config, provider, **kwargs)

    def fetch_forecasts(  # noqa: C901, PLR0913, PLR0915, PLR0917
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
        """Fetch a pure multi-model bundle (forecasts, return periods, reaches).

        Provider requests run concurrently; this orchestrator intentionally
        keeps the three provider tasks as closures over the resolved reaches.
        """
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
            float(gl_snap.get('query_lat', gl_snap['lat'])) if gl_snap else lat
        )
        gl_query_lon = (
            float(gl_snap.get('query_lon', gl_snap['lon'])) if gl_snap else lon
        )
        fh_lat = float(gl_snap['lat']) if gl_snap else lat
        fh_lon = float(gl_snap['lon']) if gl_snap else lon
        eff_river_id = reaches['geoglows'].get('river_id') or (
            parse_int(river_id) if is_geoglows_river_id(river_id) else None
        )
        eff_gauge_id = reaches['floodhub'].get('gauge_id') or gauge_id

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
                gid, gauge_meta, dist_km = self.floodhub.find_nearest_gauge(
                    fh_lat, fh_lon, target_area_km2=target_area
                )
            if not gid:
                return None, None
            is_live_fh = not hasattr(self.floodhub.fetch_forecast, '_mock_name')
            fh_cache_key = f'floodhub_fc_{gid}'
            if is_live_fh:
                cached_fh = self.flood_cache.get(fh_cache_key, max_age_s=1800)
                if (
                    isinstance(cached_fh, dict)
                    and cached_fh.get('status') == 'live'
                ):
                    return cached_fh, gid
            fc = self.floodhub.fetch_forecast(gid)
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

        def _task_glofas() -> tuple[
            dict[str, Any] | None, dict[str, Any] | None
        ]:
            if 'glofas' not in req_models:
                return None, None
            cell_lat, cell_lon = glofas_cell_center(gl_query_lat, gl_query_lon)
            rp_key = f'glofas_rp_{cell_lat:.3f}_{cell_lon:.3f}'
            fc_key = f'glofas_fc_v2_{cell_lat:.3f}_{cell_lon:.3f}'
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
                res_fc = self.glofas.fetch_forecast(
                    gl_query_lat, gl_query_lon, forecast_days=15
                )
                if is_live_gl and res_fc.get('status') == 'live':
                    self.flood_cache.put(fc_key, res_fc)
                return res_fc

            def _fetch_gl_rp() -> dict[str, Any] | None:
                res_rp = self.glofas.fetch_reanalysis_return_periods(
                    gl_query_lat, gl_query_lon
                )
                if (
                    res_rp is not None
                    and res_rp.get('source') != 'unit_test_rp'
                ):
                    self.flood_cache.put(rp_key, res_rp)
                return res_rp

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

        def _task_geoglows() -> tuple[  # noqa: C901
            dict[str, Any] | None, dict[str, Any] | None, int | None
        ]:
            if 'geoglows' not in req_models:
                return None, None, eff_river_id
            rid = eff_river_id
            if not is_geoglows_river_id(rid):
                rid = self.geoglows.fetch_river_id(lat, lon)
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
            fc_key = f'geoglows_fc_v2_{rid}'
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
                res_fc = self.geoglows.fetch_forecast(rid)
                if is_live_gg and res_fc.get('status') == 'live':
                    self.flood_cache.put(fc_key, res_fc)
                return res_fc

            def _fetch_gg_rp() -> dict[str, Any] | None:
                res_rp = self.geoglows.fetch_return_periods(rid)
                if res_rp is None:
                    res_rp = self.geoglows.fetch_retrospective_return_periods(
                        rid
                    )
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
            'lat': reaches['floodhub'].get('lat')
            or ((fh_fc or {}).get('gauge_location') or {}).get('lat'),
            'lon': reaches['floodhub'].get('lon')
            or ((fh_fc or {}).get('gauge_location') or {}).get('lon'),
            'distance_km': (fh_fc or {}).get('distance_km')
            if (fh_fc or {}).get('distance_km') is not None
            else reaches['floodhub'].get('offset_km'),
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
    'RETURN_PERIOD_YEARS',
    'MaaSDataFetcher',
    'MaaSEngine',
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
