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

"""Multi-model flood forecast comparison orchestrator (`MaaSEngine`) and SQLite caching."""

import json
import sqlite3
import threading
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

from shapely.geometry import mapping

from maas.config import (
    CAMA_GRID_RES_DEG,
    MAAS_MODEL_NAMES,
    TODAYS_EARTH_SOURCE,
    MaaSConfig,
    normalize_requested_models,
    parse_finite_float,
    parse_float_or_default,
)
from maas.floodhub import (
    FH_SEVERITY_LABELS,
    FH_SEVERITY_RANK,
    FH_SEVERITY_TO_RISK,
    FloodHubClient,
    geom_area_km2,
    round_geojson_coords,
)
from maas.geoglows import GeoGLOWSClient
from maas.glofas import GloFASClient
from maas.networks import (
    glofas_cell_center,
    glofas_cell_polygon,
    is_geoglows_river_id,
)
from maas.thresholds import (
    EXCEEDANCE_CLASSES,
    RISK_RANK,
    UNASSESSED_COLOR,
    UNASSESSED_LABEL,
    classify_exceedance,
    estimate_return_period_years,
    thresholds_from_return_periods,
)
from maas.todays_earth import (
    TodaysEarthClient,
    camaflood_unit_feature,
    emulate_camaflood_physics,
    format_todays_earth_forecast,
)

CORRIDOR_WIDTH_FACTOR: tuple[float, ...] = (1.0, 3.0, 5.0, 8.0, 10.0)
FH_DERIVED_WIDTH_FACTOR: tuple[float, ...] = (0.0, 4.0, 7.0, 10.0)


class SQLiteCache:
    """Thread-safe persistent SQLite key-value JSON cache with explicit `db_path`."""

    def __init__(self, db_path: Any, table_name: str = 'flood_cache') -> None:
        if db_path is None or str(db_path).strip() == '':
            raise ValueError('SQLiteCache requires a non-empty `db_path`.')
        if not table_name.isidentifier():
            raise ValueError(f'Invalid SQLite table name: {table_name!r}')
        from pathlib import Path

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
        if row and row[0] and (time.time() - float(row[1] or 0.0)) <= max_age_s:
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


def depth_color(depth_m: float | None) -> str:
    """Return hex color for a CaMa-Flood floodplain depth in meters."""
    if depth_m is None or depth_m <= 0.0:
        return '#bae6fd'
    if depth_m < 0.5:
        return '#38bdf8'
    if depth_m < 1.0:
        return '#0284c7'
    if depth_m < 2.0:
        return '#1d4ed8'
    return '#1e3a8a'


def channel_half_width_m(mean_discharge: Any) -> float:
    """Half the CaMa-Flood power-law channel width (floored at 40 m)."""
    q = max(parse_float_or_default(mean_discharge, 0.0), 0.0)
    return max(0.5 * max(0.40 * (q**0.75), 10.0), 40.0)


def daily_series(
    records: Sequence[Mapping[str, Any]] | None,
    key: str,
    fallback_key: str | None = None,
) -> dict[str, float]:
    """Daily means `{YYYY-MM-DD: value}` of a (sub-)daily record series."""
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


def window_peak(
    daily: Mapping[str, float],
    today_str: str | None = None,
) -> tuple[float | None, str | None]:
    """Peak `(value, date)` over today and later (whole series if nothing is current)."""
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


def spread_confidence(
    central: float | None,
    p25: float | None,
    p75: float | None,
    live: bool = True,  # noqa: FBT001, FBT002
    fallback_thresholds: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """High / Medium / Low confidence from the relative inter-quartile spread at peak."""
    if not live:
        return 'Low'
    level = 1
    if (
        central is not None
        and p25 is not None
        and p75 is not None
        and central > 0.0
    ):
        rel = max(p75 - p25, 0.0) / central
        level = 2 if rel < 0.25 else (1 if rel < 0.6 else 0)
    if fallback_thresholds:
        level = max(level - 1, 0)
    return ('Low', 'Medium', 'High')[level]


def build_consensus_row(  # noqa: PLR0913, PLR0917
    model: str,
    available: bool,  # noqa: FBT001
    status: str | None,
    peak: float | None,
    peak_time: str | None,
    rps: Mapping[str, Any] | None,
    thresholds_source: str | None,
    confidence: str,
    unit: str = 'm³/s',
    independent: bool = True,  # noqa: FBT001, FBT002
    **extra: Any,
) -> dict[str, Any]:
    """Build a standardized per-provider consensus row."""
    synthetic = available and status == 'fallback'
    cls = (
        classify_exceedance(peak, rps)
        if (available and not synthetic)
        else None
    )
    row: dict[str, Any] = {
        'model': model,
        'name': MAAS_MODEL_NAMES.get(model, model),
        'available': available,
        'status': status,
        'unit': unit,
        'peak_flow': round(peak, 2) if peak is not None else None,
        'peak_time': peak_time,
        'return_period': cls['return_period'] if cls else None,
        'return_period_yrs': (
            estimate_return_period_years(peak, rps or {})
            if (cls and unit == 'm³/s')
            else None
        ),
        'exceedance_rank': cls['rank'] if cls else None,
        'exceedance_label': (
            cls['label'] if cls else (UNASSESSED_LABEL if synthetic else None)
        ),
        'risk_level': cls['risk_level'] if cls else 'UNKNOWN',
        'confidence': confidence if available else 'N/A',
        'thresholds_source': thresholds_source,
        'independent': independent,
    }
    row.update(extra)
    return row


def reach_exceedance_summary(
    glofas_fc: Mapping[str, Any] | None,
    glofas_rp: Mapping[str, Any] | None,
    geoglows_fc: Mapping[str, Any] | None,
    geoglows_rp: Mapping[str, Any] | None,
    today_str: str | None = None,
) -> dict[str, Any]:
    """Forecast-peak return-period exceedance of the reach across live models."""
    per_model: dict[str, Any] = {}
    excluded: list[str] = []
    for key, fc, rps, value_keys in (
        (
            'glofas',
            glofas_fc,
            glofas_rp,
            ('discharge_median', 'discharge_mean'),
        ),
        ('geoglows', geoglows_fc, geoglows_rp, ('flow_med',)),
    ):
        if not fc:
            continue
        if fc.get('status') in ('fallback', 'unavailable'):
            excluded.append(key)
            continue
        peak, when = window_peak(
            daily_series(fc.get('data'), *value_keys),
            today_str=today_str,
        )
        cls = classify_exceedance(peak, rps)
        per_model[key] = {
            'peak_flow': peak,
            'peak_time': when,
            'rank': cls['rank'],
            'label': cls['label'],
            'return_period_yrs': estimate_return_period_years(peak, rps or {}),
            'thresholds_status': (rps or {}).get('status'),
        }
    if not per_model:
        out = dict(EXCEEDANCE_CLASSES[0])
        out.update(
            {
                'label': (
                    UNASSESSED_LABEL
                    if excluded
                    else 'Not assessed (no forecast)'
                ),
                'risk_level': 'UNKNOWN',
                'return_period': None,
                'color': UNASSESSED_COLOR,
                'governing_model': None,
                'per_model': {},
                'excluded_models': excluded,
                'unit': 'm³/s',
            }
        )
        return out
    gov = max(per_model, key=lambda k: per_model[k]['rank'])
    out = dict(EXCEEDANCE_CLASSES[per_model[gov]['rank']])
    out.update(
        {
            'governing_model': gov,
            'per_model': per_model,
            'excluded_models': excluded,
            'unit': 'm³/s',
        }
    )
    return out


def align_daily_series(
    daily: Mapping[str, float],
    dates: Sequence[str],
    ndigits: int = 2,
) -> list[float | None]:
    """Align a `{YYYY-MM-DD: value}` mapping onto a common `dates` list (`None` for missing dates)."""
    return [round(daily[d], ndigits) if d in daily else None for d in dates]


def geojson_feature(geom: Any, props: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a shapely geometry and properties dictionary into a GeoJSON Feature."""
    props_copy = dict(props)
    props_copy.setdefault('area_km2', geom_area_km2(geom))
    return {
        'type': 'Feature',
        'geometry': {
            'type': geom.geom_type,
            'coordinates': round_geojson_coords(mapping(geom)['coordinates']),
        },
        'properties': props_copy,
    }


def build_flood_summary(
    consensus: Sequence[Mapping[str, Any]],
    te: Mapping[str, Any] | None,
    fh_fc: Mapping[str, Any] | None,
    fh_inund: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build cross-model flood risk summary from consensus rows."""
    te_ff = (te or {}).get('flood_forecast') or {}
    independent = [
        r
        for r in consensus
        if r.get('available')
        and r.get('independent')
        and r.get('status') in ('live', 'severity_only')
    ]
    excluded = [
        str(r.get('name'))
        for r in consensus
        if r.get('available')
        and r.get('independent')
        and r.get('status') not in ('live', 'severity_only')
    ]
    n_exceed = sum(
        1 for r in independent if (r.get('exceedance_rank') or 0) >= 1
    )
    worst = (
        max(independent, key=lambda r: r.get('exceedance_rank') or 0)
        if independent
        else None
    )
    risks = [
        str(r['risk_level'])
        for r in independent
        if r.get('risk_level') in RISK_RANK
    ]
    overall = max(risks, key=lambda k: RISK_RANK[k]) if risks else 'UNKNOWN'
    agreement = (
        f'{n_exceed} of {len(independent)} independent models '
        'forecast ≥ 2-yr exceedance'
    )
    if te and te.get('emulated'):
        agreement += " (Today's Earth is emulated from GloFAS and excluded)"
    if excluded:
        agreement += f'; offline fallback excluded: {", ".join(excluded)}'
    return {
        'max_inundation_depth_m': te_ff.get('max_flood_depth_m'),
        'max_flooded_fraction_pct': te_ff.get('max_flooded_fraction_pct'),
        'peak_sfcelv_m': te_ff.get('max_sfcelv_m'),
        'peak_depth_time': te_ff.get('peak_depth_time'),
        'inundation_source': (
            (
                TODAYS_EARTH_SOURCE
                + (' — emulated' if te.get('emulated') else '')
            )
            if te
            else None
        ),
        'floodhub_severity': (
            (fh_inund or {}).get('severity') if fh_fc else 'UNAVAILABLE'
        ),
        'floodhub_trend': (fh_inund or {}).get('trend') if fh_fc else None,
        'floodhub_severity_source': (fh_inund or {}).get('severity_source'),
        'floodhub_inundation_maps_available': bool(
            (fh_inund or {}).get('inundation_maps_available')
        ),
        'floodhub_inundation_map_levels': (
            (fh_inund or {}).get('inundation_map_levels') or []
        ),
        'return_period_exceedance': (
            worst['exceedance_label'] if worst else 'Unknown'
        ),
        'return_period_exceedance_model': worst['name'] if worst else None,
        'max_exceedance_rank': (
            (worst.get('exceedance_rank') or 0) if worst else None
        ),
        'models_exceeding_2yr': n_exceed,
        'independent_models_evaluated': len(independent),
        'overall_risk_level': overall,
        'agreement': agreement,
    }


def build_aligned_timeline(
    fh_fc: Mapping[str, Any] | None,
    gl: Mapping[str, Any] | None,
    gg_fc: Mapping[str, Any] | None,
    te: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a date-aligned daily timeline across all active providers."""
    series_daily: dict[str, dict[str, dict[str, float]]] = {}
    fh_is_q = (
        str((fh_fc or {}).get('unit') or '').upper()
        == 'CUBIC_METERS_PER_SECOND'
    )
    if fh_fc and fh_fc.get('data'):
        series_daily['floodhub'] = {
            'central': daily_series(fh_fc.get('data'), 'discharge')
        }
    if gl and gl.get('data'):
        series_daily['glofas'] = {
            'central': daily_series(
                gl.get('data'), 'discharge_median', 'discharge_mean'
            ),
            'p25': daily_series(gl.get('data'), 'discharge_p25'),
            'p75': daily_series(gl.get('data'), 'discharge_p75'),
            'max': daily_series(gl.get('data'), 'discharge_max'),
            'min': daily_series(gl.get('data'), 'discharge_min'),
        }
    if gg_fc and gg_fc.get('data'):
        series_daily['geoglows'] = {
            'central': daily_series(gg_fc.get('data'), 'flow_med'),
            'p25': daily_series(gg_fc.get('data'), 'flow_25p'),
            'p75': daily_series(gg_fc.get('data'), 'flow_75p'),
            'max': daily_series(gg_fc.get('data'), 'flow_max'),
            'min': daily_series(gg_fc.get('data'), 'flow_min'),
        }
    if te and te.get('data'):
        series_daily['todays_earth'] = {
            'central': daily_series(te.get('data'), 'discharge_mean'),
            'p25': daily_series(te.get('data'), 'discharge_p25'),
            'p75': daily_series(te.get('data'), 'discharge_p75'),
            'rivout': daily_series(te.get('data'), 'rivout'),
            'fldout': daily_series(te.get('data'), 'fldout'),
            'flddph_m': daily_series(te.get('data'), 'flddph_m'),
            'fldfrc_pct': daily_series(te.get('data'), 'fldfrc_pct'),
        }
    dates = sorted(
        {d for s in series_daily.values() for col in s.values() for d in col}
    )
    timeline: dict[str, Any] = {'dates': dates, 'series': {}}
    for model, cols in series_daily.items():
        timeline['series'][model] = {
            name: align_daily_series(col, dates, 3 if name == 'flddph_m' else 2)
            for name, col in cols.items()
        }
    if 'floodhub' in timeline['series']:
        timeline['series']['floodhub']['unit'] = 'm³/s' if fh_is_q else 'm'
        timeline['series']['floodhub']['axis'] = (
            'discharge' if fh_is_q else 'stage'
        )
    timeline['status'] = {
        'floodhub': fh_fc.get('status') if fh_fc else None,
        'glofas': (gl or {}).get('status'),
        'geoglows': (gg_fc or {}).get('status'),
        'todays_earth': (
            ('emulated' if te.get('emulated') else te.get('status'))
            if te
            else None
        ),
    }
    return timeline


class MaaSEngine:
    """Orchestrates multi-model forecasts, return periods, and spatial alignment."""

    def __init__(self, config: MaaSConfig) -> None:
        if not isinstance(config, MaaSConfig):
            raise TypeError('MaaSEngine requires an explicit `MaaSConfig`.')
        self.config = config
        self.floodhub = FloodHubClient(
            api_key=config.floodhub_api_key,
            base_url=config.floodhub_base_url,
            timeout_s=config.http_timeout_s,
        )
        self.glofas = GloFASClient(
            base_url=config.glofas_base_url,
            timeout_s=config.http_timeout_s,
        )
        self.geoglows = GeoGLOWSClient(
            base_url=config.geoglows_base_url,
            timeout_s=config.http_timeout_s,
        )
        self.todays_earth = TodaysEarthClient(
            api_url=config.todays_earth_api_url,
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

    def get_watershed_polygon(
        self,
        lat: float,
        lon: float,
        fabric: str = 'glofas_cell',
        geofabric: str | None = None,
    ) -> dict[str, Any]:
        """Resolve grid-based geofabric polygon (`glofas_cell` or `camaflood_unit`)."""
        eff_fabric = (geofabric or fabric or 'glofas_cell').strip().lower()
        if eff_fabric == 'camaflood_unit':
            status = (
                'operational'
                if self.config.todays_earth_api_url.strip()
                else 'emulated'
            )
            return camaflood_unit_feature(lat, lon, service_status=status)
        if eff_fabric == 'glofas_cell':
            return glofas_cell_polygon(lat, lon)
        raise ValueError(
            f'Unsupported grid fabric {eff_fabric!r} in standalone MaaSEngine; '
            "expected 'glofas_cell' or 'camaflood_unit'."
        )

    def fetch_unified_forecast(  # noqa: PLR0913, PLR0917
        self,
        lat: float,
        lon: float,
        gauge_id: str | None = None,
        river_id: int | None = None,
        reach_id: str | None = None,
        requested_models: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch and align multi-model forecasts across requested providers."""
        t0 = time.time()
        models = normalize_requested_models(requested_models)

        fh_fc: dict[str, Any] | None = None
        if 'floodhub' in models and gauge_id:
            fh_fc = self.floodhub.fetch_forecast(gauge_id)

        gl: dict[str, Any] | None = None
        glrp: dict[str, Any] | None = None
        if 'glofas' in models or 'todays_earth' in models:
            gl = self.glofas.fetch_forecast(lat, lon, forecast_days=15)
            cell_lat, cell_lon = glofas_cell_center(lat, lon)
            rp_key = f'glofas_rp_{cell_lat:.3f}_{cell_lon:.3f}'
            glrp = self.flood_cache.get(rp_key, max_age_s=90 * 86400)
            if glrp is None:
                glrp = self.glofas.fetch_reanalysis_return_periods(lat, lon)
                if glrp is not None:
                    self.flood_cache.put(rp_key, glrp)

        gg_fc: dict[str, Any] | None = None
        gg_rp: dict[str, Any] | None = None
        eff_river_id = river_id
        if 'geoglows' in models:
            if not is_geoglows_river_id(eff_river_id):
                eff_river_id = self.geoglows.fetch_river_id(lat, lon)
            if eff_river_id is not None:
                gg_fc = self.geoglows.fetch_forecast(eff_river_id)
                rp_key = f'geoglows_rp_{eff_river_id}'
                gg_rp = self.flood_cache.get(rp_key, max_age_s=180 * 86400)

        te: dict[str, Any] | None = None
        if 'todays_earth' in models:
            if self.config.todays_earth_api_url.strip():
                te = self.todays_earth.fetch_forecast(
                    lat, lon, reach_id=reach_id
                )
            elif gl is not None:
                emu = emulate_camaflood_physics(
                    gl.get('data') or (),
                    glrp,
                    elev=10.0,
                    elev_source='default',
                    forcing_status=gl.get('status'),
                )
                te = format_todays_earth_forecast(
                    lat,
                    lon,
                    emu['series'],
                    live=False,
                    reach_id=reach_id,
                    channel_params=emu['channel_params'],
                    forcing_status=emu['forcing_status'],
                )

        models_output: dict[str, Any] = {}
        if 'floodhub' in models:
            models_output['floodhub'] = fh_fc or {
                'available': False,
                'status': 'unavailable',
                'message': 'No FloodHub gauge specified or available.',
            }
        if 'geoglows' in models and gg_fc:
            models_output['geoglows'] = {**gg_fc, 'return_periods': gg_rp}
        if 'glofas' in models and gl:
            models_output['glofas'] = {**gl, 'return_periods': glrp}
        if 'todays_earth' in models and te:
            models_output['todays_earth'] = te

        fh_th = (fh_fc or {}).get('thresholds') or {}
        if (fh_fc or {}).get('status') == 'live' and fh_th.get('warning_2yr'):
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

        consensus: list[dict[str, Any]] = []
        if 'floodhub' in models:
            if fh_fc and fh_fc.get('available'):
                fh_daily = daily_series(fh_fc.get('data'), 'discharge')
                peak, when = window_peak(fh_daily)
                fh_rps = {
                    'return_period_2': fh_th.get('warning_2yr'),
                    'return_period_5': fh_th.get('danger_5yr'),
                    'return_period_20': fh_th.get('extreme_20yr'),
                }
                consensus.append(
                    build_consensus_row(
                        'floodhub',
                        True,
                        str(fh_fc.get('status')),
                        peak,
                        when,
                        fh_rps,
                        'FloodHub gauge model thresholds',
                        'High',
                    )
                )
            else:
                consensus.append(
                    build_consensus_row(
                        'floodhub',
                        False,
                        'unavailable',
                        None,
                        None,
                        None,
                        None,
                        'N/A',
                    )
                )
        if 'glofas' in models:
            if gl and gl.get('available'):
                gl_daily = daily_series(
                    gl.get('data'), 'discharge_median', 'discharge_mean'
                )
                peak, when = window_peak(gl_daily)
                p25 = daily_series(gl.get('data'), 'discharge_p25').get(
                    when or ''
                )
                p75 = daily_series(gl.get('data'), 'discharge_p75').get(
                    when or ''
                )
                consensus.append(
                    build_consensus_row(
                        'glofas',
                        True,
                        str(gl.get('status')),
                        peak,
                        when,
                        glrp,
                        (glrp or {}).get('source'),
                        spread_confidence(peak, p25, p75),
                    )
                )
            else:
                consensus.append(
                    build_consensus_row(
                        'glofas',
                        False,
                        'unavailable',
                        None,
                        None,
                        None,
                        None,
                        'N/A',
                    )
                )
        if 'geoglows' in models:
            if gg_fc and gg_fc.get('available'):
                gg_daily = daily_series(gg_fc.get('data'), 'flow_med')
                peak, when = window_peak(gg_daily)
                p25 = daily_series(gg_fc.get('data'), 'flow_25p').get(
                    when or ''
                )
                p75 = daily_series(gg_fc.get('data'), 'flow_75p').get(
                    when or ''
                )
                consensus.append(
                    build_consensus_row(
                        'geoglows',
                        True,
                        str(gg_fc.get('status')),
                        peak,
                        when,
                        gg_rp,
                        (gg_rp or {}).get('source'),
                        spread_confidence(peak, p25, p75),
                    )
                )
            else:
                consensus.append(
                    build_consensus_row(
                        'geoglows',
                        False,
                        'unavailable',
                        None,
                        None,
                        None,
                        None,
                        'N/A',
                    )
                )
        if 'todays_earth' in models:
            if te and te.get('available'):
                te_daily = daily_series(te.get('data'), 'discharge_mean')
                peak, when = window_peak(te_daily)
                te_ff = te.get('flood_forecast') or {}
                consensus.append(
                    build_consensus_row(
                        'todays_earth',
                        True,
                        'emulated'
                        if te.get('emulated')
                        else str(te.get('status')),
                        peak,
                        when,
                        glrp,
                        'GloFAS v4 reanalysis EV1 (CaMa-Flood emulator climatology)',
                        'Low' if te.get('emulated') else 'Medium',
                        independent=not bool(te.get('emulated')),
                        emulated=bool(te.get('emulated')),
                        peak_flood_depth_m=te_ff.get('max_flood_depth_m'),
                        peak_flood_fraction_pct=te_ff.get(
                            'max_flooded_fraction_pct'
                        ),
                        peak_sfcelv_m=te_ff.get('max_sfcelv_m'),
                    )
                )
            else:
                consensus.append(
                    build_consensus_row(
                        'todays_earth',
                        False,
                        'unavailable',
                        None,
                        None,
                        None,
                        None,
                        'N/A',
                    )
                )

        flood_summary = build_flood_summary(consensus, te, fh_fc, None)
        timeline = build_aligned_timeline(fh_fc, gl, gg_fc, te)
        gl_center_lat, gl_center_lon = glofas_cell_center(lat, lon)
        inund_params: dict[str, Any] = {'lat': lat, 'lon': lon}
        if gauge_id:
            inund_params['gauge_id'] = gauge_id
        if reach_id:
            inund_params['reach_id'] = reach_id
        if eff_river_id:
            inund_params['river_id'] = eff_river_id

        return {
            'location': {
                'lat': lat,
                'lon': lon,
                'gauge_id': gauge_id,
                'river_id': eff_river_id,
                'reach_id': reach_id,
            },
            'thresholds': thresholds,
            'return_periods': {'glofas': glrp, 'geoglows': gg_rp},
            'models': models_output,
            'virtual_station': {
                'probe': {'lat': lat, 'lon': lon},
                'glofas_cell': {
                    'cell_center_lat': gl_center_lat,
                    'cell_center_lon': gl_center_lon,
                    'resolution_deg': 0.05,
                },
                'todays_earth_cell': {
                    'grid_cell_id': te.get('grid_cell_id'),
                    'cell_center_lat': te.get('cell_center_lat'),
                    'cell_center_lon': te.get('cell_center_lon'),
                    'resolution_deg': CAMA_GRID_RES_DEG,
                    'area_km2': te.get('cell_area_km2'),
                }
                if te
                else None,
            },
            'consensus': consensus,
            'flood_summary': flood_summary,
            'flood_inundation': {
                'endpoint': '/api/maas/flood-inundation?'
                + urllib.parse.urlencode(inund_params),
                'layers': [
                    'floodhub_extent',
                    'camaflood_depth',
                    'reach_exceedance',
                ],
                'reach_exceedance': reach_exceedance_summary(
                    gl, glrp, gg_fc, gg_rp
                ),
            },
            'timeline': timeline,
            'meta': {
                'models_requested': models,
                'generated_at': datetime.now(UTC).strftime(
                    '%Y-%m-%dT%H:%M:%SZ'
                ),
                'elapsed_s': round(time.time() - t0, 2),
            },
        }


__all__ = [
    'CORRIDOR_WIDTH_FACTOR',
    'FH_DERIVED_WIDTH_FACTOR',
    'FH_SEVERITY_LABELS',
    'FH_SEVERITY_RANK',
    'FH_SEVERITY_TO_RISK',
    'MaaSEngine',
    'SQLiteCache',
    'align_daily_series',
    'build_aligned_timeline',
    'build_consensus_row',
    'build_flood_summary',
    'channel_half_width_m',
    'daily_series',
    'depth_color',
    'geojson_feature',
    'reach_exceedance_summary',
    'spread_confidence',
    'window_peak',
]
