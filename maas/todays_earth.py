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

"""JAXA Today's Earth (TE-Global CaMa-Flood) client, physics emulator, and binary grid lookup."""

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests

from maas.config import (
    CAMA_GRID_RES_DEG,
    JAXA_STAC_CATALOG_URL,
    TODAYS_EARTH_SOURCE,
    TODAYS_EARTH_TIMEOUT_S,
    parse_finite_float,
)
from maas.networks import (
    cama_cell_area_km2,
    cama_cell_id,
    cama_cell_polygon,
    snap_cama_cell,
)
from maas.thresholds import gumbel_quantile_from_return_periods

TE_CATALOG_TOKENS: tuple[str, ...] = (
    'todays',
    'today',
    'te-global',
    'te_global',
    'camaflood',
    'cama-flood',
    'cama_flood',
    'rivout',
    'fldout',
    'flddph',
    'fldfrc',
    'sfcelv',
)

CAMA_FLOODPLAIN_K = 0.6
CAMA_FLDOUT_SHARE = 0.35

TODAYS_EARTH_EMULATION_NOTE = (
    'Emulated: JAXA does not publish TE-Global CaMa-Flood forecasts through a '
    "public machine-readable API (no Today's Earth collection in the JAXA Earth "
    'STAC catalog). Values are a deterministic CaMa-Flood-physics emulation '
    'driven by the GloFAS v4 forecast; set TODAYS_EARTH_API_URL to connect an '
    'operational TE-Global feed.'
)


def _series_median(values: Sequence[Any]) -> float | None:
    valid = sorted(
        v
        for v in (parse_finite_float(x) for x in values)
        if v is not None and v >= 0.0
    )
    return valid[len(valid) // 2] if valid else None


def extract_te_series(
    payload: Mapping[str, Any],
    *names: str,
) -> list[float | None] | None:
    """First list-valued field among `names` (top level or under `flood_forecast`)."""
    scopes: list[Mapping[str, Any]] = [payload]
    ff = payload.get('flood_forecast')
    if isinstance(ff, Mapping):
        scopes.append(ff)
    for scope in scopes:
        for name in names:
            val = scope.get(name)
            if isinstance(val, list) and val:
                return [parse_finite_float(v) for v in val]
    return None


def parse_todays_earth_payload(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse an operational TE-Global point-forecast JSON payload."""
    if not isinstance(payload, Mapping):
        return None, "Invalid Today's Earth payload"
    times = payload.get('timestamps') or payload.get('time')
    rivout = extract_te_series(payload, 'rivout', 'RIVOUT')
    mean = extract_te_series(payload, 'mean', 'outflw', 'OUTFLW')
    if not isinstance(times, list) or not times or not (rivout or mean):
        return None, "Today's Earth payload is missing timestamps or RIVOUT"
    n = len(times)

    def _fit(
        series: list[float | None] | None,
        default: float | None = None,
    ) -> list[float | None]:
        return (list(series or []) + [default] * n)[:n]

    fldout = _fit(extract_te_series(payload, 'fldout', 'FLDOUT'), 0.0)
    if rivout is None:
        rivout = [
            None if m is None else max(m - (f or 0.0), 0.0)
            for m, f in zip(_fit(mean), fldout, strict=False)
        ]
    rivout = _fit(rivout)
    if mean is None:
        mean = [
            None if r is None else r + (f or 0.0)
            for r, f in zip(rivout, fldout, strict=False)
        ]
    fldfrc_pct = extract_te_series(payload, 'fldfrc_pct')
    if fldfrc_pct is None:
        frac = extract_te_series(payload, 'fldfrc', 'FLDFRC')
        fldfrc_pct = (
            [None if v is None else v * 100.0 for v in frac] if frac else None
        )
    return {
        'timestamps': [str(t) for t in times],
        'mean': _fit(mean),
        'rivout': rivout,
        'fldout': fldout,
        'p25': _fit(extract_te_series(payload, 'p25')),
        'p75': _fit(extract_te_series(payload, 'p75')),
        'max': _fit(extract_te_series(payload, 'max')),
        'min': _fit(extract_te_series(payload, 'min')),
        'flddph_m': _fit(
            extract_te_series(payload, 'flddph_m', 'flddph', 'FLDDPH'), 0.0
        ),
        'fldfrc_pct': _fit(fldfrc_pct, 0.0),
        'sfcelv_m': _fit(
            extract_te_series(payload, 'sfcelv_m', 'sfcelv', 'SFCELV')
        ),
    }, None


def route_floodplain_excess(
    series: Sequence[float],
    q_bankfull: float,
    k: float = CAMA_FLOODPLAIN_K,
) -> list[float]:
    """Linear-reservoir routing of above-bankfull flow (daily explicit scheme)."""
    routed: list[float] = []
    state: float | None = None
    for q in series:
        excess = max(float(q) - q_bankfull, 0.0)
        state = excess if state is None else state + k * (excess - state)
        routed.append(state)
    return routed


def emulate_camaflood_physics(
    records: Sequence[Mapping[str, Any]],
    rps: Mapping[str, Any] | None,
    elev: float,
    elev_source: str = 'DEM',
    forcing_status: str | None = 'live',
) -> dict[str, Any]:
    """Deterministic CaMa-Flood channel/floodplain routing from GloFAS forecast records."""
    records_slice = list(records)[:6]

    def _col(name: str, fallback: str = 'discharge_mean') -> list[float]:
        out: list[float] = []
        for r in records_slice:
            v = parse_finite_float(r.get(name))
            if v is None:
                v = parse_finite_float(r.get(fallback))
            out.append(max(v or 0.0, 0.0))
        return out

    central = _col('discharge_median')
    rps_map = rps or {}
    q_clim = (
        parse_finite_float(rps_map.get('mean_flow'))
        or _series_median(central)
        or 1.0
    )
    q_clim = max(q_clim, 0.05)
    width = max(0.40 * (q_clim**0.75), 10.0)
    depth = max(0.10 * (q_clim**0.5), 1.0)
    q_bf = max(
        gumbel_quantile_from_return_periods(rps_map, 1.5) or 0.0,
        1.2 * q_clim,
        0.5,
    )
    elev_c = min(max(float(elev), 0.0), 1500.0)
    depth_scale = 1.0 + elev_c / 150.0
    f_max = (0.02 + 0.08 * math.log10(1.0 + q_clim / 10.0)) * (
        1.0 + 1.5 * math.exp(-elev_c / 30.0)
    )
    f_max = min(max(f_max, 0.02), 0.6)

    def _cama(
        series: list[float],
    ) -> tuple[list[float], list[float], list[float]]:
        routed = route_floodplain_excess(series, q_bf)
        total = [min(q, q_bf) + r for q, r in zip(series, routed, strict=False)]
        fld = [CAMA_FLDOUT_SHARE * r for r in routed]
        return total, [t - f for t, f in zip(total, fld, strict=False)], fld

    total, rivout, fldout = _cama(central)
    stage = [depth * ((max(r, 0.0) / q_bf) ** 0.6) for r in rivout]
    flddph = [max(h - depth, 0.0) for h in stage]
    fldfrc = [
        100.0 * f_max * (1.0 - math.exp(-d / depth_scale)) for d in flddph
    ]
    sfcelv = [max(max(float(elev), 0.0) - depth + h, 0.0) for h in stage]

    def _r2(xs: Sequence[float]) -> list[float]:
        return [round(x, 2) for x in xs]

    return {
        'series': {
            'timestamps': [
                f'{str(r.get("time"))[:10]}T00:00:00Z' for r in records_slice
            ],
            'mean': _r2(total),
            'rivout': _r2(rivout),
            'fldout': _r2(fldout),
            'p25': _r2(_cama(_col('discharge_p25'))[0]),
            'p75': _r2(_cama(_col('discharge_p75'))[0]),
            'max': _r2(_cama(_col('discharge_max'))[0]),
            'min': _r2(_cama(_col('discharge_min'))[0]),
            'flddph_m': [round(d, 3) for d in flddph],
            'fldfrc_pct': _r2(fldfrc),
            'sfcelv_m': _r2(sfcelv),
        },
        'channel_params': {
            'mean_flow_m3s': round(q_clim, 3),
            'bankfull_discharge_m3s': round(q_bf, 2),
            'channel_width_m': round(width, 1),
            'channel_depth_m': round(depth, 2),
            'ground_elevation_m': float(elev),
            'elevation_source': elev_source,
            'max_flooded_fraction_ceiling_pct': round(100.0 * f_max, 1),
        },
        'forcing_status': forcing_status,
        'return_period_status': rps_map.get('status'),
    }


def format_todays_earth_forecast(  # noqa: PLR0913
    lat: float,
    lon: float,
    series: Mapping[str, Any],
    *,
    live: bool,
    reach_id: str | None = None,
    channel_params: Mapping[str, Any] | None = None,
    forcing_status: str | None = None,
    live_probe: Mapping[str, Any] | None = None,
    live_error: str | None = None,
) -> dict[str, Any]:
    """Assemble the canonical Today's Earth forecast response payload."""
    cell_lat, cell_lon = snap_cama_cell(lat, lon)
    cell_id = cama_cell_id(cell_lat, cell_lon)
    times = list(series['timestamps'])
    flddph = list(series['flddph_m'])
    fldfrc = list(series['fldfrc_pct'])
    sfcelv = list(series['sfcelv_m'])
    valid_depth = [(d, i) for i, d in enumerate(flddph) if d is not None]
    peak_depth, peak_idx = max(valid_depth) if valid_depth else (0.0, None)
    valid_frac = [f for f in fldfrc if f is not None]
    valid_elev = [e for e in sfcelv if e is not None]
    data: list[dict[str, Any]] = []
    for i, t in enumerate(times):
        data.append(
            {
                'time': t,
                'discharge_mean': series['mean'][i],
                'rivout': series['rivout'][i],
                'fldout': series['fldout'][i],
                'discharge_p25': series['p25'][i],
                'discharge_p75': series['p75'][i],
                'discharge_max': series['max'][i],
                'discharge_min': series['min'][i],
                'flddph_m': flddph[i],
                'fldfrc_pct': fldfrc[i],
                'sfcelv_m': sfcelv[i],
            }
        )
    method = (
        'Operational TE-Global point forecast (TODAYS_EARTH_API_URL)'
        if live
        else (
            'CaMa-Flood physics emulator (Yamazaki et al., 2011 channel '
            'geometry, EV1 bankfull, linear floodplain reservoir) forced by '
            'GloFAS v4'
        )
    )
    return {
        'model': 'jaxa_todays_earth',
        'available': True,
        'source': TODAYS_EARTH_SOURCE,
        'status': 'live' if live else 'fallback',
        'emulated': not live,
        'method': method,
        'note': None if live else TODAYS_EARTH_EMULATION_NOTE,
        'grid_cell_id': cell_id,
        'grid_resolution_deg': CAMA_GRID_RES_DEG,
        'cell_center_lat': cell_lat,
        'cell_center_lon': cell_lon,
        'cell_area_km2': cama_cell_area_km2(cell_lat),
        'reach_id': reach_id,
        'unit': 'm³/s',
        'timestamps': times,
        'mean': list(series['mean']),
        'rivout': list(series['rivout']),
        'fldout': list(series['fldout']),
        'p25': list(series['p25']),
        'p75': list(series['p75']),
        'max': list(series['max']),
        'min': list(series['min']),
        'flood_forecast': {
            'flddph_m': flddph,
            'fldfrc_pct': fldfrc,
            'sfcelv_m': sfcelv,
            'max_flood_depth_m': round(peak_depth or 0.0, 3),
            'max_flooded_fraction_pct': (
                round(max(valid_frac), 2) if valid_frac else 0.0
            ),
            'max_sfcelv_m': round(max(valid_elev), 2) if valid_elev else None,
            'peak_depth_time': (
                times[peak_idx]
                if (peak_idx is not None and peak_depth > 0.0)
                else None
            ),
        },
        'data': data,
        'channel_params': dict(channel_params) if channel_params else None,
        'forcing_status': forcing_status,
        'live_probe': dict(live_probe) if live_probe else None,
        'live_error': live_error,
    }


def camaflood_unit_feature(
    lat: float,
    lon: float,
    service_status: str = 'emulated',
) -> dict[str, Any]:
    """GeoJSON Feature for the 0.25° CaMa-Flood unit-catchment grid cell."""
    cell_lat, cell_lon = snap_cama_cell(lat, lon)
    ring, bbox = cama_cell_polygon(cell_lat, cell_lon)
    label = "Today's Earth CaMa-Flood Unit Grid (0.25°)"
    return {
        'type': 'Feature',
        'geometry': {'type': 'Polygon', 'coordinates': [ring]},
        'properties': {
            'fabric': 'camaflood_unit',
            'fabric_name': label,
            'geofabric': 'camaflood_unit',
            'geofabric_label': label,
            'model': "JAXA Today's Earth (MATSIRO + CaMa-Flood)",
            'source': f'{TODAYS_EARTH_SOURCE} unit-catchment grid',
            'service_status': service_status,
            'grid_cell_id': cama_cell_id(cell_lat, cell_lon),
            'cell_center_lat': cell_lat,
            'cell_center_lon': cell_lon,
            'area_km2': cama_cell_area_km2(cell_lat),
            'resolution': '0.25° (~28 km)',
            'bbox': bbox,
        },
    }


def lookup_camaflood_binary_cell(
    bin_path: Path,
    lat: float,
    lon: float,
    res_deg: float = CAMA_GRID_RES_DEG,
    dtype: str = '<f4',
) -> float:
    """Read a single cell value from a flat binary CaMa-Flood global raster.

    Returns `float('nan')` for CaMa-Flood missing-value sentinels (`<= -9000`
    or `>= 1e19`).
    """
    if not bin_path.exists():
        raise FileNotFoundError(f'CaMa-Flood binary file not found: {bin_path}')
    nlat = int(round(180.0 / res_deg))
    nlon = int(round(360.0 / res_deg))
    dt = np.dtype(dtype)
    expected_bytes = nlat * nlon * dt.itemsize
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f'Unexpected CaMa-Flood binary size for {bin_path}: '
            f'expected {expected_bytes} bytes ({nlat}x{nlon}), got {actual_bytes}.'
        )
    cell_lat, cell_lon = snap_cama_cell(lat, lon, res=res_deg)
    row = min(max(int(round((90.0 - cell_lat) / res_deg - 0.5)), 0), nlat - 1)
    col = min(
        max(int(round((cell_lon + 180.0) / res_deg - 0.5)), 0),
        nlon - 1,
    )
    grid = np.fromfile(bin_path, dtype=dt).reshape((nlat, nlon))
    val = float(grid[row, col])
    if not math.isfinite(val) or val <= -9000.0 or val >= 1e19:
        return float('nan')
    return val


class TodaysEarthClient:
    """Client for JAXA Today's Earth STAC catalog and point-forecast feeds."""

    def __init__(
        self,
        api_url: str = '',
        stac_catalog_url: str = JAXA_STAC_CATALOG_URL,
        timeout_s: float = TODAYS_EARTH_TIMEOUT_S,
        session: requests.Session | None = None,
    ) -> None:
        self.api_url = api_url.strip()
        self.stac_catalog_url = stac_catalog_url
        self.timeout_s = timeout_s
        self.session = session if session is not None else requests.Session()

    def probe_stac_catalog(self) -> dict[str, Any]:
        """Probe the public JAXA Earth STAC catalog for Today's Earth collections."""
        resp = self.session.get(self.stac_catalog_url, timeout=self.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        scanned = 0
        hits: list[str] = []
        if isinstance(payload, Mapping):
            for link in payload.get('links') or ():
                if not isinstance(link, Mapping) or link.get('rel') != 'child':
                    continue
                scanned += 1
                text = f'{link.get("href", "")} {link.get("title", "")}'.lower()
                if any(tok in text for tok in TE_CATALOG_TOKENS):
                    hits.append(str(link.get('href') or ''))
        return {
            'catalog_url': self.stac_catalog_url,
            'reachable': isinstance(payload, Mapping),
            'collections_scanned': scanned,
            'todays_earth_collections': hits,
            'error': None,
            'checked_at': datetime.now(UTC).strftime(
                '%Y-%m-%dT%H:%M:%SZ'
            ),
        }

    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        reach_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch an operational TE-Global forecast from `self.api_url`."""
        if not self.api_url:
            raise ValueError(
                'TodaysEarthClient requires a non-empty `api_url` for live queries.'
            )
        params: dict[str, Any] = {'lat': lat, 'lon': lon}
        if reach_id:
            params['reach_id'] = reach_id
        resp = self.session.get(
            self.api_url, params=params, timeout=self.timeout_s
        )
        resp.raise_for_status()
        series, err = parse_todays_earth_payload(resp.json())
        if series is None:
            raise ValueError(err or "Invalid Today's Earth payload")
        return format_todays_earth_forecast(
            lat,
            lon,
            series,
            live=True,
            reach_id=reach_id,
        )


__all__ = [
    'CAMA_FLDOUT_SHARE',
    'CAMA_FLOODPLAIN_K',
    'TE_CATALOG_TOKENS',
    'TODAYS_EARTH_EMULATION_NOTE',
    'TodaysEarthClient',
    'camaflood_unit_feature',
    'emulate_camaflood_physics',
    'extract_te_series',
    'format_todays_earth_forecast',
    'lookup_camaflood_binary_cell',
    'parse_todays_earth_payload',
    'route_floodplain_excess',
]
