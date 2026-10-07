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

"""Frontend flood inundation geometry buffering, color ramps, and CaMa-Flood visual emulation."""

import math
from typing import Any

import numpy as np
from shapely import affinity
from shapely.geometry import mapping
from shapely.ops import unary_union

from maas.config import (
    TODAYS_EARTH_SOURCE,
    parse_finite_float,
    parse_float_or_default,
)
from maas.floodhub import geom_area_km2, round_geojson_coords
from maas.networks import cama_cell_area_km2, cama_cell_id, cama_cell_polygon
from maas.thresholds import gumbel_quantile_from_return_periods
from maas.todays_earth import (
    CAMA_FLDOUT_SHARE,
    CAMA_FLOODPLAIN_K,
)


def depth_color(depth_m: float | None) -> str:
    """Colour ramp for CaMa-Flood floodplain water depth (m)."""
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
    """Half the CaMa-Flood power-law channel width (floored at 40 m for visibility)."""
    q = max(parse_float_or_default(mean_discharge, 0.0), 0.0)
    return max(0.5 * max(0.40 * q**0.75, 10.0), 40.0)


def polygonal_only(geom: Any) -> Any:
    """Keep only the (Multi)Polygon part of a geometry."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    if geom.geom_type == 'GeometryCollection':
        parts = [
            g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')
        ]
        return unary_union(parts) if parts else None
    return None


def buffer_reach_corridor(
    reaches: list[dict[str, Any]],
    half_width_m: Any,
    ref_lat: float,
    clip: Any = None,
) -> Any:
    """Metric buffer of reach lines (local equirectangular scaling), optionally clipped."""
    cos_lat = max(math.cos(math.radians(ref_lat)), 0.05)
    parts = []
    for r in reaches:
        hw = half_width_m(r) if callable(half_width_m) else float(half_width_m)
        if hw <= 0:
            continue
        scaled = affinity.scale(
            r['geometry'], xfact=cos_lat, yfact=1.0, origin=(0, 0)
        )
        parts.append(scaled.buffer(hw / 111320.0, quad_segs=4))
    if not parts:
        return None
    geom = affinity.scale(
        unary_union(parts), xfact=1.0 / cos_lat, yfact=1.0, origin=(0, 0)
    )
    if clip is not None:
        geom = geom.intersection(clip)
    geom = polygonal_only(geom)
    return (
        polygonal_only(geom.simplify(0.0002, preserve_topology=True))
        if geom is not None
        else None
    )


def chain_length_km(
    reaches: list[dict[str, Any]],
    ref_lat: float,
    clip: Any = None,
) -> float:
    """Compute length in km of a reach chain under local equirectangular scaling."""
    if not reaches:
        return 0.0
    lines = unary_union([r['geometry'] for r in reaches])
    if clip is not None:
        lines = lines.intersection(clip)
    cos_lat = max(math.cos(math.radians(ref_lat)), 0.05)
    return (
        affinity.scale(lines, xfact=cos_lat, yfact=1.0, origin=(0, 0)).length
        * 111.32
    )


def route_floodplain_excess(
    series: list[float],
    q_bankfull: float,
    k: float = CAMA_FLOODPLAIN_K,
) -> list[float]:
    """Linear-reservoir routing of above-bankfull flow (daily explicit scheme)."""
    routed: list[float] = []
    state: float | None = None
    for q in series:
        excess = max(q - q_bankfull, 0.0)
        state = excess if state is None else state + k * (excess - state)
        routed.append(state)
    return routed


def emulate_camaflood_physics(
    glofas_records: list[dict[str, Any]],
    rps: dict[str, Any],
    elev: float = 80.0,
    elev_source: str = 'Open-Meteo DEM',
) -> dict[str, Any]:
    """Deterministic CaMa-Flood-style streamflow + inundation emulation from GloFAS v4."""
    records = (glofas_records or [])[:6]

    def _col(name: str, fallback: str = 'discharge_mean') -> list[float]:
        out: list[float] = []
        for r in records:
            v = parse_finite_float(r.get(name))
            if v is None:
                v = parse_finite_float(r.get(fallback))
            out.append(max(v or 0.0, 0.0))
        return out

    central = _col('discharge_median')
    med_val = float(np.median(central)) if central else 1.0
    q_clim = parse_finite_float((rps or {}).get('mean_flow')) or med_val or 1.0
    q_clim = max(q_clim, 0.05)
    width = max(0.40 * q_clim**0.75, 10.0)
    depth = max(0.10 * q_clim**0.5, 1.0)
    q_bf = max(
        gumbel_quantile_from_return_periods(rps or {}, 1.5) or 0.0,
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
        total = [min(q, q_bf) + r for q, r in zip(series, routed)]
        fld = [CAMA_FLDOUT_SHARE * r for r in routed]
        return total, [t - f for t, f in zip(total, fld)], fld

    total, rivout, fldout = _cama(central)
    stage = [depth * (max(r, 0.0) / q_bf) ** 0.6 for r in rivout]
    flddph = [max(h - depth, 0.0) for h in stage]
    fldfrc = [100.0 * f_max * (1.0 - math.exp(-d / depth_scale)) for d in flddph]
    sfcelv = [max(max(float(elev), 0.0) - depth + h, 0.0) for h in stage]
    r2 = lambda xs: [round(x, 2) for x in xs]
    return {
        'series': {
            'timestamps': [
                f"{str(r.get('time'))[:10]}T00:00:00Z" for r in records
            ],
            'mean': r2(total),
            'rivout': r2(rivout),
            'fldout': r2(fldout),
            'p25': r2(_cama(_col('discharge_p25'))[0]),
            'p75': r2(_cama(_col('discharge_p75'))[0]),
            'max': r2(_cama(_col('discharge_max'))[0]),
            'min': r2(_cama(_col('discharge_min'))[0]),
            'flddph_m': [round(d, 3) for d in flddph],
            'fldfrc_pct': r2(fldfrc),
            'sfcelv_m': r2(sfcelv),
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
        'forcing_status': 'live',
        'return_period_status': (rps or {}).get('status'),
    }


def camaflood_unit_feature(
    cell_lat: float,
    cell_lon: float,
    te_fc: dict[str, Any],
) -> dict[str, Any]:
    """Build the GeoJSON unit-cell feature for a CaMa-Flood 0.25 deg cell."""
    cell_ring, _ = cama_cell_polygon(cell_lat, cell_lon)
    cell_id = cama_cell_id(cell_lat, cell_lon)
    te_ff = te_fc.get('flood_forecast') or {}
    peak_depth = parse_float_or_default(te_ff.get('max_flood_depth_m'), 0.0)
    peak_frac = parse_float_or_default(
        te_ff.get('max_flooded_fraction_pct'), 0.0
    )
    cell_area = cama_cell_area_km2(cell_lat)
    te_source = te_fc.get('source', TODAYS_EARTH_SOURCE) + (
        ' — emulated' if te_fc.get('emulated') else ''
    )
    return {
        'type': 'Feature',
        'geometry': {'type': 'Polygon', 'coordinates': [cell_ring]},
        'properties': {
            'layer': 'camaflood_depth',
            'feature_role': 'unit_cell',
            'provider': "JAXA Today's Earth",
            'source': te_source,
            'status': te_fc.get('status'),
            'emulated': te_fc.get('emulated'),
            'grid_cell_id': cell_id,
            'label': f'CaMa-Flood unit cell {cell_id}',
            'peak_flood_depth_m': round(peak_depth, 3),
            'peak_flooded_fraction_pct': round(peak_frac, 2),
            'peak_sfcelv_m': te_ff.get('max_sfcelv_m'),
            'peak_depth_time': te_ff.get('peak_depth_time'),
            'cell_area_km2': cell_area,
            'area_km2': cell_area,
            'flooded_area_km2': round(peak_frac / 100.0 * cell_area, 2),
            'color': depth_color(peak_depth),
        },
    }


def geojson_feature(geom: Any, props: dict[str, Any]) -> dict[str, Any]:
    """Wrap a Shapely geometry into a rounded GeoJSON Feature with area_km2."""
    props_out = dict(props)
    props_out.setdefault('area_km2', geom_area_km2(geom))
    return {
        'type': 'Feature',
        'geometry': {
            'type': geom.geom_type,
            'coordinates': round_geojson_coords(mapping(geom)['coordinates']),
        },
        'properties': props_out,
    }
