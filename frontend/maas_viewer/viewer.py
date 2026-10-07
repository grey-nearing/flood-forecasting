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

"""High-level frontend viewer combining `maas.MaaSDataFetcher` with UI presentation layers."""

from datetime import UTC, datetime
import math
import time
from typing import Any
import urllib.parse

from shapely.geometry import Polygon, mapping

from frontend.maas_viewer.consensus import (
    build_aligned_timeline,
    build_consensus_row,
    build_flood_summary,
    reach_exceedance_summary,
    spread_confidence,
)
from frontend.maas_viewer.inundation import (
    buffer_reach_corridor,
    camaflood_unit_feature,
    chain_length_km,
    channel_half_width_m,
    depth_color,
    emulate_camaflood_physics,
    geojson_feature,
)
from maas.config import (
    CAMA_GRID_RES_DEG,
    TODAYS_EARTH_SOURCE,
    MaaSConfig,
    normalize_requested_models,
    parse_float_or_default,
)
from maas.fetcher import MaaSDataFetcher, daily_series, window_peak
from maas.floodhub import (
    FH_LEVEL_COLORS,
    FH_LEVEL_LABELS,
    FH_SEVERITY_LABELS,
    FH_SEVERITY_RANK,
    FH_SEVERITY_TO_RISK,
)
from maas.networks import (
    cama_cell_area_km2,
    cama_cell_id,
    cama_cell_polygon,
    glofas_cell_polygon,
    is_geoglows_river_id,
    query_hydrorivers_reaches,
    snap_cama_cell,
    trace_main_stem_chain,
)
from maas.thresholds import (
    EXCEEDANCE_CLASSES,
    thresholds_from_return_periods,
)
from maas.todays_earth import format_todays_earth_forecast

CORRIDOR_WIDTH_FACTOR: tuple[float, ...] = (1.0, 3.0, 5.0, 8.0, 10.0)
FH_DERIVED_WIDTH_FACTOR: tuple[float, ...] = (0.0, 4.0, 7.0, 10.0)


class MaaSViewer:
    """Frontend presenter that fetches raw `maas` data and builds UI visualization payloads."""

    def __init__(
        self,
        config: MaaSConfig,
        fetcher: MaaSDataFetcher | None = None,
    ) -> None:
        self.config = config
        self.fetcher = fetcher if fetcher is not None else MaaSDataFetcher(config)

    def todays_earth_service_status(self) -> str:
        """Return `'operational'` if `todays_earth_api_url` is configured, else `'emulated'`."""
        return (
            'operational'
            if self.config.todays_earth_api_url.strip()
            else 'emulated'
        )

    def render_forecast_view(  # noqa: PLR0913
        self,
        lat: float,
        lon: float,
        gauge_id: str | None = None,
        river_id: int | None = None,
        reach_id: str | None = None,
        requested_models: list[str] | None = None,
        *,
        upstream_area_km2: float | str | None = None,
        area_min_km2: float | str | None = None,
        network: str | None = None,
    ) -> dict[str, Any]:
        """Fetch raw multi-model forecasts and enrich with UI timeline, consensus, and inundation."""
        lat, lon = float(lat), float(lon)
        models = normalize_requested_models(requested_models)
        raw_bundle = self.fetcher.fetch_forecasts(
            lat=lat,
            lon=lon,
            gauge_id=gauge_id,
            river_id=river_id,
            reach_id=reach_id,
            requested_models=models,
            upstream_area_km2=upstream_area_km2,
            area_min_km2=area_min_km2,
            network=network,
        )
        models_out = dict(raw_bundle['models'])
        gl_fc = models_out.get('glofas')
        gg_fc = models_out.get('geoglows')
        fh_fc = models_out.get('floodhub')
        te_fc = models_out.get('todays_earth')
        gl_rp = (gl_fc or {}).get('return_periods') or raw_bundle[
            'return_periods'
        ].get('glofas')
        gg_rp = (gg_fc or {}).get('return_periods') or raw_bundle[
            'return_periods'
        ].get('geoglows')

        # If Today's Earth live STAC data was unavailable, run visual CaMa-Flood emulation
        if (
            'todays_earth' in models
            and (not te_fc or not te_fc.get('available'))
            and gl_fc
            and gl_fc.get('data')
        ):
            gl_cell = raw_bundle['reaches']['glofas']
            emu_lat = float(gl_cell.get('cell_center_lat') or lat)
            emu_lon = float(gl_cell.get('cell_center_lon') or lon)
            emu = emulate_camaflood_physics(
                gl_fc.get('data') or [],
                gl_rp or {},
            )
            te_fc = format_todays_earth_forecast(
                emu_lat,
                emu_lon,
                emu['series'],
                reach_id=reach_id,
                live=False,
                channel_params=emu['channel_params'],
                forcing_status=gl_fc.get('status'),
            )
            models_out['todays_earth'] = te_fc

        fh_is_q = (
            str((fh_fc or {}).get('unit') or 'CUBIC_METERS_PER_SECOND').upper()
            == 'CUBIC_METERS_PER_SECOND'
        )
        fh_th = (fh_fc or {}).get('thresholds') or {}

        consensus: list[dict[str, Any]] = []
        if 'floodhub' in models:
            live = bool(fh_fc and fh_fc.get('available') and fh_fc.get('status') == 'live')
            severity = (fh_fc or {}).get('severity')
            sev_risk = FH_SEVERITY_TO_RISK.get(str(severity or ''))
            sev_live = (
                bool(sev_risk)
                and (fh_fc or {}).get('severity_source') == 'floodhub_flood_status'
            )
            if live or sev_live:
                fh_daily = daily_series((fh_fc or {}).get('data'), 'discharge')
                peak, when = window_peak(fh_daily) if live else (None, None)
                fh_rps = (
                    {
                        'return_period_2': fh_th.get('warning_2yr'),
                        'return_period_5': fh_th.get('danger_5yr'),
                        'return_period_20': fh_th.get('extreme_20yr'),
                    }
                    if live
                    else None
                )
                row = build_consensus_row(
                    'floodhub',
                    True,
                    fh_fc.get('status') if live else 'severity_only',
                    peak,
                    when,
                    fh_rps,
                    'FloodHub gauge model thresholds'
                    if live
                    else 'FloodHub official flood status',
                    'High'
                    if (fh_fc or {}).get('quality_verified', True)
                    else 'Medium',
                    unit='m³/s' if fh_is_q else 'm',
                    value_type='discharge' if fh_is_q else 'stage',
                    severity=severity,
                    trend=(fh_fc or {}).get('trend'),
                )
                if sev_risk:
                    row['risk_level'] = sev_risk
                    row['risk_source'] = 'floodhub_severity'
                    if not live and sev_live:
                        rank = FH_SEVERITY_RANK.get(str(severity), 0)
                        row.update(
                            exceedance_rank=rank,
                            exceedance_label=FH_SEVERITY_LABELS.get(
                                str(severity), str(severity)
                            ),
                            return_period=None,
                            return_period_yrs=None,
                        )
                consensus.append(row)
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
            if gl_fc and gl_fc.get('available'):
                gl_daily = daily_series(
                    gl_fc.get('data'), 'discharge_median', 'discharge_mean'
                )
                peak, when = window_peak(gl_daily)
                p25 = daily_series(gl_fc.get('data'), 'discharge_p25').get(
                    when or ''
                )
                p75 = daily_series(gl_fc.get('data'), 'discharge_p75').get(
                    when or ''
                )
                consensus.append(
                    build_consensus_row(
                        'glofas',
                        True,
                        gl_fc.get('status'),
                        peak,
                        when,
                        gl_rp,
                        (gl_rp or {}).get('source'),
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
                gg_daily = daily_series(
                    gg_fc.get('data'), 'flow_med', 'flow_avg'
                )
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
                        gg_fc.get('status'),
                        peak,
                        when,
                        gg_rp,
                        (gg_rp or {}).get('source'),
                        spread_confidence(peak, p25, p75),
                        thresholds_status=(gg_rp or {}).get('status'),
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

        te_ff = (te_fc or {}).get('flood_forecast') or {}
        if 'todays_earth' in models:
            if te_fc and te_fc.get('available'):
                te_daily = daily_series(te_fc.get('data'), 'discharge_mean')
                peak, when = window_peak(te_daily)
                consensus.append(
                    build_consensus_row(
                        'todays_earth',
                        True,
                        'emulated'
                        if te_fc.get('emulated')
                        else te_fc.get('status'),
                        peak,
                        when,
                        gl_rp,
                        'GloFAS v4 reanalysis EV1',
                        'Low' if te_fc.get('emulated') else 'Medium',
                        independent=not bool(te_fc.get('emulated')),
                        emulated=bool(te_fc.get('emulated')),
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

        timeline = build_aligned_timeline(
            models, fh_fc, fh_is_q, gl_fc, gg_fc, te_fc
        )
        flood_summary = build_flood_summary(consensus, te_fc, fh_fc, fh_fc)
        exceedance = reach_exceedance_summary(gl_fc, gl_rp, gg_fc, gg_rp)

        inundation_features: list[dict[str, Any]] = []
        if te_fc and te_fc.get('available'):
            c_lat, c_lon = snap_cama_cell(lat, lon)
            inundation_features.append(
                camaflood_unit_feature(c_lat, c_lon, te_fc)
            )

        vs = dict(raw_bundle['virtual_station'])
        if te_fc and te_fc.get('available'):
            vs['todays_earth_cell'] = {
                **vs.get('todays_earth_cell', {}),
                'grid_cell_id': te_fc.get('grid_cell_id'),
                'cell_center_lat': te_fc.get('cell_center_lat'),
                'cell_center_lon': te_fc.get('cell_center_lon'),
                'resolution_deg': CAMA_GRID_RES_DEG,
                'area_km2': te_fc.get('cell_area_km2'),
                'status': te_fc.get('status'),
                'emulated': te_fc.get('emulated'),
                'label': (
                    f"CaMa 0.25° [{float(te_fc.get('cell_center_lat') or 0.0):.3f}, "
                    f"{float(te_fc.get('cell_center_lon') or 0.0):.3f}]"
                ),
                'model_chain': 'MATSIRO + CaMa-Flood',
            }

        eff_gid = raw_bundle['location'].get('gauge_id')
        eff_rid = raw_bundle['location'].get('river_id')
        inund_params: dict[str, Any] = {'lat': lat, 'lon': lon}
        if eff_gid:
            inund_params['gauge_id'] = eff_gid
        if reach_id:
            inund_params['reach_id'] = reach_id
        if is_geoglows_river_id(eff_rid):
            inund_params['river_id'] = eff_rid

        flood_inundation = {
            'endpoint': '/api/maas/flood-inundation?'
            + urllib.parse.urlencode(inund_params),
            'layers': ['floodhub_extent', 'camaflood_depth', 'reach_exceedance'],
            'floodhub': {
                'status': (fh_fc or {}).get('status'),
                'gauge_id': eff_gid,
                'severity': (fh_fc or {}).get('severity'),
                'trend': (fh_fc or {}).get('trend'),
                'severity_source': (fh_fc or {}).get('severity_source'),
                'issued_time': (fh_fc or {}).get('issued_time'),
                'inundation_maps_available': bool(
                    (fh_fc or {}).get('inundation_maps_available')
                ),
                'inundation_map_levels': (fh_fc or {}).get(
                    'inundation_map_levels'
                )
                or [],
                'inundation_maps_time_range': (fh_fc or {}).get(
                    'inundation_maps_time_range'
                ),
            },
            'todays_earth': {
                'grid_cell_id': (te_fc or {}).get('grid_cell_id'),
                'status': (te_fc or {}).get('status'),
                'emulated': (te_fc or {}).get('emulated'),
                'max_flood_depth_m': te_ff.get('max_flood_depth_m'),
                'max_flooded_fraction_pct': te_ff.get(
                    'max_flooded_fraction_pct'
                ),
                'max_sfcelv_m': te_ff.get('max_sfcelv_m'),
                'peak_depth_time': te_ff.get('peak_depth_time'),
            },
            'reach_exceedance': exceedance,
        }

        thresholds_by_model = {
            'floodhub': (
                {**fh_th, 'unit': (fh_fc or {}).get('unit')}
                if (fh_fc and fh_fc.get('status') == 'live')
                else None
            ),
            'glofas': (
                thresholds_from_return_periods(
                    gl_rp, str(gl_rp.get('source') or 'GloFAS v4')
                )
                if gl_rp
                else None
            ),
            'geoglows': (
                thresholds_from_return_periods(
                    gg_rp, str(gg_rp.get('source') or 'GEOGLOWS v2')
                )
                if gg_rp
                else None
            ),
            'todays_earth': (
                thresholds_from_return_periods(
                    gl_rp,
                    'GloFAS v4 reanalysis EV1 (emulator climatology)',
                )
                if (te_fc and gl_rp)
                else None
            ),
        }

        meta = {
            **raw_bundle['meta'],
            'todays_earth_service': self.todays_earth_service_status(),
        }

        return {
            **raw_bundle,
            'models': models_out,
            'virtual_station': vs,
            'thresholds_by_model': thresholds_by_model,
            'timeline': timeline,
            'consensus': consensus,
            'flood_summary': flood_summary,
            'reach_exceedance': exceedance,
            'flood_inundation': flood_inundation,
            'inundation': {
                'type': 'FeatureCollection',
                'features': inundation_features,
            },
            'meta': meta,
        }

    def render_inundation_view(
        self,
        lat: float,
        lon: float,
        gauge_id: str | None = None,
        reach_id: str | None = None,
        river_id: int | str | None = None,
    ) -> dict[str, Any]:
        """Build the unified 3-layer spatial flood-inundation GeoJSON FeatureCollection."""
        lat, lon = float(lat), float(lon)
        t0 = time.time()
        cell_lat, cell_lon = snap_cama_cell(lat, lon)
        cell_ring, cell_bbox = cama_cell_polygon(cell_lat, cell_lon)
        cell_id = cama_cell_id(cell_lat, cell_lon)
        pad = 0.03
        q_bbox = (
            min(cell_bbox['min_lon'], lon - 0.1) - pad,
            min(cell_bbox['min_lat'], lat - 0.1) - pad,
            max(cell_bbox['max_lon'], lon + 0.1) + pad,
            max(cell_bbox['max_lat'], lat + 0.1) + pad,
        )

        shp_path = (
            self.config.river_networks_dir
            / 'hydrorivers'
            / 'HydroRIVERS_v10.shp'
        )
        if not shp_path.exists():
            shp_path = (
                self.config.river_networks_dir
                / 'HydroRIVERS_v10_shp'
                / 'HydroRIVERS_v10.shp'
            )
        reaches = (
            query_hydrorivers_reaches(shp_path, *q_bbox)
            if shp_path.exists()
            else []
        )
        start, chain, snap_km = trace_main_stem_chain(
            reaches, lat, lon, reach_id=reach_id
        )

        view = self.render_forecast_view(
            lat,
            lon,
            gauge_id=gauge_id,
            river_id=int(river_id) if is_geoglows_river_id(river_id) else None,
            reach_id=reach_id,
        )
        models_out = view['models']
        te = models_out.get('todays_earth') or {}
        gg_fc = models_out.get('geoglows') or {}
        exceedance = view['reach_exceedance']

        eff_gid = view['location'].get('gauge_id') or gauge_id
        fh = self.fetcher.floodhub.fetch_inundation(
            eff_gid,
            lat,
            lon,
            include_polygons=True,
            forecast=models_out.get('floodhub'),
        )

        cell_poly = Polygon(cell_ring)
        te_ff = te.get('flood_forecast') or {}
        peak_depth = parse_float_or_default(te_ff.get('max_flood_depth_m'), 0.0)
        peak_frac = parse_float_or_default(
            te_ff.get('max_flooded_fraction_pct'), 0.0
        )
        cell_area = cama_cell_area_km2(cell_lat)
        te_source = te.get('source', TODAYS_EARTH_SOURCE) + (
            ' — emulated' if te.get('emulated') else ''
        )

        features: list[dict[str, Any]] = [
            camaflood_unit_feature(cell_lat, cell_lon, te)
        ]

        if chain:
            rank = int(exceedance.get('rank') or 0)
            factor = CORRIDOR_WIDTH_FACTOR[
                min(max(rank, 0), len(CORRIDOR_WIDTH_FACTOR) - 1)
            ]
            geom = buffer_reach_corridor(
                chain,
                lambda r: channel_half_width_m(r['mean_discharge_m3s']) * factor,
                lat,
            )
            if geom is not None:
                features.append(
                    geojson_feature(
                        geom,
                        {
                            'layer': 'reach_exceedance',
                            'provider': 'GEOGLOWS / GloFAS',
                            'source': 'HydroRIVERS main stem styled by forecast-peak return-period exceedance',
                            'label': f"Reach exceedance — {exceedance['label']}",
                            'exceedance_rank': exceedance['rank'],
                            'exceedance_label': exceedance['label'],
                            'risk_level': exceedance['risk_level'],
                            'return_period': exceedance['return_period'],
                            'governing_model': exceedance.get('governing_model'),
                            'per_model': exceedance.get('per_model'),
                            'reach_count': len(chain),
                            'hydrorivers_reach': (
                                f"HYRIV_{start['hyriv_id']}" if start else None
                            ),
                            'color': exceedance['color'],
                        },
                    )
                )

        if chain and peak_frac >= 0.1:
            length_km = chain_length_km(chain, lat, clip=cell_poly)
            if length_km >= 0.5:
                target_km2 = peak_frac / 100.0 * cell_area
                half_width_km = min(
                    max(target_km2 / (2.0 * length_km), 0.05), 12.0
                )
                geom = buffer_reach_corridor(
                    chain, half_width_km * 1000.0, lat, clip=cell_poly
                )
                if geom is not None:
                    features.append(
                        geojson_feature(
                            geom,
                            {
                                'layer': 'camaflood_depth',
                                'feature_role': 'floodplain',
                                'provider': "JAXA Today's Earth",
                                'source': te_source,
                                'status': te.get('status'),
                                'emulated': te.get('emulated'),
                                'grid_cell_id': cell_id,
                                'label': 'CaMa-Flood forecast floodplain inundation',
                                'peak_flood_depth_m': round(peak_depth, 3),
                                'peak_flooded_fraction_pct': round(
                                    peak_frac, 2
                                ),
                                'target_flooded_area_km2': round(target_km2, 2),
                                'peak_depth_time': te_ff.get('peak_depth_time'),
                                'color': depth_color(max(peak_depth, 0.01)),
                            },
                        )
                    )

        fh_polys = list(fh.get('inundation_polygons') or [])
        features.extend(fh_polys)

        fh_rank = int(fh.get('severity_rank') or 0)
        fh_derived = False
        if not fh_polys and fh_rank >= 1 and chain:
            factor = FH_DERIVED_WIDTH_FACTOR[min(fh_rank, 3)]
            geom = buffer_reach_corridor(
                chain,
                lambda r: channel_half_width_m(r['mean_discharge_m3s']) * factor,
                lat,
            )
            if geom is not None:
                fh_derived = True
                features.append(
                    geojson_feature(
                        geom,
                        {
                            'layer': 'floodhub_extent',
                            'provider': 'Google FloodHub',
                            'source': 'River corridor buffered by FloodHub severity (no official inundation map)',
                            'derived': True,
                            'gauge_id': fh.get('gauge_id'),
                            'probability_level': None,
                            'label': f"FloodHub {fh.get('severity')} zone (derived)",
                            'severity': fh.get('severity'),
                            'color': '#1a73e8',
                        },
                    )
                )

        counts: dict[str, int] = {}
        for f in features:
            layer = f['properties']['layer']
            counts[layer] = counts.get(layer, 0) + 1

        fh_summary = {
            k: v for k, v in fh.items() if k != 'inundation_polygons'
        }
        te_summary = {
            'grid_cell_id': cell_id,
            'status': te.get('status'),
            'emulated': te.get('emulated'),
            'source': te.get('source'),
            'note': te.get('note'),
            'max_flood_depth_m': te_ff.get('max_flood_depth_m'),
            'max_flooded_fraction_pct': te_ff.get('max_flooded_fraction_pct'),
            'max_sfcelv_m': te_ff.get('max_sfcelv_m'),
            'peak_depth_time': te_ff.get('peak_depth_time'),
            'cell_area_km2': cell_area,
        }
        return {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'lat': lat,
                'lon': lon,
                'gauge_id': fh.get('gauge_id') or eff_gid,
                'river_id': gg_fc.get('river_id'),
                'reach_id': f"HYRIV_{start['hyriv_id']}" if start else reach_id,
                'generated_at': datetime.now(UTC).strftime(
                    '%Y-%m-%dT%H:%M:%SZ'
                ),
                'elapsed_s': round(time.time() - t0, 2),
                'layers': {
                    'floodhub_extent': {
                        'count': counts.get('floodhub_extent', 0),
                        'status': fh.get('status'),
                        'severity': fh.get('severity'),
                        'trend': fh.get('trend'),
                        'official_maps': bool(fh_polys),
                        'derived': fh_derived,
                    },
                    'camaflood_depth': {
                        'count': counts.get('camaflood_depth', 0),
                        **te_summary,
                    },
                    'reach_exceedance': {
                        'count': counts.get('reach_exceedance', 0),
                        **exceedance,
                    },
                },
                'floodhub': fh_summary,
                'todays_earth': te_summary,
                'reach_exceedance': exceedance,
                'corridor': {
                    'dataset': 'HydroRIVERS v1.0',
                    'hydrorivers_reach': (
                        f"HYRIV_{start['hyriv_id']}" if start else None
                    ),
                    'reach_count': len(chain),
                    'snap_distance_km': snap_km,
                },
                'legend': {
                    'floodhub_extent': [
                        {
                            'label': FH_LEVEL_LABELS[k],
                            'color': FH_LEVEL_COLORS[k],
                        }
                        for k in ('HIGH', 'MEDIUM', 'LOW')
                    ],
                    'reach_exceedance': [
                        {'label': c['label'], 'color': c['color']}
                        for c in EXCEEDANCE_CLASSES
                    ],
                    'camaflood_depth': [
                        {'label': '0 m', 'color': depth_color(0.0)},
                        {'label': '< 0.5 m', 'color': depth_color(0.25)},
                        {'label': '0.5-1 m', 'color': depth_color(0.75)},
                        {'label': '1-2 m', 'color': depth_color(1.5)},
                        {'label': '> 2 m', 'color': depth_color(2.5)},
                    ],
                },
            },
        }

    def render_watershed_polygon(  # noqa: PLR0913
        self,
        lat: float,
        lon: float,
        fabric: str = 'hydroatlas_full',
        gauge_id: str | None = None,
        river_id: int | None = None,
        geofabric: str | None = None,
    ) -> dict[str, Any]:
        """Resolve the watershed polygon corresponding to the selected hydrofabric."""
        lat, lon = float(lat), float(lon)
        eff_fabric = (geofabric or fabric or 'hydroatlas_full').strip().lower()

        if eff_fabric == 'camaflood_unit':
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
                    'service_status': self.todays_earth_service_status(),
                    'grid_cell_id': cama_cell_id(cell_lat, cell_lon),
                    'cell_center_lat': cell_lat,
                    'cell_center_lon': cell_lon,
                    'area_km2': cama_cell_area_km2(cell_lat),
                    'resolution': '0.25° (~28 km)',
                    'bbox': bbox,
                },
            }

        if eff_fabric == 'glofas_cell':
            return glofas_cell_polygon(lat, lon)

        try:
            from frontend.delineator import HydroDelineator, _find_merit_shp
        except ImportError:
            try:
                from delineator import HydroDelineator, _find_merit_shp  # type: ignore[no-redef]
            except ImportError:
                HydroDelineator = None  # type: ignore[assignment]
                _find_merit_shp = None  # type: ignore[assignment]

        if eff_fabric == 'merit_reach':
            eff_rid = river_id
            if not eff_rid:
                try:
                    eff_rid = self.fetcher.geoglows.fetch_river_id(lat, lon)
                except Exception:  # noqa: BLE001
                    eff_rid = None
            if eff_rid:
                cache_key = f'merit_reach_{eff_rid}'
                cached = self.fetcher.watershed_cache.get(
                    cache_key, max_age_s=365 * 86400
                )
                if cached is not None:
                    return cached
                if _find_merit_shp is not None:
                    cat_shp = _find_merit_shp(eff_rid, 'cat')
                    if cat_shp and cat_shp.exists():
                        try:
                            import pyogrio

                            df = pyogrio.read_dataframe(
                                str(cat_shp), where=f'COMID = {eff_rid}'
                            )
                            if not df.empty:
                                geom = df.geometry.values[0]
                                area_col = (
                                    'unitarea'
                                    if 'unitarea' in df.columns
                                    else (
                                        'uparea'
                                        if 'uparea' in df.columns
                                        else None
                                    )
                                )
                                area_km2 = (
                                    float(df[area_col].values[0])
                                    if area_col
                                    else float(
                                        geom.area
                                        * 111.0
                                        * 111.0
                                        * math.cos(math.radians(lat))
                                    )
                                )
                                feature = {
                                    'type': 'Feature',
                                    'geometry': mapping(geom),
                                    'properties': {
                                        'fabric': 'merit_reach',
                                        'fabric_name': f'MERIT-Hydro Reach Catchment (COMID {eff_rid})',
                                        'model': 'GEOGLOWS ECMWF',
                                        'comid': eff_rid,
                                        'area_km2': round(area_km2, 1),
                                        'resolution': '90m MERIT-Basins',
                                    },
                                }
                                self.fetcher.watershed_cache.put(
                                    cache_key, feature
                                )
                                return feature
                        except Exception:  # noqa: BLE001
                            pass
            if HydroDelineator is not None:
                try:
                    delin = HydroDelineator('merit-hydro')
                    res = delin.delineate_catchment(
                        lat, lon, mode='unit_catchment'
                    )
                    props = res.get('properties', {})
                    area_km2 = float(props.get('area_km2', 25.0))
                    feature = {
                        'type': 'Feature',
                        'geometry': res['geometry'],
                        'properties': {
                            'fabric': 'merit_reach',
                            'fabric_name': f"MERIT Reach Catchment ({props.get('catchment_id', 'Reach')})",
                            'model': 'GEOGLOWS ECMWF',
                            'comid': eff_rid or 0,
                            'area_km2': round(area_km2, 1),
                            'resolution': '90m MERIT-Basins',
                        },
                    }
                    if eff_rid:
                        self.fetcher.watershed_cache.put(
                            f'merit_reach_{eff_rid}', feature
                        )
                    return feature
                except Exception:  # noqa: BLE001
                    pass

        cache_key = f"{eff_fabric}_{gauge_id or f'{lat:.4f}_{lon:.4f}'}"
        cached = self.fetcher.watershed_cache.get(
            cache_key, max_age_s=365 * 86400
        )
        if cached is not None:
            return cached

        if HydroDelineator is not None:
            try:
                delin = HydroDelineator('hydroatlas')
                mode = (
                    'unit_catchment'
                    if eff_fabric == 'hydroatlas_unit'
                    else 'official_ridgeline'
                )
                res = delin.delineate_catchment(lat, lon, mode=mode)
                props = res.get('properties', {})
                r_attrs = props.get('reach_attributes', {})
                hybas_id = r_attrs.get('hydrobasins_unit', gauge_id or '')
                area_km2 = float(props.get('area_km2', 0.0))
                is_full = eff_fabric == 'hydroatlas_full'
                feat_name = (
                    f"HydroATLAS Full Drainage Area ({hybas_id or 'Basin'})"
                    if is_full
                    else f"HydroATLAS Level 12 Unit Catchment ({hybas_id or 'Unit'})"
                )
                feature = {
                    'type': 'Feature',
                    'geometry': res['geometry'],
                    'properties': {
                        'fabric': eff_fabric,
                        'fabric_name': feat_name,
                        'model': 'Google FloodHub',
                        'hybas_id': hybas_id,
                        'area_km2': round(area_km2, 1),
                        'resolution': '15 arc-sec HydroATLAS',
                        'upstream_count': props.get(
                            'upstream_reaches_count', 1
                        ),
                    },
                }
                self.fetcher.watershed_cache.put(cache_key, feature)
                return feature
            except Exception:  # noqa: BLE001
                pass

        return {
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': []},
            'properties': {
                'fabric': eff_fabric,
                'available': False,
                'status': 'unavailable',
                'area_km2': 0.0,
            },
        }
