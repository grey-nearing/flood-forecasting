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

"""Frontend viewer combining `MaaSDataFetcher` with UI presentation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pyogrio
from shapely.geometry import mapping

from frontend.delineator import HydroDelineator, find_merit_shp
from frontend.maas_viewer.consensus import (
    build_aligned_timeline,
    build_consensus_row,
    build_flood_summary,
    reach_exceedance_summary,
    spread_confidence,
)
from maas.config import (
    MaaSConfig,
    normalize_requested_models,
)
from maas.fetcher import MaaSDataFetcher, daily_series, window_peak
from maas.floodhub import (
    FH_SEVERITY_LABELS,
    FH_SEVERITY_RANK,
    FH_SEVERITY_TO_RISK,
)
from maas.networks import glofas_cell_polygon
from maas.thresholds import thresholds_from_return_periods

if TYPE_CHECKING:
    from pathlib import Path

_WATERSHED_CACHE_MAX_AGE_S: float = 365.0 * 86400.0
_KM_PER_DEG: float = 111.0


class MaaSViewer:
    """Frontend presenter that fetches `maas` data and builds UI payloads."""

    def __init__(
        self,
        config: MaaSConfig,
        fetcher: MaaSDataFetcher | None = None,
    ) -> None:
        """Initialize the viewer with a `MaaSConfig` and optional fetcher."""
        self.config = config
        self.fetcher = (
            fetcher if fetcher is not None else MaaSDataFetcher(config)
        )

    def _build_floodhub_consensus_row(
        self,
        fh_fc: dict[str, Any] | None,
        fh_th: dict[str, Any],
        *,
        fh_is_q: bool,
    ) -> dict[str, Any]:
        """Build the FloodHub row for the consensus table."""
        live = bool(
            fh_fc and fh_fc.get('available') and fh_fc.get('status') == 'live'
        )
        severity = (fh_fc or {}).get('severity')
        sev_risk = FH_SEVERITY_TO_RISK.get(str(severity or ''))
        sev_live = (
            bool(sev_risk)
            and (fh_fc or {}).get('severity_source') == 'floodhub_flood_status'
        )
        if not (live or sev_live):
            return build_consensus_row(
                'floodhub',
                'unavailable',
                None,
                None,
                None,
                None,
                'N/A',
                available=False,
            )

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
        status_val = (fh_fc or {}).get('status') if live else 'severity_only'
        th_source = (
            'FloodHub gauge model thresholds'
            if live
            else 'FloodHub official flood status'
        )
        conf = (
            'High' if (fh_fc or {}).get('quality_verified', True) else 'Medium'
        )
        row = build_consensus_row(
            'floodhub',
            status_val,
            peak,
            when,
            fh_rps,
            th_source,
            conf,
            unit='m³/s' if fh_is_q else 'm',
            available=True,
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
        return row

    def render_forecast_view(  # noqa: PLR0913, PLR0917
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
        """Fetch multi-model forecasts and enrich with timeline/consensus."""
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
        gl_rp = (gl_fc or {}).get('return_periods') or raw_bundle[
            'return_periods'
        ].get('glofas')
        gg_rp = (gg_fc or {}).get('return_periods') or raw_bundle[
            'return_periods'
        ].get('geoglows')

        fh_is_q = (
            str((fh_fc or {}).get('unit') or 'CUBIC_METERS_PER_SECOND').upper()
            == 'CUBIC_METERS_PER_SECOND'
        )
        fh_th = (fh_fc or {}).get('thresholds') or {}

        consensus: list[dict[str, Any]] = []
        if 'floodhub' in models:
            consensus.append(
                self._build_floodhub_consensus_row(
                    fh_fc, fh_th, fh_is_q=fh_is_q
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
                        gl_fc.get('status'),
                        peak,
                        when,
                        gl_rp,
                        (gl_rp or {}).get('source'),
                        spread_confidence(peak, p25, p75),
                        available=True,
                    )
                )
            else:
                consensus.append(
                    build_consensus_row(
                        'glofas',
                        'unavailable',
                        None,
                        None,
                        None,
                        None,
                        'N/A',
                        available=False,
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
                        gg_fc.get('status'),
                        peak,
                        when,
                        gg_rp,
                        (gg_rp or {}).get('source'),
                        spread_confidence(peak, p25, p75),
                        available=True,
                        thresholds_status=(gg_rp or {}).get('status'),
                    )
                )
            else:
                consensus.append(
                    build_consensus_row(
                        'geoglows',
                        'unavailable',
                        None,
                        None,
                        None,
                        None,
                        'N/A',
                        available=False,
                    )
                )

        timeline = build_aligned_timeline(models, fh_fc, fh_is_q, gl_fc, gg_fc)
        flood_summary = build_flood_summary(consensus, fh_fc, fh_fc)
        exceedance = reach_exceedance_summary(gl_fc, gl_rp, gg_fc, gg_rp)
        vs = dict(raw_bundle['virtual_station'])

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
            'meta': dict(raw_bundle['meta']),
        }

    def _read_merit_unit_from_shapefile(
        self,
        cat_shp: Path,
        comid: int,
        lat: float,
    ) -> dict[str, Any] | None:
        """Read a single MERIT-Basins unit catchment from `cat_shp`."""
        df = pyogrio.read_dataframe(str(cat_shp), where=f'COMID = {int(comid)}')
        if df.empty:
            return None
        geom = df.geometry.to_numpy()[0]
        if 'unitarea' in df.columns:
            area_km2 = float(df['unitarea'].to_numpy()[0])
        elif 'uparea' in df.columns:
            area_km2 = float(df['uparea'].to_numpy()[0])
        else:
            area_km2 = float(
                geom.area
                * _KM_PER_DEG
                * _KM_PER_DEG
                * math.cos(math.radians(lat))
            )
        return {
            'type': 'Feature',
            'geometry': mapping(geom),
            'properties': {
                'fabric': 'merit_reach',
                'fabric_name': f'MERIT-Hydro Reach Catchment (COMID {comid})',
                'model': 'GEOGLOWS ECMWF',
                'comid': int(comid),
                'area_km2': round(area_km2, 1),
                'resolution': '90m MERIT-Basins',
            },
        }

    def _render_merit_watershed(
        self,
        lat: float,
        lon: float,
        river_id: int | None,
    ) -> dict[str, Any]:
        """Resolve a MERIT-Basins reach unit catchment polygon."""
        eff_rid = river_id or self.fetcher.geoglows.fetch_river_id(lat, lon)
        if eff_rid:
            cache_key = f'merit_reach_{eff_rid}'
            cached = self.fetcher.watershed_cache.get(
                cache_key, max_age_s=_WATERSHED_CACHE_MAX_AGE_S
            )
            if cached is not None:
                return cached
            cat_shp = find_merit_shp(eff_rid, 'cat')
            if cat_shp is not None and cat_shp.is_file():
                feature = self._read_merit_unit_from_shapefile(
                    cat_shp, eff_rid, lat
                )
                if feature is not None:
                    self.fetcher.watershed_cache.put(cache_key, feature)
                    return feature

        delin = HydroDelineator('merit-hydro')
        res = delin.delineate_catchment(lat, lon, mode='unit_catchment')
        props = res.get('properties') or {}
        if 'area_km2' not in props or props['area_km2'] is None:
            raise ValueError(
                'MERIT-Basins delineation result is missing area_km2.'
            )
        area_km2 = float(props['area_km2'])
        cid_label = props.get('catchment_id', 'Reach')
        feature = {
            'type': 'Feature',
            'geometry': res['geometry'],
            'properties': {
                'fabric': 'merit_reach',
                'fabric_name': f'MERIT Reach Catchment ({cid_label})',
                'model': 'GEOGLOWS ECMWF',
                'comid': eff_rid or int(props.get('outlet_unit_id') or 0),
                'area_km2': round(area_km2, 1),
                'resolution': '90m MERIT-Basins',
            },
        }
        if eff_rid:
            self.fetcher.watershed_cache.put(f'merit_reach_{eff_rid}', feature)
        return feature

    def render_watershed_polygon(  # noqa: PLR0913, PLR0917
        self,
        lat: float,
        lon: float,
        fabric: str = 'hydroatlas_full',
        gauge_id: str | None = None,
        river_id: int | None = None,
        geofabric: str | None = None,
    ) -> dict[str, Any]:
        """Resolve the watershed polygon for the selected hydrofabric."""
        lat, lon = float(lat), float(lon)
        eff_fabric = (geofabric or fabric or 'hydroatlas_full').strip().lower()

        if eff_fabric == 'glofas_cell':
            return glofas_cell_polygon(lat, lon)

        if eff_fabric == 'merit_reach':
            return self._render_merit_watershed(lat, lon, river_id)

        if eff_fabric not in ('hydroatlas_unit', 'hydroatlas_full'):
            raise ValueError(
                f'Unsupported watershed fabric {eff_fabric!r}. Expected one of '
                "'glofas_cell', 'merit_reach', 'hydroatlas_unit', "
                "'hydroatlas_full'."
            )

        cache_key = f'{eff_fabric}_{gauge_id or f"{lat:.4f}_{lon:.4f}"}'
        cached = self.fetcher.watershed_cache.get(
            cache_key, max_age_s=_WATERSHED_CACHE_MAX_AGE_S
        )
        if cached is not None:
            return cached

        delin = HydroDelineator('hydroatlas')
        mode = (
            'unit_catchment'
            if eff_fabric == 'hydroatlas_unit'
            else 'official_ridgeline'
        )
        res = delin.delineate_catchment(lat, lon, mode=mode)
        props = res.get('properties') or {}
        if 'area_km2' not in props or props['area_km2'] is None:
            raise ValueError(
                'HydroATLAS delineation result is missing area_km2.'
            )
        area_km2 = float(props['area_km2'])
        r_attrs = props.get('reach_attributes') or {}
        hybas_id = r_attrs.get('hydrobasins_unit', gauge_id or '')
        is_full = eff_fabric == 'hydroatlas_full'
        feat_name = (
            f'HydroATLAS Full Drainage Area ({hybas_id or "Basin"})'
            if is_full
            else f'HydroATLAS Level 12 Unit Catchment ({hybas_id or "Unit"})'
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
                'upstream_count': props.get('upstream_reaches_count', 1),
            },
        }
        self.fetcher.watershed_cache.put(cache_key, feature)
        return feature
