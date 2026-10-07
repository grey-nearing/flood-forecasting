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

from typing import Any

from frontend.maas_viewer.consensus import (
    build_aligned_timeline,
    build_consensus_row,
    build_flood_summary,
    reach_exceedance_summary,
    spread_confidence,
)
from frontend.maas_viewer.inundation import (
    camaflood_unit_feature,
    emulate_camaflood_physics,
)
from maas.config import MaaSConfig, normalize_requested_models
from maas.fetcher import MaaSDataFetcher, daily_series, window_peak
from maas.networks import snap_cama_cell
from maas.todays_earth import format_todays_earth_forecast


class MaaSViewer:
    """Frontend presenter that fetches raw `maas` data and builds UI visualization payloads."""

    def __init__(
        self,
        config: MaaSConfig,
        fetcher: MaaSDataFetcher | None = None,
    ) -> None:
        self.config = config
        self.fetcher = fetcher if fetcher is not None else MaaSDataFetcher(config)

    def render_forecast_view(
        self,
        lat: float,
        lon: float,
        gauge_id: str | None = None,
        river_id: int | None = None,
        reach_id: str | None = None,
        requested_models: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch raw multi-model forecasts and enrich with UI timeline, consensus, and inundation."""
        models = normalize_requested_models(requested_models)
        raw_bundle = self.fetcher.fetch_forecasts(
            lat=lat,
            lon=lon,
            gauge_id=gauge_id,
            river_id=river_id,
            reach_id=reach_id,
            requested_models=models,
        )
        models_out = dict(raw_bundle['models'])
        gl_fc = models_out.get('glofas')
        gg_fc = models_out.get('geoglows')
        fh_fc = models_out.get('floodhub')
        te_fc = models_out.get('todays_earth')
        gl_rp = (gl_fc or {}).get('return_periods')
        gg_rp = (gg_fc or {}).get('return_periods')

        # If Today's Earth live STAC data was unavailable, run visual CaMa-Flood emulation
        if (
            'todays_earth' in models
            and (not te_fc or not te_fc.get('available'))
            and gl_fc
            and gl_fc.get('data')
        ):
            emu = emulate_camaflood_physics(
                gl_fc.get('data') or [],
                gl_rp or {},
            )
            te_fc = format_todays_earth_forecast(
                lat,
                lon,
                emu['series'],
                live=False,
                channel_params=emu['channel_params'],
                forcing_status=gl_fc.get('status'),
            )
            models_out['todays_earth'] = te_fc

        consensus: list[dict[str, Any]] = []
        if 'floodhub' in models:
            if fh_fc and fh_fc.get('available'):
                fh_daily = daily_series(fh_fc.get('data'), 'discharge')
                peak, when = window_peak(fh_daily)
                consensus.append(
                    build_consensus_row(
                        'floodhub',
                        True,
                        fh_fc.get('status'),
                        peak,
                        when,
                        None,
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
        if 'glofas' in models and gl_fc:
            gl_daily = daily_series(
                gl_fc.get('data'), 'discharge_median', 'discharge_mean'
            )
            peak, when = window_peak(gl_daily)
            p25 = daily_series(gl_fc.get('data'), 'discharge_p25').get(when or '')
            p75 = daily_series(gl_fc.get('data'), 'discharge_p75').get(when or '')
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
        if 'geoglows' in models and gg_fc:
            gg_daily = daily_series(gg_fc.get('data'), 'flow_med')
            peak, when = window_peak(gg_daily)
            p25 = daily_series(gg_fc.get('data'), 'flow_25p').get(when or '')
            p75 = daily_series(gg_fc.get('data'), 'flow_75p').get(when or '')
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
                )
            )
        if 'todays_earth' in models and te_fc:
            te_daily = daily_series(te_fc.get('data'), 'discharge_mean')
            peak, when = window_peak(te_daily)
            consensus.append(
                build_consensus_row(
                    'todays_earth',
                    True,
                    'emulated' if te_fc.get('emulated') else te_fc.get('status'),
                    peak,
                    when,
                    gl_rp,
                    'GloFAS v4 reanalysis EV1',
                    'Low' if te_fc.get('emulated') else 'Medium',
                    independent=not bool(te_fc.get('emulated')),
                    emulated=bool(te_fc.get('emulated')),
                )
            )

        fh_is_q = (
            str((fh_fc or {}).get('unit') or '').upper()
            == 'CUBIC_METERS_PER_SECOND'
        )
        timeline = build_aligned_timeline(
            models, fh_fc, fh_is_q, gl_fc, gg_fc, te_fc
        )
        flood_summary = build_flood_summary(consensus, te_fc, fh_fc, None)
        exceedance = reach_exceedance_summary(gl_fc, gl_rp, gg_fc, gg_rp)

        inundation_features: list[dict[str, Any]] = []
        if te_fc:
            c_lat, c_lon = snap_cama_cell(lat, lon)
            inundation_features.append(
                camaflood_unit_feature(c_lat, c_lon, te_fc)
            )

        return {
            **raw_bundle,
            'models': models_out,
            'timeline': timeline,
            'consensus': consensus,
            'flood_summary': flood_summary,
            'reach_exceedance': exceedance,
            'inundation': {
                'type': 'FeatureCollection',
                'features': inundation_features,
            },
        }
