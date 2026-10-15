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

"""Hermetic unit tests for GloFAS reference series in `frontend.maas_viewer`.

Verifies that the selected native GloFAS series (`control` / `median` /
`mean`) is used consistently for the hydrograph `central` line, the GloFAS
consensus row, and the reach return-period exceedance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from frontend.maas_viewer import MaaSViewer
from frontend.maas_viewer.consensus import (
    build_aligned_timeline,
    reach_exceedance_summary,
)
from maas.config import MaaSConfig

_HTTP_OK = 200
_N_DAYS = 3
# Return-period levels chosen so each native series peaks in a different class:
#   control peak 90 (< 2-yr), median peak 150 (>= 2-yr), mean peak 260 (>= 5-yr)
_RPS = {
    'return_period_2': 100.0,
    'return_period_5': 200.0,
    'return_period_20': 400.0,
    'return_period_100': 800.0,
    'status': 'live',
    'source': 'unit_test_rp',
}
_PEAKS = {'control': 90.0, 'median': 150.0, 'mean': 260.0}
_RANKS = {'control': 0, 'median': 1, 'mean': 2}


def _future_days() -> list[str]:
    today = datetime.now(UTC)
    return [
        (today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(_N_DAYS)
    ]


def _glofas_forecast() -> dict[str, Any]:
    """Live GloFAS payload whose three native series peak differently."""
    days = _future_days()
    records = []
    for i, day in enumerate(days):
        # Series rise to their peak on the last day.
        frac = (i + 1) / _N_DAYS
        records.append(
            {
                'time': day,
                'discharge_deterministic': _PEAKS['control'] * frac,
                'discharge_median': _PEAKS['median'] * frac,
                'discharge_mean': _PEAKS['mean'] * frac,
                'discharge_p25': 0.8 * _PEAKS['median'] * frac,
                'discharge_p75': 1.2 * _PEAKS['median'] * frac,
                'discharge_min': 0.5 * _PEAKS['median'] * frac,
                'discharge_max': 2.0 * _PEAKS['median'] * frac,
            }
        )
    return {
        'model': 'copernicus_glofas',
        'available': True,
        'status': 'live',
        'unit': 'CUBIC_METERS_PER_SECOND',
        'data': records,
        'ensemble_members': {},
        'return_periods': dict(_RPS),
    }


def _open_meteo_payload() -> dict[str, Any]:
    """Open-Meteo `/v1/flood` JSON matching `_glofas_forecast()`."""
    days = _future_days()
    fracs = [(i + 1) / _N_DAYS for i in range(_N_DAYS)]
    return {
        'daily': {
            'time': days,
            'river_discharge': [_PEAKS['control'] * f for f in fracs],
            'river_discharge_median': [_PEAKS['median'] * f for f in fracs],
            'river_discharge_mean': [_PEAKS['mean'] * f for f in fracs],
            'river_discharge_p25': [0.8 * _PEAKS['median'] * f for f in fracs],
            'river_discharge_p75': [1.2 * _PEAKS['median'] * f for f in fracs],
            'river_discharge_min': [0.5 * _PEAKS['median'] * f for f in fracs],
            'river_discharge_max': [2.0 * _PEAKS['median'] * f for f in fracs],
        }
    }


@pytest.mark.unit
class TestGlofasReferenceConsistency:
    """Hydrograph, consensus, and exceedance must all use one GloFAS series."""

    @pytest.mark.parametrize('reference', ['control', 'median', 'mean'])
    def test_reach_exceedance_uses_selected_reference(
        self, reference: str
    ) -> None:
        """Verify the GloFAS exceedance peak comes from the chosen series."""
        summary = reach_exceedance_summary(
            _glofas_forecast(),
            _RPS,
            None,
            None,
            glofas_reference=reference,
        )
        gl = summary['per_model']['glofas']
        assert gl['peak_flow'] == pytest.approx(_PEAKS[reference])
        assert gl['rank'] == _RANKS[reference]
        assert gl['reference'] == reference
        assert summary['glofas_reference'] == reference
        assert summary['governing_model'] == 'glofas'
        assert summary['rank'] == _RANKS[reference]

    def test_reach_exceedance_defaults_to_control_and_rejects_unknown(
        self,
    ) -> None:
        """Verify the default is the control run and bad names raise."""
        summary = reach_exceedance_summary(_glofas_forecast(), _RPS, None, None)
        assert summary['per_model']['glofas']['reference'] == 'control'
        assert summary['per_model']['glofas']['peak_flow'] == pytest.approx(
            _PEAKS['control']
        )
        with pytest.raises(ValueError, match='Unsupported GloFAS reference'):
            reach_exceedance_summary(
                _glofas_forecast(), _RPS, None, None, glofas_reference='p75'
            )

    @pytest.mark.parametrize('reference', ['control', 'median', 'mean'])
    def test_timeline_central_tracks_reference_and_keeps_all_series(
        self, reference: str
    ) -> None:
        """Verify `central` follows the reference while all 3 series ship."""
        tl = build_aligned_timeline(
            ['glofas'],
            None,
            fh_is_q=True,
            gl_fc=_glofas_forecast(),
            gg_fc=None,
            glofas_reference=reference,
        )
        gl = tl['series']['glofas']
        field = {
            'control': 'deterministic',
            'median': 'median',
            'mean': 'mean',
        }[reference]
        assert gl['reference'] == reference
        assert gl['central'] == gl[field]
        assert gl['central'][-1] == pytest.approx(_PEAKS[reference])
        assert gl['deterministic'][-1] == pytest.approx(_PEAKS['control'])
        assert gl['median'][-1] == pytest.approx(_PEAKS['median'])
        assert gl['mean'][-1] == pytest.approx(_PEAKS['mean'])

    @pytest.mark.parametrize('reference', ['control', 'median', 'mean'])
    def test_render_forecast_view_is_consistent_across_outputs(
        self, tmp_path: Path, reference: str
    ) -> None:
        """Verify one reference drives consensus, exceedance, and timeline."""
        cfg = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
            floodhub_api_key='',
        )
        viewer = MaaSViewer(cfg)
        # Seed cached return periods so no reanalysis request is needed.
        viewer.fetcher.flood_cache.put('glofas_rp_38.625_-90.175', dict(_RPS))

        def fake_get(
            url: str,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> mock.MagicMock:
            resp = mock.MagicMock()
            resp.status_code = _HTTP_OK
            resp.raise_for_status.return_value = None
            resp.json.return_value = _open_meteo_payload()
            return resp

        with mock.patch('requests.Session.get', side_effect=fake_get):
            view = viewer.render_forecast_view(
                38.6270,
                -90.1994,
                requested_models=['glofas'],
                glofas_reference=reference,
            )

        assert view['glofas_reference']['key'] == reference
        assert [o['key'] for o in view['glofas_reference']['options']] == [
            'control',
            'median',
            'mean',
        ]
        row = next(r for r in view['consensus'] if r['model'] == 'glofas')
        assert row['reference'] == reference
        assert row['peak_flow'] == pytest.approx(_PEAKS[reference])
        assert row['exceedance_rank'] == _RANKS[reference]
        exceed = view['reach_exceedance']['per_model']['glofas']
        assert exceed['peak_flow'] == pytest.approx(_PEAKS[reference])
        assert exceed['rank'] == _RANKS[reference]
        gl_tl = view['timeline']['series']['glofas']
        assert gl_tl['reference'] == reference
        assert max(v for v in gl_tl['central'] if v is not None) == (
            pytest.approx(_PEAKS[reference])
        )
        # The flood summary (shown as the agreement note) follows the same peak.
        expected_exceeding = 1 if _RANKS[reference] >= 1 else 0
        assert view['flood_summary']['models_exceeding_2yr'] == (
            expected_exceeding
        )

    def test_render_forecast_view_rejects_unknown_reference(
        self, tmp_path: Path
    ) -> None:
        """Verify an unknown reference fails before any network request."""
        cfg = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
            floodhub_api_key='',
        )
        viewer = MaaSViewer(cfg)
        with (
            mock.patch('requests.Session.get') as get_mock,
            pytest.raises(ValueError, match='Unsupported GloFAS reference'),
        ):
            viewer.render_forecast_view(
                38.6270,
                -90.1994,
                requested_models=['glofas'],
                glofas_reference='ensemble',
            )
        get_mock.assert_not_called()


@pytest.mark.unit
class TestMaaSReferenceUiHooks:
    """The MaaS tab exposes the reference selector and cadence tooltips."""

    def test_index_html_has_reference_selector_and_cadence_notes(
        self,
    ) -> None:
        """Verify the selector labels, query parameter, and tooltip hooks."""
        html_text = (
            Path(__file__).resolve().parents[1] / 'static' / 'index.html'
        ).read_text(encoding='utf-8')
        assert 'id="maasGlofasReferenceSelect"' in html_text
        assert 'GloFAS Reference:' in html_text
        for label in (
            'Control (Unperturbed)',
            'Ensemble Median',
            'Ensemble Mean',
        ):
            assert label in html_text
        assert 'setMaaSGlofasReference' in html_text
        assert '&glofas_reference=' in html_text
        assert 'maasGlofasReferenceTrace' in html_text
        assert "geoglows: '3-hourly instantaneous'" in html_text
        assert "glofas: 'daily mean'" in html_text

    def test_index_html_has_prediction_dots_and_strict_dot_locking(
        self,
    ) -> None:
        """Verify prediction dots on river map and strict dot locking hooks."""
        html_text = (
            Path(__file__).resolve().parents[1] / 'static' / 'index.html'
        ).read_text(encoding='utf-8')
        assert 'MAAS_PREDICTION_DOTS_MIN_ZOOM = 8' in html_text
        assert 'maasPredictionPointsByModel' in html_text
        assert 'deriveMaaSPredictionPointsFromFeatures' in html_text
        assert 'getMaaSPredictionPointsForModel' in html_text
        assert 'ensureMaaSPredictionPointsForModel' in html_text
        assert 'zoom >= MAAS_PREDICTION_DOTS_MIN_ZOOM' in html_text
        assert 'L.circleMarker([ptLat, ptLon]' in html_text
        assert 'gProps.prediction_points' in html_text
