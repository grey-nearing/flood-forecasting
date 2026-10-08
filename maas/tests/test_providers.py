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

"""Offline unit tests for `maas` provider parsers and clients."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from maas.config import (
    convert_discharge_units,
    normalize_requested_models,
    parse_finite_float,
    parse_float_or_default,
    parse_float_or_nan,
    parse_int,
)
from maas.floodhub import (
    derive_floodhub_severity_from_forecast,
    normalize_floodhub_severity,
    normalize_floodhub_trend,
    parse_floodhub_forecast_response,
    parse_floodhub_gauges_response,
)
from maas.geoglows import (
    parse_geoglows_forecast_response,
    parse_geoglows_retrospective_response,
    parse_geoglows_return_periods_payload,
)
from maas.glofas import (
    compute_glofas_reanalysis_return_periods,
    parse_glofas_forecast_response,
)
from maas.todays_earth import (
    format_todays_earth_forecast,
    parse_todays_earth_payload,
)


class TestConfigAndParsers:
    """Tests regex-based numeric parsers, model normalization, and unit conversions."""

    def test_finite_float_and_int_parsers(self) -> None:
        assert parse_finite_float('123.45') == pytest.approx(123.45)
        assert parse_finite_float('-1.5e2') == pytest.approx(-150.0)
        assert parse_finite_float('nan') is None
        assert parse_finite_float('inf') is None
        assert parse_finite_float(True) is None
        assert parse_float_or_default('bad', 42.0) == 42.0
        assert parse_int('720010511') == 720010511
        assert parse_int('abc') is None

        nan_val = parse_float_or_nan('invalid')
        assert nan_val != nan_val  # NaN check

    def test_convert_discharge_units_and_model_aliases(self) -> None:
        assert convert_discharge_units(10.0, 'm3/s', 'cms') == pytest.approx(
            10.0
        )
        ft3s = convert_discharge_units(1.0, 'm3/s', 'ft3/s')
        assert ft3s == pytest.approx(35.3146667, rel=1e-5)

        models = normalize_requested_models(
            ['floodhub', 'camaflood', 'geoglows']
        )
        assert models == ['floodhub', 'todays_earth', 'geoglows']


class TestFloodHubProvider:
    """Tests Google FloodHub payload parsers and severity derivation."""

    def test_parse_floodhub_gauges_and_forecast(self) -> None:
        raw_gauges = {
            'floodStatuses': [
                {
                    'gaugeId': 'hybas_7120456780',
                    'gaugeLocation': {'latitude': 38.627, 'longitude': -90.199},
                    'severity': 'WARNING',
                    'forecastTrend': 'RISE',
                    'issuedTime': '2026-04-01T00:00:00Z',
                    'qualityVerified': True,
                    'source': 'HYBAS',
                }
            ]
        }
        parsed_gauges = parse_floodhub_gauges_response(raw_gauges)
        assert len(parsed_gauges) == 1
        assert parsed_gauges[0]['gauge_id'] == 'hybas_7120456780'
        assert parsed_gauges[0]['lat'] == pytest.approx(38.627)

        ref_now = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
        raw_fc = {
            'forecasts': {
                'hybas_7120456780': {
                    'forecasts': [
                        {
                            'issuedTime': '2026-04-01T00:00:00Z',
                            'forecastRanges': [
                                {
                                    'forecastStartTime': '2026-04-01T00:00:00Z',
                                    'value': 8200.0,
                                },
                                {
                                    'forecastStartTime': '2026-04-02T00:00:00Z',
                                    'value': 9800.0,
                                },
                            ],
                        }
                    ]
                }
            }
        }
        raw_meta = {
            'gaugeModels': [
                {
                    'gaugeValueUnit': 'CUBIC_METERS_PER_SECOND',
                    'thresholds': {
                        'warningLevel': 8000.0,
                        'dangerLevel': 9500.0,
                        'extremeDangerLevel': 12000.0,
                    },
                }
            ]
        }
        fc = parse_floodhub_forecast_response(
            raw_fc, raw_meta, 'hybas_7120456780', now_utc=ref_now
        )
        assert fc['status'] == 'live'
        assert len(fc['data']) == 2
        assert fc['data'][1]['discharge'] == pytest.approx(9800.0)
        assert fc['thresholds']['warning_2yr'] == pytest.approx(8000.0)
        assert fc['thresholds']['danger_5yr'] == pytest.approx(9500.0)

        sev, trend = derive_floodhub_severity_from_forecast(fc, now_utc=ref_now)
        assert sev == 'DANGER'
        assert trend == 'RISING'

    def test_normalize_severity_and_trend(self) -> None:
        assert normalize_floodhub_severity('EXTREME') == 'EXTREME_DANGER'
        assert normalize_floodhub_severity('SEVERE') == 'DANGER'
        assert normalize_floodhub_severity('ABOVE_NORMAL') == 'WARNING'
        assert normalize_floodhub_trend('RISE') == 'RISING'

    def test_empty_floodhub_forecast_returns_unavailable_without_mock_data(
        self,
    ) -> None:
        fc = parse_floodhub_forecast_response(None, None, 'hybas_7120456780')
        assert fc['status'] == 'unavailable'
        assert fc['available'] is False
        assert fc['data'] == []


class TestGloFASAndGeoGLOWSProviders:
    """Tests GloFAS v4 and GEOGLOWS v2 forecast and return-period parsers."""

    def test_parse_glofas_forecast_and_reanalysis(self) -> None:
        raw_glofas = {
            'latitude': 38.625,
            'longitude': -90.175,
            'daily': {
                'time': ['2026-04-01', '2026-04-02', '2026-04-03'],
                'river_discharge_mean': [4100.0, 4500.0, 4900.0],
                'river_discharge_median': [4100.0, 4500.0, 4900.0],
                'river_discharge_p25': [3800.0, 4100.0, 4400.0],
                'river_discharge_p75': [4400.0, 4900.0, 5400.0],
                'river_discharge_min': [3400.0, 3600.0, 3900.0],
                'river_discharge_max': [5100.0, 5800.0, 6400.0],
            },
        }
        parsed = parse_glofas_forecast_response(raw_glofas, 38.627, -90.199)
        assert parsed['status'] == 'live'
        assert len(parsed['data']) == 3
        assert parsed['data'][2]['discharge_median'] == pytest.approx(4900.0)
        assert parsed['data'][2]['discharge_min'] == pytest.approx(3900.0)
        assert parsed['data'][2]['discharge_max'] == pytest.approx(6400.0)

        # Reanalysis annual maxima across 12 years (>= 300 days/year)
        times: list[str] = []
        discharges: list[float] = []
        for yr_idx, yr in enumerate(range(2010, 2022)):
            for day in range(1, 310):
                times.append(f'{yr}-01-01')
                discharges.append(
                    1200.0 + (2000.0 + 300.0 * yr_idx if day == 150 else 0.0)
                )
        reanalysis = {
            'latitude': 38.625,
            'longitude': -90.175,
            'daily': {'time': times, 'river_discharge': discharges},
        }
        rp = compute_glofas_reanalysis_return_periods(reanalysis, end_year=2021)
        assert rp is not None
        assert rp['return_period_100'] > rp['return_period_2'] > 0.0

    def test_parse_geoglows_forecast_return_periods_and_retrospective(
        self,
    ) -> None:
        raw_fc = {
            'datetime': [
                '2026-04-01T00:00:00Z',
                '2026-04-01T06:00:00Z',
                '2026-04-02T00:00:00Z',
            ],
            'flow_med': [1200.0, '', 1650.0],
            'flow_avg': [1210.0, '', 1660.0],
            'flow_25p': [1000.0, '', 1300.0],
            'flow_75p': [1450.0, '', 2000.0],
            'flow_min': [850.0, '', 1100.0],
            'flow_max': [1800.0, '', 2600.0],
            'high_res': [1220.0, 1400.0, 1680.0],
        }
        fc = parse_geoglows_forecast_response(raw_fc, 720010511)
        assert fc['status'] == 'live'
        assert len(fc['data']) == 2  # skips intermediate blank-median/avg row
        assert fc['river_id'] == 720010511

        raw_rp = {
            'return_periods': {
                'return_period_2': {'720010511': 2500.0},
                'return_period_5': {'720010511': 3400.0},
                'return_period_10': {'720010511': 4100.0},
                'return_period_25': {'720010511': 4900.0},
                'return_period_50': {'720010511': 5600.0},
                'return_period_100': {'720010511': 6300.0},
            }
        }
        rp = parse_geoglows_return_periods_payload(raw_rp, 720010511)
        assert rp is not None
        assert rp['return_period_2'] == pytest.approx(2500.0)
        assert rp['return_period_20'] > rp['return_period_10']
        assert rp['return_period_100'] == pytest.approx(6300.0)

        retro_times: list[str] = []
        retro_vals: list[float] = []
        for yr_idx, yr in enumerate(range(2010, 2022)):
            for day in range(1, 310):
                retro_times.append(f'{yr}-05-10T00:00:00Z')
                retro_vals.append(
                    900.0 + (1800.0 + 250.0 * yr_idx if day == 100 else 0.0)
                )
        retro = parse_geoglows_retrospective_response(
            {'datetime': retro_times, '720010511': retro_vals},
            720010511,
        )
        assert retro is not None
        assert retro['years_of_record'] == 12
        assert retro['return_period_100'] > retro['return_period_2']


class TestTodaysEarthPayloadParsing:
    """Tests JAXA Today's Earth payload parsing and forecast formatting."""

    def test_todays_earth_payload_parsing_and_formatting(self) -> None:
        raw_te = {
            'timestamps': ['2026-04-01', '2026-04-02'],
            'rivout': [950.0, 1120.0],
            'flddph': [0.4, 0.8],
            'fldfrc': [0.08, 0.14],
        }
        parsed_te, err = parse_todays_earth_payload(raw_te)
        assert err is None
        assert parsed_te is not None
        assert len(parsed_te['rivout']) == 2
        assert parsed_te['fldfrc_pct'] == pytest.approx([8.0, 14.0])

        formatted = format_todays_earth_forecast(
            38.627,
            -90.199,
            parsed_te,
            live=True,
        )
        assert formatted['status'] == 'live'
        assert formatted['emulated'] is False
        assert len(formatted['data']) == 2
