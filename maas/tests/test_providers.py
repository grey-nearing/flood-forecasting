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

"""Offline unit tests for `maas` provider parsers, clients, SQLite cache, and `MaaSEngine`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from maas.config import (
    MaaSConfig,
    convert_discharge_units,
    normalize_requested_models,
    parse_finite_float,
    parse_float_or_default,
    parse_float_or_nan,
    parse_int,
)
from maas.engine import (
    MaaSEngine,
    SQLiteCache,
)
from maas.floodhub import (
    derive_floodhub_severity_from_forecast,
    geom_area_km2,
    kml_to_geometry,
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
    emulate_camaflood_physics,
    format_todays_earth_forecast,
    parse_todays_earth_payload,
    route_floodplain_excess,
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
    """Tests Google FloodHub payload parsers, severity derivation, and KML geometry conversion."""

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

    def test_normalize_severity_trend_and_kml(self) -> None:
        assert normalize_floodhub_severity('EXTREME') == 'EXTREME_DANGER'
        assert normalize_floodhub_severity('SEVERE') == 'DANGER'
        assert normalize_floodhub_severity('ABOVE_NORMAL') == 'WARNING'
        assert normalize_floodhub_trend('RISE') == 'RISING'

        kml = (
            '<kml><Polygon><outerBoundaryIs><LinearRing><coordinates>'
            '-90.20,38.60,0 -90.15,38.60,0 -90.15,38.65,0 -90.20,38.65,0 -90.20,38.60,0'
            '</coordinates></LinearRing></outerBoundaryIs></Polygon></kml>'
        )
        geom = kml_to_geometry(kml)
        assert geom is not None
        assert geom.geom_type == 'Polygon'
        assert geom_area_km2(geom) > 10.0

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


class TestTodaysEarthAndEngineAggregation:
    """Tests JAXA Today's Earth CaMa-Flood routing and `MaaSEngine` multi-model bundle assembly."""

    def test_camaflood_physics_routing_and_payload_parsing(self) -> None:
        glofas_records = [
            {
                'time': f'2026-04-0{day}',
                'discharge_median': 1200.0 + 400.0 * day,
                'discharge_mean': 1200.0 + 400.0 * day,
                'discharge_p25': 1000.0 + 300.0 * day,
                'discharge_p75': 1500.0 + 500.0 * day,
                'discharge_min': 900.0 + 250.0 * day,
                'discharge_max': 1800.0 + 600.0 * day,
            }
            for day in range(1, 7)
        ]
        rp = {
            'return_period_2': 1800.0,
            'return_period_5': 2400.0,
            'return_period_20': 3200.0,
            'mean_flow': 1100.0,
            'status': 'live',
        }
        emulated = emulate_camaflood_physics(glofas_records, rp, elev=120.0)
        series = emulated['series']
        assert len(series['rivout']) == 6
        assert len(series['flddph_m']) == 6
        assert len(series['fldfrc_pct']) == 6
        assert all(0.0 <= pct <= 100.0 for pct in series['fldfrc_pct'])

        formatted = format_todays_earth_forecast(
            38.627,
            -90.199,
            series,
            live=False,
            channel_params=emulated['channel_params'],
            forcing_status=emulated['forcing_status'],
        )
        assert formatted['status'] == 'fallback'
        assert formatted['emulated'] is True
        assert len(formatted['data']) == 6

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

        routed = route_floodplain_excess(
            [1000.0, 2000.0, 2500.0], q_bankfull=1500.0
        )
        assert len(routed) == 3
        assert routed[0] == 0.0
        assert routed[1] > 0.0

    def test_sqlite_cache_and_engine_unified_forecast(
        self, tmp_path: Path
    ) -> None:
        cache = SQLiteCache(tmp_path / 'test_cache.sqlite')
        cache.put('k1', {'val': 42})
        assert cache.get('k1', max_age_s=60.0) == {'val': 42}

        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
            floodhub_api_key='',
        )
        engine = MaaSEngine(config)

        mock_glofas = {
            'model': 'copernicus_glofas',
            'available': True,
            'status': 'live',
            'data': [
                {
                    'time': '2026-04-01',
                    'discharge_median': 2200.0,
                    'discharge_mean': 2200.0,
                    'discharge_min': 1800.0,
                    'discharge_p25': 2000.0,
                    'discharge_p75': 2500.0,
                    'discharge_max': 2900.0,
                },
                {
                    'time': '2026-04-02',
                    'discharge_median': 2600.0,
                    'discharge_mean': 2600.0,
                    'discharge_min': 2100.0,
                    'discharge_p25': 2350.0,
                    'discharge_p75': 2900.0,
                    'discharge_max': 3400.0,
                },
            ],
        }
        mock_geoglows = {
            'model': 'geoglows',
            'available': True,
            'status': 'live',
            'river_id': 720010511,
            'data': [
                {
                    'time': '2026-04-01T00:00:00Z',
                    'flow_med': 2100.0,
                    'flow_avg': 2100.0,
                    'flow_min': 1750.0,
                    'flow_25p': 1950.0,
                    'flow_75p': 2400.0,
                    'flow_max': 2750.0,
                },
                {
                    'time': '2026-04-02T00:00:00Z',
                    'flow_med': 2500.0,
                    'flow_avg': 2500.0,
                    'flow_min': 2050.0,
                    'flow_25p': 2250.0,
                    'flow_75p': 2800.0,
                    'flow_max': 3200.0,
                },
            ],
        }
        mock_rp = {
            'return_period_2': 1800.0,
            'return_period_5': 2400.0,
            'return_period_10': 2900.0,
            'return_period_20': 3500.0,
            'return_period_50': 4200.0,
            'return_period_100': 4900.0,
            'source': 'unit_test_rp',
            'status': 'live',
        }

        with (
            mock.patch.object(
                engine.glofas, 'fetch_forecast', return_value=mock_glofas
            ),
            mock.patch.object(
                engine.glofas,
                'fetch_reanalysis_return_periods',
                return_value=mock_rp,
            ),
            mock.patch.object(
                engine.geoglows, 'fetch_forecast', return_value=mock_geoglows
            ),
        ):
            engine.flood_cache.put('geoglows_rp_720010511', mock_rp)
            bundle = engine.fetch_unified_forecast(
                38.6270,
                -90.1994,
                river_id=720010511,
                requested_models=[
                    'floodhub',
                    'glofas',
                    'geoglows',
                    'todays_earth',
                ],
            )

        assert set(bundle['models'].keys()) == {
            'floodhub',
            'glofas',
            'geoglows',
            'todays_earth',
        }
        assert bundle['models']['floodhub']['status'] == 'unavailable'
        assert bundle['models']['glofas']['status'] == 'live'
        assert bundle['models']['geoglows']['status'] == 'live'
        assert bundle['models']['todays_earth']['emulated'] is True
        assert bundle['flood_summary']['overall_risk_level'] in {
            'WARNING',
            'SEVERE',
            'EXTREME',
        }
        assert 'glofas' in bundle['timeline']['series']
        assert 'geoglows' in bundle['timeline']['series']
        assert 'todays_earth' in bundle['timeline']['series']
