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

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
    _resolve_geoglows_cache_file,
    extract_geoglows_zarr_series,
    lookup_cached_geoglows_return_periods,
    parse_geoglows_forecast_response,
    parse_geoglows_retrospective_response,
    parse_geoglows_return_periods_payload,
)
from maas.glofas import (
    GloFASClient,
    compute_glofas_reanalysis_return_periods,
    parse_glofas_forecast_response,
)
from maas.thresholds import thresholds_from_return_periods

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_VAL = 42.0
_RIVER_ID = 720010511
_TWO_POINTS = 2
_THREE_POINTS = 3
_PEAK_DAY_GLOFAS = 150
_PEAK_DAY_RETRO = 100
_RETRO_YEARS = 12


class TestConfigAndParsers:
    """Tests numeric parsers, model normalization, and unit conversions."""

    def test_finite_float_and_int_parsers(self) -> None:
        """Verify numeric parsers reject NaN/Inf/booleans and parse floats."""
        assert parse_finite_float('123.45') == pytest.approx(123.45)
        assert parse_finite_float('-1.5e2') == pytest.approx(-150.0)
        assert parse_finite_float('nan') is None
        assert parse_finite_float('inf') is None
        flag_val: object = True
        assert parse_finite_float(flag_val) is None
        assert parse_float_or_default('bad', _DEFAULT_VAL) == _DEFAULT_VAL
        assert parse_int('720010511') == _RIVER_ID
        assert parse_int('abc') is None

        nan_val = parse_float_or_nan('invalid')
        assert np.isnan(nan_val)

    def test_convert_discharge_units_and_model_aliases(self) -> None:
        """Verify discharge unit conversion and model alias normalization."""
        assert convert_discharge_units(10.0, 'm3/s', 'cms') == pytest.approx(
            10.0
        )
        ft3s = convert_discharge_units(1.0, 'm3/s', 'ft3/s')
        assert ft3s == pytest.approx(35.3146667, rel=1e-5)

        models = normalize_requested_models(
            ['google_floodhub', 'copernicus_glofas', 'geoglows']
        )
        assert models == ['floodhub', 'glofas', 'geoglows']


class TestFloodHubProvider:
    """Tests Google FloodHub payload parsers and severity derivation."""

    def test_parse_floodhub_gauges_and_forecast(self) -> None:
        """Verify FloodHub gauge, forecast, and severity derivation."""
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
        assert len(fc['data']) == _TWO_POINTS
        assert fc['data'][1]['discharge'] == pytest.approx(9800.0)
        assert fc['thresholds']['warning_2yr'] == pytest.approx(8000.0)
        assert fc['thresholds']['danger_5yr'] == pytest.approx(9500.0)

        sev, trend = derive_floodhub_severity_from_forecast(fc, now_utc=ref_now)
        assert sev == 'DANGER'
        assert trend == 'RISING'

    def test_normalize_severity_and_trend(self) -> None:
        """Verify FloodHub severity and trend string normalization."""
        assert normalize_floodhub_severity('EXTREME') == 'EXTREME_DANGER'
        assert normalize_floodhub_severity('SEVERE') == 'DANGER'
        assert normalize_floodhub_severity('ABOVE_NORMAL') == 'WARNING'
        assert normalize_floodhub_trend('RISE') == 'RISING'

    def test_empty_floodhub_forecast_returns_unavailable_without_mock_data(
        self,
    ) -> None:
        """Verify missing FloodHub forecast returns unavailable with no data."""
        fc = parse_floodhub_forecast_response(None, None, 'hybas_7120456780')
        assert fc['status'] == 'unavailable'
        assert fc['available'] is False
        assert fc['data'] == []


class TestGloFASAndGeoGLOWSProviders:
    """Tests GloFAS v4 and GEOGLOWS v2 forecast and return-period parsers."""

    def test_parse_glofas_forecast_and_reanalysis(self) -> None:
        """Verify GloFAS v4 forecast parsing and reanalysis return periods."""
        raw_glofas = {
            'latitude': 38.625,
            'longitude': -90.175,
            'daily': {
                'time': ['2026-04-01', '2026-04-02', '2026-04-03'],
                'river_discharge': [4050.0, 4450.0, 4850.0],
                'river_discharge_mean': [4100.0, 4500.0, 4900.0],
                'river_discharge_median': [4100.0, 4500.0, 4900.0],
                'river_discharge_p25': [3800.0, 4100.0, 4400.0],
                'river_discharge_p75': [4400.0, 4900.0, 5400.0],
                'river_discharge_min': [3400.0, 3600.0, 3900.0],
                'river_discharge_max': [5100.0, 5800.0, 6400.0],
                'river_discharge_member01': [3900.0, 4200.0, 4600.0],
                'river_discharge_member02': [4300.0, 4800.0, 5200.0],
            },
        }
        parsed = parse_glofas_forecast_response(raw_glofas, 38.627, -90.199)
        assert parsed['status'] == 'live'
        assert len(parsed['data']) == _THREE_POINTS
        assert parsed['data'][2]['discharge_deterministic'] == pytest.approx(
            4850.0
        )
        assert parsed['data'][2]['discharge_median'] == pytest.approx(4900.0)
        assert parsed['data'][2]['discharge_min'] == pytest.approx(3900.0)
        assert parsed['data'][2]['discharge_max'] == pytest.approx(6400.0)
        assert set(parsed['ensemble_members']) == {'member01', 'member02'}
        assert parsed['ensemble_members']['member01'][2] == pytest.approx(
            4600.0
        )

        # Verify GloFASClient.fetch_forecast passes ensemble='true'
        mock_session = mock.MagicMock()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = raw_glofas
        mock_session.get.return_value = mock_resp
        client = GloFASClient(session=mock_session)
        fetched = client.fetch_forecast(38.627, -90.199, forecast_days=15)
        assert fetched['status'] == 'live'
        call_kwargs = mock_session.get.call_args.kwargs
        assert call_kwargs['params']['ensemble'] == 'true'
        assert 'river_discharge' in call_kwargs['params']['daily']

        # Verify null in river_discharge stays None (never median)
        raw_null_det = {
            'daily': {
                **raw_glofas['daily'],
                'river_discharge': [4050.0, None, 4850.0],
                'river_discharge_member01': [3900.0, None, 4600.0],
            }
        }
        parsed_null = parse_glofas_forecast_response(
            raw_null_det, 38.627, -90.199
        )
        assert parsed_null['data'][1]['discharge_deterministic'] is None
        assert parsed_null['data'][1]['discharge_median'] == pytest.approx(
            4500.0
        )
        assert parsed_null['ensemble_members']['member01'][1] is None

        # Verify missing river_discharge raises KeyError
        missing_det = {
            'daily': {
                k: v
                for k, v in raw_glofas['daily'].items()
                if k != 'river_discharge'
            }
        }
        with pytest.raises(KeyError, match='river_discharge'):
            parse_glofas_forecast_response(missing_det, 38.627, -90.199)

        # Verify mismatched member length raises ValueError
        bad_len_member = {
            'daily': {
                **raw_glofas['daily'],
                'river_discharge_member01': [3900.0, 4200.0],
            }
        }
        with pytest.raises(ValueError, match='Length mismatch'):
            parse_glofas_forecast_response(bad_len_member, 38.627, -90.199)

        # Verify non-sequence member raises TypeError
        bad_type_member = {
            'daily': {
                **raw_glofas['daily'],
                'river_discharge_member01': 'invalid',
            }
        }
        with pytest.raises(TypeError, match='Expected sequence'):
            parse_glofas_forecast_response(bad_type_member, 38.627, -90.199)

        # Reanalysis annual maxima across 12 years (>= 300 days/year)
        times: list[str] = []
        discharges: list[float] = []
        for yr_idx, yr in enumerate(range(2010, 2022)):
            for day in range(1, 310):
                times.append(f'{yr}-01-01')
                discharges.append(
                    1200.0
                    + (
                        2000.0 + 300.0 * yr_idx
                        if day == _PEAK_DAY_GLOFAS
                        else 0.0
                    )
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
        """Verify GEOGLOWS forecast, return periods, and retrospective."""
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
        fc = parse_geoglows_forecast_response(raw_fc, _RIVER_ID)
        assert fc['status'] == 'live'
        assert (
            len(fc['data']) == _TWO_POINTS
        )  # skips intermediate blank-median/avg row
        assert fc['river_id'] == _RIVER_ID

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
        rp = parse_geoglows_return_periods_payload(raw_rp, _RIVER_ID)
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
                    900.0
                    + (
                        1800.0 + 250.0 * yr_idx
                        if day == _PEAK_DAY_RETRO
                        else 0.0
                    )
                )
        retro = parse_geoglows_retrospective_response(
            {'datetime': retro_times, '720010511': retro_vals},
            _RIVER_ID,
        )
        assert retro is not None
        assert retro['years_of_record'] == _RETRO_YEARS
        assert retro['return_period_100'] > retro['return_period_2']

        # 3-level payload (e.g. 2, 5, 25 yr without explicit 100 yr) must
        # interpolate all canonical return periods so thresholds never fail.
        sparse_rp = parse_geoglows_return_periods_payload(
            {
                'return_period_2': 1800.0,
                'return_period_5': 2400.0,
                'return_period_25': 3700.0,
            },
            _RIVER_ID,
        )
        assert sparse_rp is not None
        for req_yr in (2, 5, 20, 100):
            assert f'return_period_{req_yr}' in sparse_rp
        th = thresholds_from_return_periods(sparse_rp, 'GEOGLOWS v2')
        assert th['extreme_100yr'] > th['extreme_20yr'] > th['danger_5yr']

    def test_lookup_cached_geoglows_return_periods_and_cache_resolution(
        self, tmp_path: Path
    ) -> None:
        """Verify cached GEOGLOWS return period lookup and path resolution."""
        gg_dir = tmp_path / 'geoglows_v2'
        gg_dir.mkdir(parents=True, exist_ok=True)
        npz_path = gg_dir / 'geoglows_return_periods_v1.npz'
        np.savez_compressed(
            npz_path,
            river_id=np.array([_RIVER_ID], dtype=np.int32),
            return_periods=np.array([2, 5, 25], dtype=np.int32),
            gumbel_daily=np.array([[1800.0, 2400.0, 3700.0]], dtype=np.float32),
        )
        resolved = _resolve_geoglows_cache_file(
            tmp_path, 'geoglows_return_periods_v1.npz'
        )
        assert resolved == npz_path
        cached = lookup_cached_geoglows_return_periods(_RIVER_ID, npz_path)
        assert cached is not None
        assert cached['return_period_2'] == pytest.approx(1800.0)
        assert cached['return_period_20'] > cached['return_period_5']
        assert cached['return_period_100'] > cached['return_period_25']

    def test_extract_geoglows_zarr_series_reads_reach_and_keeps_nan(
        self, tmp_path: Path
    ) -> None:
        """Verify local GEOGLOWS Zarr extraction selects a reach by `rivid`."""
        times = pd.date_range('2021-01-01', periods=_THREE_POINTS, freq='D')
        qout = np.array(
            [[10.0, 20.0, 30.0], [100.0, np.nan, 300.0]], dtype=np.float32
        ).T
        ds = xr.Dataset(
            {'Qout': (('time', 'rivid'), qout)},
            coords={
                'time': times,
                'rivid': np.array([111, _RIVER_ID], dtype=np.int64),
            },
        )
        ds.to_zarr(tmp_path / 'geoglows.zarr', mode='w')
        records = extract_geoglows_zarr_series(
            tmp_path / 'geoglows.zarr', _RIVER_ID
        )
        assert [r['time'] for r in records] == [
            '2021-01-01T00:00:00Z',
            '2021-01-02T00:00:00Z',
            '2021-01-03T00:00:00Z',
        ]
        assert records[0]['discharge'] == pytest.approx(100.0)
        assert math.isnan(records[1]['discharge'])
        assert records[2]['discharge'] == pytest.approx(300.0)
        with pytest.raises(KeyError, match='river_id 999 not found'):
            extract_geoglows_zarr_series(tmp_path / 'geoglows.zarr', 999)
