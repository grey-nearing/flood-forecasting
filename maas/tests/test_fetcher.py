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

"""Offline unit tests for `maas.fetcher`.

Covers `MaaSDataFetcher`, `SQLiteCache`, strict `daily_series` validation,
and the top-level `fetch_*` APIs (including the GloFAS reference series).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from maas.config import MaaSConfig
from maas.fetcher import (
    MaaSDataFetcher,
    SQLiteCache,
    align_daily_series,
    daily_series,
    fetch_forecasts,
    fetch_gauges,
    fetch_historical,
    fetch_return_periods,
    resolve_reaches,
    window_peak,
)

if TYPE_CHECKING:
    from pathlib import Path

_HTTP_OK = 200
_RIVER_ID = 720010511
_TWO_RECORDS = 2
_MIN_RP2_FLOW = 1000.0

_GLOFAS_FC_PAYLOAD = {
    'daily': {
        'time': ['2026-04-01', '2026-04-02'],
        'river_discharge': [2250.0, 2650.0],
        'river_discharge_median': [2200.0, 2600.0],
        'river_discharge_mean': [2300.0, 2700.0],
        'river_discharge_min': [1800.0, 2100.0],
        'river_discharge_p25': [2000.0, 2350.0],
        'river_discharge_p75': [2500.0, 2900.0],
        'river_discharge_max': [2900.0, 3400.0],
        'river_discharge_member01': [1800.0, 2100.0],
        'river_discharge_member02': [2900.0, 3400.0],
    }
}


def _mock_response(payload: object) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.status_code = _HTTP_OK
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


@pytest.mark.unit
class TestDailySeriesAndSQLiteCache:
    """Tests strict daily aggregation, alignment, window peak, and cache."""

    def test_daily_series_alignment_and_window_peak(self) -> None:
        """Verify sub-daily records average into daily means by UTC date."""
        records = [
            {'time': '2026-04-01T00:00:00Z', 'q': 100.0},
            {'time': '2026-04-01T12:00:00Z', 'q': 200.0},
            {'time': '2026-04-02T00:00:00Z', 'q': 400.0},
        ]
        ds = daily_series(records, 'q')
        assert ds == {'2026-04-01': 150.0, '2026-04-02': 400.0}
        aligned = align_daily_series(
            ds, ['2026-04-01', '2026-04-02', '2026-04-03']
        )
        assert aligned == [150.0, 400.0, None]
        peak, peak_date = window_peak(ds)
        assert peak == pytest.approx(400.0)
        assert peak_date == '2026-04-02'

    def test_daily_series_skips_missing_values_without_imputation(
        self,
    ) -> None:
        """Verify `None`/`NaN` values are skipped and never filled or zeroed."""
        records = [
            {'time': '2026-04-01', 'q': None},
            {'time': '2026-04-02', 'q': float('nan')},
            {'time': '2026-04-02T12:00:00+00:00', 'q': 30.0},
            {'time': '2026-04-03', 'q': 10},
        ]
        assert daily_series(records, 'q') == {
            '2026-04-02': 30.0,
            '2026-04-03': 10.0,
        }
        assert daily_series(None, 'q') == {}
        assert daily_series([], 'q') == {}

    def test_daily_series_rejects_malformed_timestamps(self) -> None:
        """Verify malformed or non-string `time` values raise, not skip."""
        for bad_time in ('2026-4-1', '20260401', 'April 1 2026', '2026-04-0'):
            with pytest.raises(ValueError, match='not an ISO-8601'):
                daily_series([{'time': bad_time, 'q': 1.0}], 'q')
        with pytest.raises(ValueError, match=r'month must be in 1\.\.12'):
            daily_series([{'time': '2026-13-01', 'q': 1.0}], 'q')
        with pytest.raises(TypeError, match='must be an ISO-8601 string'):
            daily_series([{'time': 1_775_000_000, 'q': 1.0}], 'q')
        with pytest.raises(KeyError, match='missing a `time` field'):
            daily_series([{'q': 1.0}], 'q')

    def test_daily_series_rejects_non_numeric_values_and_missing_keys(
        self,
    ) -> None:
        """Verify non-numeric values and missing fields fail loudly."""
        with pytest.raises(TypeError, match='must be numeric or None'):
            daily_series([{'time': '2026-04-01', 'q': '12.5'}], 'q')
        with pytest.raises(TypeError, match='must be numeric or None'):
            daily_series([{'time': '2026-04-01', 'q': True}], 'q')
        with pytest.raises(KeyError, match="missing field 'q'"):
            daily_series([{'time': '2026-04-01', 'other': 1.0}], 'q')
        with pytest.raises(TypeError, match='Expected a mapping record'):
            daily_series([('2026-04-01', 1.0)], 'q')  # type: ignore[list-item]

    def test_sqlite_cache_put_and_get(self, tmp_path: Path) -> None:
        """Verify SQLite cache round-trips payloads and misses cleanly."""
        cache = SQLiteCache(tmp_path / 'test_cache.sqlite')
        cache.put('k1', {'val': 42})
        assert cache.get('k1', max_age_s=60.0) == {'val': 42}
        assert cache.get('missing', max_age_s=60.0) is None


@pytest.mark.unit
class TestMaaSDataFetcher:
    """Tests `MaaSDataFetcher` and the top-level `fetch_*` functions."""

    def test_fetcher_forecasts_and_return_periods(self, tmp_path: Path) -> None:
        """Verify the pure backend bundle for all three providers."""
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
            floodhub_api_key='',
        )
        fetcher = MaaSDataFetcher(config)

        geoglows_fc_payload = {
            'datetime': ['2026-04-01T00:00:00Z', '2026-04-02T00:00:00Z'],
            'flow_med': [2100.0, 2500.0],
            'flow_avg': [2100.0, 2500.0],
            'flow_min': [1750.0, 2050.0],
            'flow_25p': [1950.0, 2250.0],
            'flow_75p': [2400.0, 2800.0],
            'flow_max': [2750.0, 3200.0],
        }
        cached_rp = {
            'return_period_2': 1800.0,
            'return_period_5': 2400.0,
            'return_period_10': 2900.0,
            'return_period_20': 3500.0,
            'return_period_50': 4200.0,
            'return_period_100': 4900.0,
            'source': 'unit_test_rp',
            'status': 'live',
        }

        def fake_session_get(
            url: str,
            **kwargs: object,  # noqa: ARG001
        ) -> mock.MagicMock:
            if 'forecaststats' in str(url):
                return _mock_response(geoglows_fc_payload)
            return _mock_response(_GLOFAS_FC_PAYLOAD)

        fetcher.flood_cache.put('glofas_rp_38.625_-90.175', cached_rp)
        fetcher.flood_cache.put(f'geoglows_rp_{_RIVER_ID}', cached_rp)
        with mock.patch('requests.Session.get', side_effect=fake_session_get):
            bundle = fetcher.fetch_forecasts(
                38.6270,
                -90.1994,
                river_id=_RIVER_ID,
                requested_models=[
                    'floodhub',
                    'glofas',
                    'geoglows',
                ],
            )

        assert set(bundle['models'].keys()) == {
            'floodhub',
            'glofas',
            'geoglows',
        }
        assert bundle['models']['floodhub']['status'] == 'unavailable'
        assert bundle['models']['glofas']['status'] == 'live'
        assert bundle['models']['geoglows']['status'] == 'live'
        assert bundle['thresholds']['warning_2yr'] == pytest.approx(1800.0)
        assert (
            bundle['virtual_station']['geoglows_reach']['river_id'] == _RIVER_ID
        )
        # All three native GloFAS reference series travel with every record.
        gl_record = bundle['models']['glofas']['data'][1]
        assert gl_record['discharge_deterministic'] == pytest.approx(2650.0)
        assert gl_record['discharge_median'] == pytest.approx(2600.0)
        assert gl_record['discharge_mean'] == pytest.approx(2700.0)
        # Verify pure backend bundle does NOT include frontend UI keys
        assert 'consensus' not in bundle
        assert 'flood_summary' not in bundle
        assert 'timeline' not in bundle

    def test_fetch_forecasts_glofas_reference_selects_native_series(
        self, tmp_path: Path
    ) -> None:
        """Verify `glofas_reference` drives the GloFAS daily series and peak."""
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
            floodhub_api_key='',
        )
        expected_peaks = {'control': 2650.0, 'median': 2600.0, 'mean': 2700.0}
        with mock.patch(
            'requests.Session.get',
            return_value=_mock_response(_GLOFAS_FC_PAYLOAD),
        ):
            for ref, expected_peak in expected_peaks.items():
                out = fetch_forecasts(
                    config,
                    38.6270,
                    -90.1994,
                    requested_models=['glofas'],
                    glofas_reference=ref,
                )
                assert out['glofas_reference']['key'] == ref
                assert out['peaks']['glofas']['reference'] == ref
                assert out['peaks']['glofas']['peak_flow'] == pytest.approx(
                    expected_peak
                )
                assert out['daily']['glofas']['2026-04-02'] == pytest.approx(
                    expected_peak
                )
            default_out = fetch_forecasts(
                config, 38.6270, -90.1994, requested_models=['glofas']
            )
            assert default_out['glofas_reference']['key'] == 'control'
            assert default_out['peaks']['glofas']['peak_flow'] == (
                pytest.approx(expected_peaks['control'])
            )
            with pytest.raises(ValueError, match='Unsupported GloFAS'):
                fetch_forecasts(
                    config,
                    38.6270,
                    -90.1994,
                    requested_models=['glofas'],
                    glofas_reference='p75',
                )

    def test_top_level_convenience_functions(self, tmp_path: Path) -> None:
        """Verify reaches, gauges, return periods, history, and forecasts."""
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
            floodhub_api_key='',
        )
        reaches = resolve_reaches(38.6270, -90.1994, config=config)
        assert 'glofas' in reaches
        assert 'geoglows' in reaches

        gauges = fetch_gauges(config, (38.0, -91.0, 39.0, -90.0))
        assert gauges == []

        reanalysis_dates = [
            f'{yr}-{m:02d}-{d:02d}'
            for yr in range(2000, 2015)
            for m in range(1, 12)
            for d in range(1, 29)
        ]
        reanalysis_flows = [
            1000.0 + (idx % 308) * 2.0 + (idx // 308) * 50.0
            for idx in range(len(reanalysis_dates))
        ]
        mock_rp_resp = _mock_response(
            {
                'daily': {
                    'time': reanalysis_dates,
                    'river_discharge': reanalysis_flows,
                }
            }
        )
        with mock.patch('requests.Session.get', return_value=mock_rp_resp):
            rps = fetch_return_periods(
                config, 'glofas', lat=38.6270, lon=-90.1994, method='gumbel'
            )
            assert rps is not None
            assert rps['return_period_2'] > _MIN_RP2_FLOW

        mock_hist_resp = _mock_response(
            {
                'daily': {
                    'time': ['2020-01-01', '2020-01-02'],
                    'river_discharge': [1100.0, None],
                }
            }
        )
        with mock.patch(
            'requests.Session.get',
            return_value=mock_hist_resp,
        ):
            hist = fetch_historical(config, 'glofas', lat=38.6270, lon=-90.1994)
            assert len(hist) == _TWO_RECORDS
            assert hist.iloc[0] == pytest.approx(1100.0)
            assert math.isnan(hist.iloc[1])

        mock_fc_resp = _mock_response(
            {
                'daily': {
                    'time': ['2026-04-01'],
                    'river_discharge': [1520.0],
                    'river_discharge_median': [1500.0],
                    'river_discharge_mean': [1500.0],
                    'river_discharge_min': [1200.0],
                    'river_discharge_p25': [1350.0],
                    'river_discharge_p75': [1650.0],
                    'river_discharge_max': [1800.0],
                }
            }
        )
        with mock.patch(
            'requests.Session.get',
            return_value=mock_fc_resp,
        ):
            fc = fetch_forecasts(
                config,
                38.6270,
                -90.1994,
                requested_models=['glofas'],
            )
            assert fc['models']['glofas']['status'] == 'live'
