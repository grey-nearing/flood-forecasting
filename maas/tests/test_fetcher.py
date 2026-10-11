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

"""Offline unit tests for `maas.fetcher` (`MaaSDataFetcher`, `SQLiteCache`, and top-level fetch APIs)."""

from __future__ import annotations

from pathlib import Path
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


class TestDailySeriesAndSQLiteCache:
    """Tests daily series aggregation, date alignment, window peak, and SQLite cache."""

    def test_daily_series_alignment_and_window_peak(self) -> None:
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

    def test_sqlite_cache_put_and_get(self, tmp_path: Path) -> None:
        cache = SQLiteCache(tmp_path / 'test_cache.sqlite')
        cache.put('k1', {'val': 42})
        assert cache.get('k1', max_age_s=60.0) == {'val': 42}
        assert cache.get('missing', max_age_s=60.0) is None


class TestMaaSDataFetcher:
    """Tests `MaaSDataFetcher` and top-level `resolve_reaches` / `fetch_*` functions."""

    def test_fetcher_forecasts_and_return_periods(
        self, tmp_path: Path
    ) -> None:
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
            floodhub_api_key='',
        )
        fetcher = MaaSDataFetcher(config)

        glofas_fc_payload = {
            'daily': {
                'time': ['2026-04-01', '2026-04-02'],
                'river_discharge_median': [2200.0, 2600.0],
                'river_discharge_mean': [2200.0, 2600.0],
                'river_discharge_min': [1800.0, 2100.0],
                'river_discharge_p25': [2000.0, 2350.0],
                'river_discharge_p75': [2500.0, 2900.0],
                'river_discharge_max': [2900.0, 3400.0],
            }
        }
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

        def fake_session_get(url: str, **kwargs: object) -> mock.MagicMock:
            resp = mock.MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            if 'forecaststats' in str(url):
                resp.json.return_value = geoglows_fc_payload
            else:
                resp.json.return_value = glofas_fc_payload
            return resp

        fetcher.flood_cache.put('glofas_rp_38.625_-90.175', cached_rp)
        fetcher.flood_cache.put('geoglows_rp_720010511', cached_rp)
        with mock.patch('requests.Session.get', side_effect=fake_session_get):
            bundle = fetcher.fetch_forecasts(
                38.6270,
                -90.1994,
                river_id=720010511,
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
        assert bundle['virtual_station']['geoglows_reach']['river_id'] == 720010511
        # Verify pure backend bundle does NOT include frontend UI keys
        assert 'consensus' not in bundle
        assert 'flood_summary' not in bundle
        assert 'timeline' not in bundle

    def test_top_level_convenience_functions(self, tmp_path: Path) -> None:
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
        mock_rp_resp = mock.MagicMock()
        mock_rp_resp.status_code = 200
        mock_rp_resp.raise_for_status.return_value = None
        mock_rp_resp.json.return_value = {
            'daily': {
                'time': reanalysis_dates,
                'river_discharge': reanalysis_flows,
            }
        }
        with mock.patch('requests.Session.get', return_value=mock_rp_resp):
            rps = fetch_return_periods(
                config, 'glofas', lat=38.6270, lon=-90.1994, method='gumbel'
            )
            assert rps is not None
            assert rps['return_period_2'] > 1000.0

        mock_hist_resp = mock.MagicMock()
        mock_hist_resp.status_code = 200
        mock_hist_resp.raise_for_status.return_value = None
        mock_hist_resp.json.return_value = {
            'daily': {
                'time': ['2020-01-01', '2020-01-02'],
                'river_discharge': [1100.0, 1250.0],
            }
        }
        with mock.patch(
            'requests.Session.get',
            return_value=mock_hist_resp,
        ):
            hist = fetch_historical(
                config, 'glofas', lat=38.6270, lon=-90.1994
            )
            assert len(hist) == 2

        mock_fc_resp = mock.MagicMock()
        mock_fc_resp.status_code = 200
        mock_fc_resp.raise_for_status.return_value = None
        mock_fc_resp.json.return_value = {
            'daily': {
                'time': ['2026-04-01'],
                'river_discharge_median': [1500.0],
                'river_discharge_mean': [1500.0],
                'river_discharge_min': [1200.0],
                'river_discharge_p25': [1350.0],
                'river_discharge_p75': [1650.0],
                'river_discharge_max': [1800.0],
            }
        }
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

