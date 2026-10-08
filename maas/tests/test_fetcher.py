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
                fetcher.glofas, 'fetch_forecast', return_value=mock_glofas
            ),
            mock.patch.object(
                fetcher.glofas,
                'fetch_reanalysis_return_periods',
                return_value=mock_rp,
            ),
            mock.patch.object(
                fetcher.geoglows, 'fetch_forecast', return_value=mock_geoglows
            ),
        ):
            fetcher.flood_cache.put('geoglows_rp_720010511', mock_rp)
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

        with mock.patch(
            'maas.fetcher.GloFASClient.fetch_reanalysis_return_periods',
            return_value={'return_period_2': 1500.0, 'status': 'live'},
        ):
            rps = fetch_return_periods(
                config, 'glofas', lat=38.6270, lon=-90.1994
            )
            assert rps is not None
            assert rps['return_period_2'] == pytest.approx(1500.0)

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {
            'daily': {
                'time': ['2020-01-01', '2020-01-02'],
                'river_discharge': [1100.0, 1250.0],
            }
        }
        with mock.patch(
            'requests.Session.get',
            return_value=mock_resp,
        ):
            hist = fetch_historical(
                config, 'glofas', lat=38.6270, lon=-90.1994
            )
            assert len(hist) == 2

        with mock.patch(
            'maas.fetcher.GloFASClient.fetch_forecast',
            return_value={
                'model': 'copernicus_glofas',
                'available': True,
                'status': 'live',
                'data': [],
            },
        ):
            fc = fetch_forecasts(
                config,
                38.6270,
                -90.1994,
                requested_models=['glofas'],
            )
            assert fc['models']['glofas']['status'] == 'live'
