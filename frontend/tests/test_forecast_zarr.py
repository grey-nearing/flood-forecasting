# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hermetic unit tests for real-time weather forecast Zarr extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon, mapping

from frontend.forecast_zarr import (
    FORECAST_FEATURE_SPECS,
    ForecastZarrExtractor,
)
from frontend.profile_manager import ProfileManager

_EXPECTED_HORIZON_10 = 10
_EXPECTED_HORIZON_15 = 15
_EXPECTED_BATCH_BASINS = 2
_EXPECTED_FEATURES_COUNT = 7


def _build_forecast_datasets(
    n_lead_days: int = 15,
    init_time: str = '2026-05-10T00:00:00',
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build deterministic gridded HRES and GraphCast datasets for testing."""
    init_times = pd.DatetimeIndex([init_time])
    lead_days = np.arange(1, n_lead_days + 1, dtype=np.int64)
    lats = np.linspace(41.5, 39.0, 11, dtype=np.float64)
    lons = np.linspace(-88.0, -85.5, 11, dtype=np.float64)
    shape_4d = (1, n_lead_days, len(lats), len(lons))

    ds_hres = xr.Dataset(
        data_vars={
            'hres_surface_net_solar_radiation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 210.0, dtype=np.float32),
                {'units': 'W/m^2'},
            ),
            'hres_surface_net_thermal_radiation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, -65.0, dtype=np.float32),
                {'units': 'W/m^2'},
            ),
            'pressure_surface': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 101325.0, dtype=np.float32),
                {'units': 'Pa'},
            ),
            'hres_temperature_2m': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 18.5, dtype=np.float32),
                {'units': 'degC'},
            ),
            'hres_total_precipitation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 6.0, dtype=np.float32),
                {'units': 'mm/day'},
            ),
        },
        coords={
            'init_time': init_times,
            'lead_time': ('lead_time', lead_days, {'units': 'days'}),
            'latitude': lats,
            'longitude': lons,
        },
    )

    ds_graphcast = xr.Dataset(
        data_vars={
            'graphcast_temperature_2m': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 17.25, dtype=np.float32),
                {'units': 'degC'},
            ),
            'graphcast_total_precipitation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 5.5, dtype=np.float32),
                {'units': 'mm/day'},
            ),
        },
        coords={
            'init_time': init_times,
            'lead_time': ('lead_time', lead_days, {'units': 'days'}),
            'latitude': lats,
            'longitude': lons,
        },
    )
    return ds_hres, ds_graphcast


@pytest.mark.unit
class ForecastZarrTest(unittest.TestCase):
    """Tests hermetic forecast Zarr extraction, units, and valid dates."""

    def setUp(self) -> None:
        """Set up temporary directory, fixtures, and test polygons."""
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.profile_mgr = ProfileManager(profiles_dir=self.root_path)

        ds_hres, ds_graphcast = _build_forecast_datasets(n_lead_days=15)
        self.hres_zarr = self.root_path / 'hres_fixture.zarr'
        self.gc_zarr = self.root_path / 'graphcast_fixture.zarr'
        ds_hres.to_zarr(str(self.hres_zarr), mode='w', consolidated=True)
        ds_graphcast.to_zarr(str(self.gc_zarr), mode='w', consolidated=True)

        self.poly1 = Polygon(
            [
                (-86.4, 40.7),
                (-86.1, 40.7),
                (-86.1, 40.9),
                (-86.4, 40.9),
                (-86.4, 40.7),
            ]
        )
        self.feat1 = {
            'type': 'Feature',
            'geometry': mapping(self.poly1),
            'properties': {
                'catchment_id': 'WABASH_LOGANSPORT',
                'name': 'Wabash River at Logansport, IN',
                'area_km2': 3120.0,
                'outlet': {'latitude': 40.75, 'longitude': -86.37},
            },
        }

        self.poly2 = Polygon(
            [
                (-87.5, 39.4),
                (-87.2, 39.4),
                (-87.2, 39.6),
                (-87.5, 39.6),
                (-87.5, 39.4),
            ]
        )
        self.feat2 = {
            'type': 'Feature',
            'geometry': mapping(self.poly2),
            'properties': {
                'catchment_id': 'WABASH_TERRE_HAUTE',
                'name': 'Wabash River at Terre Haute, IN',
                'area_km2': 7850.0,
                'outlet': {'latitude': 39.46, 'longitude': -87.41},
            },
        }

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()
        super().tearDown()

    def test_exact_variable_names_match_config(self) -> None:
        """Test that all 7 canonical variables and units are defined."""
        required_features = {
            'hres_surface_net_solar_radiation',
            'hres_surface_net_thermal_radiation',
            'hres_surface_pressure',
            'hres_temperature_2m',
            'hres_total_precipitation',
            'graphcast_temperature_2m',
            'graphcast_total_precipitation',
        }
        extracted_features = set(FORECAST_FEATURE_SPECS.keys())
        assert required_features == extracted_features
        assert FORECAST_FEATURE_SPECS['hres_surface_pressure']['units'] == 'kPa'
        assert (
            FORECAST_FEATURE_SPECS['hres_total_precipitation']['units']
            == 'mm/day'
        )

    def test_single_basin_forecast_extraction(self) -> None:
        """Test single-catchment extraction, kPa units, and D+(k-1) dates."""
        forecast_dir = self.profile_mgr.get_forecast_dir('user_alice')
        extractor = ForecastZarrExtractor(
            output_dir=forecast_dir,
            hres_zarr_uri=self.hres_zarr,
            graphcast_zarr_uri=self.gc_zarr,
        )

        res = extractor.fetch_and_archive(
            catchment_feature=self.feat1,
            horizon_days=_EXPECTED_HORIZON_10,
        )

        assert res['status'] == 'success'
        assert res['catchment_id'] == 'WABASH_LOGANSPORT'
        assert Path(res['zarr_path']).exists()
        assert Path(res['zarr_path']).name == 'forecast_latest.zarr'
        assert res['size_kb'] > 0
        assert not (forecast_dir / 'WABASH_LOGANSPORT').exists()

        ds = xr.open_zarr(res['zarr_path'], decode_timedelta=False)
        assert 'basin' in ds.dims
        assert 'date' in ds.dims
        assert 'lead_time' in ds.dims
        assert list(ds.lead_time.to_numpy()) == list(range(1, 11))
        assert ds.lead_time.attrs.get('units') == 'days'

        assert ds['hres_total_precipitation'].encoding.get('chunks') == (
            1,
            1,
            _EXPECTED_HORIZON_10,
        )
        assert ds['latitude'].encoding.get('chunks') == (1,)

        for feat in FORECAST_FEATURE_SPECS:
            assert feat in ds.data_vars
            assert ds[feat].shape == (1, 1, _EXPECTED_HORIZON_10)

        # Duplicate aliases must not be written
        assert 'total_precipitation' not in ds.data_vars
        assert 'temperature_2m' not in ds.data_vars
        assert '2m_temperature' not in ds.data_vars

        # Verify Pa -> kPa conversion and negative net thermal radiation
        assert ds['hres_surface_pressure'].attrs.get('units') == 'kPa'
        np.testing.assert_allclose(
            ds['hres_surface_pressure'].to_numpy(), 101.325, rtol=1e-4
        )
        np.testing.assert_allclose(
            ds['hres_surface_net_thermal_radiation'].to_numpy(),
            -65.0,
            rtol=1e-4,
        )

        # Verify valid_times: lead 1 on 2026-05-10 is 2026-05-10 (D + k - 1)
        meteogram = res['meteogram']
        assert meteogram['valid_times'][0] == '2026-05-10'
        assert meteogram['valid_times'][9] == '2026-05-19'
        assert (
            len(meteogram['hres']['precipitation_mm']) == _EXPECTED_HORIZON_10
        )
        assert (
            len(meteogram['graphcast']['precipitation_mm'])
            == _EXPECTED_HORIZON_10
        )

    def test_batch_multi_basin_forecast_extraction(self) -> None:
        """Test batch forecast extraction across multiple polygons."""
        forecast_dir = self.profile_mgr.get_forecast_dir('user_bob')
        extractor = ForecastZarrExtractor(
            output_dir=forecast_dir,
            hres_zarr_uri=self.hres_zarr,
            graphcast_zarr_uri=self.gc_zarr,
        )

        res = extractor.fetch_and_archive_batch(
            features=[self.feat1, self.feat2],
            horizon_days=_EXPECTED_HORIZON_15,
        )

        assert res['status'] == 'success'
        assert res['count'] == _EXPECTED_BATCH_BASINS
        assert res['total_basins_in_master'] == _EXPECTED_BATCH_BASINS
        assert Path(res['master_zarr_path']).exists()

        master_ds = xr.open_zarr(
            res['master_zarr_path'], decode_timedelta=False
        )
        assert len(master_ds.basin) == _EXPECTED_BATCH_BASINS
        assert len(master_ds.lead_time) == _EXPECTED_HORIZON_15
        assert 'WABASH_LOGANSPORT' in master_ds.basin.to_numpy()
        assert 'WABASH_TERRE_HAUTE' in master_ds.basin.to_numpy()

        for feat in FORECAST_FEATURE_SPECS:
            assert feat in master_ds.data_vars
            assert master_ds[feat].shape == (
                _EXPECTED_BATCH_BASINS,
                1,
                _EXPECTED_HORIZON_15,
            )

    def test_latest_issue_info_query(self) -> None:
        """Test querying forecast metadata and latest issue time."""
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path,
            hres_zarr_uri=self.hres_zarr,
            graphcast_zarr_uri=self.gc_zarr,
        )
        info = extractor.get_latest_issue_info()
        assert info['status'] == 'online'
        assert info['latest_issue_date'] == '2026-05-10'
        assert len(info['features']) == _EXPECTED_FEATURES_COUNT

    def test_missing_lead_steps_remain_nan(self) -> None:
        """Verify missing lead steps remain NaN instead of broadcasting."""
        ds_hres_short, ds_gc_short = _build_forecast_datasets(n_lead_days=3)
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path / 'short_horizon',
            datasets={'hres': ds_hres_short, 'graphcast': ds_gc_short},
        )
        res = extractor.fetch_and_archive(
            catchment_feature=self.feat1,
            horizon_days=5,
        )
        ds = xr.open_zarr(res['zarr_path'], decode_timedelta=False)
        tp = ds['hres_total_precipitation'].to_numpy()[0, 0, :]
        t2m = ds['hres_temperature_2m'].to_numpy()[0, 0, :]
        np.testing.assert_allclose(tp[:3], [6.0, 6.0, 6.0])
        assert np.all(np.isnan(tp[3:]))
        assert np.all(np.isnan(t2m[3:]))

    def test_missing_catalog_or_store_raises(self) -> None:
        """Verify missing forecast Zarr store raises FileNotFoundError."""
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path / 'missing_store',
            hres_zarr_uri=self.root_path / 'nonexistent_hres.zarr',
            graphcast_zarr_uri=self.gc_zarr,
        )
        with pytest.raises(FileNotFoundError):
            extractor.fetch_and_archive(
                catchment_feature=self.feat1, horizon_days=5
            )

    def test_missing_area_km2_raises_value_error(self) -> None:
        """Verify missing area_km2 in forecast feature raises ValueError."""
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path / 'missing_area',
            hres_zarr_uri=self.hres_zarr,
            graphcast_zarr_uri=self.gc_zarr,
        )
        bad_feat = {
            'type': 'Feature',
            'geometry': mapping(self.poly1),
            'properties': {'catchment_id': 'NO_AREA'},
        }
        with pytest.raises(ValueError, match='area_km2'):
            extractor.fetch_and_archive(
                catchment_feature=bad_feat, horizon_days=5
            )

    def test_sub_80_percent_coverage_returns_nan(self) -> None:
        """Verify < 80% valid spatial coverage in forecast grid yields NaN."""
        ds_hres, ds_gc = _build_forecast_datasets(n_lead_days=5)
        # Mask > 50% of the longitude columns with NaN
        hres_tp = ds_hres['hres_total_precipitation'].to_numpy().copy()
        hres_tp[:, :, :, :7] = np.nan
        ds_hres['hres_total_precipitation'] = (
            ds_hres['hres_total_precipitation'].dims,
            hres_tp,
            ds_hres['hres_total_precipitation'].attrs,
        )
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path / 'sparse_forecast',
            datasets={'hres': ds_hres, 'graphcast': ds_gc},
        )
        wide_poly = Polygon(
            [
                (-87.9, 39.2),
                (-85.6, 39.2),
                (-85.6, 41.3),
                (-87.9, 41.3),
                (-87.9, 39.2),
            ]
        )
        wide_feat = {
            'type': 'Feature',
            'geometry': mapping(wide_poly),
            'properties': {
                'catchment_id': 'SPARSE_FORECAST_BASIN',
                'area_km2': 9500.0,
            },
        }
        res = extractor.fetch_and_archive(
            catchment_feature=wide_feat, horizon_days=5
        )
        ds = xr.open_zarr(res['zarr_path'], decode_timedelta=False)
        tp = ds['hres_total_precipitation'].to_numpy()
        assert np.all(np.isnan(tp))


if __name__ == '__main__':
    unittest.main()
