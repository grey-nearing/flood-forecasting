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
"""Hermetic unit tests for multi-basin historical and forecast Zarr stores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon, mapping

from frontend.forecast_zarr import ForecastZarrExtractor
from frontend.historical_zarr import HistoricalZarrExtractor

_EXPECTED_BASINS_COUNT = 2
_EXPECTED_FORECAST_LEADS = 5


def _create_era5_fixture(zarr_path: Path) -> Path:
    """Create a local daily ERA5-Land Zarr fixture for hermetic tests."""
    times = pd.date_range('2021-01-01', '2021-03-31', freq='1D')
    lats = np.linspace(42.0, 39.0, 13, dtype=np.float64)
    lons = np.linspace(-89.0, -85.5, 15, dtype=np.float64)
    shape_3d = (len(times), len(lats), len(lons))

    ds = xr.Dataset(
        data_vars={
            'era5land_total_precipitation': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 5.25, dtype=np.float32),
            ),
            'era5land_temperature_2m': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 11.5, dtype=np.float32),
            ),
            'era5land_surface_pressure': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 100.8, dtype=np.float32),
            ),
            'era5land_surface_net_solar_radiation': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 160.0, dtype=np.float32),
            ),
            'era5land_surface_net_thermal_radiation': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, -55.0, dtype=np.float32),
            ),
            'era5land_dewpoint_temperature_2m': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 6.2, dtype=np.float32),
            ),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        },
    )
    ds.to_zarr(str(zarr_path), mode='w', consolidated=True)
    return zarr_path


def _create_forecast_fixtures(
    root_dir: Path,
) -> tuple[Path, Path]:
    """Create local HRES and GraphCast Zarr fixtures for hermetic tests."""
    init_times = pd.DatetimeIndex(['2026-05-10T00:00:00'])
    lead_days = np.arange(1, 11, dtype=np.int64)
    lats = np.linspace(42.0, 39.0, 13, dtype=np.float64)
    lons = np.linspace(-89.0, -85.5, 15, dtype=np.float64)
    shape_4d = (1, len(lead_days), len(lats), len(lons))

    hres_path = root_dir / 'hres_forecast.zarr'
    gc_path = root_dir / 'graphcast_forecast.zarr'

    ds_hres = xr.Dataset(
        data_vars={
            'hres_surface_net_solar_radiation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 195.0, dtype=np.float32),
                {'units': 'W/m^2'},
            ),
            'hres_surface_net_thermal_radiation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, -70.0, dtype=np.float32),
                {'units': 'W/m^2'},
            ),
            'pressure_surface': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 101200.0, dtype=np.float32),
                {'units': 'Pa'},
            ),
            'hres_temperature_2m': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 16.0, dtype=np.float32),
                {'units': 'degC'},
            ),
            'hres_total_precipitation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 4.0, dtype=np.float32),
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
    ds_hres.to_zarr(str(hres_path), mode='w', consolidated=True)

    ds_gc = xr.Dataset(
        data_vars={
            'graphcast_temperature_2m': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 15.5, dtype=np.float32),
                {'units': 'degC'},
            ),
            'graphcast_total_precipitation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 3.8, dtype=np.float32),
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
    ds_gc.to_zarr(str(gc_path), mode='w', consolidated=True)
    return hres_path, gc_path


@pytest.mark.unit
class WeatherZarrTest(unittest.TestCase):
    """Tests historical and forecast Zarr generation with hermetic fixtures."""

    def setUp(self) -> None:
        """Set up explicit catchment GeoJSON polygons and local Zarr stores."""
        super().setUp()
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

        self.era5_zarr = _create_era5_fixture(self.temp_dir / 'era5_land.zarr')
        self.hres_zarr, self.gc_zarr = _create_forecast_fixtures(self.temp_dir)

        poly1 = Polygon(
            [
                (-86.3, 40.6),
                (-85.9, 40.6),
                (-85.9, 40.9),
                (-86.3, 40.9),
                (-86.3, 40.6),
            ]
        )
        self.catchment1 = {
            'type': 'Feature',
            'geometry': mapping(poly1),
            'properties': {
                'catchment_id': 'wabash_logansport',
                'area_km2': 3120.0,
                'outlet': {'latitude': 40.75, 'longitude': -86.07},
            },
        }

        poly2 = Polygon(
            [
                (-87.6, 39.3),
                (-87.2, 39.3),
                (-87.2, 39.6),
                (-87.6, 39.6),
                (-87.6, 39.3),
            ]
        )
        self.catchment2 = {
            'type': 'Feature',
            'geometry': mapping(poly2),
            'properties': {
                'catchment_id': 'wabash_terre_haute',
                'area_km2': 7850.0,
                'outlet': {'latitude': 39.46, 'longitude': -87.41},
            },
        }

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        self.temp_dir_obj.cleanup()
        super().tearDown()

    def test_multi_basin_master_zarr_append(self) -> None:
        """Test that multiple basin requests append into Master Zarr."""
        out_dir = self.temp_dir / 'historical_out'
        extractor = HistoricalZarrExtractor(
            output_dir=out_dir,
            archive_uris={'era5': self.era5_zarr},
        )
        master_zarr_path = out_dir / 'historical_training_master.zarr'

        res1 = extractor.extract_and_archive(
            polygon_input=self.catchment1,
            start_date='2021-01-01',
            end_date='2021-03-31',
            freq='1D',
        )
        assert res1['status'] == 'success'
        assert res1['total_basins_in_master'] == 1
        assert master_zarr_path.exists()

        ds1 = xr.open_zarr(str(master_zarr_path), decode_timedelta=False)
        assert 'basin' in ds1.dims
        assert 'date' in ds1.dims
        assert len(ds1.basin) == 1
        assert ds1['era5land_total_precipitation'].shape == (1, 90)
        assert 'basin_id' in ds1.coords
        assert 'issue_time' in ds1.coords

        res2 = extractor.extract_and_archive(
            polygon_input=self.catchment2,
            start_date='2021-01-01',
            end_date='2021-03-31',
            freq='1D',
        )
        assert res2['status'] == 'success'
        assert res2['total_basins_in_master'] == _EXPECTED_BASINS_COUNT

        ds2 = xr.open_zarr(str(master_zarr_path), decode_timedelta=False)
        assert len(ds2.basin) == _EXPECTED_BASINS_COUNT
        assert ds2['era5land_total_precipitation'].shape == (
            _EXPECTED_BASINS_COUNT,
            90,
        )
        assert res1['basin_id'] in ds2.basin.to_numpy()
        assert res2['basin_id'] in ds2.basin.to_numpy()
        assert 'area_km2' in ds2
        assert 'latitude' in ds2
        assert 'longitude' in ds2

    def test_arbitrary_polygon_weather_extraction(self) -> None:
        """Test weather extraction for an arbitrary custom user polygon."""
        out_dir = self.temp_dir / 'custom_out'
        extractor = HistoricalZarrExtractor(
            output_dir=out_dir,
            archive_uris={'era5': self.era5_zarr},
        )

        custom_poly = Polygon(
            [
                (-88.5, 41.0),
                (-88.0, 41.5),
                (-87.5, 41.2),
                (-87.8, 40.8),
                (-88.5, 41.0),
            ]
        )
        custom_feature = {
            'type': 'Feature',
            'geometry': mapping(custom_poly),
            'properties': {
                'catchment_id': 'CUSTOM_USER_WATERSHED_001',
                'area_km2': 4250.0,
                'dataset_name': 'Uploaded Shapefile',
            },
        }

        res = extractor.extract_and_archive(
            polygon_input=custom_feature,
            start_date='2021-01-01',
            end_date='2021-01-31',
        )
        assert res['status'] == 'success'
        assert res['basin_id'] == 'CUSTOM_USER_WATERSHED_001'

        master_ds = xr.open_zarr(
            res['master_zarr_path'], decode_timedelta=False
        )
        assert 'CUSTOM_USER_WATERSHED_001' in master_ds.basin.to_numpy()
        assert master_ds['era5land_total_precipitation'].shape == (1, 31)
        np.testing.assert_allclose(
            master_ds['era5land_total_precipitation'].to_numpy(),
            5.25,
            rtol=1e-5,
        )

    def test_forecast_zarr_generation(self) -> None:
        """Test hermetic forecast Zarr generation with 5-day horizon."""
        out_dir = self.temp_dir / 'forecast_out'
        extractor = ForecastZarrExtractor(
            output_dir=out_dir,
            hres_zarr_uri=self.hres_zarr,
            graphcast_zarr_uri=self.gc_zarr,
        )
        res = extractor.fetch_and_archive(
            catchment_feature=self.catchment1,
            model='ifs',
            horizon_hours=120,
            step_interval_hours=6,
        )

        assert res['status'] == 'success'
        assert res['size_kb'] > 0
        assert Path(res['zarr_path']).exists()

        ds = xr.open_zarr(res['zarr_path'], decode_timedelta=False)
        assert 'hres_total_precipitation' in ds
        assert 'hres_temperature_2m' in ds
        assert 'graphcast_total_precipitation' in ds
        assert 'basin' in ds.dims
        assert 'date' in ds.dims
        assert 'lead_time' in ds.dims
        assert len(ds['lead_time']) == _EXPECTED_FORECAST_LEADS
        assert ds['hres_total_precipitation'].shape == (
            1,
            1,
            _EXPECTED_FORECAST_LEADS,
        )
        assert ds['hres_temperature_2m'].shape == (
            1,
            1,
            _EXPECTED_FORECAST_LEADS,
        )
        assert ds.lead_time.attrs.get('units') == 'days'
        assert ds['hres_surface_pressure'].attrs.get('units') == 'kPa'
        np.testing.assert_allclose(
            ds['hres_surface_pressure'].to_numpy(), 101.2, rtol=1e-4
        )


if __name__ == '__main__':
    unittest.main()
