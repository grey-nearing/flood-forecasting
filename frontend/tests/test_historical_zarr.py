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
"""Hermetic unit tests for historical weather Zarr extraction and registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import xarray as xr

from frontend.historical_zarr import HistoricalZarrExtractor
from frontend.weather_sources import (
    get_weather_source,
    list_weather_sources,
)

Polygon = shapely.geometry.Polygon

_EXPECTED_JAN_DAYS = 31
_EXPECTED_MAY_DAYS = 15
_EXPECTED_BATCH_BASINS = 2


def _write_gridded_zarr(  # noqa: PLR0913, PLR0917
    zarr_path: Path,
    var_values: dict[str, float | np.ndarray],
    start_date: str = '2020-01-01',
    end_date: str = '2020-01-31',
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
) -> Path:
    """Create a real local gridded Zarr store for hermetic testing."""
    times = pd.date_range(start_date, end_date, freq='1D')
    lat_arr = (
        lats
        if lats is not None
        else np.array([39.75, 40.0, 40.25, 40.5, 40.75], dtype=np.float64)
    )
    lon_arr = (
        lons
        if lons is not None
        else np.array([-87.75, -87.5, -87.25, -87.0, -86.75], dtype=np.float64)
    )
    shape_3d = (len(times), len(lat_arr), len(lon_arr))

    data_vars = {}
    for var_name, val in var_values.items():
        if isinstance(val, np.ndarray):
            arr = val.astype(np.float32)
        else:
            arr = np.full(shape_3d, float(val), dtype=np.float32)
        data_vars[var_name] = (['time', 'latitude', 'longitude'], arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': times,
            'latitude': lat_arr,
            'longitude': lon_arr,
        },
    )
    ds.to_zarr(str(zarr_path), mode='w', consolidated=True)
    return zarr_path


@pytest.mark.unit
class HistoricalZarrExtractorTest(unittest.TestCase):
    """Tests hermetic historical Zarr extraction and coverage rules."""

    def setUp(self) -> None:
        """Set up temporary directory and synthetic gridded Zarr stores."""
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / 'output'
        self.archives_dir = Path(self.temp_dir.name) / 'archives'
        self.archives_dir.mkdir(parents=True, exist_ok=True)

        self.cpc_zarr = _write_gridded_zarr(
            self.archives_dir / 'cpc.zarr',
            {'cpc_precipitation': 12.5},
            start_date='2020-01-01',
            end_date='2020-01-31',
        )
        self.imerg_zarr = _write_gridded_zarr(
            self.archives_dir / 'imerg.zarr',
            {'imerg_precipitation': 8.25},
            start_date='2021-05-01',
            end_date='2021-05-15',
        )
        self.era5_zarr = _write_gridded_zarr(
            self.archives_dir / 'era5.zarr',
            {
                'era5land_total_precipitation': 4.5,
                'era5land_temperature_2m': 15.2,
                'era5land_surface_pressure': 101.3,
                'era5land_surface_net_solar_radiation': 185.0,
                'era5land_surface_net_thermal_radiation': -60.0,
                'era5land_dewpoint_temperature_2m': 9.8,
            },
            start_date='2020-01-01',
            end_date='2020-01-10',
        )

        self.extractor = HistoricalZarrExtractor(
            output_dir=self.output_dir,
            archive_uris={
                'cpc': self.cpc_zarr,
                'imerg': self.imerg_zarr,
                'era5': self.era5_zarr,
            },
        )

        self.sample_poly = Polygon(
            [
                (-87.5, 40.0),
                (-87.0, 40.5),
                (-87.0, 40.0),
                (-87.5, 40.0),
            ]
        )
        self.sample_feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-87.5, 40.0],
                        [-87.0, 40.5],
                        [-87.0, 40.0],
                        [-87.5, 40.0],
                    ]
                ],
            },
            'properties': {
                'catchment_id': 'test_basin_01',
                'name': 'Test Watershed',
                'area_km2': 450.0,
            },
        }

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()
        super().tearDown()

    def test_weather_sources_registry(self) -> None:
        """Verify registered weather datasets use canonical variable names."""
        sources = list_weather_sources()
        source_ids = [s['id'] for s in sources]
        assert 'cpc' in source_ids
        assert 'imerg' in source_ids
        assert 'era5' in source_ids
        assert 'ifs' in source_ids
        assert 'graphcast' in source_ids

        cpc = get_weather_source('cpc')
        assert cpc.available_start == '1979-01-01'
        assert 'cpc_precipitation' in cpc.default_variables

        imerg = get_weather_source('imerg')
        assert imerg.available_start == '2000-06-01'
        assert 'imerg_precipitation' in imerg.default_variables

        era5 = get_weather_source('era5')
        assert 'era5land_total_precipitation' in era5.default_variables
        assert 'era5land_temperature_2m' in era5.default_variables

        ifs = get_weather_source('ifs')
        assert 'hres_total_precipitation' in ifs.default_variables
        assert 'hres_temperature_2m' in ifs.default_variables

        gc = get_weather_source('graphcast')
        assert 'graphcast_total_precipitation' in gc.default_variables

    def test_single_basin_historical_extraction(self) -> None:
        """Verify extracting weather data writes 2-D (basin, date) arrays."""
        result = self.extractor.extract_and_archive(
            polygon_input=self.sample_feature,
            basin_id='test_basin_01',
            weather_source='cpc',
            start_date='2020-01-01',
            end_date='2020-01-31',
            freq='1D',
        )

        assert result['status'] == 'success'
        assert result['basin_id'] == 'test_basin_01'
        assert result['n_timesteps'] == _EXPECTED_JAN_DAYS
        assert self.extractor.master_zarr_path.exists()
        assert not (self.output_dir / 'test_basin_01').exists()

        ds = xr.open_zarr(
            str(self.extractor.master_zarr_path), decode_timedelta=False
        )
        assert 'basin' in ds.dims
        assert 'date' in ds.dims
        assert 'lead_time' not in ds.dims
        assert list(ds.basin.to_numpy()) == ['test_basin_01']
        assert len(ds.date) == _EXPECTED_JAN_DAYS

        assert ds['cpc_precipitation'].dims == ('basin', 'date')
        assert ds['cpc_precipitation'].encoding.get('chunks') == (
            1,
            _EXPECTED_JAN_DAYS,
        )
        assert ds['cpc_precipitation'].attrs.get('units') == 'mm/day'
        assert ds['latitude'].encoding.get('chunks') == (1,)

        np.testing.assert_allclose(
            ds['cpc_precipitation'].to_numpy(),
            np.full((1, _EXPECTED_JAN_DAYS), 12.5, dtype=np.float32),
            rtol=1e-5,
        )
        assert 'basin_area' in ds.data_vars
        assert 'area_km2' in ds.data_vars

    def test_multi_basin_batch_append(self) -> None:
        """Verify appending multiple basins writes a multi-basin Zarr store."""
        feat2 = {
            'type': 'Feature',
            'geometry': self.sample_feature['geometry'],
            'properties': {
                'catchment_id': 'test_basin_02',
                'name': 'Second Watershed',
                'area_km2': 720.0,
            },
        }

        batch_res = self.extractor.extract_and_archive_batch(
            features=[self.sample_feature, feat2],
            weather_source='imerg',
            start_date='2021-05-01',
            end_date='2021-05-15',
            freq='1D',
        )

        assert batch_res['status'] == 'success'
        assert batch_res['basins_extracted_count'] == _EXPECTED_BATCH_BASINS
        assert batch_res['total_basins_in_master'] == _EXPECTED_BATCH_BASINS

        ds = xr.open_zarr(
            str(self.extractor.master_zarr_path), decode_timedelta=False
        )
        assert len(ds.basin) == _EXPECTED_BATCH_BASINS
        assert 'test_basin_01' in ds.basin.to_numpy()
        assert 'test_basin_02' in ds.basin.to_numpy()
        assert len(ds.date) == _EXPECTED_MAY_DAYS
        assert 'imerg_precipitation' in ds.data_vars
        assert ds['imerg_precipitation'].dims == ('basin', 'date')
        assert ds['imerg_precipitation'].encoding.get('chunks') == (
            1,
            _EXPECTED_MAY_DAYS,
        )
        # Verify IMERG is not multiplied by 24x
        np.testing.assert_allclose(
            ds['imerg_precipitation'].to_numpy(),
            np.full(
                (_EXPECTED_BATCH_BASINS, _EXPECTED_MAY_DAYS),
                8.25,
                dtype=np.float32,
            ),
            rtol=1e-5,
        )

    def test_missing_archive_raises_file_not_found(self) -> None:
        """Verify a missing Zarr archive raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract_and_archive(
                polygon_input=self.sample_feature,
                weather_source='cpc',
                archive_uri=self.archives_dir / 'does_not_exist.zarr',
                start_date='2020-01-01',
                end_date='2020-01-05',
            )

    def test_missing_area_km2_raises_value_error(self) -> None:
        """Verify missing or non-positive area_km2 raises ValueError."""
        bad_feature = {
            'type': 'Feature',
            'geometry': self.sample_feature['geometry'],
            'properties': {'catchment_id': 'no_area_basin'},
        }
        with pytest.raises(ValueError, match='area_km2'):
            self.extractor.extract_and_archive(
                polygon_input=bad_feature,
                weather_source='cpc',
                start_date='2020-01-01',
                end_date='2020-01-05',
            )

    def test_sub_80_percent_coverage_returns_nan(self) -> None:
        """Verify < 80% valid spatial coverage produces NaN."""
        lats = np.array([40.1, 40.3, 40.5, 40.7], dtype=np.float64)
        lons = np.array([-87.7, -87.5, -87.3, -87.1], dtype=np.float64)
        grid = np.full((5, len(lats), len(lons)), 10.0, dtype=np.float32)
        # Mask half the grid with NaN -> 50% coverage (< 80% threshold)
        grid[:, :, :2] = np.nan

        sparse_zarr = _write_gridded_zarr(
            self.archives_dir / 'sparse_cpc.zarr',
            {'cpc_precipitation': grid},
            start_date='2020-01-01',
            end_date='2020-01-05',
            lats=lats,
            lons=lons,
        )
        box_feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-87.8, 40.0],
                        [-87.0, 40.0],
                        [-87.0, 40.8],
                        [-87.8, 40.8],
                        [-87.8, 40.0],
                    ]
                ],
            },
            'properties': {
                'catchment_id': 'partial_nan_basin',
                'area_km2': 1200.0,
            },
        }
        self.extractor.extract_and_archive(
            polygon_input=box_feature,
            weather_source='cpc',
            archive_uri=sparse_zarr,
            start_date='2020-01-01',
            end_date='2020-01-05',
        )
        ds = xr.open_zarr(
            str(self.extractor.master_zarr_path), decode_timedelta=False
        )
        vals = ds['cpc_precipitation'].sel(basin='partial_nan_basin').to_numpy()
        assert np.all(np.isnan(vals))


if __name__ == '__main__':
    unittest.main()
