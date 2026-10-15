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

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon, mapping

from frontend.forecast_zarr import (
    DYNAMICAL_DATASET_IDS,
    DYNAMICAL_GRAPHCAST_DATASET_ID,
    DYNAMICAL_HRES_DATASET_ID,
    FORECAST_FEATURE_SPECS,
    ForecastZarrExtractor,
)
from frontend.profile_manager import ProfileManager

if TYPE_CHECKING:
    from collections.abc import Callable

_EXPECTED_HORIZON_10 = 10
_EXPECTED_HORIZON_15 = 15
_EXPECTED_BATCH_BASINS = 2
_EXPECTED_FEATURES_COUNT = 7
_SECONDS_PER_DAY = 86400.0

# Dataset IDs that PR #93 (af5250f2) introduced in place of the published
# catalog entries; neither has ever existed on dynamical.org (Issue #121).
_LEGACY_UNPUBLISHED_IDS = (
    'ecmwf-ifs-hres-forecast',
    'google-graphcast-forecast',
)

# ``dynamical_catalog.list()`` as published on 2026-10-15 (dynamical_catalog
# 1.0.1; the 0.8.0 catalog is a strict subset that also contains both IDs
# configured in ``frontend.forecast_zarr``).
_DYNAMICAL_CATALOG_SNAPSHOT: tuple[str, ...] = (
    'dwd-icon-eu-forecast-5-day',
    'eccc-hrdps-forecast',
    'ecmwf-aifs-ens-forecast',
    'ecmwf-aifs-single-forecast',
    'ecmwf-aifs-single-forecast-virtual',
    'ecmwf-ifs-ens-forecast-15-day-0-25-degree',
    'ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree',
    'ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree',
    'nasa-imerg-analysis-early',
    'nasa-imerg-analysis-late',
    'noaa-gefs-analysis',
    'noaa-gefs-analysis-0-25-degree-virtual',
    'noaa-gefs-forecast-10-day-0-25-degree-virtual',
    'noaa-gefs-forecast-16-day-0-5-degree-virtual',
    'noaa-gefs-forecast-35-day',
    'noaa-gefs-forecast-35-day-0-5-degree-virtual',
    'noaa-gfs-analysis',
    'noaa-gfs-analysis-virtual',
    'noaa-gfs-forecast',
    'noaa-gfs-forecast-virtual',
    'noaa-hrrr-analysis',
    'noaa-hrrr-analysis-virtual',
    'noaa-hrrr-forecast-18-hour-virtual',
    'noaa-hrrr-forecast-48-hour',
    'noaa-hrrr-forecast-48-hour-virtual',
    'noaa-mrms-conus-analysis-hourly',
    'noaa-refs-forecast-3-hourly-virtual',
    'noaa-refs-forecast-hourly-virtual',
    'noaa-rrfs-ens-forecast-virtual',
    'noaa-rrfs-forecast-18-hour-virtual',
    'noaa-rrfs-forecast-84-hour-virtual',
    'noaa-rrfs-forecast-sub-hourly-virtual',
    'ucsb-chc-chirps-analysis-final',
    'ucsb-chc-chirps-analysis-preliminary',
)


class _UnknownDatasetError(ValueError):
    """Stand-in for ``dynamical_catalog.exceptions.UnknownDatasetError``."""


@dataclasses.dataclass
class _MockDynamicalCatalog:
    """Hermetic stand-in for the ``dynamical_catalog`` module.

    Mirrors the real network boundary: ``list()`` returns the published
    dataset IDs and ``open()`` raises for anything that is not published.
    """

    datasets: tuple[str, ...]
    opener: Callable[[str], xr.Dataset]
    opened: list[str] = dataclasses.field(default_factory=list)

    def list(self) -> list[str]:
        return list(self.datasets)

    def open(self, dataset_id: str, **_kwargs: object) -> xr.Dataset:
        if dataset_id not in self.datasets:
            msg = f'Unknown dataset {dataset_id!r}'
            raise _UnknownDatasetError(msg)
        self.opened.append(dataset_id)
        return self.opener(dataset_id)


def _build_dynamical_like_datasets(
    *,
    include_net_radiation: bool,
    control_temperature_c: float = 12.0,
    perturbed_temperature_c: float = 30.0,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build IFS-ENS-like (HRES stand-in) and AIFS-like forecast datasets.

    Dimension order, coordinate dtypes, variable names, and units follow the
    dynamical.org ``ecmwf-ifs-ens-forecast-15-day-0-25-degree`` and
    ``ecmwf-aifs-single-forecast`` stores on a 0.25-degree sub-grid. Ensemble
    member 0 (the control run) carries ``control_temperature_c``; member 1
    carries ``perturbed_temperature_c`` so control-member selection is
    observable. Precipitation is a rate of exactly 1 mm/day.
    """
    init_times = pd.DatetimeIndex(
        ['2026-10-13T00:00:00', '2026-10-14T00:00:00']
    )
    lead_hours = np.arange(6, 49, 6, dtype=np.int64)
    lead_time = pd.to_timedelta(lead_hours, unit='h')
    lats = np.arange(41.5, 38.9, -0.25, dtype=np.float64)
    lons = np.arange(-88.0, -85.4, 0.25, dtype=np.float64)
    n_members = 2
    ens_dims = [
        'init_time',
        'lead_time',
        'ensemble_member',
        'latitude',
        'longitude',
    ]
    ens_shape = (
        len(init_times),
        len(lead_hours),
        n_members,
        len(lats),
        len(lons),
    )

    def _ens(value: float, units: str) -> tuple[list[str], np.ndarray, dict]:
        return (
            ens_dims,
            np.full(ens_shape, value, dtype=np.float32),
            {'units': units},
        )

    temperature = np.full(ens_shape, control_temperature_c, dtype=np.float32)
    temperature[:, :, 1, :, :] = perturbed_temperature_c
    precip_rate = np.float32(1.0 / _SECONDS_PER_DAY)
    hres_vars: dict[str, Any] = {
        'temperature_2m': (ens_dims, temperature, {'units': 'degree_Celsius'}),
        'precipitation_surface': _ens(precip_rate, 'kg m-2 s-1'),
        'pressure_surface': _ens(101325.0, 'Pa'),
        'downward_short_wave_radiation_flux_surface': _ens(250.0, 'W m-2'),
        'downward_long_wave_radiation_flux_surface': _ens(320.0, 'W m-2'),
    }
    if include_net_radiation:
        hres_vars['hres_surface_net_solar_radiation'] = _ens(210.0, 'W/m^2')
        hres_vars['hres_surface_net_thermal_radiation'] = _ens(-65.0, 'W/m^2')
    coords = {
        'init_time': init_times,
        'lead_time': lead_time,
        'latitude': lats,
        'longitude': lons,
    }
    ds_hres = xr.Dataset(
        data_vars=hres_vars,
        coords={**coords, 'ensemble_member': np.arange(n_members)},
    )

    det_dims = ['init_time', 'lead_time', 'latitude', 'longitude']
    det_shape = (len(init_times), len(lead_hours), len(lats), len(lons))
    ds_aifs = xr.Dataset(
        data_vars={
            'temperature_2m': (
                det_dims,
                np.full(det_shape, 9.5, dtype=np.float32),
                {'units': 'degree_Celsius'},
            ),
            'precipitation_surface': (
                det_dims,
                np.full(det_shape, precip_rate, dtype=np.float32),
                {'units': 'kg m-2 s-1'},
            ),
        },
        coords=coords,
    )
    return ds_hres, ds_aifs


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
        assert info['source'] == 'explicit'
        assert info['source_datasets'] == {
            'hres': str(self.hres_zarr),
            'graphcast': str(self.gc_zarr),
        }

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

    # ------------------------------------------------------------------
    # Issue #121: dynamical.org catalog dataset identifiers
    # ------------------------------------------------------------------

    def _patch_dynamical_catalog(self, catalog: _MockDynamicalCatalog) -> None:
        """Install ``catalog`` as the ``dynamical_catalog`` module."""
        for patcher in (
            mock.patch.dict(sys.modules, {'dynamical_catalog': catalog}),
            mock.patch(
                'frontend.forecast_zarr.HAS_DYNAMICAL_CATALOG', new=True
            ),
            mock.patch.dict(
                'frontend.forecast_zarr._DYNAMICAL_CACHE', {}, clear=True
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_configured_dataset_ids_are_published_catalog_entries(
        self,
    ) -> None:
        """Verify configured IDs are published and legacy IDs are not."""
        for dataset_id in DYNAMICAL_DATASET_IDS.values():
            assert dataset_id in _DYNAMICAL_CATALOG_SNAPSHOT
        for legacy_id in _LEGACY_UNPUBLISHED_IDS:
            assert legacy_id not in _DYNAMICAL_CATALOG_SNAPSHOT
            assert legacy_id not in DYNAMICAL_DATASET_IDS.values()

    def test_forecast_info_opens_published_dynamical_datasets(self) -> None:
        """Verify forecast info resolves through published catalog IDs."""
        ds_hres, ds_aifs = _build_dynamical_like_datasets(
            include_net_radiation=False
        )
        by_id = {
            DYNAMICAL_HRES_DATASET_ID: ds_hres,
            DYNAMICAL_GRAPHCAST_DATASET_ID: ds_aifs,
        }
        catalog = _MockDynamicalCatalog(
            datasets=_DYNAMICAL_CATALOG_SNAPSHOT,
            opener=lambda dataset_id: by_id[dataset_id],
        )
        self._patch_dynamical_catalog(catalog)

        extractor = ForecastZarrExtractor(output_dir=self.root_path / 'live')
        info = extractor.get_latest_issue_info()

        assert catalog.opened == [
            DYNAMICAL_HRES_DATASET_ID,
            DYNAMICAL_GRAPHCAST_DATASET_ID,
        ]
        assert info['status'] == 'online'
        assert info['source'] == 'dynamical.org'
        assert info['source_datasets'] == {
            'hres': 'ecmwf-ifs-ens-forecast-15-day-0-25-degree',
            'graphcast': 'ecmwf-aifs-single-forecast',
        }
        assert info['hres_latest_init_time'] == '2026-10-14T00:00:00'
        assert info['graphcast_latest_init_time'] == '2026-10-14T00:00:00'
        assert info['latest_issue_date'] == '2026-10-14'
        assert len(info['features']) == _EXPECTED_FEATURES_COUNT

        # A second call within the TTL is served from the session cache.
        extractor.get_latest_issue_info()
        assert len(catalog.opened) == len(DYNAMICAL_DATASET_IDS)

    def test_unpublished_dataset_id_fails_before_open(self) -> None:
        """Verify a retired catalog ID raises a descriptive error pre-open."""
        _, ds_aifs = _build_dynamical_like_datasets(include_net_radiation=False)
        catalog = _MockDynamicalCatalog(
            datasets=tuple(
                dataset_id
                for dataset_id in _DYNAMICAL_CATALOG_SNAPSHOT
                if dataset_id != DYNAMICAL_HRES_DATASET_ID
            ),
            opener=lambda _dataset_id: ds_aifs,
        )
        self._patch_dynamical_catalog(catalog)

        extractor = ForecastZarrExtractor(output_dir=self.root_path / 'gone')
        with pytest.raises(RuntimeError, match=DYNAMICAL_HRES_DATASET_ID):
            extractor.get_latest_issue_info()
        assert catalog.opened == []

    def test_partial_explicit_sources_raise_value_error(self) -> None:
        """Verify one explicit store is never paired with a catalog dataset."""
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path / 'partial',
            hres_zarr_uri=self.hres_zarr,
        )
        with pytest.raises(ValueError, match='both the hres and graphcast'):
            extractor.get_latest_issue_info()

    def test_ensemble_control_member_drives_hres_extraction(self) -> None:
        """Verify IFS ENS member 0 and dynamical units drive extraction."""
        ds_hres, ds_aifs = _build_dynamical_like_datasets(
            include_net_radiation=True
        )
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path / 'ens_control',
            datasets={'hres': ds_hres, 'graphcast': ds_aifs},
        )
        res = extractor.fetch_and_archive(
            catchment_feature=self.feat1, horizon_days=2
        )
        assert res['status'] == 'success'

        ds = xr.open_zarr(res['zarr_path'], decode_timedelta=False)
        np.testing.assert_allclose(
            ds['hres_temperature_2m'].to_numpy()[0, 0, :], [12.0, 12.0]
        )
        np.testing.assert_allclose(
            ds['hres_total_precipitation'].to_numpy()[0, 0, :],
            [1.0, 1.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            ds['hres_surface_pressure'].to_numpy()[0, 0, :],
            [101.325, 101.325],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            ds['graphcast_temperature_2m'].to_numpy()[0, 0, :], [9.5, 9.5]
        )
        assert res['meteogram']['valid_times'][0] == '2026-10-14'

    def test_downward_only_radiation_fails_loudly(self) -> None:
        """Verify downward-only fluxes are never passed off as net radiation."""
        ds_hres, ds_aifs = _build_dynamical_like_datasets(
            include_net_radiation=False
        )
        extractor = ForecastZarrExtractor(
            output_dir=self.root_path / 'no_net_radiation',
            datasets={'hres': ds_hres, 'graphcast': ds_aifs},
        )
        with pytest.raises(KeyError, match='hres_surface_net_solar_radiation'):
            extractor.fetch_and_archive(
                catchment_feature=self.feat1, horizon_days=2
            )


if __name__ == '__main__':
    unittest.main()
