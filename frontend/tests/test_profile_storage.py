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
"""Unit tests for the 13-category OpenHydroNet storage architecture."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from frontend import config
from frontend.profile_manager import ProfileManager

_EXPECTED_BASINS_COUNT = 3
_EXPECTED_INVENTORY_POLYGONS = 2
_EXPECTED_INVENTORY_VARS = 4
_EXPECTED_TIMESTEPS = 10
_AREA_TOL = 1e-2


@pytest.mark.unit
class ProfileStorageArchitectureTest(unittest.TestCase):
    """Test the 13-category storage directories, accessors, and basin splits."""

    def setUp(self) -> None:
        """Create an isolated temporary profiles directory."""
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.profiles_root = Path(self.temp_dir.name) / 'users'
        self.pm = ProfileManager(profiles_dir=self.profiles_root)

    def tearDown(self) -> None:
        """Clean up the temporary profiles directory."""
        self.temp_dir.cleanup()
        super().tearDown()

    def test_config_shared_and_legacy_constants_preserved(self) -> None:
        """Verify shared and legacy directory constants in frontend.config."""
        for attr in (
            'BASE_DIR',
            'DATA_DIR',
            'POLYGONS_DIR',
            'ATTRIBUTES_DIR',
            'HISTORICAL_DIR',
            'FORECAST_DIR',
            'REALTIME_DIR',
            'ARCHIVES_DIR',
            'BASE_LAYERS_DIR',
            'DEM_DIR',
            'RIVER_NETWORKS_DIR',
            'HYDRO_BASINS_DIR',
            'CLIMATOLOGY_DIR',
            'STATIC_DIR',
            'HYDRO_DATASETS',
            'WEATHER_CONFIG',
        ):
            assert hasattr(config, attr), (
                f'Missing legacy config constant: {attr}'
            )

        assert isinstance(config.FLOOD_FORECASTING_REPO_DIR, Path)
        assert config.SHARED_DIR == config.DATA_DIR / 'shared'
        assert config.SHARED_DEMS_DIR == config.SHARED_DIR / 'dems'
        assert config.SHARED_HYDROATLAS_DIR == config.SHARED_DIR / 'hydroatlas'
        assert (
            config.SHARED_HYDRORIVERS_DIR == config.SHARED_DIR / 'hydrorivers'
        )
        assert (
            config.SHARED_HYDROFABRICS_DIR == config.SHARED_DIR / 'hydrofabrics'
        )
        assert (
            config.SHARED_GRIDDED_ARCHIVES_DIR
            == config.SHARED_DIR / 'gridded_archives'
        )
        assert (
            config.SHARED_WEATHER_TILES_CACHE_DIR
            == config.SHARED_DIR / 'weather_tiles_cache'
        )
        assert config.SHARED_MODELS_DIR == config.SHARED_DIR / 'models'
        assert config.SHARED_MAAS_CACHE_DIR == config.SHARED_DIR / 'maas_cache'
        assert config.USERS_DIR == config.DATA_DIR / 'users'

    def test_ensure_flood_forecasting_on_sys_path(self) -> None:
        """Verify ensure_flood_forecasting_on_sys_path inserts repo and src."""
        fake_repo = Path(self.temp_dir.name) / 'fake_flood_forecasting'
        fake_src = fake_repo / 'src'
        fake_src.mkdir(parents=True, exist_ok=True)

        result = config.ensure_flood_forecasting_on_sys_path(repo_dir=fake_repo)
        assert result is True
        assert str(fake_repo) in sys.path
        assert str(fake_src) in sys.path

        sys.path.remove(str(fake_repo))
        sys.path.remove(str(fake_src))

        missing_repo = Path(self.temp_dir.name) / 'nonexistent_repo'
        assert (
            config.ensure_flood_forecasting_on_sys_path(repo_dir=missing_repo)
            is False
        )

    def test_extend_multimet_package_path_validates_multimet_dir(self) -> None:
        """Verify extend_multimet_package_path checks for a multimet package."""
        repo_with_multimet = Path(self.temp_dir.name) / 'repo_with_multimet'
        pkg_dir = repo_with_multimet / 'multimet'
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / '__init__.py').write_text('', encoding='utf-8')

        assert (
            config.extend_multimet_package_path(repo_dir=repo_with_multimet)
            is True
        )
        assert str(repo_with_multimet) in sys.path
        sys.path.remove(str(repo_with_multimet))

        empty_repo = Path(self.temp_dir.name) / 'empty_repo'
        empty_repo.mkdir(parents=True, exist_ok=True)
        assert config.extend_multimet_package_path(repo_dir=empty_repo) is False
        if str(empty_repo) in sys.path:
            sys.path.remove(str(empty_repo))

        missing_repo = Path(self.temp_dir.name) / 'lonely' / 'flood-forecasting'
        assert (
            config.extend_multimet_package_path(repo_dir=missing_repo) is False
        )

    def test_13_category_directory_provisioning_and_accessors(self) -> None:
        """Verify all 13 category directories and 21 accessor methods."""
        self.pm.login_profile('ministry_kenya')
        p_dir = self.pm.get_profile_dir('ministry_kenya')

        for legacy_sub in (
            'polygons',
            'attributes',
            'historical',
            'forecast',
            'archives',
        ):
            assert (p_dir / legacy_sub).is_dir()

        expected_subdirs = [
            p_dir / 'catchments' / 'shapes',
            p_dir / 'catchments' / 'zonal_weights',
            p_dir / 'catchments' / 'basin_lists',
            p_dir / 'statics' / 'caravan_csv',
            p_dir / 'dynamics' / 'ERA5_LAND',
            p_dir / 'dynamics' / 'CPC',
            p_dir / 'dynamics' / 'IMERG',
            p_dir / 'dynamics' / 'HRES',
            p_dir / 'dynamics' / 'GRAPHCAST',
            p_dir / 'realtime' / 'dynamics' / 'ERA5_LAND',
            p_dir / 'realtime' / 'dynamics' / 'CPC',
            p_dir / 'realtime' / 'dynamics' / 'IMERG',
            p_dir / 'realtime' / 'dynamics' / 'HRES',
            p_dir / 'realtime' / 'dynamics' / 'GRAPHCAST',
            p_dir / 'realtime' / 'scenarios',
            p_dir / 'targets' / 'uploads',
            p_dir / 'assimilation' / 'uploads',
            p_dir / 'assimilation' / 'da_states',
            p_dir / 'models' / 'hot_start',
            p_dir / 'models' / 'runs',
            p_dir / 'forecasts',
            p_dir / 'maas' / 'snapshots',
            p_dir / 'jobs',
        ]
        for directory in expected_subdirs:
            assert directory.is_dir(), (
                f'Expected directory not provisioned: {directory}'
            )

        assert (
            self.pm.get_catchments_dir('ministry_kenya') == p_dir / 'catchments'
        )
        assert (
            self.pm.get_catchment_shapes_dir('ministry_kenya')
            == p_dir / 'catchments' / 'shapes'
        )
        assert (
            self.pm.get_zonal_weights_dir('ministry_kenya')
            == p_dir / 'catchments' / 'zonal_weights'
        )
        assert (
            self.pm.get_basin_lists_dir('ministry_kenya')
            == p_dir / 'catchments' / 'basin_lists'
        )
        assert self.pm.get_statics_dir('ministry_kenya') == p_dir / 'statics'
        assert (
            self.pm.get_statics_zarr_path('ministry_kenya')
            == p_dir / 'statics' / 'attributes.zarr'
        )
        assert self.pm.get_dynamics_dir('ministry_kenya') == p_dir / 'dynamics'
        assert (
            self.pm.get_dynamics_dir('ministry_kenya', 'cpc')
            == p_dir / 'dynamics' / 'CPC'
        )
        assert (
            self.pm.get_dynamics_zarr_path('ministry_kenya', 'era5_land')
            == p_dir / 'dynamics' / 'ERA5_LAND' / 'timeseries.zarr'
        )
        assert (
            self.pm.get_realtime_dir_v2('ministry_kenya') == p_dir / 'realtime'
        )
        assert (
            self.pm.get_realtime_dynamics_zarr_path(
                'ministry_kenya', 'graphcast'
            )
            == p_dir / 'realtime' / 'dynamics' / 'GRAPHCAST' / 'timeseries.zarr'
        )
        assert self.pm.get_targets_dir('ministry_kenya') == p_dir / 'targets'
        assert (
            self.pm.get_targets_zarr_path('ministry_kenya')
            == p_dir / 'targets' / 'streamflow.zarr'
        )
        assert (
            self.pm.get_return_periods_path('ministry_kenya')
            == p_dir / 'targets' / 'return_periods.json'
        )
        assert (
            self.pm.get_assimilation_dir('ministry_kenya')
            == p_dir / 'assimilation'
        )
        assert (
            self.pm.get_assimilation_zarr_path('ministry_kenya')
            == p_dir / 'assimilation' / 'streamflow_realtime.zarr'
        )
        assert (
            self.pm.get_da_states_dir('ministry_kenya')
            == p_dir / 'assimilation' / 'da_states'
        )
        assert self.pm.get_models_dir('ministry_kenya') == p_dir / 'models'
        assert (
            self.pm.get_hot_start_dir('ministry_kenya')
            == p_dir / 'models' / 'hot_start'
        )
        assert (
            self.pm.get_model_runs_dir('ministry_kenya')
            == p_dir / 'models' / 'runs'
        )
        assert (
            self.pm.get_forecasts_dir('ministry_kenya') == p_dir / 'forecasts'
        )
        assert (
            self.pm.get_forecasts_dir('ministry_kenya', basin_id='tana_01')
            == p_dir / 'forecasts' / 'tana_01'
        )
        assert (p_dir / 'forecasts' / 'tana_01' / 'history').is_dir()
        assert self.pm.get_jobs_dir('ministry_kenya') == p_dir / 'jobs'

    def test_save_watersheds_syncs_per_basin_geojson_and_basin_lists(
        self,
    ) -> None:
        """Verify per-basin GeoJSON shapes and disjoint train/val/test split."""
        self.pm.login_profile('hydro_agency')
        features = [
            {
                'type': 'Feature',
                'id': 'basin_alpha',
                'properties': {'name': 'Alpha River', 'area_km2': 320.5},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            [36.0, -1.0],
                            [36.5, -1.0],
                            [36.5, -0.5],
                            [36.0, -1.0],
                        ]
                    ],
                },
            },
            {
                'type': 'Feature',
                'properties': {'basin_id': 'basin_beta', 'SUB_AREA': 510.0},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            [37.0, -1.0],
                            [37.5, -1.0],
                            [37.5, -0.5],
                            [37.0, -1.0],
                        ]
                    ],
                },
            },
            {
                'type': 'Feature',
                'properties': {'HYBAS_ID': 1120034560, 'UP_AREA': 1240.0},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            [38.0, -1.0],
                            [38.5, -1.0],
                            [38.5, -0.5],
                            [38.0, -1.0],
                        ]
                    ],
                },
            },
        ]

        assert (
            self.pm.save_watersheds(features, username='hydro_agency') is True
        )

        poly_ws = self.pm.get_polygons_dir('hydro_agency') / 'watersheds.json'
        catch_ws = (
            self.pm.get_catchments_dir('hydro_agency') / 'watersheds.geojson'
        )
        assert poly_ws.exists()
        assert catch_ws.exists()

        catch_fc = json.loads(catch_ws.read_text(encoding='utf-8'))
        assert catch_fc['type'] == 'FeatureCollection'
        assert len(catch_fc['features']) == _EXPECTED_BASINS_COUNT

        shapes_dir = self.pm.get_catchment_shapes_dir('hydro_agency')
        expected_ids = ['basin_alpha', 'basin_beta', '1120034560']
        for bid in expected_ids:
            sfile = shapes_dir / f'{bid}.geojson'
            assert sfile.exists(), f'Missing per-basin shape file: {sfile}'
            sdata = json.loads(sfile.read_text(encoding='utf-8'))
            assert sdata['type'] == 'Feature'
            assert sdata['id'] == bid
            assert sdata['properties']['basin_id'] == bid

        lists_dir = self.pm.get_basin_lists_dir('hydro_agency')
        for full_list_file in ('all_basins.txt', 'basins.txt'):
            fpath = lists_dir / full_list_file
            assert fpath.exists()
            lines = [
                line.strip()
                for line in fpath.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            assert lines == expected_ids

        split_sets: dict[str, set[str]] = {}
        for split_file in (
            'train_basins.txt',
            'val_basins.txt',
            'test_basins.txt',
        ):
            fpath = lists_dir / split_file
            assert fpath.exists()
            lines = [
                line.strip()
                for line in fpath.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            assert len(lines) == 1
            split_sets[split_file] = set(lines)

        assert split_sets['train_basins.txt'].isdisjoint(
            split_sets['val_basins.txt']
        )
        assert split_sets['train_basins.txt'].isdisjoint(
            split_sets['test_basins.txt']
        )
        assert split_sets['val_basins.txt'].isdisjoint(
            split_sets['test_basins.txt']
        )
        assert (
            split_sets['train_basins.txt']
            | split_sets['val_basins.txt']
            | split_sets['test_basins.txt']
        ) == set(expected_ids)

        # Stale per-basin shape file is removed when watershed list is updated.
        self.pm.save_watersheds(features[:1], username='hydro_agency')
        assert (shapes_dir / 'basin_alpha.geojson').exists()
        assert not (shapes_dir / 'basin_beta.geojson').exists()
        assert not (shapes_dir / '1120034560.geojson').exists()
        assert (lists_dir / 'all_basins.txt').read_text(
            encoding='utf-8'
        ).strip().splitlines() == ['basin_alpha']

    def test_guest_cleanup_wipes_13_category_artifacts(self) -> None:
        """Verify clear_guest_data removes guest artifacts across folders."""
        self.pm.save_watersheds(
            [
                {
                    'type': 'Feature',
                    'id': 'guest_basin_99',
                    'properties': {'area_km2': 100.0},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
                        ],
                    },
                }
            ],
            username='guest',
        )
        dummy_target = (
            self.pm.get_targets_dir('guest')
            / 'uploads'
            / 'guest_basin_99_historical.csv'
        )
        dummy_target.write_text(
            'date,discharge_cms\n2020-01-01,10.0\n', encoding='utf-8'
        )
        dummy_job = self.pm.get_jobs_dir('guest') / 'job_123.json'
        dummy_job.write_text('{}', encoding='utf-8')

        assert dummy_target.exists()
        assert dummy_job.exists()
        assert (
            self.pm.get_catchment_shapes_dir('guest') / 'guest_basin_99.geojson'
        ).exists()

        self.pm.clear_guest_data()
        assert not dummy_target.exists()
        assert not dummy_job.exists()
        assert not (
            self.pm.get_catchment_shapes_dir('guest') / 'guest_basin_99.geojson'
        ).exists()
        assert (
            self.pm.get_basin_lists_dir('guest') / 'all_basins.txt'
        ).read_text(encoding='utf-8') == ''
        assert self.pm.load_watersheds('guest') == []

    def test_get_account_summary_inventory(self) -> None:
        """Verify account summary aggregates polygons, attributes, and Zarrs."""
        self.pm.login_profile('acct_test_user')
        features = [
            {
                'type': 'Feature',
                'id': 'basin_001',
                'properties': {
                    'id': 'basin_001',
                    'name': 'Test Basin 1',
                    'area_km2': 450.25,
                    'source': 'delineated',
                    'dataset': 'merit_hydro',
                    'outlet_lat': 40.12,
                    'outlet_lon': -86.45,
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            [-86.5, 40.0],
                            [-86.0, 40.0],
                            [-86.0, 40.5],
                            [-86.5, 40.0],
                        ]
                    ],
                },
            },
            {
                'type': 'Feature',
                'id': 'basin_002',
                'properties': {
                    'id': 'basin_002',
                    'name': 'Test Basin 2',
                    'area_km2': 820.75,
                    'source': 'uploaded',
                    'dataset': 'custom_upload',
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            [-85.5, 40.0],
                            [-85.0, 40.0],
                            [-85.0, 40.5],
                            [-85.5, 40.0],
                        ]
                    ],
                },
            },
        ]
        self.pm.save_watersheds(features, username='acct_test_user')

        attr_csv = (
            self.pm.get_attributes_dir('acct_test_user')
            / 'caravan_hydroatlas_attributes.csv'
        )
        df_attr = pd.DataFrame(
            {
                'ele_mt_sav': [210.5, 340.0],
                'slp_dg_sav': [1.8, 3.2],
                'pre_mm_syr': [980.0, 1120.0],
                'area_km2': [450.25, 820.75],
            },
            index=pd.Index(['basin_001', 'basin_002'], name='basin_id'),
        )
        df_attr.to_csv(attr_csv)

        hist_zarr = (
            self.pm.get_historical_dir('acct_test_user')
            / 'historical_training_master.zarr'
        )
        dates = pd.date_range(
            '2022-01-01', periods=_EXPECTED_TIMESTEPS, freq='D'
        )
        ds_hist = xr.Dataset(
            {
                'total_precipitation_sum': (
                    ('basin', 'date'),
                    np.ones(
                        (_EXPECTED_INVENTORY_POLYGONS, _EXPECTED_TIMESTEPS),
                        dtype=np.float32,
                    ),
                ),
                'temperature_2m_mean': (
                    ('basin', 'date'),
                    np.full(
                        (_EXPECTED_INVENTORY_POLYGONS, _EXPECTED_TIMESTEPS),
                        15.0,
                        dtype=np.float32,
                    ),
                ),
                'area': (
                    ('basin',),
                    np.array([450.25, 820.75], dtype=np.float32),
                ),
            },
            coords={'basin': ['basin_001', 'basin_002'], 'date': dates},
            attrs={
                'source_dataset': 'ERA5_LAND',
                'temporal_resolution': '1D',
            },
        )
        ds_hist.to_zarr(hist_zarr, mode='w')

        rp_path = self.pm.get_return_periods_path('acct_test_user')
        rp_path.write_text(
            json.dumps(
                {
                    'basin_001': {
                        'return_periods_cms': {'Q2': 50.0, 'Q10': 120.0}
                    }
                }
            ),
            encoding='utf-8',
        )

        summary = self.pm.get_account_summary('acct_test_user')
        assert summary['username'] == 'acct_test_user'
        assert summary['is_guest'] is False

        totals = summary['totals']
        assert totals['polygon_count'] == _EXPECTED_INVENTORY_POLYGONS
        assert abs(totals['total_area_km2'] - 1271.0) < _AREA_TOL
        assert totals['attribute_basin_count'] == _EXPECTED_INVENTORY_POLYGONS
        assert totals['attribute_variable_count'] == _EXPECTED_INVENTORY_VARS
        assert totals['historical_store_count'] == 1

        coverage = {row['basin_id']: row for row in summary['basin_coverage']}
        assert 'basin_001' in coverage
        assert 'basin_002' in coverage
        assert coverage['basin_001']['has_polygon'] is True
        assert coverage['basin_001']['has_attributes'] is True
        assert coverage['basin_001']['has_historical'] is True
        assert coverage['basin_001']['has_return_periods'] is True
        assert coverage['basin_002']['has_return_periods'] is False

        hist_stores = summary['historical_training']['stores']
        assert len(hist_stores) == 1
        assert hist_stores[0]['source_dataset'] == 'ERA5_LAND'
        assert hist_stores[0]['n_timesteps'] == _EXPECTED_TIMESTEPS
        assert hist_stores[0]['start_date'] == '2022-01-01'
        assert hist_stores[0]['end_date'] == '2022-01-10'
        assert 'total_precipitation_sum' in hist_stores[0]['dynamic_variables']

    def test_account_tab_dom_elements_in_index_html(self) -> None:
        """Verify required Account tab DOM elements exist in index.html."""
        index_path = config.STATIC_DIR / 'index.html'
        html = index_path.read_text(encoding='utf-8')
        required_snippets = [
            'data-tab="tab-account"',
            'id="accountTabBtn"',
            'id="navAccountBadge"',
            'id="tab-account"',
            'id="accountTabProfileSelect"',
            'id="accountBasinCoverageTableBody"',
            'id="accountPolygonsContainer"',
            'id="accountAttributesContainer"',
            'id="accountHistoricalContainer"',
            'id="accountRealtimeAndModelsContainer"',
            'async function loadAccountSummary(',
            'function renderAccountSummaryUI(',
        ]
        for snippet in required_snippets:
            assert snippet in html, (
                f'Missing Account Tab element in index.html: {snippet}'
            )


if __name__ == '__main__':
    unittest.main()
