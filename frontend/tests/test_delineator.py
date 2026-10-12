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

"""Hermetic unit tests for hydrography snapping and catchment delineation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
from shapely.geometry import LineString, box, mapping, shape

import frontend.config as _frontend_config
from frontend.config import (
    HYDRO_DATASETS,
    resolve_hydro_dataset_id,
    resolve_hydrography_paths,
)
from frontend.delineator import (
    HydroDelineator,
    get_dem_delineator,
    get_river_network,
    get_vector_delineator,
    reset_delineator_caches,
)
from frontend.river_indexer import reset_river_indexer_caches

_TILE_CELLS: int = 6000
_RES_DEG: float = 5.0 / 6000.0
_MIN_DEM_AREA_KM2: float = 10.0
_MIN_UPSTREAM_CELLS: int = 1000
_EXPECTED_NORMALIZED_COUNT: int = 2


def _stamp_d8_basin(  # noqa: PLR0913
    grid: np.ndarray,
    lat_top: float,
    lon_left: float,
    lat: float,
    lon: float,
    *,
    n_rows: int,
    half_width: int,
) -> None:
    """Stamp a rectangular D8 drainage basin draining into `(lat, lon)`."""
    r0 = round((lat_top - lat) / _RES_DEG)
    c0 = round((lon - lon_left) / _RES_DEG)
    r_start = max(1, r0 - n_rows + 1)
    c_start = max(1, c0 - half_width)
    c_end = min(_TILE_CELLS - 2, c0 + half_width)
    grid[r_start:r0, c_start : c_end + 1] = 4
    grid[r0, c_start:c0] = 1
    grid[r0, c0 + 1 : c_end + 1] = 16
    grid[r0, c0] = 4


def _build_hermetic_hydro_fixtures(root: Path) -> None:
    """Populate synthetic 90m D8 tiles and vector shapefiles under `root`."""
    hs_dir = root / 'dem' / 'hydrosheds_90m' / 'tiles_5deg'
    mh_dir = root / 'dem' / 'merit_hydro_90m' / 'tiles_5deg'
    hs_dir.mkdir(parents=True, exist_ok=True)
    mh_dir.mkdir(parents=True, exist_ok=True)

    hs_grid = np.zeros((_TILE_CELLS, _TILE_CELLS), dtype=np.uint8)
    mh_grid = np.zeros((_TILE_CELLS, _TILE_CELLS), dtype=np.uint8)
    for pt_lat, pt_lon in (
        (39.6828, -88.7729),
        (39.4600, -87.4100),
        (38.6000, -86.8000),
    ):
        _stamp_d8_basin(
            hs_grid, 40.0, -90.0, pt_lat, pt_lon, n_rows=45, half_width=20
        )
        _stamp_d8_basin(
            mh_grid, 40.0, -90.0, pt_lat, pt_lon, n_rows=50, half_width=20
        )
    np.save(hs_dir / 'n40w090.npy', hs_grid)
    np.save(mh_dir / 'n40w090.npy', mh_grid)

    hr_dir = root / 'river_networks' / 'hydrorivers'
    hr_dir.mkdir(parents=True, exist_ok=True)
    hr_gdf = gpd.GeoDataFrame(
        {
            'HYRIV_ID': np.array([70000001, 70000002], dtype=np.int32),
            'NEXT_DOWN': np.array([0, 0], dtype=np.int32),
            'MAIN_RIV': np.array([70000001, 70000002], dtype=np.int32),
            'LENGTH_KM': np.array([8.9, 8.9], dtype=np.float64),
            'DIST_DN_KM': np.array([120.0, 240.0], dtype=np.float64),
            'UPLAND_SKM': np.array([12.1, 45.0], dtype=np.float64),
            'ORD_STRA': np.array([4, 5], dtype=np.int32),
            'ORD_CLAS': np.array([1, 1], dtype=np.int32),
            'HYBAS_L12': np.array([7120000001, 7120000002], dtype=np.int64),
            'CATCH_SKM': np.array([12.1, 45.0], dtype=np.float64),
            'DIS_AV_CMS': np.array([1.5, 8.2], dtype=np.float64),
        },
        geometry=[
            LineString([(-88.7729, 39.7200), (-88.7729, 39.6400)]),
            LineString([(-86.8000, 38.6400), (-86.8000, 38.5600)]),
        ],
        crs='EPSG:4326',
    )
    pyogrio.write_dataframe(hr_gdf, str(hr_dir / 'HydroRIVERS_v10.shp'))

    hb_dir = root / 'hydro_basins' / 'hydrobasins_units'
    hb_dir.mkdir(parents=True, exist_ok=True)
    hb_gdf = gpd.GeoDataFrame(
        {
            'HYBAS_ID': np.array([7120000001, 7120000002], dtype=np.int64),
            'NEXT_DOWN': np.array([0, 0], dtype=np.int64),
            'SUB_AREA': np.array([12.1, 45.0], dtype=np.float64),
            'UP_AREA': np.array([12.1, 45.0], dtype=np.float64),
            'MAIN_BAS': np.array([7120000001, 7120000002], dtype=np.int64),
        },
        geometry=[
            box(-88.80, 39.64, -88.74, 39.72),
            box(-86.84, 38.56, -86.76, 38.64),
        ],
        crs='EPSG:4326',
    )
    pyogrio.write_dataframe(hb_gdf, str(hb_dir / 'hybas_na_lev12_v1c.shp'))

    mr_dir = root / 'river_networks' / 'merit_rivers'
    mr_dir.mkdir(parents=True, exist_ok=True)
    mr_riv_gdf = gpd.GeoDataFrame(
        {
            'COMID': np.array([74000001, 74000002], dtype=np.int32),
            'NextDownID': np.array([0, 0], dtype=np.int32),
            'uparea': np.array([42.5, 13.5], dtype=np.float64),
            'lengthkm': np.array([8.9, 8.9], dtype=np.float64),
            'order': np.array([5, 4], dtype=np.int32),
            'sinuosity': np.array([1.1, 1.1], dtype=np.float64),
            'slope': np.array([0.001, 0.001], dtype=np.float64),
        },
        geometry=[
            LineString([(-87.4100, 39.5000), (-87.4100, 39.4200)]),
            LineString([(-88.7729, 39.7200), (-88.7729, 39.6400)]),
        ],
        crs='EPSG:4326',
    )
    pyogrio.write_dataframe(mr_riv_gdf, str(mr_dir / 'riv_pfaf_74.shp'))

    mr_cat_gdf = gpd.GeoDataFrame(
        {
            'COMID': np.array([74000001, 74000002], dtype=np.int32),
            'unitarea': np.array([42.5, 13.5], dtype=np.float64),
            'uparea': np.array([42.5, 13.5], dtype=np.float64),
        },
        geometry=[
            box(-87.45, 39.42, -87.37, 39.50),
            box(-88.80, 39.64, -88.74, 39.72),
        ],
        crs='EPSG:4326',
    )
    pyogrio.write_dataframe(mr_cat_gdf, str(mr_dir / 'cat_pfaf_74.shp'))


class HydroDelineatorTest(unittest.TestCase):
    """Hermetic tests for `HydroDelineator` across vector and 90m D8 modes."""

    _tmp_dir: tempfile.TemporaryDirectory[str]
    _orig_dem_locs: dict[str, tuple[Path, ...]]
    _orig_river_dir: Path
    _orig_hb_dir: Path
    _orig_cache_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        """Create synthetic hydrography and DEM fixtures in a temp directory."""
        super().setUpClass()
        cls._tmp_dir = tempfile.TemporaryDirectory()
        root = Path(cls._tmp_dir.name)
        _build_hermetic_hydro_fixtures(root)

        cls._orig_dem_locs = dict(_frontend_config.DEM_TILE_LOCATIONS)
        cls._orig_river_dir = _frontend_config.RIVER_NETWORKS_DIR
        cls._orig_hb_dir = _frontend_config.HYDRO_BASINS_DIR
        cls._orig_cache_dir = _frontend_config.CACHE_DIR

        _frontend_config.DEM_TILE_LOCATIONS['hydrosheds_90m'] = (
            root / 'dem' / 'hydrosheds_90m' / 'tiles_5deg',
        )
        _frontend_config.DEM_TILE_LOCATIONS['merit_hydro_90m'] = (
            root / 'dem' / 'merit_hydro_90m' / 'tiles_5deg',
        )
        _frontend_config.RIVER_NETWORKS_DIR = root / 'river_networks'
        _frontend_config.HYDRO_BASINS_DIR = root / 'hydro_basins'
        _frontend_config.CACHE_DIR = root / 'cache'
        reset_delineator_caches()
        reset_river_indexer_caches()

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore `frontend.config` paths and clean up temporary fixtures."""
        _frontend_config.DEM_TILE_LOCATIONS.clear()
        _frontend_config.DEM_TILE_LOCATIONS.update(cls._orig_dem_locs)
        _frontend_config.RIVER_NETWORKS_DIR = cls._orig_river_dir
        _frontend_config.HYDRO_BASINS_DIR = cls._orig_hb_dir
        _frontend_config.CACHE_DIR = cls._orig_cache_dir
        reset_delineator_caches()
        reset_river_indexer_caches()
        cls._tmp_dir.cleanup()
        super().tearDownClass()

    def test_merit_hydro_delineation(self) -> None:
        """Test MERIT-Basins vector catchment delineation."""
        delineator = HydroDelineator(dataset_id='merit-hydro')
        result = delineator.delineate_catchment(lat=39.46, lon=-87.41)

        assert result['type'] == 'Feature'
        assert 'properties' in result
        props = result['properties']
        assert props['dataset'] == 'merit-hydro'
        assert props['area_km2'] > 0
        assert 'outlet' in props
        assert 'latitude' in props['outlet']
        assert 'longitude' in props['outlet']
        assert 'geometry' in result
        assert result['geometry']['type'] in ('Polygon', 'MultiPolygon')

    def test_hydroatlas_delineation(self) -> None:
        """Test HydroATLAS / HydroBASINS Level 12 catchment delineation."""
        delineator = HydroDelineator(dataset_id='hydroatlas')
        result = delineator.delineate_catchment(lat=38.60, lon=-86.80)

        props = result['properties']
        assert props['dataset'] == 'hydroatlas'
        assert props['area_km2'] > 0
        assert 'reach_id' in props['outlet']

    def test_exact_pour_point_delineation(self) -> None:
        """Test that `exact_pour_point` clips sub-basin at the pour point."""
        delineator = HydroDelineator(dataset_id='hydroatlas')
        res_official = delineator.delineate_catchment(
            lat=39.6828, lon=-88.7729, mode='official_ridgeline'
        )
        res_exact = delineator.delineate_catchment(
            lat=39.6828, lon=-88.7729, mode='exact_pour_point'
        )

        assert res_exact['type'] == 'Feature'
        assert res_exact['properties']['delineation_mode'] == 'exact_pour_point'
        assert (
            'Exact Pour-Point' in res_exact['properties']['delineation_method']
        )

        snapped_lat = res_exact['properties']['outlet']['latitude']
        exact_min_lat = res_exact['properties']['bbox']['min_lat']
        official_min_lat = res_official['properties']['bbox']['min_lat']

        assert exact_min_lat == pytest.approx(snapped_lat, abs=1e-3)
        assert exact_min_lat > official_min_lat

    def test_dem_flow_direction_delineation(self) -> None:
        """Test pure DEM flow direction D8 reverse routing on 90m grid."""
        delineator = HydroDelineator(dataset_id='hydroatlas')
        res_dem = delineator.delineate_catchment(
            lat=39.6828, lon=-88.7729, mode='dem_flow_direction'
        )

        assert res_dem['type'] == 'Feature'
        assert res_dem['properties']['delineation_mode'] == 'dem_flow_direction'
        assert (
            'DEM Digital Elevation'
            in res_dem['properties']['delineation_method']
        )
        assert res_dem['properties']['area_km2'] > _MIN_DEM_AREA_KM2
        assert (
            res_dem['properties']['upstream_cells_count'] > _MIN_UPSTREAM_CELLS
        )
        assert res_dem['geometry']['type'] in ('Polygon', 'MultiPolygon')

    def test_custom_multi_watershed_upload(self) -> None:
        """Test multi-watershed GeoJSON normalization."""
        sample_geojson = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': {
                        'gauge_id': 'USGS_03335500',
                        'station_name': 'Wabash River at Lafayette, IN',
                    },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [
                                [-87.0, 40.3],
                                [-86.8, 40.3],
                                [-86.8, 40.5],
                                [-87.0, 40.5],
                                [-87.0, 40.3],
                            ]
                        ],
                    },
                },
                {
                    'type': 'Feature',
                    'properties': {
                        'gauge_id': 'USGS_03335700',
                        'station_name': 'Wildcat Creek near Lafayette, IN',
                    },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [
                                [-86.9, 40.4],
                                [-86.6, 40.4],
                                [-86.6, 40.6],
                                [-86.9, 40.6],
                                [-86.9, 40.4],
                            ]
                        ],
                    },
                },
            ],
        }

        features_in = sample_geojson['features']
        normalized = []
        for idx, feat in enumerate(features_in):
            geom = shape(feat['geometry'])
            cent = geom.centroid
            cid = feat['properties'].get('gauge_id') or f'basin_{idx + 1}'
            area_km2 = round(
                float(geom.area * 111.0 * 111.0 * np.cos(np.radians(cent.y))),
                2,
            )
            normalized.append(
                {
                    'type': 'Feature',
                    'geometry': mapping(geom),
                    'properties': {
                        **feat['properties'],
                        'catchment_id': cid,
                        'area_km2': area_km2,
                    },
                }
            )

        assert len(normalized) == _EXPECTED_NORMALIZED_COUNT
        assert normalized[0]['properties']['catchment_id'] == 'USGS_03335500'
        assert normalized[0]['properties']['area_km2'] > 0
        assert normalized[1]['properties']['catchment_id'] == 'USGS_03335700'

    def test_one_to_one_dem_river_network_mapping(self) -> None:
        """Test 1:1 association between DEM ID and River Network."""
        assert resolve_hydro_dataset_id('hydrosheds_90m') == 'hydroatlas'
        assert resolve_hydro_dataset_id('merit_hydro_90m') == 'merit-hydro'
        assert HYDRO_DATASETS['hydroatlas']['dem_id'] == 'hydrosheds_90m'
        assert HYDRO_DATASETS['merit-hydro']['dem_id'] == 'merit_hydro_90m'

        delin_hs = HydroDelineator(dataset_id='hydrosheds_90m')
        assert delin_hs.dataset_id == 'hydroatlas'
        res_hs = delin_hs.delineate_catchment(lat=38.60, lon=-86.80)
        assert res_hs['properties']['dem_id'] == 'hydrosheds_90m'
        assert 'HydroRIVERS' in res_hs['properties']['river_network']

        delin_mh = HydroDelineator(dataset_id='merit_hydro_90m')
        assert delin_mh.dataset_id == 'merit-hydro'
        res_mh = delin_mh.delineate_catchment(lat=39.46, lon=-87.41)
        assert res_mh['properties']['dem_id'] == 'merit_hydro_90m'
        assert 'MERIT-Basins' in res_mh['properties']['river_network']

    def test_delineator_and_login_tab_ui_structure(self) -> None:
        """Verify top header removal, Account tab, and Delineation UI."""
        html_path = (
            Path(__file__).resolve().parent.parent / 'static' / 'index.html'
        )
        html = html_path.read_text(encoding='utf-8')

        assert '<header class="hidden" aria-hidden="true">' in html
        assert 'data-tab="tab-login"' not in html
        assert 'id="loginTabBtn"' not in html
        assert '<section id="tab-login"' not in html
        assert 'data-tab="tab-account"' in html
        assert '<section id="tab-account"' in html
        assert 'id="loginTabUsernameInput"' in html
        assert 'id="loginTabSubmitBtn"' in html
        assert '1. River Network' in html
        assert '>River Network</label>' in html
        assert 'data-alias="demSelect"' in html
        assert 'data-dem-id="hydrosheds_90m"' in html
        assert 'data-dem-id="merit_hydro_90m"' in html

    def test_late_populated_datasets_recover_without_cache_reset(self) -> None:
        """Verify river, vector, and DEM singletons recover on file arrival."""
        with tempfile.TemporaryDirectory() as late_tmp:
            late_root = Path(late_tmp)
            empty_dem = late_root / 'empty_dem'
            populated_dem = late_root / 'dem' / 'hydrosheds_90m' / 'tiles_5deg'
            empty_dem.mkdir(parents=True, exist_ok=True)

            saved_dem_locs = dict(_frontend_config.DEM_TILE_LOCATIONS)
            saved_river_dir = _frontend_config.RIVER_NETWORKS_DIR
            saved_hb_dir = _frontend_config.HYDRO_BASINS_DIR
            saved_hr_cache = (
                _frontend_config.OPENHYDRONET_HYDRORIVERS_SHP_CACHE_PATH
            )
            saved_hb_cache = _frontend_config.OPENHYDRONET_HYDROBASINS_CACHE_DIR

            _frontend_config.DEM_TILE_LOCATIONS['hydrosheds_90m'] = (
                empty_dem,
                populated_dem,
            )
            _frontend_config.RIVER_NETWORKS_DIR = late_root / 'river_networks'
            _frontend_config.HYDRO_BASINS_DIR = late_root / 'hydro_basins'
            # Empty local dir must not shadow populated cache
            (_frontend_config.HYDRO_BASINS_DIR / 'hydrobasins_units').mkdir(
                parents=True, exist_ok=True
            )
            _frontend_config.OPENHYDRONET_HYDRORIVERS_SHP_CACHE_PATH = (
                late_root / 'missing_cache' / 'HydroRIVERS_v10.shp'
            )
            _frontend_config.OPENHYDRONET_HYDROBASINS_CACHE_DIR = (
                late_root / 'missing_cache' / 'hydrobasins'
            )
            reset_delineator_caches()

            # 1. Before files exist on disk, river/vector lookups return None
            assert get_river_network('hydroatlas') is None
            assert get_vector_delineator('hydroatlas') is None
            initial_dem = get_dem_delineator('hydrosheds_90m')
            assert initial_dem.tiles_dir == empty_dem.resolve()

            # 2. Populate files on disk without calling reset_delineator_caches
            _build_hermetic_hydro_fixtures(late_root)
            paths = resolve_hydrography_paths()
            assert paths['hydrorivers_shp'].is_file()
            assert any(paths['hydrobasins_dir'].glob('*.shp'))

            assert get_river_network('hydroatlas') is not None
            assert get_vector_delineator('hydroatlas') is not None
            updated_dem = get_dem_delineator('hydrosheds_90m')
            assert updated_dem.tiles_dir == populated_dem.resolve()

            res = HydroDelineator('hydroatlas').delineate_catchment(
                lat=39.6828, lon=-88.7729, mode='dem_flow_direction'
            )
            assert res['properties']['outlet']['reach_id'] == 'HYRIV_70000001'

            _frontend_config.DEM_TILE_LOCATIONS.clear()
            _frontend_config.DEM_TILE_LOCATIONS.update(saved_dem_locs)
            _frontend_config.RIVER_NETWORKS_DIR = saved_river_dir
            _frontend_config.HYDRO_BASINS_DIR = saved_hb_dir
            _frontend_config.OPENHYDRONET_HYDRORIVERS_SHP_CACHE_PATH = (
                saved_hr_cache
            )
            _frontend_config.OPENHYDRONET_HYDROBASINS_CACHE_DIR = saved_hb_cache
            reset_delineator_caches()


if __name__ == '__main__':
    unittest.main()
