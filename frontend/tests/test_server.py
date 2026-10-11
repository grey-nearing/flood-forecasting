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
"""Hermetic integration tests for OpenHydroNet HTTP server endpoints."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, box

import frontend.delineator as delineator_mod
import frontend.forecast_zarr as forecast_zarr_mod
import frontend.river_indexer as river_indexer_mod
import frontend.server as server_mod
import frontend.static_attributes as static_attr_mod
from frontend import tests as _isolated_profiles  # noqa: F401
from frontend.dem_tile_server import DemTileServer
from frontend.server import OpenHydroNetHandler
from frontend.weather_sources import WEATHER_SOURCES
from multimet.catchment_delineation import (
    RES_DEG,
    TILE_CELLS,
    DemDelineator,
    RiverNetwork,
)
from multimet.static_extractor import StaticAttributesExtractor

_HTTP_OK = 200
_MIN_DATASETS = 2
_MIN_ELEVATION_M = 100.0
_MIN_PRECIP_MM = 500.0
_MIN_ARIDITY = 0.1
_EXPECTED_JANUARY_DAYS = 31
_EXPECTED_BATCH_COUNT = 4
_EXPECTED_REMAINING_AFTER_DELETE = 3
_MIN_PNG_BYTES = 1000
_SOUTH_D8 = 4
_WEST_D8 = 16
_EAST_D8 = 1


def _populate_d8_basin(
    tile_arr: np.ndarray,
    center_row: int,
    center_col: int,
    half_width: int = 8,
) -> None:
    """Create a synthetic rectangular D8 drainage basin around (row, col)."""
    for r in range(center_row - half_width, center_row):
        for c in range(center_col - half_width, center_col + half_width + 1):
            tile_arr[r, c] = _SOUTH_D8
    for c in range(center_col - half_width, center_col):
        tile_arr[center_row, c] = _EAST_D8
    for c in range(center_col + 1, center_col + half_width + 1):
        tile_arr[center_row, c] = _WEST_D8


def _build_hermetic_fixtures(root: Path) -> dict[str, Any]:
    """Create synthetic DEM, river, BasinATLAS, and Zarr fixtures in root."""
    # 1. Synthetic D8 flow-direction tiles (n45w090.npy and n40w090.npy)
    d8_dir = root / 'd8_tiles_5deg'
    d8_dir.mkdir(parents=True, exist_ok=True)

    tile_n45 = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    row_4075 = round((45.0 - 40.75) / RES_DEG)
    col_8607 = round((-86.07 - (-90.0)) / RES_DEG)
    _populate_d8_basin(tile_n45, row_4075, col_8607, half_width=10)
    np.save(d8_dir / 'n45w090.npy', tile_n45)

    tile_n40 = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    row_3995 = round((40.0 - 39.95) / RES_DEG)
    col_8625 = round((-86.25 - (-90.0)) / RES_DEG)
    _populate_d8_basin(tile_n40, row_3995, col_8625, half_width=10)
    np.save(d8_dir / 'n40w090.npy', tile_n40)

    # 2. Synthetic Elevation tiles for DemTileServer (8/65/97.png)
    elv_dir = root / 'elevation_tiles_5deg'
    elv_dir.mkdir(parents=True, exist_ok=True)
    row_grad = np.linspace(150, 450, TILE_CELLS, dtype=np.int16)[:, None]
    col_grad = np.linspace(0, 120, TILE_CELLS, dtype=np.int16)[None, :]
    elv_tile = (row_grad + col_grad).astype(np.int16)
    np.save(elv_dir / 'n45w090.npy', elv_tile)
    np.save(elv_dir / 'n40w090.npy', elv_tile)

    # 3. Synthetic RiverNetwork shapefile
    rivers_dir = root / 'rivers'
    rivers_dir.mkdir(parents=True, exist_ok=True)
    rivers_shp = rivers_dir / 'HydroRIVERS_v10_na.shp'
    rivers_gdf = gpd.GeoDataFrame(
        {
            'HYRIV_ID': [70001, 70002],
            'NEXT_DOWN': [70002, 0],
            'MAIN_RIV': [70002, 70002],
            'LENGTH_KM': [4.5, 6.0],
            'DIST_DN_KM': [6.0, 0.0],
            'DIST_UP_KM': [4.5, 10.5],
            'CATCH_SKM': [2.5, 3.0],
            'UPLAND_SKM': [2.5, 5.5],
            'DIS_AV_CMS': [1.5, 3.2],
            'ORD_STRA': [5, 6],
            'ORD_CLAS': [1, 1],
            'ORD_FLOW': [5, 4],
            'HYBAS_L12': [712000001, 712000002],
        },
        geometry=[
            LineString([(-86.08, 40.76), (-86.07, 40.75), (-86.06, 40.74)]),
            LineString([(-86.26, 39.96), (-86.25, 39.95), (-86.24, 39.94)]),
        ],
        crs='EPSG:4326',
    )
    rivers_gdf.to_file(rivers_shp)

    # 4. Synthetic BasinATLAS shapefile & ERA5 climate table
    poly1 = box(-89.0, 38.5, -86.2, 42.0)
    poly2 = box(-86.2, 38.5, -84.0, 42.0)
    hydroatlas_gdf = gpd.GeoDataFrame(
        {
            'HYBAS_ID': [712000001, 712000002],
            'NEXT_DOWN': [712000002, 0],
            'SUB_AREA': [5000.0, 5000.0],
            'UP_AREA': [5000.0, 10000.0],
            'ele_mt_sav': [220.0, 380.0],
            'slp_dg_sav': [15.0, 25.0],
            'pre_mm_syr': [950.0, 1100.0],
            'tmp_dc_syr': [110.0, 130.0],
            'ari_ix_sav': [80.0, 110.0],
            'cly_pc_sav': [24.0, 32.0],
            'snd_pc_sav': [42.0, 36.0],
            'slt_pc_sav': [34.0, 32.0],
            'for_pc_sse': [45.0, 30.0],
            'crp_pc_sse': [35.0, 45.0],
            'urb_pc_sse': [8.0, 12.0],
            'gwt_cm_sav': [150.0, 220.0],
            'swc_pc_syr': [55.0, 62.0],
            'inu_pc_smx': [4.0, 7.0],
            'glc_cl_smj': [4, 12],
            'clz_cl_smj': [7, 9],
            'lit_cl_smj': [1, 3],
            'dis_m3_pyr': [25.0, 65.0],
            'run_mm_syr': [350.0, 480.0],
        },
        geometry=[poly1, poly2],
        crs='EPSG:4326',
    )
    hydroatlas_shp = root / 'synthetic_hydroatlas.shp'
    hydroatlas_gdf.to_file(hydroatlas_shp)

    era5_cache = root / 'era5_cache'
    era5_cache.mkdir(parents=True, exist_ok=True)
    rec1 = {
        'gauge_id': 'hybas_712000001',
        'p_mean': 3.2,
        'pet_mean': 2.1,
        'aridity': 0.65,
        'frac_snow': 0.12,
        'moisture_index': 0.42,
        'seasonality': 0.25,
        'high_prec_freq': 0.05,
        'high_prec_dur': 1.3,
        'low_prec_freq': 0.55,
        'low_prec_dur': 3.2,
    }
    rec2 = {
        'gauge_id': 'hybas_712000002',
        'p_mean': 4.1,
        'pet_mean': 2.4,
        'aridity': 0.58,
        'frac_snow': 0.15,
        'moisture_index': 0.51,
        'seasonality': 0.30,
        'high_prec_freq': 0.07,
        'high_prec_dur': 1.5,
        'low_prec_freq': 0.48,
        'low_prec_dur': 2.9,
    }
    (era5_cache / 'na_climate_indices.txt').write_text(
        json.dumps(rec1) + '\n' + json.dumps(rec2) + '\n',
        encoding='utf-8',
    )

    # 5. Synthetic Historical ERA5 Gridded Zarr Store (2021-01-01 to 2021-01-31)
    era5_zarr = root / 'era5_historical.zarr'
    times = pd.date_range('2021-01-01', '2021-01-31', freq='1D')
    lats = np.linspace(38.5, 42.0, 15, dtype=np.float64)
    lons = np.linspace(-89.0, -84.0, 21, dtype=np.float64)
    shape_3d = (len(times), len(lats), len(lons))
    ds_era5 = xr.Dataset(
        data_vars={
            'era5land_total_precipitation': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 4.2, dtype=np.float32),
            ),
            'era5land_temperature_2m': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 14.5, dtype=np.float32),
            ),
            'era5land_surface_pressure': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 101.3, dtype=np.float32),
            ),
            'era5land_surface_net_solar_radiation': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 180.0, dtype=np.float32),
            ),
            'era5land_surface_net_thermal_radiation': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, -55.0, dtype=np.float32),
            ),
            'era5land_dewpoint_temperature_2m': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 8.5, dtype=np.float32),
            ),
            'era5land_potential_evaporation_FAO_PENMAN_MONTEITH': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 2.2, dtype=np.float32),
            ),
            'potential_evaporation': (
                ['time', 'latitude', 'longitude'],
                np.full(shape_3d, 2.5, dtype=np.float32),
            ),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        },
    )
    ds_era5.to_zarr(str(era5_zarr), mode='w', consolidated=True)

    # 6. Synthetic HRES and GraphCast Forecast Datasets
    init_times = pd.DatetimeIndex(['2026-05-10T00:00:00'])
    lead_days = np.arange(1, 16, dtype=np.int64)
    fc_lats = np.linspace(42.0, 38.5, 15, dtype=np.float64)
    fc_lons = np.linspace(-89.0, -84.0, 21, dtype=np.float64)
    shape_4d = (1, len(lead_days), len(fc_lats), len(fc_lons))

    ds_hres = xr.Dataset(
        data_vars={
            'hres_surface_net_solar_radiation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 205.0, dtype=np.float32),
                {'units': 'W/m^2'},
            ),
            'hres_surface_net_thermal_radiation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, -60.0, dtype=np.float32),
                {'units': 'W/m^2'},
            ),
            'pressure_surface': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 101325.0, dtype=np.float32),
                {'units': 'Pa'},
            ),
            'hres_temperature_2m': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 17.5, dtype=np.float32),
                {'units': 'degC'},
            ),
            'hres_total_precipitation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 5.2, dtype=np.float32),
                {'units': 'mm/day'},
            ),
        },
        coords={
            'init_time': init_times,
            'lead_time': ('lead_time', lead_days, {'units': 'days'}),
            'latitude': fc_lats,
            'longitude': fc_lons,
        },
    )

    ds_graphcast = xr.Dataset(
        data_vars={
            'graphcast_temperature_2m': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 16.8, dtype=np.float32),
                {'units': 'degC'},
            ),
            'graphcast_total_precipitation': (
                ['init_time', 'lead_time', 'latitude', 'longitude'],
                np.full(shape_4d, 4.8, dtype=np.float32),
                {'units': 'mm/day'},
            ),
        },
        coords={
            'init_time': init_times,
            'lead_time': ('lead_time', lead_days, {'units': 'days'}),
            'latitude': fc_lats,
            'longitude': fc_lons,
        },
    )

    return {
        'd8_dir': d8_dir,
        'elv_dir': elv_dir,
        'rivers_shp': rivers_shp,
        'hydroatlas_shp': hydroatlas_shp,
        'era5_cache': era5_cache,
        'era5_zarr': era5_zarr,
        'ds_hres': ds_hres,
        'ds_graphcast': ds_graphcast,
    }


@pytest.mark.unit
class ServerApiIntegrationTest(unittest.TestCase):
    """Test OpenHydroNet server endpoints against synthetic fixtures."""

    _temp_dir: tempfile.TemporaryDirectory[str]
    _stack: ExitStack
    _orig_era5_zarr_path: str | None
    server: ThreadingHTTPServer
    port: int
    thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        """Provision synthetic fixtures and start the local HTTP server."""
        super().setUpClass()
        cls._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls._stack = ExitStack()
        fixtures = _build_hermetic_fixtures(Path(cls._temp_dir.name))

        delineator_mod.reset_delineator_caches()
        river_indexer_mod.reset_river_indexer_caches()
        static_attr_mod.reset_attributes_extractor()
        server_mod._river_indexers.clear()  # noqa: SLF001
        server_mod._delineators.clear()  # noqa: SLF001

        dem_tile_srv = DemTileServer(tiles_dir=fixtures['elv_dir'])
        dem_hs = DemDelineator(
            tiles_dir=fixtures['d8_dir'],
            dataset='hydrosheds_90m',
        )
        dem_mh = DemDelineator(
            tiles_dir=fixtures['d8_dir'],
            dataset='merit_hydro_90m',
        )
        synth_net = RiverNetwork.from_hydrorivers(fixtures['rivers_shp'])
        synth_extractor = StaticAttributesExtractor(
            gdb_path=fixtures['hydroatlas_shp'],
            era5_cache_dir=fixtures['era5_cache'],
            gridded_era5_uri=str(fixtures['era5_zarr']),
        )

        cls._stack.enter_context(
            mock.patch.object(
                server_mod,
                'get_dem_tile_server',
                return_value=dem_tile_srv,
            )
        )
        cls._stack.enter_context(
            mock.patch.object(
                delineator_mod,
                'get_dem_delineator',
                side_effect=lambda dem_id='hydrosheds_90m': (
                    dem_mh if 'merit' in str(dem_id) else dem_hs
                ),
            )
        )
        cls._stack.enter_context(
            mock.patch.object(
                delineator_mod,
                'get_river_network',
                return_value=synth_net,
            )
        )
        cls._stack.enter_context(
            mock.patch.object(
                river_indexer_mod,
                'get_river_network',
                return_value=synth_net,
            )
        )
        cls._stack.enter_context(
            mock.patch.object(
                static_attr_mod,
                'get_attributes_extractor',
                return_value=synth_extractor,
            )
        )
        cls._stack.enter_context(
            mock.patch.object(
                server_mod,
                'get_attributes_extractor',
                return_value=synth_extractor,
            )
        )
        cls._stack.enter_context(
            mock.patch.object(
                forecast_zarr_mod,
                '_get_dynamical_datasets',
                return_value=(fixtures['ds_hres'], fixtures['ds_graphcast']),
            )
        )

        cls._orig_era5_zarr_path = WEATHER_SOURCES['era5'].zarr_path
        WEATHER_SOURCES['era5'].zarr_path = str(fixtures['era5_zarr'])

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), OpenHydroNetHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down the server and restore module caches."""
        import gc

        cls.server.shutdown()
        cls.server.server_close()
        WEATHER_SOURCES['era5'].zarr_path = cls._orig_era5_zarr_path
        cls._stack.close()
        static_attr_mod.reset_attributes_extractor()
        delineator_mod.reset_delineator_caches()
        river_indexer_mod.reset_river_indexer_caches()
        server_mod._river_indexers.clear()  # noqa: SLF001
        server_mod._delineators.clear()  # noqa: SLF001
        gc.collect()
        cls._temp_dir.cleanup()
        super().tearDownClass()

    def _get(self, path: str) -> tuple[int, Any]:
        url = f'http://127.0.0.1:{self.port}{path}'
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = resp.read().decode('utf-8')
            status = resp.status
            content_type = resp.headers.get('Content-Type') or ''
            if 'application/json' in content_type:
                return status, json.loads(data)
            return status, data

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        url = f'http://127.0.0.1:{self.port}{path}'
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = resp.read().decode('utf-8')
            status = resp.status
            return status, json.loads(data)

    def _get_bytes(self, path: str) -> tuple[int, str | None, bytes]:
        url = f'http://127.0.0.1:{self.port}{path}'
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = resp.read()
            status = resp.status
            content_type = resp.headers.get('Content-Type')
            return status, content_type, data

    def test_datasets_endpoint(self) -> None:
        """Verify GET /api/datasets returns hydroatlas and merit-hydro."""
        status, data = self._get('/api/datasets')
        assert status == _HTTP_OK
        assert 'datasets' in data
        assert len(data['datasets']) >= _MIN_DATASETS
        dataset_ids = [d['id'] for d in data['datasets']]
        assert 'merit-hydro' in dataset_ids
        assert 'hydroatlas' in dataset_ids

    def test_weather_sources_endpoint(self) -> None:
        """Verify GET /api/weather/sources returns canonical weather sources."""
        status, data = self._get('/api/weather/sources')
        assert status == _HTTP_OK
        assert 'sources' in data
        sources = {s['id']: s for s in data['sources']}
        assert 'era5' in sources
        assert 'cpc' in sources
        assert 'imerg' in sources
        assert 'ifs' in sources
        assert 'graphcast' in sources

        cpc = sources['cpc']
        assert cpc['start_date'] == '1979-01-01'
        assert cpc['end_date'] > '2024-01-01'
        assert cpc['total_years'] > _MIN_DATASETS

    def test_attributes_schema_and_extract_endpoints(self) -> None:
        """Verify attributes schema and polygon attribute extraction APIs."""
        status, schema_data = self._get('/api/attributes/schema')
        assert status == _HTTP_OK
        assert 'attributes' in schema_data
        assert 'ele_mt_sav' in schema_data['attributes']
        assert 'cly_pc_sav' in schema_data['attributes']
        assert 'for_pc_sse' in schema_data['attributes']
        assert 'p_mean' in schema_data['attributes']
        assert 'pet_mean_ERA5_LAND' in schema_data['attributes']
        assert 'aridity_ERA5_LAND' in schema_data['attributes']
        assert 'frac_snow' in schema_data['attributes']

        sample_poly = {
            'type': 'Polygon',
            'coordinates': [
                [
                    [-86.5, 40.2],
                    [-86.1, 40.2],
                    [-86.1, 40.6],
                    [-86.5, 40.6],
                    [-86.5, 40.2],
                ]
            ],
        }
        extract_status, extract_data = self._post(
            '/api/attributes/extract',
            {
                'catchment_id': 'test_basin_attribs_01',
                'polygon': sample_poly,
            },
        )
        assert extract_status == _HTTP_OK
        assert 'summary' in extract_data
        assert 'categories' in extract_data
        assert 'attributes' in extract_data
        assert extract_data['intersected_subbasins_count'] > 0
        assert extract_data['summary']['elevation_mean_m'] > _MIN_ELEVATION_M
        assert extract_data['summary']['annual_precip_mm'] > _MIN_PRECIP_MM
        assert extract_data['attributes']['p_mean'] > 1.0
        assert extract_data['attributes']['pet_mean'] > 1.0
        assert extract_data['summary']['era5_p_mean_mm_day'] > 1.0
        assert extract_data['summary']['era5_fao_aridity'] > _MIN_ARIDITY
        assert 'Topography' in extract_data['categories']
        assert 'Soils' in extract_data['categories']
        assert 'Climate' in extract_data['categories']

    def test_rivers_endpoint(self) -> None:
        """Verify GET /api/hydro/rivers returns reaches within the bbox."""
        status, data = self._get(
            '/api/hydro/rivers?dataset=merit-hydro&bbox=-88.5,37.5,-84.5,41.8&zoom=7'
        )
        assert status == _HTTP_OK
        assert data['type'] == 'FeatureCollection'
        assert 'features' in data
        assert len(data['features']) > 0
        first_reach = data['features'][0]
        assert 'properties' in first_reach
        assert 'stream_order' in first_reach['properties']
        assert 'upstream_area_km2' in first_reach['properties']

    def test_full_workflow_delineate_and_extract_weather(self) -> None:
        """Verify delineate, historical Zarr, forecast Zarr, and upload APIs."""
        delin_status, delin_data = self._post(
            '/api/delineate',
            {
                'dataset': 'hydroatlas',
                'latitude': 40.75,
                'longitude': -86.07,
            },
        )
        assert delin_status == _HTTP_OK
        assert delin_data['type'] == 'Feature'
        catchment_id = delin_data['properties']['catchment_id']
        assert catchment_id.startswith('catchment_hydroatlas_')

        hist_status, hist_data = self._post(
            '/api/weather/historical',
            {
                'catchment_id': catchment_id,
                'start_date': '2021-01-01',
                'end_date': '2021-01-31',
                'frequency': '1D',
            },
        )
        assert hist_status == _HTTP_OK
        assert hist_data['status'] == 'success'
        assert 'preview' in hist_data
        assert len(hist_data['preview']['timestamps']) == _EXPECTED_JANUARY_DAYS
        assert 'master_zarr_path' in hist_data

        fcst_status, fcst_data = self._post(
            '/api/weather/forecast',
            {
                'catchment_id': catchment_id,
                'model': 'ifs',
                'horizon_hours': 72,
            },
        )
        assert fcst_status == _HTTP_OK
        assert fcst_data['status'] == 'success'
        assert 'meteogram' in fcst_data

        custom_poly_geojson = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-88.0, 41.0],
                        [-87.5, 41.0],
                        [-87.5, 41.5],
                        [-88.0, 41.5],
                        [-88.0, 41.0],
                    ]
                ],
            },
            'properties': {
                'catchment_id': 'CUSTOM_USER_BASIN_TEST_99',
                'name': 'Custom Illinois Watershed',
            },
        }
        poly_status, poly_data = self._post(
            '/api/polygons/upload', custom_poly_geojson
        )
        assert poly_status == _HTTP_OK
        assert poly_data['type'] == 'FeatureCollection'
        assert (
            poly_data['features'][0]['properties']['catchment_id']
            == 'CUSTOM_USER_BASIN_TEST_99'
        )

        custom_hist_status, custom_hist_data = self._post(
            '/api/weather/historical',
            {
                'catchment_id': 'CUSTOM_USER_BASIN_TEST_99',
                'start_date': '2021-01-01',
                'end_date': '2021-01-31',
            },
        )
        assert custom_hist_status == _HTTP_OK
        assert custom_hist_data['status'] == 'success'
        assert custom_hist_data['total_basins_in_master'] >= _MIN_DATASETS

        arch_status, arch_data = self._get('/api/archives')
        assert arch_status == _HTTP_OK
        archives = arch_data['archives']
        assert len(archives) > 0

    def test_watershed_registry_management_and_batch_extraction(self) -> None:
        """Verify watershed registry CRUD and batch static extraction."""
        self._post('/api/watersheds/clear', {})

        s1, d1 = self._post(
            '/api/delineate',
            {'dataset': 'hydroatlas', 'latitude': 40.75, 'longitude': -86.07},
        )
        assert s1 == _HTTP_OK
        id1 = d1['properties']['catchment_id']

        s2, d2 = self._post(
            '/api/delineate',
            {'dataset': 'hydroatlas', 'latitude': 39.95, 'longitude': -86.25},
        )
        assert s2 == _HTTP_OK
        id2 = d2['properties']['catchment_id']

        fc = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [
                                [-87.5, 40.0],
                                [-87.0, 40.0],
                                [-87.0, 40.5],
                                [-87.5, 40.5],
                                [-87.5, 40.0],
                            ]
                        ],
                    },
                    'properties': {
                        'catchment_id': 'UPLOADED_BASIN_A',
                        'name': 'Uploaded Basin A',
                    },
                },
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [
                                [-86.0, 39.0],
                                [-85.5, 39.0],
                                [-85.5, 39.5],
                                [-86.0, 39.5],
                                [-86.0, 39.0],
                            ]
                        ],
                    },
                    'properties': {
                        'catchment_id': 'UPLOADED_BASIN_B',
                        'name': 'Uploaded Basin B',
                    },
                },
            ],
        }
        s3, _ = self._post('/api/polygons/upload', fc)
        assert s3 == _HTTP_OK

        ws_status, ws_data = self._get('/api/watersheds')
        assert ws_status == _HTTP_OK
        assert ws_data['count'] == _EXPECTED_BATCH_COUNT
        cids = [f['properties']['catchment_id'] for f in ws_data['features']]
        assert id1 in cids
        assert id2 in cids
        assert 'UPLOADED_BASIN_A' in cids
        assert 'UPLOADED_BASIN_B' in cids

        attr_status, attr_data = self._post(
            '/api/attributes/extract', {'all_active': True}
        )
        assert attr_status == _HTTP_OK
        assert attr_data['batch'] is True
        assert attr_data['count'] == _EXPECTED_BATCH_COUNT

        del_status, del_data = self._post(
            '/api/watersheds/delete', {'catchment_id': 'UPLOADED_BASIN_A'}
        )
        assert del_status == _HTTP_OK
        assert del_data['status'] == 'deleted'
        assert del_data['remaining_count'] == _EXPECTED_REMAINING_AFTER_DELETE

        clr_status, clr_data = self._post('/api/watersheds/clear', {})
        assert clr_status == _HTTP_OK
        assert clr_data['remaining_count'] == 0

    def test_dem_tiles_endpoint(self) -> None:
        """Verify GET /api/tiles/dem/{z}/{x}/{y}.png renders hillshade PNGs."""
        status, ctype, data = self._get_bytes('/api/tiles/dem/8/65/97.png')
        assert status == _HTTP_OK
        assert ctype == 'image/png'
        assert data.startswith(b'\x89PNG\r\n\x1a\n')
        assert len(data) > _MIN_PNG_BYTES

        status2, ctype2, data2 = self._get_bytes('/api/tiles/dem/8/0/0.png')
        assert status2 == _HTTP_OK
        assert ctype2 == 'image/png'
        assert data2.startswith(b'\x89PNG\r\n\x1a\n')

    def test_serve_frontend(self) -> None:
        """Verify GET / serves the OpenHydroNet index.html."""
        status, html = self._get('/')
        assert status == _HTTP_OK
        assert 'OpenHydroNet' in html


if __name__ == '__main__':
    unittest.main()
