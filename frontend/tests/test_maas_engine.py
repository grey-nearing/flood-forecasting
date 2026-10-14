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

"""Unit and integration tests for MaaS frontend adapter, viewer, and API."""

from __future__ import annotations

import json
import socketserver
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

import frontend.config as _frontend_config
import maas.fetcher as _maas_fetcher
from frontend import maas_engine
from frontend.delineator import reset_delineator_caches
from frontend.maas_live_status import (
    _LIVE_STATUS,
    _extract_current_floodhub_flow,
    _load_geoglows_rp_arrays,
    compute_live_flow_metrics,
    enrich_network_features_live,
    grid_feature_key,
)
from frontend.maas_networks import reset_maas_network_caches
from frontend.maas_viewer import (
    MaaSViewer,
    build_consensus_row,
    reach_exceedance_summary,
)
from frontend.river_indexer import reset_river_indexer_caches
from frontend.server import EarthkitHydroHandler
from maas.config import (
    GLOFAS_NLON,
    GLOFAS_RES_DEG,
    MaaSConfig,
    parse_finite_float,
)
from maas.floodhub import FloodHubClient
from maas.networks import (
    FLOODHUB_PYRAMID_LOD,
    GEOGLOWS_LOD,
    GLOFAS_LOD,
    pyramid_signature,
    save_network_pyramid,
)
from maas.thresholds import (
    classify_exceedance,
    estimate_return_period_years,
)
from multimet.catchment_delineation import RES_DEG, TILE_CELLS

_HTTP_OK = 200
_POLYGON_RING_LEN = 5
_SOUTH_D8 = 4
_WEST_D8 = 16
_EAST_D8 = 1
_MIN_CROSS_NETWORK_AREA_KM2 = 500000.0
_EXPECTED_NORMAL_MIN_PCT = 25.0
_EXPECTED_NORMAL_MAX_PCT = 75.0
_EXPECTED_HIGH_MAX_PCT = 90.0
_EXPECTED_WARN_MIN_PCT = 90.0
_EXPECTED_DANGER_MIN_PCT = 95.0
_EXPECTED_EXTREME_MIN_PCT = 99.0
_EXPECTED_RP50_MIN_PCT = 99.5
_EXPECTED_CURRENT_FH_FLOW = 120.0


def _populate_d8_basin(
    tile_arr: np.ndarray,
    center_row: int,
    center_col: int,
    half_width: int = 12,
) -> None:
    """Create a synthetic rectangular D8 drainage basin around `(row, col)`."""
    for r in range(center_row - half_width, center_row):
        for c in range(center_col - half_width, center_col + half_width + 1):
            tile_arr[r, c] = _SOUTH_D8
    for c in range(center_col - half_width, center_col):
        tile_arr[center_row, c] = _EAST_D8
    for c in range(center_col + 1, center_col + half_width + 1):
        tile_arr[center_row, c] = _WEST_D8


def _build_synthetic_maas_fixtures(root: Path) -> dict[str, Any]:
    """Build synthetic pyramids and hydrography fixtures in `root`."""
    river_dir = root / 'river_networks'
    cache_dir = root / 'cache'
    glofas_dir = river_dir / 'glofas_v4'
    geoglows_dir = river_dir / 'geoglows_v2'
    glofas_dir.mkdir(parents=True, exist_ok=True)
    geoglows_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    coords = np.array([[89.825, 23.925], [89.875, 23.875]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int64)
    bbox = np.array([[89.825, 23.875, 89.875, 23.925]], dtype=np.float32)
    amin = np.array([1480000.0], dtype=np.float32)
    amax = np.array([1480000.0], dtype=np.float32)

    gl_levels = [
        {
            'coords': coords,
            'offsets': offsets,
            'bbox': bbox,
            'amin': amin,
            'amax': amax,
        }
        for r in GLOFAS_LOD
        if r[2] is not None
    ]
    gl_row = round((90.0 - 23.875) / GLOFAS_RES_DEG - 0.5)
    gl_col = round((89.875 + 180.0) / GLOFAS_RES_DEG - 0.5)
    save_network_pyramid(
        glofas_dir / 'glofas_network_v1.npz',
        gl_levels,
        pyramid_signature(GLOFAS_LOD),
        cell_lin=np.array([gl_row * GLOFAS_NLON + gl_col], dtype=np.int64),
        cell_area=np.array([1480000.0], dtype=np.float32),
    )

    fh_levels = [
        {
            'coords': coords,
            'offsets': offsets,
            'bbox': bbox,
            'amin': amin,
            'amax': amax,
            'river_id': np.array([40001], dtype=np.int32),
            'hybas_l12': np.array([4120012340], dtype=np.int64),
            'has_forecast': np.array([1], dtype=np.uint8),
            'stream_order': np.array([8], dtype=np.int8),
        }
        for r in FLOODHUB_PYRAMID_LOD
        if r[2] is not None
    ]
    save_network_pyramid(
        glofas_dir / 'floodhub_network_v1.npz',
        fh_levels,
        pyramid_signature(FLOODHUB_PYRAMID_LOD),
        reach_lat=np.array([23.875], dtype=np.float32),
        reach_lon=np.array([89.875], dtype=np.float32),
        reach_hyriv_id=np.array([40001], dtype=np.int32),
        reach_hybas_l12=np.array([4120012340], dtype=np.int64),
        reach_area=np.array([1480000.0], dtype=np.float32),
    )

    gg_levels = [
        {
            'coords': coords,
            'offsets': offsets,
            'bbox': bbox,
            'amin': amin,
            'amax': amax,
            'river_id': np.array([720010511], dtype=np.int32),
        }
        for r in GEOGLOWS_LOD
        if r[2] is not None
    ]
    save_network_pyramid(
        geoglows_dir / 'geoglows_network_v1.npz',
        gg_levels,
        pyramid_signature(GEOGLOWS_LOD),
        reach_lat=np.array([23.875], dtype=np.float32),
        reach_lon=np.array([89.875], dtype=np.float32),
        reach_lon0=np.array([89.825], dtype=np.float32),
        reach_lat0=np.array([23.925], dtype=np.float32),
        reach_lon1=np.array([89.875], dtype=np.float32),
        reach_lat1=np.array([23.875], dtype=np.float32),
        reach_linkno=np.array([720010511], dtype=np.int32),
        reach_area=np.array([1480000.0], dtype=np.float32),
    )
    np.savez_compressed(
        geoglows_dir / 'geoglows_return_periods_v1.npz',
        river_id=np.array([720010511], dtype=np.int32),
        periods=np.array([2, 5, 10, 25, 50, 100], dtype=np.int32),
        gumbel=np.array(
            [[1800.0, 2400.0, 2900.0, 3700.0, 4200.0, 4900.0]],
            dtype=np.float32,
        ),
    )

    # Synthetic D8 tile n45w090.npy for (40.45, -86.95) and hydrography files
    dem_hs = root / 'dem_tiles' / 'hydrosheds_90m'
    dem_mh = root / 'dem_tiles' / 'merit_hydro_90m'
    dem_hs.mkdir(parents=True, exist_ok=True)
    dem_mh.mkdir(parents=True, exist_ok=True)
    tile = np.zeros((TILE_CELLS, TILE_CELLS), dtype=np.uint8)
    row_4045 = round((45.0 - 40.45) / RES_DEG)
    col_8695 = round((-86.95 - (-90.0)) / RES_DEG)
    _populate_d8_basin(tile, row_4045, col_8695, half_width=14)
    np.save(dem_hs / 'n45w090.npy', tile)
    np.save(dem_mh / 'n45w090.npy', tile)

    hr_dir = root / 'HydroRIVERS_v10_shp'
    hr_dir.mkdir(parents=True, exist_ok=True)
    hr_shp = hr_dir / 'HydroRIVERS_v10.shp'
    gpd.GeoDataFrame(
        {
            'HYRIV_ID': [70050001],
            'NEXT_DOWN': [0],
            'MAIN_RIV': [70050001],
            'LENGTH_KM': [6.5],
            'DIST_DN_KM': [0.0],
            'DIST_UP_KM': [12.0],
            'CATCH_SKM': [32.0],
            'UPLAND_SKM': [85.0],
            'DIS_AV_CMS': [4.2],
            'ORD_STRA': [4],
            'ORD_CLAS': [1],
            'ORD_FLOW': [5],
            'HYBAS_L12': [7120050010],
        },
        geometry=[
            LineString([(-86.96, 40.46), (-86.95, 40.45), (-86.94, 40.44)])
        ],
        crs='EPSG:4326',
    ).to_file(hr_shp)

    hb_dir = root / 'hydrobasins_lev12'
    hb_dir.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {
            'HYBAS_ID': [7120050010],
            'NEXT_DOWN': [0],
            'NEXT_SINK': [7120050010],
            'MAIN_BAS': [7120050010],
            'DIST_SINK': [0.0],
            'DIST_MAIN': [0.0],
            'SUB_AREA': [85.0],
            'UP_AREA': [85.0],
            'PFAF_ID': [742101000100],
            'ENDO': [0],
            'COAST': [0],
            'ORDER_': [4],
            'SORT': [1],
        },
        geometry=[box(-87.05, 40.35, -86.85, 40.55)],
        crs='EPSG:4326',
    ).to_file(hb_dir / 'hybas_na_lev12_v1c.shp')

    merit_dir = root / 'merit_basins'
    merit_dir.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {
            'COMID': [71000092],
            'unitarea': [42.5],
            'uparea': [120.0],
        },
        geometry=[box(90.60, 23.20, 90.68, 23.26)],
        crs='EPSG:4326',
    ).to_file(merit_dir / 'cat_pfaf_71_MERIT_Hydro_v07_Basins_v01.shp')

    return {
        'river_networks_dir': river_dir,
        'cache_dir': cache_dir,
        'dem_hs': dem_hs,
        'dem_mh': dem_mh,
        'hydrorivers_shp': hr_shp,
        'hydrobasins_dir': hb_dir,
        'merit_basins_dir': merit_dir,
    }


def _fake_requests_get(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float | None = None,  # noqa: ARG001
    **kwargs: object,  # noqa: ARG001
) -> mock.MagicMock:
    """Hermetic HTTP GET responder for Open-Meteo, GEOGLOWS, and FloodHub."""
    today = datetime.now(UTC)
    resp = mock.MagicMock()
    resp.status_code = _HTTP_OK
    resp.raise_for_status.return_value = None
    prm = params or {}
    payload: dict[str, Any] = {'floodStatuses': []}

    if 'open-meteo.com' in url or '/v1/flood' in url:
        if 'start_date' in prm:
            years = list(range(1984, 2024))
            payload = {
                'latitude': float(prm.get('latitude', 23.875)),
                'longitude': float(prm.get('longitude', 89.875)),
                'daily': {
                    'time': [f'{yr}-07-15' for yr in years],
                    'river_discharge': [
                        1800.0 + 80.0 * (yr - 1984) for yr in years
                    ],
                },
            }
        else:
            days = [
                (today + timedelta(days=d)).strftime('%Y-%m-%d')
                for d in range(15)
            ]
            payload = {
                'daily': {
                    'time': days,
                    'river_discharge': [2250.0 + 100.0 * d for d in range(15)],
                    'river_discharge_mean': [
                        2250.0 + 100.0 * d for d in range(15)
                    ],
                    'river_discharge_median': [
                        2200.0 + 100.0 * d for d in range(15)
                    ],
                    'river_discharge_max': [
                        2900.0 + 130.0 * d for d in range(15)
                    ],
                    'river_discharge_min': [
                        1800.0 + 80.0 * d for d in range(15)
                    ],
                    'river_discharge_p25': [
                        2000.0 + 90.0 * d for d in range(15)
                    ],
                    'river_discharge_p75': [
                        2500.0 + 110.0 * d for d in range(15)
                    ],
                    'river_discharge_member01': [
                        2150.0 + 95.0 * d for d in range(15)
                    ],
                    'river_discharge_member02': [
                        2350.0 + 105.0 * d for d in range(15)
                    ],
                }
            }
    elif '/getriverid' in url:
        payload = {'river_id': 720010511}
    elif '/forecaststats/' in url:
        payload = {
            'datetime': [
                (today + timedelta(days=d)).strftime('%Y-%m-%dT00:00:00Z')
                for d in range(15)
            ],
            'flow_med': [2100.0 + 95.0 * d for d in range(15)],
            'flow_avg': [2150.0 + 95.0 * d for d in range(15)],
            'flow_min': [1750.0 + 70.0 * d for d in range(15)],
            'flow_25p': [1950.0 + 85.0 * d for d in range(15)],
            'flow_75p': [2400.0 + 105.0 * d for d in range(15)],
            'flow_max': [2750.0 + 120.0 * d for d in range(15)],
        }
    elif '/returnperiods/' in url:
        payload = {
            'return_period_2': 1800.0,
            'return_period_5': 2400.0,
            'return_period_10': 2900.0,
            'return_period_20': 3500.0,
            'return_period_50': 4200.0,
            'return_period_100': 4900.0,
        }
    elif 'gauges:queryGaugeForecasts' in url:
        gid = str(prm.get('gaugeIds') or 'hybas_7120012340')
        payload = {
            'forecasts': {
                gid: {
                    'forecasts': [
                        {
                            'issuedTime': today.strftime('%Y-%m-%dT00:00:00Z'),
                            'forecastRanges': [
                                {
                                    'forecastStartTime': (
                                        today + timedelta(days=d)
                                    ).strftime('%Y-%m-%dT00:00:00Z'),
                                    'forecastEndTime': (
                                        today + timedelta(days=d + 1)
                                    ).strftime('%Y-%m-%dT00:00:00Z'),
                                    'value': 1950.0 + 120.0 * d,
                                }
                                for d in range(7)
                            ],
                        }
                    ]
                }
            }
        }
    elif 'gaugeModels:batchGet' in url:
        payload = {
            'gaugeModels': [
                {
                    'gaugeValueUnit': 'CUBIC_METERS_PER_SECOND',
                    'thresholds': {
                        'warningLevel': 1800.0,
                        'dangerLevel': 2400.0,
                        'extremeDangerLevel': 3500.0,
                    },
                }
            ]
        }

    resp.json.return_value = payload
    return resp


def _fake_requests_post(
    url: str,  # noqa: ARG001
    json: dict[str, Any] | None = None,  # noqa: ARG001
    timeout: float | None = None,  # noqa: ARG001
    **kwargs: object,  # noqa: ARG001
) -> mock.MagicMock:
    """Hermetic HTTP POST responder for FloodHub area gauge search."""
    today = datetime.now(UTC)
    resp = mock.MagicMock()
    resp.status_code = _HTTP_OK
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        'floodStatuses': [
            {
                'gaugeId': 'hybas_4120012340',
                'gaugeLocation': {'latitude': 23.88, 'longitude': 89.88},
                'severity': 'NO_FLOODING',
                'forecastTrend': 'STEADY',
                'issuedTime': today.strftime('%Y-%m-%dT00:00:00Z'),
                'qualityVerified': True,
                'source': 'HYBAS',
            }
        ]
    }
    return resp


def _enter_hermetic_maas_context(
    stack: ExitStack,
    fixtures: dict[str, Any],
) -> None:
    """Patch config paths and HTTP session boundary onto `stack`."""
    import frontend.delineator as _delineator  # noqa: PLC0415

    dem_hs = fixtures['dem_hs']
    dem_mh = fixtures['dem_mh']
    hydro_paths = {
        'hydrorivers_shp': fixtures['hydrorivers_shp'],
        'hydrobasins_dir': fixtures['hydrobasins_dir'],
        'merit_basins_dir': fixtures['merit_basins_dir'],
        'merit_rivers_dir': fixtures['merit_basins_dir'],
        'merit_catchments_dir': fixtures['merit_basins_dir'],
    }
    test_cfg = MaaSConfig(
        cache_dir=fixtures['cache_dir'],
        river_networks_dir=fixtures['river_networks_dir'],
        floodhub_api_key='test-key',
    )
    stack.enter_context(
        mock.patch.object(
            _frontend_config,
            'RIVER_NETWORKS_DIR',
            fixtures['river_networks_dir'],
        )
    )
    stack.enter_context(
        mock.patch.object(_frontend_config, 'CACHE_DIR', fixtures['cache_dir'])
    )
    stack.enter_context(
        mock.patch.dict(
            _frontend_config.DEM_TILE_LOCATIONS,
            {'hydrosheds_90m': (dem_hs,), 'merit_hydro_90m': (dem_mh,)},
        )
    )
    for target_mod in (_frontend_config, _delineator):
        stack.enter_context(
            mock.patch.object(
                target_mod,
                'resolve_dem_tiles_dir',
                side_effect=lambda dem_id='hydrosheds_90m': (
                    dem_mh if 'merit' in str(dem_id) else dem_hs
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                target_mod,
                'resolve_hydrography_paths',
                return_value=hydro_paths,
            )
        )
    stack.enter_context(
        mock.patch.object(maas_engine, 'get_maas_config', return_value=test_cfg)
    )
    stack.enter_context(
        mock.patch('requests.Session.get', side_effect=_fake_requests_get)
    )
    stack.enter_context(
        mock.patch('requests.Session.post', side_effect=_fake_requests_post)
    )
    reset_maas_network_caches()
    reset_delineator_caches()
    reset_river_indexer_caches()
    _maas_fetcher._PYRAMID_MEM_CACHE.clear()  # noqa: SLF001


class TestMaaSWatershedAndAdapter(unittest.TestCase):
    """Tests watershed polygon generation and `maas_engine` delegation."""

    _tmp_dir: tempfile.TemporaryDirectory[str]
    _stack: ExitStack

    @classmethod
    def setUpClass(cls) -> None:
        """Provision synthetic hydrography and pyramid fixtures."""
        super().setUpClass()
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._stack = ExitStack()
        fixtures = _build_synthetic_maas_fixtures(Path(cls._tmp_dir.name))
        _enter_hermetic_maas_context(cls._stack, fixtures)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up synthetic fixtures and reset caches."""
        cls._stack.close()
        reset_maas_network_caches()
        reset_delineator_caches()
        reset_river_indexer_caches()
        _maas_fetcher._PYRAMID_MEM_CACHE.clear()  # noqa: SLF001
        cls._tmp_dir.cleanup()
        super().tearDownClass()

    def test_get_maas_watershed_polygon_glofas(self) -> None:
        """Verify GloFAS 0.05-deg cell watershed polygon."""
        feat = maas_engine.get_maas_watershed_polygon(
            23.23, 90.64, fabric='glofas_cell'
        )
        assert feat.get('type') == 'Feature'
        geom = feat.get('geometry', {})
        assert geom.get('type') == 'Polygon'
        assert len(geom.get('coordinates', [[]])[0]) == _POLYGON_RING_LEN
        props = feat.get('properties', {})
        assert props.get('fabric') == 'glofas_cell'
        assert props.get('area_km2', 0) > 0

    def test_get_maas_watershed_polygon_merit_and_hydroatlas(self) -> None:
        """Verify MERIT-Basins and HydroATLAS watershed polygons."""
        feat = maas_engine.get_maas_watershed_polygon(
            23.23, 90.64, fabric='merit_reach', river_id=71000092
        )
        assert feat.get('type') == 'Feature'
        assert feat.get('properties', {}).get('fabric') == 'merit_reach'

        unit_feat = maas_engine.get_maas_watershed_polygon(
            40.45, -86.95, fabric='hydroatlas_unit'
        )
        assert unit_feat.get('type') == 'Feature'
        assert (
            unit_feat.get('properties', {}).get('fabric') == 'hydroatlas_unit'
        )

        full_feat = maas_engine.get_maas_watershed_polygon(
            40.45, -86.95, fabric='hydroatlas_full'
        )
        assert full_feat.get('type') == 'Feature'
        assert (
            full_feat.get('properties', {}).get('fabric') == 'hydroatlas_full'
        )

        # Missing merit-hydro vector shapefiles must raise ValueError loudly
        # rather than silently substituting a different hydrofabric.
        with (
            mock.patch(
                'frontend.maas_viewer.viewer.find_merit_shp', return_value=None
            ),
            mock.patch(
                'frontend.delineator.get_vector_delineator',
                return_value=None,
            ),
            pytest.raises(ValueError, match='merit-hydro'),
        ):
            maas_engine.get_maas_watershed_polygon(
                40.45, -86.95, fabric='merit_reach', river_id=999999999
            )


class TestMaaSContracts(unittest.TestCase):
    """Deterministic tests for `maas` and `frontend.maas_viewer` contracts."""

    @staticmethod
    def _iso(dt: datetime) -> str:
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    def _issue(
        self,
        issued: datetime,
        starts: list[datetime],
        values: list[Any],
    ) -> dict[str, Any]:
        return {
            'issuedTime': self._iso(issued),
            'forecastRanges': [
                {
                    'forecastStartTime': self._iso(s),
                    'forecastEndTime': self._iso(s + timedelta(days=1)),
                    'value': v,
                }
                for s, v in zip(starts, values, strict=False)
            ],
        }

    def test_parse_finite_float_rejects_nan_strings(self) -> None:
        """Verify NaN, Inf, and non-numeric strings return None."""
        for bad in (
            'NaN',
            'nan',
            'inf',
            '-Infinity',
            '',
            None,
            'abc',
            float('nan'),
        ):
            assert parse_finite_float(bad) is None, repr(bad)
        assert parse_finite_float('1.5') == 1.5  # noqa: PLR2004
        assert parse_finite_float(2) == 2.0  # noqa: PLR2004

    def test_estimate_return_period_years_consistent_with_levels(self) -> None:
        """Verify return-period interpolation agrees with exceedance ranks."""
        rps = {
            'return_period_2': 47.7,
            'return_period_5': 127.2,
            'return_period_20': 298.6,
            'status': 'live',
        }
        for below in (0.0, 0.6, 47.0, None):
            assert estimate_return_period_years(below, rps) is None
        assert (
            estimate_return_period_years(100.0, {'return_period_2': 50.0})
            is None
        )
        assert estimate_return_period_years(47.7, rps) == 2.0  # noqa: PLR2004
        assert estimate_return_period_years(127.2, rps) == 5.0  # noqa: PLR2004
        assert estimate_return_period_years(298.6, rps) == 20.0  # noqa: PLR2004
        rp_mid = estimate_return_period_years(200.0, rps)
        assert rp_mid is not None
        assert 5.0 < rp_mid < 20.0  # noqa: PLR2004
        rp_high = estimate_return_period_years(400.0, rps)
        assert rp_high is not None
        assert rp_high > 20.0  # noqa: PLR2004
        assert estimate_return_period_years(1e9, rps) == 1000.0  # noqa: PLR2004
        for q in (60.0, 150.0, 320.0):
            rank = classify_exceedance(q, rps)['rank']
            rp_est = estimate_return_period_years(q, rps)
            assert rp_est is not None
            assert rp_est >= {1: 2.0, 2: 5.0, 3: 20.0}[rank]

    def test_consensus_row_and_reach_exceedance_summary(self) -> None:
        """Verify live vs unavailable model consensus and reach exceedance."""
        rps = {
            'return_period_2': 0.5,
            'return_period_5': 1.0,
            'return_period_20': 2.0,
            'status': 'computed',
        }
        live = build_consensus_row(
            'geoglows',
            'live',
            58.0,
            '2026-09-30',
            rps,
            'src',
            'High',
            available=True,
        )
        assert live['risk_level'] == 'EXTREME'
        unavail = build_consensus_row(
            'geoglows',
            'unavailable',
            None,
            None,
            rps,
            'src',
            'N/A',
            available=False,
        )
        assert unavail['risk_level'] == 'UNKNOWN'
        assert unavail['exceedance_rank'] is None
        assert unavail['exceedance_label'] is None

        today = datetime.now(UTC)

        def forecast(status: str, key: str, value: float) -> dict[str, Any]:
            days = [
                (today + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(3)
            ]
            return {
                'status': status,
                'data': [{'time': d, key: value} for d in days],
            }

        geoglows_unavail = {'status': 'unavailable', 'data': []}
        summary = reach_exceedance_summary(
            forecast('live', 'discharge_median', 0.2),
            rps,
            geoglows_unavail,
            rps,
        )
        assert summary['governing_model'] == 'glofas'
        assert summary['rank'] == 0
        assert summary['excluded_models'] == []

    def test_floodhub_client_skips_nan_and_returns_unavailable(self) -> None:
        """Verify FloodHubClient skips NaN issues and never fabricates data."""
        now = datetime.now(UTC)
        gid = 'TEST_FH_GAUGE'
        starts = [now + timedelta(days=d) for d in range(3)]
        finite = self._issue(
            now - timedelta(hours=6), starts, [10.0, 20.5, 30.25]
        )
        nan_issue = self._issue(
            now - timedelta(hours=1), starts, ['NaN', 'NaN', 'NaN']
        )
        thresholds = {
            'warningLevel': 12.0,
            'dangerLevel': 'NaN',
            'extremeDangerLevel': 30.0,
        }

        session = mock.MagicMock()

        def fake_get(
            url: str,
            params: dict[str, Any] | None = None,  # noqa: ARG001
            timeout: float = 12.0,  # noqa: ARG001
        ) -> mock.MagicMock:
            resp = mock.MagicMock()
            resp.status_code = _HTTP_OK
            if 'gauges:queryGaugeForecasts' in url:
                resp.json.return_value = {
                    'forecasts': {gid: {'forecasts': [finite, nan_issue]}}
                }
            elif 'gaugeModels:batchGet' in url:
                resp.json.return_value = {
                    'gaugeModels': [
                        {
                            'gaugeValueUnit': 'CUBIC_METERS_PER_SECOND',
                            'thresholds': thresholds,
                        }
                    ]
                }
            return resp

        session.get.side_effect = fake_get
        client = FloodHubClient(api_key='test-key', session=session)
        res = client.fetch_forecast(gid, now_utc=now)
        assert res['status'] == 'live'
        assert res['nan_issues_skipped'] == 1
        assert [p['discharge'] for p in res['data']] == [10.0, 20.5, 30.25]
        assert res['thresholds'] == {
            'warning_2yr': 12.0,
            'danger_5yr': None,
            'extreme_20yr': 30.0,
        }

        all_nan = [
            self._issue(now - timedelta(hours=h), starts, ['NaN'] * 3)
            for h in (12, 6, 1)
        ]
        session.get.side_effect = lambda url, **_kw: mock.MagicMock(
            status_code=_HTTP_OK,
            json=mock.MagicMock(
                return_value={'forecasts': {gid: {'forecasts': all_nan}}}
                if 'queryGaugeForecasts' in url
                else {'gaugeModels': []}
            ),
        )
        res_nan = client.fetch_forecast(gid, now_utc=now)
        assert not res_nan['available']
        assert res_nan['status'] == 'unavailable'
        assert res_nan['data'] == []
        assert res_nan['nan_issues_skipped'] == 3  # noqa: PLR2004

    def test_maas_viewer_render_forecast_view(self) -> None:
        """Verify MaaSViewer renders forecasts by mocking HTTP boundary."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = MaaSConfig(
                cache_dir=tmp_path / 'cache',
                river_networks_dir=tmp_path / 'river_networks',
                floodhub_api_key='test-key',
            )
            with (
                mock.patch(
                    'requests.Session.get', side_effect=_fake_requests_get
                ),
                mock.patch(
                    'requests.Session.post', side_effect=_fake_requests_post
                ),
            ):
                viewer = MaaSViewer(cfg)
                view = viewer.render_forecast_view(
                    38.6270,
                    -90.1994,
                    gauge_id='hybas_7120012340',
                    requested_models=['floodhub', 'glofas', 'geoglows'],
                )
                assert set(view['models']) == {'floodhub', 'glofas', 'geoglows'}
                assert [r['model'] for r in view['consensus']] == [
                    'floodhub',
                    'glofas',
                    'geoglows',
                ]
                assert 'overall_risk_level' in view['flood_summary']
                timeline = view['timeline']
                assert len(timeline['dates']) > 0
                for key, series in timeline['series'].items():
                    assert 'central' in series, key
                    assert len(series['central']) == len(timeline['dates']), key
                assert 'status' in timeline


class TestMaaSServerEndpoints(unittest.TestCase):
    """Integration tests for `/api/maas/*` HTTP endpoints."""

    _tmp_dir: tempfile.TemporaryDirectory[str]
    _stack: ExitStack
    server: socketserver.TCPServer
    port: int
    thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        """Provision synthetic pyramids, mock HTTP, and start test server."""
        super().setUpClass()
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._stack = ExitStack()
        fixtures = _build_synthetic_maas_fixtures(Path(cls._tmp_dir.name))
        _enter_hermetic_maas_context(cls._stack, fixtures)

        cls.server = socketserver.TCPServer(
            ('127.0.0.1', 0), EarthkitHydroHandler
        )
        cls.port = int(cls.server.server_address[1])
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down test server and clean up synthetic fixtures."""
        cls.server.shutdown()
        cls.server.server_close()
        cls._stack.close()
        reset_maas_network_caches()
        reset_delineator_caches()
        reset_river_indexer_caches()
        _maas_fetcher._PYRAMID_MEM_CACHE.clear()  # noqa: SLF001
        cls._tmp_dir.cleanup()
        super().tearDownClass()

    def _get(
        self, path: str, timeout: float = 30.0
    ) -> tuple[int, dict[str, Any]]:
        url = f'http://127.0.0.1:{self.port}{path}'
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode('utf-8'))

    def test_api_maas_models(self) -> None:
        """Verify GET /api/maas/models returns all 3 model providers."""
        status, data = self._get('/api/maas/models')
        assert status == _HTTP_OK
        assert 'models' in data
        model_ids = [m['id'] for m in data['models']]
        assert model_ids == ['floodhub', 'glofas', 'geoglows']

    def test_api_maas_gauges(self) -> None:
        """Verify GET /api/maas/gauges queries FloodHub via HTTP boundary."""
        status, data = self._get(
            '/api/maas/gauges'
            '?min_lat=22.0&min_lon=88.0&max_lat=25.5&max_lon=91.5'
        )
        assert status == _HTTP_OK
        assert data['count'] == 1

    def test_api_maas_reaches(self) -> None:
        """Verify GET /api/maas/reaches resolves GloFAS and GEOGLOWS reaches."""
        status, data = self._get('/api/maas/reaches?lat=23.875&lon=89.875')
        assert status == _HTTP_OK
        assert 'glofas' in data
        assert 'geoglows' in data

    def test_api_maas_watershed(self) -> None:
        """Verify GET /api/maas/watershed returns GeoJSON polygon."""
        status, data = self._get(
            '/api/maas/watershed?lat=23.23&lon=90.64&fabric=glofas_cell'
        )
        assert status == _HTTP_OK
        assert data.get('type') == 'Feature'
        assert data.get('properties', {}).get('fabric') == 'glofas_cell'

    def test_api_maas_forecast_http_boundary_mocked(self) -> None:
        """Verify GET /api/maas/forecast with HTTP boundary mocking."""
        status, data = self._get(
            '/api/maas/forecast'
            '?lat=32.756&lon=-117.252&gauge_id=hybas_7120012340'
        )
        assert status == _HTTP_OK
        assert set(data['models']) == {'floodhub', 'glofas', 'geoglows'}
        assert [r['model'] for r in data['consensus']] == [
            'floodhub',
            'glofas',
            'geoglows',
        ]
        assert 'glofas' in data['timeline']['series']
        gl_tl = data['timeline']['series']['glofas']
        assert 'deterministic' in gl_tl
        assert 'median' in gl_tl
        assert 'members' in gl_tl
        assert set(gl_tl['members']) == {'member01', 'member02'}

        status_sub, data_sub = self._get(
            '/api/maas/forecast?lat=32.756&lon=-117.252&models=glofas,geoglows'
        )
        assert status_sub == _HTTP_OK
        assert [r['model'] for r in data_sub['consensus']] == [
            'glofas',
            'geoglows',
        ]

    def test_api_maas_network_telescoping_all_models(self) -> None:
        """Verify GET /api/maas/network returns features for all 3 models."""
        for model in ('floodhub', 'glofas', 'geoglows'):
            status, data = self._get(
                f'/api/maas/network?model={model}&bbox=88.0,22.0,91.5,25.5&zoom=8'
            )
            assert status == _HTTP_OK
            assert data.get('type') == 'FeatureCollection'
            assert (data.get('properties') or {}).get('model') == model
            assert len(data.get('features') or []) > 0

    def test_api_maas_cross_network_reach_matching(self) -> None:
        """Verify 3-way cross-network reach snapping across pyramids."""
        for src_net in ('glofas', 'geoglows', 'floodhub'):
            status, data = self._get(
                '/api/maas/forecast'
                f'?lat=23.875&lon=89.875&network={src_net}'
                '&upstream_area_km2=1480000'
            )
            assert status == _HTTP_OK
            vs = data.get('virtual_station') or {}
            gl_area = (vs.get('glofas_cell') or {}).get(
                'upstream_area_km2'
            ) or 0
            gg_area = (vs.get('geoglows_reach') or {}).get(
                'upstream_area_km2'
            ) or 0
            fh_gauge = (vs.get('floodhub_gauge') or {}).get('gauge_id')
            fh_area = (vs.get('floodhub_gauge') or {}).get(
                'upstream_area_km2'
            ) or 0
            assert gl_area > _MIN_CROSS_NETWORK_AREA_KM2
            assert gg_area > _MIN_CROSS_NETWORK_AREA_KM2
            assert fh_gauge == 'hybas_4120012340'
            assert fh_area > _MIN_CROSS_NETWORK_AREA_KM2

        # Also verify snapping when upstream_area_km2 is omitted
        status_no_area, data_no_area = self._get(
            '/api/maas/forecast?lat=23.875&lon=89.875&network=geoglows'
        )
        assert status_no_area == _HTTP_OK
        reaches_no_area = data_no_area.get('reaches') or {}
        assert (
            reaches_no_area.get('floodhub') or {}
        ).get('gauge_id') == 'hybas_4120012340'
        assert (
            (reaches_no_area.get('glofas') or {}).get('upstream_area_km2') or 0
        ) > _MIN_CROSS_NETWORK_AREA_KM2

    def test_maas_multi_hydrofabric_ui_controls_present(self) -> None:
        """Verify multi-hydrofabric toggle and GloFAS ensemble hooks in UI."""
        html_path = (
            Path(__file__).resolve().parents[1] / 'static' / 'index.html'
        )
        html_text = html_path.read_text(encoding='utf-8')
        assert 'selectedMaaSMapModels' in html_text
        assert 'toggleMaaSMapModel' in html_text
        assert 'updateMaaSSnappedReachMarkers' in html_text
        assert 'maasModelSubLayers' in html_text
        assert 'maasGlofasEnsembleEnabled' in html_text
        assert 'toggleMaaSGlofasEnsemble' in html_text
        assert 'appendMaaSGlofasEnsembleDatasets' in html_text

    def test_live_network_status_coloring(self) -> None:
        """Verify live flow percentile bins and exceedance rank coloring."""
        m_low = compute_live_flow_metrics(2.0, q2=200.0, q5=350.0, q20=600.0)
        assert m_low['flow_percentile'] < _EXPECTED_NORMAL_MIN_PCT
        assert m_low['exceedance_rank'] == 0

        m_norm = compute_live_flow_metrics(57.0, q2=200.0, q5=350.0, q20=600.0)
        assert m_norm['flow_percentile'] >= _EXPECTED_NORMAL_MIN_PCT
        assert m_norm['flow_percentile'] < _EXPECTED_NORMAL_MAX_PCT
        assert m_norm['exceedance_rank'] == 0

        m_high = compute_live_flow_metrics(150.0, q2=200.0, q5=350.0, q20=600.0)
        assert m_high['flow_percentile'] >= _EXPECTED_NORMAL_MAX_PCT
        assert m_high['flow_percentile'] < _EXPECTED_HIGH_MAX_PCT
        assert m_high['exceedance_rank'] == 0

        m_warn = compute_live_flow_metrics(240.0, q2=200.0, q5=350.0, q20=600.0)
        assert m_warn['flow_percentile'] >= _EXPECTED_WARN_MIN_PCT
        assert m_warn['exceedance_rank'] == 1

        m_danger = compute_live_flow_metrics(
            400.0, q2=200.0, q5=350.0, q20=600.0
        )
        assert m_danger['flow_percentile'] >= _EXPECTED_DANGER_MIN_PCT
        assert m_danger['exceedance_rank'] == 2  # noqa: PLR2004

        m_ext = compute_live_flow_metrics(650.0, q2=200.0, q5=350.0, q20=600.0)
        assert m_ext['flow_percentile'] >= _EXPECTED_EXTREME_MIN_PCT
        assert m_ext['exceedance_rank'] == 3  # noqa: PLR2004

        ranges = [
            {
                'forecastStartTime': '2026-10-08T00:00:00Z',
                'forecastEndTime': '2026-10-09T00:00:00Z',
                'value': 120.0,
            },
            {
                'forecastStartTime': '2026-10-12T00:00:00Z',
                'forecastEndTime': '2026-10-13T00:00:00Z',
                'value': 950.0,
            },
        ]
        assert (
            _extract_current_floodhub_flow(
                [{'forecastRanges': ranges}], '2026-10-08T12:00:00Z'
            )
            == _EXPECTED_CURRENT_FH_FLOW
        )

        _LIVE_STATUS['geoglows']['760021611'] = m_warn
        feats = [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[-90.0, 38.0], [-90.1, 38.1]],
                },
                'properties': {
                    'river_id': 760021611,
                    'upstream_area_km2': 50000.0,
                },
            }
        ]
        colored, pending = enrich_network_features_live(
            'geoglows', feats, trigger_fetch=False
        )
        assert colored == 1
        assert pending == 0
        assert feats[0]['properties']['exceedance_rank'] == 1
        assert (
            feats[0]['properties']['flow_percentile'] >= _EXPECTED_WARN_MIN_PCT
        )

        m_rp50 = compute_live_flow_metrics(
            1100.0, q2=200.0, q5=350.0, q20=600.0, q50=950.0
        )
        assert m_rp50['exceedance_rank'] == 4  # noqa: PLR2004
        assert m_rp50['flow_percentile'] >= _EXPECTED_RP50_MIN_PCT

        coarse_coords = [[89.5, 24.5], [89.875, 23.875]]
        fine_coords = [
            [89.5, 24.5],
            [89.62, 24.31],
            [89.74, 24.12],
            [89.875, 23.875],
        ]
        assert grid_feature_key(coarse_coords, 45000.0) == grid_feature_key(
            fine_coords, 45000.0
        )

        rp_arrays = _load_geoglows_rp_arrays()
        assert rp_arrays is not None
        assert len(rp_arrays[0]) >= 1


if __name__ == '__main__':
    unittest.main()
