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
"""Hermetic unit and integration tests for the OpenHydroNet Weather Engine."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontend import profile_manager as pm
from frontend import weather_engine as we
from frontend.server import OpenHydroNetHandler
from frontend.weather_engine import (
    SUPPORTED_MODELS,
    SUPPORTED_VARIABLES,
    _rate_file_steps,
    generate_raster_tile,
    get_catchment_weather_summary,
    get_weather_probe,
    get_wind_vectors,
)

_MIN_TILE_BYTES = 200
_STEP_2_HOURS = 6
_STEP_4_HOURS = 12
_PROBE_LAT = 40.42
_PROBE_LON = -86.92
_PROBE_STEPS = 81
_BASIN_AREA_KM2 = 4500.0
_HTTP_OK = 200
_MIN_MODELS_COUNT = 4
_API_PROBE_LAT = 45.0
_API_PROBE_LON = 5.0


def _populate_synthetic_weather_streams(stream_dir: Path) -> None:
    """Write deterministic binary weather streams for hermetic engine tests."""
    stream_dir.mkdir(parents=True, exist_ok=True)
    leads = list(range(0, 241, 6))  # 41 steps (0h..240h)
    plane = (we.N_LAT, we.N_LON)
    n_steps = len(leads)

    precip = np.full((n_steps, *plane), 2.5, dtype=np.float16)
    precip[0] = 0.0
    temp = np.linspace(-10.0, 25.0, we.N_LAT, dtype=np.float16)[None, :, None]
    temp_3d = np.broadcast_to(temp, (n_steps, *plane)).copy()
    u10 = np.full((n_steps, *plane), 4.0, dtype=np.float16)
    v10 = np.full((n_steps, *plane), -3.0, dtype=np.float16)
    lon_grid = np.linspace(0.0, 20.0, we.N_LON, dtype=np.float16)[None, None, :]
    mslp_3d = np.broadcast_to(lon_grid, (n_steps, *plane)).copy()

    datasets_meta: dict[str, dict[str, Any]] = {}
    for model_key in (
        'ecmwf_ifs',
        'ecmwf_hres',
        'ecmwf_aifs',
        'nasa_imerg',
        'noaa_cpc',
    ):
        precip.tofile(stream_dir / f'{model_key}_precip.bin')
        temp_3d.tofile(stream_dir / f'{model_key}_temp.bin')
        u10.tofile(stream_dir / f'{model_key}_u10.bin')
        v10.tofile(stream_dir / f'{model_key}_v10.bin')
        mslp_3d.tofile(stream_dir / f'{model_key}_mslp.bin')
        datasets_meta[f'{model_key}_forecast'] = {
            'model': model_key,
            'init_time': '2026-05-10T00:00:00Z',
            'lead_steps': n_steps,
            'lead_hours': leads,
            'streams': ['precip', 'temp', 'u10', 'v10', 'mslp'],
            'downloaded_utc': '2026-05-10T06:00:00Z',
        }

    (stream_dir / we.RUN_METADATA_FILE).write_text(
        json.dumps({'datasets': datasets_meta}), encoding='utf-8'
    )


class _HermeticWeatherStreamTestCase(unittest.TestCase):
    """Base test case provisioning hermetic binary weather streams."""

    _tmp_dir: tempfile.TemporaryDirectory[str]
    _saved_dirs: list[Path]
    _saved_warm: str | None

    @classmethod
    def setUpClass(cls) -> None:
        """Create temporary weather stream files and bind WeatherEngine."""
        super().setUpClass()
        cls._tmp_dir = tempfile.TemporaryDirectory()
        stream_dir = Path(cls._tmp_dir.name) / 'current'
        _populate_synthetic_weather_streams(stream_dir)

        cls._saved_dirs = list(we.CANDIDATE_DATA_DIRS)
        cls._saved_warm = os.environ.get('OPENHYDRONET_WEATHER_WARM_CACHE')
        os.environ['OPENHYDRONET_WEATHER_WARM_CACHE'] = '0'
        we.CANDIDATE_DATA_DIRS[:] = [stream_dir]
        we._INITIALIZED = False  # noqa: SLF001
        we.init_weather_streams()

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore WeatherEngine candidate directories and clean up."""
        we.CANDIDATE_DATA_DIRS[:] = cls._saved_dirs
        if cls._saved_warm is None:
            os.environ.pop('OPENHYDRONET_WEATHER_WARM_CACHE', None)
        else:
            os.environ['OPENHYDRONET_WEATHER_WARM_CACHE'] = cls._saved_warm
        we._INITIALIZED = False  # noqa: SLF001
        cls._tmp_dir.cleanup()
        super().tearDownClass()


@pytest.mark.unit
class WeatherEngineUnitTest(_HermeticWeatherStreamTestCase):
    """Tests core computational weather engine functions."""

    def test_supported_models_and_variables(self) -> None:
        """Verify supported model keys and meteorological variables."""
        for model_key in (
            'ecmwf_hres',
            'ecmwf_ifs',
            'ecmwf_aifs',
            'noaa_gfs',
            'noaa_gefs',
            'noaa_hrrr',
            'nasa_imerg',
            'noaa_cpc',
        ):
            assert model_key in SUPPORTED_MODELS
        assert 'graphcast' not in SUPPORTED_MODELS

        assert 'precipitation' in SUPPORTED_VARIABLES
        assert 'temperature' in SUPPORTED_VARIABLES
        assert 'wind' in SUPPORTED_VARIABLES
        assert 'pressure' in SUPPORTED_VARIABLES

    def test_generate_raster_tiles_png(self) -> None:
        """Verify Web Mercator PNG tile generation across variables."""
        precip_tile = generate_raster_tile(
            'ecmwf_ifs', 'precipitation', 2, 2, 1, 1
        )
        assert precip_tile.startswith(b'\x89PNG\r\n\x1a\n')
        assert len(precip_tile) > _MIN_TILE_BYTES

        temp_tile = generate_raster_tile('ecmwf_ifs', 'temperature', 6, 3, 2, 2)
        assert temp_tile.startswith(b'\x89PNG\r\n\x1a\n')
        assert len(temp_tile) > _MIN_TILE_BYTES

        press_tile = generate_raster_tile('ecmwf_ifs', 'pressure', 12, 2, 1, 1)
        assert press_tile.startswith(b'\x89PNG\r\n\x1a\n')

    def test_get_wind_vectors(self) -> None:
        """Verify downsampled U/V wind vector extraction."""
        vectors = get_wind_vectors('ecmwf_ifs', step_idx=2, subsample=2)
        assert 'header' in vectors
        assert 'u' in vectors
        assert 'v' in vectors

        header = vectors['header']
        assert header['model'] == 'ecmwf_ifs'
        assert header['step_hours'] == _STEP_2_HOURS
        assert header['nx'] > 0
        assert header['ny'] > 0
        assert len(vectors['u']) == header['nx'] * header['ny']
        assert len(vectors['v']) == header['nx'] * header['ny']

    def test_get_weather_probe(self) -> None:
        """Verify multi-model point meteogram probe extraction."""
        probe = get_weather_probe(_PROBE_LAT, _PROBE_LON)
        assert probe['latitude'] == _PROBE_LAT
        assert probe['longitude'] == _PROBE_LON
        assert len(probe['lead_hours']) == _PROBE_STEPS
        assert 'models' in probe
        assert 'ecmwf_hres' in probe['models']
        assert 'ecmwf_ifs' in probe['models']
        assert 'ecmwf_aifs' in probe['models']
        assert 'nasa_imerg' in probe['models']
        assert 'noaa_cpc' in probe['models']
        assert 'graphcast' not in probe['models']

        ifs = probe['models']['ecmwf_ifs']
        assert len(ifs['precip_rate_mmh']) == _PROBE_STEPS
        assert len(ifs['accum_precip_mm']) == _PROBE_STEPS
        assert len(ifs['temp_c']) == _PROBE_STEPS
        assert len(ifs['wind_speed_mps']) == _PROBE_STEPS
        assert len(ifs['pressure_hpa']) == _PROBE_STEPS

    def test_get_catchment_weather_summary(self) -> None:
        """Verify area-weighted catchment weather summary calculation."""
        sample_basin = {
            'id': 'test_basin',
            'properties': {
                'catchment_id': 'test_basin',
                'area_km2': _BASIN_AREA_KM2,
                'outlet_latitude': 40.5,
                'outlet_longitude': -86.5,
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-87.0, 40.0],
                        [-86.0, 40.0],
                        [-86.0, 41.0],
                        [-87.0, 41.0],
                        [-87.0, 40.0],
                    ]
                ],
            },
        }
        summary = get_catchment_weather_summary(sample_basin, step_idx=4)
        assert summary['catchment_id'] == 'test_basin'
        assert summary['area_km2'] == _BASIN_AREA_KM2
        assert summary['step_hours'] == _STEP_4_HOURS
        assert summary['basin_mean_precip_mmh'] >= 0.0
        assert summary['basin_accumulated_10d_mm'] > 0.0
        assert 'centroid' in summary

    def test_rain_rate_steps_cover_the_viewer_interval(self) -> None:
        """Verify file-step lookup for 1-hourly and 6-hourly rain rates."""
        hourly = {
            'lead_hours': [*range(121), 123, 126],
            'archived_run': True,
        }
        assert _rate_file_steps(hourly, 51) == [49, 50, 51]
        assert _rate_file_steps(hourly, 0) == [1]
        assert _rate_file_steps(hourly, 123) == [121]
        assert _rate_file_steps(hourly, 129) == []

        six_hourly = {'lead_hours': [0, 6, 12, 18], 'archived_run': True}
        assert _rate_file_steps(six_hourly, 0) == [1]
        assert _rate_file_steps(six_hourly, 3) == [1]
        assert _rate_file_steps(six_hourly, 9) == [2]
        assert _rate_file_steps(six_hourly, 21) == []


@pytest.mark.integration
class WeatherServerIntegrationTest(_HermeticWeatherStreamTestCase):
    """Integration test testing HTTP API endpoints over live loopback server."""

    server: ThreadingHTTPServer
    port: int
    thread: threading.Thread
    _saved_profile_mgr: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Start a local HTTP server with hermetic profile and stream dirs."""
        super().setUpClass()
        profiles_dir = Path(cls._tmp_dir.name) / 'profiles'
        cls._saved_profile_mgr = getattr(pm, '_PROFILE_MANAGER', None)
        temp_mgr = pm.ProfileManager(profiles_dir=profiles_dir)
        pm._PROFILE_MANAGER = temp_mgr  # noqa: SLF001

        sample_basin = {
            'id': 'global_default',
            'properties': {
                'catchment_id': 'global_default',
                'area_km2': _BASIN_AREA_KM2,
                'outlet_latitude': 40.5,
                'outlet_longitude': -86.5,
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-87.0, 40.0],
                        [-86.0, 40.0],
                        [-86.0, 41.0],
                        [-87.0, 41.0],
                        [-87.0, 40.0],
                    ]
                ],
            },
        }
        pm.get_profile_manager().save_watersheds([sample_basin], 'guest')

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), OpenHydroNetHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down the HTTP server and restore the profile manager."""
        cls.server.shutdown()
        cls.server.server_close()
        pm._PROFILE_MANAGER = cls._saved_profile_mgr  # noqa: SLF001
        super().tearDownClass()

    def _get(self, path: str) -> tuple[int, Any]:
        url = f'http://127.0.0.1:{self.port}{path}'
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
            data = resp.read()
            content_type = resp.headers.get('Content-Type', '')
            status = resp.status
            if 'application/json' in content_type:
                return status, json.loads(data.decode('utf-8'))
            return status, data

    def test_api_weather_models(self) -> None:
        """Verify `/api/weather/models` returns model and variable specs."""
        status, data = self._get('/api/weather/models')
        assert status == _HTTP_OK
        assert 'models' in data
        assert 'variables' in data
        assert len(data['models']) >= _MIN_MODELS_COUNT

    def test_api_weather_tiles_png(self) -> None:
        """Verify `/api/weather/tiles/...` returns a valid PNG image."""
        status, png_bytes = self._get(
            '/api/weather/tiles/ecmwf_ifs/precipitation/0/2/1/1.png'
        )
        assert status == _HTTP_OK
        assert png_bytes.startswith(b'\x89PNG\r\n\x1a\n')

    def test_api_weather_wind_vectors(self) -> None:
        """Verify `/api/weather/wind-vectors` returns U/V arrays."""
        status, data = self._get(
            '/api/weather/wind-vectors?model=ecmwf_ifs&step=0&subsample=4'
        )
        assert status == _HTTP_OK
        assert 'header' in data
        assert 'u' in data
        assert 'v' in data

    def test_api_weather_probe(self) -> None:
        """Verify `/api/weather/probe` returns multi-model meteograms."""
        status, data = self._get('/api/weather/probe?lat=45.0&lon=5.0')
        assert status == _HTTP_OK
        assert data['latitude'] == _API_PROBE_LAT
        assert data['longitude'] == _API_PROBE_LON
        assert 'models' in data

    def test_api_weather_catchment_summary(self) -> None:
        """Verify `/api/weather/catchment-summary` returns basin metrics."""
        status, data = self._get('/api/weather/catchment-summary?step=0')
        assert status == _HTTP_OK
        assert 'basin_mean_precip_mmh' in data
        assert 'basin_accumulated_10d_mm' in data


if __name__ == '__main__':
    unittest.main()
