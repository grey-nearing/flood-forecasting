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

"""Tests for the Geographical Features tab (openhydronet static_extractor)."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import unittest
import urllib.request
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

from frontend import tests as _isolated_profiles  # noqa: F401
from frontend.server import EarthkitHydroHandler
from frontend.static_attributes import (
    ATTRIBUTE_REGISTRY,
    build_attribute_card_payload,
    get_attributes_extractor,
    reset_attributes_extractor,
)
from frontend.tests.test_static_attributes import (
    build_synthetic_hydroatlas_fixtures,
)

_HTTP_OK = 200
_EXPECTED_SOME_COUNT = 2
_EXPECTED_ALL_COUNT = 3
_MIN_SCHEMA_ATTRIBUTES = 30

CATCHMENT_1: dict[str, Any] = {
    'type': 'Feature',
    'properties': {
        'catchment_id': 'geo_test_basin_1',
        'name': 'Upper Colorado Headwaters',
        'area_km2': 215.0,
        'source': 'delineated',
    },
    'geometry': {
        'type': 'Polygon',
        'coordinates': [
            [
                [-106.20, 39.50],
                [-106.00, 39.50],
                [-106.00, 39.70],
                [-106.20, 39.70],
                [-106.20, 39.50],
            ]
        ],
    },
}

CATCHMENT_2: dict[str, Any] = {
    'type': 'Feature',
    'properties': {
        'catchment_id': 'geo_test_basin_2',
        'name': 'Blue River Sub-basin',
        'area_km2': 180.0,
        'source': 'delineated',
    },
    'geometry': {
        'type': 'Polygon',
        'coordinates': [
            [
                [-106.05, 39.60],
                [-105.85, 39.60],
                [-105.85, 39.80],
                [-106.05, 39.80],
                [-106.05, 39.60],
            ]
        ],
    },
}

CATCHMENT_3: dict[str, Any] = {
    'type': 'Feature',
    'properties': {
        'catchment_id': 'geo_test_basin_3',
        'name': 'Eagle River Valley',
        'area_km2': 240.0,
        'source': 'uploaded',
    },
    'geometry': {
        'type': 'Polygon',
        'coordinates': [
            [
                [-106.50, 39.55],
                [-106.30, 39.55],
                [-106.30, 39.75],
                [-106.50, 39.75],
                [-106.50, 39.55],
            ]
        ],
    },
}


class GeoFeaturesTabTest(unittest.TestCase):
    """Hermetic integration tests for the Geographical Features tab."""

    _tmp_dir: tempfile.TemporaryDirectory[str]
    _stack: ExitStack
    server: ThreadingHTTPServer
    port: int
    thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        """Provision synthetic HydroATLAS fixtures and start test server."""
        super().setUpClass()
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._stack = ExitStack()
        hydroatlas_dir, era5_dir = build_synthetic_hydroatlas_fixtures(
            Path(cls._tmp_dir.name)
        )
        cls._stack.enter_context(
            mock.patch.dict(
                os.environ,
                {
                    'HYDROATLAS_GDB_PATH': str(hydroatlas_dir),
                    'ERA5_CLIMATE_CACHE_DIR': str(era5_dir),
                },
            )
        )
        reset_attributes_extractor()

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), EarthkitHydroHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down test server and clean up synthetic HydroATLAS fixtures."""
        cls.server.shutdown()
        cls.server.server_close()
        reset_attributes_extractor()
        cls._stack.close()
        cls._tmp_dir.cleanup()
        super().tearDownClass()

    def _get(
        self, path: str, headers: dict[str, str] | None = None
    ) -> tuple[int, Any]:
        url = f'http://127.0.0.1:{self.port}{path}'
        req = urllib.request.Request(url, headers=headers or {}, method='GET')
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = resp.read().decode('utf-8')
            status = resp.status
            content_type = resp.headers.get_content_type()
            parsed: Any = (
                json.loads(data) if content_type == 'application/json' else data
            )
            return status, parsed

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        url = f'http://127.0.0.1:{self.port}{path}'
        body = json.dumps(payload).encode('utf-8')
        req_headers = {'Content-Type': 'application/json'}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(
            url,
            data=body,
            headers=req_headers,
            method='POST',
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = resp.read().decode('utf-8')
            status = resp.status
            return status, json.loads(data)

    def test_static_extractor_openhydronet_integration(self) -> None:
        """Verify StaticAttributesExtractor uses openhydronet extractor."""
        extractor = get_attributes_extractor()
        res = build_attribute_card_payload(
            extractor.extract_attributes_for_polygon(
                CATCHMENT_1,
                catchment_id='geo_test_basin_1',
                era5_source='hybas',
            )
        )
        assert res['status'] == 'success'
        assert res['catchment_id'] == 'geo_test_basin_1'
        assert 'flat_attributes' in res
        assert 'categories' in res
        assert 'summary' in res
        flat = res['flat_attributes']
        for key in (
            'ele_mt_sav',
            'pre_mm_syr',
            'for_pc_sse',
            'cly_pc_sav',
            'p_mean',
            'aridity_ERA5_LAND',
        ):
            assert key in flat

    def test_extract_one_some_and_all_catchments(self) -> None:
        """Verify extracting attributes for One, Some, and All catchments."""
        headers = {'X-User-Profile': 'test_geo_user'}
        self._post('/api/watersheds/clear', {}, headers=headers)

        status, _ = self._post(
            '/api/polygons/upload',
            {
                'type': 'FeatureCollection',
                'features': [CATCHMENT_1, CATCHMENT_2, CATCHMENT_3],
            },
            headers=headers,
        )
        assert status == _HTTP_OK

        status, d_one = self._post(
            '/api/attributes/extract',
            {
                'catchment_id': 'geo_test_basin_1',
                'polygon': CATCHMENT_1,
            },
            headers=headers,
        )
        assert status == _HTTP_OK
        assert d_one['status'] == 'success'
        assert d_one['catchment_id'] == 'geo_test_basin_1'
        assert 'geo_test_basin_1' in d_one['results_by_id']

        status, d_some = self._post(
            '/api/attributes/extract',
            {
                'catchment_ids': ['geo_test_basin_2', 'geo_test_basin_3'],
                'features': [CATCHMENT_2, CATCHMENT_3],
            },
            headers=headers,
        )
        assert status == _HTTP_OK
        assert d_some['status'] == 'success'
        assert d_some['batch']
        assert d_some['count'] == _EXPECTED_SOME_COUNT
        assert {
            'geo_test_basin_1',
            'geo_test_basin_2',
            'geo_test_basin_3',
        }.issubset(set(d_some['results_by_id'].keys()))

        status, d_all = self._post(
            '/api/attributes/extract',
            {'all_active': True},
            headers=headers,
        )
        assert status == _HTTP_OK
        assert d_all['status'] == 'success'
        assert d_all['count'] == _EXPECTED_ALL_COUNT

        status, d_get = self._get('/api/attributes/extracted', headers=headers)
        assert status == _HTTP_OK
        assert d_get['count'] == _EXPECTED_ALL_COUNT
        assert 'geo_test_basin_1' in d_get['results_by_id']
        assert 'geo_test_basin_2' in d_get['results_by_id']
        assert 'geo_test_basin_3' in d_get['results_by_id']

        self._post('/api/watersheds/clear', {}, headers=headers)

    def test_raw_hydroatlas_attribute_map_layer_endpoint(self) -> None:
        """Verify GET /api/attributes/map-layer returns colored polygons."""
        s_status, schema = self._get('/api/attributes/schema')
        assert s_status == _HTTP_OK
        assert schema['extractor_source'] == 'multimet.static_extractor'
        assert isinstance(schema['attribute_list'], list)
        assert len(schema['attribute_list']) > _MIN_SCHEMA_ATTRIBUTES

        for attr_key in (
            'ele_mt_sav',
            'pre_mm_syr',
            'for_pc_sse',
            'cly_pc_sav',
            'p_mean',
            'pet_mean_FAO_PM',
        ):
            status, fc = self._get(
                '/api/attributes/map-layer'
                f'?bbox=-106.5,39.4,-105.8,39.9&zoom=8&attribute={attr_key}'
            )
            assert status == _HTTP_OK
            assert fc['type'] == 'FeatureCollection'
            props = fc['properties']
            assert props['attribute'] == attr_key
            assert props['unit'] == ATTRIBUTE_REGISTRY[attr_key].physical_unit
            assert props['count'] > 0
            assert props['min_value'] is not None
            assert props['mean_value'] is not None
            assert props['max_value'] is not None
            first_feat = fc['features'][0]
            assert first_feat['type'] == 'Feature'
            assert 'HYBAS_ID' in first_feat['properties']
            assert 'value' in first_feat['properties']

    def test_geo_features_tab_dom_and_map_reparenting(self) -> None:
        """Verify #tab-geo-features DOM contains Map Viewer and controls."""
        status, html = self._get('/')
        assert status == _HTTP_OK

        required_ids = [
            'tab-geo-features',
            'geoFeaturesMapSlot',
            'geoSelectModeAll',
            'geoSelectModeSome',
            'geoSelectModeOne',
            'geoSelectMissingBtn',
            'geoCheckAllBtn',
            'geoUncheckAllBtn',
            'geoCatchmentSelectorList',
            'extractAttrsBtn',
            'extractAttributesBtn',
            'attrsResult',
            'openAttrExplorerBtn',
            'downloadCaravanCsvBtn',
            'geoAttrSelect',
            'geoLayerModeExtracted',
            'geoLayerModeRaw',
            'geoLayerModeBoth',
            'geoRawLevelSelect',
            'geoRawOpacitySlider',
            'geoExtractedValuesList',
            'geoMapLegendCard',
            'geoLegendGradientBar',
            'modalCatchmentSelect',
        ]
        for elem_id in required_ids:
            assert re.search(rf'id=["\']{re.escape(elem_id)}["\']', html), (
                f'Missing #{elem_id} in Geographical Features tab'
            )

        assert 'geoFeaturesMapSlot.prepend(mapEl)' in html
        assert 'initGeoFeaturesMapLayers()' in html
        assert 'syncDefaultUnextractedSelection()' in html
        assert '/api/attributes/map-layer' in html


if __name__ == '__main__':
    unittest.main()
