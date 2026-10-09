"""Tests for the Geographical Features tab (googlehydrology static_extractor + Map Viewer)."""

from http.server import ThreadingHTTPServer
import json
import re
import threading
import time
import unittest
from unittest.mock import patch
import urllib.parse
import urllib.request

from frontend.server import EarthkitHydroHandler
from frontend.static_attributes import (
    ATTRIBUTE_REGISTRY,
    StaticAttributesExtractor,
)
# Importing the tests package points profiles at a temp folder (never real accounts).
from frontend import tests as _isolated_profiles  # pylint: disable=unused-import

CATCHMENT_1 = {
    "type": "Feature",
    "properties": {
        "catchment_id": "geo_test_basin_1",
        "name": "Upper Colorado Headwaters",
        "area_km2": 215.0,
        "source": "delineated",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-106.20, 39.50],
            [-106.00, 39.50],
            [-106.00, 39.70],
            [-106.20, 39.70],
            [-106.20, 39.50],
        ]],
    },
}

CATCHMENT_2 = {
    "type": "Feature",
    "properties": {
        "catchment_id": "geo_test_basin_2",
        "name": "Blue River Sub-basin",
        "area_km2": 180.0,
        "source": "delineated",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-106.05, 39.60],
            [-105.85, 39.60],
            [-105.85, 39.80],
            [-106.05, 39.80],
            [-106.05, 39.60],
        ]],
    },
}

CATCHMENT_3 = {
    "type": "Feature",
    "properties": {
        "catchment_id": "geo_test_basin_3",
        "name": "Eagle River Valley",
        "area_km2": 240.0,
        "source": "uploaded",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-106.50, 39.55],
            [-106.30, 39.55],
            [-106.30, 39.75],
            [-106.50, 39.75],
            [-106.50, 39.55],
        ]],
    },
}


class GeoFeaturesTabTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EarthkitHydroHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path: str, headers: dict | None = None):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode("utf-8")
            status = resp.status
            try:
                parsed = json.loads(data)
            except Exception:
                parsed = data
            return status, parsed

    def _post(self, path: str, payload: dict, headers: dict | None = None):
        url = f"http://127.0.0.1:{self.port}{path}"
        body = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(
            url,
            data=body,
            headers=req_headers,
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode("utf-8")
            status = resp.status
            return status, json.loads(data)

    def test_static_extractor_googlehydrology_integration(self):
        """Verify StaticAttributesExtractor uses the googlehydrology static_extractor package."""
        from frontend.static_attributes import get_attributes_extractor, build_attribute_card_payload
        extractor = get_attributes_extractor()
        res = build_attribute_card_payload(extractor.extract_attributes_for_polygon(CATCHMENT_1, catchment_id="geo_test_basin_1", era5_source="hybas"))
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["catchment_id"], "geo_test_basin_1")
        self.assertIn("flat_attributes", res)
        self.assertIn("categories", res)
        self.assertIn("summary", res)
        flat = res["flat_attributes"]
        for key in ("ele_mt_sav", "pre_mm_syr", "for_pc_sse", "cly_pc_sav", "p_mean", "aridity_ERA5_LAND"):
            self.assertIn(key, flat)

    def test_extract_one_some_and_all_catchments(self):
        """Verify extracting attributes for One, Some, and All catchments and retrieving persisted state."""
        headers = {"X-User-Profile": "test_geo_user"}
        self._post("/api/watersheds/clear", {}, headers=headers)

        # Upload 3 catchments to active profile
        status, _ = self._post(
            "/api/polygons/upload",
            {
                "type": "FeatureCollection",
                "features": [CATCHMENT_1, CATCHMENT_2, CATCHMENT_3],
            },
            headers=headers,
        )
        self.assertEqual(status, 200)

        # 1. Extract ONE catchment
        status, d_one = self._post(
            "/api/attributes/extract",
            {
                "catchment_id": "geo_test_basin_1",
                "polygon": CATCHMENT_1,
            },
            headers=headers,
        )
        self.assertEqual(status, 200)
        self.assertEqual(d_one["status"], "success")
        self.assertEqual(d_one["catchment_id"], "geo_test_basin_1")
        self.assertIn("geo_test_basin_1", d_one["results_by_id"])

        # 2. Extract SOME catchments (subset of 2 catchments via catchment_ids)
        status, d_some = self._post(
            "/api/attributes/extract",
            {
                "catchment_ids": ["geo_test_basin_2", "geo_test_basin_3"],
                "features": [CATCHMENT_2, CATCHMENT_3],
            },
            headers=headers,
        )
        self.assertEqual(status, 200)
        self.assertEqual(d_some["status"], "success")
        self.assertTrue(d_some["batch"])
        self.assertEqual(d_some["count"], 2)
        self.assertTrue(
            {
                "geo_test_basin_1",
                "geo_test_basin_2",
                "geo_test_basin_3",
            }.issubset(set(d_some["results_by_id"].keys()))
        )

        # 3. Extract ALL active catchments
        status, d_all = self._post(
            "/api/attributes/extract",
            {"all_active": True},
            headers=headers,
        )
        self.assertEqual(status, 200)
        self.assertEqual(d_all["status"], "success")
        self.assertEqual(d_all["count"], 3)

        # 4. Verify GET /api/attributes/extracted returns all 3 persisted catchments
        status, d_get = self._get("/api/attributes/extracted", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(d_get["count"], 3)
        self.assertIn("geo_test_basin_1", d_get["results_by_id"])
        self.assertIn("geo_test_basin_2", d_get["results_by_id"])
        self.assertIn("geo_test_basin_3", d_get["results_by_id"])

        # Clean up test user watersheds
        self._post("/api/watersheds/clear", {}, headers=headers)

    def test_raw_hydroatlas_attribute_map_layer_endpoint(self):
        """Verify GET /api/attributes/map-layer returns colored BasinATLAS polygons for multiple attributes."""
        s_status, schema = self._get("/api/attributes/schema")
        self.assertEqual(s_status, 200)
        self.assertEqual(schema["extractor_source"], "multimet.static_extractor")
        self.assertIsInstance(schema["attribute_list"], list)
        self.assertGreater(len(schema["attribute_list"]), 30)

        for attr_key in ("ele_mt_sav", "pre_mm_syr", "for_pc_sse", "cly_pc_sav", "p_mean", "pet_mean_FAO_PM"):
            status, fc = self._get(
                f"/api/attributes/map-layer?bbox=-106.5,39.4,-105.8,39.9&zoom=8&attribute={attr_key}"
            )
            self.assertEqual(status, 200)
            self.assertEqual(fc["type"], "FeatureCollection")
            props = fc["properties"]
            self.assertEqual(props["attribute"], attr_key)
            self.assertEqual(props["unit"], ATTRIBUTE_REGISTRY[attr_key].physical_unit)
            self.assertGreater(props["count"], 0)
            self.assertIsNotNone(props["min_value"])
            self.assertIsNotNone(props["mean_value"])
            self.assertIsNotNone(props["max_value"])
            first_feat = fc["features"][0]
            self.assertEqual(first_feat["type"], "Feature")
            self.assertIn("HYBAS_ID", first_feat["properties"])
            self.assertIn("value", first_feat["properties"])

    def test_geo_features_tab_dom_and_map_reparenting(self):
        """Verify #tab-geo-features contains the Map Viewer slot, All/Some/One selector, and attribute layer controls."""
        status, html = self._get("/")
        self.assertEqual(status, 200)

        required_ids = [
            "tab-geo-features",
            "geoFeaturesMapSlot",
            "geoSelectModeAll",
            "geoSelectModeSome",
            "geoSelectModeOne",
            "geoSelectMissingBtn",
            "geoCheckAllBtn",
            "geoUncheckAllBtn",
            "geoCatchmentSelectorList",
            "extractAttrsBtn",
            "extractAttributesBtn",
            "attrsResult",
            "openAttrExplorerBtn",
            "downloadCaravanCsvBtn",
            "geoAttrSelect",
            "geoLayerModeExtracted",
            "geoLayerModeRaw",
            "geoLayerModeBoth",
            "geoRawLevelSelect",
            "geoRawOpacitySlider",
            "geoExtractedValuesList",
            "geoMapLegendCard",
            "geoLegendGradientBar",
            "modalCatchmentSelect",
        ]
        for elem_id in required_ids:
            self.assertRegex(
                html,
                rf'id=["\']{re.escape(elem_id)}["\']',
                f"Missing #{elem_id} in Geographical Features tab",
            )

        self.assertIn("geoFeaturesMapSlot.prepend(mapEl)", html)
        self.assertIn("initGeoFeaturesMapLayers()", html)
        self.assertIn("syncDefaultUnextractedSelection()", html)
        self.assertIn("/api/attributes/map-layer", html)


if __name__ == "__main__":
    unittest.main()
