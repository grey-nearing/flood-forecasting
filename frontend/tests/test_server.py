"""Integration tests for Earthkit Hydro server endpoints using standard library urllib."""

import unittest
from unittest.mock import patch
import threading
import time
import json
import urllib.request
import urllib.parse
from http.server import ThreadingHTTPServer

from frontend.server import EarthkitHydroHandler
# Importing the tests package points profiles at a temp folder (never real accounts).
from frontend import tests as _isolated_profiles  # pylint: disable=unused-import


class ServerApiIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EarthkitHydroHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)  # Allow server to initialize

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path: str):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode("utf-8")
            status = resp.status
            try:
                parsed = json.loads(data)
            except Exception:
                parsed = data
            return status, parsed

    def _post(self, path: str, payload: dict):
        url = f"http://127.0.0.1:{self.port}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode("utf-8")
            status = resp.status
            return status, json.loads(data)

    def test_datasets_endpoint(self):
        status, data = self._get("/api/datasets")
        self.assertEqual(status, 200)
        self.assertIn("datasets", data)
        self.assertGreaterEqual(len(data["datasets"]), 2)
        dataset_ids = [d["id"] for d in data["datasets"]]
        self.assertIn("merit-hydro", dataset_ids)
        self.assertIn("hydroatlas", dataset_ids)

    def test_weather_sources_endpoint(self):
        status, data = self._get("/api/weather/sources")
        self.assertEqual(status, 200)
        self.assertIn("sources", data)
        sources = {s["id"]: s for s in data["sources"]}
        self.assertIn("era5", sources)
        self.assertIn("era5-land", sources)
        self.assertIn("chirps", sources)
        
        era5 = sources["era5"]
        self.assertIn(era5["start_date"], ("1940-01-01", "1950-01-01"))
        self.assertTrue(era5["end_date"] > "2024-01-01")
        self.assertGreater(era5["total_years"], 70)

    def test_attributes_schema_and_extract_endpoints(self):
        # 1. Test Schema endpoint
        status, schema_data = self._get("/api/attributes/schema")
        self.assertEqual(status, 200)
        self.assertIn("attributes", schema_data)
        self.assertIn("ele_mt_sav", schema_data["attributes"])
        self.assertIn("cly_pc_sav", schema_data["attributes"])
        self.assertIn("for_pc_sse", schema_data["attributes"])
        # Verify ERA5-Land climate indices in schema
        self.assertIn("p_mean", schema_data["attributes"])
        self.assertIn("pet_mean_ERA5_LAND", schema_data["attributes"])
        self.assertIn("aridity_ERA5_LAND", schema_data["attributes"])
        self.assertIn("frac_snow", schema_data["attributes"])

        # 2. Test Extraction endpoint for a custom polygon
        sample_poly = {
            "type": "Polygon",
            "coordinates": [[
                [-86.5, 40.2],
                [-86.1, 40.2],
                [-86.1, 40.6],
                [-86.5, 40.6],
                [-86.5, 40.2]
            ]]
        }
        extract_status, extract_data = self._post(
            "/api/attributes/extract",
            {
                "catchment_id": "test_basin_attribs_01",
                "polygon": sample_poly
            }
        )
        self.assertEqual(extract_status, 200)
        self.assertIn("summary", extract_data)
        self.assertIn("categories", extract_data)
        self.assertIn("attributes", extract_data)
        self.assertGreater(extract_data["intersected_subbasins_count"], 0)
        self.assertGreater(extract_data["summary"]["elevation_mean_m"], 100)
        self.assertGreater(extract_data["summary"]["annual_precip_mm"], 500)
        # Verify ERA5-Land values
        self.assertGreater(extract_data["attributes"]["p_mean"], 1.0)
        self.assertGreater(extract_data["attributes"]["pet_mean"], 1.0)
        self.assertGreater(extract_data["summary"]["era5_p_mean_mm_day"], 1.0)
        self.assertGreater(extract_data["summary"]["era5_fao_aridity"], 0.1)
        self.assertIn("Topography", extract_data["categories"])
        self.assertIn("Soils", extract_data["categories"])
        self.assertIn("Climate", extract_data["categories"])

    def test_rivers_endpoint(self):
        # Query rivers in Wabash River bounding box
        status, data = self._get("/api/hydro/rivers?dataset=merit-hydro&bbox=-88.5,37.5,-84.5,41.8&zoom=7")
        self.assertEqual(status, 200)
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertIn("features", data)
        self.assertGreater(len(data["features"]), 0)
        first_reach = data["features"][0]
        self.assertIn("properties", first_reach)
        self.assertIn("stream_order", first_reach["properties"])
        self.assertIn("upstream_area_km2", first_reach["properties"])

    def test_full_workflow_delineate_and_extract_weather(self):
        # 1. Delineate Catchment (Wabash River at Logansport, IN)
        delin_status, delin_data = self._post(
            "/api/delineate",
            {
                "dataset": "hydroatlas",
                "latitude": 40.75,
                "longitude": -86.07,
            }
        )
        self.assertEqual(delin_status, 200)
        self.assertEqual(delin_data["type"], "Feature")
        catchment_id = delin_data["properties"]["catchment_id"]
        self.assertTrue(catchment_id.startswith("catchment_hydroatlas_"))

        # 2. Generate Historical Training Zarr Archive
        hist_status, hist_data = self._post(
            "/api/weather/historical",
            {
                "catchment_id": catchment_id,
                "start_date": "2021-01-01",
                "end_date": "2021-01-31",
                "frequency": "1D",
            }
        )
        self.assertEqual(hist_status, 200)
        self.assertEqual(hist_data["status"], "success")
        self.assertIn("preview", hist_data)
        self.assertEqual(len(hist_data["preview"]["timestamps"]), 31)
        self.assertIn("master_zarr_path", hist_data)

        # 3. Fetch Real-Time Forecast Inference Zarr Store
        fcst_status, fcst_data = self._post(
            "/api/weather/forecast",
            {
                "catchment_id": catchment_id,
                "model": "ifs",
                "horizon_hours": 72,
            }
        )
        self.assertEqual(fcst_status, 200)
        self.assertEqual(fcst_data["status"], "success")
        self.assertIn("meteogram", fcst_data)

        # 4. Upload Arbitrary Custom Polygon
        custom_poly_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-88.0, 41.0], [-87.5, 41.0], [-87.5, 41.5], [-88.0, 41.5], [-88.0, 41.0]]]
            },
            "properties": {
                "catchment_id": "CUSTOM_USER_BASIN_TEST_99",
                "name": "Custom Illinois Watershed"
            }
        }
        poly_status, poly_data = self._post("/api/polygons/upload", custom_poly_geojson)
        self.assertEqual(poly_status, 200)
        self.assertEqual(poly_data["type"], "FeatureCollection")
        self.assertEqual(poly_data["features"][0]["properties"]["catchment_id"], "CUSTOM_USER_BASIN_TEST_99")

        # 5. Extract Historical Weather for Custom Polygon (Appends to Master Zarr)
        custom_hist_status, custom_hist_data = self._post(
            "/api/weather/historical",
            {
                "catchment_id": "CUSTOM_USER_BASIN_TEST_99",
                "start_date": "2021-01-01",
                "end_date": "2021-01-31",
            }
        )
        self.assertEqual(custom_hist_status, 200)
        self.assertEqual(custom_hist_data["status"], "success")
        self.assertGreaterEqual(custom_hist_data["total_basins_in_master"], 2)

        # 6. List Archives on Disk
        arch_status, arch_data = self._get("/api/archives")
        self.assertEqual(arch_status, 200)
        archives = arch_data["archives"]
        self.assertGreater(len(archives), 0)

    def test_watershed_registry_management_and_batch_extraction(self):
        # 1. Clear session
        self._post("/api/watersheds/clear", {})

        # 2. Delineate 2 distinct basins
        s1, d1 = self._post("/api/delineate", {"dataset": "hydroatlas", "latitude": 40.75, "longitude": -86.07})
        self.assertEqual(s1, 200)
        id1 = d1["properties"]["catchment_id"]

        s2, d2 = self._post("/api/delineate", {"dataset": "hydroatlas", "latitude": 39.95, "longitude": -86.25})
        self.assertEqual(s2, 200)
        id2 = d2["properties"]["catchment_id"]

        # 3. Upload a 2-basin GeoJSON FeatureCollection
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-87.5, 40.0], [-87.0, 40.0], [-87.0, 40.5], [-87.5, 40.5], [-87.5, 40.0]]]
                    },
                    "properties": {"catchment_id": "UPLOADED_BASIN_A", "name": "Uploaded Basin A"}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-86.0, 39.0], [-85.5, 39.0], [-85.5, 39.5], [-86.0, 39.5], [-86.0, 39.0]]]
                    },
                    "properties": {"catchment_id": "UPLOADED_BASIN_B", "name": "Uploaded Basin B"}
                }
            ]
        }
        s3, d3 = self._post("/api/polygons/upload", fc)
        self.assertEqual(s3, 200)

        # 4. Verify GET /api/watersheds contains all 4 basins
        ws_status, ws_data = self._get("/api/watersheds")
        self.assertEqual(ws_status, 200)
        self.assertEqual(ws_data["count"], 4)
        cids = [f["properties"]["catchment_id"] for f in ws_data["features"]]
        self.assertIn(id1, cids)
        self.assertIn(id2, cids)
        self.assertIn("UPLOADED_BASIN_A", cids)
        self.assertIn("UPLOADED_BASIN_B", cids)

        # 5. Batch Extract Static Attributes for all 4 basins
        attr_status, attr_data = self._post("/api/attributes/extract", {"all_active": True})
        self.assertEqual(attr_status, 200)
        self.assertTrue(attr_data["batch"])
        self.assertEqual(attr_data["count"], 4)

        # 6. Delete 1 individual watershed
        del_status, del_data = self._post("/api/watersheds/delete", {"catchment_id": "UPLOADED_BASIN_A"})
        self.assertEqual(del_status, 200)
        self.assertEqual(del_data["status"], "deleted")
        self.assertEqual(del_data["remaining_count"], 3)

        # 7. Clear all remaining watersheds
        clr_status, clr_data = self._post("/api/watersheds/clear", {})
        self.assertEqual(clr_status, 200)
        self.assertEqual(clr_data["remaining_count"], 0)

    def _get_bytes(self, path: str):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            status = resp.status
            content_type = resp.headers.get("Content-Type")
            return status, content_type, data

    def test_dem_tiles_endpoint(self):
        # 1. Test rendering valid DEM tile 8/65/97 (Indiana)
        status, ctype, data = self._get_bytes("/api/tiles/dem/8/65/97.png")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "image/png")
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(data), 1000)

        # 2. Test rendering ocean / outside tile
        status2, ctype2, data2 = self._get_bytes("/api/tiles/dem/8/0/0.png")
        self.assertEqual(status2, 200)
        self.assertEqual(ctype2, "image/png")
        self.assertTrue(data2.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_serve_frontend(self):
        status, html = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("OpenHydroNet", html)


if __name__ == "__main__":
    unittest.main()
