"""Unit and integration tests for Earthkit Hydro Windy-style Weather Engine."""

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
import unittest
import urllib.request

# Ensure workspace root path is resolvable
_ws_root = str(Path(__file__).resolve().parents[6])
if _ws_root not in sys.path:
  sys.path.insert(0, _ws_root)

from frontend.server import EarthkitHydroHandler
from frontend.weather_engine import (
    _rate_file_steps,
    generate_raster_tile,
    get_catchment_weather_summary,
    get_weather_probe,
    get_wind_vectors,
    SUPPORTED_MODELS,
    SUPPORTED_VARIABLES,
)


class WeatherEngineUnitTest(unittest.TestCase):
  """Tests core computational weather engine functions."""

  def test_supported_models_and_variables(self):
    for model_key in (
        "ecmwf_hres",
        "ecmwf_ifs",
        "ecmwf_aifs",
        "noaa_gfs",
        "noaa_gefs",
        "noaa_hrrr",
        "nasa_imerg",
        "noaa_cpc",
    ):
      self.assertIn(model_key, SUPPORTED_MODELS)
    self.assertNotIn("graphcast", SUPPORTED_MODELS)

    self.assertIn("precipitation", SUPPORTED_VARIABLES)
    self.assertIn("temperature", SUPPORTED_VARIABLES)
    self.assertIn("wind", SUPPORTED_VARIABLES)
    self.assertIn("pressure", SUPPORTED_VARIABLES)

  def test_generate_raster_tiles_png(self):
    # Test precipitation tile at zoom 2
    precip_tile = generate_raster_tile("ecmwf_ifs", "precipitation", 0, 2, 1, 1)
    self.assertTrue(precip_tile.startswith(b"\x89PNG\r\n\x1a\n"))
    self.assertGreater(len(precip_tile), 500)

    # Test temperature tile at zoom 3
    temp_tile = generate_raster_tile("ecmwf_ifs", "temperature", 6, 3, 2, 2)
    self.assertTrue(temp_tile.startswith(b"\x89PNG\r\n\x1a\n"))
    self.assertGreater(len(temp_tile), 1000)

    # Test pressure isobar tile
    press_tile = generate_raster_tile("ecmwf_ifs", "pressure", 12, 2, 1, 1)
    self.assertTrue(press_tile.startswith(b"\x89PNG\r\n\x1a\n"))

  def test_get_wind_vectors(self):
    vectors = get_wind_vectors("ecmwf_ifs", step_idx=2, subsample=2)
    self.assertIn("header", vectors)
    self.assertIn("u", vectors)
    self.assertIn("v", vectors)

    header = vectors["header"]
    self.assertEqual(header["model"], "ecmwf_ifs")
    self.assertEqual(header["step_hours"], 6)
    self.assertGreater(header["nx"], 0)
    self.assertGreater(header["ny"], 0)
    self.assertEqual(len(vectors["u"]), header["nx"] * header["ny"])
    self.assertEqual(len(vectors["v"]), header["nx"] * header["ny"])

  def test_get_weather_probe(self):
    probe = get_weather_probe(40.42, -86.92)  # West Lafayette / Wabash River
    self.assertEqual(probe["latitude"], 40.42)
    self.assertEqual(probe["longitude"], -86.92)
    self.assertEqual(len(probe["lead_hours"]), 81)
    self.assertIn("models", probe)
    self.assertIn("ecmwf_hres", probe["models"])
    self.assertIn("ecmwf_ifs", probe["models"])
    self.assertIn("ecmwf_aifs", probe["models"])
    self.assertIn("nasa_imerg", probe["models"])
    self.assertIn("noaa_cpc", probe["models"])
    self.assertNotIn("graphcast", probe["models"])

    ifs = probe["models"]["ecmwf_ifs"]
    self.assertEqual(len(ifs["precip_rate_mmh"]), 81)
    self.assertEqual(len(ifs["accum_precip_mm"]), 81)
    self.assertEqual(len(ifs["temp_c"]), 81)
    self.assertEqual(len(ifs["wind_speed_mps"]), 81)
    self.assertEqual(len(ifs["pressure_hpa"]), 81)

  def test_get_catchment_weather_summary(self):
    sample_basin = {
        "id": "test_basin",
        "properties": {
            "catchment_id": "test_basin",
            "area_km2": 4500.0,
            "outlet_latitude": 40.5,
            "outlet_longitude": -86.5,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-87.0, 40.0],
                [-86.0, 40.0],
                [-86.0, 41.0],
                [-87.0, 41.0],
                [-87.0, 40.0],
            ]],
        },
    }
    summary = get_catchment_weather_summary(sample_basin, step_idx=4)
    self.assertEqual(summary["catchment_id"], "test_basin")
    self.assertEqual(summary["area_km2"], 4500.0)
    self.assertEqual(summary["step_hours"], 12)
    self.assertGreaterEqual(summary["basin_mean_precip_mmh"], 0.0)
    self.assertGreater(summary["basin_accumulated_10d_mm"], 0.0)
    self.assertIn("centroid", summary)

  def test_rain_rate_steps_cover_the_viewer_interval(self):
    # GFS-like: hourly to +120 h, then 3-hourly. Plane i ends at lead_hours[i].
    hourly = {"lead_hours": list(range(121)) + [123, 126], "archived_run": True}
    self.assertEqual(_rate_file_steps(hourly, 51), [49, 50, 51])
    self.assertEqual(_rate_file_steps(hourly, 0), [1])
    self.assertEqual(_rate_file_steps(hourly, 123), [121])
    self.assertEqual(_rate_file_steps(hourly, 129), [])
    # AIFS-like: 6-hourly, so a 3-hourly step uses the interval covering it.
    six_hourly = {"lead_hours": [0, 6, 12, 18], "archived_run": True}
    self.assertEqual(_rate_file_steps(six_hourly, 0), [1])
    self.assertEqual(_rate_file_steps(six_hourly, 3), [1])
    self.assertEqual(_rate_file_steps(six_hourly, 9), [2])
    self.assertEqual(_rate_file_steps(six_hourly, 21), [])


class WeatherServerIntegrationTest(unittest.TestCase):
  """Integration test testing HTTP API endpoints over live loopback server."""

  @classmethod
  def setUpClass(cls):
    cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EarthkitHydroHandler)
    cls.port = cls.server.server_port
    cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.thread.start()
    time.sleep(0.1)

    # Save a valid polygon watershed so the server doesn't fallback to a Point
    from frontend.profile_manager import get_profile_manager
    sample_basin = {
        "id": "global_default",
        "properties": {
            "catchment_id": "global_default",
            "area_km2": 4500.0,
            "outlet_latitude": 40.5,
            "outlet_longitude": -86.5,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-87.0, 40.0],
                [-86.0, 40.0],
                [-86.0, 41.0],
                [-87.0, 41.0],
                [-87.0, 40.0],
            ]],
        },
    }
    get_profile_manager().save_watersheds([sample_basin], "guest")

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def _get(self, path: str):
    url = f"http://127.0.0.1:{self.port}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
      data = resp.read()
      content_type = resp.headers.get("Content-Type", "")
      status = resp.status
      if "application/json" in content_type:
        return status, json.loads(data.decode("utf-8"))
      return status, data

  def test_api_weather_models(self):
    status, data = self._get("/api/weather/models")
    self.assertEqual(status, 200)
    self.assertIn("models", data)
    self.assertIn("variables", data)
    self.assertGreaterEqual(len(data["models"]), 4)

  def test_api_weather_tiles_png(self):
    status, png_bytes = self._get(
        "/api/weather/tiles/ecmwf_ifs/precipitation/0/2/1/1.png"
    )
    self.assertEqual(status, 200)
    self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

  def test_api_weather_wind_vectors(self):
    status, data = self._get("/api/weather/wind-vectors?model=ecmwf_ifs&step=0&subsample=4")
    self.assertEqual(status, 200)
    self.assertIn("header", data)
    self.assertIn("u", data)
    self.assertIn("v", data)

  def test_api_weather_probe(self):
    status, data = self._get("/api/weather/probe?lat=45.0&lon=5.0")
    self.assertEqual(status, 200)
    self.assertEqual(data["latitude"], 45.0)
    self.assertEqual(data["longitude"], 5.0)
    self.assertIn("models", data)

  def test_api_weather_catchment_summary(self):
    status, data = self._get("/api/weather/catchment-summary?step=0")
    self.assertEqual(status, 200)
    self.assertIn("basin_mean_precip_mmh", data)
    self.assertIn("basin_accumulated_10d_mm", data)

  def test_api_weather_radar_times(self):
    status, data = self._get("/api/weather/radar-times")
    self.assertEqual(status, 200)
    self.assertIn("radar", data)


if __name__ == "__main__":
  unittest.main()
