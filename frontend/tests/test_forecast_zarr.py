"""Unit tests for real-time weather forecast extraction from dynamical.org for Open Hydro Net inference."""

from pathlib import Path
import shutil
import tempfile
import unittest
from shapely.geometry import Polygon, mapping
import xarray as xr

from frontend.forecast_zarr import (
    ForecastZarrExtractor,
    FORECAST_FEATURE_SPECS,
)
from frontend.profile_manager import ProfileManager


class ForecastZarrTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = Path(tempfile.mkdtemp())
    self.profile_mgr = ProfileManager(profiles_dir=self.temp_dir)

    # Test Catchment 1 (Logansport, IN)
    self.poly1 = Polygon([
        (-86.4, 40.7),
        (-86.1, 40.7),
        (-86.1, 40.9),
        (-86.4, 40.9),
        (-86.4, 40.7),
    ])
    self.feat1 = {
        "type": "Feature",
        "geometry": mapping(self.poly1),
        "properties": {
            "catchment_id": "WABASH_LOGANSPORT",
            "name": "Wabash River at Logansport, IN",
            "area_km2": 3120.0,
            "outlet": {"latitude": 40.75, "longitude": -86.37},
        },
    }

    # Test Catchment 2 (Terre Haute, IN)
    self.poly2 = Polygon([
        (-87.5, 39.4),
        (-87.2, 39.4),
        (-87.2, 39.6),
        (-87.5, 39.6),
        (-87.5, 39.4),
    ])
    self.feat2 = {
        "type": "Feature",
        "geometry": mapping(self.poly2),
        "properties": {
            "catchment_id": "WABASH_TERRE_HAUTE",
            "name": "Wabash River at Terre Haute, IN",
            "area_km2": 7850.0,
            "outlet": {"latitude": 39.46, "longitude": -87.41},
        },
    }

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_exact_variable_names_match_config(self):
    """Tests that all 7 variables defined in floodhub-settings-config.yml are present in FORECAST_FEATURE_SPECS."""
    required_features = {
        # HRES
        "hres_surface_net_solar_radiation",
        "hres_surface_net_thermal_radiation",
        "hres_surface_pressure",
        "hres_temperature_2m",
        "hres_total_precipitation",
        # GraphCast (proxied by AIFS)
        "graphcast_temperature_2m",
        "graphcast_total_precipitation",
    }
    extracted_features = set(FORECAST_FEATURE_SPECS.keys())
    self.assertTrue(
        required_features.issubset(extracted_features),
        f"Missing required features: {required_features - extracted_features}",
    )

  def test_single_basin_forecast_extraction(self):
    """Tests single-catchment forecast extraction and unified Zarr store creation chunked along basin."""
    forecast_dir = self.profile_mgr.get_forecast_dir("user_alice")
    extractor = ForecastZarrExtractor(output_dir=forecast_dir)

    res = extractor.fetch_and_archive(
        catchment_feature=self.feat1,
        horizon_days=10,
    )

    self.assertEqual(res["status"], "success")
    self.assertEqual(res["catchment_id"], "WABASH_LOGANSPORT")
    self.assertTrue(Path(res["zarr_path"]).exists())
    self.assertEqual(Path(res["zarr_path"]).name, "forecast_latest.zarr")
    self.assertEqual(Path(res["zarr_path"]).parent.name, "forecast")
    self.assertGreater(res["size_kb"], 0)

    # Verify no redundant per-basin subfolder exists
    self.assertFalse((forecast_dir / "WABASH_LOGANSPORT").exists())

    # Verify single catchment Zarr store schema
    ds = xr.open_zarr(res["zarr_path"], decode_timedelta=False)
    self.assertIn("basin", ds.dims)
    self.assertIn("date", ds.dims)
    self.assertIn("lead_time", ds.dims)
    self.assertEqual(len(ds.lead_time), 10)
    self.assertEqual(ds.lead_time.attrs.get("units"), "days")

    # Verify chunking along basin dimension (chunk size 1 along basin)
    self.assertEqual(ds["hres_total_precipitation"].encoding.get("chunks"), (1, 1, 10))
    self.assertEqual(ds["latitude"].encoding.get("chunks"), (1,))

    # Check all 7 dynamic weather forcings
    for feat in FORECAST_FEATURE_SPECS:
      self.assertIn(feat, ds.data_vars)
      self.assertEqual(ds[feat].shape, (1, 1, 10))

    # Check metadata variables
    self.assertIn("area_km2", ds)
    self.assertIn("latitude", ds)
    self.assertIn("longitude", ds)
    self.assertIn("total_precipitation", ds)
    self.assertIn("temperature_2m", ds)

    # Check meteogram payload
    meteogram = res.get("meteogram", {})
    self.assertIn("hres", meteogram)
    self.assertIn("graphcast", meteogram)
    self.assertEqual(len(meteogram["hres"]["precipitation_mm"]), 10)
    self.assertEqual(len(meteogram["graphcast"]["precipitation_mm"]), 10)

  def test_batch_multi_basin_forecast_extraction(self):
    """Tests batch forecast extraction across multiple polygons into unified store in forecast/."""
    forecast_dir = self.profile_mgr.get_forecast_dir("user_bob")
    extractor = ForecastZarrExtractor(output_dir=forecast_dir)

    res = extractor.fetch_and_archive_batch(
        features=[self.feat1, self.feat2],
        horizon_days=15,
    )

    self.assertEqual(res["status"], "success")
    self.assertEqual(res["count"], 2)
    self.assertEqual(res["total_basins_in_master"], 2)
    self.assertTrue(Path(res["master_zarr_path"]).exists())
    self.assertEqual(Path(res["master_zarr_path"]).name, "forecast_latest.zarr")
    self.assertEqual(Path(res["master_zarr_path"]).parent.name, "forecast")

    # Verify no redundant per-basin subfolders exist
    self.assertFalse((forecast_dir / "WABASH_LOGANSPORT").exists())
    self.assertFalse((forecast_dir / "WABASH_TERRE_HAUTE").exists())

    # Verify unified forecast Zarr store
    master_ds = xr.open_zarr(res["master_zarr_path"], decode_timedelta=False)
    self.assertEqual(len(master_ds.basin), 2)
    self.assertEqual(len(master_ds.lead_time), 15)
    self.assertIn("WABASH_LOGANSPORT", master_ds.basin.values)
    self.assertIn("WABASH_TERRE_HAUTE", master_ds.basin.values)

    # Verify chunking along basin dimension
    self.assertEqual(master_ds["hres_total_precipitation"].encoding.get("chunks"), (1, 1, 15))
    self.assertEqual(master_ds["latitude"].encoding.get("chunks"), (1,))

    for feat in FORECAST_FEATURE_SPECS:
      self.assertIn(feat, master_ds.data_vars)
      self.assertEqual(master_ds[feat].shape, (2, 1, 15))

  def test_latest_issue_info_query(self):
    """Tests querying dynamical.org metadata and latest issue time."""
    extractor = ForecastZarrExtractor(output_dir=self.temp_dir)
    info = extractor.get_latest_issue_info()
    self.assertIn(info["status"], ["online", "fallback"])
    self.assertIn("latest_issue_date", info)
    self.assertIn("features", info)
    self.assertEqual(len(info["features"]), 7)


if __name__ == "__main__":
  unittest.main()
