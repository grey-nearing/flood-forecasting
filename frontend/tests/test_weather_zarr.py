"""Unit tests for multi-basin appendable historical training Zarr archives, arbitrary polygon ingestion, and real-time forecast stores."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from shapely.geometry import Polygon, mapping
import xarray as xr

from frontend.delineator import HydroDelineator
from frontend.forecast_zarr import ForecastZarrExtractor
from frontend.historical_zarr import HistoricalZarrExtractor


class WeatherZarrTest(unittest.TestCase):

  def setUp(self):
    os.environ["OPENHYDRONET_OFFLINE_TESTS"] = "1"
    self.temp_dir = Path(tempfile.mkdtemp())
    self.delineator = HydroDelineator(dataset_id="hydroatlas")
    self.catchment1 = self.delineator.delineate_catchment(
        lat=40.75, lon=-86.07
    )  # Wabash at Logansport
    self.catchment2 = self.delineator.delineate_catchment(
        lat=39.46, lon=-87.41
    )  # Wabash at Terre Haute

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_multi_basin_master_zarr_append(self):
    """Tests that multiple basin requests cleanly append into a single Master Zarr store with (basin, date, lead_time) dims."""
    extractor = HistoricalZarrExtractor(output_dir=self.temp_dir)
    master_zarr_path = self.temp_dir / "historical_training_master.zarr"

    # 1. Request for Basin 1
    res1 = extractor.extract_and_archive(
        polygon_input=self.catchment1,
        start_date="2021-01-01",
        end_date="2021-03-31",
        freq="1D",
    )
    self.assertEqual(res1["status"], "success")
    self.assertEqual(res1["total_basins_in_master"], 1)
    self.assertTrue(master_zarr_path.exists())

    # Verify initial Master Zarr store
    ds1 = xr.open_zarr(str(master_zarr_path), decode_timedelta=False)
    self.assertIn("basin", ds1.dims)
    self.assertIn("date", ds1.dims)
    self.assertIn("lead_time", ds1.dims)
    self.assertEqual(len(ds1.basin), 1)
    self.assertEqual(ds1.total_precipitation.shape, (1, 90, 1))
    self.assertIn("basin_id", ds1.coords)
    self.assertIn("issue_time", ds1.coords)
    self.assertEqual(ds1.lead_time.attrs.get("units"), "days")

    # 2. Request for Basin 2 (Should recognize existing store and append)
    res2 = extractor.extract_and_archive(
        polygon_input=self.catchment2,
        start_date="2021-01-01",
        end_date="2021-03-31",
        freq="1D",
    )
    self.assertEqual(res2["status"], "success")
    self.assertEqual(res2["total_basins_in_master"], 2)

    # Verify updated Master Zarr store has both basins
    ds2 = xr.open_zarr(str(master_zarr_path), decode_timedelta=False)
    self.assertEqual(len(ds2.basin), 2)
    self.assertEqual(ds2.total_precipitation.shape, (2, 90, 1))
    self.assertIn(res1["basin_id"], ds2.basin.values)
    self.assertIn(res2["basin_id"], ds2.basin.values)
    self.assertIn("area_km2", ds2)
    self.assertIn("latitude", ds2)
    self.assertIn("longitude", ds2)

  def test_arbitrary_polygon_weather_extraction(self):
    """Tests that weather extraction works for any arbitrary custom user polygon."""
    extractor = HistoricalZarrExtractor(output_dir=self.temp_dir)

    # Create an arbitrary polygon (e.g. Custom experimental watershed)
    custom_poly = Polygon([
        (-88.5, 41.0),
        (-88.0, 41.5),
        (-87.5, 41.2),
        (-87.8, 40.8),
        (-88.5, 41.0),
    ])
    custom_feature = {
        "type": "Feature",
        "geometry": mapping(custom_poly),
        "properties": {
            "catchment_id": "CUSTOM_USER_WATERSHED_001",
            "area_km2": 4250.0,
            "dataset_name": "Uploaded Shapefile",
        },
    }

    res = extractor.extract_and_archive(
        polygon_input=custom_feature,
        start_date="2021-01-01",
        end_date="2021-01-31",
    )
    self.assertEqual(res["status"], "success")
    self.assertEqual(res["basin_id"], "CUSTOM_USER_WATERSHED_001")

    # Verify Master Zarr includes custom basin with 3D daily aggregated array
    master_ds = xr.open_zarr(res["master_zarr_path"], decode_timedelta=False)
    self.assertIn("CUSTOM_USER_WATERSHED_001", master_ds.basin.values)
    self.assertEqual(master_ds.total_precipitation.shape, (1, 31, 1))
    self.assertIn("basin_id", master_ds.coords)
    self.assertIn("issue_time", master_ds.coords)

  def test_forecast_zarr_generation(self):
    extractor = ForecastZarrExtractor(output_dir=self.temp_dir)
    res = extractor.fetch_and_archive(
        catchment_feature=self.catchment1,
        model="ifs",
        horizon_hours=120,
        step_interval_hours=6,
    )

    self.assertEqual(res["status"], "success")
    self.assertGreater(res["size_kb"], 0)
    self.assertTrue(Path(res["zarr_path"]).exists())

    # Validate Zarr store format matches flood-forecasting PR 271 Multimet
    ds = xr.open_zarr(res["zarr_path"], decode_timedelta=False)
    self.assertIn("total_precipitation", ds)
    self.assertIn("temperature_2m", ds)
    self.assertIn("2m_temperature", ds)
    self.assertIn("basin", ds.dims)
    self.assertIn("date", ds.dims)
    self.assertIn("lead_time", ds.dims)
    self.assertEqual(len(ds["lead_time"]), 5)  # 120h = 5 daily lead times
    self.assertEqual(ds.total_precipitation.shape, (1, 1, 5))
    self.assertEqual(ds.temperature_2m.shape, (1, 1, 5))
    self.assertEqual(ds.lead_time.attrs.get("units"), "days")


if __name__ == "__main__":
  unittest.main()
