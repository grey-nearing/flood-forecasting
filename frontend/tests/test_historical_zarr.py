"""Unit tests for historical weather Zarr extraction and multi-source registry."""

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import shapely.geometry
Polygon = shapely.geometry.Polygon
import xarray as xr

from frontend.historical_zarr import HistoricalZarrExtractor
from frontend.weather_sources import WEATHER_SOURCES, get_weather_source, list_weather_sources


class HistoricalZarrExtractorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["UNITTEST_ON_FORGE"] = "1"
    self.temp_dir = tempfile.TemporaryDirectory()
    self.output_dir = Path(self.temp_dir.name)
    self.extractor = HistoricalZarrExtractor(output_dir=self.output_dir)

    # Sample triangle polygon representing a watershed
    self.sample_poly = Polygon([
        (-87.5, 40.0),
        (-87.0, 40.5),
        (-87.0, 40.0),
        (-87.5, 40.0),
    ])
    self.sample_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-87.5, 40.0],
                [-87.0, 40.5],
                [-87.0, 40.0],
                [-87.5, 40.0],
            ]],
        },
        "properties": {
            "catchment_id": "test_basin_01",
            "name": "Test Watershed",
            "area_km2": 450.0,
        },
    }

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def test_weather_sources_registry(self):
    """Verifies all required weather datasets are registered with dynamic bounds."""
    sources = list_weather_sources()
    source_ids = [s["id"] for s in sources]
    self.assertIn("cpc", source_ids)
    self.assertIn("imerg", source_ids)
    self.assertIn("era5", source_ids)
    self.assertIn("ifs", source_ids)
    self.assertIn("graphcast", source_ids)

    cpc = get_weather_source("cpc")
    self.assertEqual(cpc.available_start, "1979-01-01")
    self.assertIn("cpc_precipitation", cpc.default_variables)

    imerg = get_weather_source("imerg")
    self.assertEqual(imerg.available_start, "2000-06-01")
    self.assertIn("imerg_precipitation", imerg.default_variables)

    ifs = get_weather_source("ifs")
    self.assertIn("hres_total_precipitation", ifs.default_variables)
    self.assertIn("hres_temperature_2m", ifs.default_variables)

    gc = get_weather_source("graphcast")
    self.assertIn("graphcast_total_precipitation", gc.default_variables)

  def test_single_basin_historical_extraction(self):
    """Verifies extracting weather data for a single polygon creates compliant PR 271 Zarr chunked by basin."""
    result = self.extractor.extract_and_archive(
        polygon_input=self.sample_feature,
        basin_id="test_basin_01",
        weather_source="cpc",
        start_date="2020-01-01",
        end_date="2020-01-31",
        freq="1D",
    )

    self.assertEqual(result["status"], "success")
    self.assertEqual(result["basin_id"], "test_basin_01")
    self.assertEqual(result["n_timesteps"], 31)
    self.assertTrue(self.extractor.master_zarr_path.exists())

    # Verify no redundant per-basin subfolder exists
    self.assertFalse((self.output_dir / "test_basin_01").exists())

    # Verify Zarr Dataset schema & chunking
    ds = xr.open_zarr(str(self.extractor.master_zarr_path))
    self.assertIn("basin", ds.dims)
    self.assertIn("date", ds.dims)
    self.assertIn("lead_time", ds.dims)
    self.assertEqual(list(ds.basin.values), ["test_basin_01"])
    self.assertEqual(len(ds.date), 31)
    self.assertEqual(len(ds.lead_time), 1)

    # Verify basin chunking
    self.assertEqual(ds["cpc_precipitation"].encoding.get("chunks"), (1, 31, 1))
    self.assertEqual(ds["latitude"].encoding.get("chunks"), (1,))

    self.assertIn("cpc_precipitation", ds.data_vars)
    self.assertIn("basin_area", ds.data_vars)
    self.assertIn("latitude", ds.data_vars)
    self.assertIn("longitude", ds.data_vars)

  def test_multi_basin_batch_append(self):
    """Verifies appending multiple basins creates a multi-basin master Zarr archive chunked by basin."""
    feat2 = {
        "type": "Feature",
        "geometry": self.sample_feature["geometry"],
        "properties": {
            "catchment_id": "test_basin_02",
            "name": "Second Watershed",
            "area_km2": 720.0,
        },
    }

    batch_res = self.extractor.extract_and_archive_batch(
        features=[self.sample_feature, feat2],
        weather_source="imerg",
        start_date="2021-05-01",
        end_date="2021-05-15",
        freq="1D",
    )

    self.assertEqual(batch_res["status"], "success")
    self.assertEqual(batch_res["basins_extracted_count"], 2)
    self.assertEqual(batch_res["total_basins_in_master"], 2)

    # Verify no redundant per-basin subfolders exist
    self.assertFalse((self.output_dir / "test_basin_01").exists())
    self.assertFalse((self.output_dir / "test_basin_02").exists())

    ds = xr.open_zarr(str(self.extractor.master_zarr_path))
    self.assertEqual(len(ds.basin), 2)
    self.assertIn("test_basin_01", ds.basin.values)
    self.assertIn("test_basin_02", ds.basin.values)
    self.assertEqual(len(ds.date), 15)
    self.assertIn("imerg_precipitation", ds.data_vars)

    # Verify basin chunking
    self.assertEqual(ds["imerg_precipitation"].encoding.get("chunks"), (1, 15, 1))
    self.assertEqual(ds["latitude"].encoding.get("chunks"), (1,))


if __name__ == "__main__":
  unittest.main()
