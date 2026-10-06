"""Unit tests for CNS Zarr importer."""

from pathlib import Path
import tempfile
from unittest import mock
try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest
import numpy as np
import pandas as pd
import xarray as xr

from frontend.cns_importer import CNSZarrImporter


class CNSZarrImporterTest(absltest.TestCase):

  def test_invalid_cns_path_raises(self):
    """Verifies that invalid or non-CNS paths raise ValueError."""
    with tempfile.TemporaryDirectory() as td:
      importer = CNSZarrImporter(output_dir=Path(td))
      with self.assertRaises(ValueError):
        importer.import_zarr_from_cns("")
      with self.assertRaises(ValueError):
        importer.import_zarr_from_cns("/tmp/local_path.zarr")

  @mock.patch("subprocess.run")
  def test_import_zarr_from_cns_mock(self, mock_run):
    """Verifies successful import workflow with synthetic Zarr store."""
    # Mock subprocess.run for fileutil check and copy
    mock_run.return_value = mock.MagicMock(returncode=0, stderr="")

    with tempfile.TemporaryDirectory() as td:
      out_dir = Path(td)
      importer = CNSZarrImporter(output_dir=out_dir)

      # Create a synthetic Zarr store at the destination to simulate downloaded store
      target_zarr = out_dir / "historical_training_master.zarr"
      dates = pd.date_range("2020-01-01", periods=10, freq="1D")
      basins = ["us_03338780", "us_03339000"]
      lead_times = [0]

      ds = xr.Dataset(
          data_vars={
              "cpc_precipitation": (
                  ["basin", "date", "lead_time"],
                  np.ones((2, 10, 1), dtype=np.float32) * 5.0,
              ),
              "temperature_2m": (
                  ["basin", "date", "lead_time"],
                  np.ones((2, 10, 1), dtype=np.float32) * 15.0,
              ),
          },
          coords={
              "basin": basins,
              "date": dates,
              "lead_time": lead_times,
          },
      )
      ds.to_zarr(str(target_zarr))

      result = importer.import_zarr_from_cns(
          cns_path="/cns/iz-d/home/floods/test_master.zarr",
          dest_filename="historical_training_master.zarr",
          overwrite=False,
      )

      self.assertEqual(result["status"], "completed")
      self.assertEqual(result["total_basins"], 2)
      self.assertListEqual(result["basins"], basins)
      self.assertEqual(result["n_timesteps"], 10)
      self.assertIn("cpc_precipitation", result["variables"])
      self.assertIn("temperature_2m", result["variables"])
      self.assertIn("preview", result)
      self.assertEqual(result["preview"]["sample_basin_id"], "us_03338780")
      self.assertEqual(len(result["preview"]["timestamps"]), 10)


if __name__ == "__main__":
  absltest.main()
