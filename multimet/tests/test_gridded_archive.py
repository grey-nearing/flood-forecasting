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

"""Unit tests for gridded Zarr archive extraction and strict data-quality invariants."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import xarray as xr

from multimet.timeseries_extractors.config import (
    DEFAULT_STORAGE_PATHS,
    MISSING_FRACTION_VAR,
    PRODUCT_BANDS,
    Product,
)
from multimet.timeseries_extractors.cpc import CPCExtractor
from multimet.timeseries_extractors.era5_land import ERA5LandExtractor
from multimet.utils.geometry import load_basin_geometries
from multimet.timeseries_extractors.gridded_archive import (
    _WARNED_MISSING_VARS,
    GriddedArchiveError,
    extract_from_archive,
    open_gridded_archive,
)
from multimet.timeseries_extractors.hres import HRESExtractor
from multimet.timeseries_extractors.imerg import IMERGExtractor
from multimet.timeseries_extractors.runner import extract_multimet_serial
from multimet.timeseries_extractors.zarr_writer import check_zarr_store_exists
from multimet.utils.zonal import ZonalWeightCalculator, ZonalWeightMatrix

pytestmark = pytest.mark.unit


@pytest.fixture
def basins_gdf() -> gpd.GeoDataFrame:
  path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  return load_basin_geometries(path)


def test_no_hardcoded_gs_bucket_paths_in_config():
  """Verifies that all hardcoded gs:// bucket paths have been stripped from DEFAULT_STORAGE_PATHS."""
  for prod, paths_map in DEFAULT_STORAGE_PATHS.items():
    for key, val in paths_map.items():
      assert not str(val).startswith(("gs://", "gcs://")), (
          f"Found prohibited hardcoded GCS bucket URI in "
          f"DEFAULT_STORAGE_PATHS[{prod}][{key!r}] = {val!r}"
      )


def test_out_of_domain_basin_never_snaps_to_nearest_cell():
  """Verifies that a basin outside the grid domain gets zero weight and NaN, not a distant pixel."""
  # Grid covering North America only: lats [35..45], lons [-95..-80]
  lats = np.linspace(35.0, 45.0, 21, dtype=np.float64)
  lons = np.linspace(-95.0, -80.0, 31, dtype=np.float64)

  in_domain_poly = shapely.geometry.box(-88.0, 39.0, -87.0, 40.0)
  out_of_domain_poly = shapely.geometry.box(10.0, 48.0, 11.0, 49.0)  # Europe

  gdf = gpd.GeoDataFrame(
      {"geometry": [in_domain_poly, out_of_domain_poly]},
      index=["basin_us", "basin_europe"],
      crs="EPSG:4326",
  )

  calc = ZonalWeightCalculator(lats, lons, cell_res_lat=0.5, cell_res_lon=0.5)
  lat_eu, lon_eu, w_eu = calc.compute_weights(
      "basin_europe", out_of_domain_poly
  )
  assert len(w_eu) == 0
  assert len(lat_eu) == 0
  assert len(lon_eu) == 0

  matrix = ZonalWeightMatrix.from_geodataframe(
      gdf, lats, lons, cell_res_lat=0.5, cell_res_lon=0.5
  )
  assert bool(matrix.has_weights[0]) is True
  assert bool(matrix.has_weights[1]) is False

  grid_2d = np.full((len(lats), len(lons)), 25.0, dtype=np.float32)
  vals, missing = matrix.reduce_2d_with_coverage(grid_2d)
  assert np.isclose(vals[0], 25.0)
  assert np.isclose(missing[0], 0.0)
  assert np.isnan(vals[1])
  assert np.isclose(missing[1], 1.0)


def test_partial_coverage_missing_fraction_matches_cookie_cutter():
  """Verifies 80% minimum valid basin coverage threshold and missing_fraction reporting."""
  lats = np.array([40.5], dtype=np.float64)
  lons = np.array([-89.5, -88.5, -87.5, -86.5, -85.5], dtype=np.float64)
  # Basin covering all 5 equal-latitude cells equally (20% weight each)
  poly = shapely.geometry.box(-90.0, 40.0, -85.0, 41.0)
  gdf = gpd.GeoDataFrame(
      {"geometry": [poly]}, index=["coastal_basin"], crs="EPSG:4326"
  )
  matrix = ZonalWeightMatrix.from_geodataframe(
      gdf, lats, lons, cell_res_lat=1.0, cell_res_lon=1.0
  )

  # 4 of 5 cells valid (80% >= 80% threshold): renormalizes valid cells to 10.0, missing = 0.20
  grid_80 = np.array([[10.0, 10.0, 10.0, 10.0, np.nan]], dtype=np.float32)
  vals_80, missing_80 = matrix.reduce_2d_with_coverage(grid_80)
  assert np.isclose(vals_80[0], 10.0, atol=1e-4)
  assert np.isclose(float(missing_80[0]), 0.20, atol=1e-4)

  # 2 of 5 cells valid (40% < 80% threshold): value must be NaN, missing = 0.60
  grid_40 = np.array([[10.0, 10.0, np.nan, np.nan, np.nan]], dtype=np.float32)
  vals_40, missing_40 = matrix.reduce_2d_with_coverage(grid_40)
  assert np.isnan(vals_40[0])
  assert np.isclose(float(missing_40[0]), 0.60, atol=1e-4)


def test_check_zarr_store_exists_does_not_mask_storage_errors(monkeypatch):
  """Verifies that storage OSErrors propagate directly and are never masked as False."""
  import fsspec

  class ValidFS:
    def exists(self, path: str) -> bool:
      return path.endswith("zarr.json")

  monkeypatch.setattr(
      fsspec.core,
      "url_to_fs",
      lambda url: (ValidFS(), url.removeprefix("gs://")),
  )
  assert check_zarr_store_exists("gs://open-multimet/test.zarr") is True

  # Storage 403 must raise OSError directly, NEVER return False
  class AlwaysFailingFS:
    def exists(self, path: str) -> bool:
      raise OSError("403 Forbidden: persistent permission failure")

  monkeypatch.setattr(
      fsspec.core,
      "url_to_fs",
      lambda url: (AlwaysFailingFS(), url.removeprefix("gs://")),
  )
  with pytest.raises(OSError, match="403 Forbidden"):
    check_zarr_store_exists("gs://open-multimet/test.zarr")


def test_multi_band_missing_fraction_uses_maximum_across_bands(
    tmp_path, basins_gdf
):
  """Verifies that missing_fraction is aggregated via np.maximum across bands."""
  lats = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
  lons = np.linspace(-179.75, 179.75, 720, dtype=np.float32)
  times = pd.date_range("2022-05-01", "2022-05-02", freq="D")

  precip = np.full((len(times), len(lats), len(lons)), 12.5, dtype=np.float32)
  # Second band is completely NaN on day 1
  stations = np.full((len(times), len(lats), len(lons)), 5.0, dtype=np.float32)
  stations[0, :, :] = np.nan

  ds_archive = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["time", "latitude", "longitude"], precip),
          "cpc_num_stations": (["time", "latitude", "longitude"], stations),
      },
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  )
  store_path = tmp_path / "cpc_missing_band.zarr"
  ds_archive.to_zarr(store_path)

  ext = CPCExtractor(data_dir=str(store_path), source="archive")
  ds_out = ext.extract_for_basins(
      basins_gdf, start_date="2022-05-01", end_date="2022-05-02"
  )
  # Day 0 has stations=NaN (missing=1.0) and precip=12.5 (missing=0.0) -> max is 1.0
  assert np.allclose(ds_out["cpc_missing_fraction"].values[:, 0], 1.0)
  assert np.allclose(ds_out["cpc_missing_fraction"].values[:, 1], 0.0)


def test_cpc_archive_extraction_and_deduplicated_warning(
    tmp_path, basins_gdf, caplog
):
  """Tests CPC gridded archive extraction and single-warning behavior when cpc_num_stations is absent."""
  _WARNED_MISSING_VARS.clear()
  lats = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
  lons = np.linspace(-179.75, 179.75, 720, dtype=np.float32)
  times = pd.date_range("2022-05-01", "2022-05-03", freq="D")

  precip = np.full((len(times), len(lats), len(lons)), 12.5, dtype=np.float32)
  ds_archive = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["time", "latitude", "longitude"], precip),
      },
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  )
  store_path = tmp_path / "cpc_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = CPCExtractor(data_dir=str(store_path), source="archive")
  with caplog.at_level(logging.WARNING):
    ds_out1 = ext.extract_for_basins(
        basins_gdf, start_date="2022-05-01", end_date="2022-05-03"
    )
    ds_out2 = ext.extract_for_basins(
        basins_gdf, start_date="2022-05-01", end_date="2022-05-03"
    )

  assert "cpc_precipitation" in ds_out1.data_vars
  assert "cpc_num_stations" in ds_out1.data_vars
  assert "cpc_missing_fraction" in ds_out1.data_vars
  assert np.allclose(ds_out1["cpc_precipitation"].values, 12.5, atol=1e-3)
  assert np.all(np.isnan(ds_out1["cpc_num_stations"].values))
  assert np.allclose(ds_out1["cpc_missing_fraction"].values, 0.0)

  # Warning for missing cpc_num_stations must be logged exactly ONCE across both calls
  station_warnings = [
      rec
      for rec in caplog.records
      if "cpc_num_stations" in rec.getMessage()
  ]
  assert len(station_warnings) == 1


def test_era5_land_archive_extraction_with_min_max_temperature(
    tmp_path, basins_gdf
):
  """Tests ERA5-Land archive extraction on descending latitudes with all 17 bands."""
  lats = np.linspace(90.0, -90.0, 181, dtype=np.float32)  # descending
  lons = np.linspace(-180.0, 179.0, 360, dtype=np.float32)
  times = pd.date_range("2021-07-01", "2021-07-02", freq="D")

  data_vars = {}
  for idx, band in enumerate(PRODUCT_BANDS[Product.ERA5_LAND]):
    val = 10.0 + float(idx)
    if band == "era5land_temperature_2m_min":
      val = 14.0
    elif band == "era5land_temperature_2m":
      val = 20.0
    elif band == "era5land_temperature_2m_max":
      val = 26.0
    data_vars[band] = (
        ["time", "latitude", "longitude"],
        np.full((len(times), len(lats), len(lons)), val, dtype=np.float32),
    )

  ds_archive = xr.Dataset(
      data_vars=data_vars,
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  )
  store_path = tmp_path / "era5_land_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = ERA5LandExtractor(data_dir=str(store_path), source="archive")
  ds_out = ext.extract_for_basins(
      basins_gdf, start_date="2021-07-01", end_date="2021-07-02"
  )
  for band in PRODUCT_BANDS[Product.ERA5_LAND]:
    assert band in ds_out.data_vars
  assert MISSING_FRACTION_VAR[Product.ERA5_LAND] in ds_out.data_vars
  assert np.allclose(ds_out["era5land_temperature_2m_min"].values, 14.0, atol=1e-3)
  assert np.allclose(ds_out["era5land_temperature_2m"].values, 20.0, atol=1e-3)
  assert np.allclose(ds_out["era5land_temperature_2m_max"].values, 26.0, atol=1e-3)
  assert np.allclose(ds_out["era5land_missing_fraction"].values, 0.0)


def test_hres_archive_extraction_unit_conversion_and_lon_wrapping(
    tmp_path, basins_gdf
):
  """Tests HRES archive extraction from [0, 360) longitudes and native ECMWF units."""
  lats = np.linspace(-90.0, 90.0, 181, dtype=np.float32)
  lons = np.linspace(0.0, 359.0, 360, dtype=np.float32)  # [0, 360) convention
  times = pd.date_range("2023-03-01", "2023-03-02", freq="D")
  leads = np.arange(1, 11, dtype=np.int32)
  shape = (len(times), len(leads), len(lats), len(lons))

  ds_archive = xr.Dataset(
      data_vars={
          "temperature_2m": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 293.15, dtype=np.float32),  # 20.0 degC
          ),
          "surface_pressure": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 101325.0, dtype=np.float32),  # 101.325 kPa
          ),
          "total_precipitation": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 0.015, dtype=np.float32),  # 15.0 mm
          ),
          "surface_net_solar_radiation": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, 86400.0 * 200.0, dtype=np.float32),  # 200.0 W/m^2
          ),
          "surface_net_thermal_radiation": (
              ["time", "lead_time", "latitude", "longitude"],
              np.full(shape, -86400.0 * 50.0, dtype=np.float32),  # -50.0 W/m^2
          ),
      },
      coords={
          "time": times.values,
          "lead_time": leads,
          "latitude": lats,
          "longitude": lons,
      },
  )
  store_path = tmp_path / "hres_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = HRESExtractor(data_dir=str(store_path), source="archive")
  ds_out = ext.extract_for_basins(
      basins_gdf, start_date="2023-03-01", end_date="2023-03-02"
  )
  assert ds_out["hres_temperature_2m"].shape == (len(basins_gdf), 2, 10)
  assert np.issubdtype(ds_out["lead_time"].dtype, np.timedelta64)
  assert np.allclose(ds_out["hres_temperature_2m"].values, 20.0, atol=1e-2)
  assert np.allclose(ds_out["hres_surface_pressure"].values, 101.325, atol=1e-2)
  assert np.allclose(ds_out["hres_total_precipitation"].values, 15.0, atol=1e-2)
  assert np.allclose(
      ds_out["hres_surface_net_solar_radiation"].values, 200.0, atol=1e-2
  )
  assert np.allclose(
      ds_out["hres_surface_net_thermal_radiation"].values, -50.0, atol=1e-2
  )
  assert np.allclose(ds_out["hres_missing_fraction"].values, 0.0)


def test_strict_validation_missing_dates_and_non_overlapping_window(
    tmp_path, basins_gdf
):
  """Verifies that omitting dates or requesting dates outside the archive raises."""
  lats = np.linspace(-89.75, 89.75, 36, dtype=np.float32)
  lons = np.linspace(-179.75, 179.75, 72, dtype=np.float32)
  times = pd.date_range("2020-01-01", "2020-01-05", freq="D")
  ds_archive = xr.Dataset(
      data_vars={
          "imerg_precipitation": (
              ["time", "latitude", "longitude"],
              np.full((len(times), len(lats), len(lons)), 4.0, dtype=np.float32),
          ),
      },
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  )
  store_path = tmp_path / "imerg_daily_surface.zarr"
  ds_archive.to_zarr(store_path)

  ext = IMERGExtractor(data_dir=str(store_path), source="archive")
  with pytest.raises(ValueError, match="start_date and end_date"):
    ext.extract_for_basins(basins_gdf)

  with pytest.raises(GriddedArchiveError, match="no overlap"):
    ext.extract_for_basins(
        basins_gdf, start_date="2025-01-01", end_date="2025-01-05"
    )


def test_serial_runner_with_archive_stores(tmp_path, basins_gdf):
  """Tests extract_multimet_serial end-to-end using explicit archive_stores."""
  lats = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
  lons = np.linspace(-179.75, 179.75, 720, dtype=np.float32)
  times = pd.date_range("2022-01-01", "2022-01-02", freq="D")

  cpc_store = tmp_path / "cpc_archive.zarr"
  xr.Dataset(
      data_vars={
          "cpc_precipitation": (
              ["time", "latitude", "longitude"],
              np.full((2, 360, 720), 6.0, dtype=np.float32),
          ),
          "cpc_num_stations": (
              ["time", "latitude", "longitude"],
              np.full((2, 360, 720), 4.0, dtype=np.float32),
          ),
      },
      coords={"time": times.values, "latitude": lats, "longitude": lons},
  ).to_zarr(cpc_store)

  out_dir = tmp_path / "output_stores"
  stores = extract_multimet_serial(
      basins=basins_gdf,
      output_dir=out_dir,
      products=["CPC"],
      start_date="2022-01-01",
      end_date="2022-01-02",
      source="archive",
      archive_stores={"CPC": str(cpc_store)},
  )
  assert "CPC" in stores
  ds_written = xr.open_zarr(stores["CPC"])
  assert np.allclose(ds_written["cpc_precipitation"].values, 6.0, atol=1e-3)
  assert np.allclose(ds_written["cpc_num_stations"].values, 4.0, atol=1e-3)
  assert np.allclose(ds_written["cpc_missing_fraction"].values, 0.0, atol=1e-3)


def test_runner_requires_explicit_basins_and_output_flags():
  """Verifies that runner CLI has no default output/input paths and requires explicit flags."""
  from multimet.timeseries_extractors.runner import _build_parser

  parser = _build_parser()

  # Omitting --basins_path or --output_dir must fail
  with pytest.raises(SystemExit):
    parser.parse_args(["--start_date", "2022-01-01", "--end_date", "2022-01-02"])

  with pytest.raises(SystemExit):
    parser.parse_args([
        "--basins_path",
        "/tmp/basins.geojson",
        "--start_date",
        "2022-01-01",
        "--end_date",
        "2022-01-02",
    ])

  args = parser.parse_args([
      "--basins_path",
      "/tmp/basins.geojson",
      "--output_dir",
      "/tmp/out_stores",
      "--start_date",
      "2022-01-01",
      "--end_date",
      "2022-01-02",
  ])
  assert args.basins_path == ["/tmp/basins.geojson"]
  assert args.output_dir == "/tmp/out_stores"

