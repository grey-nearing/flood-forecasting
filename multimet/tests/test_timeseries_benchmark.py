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

"""Mock-free unit tests for gridded archive and timeseries benchmark modules."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from multimet.gridded_archive_builders import benchmark as grid_bench
from multimet.gridded_archive_builders.build_cpc_archive import (
    CPC_LATS,
    CPC_LONS,
    CPC_VARIABLE,
)
from multimet.gridded_archive_builders.build_imerg_archive import (
    IMERG_LATS,
    IMERG_LONS,
    IMERG_VARIABLE,
)
from multimet.timeseries_extractors import benchmark as ts_bench
from multimet.timeseries_extractors.config import Product
from multimet.timeseries_extractors.gridded_archive import extract_from_archive
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import xarray as xr

pytestmark = pytest.mark.unit


def _write_synthetic_imerg_daily_nc4(
    nc_path: Path, date_str: str, grid_lat_lon: np.ndarray
) -> None:
  """Writes a synthetic V07 IMERG daily NetCDF-4 granule with shape (time, lon, lat)."""
  transposed = grid_lat_lon.T[np.newaxis, :, :].astype(np.float32)
  ds = xr.Dataset(
      data_vars={
          "precipitation": (["time", "lon", "lat"], transposed),
      },
      coords={
          "time": pd.DatetimeIndex([date_str]),
          "lon": IMERG_LONS,
          "lat": IMERG_LATS,
      },
  )
  ds.to_netcdf(nc_path, engine="netcdf4")


def test_gridded_archive_benchmark_imerg_local_exact_parity(tmp_path: Path):
  """Tests rebuilding a 2-day IMERG archive from local nc4 files vs reference Zarr."""
  local_nc_dir = tmp_path / "imerg_nc4"
  local_nc_dir.mkdir(parents=True, exist_ok=True)

  dates = pd.date_range("2024-01-01", "2024-01-02", freq="1D")
  grids = []
  for idx, dt in enumerate(dates):
    grid = np.full(
        (len(IMERG_LATS), len(IMERG_LONS)), float(idx + 2.5), dtype=np.float32
    )
    grid[0:10, 0:10] = np.nan
    grids.append(grid)
    ymd = dt.strftime("%Y%m%d")
    nc_file = (
        local_nc_dir / f"3B-DAY-E.MS.MRG.3IMERG.{ymd}-S000000-E235959.V07B.nc4"
    )
    _write_synthetic_imerg_daily_nc4(
        nc_file, dt.strftime("%Y-%m-%d"), grid
    )

  ref_zarr = tmp_path / "ref_imerg.zarr"
  ds_ref = xr.Dataset(
      data_vars={
          IMERG_VARIABLE: (
              ["time", "latitude", "longitude"],
              np.stack(grids, axis=0),
          )
      },
      coords={
          "time": dates.values,
          "latitude": IMERG_LATS,
          "longitude": IMERG_LONS,
      },
  )
  ds_ref.to_zarr(ref_zarr)

  out_dir = tmp_path / "bench_imerg_out"
  grid_bench.main([
      "--product",
      "IMERG",
      "--start-date",
      "2024-01-01",
      "--end-date",
      "2024-01-02",
      "--imerg-source",
      "local",
      "--imerg-local-dir",
      str(local_nc_dir),
      "--imerg-local-format",
      "nc4",
      "--reference-zarr",
      str(ref_zarr),
      "--output-dir",
      str(out_dir),
      "--num-workers",
      "1",
  ])

  summary_csv = out_dir / "summary_metrics.csv"
  daily_csv = out_dir / "daily_metrics.csv"
  report_md = out_dir / "benchmark_report.md"
  assert summary_csv.exists()
  assert daily_csv.exists()
  assert report_md.exists()

  summary_df = pd.read_csv(summary_csv)
  assert len(summary_df) == 1
  row = summary_df.iloc[0]
  assert row["product"] == "IMERG"
  assert row["variable"] == IMERG_VARIABLE
  assert bool(row["time_exact_match"]) is True
  assert bool(row["lat_exact_match"]) is True
  assert bool(row["lon_exact_match"]) is True
  assert np.isclose(row["rebuilt_only_nan_pct"], 0.0)
  assert np.isclose(row["ref_only_nan_pct"], 0.0)
  assert np.isclose(row["mae"], 0.0)
  assert np.isclose(row["rmse"], 0.0)
  assert np.isclose(row["pearson_r"], 1.0)
  assert np.isclose(row["frac_exact_match"], 1.0)


def test_compare_gridded_archives_detects_bias_and_nan_mismatch(tmp_path: Path):
  """Tests compare_gridded_archives with controlled numerical perturbation and NaN diff."""
  dates = pd.date_range("2020-01-01", "2020-01-03", freq="1D")
  rng = np.random.default_rng(42)
  ref_data = rng.uniform(
      0.0, 50.0, size=(len(dates), len(CPC_LATS), len(CPC_LONS))
  ).astype(np.float32)
  ref_data[:, 0:5, 0:5] = np.nan

  reb_data = ref_data.copy() + np.float32(1e-3)
  # Introduce 2 rebuilt-only NaN cells
  reb_data[0, 10, 10] = np.nan
  reb_data[1, 20, 20] = np.nan

  ref_zarr = tmp_path / "cpc_ref.zarr"
  reb_zarr = tmp_path / "cpc_reb.zarr"

  for path, arr in ((ref_zarr, ref_data), (reb_zarr, reb_data)):
    xr.Dataset(
        data_vars={CPC_VARIABLE: (["time", "latitude", "longitude"], arr)},
        coords={
            "time": dates.values,
            "latitude": CPC_LATS,
            "longitude": CPC_LONS,
        },
    ).to_zarr(path)

  summary_df, daily_df = grid_bench.compare_gridded_archives(
      product="CPC",
      rebuilt_zarr=str(reb_zarr),
      reference_zarr=str(ref_zarr),
      start_date="2020-01-01",
      end_date="2020-01-03",
  )
  assert len(summary_df) == 1
  assert len(daily_df) == 3
  row = summary_df.iloc[0]
  assert int(row["rebuilt_only_nan_count"]) == 2
  assert int(row["ref_only_nan_count"]) == 0
  assert np.isclose(row["bias"], 1e-3, atol=1e-5)
  assert np.isclose(row["mae"], 1e-3, atol=1e-5)
  assert np.isclose(row["pearson_r"], 1.0, atol=1e-6)


def test_timeseries_benchmark_end_to_end_with_polygon_revision(tmp_path: Path):
  """Tests timeseries_extractors/benchmark.py across CPC and HRES with revised polygons."""
  # Create 3 basins:
  # 1. camels_01: unrevised 1x1 deg box [10..11 E, 45..46 N]
  # 2. camelscl_rev: polygon in dataset is revised to [20..22 E, -35..-33 N] (4 deg^2),
  #    while canonical store was extracted from unrevised [20..20.5 E, -35..-34.5 N] (0.25 deg^2)
  # 3. unmatched_99: not present in canonical store
  poly_camels = shapely.geometry.box(10.0, 45.0, 11.0, 46.0)
  poly_cl_unrevised = shapely.geometry.box(20.0, -35.0, 20.5, -34.5)
  poly_cl_revised = shapely.geometry.box(20.0, -35.0, 22.0, -33.0)
  poly_unmatched = shapely.geometry.box(-100.0, 35.0, -99.0, 36.0)

  area_camels = (
      poly_camels.area
      * (111.0**2)
      * float(np.cos(np.radians(poly_camels.centroid.y)))
  )
  area_cl_unrev = (
      poly_cl_unrevised.area
      * (111.0**2)
      * float(np.cos(np.radians(poly_cl_unrevised.centroid.y)))
  )

  bench_df = pd.DataFrame({
      "gauge_id": ["camels_01", "camelscl_rev", "unmatched_99"],
      "dataset": ["camels", "camelscl", "camels"],
      "size_tier": ["4_large", "3_medium", "4_large"],
      "ref_area_km2": [area_camels, area_cl_unrev, area_camels],
      "geometry_wkt": [
          poly_camels.wkt,
          poly_cl_revised.wkt,
          poly_unmatched.wkt,
      ],
  })
  dataset_parquet = tmp_path / "benchmark_basins.parquet"
  bench_df.to_parquet(dataset_parquet, index=False)

  loaded_gdf = ts_bench.load_benchmark_dataset(str(dataset_parquet))
  assert len(loaded_gdf) == 3
  assert bool(loaded_gdf.loc["camels_01", "is_geometry_revised"]) is False
  assert bool(loaded_gdf.loc["camelscl_rev", "is_geometry_revised"]) is True

  # Build synthetic gridded archives for CPC and HRES over two 2-day windows:
  # Window 1: 2020-06-01..2020-06-02 (HRES Tier 1)
  # Window 2: 2023-09-01..2023-09-02 (HRES Tier 3 - radiation NaN)
  dates_w1 = pd.date_range("2020-06-01", "2020-06-02", freq="1D")
  dates_w2 = pd.date_range("2023-09-01", "2023-09-02", freq="1D")
  all_dates = dates_w1.append(dates_w2)

  lats = np.linspace(-89.5, 89.5, 180, dtype=np.float32)
  lons = np.linspace(-179.5, 179.5, 360, dtype=np.float32)
  lon_grid, lat_grid = np.meshgrid(lons, lats)

  # Spatial gradient so revised polygon produces different basin average than unrevised
  spatial_pattern = (0.2 * np.abs(lat_grid) + 0.3 * np.abs(lon_grid)).astype(
      np.float32
  )
  time_factors = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)[
      :, np.newaxis, np.newaxis
  ]
  cpc_3d = time_factors * spatial_pattern[np.newaxis, :, :]

  cpc_grid_zarr = tmp_path / "cpc_grid.zarr"
  xr.Dataset(
      data_vars={
          "cpc_precipitation": (["time", "latitude", "longitude"], cpc_3d)
      },
      coords={"time": all_dates.values, "latitude": lats, "longitude": lons},
  ).to_zarr(cpc_grid_zarr)

  leads = np.arange(1, 11, dtype=np.int64)
  lead_factors = np.linspace(1.0, 1.9, 10, dtype=np.float32)[
      np.newaxis, :, np.newaxis, np.newaxis
  ]
  hres_base_4d = (
      time_factors[:, np.newaxis, :, :]
      * lead_factors
      * spatial_pattern[np.newaxis, np.newaxis, :, :]
  )
  # In Tier 3 (indices 2 and 3), solar and thermal radiation are NaN
  hres_rad_4d = hres_base_4d.copy()
  hres_rad_4d[2:, :, :, :] = np.nan

  hres_grid_zarr = tmp_path / "hres_grid.zarr"
  xr.Dataset(
      data_vars={
          "hres_surface_net_solar_radiation": (
              ["time", "lead_time", "latitude", "longitude"],
              hres_rad_4d,
          ),
          "hres_surface_net_thermal_radiation": (
              ["time", "lead_time", "latitude", "longitude"],
              hres_rad_4d,
          ),
          "hres_surface_pressure": (
              ["time", "lead_time", "latitude", "longitude"],
              hres_base_4d + np.float32(100.0),
          ),
          "hres_temperature_2m": (
              ["time", "lead_time", "latitude", "longitude"],
              hres_base_4d,
          ),
          "hres_total_precipitation": (
              ["time", "lead_time", "latitude", "longitude"],
              hres_base_4d,
          ),
      },
      coords={
          "time": all_dates.values,
          "lead_time": leads,
          "latitude": lats,
          "longitude": lons,
      },
  ).to_zarr(hres_grid_zarr)

  # Build canonical Zarr stores using unrevised geometries for camels_01 and camelscl_rev
  canon_gdf = gpd.GeoDataFrame(
      {"geometry": [poly_camels, poly_cl_unrevised]},
      index=["camels_01", "camelscl_rev"],
      crs="EPSG:4326",
  )
  canon_root = tmp_path / "canonical_v1_1"
  canon_writer = MultiMetZarrWriter(canon_root)

  for prod_enum, grid_zarr in (
      (Product.CPC, cpc_grid_zarr),
      (Product.HRES, hres_grid_zarr),
  ):
    for d_win in (dates_w1, dates_w2):
      ds_can_part = extract_from_archive(
          prod_enum,
          grid_zarr,
          canon_gdf,
          d_win[0].strftime("%Y-%m-%d"),
          d_win[-1].strftime("%Y-%m-%d"),
          use_bounding_box=False,
      )
      canon_writer.write_or_append(ds_can_part, prod_enum)

  out_dir = tmp_path / "ts_bench_out"
  reconstructed_dir = tmp_path / "reconstructed_zarr"

  ts_bench.main([
      "--dataset",
      str(dataset_parquet),
      "--canonical-dir",
      str(canon_root),
      "--archive-store",
      f"CPC={cpc_grid_zarr}",
      "--archive-store",
      f"HRES={hres_grid_zarr}",
      "--date-windows",
      "2020-06-01:2020-06-02",
      "2023-09-01:2023-09-02",
      "--save-reconstructed-zarr",
      str(reconstructed_dir),
      "--output-dir",
      str(out_dir),
      "--num-workers",
      "2",
  ])

  for expected_file in (
      "summary_metrics.csv",
      "metrics_by_size_tier.csv",
      "metrics_by_dataset.csv",
      "metrics_by_window.csv",
      "hres_metrics_by_lead.csv",
      "per_basin_metrics.parquet",
      "benchmark_report.md",
  ):
    assert (out_dir / expected_file).exists(), f"Missing {expected_file}"

  assert (reconstructed_dir / "CPC" / "timeseries.zarr").exists()
  assert (reconstructed_dir / "HRES" / "timeseries.zarr").exists()

  summary_df = pd.read_csv(out_dir / "summary_metrics.csv")
  # Check unrevised_geometry subset (camels_01) achieves exact reconstruction
  unrev_cpc = summary_df[
      (summary_df["product"] == "CPC")
      & (summary_df["basin_subset"] == "unrevised_geometry")
  ].iloc[0]
  assert int(unrev_cpc["n_basins"]) == 1
  assert np.isclose(unrev_cpc["mae"], 0.0, atol=1e-6)
  assert np.isclose(unrev_cpc["rmse"], 0.0, atol=1e-6)
  assert np.isclose(unrev_cpc["pearson_r"], 1.0, atol=1e-6)
  assert np.isclose(unrev_cpc["median_nse"], 1.0, atol=1e-6)
  assert np.isclose(unrev_cpc["median_kge"], 1.0, atol=1e-6)

  # Check all_matched subset (includes camelscl_rev with revised polygon) has non-zero MAE
  all_cpc = summary_df[
      (summary_df["product"] == "CPC")
      & (summary_df["basin_subset"] == "all_matched")
  ].iloc[0]
  assert int(all_cpc["n_basins"]) == 2
  assert float(all_cpc["mae"]) > 0.01

  # Check HRES Tier 3 radiation is 100% Both NaN
  win_df = pd.read_csv(out_dir / "metrics_by_window.csv")
  tier3_solar = win_df[
      (win_df["product"] == "HRES")
      & (win_df["variable"] == "hres_surface_net_solar_radiation")
      & (win_df["basin_subset"] == "unrevised_geometry")
      & (win_df["window"] == "2023-09-01:2023-09-02")
  ].iloc[0]
  assert np.isclose(tier3_solar["both_nan_pct"], 100.0)
  assert np.isclose(tier3_solar["both_valid_pct"], 0.0)

  # Check HRES lead breakdown has 10 lead days
  lead_df = pd.read_csv(out_dir / "hres_metrics_by_lead.csv")
  t2m_leads = lead_df[
      (lead_df["variable"] == "hres_temperature_2m")
      & (lead_df["basin_subset"] == "unrevised_geometry")
      & (lead_df["upstream_tier"] == "ALL")
  ]
  assert len(t2m_leads) == 10
  assert np.allclose(t2m_leads["mae"].to_numpy(), 0.0, atol=1e-6)

  # Verify ERA5_LAND and GRAPHCAST are explicitly rejected
  for forbidden in ("ERA5_LAND", "GRAPHCAST"):
    with pytest.raises(ValueError, match="Unsupported benchmark product"):
      ts_bench.run_timeseries_benchmark(
          dataset_path=str(dataset_parquet),
          canonical_dir=str(canon_root),
          archive_stores={forbidden: str(cpc_grid_zarr)},
          date_windows=["2020-06-01:2020-06-02"],
          output_dir=str(tmp_path / "forbidden_out"),
      )


def test_timeseries_benchmark_preserves_none_geometry_as_extracted_only_nan(tmp_path: Path):
  """Verifies that None geometry_wkt rows are retained, never fall back to canonical, and flag extracted_only_nan."""
  poly_valid = shapely.geometry.box(10.0, 45.0, 11.0, 46.0)
  poly_failed_canon = shapely.geometry.box(12.0, 45.0, 13.0, 46.0)
  area_valid = (
      poly_valid.area
      * (111.0**2)
      * float(np.cos(np.radians(poly_valid.centroid.y)))
  )

  bench_df = pd.DataFrame({
      "gauge_id": ["camels_ok", "camels_failed_delin"],
      "dataset": ["camels", "camels"],
      "size_tier": ["4_large", "4_large"],
      "ref_area_km2": [area_valid, area_valid],
      "geometry_wkt": [poly_valid.wkt, None],
  })
  dataset_parquet = tmp_path / "cascade_basins.parquet"
  bench_df.to_parquet(dataset_parquet, index=False)

  dates = pd.date_range("2020-06-01", "2020-06-02", freq="1D")
  lats = np.linspace(-89.5, 89.5, 180, dtype=np.float32)
  lons = np.linspace(-179.5, 179.5, 360, dtype=np.float32)
  cpc_3d = np.full((len(dates), len(lats), len(lons)), 5.0, dtype=np.float32)
  cpc_grid_zarr = tmp_path / "cpc_grid_none.zarr"
  xr.Dataset(
      data_vars={"cpc_precipitation": (["time", "latitude", "longitude"], cpc_3d)},
      coords={"time": dates.values, "latitude": lats, "longitude": lons},
  ).to_zarr(cpc_grid_zarr)

  canon_gdf = gpd.GeoDataFrame(
      {"geometry": [poly_valid, poly_failed_canon]},
      index=["camels_ok", "camels_failed_delin"],
      crs="EPSG:4326",
  )
  canon_root = tmp_path / "canon_none_test"
  canon_writer = MultiMetZarrWriter(canon_root)
  ds_can = extract_from_archive(
      Product.CPC,
      cpc_grid_zarr,
      canon_gdf,
      "2020-06-01",
      "2020-06-02",
      use_bounding_box=False,
  )
  canon_writer.write_or_append(ds_can, Product.CPC)

  out_dir = tmp_path / "ts_none_out"
  res = ts_bench.run_timeseries_benchmark(
      dataset_path=str(dataset_parquet),
      canonical_dir=str(canon_root),
      archive_stores={"CPC": str(cpc_grid_zarr)},
      date_windows=["2020-06-01:2020-06-02"],
      output_dir=str(out_dir),
      num_workers=1,
  )
  summary_df = res["summary"]
  all_row = summary_df[summary_df["basin_subset"] == "all_matched"].iloc[0]
  assert int(all_row["n_basins"]) == 2
  assert int(all_row["total_points"]) == 4
  assert int(all_row["both_valid_count"]) == 2
  assert int(all_row["extracted_only_nan_count"]) == 2
  assert bool(all_row["has_extracted_only_nan_discrepancy"]) is True

  report_text = (out_dir / "benchmark_report.md").read_text(encoding="utf-8")
  assert "DISCREPANCY (2)" in report_text
