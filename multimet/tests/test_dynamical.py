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

"""Unit tests for dynamical.org Icechunk loaders and catchment timeseries extractors."""

from __future__ import annotations

from pathlib import Path
import types
from typing import Dict, Sequence, Set, Tuple

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
import xarray as xr

from multimet.timeseries_extractors.config import (
    ENSEMBLE_STAT_SUFFIXES,
    PRODUCT_BANDS,
    Product,
)
import multimet.timeseries_extractors.dynamical as dyn_mod
from multimet.timeseries_extractors.dynamical import (
    AIFSEnsExtractor,
    AIFSExtractor,
    DynamicalDataLoader,
    DynamicalIMERGExtractor,
    GEFSExtractor,
    GFSExtractor,
    IFSEnsExtractor,
    _build_compressed_weight_matrix,
    _gather_active_cells,
    find_latest_dynamical_forecast_date,
    list_catalog_datasets,
)
from multimet.timeseries_extractors.realtime import fetch_realtime_multimet
from multimet.timeseries_extractors.runner import extract_multimet_serial
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter
from multimet.utils.zonal import ZonalWeightMatrix

pytestmark = pytest.mark.unit


def _make_disjoint_basins_gdf() -> gpd.GeoDataFrame:
  """Creates two disjoint basins separated by empty (ocean/unassigned) grid chunks."""
  gdf = gpd.GeoDataFrame(
      {
          "basin_id": ["basin_west", "basin_east"],
          "geometry": [
              box(-120.0, 40.0, -119.5, 40.5),
              box(-75.0, 35.0, -74.5, 35.5),
          ],
      },
      crs="EPSG:4326",
  ).set_index("basin_id")
  return gdf


def _build_synthetic_ecmwf_dataset(
    init_times: Sequence[str],
    lead_hours: Sequence[int],
    ensemble_members: int | None = None,
    unpopulated_last_init: bool = False,
) -> xr.Dataset:
  """Builds a synthetic ECMWF (AIFS / AIFS_ENS / IFS_ENS) dataset on a descending lat grid."""
  lats = np.linspace(42.0, 34.0, 33, dtype=np.float64)  # 0.25 deg descending
  lons = np.linspace(-121.0, -73.0, 193, dtype=np.float64)  # 0.25 deg ascending
  init_idx = pd.to_datetime(init_times)
  leads = np.array([ np.timedelta64(h, "h") for h in lead_hours ], dtype="timedelta64[ns]")

  n_t = len(init_idx)
  n_l = len(leads)
  n_y = len(lats)
  n_x = len(lons)

  # Base physical values in upstream ECMWF units:
  # temperature_2m = 20.0 degC, dewpoint_2m = 10.0 degC, pressure = 100000 Pa (100 kPa),
  # precip_rate = 2.0 / 86400 kg m-2 s-1 (2.0 mm/day), sw = 250 W/m2, lw = 320 W/m2,
  # u10 = 3.5 m/s, v10 = -1.5 m/s.
  raw_base = {
      "dew_point_temperature_2m": 10.0,
      "downward_long_wave_radiation_flux_surface": 320.0,
      "downward_short_wave_radiation_flux_surface": 250.0,
      "pressure_surface": 100000.0,
      "temperature_2m": 20.0,
      "precipitation_surface": 2.0 / 86400.0,
      "wind_u_10m": 3.5,
      "wind_v_10m": -1.5,
  }
  step_interval_vars = {
      "downward_long_wave_radiation_flux_surface",
      "downward_short_wave_radiation_flux_surface",
      "precipitation_surface",
  }

  coords: Dict[str, object] = {
      "init_time": init_idx,
      "lead_time": leads,
      "latitude": lats,
      "longitude": lons,
  }
  data_vars: Dict[str, Tuple[Tuple[str, ...], da.Array]] = {}

  if ensemble_members is None:
    dims = ("init_time", "lead_time", "latitude", "longitude")
    chunks = (1, n_l, 8, 16)
    for var_name, base_val in raw_base.items():
      arr = np.full((n_t, n_l, n_y, n_x), base_val, dtype=np.float32)
      # Add a deterministic longitudinal gradient (+1.0 on the eastern half)
      arr[:, :, :, n_x // 2 :] += (
          (1.0 / 86400.0) if var_name == "precipitation_surface"
          else (1000.0 if var_name == "pressure_surface" else 1.0)
      )
      # Step-interval variables are NaN at lead_time == 0h in live dynamical.org stores
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          arr[:, l_i, :, :] = np.nan
      if unpopulated_last_init:
        arr[-1, -1, :, :] = np.nan
      data_vars[var_name] = (dims, da.from_array(arr, chunks=chunks))
  else:
    coords["ensemble_member"] = np.arange(ensemble_members, dtype=np.int64)
    dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    chunks = (1, n_l, ensemble_members, 8, 16)
    member_offsets = np.arange(ensemble_members, dtype=np.float32)
    for var_name, base_val in raw_base.items():
      arr = np.full(
          (n_t, n_l, ensemble_members, n_y, n_x), base_val, dtype=np.float32
      )
      scale = (
          (1.0 / 86400.0) if var_name == "precipitation_surface"
          else (1000.0 if var_name == "pressure_surface" else 1.0)
      )
      arr += member_offsets[None, None, :, None, None] * scale
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          arr[:, l_i, :, :, :] = np.nan
      if unpopulated_last_init:
        arr[-1, -1, :, :, :] = np.nan
      data_vars[var_name] = (dims, da.from_array(arr, chunks=chunks))

  return xr.Dataset(data_vars, coords=coords)


def _build_synthetic_noaa_dataset(
    init_times: Sequence[str],
    lead_hours: Sequence[int],
    ensemble_members: int | None = None,
) -> xr.Dataset:
  """Builds a synthetic NOAA (GFS / GEFS) dataset with 9 surface vars and (init, member, lead) order."""
  lats = np.linspace(42.0, 34.0, 33, dtype=np.float64)
  lons = np.linspace(-121.0, -73.0, 193, dtype=np.float64)
  init_idx = pd.to_datetime(init_times)
  leads = np.array([ np.timedelta64(h, "h") for h in lead_hours ], dtype="timedelta64[ns]")

  n_t = len(init_idx)
  n_l = len(leads)
  n_y = len(lats)
  n_x = len(lons)

  raw_base = {
      "downward_long_wave_radiation_flux_surface": 300.0,
      "downward_short_wave_radiation_flux_surface": 200.0,
      "pressure_surface": 98000.0,
      "temperature_2m": 15.0,
      "maximum_temperature_2m": 18.0,
      "minimum_temperature_2m": 12.0,
      "precipitation_surface": 4.0 / 86400.0,
      "wind_u_10m": 2.0,
      "wind_v_10m": 1.0,
  }
  step_interval_vars = {
      "downward_long_wave_radiation_flux_surface",
      "downward_short_wave_radiation_flux_surface",
      "maximum_temperature_2m",
      "minimum_temperature_2m",
      "precipitation_surface",
  }

  coords: Dict[str, object] = {
      "init_time": init_idx,
      "lead_time": leads,
      "latitude": lats,
      "longitude": lons,
  }
  data_vars: Dict[str, Tuple[Tuple[str, ...], da.Array]] = {}

  if ensemble_members is None:
    dims = ("init_time", "lead_time", "latitude", "longitude")
    chunks = (1, n_l, 8, 16)
    for var_name, base_val in raw_base.items():
      arr = np.full((n_t, n_l, n_y, n_x), base_val, dtype=np.float32)
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          arr[:, l_i, :, :] = np.nan
        elif h > 0:
          hour_of_day = ((h - 1) % 24) + 1
          if var_name == "maximum_temperature_2m":
            # Peak of base_val + 6.0 occurs when hour_of_day == 24
            arr[:, l_i, :, :] = base_val + (6.0 if hour_of_day == 24 else 1.0)
          elif var_name == "minimum_temperature_2m":
            # Trough of base_val - 5.0 occurs when hour_of_day == 24
            arr[:, l_i, :, :] = base_val - (5.0 if hour_of_day == 24 else 1.0)
      data_vars[var_name] = (dims, da.from_array(arr, chunks=chunks))
  else:
    # GEFS has dimension order (init_time, ensemble_member, lead_time, latitude, longitude)
    coords["ensemble_member"] = np.arange(ensemble_members, dtype=np.int64)
    dims = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")
    chunks = (1, ensemble_members, n_l, 8, 16)
    member_offsets = np.arange(ensemble_members, dtype=np.float32)
    for var_name, base_val in raw_base.items():
      arr = np.full(
          (n_t, ensemble_members, n_l, n_y, n_x), base_val, dtype=np.float32
      )
      scale = (
          (1.0 / 86400.0) if var_name == "precipitation_surface"
          else (1000.0 if var_name == "pressure_surface" else 1.0)
      )
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          arr[:, :, l_i, :, :] = np.nan
        elif h > 0:
          hour_of_day = ((h - 1) % 24) + 1
          if var_name == "maximum_temperature_2m":
            arr[:, :, l_i, :, :] = (
                base_val
                + (6.0 if hour_of_day == 24 else 1.0)
                + member_offsets[None, :, None, None] * scale
            )
          elif var_name == "minimum_temperature_2m":
            arr[:, :, l_i, :, :] = (
                base_val
                - (5.0 if hour_of_day == 24 else 1.0)
                + member_offsets[None, :, None, None] * scale
            )
          else:
            arr[:, :, l_i, :, :] = (
                base_val + member_offsets[None, :, None, None] * scale
            )
        else:
          arr[:, :, l_i, :, :] = (
              base_val + member_offsets[None, :, None, None] * scale
          )
      data_vars[var_name] = (dims, da.from_array(arr, chunks=chunks))

  return xr.Dataset(data_vars, coords=coords)


def test_active_chunk_compressed_csr_is_bit_for_bit_exact_and_skips_inactive_chunks():
  """Verifies compressed CSR + active-chunk gather matches full-grid reduction and skips ocean chunks."""
  gdf = _make_disjoint_basins_gdf()
  lats = np.linspace(42.0, 34.0, 33, dtype=np.float64)
  lons = np.linspace(-121.0, -73.0, 193, dtype=np.float64)

  full_weights = ZonalWeightMatrix.from_geodataframe(gdf, lats, lons)
  active_cols, compressed_weights = _build_compressed_weight_matrix(full_weights)
  assert len(active_cols) < len(lats) * len(lons)

  rng = np.random.default_rng(42)
  raw_grid = rng.uniform(0.0, 50.0, size=(4, len(lats), len(lons))).astype(
      np.float32
  )
  # Inject a few NaNs inside and outside the basins
  raw_grid[1, 6, 4] = np.nan
  raw_grid[2, 15, 90] = np.nan

  accessed_chunks: Set[Tuple[int, int]] = set()

  def _track_block(block: np.ndarray, block_info=None) -> np.ndarray:
    if block_info is not None and 0 in block_info:
      loc = block_info[0]["chunk-location"]
      accessed_chunks.add((int(loc[1]), int(loc[2])))
    return block

  dask_raw = da.from_array(raw_grid, chunks=(4, 8, 16))
  tracked_dask = da.map_blocks(_track_block, dask_raw, dtype=np.float32)
  ds_xr = xr.Dataset(
      {"temperature_2m": (("time", "latitude", "longitude"), tracked_dask)},
      coords={"time": np.arange(4), "latitude": lats, "longitude": lons},
  )

  gathered = _gather_active_cells(
      ds_xr, ["temperature_2m"], active_cols, len(lons)
  )
  active_cells = gathered["temperature_2m"]
  assert active_cells.shape == (4, len(active_cols))

  # Total spatial chunks = ceil(33/8) * ceil(193/16) = 5 * 13 = 65.
  # Only the few chunks touching basin_west (lon ~ -120) and basin_east (lon ~ -75) may be accessed.
  assert len(accessed_chunks) <= 6
  middle_lon_chunks = {c_lon for _, c_lon in accessed_chunks if 2 <= c_lon <= 10}
  assert not middle_lon_chunks, (
      f"Intervening non-basin longitude chunks were unexpectedly accessed: {middle_lon_chunks}"
  )

  mean_compressed, miss_compressed = compressed_weights.reduce_3d_with_coverage(
      active_cells[:, :, np.newaxis]
  )
  mean_full, miss_full = full_weights.reduce_3d_with_coverage(raw_grid)

  np.testing.assert_array_equal(mean_compressed, mean_full)
  np.testing.assert_array_equal(miss_compressed, miss_full)


def test_dynamical_data_loader_spatial_subset_and_timeseries(
    monkeypatch: pytest.MonkeyPatch,
):
  """Verifies DynamicalDataLoader.compute_spatial_slices, load_spatial_subset, and extract_basin_timeseries."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 49, 6))
  ds = _build_synthetic_ecmwf_dataset(["2026-03-01T00:00:00"], lead_hours)
  loader = DynamicalDataLoader("ecmwf-aifs-single-forecast", ds=ds)

  monkeypatch.setattr(
      dyn_mod,
      "dynamical_catalog",
      types.SimpleNamespace(
          list=lambda: ["ecmwf-aifs-single-forecast", "noaa-gfs-forecast"],
          open=lambda _id: ds,
      ),
  )
  assert "ecmwf-aifs-single-forecast" in list_catalog_datasets()
  slices = loader.compute_spatial_slices(gdf, buffer=0.25)
  lat_slice = slices["latitude"]
  lon_slice = slices["longitude"]
  assert lat_slice.start < lat_slice.stop
  assert lon_slice.start < lon_slice.stop

  sub = loader.load_spatial_subset(
      gdf,
      variables=["temperature_2m"],
      start_date="2026-03-01",
      end_date="2026-03-01",
  )
  assert "temperature_2m" in sub.data_vars

  ts = loader.extract_basin_timeseries(
      gdf,
      variables=["temperature_2m"],
      start_date="2026-03-01",
      end_date="2026-03-01",
  )
  assert ts["temperature_2m"].dims == ("basin", "date", "lead_time")
  assert list(ts["basin"].values) == ["basin_west", "basin_east"]
  np.testing.assert_allclose(ts["temperature_2m"].sel(basin="basin_west"), 20.0)
  np.testing.assert_allclose(ts["temperature_2m"].sel(basin="basin_east"), 21.0)


def test_aifs_extractor_variables_units_and_lead0_nan():
  """Verifies AIFSExtractor extracts all 8 ECMWF variables with Caravan units and handles NaN at lead=0h."""
  gdf = _make_disjoint_basins_gdf()
  # 6-hourly steps up to 240h (10 days)
  lead_hours = list(range(0, 241, 6))
  ds = _build_synthetic_ecmwf_dataset(
      ["2026-03-01T00:00:00", "2026-03-02T00:00:00"], lead_hours
  )
  loader = DynamicalDataLoader(AIFSExtractor.DEFAULT_DATASET_ID, ds=ds)
  extractor = AIFSExtractor(loader=loader)

  out = extractor.extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-02"
  )
  assert set(PRODUCT_BANDS[Product.AIFS]).issubset(set(out.data_vars))
  assert "aifs_missing_fraction" in out.data_vars
  assert out.sizes["basin"] == 2
  assert out.sizes["date"] == 2
  assert out.sizes["lead_time"] == 10

  # Verify basin_west (western half) and basin_east (eastern half, +1 offset)
  np.testing.assert_allclose(
      out["aifs_temperature_2m"].sel(basin="basin_west"), 20.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["aifs_temperature_2m"].sel(basin="basin_east"), 21.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["aifs_dewpoint_temperature_2m"].sel(basin="basin_west"),
      10.0,
      rtol=1e-5,
  )
  # Pressure converted from Pa (100000, 101000) to kPa (100.0, 101.0)
  np.testing.assert_allclose(
      out["aifs_surface_pressure"].sel(basin="basin_west"), 100.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["aifs_surface_pressure"].sel(basin="basin_east"), 101.0, rtol=1e-5
  )
  # Precip rate (2.0/86400 kg m-2 s-1) integrated over 24h -> 2.0 mm/day (west) and 3.0 mm/day (east)
  np.testing.assert_allclose(
      out["aifs_total_precipitation"].sel(basin="basin_west"), 2.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["aifs_total_precipitation"].sel(basin="basin_east"), 3.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["aifs_downward_short_wave_radiation"].sel(basin="basin_west"),
      250.0,
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["aifs_downward_long_wave_radiation"].sel(basin="basin_west"),
      320.0,
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["aifs_u_component_of_wind_10m"].sel(basin="basin_west"),
      3.5,
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["aifs_v_component_of_wind_10m"].sel(basin="basin_west"),
      -1.5,
      rtol=1e-5,
  )
  np.testing.assert_allclose(out["aifs_missing_fraction"], 0.0, atol=1e-6)


def test_gfs_extractor_nonuniform_leads_min_max_temp_and_spinup_cutoff():
  """Verifies GFSExtractor handles 1h+3h steps, daily Tmin/Tmax, and spinup_only_before."""
  gdf = _make_disjoint_basins_gdf()
  # GFS has 1-hourly steps for day 1 (0..24h) and 3-hourly steps for days 2..10 (27..240h)
  lead_hours = list(range(0, 25, 1)) + list(range(27, 241, 3))
  ds = _build_synthetic_noaa_dataset(
      ["2026-03-01T00:00:00", "2026-03-02T00:00:00"], lead_hours
  )
  loader = DynamicalDataLoader(GFSExtractor.DEFAULT_DATASET_ID, ds=ds)
  extractor = GFSExtractor(loader=loader)

  out = extractor.extract_for_basins(
      gdf,
      start_date="2026-03-01",
      end_date="2026-03-02",
      spinup_only_before="2026-03-02",
  )
  assert set(PRODUCT_BANDS[Product.GFS]).issubset(set(out.data_vars))

  # On 2026-03-01 (before spinup_only_before), only lead_time index 0 (1D) is populated
  d0 = out.isel(date=0)
  assert np.all(np.isfinite(d0["gfs_total_precipitation"].isel(lead_time=0)))
  assert np.all(np.isnan(d0["gfs_total_precipitation"].isel(lead_time=slice(1, None))))
  np.testing.assert_allclose(
      d0["gfs_missing_fraction"].isel(lead_time=0), 0.0, atol=1e-6
  )
  np.testing.assert_allclose(
      d0["gfs_missing_fraction"].isel(lead_time=slice(1, None)), 1.0
  )

  # On 2026-03-02 (>= spinup_only_before), all 10 lead days are populated
  d1 = out.isel(date=1)
  assert np.all(np.isfinite(d1["gfs_total_precipitation"]))
  np.testing.assert_allclose(d1["gfs_total_precipitation"], 4.0, rtol=1e-5)
  np.testing.assert_allclose(d1["gfs_surface_pressure"], 98.0, rtol=1e-5)
  # Tmax = 18.0 + 6.0 = 24.0; Tmin = 12.0 - 5.0 = 7.0
  np.testing.assert_allclose(d1["gfs_temperature_2m_max"], 24.0, rtol=1e-5)
  np.testing.assert_allclose(d1["gfs_temperature_2m_min"], 7.0, rtol=1e-5)
  np.testing.assert_allclose(d1["gfs_missing_fraction"], 0.0, atol=1e-6)


def test_ensemble_extractors_summary_stats_and_optional_4d_members(tmp_path: Path):
  """Verifies IFS_ENS, AIFS_ENS, and GEFS compute all 7 ensemble stats and optional 4D member arrays."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 241, 6))
  n_members = 11  # offsets 0, 1, 2, ..., 10

  # 1. IFS_ENS (init_time, lead_time, ensemble_member, latitude, longitude)
  ifs_ds = _build_synthetic_ecmwf_dataset(
      ["2026-03-01T00:00:00"], lead_hours, ensemble_members=n_members
  )
  ifs_loader = DynamicalDataLoader(IFSEnsExtractor.DEFAULT_DATASET_ID, ds=ifs_ds)
  ifs_extractor = IFSEnsExtractor(
      loader=ifs_loader, include_ensemble_members=True
  )
  ifs_out = ifs_extractor.extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-01"
  )

  assert len(PRODUCT_BANDS[Product.IFS_ENS]) == 8 * len(ENSEMBLE_STAT_SUFFIXES)
  assert set(PRODUCT_BANDS[Product.IFS_ENS]).issubset(set(ifs_out.data_vars))
  assert "ifs_ens_total_precipitation_ensemble" in ifs_out.data_vars
  assert ifs_out["ifs_ens_total_precipitation_ensemble"].dims == (
      "basin",
      "date",
      "ensemble_member",
      "lead_time",
  )
  assert ifs_out.sizes["ensemble_member"] == n_members

  # For total_precipitation on basin_west: members have values 2.0 + m for m in 0..10 -> [2, 3, ..., 12]
  expected_members = np.arange(2.0, 2.0 + n_members, dtype=np.float32)
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_mean"].sel(basin="basin_west"),
      float(np.mean(expected_members)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_std"].sel(basin="basin_west"),
      float(np.std(expected_members, ddof=0)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_min"].sel(basin="basin_west"),
      float(np.min(expected_members)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_max"].sel(basin="basin_west"),
      float(np.max(expected_members)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_p10"].sel(basin="basin_west"),
      float(np.percentile(expected_members, 10)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_p50"].sel(basin="basin_west"),
      float(np.percentile(expected_members, 50)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_p90"].sel(basin="basin_west"),
      float(np.percentile(expected_members, 90)),
      rtol=1e-5,
  )

  # Verify Zarr writer round-trip with both 3D summary stats and 4D ensemble members
  writer = MultiMetZarrWriter(str(tmp_path))
  store_path = writer.write_or_append(ifs_out, Product.IFS_ENS)
  with xr.open_zarr(store_path) as reloaded:
    assert "ifs_ens_total_precipitation_mean" in reloaded.data_vars
    assert "ifs_ens_total_precipitation_ensemble" in reloaded.data_vars
    assert reloaded["ifs_ens_total_precipitation_ensemble"].shape == (
        2,
        1,
        n_members,
        10,
    )

  # 2. GEFS (init_time, ensemble_member, lead_time, latitude, longitude) with 9 base vars (63 bands)
  gefs_ds = _build_synthetic_noaa_dataset(
      ["2026-03-01T00:00:00"], lead_hours, ensemble_members=n_members
  )
  gefs_loader = DynamicalDataLoader(GEFSExtractor.DEFAULT_DATASET_ID, ds=gefs_ds)
  gefs_extractor = GEFSExtractor(loader=gefs_loader)
  gefs_out = gefs_extractor.extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-01"
  )
  assert len(PRODUCT_BANDS[Product.GEFS]) == 9 * len(ENSEMBLE_STAT_SUFFIXES)
  assert set(PRODUCT_BANDS[Product.GEFS]).issubset(set(gefs_out.data_vars))
  # Tmax per member = 24.0 + m for m in 0..10 -> mean is 29.0, min is 24.0, max is 34.0
  np.testing.assert_allclose(
      gefs_out["gefs_temperature_2m_max_mean"], 29.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      gefs_out["gefs_temperature_2m_max_min"], 24.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      gefs_out["gefs_temperature_2m_max_max"], 34.0, rtol=1e-5
  )

  # 3. AIFS_ENS
  aifs_ens_loader = DynamicalDataLoader(
      AIFSEnsExtractor.DEFAULT_DATASET_ID, ds=ifs_ds
  )
  aifs_ens_out = AIFSEnsExtractor(loader=aifs_ens_loader).extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-01"
  )
  assert set(PRODUCT_BANDS[Product.AIFS_ENS]).issubset(
      set(aifs_ens_out.data_vars)
  )


def test_dynamical_imerg_extractor_half_hourly_to_daily_and_incomplete_day():
  """Verifies DynamicalIMERGExtractor aggregates 48 half-hour steps to mm/day and flags incomplete days."""
  gdf = _make_disjoint_basins_gdf()
  lats = np.linspace(42.0, 34.0, 33, dtype=np.float64)
  lons = np.linspace(-121.0, -73.0, 193, dtype=np.float64)

  # 2 full days (96 half-hours) + 1 incomplete day (only 20 half-hours on 2026-03-03)
  times_full = pd.date_range(
      "2026-03-01T00:00:00", "2026-03-02T23:30:00", freq="30min"
  )
  times_partial = pd.date_range(
      "2026-03-03T00:00:00", periods=20, freq="30min"
  )
  times = times_full.append(times_partial)

  # Constant rate in kg m-2 s-1 (= mm/s):
  # 36.0 / 86400.0 mm/s * 1800s * 48 steps = 36.0 mm/day on West,
  # 84.0 / 86400.0 mm/s * 1800s * 48 steps = 84.0 mm/day on East
  arr = np.full(
      (len(times), len(lats), len(lons)), 36.0 / 86400.0, dtype=np.float32
  )
  arr[:, :, len(lons) // 2 :] = 84.0 / 86400.0
  imerg_ds = xr.Dataset(
      {
          "precipitation_surface": (
              ("time", "latitude", "longitude"),
              da.from_array(arr, chunks=(48, 8, 16)),
          )
      },
      coords={"time": times, "latitude": lats, "longitude": lons},
  )

  loader = DynamicalDataLoader(
      DynamicalIMERGExtractor.DEFAULT_DATASET_ID, ds=imerg_ds
  )
  extractor = DynamicalIMERGExtractor(loader=loader, batch_days=1)
  out = extractor.extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-03"
  )

  assert "dynamical_imerg_precipitation" in out.data_vars
  assert "dynamical_imerg_missing_fraction" in out.data_vars
  assert "imerg_precipitation" not in out.data_vars
  assert out["dynamical_imerg_precipitation"].dims == ("basin", "date")
  assert out.sizes["date"] == 3

  # Days 0 and 1 have all 48 half-hour steps
  np.testing.assert_allclose(
      out["dynamical_imerg_precipitation"].sel(basin="basin_west").isel(date=[0, 1]),
      36.0,
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["dynamical_imerg_precipitation"].sel(basin="basin_east").isel(date=[0, 1]),
      84.0,
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["dynamical_imerg_missing_fraction"].isel(date=[0, 1]), 0.0, atol=1e-6
  )

  # Day 2 (2026-03-03) has only 20/48 half-hours -> rejected as incomplete (NaN, missing_fraction=1.0)
  assert np.all(np.isnan(out["dynamical_imerg_precipitation"].isel(date=2)))
  np.testing.assert_allclose(
      out["dynamical_imerg_missing_fraction"].isel(date=2), 1.0
  )


def test_find_latest_dynamical_forecast_date_and_realtime_fetcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  """Verifies find_latest_dynamical_forecast_date skips unpopulated runs and integrates with fetch_realtime_multimet."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 241, 6))
  # 2026-03-03T00:00:00 has NaN at 240h (unpopulated_last_init=True), so latest complete 10d 00z is 2026-03-02
  ds_aifs = _build_synthetic_ecmwf_dataset(
      [
          "2026-03-01T00:00:00",
          "2026-03-02T00:00:00",
          "2026-03-02T12:00:00",
          "2026-03-03T00:00:00",
      ],
      lead_hours,
      unpopulated_last_init=True,
  )
  aifs_loader = DynamicalDataLoader(AIFSExtractor.DEFAULT_DATASET_ID, ds=ds_aifs)

  latest_dt = find_latest_dynamical_forecast_date(
      AIFSExtractor.DEFAULT_DATASET_ID,
      reference_date="2026-03-03",
      require_full_10d=True,
      loader=aifs_loader,
  )
  assert latest_dt == pd.Timestamp("2026-03-02")

  result = fetch_realtime_multimet(
      basins=gdf,
      output_dir=tmp_path / "realtime_out",
      mode="hotstart",
      reference_date="2026-03-02",
      lookback_days=1,
      products=["AIFS"],
      full_forecast_days=1,
      dynamical_loaders={"AIFS": aifs_loader},
  )
  assert result.reference_date == pd.Timestamp("2026-03-02")
  assert "AIFS" in result.stores
  with xr.open_zarr(result.stores["AIFS"]) as zds:
    assert zds.sizes["date"] == 2
    assert zds.sizes["lead_time"] == 10
    # Spin-up date 2026-03-01 has lead_time=1D populated and leads 2D..10D NaN
    assert np.all(
        np.isfinite(zds["aifs_total_precipitation"].isel(date=0, lead_time=0))
    )
    assert np.all(
        np.isnan(
            zds["aifs_total_precipitation"].isel(date=0, lead_time=slice(1, None))
        )
    )
    # Reference date 2026-03-02 has all 10 lead days populated
    assert np.all(np.isfinite(zds["aifs_total_precipitation"].isel(date=1)))

  # Also verify serial historical runner integration
  monkeypatch.setattr(
      dyn_mod,
      "dynamical_catalog",
      types.SimpleNamespace(open=lambda _id: ds_aifs),
  )
  serial_stores = extract_multimet_serial(
      basins=gdf,
      start_date="2026-03-01",
      end_date="2026-03-02",
      output_dir=str(tmp_path / "serial_out"),
      products=["AIFS"],
  )
  assert "AIFS" in serial_stores
  with xr.open_zarr(serial_stores["AIFS"]) as sds:
    assert np.all(np.isfinite(sds["aifs_total_precipitation"]))


def test_precomputed_weights_matrix_cropping_with_stripped_worker_gdf():
  """Verifies Dask worker path where precomputed 0.5-deg buffered weights_matrix is cropped with stripped worker_gdf."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 49, 6))
  ds_aifs = _build_synthetic_ecmwf_dataset(["2026-03-01T00:00:00"], lead_hours)
  loader = DynamicalDataLoader(AIFSExtractor.DEFAULT_DATASET_ID, ds=ds_aifs)
  extractor = AIFSExtractor(loader=loader, lead_days=2)

  # Driver precomputes weights on a wider (0.5 deg buffered) grid slice
  wide_slices = loader.compute_spatial_slices(gdf, buffer=0.5)
  wide_lats = ds_aifs.latitude.values[wide_slices["latitude"]]
  wide_lons = ds_aifs.longitude.values[wide_slices["longitude"]]
  precomputed_wm = ZonalWeightMatrix.from_geodataframe(
      gdf, wide_lats, wide_lons, cell_res_lat=0.25, cell_res_lon=0.25
  )

  # Dask runner strips geometries to [bbox_geom, None] before scattering to workers
  minx, miny, maxx, maxy = gdf.total_bounds
  worker_gdf = gpd.GeoDataFrame(
      {"geometry": [box(minx, miny, maxx, maxy), None]},
      index=gdf.index,
      crs="EPSG:4326",
  )

  out_worker = extractor.extract_for_basins(
      worker_gdf,
      start_date="2026-03-01",
      end_date="2026-03-01",
      weights_matrix=precomputed_wm,
  )
  out_direct = extractor.extract_for_basins(
      gdf,
      start_date="2026-03-01",
      end_date="2026-03-01",
  )
  np.testing.assert_allclose(
      out_worker["aifs_total_precipitation"].values,
      out_direct["aifs_total_precipitation"].values,
      rtol=1e-6,
  )

