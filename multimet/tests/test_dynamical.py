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

import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple
import warnings

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
import xarray as xr

from multimet.timeseries_extractors.config import (
    ENSEMBLE_MEMBER_BANDS,
    ENSEMBLE_MISSING_FRACTION_VAR,
    ENSEMBLE_STAT_SUFFIXES,
    PRODUCT_BANDS,
    Product,
)
from multimet.timeseries_extractors.dask_runner import extract_product_dask
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
import multimet.timeseries_extractors.realtime as realtime_mod
from multimet.timeseries_extractors.realtime import (
    RealtimeForcingFetcher,
    fetch_realtime_multimet,
)
from multimet.timeseries_extractors.runner import extract_multimet_serial
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter
from multimet.utils.zonal import ZonalWeightMatrix

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class _MockDynamicalCatalog:
  """Typed mock for dynamical_catalog module in unit tests."""

  datasets: Tuple[str, ...] = (
      "ecmwf-aifs-single-forecast",
      "ecmwf-aifs-ens-forecast",
      "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
      "noaa-gfs-forecast",
      "noaa-gefs-forecast-35-day",
      "nasa-imerg-analysis-early",
  )
  opener: Callable[[str], xr.Dataset] | None = None

  def list(self) -> List[str]:
    return list(self.datasets)

  def open(self, dataset_id: str) -> xr.Dataset:
    if self.opener is None:
      raise RuntimeError(f"No mock dataset opener configured for {dataset_id}")
    return self.opener(dataset_id)


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
  """Builds a lazy Dask-backed synthetic ECMWF dataset on the native 0.25-deg global grid (721 x 1440)."""
  lats = np.linspace(90.0, -90.0, 721, dtype=np.float64)
  lons = np.linspace(-180.0, 179.75, 1440, dtype=np.float64)
  init_idx = pd.to_datetime(init_times)
  leads = np.array(
      [np.timedelta64(h, "h") for h in lead_hours], dtype="timedelta64[ns]"
  )

  n_t = len(init_idx)
  n_l = len(leads)
  n_y = len(lats)
  n_x = len(lons)

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
  east_mask = da.from_array((lons >= -97.0).astype(np.float32), chunks=16)

  if ensemble_members is None:
    dims = ("init_time", "lead_time", "latitude", "longitude")
    chunks = (1, n_l, 8, 16)
    for var_name, base_val in raw_base.items():
      east_delta = (
          (1.0 / 86400.0)
          if var_name == "precipitation_surface"
          else (1000.0 if var_name == "pressure_surface" else 1.0)
      )
      time_lead = np.full((n_t, n_l), base_val, dtype=np.float32)
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          time_lead[:, l_i] = np.nan
      if unpopulated_last_init:
        time_lead[-1, -1] = np.nan
      tl_da = da.from_array(time_lead[:, :, None, None], chunks=(1, n_l, 1, 1))
      spatial_da = da.zeros((1, 1, n_y, n_x), dtype=np.float32, chunks=(1, 1, 8, 16))
      arr_da = (tl_da + spatial_da + east_mask[None, None, None, :] * np.float32(east_delta)).rechunk(chunks)
      data_vars[var_name] = (dims, arr_da)
  else:
    coords["ensemble_member"] = np.arange(ensemble_members, dtype=np.int64)
    dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    chunks = (1, n_l, ensemble_members, 8, 16)
    member_offsets = np.arange(ensemble_members, dtype=np.float32)
    for var_name, base_val in raw_base.items():
      scale = (
          (1.0 / 86400.0)
          if var_name == "precipitation_surface"
          else (1000.0 if var_name == "pressure_surface" else 1.0)
      )
      tlm = (
          np.full((n_t, n_l, ensemble_members), base_val, dtype=np.float32)
          + member_offsets[None, None, :] * np.float32(scale)
      )
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          tlm[:, l_i, :] = np.nan
      if unpopulated_last_init:
        tlm[-1, -1, :] = np.nan
      tlm_da = da.from_array(
          tlm[:, :, :, None, None], chunks=(1, n_l, ensemble_members, 1, 1)
      )
      spatial_da = da.zeros(
          (1, 1, 1, n_y, n_x), dtype=np.float32, chunks=(1, 1, 1, 8, 16)
      )
      arr_da = (tlm_da + spatial_da).rechunk(chunks)
      data_vars[var_name] = (dims, arr_da)

  return xr.Dataset(data_vars, coords=coords)


def _build_synthetic_noaa_dataset(
    init_times: Sequence[str],
    lead_hours: Sequence[int],
    ensemble_members: int | None = None,
) -> xr.Dataset:
  """Builds a lazy Dask-backed synthetic NOAA dataset on the native 0.25-deg global grid (721 x 1440)."""
  lats = np.linspace(90.0, -90.0, 721, dtype=np.float64)
  lons = np.linspace(-180.0, 179.75, 1440, dtype=np.float64)
  init_idx = pd.to_datetime(init_times)
  leads = np.array(
      [np.timedelta64(h, "h") for h in lead_hours], dtype="timedelta64[ns]"
  )

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
      tl = np.full((n_t, n_l), base_val, dtype=np.float32)
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          tl[:, l_i] = np.nan
        elif h > 0:
          hour_of_day = ((h - 1) % 24) + 1
          if var_name == "maximum_temperature_2m":
            tl[:, l_i] = base_val + (6.0 if hour_of_day == 24 else 1.0)
          elif var_name == "minimum_temperature_2m":
            tl[:, l_i] = base_val - (5.0 if hour_of_day == 24 else 1.0)
      tl_da = da.from_array(tl[:, :, None, None], chunks=(1, n_l, 1, 1))
      spatial_da = da.zeros(
          (1, 1, n_y, n_x), dtype=np.float32, chunks=(1, 1, 8, 16)
      )
      data_vars[var_name] = (dims, (tl_da + spatial_da).rechunk(chunks))
  else:
    coords["ensemble_member"] = np.arange(ensemble_members, dtype=np.int64)
    dims = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")
    chunks = (1, ensemble_members, n_l, 8, 16)
    member_offsets = np.arange(ensemble_members, dtype=np.float32)
    for var_name, base_val in raw_base.items():
      scale = (
          (1.0 / 86400.0)
          if var_name == "precipitation_surface"
          else (1000.0 if var_name == "pressure_surface" else 1.0)
      )
      tml = np.full(
          (n_t, ensemble_members, n_l), base_val, dtype=np.float32
      )
      for l_i, h in enumerate(lead_hours):
        if h == 0 and var_name in step_interval_vars:
          tml[:, :, l_i] = np.nan
        elif h > 0:
          hour_of_day = ((h - 1) % 24) + 1
          if var_name == "maximum_temperature_2m":
            tml[:, :, l_i] = (
                base_val
                + (6.0 if hour_of_day == 24 else 1.0)
                + member_offsets[None, :] * scale
            )
          elif var_name == "minimum_temperature_2m":
            tml[:, :, l_i] = (
                base_val
                - (5.0 if hour_of_day == 24 else 1.0)
                + member_offsets[None, :] * scale
            )
          else:
            tml[:, :, l_i] = base_val + member_offsets[None, :] * scale
        else:
          tml[:, :, l_i] = base_val + member_offsets[None, :] * scale
      tml_da = da.from_array(
          tml[:, :, :, None, None], chunks=(1, ensemble_members, n_l, 1, 1)
      )
      spatial_da = da.zeros(
          (1, 1, 1, n_y, n_x), dtype=np.float32, chunks=(1, 1, 1, 8, 16)
      )
      data_vars[var_name] = (dims, (tml_da + spatial_da).rechunk(chunks))

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
      _MockDynamicalCatalog(opener=lambda _id: ds),
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
  np.testing.assert_allclose(
      out["aifs_surface_pressure"].sel(basin="basin_west"), 100.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["aifs_surface_pressure"].sel(basin="basin_east"), 101.0, rtol=1e-5
  )
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


def test_trapezoidal_state_integration_and_noaa_6h_reset_radiation_mean():
  """Verifies trapezoidal integration over [(d-1)*24h, d*24h] for state vars and 6h reset mean for NOAA radiation."""
  gdf = _make_disjoint_basins_gdf()
  # Use 3-hourly steps across 2 lead days (0..48h)
  lead_hours = list(range(0, 49, 3))
  ds = _build_synthetic_noaa_dataset(["2026-03-01T00:00:00"], lead_hours)

  # Make temperature_2m vary across lead day 1 [0..24h]:
  # h=0 -> 10.0 C, h=3..21 -> 20.0 C, h=24 -> 30.0 C.
  # Trapezoidal integral over [0, 24]:
  # (1.5*10 + 3*(20*7) + 1.5*30) / 24 = (15 + 420 + 45) / 24 = 480 / 24 = 20.0 C
  # (whereas a naive right-endpoint mean over h=3..24 would give (20*7 + 30)/8 = 21.25 C).
  temp_profile = np.full((1, len(lead_hours), 1, 1), 20.0, dtype=np.float32)
  temp_profile[:, 0, :, :] = 10.0  # h=0
  temp_profile[:, 8, :, :] = 30.0  # h=24 (also start of day 2!)
  temp_profile[:, 16, :, :] = 10.0  # h=48
  ds["temperature_2m"] = (
      ("init_time", "lead_time", "latitude", "longitude"),
      (
          da.from_array(temp_profile, chunks=(1, len(lead_hours), 1, 1))
          + da.zeros((1, 1, ds.sizes["latitude"], ds.sizes["longitude"]), dtype=np.float32, chunks=(1, 1, 8, 16))
      ).rechunk((1, len(lead_hours), 8, 16)),
  )

  # Make NOAA downward_short_wave_radiation_flux_surface alternate between:
  # - 3h intermediate steps (h=3, 9, 15, 21, ...): 100.0 W/m^2 (0-3h means)
  # - 6h synoptic reset steps (h=6, 12, 18, 24, ...): 240.0 W/m^2 (0-6h means)
  # Exact daily mean over 24h is the mean of the four 6h reset steps = 240.0 W/m^2
  # (whereas averaging all eight 3h+6h steps would double-count the first 3h and give 170.0 W/m^2).
  sw_profile = np.full((1, len(lead_hours), 1, 1), np.nan, dtype=np.float32)
  for l_i, h in enumerate(lead_hours):
    if h > 0:
      sw_profile[:, l_i, :, :] = 240.0 if (h % 6 == 0) else 100.0
  ds["downward_short_wave_radiation_flux_surface"] = (
      ("init_time", "lead_time", "latitude", "longitude"),
      (
          da.from_array(sw_profile, chunks=(1, len(lead_hours), 1, 1))
          + da.zeros((1, 1, ds.sizes["latitude"], ds.sizes["longitude"]), dtype=np.float32, chunks=(1, 1, 8, 16))
      ).rechunk((1, len(lead_hours), 8, 16)),
  )

  loader = DynamicalDataLoader(GFSExtractor.DEFAULT_DATASET_ID, ds=ds)
  extractor = GFSExtractor(loader=loader, lead_days=2)
  out = extractor.extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-01"
  )

  np.testing.assert_allclose(
      out["gfs_temperature_2m"].isel(date=0, lead_time=0), 20.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["gfs_temperature_2m"].isel(date=0, lead_time=1), 20.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      out["gfs_downward_short_wave_radiation"].isel(date=0), 240.0, rtol=1e-5
  )


def test_gfs_extractor_nonuniform_leads_min_max_temp_and_spinup_cutoff():
  """Verifies GFSExtractor handles 1h+3h steps, daily Tmin/Tmax, and spinup_only_before."""
  gdf = _make_disjoint_basins_gdf()
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

  d0 = out.isel(date=0)
  assert np.all(np.isfinite(d0["gfs_total_precipitation"].isel(lead_time=0)))
  assert np.all(np.isnan(d0["gfs_total_precipitation"].isel(lead_time=slice(1, None))))
  np.testing.assert_allclose(
      d0["gfs_missing_fraction"].isel(lead_time=0), 0.0, atol=1e-6
  )
  np.testing.assert_allclose(
      d0["gfs_missing_fraction"].isel(lead_time=slice(1, None)), 1.0
  )

  d1 = out.isel(date=1)
  assert np.all(np.isfinite(d1["gfs_total_precipitation"]))
  np.testing.assert_allclose(d1["gfs_total_precipitation"], 4.0, rtol=1e-5)
  np.testing.assert_allclose(d1["gfs_surface_pressure"], 98.0, rtol=1e-5)
  np.testing.assert_allclose(d1["gfs_temperature_2m_max"], 24.0, rtol=1e-5)
  np.testing.assert_allclose(d1["gfs_temperature_2m_min"], 7.0, rtol=1e-5)
  np.testing.assert_allclose(d1["gfs_missing_fraction"], 0.0, atol=1e-6)


def test_ensemble_extractors_summary_stats_and_optional_4d_members(tmp_path: Path):
  """Verifies IFS_ENS, AIFS_ENS, and GEFS compute all 7 ensemble stats (ddof=1) and optional 4D member arrays."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 241, 6))
  n_members = 11

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
  assert set(ENSEMBLE_MEMBER_BANDS[Product.IFS_ENS]).issubset(
      set(ifs_out.data_vars)
  )
  assert ENSEMBLE_MISSING_FRACTION_VAR[Product.IFS_ENS] in ifs_out.data_vars
  assert ifs_out["ifs_ens_total_precipitation_ensemble"].dims == (
      "basin",
      "date",
      "ensemble_member",
      "lead_time",
  )
  assert ifs_out["ifs_ens_missing_fraction_ensemble"].dims == (
      "basin",
      "date",
      "ensemble_member",
      "lead_time",
  )
  assert ifs_out.sizes["ensemble_member"] == n_members

  expected_members = np.arange(2.0, 2.0 + n_members, dtype=np.float32)
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_mean"].sel(basin="basin_west"),
      float(np.mean(expected_members)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      ifs_out["ifs_ens_total_precipitation_std"].sel(basin="basin_west"),
      float(np.std(expected_members, ddof=1)),
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

  writer = MultiMetZarrWriter(str(tmp_path))
  store_path = writer.write_or_append(ifs_out, Product.IFS_ENS)
  with xr.open_zarr(store_path) as reloaded:
    assert "ifs_ens_total_precipitation_mean" in reloaded.data_vars
    assert "ifs_ens_total_precipitation_ensemble" in reloaded.data_vars
    assert "ifs_ens_missing_fraction_ensemble" in reloaded.data_vars
    assert reloaded["ifs_ens_total_precipitation_ensemble"].shape == (
        2,
        1,
        n_members,
        10,
    )
    assert reloaded["ifs_ens_missing_fraction_ensemble"].shape == (
        2,
        1,
        n_members,
        10,
    )

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
  np.testing.assert_allclose(
      gefs_out["gefs_temperature_2m_max_mean"], 29.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      gefs_out["gefs_temperature_2m_max_min"], 24.0, rtol=1e-5
  )
  np.testing.assert_allclose(
      gefs_out["gefs_temperature_2m_max_max"], 34.0, rtol=1e-5
  )

  aifs_ens_loader = DynamicalDataLoader(
      AIFSEnsExtractor.DEFAULT_DATASET_ID, ds=ifs_ds
  )
  aifs_ens_out = AIFSEnsExtractor(loader=aifs_ens_loader).extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-01"
  )
  assert set(PRODUCT_BANDS[Product.AIFS_ENS]).issubset(
      set(aifs_ens_out.data_vars)
  )


def test_ensemble_partial_and_all_nan_member_dropout_without_warnings():
  """Verifies missing ensemble members are dropped cleanly (ddof=1 requires >=2 valid, stats require >=1) with zero RuntimeWarnings."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 73, 6))  # 3 lead days (0..72h)
  n_members = 3
  ds = _build_synthetic_ecmwf_dataset(
      ["2026-03-01T00:00:00"], lead_hours, ensemble_members=n_members
  )

  # Inject member-specific NaNs in precipitation_surface:
  # - Lead day 1 (h=6..24, indices 1..4): member 0 is NaN, members 1 & 2 are valid (3.0 and 4.0 mm/day)
  # - Lead day 2 (h=30..48, indices 5..8): members 0 & 2 are NaN, only member 1 is valid (3.0 mm/day)
  # - Lead day 3 (h=54..72, indices 9..12): all 3 members are NaN
  tlm = np.full((1, len(lead_hours), n_members), np.nan, dtype=np.float32)
  for m in range(n_members):
    tlm[0, 1:, m] = (2.0 + m) / 86400.0
  tlm[0, 1:5, 0] = np.nan
  tlm[0, 5:9, 0] = np.nan
  tlm[0, 5:9, 2] = np.nan
  tlm[0, 9:13, :] = np.nan

  tlm_da = da.from_array(
      tlm[:, :, :, None, None], chunks=(1, len(lead_hours), n_members, 1, 1)
  )
  spatial_da = da.zeros(
      (1, 1, 1, ds.sizes["latitude"], ds.sizes["longitude"]),
      dtype=np.float32,
      chunks=(1, 1, 1, 8, 16),
  )
  ds["precipitation_surface"] = (
      ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
      (tlm_da + spatial_da).rechunk((1, len(lead_hours), n_members, 8, 16)),
  )

  loader = DynamicalDataLoader(IFSEnsExtractor.DEFAULT_DATASET_ID, ds=ds)
  extractor = IFSEnsExtractor(
      loader=loader, lead_days=3, include_ensemble_members=True
  )

  with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    out = extractor.extract_for_basins(
        gdf, start_date="2026-03-01", end_date="2026-03-01"
    )

  # Lead day 1 (index 0): valid members are [3.0, 4.0]
  np.testing.assert_allclose(
      out["ifs_ens_total_precipitation_mean"].isel(date=0, lead_time=0),
      3.5,
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["ifs_ens_total_precipitation_std"].isel(date=0, lead_time=0),
      float(np.std([3.0, 4.0], ddof=1)),
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["ifs_ens_missing_fraction_ensemble"].isel(basin=0, date=0, lead_time=0),
      [1.0, 0.0, 0.0],
      atol=1e-6,
  )
  np.testing.assert_allclose(
      out["ifs_ens_missing_fraction"].isel(date=0, lead_time=0),
      1.0 / 3.0,
      rtol=1e-5,
  )

  # Lead day 2 (index 1): only 1 valid member [3.0] -> mean/min/max/p50 = 3.0, std(ddof=1) = NaN
  np.testing.assert_allclose(
      out["ifs_ens_total_precipitation_mean"].isel(date=0, lead_time=1),
      3.0,
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      out["ifs_ens_total_precipitation_p50"].isel(date=0, lead_time=1),
      3.0,
      rtol=1e-5,
  )
  assert np.all(
      np.isnan(out["ifs_ens_total_precipitation_std"].isel(date=0, lead_time=1))
  )
  np.testing.assert_allclose(
      out["ifs_ens_missing_fraction"].isel(date=0, lead_time=1),
      2.0 / 3.0,
      rtol=1e-5,
  )

  # Lead day 3 (index 2): 0 valid members -> all stats NaN, missing_fraction = 1.0
  for suffix in ENSEMBLE_STAT_SUFFIXES:
    assert np.all(
        np.isnan(
            out[f"ifs_ens_total_precipitation_{suffix}"].isel(date=0, lead_time=2)
        )
    )
  np.testing.assert_allclose(
      out["ifs_ens_missing_fraction"].isel(date=0, lead_time=2),
      1.0,
      atol=1e-6,
  )


def test_dynamical_imerg_extractor_half_hourly_to_daily_and_incomplete_day():
  """Verifies DynamicalIMERGExtractor aggregates 48 half-hour steps on a 0.1-deg global grid (1800 x 3600) and aligns 30-day windows."""
  gdf = _make_disjoint_basins_gdf()
  lats = np.linspace(89.95, -89.95, 1800, dtype=np.float64)
  lons = np.linspace(-179.95, 179.95, 3600, dtype=np.float64)

  times_full = pd.date_range(
      "2026-03-01T00:00:00", "2026-03-02T23:30:00", freq="30min"
  )
  times_partial = pd.date_range(
      "2026-03-03T00:00:00", periods=20, freq="30min"
  )
  times = times_full.append(times_partial)

  east_mask = da.from_array((lons >= -97.0).astype(np.float32), chunks=30)
  base_rate = np.float32(36.0 / 86400.0)
  east_delta = np.float32((84.0 - 36.0) / 86400.0)
  time_da = da.full(
      (len(times), 1, 1), base_rate, dtype=np.float32, chunks=(48, 1, 1)
  )
  spatial_da = da.zeros(
      (1, len(lats), len(lons)), dtype=np.float32, chunks=(1, 15, 30)
  )
  arr_da = (
      time_da + spatial_da + east_mask[None, None, :] * east_delta
  ).rechunk((48, 15, 30))

  imerg_ds = xr.Dataset(
      {"precipitation_surface": (("time", "latitude", "longitude"), arr_da)},
      coords={"time": times, "latitude": lats, "longitude": lons},
  )

  loader = DynamicalDataLoader(
      DynamicalIMERGExtractor.DEFAULT_DATASET_ID, ds=imerg_ds
  )
  # Verify _iter_chunk_aligned_windows never straddles a 30-day boundary even when batch_days < 30
  # Anchor is 2026-03-01; chunk 0 is 2026-03-01..2026-03-30, chunk 1 starts 2026-03-31.
  windows = DynamicalIMERGExtractor(
      loader=loader, batch_days=5
  )._iter_chunk_aligned_windows(
      pd.Timestamp("2026-03-28"), pd.Timestamp("2026-04-02")
  )
  assert windows == [
      (pd.Timestamp("2026-03-28"), pd.Timestamp("2026-03-30")),
      (pd.Timestamp("2026-03-31"), pd.Timestamp("2026-04-02")),
  ]

  extractor = DynamicalIMERGExtractor(loader=loader, batch_days=1)
  out = extractor.extract_for_basins(
      gdf, start_date="2026-03-01", end_date="2026-03-03"
  )

  assert "dynamical_imerg_precipitation" in out.data_vars
  assert "dynamical_imerg_missing_fraction" in out.data_vars
  assert "imerg_precipitation" not in out.data_vars
  assert out["dynamical_imerg_precipitation"].dims == ("basin", "date")
  assert out.sizes["date"] == 3

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

  assert np.all(np.isnan(out["dynamical_imerg_precipitation"].isel(date=2)))
  np.testing.assert_allclose(
      out["dynamical_imerg_missing_fraction"].isel(date=2), 1.0
  )


def test_find_latest_dynamical_forecast_date_and_realtime_fetcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  """Verifies find_latest_dynamical_forecast_date, multi-product t0 resolution, and hotstart 4D ensemble updates."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 241, 6))
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

  # Verify multi-product reference date resolution returns min(HRES_t0, AIFS_t0)
  monkeypatch.setattr(
      realtime_mod,
      "find_latest_hres_open_data_date",
      lambda **kwargs: pd.Timestamp("2026-03-03"),
  )
  mixed_fetcher = RealtimeForcingFetcher(
      output_dir=tmp_path / "mixed_check",
      dynamical_loaders={"AIFS": aifs_loader},
  )
  assert mixed_fetcher.resolve_reference_date(
      "latest", products=["HRES", "AIFS"]
  ) == pd.Timestamp("2026-03-02")

  # Verify nowcast-only products=["DYNAMICAL_IMERG"] does not query HRES
  def _forbid_hres(**kwargs) -> pd.Timestamp:
    raise AssertionError("HRES open data should not be queried for DYNAMICAL_IMERG")

  monkeypatch.setattr(realtime_mod, "find_latest_hres_open_data_date", _forbid_hres)
  imerg_times = pd.date_range(
      "2026-03-01T00:00:00", "2026-03-02T23:30:00", freq="30min"
  )
  imerg_ds = xr.Dataset(
      {
          "precipitation_surface": (
              ("time", "latitude", "longitude"),
              da.zeros((len(imerg_times), 4, 4), dtype=np.float32),
          )
      },
      coords={
          "time": imerg_times,
          "latitude": np.array([40.5, 40.0, 35.5, 35.0]),
          "longitude": np.array([-120.0, -119.5, -75.0, -74.5]),
      },
  )
  imerg_loader = DynamicalDataLoader(
      DynamicalIMERGExtractor.DEFAULT_DATASET_ID, ds=imerg_ds
  )
  nowcast_fetcher = RealtimeForcingFetcher(
      output_dir=tmp_path / "nowcast_check",
      dynamical_loaders={"DYNAMICAL_IMERG": imerg_loader},
  )
  assert nowcast_fetcher.resolve_reference_date(
      "latest", products=["DYNAMICAL_IMERG"]
  ) == pd.Timestamp("2026-03-02")

  # Verify incremental hotstart with 4D ensemble members (same-date rewrite + next-day append)
  ds_ifs_ens = _build_synthetic_ecmwf_dataset(
      ["2026-03-01T00:00:00", "2026-03-02T00:00:00"],
      lead_hours,
      ensemble_members=4,
  )
  ifs_ens_loader = DynamicalDataLoader(
      IFSEnsExtractor.DEFAULT_DATASET_ID, ds=ds_ifs_ens
  )
  rt_ens_dir = tmp_path / "realtime_ens"
  res_day1 = fetch_realtime_multimet(
      basins=gdf,
      output_dir=rt_ens_dir,
      mode="hotstart",
      reference_date="2026-03-01",
      lookback_days=0,
      products=["IFS_ENS"],
      full_forecast_days=1,
      include_ensemble_members=True,
      dynamical_loaders={"IFS_ENS": ifs_ens_loader},
  )
  # Repeat on same date (direct chunk overwrite) and then advance to 2026-03-02 (append_dates)
  fetch_realtime_multimet(
      basins=gdf,
      output_dir=rt_ens_dir,
      mode="hotstart",
      reference_date="2026-03-01",
      lookback_days=0,
      products=["IFS_ENS"],
      full_forecast_days=1,
      include_ensemble_members=True,
      dynamical_loaders={"IFS_ENS": ifs_ens_loader},
  )
  res_day2 = fetch_realtime_multimet(
      basins=gdf,
      output_dir=rt_ens_dir,
      mode="hotstart",
      reference_date="2026-03-02",
      lookback_days=1,
      products=["IFS_ENS"],
      full_forecast_days=2,
      include_ensemble_members=True,
      dynamical_loaders={"IFS_ENS": ifs_ens_loader},
  )
  with xr.open_zarr(res_day2.stores["IFS_ENS"]) as z_ens:
    assert z_ens.sizes["date"] == 2
    assert z_ens.sizes["ensemble_member"] == 4
    assert np.all(
        np.isfinite(z_ens["ifs_ens_total_precipitation_ensemble"].values)
    )
    assert np.all(
        np.isfinite(z_ens["ifs_ens_missing_fraction_ensemble"].values)
    )

  # Also verify serial and Dask runners with dynamical products
  monkeypatch.setattr(
      dyn_mod,
      "dynamical_catalog",
      _MockDynamicalCatalog(
          opener=lambda dataset_id: (
              ds_ifs_ens
              if dataset_id == IFSEnsExtractor.DEFAULT_DATASET_ID
              else ds_aifs
          )
      ),
  )
  serial_stores = extract_multimet_serial(
      basins=gdf,
      start_date="2026-03-01",
      end_date="2026-03-02",
      output_dir=str(tmp_path / "serial_out"),
      products=["AIFS", "IFS_ENS"],
      include_ensemble_members=True,
  )
  with xr.open_zarr(serial_stores["IFS_ENS"]) as sds:
    assert np.all(np.isfinite(sds["ifs_ens_total_precipitation_mean"].values))
    assert np.all(
        np.isfinite(sds["ifs_ens_total_precipitation_ensemble"].values)
    )
    assert np.all(np.isfinite(sds["ifs_ens_missing_fraction_ensemble"].values))

  # Verify Dask runner with in-process worker execution on 4D ensemble output
  import distributed

  with distributed.Client(
      processes=False,
      n_workers=1,
      threads_per_worker=1,
      dashboard_address=None,
  ) as dask_client:
    dask_store = extract_product_dask(
        product="IFS_ENS",
        basins=gdf,
        output_dir=str(tmp_path / "dask_out"),
        start_date="2026-03-01",
        end_date="2026-03-02",
        client=dask_client,
        include_ensemble_members=True,
        show_progress=False,
        loader=ifs_ens_loader,
    )
  with xr.open_zarr(dask_store) as dds:
    assert dds.sizes["date"] == 2
    assert dds.sizes["ensemble_member"] == 4
    assert np.all(np.isfinite(dds["ifs_ens_total_precipitation_mean"].values))
    assert np.all(
        np.isfinite(dds["ifs_ens_total_precipitation_ensemble"].values)
    )
    assert np.all(np.isfinite(dds["ifs_ens_missing_fraction_ensemble"].values))


def test_precomputed_weights_matrix_cropping_with_stripped_worker_gdf():
  """Verifies Dask worker path where precomputed 0.5-deg buffered weights_matrix is cropped with stripped worker_gdf."""
  gdf = _make_disjoint_basins_gdf()
  lead_hours = list(range(0, 49, 6))
  ds_aifs = _build_synthetic_ecmwf_dataset(["2026-03-01T00:00:00"], lead_hours)
  loader = DynamicalDataLoader(AIFSExtractor.DEFAULT_DATASET_ID, ds=ds_aifs)
  extractor = AIFSExtractor(loader=loader, lead_days=2)

  wide_slices = loader.compute_spatial_slices(gdf, buffer=0.5)
  wide_lats = ds_aifs.latitude.values[wide_slices["latitude"]]
  wide_lons = ds_aifs.longitude.values[wide_slices["longitude"]]
  precomputed_wm = ZonalWeightMatrix.from_geodataframe(
      gdf, wide_lats, wide_lons, cell_res_lat=0.25, cell_res_lon=0.25
  )

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
