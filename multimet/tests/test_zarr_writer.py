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

"""Unit tests for MultiMet Zarr store writer and schema validator."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet.timeseries_extractors.config import (
    PRODUCT_BANDS,
    Product,
)
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter


pytestmark = pytest.mark.unit


@pytest.fixture
def writer(tmp_path) -> MultiMetZarrWriter:
  return MultiMetZarrWriter(tmp_path)


def test_write_and_append_nowcast(writer: MultiMetZarrWriter):
  basins_1 = ["basin_A", "basin_B"]
  dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
  precip_1 = np.random.rand(len(basins_1), len(dates)).astype(np.float32)

  ds1 = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], precip_1),
      },
      coords={
          "basin": basins_1,
          "date": dates.values,
      },
  )

  # 1. Write initial dataset
  store_path = writer.write_or_append(ds1, Product.CPC)
  assert Path(store_path).as_posix().endswith("CPC/timeseries.zarr")

  read_ds = xr.open_zarr(store_path)
  assert list(read_ds["basin"].values) == basins_1
  assert len(read_ds["date"]) == 10
  assert read_ds["cpc_precipitation"].dtype == np.float32

  # 2. Append new basin_C
  basins_2 = ["basin_C"]
  precip_2 = np.random.rand(len(basins_2), len(dates)).astype(np.float32)
  ds2 = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], precip_2),
      },
      coords={
          "basin": basins_2,
          "date": dates.values,
      },
  )

  writer.write_or_append(ds2, Product.CPC)

  read_ds2 = xr.open_zarr(store_path)
  assert list(read_ds2["basin"].values) == ["basin_A", "basin_B", "basin_C"]
  assert read_ds2["cpc_precipitation"].shape == (3, 10)


def test_write_and_append_forecast(writer: MultiMetZarrWriter):
  basins_1 = ["basin_X"]
  dates = pd.date_range("2021-01-01", "2021-01-05", freq="D")
  leads = pd.to_timedelta(range(1, 11), unit="D")
  shape = (len(basins_1), len(dates), len(leads))

  data_vars = {
      band: (
          ["basin", "date", "lead_time"],
          np.random.rand(*shape).astype(np.float32),
      )
      for band in PRODUCT_BANDS[Product.HRES]
  }

  ds = xr.Dataset(
      data_vars=data_vars,
      coords={
          "basin": basins_1,
          "date": dates.values,
          "lead_time": leads.values,
      },
  )

  store_path = writer.write_or_append(ds, Product.HRES)
  assert Path(store_path).as_posix().endswith("HRES/timeseries.zarr")

  read_ds = xr.open_zarr(store_path)
  assert "lead_time" in read_ds.coords
  assert len(read_ds["lead_time"]) == 10
  for band in PRODUCT_BANDS[Product.HRES]:
    assert band in read_ds.data_vars


def test_schema_validation_failure(writer: MultiMetZarrWriter):
  # Missing band
  ds_bad = xr.Dataset(
      data_vars={"wrong_var": (["basin", "date"], np.zeros((2, 2), dtype=np.float32))},
      coords={"basin": ["b1", "b2"], "date": pd.date_range("2020-01-01", periods=2)},
  )
  with pytest.raises(ValueError):
    writer.write_or_append(ds_bad, Product.CPC)


def test_append_dates_nowcast(writer: MultiMetZarrWriter):
  basins = ["basin_1", "basin_2"]
  dates_initial = pd.date_range("2020-01-01", periods=5, freq="D")
  ds_init = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((2, 5), dtype=np.float32)),
      },
      coords={"basin": basins, "date": dates_initial.values},
  )
  store_path = writer.write_or_append(ds_init, Product.CPC)

  # Append 3 new dates
  dates_new = pd.date_range("2020-01-06", periods=3, freq="D")
  start_idx, end_idx = writer.append_dates(Product.CPC, dates_new)
  assert start_idx == 5
  assert end_idx == 8

  ds_extended = xr.open_zarr(store_path)
  assert len(ds_extended["date"]) == 8
  assert ds_extended["cpc_precipitation"].shape == (2, 8)
  # Verify original values preserved
  assert np.all(ds_extended["cpc_precipitation"].values[:, :5] == 1.0)
  # Verify new slots are NaN before write
  assert np.all(np.isnan(ds_extended["cpc_precipitation"].values[:, 5:]))

  # Verify direct chunk write into new slots works
  writer.write_direct_chunk(Product.CPC, "cpc_precipitation", 5, np.array([42.0, 43.0], dtype=np.float32))
  writer.consolidate_metadata(Product.CPC)

  ds_written = xr.open_zarr(store_path)
  assert np.all(ds_written["cpc_precipitation"].values[:, 5] == [42.0, 43.0])


def test_append_basins_nowcast(writer: MultiMetZarrWriter):
  basins_initial = ["basin_A", "basin_B"]
  dates = pd.date_range("2020-01-01", periods=5, freq="D")
  ds_init = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((2, 5), dtype=np.float32)),
      },
      coords={"basin": basins_initial, "date": dates.values},
  )
  store_path = writer.write_or_append(ds_init, Product.CPC)

  # Append new basin_C and basin_D over the SAME dates
  basins_new = ["basin_C", "basin_D"]
  ds_new = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.full((2, 5), 88.0, dtype=np.float32)),
      },
      coords={"basin": basins_new, "date": dates.values},
  )
  writer.append_basins(Product.CPC, ds_new)

  ds_result = xr.open_zarr(store_path)
  assert list(ds_result["basin"].values) == ["basin_A", "basin_B", "basin_C", "basin_D"]
  assert ds_result["cpc_precipitation"].shape == (4, 5)
  assert np.all(ds_result["cpc_precipitation"].sel(basin="basin_C").values == 88.0)


def test_reject_simultaneous_2d_expansion(writer: MultiMetZarrWriter):
  basins_initial = ["basin_1", "basin_2"]
  dates_initial = pd.date_range("2020-01-01", periods=5, freq="D")
  ds_init = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((2, 5), dtype=np.float32)),
      },
      coords={"basin": basins_initial, "date": dates_initial.values},
  )
  writer.write_or_append(ds_init, Product.CPC)

  # Attempt to append BOTH new basins AND new dates
  basins_new = ["basin_3"]
  dates_new = pd.date_range("2020-01-06", periods=3, freq="D")
  ds_both = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], np.ones((1, 3), dtype=np.float32)),
      },
      coords={"basin": basins_new, "date": dates_new.values},
  )

  with pytest.raises(ValueError, match="Cannot add both new basins and new dates"):
    writer.write_or_append(ds_both, Product.CPC)


def test_expand_date_range_prepending_and_overlap(writer: MultiMetZarrWriter):
  basins = ["basin_1"]
  dates = pd.date_range("2020-01-10", periods=5, freq="D")
  ds = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 5), dtype=np.float32))},
      coords={"basin": basins, "date": dates.values},
  )
  writer.write_or_append(ds, Product.CPC)

  # 1. Prepend earlier dates: 2020-01-01 to 2020-01-03
  earlier_dates = pd.date_range("2020-01-01", periods=3, freq="D")
  all_dates, indices = writer.expand_date_range(Product.CPC, earlier_dates)

  assert len(all_dates) == 8
  assert indices == [0, 1, 2]

  store_path = writer.get_store_path(Product.CPC)
  with xr.open_zarr(store_path) as res_ds:
    # First 3 days should be NaN (unwritten skeleton)
    assert np.isnan(res_ds["cpc_precipitation"].values[0, :3]).all()
    # Days 3..8 (the original 5 dates) should be preserved
    np.testing.assert_array_equal(
        res_ds["cpc_precipitation"].values[0, 3:],
        np.ones(5, dtype=np.float32),
    )

  # 2. Prepending via append_dates also seamlessly delegates to expand_date_range
  even_earlier = pd.date_range("2019-12-30", periods=2, freq="D")
  start_idx, end_idx = writer.append_dates(Product.CPC, even_earlier)
  assert start_idx == 0
  assert end_idx == 2
  info = writer.get_store_info(Product.CPC)
  assert len(info["dates"]) == 10
  with xr.open_zarr(store_path) as res_ds2:
    # Original data at 2020-01-10..14 should now be shifted to indices 5..10
    np.testing.assert_array_equal(
        res_ds2["cpc_precipitation"].values[0, 5:],
        np.ones(5, dtype=np.float32),
    )
    # The earlier 5 days should be NaN
    assert np.isnan(res_ds2["cpc_precipitation"].values[0, :5]).all()

  # 3. Partial overlap without expanding date range
  overlap_dates = pd.date_range("2020-01-10", periods=3, freq="D")
  all_dates3, indices3 = writer.expand_date_range(Product.CPC, overlap_dates)
  assert len(all_dates3) == 10
  assert indices3 == [5, 6, 7]


def test_reject_mismatched_dates_on_append_basins(writer: MultiMetZarrWriter):
  basins = ["basin_1"]
  dates = pd.date_range("2020-01-01", periods=5, freq="D")
  ds = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 5), dtype=np.float32))},
      coords={"basin": basins, "date": dates.values},
  )
  writer.write_or_append(ds, Product.CPC)

  ds_mismatched_dates = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 4), dtype=np.float32))},
      coords={"basin": ["basin_2"], "date": pd.date_range("2020-01-01", periods=4).values},
  )
  with pytest.raises(ValueError, match="dates mismatch"):
    writer.append_basins(Product.CPC, ds_mismatched_dates)


def test_write_or_append_prepending_and_rewriting(writer: MultiMetZarrWriter):
  """Verifies write_or_append supports prepending earlier dates and updating overlapping dates."""
  basins = ["basin_1"]
  # Initial dataset: 2020-01-05 to 2020-01-07 (3 days, all 1.0)
  dates_init = pd.date_range("2020-01-05", periods=3, freq="D")
  ds_init = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.ones((1, 3), dtype=np.float32))},
      coords={"basin": basins, "date": dates_init.values},
  )
  writer.write_or_append(ds_init, Product.CPC)

  # Second dataset: 2020-01-03 to 2020-01-05 (prepending 03, 04, overlapping 05 with 2.0)
  dates_update = pd.date_range("2020-01-03", periods=3, freq="D")
  ds_update = xr.Dataset(
      data_vars={"cpc_precipitation": (["basin", "date"], np.full((1, 3), 2.0, dtype=np.float32))},
      coords={"basin": basins, "date": dates_update.values},
  )
  writer.write_or_append(ds_update, Product.CPC)

  store_path = writer.get_store_path(Product.CPC)
  with xr.open_zarr(store_path) as res_ds:
    # Dates should now span 2020-01-03 to 2020-01-07 (5 days)
    assert len(res_ds["date"]) == 5
    vals = res_ds["cpc_precipitation"].values[0]
    # 2020-01-03, 04, 05 should be 2.0
    np.testing.assert_array_equal(vals[:3], [2.0, 2.0, 2.0])
    # 2020-01-06, 07 should still be 1.0
    np.testing.assert_array_equal(vals[3:], [1.0, 1.0])


def test_expand_date_range_forecast_product(writer: MultiMetZarrWriter):
  """Verifies date expansion (prepending and postpending) works on 3D forecast stores."""
  basins = ["basin_A"]
  dates = pd.date_range("2020-01-05", periods=2, freq="D")
  leads = pd.to_timedelta(range(1, 11), unit="D")
  shape = (len(basins), len(dates), len(leads))
  data_vars = {
      band: (
          ["basin", "date", "lead_time"],
          np.ones(shape, dtype=np.float32),
      )
      for band in PRODUCT_BANDS[Product.HRES]
  }
  ds = xr.Dataset(
      data_vars=data_vars,
      coords={"basin": basins, "date": dates.values, "lead_time": leads.values},
  )
  writer.write_or_append(ds, Product.HRES)

  # Prepend 2 days and postpend 1 day
  req_dates = pd.date_range("2020-01-03", periods=5, freq="D")
  all_dates, indices = writer.expand_date_range(Product.HRES, req_dates)
  assert len(all_dates) == 5
  assert indices == [0, 1, 2, 3, 4]

  store_path = writer.get_store_path(Product.HRES)
  with xr.open_zarr(store_path) as res_ds:
    vals = res_ds["hres_surface_net_solar_radiation"].values
    assert vals.shape == (1, 5, 10)
    # Days 0, 1 (prepended 03, 04) should be NaN
    assert np.isnan(vals[0, :2, :]).all()
    # Days 2, 3 (original 05, 06) should be 1.0
    np.testing.assert_array_equal(vals[0, 2:4, :], np.ones((2, 10), dtype=np.float32))
    # Day 4 (postpended 07) should be NaN
    assert np.isnan(vals[0, 4, :]).all()


def test_initialize_zarr_store_local_and_remote_memory_roundtrip(tmp_path):
  """Verifies initialize_zarr_store skeleton on local and memory:// stores, including re-init cleanup."""
  import zarr
  from multimet.timeseries_extractors.zarr_writer import check_zarr_store_exists

  basins = ["b_01", "b_02", "b_03"]
  dates = pd.date_range("2022-01-01", periods=4, freq="D")

  # 1. Remote memory:// store initialization (Nowcast ERA5_LAND)
  mem_root = "memory://test_init_zarr_store_roundtrip"
  mem_writer = MultiMetZarrWriter(mem_root)
  store_uri = mem_writer.initialize_zarr_store(
      Product.ERA5_LAND,
      basin_ids=basins,
      dates=list(dates),
      extra_attrs={"source_test": "unit"},
  )
  assert check_zarr_store_exists(store_uri)

  with xr.open_zarr(store_uri, consolidated=True) as ds:
    assert list(ds["basin"].values) == basins
    assert len(ds["date"]) == 4
    assert ds.attrs.get("source_test") == "unit"
    for band in PRODUCT_BANDS[Product.ERA5_LAND]:
      assert band in ds.data_vars
      assert ds[band].dtype == np.float32
      assert ds[band].shape == (3, 4)
    # Unwritten skeleton chunks must read back as NaN
    assert np.all(np.isnan(ds["era5land_total_precipitation"].values))

  # Write a direct chunk to day 1 and verify it persists
  mem_writer.write_direct_chunk(
      Product.ERA5_LAND,
      "era5land_total_precipitation",
      1,
      np.array([10.0, 20.0, 30.0], dtype=np.float32),
  )
  with xr.open_zarr(store_uri, consolidated=True) as ds_written:
    np.testing.assert_allclose(
        ds_written["era5land_total_precipitation"].values[:, 1],
        [10.0, 20.0, 30.0],
    )
    assert np.all(np.isnan(ds_written["era5land_total_precipitation"].values[:, 0]))

  # Re-initializing the same remote memory:// store must wipe the previously written day-1 chunk
  mem_writer_reinit = MultiMetZarrWriter(mem_root)
  mem_writer_reinit.initialize_zarr_store(
      Product.ERA5_LAND,
      basin_ids=basins,
      dates=list(dates),
  )
  with xr.open_zarr(store_uri, consolidated=True) as ds_reinit:
    assert np.all(np.isnan(ds_reinit["era5land_total_precipitation"].values))

  # 2. Local Forecast store with include_ensemble_members=True (GEFS)
  local_writer = MultiMetZarrWriter(tmp_path / "ens_init")
  ens_store = local_writer.initialize_zarr_store(
      Product.GEFS,
      basin_ids=["b_ens"],
      dates=list(dates[:2]),
      include_ensemble_members=True,
      ensemble_members=[0, 1, 2],
  )
  with xr.open_zarr(ens_store, consolidated=True) as ds_ens:
    assert "ensemble_member" in ds_ens.coords
    assert list(ds_ens["ensemble_member"].values) == [0, 1, 2]
    assert ds_ens["gefs_total_precipitation_ensemble"].shape == (1, 2, 3, 10)
    assert ds_ens["gefs_total_precipitation_ensemble"].dtype == np.float32
  z_grp = zarr.open_group(ens_store, mode="r")
  assert "_f32_template" not in z_grp


def test_write_or_append_2d_expansion_and_basin_dtype_widening(
    writer: MultiMetZarrWriter,
):
  """Verifies write_or_append supports adding new basins with shifted dates and varying string lengths."""
  basins_init = [
      "catchment_hydroatlas_POINT_38526_81631_38525_81630",
      "catchment_hydroatlas_HYRIV_70689198_35212_81698",
  ]
  dates_init = pd.date_range("2025-10-12", periods=3, freq="D")
  leads = pd.to_timedelta(range(1, 11), unit="D")
  ds1 = xr.Dataset(
      data_vars={
          band: (
              ["basin", "date", "lead_time"],
              np.ones((2, 3, 10), dtype=np.float32),
          )
          for band in PRODUCT_BANDS[Product.HRES]
      },
      coords={
          "basin": basins_init,
          "date": dates_init.values,
          "lead_time": leads.values,
      },
  )
  store_path = writer.write_or_append(ds1, Product.HRES)

  # 1. Add a 3rd basin (shorter string ID <U47 vs <U50) AND shifted dates (2025-10-13..2025-10-15)
  basins_next = basins_init + [
      "catchment_hydroatlas_HYRIV_70674197_35782_81497"
  ]
  dates_next = pd.date_range("2025-10-13", periods=3, freq="D")
  ds2 = xr.Dataset(
      data_vars={
          band: (
              ["basin", "date", "lead_time"],
              np.full((3, 3, 10), 2.0, dtype=np.float32),
          )
          for band in PRODUCT_BANDS[Product.HRES]
      },
      coords={
          "basin": basins_next,
          "date": dates_next.values,
          "lead_time": leads.values,
      },
  )
  writer.write_or_append(
      ds2, Product.HRES, preserve_existing_valid=True, allow_2d_expansion=True
  )

  # 2. Add a single new basin with an even longer string ID (<U60) on a new date with allow_2d_expansion=True
  long_basin = "catchment_hydroatlas_VERY_LONG_CUSTOM_BASIN_IDENTIFIER_99999"
  dates_single = pd.date_range("2025-10-16", periods=1, freq="D")
  ds3 = xr.Dataset(
      data_vars={
          band: (
              ["basin", "date", "lead_time"],
              np.full((1, 1, 10), 5.0, dtype=np.float32),
          )
          for band in PRODUCT_BANDS[Product.HRES]
      },
      coords={
          "basin": [long_basin],
          "date": dates_single.values,
          "lead_time": leads.values,
      },
  )
  writer.write_or_append(
      ds3, Product.HRES, preserve_existing_valid=True, allow_2d_expansion=True
  )

  with xr.open_zarr(store_path) as res_ds:
    assert list(res_ds["basin"].values) == basins_next + [long_basin]
    assert res_ds["hres_total_precipitation"].shape == (4, 5, 10)
    # New basin at 2025-10-12 (before its extraction window) must be NaN
    assert np.isnan(
        res_ds["hres_total_precipitation"]
        .sel(
            basin="catchment_hydroatlas_HYRIV_70674197_35782_81497",
            date="2025-10-12",
        )
        .values
    ).all()
    # New basin at 2025-10-15 must be 2.0
    np.testing.assert_allclose(
        res_ds["hres_total_precipitation"]
        .sel(
            basin="catchment_hydroatlas_HYRIV_70674197_35782_81497",
            date="2025-10-15",
        )
        .values,
        2.0,
    )
    # Long basin ID must be preserved without truncation and have 5.0 on 2025-10-16
    np.testing.assert_allclose(
        res_ds["hres_total_precipitation"]
        .sel(basin=long_basin, date="2025-10-16")
        .values,
        5.0,
    )


def test_write_or_append_rejects_2d_expansion_when_disallowed(
    writer: MultiMetZarrWriter,
):
  """Verifies write_or_append raises ValueError on partial-overlap 2D expansion when allow_2d_expansion=False."""
  dates_1 = pd.date_range("2025-11-01", periods=2, freq="D")
  ds1 = xr.Dataset(
      data_vars={
          band: (["basin", "date"], np.ones((2, 2), dtype=np.float32))
          for band in PRODUCT_BANDS[Product.IMERG]
      },
      coords={"basin": ["b1", "b2"], "date": dates_1.values},
  )
  writer.write_or_append(ds1, Product.IMERG)

  dates_2 = pd.date_range("2025-11-02", periods=2, freq="D")
  ds2 = xr.Dataset(
      data_vars={
          band: (["basin", "date"], np.ones((2, 2), dtype=np.float32))
          for band in PRODUCT_BANDS[Product.IMERG]
      },
      coords={"basin": ["b2", "b3"], "date": dates_2.values},
  )
  with pytest.raises(ValueError, match="Cannot add both new basins and new dates"):
    writer.write_or_append(ds2, Product.IMERG, allow_2d_expansion=False)


def test_write_or_append_sorts_unsorted_dates_and_preserves_untouched_basins(
    writer: MultiMetZarrWriter,
):
  """Verifies unsorted incoming dates are sorted and subset overwrite_existing_basins preserves other basins."""
  unsorted_dates = pd.to_datetime(["2025-12-03", "2025-12-01", "2025-12-02"])
  vals = np.array([[30.0, 10.0, 20.0], [300.0, 100.0, 200.0]], dtype=np.float32)
  ds1 = xr.Dataset(
      data_vars={"imerg_precipitation": (["basin", "date"], vals)},
      coords={"basin": ["b1", "b2"], "date": unsorted_dates.values},
  )
  store_path = writer.write_or_append(ds1, Product.IMERG)

  with xr.open_zarr(store_path) as ds_sorted:
    assert list(pd.to_datetime(ds_sorted["date"].values)) == list(
        pd.date_range("2025-12-01", periods=3, freq="D")
    )
    np.testing.assert_allclose(
        ds_sorted["imerg_precipitation"].sel(basin="b1").values,
        [10.0, 20.0, 30.0],
    )

  # Overwrite only b1 on 2025-12-02 while preserving b2
  ds_sub = xr.Dataset(
      data_vars={
          "imerg_precipitation": (
              ["basin", "date"],
              np.array([[99.0]], dtype=np.float32),
          )
      },
      coords={
          "basin": ["b1"],
          "date": pd.to_datetime(["2025-12-02"]).values,
      },
  )
  writer.write_or_append(
      ds_sub,
      Product.IMERG,
      overwrite_existing_basins=True,
      preserve_existing_valid=False,
  )
  with xr.open_zarr(store_path) as ds_after:
    np.testing.assert_allclose(
        ds_after["imerg_precipitation"].sel(basin="b1").values,
        [10.0, 99.0, 30.0],
    )
    np.testing.assert_allclose(
        ds_after["imerg_precipitation"].sel(basin="b2").values,
        [100.0, 200.0, 300.0],
    )


def test_inspect_store_last_valid_date_tail_and_optional_bands(tmp_path):
  """Verifies inspect_store_last_valid_date checks optional bands consistently across tail and history."""
  from multimet.timeseries_extractors.realtime import inspect_store_last_valid_date

  writer = MultiMetZarrWriter(tmp_path / "rt_inspect")
  dates = pd.date_range("2026-01-01", periods=20, freq="D")
  precip = np.ones((1, 20), dtype=np.float32)
  # Last 14 days (indices 6..19) have NaN in optional band cpc_num_stations,
  # while days 0..5 have valid cpc_num_stations.
  stations = np.full((1, 20), np.nan, dtype=np.float32)
  stations[:, :6] = 5.0
  ds_cpc = xr.Dataset(
      data_vars={
          "cpc_precipitation": (["basin", "date"], precip),
          "cpc_num_stations": (["basin", "date"], stations),
      },
      coords={"basin": ["b1"], "date": dates.values},
  )
  writer.write_or_append(ds_cpc, Product.CPC)

  info = inspect_store_last_valid_date(
      writer, Product.CPC, basin_ids=["b1"]
  )
  assert info == pd.Timestamp("2026-01-06")


def test_imerg_version_cutover_and_missing_file_handling(monkeypatch):
  """Verifies IMERG V07B/V07C cutover and interior vs trailing missing-date handling."""
  import geopandas as gpd
  from shapely.geometry import box
  from multimet.timeseries_extractors.imerg import (
      IMERG_V07C_CUTOVER_DATE,
      IMERGExtractor,
  )

  gdf = gpd.GeoDataFrame(
      {"geometry": [box(-80.0, 35.0, -79.5, 35.5)]},
      index=pd.Index(["b1"], name="basin"),
      crs="EPSG:4326",
  )
  ext = IMERGExtractor(source="gesdisc")
  assert ext._resolve_imerg_version_for_date("20260303") == "V07B"
  assert ext._resolve_imerg_version_for_date(IMERG_V07C_CUTOVER_DATE) == "V07C"
  assert ext._resolve_imerg_version_for_date("20260401") == "V07C"

  # Interior missing date older than MAX_PUBLICATION_LAG_DAYS must raise FileNotFoundError
  monkeypatch.setattr(ext, "_has_daily_file", lambda dt: False)
  with pytest.raises(FileNotFoundError, match="Missing IMERG daily"):
    ext.extract_for_basins(gdf, start_date="2026-01-01", end_date="2026-01-15")




