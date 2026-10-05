# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end integration tests for the CPC and IMERG gridded archive builders."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from multimet.gridded_archive_builders.build_cpc_archive import (
    CPC_VARIABLE,
    build_cpc_archive,
)
from multimet.gridded_archive_builders.build_imerg_archive import (
    IMERG_ATTRS,
    IMERG_LATS,
    IMERG_LONS,
    IMERG_VARIABLE,
    LAT_COUNT,
    LON_COUNT,
    build_imerg_archive,
)
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = pytest.mark.integration


def open_store(path: str) -> xr.Dataset:
  """Loads a local Zarr store fully into memory so it can be closed."""
  with xr.open_zarr(path, consolidated=False) as store:
    return store.load()


def _write_imerg_nc4_day(
    local_dir: Path, date_iso: str, value: float
) -> Path:
  """Writes a full-resolution (1800, 3600) daily IMERG NetCDF-4 file."""
  dt = pd.Timestamp(date_iso)
  d_str = dt.strftime("%Y%m%d")
  out_path = local_dir / f"imerg_{d_str}.nc4"
  ds = xr.Dataset(
      {
          "precipitation": (
              ["time", "lat", "lon"],
              np.full((1, LAT_COUNT, LON_COUNT), value, dtype=np.float32),
          )
      },
      coords={
          "time": [dt],
          "lat": IMERG_LATS,
          "lon": IMERG_LONS,
      },
  )
  ds.to_netcdf(out_path)
  return out_path


class TestCPCArchiveEndToEnd:
  """Full ``build_cpc_archive`` runs against a local store."""

  def test_single_year_build(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-05")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-05",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    assert len(store["time"]) == 5
    assert store[CPC_VARIABLE].shape == (5, 360, 720)
    assert list(pd.to_datetime(store["time"].values)) == list(
        pd.date_range("2020-01-01", "2020-01-05")
    )

  def test_incomplete_historical_year_without_end_date_raises_value_error(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-05")
    target = str(tmp_path / "cpc_incomplete.zarr")

    with pytest.raises(ValueError, match="before end of year 2020-12-31"):
      build_cpc_archive(
          target_zarr=target,
          start_year=2020,
          end_year=2020,
          cache_dir=str(psl_cache),
          num_workers=1,
      )

  def test_multi_year_build_is_chronological(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-12-29", "2020-12-31")
    write_psl_year(2021, "2021-01-01", "2021-01-03")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2021,
        start_date="2020-12-29",
        end_date="2021-01-03",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert len(times) == 6
    assert times.is_monotonic_increasing
    assert times[0] == pd.Timestamp("2020-12-29")
    assert times[-1] == pd.Timestamp("2021-01-03")

  def test_non_contiguous_multi_year_gap_raises_value_error(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-03")
    write_psl_year(2021, "2021-01-01", "2021-01-03")
    target = str(tmp_path / "cpc_gap.zarr")

    with pytest.raises(ValueError, match="before end of year 2020-12-31"):
      build_cpc_archive(
          target_zarr=target,
          start_year=2020,
          end_year=2021,
          end_date="2021-01-03",
          cache_dir=str(psl_cache),
          num_workers=1,
      )

  def test_values_land_on_the_right_dates(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-04")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-04",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    for i, day in enumerate(pd.date_range("2020-01-01", "2020-01-04")):
      slice_values = store[CPC_VARIABLE].sel(time=day).values
      assert float(np.nanmin(slice_values)) == pytest.approx(1.0 + i)
      assert float(np.nanmax(slice_values)) == pytest.approx(1.0 + i)

  def test_date_filters_bound_the_build(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-10")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        start_date="2020-01-03",
        end_date="2020-01-06",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert times[0] == pd.Timestamp("2020-01-03")
    assert times[-1] == pd.Timestamp("2020-01-06")
    assert len(times) == 4

  def test_resume_appends_only_new_days(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-10")
    target = str(tmp_path / "cpc.zarr")
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-04",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-08",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert list(times) == list(pd.date_range("2020-01-01", "2020-01-08"))
    assert len(set(times)) == len(times)

  def test_resume_rejects_start_date_gap(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-10")
    target = str(tmp_path / "cpc_resume_gap.zarr")
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-04",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    with pytest.raises(ValueError, match="expected 2020-01-05"):
      build_cpc_archive(
          target_zarr=target,
          start_year=2020,
          end_year=2020,
          start_date="2020-01-06",
          end_date="2020-01-08",
          cache_dir=str(psl_cache),
          num_workers=1,
      )

  def test_resume_does_not_disturb_existing_values(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-01-01", "2020-01-06")
    target = str(tmp_path / "cpc.zarr")
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-03",
        cache_dir=str(psl_cache),
        num_workers=1,
    )
    before = open_store(target)[CPC_VARIABLE].isel(time=slice(0, 3)).values

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-06",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    after = open_store(target)[CPC_VARIABLE].isel(time=slice(0, 3)).values
    np.testing.assert_array_equal(before, after)

  def test_overwrite_rebuilds_from_scratch(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    target = str(tmp_path / "cpc.zarr")
    write_psl_year(2020, "2020-01-01", "2020-01-06")
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-06",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    write_psl_year(2020, "2020-01-01", "2020-01-02", fill_value=50.0)
    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-02",
        cache_dir=str(psl_cache),
        num_workers=1,
        overwrite=True,
    )

    store = open_store(target)
    assert len(store["time"]) == 2
    assert float(store[CPC_VARIABLE].isel(time=0).min()) == pytest.approx(50.0)

  def test_missing_cells_survive_as_nan(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(
        2020, "2020-01-01", "2020-01-02", missing_cells=[(0, 0, 0), (1, 5, 5)]
    )
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-02",
        cache_dir=str(psl_cache),
        num_workers=1,
    )

    store = open_store(target)
    assert int(np.isnan(store[CPC_VARIABLE].values).sum()) == 2

  def test_cleanup_cache_removes_downloads(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    netcdf = write_psl_year(2020, "2020-01-01", "2020-01-02")
    target = str(tmp_path / "cpc.zarr")

    build_cpc_archive(
        target_zarr=target,
        start_year=2020,
        end_year=2020,
        end_date="2020-01-02",
        cache_dir=str(psl_cache),
        cleanup_cache=True,
        num_workers=1,
    )

    assert not netcdf.exists()
    assert len(open_store(target)["time"]) == 2

  @pytest.mark.slow
  def test_parallel_workers_produce_the_same_store(
      self, write_psl_year: Callable[..., Path], psl_cache: Path, tmp_path: Path
  ) -> None:
    write_psl_year(2020, "2020-12-29", "2020-12-31")
    write_psl_year(2021, "2021-01-01", "2021-01-03")
    serial_target = str(tmp_path / "serial.zarr")
    parallel_target = str(tmp_path / "parallel.zarr")

    build_cpc_archive(
        target_zarr=serial_target,
        start_year=2020,
        end_year=2021,
        start_date="2020-12-29",
        end_date="2021-01-03",
        cache_dir=str(psl_cache),
        num_workers=1,
    )
    build_cpc_archive(
        target_zarr=parallel_target,
        start_year=2020,
        end_year=2021,
        start_date="2020-12-29",
        end_date="2021-01-03",
        cache_dir=str(psl_cache),
        num_workers=2,
    )

    serial = open_store(serial_target)
    parallel = open_store(parallel_target)
    assert len(parallel["time"]) == 6
    xr.testing.assert_identical(serial, parallel)


class TestIMERGArchiveEndToEnd:
  """Full ``build_imerg_archive`` runs against a local store at native (1800, 3600) resolution."""

  def test_build_resume_and_in_place_update(self, tmp_path: Path) -> None:
    local_dir = tmp_path / "local_nc"
    local_dir.mkdir()
    for d_idx, d_iso in enumerate(
        ["2024-01-01", "2024-01-02", "2024-01-03"], start=1
    ):
      _write_imerg_nc4_day(local_dir, d_iso, float(d_idx))

    target_store = str(tmp_path / "imerg_out.zarr")
    cache_dir = tmp_path / "temp_cache"
    cache_dir.mkdir()

    # 1. Build initial archive for 2024-01-01 to 2024-01-03.
    build_imerg_archive(
        target_zarr=target_store,
        start_date="2024-01-01",
        end_date="2024-01-03",
        source_type="local",
        local_format="nc4",
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        cleanup_cache=True,
        batch_size=2,
        num_workers=1,
    )
    assert not cache_dir.exists()

    store = open_store(target_store)
    assert len(store["time"]) == 3
    assert store[IMERG_VARIABLE].shape == (3, LAT_COUNT, LON_COUNT)
    assert store.attrs["title"] == IMERG_ATTRS["title"]
    assert float(store[IMERG_VARIABLE].isel(time=0).mean()) == pytest.approx(
        1.0
    )
    assert float(store[IMERG_VARIABLE].isel(time=2).mean()) == pytest.approx(
        3.0
    )

    # 2. Add 2024-01-04 and resume up to 2024-01-04.
    _write_imerg_nc4_day(local_dir, "2024-01-04", 4.0)

    build_imerg_archive(
        target_zarr=target_store,
        start_date="2024-01-01",
        end_date="2024-01-04",
        source_type="local",
        local_format="nc4",
        local_dir=str(local_dir),
        batch_size=2,
        num_workers=1,
    )
    store = open_store(target_store)
    assert len(store["time"]) == 4
    assert float(store[IMERG_VARIABLE].isel(time=3).mean()) == pytest.approx(
        4.0
    )

    # 3. Missing date (2024-01-05) raises FileNotFoundError instead of writing NaNs.
    with pytest.raises(FileNotFoundError, match="2024-01-05"):
      build_imerg_archive(
          target_zarr=target_store,
          start_date="2024-01-01",
          end_date="2024-01-05",
          source_type="local",
          local_format="nc4",
          local_dir=str(local_dir),
          batch_size=2,
          num_workers=1,
      )
    assert len(open_store(target_store)["time"]) == 4

    # 4. In-place update of 2024-01-02 with updated valid data.
    _write_imerg_nc4_day(local_dir, "2024-01-02", 99.0)

    build_imerg_archive(
        target_zarr=target_store,
        start_date="2024-01-02",
        end_date="2024-01-02",
        source_type="local",
        local_format="nc4",
        local_dir=str(local_dir),
        in_place=True,
        num_workers=1,
    )
    store = open_store(target_store)
    assert float(store[IMERG_VARIABLE].isel(time=1).mean()) == pytest.approx(
        99.0
    )
