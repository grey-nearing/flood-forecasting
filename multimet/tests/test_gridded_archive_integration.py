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
from multimet.tests.conftest import PSL_LATS, PSL_LONS
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

  def test_extend_archive_missing_store_raises_file_not_found(
      self, psl_cache: Path, tmp_path: Path
  ) -> None:
    target = str(tmp_path / "nonexistent_cpc.zarr")
    with pytest.raises(FileNotFoundError, match="--extend_archive"):
      build_cpc_archive(
          target_zarr=target,
          start_year=2020,
          end_year=2020,
          cache_dir=str(psl_cache),
          extend_archive=True,
          num_workers=1,
      )

  def test_extend_archive_with_overwrite_raises_value_error(
      self, psl_cache: Path, tmp_path: Path
  ) -> None:
    target = str(tmp_path / "cpc.zarr")
    with pytest.raises(ValueError, match="cannot be used together"):
      build_cpc_archive(
          target_zarr=target,
          start_year=2020,
          end_year=2020,
          cache_dir=str(psl_cache),
          extend_archive=True,
          overwrite=True,
          num_workers=1,
      )

  def test_extend_archive_ignores_stale_cache_and_trims_active_year_tail(
      self,
      write_psl_year: Callable[..., Path],
      psl_cache: Path,
      tmp_path: Path,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    # 1. Build initial store through 2026-01-02 using cached precip.2026.nc.
    write_psl_year(2026, "2026-01-01", "2026-01-02")
    target = str(tmp_path / "cpc_extend.zarr")
    build_cpc_archive(
        target_zarr=target,
        start_year=2026,
        end_year=2026,
        end_date="2026-01-02",
        cache_dir=str(psl_cache),
        num_workers=1,
        reference_date="2026-01-03",
    )
    assert len(open_store(target)["time"]) == 2

    # 2. Now upstream has data through 2026-01-05 (with trailing NaNs 01-06..01-08),
    # while psl_cache still holds the stale 2-day precip.2026.nc.
    def fake_download(url: str, dest_path: str, **kwargs: object) -> str:
      dates = pd.date_range("2026-01-01", "2026-01-08", freq="1D")
      data = np.full((len(dates), 360, 720), 7.0, dtype=np.float32)
      data[5:] = np.nan  # Valid through 2026-01-05 (lag = 2 days from Jan 7)
      ds = xr.Dataset(
          data_vars={"precip": (["time", "lat", "lon"], data)},
          coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
      )
      ds.to_netcdf(dest_path)
      ds.close()
      return dest_path

    monkeypatch.setattr(
        "multimet.gridded_archive_builders.build_cpc_archive.download_http_file",
        fake_download,
    )

    build_cpc_archive(
        target_zarr=target,
        start_year=2026,
        end_year=2026,
        cache_dir=str(psl_cache),
        extend_archive=True,
        num_workers=1,
        reference_date="2026-01-07",
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert list(times) == list(pd.date_range("2026-01-01", "2026-01-05"))
    assert float(store[CPC_VARIABLE].sel(time="2026-01-05").mean()) == (
        pytest.approx(7.0)
    )

  def test_extend_archive_handles_early_january_rollover_when_new_year_unpublished(
      self,
      write_psl_year: Callable[..., Path],
      psl_cache: Path,
      tmp_path: Path,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    # Build initial store through 2025-12-28.
    write_psl_year(2025, "2025-12-27", "2025-12-30")
    target = str(tmp_path / "cpc_rollover.zarr")
    build_cpc_archive(
        target_zarr=target,
        start_year=2025,
        end_year=2025,
        start_date="2025-12-27",
        end_date="2025-12-28",
        cache_dir=str(psl_cache),
        num_workers=1,
        reference_date="2025-12-29",
    )

    # On 2026-01-02 (reference_date = 2026-01-02), precip.2025.nc has valid data
    # through 2025-12-30 and trailing NaN on 2025-12-31, while precip.2026.nc is
    # not yet published on NOAA PSL (HTTP 404).
    dates_2025 = pd.date_range("2025-12-27", "2025-12-31", freq="1D")
    data_2025 = np.full((len(dates_2025), 360, 720), 4.0, dtype=np.float32)
    data_2025[-1] = np.nan  # Dec 31 unpublished
    ds_2025 = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data_2025)},
        coords={"time": dates_2025, "lat": PSL_LATS, "lon": PSL_LONS},
    )
    ds_2025.to_netcdf(psl_cache / "precip.2025.nc")
    ds_2025.close()

    monkeypatch.setattr(
        "multimet.gridded_archive_builders.build_cpc_archive.check_http_url_exists",
        lambda url, **kw: False,
    )
    monkeypatch.setattr(
        "multimet.gridded_archive_builders.build_cpc_archive.download_http_file",
        lambda url, dest_path, **kw: dest_path,
    )

    build_cpc_archive(
        target_zarr=target,
        start_year=2025,
        end_year=2026,
        cache_dir=str(psl_cache),
        num_workers=1,
        reference_date="2026-01-02",
    )

    store = open_store(target)
    times = pd.to_datetime(store["time"].values)
    assert list(times) == list(pd.date_range("2025-12-27", "2025-12-30"))


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

    # 5. Extend archive without --end_date: auto-discovers latest published date
    # (2024-01-05) within 7 days of reference_date="2024-01-08".
    _write_imerg_nc4_day(local_dir, "2024-01-05", 5.0)
    build_imerg_archive(
        target_zarr=target_store,
        start_date="2024-01-01",
        source_type="local",
        local_format="nc4",
        local_dir=str(local_dir),
        extend_archive=True,
        num_workers=1,
        reference_date="2024-01-08",
    )
    store = open_store(target_store)
    assert len(store["time"]) == 5
    assert pd.Timestamp(store["time"].values[-1]) == pd.Timestamp("2024-01-05")

    # 6. Interior gap before latest published date (2024-01-07 exists, 2024-01-06
    # missing) raises FileNotFoundError for the missing interior date.
    _write_imerg_nc4_day(local_dir, "2024-01-07", 7.0)
    with pytest.raises(FileNotFoundError, match="2024-01-06"):
      build_imerg_archive(
          target_zarr=target_store,
          start_date="2024-01-01",
          source_type="local",
          local_format="nc4",
          local_dir=str(local_dir),
          extend_archive=True,
          num_workers=1,
          reference_date="2024-01-08",
      )

