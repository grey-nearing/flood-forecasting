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

"""Unit tests for :mod:`multimet.gridded_archive_builders.build_cpc_archive`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from multimet.gridded_archive_builders.build_cpc_archive import (
    CPC_LATS,
    CPC_LONS,
    CPC_VARIABLE,
    ensure_psl_cpc_netcdf,
    process_cpc_netcdf_to_dataset,
    write_batch_to_zarr,
)
from multimet.gridded_archive_builders.test.conftest import PSL_LATS, PSL_LONS
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_year(write_psl_year: Callable[..., Path]) -> Path:
  """A three day synthetic 2020 file with one missing cell on day 0."""
  return write_psl_year(
      2020, "2020-01-01", "2020-01-03", missing_cells=[(0, 0, 0)]
  )


class TestGridStandardization:
  """The PSL -> Caravan MultiMet spatial rewrite and input validation."""

  def test_output_schema(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(str(sample_year))

    assert dataset is not None
    assert CPC_VARIABLE in dataset.data_vars
    assert dataset[CPC_VARIABLE].dims == ("time", "latitude", "longitude")
    assert dataset[CPC_VARIABLE].shape == (3, 360, 720)
    assert dataset[CPC_VARIABLE].dtype == np.float32

  def test_latitude_is_flipped_to_ascending(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(str(sample_year))

    assert dataset is not None
    np.testing.assert_allclose(dataset["latitude"].values, CPC_LATS)
    assert dataset["latitude"].values[0] == pytest.approx(-89.75)
    assert dataset["latitude"].values[-1] == pytest.approx(89.75)
    assert np.all(np.diff(dataset["latitude"].values) > 0)

  def test_longitude_is_rolled_to_signed_range(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(str(sample_year))

    assert dataset is not None
    np.testing.assert_allclose(dataset["longitude"].values, CPC_LONS)
    assert dataset["longitude"].values[0] == pytest.approx(-179.75)
    assert dataset["longitude"].values[-1] == pytest.approx(179.75)
    assert np.all(np.diff(dataset["longitude"].values) > 0)

  def test_values_follow_their_coordinates(self, sample_year: Path) -> None:
    """A cell must keep its geographic identity through the axis rewrite."""
    dates = pd.date_range("2020-01-01", "2020-01-01", freq="D")
    psl_lat_index, psl_lon_index = 100, 500
    data = np.zeros((1, len(PSL_LATS), len(PSL_LONS)), dtype=np.float32)
    data[0, psl_lat_index, psl_lon_index] = 42.0
    source = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
    )
    path = sample_year.parent / "precip.1999.nc"
    source.to_netcdf(path)
    source.close()

    dataset = process_cpc_netcdf_to_dataset(str(path))
    assert dataset is not None

    expected_lat = float(PSL_LATS[psl_lat_index])
    raw_lon = float(PSL_LONS[psl_lon_index])
    expected_lon = raw_lon - 360.0 if raw_lon > 180.0 else raw_lon
    marked = dataset[CPC_VARIABLE].sel(
        latitude=expected_lat, longitude=expected_lon, method="nearest"
    )
    assert float(marked.isel(time=0)) == pytest.approx(42.0)
    assert float(dataset[CPC_VARIABLE].isel(time=0).sum()) == pytest.approx(
        42.0
    )

  def test_missing_values_become_nan(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(str(sample_year))
    assert dataset is not None
    values = dataset[CPC_VARIABLE].values

    assert np.isnan(values[0, 359, 360])
    assert np.count_nonzero(np.isnan(values)) == 1

  def test_negative_sentinels_become_nan_while_preserving_valid_cells(
      self, sample_year: Path
  ) -> None:
    dates = pd.date_range("2001-01-01", "2001-01-01", freq="D")
    data = np.full((1, len(PSL_LATS), len(PSL_LONS)), 5.0, dtype=np.float32)
    data[0, :180, :] = -1.0
    source = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
    )
    path = sample_year.parent / "precip.2001.nc"
    source.to_netcdf(path)
    source.close()

    dataset = process_cpc_netcdf_to_dataset(str(path))
    assert dataset is not None
    vals = dataset[CPC_VARIABLE].values
    # Raw indices 0..179 (northern hemisphere) become indices 180..359 after flip.
    assert bool(np.isnan(vals[0, 180:, :]).all())
    assert np.allclose(vals[0, :180, :], 5.0)

  def test_all_nan_file_raises_value_error(
      self, write_psl_year: Callable[..., Path]
  ) -> None:
    path = write_psl_year(2002, "2002-01-01", "2002-01-02", fill_value=-10.0)
    with pytest.raises(ValueError, match="no finite precipitation values"):
      process_cpc_netcdf_to_dataset(str(path), trim_trailing_unpublished=True)

  def test_interior_all_nan_day_raises_value_error(
      self, sample_year: Path
  ) -> None:
    dates = pd.date_range("2025-01-01", "2025-01-04", freq="D")
    data = np.ones((4, len(PSL_LATS), len(PSL_LONS)), dtype=np.float32)
    # Day 1 is an interior all-NaN day; day 3 is a trailing unpublished day.
    data[1] = -9.96921e36
    data[3] = -9.96921e36
    source = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
    )
    path = sample_year.parent / "precip.2025.nc"
    source.to_netcdf(path)
    source.close()

    with pytest.raises(ValueError, match="2025-01-02"):
      process_cpc_netcdf_to_dataset(str(path), trim_trailing_unpublished=True)

  def test_already_ascending_latitude_raises_value_error(
      self, sample_year: Path
  ) -> None:
    dates = pd.date_range("2024-01-01", "2024-01-01", freq="D")
    data = np.ones((1, len(CPC_LATS), len(PSL_LONS)), dtype=np.float32)
    source = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": CPC_LATS, "lon": PSL_LONS},
    )
    path = sample_year.parent / "precip.bad_lat.nc"
    source.to_netcdf(path)
    source.close()

    with pytest.raises(ValueError, match="Unexpected latitude coordinates"):
      process_cpc_netcdf_to_dataset(str(path))

  def test_signed_longitude_input_raises_value_error(
      self, sample_year: Path
  ) -> None:
    dates = pd.date_range("2024-01-01", "2024-01-01", freq="D")
    data = np.ones((1, len(PSL_LATS), len(CPC_LONS)), dtype=np.float32)
    source = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": PSL_LATS, "lon": CPC_LONS},
    )
    path = sample_year.parent / "precip.bad_lon.nc"
    source.to_netcdf(path)
    source.close()

    with pytest.raises(ValueError, match="Unexpected longitude coordinates"):
      process_cpc_netcdf_to_dataset(str(path))

  def test_non_contiguous_dates_raise_value_error(
      self, sample_year: Path
  ) -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"])
    data = np.ones((2, len(PSL_LATS), len(PSL_LONS)), dtype=np.float32)
    source = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
    )
    path = sample_year.parent / "precip.gap.nc"
    source.to_netcdf(path)
    source.close()

    with pytest.raises(ValueError, match="contiguous daily"):
      process_cpc_netcdf_to_dataset(str(path))

  def test_trims_trailing_unpublished_nan_days(
      self, sample_year: Path
  ) -> None:
    dates = pd.date_range("2026-01-01", "2026-01-05", freq="D")
    data = np.ones((5, len(PSL_LATS), len(PSL_LONS)), dtype=np.float32)
    # Days 3 and 4 are future pre-allocated days filled with negative sentinel.
    data[3:] = -9.96921e36
    source = xr.Dataset(
        data_vars={"precip": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
    )
    path = sample_year.parent / "precip.2026.nc"
    source.to_netcdf(path)
    source.close()

    dataset = process_cpc_netcdf_to_dataset(
        str(path), trim_trailing_unpublished=True
    )
    assert dataset is not None
    assert len(dataset["time"]) == 3
    assert pd.Timestamp(dataset["time"].values[-1]) == pd.Timestamp(
        "2026-01-03"
    )

  def test_timestamps_are_normalized_to_midnight(
      self, sample_year: Path
  ) -> None:
    dataset = process_cpc_netcdf_to_dataset(str(sample_year))
    assert dataset is not None
    times = pd.to_datetime(dataset["time"].values)

    assert list(times) == list(pd.date_range("2020-01-01", "2020-01-03"))
    assert (times.normalize() == times).all()

  def test_global_metadata_is_attached(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(str(sample_year))
    assert dataset is not None

    assert dataset.attrs["product"] == "CPC"
    assert dataset.attrs["spatial_resolution"] == "0.50 degree"
    assert "NOAA" in dataset.attrs["institution"]
    assert dataset[CPC_VARIABLE].attrs["units"] == "mm/day"


class TestDateFiltering:
  """``target_start_date`` / ``target_end_date`` bounds."""

  def test_filters_to_a_single_day(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(
        str(sample_year),
        target_start_date=pd.Timestamp("2020-01-02"),
        target_end_date=pd.Timestamp("2020-01-02"),
    )

    assert dataset is not None
    assert len(dataset["time"]) == 1
    assert pd.Timestamp(dataset["time"].values[0]) == pd.Timestamp("2020-01-02")

  def test_bounds_are_inclusive(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(
        str(sample_year),
        target_start_date=pd.Timestamp("2020-01-01"),
        target_end_date=pd.Timestamp("2020-01-03"),
    )

    assert dataset is not None
    assert len(dataset["time"]) == 3

  def test_returns_none_when_no_dates_match(self, sample_year: Path) -> None:
    dataset = process_cpc_netcdf_to_dataset(
        str(sample_year), target_start_date=pd.Timestamp("2021-01-01")
    )

    assert dataset is None


class TestWriteBatchToZarr:
  """The local/remote-agnostic Zarr write primitive."""

  def test_initial_write_then_append(
      self, sample_year: Path, tmp_path: Path
  ) -> None:
    target = str(tmp_path / "cpc.zarr")

    first = process_cpc_netcdf_to_dataset(
        str(sample_year),
        target_start_date=pd.Timestamp("2020-01-01"),
        target_end_date=pd.Timestamp("2020-01-01"),
    )
    assert first is not None
    write_batch_to_zarr(first, target, is_initial_write=True)

    with xr.open_zarr(target, consolidated=False) as store:
      assert len(store["time"]) == 1
      assert store[CPC_VARIABLE].shape == (1, 360, 720)

    rest = process_cpc_netcdf_to_dataset(
        str(sample_year),
        target_start_date=pd.Timestamp("2020-01-02"),
        target_end_date=pd.Timestamp("2020-01-03"),
    )
    assert rest is not None
    write_batch_to_zarr(rest, target, is_initial_write=False)

    with xr.open_zarr(target, consolidated=False) as store:
      assert len(store["time"]) == 3
      assert store[CPC_VARIABLE].shape == (3, 360, 720)
      times = pd.to_datetime(store["time"].values)
      assert list(times) == list(pd.date_range("2020-01-01", "2020-01-03"))

  def test_append_preserves_previously_written_values(
      self, sample_year: Path, tmp_path: Path
  ) -> None:
    target = str(tmp_path / "cpc.zarr")
    full = process_cpc_netcdf_to_dataset(str(sample_year))
    assert full is not None

    write_batch_to_zarr(
        full.isel(time=slice(0, 1)), target, is_initial_write=True
    )
    write_batch_to_zarr(full.isel(time=slice(1, 3)), target)

    with xr.open_zarr(target, consolidated=False) as store:
      xr.testing.assert_allclose(store[CPC_VARIABLE], full[CPC_VARIABLE])

  def test_chunking_is_capped_at_thirty_days(
      self, sample_year: Path, tmp_path: Path
  ) -> None:
    target = str(tmp_path / "cpc.zarr")
    first = process_cpc_netcdf_to_dataset(
        str(sample_year),
        target_end_date=pd.Timestamp("2020-01-02"),
    )
    assert first is not None

    write_batch_to_zarr(first, target, is_initial_write=True)

    with xr.open_zarr(target, consolidated=False) as store:
      time_chunk = store[CPC_VARIABLE].encoding["chunks"][0]
    assert time_chunk == min(30, len(first["time"]))


class TestEnsurePslNetcdf:
  """Download caching and HTTP error handling."""

  def test_returns_cached_file_without_downloading(
      self, sample_year: Path, psl_cache: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    def explode(*args: object, **kwargs: object) -> None:
      raise AssertionError("ensure_psl_cpc_netcdf must not hit the network")

    monkeypatch.setattr("requests.get", explode)

    resolved = ensure_psl_cpc_netcdf(2020, cache_dir=str(psl_cache))

    assert Path(resolved) == sample_year

  def test_raises_file_not_found_on_http_404(
      self, psl_cache: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    class _Fake404Response:
      status_code = 404

      def __enter__(self) -> _Fake404Response:
        return self

      def __exit__(self, *args: object) -> None:
        pass

    monkeypatch.setattr("requests.get", lambda *a, **k: _Fake404Response())

    with pytest.raises(FileNotFoundError, match="HTTP 404"):
      ensure_psl_cpc_netcdf(2099, cache_dir=str(psl_cache))
