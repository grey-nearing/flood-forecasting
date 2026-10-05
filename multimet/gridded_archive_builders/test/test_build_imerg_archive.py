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

"""Unit tests for :mod:`multimet.gridded_archive_builders.build_imerg_archive`."""

from __future__ import annotations

from pathlib import Path

import h5py
from multimet.gridded_archive_builders import (
    build_imerg_archive as imerg_module,
)
from multimet.gridded_archive_builders.build_imerg_archive import (
    EXPECTED_HHR_START_TOKENS,
    GESDISCImergSource,
    IMERG_LATS,
    IMERG_LONS,
    LAT_COUNT,
    LON_COUNT,
    LocalImergSource,
    parse_imerg_netcdf_to_grid,
)
from multimet.utils import http as http_module
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = pytest.mark.unit


def write_synthetic_imerg_nc4(
    path: Path,
    fill_value: float = 5.0,
    var_name: str = "precipitation",
    transpose_dims: bool = True,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    date: str = "2024-01-01",
    include_time: bool = True,
) -> None:
  """Writes a synthetic daily IMERG NetCDF-4 file at native (1800, 3600) resolution."""
  lat_coords = IMERG_LATS if lats is None else lats
  lon_coords = IMERG_LONS if lons is None else lons
  if transpose_dims:
    if include_time:
      data = np.full(
          (1, len(lon_coords), len(lat_coords)), fill_value, dtype=np.float32
      )
      if fill_value >= 0.0:
        data[0, 0, 0] = -9999.9  # Sentinel to verify NaN masking
      ds = xr.Dataset(
          {var_name: (["time", "lon", "lat"], data)},
          coords={
              "time": [pd.Timestamp(date)],
              "lon": lon_coords,
              "lat": lat_coords,
          },
      )
    else:
      data = np.full(
          (len(lon_coords), len(lat_coords)), fill_value, dtype=np.float32
      )
      ds = xr.Dataset(
          {var_name: (["lon", "lat"], data)},
          coords={"lon": lon_coords, "lat": lat_coords},
      )
  else:
    if include_time:
      data = np.full(
          (1, len(lat_coords), len(lon_coords)), fill_value, dtype=np.float32
      )
      ds = xr.Dataset(
          {var_name: (["time", "lat", "lon"], data)},
          coords={
              "time": [pd.Timestamp(date)],
              "lat": lat_coords,
              "lon": lon_coords,
          },
      )
    else:
      data = np.full(
          (len(lat_coords), len(lon_coords)), fill_value, dtype=np.float32
      )
      ds = xr.Dataset(
          {var_name: (["lat", "lon"], data)},
          coords={"lat": lat_coords, "lon": lon_coords},
      )
  ds.to_netcdf(path)


class TestNetCDFParsing:
  """Tests ``parse_imerg_netcdf_to_grid`` validation and transposition."""

  def test_transposes_lon_lat_and_masks_sentinels(self, tmp_path: Path) -> None:
    nc_path = tmp_path / "sample.nc4"
    write_synthetic_imerg_nc4(
        nc_path, fill_value=12.5, transpose_dims=True, date="2024-01-01"
    )

    grid = parse_imerg_netcdf_to_grid(
        str(nc_path), expected_date=pd.Timestamp("2024-01-01")
    )

    assert grid.shape == (LAT_COUNT, LON_COUNT)
    assert grid.dtype == np.float32
    assert np.isnan(grid[0, 0])
    assert float(grid[1, 1]) == pytest.approx(12.5)

  def test_rejects_mismatched_internal_time_coordinate(
      self, tmp_path: Path
  ) -> None:
    nc_path = tmp_path / "wrong_date.nc4"
    write_synthetic_imerg_nc4(
        nc_path, fill_value=5.0, transpose_dims=True, date="2024-01-01"
    )

    with pytest.raises(ValueError, match="does not match expected date"):
      parse_imerg_netcdf_to_grid(
          str(nc_path), expected_date=pd.Timestamp("2024-01-02")
      )

  def test_rejects_missing_time_coordinate_when_expected_date_set(
      self, tmp_path: Path
  ) -> None:
    nc_path = tmp_path / "no_time.nc4"
    write_synthetic_imerg_nc4(
        nc_path, fill_value=5.0, transpose_dims=False, include_time=False
    )

    with pytest.raises(KeyError, match="'time' coordinate"):
      parse_imerg_netcdf_to_grid(
          str(nc_path), expected_date=pd.Timestamp("2024-01-01")
      )

  def test_rejects_legacy_v06_precipitation_cal_variable(
      self, tmp_path: Path
  ) -> None:
    nc_path = tmp_path / "sample_v06.nc4"
    write_synthetic_imerg_nc4(
        nc_path,
        fill_value=3.25,
        var_name="precipitationCal",
        transpose_dims=False,
    )

    with pytest.raises(KeyError, match="precipitation"):
      parse_imerg_netcdf_to_grid(str(nc_path))

  def test_rejects_inverted_latitude_coordinates(self, tmp_path: Path) -> None:
    nc_path = tmp_path / "flipped_lat.nc4"
    write_synthetic_imerg_nc4(
        nc_path,
        fill_value=2.0,
        transpose_dims=False,
        lats=IMERG_LATS[::-1],
    )

    with pytest.raises(ValueError, match="Latitude coordinate"):
      parse_imerg_netcdf_to_grid(str(nc_path))

  def test_rejects_shifted_longitude_coordinates(self, tmp_path: Path) -> None:
    nc_path = tmp_path / "shifted_lon.nc4"
    write_synthetic_imerg_nc4(
        nc_path,
        fill_value=2.0,
        transpose_dims=False,
        lons=np.linspace(0.05, 359.95, LON_COUNT, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Longitude coordinate"):
      parse_imerg_netcdf_to_grid(str(nc_path))

  def test_rejects_all_nan_daily_grid(self, tmp_path: Path) -> None:
    nc_path = tmp_path / "all_nan.nc4"
    write_synthetic_imerg_nc4(
        nc_path,
        fill_value=-9999.9,
        transpose_dims=False,
    )

    with pytest.raises(ValueError, match="all-NaN"):
      parse_imerg_netcdf_to_grid(str(nc_path))


class TestGESDISCSourceAndCleanup:
  """Tests ``GESDISCImergSource`` CMR discovery, caching, and conflict detection."""

  def test_resolves_cached_v07c_and_cleans_up_file(
      self, tmp_path: Path
  ) -> None:
    cache_dir = tmp_path / "imerg_cache"
    cache_dir.mkdir()
    fn = "3B-DAY-E.MS.MRG.3IMERG.20260201-S000000-E235959.V07C.nc4"
    staged_file = cache_dir / fn
    write_synthetic_imerg_nc4(
        staged_file, fill_value=7.0, date="2026-02-01"
    )

    source = GESDISCImergSource(cache_dir=str(cache_dir), cleanup_cache=True)
    grid = source.extract_date(pd.Timestamp("2026-02-01"))

    assert grid.shape == (LAT_COUNT, LON_COUNT)
    assert float(np.nanmax(grid)) == pytest.approx(7.0)
    assert not staged_file.exists()

  def test_rejects_conflicting_cached_versions(self, tmp_path: Path) -> None:
    cache_dir = tmp_path / "imerg_cache"
    cache_dir.mkdir()
    for suffix in ("V07B", "V07C"):
      fn = f"3B-DAY-E.MS.MRG.3IMERG.20260201-S000000-E235959.{suffix}.nc4"
      write_synthetic_imerg_nc4(
          cache_dir / fn, fill_value=7.0, date="2026-02-01"
      )

    source = GESDISCImergSource(cache_dir=str(cache_dir))
    with pytest.raises(ValueError, match="Multiple conflicting cached"):
      source.extract_date(pd.Timestamp("2026-02-01"))

  def test_raises_file_not_found_when_cmr_returns_no_granules(
      self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setattr(imerg_module, "query_cmr_granules", lambda *a, **k: [])
    source = GESDISCImergSource(cache_dir=str(tmp_path / "empty_cache"))
    with pytest.raises(FileNotFoundError, match="No published IMERG V07"):
      source.extract_date(pd.Timestamp("2026-02-01"))


class TestLocalSource:
  """Tests ``LocalImergSource`` explicit format selection and granule checks."""

  def test_nonexistent_local_dir_raises_immediately(
      self, tmp_path: Path
  ) -> None:
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
      LocalImergSource(str(missing_dir))

  def test_reads_local_nc4(self, tmp_path: Path) -> None:
    nc_file = (
        tmp_path / "3B-DAY-E.MS.MRG.3IMERG.20230510-S000000-E235959.V07B.nc4"
    )
    write_synthetic_imerg_nc4(
        nc_file, fill_value=4.5, date="2023-05-10"
    )

    source = LocalImergSource(str(tmp_path), local_format="nc4")
    grid = source.extract_date(pd.Timestamp("2023-05-10"))

    assert float(np.nanmax(grid)) == pytest.approx(4.5)

  def test_ignores_filename_with_non_delimited_date_substring(
      self, tmp_path: Path
  ) -> None:
    unrelated_file = tmp_path / "run_id_9202305109_output.nc4"
    write_synthetic_imerg_nc4(
        unrelated_file, fill_value=4.5, date="2023-05-10"
    )

    source = LocalImergSource(str(tmp_path), local_format="nc4")
    with pytest.raises(FileNotFoundError, match="No local IMERG NetCDF-4"):
      source.extract_date(pd.Timestamp("2023-05-10"))

  def test_rejects_conflicting_local_nc4_files(self, tmp_path: Path) -> None:
    for suffix in ("V07B", "V07C"):
      nc_file = (
          tmp_path
          / f"3B-DAY-E.MS.MRG.3IMERG.20230510-S000000-E235959.{suffix}.nc4"
      )
      write_synthetic_imerg_nc4(
          nc_file, fill_value=4.5, date="2023-05-10"
      )

    source = LocalImergSource(str(tmp_path), local_format="nc4")
    with pytest.raises(ValueError, match="Multiple conflicting local"):
      source.extract_date(pd.Timestamp("2023-05-10"))

  def test_nc4_mode_does_not_fall_back_to_h5_files(
      self, tmp_path: Path
  ) -> None:
    month_dir = tmp_path / "202305"
    month_dir.mkdir()
    for idx, token in enumerate(sorted(EXPECTED_HHR_START_TOKENS)):
      fpath = (
          month_dir
          / f"3B-HHR-E.MS.MRG.3IMERG.20230515-S{token}.{idx:04d}.V07B.RT-H5"
      )
      fpath.write_bytes(b"placeholder")

    source = LocalImergSource(str(tmp_path), local_format="nc4")
    with pytest.raises(FileNotFoundError, match="No local IMERG NetCDF-4"):
      source.extract_date(pd.Timestamp("2023-05-15"))

  def test_accumulates_48_half_hourly_h5_granules_and_masks_partial_missing_cells(
      self, tmp_path: Path
  ) -> None:
    month_dir = tmp_path / "202305"
    month_dir.mkdir()

    # Write 48 full-resolution (3600, 1800) half-hourly granules with all 48
    # unique start tokens. Rate = 2.0 mm/hr -> daily total = 48.0 mm.
    # Set cell (0, 0) to missing (-9999.9) in JUST ONE granule (idx == 17).
    base_arr = np.full((LON_COUNT, LAT_COUNT), 2.0, dtype=np.float32)
    for idx, token in enumerate(sorted(EXPECTED_HHR_START_TOKENS)):
      fpath = (
          month_dir
          / f"3B-HHR-E.MS.MRG.3IMERG.20230515-S{token}.{idx:04d}.V07B.RT-H5"
      )
      with h5py.File(fpath, "w") as h5:
        grp = h5.create_group("Grid")
        grp.create_dataset("lat", data=IMERG_LATS)
        grp.create_dataset("lon", data=IMERG_LONS)
        if idx == 17:
          arr = base_arr.copy()
          arr[0, 0] = -9999.9
          grp.create_dataset("precipitation", data=arr)
        else:
          grp.create_dataset("precipitation", data=base_arr)

    source = LocalImergSource(
        str(tmp_path), local_format="h5", granule_workers=4
    )
    grid = source.extract_date(pd.Timestamp("2023-05-15"))

    assert grid.shape == (LAT_COUNT, LON_COUNT)
    assert np.isnan(grid[0, 0]), "Cell with 47/48 valid half-hours must be NaN"
    assert np.allclose(grid[1:, :], 48.0)

  def test_rejects_duplicate_half_hour_h5_granules_even_when_count_is_48(
      self, tmp_path: Path
  ) -> None:
    month_dir = tmp_path / "202305"
    month_dir.mkdir()
    tokens = sorted(EXPECTED_HHR_START_TOKENS)
    # Replace token[1] ("003000") with a duplicate of token[0] ("000000") under a
    # different version suffix so there are still 48 files total.
    tokens[1] = tokens[0]
    for idx, token in enumerate(tokens):
      suffix = "V07C" if idx == 1 else "V07B"
      fpath = (
          month_dir
          / f"3B-HHR-E.MS.MRG.3IMERG.20230515-S{token}.{idx:04d}.{suffix}.RT-H5"
      )
      fpath.write_bytes(b"placeholder")

    source = LocalImergSource(str(tmp_path), local_format="h5")
    with pytest.raises(ValueError, match="48 unique 30-minute intervals"):
      source.extract_date(pd.Timestamp("2023-05-15"))


class TestCMRQuery:
  """Tests ``query_cmr_granules`` link filtering."""

  def test_query_cmr_granules_parses_data_links(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    class _FakeResp:

      def raise_for_status(self) -> None:
        pass

      def json(self) -> dict:
        return {
            "feed": {
                "entry": [{
                    "links": [
                        {
                            "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                            "href": "https://gpm1.gesdisc.eosdis.nasa.gov/data/3B-DAY-E.V07B.nc4",
                        },
                        {
                            "rel": (
                                "http://esipfed.org/ns/fedsearch/1.1/metadata#"
                            ),
                            "href": "https://gpm1.gesdisc.eosdis.nasa.gov/data/3B-DAY-E.V07B.nc4.xml",
                        },
                    ]
                }]
            }
        }

    monkeypatch.setattr(
        http_module.requests, "get", lambda *a, **k: _FakeResp()
    )
    urls = imerg_module.query_cmr_granules(
        imerg_module.IMERG_DAILY_SHORT_NAME, pd.Timestamp("2024-06-01")
    )
    assert urls == [
        "https://gpm1.gesdisc.eosdis.nasa.gov/data/3B-DAY-E.V07B.nc4"
    ]

  def test_gesdisc_find_latest_published_date_steps_back_within_7_days(
      self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    def fake_query(
        short_name: str, date: pd.Timestamp, **kwargs: object
    ) -> list[str]:
      if pd.Timestamp(date).normalize() <= pd.Timestamp("2026-01-08"):
        return [
            f"https://gpm1.gesdisc.eosdis.nasa.gov/3B-DAY-E.{date.strftime('%Y%m%d')}.nc4"
        ]
      return []

    monkeypatch.setattr(imerg_module, "query_cmr_granules", fake_query)
    source = imerg_module.GESDISCImergSource(cache_dir=str(tmp_path))
    latest = source.find_latest_published_date(
        pd.Timestamp("2026-01-10"), max_lag_days=7
    )
    assert latest == pd.Timestamp("2026-01-08")

  def test_gesdisc_find_latest_published_date_raises_when_lag_exceeds_7_days(
      self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setattr(
        imerg_module, "query_cmr_granules", lambda *a, **k: []
    )
    source = imerg_module.GESDISCImergSource(cache_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="max allowed lag: 7 days"):
      source.find_latest_published_date(
          pd.Timestamp("2026-01-10"), max_lag_days=7
      )

  def test_gesdisc_force_download_replaces_cached_file(
      self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    dt = pd.Timestamp("2024-06-01")
    stale_path = tmp_path / "3B-DAY-E.20240601-S000000.V07B.nc4"
    write_synthetic_imerg_nc4(
        stale_path,
        date="2024-06-01",
        fill_value=1.0,
    )
    fresh_source_dir = tmp_path / "fresh"
    fresh_source_dir.mkdir()
    fresh_file = fresh_source_dir / "3B-DAY-E.20240601-S000000.V07B.nc4"
    write_synthetic_imerg_nc4(
        fresh_file,
        date="2024-06-01",
        fill_value=42.0,
    )

    monkeypatch.setattr(
        imerg_module,
        "query_cmr_granules",
        lambda *a, **k: [
            "https://gpm1.gesdisc.eosdis.nasa.gov/3B-DAY-E.20240601-S000000.V07B.nc4"
        ],
    )

    def fake_download(
        url: str, dest_path: str, **kwargs: object
    ) -> str:
      Path(dest_path).write_bytes(fresh_file.read_bytes())
      return dest_path

    monkeypatch.setattr(imerg_module, "download_daily_imerg", fake_download)

    source = imerg_module.GESDISCImergSource(
        cache_dir=str(tmp_path), force_download=True
    )
    assert stale_path.exists()
    grid = source.extract_date(dt)
    assert float(np.nanmean(grid)) == pytest.approx(42.0)

  def test_cli_parses_extend_archive_flags(self) -> None:
    parser = imerg_module.build_arg_parser()
    args_u = parser.parse_args(
        ["--target_zarr", "/tmp/imerg.zarr", "--extend_archive"]
    )
    assert args_u.extend_archive is True

    args_h = parser.parse_args(
        ["--target_zarr", "/tmp/imerg.zarr", "--extend-archive"]
    )
    assert args_h.extend_archive is True

