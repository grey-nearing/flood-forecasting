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

"""Unit tests for real-time meteorological forcing fetcher (Cold-Start & Hot-Start)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import multimet.timeseries_extractors.dynamical as dynamical_mod
from multimet.timeseries_extractors.config import PRODUCT_BANDS, Product
from multimet.timeseries_extractors.hres import (
    OPEN_DATA_GRID_SHAPE,
    HRESExtractor,
    decode_grib2_message,
    find_latest_hres_open_data_date,
)
from multimet.timeseries_extractors.realtime import (
    DEFAULT_COLDSTART_LOOKBACK_DAYS,
    RealtimeForcingFetcher,
    build_arg_parser,
    fetch_realtime_multimet,
    inspect_store_last_valid_date,
    main as realtime_main,
    read_hot_start_state_date,
)
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter
from multimet.utils.geometry import load_basin_geometries

pytestmark = pytest.mark.unit

# Standard WMO GRIB2 Section 0-6 headers for 0.25-deg global (721x1440) grid_ieee messages.
_GRIB2_IEEE_HEADERS: Dict[str, bytes] = {
    "2t": base64.b64decode(
        "R1JJQv//AAIAAAAAAD9fKgAAABUBAGIAAAQAAQfXAxcMAAAAAgAAAEgDAAAP16AAAAAABv///////////////////wAABaAAAALRAAAAAP////8Dk4cAAAAAADAAAAAAAcnDgAAehIAAHoSAAAAAACIEAAAAAAAAAP+AAAAAAQAAAABnAAAAAAL///////8AAAAMBQAP16AABAEAAAAGBv8AP16FBw=="
    ),
    "sp": base64.b64decode(
        "R1JJQv//AAIAAAAAAD9fKgAAABUBAGIAAAQAAQfXAxcMAAAAAgAAAEgDAAAP16AAAAAABv///////////////////wAABaAAAALRAAAAAP////8Dk4cAAAAAADAAAAAAAcnDgAAehIAAHoSAAAAAACIEAAAAAAMAAP+AAAAAAQAAAAAB//////////////8AAAAMBQAP16AABAEAAAAGBv8AP16FBw=="
    ),
    "tp": base64.b64decode(
        "R1JJQv//AAIAAAAAAD9fQgAAABUBAGIAAAQBAQfXAxcMAAAAAgAAAEgDAAAP16AAAAAABv///////////////////wAABaAAAALRAAAAAP////8Dk4cAAAAAADAAAAAAAcnDgAAehIAAHoSAAAAAADoEAAAACAHBAP+AAAAAAQAAAAAB//////////////8H1wMXDAAAAQAAAAABAgEAAAAA/wAAAAAAAAAMBQAP16AABAEAAAAGBv8AP16FBw=="
    ),
    "ssr": base64.b64decode(
        "R1JJQv//wAIAAAAAAD9fKgAAABUBAGIAAAQAAQfXAxcMAAAAAgAAAEgDAAAP16AAAAAABv///////////////////wAABaAAAALRAAAAAP////8Dk4cAAAAAADAAAAAAAcnDgAAehIAAHoSAAAAAACIEAAAAALSwAP+AAAAAAQAAAAAB//////////////8AAAAMBQAP16AABAEAAAAGBv8AP16FBw=="
    ),
    "str": base64.b64decode(
        "R1JJQv//wAIAAAAAAD9fKgAAABUBAGIAAAQAAQfXAxcMAAAAAgAAAEgDAAAP16AAAAAABv///////////////////wAABaAAAALRAAAAAP////8Dk4cAAAAAADAAAAAAAcnDgAAehIAAHoSAAAAAACIEAAAAALSxAP+AAAAAAQAAAAAB//////////////8AAAAMBQAP16AABAEAAAAGBv8AP16FBw=="
    ),
}


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


def _encode_test_grib2_message(
    param: str,
    val_nh: float,
    val_sh: float,
    shape: Tuple[int, int] = OPEN_DATA_GRID_SHAPE,
    nan_rows: Optional[slice] = None,
) -> bytes:
  """Encodes a real WMO GRIB2 message on the 0.25-deg global grid (IEEE 754 float32)."""
  if shape != OPEN_DATA_GRID_SHAPE:
    raise ValueError(f"Expected shape {OPEN_DATA_GRID_SHAPE}, got {shape}")
  header = _GRIB2_IEEE_HEADERS[param]
  arr = np.full(shape, val_nh, dtype=">f4")
  # In raw ECMWF GRIB2, row 0 is +90N and row 720 is -90S.
  # Rows 361..720 are Southern Hemisphere (< 0 lat).
  arr[shape[0] // 2 + 1 :, :] = val_sh
  if nan_rows is not None:
    arr[nan_rows, :] = np.nan
  return header + arr.tobytes() + b"7777"


def _physical_values_for_param_and_step(
    param: str, step: int
) -> Tuple[float, float]:
  """Returns deterministic (Northern Hemisphere, Southern Hemisphere) WMO values."""
  days_elapsed = step / 24.0
  if param == "2t":
    # 293.15 K -> 20.0 degC in Northern Hemisphere; 250.0 K in Southern Hemisphere
    return 293.15, 250.0
  if param == "sp":
    # 101325 Pa -> 101.325 kPa
    return 101325.0, 50000.0
  if param == "tp":
    # Cumulative 0.005 m per lead day -> 5.0 mm/day after deaccumulation
    return 0.005 * days_elapsed, 0.0
  if param == "ssr":
    # Cumulative 200 W/m^2 * 86400 s per lead day -> 200.0 W/m^2
    return 200.0 * 86400.0 * days_elapsed, 0.0
  if param == "str":
    # Cumulative -50 W/m^2 * 86400 s per lead day -> -50.0 W/m^2
    return -50.0 * 86400.0 * days_elapsed, 0.0
  raise ValueError(f"Unexpected param {param!r}")


class FakeECMWFOpenDataFS:
  """In-memory mock of gs://ecmwf-open-data serving real ecCodes GRIB2 payloads."""

  _STEP_CACHE: Dict[int, Tuple[bytes, bytes]] = {}
  _PARAMS: Tuple[str, ...] = ("2t", "sp", "tp", "ssr", "str")

  def __init__(
      self,
      available_dates: Sequence[str],
      published_steps: Optional[Sequence[int]] = None,
      custom_step_values: Optional[
          Dict[Tuple[str, int], Tuple[float, float]]
      ] = None,
  ):
    self.available_dates = {
        pd.to_datetime(d).strftime("%Y%m%d") for d in available_dates
    }
    self.published_steps = (
        set(published_steps)
        if published_steps is not None
        else set(range(24, 241, 24))
    )
    self.custom_step_values = custom_step_values or {}
    self._local_step_cache: Dict[int, Tuple[bytes, bytes]] = {}
    self.requested_index_paths: List[str] = []
    self.requested_grib_ranges: List[Tuple[str, int, int]] = []

  def _parse_date_and_step(self, path: str) -> Tuple[str, int]:
    fname = path.rsplit("/", 1)[-1]
    date_str = fname[:8]
    step_part = fname.split("-")[1]  # e.g. "24h"
    step = int(step_part.rstrip("h"))
    return date_str, step

  def exists(self, path: str) -> bool:
    date_str, step = self._parse_date_and_step(path)
    return date_str in self.available_dates and step in self.published_steps

  def _get_step_index_and_grib(self, step: int) -> Tuple[bytes, bytes]:
    if not self.custom_step_values and step in self._STEP_CACHE:
      return self._STEP_CACHE[step]
    if step in self._local_step_cache:
      return self._local_step_cache[step]

    lines: List[str] = []
    chunks: List[bytes] = []
    offset = 0
    for param in self._PARAMS:
      if (param, step) in self.custom_step_values:
        val_nh, val_sh = self.custom_step_values[(param, step)]
      else:
        val_nh, val_sh = _physical_values_for_param_and_step(param, step)
      msg = _encode_test_grib2_message(param, val_nh, val_sh)
      lines.append(
          json.dumps({
              "param": param,
              "levtype": "sfc",
              "step": str(step),
              "_offset": offset,
              "_length": len(msg),
          })
      )
      chunks.append(msg)
      offset += len(msg)

    idx_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    grib_bytes = b"".join(chunks)
    if not self.custom_step_values:
      self._STEP_CACHE[step] = (idx_bytes, grib_bytes)
    else:
      self._local_step_cache[step] = (idx_bytes, grib_bytes)
    return idx_bytes, grib_bytes

  def cat(self, paths: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in paths:
      self.requested_index_paths.append(p)
      date_str, step = self._parse_date_and_step(p)
      if date_str not in self.available_dates or step not in self.published_steps:
        raise FileNotFoundError(p)
      idx_bytes, _ = self._get_step_index_and_grib(step)
      out[p] = idx_bytes
    return out

  def cat_ranges(
      self, paths: Sequence[str], starts: Sequence[int], ends: Sequence[int]
  ) -> List[bytes]:
    blobs: List[bytes] = []
    for p, s, e in zip(paths, starts, ends):
      self.requested_grib_ranges.append((p, s, e))
      date_str, step = self._parse_date_and_step(p)
      if date_str not in self.available_dates or step not in self.published_steps:
        raise FileNotFoundError(p)
      _, grib_bytes = self._get_step_index_and_grib(step)
      blobs.append(grib_bytes[s:e])
    return blobs


def test_decode_grib2_message_bitmap_and_param_validation():
  """Verifies real ecCodes GRIB2 decoding, bitmap NaN conversion, and shortName validation."""
  msg = _encode_test_grib2_message(
      "2t",
      val_nh=290.0,
      val_sh=260.0,
      nan_rows=slice(100, 120),
  )
  decoded = decode_grib2_message(
      msg, OPEN_DATA_GRID_SHAPE, param="2t", context="test_bitmap"
  )
  assert decoded.shape == OPEN_DATA_GRID_SHAPE
  assert np.all(np.isnan(decoded[100:120, :]))
  assert np.allclose(decoded[:100, :], 290.0, atol=1e-2)
  assert np.allclose(decoded[361:, :], 260.0, atol=1e-2)

  with pytest.raises(ValueError, match="GRIB2 parameter mismatch"):
    decode_grib2_message(
        msg, OPEN_DATA_GRID_SHAPE, param="tp", context="wrong_param"
    )


def test_find_latest_hres_open_data_date():
  """Verifies walking backwards from reference_date to find newest published 00z run."""
  fs = FakeECMWFOpenDataFS(available_dates=["2026-09-25", "2026-09-26"])
  latest = find_latest_hres_open_data_date(
      reference_date="2026-09-28",
      max_lookback_days=5,
      fs=fs,
  )
  assert latest == pd.Timestamp("2026-09-26")

  with pytest.raises(FileNotFoundError, match="No published ECMWF Open Data"):
    find_latest_hres_open_data_date(
        reference_date="2026-10-15",
        max_lookback_days=3,
        fs=fs,
    )


def test_hres_open_data_extraction_and_spinup_1d_optimization(basins_gdf):
  """Verifies real GRIB2 decoding, unit conversion, lat-flip, and 10x spinup_only_before optimization."""
  fs = FakeECMWFOpenDataFS(
      available_dates=["2026-09-25", "2026-09-26", "2026-09-27"]
  )
  ext = HRESExtractor(source="open_data", fs=fs)

  # Extract 4 days where 2026-09-25 and 2026-09-26 are spin-up (lead 1D only),
  # 2026-09-27 is the forecast issue date (all 10 leads), and 2026-09-28 is unpublished.
  ds = ext.extract_for_basins(
      basins_gdf,
      start_date="2026-09-25",
      end_date="2026-09-28",
      spinup_only_before="2026-09-27",
  )

  assert ds["hres_temperature_2m"].shape == (len(basins_gdf), 4, 10)

  # 1. Spin-up days (indices 0 and 1: 2026-09-25, 2026-09-26):
  #    Only lead_time=1D (index 0) was downloaded; leads 2D..10D are strictly NaN.
  for d_idx in (0, 1):
    assert np.allclose(
        ds["hres_temperature_2m"].values[:, d_idx, 0], 20.0, atol=1e-2
    )
    assert np.allclose(
        ds["hres_surface_pressure"].values[:, d_idx, 0], 101.325, atol=1e-2
    )
    assert np.allclose(
        ds["hres_total_precipitation"].values[:, d_idx, 0], 5.0, atol=1e-2
    )
    assert np.allclose(
        ds["hres_surface_net_solar_radiation"].values[:, d_idx, 0],
        200.0,
        atol=1e-1,
    )
    assert np.allclose(
        ds["hres_surface_net_thermal_radiation"].values[:, d_idx, 0],
        -50.0,
        atol=1e-1,
    )
    assert np.all(np.isnan(ds["hres_temperature_2m"].values[:, d_idx, 1:]))
    assert np.all(np.isnan(ds["hres_total_precipitation"].values[:, d_idx, 1:]))
    assert np.allclose(
        ds["hres_missing_fraction"].values[:, d_idx, 0], 0.0, atol=1e-4
    )
    assert np.allclose(
        ds["hres_missing_fraction"].values[:, d_idx, 1:], 1.0, atol=1e-4
    )

  # 2. Forecast issue date (index 2: 2026-09-27): all 10 lead days are populated!
  assert np.allclose(
      ds["hres_temperature_2m"].values[:, 2, :], 20.0, atol=1e-2
  )
  assert np.allclose(
      ds["hres_surface_pressure"].values[:, 2, :], 101.325, atol=1e-2
  )
  assert np.allclose(
      ds["hres_total_precipitation"].values[:, 2, :], 5.0, atol=1e-2
  )
  assert np.allclose(
      ds["hres_surface_net_solar_radiation"].values[:, 2, :], 200.0, atol=1e-1
  )
  assert np.allclose(
      ds["hres_surface_net_thermal_radiation"].values[:, 2, :],
      -50.0,
      atol=1e-1,
  )
  assert np.allclose(
      ds["hres_missing_fraction"].values[:, 2, :], 0.0, atol=1e-4
  )

  # 3. Unpublished trailing date (index 3: 2026-09-28): all NaN and missing_fraction=1.0
  assert np.all(np.isnan(ds["hres_temperature_2m"].values[:, 3, :]))
  assert np.allclose(
      ds["hres_missing_fraction"].values[:, 3, :], 1.0, atol=1e-4
  )

  # Confirm spin-up days only fetched step=24h
  sep25_steps = [
      p for p in fs.requested_index_paths if "/20260925/" in p
  ]
  sep26_steps = [
      p for p in fs.requested_index_paths if "/20260926/" in p
  ]
  sep27_steps = [
      p for p in fs.requested_index_paths if "/20260927/" in p
  ]
  assert len(sep25_steps) == 1
  assert len(sep26_steps) == 1
  assert len(sep27_steps) == 10


def test_hres_open_data_subdaily_averaging_and_incomplete_run_errors(basins_gdf):
  """Verifies sub-daily 3h step averaging for 2t/sp and FileNotFoundError on incomplete or empty runs."""
  # Publish 3h sub-daily steps (3, 6, ..., 24) for 2026-09-25 where 2t alternates
  # between 283.15 K (10 C) and 303.15 K (30 C), averaging to 20.0 C.
  subdaily_steps = [3, 6, 9, 12, 15, 18, 21, 24]
  custom_vals: Dict[Tuple[str, int], Tuple[float, float]] = {}
  for idx, s in enumerate(subdaily_steps):
    temp_k = 283.15 if idx % 2 == 0 else 303.15
    custom_vals[("2t", s)] = (temp_k, 250.0)

  fs = FakeECMWFOpenDataFS(
      available_dates=["2026-09-25"],
      published_steps=subdaily_steps,
      custom_step_values=custom_vals,
  )
  ext = HRESExtractor(source="open_data", fs=fs)
  ds = ext.extract_for_basins_open_data(
      basins_gdf,
      start_date="2026-09-25",
      end_date="2026-09-25",
      spinup_only_before="2026-09-26",
  )
  assert np.allclose(ds["hres_temperature_2m"].values[:, 0, 0], 20.0, atol=1e-2)
  assert np.allclose(ds["hres_total_precipitation"].values[:, 0, 0], 5.0, atol=1e-2)

  # If zero dates in the window are published, raise FileNotFoundError
  with pytest.raises(FileNotFoundError, match="No published ECMWF Open Data"):
    ext.extract_for_basins_open_data(
        basins_gdf,
        start_date="2026-10-01",
        end_date="2026-10-02",
    )

  # If step 24h is published but step 48h..240h are missing when 10 leads are requested,
  # raise FileNotFoundError instead of silently returning partial leads.
  with pytest.raises(FileNotFoundError, match="Incomplete ECMWF Open Data HRES run"):
    ext.extract_for_basins_open_data(
        basins_gdf,
        start_date="2026-09-25",
        end_date="2026-09-25",
        spinup_only_before=None,
    )


def test_read_hot_start_state_date_from_file_and_dir(tmp_path):
  """Verifies reading saved hot-start timestamps and raising on invalid inputs."""
  state_dir = tmp_path / "hot_start"
  state_dir.mkdir()

  with pytest.raises(FileNotFoundError, match="No .npz hot-start state files"):
    read_hot_start_state_date(state_dir)

  bad_txt = tmp_path / "not_npz.txt"
  bad_txt.write_text("hello")
  with pytest.raises(ValueError, match="Expected a .npz hot-start state file"):
    read_hot_start_state_date(bad_txt)

  bad_npz = tmp_path / "missing_date_key.npz"
  np.savez(bad_npz, h=np.zeros((1, 64)))
  with pytest.raises(ValueError, match="does not contain any recognized date key"):
    read_hot_start_state_date(bad_npz)

  f1 = state_dir / "state_basin_1.npz"
  f2 = state_dir / "state_basin_2.npz"
  np.savez(f1, h=np.zeros((1, 64)), date=np.array("2026-09-24"))
  np.savez(f2, h=np.zeros((1, 64)), last_date=np.array("2026-09-22"))

  assert read_hot_start_state_date(f1) == pd.Timestamp("2026-09-24")
  # Directory returns the minimum (earliest) date across all basin states
  assert read_hot_start_state_date(state_dir) == pd.Timestamp("2026-09-22")


def _build_synthetic_imerg_half_hourly_cube(
    basins_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    daily_mm: float,
    trailing_incomplete_day: bool = False,
) -> xr.Dataset:
  """Builds a real 3D half-hourly IMERG dataset on a 0.1-deg grid covering basins_gdf."""
  minx, miny, maxx, maxy = basins_gdf.total_bounds
  lats = np.arange(np.ceil(maxy) + 1.0, np.floor(miny) - 1.0, -0.1, dtype=np.float32)
  lons = np.arange(np.floor(minx) - 1.0, np.ceil(maxx) + 1.0, 0.1, dtype=np.float32)
  end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=30)
  times = pd.date_range(start_date, end_ts, freq="30min")
  flux_val = np.float32(daily_mm / 86400.0)
  data = np.full((len(times), len(lats), len(lons)), flux_val, dtype=np.float32)
  if trailing_incomplete_day:
    # Set the final day's last 12 half-hourly steps to NaN to simulate upstream latency
    data[-12:, :, :] = np.nan
  return xr.Dataset(
      data_vars={
          "precipitation_surface": (["time", "latitude", "longitude"], data)
      },
      coords={
          "time": times.values,
          "latitude": lats,
          "longitude": lons,
      },
  )


def _write_synthetic_cpc_netcdf(
    target_path: Path,
    basins_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    daily_mm: float,
    trailing_nan_day: bool = False,
) -> None:
  """Writes a real NOAA PSL CPC yearly NetCDF file on the global 0.5-deg 0..360 longitude grid."""
  del basins_gdf
  lats = np.arange(89.75, -90.0, -0.5, dtype=np.float32)
  lons = np.arange(0.25, 360.0, 0.5, dtype=np.float32)
  times = pd.date_range(start_date, end_date, freq="D")
  data = np.full((len(times), len(lats), len(lons)), daily_mm, dtype=np.float32)
  if trailing_nan_day:
    data[-1, :, :] = np.nan
  ds = xr.Dataset(
      data_vars={"precip": (["time", "lat", "lon"], data)},
      coords={"time": times.values, "lat": lats, "lon": lons},
  )
  target_path.parent.mkdir(parents=True, exist_ok=True)
  ds.to_netcdf(target_path)


def test_coldstart_and_hotstart_end_to_end_workflow(
    tmp_path, basins_gdf, monkeypatch
):
  """Tests Cold-Start initialization followed by Hot-Start incremental append & NaN healing."""
  basin_ids = [str(b) for b in basins_gdf.index]
  out_dir = tmp_path / "realtime_forcing"
  cpc_cache_dir = tmp_path / "cpc_cache"

  all_dates = [f"2026-09-{d:02d}" for d in range(21, 28)]
  fs = FakeECMWFOpenDataFS(available_dates=all_dates)

  # Prepare Run 1 upstream feeds (2026-09-21..2026-09-25 where 2026-09-25 is incomplete/NaN)
  imerg_cube_run1 = _build_synthetic_imerg_half_hourly_cube(
      basins_gdf,
      start_date="2026-09-21",
      end_date="2026-09-25",
      daily_mm=4.0,
      trailing_incomplete_day=True,
  )
  monkeypatch.setitem(
      dynamical_mod._GLOBAL_DATASET_CACHE,
      "nasa-imerg-analysis-early",
      imerg_cube_run1,
  )

  cpc_state = {"run": 1}

  def _fake_cpc_download(url: str, target_path: str, **kwargs) -> str:
    del url, kwargs
    if cpc_state["run"] == 1:
      _write_synthetic_cpc_netcdf(
          Path(target_path),
          basins_gdf,
          start_date="2026-09-21",
          end_date="2026-09-25",
          daily_mm=3.0,
          trailing_nan_day=True,
      )
    else:
      _write_synthetic_cpc_netcdf(
          Path(target_path),
          basins_gdf,
          start_date="2026-09-21",
          end_date="2026-09-27",
          daily_mm=6.0,
          trailing_nan_day=False,
      )
    return target_path

  monkeypatch.setattr(
      "multimet.utils.cpc.download_http_file", _fake_cpc_download
  )

  # 1. Execute Cold-Start up to 2026-09-25 (with lookback_days=4 -> 2026-09-21..2026-09-25)
  cold_res = fetch_realtime_multimet(
      basins=basins_gdf,
      output_dir=out_dir,
      mode="coldstart",
      reference_date="2026-09-25",
      lookback_days=4,
      full_forecast_days=1,
      cpc_cache_dir=str(cpc_cache_dir),
      hres_fs=fs,
  )
  assert set(cold_res.keys()) == {"HRES", "IMERG", "CPC"}
  assert cold_res.reference_date == pd.Timestamp("2026-09-25")
  assert cold_res.product_windows["HRES"] == (
      pd.Timestamp("2026-09-21"),
      pd.Timestamp("2026-09-25"),
  )

  # Verify on-disk Zarr stores after Cold-Start
  with xr.open_zarr(cold_res["HRES"]) as ds_hres_1:
    assert len(ds_hres_1["date"]) == 5
    # 2026-09-21..24 have lead 1D valid and leads 2D..10D strictly NaN
    assert np.all(~np.isnan(ds_hres_1["hres_temperature_2m"].values[:, :4, 0]))
    assert np.all(np.isnan(ds_hres_1["hres_temperature_2m"].values[:, :4, 1:]))
    assert np.allclose(ds_hres_1["hres_missing_fraction"].values[:, :4, 0], 0.0)
    assert np.allclose(ds_hres_1["hres_missing_fraction"].values[:, :4, 1:], 1.0)
    # 2026-09-25 (index 4) has all 10 lead days valid with missing_fraction=0.0!
    assert np.all(~np.isnan(ds_hres_1["hres_temperature_2m"].values[:, 4, :]))
    assert np.allclose(ds_hres_1["hres_missing_fraction"].values[:, 4, :], 0.0)

  with xr.open_zarr(cold_res["IMERG"]) as ds_imerg_1:
    assert len(ds_imerg_1["date"]) == 5
    # 2026-09-25 (index 4) is strictly NaN due to simulated 1-day lag
    assert np.allclose(ds_imerg_1["imerg_precipitation"].values[:, :4], 4.0, atol=1e-3)
    assert np.all(np.isnan(ds_imerg_1["imerg_precipitation"].values[:, 4]))
    assert np.allclose(ds_imerg_1["imerg_missing_fraction"].values[:, :4], 0.0)
    assert np.allclose(ds_imerg_1["imerg_missing_fraction"].values[:, 4], 1.0)

  with xr.open_zarr(cold_res["CPC"]) as ds_cpc_1:
    assert len(ds_cpc_1["date"]) == 5
    assert np.allclose(ds_cpc_1["cpc_precipitation"].values[:, :4], 3.0, atol=1e-3)
    assert np.all(np.isnan(ds_cpc_1["cpc_precipitation"].values[:, 4]))
    assert np.allclose(ds_cpc_1["cpc_missing_fraction"].values[:, :4], 0.0)
    assert np.allclose(ds_cpc_1["cpc_missing_fraction"].values[:, 4], 1.0)

  # Confirm inspect_store_last_valid_date backs up over the trailing NaN on 2026-09-25
  writer = MultiMetZarrWriter(out_dir)
  assert inspect_store_last_valid_date(writer, Product.IMERG, basin_ids) == pd.Timestamp(
      "2026-09-24"
  )
  assert inspect_store_last_valid_date(writer, Product.CPC, basin_ids) == pd.Timestamp(
      "2026-09-24"
  )
  assert inspect_store_last_valid_date(writer, Product.HRES, basin_ids) == pd.Timestamp(
      "2026-09-25"
  )

  # 2. Advance upstream feeds to Run 2 (2026-09-27 published, 2026-09-25 healed)
  cpc_state["run"] = 2
  imerg_cube_run2 = _build_synthetic_imerg_half_hourly_cube(
      basins_gdf,
      start_date="2026-09-21",
      end_date="2026-09-27",
      daily_mm=8.0,
      trailing_incomplete_day=False,
  )
  monkeypatch.setitem(
      dynamical_mod._GLOBAL_DATASET_CACHE,
      "nasa-imerg-analysis-early",
      imerg_cube_run2,
  )

  hot_res = fetch_realtime_multimet(
      basins=basins_gdf,
      output_dir=out_dir,
      mode="hotstart",
      reference_date="2026-09-27",
      full_forecast_days=1,
      cpc_cache_dir=str(cpc_cache_dir),
      hres_fs=fs,
  )
  assert hot_res.product_windows["IMERG"] == (
      pd.Timestamp("2026-09-24"),
      pd.Timestamp("2026-09-27"),
  )
  assert hot_res.product_windows["CPC"] == (
      pd.Timestamp("2026-09-24"),
      pd.Timestamp("2026-09-27"),
  )
  assert hot_res.product_windows["HRES"] == (
      pd.Timestamp("2026-09-25"),
      pd.Timestamp("2026-09-27"),
  )

  with xr.open_zarr(hot_res["IMERG"]) as ds_imerg_2:
    assert len(ds_imerg_2["date"]) == 7  # 2026-09-21 .. 2026-09-27
    # 2026-09-21..24 preserved as 4.0; 2026-09-25 (index 4) healed to 8.0; 26, 27 are 8.0!
    assert np.allclose(ds_imerg_2["imerg_precipitation"].values[:, :4], 4.0, atol=1e-3)
    assert np.allclose(ds_imerg_2["imerg_precipitation"].values[:, 4:], 8.0, atol=1e-3)
    assert np.allclose(ds_imerg_2["imerg_missing_fraction"].values, 0.0)

  with xr.open_zarr(hot_res["CPC"]) as ds_cpc_2:
    assert len(ds_cpc_2["date"]) == 7
    assert np.allclose(ds_cpc_2["cpc_precipitation"].values[:, :4], 3.0, atol=1e-3)
    assert np.allclose(ds_cpc_2["cpc_precipitation"].values[:, 4:], 6.0, atol=1e-3)
    assert np.allclose(ds_cpc_2["cpc_missing_fraction"].values, 0.0)

  with xr.open_zarr(hot_res["HRES"]) as ds_hres_2:
    assert len(ds_hres_2["date"]) == 7  # 2026-09-21 .. 2026-09-27
    # 2026-09-25 (index 4) STILL has all 10 lead days valid (missing_fraction=0.0, not clobbered by 1D spin-up!)
    assert np.all(~np.isnan(ds_hres_2["hres_temperature_2m"].values[:, 4, :]))
    assert np.allclose(ds_hres_2["hres_missing_fraction"].values[:, 4, :], 0.0)
    # 2026-09-26 (index 5) has lead 1D valid
    assert np.all(~np.isnan(ds_hres_2["hres_temperature_2m"].values[:, 5, 0]))
    # 2026-09-27 (index 6) has all 10 lead days valid!
    assert np.all(~np.isnan(ds_hres_2["hres_temperature_2m"].values[:, 6, :]))


def test_inspect_store_last_valid_date_checks_all_bands_and_all_basins(tmp_path):
  """Verifies inspect_store_last_valid_date rejects dates where a secondary band or single basin is NaN."""
  writer = MultiMetZarrWriter(tmp_path / "strict_check")
  basins = ["basin_1", "basin_2"]
  dates = pd.date_range("2026-09-21", "2026-09-23", freq="D")
  leads = pd.to_timedelta(range(1, 11), unit="D")

  data_vars = {}
  for band in PRODUCT_BANDS[Product.HRES]:
    arr = np.ones((2, 3, 10), dtype=np.float32)
    if band == "hres_surface_net_thermal_radiation":
      # Corrupt only a non-primary band on 2026-09-23 (index 2)
      arr[:, 2, :] = np.nan
    elif band == "hres_temperature_2m":
      # Corrupt only basin_2 on 2026-09-22 (index 1) at lead_time=5D
      arr[1, 1, 4] = np.nan
    data_vars[band] = (["basin", "date", "lead_time"], arr)
  data_vars["hres_missing_fraction"] = (
      ["basin", "date", "lead_time"],
      np.zeros((2, 3, 10), dtype=np.float32),
  )

  ds = xr.Dataset(
      data_vars=data_vars,
      coords={"basin": basins, "date": dates.values, "lead_time": leads.values},
  )
  writer.write_or_append(ds, Product.HRES)

  # When only lead 1D is required, 2026-09-23 is rejected (thermal radiation is NaN),
  # and 2026-09-22 is accepted (lead 1D is valid across all bands and basins).
  assert inspect_store_last_valid_date(
      writer, Product.HRES, basins, require_all_leads_on_last_date=False
  ) == pd.Timestamp("2026-09-22")

  # When all 10 leads are required, 2026-09-22 is ALSO rejected because basin_2
  # has NaN at lead 5D, backing up to 2026-09-21!
  assert inspect_store_last_valid_date(
      writer, Product.HRES, basins, require_all_leads_on_last_date=True
  ) == pd.Timestamp("2026-09-21")


def test_default_coldstart_lookback_and_validation_errors(tmp_path, basins_gdf):
  """Verifies 365-day default coldstart lookback and strict validation on imerg_source / weights_cache."""
  with pytest.raises(ValueError, match="Invalid imerg_source"):
    RealtimeForcingFetcher(tmp_path, imerg_source="auto")

  fetcher = RealtimeForcingFetcher(tmp_path)
  basin_ids = [str(b) for b in basins_gdf.index]
  ref_dt = pd.Timestamp("2026-09-27")
  start_dt, end_dt = fetcher.plan_product_window(
      Product.HRES,
      basin_ids=basin_ids,
      reference_date=ref_dt,
      mode="coldstart",
  )
  assert end_dt == ref_dt
  assert (end_dt - start_dt).days == DEFAULT_COLDSTART_LOOKBACK_DAYS == 365
  assert start_dt == pd.Timestamp("2025-09-27")

  with pytest.raises(FileNotFoundError, match="Weights cache path does not exist"):
    fetcher.fetch(
        basins_gdf,
        mode="hotstart",
        reference_date="2026-09-27",
        products=["HRES"],
        weights_cache=str(tmp_path / "nonexistent_weights.npz"),
    )


def test_cli_arg_parser_and_main(tmp_path, monkeypatch):
  """Verifies CLI argument parsing and execution via python -m multimet.timeseries_extractors.realtime."""
  geojson_path = (
      Path(__file__).parent
      / "test_data"
      / "shapefiles"
      / "us"
      / "us_basin_shapes.geojson"
  )
  out_dir = tmp_path / "cli_out"

  parser = build_arg_parser()
  args = parser.parse_args([
      "--basins_path",
      str(geojson_path),
      "--output_dir",
      str(out_dir),
      "--mode",
      "coldstart",
      "--reference_date",
      "2026-09-27",
      "--lookback_days",
      "2",
      "--products",
      "HRES",
  ])
  assert args.mode == "coldstart"
  assert args.reference_date == "2026-09-27"
  assert args.lookback_days == 2

  fs = FakeECMWFOpenDataFS(
      available_dates=["2026-09-25", "2026-09-26", "2026-09-27"]
  )
  monkeypatch.setattr(
      "multimet.timeseries_extractors.hres.HRESExtractor._get_open_data_fs",
      lambda self: fs,
  )

  res = realtime_main([
      "--basins_path",
      str(geojson_path),
      "--output_dir",
      str(out_dir),
      "--mode",
      "coldstart",
      "--reference_date",
      "2026-09-27",
      "--lookback_days",
      "2",
      "--products",
      "HRES",
  ])
  assert "HRES" in res
  summary = res.summary()
  assert summary["mode"] == "coldstart"
  assert summary["reference_date"] == "2026-09-27"
  assert summary["start_date"] == "2026-09-25"
  assert summary["end_date"] == "2026-09-27"


