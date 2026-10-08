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

"""Unit and integration tests for multimet.weather_fetcher.sync and CLI."""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import xarray as xr

from multimet.weather_fetcher import cli
from multimet.weather_fetcher.config import (
    MAX_LEAD_HOURS,
    MSLP_OFFSET_HPA,
    N_LAT,
    N_LON,
    output_lead_hours,
    run_lead_hours,
    RUN_METADATA_FILE,
    to_stored_units,
)
from multimet.weather_fetcher.fetcher import WeatherDataFetcher
from multimet.weather_fetcher.sync import (
    _build_2d_to_global_reprojector,
    _extract_hres_streams,
    _latest_hres_run_info,
    _resample_1d_rectilinear_to_global,
    aggregate_rates,
    current_run_dir,
    download_model_run,
    DYNAMICAL_MODELS,
    IncompleteRunError,
    is_plane_complete,
    latest_init_time_str,
    list_available_runs,
    prune_old_runs,
    swap_current_symlink,
    sync_all_models,
    WeatherSynchronizer,
)

_UTC = datetime.timezone.utc
_PSL_FILL = np.float32(-9.96921e36)
_GRID_LATS = np.linspace(90.0, -90.0, N_LAT)
_GRID_LONS = np.linspace(-180.0, 180.0, N_LON, endpoint=False)


def _grid_index(lat: float, lon: float) -> Tuple[int, int]:
  """Row/column of the viewer grid cell centred at (lat, lon)."""
  return int(round((90.0 - lat) / 0.25)), int(round((lon + 180.0) / 0.25))


# ---------------------------------------------------------------------------
# Synthetic upstream datasets at native resolution
# ---------------------------------------------------------------------------


def _build_native_forecast_dataset(
    init_times: Sequence[np.datetime64],
    lead_hours: Sequence[int],
    members: bool = False,
    rate_kg_m2_s: float = 1.0 / 3600.0,
) -> xr.Dataset:
  """Constructs a native 721 x 1440 dynamical.org-schema forecast Dataset.

  Like the upstream stores, precipitation is ``NaN`` at lead 0 (no preceding
  step) and temperature differs per init so the newest run is identifiable.
  """
  shape = (len(init_times), len(lead_hours), N_LAT, N_LON)
  precip = np.full(shape, rate_kg_m2_s, np.float32)
  if int(lead_hours[0]) == 0:
    precip[:, 0] = np.nan
  temp = np.empty(shape, np.float32)
  for i in range(len(init_times)):
    temp[i] = 12.5 + i
  data = {
      "precipitation_surface": (precip, "kg m-2 s-1"),
      "temperature_2m": (temp, "degree_Celsius"),
      "pressure_reduced_to_mean_sea_level": (
          np.full(shape, 101325.0, np.float32),
          "Pa",
      ),
      "wind_u_10m": (np.full(shape, 3.0, np.float32), "m s-1"),
      "wind_v_10m": (np.full(shape, -4.0, np.float32), "m s-1"),
  }
  coords: Dict[str, Any] = {
      "init_time": np.array(init_times, dtype="datetime64[ns]"),
      "lead_time": np.array(lead_hours, dtype="timedelta64[h]").astype(
          "timedelta64[ns]"
      ),
      "latitude": np.linspace(90.0, -90.0, N_LAT, dtype=np.float32),
      "longitude": np.linspace(
          -180.0, 180.0, N_LON, endpoint=False, dtype=np.float32
      ),
  }
  dims: Tuple[str, ...] = ("init_time", "lead_time", "latitude", "longitude")
  if members:
    coords["ensemble_member"] = np.arange(3)
    dims = (
        "init_time", "lead_time", "ensemble_member", "latitude", "longitude"
    )
    data = {
        k: (np.repeat(v[:, :, None], 3, axis=2), u)
        for k, (v, u) in data.items()
    }
  return xr.Dataset(
      {k: (dims, v, {"units": u}) for k, (v, u) in data.items()},
      coords=coords,
  )


def _hrrr_native_latlon() -> Tuple[np.ndarray, np.ndarray]:
  """Cell-centre latitude/longitude of the native 1059 x 1799 HRRR grid.

  Uses the Lambert conformal parameters published in the dynamical.org store
  (``spatial_ref``: lat_0 = 38.5, lon_0 = -97.5, lat_1 = lat_2 = 38.5,
  sphere R = 6371229 m, 3 km cells, GeoTransform origin
  (-2699020.14, 1588193.85)).
  """
  import pyproj

  lcc = pyproj.CRS.from_proj4(
      "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5"
      " +R=6371229 +units=m +no_defs"
  )
  geo = pyproj.CRS.from_proj4("+proj=longlat +R=6371229 +no_defs")
  transformer = pyproj.Transformer.from_crs(lcc, geo, always_xy=True)
  x = -2699020.142521929 + 3000.0 * (np.arange(1799) + 0.5)
  y = 1588193.847443335 - 3000.0 * (np.arange(1059) + 0.5)
  xx, yy = np.meshgrid(x, y)
  lon, lat = transformer.transform(xx, yy)
  return lat.astype(np.float32), lon.astype(np.float32)


def _build_hrrr_dataset(
    init_time: np.datetime64, lead_hours: Sequence[int]
) -> xr.Dataset:
  """Constructs a native-grid HRRR Dataset with longitude-dependent fields."""
  lat2d, lon2d = _hrrr_native_latlon()
  shape = (1, len(lead_hours)) + lat2d.shape
  temp = np.broadcast_to(
      (20.0 + 0.1 * lon2d).astype(np.float32), shape
  ).copy()
  precip = np.full(shape, 2.0 / 3600.0, np.float32)
  precip[:, 0] = np.nan
  dims = ("init_time", "lead_time", "y", "x")
  return xr.Dataset(
      {
          "precipitation_surface": (dims, precip, {"units": "kg m-2 s-1"}),
          "temperature_2m": (dims, temp, {"units": "degree_Celsius"}),
          "pressure_reduced_to_mean_sea_level": (
              dims,
              np.full(shape, 101500.0, np.float32),
              {"units": "Pa"},
          ),
          "wind_u_10m": (
              dims, np.full(shape, 4.0, np.float32), {"units": "m s-1"}
          ),
          "wind_v_10m": (
              dims, np.full(shape, 3.0, np.float32), {"units": "m s-1"}
          ),
      },
      coords={
          "init_time": np.array([init_time], dtype="datetime64[ns]"),
          "lead_time": np.array(lead_hours, dtype="timedelta64[h]").astype(
              "timedelta64[ns]"
          ),
          "latitude": (("y", "x"), lat2d),
          "longitude": (("y", "x"), lon2d),
      },
  )


def _build_imerg_dataset(
    start: np.datetime64,
    n_frames: int,
    frame_rate_mm_h: Callable[[int], float],
) -> xr.Dataset:
  """Constructs a native 1800 x 3600 half-hourly IMERG-schema Dataset."""
  times = start + np.arange(n_frames) * np.timedelta64(30, "m")
  data = np.empty((n_frames, 1800, 3600), np.float32)
  for k in range(n_frames):
    data[k] = np.float32(frame_rate_mm_h(k) / 3600.0)
  return xr.Dataset(
      {
          "precipitation_surface": (
              ("time", "latitude", "longitude"),
              data,
              {"units": "kg m-2 s-1"},
          )
      },
      coords={
          "time": times.astype("datetime64[ns]"),
          "latitude": np.linspace(89.95, -89.95, 1800, dtype=np.float32),
          "longitude": np.linspace(-179.95, 179.95, 3600, dtype=np.float32),
      },
  )


def _write_psl_cpc_file(
    path: Path,
    dates: Sequence[datetime.date],
    valid_dates: Sequence[datetime.date],
) -> None:
  """Writes a NOAA PSL style ``precip.<year>.nc`` (360 x 720, 0.25..359.75).

  Land (eastern hemisphere columns) holds ``day-of-month`` mm/day on valid
  dates; oceans (western hemisphere) and unpublished days are fill values.
  """
  precip = np.full((len(dates), 360, 720), np.nan, np.float32)
  valid = set(valid_dates)
  for k, day in enumerate(dates):
    if day in valid:
      precip[k, :, :360] = float(day.day)
  ds = xr.Dataset(
      {
          "precip": (
              ("time", "lat", "lon"),
              precip,
              {"units": "mm", "long_name": "Daily total of precipitation"},
          )
      },
      coords={
          "time": np.array(
              [np.datetime64(d.isoformat()) for d in dates],
              dtype="datetime64[ns]",
          ),
          "lat": np.linspace(89.75, -89.75, 360, dtype=np.float32),
          "lon": np.linspace(0.25, 359.75, 720, dtype=np.float32),
      },
  )
  ds.to_netcdf(path, encoding={"precip": {"_FillValue": _PSL_FILL}})


class _CpcUpstream:
  """Fake NOAA PSL server: ``download_http_file`` copies prepared files."""

  def __init__(self, staging: Path, published_years: Sequence[int]):
    self.staging = staging
    self.published_years = set(published_years)
    self.downloads: List[str] = []

  def exists(self, url: str, **_kwargs: Any) -> bool:
    return int(url.rsplit(".", 2)[-2]) in self.published_years

  def download(self, url: str, dest_path: str, **_kwargs: Any) -> str:
    year = int(url.rsplit(".", 2)[-2])
    if year not in self.published_years:
      raise FileNotFoundError(f"not published (HTTP 404): {url}")
    self.downloads.append(url)
    shutil.copyfile(self.staging / f"precip.{year}.nc", dest_path)
    return dest_path


def _grib2_message(values: np.ndarray, category: int, number: int) -> bytes:
  """Minimal GRIB2 message (sections 0, 1, 4, 7) with big-endian float32."""
  data = np.asarray(values, dtype=">f4").tobytes()
  sec1 = struct.pack(">IB", 21, 1) + bytes(16)
  sec4 = struct.pack(">IB", 11, 4) + bytes(4) + bytes([category, number])
  sec7 = struct.pack(">IB", 5 + len(data), 7) + data
  body = sec1 + sec4 + sec7
  total_len = 16 + len(body) + 4
  sec0 = b"GRIB" + bytes(2) + bytes([0, 2]) + struct.pack(">Q", total_len)
  return sec0 + body + b"7777"


class _FakeOpenDataGcs:
  """gcsfs stand-in serving one synthetic HRES run from byte ranges."""

  _PARAM_CODES = {
      "tp": (1, 52),
      "2t": (0, 0),
      "msl": (3, 1),
      "10u": (2, 2),
      "10v": (2, 3),
  }

  def __init__(self, date_str: str, cycle: str, max_lead: int):
    self.date_str, self.cycle, self.max_lead = date_str, cycle, max_lead
    self._files: Dict[str, bytes] = {}
    self._index: Dict[str, str] = {}

  def _prefix(self, lead_h: int) -> str:
    hh = self.cycle[:2]
    return (
        f"ecmwf-open-data/{self.date_str}/{self.cycle}/ifs/0p25/oper/"
        f"{self.date_str}{hh}0000-{lead_h}h-oper-fc"
    )

  @staticmethod
  def field(param: str, lead_h: int) -> np.ndarray:
    values = {
        "tp": 0.001 * lead_h,  # 1 mm accumulated per hour, in metres
        "2t": 293.15,
        "msl": 101325.0,
        "10u": 3.0,
        "10v": -4.0,
    }[param]
    return np.full((N_LAT, N_LON), values, np.float32)

  def _build(self, lead_h: int) -> None:
    prefix = self._prefix(lead_h)
    if prefix in self._index:
      return
    blob = b""
    lines = []
    for param, (cat, num) in self._PARAM_CODES.items():
      msg = _grib2_message(self.field(param, lead_h), cat, num)
      lines.append(
          json.dumps({
              "param": param,
              "levtype": "sfc",
              "_offset": len(blob),
              "_length": len(msg),
          })
      )
      blob += msg
    self._files[f"{prefix}.grib2"] = blob
    self._index[prefix] = "\n".join(lines) + "\n"

  def exists(self, path: str) -> bool:
    return path == f"{self._prefix(self.max_lead)}.index"

  def cat(self, path: str) -> bytes:
    prefix = path[: -len(".index")]
    lead_h = int(prefix.rsplit("-", 3)[1][:-1])
    self._build(lead_h)
    return self._index[prefix].encode("utf-8")

  def cat_file(self, path: str, start: int, end: int) -> bytes:
    return self._files[path][start:end]


# ---------------------------------------------------------------------------
# Temporal aggregation and unit conversions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_lead_hours_and_run_lead_hours() -> None:
  """3-hourly lead filtering up to 240 h and per-model stored lead labels."""
  gfs = list(range(121)) + list(range(123, 385, 3))
  out = output_lead_hours(gfs)
  assert out[:4] == [0, 3, 6, 9]
  assert out[-1] == 240
  assert len(out) == 81
  assert output_lead_hours([0, 6, 12, 246]) == [0, 6, 12]

  assert run_lead_hours("noaa_gfs", 81)[-1] == 240
  assert run_lead_hours("ecmwf_hres", 65)[-1] == 240
  assert run_lead_hours("ecmwf_hres", 49)[-1] == 144
  assert run_lead_hours("ecmwf_hres", 50)[-1] == 150
  assert run_lead_hours("ecmwf_aifs", 41) == list(range(0, 241, 6))
  assert run_lead_hours("noaa_hrrr", 17) == list(range(0, 49, 3))
  assert run_lead_hours("noaa_cpc", 11) == list(range(0, 241, 24))
  with pytest.raises(KeyError):
    run_lead_hours("graphcast", 81)
  with pytest.raises(ValueError):
    run_lead_hours("noaa_hrrr", 18)


@pytest.mark.unit
def test_aggregate_rates_preserves_nan_and_requires_full_coverage() -> None:
  """Hourly means, NaN propagation, gaps -> NaN and the lead-0 domain mask."""
  rates = np.array([np.nan, 1, 1, 1, 4, 4, 4], np.float32)[:, None, None]
  agg = aggregate_rates(rates, [0, 1, 2, 3, 4, 5, 6], [0, 3, 6])
  np.testing.assert_allclose(agg[:, 0, 0], [0.0, 1.0, 4.0])

  mixed = aggregate_rates(
      np.array([0, 2, 2, 1], np.float32)[:, None, None], [0, 1, 2, 5], [0, 2, 5]
  )
  np.testing.assert_allclose(mixed[:, 0, 0], [0.0, 2.0, 1.0])

  # A NaN cell in one hourly step makes the 3 h mean NaN (never a partial sum).
  nan_cell = np.array([np.nan, 1.0, np.nan, 1.0], np.float32)[:, None, None]
  assert np.isnan(aggregate_rates(nan_cell, [0, 1, 2, 3], [0, 3])[1, 0, 0])

  # Missing input steps inside an interval make the whole plane NaN.
  gap = aggregate_rates(
      np.array([np.nan, 3.0, 3.0], np.float32)[:, None, None], [0, 1, 2], [0, 3]
  )
  assert np.isnan(gap[1, 0, 0])
  six_hourly = aggregate_rates(
      np.array([np.nan, 2.0, 4.0], np.float32)[:, None, None],
      [0, 6, 12],
      [0, 6, 12],
  )
  np.testing.assert_allclose(six_hourly[:, 0, 0], [0.0, 2.0, 4.0])

  # Lead 0 is zero inside the domain of the first informative plane only.
  domain = np.full((2, 2, 2), np.nan, np.float32)
  domain[1, 0, 0] = 5.0
  lead0 = aggregate_rates(domain, [0, 3], [0, 3])
  assert lead0[0, 0, 0] == 0.0
  assert np.isnan(lead0[0, 1, 1])
  assert lead0[1, 0, 0] == 5.0

  with pytest.raises(ValueError):
    aggregate_rates(rates, [0, 1, 2], [0, 3])
  with pytest.raises(ValueError):
    aggregate_rates(rates[:3], [0, 2, 1], [0, 3])
  with pytest.raises(ValueError):
    aggregate_rates(rates[:3], [3, 4, 5], [0, 3])


@pytest.mark.unit
def test_to_stored_units_conversions_nan_and_errors() -> None:
  """Explicit upstream units, NaN preservation and loud failures."""
  np.testing.assert_allclose(
      to_stored_units("precip", [1.0 / 3600.0], "kg m-2 s-1").astype(float),
      [1.0],
      rtol=1e-3,
  )
  np.testing.assert_allclose(
      to_stored_units("precip", [48.0], "mm/day").astype(float),
      [2.0],
      rtol=1e-3,
  )
  np.testing.assert_allclose(
      to_stored_units("precip", [1.0 / 3600.0]).astype(float), [1.0], rtol=1e-3
  )
  precip = to_stored_units("precip", [np.nan, -0.5, 2.0], "mm/h").astype(float)
  assert np.isnan(precip[0])
  assert precip[1] == 0.0
  assert precip[2] == pytest.approx(2.0, abs=1e-3)

  assert float(to_stored_units("temp", [293.15], "K")[0]) == pytest.approx(
      20.0, abs=0.02
  )
  assert float(to_stored_units("temp", [20.0], "degree_Celsius")[0]) == (
      pytest.approx(20.0, abs=0.02)
  )
  assert np.isnan(to_stored_units("temp", [np.nan], "K")[0])

  stored_mslp = float(to_stored_units("mslp", [101325.0], "Pa")[0])
  assert pytest.approx(stored_mslp + MSLP_OFFSET_HPA, abs=0.1) == 1013.25
  assert float(to_stored_units("mslp", [1013.25], "hPa")[0]) == pytest.approx(
      13.25, abs=0.02
  )
  assert float(to_stored_units("u10", [3.0], "m s-1")[0]) == 3.0

  with pytest.raises(ValueError):
    to_stored_units("precip", [1.0], "inches")
  with pytest.raises(ValueError):
    to_stored_units("temp", [1.0], "F")
  with pytest.raises(ValueError):
    to_stored_units("mslp", [1.0], "atm")
  with pytest.raises(ValueError):
    to_stored_units("humidity", [1.0])


# ---------------------------------------------------------------------------
# Spatial regridding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resample_fine_grid_averages_cells_and_applies_coverage_rule() -> None:
  """Native IMERG 0.1 deg grid -> 0.25 deg cell means with NaN handling."""
  lats = np.linspace(89.95, -89.95, 1800)
  lons = np.linspace(-179.95, 179.95, 3600)
  lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
  field = (
      10.0 * np.cos(np.radians(lat2d)) * np.sin(np.radians(lon2d))
      + 0.1 * lat2d
  ).astype(np.float32)
  # Target (40.0N, 20.0E) collects 2 x 2 source cells: one NaN -> 3/4 < 0.8.
  field[499, 1999] = np.nan
  # Target (40.25N, 20.25E) collects 3 x 3 source cells: one NaN -> 8/9 ok.
  field[498, 2001] = np.nan

  out = _resample_1d_rectilinear_to_global(field[None], lats, lons)
  assert out.shape == (1, N_LAT, N_LON)
  expected = 10.0 * np.cos(np.radians(_GRID_LATS))[:, None] * np.sin(
      np.radians(_GRID_LONS)
  )[None, :] + 0.1 * _GRID_LATS[:, None]
  err = np.abs(out[0] - expected)
  i_nan, j_nan = _grid_index(40.0, 20.0)
  i_ok, j_ok = _grid_index(40.25, 20.25)
  assert np.isnan(out[0, i_nan, j_nan])
  assert np.isfinite(out[0, i_ok, j_ok])
  err[i_nan, j_nan] = 0.0
  assert float(np.nanmax(err)) < 0.05
  assert float(np.isnan(out).mean()) < 1e-5

  # Planes already on the viewer grid pass through untouched (0..360 lons
  # are recognised after wrapping).
  native = np.random.default_rng(0).random((1, N_LAT, N_LON)).astype(np.float32)
  same = _resample_1d_rectilinear_to_global(native, _GRID_LATS, _GRID_LONS)
  assert same is native
  with pytest.raises(ValueError):
    _resample_1d_rectilinear_to_global(native, _GRID_LATS[:-1], _GRID_LONS)


@pytest.mark.unit
def test_resample_coarse_grid_picks_containing_cell_and_keeps_nan() -> None:
  """Native CPC 0.5 deg grid (0.25..359.75 lons, 89.75..-89.75 lats)."""
  lats = np.linspace(89.75, -89.75, 360)
  lons = np.linspace(0.25, 359.75, 720)
  src = np.full((1, 360, 720), np.nan, np.float32)
  src[:, :, :360] = 1.0  # eastern hemisphere "land"
  src[:, :180, :360] += 10.0  # northern hemisphere +10
  src[:, 100:110, :] = np.nan  # missing latitude band
  out = _resample_1d_rectilinear_to_global(src, lats, lons)

  # Target centres at x.25 / x.75 lie strictly inside one 0.5 deg source cell
  # (centres at x.00 / x.50 sit on source cell edges, where either neighbour
  # is an acceptable pick). Source row 179 (0.25N) is the last "+10" row.
  def _at(lat: float, lon: float) -> float:
    i, j = _grid_index(lat, lon)
    return float(out[0, i, j])

  assert _at(-0.25, 90.25) == 1.0
  assert _at(0.25, 90.25) == 11.0
  assert _at(40.25, 90.25) == 11.0
  assert np.isnan(_at(0.25, -90.25))
  # 100..110 source rows = 39.75N .. 35.25N -> NaN band
  assert np.isnan(_at(37.75, 90.25))
  assert np.isfinite(_at(42.25, 90.25))
  assert np.isfinite(_at(34.75, 90.25))
  # Columns at exactly -180.0 / 0.0 sit on source cell edges (tie between the
  # wrapped 179.75 cell and the 0.25 cell); skip them.
  east = out[0, :, N_LON // 2 + 1 :]
  west = out[0, :, 1 : N_LON // 2]
  assert np.isnan(west).all()
  # ~10 source rows (5 deg = 20 target rows, +-1 for edge ties) are NaN.
  assert 0.94 < float(np.isfinite(east[:400]).mean()) < 0.96
  assert np.isfinite(east[:100]).all()
  # Regional grids leave the rest of the globe NaN.
  regional = np.full((1, 100, 100), 7.0, np.float32)
  res = _resample_1d_rectilinear_to_global(
      regional, np.linspace(39.95, 30.05, 100), np.linspace(-99.95, -90.05, 100)
  )
  assert res[0, _grid_index(35.0, -95.0)[0], _grid_index(35.0, -95.0)[1]] == 7.0
  i_far, j_far = _grid_index(35.0, 10.0)
  assert np.isnan(res[0, i_far, j_far])
  assert float(np.isfinite(res).mean()) < 0.01


@pytest.mark.unit
def test_hrrr_reprojector_on_native_lambert_grid() -> None:
  """Native 1059 x 1799 HRRR grid -> 0.25 deg cell means, NaN outside CONUS."""
  lat2d, lon2d = _hrrr_native_latlon()
  assert lat2d.min() == pytest.approx(21.14, abs=0.05)
  assert lat2d.max() == pytest.approx(52.62, abs=0.05)
  assert lon2d.min() == pytest.approx(-134.10, abs=0.05)
  assert lon2d.max() == pytest.approx(-60.92, abs=0.05)

  reprojector = _build_2d_to_global_reprojector(lat2d, lon2d)
  plane = (2.0 * lat2d + lon2d).astype(np.float32)
  out = reprojector.apply(plane[None])
  assert out.shape == (1, N_LAT, N_LON)
  for lat, lon in [(38.0, -90.0), (45.0, -110.0), (30.0, -85.0)]:
    i, j = _grid_index(lat, lon)
    assert out[0, i, j] == pytest.approx(2.0 * lat + lon, abs=0.05)
  i_eu, j_eu = _grid_index(50.0, 10.0)
  assert np.isnan(out[0, i_eu, j_eu])
  assert not reprojector.valid_mask[i_eu, j_eu]
  covered = reprojector.valid_mask.sum()
  assert 20000 < covered < 40000  # ~3.9 Mkm2 CONUS domain / ~500 km2 cells

  # Every finite output cell lies within the source lat/lon envelope.
  rows, cols = np.nonzero(np.isfinite(out[0]))
  assert _GRID_LATS[rows].min() >= lat2d.min() - 0.25
  assert _GRID_LONS[cols].max() <= lon2d.max() + 0.25
  # NaN source cells propagate through the coverage rule.
  hole = plane.copy()
  hole[500:600, 900:1000] = np.nan
  out_hole = reprojector.apply(hole)
  assert np.isnan(out_hole).sum() > np.isnan(out[0]).sum()

  with pytest.raises(ValueError):
    _build_2d_to_global_reprojector(
        np.full((4, 4), np.nan), np.full((4, 4), np.nan)
    )
  with pytest.raises(ValueError):
    reprojector.apply(np.zeros((10, 10), np.float32))


@pytest.mark.unit
def test_is_plane_complete_checks_every_lead() -> None:
  """Interior all-NaN planes and under-populated last planes are detected."""
  planes = np.ones((5, 4, 4), np.float32)
  assert is_plane_complete(planes, "temp")
  planes[2] = np.nan
  assert not is_plane_complete(planes, "temp")
  precip = np.ones((5, 4, 4), np.float32)
  precip[0] = np.nan  # lead 0 of rate fields is NaN upstream
  assert is_plane_complete(precip, "precip")
  precip[-1, :2] = np.nan
  assert not is_plane_complete(precip, "precip")
  assert not is_plane_complete(np.full((3, 4, 4), np.nan, np.float32), "temp")


# ---------------------------------------------------------------------------
# dynamical.org forecast synchronisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sync_atomic_swap_incremental_update_and_hot_reload(
    tmp_path: Path,
) -> None:
  """Native-grid sync, hard-linked carry-over, atomic swap and engine reload."""
  t0 = np.datetime64("2026-09-29T00:00")
  t1 = np.datetime64("2026-09-29T06:00")
  catalogs = {
      "noaa-gfs-forecast": _build_native_forecast_dataset(
          [t0], [0, 1, 2, 3, 6]
      ),
      "ecmwf-aifs-single-forecast": _build_native_forecast_dataset(
          [t0], [0, 6]
      ),
  }
  open_ds = lambda _cat, dataset_id: catalogs[dataset_id]
  models = ["noaa_gfs", "ecmwf_aifs"]

  synchronizer = WeatherSynchronizer(data_dir=tmp_path, models=models)
  status = synchronizer.sync_all(open_dataset=open_ds, log=lambda _: None)
  assert status["last_result"] == "updated"
  assert status["errors"] == {}
  assert status["sources"] == {
      "noaa_gfs": "dynamical.org",
      "ecmwf_aifs": "dynamical.org",
  }

  run1 = current_run_dir(tmp_path)
  assert run1 is not None
  meta = json.loads((run1 / RUN_METADATA_FILE).read_text(encoding="utf-8"))
  gfs = meta["datasets"]["noaa_gfs_forecast"]
  assert gfs["model"] == "noaa_gfs"
  assert gfs["lead_hours"] == [0, 3, 6]
  assert gfs["lead_steps"] == 3
  assert gfs["source"] == "dynamical.org"
  assert meta["datasets"]["ecmwf_aifs_single_forecast"]["lead_hours"] == [0, 6]

  precip = np.fromfile(run1 / "noaa_gfs_precip.bin", np.float16).reshape(
      3, N_LAT, N_LON
  )
  assert pytest.approx(float(precip[1, 0, 0]), abs=0.02) == 1.0
  assert float(precip[0, 0, 0]) == 0.0
  temp = np.fromfile(run1 / "noaa_gfs_temp.bin", np.float16).reshape(
      3, N_LAT, N_LON
  )
  assert np.allclose(temp.astype(float), 12.5)
  mslp = np.fromfile(run1 / "noaa_gfs_mslp.bin", np.float16)
  assert pytest.approx(float(mslp[0]) + MSLP_OFFSET_HPA, abs=0.1) == 1013.25

  fetcher = WeatherDataFetcher(tmp_path)
  assert fetcher.get_model_info("ecmwf_aifs")["init_time"] == (
      "2026-09-29T00:00:00Z"
  )
  assert not fetcher.reload_if_changed()

  status2 = synchronizer.sync_all(open_dataset=open_ds, log=lambda _: None)
  assert status2["last_result"] == "up_to_date"
  assert current_run_dir(tmp_path) == run1

  # New GFS run -> GFS refreshed, AIFS hard-linked over, symlink swapped.
  catalogs["noaa-gfs-forecast"] = _build_native_forecast_dataset(
      [t0, t1], [0, 1, 2, 3, 6]
  )
  status3 = synchronizer.sync_all(open_dataset=open_ds, log=lambda _: None)
  assert status3["last_result"] == "updated"
  assert status3["updated_models"] == ["noaa_gfs"]
  run2 = current_run_dir(tmp_path)
  assert run2 is not None and run2 != run1
  assert os.path.islink(tmp_path / "current")
  carried = run2 / "ecmwf_aifs_precip.bin"
  assert carried.stat().st_ino == (run1 / "ecmwf_aifs_precip.bin").stat().st_ino
  temp2 = np.fromfile(run2 / "noaa_gfs_temp.bin", np.float16)
  assert np.allclose(temp2.astype(float), 13.5)
  assert not any(
      p.name.endswith(".partial") for p in (tmp_path / "runs").iterdir()
  )
  runs = list_available_runs(tmp_path)
  assert [r["is_current"] for r in runs] == [False, True]
  assert runs[1]["models"]["noaa_gfs"]["init_time"] == "2026-09-29T06:00:00Z"

  assert fetcher.reload_if_changed()
  assert fetcher.get_model_info("noaa_gfs")["init_time"] == (
      "2026-09-29T06:00:00Z"
  )


@pytest.mark.unit
def test_incomplete_run_rejected_and_previous_preserved(tmp_path: Path) -> None:
  """Unpublished last leads or interior NaN planes never replace a good run."""
  t0 = np.datetime64("2026-09-29T00:00")
  t1 = np.datetime64("2026-09-29T12:00")
  good = _build_native_forecast_dataset([t0], [0, 6])
  sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_aifs"],
      open_dataset=lambda _c, _i: good,
      log=lambda _: None,
  )
  run1 = current_run_dir(tmp_path)
  assert run1 is not None

  partial = _build_native_forecast_dataset([t0, t1], [0, 6])
  for var in partial.data_vars:
    partial[var].values[1, -1] = np.nan
  with pytest.raises(IncompleteRunError):
    download_model_run(
        partial,
        "ecmwf_aifs",
        DYNAMICAL_MODELS["ecmwf_aifs"],
        tmp_path / "direct_out",
        log=lambda _: None,
    )
  status = sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_aifs"],
      open_dataset=lambda _c, _i: partial,
      log=lambda _: None,
  )
  assert status["last_result"] == "error"
  assert "ecmwf_aifs" in status["errors"]
  assert current_run_dir(tmp_path) == run1
  assert not any(
      p.name.endswith(".partial") for p in (tmp_path / "runs").iterdir()
  )

  interior = _build_native_forecast_dataset([t1], [0, 6, 12, 18])
  interior["temperature_2m"].values[0, 2] = np.nan
  with pytest.raises(IncompleteRunError, match="lead 12 h"):
    download_model_run(
        interior,
        "ecmwf_aifs",
        DYNAMICAL_MODELS["ecmwf_aifs"],
        tmp_path / "direct_out2",
        log=lambda _: None,
    )

  # Partial failure of one model still publishes the other ("partial").
  gfs = _build_native_forecast_dataset([t1], [0, 1, 2, 3])
  status_partial = sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_aifs", "noaa_gfs"],
      open_dataset=lambda _c, ds_id: (
          partial if ds_id == "ecmwf-aifs-single-forecast" else gfs
      ),
      log=lambda _: None,
  )
  assert status_partial["last_result"] == "partial"
  assert status_partial["updated_models"] == ["noaa_gfs"]
  run2 = current_run_dir(tmp_path)
  assert run2 is not None and run2 != run1
  assert (run2 / "ecmwf_aifs_temp.bin").exists()  # carried from run1
  assert status_partial["models"]["ecmwf_aifs"]["init_time"] == (
      "2026-09-29T00:00:00"
  )


@pytest.mark.unit
def test_ensemble_selects_control_member_and_cli_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Ensemble control member (member 0) extraction and CLI --status."""
  t0 = np.datetime64("2026-09-29T00:00")
  ds = _build_native_forecast_dataset([t0], [0, 3, 6], members=True)
  ds["temperature_2m"].values[:, :, 1:] = 99.0

  sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_ifs"],
      open_dataset=lambda _c, _i: ds,
      log=lambda _: None,
  )
  run_dir = current_run_dir(tmp_path)
  assert run_dir is not None
  temp = np.fromfile(run_dir / "ecmwf_ifs_temp.bin", np.float16)
  assert np.allclose(temp.astype(float), 12.5)
  meta = json.loads((run_dir / RUN_METADATA_FILE).read_text(encoding="utf-8"))
  entry = meta["datasets"]["ecmwf_ifs_ens_forecast_15_day_0_25_degree"]
  assert entry["ensemble_member"] == 0

  rc = cli.main(["--data-dir", str(tmp_path), "--status"])
  assert rc == 0
  captured = json.loads(capsys.readouterr().out)
  assert captured["last_result"] == "updated"
  assert "ecmwf_ifs" in captured["models"]

  missing_member = ds.copy()
  cfg = dict(DYNAMICAL_MODELS["ecmwf_ifs"], ensemble_member=7)
  with pytest.raises(ValueError, match="ensemble member 7"):
    download_model_run(
        missing_member, "ecmwf_ifs", cfg, tmp_path / "bad", log=lambda _: None
    )


@pytest.mark.unit
def test_hrrr_sync_on_native_grid(tmp_path: Path) -> None:
  """HRRR 3 km CONUS planes are binned onto the viewer grid; outside is NaN."""
  hrrr = _build_hrrr_dataset(np.datetime64("2026-09-29T00:00"), [0, 1, 2, 3])
  status = sync_all_models(
      data_dir=tmp_path,
      models=["noaa_hrrr"],
      open_dataset=lambda _c, _i: hrrr,
      log=lambda _: None,
  )
  assert status["last_result"] == "updated"
  run_dir = current_run_dir(tmp_path)
  assert run_dir is not None
  meta = json.loads((run_dir / RUN_METADATA_FILE).read_text(encoding="utf-8"))
  assert meta["datasets"]["noaa_hrrr_forecast_48_hour"]["lead_hours"] == [0, 3]

  temp = np.fromfile(run_dir / "noaa_hrrr_temp.bin", np.float16).reshape(
      2, N_LAT, N_LON
  )
  precip = np.fromfile(run_dir / "noaa_hrrr_precip.bin", np.float16).reshape(
      2, N_LAT, N_LON
  )
  i_us, j_us = _grid_index(38.0, -90.0)
  i_eu, j_eu = _grid_index(50.0, 10.0)
  assert float(temp[1, i_us, j_us]) == pytest.approx(20.0 - 9.0, abs=0.1)
  assert np.isnan(temp[1, i_eu, j_eu])
  assert float(precip[1, i_us, j_us]) == pytest.approx(2.0, abs=0.02)
  assert float(precip[0, i_us, j_us]) == 0.0
  assert np.isnan(precip[0, i_eu, j_eu])
  assert np.isnan(precip[1, i_eu, j_eu])


# ---------------------------------------------------------------------------
# NASA IMERG half-hourly analysis
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_latest_init_time_str_uses_utc_aligned_three_hour_bins() -> None:
  """The window ends at the last 3 h boundary fully covered by frames."""
  start = np.datetime64("2026-09-29T00:00")
  times = start + np.arange(42) * np.timedelta64(30, "m")  # last 20:30
  ds = xr.Dataset(coords={"time": times.astype("datetime64[ns]")})
  assert latest_init_time_str(ds) == "2026-09-19T21:00:00"
  ds_short = xr.Dataset(coords={"time": times[:-1].astype("datetime64[ns]")})
  assert latest_init_time_str(ds_short) == "2026-09-19T18:00:00"


@pytest.mark.unit
def test_imerg_bins_average_all_half_hours_and_require_six_frames(
    tmp_path: Path,
) -> None:
  """Rain only in :30 frames -> 5 mm/h; incomplete bins and NaN cells -> NaN."""
  start = np.datetime64("2026-09-29T00:00")
  # 13 frames: 00:00 .. 06:00 -> complete bins ending 03:00 and 06:00.
  imerg = _build_imerg_dataset(start, 13, lambda k: 10.0 if k % 2 else 0.0)
  imerg["precipitation_surface"].values[2, 499, 1999] = np.nan  # 01:00 frame
  status = sync_all_models(
      data_dir=tmp_path,
      models=["nasa_imerg"],
      open_dataset=lambda _c, _i: imerg,
      log=lambda _: None,
  )
  assert status["last_result"] == "updated"
  run_dir = current_run_dir(tmp_path)
  assert run_dir is not None
  meta = json.loads((run_dir / RUN_METADATA_FILE).read_text(encoding="utf-8"))
  entry = meta["datasets"]["nasa_imerg_analysis_early"]
  assert entry["init_time"] == "2026-09-19T06:00:00"
  assert entry["lead_hours"] == list(range(0, MAX_LEAD_HOURS + 1, 3))
  planes = np.fromfile(run_dir / "nasa_imerg_precip.bin", np.float16).reshape(
      81, N_LAT, N_LON
  )
  i, j = _grid_index(40.0, 20.0)
  assert float(planes[80, 100, 100]) == pytest.approx(5.0, abs=0.01)
  assert float(planes[79, 100, 100]) == pytest.approx(5.0, abs=0.01)
  assert np.isnan(planes[79, i, j])  # NaN half-hour cell -> NaN 3 h mean
  assert float(planes[80, i, j]) == pytest.approx(5.0, abs=0.01)
  assert np.isnan(planes[1:79]).all()  # bins before the archive start
  assert float(planes[0, 100, 100]) == 0.0
  assert float(np.isnan(planes[80]).mean()) < 1e-5

  # Dropping one half-hour frame leaves that bin NaN instead of a 5/6 mean.
  short = imerg.isel(time=[k for k in range(13) if k != 8])
  status2 = sync_all_models(
      data_dir=tmp_path / "second",
      models=["nasa_imerg"],
      open_dataset=lambda _c, _i: short,
      log=lambda _: None,
  )
  assert status2["last_result"] == "updated"
  run2 = current_run_dir(tmp_path / "second")
  assert run2 is not None
  planes2 = np.fromfile(run2 / "nasa_imerg_precip.bin", np.float16).reshape(
      81, N_LAT, N_LON
  )
  assert np.isnan(planes2[80]).all()
  assert float(planes2[79, 100, 100]) == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# NOAA CPC daily gauge analysis
# ---------------------------------------------------------------------------


def _install_cpc_upstream(
    monkeypatch: pytest.MonkeyPatch, staging: Path, years: Sequence[int]
) -> _CpcUpstream:
  upstream = _CpcUpstream(staging, years)
  monkeypatch.setattr(
      "multimet.utils.cpc.download_http_file", upstream.download
  )
  monkeypatch.setattr(
      "multimet.weather_fetcher.sync.check_http_url_exists", upstream.exists
  )
  return upstream


@pytest.mark.unit
def test_cpc_window_stitches_years_refreshes_cache_and_keeps_oceans_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Year-boundary stitching, 12Z day ends, mm/day -> mm/h, cache refresh."""
  staging = tmp_path / "psl"
  staging.mkdir()
  dec = [
      datetime.date(2025, 12, 20) + datetime.timedelta(days=k)
      for k in range(12)
  ]
  jan = [
      datetime.date(2026, 1, 1) + datetime.timedelta(days=k)
      for k in range(10)
  ]
  _write_psl_cpc_file(staging / "precip.2025.nc", dec, dec)
  _write_psl_cpc_file(staging / "precip.2026.nc", jan, jan[:5])
  upstream = _install_cpc_upstream(monkeypatch, staging, [2025, 2026])

  data_dir = tmp_path / "data"
  cache_dir = tmp_path / "cpc_cache"
  now = datetime.datetime(2026, 1, 8, 10, tzinfo=_UTC)
  status = sync_all_models(
      data_dir=data_dir,
      models=["noaa_cpc"],
      cpc_cache_dir=cache_dir,
      now=now,
      log=lambda _: None,
  )
  assert status["last_result"] == "updated"
  assert status["sources"] == {"noaa_cpc": "NOAA PSL (downloads.psl.noaa.gov)"}
  assert len(upstream.downloads) == 2
  assert (cache_dir / "precip.2025.nc").exists()

  run_dir = current_run_dir(data_dir)
  assert run_dir is not None
  meta = json.loads((run_dir / RUN_METADATA_FILE).read_text(encoding="utf-8"))
  entry = meta["datasets"]["noaa_cpc_unified_gauge_precip"]
  assert entry["init_time"] == "2025-12-26T12:00:00"
  assert entry["lead_hours"] == list(range(0, 241, 24))
  assert entry["day_dates"][0] == "2025-12-27"
  assert entry["day_dates"][-1] == "2026-01-05"

  planes = np.fromfile(run_dir / "noaa_cpc_precip.bin", np.float16).reshape(
      11, N_LAT, N_LON
  )
  i_land, j_land = _grid_index(40.0, 90.0)
  i_sea, j_sea = _grid_index(40.0, -90.0)
  assert float(planes[1, i_land, j_land]) == pytest.approx(
      27.0 / 24.0, abs=0.01
  )
  assert float(planes[5, i_land, j_land]) == pytest.approx(
      31.0 / 24.0, abs=0.01
  )
  assert float(planes[6, i_land, j_land]) == pytest.approx(1.0 / 24.0, abs=0.01)
  assert float(planes[10, i_land, j_land]) == pytest.approx(
      5.0 / 24.0, abs=0.01
  )
  assert float(planes[0, i_land, j_land]) == 0.0
  assert np.isnan(planes[:, i_sea, j_sea]).all()
  assert float(np.isnan(planes[1]).mean()) == pytest.approx(0.5, abs=0.01)

  # Fresh cache: nothing new upstream -> up to date, no download.
  status2 = sync_all_models(
      data_dir=data_dir,
      models=["noaa_cpc"],
      cpc_cache_dir=cache_dir,
      now=now,
      log=lambda _: None,
  )
  assert status2["last_result"] == "up_to_date"
  assert len(upstream.downloads) == 2

  # Stale cache (older than CPC_CACHE_REFRESH_HOURS) is downloaded again and
  # newly published days (Jan 6) move the window forward.
  _write_psl_cpc_file(staging / "precip.2026.nc", jan, jan[:6])
  old = (now - datetime.timedelta(hours=10)).timestamp()
  for name in ("precip.2025.nc", "precip.2026.nc"):
    os.utime(cache_dir / name, (old, old))
  status3 = sync_all_models(
      data_dir=data_dir,
      models=["noaa_cpc"],
      cpc_cache_dir=cache_dir,
      now=now,
      log=lambda _: None,
  )
  assert status3["last_result"] == "updated"
  assert len(upstream.downloads) == 4
  assert status3["models"]["noaa_cpc"]["init_time"] == "2025-12-27T12:00:00"


@pytest.mark.unit
def test_cpc_falls_back_to_previous_year_until_new_file_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Early January: the new year's NetCDF does not exist upstream yet."""
  staging = tmp_path / "psl"
  staging.mkdir()
  dec = [
      datetime.date(2025, 12, 20) + datetime.timedelta(days=k)
      for k in range(12)
  ]
  _write_psl_cpc_file(staging / "precip.2025.nc", dec, dec)
  upstream = _install_cpc_upstream(monkeypatch, staging, [2025])

  logs: List[str] = []
  status = sync_all_models(
      data_dir=tmp_path / "data",
      models=["noaa_cpc"],
      now=datetime.datetime(2026, 1, 2, 6, tzinfo=_UTC),
      log=logs.append,
  )
  assert status["last_result"] == "updated"
  assert upstream.downloads == [
      "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.2025.nc"
  ]
  assert any("not published yet" in line for line in logs)
  assert status["models"]["noaa_cpc"]["init_time"] == "2025-12-21T12:00:00"
  assert (tmp_path / "data" / "cpc_cache" / "precip.2025.nc").exists()


# ---------------------------------------------------------------------------
# ECMWF Open Data HRES
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(
    importlib.util.find_spec("eccodes") is not None,
    reason="synthetic GRIB2 messages only decode through the raw parser",
)
def test_hres_open_data_extraction_with_synthetic_grib2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """tp de-accumulation (m -> mm/h), K -> degC, Pa -> hPa on 721 x 1440."""
  for mod_name, mod in list(sys.modules.items()):
    if mod_name == "rasterio" or mod_name.startswith("rasterio."):
      if getattr(mod, "__spec__", None) is None:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
  now = datetime.datetime(2026, 9, 29, 15, tzinfo=_UTC)
  gcs = _FakeOpenDataGcs("20260929", "00z", MAX_LEAD_HOURS)
  assert _latest_hres_run_info(gcs, now=now) == (
      "20260929",
      "00z",
      "2026-09-29T00:00:00",
  )
  assert _latest_hres_run_info(
      _FakeOpenDataGcs("20260920", "12z", MAX_LEAD_HOURS), now=now
  ) is None

  out_dir = tmp_path / "stage"
  out_dir.mkdir()
  entry, err = _extract_hres_streams(
      "ecmwf_hres",
      DYNAMICAL_MODELS["ecmwf_hres"],
      out_dir,
      log=lambda _: None,
      fs=gcs,
      leads=[0, 3, 6, 144, 150],
      max_workers=2,
      now=now,
  )
  assert err is None and entry is not None
  assert entry["init_time"] == "2026-09-29T00:00:00"
  assert entry["lead_hours"] == [0, 3, 6, 144, 150]
  assert entry["source"] == "gs://ecmwf-open-data"
  precip = np.fromfile(out_dir / "ecmwf_hres_precip.bin", np.float16).reshape(
      5, N_LAT, N_LON
  )
  np.testing.assert_allclose(
      precip[:, 100, 200].astype(float), [0.0, 1.0, 1.0, 1.0, 1.0], atol=0.01
  )
  temp = np.fromfile(out_dir / "ecmwf_hres_temp.bin", np.float16)
  assert np.allclose(temp.astype(float), 20.0, atol=0.02)
  mslp = np.fromfile(out_dir / "ecmwf_hres_mslp.bin", np.float16)
  assert np.allclose(mslp.astype(float) + MSLP_OFFSET_HPA, 1013.25, atol=0.1)
  v10 = np.fromfile(out_dir / "ecmwf_hres_v10.bin", np.float16)
  assert np.allclose(v10.astype(float), -4.0)

  no_run = _FakeOpenDataGcs("20260920", "12z", MAX_LEAD_HOURS)
  entry_none, err_none = _extract_hres_streams(
      "ecmwf_hres",
      DYNAMICAL_MODELS["ecmwf_hres"],
      out_dir,
      log=lambda _: None,
      fs=no_run,
      leads=[0, 3],
      now=now,
  )
  assert entry_none is None and isinstance(err_none, IncompleteRunError)


# ---------------------------------------------------------------------------
# Run directory housekeeping and concurrency
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prune_keeps_partial_dirs_and_swap_requires_existing_run(
    tmp_path: Path,
) -> None:
  """prune_old_runs never touches staging dirs; swap refuses missing runs."""
  runs = tmp_path / "runs"
  for name in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
    (runs / name).mkdir(parents=True)
  foreign = runs / f"20260104T000000Z.{os.getpid() + 1}.partial"
  foreign.mkdir()
  (foreign / "noaa_gfs_precip.bin").write_bytes(b"x")

  with pytest.raises(FileNotFoundError):
    swap_current_symlink(tmp_path, "does_not_exist")
  assert not os.path.lexists(tmp_path / "current")

  swap_current_symlink(tmp_path, "20260103T000000Z")
  prune_old_runs(tmp_path, keep_previous=1)
  remaining = sorted(p.name for p in runs.iterdir())
  assert remaining == [
      "20260102T000000Z",
      "20260103T000000Z",
      foreign.name,
  ]
  assert current_run_dir(tmp_path) == (runs / "20260103T000000Z").resolve()


def _dead_pid() -> int:
  proc = Path("/proc")
  pid = 4194303
  while proc.is_dir() and (proc / str(pid)).exists():
    pid -= 1
  return pid


@pytest.mark.unit
def test_busy_detection_and_stale_staging_cleanup(tmp_path: Path) -> None:
  """A live staging dir makes sync report busy; dead ones are removed."""
  runs = tmp_path / "runs"
  live = runs / f"20260101T000000Z.{os.getpid()}.partial"
  live.mkdir(parents=True)
  ds = _build_native_forecast_dataset(
      [np.datetime64("2026-09-29T00:00")], [0, 6]
  )
  status = sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_aifs"],
      open_dataset=lambda _c, _i: ds,
      log=lambda _: None,
  )
  assert status["last_result"] == "busy"
  assert live.is_dir()
  assert current_run_dir(tmp_path) is None
  assert cli.main(["--data-dir", str(tmp_path), "--status"]) == 0
  live.rmdir()

  stale = runs / f"20260101T000000Z.{_dead_pid()}.partial"
  stale.mkdir()
  old = (datetime.datetime.now(_UTC) - datetime.timedelta(hours=30)).timestamp()
  os.utime(stale, (old, old))
  status2 = sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_aifs"],
      open_dataset=lambda _c, _i: ds,
      log=lambda _: None,
  )
  assert status2["last_result"] == "updated"
  assert not stale.exists()
  assert current_run_dir(tmp_path) is not None


@pytest.mark.unit
def test_swap_current_symlink_windows_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """swap uses target_is_directory=True and unlinks the old link on nt."""
  (tmp_path / "runs" / "run1").mkdir(parents=True)
  (tmp_path / "runs" / "run2").mkdir(parents=True)

  calls: List[bool] = []
  real_symlink = os.symlink
  real_replace = os.replace

  def tracked_symlink(
      src: os.PathLike[str] | str,
      dst: os.PathLike[str] | str,
      target_is_directory: bool = False,
      *,
      dir_fd: Optional[int] = None,
  ) -> None:
    calls.append(target_is_directory)
    real_symlink(
        src, dst, target_is_directory=target_is_directory, dir_fd=dir_fd
    )

  def strict_windows_replace(
      src: os.PathLike[str] | str, dst: os.PathLike[str] | str
  ) -> None:
    if os.path.lexists(dst):
      raise PermissionError(
          "[WinError 5] Access is denied when replacing existing directory"
          " symlink"
      )
    real_replace(src, dst)

  monkeypatch.setattr(os, "symlink", tracked_symlink)
  monkeypatch.setattr(os, "replace", strict_windows_replace)
  # ``pathlib.Path()`` picks WindowsPath when os.name == "nt", which cannot be
  # instantiated on POSIX; pin the module-level name to the platform class so
  # only the ``os.name`` branch of swap_current_symlink is emulated.
  monkeypatch.setattr("multimet.weather_fetcher.sync.Path", type(tmp_path))
  monkeypatch.setattr(os, "name", "nt")

  link1 = swap_current_symlink(tmp_path, "run1")
  assert link1.resolve() == (tmp_path / "runs" / "run1").resolve()
  assert calls == [True]
  link2 = swap_current_symlink(tmp_path, "run2")
  assert link2.resolve() == (tmp_path / "runs" / "run2").resolve()
  assert calls == [True, True]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_requires_data_dir_and_maps_results_to_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  """No hidden default directory; partial/error results exit non-zero."""
  with pytest.raises(SystemExit) as exc:
    cli.main([])
  assert exc.value.code == 2

  assert cli.main(["--data-dir", str(tmp_path), "--status"]) == 0
  assert json.loads(capsys.readouterr().out) == {}

  with pytest.raises(ValueError, match="Unsupported weather model"):
    cli.main(["--data-dir", str(tmp_path), "--models", "graphcast"])

  seen: Dict[str, Any] = {}

  def fake_sync(**kwargs: Any) -> Dict[str, Any]:
    seen.update(kwargs)
    return {"last_result": seen.pop("expected")}

  monkeypatch.setattr(cli, "sync_all_models", fake_sync)
  for result, code in (
      ("updated", 0),
      ("up_to_date", 0),
      ("busy", 0),
      ("partial", 1),
      ("error", 1),
  ):
    seen["expected"] = result
    rc = cli.main([
        "--data-dir",
        str(tmp_path),
        "--cpc-cache-dir",
        str(tmp_path / "cpc"),
        "--models",
        "noaa_gfs,noaa_cpc",
        "--force",
    ])
    assert rc == code, result
    assert seen["models"] == ["noaa_gfs", "noaa_cpc"]
    assert seen["force"] is True
    assert seen["cpc_cache_dir"] == str(tmp_path / "cpc")
    assert seen["data_dir"] == tmp_path.resolve()
