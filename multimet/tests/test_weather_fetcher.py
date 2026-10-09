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

"""Unit, regression, and canary tests for multimet.weather_fetcher.

Fixtures write native `721 x 1440` float16 planes whose values are exact in
float16 (multiples of 0.125 below 256) and vary by row, column and plane, so
every assertion below compares against an independently computed value.
"""

from __future__ import annotations

import ast
import gc
import json
import math
import os
import threading
import time
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import multimet.weather_fetcher as wf
from multimet.weather_fetcher.config import (
  DYNAMICAL_MODELS,
  MSLP_OFFSET_HPA,
  N_LAT,
  N_LON,
  RUN_METADATA_FILE,
  SUPPORTED_MODELS,
  SUPPORTED_VARIABLES,
  from_stored_units,
  to_stored_units,
)
from multimet.weather_fetcher.fetcher import (
  WeatherDataFetcher,
  bilinear_sample_grid,
  clear_accum_grid_cache,
  close_unreferenced_mmaps,
  compute_accumulated_precip_grid,
  compute_wind_speed_and_direction,
  extract_accumulation_series,
  extract_point_value,
  fetch_wind_grid,
  file_step_for_lead,
  geojson_polygon_to_shapely,
  grid_indices,
  rate_file_steps,
  scan_streams,
)

ROWS = np.arange(N_LAT, dtype=np.int64)[:, None]
COLS = np.arange(N_LON, dtype=np.int64)[None, :]

# Probe location used throughout: lat 40.42 -> row 198 (40.5 N), lon -86.92 ->
# col 372 (87.0 W).
PROBE_LAT, PROBE_LON = 40.42, -86.92
PROBE_ROW, PROBE_COL = 198, 372

# noaa_gefs catchment fixture: 10 mm/h in a 2 x 2 block of cells, dry elsewhere.
RAIN_ROWS = (197, 198)  # 40.75 N, 40.5 N
RAIN_COLS = (373, 374)  # 86.75 W, 86.5 W
RAIN_MMH = 10.0

# noaa_hrrr fixture: finite only inside a CONUS-like box, NaN elsewhere.
HRRR_ROWS = (160, 260)  # 50 N .. 25 N
HRRR_COLS = (220, 460)  # 125 W .. 65 W
HRRR_RATE, HRRR_TEMP, HRRR_U, HRRR_V, HRRR_MSLP = 2.0, 15.0, 3.0, 4.0, 5.0

INIT_TIME = "2026-09-29T00:00:00"


def _lat_of_row(row: int) -> float:
  return 90.0 - 0.25 * row


def _lon_of_col(col: int) -> float:
  return -180.0 + 0.25 * col


def _precip_plane(i: int) -> np.ndarray:
  """Interval-mean rain rate of plane `i` (mm/h); plane 0 is zero."""
  if i == 0:
    return np.zeros((N_LAT, N_LON), dtype=np.float16)
  return (0.125 * ((ROWS + COLS + i) % 8)).astype(np.float16)


def _temp_plane(i: int) -> np.ndarray:
  return (10.0 + 0.25 * (ROWS % 16) + 0.125 * (COLS % 8) + 0.5 * i).astype(
      np.float16
  )


def _mslp_plane(i: int) -> np.ndarray:
  """Stored MSLP (hPa minus the 1000 hPa offset)."""
  return (0.25 * ((ROWS + 2 * COLS) % 40) - 5.0 + 0.125 * i).astype(np.float16)


def _u10_plane() -> np.ndarray:
  return (0.5 * ((ROWS + COLS) % 16) - 4.0).astype(np.float16)


def _v10_plane() -> np.ndarray:
  return (0.25 * ((ROWS - COLS) % 16) - 2.0).astype(np.float16)


def _expected_precip(i: int, row: int, col: int) -> float:
  return 0.0 if i == 0 else 0.125 * ((row + col + i) % 8)


def _expected_temp(i: int, row: int, col: int) -> float:
  return 10.0 + 0.25 * (row % 16) + 0.125 * (col % 8) + 0.5 * i


def _expected_pressure(i: int, row: int, col: int) -> float:
  return 0.25 * ((row + 2 * col) % 40) - 5.0 + 0.125 * i + MSLP_OFFSET_HPA


def _expected_u10(row: int, col: int) -> float:
  return 0.5 * ((row + col) % 16) - 4.0


def _expected_v10(row: int, col: int) -> float:
  return 0.25 * ((row - col) % 16) - 2.0


def _gefs_precip_plane(i: int) -> np.ndarray:
  plane = np.zeros((N_LAT, N_LON), dtype=np.float16)
  if i > 0:
    r0, r1 = RAIN_ROWS
    c0, c1 = RAIN_COLS
    plane[r0 : r1 + 1, c0 : c1 + 1] = RAIN_MMH
  return plane


def _gefs_temp_plane(i: int) -> np.ndarray:
  del i
  return np.broadcast_to(20.0 + 0.125 * (ROWS % 4), (N_LAT, N_LON)).astype(
      np.float16
  )


def _hrrr_plane(value: float, i: int, is_precip: bool) -> np.ndarray:
  plane = np.full((N_LAT, N_LON), np.nan, dtype=np.float16)
  fill = 0.0 if (is_precip and i == 0) else value
  plane[HRRR_ROWS[0] : HRRR_ROWS[1], HRRR_COLS[0] : HRRR_COLS[1]] = fill
  return plane


def _write_stream(run_dir: Path, stream_id: str, planes: Sequence[np.ndarray]):
  with open(run_dir / f"{stream_id}.bin", "wb") as f_out:
    for plane in planes:
      f_out.write(np.ascontiguousarray(plane, dtype=np.float16).tobytes())


def _write_meta(
    run_dir: Path, datasets: dict[str, dict[str, Any]], downloaded: str
) -> None:
  meta = {
      "status": "HEALTHY",
      "source": "test",
      "last_updated_utc": downloaded,
      "datasets": datasets,
  }
  (run_dir / RUN_METADATA_FILE).write_text(
      json.dumps(meta, indent=2), encoding="utf-8"
  )


def _dataset_entry(
    model: str,
    init_time: str,
    leads: Sequence[int],
    streams: Sequence[str],
    downloaded: str,
) -> dict[str, Any]:
  return {
      "model": model,
      "init_time": init_time,
      "lead_steps": len(leads),
      "lead_hours": list(leads),
      "streams": list(streams),
      "downloaded_utc": downloaded,
      "mslp_offset_hpa": MSLP_OFFSET_HPA,
  }


def _point_current(root: Path, run_name: str) -> None:
  tmp_link = root / "current.tmp"
  if os.path.lexists(tmp_link):
    tmp_link.unlink()
  os.symlink(Path("runs") / run_name, tmp_link, target_is_directory=True)
  os.replace(tmp_link, root / "current")


def _write_model(
    run_dir: Path,
    model: str,
    leads: Sequence[int],
    stream_planes: dict[str, list[np.ndarray]],
) -> None:
  for suffix, planes in stream_planes.items():
    assert len(planes) == len(leads)
    _write_stream(run_dir, f"{model}_{suffix}", planes)


def _write_main_run(root: Path, run_name: str = "20260929T000000Z") -> Path:
  """Writes a run with a 3-hourly, a 6-hourly, and a precip/temp-only model."""
  run_dir = root / "runs" / run_name
  run_dir.mkdir(parents=True, exist_ok=True)
  downloaded = "2026-09-29T03:00:00Z"
  ifs_leads = list(range(0, 25, 3))
  aifs_leads = list(range(0, 25, 6))
  gefs_leads = list(range(0, 25, 3))
  _write_model(
      run_dir,
      "ecmwf_ifs",
      ifs_leads,
      {
          "precip": [_precip_plane(i) for i in range(len(ifs_leads))],
          "temp": [_temp_plane(i) for i in range(len(ifs_leads))],
          "mslp": [_mslp_plane(i) for i in range(len(ifs_leads))],
          "u10": [_u10_plane() for _ in ifs_leads],
          "v10": [_v10_plane() for _ in ifs_leads],
      },
  )
  _write_model(
      run_dir,
      "ecmwf_aifs",
      aifs_leads,
      {
          "precip": [_precip_plane(i) for i in range(len(aifs_leads))],
          "temp": [_temp_plane(i) for i in range(len(aifs_leads))],
          "mslp": [_mslp_plane(i) for i in range(len(aifs_leads))],
          "u10": [_u10_plane() for _ in aifs_leads],
          "v10": [_v10_plane() for _ in aifs_leads],
      },
  )
  _write_model(
      run_dir,
      "noaa_gefs",
      gefs_leads,
      {
          "precip": [_gefs_precip_plane(i) for i in range(len(gefs_leads))],
          "temp": [_gefs_temp_plane(i) for i in range(len(gefs_leads))],
      },
  )
  all_streams = ("precip", "temp", "mslp", "u10", "v10")
  _write_meta(
      run_dir,
      {
          "ecmwf_ifs_ens_forecast_15_day_0_25_degree": _dataset_entry(
              "ecmwf_ifs", INIT_TIME, ifs_leads, all_streams, downloaded
          ),
          "ecmwf_aifs_single_forecast": _dataset_entry(
              "ecmwf_aifs", INIT_TIME, aifs_leads, all_streams, downloaded
          ),
          "noaa_gefs_forecast_35_day": _dataset_entry(
              "noaa_gefs", INIT_TIME, gefs_leads, ("precip", "temp"), downloaded
          ),
      },
      downloaded,
  )
  _point_current(root, run_name)
  return run_dir


def _write_hrrr_run(root: Path) -> Path:
  run_dir = root / "runs" / "20260929T000000Z"
  run_dir.mkdir(parents=True, exist_ok=True)
  leads = [0, 3, 6]
  _write_model(
      run_dir,
      "noaa_hrrr",
      leads,
      {
          "precip": [_hrrr_plane(HRRR_RATE, i, True) for i in range(3)],
          "temp": [_hrrr_plane(HRRR_TEMP, i, False) for i in range(3)],
          "mslp": [_hrrr_plane(HRRR_MSLP, i, False) for i in range(3)],
          "u10": [_hrrr_plane(HRRR_U, i, False) for i in range(3)],
          "v10": [_hrrr_plane(HRRR_V, i, False) for i in range(3)],
      },
  )
  downloaded = "2026-09-29T02:00:00Z"
  _write_meta(
      run_dir,
      {
          "noaa_hrrr_forecast_48_hour": _dataset_entry(
              "noaa_hrrr",
              INIT_TIME,
              leads,
              ("precip", "temp", "mslp", "u10", "v10"),
              downloaded,
          )
      },
      downloaded,
  )
  _point_current(root, "20260929T000000Z")
  return run_dir


def _write_small_run(
    root: Path,
    run_name: str,
    init_time: str,
    rain_mmh: float,
    temp_c: float = 18.5,
) -> Path:
  """Writes a two-plane (0 h, 3 h) ecmwf_aifs run used by reload tests."""
  run_dir = root / "runs" / run_name
  run_dir.mkdir(parents=True, exist_ok=True)
  leads = [0, 3]
  zero = np.zeros((N_LAT, N_LON), dtype=np.float16)
  _write_model(
      run_dir,
      "ecmwf_aifs",
      leads,
      {
          "precip": [zero, np.full((N_LAT, N_LON), rain_mmh, np.float16)],
          "temp": [np.full((N_LAT, N_LON), temp_c, np.float16)] * 2,
          "u10": [np.full((N_LAT, N_LON), 6.0, np.float16)] * 2,
          "v10": [np.full((N_LAT, N_LON), 8.0, np.float16)] * 2,
      },
  )
  downloaded = f"{init_time}Z"
  _write_meta(
      run_dir,
      {
          "ecmwf_aifs_single_forecast": _dataset_entry(
              "ecmwf_aifs",
              init_time,
              leads,
              ("precip", "temp", "u10", "v10"),
              downloaded,
          )
      },
      downloaded,
  )
  _point_current(root, run_name)
  return run_dir


def _box(min_lon: float, min_lat: float, max_lon: float, max_lat: float):
  return [[
      [min_lon, min_lat],
      [max_lon, min_lat],
      [max_lon, max_lat],
      [min_lon, max_lat],
      [min_lon, min_lat],
  ]]


def _feature(
    geometry: dict[str, Any] | None,
    feature_id: str | None = "basin",
    area_km2: float | None = None,
) -> dict[str, Any]:
  props: dict[str, Any] = {}
  if area_km2 is not None:
    props["area_km2"] = area_km2
  feature: dict[str, Any] = {"properties": props, "geometry": geometry}
  if feature_id is not None:
    feature["id"] = feature_id
  return feature


def _cos(row: int) -> float:
  return math.cos(math.radians(_lat_of_row(row)))


@pytest.fixture(scope="module")
def main_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
  root = tmp_path_factory.mktemp("weather_main")
  _write_main_run(root)
  return root


@pytest.fixture(scope="module")
def main_fetcher(main_root: Path):
  with WeatherDataFetcher(main_root) as fetcher:
    yield fetcher


@pytest.fixture
def hrrr_fetcher(tmp_path: Path):
  _write_hrrr_run(tmp_path)
  with WeatherDataFetcher(tmp_path) as fetcher:
    yield fetcher


@pytest.mark.unit
def test_zero_try_except_and_zero_ui_rendering_in_weather_fetcher_package() -> (
    None
):
  """Verifies zero try/except blocks and zero rendering code in the package."""
  pkg_dir = Path(wf.__file__).resolve().parent
  py_files = sorted(pkg_dir.glob("*.py"))
  assert {p.name for p in py_files} >= {
      "__init__.py",
      "cli.py",
      "config.py",
      "fetcher.py",
      "sync.py",
  }
  forbidden_ui_tokens = (
      "make_png_bytes",
      "encode_rgba_png",
      "encode_indexed_png",
      "colorize_rgba",
      "colorize_indexed",
      "render_colorbar_lut_png",
      "tile_coordinates",
      "frame_coordinates",
  )
  for py_file in py_files:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert not try_nodes, (
        f"Found forbidden try/except node(s) in {py_file.name}: "
        f"lines {[n.lineno for n in try_nodes]}"
    )
    for token in forbidden_ui_tokens:
      assert (
          token not in source
      ), f"Found forbidden UI rendering symbol {token!r} in {py_file.name}"


@pytest.mark.unit
def test_package_exports_resolve() -> None:
  """Every name in `__all__` is importable and the removed API is gone."""
  for name in wf.__all__:
    assert hasattr(wf, name), name
  for removed in (
      "WeatherSource",
      "WEATHER_SOURCES",
      "get_weather_source",
      "list_weather_sources",
      "parse_zarr_metadata_time_extent",
      "extract_point_probe",
      "extract_catchment_weather_summary",
  ):
    assert not hasattr(wf, removed), removed


@pytest.mark.unit
def test_config_catalog_and_unit_conversions() -> None:
  """Tests model catalog, variable catalog, and physical unit conversions."""
  expected_models = (
      "ecmwf_hres",
      "ecmwf_ifs",
      "ecmwf_aifs",
      "noaa_gfs",
      "noaa_gefs",
      "noaa_hrrr",
      "nasa_imerg",
      "noaa_cpc",
  )
  assert tuple(SUPPORTED_MODELS) == expected_models
  for model_key in expected_models:
    assert model_key in DYNAMICAL_MODELS
  assert "graphcast" not in SUPPORTED_MODELS
  for var_key in (
      "precipitation",
      "accumulated_precip",
      "temperature",
      "wind",
      "pressure",
  ):
    assert var_key in SUPPORTED_VARIABLES

  precip_stored = to_stored_units("precip", np.array([2.0 / 3600.0, -0.5]))
  np.testing.assert_allclose(precip_stored.astype(float), [2.0, 0.0], rtol=1e-3)
  mslp_stored = to_stored_units("mslp", np.array([101325.0]))
  np.testing.assert_allclose(
      from_stored_units("mslp", mslp_stored), [1013.25], rtol=1e-3
  )
  temp_k = to_stored_units("temp", np.array([273.15, 293.15]))
  np.testing.assert_allclose(temp_k.astype(float), [0.0, 20.0], atol=0.05)


@pytest.mark.unit
def test_file_step_for_lead_exact_for_state_variables() -> None:
  """State variables map only to exactly stored leads; rates to intervals."""
  info = {"lead_hours": [0, 6, 12], "archived_run": True}
  assert file_step_for_lead(info, 0, is_rate=False) == 0
  assert file_step_for_lead(info, 3, is_rate=False) is None
  assert file_step_for_lead(info, 6, is_rate=False) == 1
  assert file_step_for_lead(info, 12, is_rate=False) == 2
  assert file_step_for_lead(info, 13, is_rate=False) is None
  assert file_step_for_lead(info, -3, is_rate=False) is None
  assert file_step_for_lead(info, 0, is_rate=True) == 1
  assert file_step_for_lead(info, 3, is_rate=True) == 1
  assert file_step_for_lead(info, 6, is_rate=True) == 1
  assert file_step_for_lead(info, 7, is_rate=True) == 2
  assert file_step_for_lead(info, 12, is_rate=True) == 2
  assert file_step_for_lead(info, 13, is_rate=True) is None


@pytest.mark.unit
def test_rate_file_steps_interval_coverage() -> None:
  """Tests rate_file_steps for hourly and 6-hourly forecast schedules."""
  hourly = {"lead_hours": list(range(121)) + [123, 126], "archived_run": True}
  assert rate_file_steps(hourly, 51) == [49, 50, 51]
  assert rate_file_steps(hourly, 0) == [1]
  assert rate_file_steps(hourly, 123) == [121]
  assert rate_file_steps(hourly, 129) == []

  six_hourly = {"lead_hours": [0, 6, 12, 18], "archived_run": True}
  assert rate_file_steps(six_hourly, 0) == [1]
  assert rate_file_steps(six_hourly, 3) == [1]
  assert rate_file_steps(six_hourly, 9) == [2]
  assert rate_file_steps(six_hourly, 21) == []


@pytest.mark.unit
def test_grid_indices_rounding_wrap_and_validation() -> None:
  """Half-cell positions round up, longitudes wrap, bad coordinates raise."""
  rows, cols = grid_indices(
      np.array([90.0, 89.875, 40.42, -90.0]),
      np.array([-180.0, -179.875, 179.875, 360.0]),
  )
  assert rows.tolist() == [0, 1, PROBE_ROW, N_LAT - 1]
  assert cols.tolist() == [0, 1, 0, 720]
  for bad_lat, bad_lon in ((90.5, 0.0), (0.0, -181.0), (np.nan, 0.0)):
    with pytest.raises(ValueError, match="must be finite"):
      grid_indices(np.array([bad_lat]), np.array([bad_lon]))


@pytest.mark.unit
def test_bilinear_interpolation_reproduces_linear_fields() -> None:
  """A bilinear field is reproduced exactly at arbitrary points."""
  lats = np.linspace(90.0, -90.0, N_LAT, dtype=np.float64)
  lons = np.linspace(-180.0, 179.75, N_LON, dtype=np.float64)
  plane = (lats[:, None] * 0.5 + lons[None, :] * 0.1).astype(np.float32)
  target_lats = np.array([45.125, 0.0, -33.875, 90.0, -90.0])
  target_lons = np.array([-86.875, 0.125, 151.125, 10.3])
  sampled = bilinear_sample_grid(plane, target_lats, target_lons)
  expected = target_lats[:, None] * 0.5 + target_lons[None, :] * 0.1
  np.testing.assert_allclose(sampled, expected, atol=1e-3)


@pytest.mark.unit
def test_bilinear_interpolation_wraps_antimeridian_and_masks_nan() -> None:
  """Longitudes wrap at 180 and NaN neighbours never leak into samples."""
  plane = np.zeros((N_LAT, N_LON), dtype=np.float32)
  plane[:, N_LON - 1] = 5.0
  plane[:, 0] = 7.0
  wrapped = bilinear_sample_grid(plane, np.array([10.0]), np.array([179.875]))
  assert pytest.approx(float(wrapped[0, 0]), abs=1e-6) == 6.0

  plane[10, 10] = np.nan
  lat10 = _lat_of_row(10)
  at_nan_node = bilinear_sample_grid(
      plane, np.array([lat10]), np.array([_lon_of_col(10)])
  )
  assert np.isnan(at_nan_node[0, 0])
  half_nan = bilinear_sample_grid(
      plane, np.array([lat10]), np.array([_lon_of_col(10) + 0.125])
  )
  assert np.isnan(half_nan[0, 0])  # only 50% of the weight is finite
  mostly_valid = bilinear_sample_grid(
      plane, np.array([lat10]), np.array([_lon_of_col(10) + 0.225])
  )
  assert pytest.approx(float(mostly_valid[0, 0]), abs=1e-6) == 0.0
  with pytest.raises(ValueError, match="Latitudes"):
    bilinear_sample_grid(plane, np.array([91.0]), np.array([0.0]))


@pytest.mark.unit
def test_wind_speed_and_direction_convention() -> None:
  """Direction is the meteorological 'blowing from' angle in `[0, 360)`."""
  speed, direction = compute_wind_speed_and_direction(
      np.array([0.0, -1.0, 0.0, 1.0, 6.0]), np.array([-1.0, 0.0, 1.0, 0.0, 8.0])
  )
  np.testing.assert_allclose(speed, [1.0, 1.0, 1.0, 1.0, 10.0])
  np.testing.assert_allclose(
      direction, [0.0, 90.0, 180.0, 270.0, 216.8699], atol=1e-3
  )


@pytest.mark.unit
def test_fetcher_requires_explicit_data_dir_and_rejects_unsynced_models(
    tmp_path: Path,
) -> None:
  """Verifies the explicit data_dir requirement and no synthetic fallbacks."""
  with pytest.raises(ValueError):
    WeatherDataFetcher(None)  # type: ignore[arg-type]
  with pytest.raises(ValueError):
    WeatherDataFetcher("")

  with WeatherDataFetcher(tmp_path) as empty_fetcher:
    info = empty_fetcher.get_model_info("ecmwf_aifs")
    assert info["data_source"] == "unavailable"
    assert info["real_variables"] == []
    assert info["stored_lead_hours"] is None
    with pytest.raises(FileNotFoundError):
      empty_fetcher.fetch_forecast_grid("ecmwf_aifs", "precipitation", 0)
    with pytest.raises(FileNotFoundError):
      empty_fetcher.fetch_wind_grid("ecmwf_aifs", step_idx=1)
    with pytest.raises(FileNotFoundError):
      empty_fetcher.fetch_point_timeseries(40.0, -86.0)
    with pytest.raises(FileNotFoundError):
      empty_fetcher.fetch_point_timeseries(40.0, -86.0, models=["noaa_gfs"])
    with pytest.raises(FileNotFoundError):
      empty_fetcher.fetch_catchment_summary(
          _feature({"type": "Polygon", "coordinates": _box(-87, 40, -86, 41)}),
          step_idx=0,
          model_key="ecmwf_aifs",
      )
    status = empty_fetcher.get_sync_status()
    assert status["sync_status_found"] is False
    assert status["check_interval_minutes"] is None
    with pytest.raises(ValueError, match="Unknown weather model"):
      empty_fetcher.get_model_info("graphcast")


@pytest.mark.unit
def test_scan_streams_rejects_corrupt_files_and_inconsistent_metadata(
    tmp_path: Path,
) -> None:
  """Truncated, empty, undescribed, and mislabelled streams raise."""
  plane_bytes = N_LAT * N_LON * 2
  assert scan_streams(tmp_path / "missing") == ({}, {}, {})
  assert scan_streams(tmp_path) == ({}, {}, {})

  run_dir = tmp_path / "run"
  run_dir.mkdir()
  zero = np.zeros((N_LAT, N_LON), dtype=np.float16)
  _write_stream(run_dir, "noaa_gfs_precip", [zero, zero])
  with pytest.raises(FileNotFoundError, match=RUN_METADATA_FILE):
    scan_streams(run_dir)

  entry = _dataset_entry(
      "noaa_gfs", INIT_TIME, [0, 3], ("precip",), "2026-09-29T03:00:00Z"
  )
  _write_meta(run_dir, {"noaa_gfs_forecast": entry}, "2026-09-29T03:00:00Z")
  handles, infos, arrays = scan_streams(run_dir)
  assert infos["noaa_gfs_precip"]["lead_hours"] == [0, 3]
  assert infos["noaa_gfs_precip"]["size_bytes"] == 2 * plane_bytes
  assert arrays["noaa_gfs_precip"].shape == (2, N_LAT, N_LON)
  del arrays
  assert close_unreferenced_mmaps(handles) == 1

  _write_meta(
      run_dir,
      {
          "noaa_gfs_forecast": {
              **entry,
              "lead_hours": [0, 3, 6],
              "lead_steps": 3,
          }
      },
      "2026-09-29T03:00:00Z",
  )
  with pytest.raises(ValueError, match="lists 3 lead hours"):
    scan_streams(run_dir)

  _write_meta(
      run_dir,
      {"noaa_gfs_forecast": {**entry, "lead_hours": [3, 0]}},
      "2026-09-29T03:00:00Z",
  )
  with pytest.raises(ValueError, match="strictly increasing"):
    scan_streams(run_dir)

  _write_meta(
      run_dir,
      {"noaa_gfs_forecast": {**entry, "model": "noaa_gefs"}},
      "2026-09-29T03:00:00Z",
  )
  with pytest.raises(ValueError, match="not described"):
    scan_streams(run_dir)

  _write_meta(run_dir, {"noaa_gfs_forecast": entry}, "2026-09-29T03:00:00Z")
  with open(run_dir / "noaa_gfs_precip.bin", "r+b") as f_bin:
    f_bin.truncate(2 * plane_bytes - 10)
  with pytest.raises(ValueError, match="truncated or corrupt"):
    scan_streams(run_dir)
  with open(run_dir / "noaa_gfs_precip.bin", "r+b") as f_bin:
    f_bin.truncate(0)
  with pytest.raises(ValueError, match="truncated or corrupt"):
    scan_streams(run_dir)


@pytest.mark.unit
def test_model_info_reports_stored_leads_and_missing_variables(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Model info reflects exactly what was synced, per model."""
  ifs = main_fetcher.get_model_info("ecmwf_ifs")
  assert ifs["data_source"] == "archived_run"
  assert ifs["init_time"] == f"{INIT_TIME}Z"
  assert ifs["max_lead_hours"] == 24
  assert ifs["stored_lead_hours"] == list(range(0, 25, 3))
  assert ifs["real_variables"] == [
      "precipitation",
      "accumulated_precip",
      "temperature",
      "pressure",
      "wind",
  ]
  assert ifs["missing_variables"] == []

  aifs = main_fetcher.get_model_info("ecmwf_aifs")
  assert aifs["stored_lead_hours"] == [0, 6, 12, 18, 24]

  gefs = main_fetcher.get_model_info("noaa_gefs")
  assert gefs["missing_variables"] == ["pressure", "wind"]

  gfs = main_fetcher.get_model_info("noaa_gfs")
  assert gfs["data_source"] == "unavailable"
  assert gfs["max_lead_hours"] == 0

  by_key = {m["id"]: m for m in main_fetcher.get_all_models_info()}
  assert set(by_key) == set(SUPPORTED_MODELS)
  assert by_key["ecmwf_ifs"]["real_variables"] == ifs["real_variables"]


@pytest.mark.unit
def test_native_grid_fetch_matches_stored_planes(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Rates, state variables, and accumulations match the written planes."""
  rate = main_fetcher.fetch_forecast_grid("ecmwf_ifs", "precipitation", 2)
  assert rate.shape == (N_LAT, N_LON)
  np.testing.assert_array_equal(rate, _precip_plane(2).astype(np.float32))

  rate0 = main_fetcher.fetch_forecast_grid("ecmwf_ifs", "precipitation", 0)
  np.testing.assert_array_equal(rate0, _precip_plane(1).astype(np.float32))

  temp = main_fetcher.fetch_forecast_grid("ecmwf_ifs", "temperature", 3)
  np.testing.assert_array_equal(temp, _temp_plane(3).astype(np.float32))

  pressure = main_fetcher.fetch_forecast_grid("ecmwf_ifs", "pressure", 1)
  np.testing.assert_allclose(
      pressure, _mslp_plane(1).astype(np.float32) + MSLP_OFFSET_HPA, atol=1e-4
  )

  accum = main_fetcher.fetch_forecast_grid("ecmwf_ifs", "accumulated_precip", 3)
  expected_accum = 3.0 * sum(
      _precip_plane(i).astype(np.float32) for i in (1, 2, 3)
  )
  np.testing.assert_allclose(accum, expected_accum, atol=1e-4)
  accum0 = main_fetcher.fetch_forecast_grid(
      "ecmwf_ifs", "accumulated_precip", 0
  )
  assert float(np.abs(accum0).max()) == 0.0

  assert main_fetcher.fetch_forecast_grid("ecmwf_ifs", "temperature", 9) is None
  assert (
      main_fetcher.fetch_forecast_grid("ecmwf_ifs", "accumulated_precip", 9)
      is None
  )
  with pytest.raises(ValueError, match="step_idx"):
    main_fetcher.fetch_forecast_grid("ecmwf_ifs", "temperature", -1)
  with pytest.raises(ValueError, match="Unsupported"):
    main_fetcher.fetch_forecast_grid("ecmwf_ifs", "wind", 1)
  with pytest.raises(FileNotFoundError):
    main_fetcher.fetch_forecast_grid("noaa_gefs", "pressure", 1)


@pytest.mark.unit
def test_custom_grid_fetch_nearest_and_bilinear(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Custom lat/lon targets sample nearest cells or bilinear blends."""
  lats = np.array([_lat_of_row(40), _lat_of_row(40) - 0.125])
  lons = np.array([_lon_of_col(100), _lon_of_col(100) + 0.125])
  nearest = main_fetcher.fetch_forecast_grid(
      "ecmwf_ifs", "temperature", 2, lats=lats, lons=lons
  )
  assert nearest.shape == (2, 2)
  assert nearest[0, 0] == _expected_temp(2, 40, 100)
  assert nearest[1, 1] == _expected_temp(2, 41, 101)

  blended = main_fetcher.fetch_forecast_grid(
      "ecmwf_ifs", "temperature", 2, lats=lats, lons=lons, bilinear=True
  )
  assert blended[0, 0] == _expected_temp(2, 40, 100)
  expected_mid = 0.25 * sum(
      _expected_temp(2, r, c) for r in (40, 41) for c in (100, 101)
  )
  assert pytest.approx(float(blended[1, 1]), abs=1e-5) == expected_mid


@pytest.mark.unit
def test_six_hourly_model_never_substitutes_neighbouring_leads(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """A 6-hourly model yields None for state variables at +3 h steps."""
  assert (
      main_fetcher.fetch_forecast_grid("ecmwf_aifs", "temperature", 1) is None
  )
  assert main_fetcher.fetch_forecast_grid("ecmwf_aifs", "pressure", 3) is None
  temp6 = main_fetcher.fetch_forecast_grid("ecmwf_aifs", "temperature", 2)
  np.testing.assert_array_equal(temp6, _temp_plane(1).astype(np.float32))

  # Rates keep interval semantics: +3 h lies in the (0, 6] h interval.
  rate3 = main_fetcher.fetch_forecast_grid("ecmwf_aifs", "precipitation", 1)
  np.testing.assert_array_equal(rate3, _precip_plane(1).astype(np.float32))
  accum3 = main_fetcher.fetch_forecast_grid(
      "ecmwf_aifs", "accumulated_precip", 1
  )
  assert float(np.abs(accum3).max()) == 0.0  # no stored lead <= 3 h beyond 0
  accum6 = main_fetcher.fetch_forecast_grid(
      "ecmwf_aifs", "accumulated_precip", 2
  )
  np.testing.assert_allclose(
      accum6, 6.0 * _precip_plane(1).astype(np.float32), atol=1e-4
  )

  with pytest.raises(ValueError, match="not stored"):
    main_fetcher.fetch_wind_grid("ecmwf_aifs", step_idx=1)
  probe = main_fetcher.fetch_point_timeseries(
      PROBE_LAT, PROBE_LON, models=["ecmwf_aifs"]
  )
  curves = probe["models"]["ecmwf_aifs"]
  assert curves["stored_lead_hours"] == [0, 6, 12, 18, 24]
  assert curves["temp_c"][1] is None
  assert curves["wind_speed_mps"][1] is None
  assert curves["pressure_hpa"][1] is None
  assert curves["temp_c"][2] == _expected_temp(1, PROBE_ROW, PROBE_COL)
  assert pytest.approx(
      curves["precip_rate_mmh"][1], abs=0.0051
  ) == _expected_precip(1, PROBE_ROW, PROBE_COL)


@pytest.mark.unit
def test_wind_grid_global_bbox_and_antimeridian(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Wind grids cover the globe, viewports, and antimeridian-crossing boxes."""
  wind = main_fetcher.fetch_wind_grid("ecmwf_ifs", step_idx=2, subsample=2)
  header = wind["header"]
  assert (header["nx"], header["ny"]) == (180, 91)
  assert (header["lo1"], header["la1"], header["dx"]) == (-180.0, 90.0, 2.0)
  assert header["valid_time"] == "2026-09-29T06:00:00Z"
  assert header["missing_count"] == 0
  assert len(wind["u"]) == len(wind["v"]) == 180 * 91
  assert wind["u"][0] == _expected_u10(0, 0)
  assert wind["v"][0] == _expected_v10(0, 0)
  # Second row (lat 88), third column (lon -176): row 8, col 16.
  assert wind["u"][180 + 2] == _expected_u10(8, 16)

  bbox = main_fetcher.fetch_wind_grid(
      "ecmwf_ifs", step_idx=2, subsample=1, bbox=(-90.0, 35.0, -80.0, 45.0)
  )
  assert (bbox["header"]["nx"], bbox["header"]["ny"]) == (11, 11)
  assert bbox["header"]["la1"] == 45.0
  # First sample: lat 45 (row 180), lon -90 (col 360).
  assert bbox["u"][0] == _expected_u10(180, 360)

  crossing = main_fetcher.fetch_wind_grid(
      "ecmwf_ifs", step_idx=0, subsample=1, bbox=(170.0, -10.0, -170.0, 10.0)
  )
  assert crossing["header"]["nx"] == 21
  assert crossing["header"]["lo1"] == 170.0
  # Header lon 185 (index 15) samples lon -175: row 320 (10 N), col 20.
  assert crossing["u"][15] == _expected_u10(320, 20)

  with pytest.raises(ValueError, match="subsample"):
    main_fetcher.fetch_wind_grid("ecmwf_ifs", step_idx=0, subsample=5)
  with pytest.raises(ValueError, match="subsample"):
    main_fetcher.fetch_wind_grid("ecmwf_ifs", step_idx=0, subsample=0)
  with pytest.raises(ValueError, match="min_lat"):
    main_fetcher.fetch_wind_grid(
        "ecmwf_ifs", step_idx=0, bbox=(-90.0, 45.0, -80.0, 35.0)
    )
  with pytest.raises(FileNotFoundError):
    main_fetcher.fetch_wind_grid("noaa_gefs", step_idx=0)


@pytest.mark.unit
def test_point_timeseries_values_and_strictness(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Meteogram curves match the stored planes at the nearest grid cell."""
  probe = main_fetcher.fetch_point_timeseries(PROBE_LAT, PROBE_LON)
  assert set(probe["models"]) == {"ecmwf_ifs", "ecmwf_aifs", "noaa_gefs"}
  assert probe["lead_hours"] == [3 * i for i in range(81)]
  ifs = probe["models"]["ecmwf_ifs"]
  assert ifs["name"] == SUPPORTED_MODELS["ecmwf_ifs"]["name"]
  assert ifs["stored_lead_hours"] == list(range(0, 25, 3))
  for key in (
      "precip_rate_mmh",
      "accum_precip_mm",
      "temp_c",
      "wind_speed_mps",
      "wind_direction_deg",
      "pressure_hpa",
  ):
    assert len(ifs[key]) == 81
    assert all(v is None for v in ifs[key][9:]), key

  for k in range(9):
    plane = max(k, 1)
    assert pytest.approx(
        ifs["precip_rate_mmh"][k], abs=0.0051
    ) == _expected_precip(plane, PROBE_ROW, PROBE_COL)
    expected_accum = 3.0 * sum(
        _expected_precip(i, PROBE_ROW, PROBE_COL) for i in range(1, k + 1)
    )
    assert pytest.approx(ifs["accum_precip_mm"][k], abs=0.051) == expected_accum
    assert ifs["temp_c"][k] == _expected_temp(k, PROBE_ROW, PROBE_COL)
    assert (
        pytest.approx(ifs["pressure_hpa"][k], abs=0.051)
        == _expected_pressure(k, PROBE_ROW, PROBE_COL)
    )

  u_exp = _expected_u10(PROBE_ROW, PROBE_COL)
  v_exp = _expected_v10(PROBE_ROW, PROBE_COL)
  speed, direction = compute_wind_speed_and_direction(
      np.array([u_exp]), np.array([v_exp])
  )
  assert ifs["wind_speed_mps"][4] == round(float(speed[0]), 1)
  assert ifs["wind_direction_deg"][4] == int(direction[0])

  gefs = probe["models"]["noaa_gefs"]
  assert gefs["precip_rate_mmh"][1] == 0.0
  assert pytest.approx(gefs["temp_c"][1], abs=0.051) == 20.0 + 0.125 * (
      PROBE_ROW % 4
  )
  assert gefs["wind_speed_mps"][1] is None
  assert gefs["pressure_hpa"][1] is None

  with pytest.raises(FileNotFoundError):
    main_fetcher.fetch_point_timeseries(
        PROBE_LAT, PROBE_LON, models=["ecmwf_ifs", "noaa_gfs"]
    )
  lenient = main_fetcher.fetch_point_timeseries(
      PROBE_LAT, PROBE_LON, models=["ecmwf_ifs", "noaa_gfs"], strict=False
  )
  assert lenient["models"]["noaa_gfs"]["data_source"] == "unavailable"
  assert all(v is None for v in lenient["models"]["noaa_gfs"]["temp_c"])
  with pytest.raises(FileNotFoundError):
    main_fetcher.fetch_point_timeseries(
        PROBE_LAT, PROBE_LON, models=["noaa_gfs"], strict=False
    )
  with pytest.raises(ValueError, match="Unknown weather model"):
    main_fetcher.fetch_point_timeseries(PROBE_LAT, PROBE_LON, models=["gfs"])
  with pytest.raises(ValueError, match="Latitudes"):
    main_fetcher.fetch_point_timeseries(95.0, PROBE_LON)


@pytest.mark.unit
def test_catchment_summary_is_exactly_area_weighted(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Basin means weight cells by intersection area times cos(latitude)."""
  rain_w = sum(_cos(r) for r in RAIN_ROWS)
  all_rows = (196, 197, 198, 199)

  # Aligned 4 x 4 block of cells around the 2 x 2 rain block.
  aligned = _feature(
      {
          "type": "Polygon",
          "coordinates": _box(-87.125, 40.125, -86.125, 41.125),
      },
      area_km2=4500.0,
  )
  summary = main_fetcher.fetch_catchment_summary(
      aligned, step_idx=2, model_key="noaa_gefs"
  )
  expected = RAIN_MMH * 2 * rain_w / (4 * sum(_cos(r) for r in all_rows))
  assert summary["catchment_id"] == "basin"
  assert summary["grid_cells"] == 16
  assert summary["area_km2"] == 4500.0
  assert summary["area_km2_source"] == "properties"
  assert summary["valid_time_utc"] == "2026-09-29T06:00:00Z"
  assert summary["accumulation_hours"] == 24
  assert pytest.approx(summary["basin_mean_precip_mmh"], abs=0.006) == expected
  assert summary["basin_max_precip_mmh"] == RAIN_MMH
  assert (
      pytest.approx(summary["basin_accumulated_10d_mm"], abs=0.06)
      == 24.0 * expected
  )
  assert summary["basin_mean_temp_c"] is not None
  assert summary["missing_area_fraction"] == {
      "precipitation": 0.0,
      "accumulated_precip": 0.0,
      "temperature": 0.0,
  }
  assert summary["centroid"] == {"latitude": 40.625, "longitude": -86.625}

  # Offset by half a cell: edge rows/columns count with half weight.
  offset = _feature(
      {"type": "Polygon", "coordinates": _box(-87.0, 40.25, -86.25, 41.0)},
      feature_id="offset",
  )
  summary = main_fetcher.fetch_catchment_summary(
      offset, step_idx=2, model_key="noaa_gefs"
  )
  row_w = 0.5 * _cos(196) + _cos(197) + _cos(198) + 0.5 * _cos(199)
  expected = RAIN_MMH * 2 * rain_w / (3.0 * row_w)
  assert pytest.approx(summary["basin_mean_precip_mmh"], abs=0.006) == expected
  assert summary["area_km2_source"] == "geometry"
  assert 5000.0 < summary["area_km2"] < 5400.0


@pytest.mark.unit
def test_catchment_summary_holes_multipolygons_and_subgrid_basins(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Interior rings are excluded; MultiPolygons and tiny basins work."""
  rain_box = _box(-86.875, 40.375, -86.375, 40.875)
  with_hole = _feature({
      "type": "Polygon",
      "coordinates": _box(-87.0, 40.25, -86.25, 41.0) + rain_box,
  })
  summary = main_fetcher.fetch_catchment_summary(
      with_hole, step_idx=2, model_key="noaa_gefs"
  )
  assert summary["basin_mean_precip_mmh"] == 0.0
  assert summary["basin_max_precip_mmh"] == 0.0
  assert summary["grid_cells"] == 12

  multi = _feature({
      "type": "MultiPolygon",
      "coordinates": [rain_box, _box(-85.125, 40.375, -84.625, 40.875)],
  })
  summary = main_fetcher.fetch_catchment_summary(
      multi, step_idx=2, model_key="noaa_gefs"
  )
  assert summary["grid_cells"] == 8
  assert pytest.approx(summary["basin_mean_precip_mmh"], abs=1e-6) == 5.0

  tiny = _feature({
      "type": "Polygon",
      "coordinates": _box(-86.775, 40.725, -86.725, 40.775),
  })
  summary = main_fetcher.fetch_catchment_summary(
      tiny, step_idx=2, model_key="noaa_gefs"
  )
  assert summary["grid_cells"] == 1
  assert summary["basin_mean_precip_mmh"] == RAIN_MMH
  assert 15.0 < summary["area_km2"] < 30.0


@pytest.mark.unit
def test_catchment_summary_rejects_bad_features(
    main_fetcher: WeatherDataFetcher,
) -> None:
  """Points, missing geometry, missing ids, and antimeridian spans raise."""
  square = _box(-87.0, 40.0, -86.0, 41.0)
  with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
    main_fetcher.fetch_catchment_summary(
        _feature({"type": "Point", "coordinates": [-86.5, 40.5]}),
        step_idx=0,
        model_key="noaa_gefs",
    )
  with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
    main_fetcher.fetch_catchment_summary(
        _feature(None), step_idx=0, model_key="noaa_gefs"
    )
  with pytest.raises(ValueError, match="no coordinates"):
    main_fetcher.fetch_catchment_summary(
        _feature({"type": "Polygon", "coordinates": []}),
        step_idx=0,
        model_key="noaa_gefs",
    )
  with pytest.raises(KeyError, match="catchment_id"):
    main_fetcher.fetch_catchment_summary(
        _feature({"type": "Polygon", "coordinates": square}, feature_id=None),
        step_idx=0,
        model_key="noaa_gefs",
    )
  with pytest.raises(ValueError, match="antimeridian"):
    main_fetcher.fetch_catchment_summary(
        _feature({
            "type": "Polygon",
            "coordinates": _box(170.0, -5.0, -170.0, 5.0),
        }),
        step_idx=0,
        model_key="noaa_gefs",
    )
  with pytest.raises(ValueError, match="EPSG:4326"):
    geojson_polygon_to_shapely(
        {"type": "Polygon", "coordinates": _box(0.0, 0.0, 200.0, 10.0)}
    )
  bowtie = {
      "type": "Polygon",
      "coordinates": [[[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]]],
  }
  with pytest.raises(ValueError, match="not a valid polygon"):
    geojson_polygon_to_shapely(bowtie)
  with pytest.raises(ValueError, match="area_km2"):
    main_fetcher.fetch_catchment_summary(
        _feature({"type": "Polygon", "coordinates": square}, area_km2=0.0),
        step_idx=0,
        model_key="noaa_gefs",
    )
  with pytest.raises(FileNotFoundError):
    main_fetcher.fetch_catchment_summary(
        _feature({"type": "Polygon", "coordinates": square}),
        step_idx=0,
        model_key="noaa_gfs",
    )


@pytest.mark.unit
def test_masked_cells_stay_missing_everywhere(
    hrrr_fetcher: WeatherDataFetcher,
) -> None:
  """NaN cells of a regional model are NaN/None in every product."""
  inside = (_lat_of_row(200), _lon_of_col(300))  # 40 N, 105 W
  outside = (_lat_of_row(200), _lon_of_col(760))  # 40 N, 10 E
  in_rc = (200, 300)
  out_rc = (200, 760)

  rate = hrrr_fetcher.fetch_forecast_grid("noaa_hrrr", "precipitation", 1)
  assert rate[in_rc] == HRRR_RATE
  assert np.isnan(rate[out_rc])
  assert int(np.isfinite(rate).sum()) == (HRRR_ROWS[1] - HRRR_ROWS[0]) * (
      HRRR_COLS[1] - HRRR_COLS[0]
  )

  accum0 = hrrr_fetcher.fetch_forecast_grid(
      "noaa_hrrr", "accumulated_precip", 0
  )
  assert accum0[in_rc] == 0.0
  assert np.isnan(accum0[out_rc])
  accum = hrrr_fetcher.fetch_forecast_grid("noaa_hrrr", "accumulated_precip", 2)
  assert accum[in_rc] == 2 * 3.0 * HRRR_RATE
  assert np.isnan(accum[out_rc])

  temp = hrrr_fetcher.fetch_forecast_grid("noaa_hrrr", "temperature", 1)
  assert np.isnan(temp[out_rc])
  edge_lat = _lat_of_row(HRRR_ROWS[0])
  edge_lon = _lon_of_col(HRRR_COLS[0])
  blended = hrrr_fetcher.fetch_forecast_grid(
      "noaa_hrrr",
      "temperature",
      1,
      lats=np.array([edge_lat, edge_lat + 0.125]),
      lons=np.array([edge_lon, edge_lon - 0.125]),
      bilinear=True,
  )
  assert blended[0, 0] == HRRR_TEMP  # domain-edge node keeps its value
  assert np.isnan(blended[1, 1])  # halfway into the masked region

  arrays, stream_info = hrrr_fetcher.snapshot()
  assert (
      extract_point_value(
          arrays, stream_info, "noaa_hrrr", "temperature", 3, *outside
      )
      is None
  )
  assert extract_accumulation_series(
      arrays, stream_info, "noaa_hrrr", *outside, [0, 3, 6, 9]
  ) == [None, None, None, None]
  assert extract_accumulation_series(
      arrays, stream_info, "noaa_hrrr", *inside, [0, 3, 6, 9]
  ) == [0.0, 6.0, 12.0, None]

  probe = hrrr_fetcher.fetch_point_timeseries(*outside)["models"]["noaa_hrrr"]
  for key in ("precip_rate_mmh", "accum_precip_mm", "temp_c", "wind_speed_mps"):
    assert all(v is None for v in probe[key]), key
  probe_in = hrrr_fetcher.fetch_point_timeseries(*inside)["models"]["noaa_hrrr"]
  assert probe_in["accum_precip_mm"][:3] == [0.0, 6.0, 12.0]
  assert probe_in["wind_speed_mps"][1] == 5.0
  assert probe_in["pressure_hpa"][1] == HRRR_MSLP + MSLP_OFFSET_HPA

  wind = hrrr_fetcher.fetch_wind_grid("noaa_hrrr", step_idx=1, subsample=4)
  assert wind["header"]["missing_count"] == sum(
      1 for v in wind["u"] if v is None
  )
  assert wind["header"]["missing_count"] > 0
  # lat 40 is row 12 (4 deg steps from 90), lon -104 is col 19 -> inside.
  inside_idx = 12 * wind["header"]["nx"] + 19
  assert (wind["u"][inside_idx], wind["v"][inside_idx]) == (HRRR_U, HRRR_V)
  assert wind["u"][0] is None

  box_in = _feature(
      {"type": "Polygon", "coordinates": _box(-105.0, 38.0, -103.0, 40.0)}
  )
  summary = hrrr_fetcher.fetch_catchment_summary(
      box_in, step_idx=2, model_key="noaa_hrrr"
  )
  assert summary["basin_mean_precip_mmh"] == HRRR_RATE
  assert summary["basin_accumulated_10d_mm"] == 2 * 3.0 * HRRR_RATE
  assert summary["basin_mean_temp_c"] == HRRR_TEMP

  # Mostly outside the domain (east of 65 W): below 80% coverage -> None.
  box_edge = _feature(
      {"type": "Polygon", "coordinates": _box(-66.0, 38.0, -60.0, 40.0)},
      feature_id="edge",
  )
  summary = hrrr_fetcher.fetch_catchment_summary(
      box_edge, step_idx=2, model_key="noaa_hrrr"
  )
  assert summary["basin_mean_precip_mmh"] is None
  assert summary["basin_max_precip_mmh"] is None
  assert summary["basin_accumulated_10d_mm"] is None
  assert summary["basin_mean_temp_c"] is None
  assert 0.7 < summary["missing_area_fraction"]["precipitation"] < 0.9


@pytest.mark.unit
def test_wind_grid_requires_recorded_init_time() -> None:
  """No valid time is invented when the run metadata lacks an init time."""
  plane = np.zeros((1, N_LAT, N_LON), dtype=np.float16)
  arrays = {"ecmwf_ifs_u10": plane, "ecmwf_ifs_v10": plane}
  info = {
      "model": "ecmwf_ifs",
      "lead_hours": [0],
      "archived_run": True,
      "init_time": None,
      "offset": 0.0,
  }
  stream_info = {"ecmwf_ifs_u10": dict(info), "ecmwf_ifs_v10": dict(info)}
  with pytest.raises(ValueError, match="init time"):
    fetch_wind_grid(arrays, stream_info, "ecmwf_ifs", step_idx=0)


@pytest.mark.unit
def test_accumulation_cache_is_read_only_and_keyed_per_file(
    tmp_path: Path,
) -> None:
  """Cached totals are immutable and never served for a re-synced file."""
  clear_accum_grid_cache()
  _write_small_run(tmp_path, "r1", INIT_TIME, rain_mmh=4.0)
  with WeatherDataFetcher(tmp_path) as fetcher:
    arrays, stream_info = fetcher.snapshot()
    first, res = compute_accumulated_precip_grid(
        arrays, stream_info, "ecmwf_aifs", 3
    )
    assert res == 0.25
    assert first.flags.writeable is False
    assert float(first[0, 0]) == 12.0
    again, _ = compute_accumulated_precip_grid(
        arrays, stream_info, "ecmwf_aifs", 3
    )
    assert again is first
    assert (
        compute_accumulated_precip_grid(arrays, stream_info, "ecmwf_aifs", 6)
        is None
    )

    # Same init time, new run directory with different data.
    _write_small_run(tmp_path, "r2", INIT_TIME, rain_mmh=1.0)
    assert fetcher.reload_if_changed() is True
    arrays, stream_info = fetcher.snapshot()
    updated, _ = compute_accumulated_precip_grid(
        arrays, stream_info, "ecmwf_aifs", 3
    )
    assert float(updated[0, 0]) == 3.0
    assert float(first[0, 0]) == 12.0

    clear_accum_grid_cache()
    fresh, _ = compute_accumulated_precip_grid(
        arrays, stream_info, "ecmwf_aifs", 3
    )
    assert fresh is not updated
    np.testing.assert_array_equal(fresh, updated)


@pytest.mark.unit
def test_hot_reload_keeps_in_flight_views_valid(tmp_path: Path) -> None:
  """Readers holding arrays from the previous run are never invalidated."""
  _write_small_run(tmp_path, "r1", "2026-09-29T00:00:00", rain_mmh=4.0)
  fetcher = WeatherDataFetcher(tmp_path)
  assert fetcher.reload_if_changed() is False
  old_arrays, old_info = fetcher.snapshot()
  live_view = old_arrays["ecmwf_aifs_precip"][1]
  del old_arrays  # keep only one in-flight view alive
  old_handles = {
      sid: weakref.ref(handle[0]) for sid, handle in fetcher.handles.items()
  }

  _write_small_run(tmp_path, "r2", "2026-09-29T06:00:00", rain_mmh=2.0)
  assert fetcher.reload_if_changed() is True
  assert fetcher.reload_if_changed() is False
  assert fetcher.get_model_info("ecmwf_aifs")["init_time"] == (
      "2026-09-29T06:00:00Z"
  )
  assert float(fetcher.arrays["ecmwf_aifs_precip"][1][0, 0]) == 2.0
  assert float(live_view[0, 0]) == 4.0
  assert old_info["ecmwf_aifs_precip"]["init_time"] == "2026-09-29T00:00:00Z"

  # Streams without live views were closed on reload; the viewed one is open
  # until its last view disappears.
  precip_mm = old_handles["ecmwf_aifs_precip"]()
  assert precip_mm is not None and not precip_mm.closed
  for sid in ("ecmwf_aifs_temp", "ecmwf_aifs_u10", "ecmwf_aifs_v10"):
    retired = old_handles[sid]()
    assert retired is None or retired.closed, sid
  del precip_mm, live_view
  gc.collect()
  assert old_handles["ecmwf_aifs_precip"]() is None

  fetcher.close()
  assert fetcher.arrays == {}
  assert fetcher.handles == {}
  assert fetcher.get_model_info("ecmwf_aifs")["data_source"] == "unavailable"


@pytest.mark.unit
def test_close_defers_only_maps_with_live_views(tmp_path: Path) -> None:
  """`close()` unmaps idle streams now and in-use streams once released."""
  _write_small_run(tmp_path, "r1", INIT_TIME, rain_mmh=4.0)
  with WeatherDataFetcher(tmp_path) as fetcher:
    refs = {sid: weakref.ref(h[0]) for sid, h in fetcher.handles.items()}
    kept = fetcher.arrays["ecmwf_aifs_temp"]
  temp_mm = refs["ecmwf_aifs_temp"]()
  assert temp_mm is not None and not temp_mm.closed
  for sid in ("ecmwf_aifs_precip", "ecmwf_aifs_u10", "ecmwf_aifs_v10"):
    idle = refs[sid]()
    assert idle is None or idle.closed, sid
  assert float(kept[0, 0, 0]) == 18.5  # still readable after close()
  del kept, temp_mm
  gc.collect()
  assert refs["ecmwf_aifs_temp"]() is None


@pytest.mark.unit
def test_concurrent_reads_during_hot_reload_never_fail(tmp_path: Path) -> None:
  """Grid, probe, and wind requests keep succeeding while runs are swapped."""
  _write_small_run(tmp_path, "r1", "2026-09-29T00:00:00", rain_mmh=4.0)
  fetcher = WeatherDataFetcher(tmp_path)
  errors: list[str] = []
  stop = threading.Event()
  completed = [0] * 6

  def reader(idx: int) -> None:
    while not stop.is_set():
      grid = fetcher.fetch_forecast_grid("ecmwf_aifs", "accumulated_precip", 1)
      assert grid is not None and math.isfinite(float(grid[idx, idx]))
      probe = fetcher.fetch_point_timeseries(
          40.0 + idx, -86.0, models=["ecmwf_aifs"]
      )
      assert probe["models"]["ecmwf_aifs"]["temp_c"][1] == 18.5
      wind = fetcher.fetch_wind_grid("ecmwf_aifs", step_idx=1, subsample=4)
      assert wind["header"]["missing_count"] == 0
      completed[idx] += 1

  import concurrent.futures
  def guarded(idx: int) -> None:
    reader(idx)

  executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
  futures = [executor.submit(guarded, i) for i in range(6)]
  reloads = 0
  for k in range(2, 6):
    _write_small_run(
        tmp_path, f"r{k}", f"2026-09-29T{6 * (k % 4):02d}:00:00", float(k)
    )
    reloads += int(fetcher.reload_if_changed())
    time.sleep(0.05)
  deadline = time.time() + 0.5
  while time.time() < deadline and not stop.is_set():
    time.sleep(0.02)
  stop.set()
  for future in futures:
      exc = future.exception()
      if exc:
          errors.append(repr(exc))
  executor.shutdown(wait=True)
  fetcher.close()
  assert not errors, errors
  assert reloads == 4
  assert all(count > 0 for count in completed), completed


@pytest.mark.canary
def test_canary_dynamical_stac_catalog_reachable() -> None:
  """Live canary check verifying dynamical.org STAC catalog is reachable."""
  import importlib.util
  import urllib.request

  from multimet.weather_fetcher.config import STAC_CATALOG_URL

  with urllib.request.urlopen(STAC_CATALOG_URL, timeout=15) as resp:
    catalog_json = json.loads(resp.read().decode("utf-8"))
  assert catalog_json.get("id") == "dynamical-org"
  if importlib.util.find_spec("pystac") is not None:
    cat = wf.open_dynamical_catalog()
    assert cat is not None
    child = cat.get_child("noaa-gfs-forecast")
    assert child is not None
def test_to_xarray_physical_units_and_coords(main_fetcher: Any) -> None:
  """Verifies physical units, coordinates, NaN preservation, and leads."""
  ds = main_fetcher.to_xarray("ecmwf_ifs")

  assert "precipitation" in ds
  assert "temperature" in ds
  assert ds["precipitation"].attrs["units"] == "mm/h"
  assert ds["temperature"].attrs["units"] == "degC"

  assert "latitude" in ds.coords
  assert "longitude" in ds.coords
  assert "lead_time" in ds.coords
  assert "valid_time" in ds.coords

  assert ds.sizes["latitude"] == 721
  assert ds.sizes["longitude"] == 1440
  assert ds.sizes["lead_time"] > 0

  import pandas as pd

  assert ds.lead_time.values[0] == pd.Timedelta(hours=0)
  assert "init_time" in ds.attrs


def test_hres_extract_for_basins_zarr_preserves_continuous_missing_fraction(
    tmp_path: Path,
) -> None:
  """Verifies extract_for_basins_zarr preserves continuous spatial missing fraction."""
  import geopandas as gpd
  import pandas as pd
  import shapely.geometry
  import xarray as xr

  from multimet.timeseries_extractors.hres import HRESExtractor
  from multimet.utils.zonal import ZonalWeightCalculator

  zarr_path = tmp_path / "hres_synth.zarr"

  # 1 row (lat=0.5) x 10 cols (lon=0.5..9.5) over 240 hourly steps from 2016-01-01
  t2m_data = np.full((240, 1, 10), 283.15, dtype=np.float32)
  # Mask 1 of 10 equal-area cells (10% missing, 90% valid >= 80% coverage threshold)
  t2m_data[:, 0, 0] = np.nan
  xr.Dataset({"2m_temperature": (["time", "lat", "lon"], t2m_data)}).to_zarr(
      zarr_path, mode="w"
  )

  extractor = HRESExtractor(data_dir=str(zarr_path), source="zarr")
  extractor.lats = np.array([0.5], dtype=np.float64)
  extractor.lons = np.arange(0.5, 10.0, 1.0, dtype=np.float64)
  extractor.sort_lon_idx = np.arange(10)
  extractor.zonal_calc = ZonalWeightCalculator(
      extractor.lats, extractor.lons, cell_res_lat=1.0, cell_res_lon=1.0
  )

  basins_gdf = gpd.GeoDataFrame(
      {"geometry": [shapely.geometry.box(0.0, 0.0, 10.0, 1.0)]},
      index=pd.Index(["basin_partial"], name="basin"),
      crs="EPSG:4326",
  )

  ds = extractor.extract_for_basins_zarr(
      basins_gdf, start_date="2016-01-01", end_date="2016-01-01"
  )
  temp_vals = ds["hres_temperature_2m"].values
  miss_vals = ds["hres_missing_fraction"].values
  assert temp_vals.shape == (1, 1, 10)
  assert miss_vals.shape == (1, 1, 10)
  np.testing.assert_allclose(temp_vals, 10.0, atol=1e-4)
  np.testing.assert_allclose(miss_vals, 0.10, atol=1e-5)

