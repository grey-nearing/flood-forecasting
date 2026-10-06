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

"""Unit, regression, and canary tests for multimet.weather_viewer."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

import multimet.weather_viewer as wv
from multimet.weather_viewer.config import (
    DYNAMICAL_MODELS,
    FRAME_SIZE,
    from_stored_units,
    get_weather_source,
    list_weather_sources,
    MSLP_OFFSET_HPA,
    N_LAT,
    N_LON,
    parse_zarr_metadata_time_extent,
    RUN_METADATA_FILE,
    SUPPORTED_MODELS,
    SUPPORTED_VARIABLES,
    to_stored_units,
)
from multimet.weather_viewer.probe import WeatherViewerEngine
from multimet.weather_viewer.tiles import (
    bilinear_sample_grid,
     rate_file_steps,
    render_colorbar_lut_png,
)
from multimet.weather_viewer.wind import compute_wind_speed_and_direction


def _write_native_synced_run(
    root: Path,
    run_name: str = "20260929T000000Z",
    init_time: str = "2026-09-29T00:00:00",
    rain_mmh: float = 4.0,
) -> Path:
  """Writes a native-resolution (721 x 1440, 0.25 deg) synced run to root."""
  run_dir = root / "runs" / run_name
  run_dir.mkdir(parents=True, exist_ok=True)
  leads = [0, 3, 6, 9, 12]
  shape = (len(leads), N_LAT, N_LON)

  precip = np.zeros(shape, dtype=np.float16)
  precip[1:, :, :] = rain_mmh
  precip.tofile(run_dir / "ecmwf_aifs_precip.bin")
  precip.tofile(run_dir / "ecmwf_ifs_precip.bin")

  temp = np.full(shape, 18.5, dtype=np.float16)
  temp.tofile(run_dir / "ecmwf_aifs_temp.bin")
  temp.tofile(run_dir / "ecmwf_ifs_temp.bin")

  mslp = np.full(shape, 13.25, dtype=np.float16)  # 1013.25 - 1000.0 hPa
  mslp.tofile(run_dir / "ecmwf_aifs_mslp.bin")

  u10 = np.full(shape, 6.0, dtype=np.float16)
  v10 = np.full(shape, 8.0, dtype=np.float16)
  u10.tofile(run_dir / "ecmwf_aifs_u10.bin")
  v10.tofile(run_dir / "ecmwf_aifs_v10.bin")

  meta = {
      "status": "HEALTHY",
      "source": "dynamical.org",
      "last_updated_utc": "2026-09-29T03:00:00Z",
      "datasets": {
          "ecmwf_aifs_single_forecast": {
              "model": "ecmwf_aifs",
              "init_time": init_time,
              "lead_steps": len(leads),
              "lead_hours": leads,
              "streams": ["precip", "temp", "mslp", "u10", "v10"],
              "downloaded_utc": "2026-09-29T03:00:00Z",
              "mslp_offset_hpa": MSLP_OFFSET_HPA,
          },
          "ecmwf_ifs_ens_forecast_15_day_0_25_degree": {
              "model": "ecmwf_ifs",
              "init_time": init_time,
              "lead_steps": len(leads),
              "lead_hours": leads,
              "streams": ["precip", "temp"],
              "downloaded_utc": "2026-09-29T03:00:00Z",
          },
      },
  }
  (run_dir / RUN_METADATA_FILE).write_text(
      json.dumps(meta, indent=2), encoding="utf-8"
  )
  tmp_link = root / "current.tmp"
  if os.path.lexists(tmp_link):
    tmp_link.unlink()
  os.symlink(Path("runs") / run_name, tmp_link)
  os.replace(tmp_link, root / "current")
  return run_dir


@pytest.mark.unit
def test_zero_try_except_in_weather_viewer_package() -> None:
  """Verifies that zero try/except/finally blocks exist in multimet/weather_viewer/."""
  pkg_dir = Path(wv.__file__).resolve().parent
  py_files = sorted(pkg_dir.glob("*.py"))
  assert len(py_files) >= 6
  for py_file in py_files:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert not try_nodes, (
        f"Found forbidden try/except node(s) in {py_file.name}: "
        f"lines {[n.lineno for n in try_nodes]}"
    )


@pytest.mark.unit
def test_config_catalog_and_unit_conversions() -> None:
  """Tests model catalog, variable catalog, and physical unit conversions."""
  for model_key in (
      "ecmwf_ifs",
      "ecmwf_aifs",
      "noaa_gfs",
      "noaa_gefs",
      "noaa_hrrr",
      "graphcast",
  ):
    assert model_key in SUPPORTED_MODELS

  for dyn_key in (
      "ecmwf_ifs",
      "ecmwf_aifs",
      "noaa_gfs",
      "noaa_gefs",
      "noaa_hrrr",
  ):
    assert dyn_key in DYNAMICAL_MODELS

  for var_key in (
      "precipitation",
      "accumulated_precip",
      "temperature",
      "wind",
      "pressure",
  ):
    assert var_key in SUPPORTED_VARIABLES

  # mm/s -> mm/h
  precip_stored = to_stored_units("precip", np.array([2.0 / 3600.0, -0.5]))
  np.testing.assert_allclose(precip_stored.astype(float), [2.0, 0.0], rtol=1e-3)

  # Pa -> hPa - 1000 -> hPa
  mslp_stored = to_stored_units("mslp", np.array([101325.0]))
  restored = from_stored_units("mslp", mslp_stored)
  np.testing.assert_allclose(restored, [1013.25], rtol=1e-3)

  # Kelvin -> Celsius
  temp_k = to_stored_units("temp", np.array([273.15, 293.15]))
  np.testing.assert_allclose(temp_k.astype(float), [0.0, 20.0], atol=0.05)


@pytest.mark.unit
def test_weather_sources_and_zarr_metadata_extent() -> None:
  """Tests WeatherSource serialization and Zarr .zmetadata time parsing."""
  sources = list_weather_sources()
  assert len(sources) >= 8
  era5 = get_weather_source("era5")
  assert era5.id == "era5"

  sample_zmetadata = json.dumps({
      "metadata": {
          "time/.zarray": {"shape": [366]},
          "time/.zattrs": {"units": "days since 2020-01-01 00:00:00"},
      }
  })
  extent = parse_zarr_metadata_time_extent(sample_zmetadata)
  assert extent is not None
  assert extent["start_date"] == "2020-01-01"
  assert extent["end_date"] == "2020-12-31"
  assert extent["n_timesteps"] == 366


@pytest.mark.unit
def test_bilinear_interpolation_and_colorbar_lut() -> None:
  """Tests bilinear spatial interpolation on a 721x1440 native grid and LUT rendering."""
  lats = np.linspace(90.0, -90.0, N_LAT, dtype=np.float32)
  lons = np.linspace(-180.0, 179.75, N_LON, dtype=np.float32)
  plane = (lats[:, None] * 0.5 + lons[None, :] * 0.1).astype(np.float32)

  target_lats = np.array([45.125, 0.0, -33.875])
  target_lons = np.array([-86.875, 0.125, 151.125])
  sampled = bilinear_sample_grid(plane, target_lats, target_lons)
  expected = target_lats[:, None] * 0.5 + target_lons[None, :] * 0.1
  np.testing.assert_allclose(sampled, expected, atol=1e-3)

  for var_key in SUPPORTED_VARIABLES:
    lut_png = render_colorbar_lut_png(var_key, width=128, height=8)
    assert lut_png.startswith(b"\x89PNG\r\n\x1a\n")
    w, h = struct.unpack("!II", lut_png[16:24])
    assert (w, h) == (128, 8)


@pytest.mark.unit
def test_engine_requires_explicit_data_dir_and_rejects_unsynced_models(
    tmp_path: Path,
) -> None:
  """Verifies Rule 2 & Rule 3: explicit data_dir required and no synthetic fallbacks."""
  with pytest.raises(ValueError, match="explicit data_dir"):
    WeatherViewerEngine(None)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="explicit data_dir"):
    WeatherViewerEngine("")

  empty_engine = WeatherViewerEngine(tmp_path)
  info = empty_engine.get_model_info("ecmwf_aifs")
  assert info["data_source"] == "unavailable"
  assert info["real_variables"] == []

  with pytest.raises(FileNotFoundError):
    empty_engine.render_tile("ecmwf_aifs", "precipitation", 0, 2, 1, 1)

  with pytest.raises(FileNotFoundError):
    empty_engine.get_frame_index("ecmwf_aifs", "precipitation")

  with pytest.raises(FileNotFoundError):
    empty_engine.get_wind_vectors("ecmwf_aifs", step_idx=1)

  with pytest.raises(FileNotFoundError):
    empty_engine.probe_point(40.0, -86.0)

  with pytest.raises(FileNotFoundError):
    empty_engine.summarize_catchment(
        {"id": "b1", "geometry": {"type": "Point", "coordinates": [-86.0, 40.0]}}
    )


@pytest.mark.unit
def test_engine_tiles_frames_wind_probe_and_catchment_on_native_grid(
    tmp_path: Path,
) -> None:
  """Tests end-to-end WeatherViewerEngine operations on a native 721x1440 synced run."""
  _write_native_synced_run(tmp_path, rain_mmh=5.0)
  engine = WeatherViewerEngine(tmp_path)

  # 1. Model metadata
  aifs_info = engine.get_model_info("ecmwf_aifs")
  assert aifs_info["data_source"] == "archived_run"
  assert aifs_info["init_time"] == "2026-09-29T00:00:00Z"
  assert "pressure" in aifs_info["real_variables"]
  assert "wind" in aifs_info["real_variables"]

  # 2. Web Mercator XYZ tiles (bilinear and nearest)
  for var_key in ("precipitation", "accumulated_precip", "temperature", "pressure"):
    tile_png = engine.render_tile(
        "ecmwf_aifs", var_key, step_idx=2, z=2, x=1, y=1, bilinear=True
    )
    assert tile_png.startswith(b"\x89PNG\r\n\x1a\n")
    w, h = struct.unpack("!II", tile_png[16:24])
    assert (w, h) == (256, 256)

  # 3. Animation frame index & frame rendering
  idx = engine.get_frame_index("ecmwf_aifs", "precipitation")
  assert idx["frame_steps"] == [1, 2, 3, 4]
  frame_png = engine.render_frame("ecmwf_aifs", "precipitation", step_idx=2)
  assert frame_png.startswith(b"\x89PNG\r\n\x1a\n")
  fw, fh = struct.unpack("!II", frame_png[16:24])
  assert (fw, fh) == (FRAME_SIZE, FRAME_SIZE)

  # 4. Wind vectors (global & viewport bbox)
  wind_global = engine.get_wind_vectors("ecmwf_aifs", step_idx=2, subsample=2)
  assert wind_global["header"]["data_source"] == "archived_run"
  assert len(wind_global["u"]) == wind_global["header"]["nx"] * wind_global["header"]["ny"]
  assert pytest.approx(wind_global["u"][0], abs=0.05) == 6.0
  assert pytest.approx(wind_global["v"][0], abs=0.05) == 8.0

  wind_bbox = engine.get_wind_vectors(
      "ecmwf_aifs", step_idx=2, subsample=1, bbox=(-90.0, 35.0, -80.0, 45.0)
  )
  assert wind_bbox["header"]["nx"] == 11
  assert wind_bbox["header"]["ny"] == 11

  spd, direc = compute_wind_speed_and_direction(np.array([6.0]), np.array([8.0]))
  assert pytest.approx(float(spd[0]), abs=1e-4) == 10.0
  assert 0.0 <= float(direc[0]) < 360.0

  # 5. Point probe
  probe = engine.probe_point(40.42, -86.92)
  assert "ecmwf_aifs" in probe["models"]
  assert "ecmwf_ifs" in probe["models"]
  aifs_p = probe["models"]["ecmwf_aifs"]
  assert pytest.approx(aifs_p["precip_rate_mmh"][1], abs=0.05) == 5.0
  assert pytest.approx(aifs_p["accum_precip_mm"][2], abs=0.1) == 30.0
  assert pytest.approx(aifs_p["temp_c"][1], abs=0.1) == 18.5
  assert pytest.approx(aifs_p["wind_speed_mps"][1], abs=0.1) == 10.0
  assert pytest.approx(aifs_p["pressure_hpa"][1], abs=0.2) == 1013.2

  # 6. Catchment summary
  basin = {
      "id": "wabash_test",
      "properties": {"catchment_id": "wabash_test", "area_km2": 3200.0},
      "geometry": {
          "type": "Polygon",
          "coordinates": [[
              [-87.0, 40.0],
              [-86.0, 40.0],
              [-86.0, 41.0],
              [-87.0, 41.0],
              [-87.0, 40.0],
          ]],
      },
  }
  summary = engine.summarize_catchment(basin, step_idx=2, model_key="ecmwf_aifs")
  assert summary["catchment_id"] == "wabash_test"
  assert pytest.approx(summary["basin_mean_precip_mmh"], abs=0.05) == 5.0
  assert pytest.approx(summary["basin_mean_temp_c"], abs=0.1) == 18.5
  assert pytest.approx(summary["basin_accumulated_10d_mm"], abs=0.2) == 60.0


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


@pytest.mark.canary
def test_canary_dynamical_stac_catalog_reachable() -> None:
  """Live canary check verifying dynamical.org STAC catalog is reachable."""
  cat = wv.open_dynamical_catalog()
  assert cat is not None
  child = cat.get_child("noaa-gfs-forecast")
  assert child is not None
