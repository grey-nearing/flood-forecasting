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

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import xarray as xr

from multimet.weather_fetcher import cli
from multimet.weather_fetcher.config import (
    MSLP_OFFSET_HPA,
    N_LAT,
    N_LON,
    output_lead_hours,
    RUN_METADATA_FILE,
    to_stored_units,
)
from multimet.weather_fetcher.fetcher import WeatherDataFetcher
from multimet.weather_fetcher.sync import (
    aggregate_rates,
    current_run_dir,
    download_model_run,
    DYNAMICAL_MODELS,
    IncompleteRunError,
    swap_current_symlink,
    sync_all_models,
    WeatherSynchronizer,
)


def _build_native_forecast_dataset(
    init_times: Sequence[np.datetime64],
    lead_hours: Sequence[int],
    members: bool = False,
) -> xr.Dataset:
  """Constructs a native 721 x 1440 (0.25 deg) dynamical.org-schema Dataset."""
  shape = (len(init_times), len(lead_hours), N_LAT, N_LON)
  coords = {
      "init_time": np.array(init_times, dtype="datetime64[ns]"),
      "lead_time": np.array(lead_hours, dtype="timedelta64[h]").astype(
          "timedelta64[ns]"
      ),
      "latitude": np.linspace(90.0, -90.0, N_LAT, dtype=np.float32),
      "longitude": np.linspace(
          -180.0, 180.0, N_LON, endpoint=False, dtype=np.float32
      ),
  }
  dims = ("init_time", "lead_time", "latitude", "longitude")
  data = {
      "precipitation_surface": np.full(shape, 1.0 / 3600.0, np.float32),
      "temperature_2m": np.full(shape, 12.5, np.float32),
      "pressure_reduced_to_mean_sea_level": np.full(
          shape, 101325.0, np.float32
      ),
      "wind_u_10m": np.full(shape, 3.0, np.float32),
      "wind_v_10m": np.full(shape, -4.0, np.float32),
  }
  if members:
    coords["ensemble_member"] = np.arange(3)
    dims = (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    data = {k: np.repeat(v[:, :, None], 3, axis=2) for k, v in data.items()}
  return xr.Dataset({k: (dims, v) for k, v in data.items()}, coords=coords)


@pytest.mark.unit
def test_output_lead_hours_and_rate_aggregation() -> None:
  """Tests 3-hourly lead filtering up to 240h and sub-step rate aggregation."""
  gfs = list(range(121)) + list(range(123, 385, 3))
  out = output_lead_hours(gfs)
  assert out[:4] == [0, 3, 6, 9]
  assert out[-1] == 240
  assert len(out) == 81
  assert output_lead_hours([0, 6, 12, 246]) == [0, 6, 12]

  leads = [0, 1, 2, 3, 4, 5, 6]
  rates = np.array([np.nan, 1, 1, 1, 4, 4, 4], np.float32)[:, None, None]
  agg = aggregate_rates(rates, leads, [0, 3, 6])
  np.testing.assert_allclose(agg[:, 0, 0], [0.0, 1.0, 4.0])

  mixed = aggregate_rates(
      np.array([0, 2, 2, 1], np.float32)[:, None, None],
      [0, 1, 2, 5],
      [0, 2, 5],
  )
  np.testing.assert_allclose(mixed[:, 0, 0], [0.0, 2.0, 1.0])


@pytest.mark.unit
def test_unit_conversions_in_sync() -> None:
  """Tests precipitation and MSLP float16 unit conversions."""
  np.testing.assert_allclose(
      to_stored_units("precip", [1.0 / 3600.0]).astype(float),
      [1.0],
      rtol=1e-3,
  )
  stored_mslp = float(to_stored_units("mslp", [101325.0])[0])
  assert pytest.approx(stored_mslp + MSLP_OFFSET_HPA, abs=0.1) == 1013.25


@pytest.mark.unit
def test_sync_atomic_swap_incremental_update_and_hot_reload(
    tmp_path: Path,
) -> None:
  """Tests native-grid sync, incremental run carry-over, atomic symlink swap, and engine reload."""
  t0 = np.datetime64("2026-09-29T00:00")
  t1 = np.datetime64("2026-09-29T06:00")
  catalogs = {
      "noaa-gfs-forecast": _build_native_forecast_dataset([t0], [0, 1, 2, 3, 6]),
      "ecmwf-aifs-single-forecast": _build_native_forecast_dataset(
          [t0], [0, 6]
      ),
  }
  open_ds = lambda _cat, dataset_id: catalogs[dataset_id]
  models = ["noaa_gfs", "ecmwf_aifs"]

  synchronizer = WeatherSynchronizer(data_dir=tmp_path, models=models)
  status = synchronizer.sync_all(
      catalog=object(), open_dataset=open_ds, log=lambda _: None
  )
  assert status["last_result"] == "updated"

  run1 = current_run_dir(tmp_path)
  assert run1 is not None
  meta = json.loads((run1 / RUN_METADATA_FILE).read_text(encoding="utf-8"))
  gfs = meta["datasets"]["noaa_gfs_forecast"]
  assert gfs["model"] == "noaa_gfs"
  assert gfs["lead_hours"] == [0, 3, 6]

  precip = np.fromfile(run1 / "noaa_gfs_precip.bin", np.float16).reshape(
      3, N_LAT, N_LON
  )
  assert pytest.approx(float(precip[1, 0, 0]), abs=0.02) == 1.0
  assert float(precip[0, 0, 0]) == 0.0

  fetcher = WeatherDataFetcher(tmp_path)
  assert fetcher.get_model_info("ecmwf_aifs")["init_time"] == "2026-09-29T00:00:00Z"
  assert not fetcher.reload_if_changed()

  # Second check with no upstream changes -> up_to_date
  status2 = synchronizer.sync_all(
      catalog=object(), open_dataset=open_ds, log=lambda _: None
  )
  assert status2["last_result"] == "up_to_date"
  assert current_run_dir(tmp_path) == run1

  # New GFS run published -> updates GFS, carries over AIFS, swaps current symlink
  catalogs["noaa-gfs-forecast"] = _build_native_forecast_dataset(
      [t0, t1], [0, 1, 2, 3, 6]
  )
  status3 = synchronizer.sync_all(
      catalog=object(), open_dataset=open_ds, log=lambda _: None
  )
  assert status3["updated_models"] == ["noaa_gfs"]
  run2 = current_run_dir(tmp_path)
  assert run2 is not None and run2 != run1
  assert (run2 / "ecmwf_aifs_precip.bin").exists()
  assert os.path.islink(tmp_path / "current")

  assert fetcher.reload_if_changed()
  assert fetcher.get_model_info("noaa_gfs")["init_time"] == "2026-09-29T06:00:00Z"


@pytest.mark.unit
def test_incomplete_run_rejected_and_previous_preserved(
    tmp_path: Path,
) -> None:
  """Verifies that incomplete final lead planes raise IncompleteRunError or preserve the previous run."""
  t0 = np.datetime64("2026-09-29T00:00")
  t1 = np.datetime64("2026-09-29T12:00")
  good = _build_native_forecast_dataset([t0], [0, 6])
  sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_aifs"],
      catalog=object(),
      open_dataset=lambda _c, _i: good,
      log=lambda _: None,
  )
  run1 = current_run_dir(tmp_path)

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
      catalog=object(),
      open_dataset=lambda _c, _i: partial,
      log=lambda _: None,
  )
  assert status["last_result"] == "error"
  assert current_run_dir(tmp_path) == run1


@pytest.mark.unit
def test_ensemble_selects_control_member_and_cli_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verifies ensemble control member (member 0) extraction and CLI --status."""
  t0 = np.datetime64("2026-09-29T00:00")
  ds = _build_native_forecast_dataset([t0], [0, 3, 6], members=True)
  ds["temperature_2m"].values[:, :, 1:] = 99.0

  sync_all_models(
      data_dir=tmp_path,
      models=["ecmwf_ifs"],
      catalog=object(),
      open_dataset=lambda _c, _i: ds,
      log=lambda _: None,
  )
  run_dir = current_run_dir(tmp_path)
  assert run_dir is not None
  temp = np.fromfile(run_dir / "ecmwf_ifs_temp.bin", np.float16)
  assert np.allclose(temp.astype(float), 12.5)

  rc = cli.main(["--data-dir", str(tmp_path), "--status"])
  assert rc == 0
  captured = json.loads(capsys.readouterr().out)
  assert captured["last_result"] == "updated"
  assert "ecmwf_ifs" in captured["models"]


@pytest.mark.unit
def test_swap_current_symlink_windows_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies swap_current_symlink passes target_is_directory=True and unlinks existing link on Windows."""
  (tmp_path / "runs" / "run1").mkdir(parents=True)
  (tmp_path / "runs" / "run2").mkdir(parents=True)

  calls: list[bool] = []
  real_symlink = os.symlink
  real_replace = os.replace

  def tracked_symlink(
      src: os.PathLike[str] | str,
      dst: os.PathLike[str] | str,
      target_is_directory: bool = False,
      *,
      dir_fd: int | None = None,
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

  monkeypatch.setattr(
      "multimet.weather_fetcher.sync.Path", type(tmp_path)
  )
  monkeypatch.setattr(os, "symlink", tracked_symlink)
  monkeypatch.setattr(os, "replace", strict_windows_replace)
  monkeypatch.setattr(os, "name", "nt")

  link1 = swap_current_symlink(tmp_path, "run1")
  assert link1.resolve() == (tmp_path / "runs" / "run1").resolve()
  assert calls == [True]

  link2 = swap_current_symlink(tmp_path, "run2")
  assert link2.resolve() == (tmp_path / "runs" / "run2").resolve()
  assert calls == [True, True]

