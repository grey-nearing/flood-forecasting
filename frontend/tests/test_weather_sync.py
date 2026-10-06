"""Tests for the automatic forecast update (weather_sync) and animation frames."""

import datetime
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import threading
import unittest

import numpy as np

_ws_root = str(Path(__file__).resolve().parents[6])
if _ws_root not in sys.path:
  sys.path.insert(0, _ws_root)

from frontend import weather_engine as we
from frontend import weather_sync as ws


def _fake_forecast(init_times, lead_hours, n_lat=4, n_lon=8, members=False):
  """Small dynamical.org-like dataset (rates in mm/s, temperature in C, MSLP in Pa)."""
  import xarray as xr  # pylint: disable=g-import-not-at-top

  shape = (len(init_times), len(lead_hours), n_lat, n_lon)
  coords = {
      "init_time": np.array(init_times, dtype="datetime64[ns]"),
      "lead_time": np.array(lead_hours, dtype="timedelta64[h]").astype("timedelta64[ns]"),
      "latitude": np.linspace(90, -90, n_lat),
      "longitude": np.linspace(-180, 180, n_lon, endpoint=False),
  }
  dims = ("init_time", "lead_time", "latitude", "longitude")
  data = {
      "precipitation_surface": np.full(shape, 1.0 / 3600.0, np.float32),  # 1 mm/h
      "temperature_2m": np.full(shape, 12.5, np.float32),
      "pressure_reduced_to_mean_sea_level": np.full(shape, 101325.0, np.float32),
      "wind_u_10m": np.full(shape, 3.0, np.float32),
      "wind_v_10m": np.full(shape, -4.0, np.float32),
  }
  if members:
    coords["ensemble_member"] = np.arange(3)
    dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    data = {k: np.repeat(v[:, :, None], 3, axis=2) for k, v in data.items()}
  return xr.Dataset({k: (dims, v) for k, v in data.items()}, coords=coords)


class WeatherSyncTest(unittest.TestCase):

  def setUp(self):
    self.root = tempfile.mkdtemp(prefix="ekh_weather_sync_")

  def tearDown(self):
    shutil.rmtree(self.root, ignore_errors=True)

  def test_output_lead_hours_keeps_viewer_steps_up_to_240h(self):
    gfs = list(range(121)) + list(range(123, 385, 3))
    out = ws.output_lead_hours(gfs)
    self.assertEqual(out[:4], [0, 3, 6, 9])
    self.assertEqual(out[-1], 240)
    self.assertEqual(len(out), 81)
    self.assertEqual(ws.output_lead_hours([0, 6, 12, 246]), [0, 6, 12])

  def test_aggregate_rates_averages_hourly_rain_over_each_step(self):
    leads = [0, 1, 2, 3, 4, 5, 6]
    rates = np.array([np.nan, 1, 1, 1, 4, 4, 4], np.float32)[:, None, None]
    out = ws.aggregate_rates(rates, leads, [0, 3, 6])
    np.testing.assert_allclose(out[:, 0, 0], [0.0, 1.0, 4.0])
    # Mixed 1 h and 3 h rates: (1*2 + 3*1) / 3.
    out = ws.aggregate_rates(np.array([0, 2, 2, 1], np.float32)[:, None, None], [0, 1, 2, 5], [0, 2, 5])
    np.testing.assert_allclose(out[:, 0, 0], [0.0, 2.0, 1.0])

  def test_units(self):
    np.testing.assert_allclose(ws.to_stored_units("precip", [1 / 3600.0]).astype(float), [1.0], rtol=1e-3)
    self.assertAlmostEqual(float(ws.to_stored_units("mslp", [101325.0])[0]) + ws.MSLP_OFFSET_HPA, 1013.25, places=1)

  def test_sync_downloads_new_runs_only_and_swaps_atomically(self):
    t0 = np.datetime64("2026-09-29T00:00")
    t1 = np.datetime64("2026-09-29T06:00")
    hourly = list(range(0, 13))
    catalogs = {
        "noaa-gfs-forecast": _fake_forecast([t0], hourly),
        "ecmwf-aifs-single-forecast": _fake_forecast([t0], [0, 6, 12]),
    }
    open_ds = lambda _cat, dataset_id: catalogs[dataset_id]
    models = ["noaa_gfs", "ecmwf_aifs"]

    status = ws.sync_latest(root=self.root, models=models, catalog=object(), open_dataset=open_ds, log=lambda m: None)
    self.assertEqual(status["last_result"], "updated")
    run1 = ws.current_run_dir(self.root)
    meta = json.loads(Path(run1, ws.RUN_METADATA_FILE).read_text())
    gfs = meta["datasets"]["noaa_gfs_forecast"]
    self.assertEqual(gfs["model"], "noaa_gfs")
    self.assertEqual(gfs["lead_hours"], [0, 3, 6, 9, 12])
    precip = np.fromfile(Path(run1, "noaa_gfs_precip.bin"), np.float16).reshape(5, 4, 8)
    self.assertAlmostEqual(float(precip[2, 0, 0]), 1.0, places=2)
    self.assertEqual(float(precip[0, 0, 0]), 0.0)

    # Nothing new: no new run directory.
    status = ws.sync_latest(root=self.root, models=models, catalog=object(), open_dataset=open_ds, log=lambda m: None)
    self.assertEqual(status["last_result"], "up_to_date")
    self.assertEqual(ws.current_run_dir(self.root), run1)

    # New GFS run only: AIFS files are carried over, "current" moves.
    catalogs["noaa-gfs-forecast"] = _fake_forecast([t0, t1], hourly)
    status = ws.sync_latest(root=self.root, models=models, catalog=object(), open_dataset=open_ds, log=lambda m: None)
    self.assertEqual(status["updated_models"], ["noaa_gfs"])
    run2 = ws.current_run_dir(self.root)
    self.assertNotEqual(run2, run1)
    self.assertTrue(Path(run2, "ecmwf_aifs_precip.bin").exists())
    self.assertEqual(status["models"]["noaa_gfs"]["init_time"], "2026-09-29T06:00:00")
    self.assertEqual(status["models"]["ecmwf_aifs"]["init_time"], "2026-09-29T00:00:00")
    self.assertTrue(os.path.islink(os.path.join(self.root, "current")))

  def test_incomplete_run_keeps_previous(self):
    t0 = np.datetime64("2026-09-29T00:00")
    t1 = np.datetime64("2026-09-29T12:00")
    good = _fake_forecast([t0], [0, 6, 12])
    open_ds = lambda _cat, _id: good
    ws.sync_latest(root=self.root, models=["ecmwf_aifs"], catalog=object(), open_dataset=open_ds, log=lambda m: None)
    run1 = ws.current_run_dir(self.root)

    partial = _fake_forecast([t0, t1], [0, 6, 12])
    for var in partial.data_vars:
      partial[var].values[1, -1] = np.nan  # last lead of the new run not published yet
    open_ds = lambda _cat, _id: partial
    status = ws.sync_latest(root=self.root, models=["ecmwf_aifs"], catalog=object(), open_dataset=open_ds, log=lambda m: None)
    self.assertEqual(status["last_result"], "error")
    self.assertEqual(ws.current_run_dir(self.root), run1)

  def test_ensemble_uses_control_member(self):
    t0 = np.datetime64("2026-09-29T00:00")
    ds = _fake_forecast([t0], [0, 3, 6], members=True)
    ds["temperature_2m"].values[:, :, 1:] = 99.0  # other members differ
    ws.sync_latest(root=self.root, models=["ecmwf_ifs"], catalog=object(),
                   open_dataset=lambda _c, _i: ds, log=lambda m: None)
    temp = np.fromfile(Path(ws.current_run_dir(self.root), "ecmwf_ifs_temp.bin"), np.float16)
    self.assertTrue(np.allclose(temp.astype(float), 12.5))


class WeatherEngineReloadAndFramesTest(unittest.TestCase):
  """Hot reload of a new run and whole-world animation frames, on a fake 0.25 deg run."""

  def setUp(self):
    self.root = Path(tempfile.mkdtemp(prefix="ekh_weather_engine_"))
    self._saved = (list(we.CANDIDATE_DATA_DIRS), os.environ.get("EARTHKIT_WEATHER_WARM_CACHE"))
    os.environ["EARTHKIT_WEATHER_WARM_CACHE"] = "0"
    we.CANDIDATE_DATA_DIRS[:] = [self.root / "current"]
    we._INITIALIZED = False
    # A frame pre-render thread started by an earlier test stops once the
    # loaded streams change; wait for it so it cannot re-initialise mid-test.
    we.STREAM_INFO = {}
    for thread in threading.enumerate():
      if thread.name == "weather-cache-warmup":
        thread.join(timeout=60)
    we._INITIALIZED = False

  def tearDown(self):
    we.CANDIDATE_DATA_DIRS[:] = self._saved[0]
    if self._saved[1] is None:
      os.environ.pop("EARTHKIT_WEATHER_WARM_CACHE", None)
    else:
      os.environ["EARTHKIT_WEATHER_WARM_CACHE"] = self._saved[1]
    we._INITIALIZED = False  # Next user re-reads the real data.
    shutil.rmtree(self.root, ignore_errors=True)

  def _write_run(self, name, init_time, rain_mmh):
    run = self.root / "runs" / name
    run.mkdir(parents=True)
    leads = [0, 6, 12, 18, 24]
    plane = (we.N_LAT, we.N_LON)
    precip = np.zeros((len(leads),) + plane, np.float16)
    precip[1:, 300:400, 500:700] = rain_mmh
    precip.tofile(run / "ecmwf_aifs_precip.bin")
    np.full((len(leads),) + plane, 15.0, np.float16).tofile(run / "ecmwf_aifs_temp.bin")
    np.full((len(leads),) + plane, 13.0, np.float16).tofile(run / "ecmwf_aifs_mslp.bin")  # 1013 hPa
    meta = {"datasets": {"ecmwf_aifs_single_forecast": {
        "model": "ecmwf_aifs", "init_time": init_time, "lead_steps": len(leads),
        "lead_hours": leads, "streams": ["precip", "temp", "mslp"],
        "downloaded_utc": "2026-09-29T20:00:00Z"}}}
    (run / we.RUN_METADATA_FILE).write_text(json.dumps(meta))
    tmp = self.root / "current.tmp"
    if tmp.is_symlink():
      tmp.unlink()
    os.symlink(Path("runs") / name, tmp)
    os.replace(tmp, self.root / "current")

  def test_frames_skip_duplicate_steps_and_reload_picks_up_new_run(self):
    self._write_run("r1", "2026-09-29T00:00:00", 3.0)
    info = we.get_model_data_info("ecmwf_aifs")
    self.assertEqual(info["init_time"], "2026-09-29T00:00:00Z")
    self.assertEqual(info["max_lead_hours"], 24)
    self.assertIn("pressure", info["real_variables"])
    self.assertEqual(info["downloaded_utc"], "2026-09-29T20:00:00Z")

    index = we.get_frame_index("ecmwf_aifs", "precipitation")
    self.assertEqual(index["frame_steps"], [2, 4, 6, 8])  # 6-hourly model: one frame per 6 h
    self.assertEqual(index["step_frames"][:9], [2, 2, 2, 4, 4, 6, 6, 8, 8])
    self.assertIsNone(index["step_frames"][9])  # beyond the end of the run
    temp_index = we.get_frame_index("ecmwf_aifs", "temperature")
    self.assertEqual(temp_index["frame_steps"], [0, 2, 4, 6, 8])

    png = we.generate_weather_frame("ecmwf_aifs", "precipitation", 3)
    self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
    width, height, depth, color_type = struct.unpack("!IIBB", png[16:26])
    self.assertEqual((width, height, depth, color_type), (we.FRAME_SIZE, we.FRAME_SIZE, 8, 3))
    self.assertIs(we.generate_weather_frame("ecmwf_aifs", "precipitation", 4), png)  # cached, same frame
    self.assertTrue(we.generate_weather_frame("ecmwf_aifs", "pressure", 2).startswith(b"\x89PNG"))
    point = we._point_value("ecmwf_aifs", "pressure", 6, 0.0, 0.0)
    self.assertAlmostEqual(point, 1013.0, places=1)

    self.assertFalse(we.reload_if_changed())
    self._write_run("r2", "2026-09-29T12:00:00", 8.0)
    self.assertTrue(we.reload_if_changed())
    self.assertEqual(we.get_model_data_info("ecmwf_aifs")["init_time"], "2026-09-29T12:00:00Z")
    self.assertIsNot(we.generate_weather_frame("ecmwf_aifs", "precipitation", 4), png)

  def test_frame_index_rejects_unknown_layers(self):
    with self.assertRaises(ValueError):
      we.get_frame_index("ecmwf_aifs", "radar")
    with self.assertRaises(ValueError):
      we.get_frame_index("nope", "precipitation")


if __name__ == "__main__":
  unittest.main()
