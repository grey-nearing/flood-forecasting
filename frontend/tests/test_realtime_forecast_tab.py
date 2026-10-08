"""Tests for the Real-Time Forecasting & Data Assimilation Tab and Service."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

import sys

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[2])
if _WORKSPACE_ROOT not in sys.path:
  sys.path.insert(0, _WORKSPACE_ROOT)

# Importing the tests package first points profiles at a temp folder (never real accounts).
from frontend import tests as _isolated_profiles  # pylint: disable=unused-import,g-import-not-at-top
from frontend import profile_manager  # pylint: disable=g-import-not-at-top
from frontend import realtime_forecast_service  # pylint: disable=g-import-not-at-top
from frontend.server import EarthkitHydroHandler  # pylint: disable=g-import-not-at-top


TEST_USER = "test_fc_realtime_user"
TEST_BASIN = "camels_01022500"
TEST_BASIN_2 = "camels_01031500"

TEST_WATERSHED_FEATURE = {
    "type": "Feature",
    "id": TEST_BASIN,
    "properties": {
        "catchment_id": TEST_BASIN,
        "name": "Narraguagus River at Cherryfield, ME",
        "area_km2": 573.6,
        "source": "delineated",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-68.05, 44.60],
            [-67.85, 44.60],
            [-67.85, 44.80],
            [-68.05, 44.80],
            [-68.05, 44.60],
        ]],
    },
}

TEST_WATERSHED_FEATURE_2 = {
    "type": "Feature",
    "id": TEST_BASIN_2,
    "properties": {
        "catchment_id": TEST_BASIN_2,
        "name": "Piscataquis River near Dover-Foxcroft, ME",
        "area_km2": 771.8,
        "source": "delineated",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-69.30, 45.10],
            [-69.10, 45.10],
            [-69.10, 45.25],
            [-69.30, 45.25],
            [-69.30, 45.10],
        ]],
    },
}


def _populate_synthetic_realtime_zarr_stores(
    profile_dir: Path,
    basin_id: str | list[str] = TEST_BASIN,
    issue_date_str: str = "2026-03-30",
    spinup_days: int = 365,
    forecast_horizon: int = 10,
    include_extra_forecast_streams: bool = True,
) -> None:
  """Populates real-time Zarr stores (including NaN latency gaps and multiple streams)."""
  basin_ids = [basin_id] if isinstance(basin_id, str) else list(basin_id)
  n_basins = len(basin_ids)
  t0 = pd.Timestamp(issue_date_str)
  start_date = t0 - pd.Timedelta(days=spinup_days)
  end_date = t0 + pd.Timedelta(days=forecast_horizon - 1)
  nowcast_dates = pd.date_range(start_date, t0 - pd.Timedelta(days=1), freq="D")
  all_dates = pd.date_range(start_date, end_date, freq="D")

  rt_dyn_dir = profile_dir / "realtime" / "dynamics"
  rt_dyn_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(42)

  # 1. CPC (nowcast precip with 2-day real-time latency -> trailing 2 days are NaN)
  cpc_precip = rng.uniform(
      0.0, 18.0, size=(n_basins, len(nowcast_dates))
  ).astype(np.float32)
  cpc_precip[:, -2:] = np.nan
  cpc_ds = xr.Dataset(
      {
          "cpc_precip": (("basin", "date"), cpc_precip),
          "cpc_precip_missing_fraction": (
              ("basin", "date"),
              np.where(np.isnan(cpc_precip), 1.0, 0.0).astype(np.float32),
          ),
      },
      coords={"basin": basin_ids, "date": nowcast_dates.values},
  )
  cpc_path = rt_dyn_dir / "CPC" / "timeseries.zarr"
  if cpc_path.exists():
    shutil.rmtree(cpc_path)
  cpc_path.parent.mkdir(parents=True, exist_ok=True)
  cpc_ds.to_zarr(cpc_path, mode="w", zarr_format=2, consolidated=True)

  # 2. IMERG (nowcast precip with 1-day real-time latency -> last day is NaN)
  imerg_precip = rng.uniform(
      0.0, 22.0, size=(n_basins, len(nowcast_dates))
  ).astype(np.float32)
  imerg_precip[:, -1] = np.nan
  imerg_ds = xr.Dataset(
      {
          "imerg_precip": (("basin", "date"), imerg_precip),
          "imerg_precip_missing_fraction": (
              ("basin", "date"),
              np.where(np.isnan(imerg_precip), 1.0, 0.0).astype(np.float32),
          ),
      },
      coords={"basin": basin_ids, "date": nowcast_dates.values},
  )
  imerg_path = rt_dyn_dir / "IMERG" / "timeseries.zarr"
  if imerg_path.exists():
    shutil.rmtree(imerg_path)
  imerg_path.parent.mkdir(parents=True, exist_ok=True)
  imerg_ds.to_zarr(imerg_path, mode="w", zarr_format=2, consolidated=True)

  # 3. HRES (hindcast lead_time=1d across spinup + 10-day forecast on t0)
  lead_times = pd.to_timedelta(np.arange(1, forecast_horizon + 1), unit="D")
  hres_vars = [
      "hres_temperature_2m",
      "hres_total_precipitation",
      "hres_surface_net_solar_radiation",
      "hres_surface_net_thermal_radiation",
      "hres_surface_pressure",
      "hres_specific_humidity",
      "hres_u_component_of_wind_10m",
      "hres_v_component_of_wind_10m",
  ]
  hres_data = {}
  t0_idx = int(np.where(all_dates == t0)[0][0])
  for v in hres_vars:
    arr = np.full(
        (n_basins, len(all_dates), forecast_horizon), np.nan, dtype=np.float32
    )
    # 1-day lead time populated for spin-up through t0
    if "temperature" in v:
      arr[:, : t0_idx + 1, 0] = rng.uniform(
          2.0, 16.0, size=(n_basins, t0_idx + 1)
      )
      arr[:, t0_idx, :] = np.linspace(8.0, 14.5, forecast_horizon)
    elif "precipitation" in v:
      arr[:, : t0_idx + 1, 0] = rng.uniform(
          0.0, 15.0, size=(n_basins, t0_idx + 1)
      )
      arr[:, t0_idx, :] = np.array(
          [3.5, 12.0, 25.0, 8.0, 1.5, 0.0, 4.0, 9.5, 2.0, 0.5],
          dtype=np.float32,
      )
    elif "pressure" in v:
      arr[:, : t0_idx + 1, 0] = 101325.0
      arr[:, t0_idx, :] = 101200.0
    else:
      arr[:, : t0_idx + 1, 0] = rng.uniform(
          0.5, 5.0, size=(n_basins, t0_idx + 1)
      )
      arr[:, t0_idx, :] = rng.uniform(0.5, 5.0, size=(n_basins, forecast_horizon))
    hres_data[v] = (("basin", "date", "lead_time"), arr)

  hres_ds = xr.Dataset(
      hres_data,
      coords={
          "basin": basin_ids,
          "date": all_dates.values,
          "lead_time": lead_times.values,
      },
  )
  hres_path = rt_dyn_dir / "HRES" / "timeseries.zarr"
  if hres_path.exists():
    shutil.rmtree(hres_path)
  hres_path.parent.mkdir(parents=True, exist_ok=True)
  hres_ds.to_zarr(hres_path, mode="w", zarr_format=2, consolidated=True)

  # 4. Additional dynamical.org forecast streams (ECMWF_AIFS and NOAA_GFS)
  if include_extra_forecast_streams:
    for prod_name, prefix in [("ECMWF_AIFS", "aifs"), ("NOAA_GFS", "gfs")]:
      p_arr = np.full(
          (n_basins, len(all_dates), forecast_horizon), np.nan, dtype=np.float32
      )
      t_arr = np.full(
          (n_basins, len(all_dates), forecast_horizon), np.nan, dtype=np.float32
      )
      p_arr[:, t0_idx, :] = np.linspace(1.0, 18.0, forecast_horizon)
      t_arr[:, t0_idx, :] = np.linspace(6.0, 15.0, forecast_horizon)
      # Put a deliberate NaN at lead day 9 to verify forecast NaNs are preserved
      p_arr[:, t0_idx, 8] = np.nan
      extra_ds = xr.Dataset(
          {
              f"{prefix}_total_precipitation": (
                  ("basin", "date", "lead_time"),
                  p_arr,
              ),
              f"{prefix}_temperature_2m": (
                  ("basin", "date", "lead_time"),
                  t_arr,
              ),
          },
          coords={
              "basin": basin_ids,
              "date": all_dates.values,
              "lead_time": lead_times.values,
          },
      )
      extra_path = rt_dyn_dir / prod_name / "timeseries.zarr"
      if extra_path.exists():
        shutil.rmtree(extra_path)
      extra_path.parent.mkdir(parents=True, exist_ok=True)
      extra_ds.to_zarr(extra_path, mode="w", zarr_format=2, consolidated=True)


class RealtimeForecastTabTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EarthkitHydroHandler)
    cls.port = cls.server.server_port
    cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.thread.start()
    time.sleep(0.1)

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def setUp(self):
    self.pm = profile_manager.get_profile_manager()
    self.profile_dir = self.pm.get_profile_dir(TEST_USER)
    if self.profile_dir.exists():
      shutil.rmtree(self.profile_dir)
    self.profile_dir = self.pm.get_profile_dir(TEST_USER)
    self.pm.save_watersheds(
        [TEST_WATERSHED_FEATURE, TEST_WATERSHED_FEATURE_2], username=TEST_USER
    )

  def tearDown(self):
    if self.profile_dir.exists():
      shutil.rmtree(self.profile_dir)

  def _request(self, method: str, path: str, body: dict | None = None):
    url = f"http://127.0.0.1:{self.port}{path}"
    headers = {"X-User-Profile": TEST_USER}
    data = None
    if body is not None:
      data = json.dumps(body).encode("utf-8")
      headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
      with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)
    except urllib.error.HTTPError as err:
      raw = err.read().decode("utf-8")
      return err.code, json.loads(raw)

  def test_ui_html_contains_simplified_forecast_controls_and_preserves_legacy_ids(
      self,
  ):
    """Verifies #tab-forecasting has the multi-catchment and product/variable controls and keeps legacy IDs."""
    url = f"http://127.0.0.1:{self.port}/"
    with urllib.request.urlopen(url) as resp:
      html = resp.read().decode("utf-8")

    required_ids = [
        "tab-forecasting",
        "fcCatchmentSelectorList",
        "fcSelectAllBasinsBtn",
        "fcSelectNoneBasinsBtn",
        "fcSelectedCatchmentCountBadge",
        "fcInputProductsChecklist",
        "fcModeColdBtn",
        "fcModeHotBtn",
        "fcIssueDateText",
        "fcSavedStateDateText",
        "fcExpectedStateDateText",
        "fcStateValidationBanner",
        "fcFetchRealtimeBtn",
        "fcRunModelBtn",
        "fcRunBlockedReason",
        "fcNowcastProductSelect",
        "fcForecastProductSelect",
        "fcForecastVariableSelect",
        "fcLeadTimeSummaryTable",
        "fcHydroCatchmentTabs",
        "whatIfHyetographCanvas",
        "masterHydrographCanvas",
        "fcHydroPeakBadge",
        # Legacy IDs preserved for backwards compatibility
        "fetchForecastBtn",
        "executionModeSelect",
        "modelCheckpointSelect",
    ]
    for dom_id in required_ids:
      self.assertIn(f'id="{dom_id}"', html, f"Missing DOM ID: {dom_id}")

  def test_multi_stream_discovery_and_strict_no_fake_data_nan_preservation(self):
    """Verifies IMERG/CPC/HRES/ECMWF_AIFS/NOAA_GFS discovery, all-variable extraction, and strict NaN preservation."""
    _populate_synthetic_realtime_zarr_stores(
        self.profile_dir,
        basin_id=TEST_BASIN,
        issue_date_str="2026-03-30",
    )

    status_code, payload = self._request(
        "GET", f"/api/forecast/status?catchment_id={TEST_BASIN}"
    )
    self.assertEqual(status_code, 200)
    self.assertEqual(payload["status"], "success")
    self.assertEqual(payload["issue_date"], "2026-03-30")
    self.assertEqual(payload["expected_state_date"], "2026-03-29")

    # Verify nowcast streams include CPC, IMERG, and 1D hindcasts
    nowcast_products = payload["nowcast_products"]
    self.assertIn("CPC", nowcast_products)
    self.assertIn("IMERG", nowcast_products)
    self.assertIn("HRES_1D", nowcast_products)

    # Verify trailing latency NaNs in CPC (last 2 days) and IMERG (last 1 day) are None (never filled!)
    cpc_series = nowcast_products["CPC"]["series"]
    self.assertEqual(len(cpc_series), 14)
    self.assertEqual(cpc_series[-1]["date"], "2026-03-29")
    self.assertIsNone(cpc_series[-1]["precip_mm"])
    self.assertIsNone(cpc_series[-2]["precip_mm"])
    self.assertIsNotNone(cpc_series[-3]["precip_mm"])

    imerg_series = nowcast_products["IMERG"]["series"]
    self.assertEqual(len(imerg_series), 14)
    self.assertEqual(imerg_series[-1]["date"], "2026-03-29")
    self.assertIsNone(imerg_series[-1]["precip_mm"])
    self.assertIsNotNone(imerg_series[-2]["precip_mm"])

    # Verify forecast streams include HRES, ECMWF_AIFS, and NOAA_GFS
    forecast_products = payload["forecast_products"]
    self.assertIn("HRES", forecast_products)
    self.assertIn("ECMWF_AIFS", forecast_products)
    self.assertIn("NOAA_GFS", forecast_products)

    # Verify default variable mode is precip_and_temp and all HRES variables are exposed
    hres_prod = forecast_products["HRES"]
    self.assertEqual(hres_prod["default_variable_mode"], "precip_and_temp")
    self.assertTrue(hres_prod["has_precip"])
    self.assertTrue(hres_prod["has_temp"])
    hres_var_names = [v["name"] for v in hres_prod["variables"]]
    self.assertIn("hres_total_precipitation", hres_var_names)
    self.assertIn("hres_temperature_2m", hres_var_names)
    self.assertIn("hres_surface_net_solar_radiation", hres_var_names)
    self.assertIn("hres_surface_pressure", hres_var_names)
    self.assertIn("hres_u_component_of_wind_10m", hres_var_names)

    # Verify Guy Shalev / PR #333 lead time mapping: lead_time=1d -> 2026-03-30 (t0, issue date)
    hres_fc = hres_prod["series"]
    self.assertEqual(len(hres_fc), 10)
    self.assertEqual(hres_fc[0]["lead_time_days"], 1)
    self.assertEqual(hres_fc[0]["date"], "2026-03-30")
    self.assertTrue(hres_fc[0]["is_issue_date"])
    self.assertAlmostEqual(hres_fc[0]["precip_mm"], 3.5, places=3)
    self.assertAlmostEqual(hres_fc[0]["temperature_c"], 8.0, places=3)
    self.assertAlmostEqual(
        hres_fc[0]["values"]["hres_surface_pressure"], 101200.0, places=1
    )
    self.assertEqual(hres_fc[-1]["lead_time_days"], 10)
    self.assertEqual(hres_fc[-1]["date"], "2026-04-08")
    self.assertFalse(hres_fc[-1]["is_issue_date"])
    self.assertAlmostEqual(hres_fc[-1]["temperature_c"], 14.5, places=3)

    # Verify 1D spin-up series on the forecast product is also populated with all variables
    self.assertEqual(len(hres_prod["hindcast_series"]), 14)
    self.assertIn(
        "hres_surface_net_solar_radiation",
        hres_prod["hindcast_series"][-1]["values"],
    )

    # Verify deliberate NaN at index 8 in ECMWF_AIFS is preserved as None
    aifs_fc = forecast_products["ECMWF_AIFS"]["series"]
    self.assertIsNone(aifs_fc[8]["precip_mm"])
    self.assertIsNotNone(aifs_fc[8]["temperature_c"])

  def test_state_date_validation_and_cold_to_hot_start_model_execution(self):
    """Verifies Hot-Start blocking when state is missing/stale, Cold-Start state saving, and Hot-Start parity."""
    _populate_synthetic_realtime_zarr_stores(
        self.profile_dir,
        basin_id=TEST_BASIN,
        issue_date_str="2026-03-30",
    )

    # 1. With no saved state file, Hot-Start must be blocked
    code, status_payload = self._request(
        "GET", f"/api/forecast/status?catchment_id={TEST_BASIN}"
    )
    self.assertEqual(code, 200)
    self.assertFalse(status_payload["saved_state"]["exists"])
    self.assertFalse(status_payload["is_state_from_previous_day"])
    self.assertFalse(status_payload["can_run_hotstart"])
    self.assertTrue(status_payload["can_run_coldstart"])

    code, blocked_res = self._request(
        "POST",
        "/api/forecast/run-model",
        {"catchment_id": TEST_BASIN, "mode": "hotstart"},
    )
    self.assertEqual(code, 400)
    self.assertIn(
        "cannot run model in hot-start mode", blocked_res["message"].lower()
    )

    # 2. If a saved state file exists with a stale date (2026-03-28 != 2026-03-29), Hot-Start must STILL be blocked
    state_path = realtime_forecast_service.get_state_file_path(
        TEST_USER, TEST_BASIN
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        state_path,
        h_n=np.zeros((1, 1, 64), dtype=np.float32),
        c_n=np.zeros((1, 1, 64), dtype=np.float32),
        date=np.array("2026-03-28", dtype="U10"),
    )
    code, stale_status = self._request(
        "GET", f"/api/forecast/status?catchment_id={TEST_BASIN}"
    )
    self.assertEqual(code, 200)
    self.assertTrue(stale_status["saved_state"]["exists"])
    self.assertEqual(stale_status["saved_state"]["state_date"], "2026-03-28")
    self.assertFalse(stale_status["is_state_from_previous_day"])
    self.assertFalse(stale_status["can_run_hotstart"])

    code, stale_run_res = self._request(
        "POST",
        "/api/forecast/run-model",
        {"catchment_id": TEST_BASIN, "mode": "hotstart"},
    )
    self.assertEqual(code, 400)
    self.assertIn("2026-03-28", stale_run_res["message"])
    self.assertIn("2026-03-29", stale_run_res["message"])

    # 3. Run Cold-Start model: should succeed, return 10-day forecast, and save state stamped 2026-03-29
    code, cold_res = self._request(
        "POST",
        "/api/forecast/run-model",
        {"catchment_id": TEST_BASIN, "mode": "coldstart"},
    )
    self.assertEqual(code, 200, f"Cold-start failed: {cold_res}")
    self.assertEqual(cold_res["status"], "success")
    self.assertEqual(cold_res["mode"], "coldstart")
    self.assertEqual(cold_res["issue_date"], "2026-03-30")
    self.assertEqual(len(cold_res["forecast"]), cold_res["lead_time_days"])
    self.assertGreater(len(cold_res["hindcast_tail"]), 0)
    self.assertEqual(
        cold_res["saved_state"]["state_date"], "2026-03-29"
    )
    self.assertTrue(
        cold_res["post_run_status"]["is_state_from_previous_day"]
    )
    self.assertTrue(cold_res["post_run_status"]["can_run_hotstart"])

    # 4. Now run Hot-Start model: should succeed and match Cold-Start forecast!
    code, hot_res = self._request(
        "POST",
        "/api/forecast/run-model",
        {"catchment_id": TEST_BASIN, "mode": "hotstart"},
    )
    self.assertEqual(code, 200, f"Hot-start failed: {hot_res}")
    self.assertEqual(hot_res["status"], "success")
    self.assertEqual(hot_res["mode"], "hotstart")
    self.assertEqual(len(hot_res["forecast"]), hot_res["lead_time_days"])

    cold_q = [pt["discharge_cms"] for pt in cold_res["forecast"]]
    hot_q = [pt["discharge_cms"] for pt in hot_res["forecast"]]
    np.testing.assert_allclose(cold_q, hot_q, rtol=1e-4, atol=1e-4)

  def test_multi_catchment_selection_and_strict_yesterday_state_gating(self):
    """Verifies multi-catchment batch status, fetch, and forecast require EVERY selected catchment to have yesterday's state for Hot-Start."""
    _populate_synthetic_realtime_zarr_stores(
        self.profile_dir,
        basin_id=[TEST_BASIN, TEST_BASIN_2],
        issue_date_str="2026-03-30",
    )

    # Give TEST_BASIN a valid yesterday state (2026-03-29), but give TEST_BASIN_2 an older state (2026-03-27)
    state_path_1 = realtime_forecast_service.get_state_file_path(
        TEST_USER, TEST_BASIN
    )
    state_path_2 = realtime_forecast_service.get_state_file_path(
        TEST_USER, TEST_BASIN_2
    )
    state_path_1.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        state_path_1,
        h_n=np.zeros((1, 1, 64), dtype=np.float32),
        c_n=np.zeros((1, 1, 64), dtype=np.float32),
        date=np.array("2026-03-29", dtype="U10"),
    )
    np.savez(
        state_path_2,
        h_n=np.zeros((1, 1, 64), dtype=np.float32),
        c_n=np.zeros((1, 1, 64), dtype=np.float32),
        date=np.array("2026-03-27", dtype="U10"),
    )

    # Status for both catchments must report can_run_hotstart=False because TEST_BASIN_2 state is from 2026-03-27
    code, multi_status = self._request(
        "GET",
        f"/api/forecast/status?catchment_id={TEST_BASIN}&catchment_ids={TEST_BASIN},{TEST_BASIN_2}",
    )
    self.assertEqual(code, 200)
    self.assertFalse(multi_status["all_selected_have_previous_day_state"])
    self.assertFalse(multi_status["can_run_hotstart"])
    self.assertFalse(multi_status["can_fetch_hotstart"])
    missing_ids = [
        item["catchment_id"]
        for item in multi_status["missing_or_stale_state_catchments"]
    ]
    self.assertIn(TEST_BASIN_2, missing_ids)
    self.assertNotIn(TEST_BASIN, missing_ids)
    self.assertEqual(
        multi_status["missing_or_stale_state_catchments"][0]["state_date"],
        "2026-03-27",
    )

    # Attempting to fetch or run in hotstart mode for [TEST_BASIN, TEST_BASIN_2] must fail with HTTP 400
    code, blocked_fetch = self._request(
        "POST",
        "/api/forecast/fetch-realtime",
        {
            "catchment_id": TEST_BASIN,
            "catchment_ids": [TEST_BASIN, TEST_BASIN_2],
            "mode": "hotstart",
        },
    )
    self.assertEqual(code, 400)
    self.assertIn(TEST_BASIN_2, blocked_fetch["message"])
    self.assertIn("2026-03-27", blocked_fetch["message"])

    code, blocked_run = self._request(
        "POST",
        "/api/forecast/run-model",
        {
            "catchment_id": TEST_BASIN,
            "catchment_ids": [TEST_BASIN, TEST_BASIN_2],
            "mode": "hotstart",
        },
    )
    self.assertEqual(code, 400)
    self.assertIn(TEST_BASIN_2, blocked_run["message"])
    self.assertIn("2026-03-27", blocked_run["message"])

    # Run Cold-Start across both catchments simultaneously -> saves 2026-03-29 state for both
    code, batch_cold = self._request(
        "POST",
        "/api/forecast/run-model",
        {
            "catchment_id": TEST_BASIN,
            "catchment_ids": [TEST_BASIN, TEST_BASIN_2],
            "mode": "coldstart",
        },
    )
    self.assertEqual(code, 200, f"Batch cold-start failed: {batch_cold}")
    self.assertEqual(batch_cold["catchment_ids"], [TEST_BASIN, TEST_BASIN_2])
    self.assertIn(TEST_BASIN, batch_cold["results_by_id"])
    self.assertIn(TEST_BASIN_2, batch_cold["results_by_id"])
    self.assertEqual(
        batch_cold["results_by_id"][TEST_BASIN_2]["saved_state"]["state_date"],
        "2026-03-29",
    )

    # Now Hot-Start across both catchments must succeed
    code, batch_hot = self._request(
        "POST",
        "/api/forecast/run-model",
        {
            "catchment_id": TEST_BASIN,
            "catchment_ids": [TEST_BASIN, TEST_BASIN_2],
            "mode": "hotstart",
        },
    )
    self.assertEqual(code, 200, f"Batch hot-start failed: {batch_hot}")
    self.assertEqual(batch_hot["mode"], "hotstart")
    for cid in (TEST_BASIN, TEST_BASIN_2):
      cold_q = [
          pt["discharge_cms"]
          for pt in batch_cold["results_by_id"][cid]["forecast"]
      ]
      hot_q = [
          pt["discharge_cms"]
          for pt in batch_hot["results_by_id"][cid]["forecast"]
      ]
      np.testing.assert_allclose(cold_q, hot_q, rtol=1e-4, atol=1e-4)

  def test_fetch_realtime_endpoint_calls_multimet_realtime_with_mode(self):
    """Verifies POST /api/forecast/fetch-realtime passes mode and multiple basins to multimet.realtime."""
    _populate_synthetic_realtime_zarr_stores(
        self.profile_dir,
        basin_id=[TEST_BASIN, TEST_BASIN_2],
        issue_date_str="2026-03-30",
    )

    fake_summary = {
        "mode": "coldstart",
        "reference_date": pd.Timestamp("2026-03-30"),
        "products": {
            "CPC": self.profile_dir / "realtime" / "dynamics" / "CPC" / "timeseries.zarr",
            "IMERG": self.profile_dir / "realtime" / "dynamics" / "IMERG" / "timeseries.zarr",
            "HRES": self.profile_dir / "realtime" / "dynamics" / "HRES" / "timeseries.zarr",
        },
        "windows": {
            "CPC": (pd.Timestamp("2025-03-30"), pd.Timestamp("2026-03-29")),
            "IMERG": (pd.Timestamp("2025-03-30"), pd.Timestamp("2026-03-29")),
            "HRES": (pd.Timestamp("2025-03-30"), pd.Timestamp("2026-03-30")),
        },
    }

    with mock.patch.object(
        realtime_forecast_service,
        "_load_multimet_realtime",
    ) as mock_loader:
      mock_mm_config = mock.MagicMock()
      mock_mm_realtime = mock.MagicMock()
      mock_mm_realtime.fetch_realtime_multimet.return_value = fake_summary
      mock_loader.return_value = (mock_mm_config, mock_mm_realtime)

      code, fetch_res = self._request(
          "POST",
          "/api/forecast/fetch-realtime",
          {
              "catchment_id": TEST_BASIN,
              "catchment_ids": [TEST_BASIN, TEST_BASIN_2],
              "mode": "coldstart",
          },
      )
      self.assertEqual(code, 200, f"Fetch endpoint failed: {fetch_res}")
      self.assertEqual(fetch_res["status"], "success")
      self.assertEqual(fetch_res["fetch_summary"]["mode"], "coldstart")
      self.assertEqual(
          fetch_res["fetch_summary"]["catchment_ids"],
          [TEST_BASIN, TEST_BASIN_2],
      )
      self.assertIn("HRES", fetch_res["fetch_summary"]["products_written"])
      mock_mm_realtime.fetch_realtime_multimet.assert_called_once()
      call_kwargs = mock_mm_realtime.fetch_realtime_multimet.call_args.kwargs
      self.assertEqual(call_kwargs["mode"], "coldstart")

  def test_load_multimet_realtime_reload_preserves_product_enum_identity(self):
    """Regression test: reloading multimet modules must never cause KeyError(<Product.HRES: 'HRES'>)."""
    from multimet.timeseries_extractors import dynamical as mm_dynamical
    from multimet.timeseries_extractors import hres as mm_hres
    from multimet.timeseries_extractors import zarr_writer as mm_zw

    mm_config, mm_realtime = realtime_forecast_service._load_multimet_realtime(
        reload_modules=True
    )
    self.assertIs(mm_realtime.Product, mm_config.Product)
    self.assertIs(mm_realtime.Product, mm_zw.Product)
    self.assertIs(mm_realtime.Product, mm_hres.Product)
    self.assertIs(mm_realtime.Product, mm_dynamical.Product)

    for prod_name in ("HRES", "AIFS", "GFS", "GEFS", "IFS_ENS", "IMERG", "CPC"):
      prod_enum = mm_realtime.Product[prod_name]
      self.assertIn(prod_enum, mm_zw.PRODUCT_TYPES)
      self.assertIn(prod_enum, mm_zw.PRODUCT_BANDS)


if __name__ == "__main__":
  unittest.main()


