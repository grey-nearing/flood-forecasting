#!/usr/bin/env python3
"""Unit and Integration Tests for MaaS Frontend Adapter, Viewer, Networks & API Endpoints."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request

from shapely.geometry import LineString

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(PARENT_DIR)

for p in (PARENT_DIR, REPO_ROOT):
  if p not in sys.path:
    sys.path.insert(0, p)

from frontend import maas_engine
from frontend.maas_viewer import (
    MaaSViewer,
    buffer_reach_corridor,
    build_consensus_row,
    camaflood_unit_feature,
    chain_length_km,
    emulate_camaflood_physics,
    reach_exceedance_summary,
    route_floodplain_excess,
)
from frontend.server import EarthkitHydroHandler
from maas.config import MaaSConfig, parse_finite_float
from maas.floodhub import FloodHubClient
from maas.thresholds import (
    UNASSESSED_LABEL,
    classify_exceedance,
    estimate_return_period_years,
)


def _mock_glofas_forecast() -> dict:
  today = datetime.now(timezone.utc)
  return {
      "model": "copernicus_glofas",
      "available": True,
      "status": "live",
      "lat": 23.875,
      "lon": 89.875,
      "data": [
          {
              "time": (today + timedelta(days=d)).strftime("%Y-%m-%d"),
              "discharge_median": 2200.0 + 100.0 * d,
              "discharge_mean": 2250.0 + 100.0 * d,
              "discharge_min": 1800.0 + 80.0 * d,
              "discharge_p25": 2000.0 + 90.0 * d,
              "discharge_p75": 2500.0 + 110.0 * d,
              "discharge_max": 2900.0 + 130.0 * d,
          }
          for d in range(15)
      ],
  }


def _mock_geoglows_forecast(river_id: int = 720010511) -> dict:
  today = datetime.now(timezone.utc)
  return {
      "model": "geoglows",
      "available": True,
      "status": "live",
      "river_id": river_id,
      "data": [
          {
              "time": (today + timedelta(days=d)).strftime("%Y-%m-%dT00:00:00Z"),
              "flow_med": 2100.0 + 95.0 * d,
              "flow_avg": 2150.0 + 95.0 * d,
              "flow_min": 1750.0 + 70.0 * d,
              "flow_25p": 1950.0 + 85.0 * d,
              "flow_75p": 2400.0 + 105.0 * d,
              "flow_max": 2750.0 + 120.0 * d,
          }
          for d in range(15)
      ],
  }


def _mock_floodhub_forecast(gauge_id: str = "hybas_7120012340") -> dict:
  today = datetime.now(timezone.utc)
  return {
      "model": "google_floodhub",
      "available": True,
      "status": "live",
      "gauge_id": gauge_id,
      "issued_time": today.strftime("%Y-%m-%dT00:00:00Z"),
      "unit": "CUBIC_METERS_PER_SECOND",
      "thresholds": {
          "warning_2yr": 1800.0,
          "danger_5yr": 2400.0,
          "extreme_20yr": 3500.0,
      },
      "data": [
          {
              "time": (today + timedelta(days=d)).strftime("%Y-%m-%dT00:00:00Z"),
              "discharge": 1950.0 + 120.0 * d,
          }
          for d in range(7)
      ],
      "fallback_reason": None,
      "nan_issues_skipped": 0,
  }


def _mock_return_periods() -> dict:
  return {
      "return_period_2": 1800.0,
      "return_period_5": 2400.0,
      "return_period_10": 2900.0,
      "return_period_20": 3500.0,
      "return_period_25": 3650.0,
      "return_period_50": 4200.0,
      "return_period_100": 4900.0,
      "source": "unit_test_rp",
      "status": "live",
      "unit": "m³/s",
  }


class TestMaaSWatershedAndAdapter(unittest.TestCase):
  """Tests watershed polygon generation and `maas_engine` delegation to `maas`."""

  def test_get_maas_watershed_polygon_glofas(self):
    feat = maas_engine.get_maas_watershed_polygon(23.23, 90.64, fabric="glofas_cell")
    self.assertEqual(feat.get("type"), "Feature")
    geom = feat.get("geometry", {})
    self.assertEqual(geom.get("type"), "Polygon")
    self.assertEqual(len(geom.get("coordinates", [[]])[0]), 5)
    props = feat.get("properties", {})
    self.assertEqual(props.get("fabric"), "glofas_cell")
    self.assertGreater(props.get("area_km2", 0), 0)

  def test_get_maas_watershed_polygon_camaflood_unit(self):
    feat = maas_engine.get_maas_watershed_polygon(23.23, 90.64, geofabric="camaflood_unit")
    self.assertEqual(feat.get("type"), "Feature")
    geom = feat.get("geometry", {})
    self.assertEqual(geom.get("type"), "Polygon")
    ring = geom["coordinates"][0]
    self.assertEqual(len(ring), 5)
    self.assertEqual(ring[0], ring[-1])
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    self.assertAlmostEqual(max(lons) - min(lons), 0.25, places=4)
    self.assertAlmostEqual(max(lats) - min(lats), 0.25, places=4)
    props = feat.get("properties", {})
    self.assertEqual(props.get("geofabric"), "camaflood_unit")
    self.assertEqual(props.get("fabric"), "camaflood_unit")
    self.assertEqual(props.get("grid_cell_id"), "cama_025_23.125_90.625")
    self.assertTrue(600 < props.get("area_km2", 0) < 800)

  def test_get_maas_watershed_polygon_merit_and_hydroatlas(self):
    feat = maas_engine.get_maas_watershed_polygon(23.23, 90.64, fabric="merit_reach", river_id=71000092)
    self.assertEqual(feat.get("type"), "Feature")
    self.assertEqual(feat.get("properties", {}).get("fabric"), "merit_reach")

    unit_feat = maas_engine.get_maas_watershed_polygon(40.45, -86.95, fabric="hydroatlas_unit")
    self.assertEqual(unit_feat.get("type"), "Feature")
    self.assertEqual(unit_feat.get("properties", {}).get("fabric"), "hydroatlas_unit")

    full_feat = maas_engine.get_maas_watershed_polygon(40.45, -86.95, fabric="hydroatlas_full")
    self.assertEqual(full_feat.get("type"), "Feature")
    self.assertEqual(full_feat.get("properties", {}).get("fabric"), "hydroatlas_full")


class TestMaaSContracts(unittest.TestCase):
  """Deterministic tests for `maas` and `frontend.maas_viewer` contracts."""

  @staticmethod
  def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

  def _issue(self, issued, starts, values):
    return {
        "issuedTime": self._iso(issued),
        "forecastRanges": [
            {
                "forecastStartTime": self._iso(s),
                "forecastEndTime": self._iso(s + timedelta(days=1)),
                "value": v,
            }
            for s, v in zip(starts, values)
        ],
    }

  def test_parse_finite_float_rejects_nan_strings(self):
    for bad in ("NaN", "nan", "inf", "-Infinity", "", None, "abc", float("nan")):
      self.assertIsNone(parse_finite_float(bad), repr(bad))
    self.assertEqual(parse_finite_float("1.5"), 1.5)
    self.assertEqual(parse_finite_float(2), 2.0)

  def test_estimate_return_period_years_consistent_with_levels(self):
    rps = {
        "return_period_2": 47.7,
        "return_period_5": 127.2,
        "return_period_20": 298.6,
        "status": "live",
    }
    for below in (0.0, 0.6, 47.0, None):
      self.assertIsNone(estimate_return_period_years(below, rps), repr(below))
    self.assertIsNone(estimate_return_period_years(100.0, {"return_period_2": 50.0}))
    self.assertEqual(estimate_return_period_years(47.7, rps), 2.0)
    self.assertEqual(estimate_return_period_years(127.2, rps), 5.0)
    self.assertEqual(estimate_return_period_years(298.6, rps), 20.0)
    self.assertTrue(5.0 < estimate_return_period_years(200.0, rps) < 20.0)
    self.assertGreater(estimate_return_period_years(400.0, rps), 20.0)
    self.assertEqual(estimate_return_period_years(1e9, rps), 1000.0)
    for q in (60.0, 150.0, 320.0):
      rank = classify_exceedance(q, rps)["rank"]
      self.assertGreaterEqual(
          estimate_return_period_years(q, rps), {1: 2.0, 2: 5.0, 3: 20.0}[rank]
      )

  def test_consensus_row_and_reach_exceedance_summary(self):
    rps = {
        "return_period_2": 0.5,
        "return_period_5": 1.0,
        "return_period_20": 2.0,
        "status": "computed",
    }
    live = build_consensus_row(
        "geoglows", True, "live", 58.0, "2026-09-30", rps, "src", "High"
    )
    self.assertEqual(live["risk_level"], "EXTREME")
    fallback = build_consensus_row(
        "geoglows", True, "fallback", 58.0, "2026-09-30", rps, "src", "Low"
    )
    self.assertEqual(fallback["risk_level"], "UNKNOWN")
    self.assertIsNone(fallback["exceedance_rank"])
    self.assertEqual(fallback["exceedance_label"], UNASSESSED_LABEL)

    today = datetime.now(timezone.utc)

    def forecast(status, key, value):
      days = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]
      return {"status": status, "data": [{"time": d, key: value} for d in days]}

    geoglows_fallback = forecast("fallback", "flow_med", 58.0)
    summary = reach_exceedance_summary(
        forecast("live", "discharge_median", 0.2), rps, geoglows_fallback, rps
    )
    self.assertEqual(summary["governing_model"], "glofas")
    self.assertEqual(summary["rank"], 0)
    self.assertEqual(summary["excluded_models"], ["geoglows"])

  def test_floodhub_client_skips_nan_and_returns_unavailable_without_synthetic_fallback(self):
    now = datetime.now(timezone.utc)
    gid = "TEST_FH_GAUGE"
    starts = [now + timedelta(days=d) for d in range(3)]
    finite = self._issue(now - timedelta(hours=6), starts, [10.0, 20.5, 30.25])
    nan_issue = self._issue(now - timedelta(hours=1), starts, ["NaN", "NaN", "NaN"])
    thresholds = {"warningLevel": 12.0, "dangerLevel": "NaN", "extremeDangerLevel": 30.0}

    session = mock.MagicMock()

    def fake_get(url, params=None, timeout=12.0):  # pylint: disable=unused-argument
      resp = mock.MagicMock()
      resp.status_code = 200
      if "gauges:queryGaugeForecasts" in url:
        resp.json.return_value = {"forecasts": {gid: {"forecasts": [finite, nan_issue]}}}
      elif "gaugeModels:batchGet" in url:
        resp.json.return_value = {
            "gaugeModels": [
                {"gaugeValueUnit": "CUBIC_METERS_PER_SECOND", "thresholds": thresholds}
            ]
        }
      return resp

    session.get.side_effect = fake_get
    client = FloodHubClient(api_key="test-key", session=session)
    res = client.fetch_forecast(gid, now_utc=now)
    self.assertEqual(res["status"], "live")
    self.assertEqual(res["nan_issues_skipped"], 1)
    self.assertEqual([p["discharge"] for p in res["data"]], [10.0, 20.5, 30.25])
    self.assertEqual(
        res["thresholds"],
        {"warning_2yr": 12.0, "danger_5yr": None, "extreme_20yr": 30.0},
    )

    # All-NaN issues -> status="unavailable", empty data (no synthetic fallback!)
    all_nan = [self._issue(now - timedelta(hours=h), starts, ["NaN"] * 3) for h in (12, 6, 1)]
    session.get.side_effect = lambda url, **kw: mock.MagicMock(
        status_code=200,
        json=mock.MagicMock(
            return_value={"forecasts": {gid: {"forecasts": all_nan}}}
            if "queryGaugeForecasts" in url
            else {"gaugeModels": []}
        ),
    )
    res_nan = client.fetch_forecast(gid, now_utc=now)
    self.assertFalse(res_nan["available"])
    self.assertEqual(res_nan["status"], "unavailable")
    self.assertEqual(res_nan["data"], [])
    self.assertEqual(res_nan["nan_issues_skipped"], 3)

  def test_todays_earth_operational_feed_via_maas(self):
    feed = {
        "timestamps": [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ],
        "rivout": [100.0, 150.0, 120.0],
        "fldout": [0.0, 30.0, 10.0],
        "flddph": [0.0, 0.8, 0.3],
        "fldfrc": [0.0, 0.12, 0.05],
        "sfcelv": [5.0, 6.2, 5.6],
    }
    mock_resp = mock.MagicMock(status_code=200)
    mock_resp.json.return_value = feed
    with (
        mock.patch.dict(os.environ, {"TODAYS_EARTH_API_URL": "http://te-feed.invalid/point"}),
        mock.patch("requests.Session.get", return_value=mock_resp),
    ):
      self.assertEqual(maas_engine.todays_earth_service_status(), "operational")
      te = maas_engine.fetch_todays_earth_forecast(-12.34, 56.78, reach_id="TEST_REACH")
    self.assertEqual(te["status"], "live")
    self.assertFalse(te["emulated"])
    self.assertEqual(te["mean"], [100.0, 180.0, 130.0])
    ff = te["flood_forecast"]
    self.assertAlmostEqual(ff["max_flood_depth_m"], 0.8)
    self.assertAlmostEqual(ff["max_flooded_fraction_pct"], 12.0)

  def test_inundation_corridor_buffering_and_camaflood_emulation(self):
    chain = [
        {"geometry": LineString([[-90.30, 38.65], [-90.25, 38.63]])},
        {"geometry": LineString([[-90.25, 38.63], [-90.20, 38.61]])},
        {"geometry": LineString([[-90.20, 38.61], [-90.15, 38.59]])},
    ]
    total_len = chain_length_km(chain, ref_lat=38.63)
    self.assertGreater(total_len, 12.0)

    poly = buffer_reach_corridor(chain, half_width_m=600.0, ref_lat=38.63)
    self.assertIsNotNone(poly)
    self.assertIn(poly.geom_type, ("Polygon", "MultiPolygon"))
    self.assertGreater(poly.area, 0.0)

    routed = route_floodplain_excess([1000.0, 2000.0, 2500.0], q_bankfull=1500.0)
    self.assertEqual(len(routed), 3)
    self.assertEqual(routed[0], 0.0)
    self.assertGreater(routed[1], 0.0)

    glofas_records = _mock_glofas_forecast()["data"]
    rp = _mock_return_periods()
    emulated = emulate_camaflood_physics(glofas_records, rp, elev=120.0)
    self.assertEqual(len(emulated["series"]["rivout"]), 6)
    self.assertEqual(len(emulated["series"]["flddph_m"]), 6)

    unit_feat = camaflood_unit_feature(
        38.625,
        -90.125,
        {
            "status": "emulated",
            "emulated": True,
            "flood_forecast": {
                "max_flood_depth_m": 0.85,
                "max_flooded_fraction_pct": 12.5,
            },
        },
    )
    self.assertEqual(unit_feat["type"], "Feature")
    self.assertEqual(unit_feat["properties"]["layer"], "camaflood_depth")

  def test_maas_viewer_render_forecast_and_inundation_views(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      cfg = MaaSConfig(
          cache_dir=tmp_path / "cache",
          river_networks_dir=tmp_path / "river_networks",
          floodhub_api_key="test-key",
      )
      viewer = MaaSViewer(cfg)
      mock_gl = _mock_glofas_forecast()
      mock_gg = _mock_geoglows_forecast()
      mock_fh = _mock_floodhub_forecast()
      mock_rp = _mock_return_periods()

      with (
          mock.patch.object(viewer.fetcher.glofas, "fetch_forecast", return_value=mock_gl),
          mock.patch.object(
              viewer.fetcher.glofas, "fetch_reanalysis_return_periods", return_value=mock_rp
          ),
          mock.patch.object(viewer.fetcher.geoglows, "fetch_river_id", return_value=720010511),
          mock.patch.object(viewer.fetcher.geoglows, "fetch_forecast", return_value=mock_gg),
          mock.patch.object(
              viewer.fetcher.geoglows, "fetch_return_periods", return_value=mock_rp
          ),
          mock.patch.object(viewer.fetcher.floodhub, "fetch_forecast", return_value=mock_fh),
          mock.patch.object(viewer.fetcher.floodhub, "fetch_flood_status", return_value=None),
      ):
        view = viewer.render_forecast_view(
            38.6270,
            -90.1994,
            gauge_id="hybas_7120012340",
            requested_models=["floodhub", "glofas", "geoglows", "todays_earth"],
        )
        self.assertEqual(
            set(view["models"]), {"floodhub", "glofas", "geoglows", "todays_earth"}
        )
        self.assertEqual(
            [r["model"] for r in view["consensus"]],
            ["floodhub", "glofas", "geoglows", "todays_earth"],
        )
        self.assertIn("overall_risk_level", view["flood_summary"])
        timeline = view["timeline"]
        self.assertGreater(len(timeline["dates"]), 0)
        for key, series in timeline["series"].items():
          self.assertIn("central", series, key)
          self.assertEqual(len(series["central"]), len(timeline["dates"]), key)
        self.assertIn("status", timeline)

        inund_fc = viewer.render_inundation_view(
            38.6270, -90.1994, gauge_id="hybas_7120012340"
        )
        self.assertEqual(inund_fc["type"], "FeatureCollection")
        self.assertGreater(len(inund_fc["features"]), 0)
        self.assertEqual(
            set(inund_fc["metadata"]["layers"]),
            {"floodhub_extent", "camaflood_depth", "reach_exceedance"},
        )


class TestMaaSServerEndpoints(unittest.TestCase):
  """Integration tests for `/api/maas/*` HTTP endpoints."""

  @classmethod
  def setUpClass(cls):
    cls.server = socketserver.TCPServer(("127.0.0.1", 0), EarthkitHydroHandler)
    cls.port = int(cls.server.server_address[1])
    cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.thread.start()
    time.sleep(0.05)

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def _get(self, path, timeout=30.0):
    url = f"http://127.0.0.1:{self.port}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return resp.status, json.loads(resp.read().decode("utf-8"))

  def _get_status(self, path):
    try:
      return self._get(path)[0]
    except urllib.error.HTTPError as e:
      return e.code

  def test_api_maas_models(self):
    status, data = self._get("/api/maas/models")
    self.assertEqual(status, 200)
    self.assertIn("models", data)
    model_ids = [m["id"] for m in data["models"]]
    self.assertEqual(model_ids, ["floodhub", "glofas", "geoglows", "todays_earth"])

  def test_api_maas_gauges(self):
    with mock.patch(
        "frontend.maas_engine.fetch_floodhub_gauges_bbox",
        return_value=[
            {"gauge_id": "hybas_1", "lat": 32.7, "lon": -117.2, "severity": "NO_FLOODING"}
        ],
    ):
      status, data = self._get(
          "/api/maas/gauges?min_lat=32.5&min_lon=-117.5&max_lat=33.0&max_lon=-117.0"
      )
    self.assertEqual(status, 200)
    self.assertEqual(data["count"], 1)

  def test_api_maas_reaches(self):
    status, data = self._get("/api/maas/reaches?lat=23.875&lon=89.875")
    self.assertEqual(status, 200)
    self.assertIn("glofas", data)
    self.assertIn("todays_earth", data)

  def test_api_maas_watershed(self):
    status, data = self._get("/api/maas/watershed?lat=23.23&lon=90.64&fabric=glofas_cell")
    self.assertEqual(status, 200)
    self.assertEqual(data.get("type"), "Feature")
    self.assertEqual(data.get("properties", {}).get("fabric"), "glofas_cell")

    status, data = self._get("/api/maas/watershed?lat=23.23&lon=90.64&geofabric=camaflood_unit")
    self.assertEqual(status, 200)
    self.assertEqual(data.get("properties", {}).get("grid_cell_id"), "cama_025_23.125_90.625")

  def test_api_maas_forecast_and_inundation_mocked(self):
    mock_gl = _mock_glofas_forecast()
    mock_gg = _mock_geoglows_forecast()
    mock_fh = _mock_floodhub_forecast()
    mock_rp = _mock_return_periods()
    with (
        mock.patch("maas.glofas.GloFASClient.fetch_forecast", return_value=mock_gl),
        mock.patch(
            "maas.glofas.GloFASClient.fetch_reanalysis_return_periods", return_value=mock_rp
        ),
        mock.patch("maas.geoglows.GeoGLOWSClient.fetch_river_id", return_value=720010511),
        mock.patch("maas.geoglows.GeoGLOWSClient.fetch_forecast", return_value=mock_gg),
        mock.patch(
            "maas.geoglows.GeoGLOWSClient.fetch_return_periods", return_value=mock_rp
        ),
        mock.patch("maas.floodhub.FloodHubClient.fetch_forecast", return_value=mock_fh),
        mock.patch("maas.floodhub.FloodHubClient.fetch_flood_status", return_value=None),
    ):
      status, data = self._get(
          "/api/maas/forecast?lat=32.756&lon=-117.252&gauge_id=hybas_7120012340"
      )
      self.assertEqual(status, 200)
      self.assertEqual(
          set(data["models"]), {"floodhub", "glofas", "geoglows", "todays_earth"}
      )
      self.assertEqual(
          [r["model"] for r in data["consensus"]],
          ["floodhub", "glofas", "geoglows", "todays_earth"],
      )
      self.assertIn("todays_earth", data["timeline"]["series"])

      status_sub, data_sub = self._get(
          "/api/maas/forecast?lat=32.756&lon=-117.252&models=glofas,todays_earth"
      )
      self.assertEqual(status_sub, 200)
      self.assertEqual(
          [r["model"] for r in data_sub["consensus"]], ["glofas", "todays_earth"]
      )

      status_inund, data_inund = self._get(
          "/api/maas/flood-inundation?lat=32.756&lon=-117.252&gauge_id=hybas_7120012340"
      )
      self.assertEqual(status_inund, 200)
      self.assertEqual(data_inund.get("type"), "FeatureCollection")
      self.assertGreater(len(data_inund.get("features", [])), 0)

  def test_api_maas_flood_inundation_rejects_bad_coordinates(self):
    self.assertEqual(self._get_status("/api/maas/flood-inundation?lat=abc&lon=1"), 400)

  def test_api_maas_network_telescoping_all_models(self):
    for model in ("floodhub", "glofas", "geoglows", "todays_earth"):
      status, data = self._get(
          f"/api/maas/network?model={model}&bbox=88.0,22.0,91.5,25.5&zoom=8"
      )
      self.assertEqual(status, 200)
      self.assertEqual(data.get("type"), "FeatureCollection")
      self.assertEqual((data.get("properties") or {}).get("model"), model)
      self.assertGreater(len(data.get("features") or []), 0)

  def test_api_maas_cross_network_reach_matching(self):
    mock_gl = _mock_glofas_forecast()
    mock_gg = _mock_geoglows_forecast()
    mock_fh = _mock_floodhub_forecast()
    mock_rp = _mock_return_periods()
    with (
        mock.patch("maas.glofas.GloFASClient.fetch_forecast", return_value=mock_gl),
        mock.patch(
            "maas.glofas.GloFASClient.fetch_reanalysis_return_periods", return_value=mock_rp
        ),
        mock.patch("maas.geoglows.GeoGLOWSClient.fetch_forecast", return_value=mock_gg),
        mock.patch(
            "maas.geoglows.GeoGLOWSClient.fetch_return_periods", return_value=mock_rp
        ),
        mock.patch(
            "maas.floodhub.FloodHubClient.find_nearest_gauge",
            return_value=("hybas_4120012340", {"lat": 23.88, "lon": 89.88}, 1.2),
        ),
        mock.patch("maas.floodhub.FloodHubClient.fetch_forecast", return_value=mock_fh),
        mock.patch("maas.floodhub.FloodHubClient.fetch_flood_status", return_value=None),
    ):
      status, data = self._get(
          "/api/maas/forecast?lat=23.875&lon=89.875&network=todays_earth&upstream_area_km2=1480000"
      )
    self.assertEqual(status, 200)
    vs = data.get("virtual_station") or {}
    self.assertGreater((vs.get("glofas_cell") or {}).get("upstream_area_km2") or 0, 500000)
    self.assertGreater(
        (vs.get("geoglows_reach") or {}).get("upstream_area_km2") or 0, 500000
    )


if __name__ == "__main__":
  unittest.main()
