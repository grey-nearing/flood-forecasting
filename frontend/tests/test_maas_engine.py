#!/usr/bin/env python3
"""Unit and Integration Tests for MaaS Engine & API Endpoints."""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock
import urllib.error
import urllib.request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../.."))

if PARENT_DIR not in sys.path:
  sys.path.insert(0, PARENT_DIR)
if WORKSPACE_ROOT not in sys.path:
  sys.path.insert(0, WORKSPACE_ROOT)

import maas_engine
from server import EarthkitHydroHandler


class TestMaaSEngine(unittest.TestCase):

  def test_fetch_floodhub_gauges_bbox(self):
    """Verifies querying FloodHub gauges in a bounding box."""
    gauges = maas_engine.fetch_floodhub_gauges_bbox(32.5, -117.5, 33.0, -117.0)
    self.assertIsInstance(gauges, list)
    self.assertGreater(len(gauges), 0)
    g = gauges[0]
    self.assertIn("gauge_id", g)
    self.assertIn("lat", g)
    self.assertIn("lon", g)
    self.assertIn("severity", g)

  def test_fetch_floodhub_forecast(self):
    """Verifies querying 7-day forecast for a FloodHub gauge."""
    res = maas_engine.fetch_floodhub_forecast("hybas_7120012340")
    self.assertTrue(res.get("available"))
    self.assertEqual(res.get("model"), "google_floodhub")
    self.assertIn("thresholds", res)
    data = res.get("data", [])
    self.assertIsInstance(data, list)
    self.assertGreater(len(data), 0)
    self.assertIn("time", data[0])
    self.assertIn("discharge", data[0])

  def test_fetch_geoglows_forecast(self):
    """Verifies querying GEOGLOWS 15-day ensemble forecast."""
    res = maas_engine.fetch_geoglows_forecast(32.75, -117.25)
    self.assertTrue(res.get("available"))
    self.assertEqual(res.get("model"), "geoglows")
    self.assertIn("river_id", res)
    data = res.get("data", [])
    self.assertIsInstance(data, list)
    self.assertGreater(len(data), 0)
    self.assertIn("flow_med", data[0])

  def test_fetch_glofas_forecast(self):
    """Verifies querying GloFAS 15-day ensemble daily statistics."""
    res = maas_engine.fetch_glofas_forecast(32.75, -117.25, forecast_days=15)
    self.assertTrue(res.get("available"))
    self.assertEqual(res.get("model"), "copernicus_glofas")
    data = res.get("data", [])
    self.assertIsInstance(data, list)
    self.assertGreater(len(data), 0)
    self.assertIn("discharge_mean", data[0])

  def test_aggregate_maas_forecast(self):
    """Verifies aggregating FloodHub, GEOGLOWS, and GloFAS forecasts."""
    agg = maas_engine.aggregate_maas_forecast(
        32.756, -117.252, gauge_id="hybas_7120012340"
    )
    self.assertIn("location", agg)
    self.assertIn("thresholds", agg)
    self.assertIn("models", agg)
    models = agg["models"]
    self.assertIn("floodhub", models)
    self.assertIn("geoglows", models)
    self.assertIn("glofas", models)

  def test_get_maas_watershed_polygon_glofas(self):
    """Verifies GloFAS 0.05° grid-cell polygon generation."""
    feat = maas_engine.get_maas_watershed_polygon(23.23, 90.64, fabric="glofas_cell")
    self.assertEqual(feat.get("type"), "Feature")
    geom = feat.get("geometry", {})
    self.assertEqual(geom.get("type"), "Polygon")
    self.assertEqual(len(geom.get("coordinates", [[]])[0]), 5)
    props = feat.get("properties", {})
    self.assertEqual(props.get("fabric"), "glofas_cell")
    self.assertIn("area_km2", props)
    self.assertGreater(props.get("area_km2", 0), 0)

  def test_get_maas_watershed_polygon_merit(self):
    """Verifies MERIT-Hydro reach catchment polygon retrieval."""
    feat = maas_engine.get_maas_watershed_polygon(23.23, 90.64, fabric="merit_reach", river_id=71000092)
    self.assertEqual(feat.get("type"), "Feature")
    self.assertIn(feat.get("geometry", {}).get("type"), ["Polygon", "MultiPolygon"])
    props = feat.get("properties", {})
    self.assertEqual(props.get("fabric"), "merit_reach")
    self.assertEqual(props.get("comid"), 71000092)
    self.assertGreater(props.get("area_km2", 0), 0)

  def test_get_maas_watershed_polygon_hydroatlas(self):
    """Verifies HydroATLAS full drainage and unit catchment polygon retrieval."""
    # Unit catchment
    unit_feat = maas_engine.get_maas_watershed_polygon(40.45, -86.95, fabric="hydroatlas_unit")
    self.assertEqual(unit_feat.get("type"), "Feature")
    self.assertIn(unit_feat.get("geometry", {}).get("type"), ["Polygon", "MultiPolygon"])
    self.assertEqual(unit_feat.get("properties", {}).get("fabric"), "hydroatlas_unit")

    # Full drainage
    full_feat = maas_engine.get_maas_watershed_polygon(40.45, -86.95, fabric="hydroatlas_full")
    self.assertEqual(full_feat.get("type"), "Feature")
    self.assertIn(full_feat.get("geometry", {}).get("type"), ["Polygon", "MultiPolygon"])
    self.assertEqual(full_feat.get("properties", {}).get("fabric"), "hydroatlas_full")

  def test_get_maas_watershed_polygon_caching(self):
    """Verifies persistent SQLite caching for watershed polygons."""
    t0 = time.time()
    feat1 = maas_engine.get_maas_watershed_polygon(23.23, 90.64, fabric="glofas_cell")
    t1 = time.time()
    feat2 = maas_engine.get_maas_watershed_polygon(23.23, 90.64, fabric="glofas_cell")
    t2 = time.time()
    self.assertEqual(feat1["properties"]["cell_center_lat"], feat2["properties"]["cell_center_lat"])
    self.assertEqual(feat1["geometry"]["coordinates"], feat2["geometry"]["coordinates"])

  # ---------------------------------------------------------------------------
  # Phase 6: JAXA Today's Earth, return periods, flood inundation (live APIs)
  # ---------------------------------------------------------------------------

  def test_fetch_todays_earth_forecast_structure(self):
    """Verifies the Today's Earth (CaMa-Flood) streamflow + flood-inundation forecast contract."""
    te = maas_engine.fetch_todays_earth_forecast(23.23, 90.64)
    self.assertEqual(te.get("model"), "jaxa_todays_earth")
    self.assertIn(te.get("status"), ("live", "fallback"))
    self.assertIsInstance(te.get("emulated"), bool)
    self.assertTrue(te.get("source"))
    self.assertEqual(te.get("grid_cell_id"), "cama_025_23.125_90.625")
    self.assertEqual(te.get("unit"), "m³/s")
    n = len(te.get("timestamps") or [])
    self.assertGreater(n, 0)
    for key in ("mean", "rivout", "fldout", "p25", "p75", "max", "min"):
      self.assertEqual(len(te.get(key) or []), n, key)
    for mean, rivout, fldout in zip(te["mean"], te["rivout"], te["fldout"]):
      if None not in (mean, rivout, fldout):
        self.assertAlmostEqual(mean, rivout + fldout, delta=0.02)  # mean = RIVOUT + FLDOUT
    ff = te.get("flood_forecast") or {}
    for key in ("flddph_m", "fldfrc_pct", "sfcelv_m"):
      self.assertEqual(len(ff.get(key) or []), n, key)
    self.assertGreaterEqual(ff.get("max_flood_depth_m"), 0.0)
    self.assertGreaterEqual(ff.get("max_flooded_fraction_pct"), 0.0)
    self.assertLessEqual(ff.get("max_flooded_fraction_pct"), 100.0)
    self.assertTrue(all(v is None or v >= 0.0 for v in ff["flddph_m"]))
    self.assertTrue(all(v is None or v >= 0.0 for v in ff["sfcelv_m"]))

  def test_fetch_geoglows_return_periods_contract(self):
    """Verifies GEOGLOWS return-period thresholds (official, computed or scaled fallback)."""
    rp = maas_engine.fetch_geoglows_return_periods(710437069, mean_flow=1.0)
    self.assertIn(rp.get("status"), ("live", "computed", "fallback"))
    self.assertEqual(rp.get("unit"), "m³/s")
    values = [rp.get(f"return_period_{k}") for k in (2, 5, 10, 25, 50, 100)]
    self.assertTrue(all(isinstance(v, (int, float)) and v > 0 for v in values), values)
    self.assertEqual(values, sorted(values))

  def test_fetch_floodhub_inundation_contract(self):
    """Verifies FloodHub severity / trend normalization and the thresholds contract."""
    res = maas_engine.fetch_floodhub_inundation("hybas_7120012340", 32.756, -117.252, include_polygons=False)
    self.assertEqual(res.get("provider"), "floodhub")
    self.assertEqual(res.get("gauge_id"), "hybas_7120012340")
    self.assertIn(res.get("severity"), ("NO_FLOODING", "WARNING", "DANGER", "EXTREME_DANGER", "UNKNOWN"))
    self.assertIn(res.get("trend"), ("RISING", "FALLING", "STEADY", "UNKNOWN"))
    thresholds = res.get("thresholds") or {}
    for key in ("warning_level", "danger_level", "extreme_danger_level", "unit"):
      self.assertIn(key, thresholds)
    self.assertEqual(res.get("inundation_polygons"), [])  # include_polygons=False

  def test_get_unified_maas_forecast_four_providers(self):
    """Verifies the unified 4-provider forecast: models, consensus rows and flood summary."""
    d = maas_engine.get_unified_maas_forecast(32.756, -117.252, gauge_id="hybas_7120012340")
    self.assertEqual(set(d["models"]), {"floodhub", "glofas", "geoglows", "todays_earth"})
    for key in ("floodhub", "glofas", "geoglows"):
      self.assertIn("data", d["models"][key])  # legacy aggregate contract preserved
    rows = d.get("consensus") or []
    self.assertEqual([r["model"] for r in rows], ["floodhub", "glofas", "geoglows", "todays_earth"])
    for r in rows:
      for key in ("name", "available", "status", "peak_flow", "peak_time", "return_period", "confidence",
                  "risk_level"):
        self.assertIn(key, r)
      self.assertIn(r["risk_level"], ("NORMAL", "WARNING", "SEVERE", "EXTREME", "UNKNOWN"))
    self.assertIn("peak_flood_depth_m", rows[3])
    self.assertIn("peak_flood_fraction_pct", rows[3])
    fs = d.get("flood_summary") or {}
    for key in ("max_inundation_depth_m", "max_flooded_fraction_pct", "floodhub_severity",
                "return_period_exceedance", "overall_risk_level", "agreement"):
      self.assertIn(key, fs)
    self.assertIn(fs["overall_risk_level"], ("NORMAL", "WARNING", "SEVERE", "EXTREME", "UNKNOWN"))
    self.assertTrue(d["virtual_station"]["todays_earth_cell"]["grid_cell_id"].startswith("cama_025_"))
    self.assertTrue(d["flood_inundation"]["endpoint"].startswith("/api/maas/flood-inundation?"))
    for key in ("warning_2yr", "danger_5yr", "extreme_20yr", "source"):
      self.assertIn(key, d["thresholds"])
    timeline = d["timeline"]
    self.assertGreater(len(timeline["dates"]), 0)
    for key, series in timeline["series"].items():
      self.assertEqual(len(series["central"]), len(timeline["dates"]), key)

  def test_get_unified_maas_forecast_model_subset(self):
    """Verifies provider selection; an emulated Today's Earth row is not counted as independent."""
    d = maas_engine.get_unified_maas_forecast(32.756, -117.252, requested_models=["glofas", "todays_earth"])
    self.assertEqual(d["meta"]["models_requested"], ["glofas", "todays_earth"])
    self.assertEqual([r["model"] for r in d["consensus"]], ["glofas", "todays_earth"])
    te_row = d["consensus"][1]
    if te_row.get("emulated"):
      self.assertFalse(te_row["independent"])

  def test_get_maas_flood_inundation_feature_collection(self):
    """Verifies the 3-layer flood-inundation GeoJSON FeatureCollection."""
    fc = maas_engine.get_maas_flood_inundation(32.756, -117.252, gauge_id="hybas_7120012340")
    self.assertEqual(fc.get("type"), "FeatureCollection")
    feats = fc.get("features") or []
    self.assertGreater(len(feats), 0)
    layers = set()
    for f in feats:
      self.assertEqual(f.get("type"), "Feature")
      self.assertIn(f["geometry"]["type"], ("Polygon", "MultiPolygon"))
      self.assertIn(f["properties"]["layer"], ("floodhub_extent", "camaflood_depth", "reach_exceedance"))
      layers.add(f["properties"]["layer"])
    self.assertIn("camaflood_depth", layers)
    unit_cells = [f for f in feats if f["properties"].get("feature_role") == "unit_cell"]
    self.assertEqual(len(unit_cells), 1)
    for key in ("peak_flood_depth_m", "peak_flooded_fraction_pct", "grid_cell_id", "source", "color"):
      self.assertIn(key, unit_cells[0]["properties"])
    meta_layers = (fc.get("metadata") or {}).get("layers") or {}
    self.assertEqual(set(meta_layers), {"floodhub_extent", "camaflood_depth", "reach_exceedance"})
    for name, info in meta_layers.items():
      self.assertEqual(info["count"], sum(1 for f in feats if f["properties"]["layer"] == name), name)


class TestMaaSOfflineContracts(unittest.TestCase):
  """Deterministic Phase-6 checks that need no network access (upstream APIs are mocked)."""

  @staticmethod
  def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

  def _issue(self, issued, starts, values):
    return {
        "issuedTime": self._iso(issued),
        "forecastRanges": [
            {"forecastStartTime": self._iso(s), "forecastEndTime": self._iso(s + timedelta(days=1)), "value": v}
            for s, v in zip(starts, values)
        ],
    }

  @staticmethod
  def _mock_floodhub(gauge_id, issues, thresholds=None):
    def fake_get(url, params=None, timeout=10):  # pylint: disable=unused-argument
      if "gauges:queryGaugeForecasts" in url:
        return {"forecasts": {gauge_id: {"forecasts": issues}}}
      if "gaugeModels:batchGet" in url:
        return {"gaugeModels": [{"gaugeValueUnit": "CUBIC_METERS_PER_SECOND", "thresholds": thresholds or {}}]}
      return None

    return mock.patch.object(maas_engine, "_http_get_json", side_effect=fake_get)

  def test_safe_float_or_none_rejects_nan_strings(self):
    for bad in ("NaN", "nan", "inf", "-Infinity", "", None, "abc", float("nan")):
      self.assertIsNone(maas_engine._safe_float_or_none(bad), repr(bad))
    self.assertEqual(maas_engine._safe_float_or_none("1.5"), 1.5)
    self.assertEqual(maas_engine._safe_float_or_none(2), 2.0)

  def test_estimate_return_period_yrs_consistent_with_levels(self):
    """Return periods are exact at the known levels, monotone, and not reported below the 2-yr level."""
    est = maas_engine._estimate_return_period_yrs
    rps = {"return_period_2": 47.7, "return_period_5": 127.2, "return_period_20": 298.6, "status": "live"}
    # Far below the 2-yr level: "< 2-yr" (no EV1 lower-tail artefact such as ~1.6-yr for 0.6 m³/s).
    for below in (0.0, 0.6, 47.0, None):
      self.assertIsNone(est(below, rps), repr(below))
    self.assertIsNone(est(100.0, {"return_period_2": 50.0}))  # a single level cannot be interpolated
    self.assertEqual(est(47.7, rps), 2.0)
    self.assertEqual(est(127.2, rps), 5.0)
    self.assertEqual(est(298.6, rps), 20.0)
    self.assertTrue(5.0 < est(200.0, rps) < 20.0)
    self.assertGreater(est(400.0, rps), 20.0)
    self.assertEqual(est(1e9, rps), 1000.0)
    values = [est(q, rps) for q in (50.0, 80.0, 127.2, 150.0, 250.0, 298.6, 350.0, 600.0)]
    self.assertEqual(values, sorted(values))
    for q in (60.0, 150.0, 320.0):  # consistent with the exceedance class of the same peak
      rank = maas_engine._classify_exceedance(q, rps)["rank"]
      self.assertGreaterEqual(est(q, rps), {1: 2.0, 2: 5.0, 3: 20.0}[rank])

  def test_synthetic_fallbacks_are_not_classified(self):
    """Regression: a synthetic fallback series is shown but never judged against a real climatology."""
    rps = {"return_period_2": 0.5, "return_period_5": 1.0, "return_period_20": 2.0, "status": "computed"}
    live = maas_engine._consensus_row("geoglows", True, "live", 58.0, "2026-09-30", rps, "src", "High")
    self.assertEqual(live["risk_level"], "EXTREME")
    fallback = maas_engine._consensus_row("geoglows", True, "fallback", 58.0, "2026-09-30", rps, "src", "Low")
    self.assertEqual(fallback["risk_level"], "UNKNOWN")
    self.assertIsNone(fallback["exceedance_rank"])
    self.assertIsNone(fallback["return_period_yrs"])
    self.assertEqual(fallback["exceedance_label"], maas_engine._UNASSESSED_LABEL)
    self.assertEqual(fallback["peak_flow"], 58.0)  # still displayed

    today = datetime.now(timezone.utc)

    def forecast(status, key, value):
      days = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]
      return {"status": status, "data": [{"time": d, key: value} for d in days]}

    geoglows_fallback = forecast("fallback", "flow_med", 58.0)
    summary = maas_engine._reach_exceedance_summary(
        forecast("live", "discharge_median", 0.2), rps, geoglows_fallback, rps)
    self.assertEqual(summary["governing_model"], "glofas")
    self.assertEqual(summary["rank"], 0)
    self.assertEqual(summary["excluded_models"], ["geoglows"])
    self.assertNotIn("geoglows", summary["per_model"])
    no_live = maas_engine._reach_exceedance_summary(
        forecast("fallback", "discharge_median", 58.0), rps, geoglows_fallback, rps)
    self.assertEqual(no_live["risk_level"], "UNKNOWN")
    self.assertEqual(no_live["label"], maas_engine._UNASSESSED_LABEL)
    self.assertEqual(no_live["per_model"], {})
    self.assertEqual(no_live["excluded_models"], ["glofas", "geoglows"])

  def test_floodhub_forecast_skips_nan_issues(self):
    """Regression: FloodHub encodes missing values as "NaN"; the latest finite, fresh issue is used."""
    now = datetime.now(timezone.utc)
    gid = f"TEST_FH_NAN_SKIP_{time.time_ns()}"
    starts = [now + timedelta(days=d) for d in range(3)]
    finite = self._issue(now - timedelta(hours=6), starts, [10.0, 20.5, 30.25])
    nan_issue = self._issue(now - timedelta(hours=1), starts, ["NaN", "NaN", "NaN"])
    thresholds = {"warningLevel": 12.0, "dangerLevel": "NaN", "extremeDangerLevel": 30.0}
    with self._mock_floodhub(gid, [finite, nan_issue], thresholds):
      res = maas_engine.fetch_floodhub_forecast(gid)
    self.assertEqual(res["status"], "live")
    self.assertEqual(res["nan_issues_skipped"], 1)
    self.assertIsNone(res["fallback_reason"])
    self.assertEqual(res["issued_time"], finite["issuedTime"])
    self.assertEqual([p["discharge"] for p in res["data"]], [10.0, 20.5, 30.25])
    self.assertEqual(res["thresholds"], {"warning_2yr": 12.0, "danger_5yr": None, "extreme_20yr": 30.0})

  def test_floodhub_forecast_all_nan_falls_back(self):
    now = datetime.now(timezone.utc)
    gid = f"TEST_FH_ALL_NAN_{time.time_ns()}"
    starts = [now + timedelta(days=d) for d in range(3)]
    issues = [self._issue(now - timedelta(hours=h), starts, ["NaN"] * 3) for h in (12, 6, 1)]
    with self._mock_floodhub(gid, issues):
      res = maas_engine.fetch_floodhub_forecast(gid)
    self.assertEqual(res["status"], "fallback")
    self.assertEqual(res["nan_issues_skipped"], 3)
    self.assertIn("NaN", res["fallback_reason"])
    self.assertGreater(len(res["data"]), 0)  # flagged synthetic series keeps the legacy `data` contract

  def test_floodhub_forecast_stale_issue_falls_back(self):
    now = datetime.now(timezone.utc)
    gid = f"TEST_FH_STALE_{time.time_ns()}"
    starts = [now - timedelta(days=5 - d) for d in range(3)]  # last value 3 days ago
    with self._mock_floodhub(gid, [self._issue(now - timedelta(days=5), starts, [5.0, 6.0, 7.0])]):
      res = maas_engine.fetch_floodhub_forecast(gid)
    self.assertEqual(res["status"], "fallback")
    self.assertIn("stale", res["fallback_reason"])

  def test_todays_earth_operational_feed_adapter(self):
    """A configured TE-Global feed (TODAYS_EARTH_API_URL) is used as-is: status=live, not emulated."""
    feed = {
        "timestamps": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
        "rivout": [100.0, 150.0, 120.0],
        "fldout": [0.0, 30.0, 10.0],
        "flddph": [0.0, 0.8, 0.3],
        "fldfrc": [0.0, 0.12, 0.05],
        "sfcelv": [5.0, 6.2, 5.6],
    }

    def fake_get(url, params=None, timeout=10):  # pylint: disable=unused-argument
      return (feed, None) if url.startswith("http://te-feed.invalid") else (None, "offline")

    with mock.patch.dict(os.environ, {"TODAYS_EARTH_API_URL": "http://te-feed.invalid/point"}), \
        mock.patch.object(maas_engine, "_http_get_json_with_error", side_effect=fake_get):
      self.assertEqual(maas_engine.todays_earth_service_status(), "operational")
      te = maas_engine.fetch_todays_earth_forecast(-12.34, 56.78, reach_id="TEST_REACH")
    self.assertEqual(te["status"], "live")
    self.assertFalse(te["emulated"])
    self.assertIsNone(te["note"])
    self.assertEqual(te["mean"], [100.0, 180.0, 130.0])  # RIVOUT + FLDOUT
    ff = te["flood_forecast"]
    for got, want in zip(ff["fldfrc_pct"], [0.0, 12.0, 5.0]):  # FLDFRC 0-1 -> %
      self.assertAlmostEqual(got, want)
    self.assertAlmostEqual(ff["max_flood_depth_m"], 0.8)
    self.assertAlmostEqual(ff["max_flooded_fraction_pct"], 12.0)
    self.assertEqual(ff["peak_depth_time"], "2026-01-02T00:00:00Z")

  def test_get_maas_watershed_polygon_camaflood_unit(self):
    """Verifies the JAXA Today's Earth CaMa-Flood 0.25° unit-grid polygon (geofabric alias)."""
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
    self.assertTrue(min(lons) <= 90.64 <= max(lons) and min(lats) <= 23.23 <= max(lats))
    props = feat.get("properties", {})
    self.assertEqual(props.get("geofabric"), "camaflood_unit")
    self.assertEqual(props.get("fabric"), "camaflood_unit")
    self.assertEqual(props.get("geofabric_label"), "Today's Earth CaMa-Flood Unit Grid (0.25°)")
    self.assertEqual(props.get("grid_cell_id"), "cama_025_23.125_90.625")
    self.assertTrue(props.get("source"))
    self.assertTrue(600 < props.get("area_km2", 0) < 800, props.get("area_km2"))
    legacy = maas_engine.get_maas_watershed_polygon(23.23, 90.64, fabric="camaflood_unit")
    self.assertEqual(legacy["geometry"], feat["geometry"])


class TestMaaSServerEndpoints(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.server = socketserver.TCPServer(("127.0.0.1", 0), EarthkitHydroHandler)
    cls.port = int(cls.server.server_address[1])
    cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.thread.start()
    time.sleep(0.1)

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def _get(self, path, timeout=60.0):
    # The unified forecast / flood-inundation routes query 4 live providers in parallel.
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
    self.assertIn("floodhub", model_ids)
    self.assertIn("glofas", model_ids)
    self.assertIn("geoglows", model_ids)

  def test_api_maas_gauges(self):
    status, data = self._get("/api/maas/gauges?min_lat=32.5&min_lon=-117.5&max_lat=33.0&max_lon=-117.0")
    self.assertEqual(status, 200)
    self.assertIn("gauges", data)
    self.assertGreater(data.get("count", 0), 0)

  def test_api_maas_forecast(self):
    status, data = self._get("/api/maas/forecast?lat=32.756&lon=-117.252&gauge_id=hybas_7120012340")
    self.assertEqual(status, 200)
    self.assertIn("models", data)
    self.assertIn("floodhub", data["models"])
    self.assertIn("glofas", data["models"])
    self.assertIn("geoglows", data["models"])

  def test_api_maas_watershed(self):
    # GloFAS cell endpoint
    status, data = self._get("/api/maas/watershed?lat=23.23&lon=90.64&fabric=glofas_cell")
    self.assertEqual(status, 200)
    self.assertEqual(data.get("type"), "Feature")
    self.assertEqual(data.get("properties", {}).get("fabric"), "glofas_cell")

    # HydroATLAS full endpoint
    status, data = self._get("/api/maas/watershed?lat=40.45&lon=-86.95&fabric=hydroatlas_full")
    self.assertEqual(status, 200)
    self.assertEqual(data.get("type"), "Feature")
    self.assertEqual(data.get("properties", {}).get("fabric"), "hydroatlas_full")

  # ---------------------------------------------------------------------------
  # Phase 6 endpoints
  # ---------------------------------------------------------------------------

  def test_api_maas_models_lists_todays_earth(self):
    status, data = self._get("/api/maas/models")
    self.assertEqual(status, 200)
    te = [m for m in data["models"] if m["id"] == "todays_earth"]
    self.assertEqual(len(te), 1)
    self.assertIn(te[0].get("status"), ("emulated", "operational"))

  def test_api_maas_forecast_four_providers(self):
    status, data = self._get("/api/maas/forecast?lat=32.756&lon=-117.252&gauge_id=hybas_7120012340")
    self.assertEqual(status, 200)
    self.assertEqual(set(data["models"]), {"floodhub", "glofas", "geoglows", "todays_earth"})
    self.assertEqual([r["model"] for r in data["consensus"]], ["floodhub", "glofas", "geoglows", "todays_earth"])
    self.assertIn("overall_risk_level", data["flood_summary"])
    self.assertIn("todays_earth_cell", data["virtual_station"])
    self.assertIn("todays_earth", data["timeline"]["series"])
    self.assertTrue(data["flood_inundation"]["endpoint"].startswith("/api/maas/flood-inundation?"))

  def test_api_maas_forecast_model_subset(self):
    status, data = self._get("/api/maas/forecast?lat=32.756&lon=-117.252&models=glofas,todays_earth")
    self.assertEqual(status, 200)
    self.assertEqual([r["model"] for r in data["consensus"]], ["glofas", "todays_earth"])

  def test_api_maas_watershed_camaflood_unit(self):
    status, data = self._get("/api/maas/watershed?lat=23.23&lon=90.64&geofabric=camaflood_unit")
    self.assertEqual(status, 200)
    self.assertEqual(data.get("type"), "Feature")
    props = data.get("properties", {})
    self.assertEqual(props.get("geofabric"), "camaflood_unit")
    self.assertEqual(props.get("grid_cell_id"), "cama_025_23.125_90.625")
    self.assertGreater(props.get("area_km2", 0), 0)

  def test_api_maas_flood_inundation(self):
    status, data = self._get("/api/maas/flood-inundation?lat=32.756&lon=-117.252&gauge_id=hybas_7120012340")
    self.assertEqual(status, 200)
    self.assertEqual(data.get("type"), "FeatureCollection")
    self.assertGreater(len(data.get("features", [])), 0)
    self.assertEqual(set(data["metadata"]["layers"]), {"floodhub_extent", "camaflood_depth", "reach_exceedance"})
    for f in data["features"]:
      self.assertIn(f["properties"]["layer"], ("floodhub_extent", "camaflood_depth", "reach_exceedance"))

  def test_api_maas_flood_inundation_rejects_bad_coordinates(self):
    self.assertEqual(self._get_status("/api/maas/flood-inundation?lat=abc&lon=1"), 400)

  def test_api_maas_network_all_models(self):
    for model in ("floodhub", "glofas", "geoglows", "todays_earth"):
      status, data = self._get(f"/api/maas/network?model={model}&bbox=88.0,22.0,91.5,25.5&zoom=8")
      self.assertEqual(status, 200)
      self.assertEqual(data.get("type"), "FeatureCollection")
      self.assertEqual((data.get("properties") or {}).get("model"), model)
      self.assertGreater(len(data.get("features") or []), 0)

  def test_api_maas_cross_network_reach_matching_and_horizons(self):
    status, data = self._get(
        "/api/maas/forecast?lat=23.875&lon=89.875&network=todays_earth&upstream_area_km2=1480000"
    )
    self.assertEqual(status, 200)
    models = data.get("models") or {}
    tl_series = (data.get("timeline") or {}).get("series") or {}
    for m in ("floodhub", "glofas", "geoglows", "todays_earth"):
      self.assertIn(m, models)
      self.assertIn(m, tl_series)
      finite = [v for v in (tl_series[m].get("central") or []) if v is not None]
      self.assertGreater(len(finite), 0, m)
    te_points = ((models["todays_earth"].get("streamflow") or {}).get("data") or [])
    self.assertLessEqual(len(te_points), 6)
    self.assertGreaterEqual(len(models["glofas"].get("data") or []), 14)
    self.assertEqual(models["floodhub"].get("unit"), "CUBIC_METERS_PER_SECOND")
    self.assertEqual(tl_series["floodhub"].get("unit"), "m³/s")
    vs = data.get("virtual_station") or {}
    self.assertGreater((vs.get("glofas_cell") or {}).get("upstream_area_km2") or 0, 500000)
    self.assertGreater((vs.get("geoglows_reach") or {}).get("upstream_area_km2") or 0, 500000)


if __name__ == "__main__":
  unittest.main()




