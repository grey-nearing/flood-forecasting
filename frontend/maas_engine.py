#!/usr/bin/env python3
"""Models-as-a-Service (MaaS) Engine for Earthkit Hydro Web.

Integrates real-time operational streamflow predictions from:
  1. Google FloodHub (FloodForecasting v1 API)
  2. GEOGLOWS ECMWF (v2 REST API)
  3. Copernicus GloFAS (Open-Meteo REST Pipeline & CEMS)
  4. JAXA Today's Earth (TE-Global MATSIRO + CaMa-Flood), with a deterministic
     CaMa-Flood emulator fallback when no TE-Global feed is reachable.

Features:
  - Viewport-bounded gauge search with severity mapping.
  - Lat/Lon reach snapping and coordinate-to-gauge discovery.
  - Synchronized multi-model forecast aggregation with unified units (m³/s).
  - Return period threshold normalization (2-yr, 5-yr, 20-yr).
  - In-memory thread-safe TTL caching to respect external API limits.
  - Robust offline fallback generation ensuring UI resilience.
  - Model-specific return periods (GEOGLOWS v2 and GloFAS v4 reanalysis EV1
    fits) with persistent SQLite caching.
  - Spatial flood inundation layers: FloodHub inundation maps, CaMa-Flood
    floodplain depth / flooded-fraction cells, and reach return-period
    exceedance corridors.
"""

import concurrent.futures
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

try:
  from frontend.config import DATA_DIR
  from frontend.delineator import HydroDelineator, _find_merit_shp
except ImportError:
  try:
    from config import DATA_DIR
    from delineator import HydroDelineator, _find_merit_shp
  except ImportError:
    DATA_DIR = Path.home() / ".cache" / "openhydronet" / "data"
    HydroDelineator = None
    _find_merit_shp = None

logger = logging.getLogger(__name__)

# Default API Key read from environment for FloodForecasting v1 API
DEFAULT_FLOODHUB_KEY = os.environ.get("FLOODHUB_API_KEY", "")
FLOODHUB_BASE_URL = "https://floodforecasting.googleapis.com/v1"
GEOGLOWS_BASE_URL = "https://geoglows.ecmwf.int/api/v2"
GLOFAS_BASE_URL = "https://flood-api.open-meteo.com/v1/flood"

# JAXA Earth API public STAC catalog (Cloud-Optimized GeoTIFF collections).
# It is probed for a Today's Earth (TE-Global CaMa-Flood) collection; none is
# published there as of 2026-09 (the documented `/api/stac/v1` path is 404).
JAXA_STAC_CATALOG_URL = "https://data.earth.jaxa.jp/stac/cog/v1/catalog.json"
# Optional operator-provided TE-Global point-forecast endpoint. The JSON
# contract is documented in `fetch_todays_earth_forecast()`.
TODAYS_EARTH_API_URL_ENV = "TODAYS_EARTH_API_URL"
TODAYS_EARTH_TIMEOUT_S = 6
TODAYS_EARTH_SOURCE = "JAXA Today's Earth (TE-Global CaMa-Flood)"
CAMA_GRID_RES_DEG = 0.25
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

WATERSHED_CACHE_DB = DATA_DIR / "cache" / "maas_watershed_cache.sqlite"
FLOOD_CACHE_DB = DATA_DIR / "cache" / "maas_flood_cache.sqlite"

# In-memory thread-safe TTL cache
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _get_floodhub_key() -> str:
  return os.environ.get("FLOODHUB_API_KEY", DEFAULT_FLOODHUB_KEY)


def get_cached(key: str) -> Optional[Any]:
  with _CACHE_LOCK:
    entry = _CACHE.get(key)
    if entry and time.time() < entry["expires_at"]:
      return entry["value"]
    elif entry:
      del _CACHE[key]
  return None


def set_cached(key: str, value: Any, ttl_seconds: int = 3600) -> None:
  with _CACHE_LOCK:
    _CACHE[key] = {
        "value": value,
        "expires_at": time.time() + ttl_seconds,
    }


# Synthetic offline fallbacks are cached only briefly, so a transient upstream
# failure does not pin generated series (and their flood classes) for an hour.
FALLBACK_CACHE_TTL_S = 120


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
  """Performs an HTTP GET request and returns parsed JSON."""
  if params:
    query_str = urllib.parse.urlencode(params)
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}{query_str}"

  req = urllib.request.Request(
      url,
      headers={
          "User-Agent": "EarthkitHydroWeb/1.0",
          "Accept": "application/json",
      },
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout) as response:
      if response.status == 200:
        data = response.read().decode("utf-8")
        return json.loads(data)
  except Exception as e:
    logger.debug("HTTP GET failed for %s: %s", url, e)
  return None


def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 10) -> Optional[Dict[str, Any]]:
  """Performs an HTTP POST request with JSON body."""
  data_bytes = json.dumps(payload).encode("utf-8")
  req = urllib.request.Request(
      url,
      data=data_bytes,
      headers={
          "User-Agent": "EarthkitHydroWeb/1.0",
          "Content-Type": "application/json",
          "Accept": "application/json",
      },
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout) as response:
      if response.status == 200:
        data = response.read().decode("utf-8")
        return json.loads(data)
  except Exception as e:
    logger.debug("HTTP POST failed for %s: %s", url, e)
  return None


# ---------------------------------------------------------------------------
# 1. Google FloodHub API Client
# ---------------------------------------------------------------------------

def fetch_floodhub_gauges_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    page_size: int = 100,
    include_non_verified: bool = True,
) -> List[Dict[str, Any]]:
  """Fetches active FloodHub gauges and their real-time flood status for a viewport bounding box."""
  cache_key = f"fh_bbox_{min_lat:.2f}_{min_lon:.2f}_{max_lat:.2f}_{max_lon:.2f}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached

  key = _get_floodhub_key()
  url = f"{FLOODHUB_BASE_URL}/floodStatus:searchLatestFloodStatusByArea?key={key}"

  # Construct a counter-clockwise spherical polygon loop
  payload = {
      "loop": {
          "vertices": [
              {"latitude": min_lat, "longitude": min_lon},
              {"latitude": max_lat, "longitude": min_lon},
              {"latitude": max_lat, "longitude": max_lon},
              {"latitude": min_lat, "longitude": max_lon},
          ]
      },
      "pageSize": page_size,
      "includeNonQualityVerified": include_non_verified,
  }

  res = _http_post_json(url, payload, timeout=12)
  gauges: List[Dict[str, Any]] = []

  if res and "floodStatuses" in res:
    for status in res["floodStatuses"]:
      loc = status.get("gaugeLocation", {})
      gid = status.get("gaugeId", "")
      severity = status.get("severity", "UNKNOWN")
      gauges.append({
          "gauge_id": gid,
          "lat": loc.get("latitude", 0.0),
          "lon": loc.get("longitude", 0.0),
          "severity": severity,
          "forecast_trend": status.get("forecastTrend", "NO_CHANGE"),
          "issued_time": status.get("issuedTime", ""),
          "quality_verified": status.get("qualityVerified", False),
          "source": status.get("source", "HYBAS"),
      })

  # Fallback: if external API fails, generate realistic sample gauges for testing
  if not gauges:
    gauges = _generate_sample_floodhub_gauges(min_lat, min_lon, max_lat, max_lon)

  set_cached(cache_key, gauges, ttl_seconds=900)  # 15 minutes TTL for gauges
  return gauges


def fetch_floodhub_forecast(gauge_id: str) -> Dict[str, Any]:
  """Queries the 7-day forecast and return period thresholds for a specific FloodHub gauge."""
  cache_key = f"fh_forecast_{gauge_id}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached

  key = _get_floodhub_key()
  now_utc = datetime.now(timezone.utc)
  start_str = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
  # `issuedTimeEnd` is a date bound: use tomorrow so today's issues are included.
  end_str = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")

  # 1. Fetch Forecast Timeseries
  f_url = f"{FLOODHUB_BASE_URL}/gauges:queryGaugeForecasts"
  f_params = {
      "key": key,
      "gaugeIds": gauge_id,
      "issuedTimeStart": start_str,
      "issuedTimeEnd": end_str,
  }

  f_res = _http_get_json(f_url, f_params, timeout=12)
  forecast_series: List[Dict[str, Any]] = []
  latest_issue_time = ""
  fallback_reason = None
  nan_issues_skipped = 0

  if f_res and "forecasts" in f_res and gauge_id in f_res["forecasts"]:
    g_data = f_res["forecasts"][gauge_id]
    fcasts = g_data.get("forecasts", [])
    # FloodHub encodes missing values as the string "NaN". Use the most recent
    # issue that still carries finite values, provided it covers the last 24 h.
    fresh_cutoff = (now_utc - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for issue in reversed(fcasts):
      points = []
      for rng in issue.get("forecastRanges", []):
        start_t = rng.get("forecastStartTime", "")
        val = _safe_float_or_none(rng.get("value"))
        if start_t and val is not None:
          points.append({"time": start_t, "discharge": round(val, 2)})
      if not points:
        nan_issues_skipped += 1
        continue
      if max(p["time"] for p in points) >= fresh_cutoff:
        forecast_series = points
        latest_issue_time = issue.get("issuedTime", "")
      else:
        fallback_reason = f"latest finite FloodHub forecast ({issue.get('issuedTime', '')}) is stale"
      break
    if not forecast_series and not fallback_reason:
      fallback_reason = "FloodHub forecast values are all NaN" if fcasts else "no FloodHub forecasts issued"
  else:
    fallback_reason = "FloodHub forecast API unavailable"

  # 2. Fetch Model Metadata & Return Period Thresholds
  m_url = f"{FLOODHUB_BASE_URL}/gaugeModels:batchGet?key={key}&names=gaugeModels/{gauge_id}"
  m_res = _http_get_json(m_url, timeout=10)
  thresholds = {"warning_2yr": None, "danger_5yr": None, "extreme_20yr": None}
  unit = "CUBIC_METERS_PER_SECOND"

  if m_res and "gaugeModels" in m_res and m_res["gaugeModels"]:
    m_info = m_res["gaugeModels"][0]
    unit = m_info.get("gaugeValueUnit", "CUBIC_METERS_PER_SECOND")
    th = m_info.get("thresholds", {})
    for out_key, api_key in (
        ("warning_2yr", "warningLevel"), ("danger_5yr", "dangerLevel"), ("extreme_20yr", "extremeDangerLevel"),
    ):
      val = _safe_float_or_none(th.get(api_key))
      thresholds[out_key] = round(val, 2) if val is not None else None

  # Fallback if external API returned empty (synthetic series + matching synthetic thresholds).
  status = "live"
  if not forecast_series:
    forecast_series, thresholds = _generate_sample_floodhub_forecast(gauge_id)
    latest_issue_time = now_utc.strftime("%Y-%m-%dT%H:00:00Z")
    status = "fallback"

  result = {
      "model": "google_floodhub",
      "available": True,
      "status": status,
      "gauge_id": gauge_id,
      "issued_time": latest_issue_time,
      "unit": unit,
      "thresholds": thresholds,
      "data": forecast_series,
      "fallback_reason": fallback_reason if status == "fallback" else None,
      "nan_issues_skipped": nan_issues_skipped,
  }

  set_cached(cache_key, result, ttl_seconds=3600 if status == "live" else FALLBACK_CACHE_TTL_S)
  return result


# ---------------------------------------------------------------------------
# 2. GEOGLOWS ECMWF v2 API Client
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
  try:
    if val is None or val == "":
      return default
    return float(val)
  except (ValueError, TypeError):
    return default


def _safe_float_or_none(val: Any) -> Optional[float]:
  """Parses a numeric value, returning None for blanks, NaN/inf, or junk."""
  try:
    if val is None or val == "":
      return None
    f = float(val)
    return f if math.isfinite(f) else None
  except (ValueError, TypeError):
    return None


def fetch_geoglows_river_id(lat: float, lon: float) -> Optional[int]:
  """Queries GEOGLOWS /getriverid to snap coordinates to a 9-digit COMID reach."""
  cache_reach_key = f"gg_reach_{lat:.3f}_{lon:.3f}"
  river_id = get_cached(cache_reach_key)
  if river_id is not None:
    try:
      return int(river_id)
    except (ValueError, TypeError):
      pass

  reach_res = _http_get_json(f"{GEOGLOWS_BASE_URL}/getriverid", {"lat": lat, "lon": lon}, timeout=8)
  if reach_res and "river_id" in reach_res:
    try:
      rid = int(reach_res["river_id"])
      set_cached(cache_reach_key, rid, ttl_seconds=86400 * 30)
      return rid
    except (ValueError, TypeError):
      pass
  return None


def fetch_geoglows_forecast(lat: float, lon: float, river_id: Optional[int] = None) -> Dict[str, Any]:
  """Fetches the 15-day hourly streamflow forecast from GEOGLOWS ECMWF Streamflow Service."""
  # 1. Resolve river_id if not supplied
  if not river_id:
    river_id = fetch_geoglows_river_id(lat, lon)

  if not river_id:
    river_id = int(abs(lat * 10000) + abs(lon * 10000))

  cache_key = f"gg_forecast_{river_id}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached

  # 2. Fetch forecaststats in JSON format
  f_url = f"{GEOGLOWS_BASE_URL}/forecaststats/{river_id}"
  f_res = _http_get_json(f_url, {"format": "json"}, timeout=12)

  records: List[Dict[str, Any]] = []
  if f_res and "datetime" in f_res:
    times = f_res.get("datetime", [])
    meds = f_res.get("flow_med", [])
    avgs = f_res.get("flow_avg", [])
    maxs = f_res.get("flow_max", [])
    mins = f_res.get("flow_min", [])
    p25s = f_res.get("flow_25p", [])
    p75s = f_res.get("flow_75p", [])
    high_res = f_res.get("high_res", [])

    for i in range(len(times)):
      med_raw = meds[i] if i < len(meds) else None
      avg_raw = avgs[i] if i < len(avgs) else None
      # Upstream publishes the 51-member ensemble statistics only at 3-hourly
      # steps; intermediate hourly rows carry blank strings that would
      # otherwise be coerced into spurious 0.0 m³/s values. Skip them.
      if _safe_float_or_none(med_raw) is None and _safe_float_or_none(avg_raw) is None:
        continue
      records.append({
          "time": times[i],
          "flow_med": round(_safe_float(med_raw), 2),
          "flow_avg": round(_safe_float(avg_raw), 2),
          "flow_max": round(_safe_float(maxs[i] if i < len(maxs) else None), 2),
          "flow_min": round(_safe_float(mins[i] if i < len(mins) else None), 2),
          "flow_25p": round(_safe_float(p25s[i] if i < len(p25s) else None), 2),
          "flow_75p": round(_safe_float(p75s[i] if i < len(p75s) else None), 2),
          "high_res": round(_safe_float(high_res[i] if i < len(high_res) else None), 2),
      })

  # Fallback if GEOGLOWS upstream is slow/unreachable
  status = "live"
  if not records:
    records = _generate_sample_geoglows_forecast(river_id)
    status = "fallback"

  result = {
      "model": "geoglows",
      "available": True,
      "status": status,
      "river_id": river_id,
      "unit": "CUBIC_METERS_PER_SECOND",
      "data": records,
  }

  set_cached(cache_key, result, ttl_seconds=3600 if status == "live" else FALLBACK_CACHE_TTL_S)
  return result


# ---------------------------------------------------------------------------
# 3. Copernicus GloFAS API Client
# ---------------------------------------------------------------------------

def fetch_glofas_forecast(lat: float, lon: float, forecast_days: int = 15) -> Dict[str, Any]:
  """Fetches 15-day GloFAS ensemble forecast from the low-latency Open-Meteo GloFAS pipeline."""
  cache_key = f"glofas_forecast_{lat:.3f}_{lon:.3f}_{forecast_days}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached

  params = {
      "latitude": lat,
      "longitude": lon,
      "daily": "river_discharge,river_discharge_mean,river_discharge_median,river_discharge_max,river_discharge_min,river_discharge_p25,river_discharge_p75",
      "forecast_days": forecast_days,
  }

  res = _http_get_json(GLOFAS_BASE_URL, params, timeout=10)
  records: List[Dict[str, Any]] = []

  if res and "daily" in res:
    daily = res["daily"]
    times = daily.get("time", [])
    means = daily.get("river_discharge_mean", [])
    meds = daily.get("river_discharge_median", [])
    maxs = daily.get("river_discharge_max", [])
    mins = daily.get("river_discharge_min", [])
    p25s = daily.get("river_discharge_p25", [])
    p75s = daily.get("river_discharge_p75", [])

    for i in range(len(times)):
      records.append({
          "time": times[i],
          "discharge_mean": round(means[i], 2) if i < len(means) and means[i] is not None else 0.0,
          "discharge_median": round(meds[i], 2) if i < len(meds) and meds[i] is not None else 0.0,
          "discharge_max": round(maxs[i], 2) if i < len(maxs) and maxs[i] is not None else 0.0,
          "discharge_min": round(mins[i], 2) if i < len(mins) and mins[i] is not None else 0.0,
          "discharge_p25": round(p25s[i], 2) if i < len(p25s) and p25s[i] is not None else 0.0,
          "discharge_p75": round(p75s[i], 2) if i < len(p75s) and p75s[i] is not None else 0.0,
      })

  status = "live"
  if not records:
    records = _generate_sample_glofas_forecast(lat, lon, forecast_days)
    status = "fallback"

  result = {
      "model": "copernicus_glofas",
      "available": True,
      "status": status,
      "lat": lat,
      "lon": lon,
      "unit": "CUBIC_METERS_PER_SECOND",
      "data": records,
  }

  set_cached(cache_key, result, ttl_seconds=3600 if status == "live" else FALLBACK_CACHE_TTL_S)
  return result


# ---------------------------------------------------------------------------
# 4. Multi-Model Aggregator & Virtual Station Resolver
# ---------------------------------------------------------------------------

def aggregate_maas_forecast(
    lat: float,
    lon: float,
    gauge_id: Optional[str] = None,
    river_id: Optional[int] = None,
    requested_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Aggregates streamflow forecasts across FloodHub, GEOGLOWS, and GloFAS into a unified response."""
  if not requested_models:
    requested_models = ["floodhub", "geoglows", "glofas"]

  # Auto-resolve nearest FloodHub gauge if not provided
  if "floodhub" in requested_models and not gauge_id:
    nearby_gauges = fetch_floodhub_gauges_bbox(lat - 0.25, lon - 0.25, lat + 0.25, lon + 0.25)
    if nearby_gauges:
      gauge_id = nearby_gauges[0]["gauge_id"]

  models_output: Dict[str, Any] = {}
  master_thresholds = {"warning_2yr": None, "danger_5yr": None, "extreme_20yr": None}

  # 1. FloodHub
  if "floodhub" in requested_models:
    if gauge_id:
      fh_res = fetch_floodhub_forecast(gauge_id)
      models_output["floodhub"] = fh_res
      if fh_res.get("thresholds"):
        master_thresholds = fh_res["thresholds"]
    else:
      models_output["floodhub"] = {"available": False, "message": "No FloodHub gauge near coordinates."}

  # 2. GEOGLOWS
  if "geoglows" in requested_models:
    gg_res = fetch_geoglows_forecast(lat, lon, river_id=river_id)
    models_output["geoglows"] = gg_res

  # 3. GloFAS
  if "glofas" in requested_models:
    gl_res = fetch_glofas_forecast(lat, lon, forecast_days=15)
    models_output["glofas"] = gl_res

  # Provide fallback thresholds if FloodHub didn't supply them
  if master_thresholds["warning_2yr"] is None:
    base_flow = 15.0
    master_thresholds = {
        "warning_2yr": round(base_flow * 2.5, 1),
        "danger_5yr": round(base_flow * 5.0, 1),
        "extreme_20yr": round(base_flow * 9.5, 1),
    }

  return {
      "location": {
          "lat": lat,
          "lon": lon,
          "gauge_id": gauge_id,
          "river_id": river_id,
      },
      "thresholds": master_thresholds,
      "models": models_output,
  }


# ---------------------------------------------------------------------------
# 4b. Return-Period Thresholds (GEOGLOWS v2 & GloFAS v4) + Persistent Cache
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015329
RETURN_PERIOD_YEARS = (2, 5, 10, 20, 25, 50, 100)
# Index-flood regionalisation, used only when no reanalysis record is reachable:
# mean annual flood ~= 3.3 x reference flow, annual-maximum CV ~= 0.45.
_INDEX_FLOOD_MAF_RATIO = 3.3
_INDEX_FLOOD_CV = 0.45
GEOGLOWS_RP_OFFICIAL_TIMEOUT_S = 4
GEOGLOWS_RP_UNHEALTHY_BACKOFF_S = 1800
GEOGLOWS_RETRO_TIMEOUT_S = 60
_GEOGLOWS_RP_ENDPOINT_STATE: Dict[str, Any] = {"unhealthy_until": 0.0, "last_error": None}
_RP_JOBS_LOCK = threading.Lock()
_RP_JOBS_IN_FLIGHT: Dict[int, threading.Thread] = {}
_FLOOD_CACHE_LOCK = threading.Lock()
_FLOOD_CACHE_STATE = {"ready": False}


def _http_get_json_with_error(
    url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10
) -> Tuple[Optional[Any], Optional[str]]:
  """Performs an HTTP GET returning (parsed JSON or None, short error text or None)."""
  if params:
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}{urllib.parse.urlencode(params)}"
  req = urllib.request.Request(
      url, headers={"User-Agent": "EarthkitHydroWeb/1.0", "Accept": "application/json"}
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout) as response:
      return json.loads(response.read().decode("utf-8")), None
  except urllib.error.HTTPError as e:
    detail = ""
    try:
      detail = e.read().decode("utf-8", "replace").strip()
    except Exception:
      pass
    return None, f"HTTP {e.code} {detail}"[:300].strip()
  except Exception as e:
    return None, f"{type(e).__name__}: {e}"[:300]


def _init_flood_cache_db() -> bool:
  """Creates the persistent flood / return-period cache table (idempotent)."""
  if _FLOOD_CACHE_STATE["ready"]:
    return True
  with _FLOOD_CACHE_LOCK:
    if not _FLOOD_CACHE_STATE["ready"]:
      try:
        FLOOD_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(FLOOD_CACHE_DB), timeout=5)) as conn:
          conn.execute(
              "CREATE TABLE IF NOT EXISTS flood_cache ("
              "cache_key TEXT PRIMARY KEY, payload TEXT, created_at REAL)"
          )
          conn.commit()
        _FLOOD_CACHE_STATE["ready"] = True
      except Exception as e:
        logger.debug("Failed to initialize flood SQLite cache: %s", e)
  return _FLOOD_CACHE_STATE["ready"]


def _flood_cache_get(cache_key: str, max_age_s: float) -> Optional[Dict[str, Any]]:
  """Reads a JSON payload from the persistent flood cache if it is fresh enough."""
  try:
    if not FLOOD_CACHE_DB.exists():
      return None
    with closing(sqlite3.connect(str(FLOOD_CACHE_DB), timeout=5)) as conn:
      row = conn.execute(
          "SELECT payload, created_at FROM flood_cache WHERE cache_key = ?", (cache_key,)
      ).fetchone()
    if row and row[0] and (time.time() - float(row[1] or 0.0)) <= max_age_s:
      payload = json.loads(row[0])
      return payload if isinstance(payload, dict) else None
  except Exception as e:
    logger.debug("Flood SQLite cache read failed for %s: %s", cache_key, e)
  return None


def _flood_cache_put(cache_key: str, payload: Dict[str, Any]) -> None:
  """Persists a JSON-serialisable payload in the flood cache."""
  if not _init_flood_cache_db():
    return
  try:
    with closing(sqlite3.connect(str(FLOOD_CACHE_DB), timeout=5)) as conn:
      conn.execute(
          "INSERT OR REPLACE INTO flood_cache (cache_key, payload, created_at) VALUES (?, ?, ?)",
          (cache_key, json.dumps(payload), time.time()),
      )
      conn.commit()
  except Exception as e:
    logger.debug("Flood SQLite cache write failed for %s: %s", cache_key, e)


def _gumbel_frequency_factor(return_period_yrs: float) -> float:
  """EV1 (Gumbel) frequency factor K_T for the method of moments (Chow, 1951)."""
  t = float(return_period_yrs)
  return -(math.sqrt(6.0) / math.pi) * (_EULER_GAMMA + math.log(math.log(t / (t - 1.0))))


def _annual_maxima(times: List[Any], values: List[Any], min_valid_days: int = 300) -> List[float]:
  """Calendar-year maxima of a daily series (years with >= min_valid_days valid values)."""
  by_year: Dict[str, List[float]] = {}
  for t, v in zip(times, values):
    fv = _safe_float_or_none(v)
    if fv is None or fv < 0:
      continue
    year = str(t)[:4]
    if year.isdigit():
      by_year.setdefault(year, []).append(fv)
  return [max(vals) for _, vals in sorted(by_year.items()) if len(vals) >= min_valid_days]


def _gumbel_return_periods(annual_maxima: List[float]) -> Optional[Dict[str, float]]:
  """Fits EV1 by moments to annual maxima; returns {'return_period_T': Q_T} in m³/s."""
  n = len(annual_maxima)
  if n < 8:
    return None
  mean = sum(annual_maxima) / n
  if mean <= 1e-3:
    return None
  std = math.sqrt(max(sum((x - mean) ** 2 for x in annual_maxima) / (n - 1), 0.0))
  return {
      f"return_period_{t}": round(max(mean + _gumbel_frequency_factor(t) * std, 0.0), 2)
      for t in RETURN_PERIOD_YEARS
  }


def _ev1_fit_line(rps: Dict[str, Any]) -> Optional[Tuple[float, float]]:
  """Least-squares EV1 line Q = a + b K_T through known `return_period_T` levels."""
  pts = []
  for key, val in (rps or {}).items():
    if not str(key).startswith("return_period_"):
      continue
    try:
      t = float(str(key).rsplit("_", 1)[1])
    except ValueError:
      continue
    fv = _safe_float_or_none(val)
    if fv is not None and t > 1.0:
      pts.append((_gumbel_frequency_factor(t), fv))
  if len(pts) < 2:
    return None
  n = len(pts)
  mean_k = sum(p[0] for p in pts) / n
  mean_q = sum(p[1] for p in pts) / n
  sxx = sum((p[0] - mean_k) ** 2 for p in pts)
  if sxx <= 1e-12:
    return None
  slope = sum((p[0] - mean_k) * (p[1] - mean_q) for p in pts) / sxx
  return mean_q - slope * mean_k, slope


def _gumbel_quantile_from_return_periods(rps: Dict[str, Any], return_period_yrs: float) -> Optional[float]:
  """Interpolates Q_T from known return levels via the EV1 line Q = a + b K_T."""
  line = _ev1_fit_line(rps)
  if line is None:
    return None
  return round(max(line[0] + line[1] * _gumbel_frequency_factor(return_period_yrs), 0.0), 2)


def _known_return_levels(rps: Dict[str, Any]) -> List[Tuple[float, float]]:
  """Sorted, strictly increasing (T, Q_T) pairs (T >= 2 yr, Q_T > 0) from `return_period_T` keys."""
  pts = []
  for key, val in (rps or {}).items():
    if not str(key).startswith("return_period_"):
      continue
    try:
      t = float(str(key).rsplit("_", 1)[1])
    except ValueError:
      continue
    fv = _safe_float_or_none(val)
    if fv is not None and fv > 0 and t >= 2.0:
      pts.append((t, fv))
  levels: List[Tuple[float, float]] = []
  for t, q in sorted(pts):
    if not levels or q > levels[-1][1]:
      levels.append((t, q))
  return levels


def _estimate_return_period_yrs(value: Optional[float], rps: Dict[str, Any]) -> Optional[float]:
  """Estimates the return period (years) of a flow value from the model's known return levels.

  The EV1 reduced variate K_T is interpolated piecewise-linearly between consecutive known
  levels, so the estimate is exact at each level and consistent with `_classify_exceedance`;
  above the highest level it is extrapolated along the last segment (capped at 1000 yr).
  Flows below the lowest (2-yr) level return None ("< 2-yr"): the levels describe annual
  maxima, and inverting the unbounded EV1 lower tail there gives misleading 1-2 yr values
  (e.g. ~1.6-yr for 0.6 m³/s where the 2-yr level is 48 m³/s).
  """
  if value is None:
    return None
  levels = _known_return_levels(rps)
  v = float(value)
  if len(levels) < 2 or v < levels[0][1]:
    return None
  ks = [(_gumbel_frequency_factor(t), q) for t, q in levels]
  (k0, q0), (k1, q1) = ks[-2], ks[-1]
  for (ka, qa), (kb, qb) in zip(ks, ks[1:]):
    if v <= qb:
      (k0, q0), (k1, q1) = (ka, qa), (kb, qb)
      break
  k = k0 + (v - q0) * (k1 - k0) / (q1 - q0)
  u = -(math.pi / math.sqrt(6.0)) * k - _EULER_GAMMA
  if u > 30.0:
    return 1.0
  if u < -30.0:
    return 1000.0
  denom = 1.0 - math.exp(-math.exp(u))
  if denom <= 1e-12:
    return 1000.0
  return round(min(max(1.0 / denom, 1.0), 1000.0), 1)


def _scaled_return_periods(reference_flow: float) -> Dict[str, float]:
  """Deterministic index-flood return periods scaled from a reference flow (m³/s)."""
  maf = max(float(reference_flow or 0.0), 0.1) * _INDEX_FLOOD_MAF_RATIO
  return {
      f"return_period_{t}": round(max(maf * (1.0 + _gumbel_frequency_factor(t) * _INDEX_FLOOD_CV), 0.0), 2)
      for t in RETURN_PERIOD_YEARS
  }


def _thresholds_from_return_periods(rps: Dict[str, Any], source: str) -> Dict[str, Any]:
  """Maps return-period levels onto the MaaS 2/5/20/100-yr threshold contract."""
  return {
      "warning_2yr": rps.get("return_period_2"),
      "danger_5yr": rps.get("return_period_5"),
      "extreme_20yr": rps.get("return_period_20"),
      "extreme_100yr": rps.get("return_period_100"),
      "source": source,
      "unit": "m³/s",
  }


def _series_median(values: List[Any]) -> Optional[float]:
  valid = sorted(v for v in (_safe_float_or_none(x) for x in values) if v is not None and v >= 0)
  return valid[len(valid) // 2] if valid else None


def _glofas_cell_center(lat: float, lon: float, res: float = 0.05) -> Tuple[float, float]:
  """Centre of the GloFAS v4 0.05° grid cell containing (lat, lon)."""
  return (
      round(math.floor(lat / res) * res + res / 2.0, 3),
      round(math.floor(lon / res) * res + res / 2.0, 3),
  )


def fetch_glofas_return_periods(lat: float, lon: float) -> Dict[str, Any]:
  """GloFAS v4 return periods (m³/s) from an EV1 fit to the 1984-onward reanalysis.

  The daily reanalysis is read from the Open-Meteo GloFAS archive, cached in
  memory and persisted to SQLite (90 days). If the archive is unreachable, an
  index-flood scaling of the GloFAS forecast median is returned with
  `status="fallback"`.
  """
  cell_lat, cell_lon = _glofas_cell_center(lat, lon)
  cache_key = f"glofas_rp_{cell_lat:.3f}_{cell_lon:.3f}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached
  persisted = _flood_cache_get(cache_key, max_age_s=90 * 86400)
  if persisted:
    set_cached(cache_key, persisted, ttl_seconds=86400)
    return persisted

  end_year = datetime.now(timezone.utc).year - 1
  res = _http_get_json(
      GLOFAS_BASE_URL,
      {
          "latitude": lat,
          "longitude": lon,
          "daily": "river_discharge",
          "start_date": "1984-01-01",
          "end_date": f"{end_year}-12-31",
      },
      timeout=10,
  )
  if isinstance(res, dict) and isinstance(res.get("daily"), dict):
    daily = res["daily"]
    values = daily.get("river_discharge") or []
    annual_max = _annual_maxima(daily.get("time") or [], values)
    rps = _gumbel_return_periods(annual_max)
    if rps:
      valid = [v for v in (_safe_float_or_none(x) for x in values) if v is not None and v >= 0]
      result = {
          "provider": "glofas",
          "status": "live",
          "source": f"Copernicus GloFAS v4 reanalysis (Open-Meteo, 1984-{end_year})",
          "method": "EV1 (Gumbel, method of moments) fit to calendar-year maxima",
          "years_of_record": len(annual_max),
          "mean_flow": round(sum(valid) / len(valid), 3) if valid else None,
          "grid_lat": res.get("latitude"),
          "grid_lon": res.get("longitude"),
          "unit": "m³/s",
          **rps,
      }
      set_cached(cache_key, result, ttl_seconds=86400)
      _flood_cache_put(cache_key, result)
      return result

  forecast = fetch_glofas_forecast(lat, lon, forecast_days=15)
  ref = _series_median([r.get("discharge_median") for r in forecast.get("data", [])]) or 10.0
  result = {
      "provider": "glofas",
      "status": "fallback",
      "source": "Index-flood scaling of the GloFAS forecast median (reanalysis unreachable)",
      "method": f"Q_T = {_INDEX_FLOOD_MAF_RATIO} Q_ref (1 + {_INDEX_FLOOD_CV} K_T)",
      "years_of_record": 0,
      "mean_flow": round(ref, 3),
      "grid_lat": None,
      "grid_lon": None,
      "unit": "m³/s",
      **_scaled_return_periods(ref),
  }
  set_cached(cache_key, result, ttl_seconds=900)
  return result


def _is_geoglows_river_id(value: Any) -> bool:
  """GEOGLOWS v2 (TDX-Hydro) river IDs are 9-digit integers."""
  try:
    v = int(value)
  except (TypeError, ValueError):
    return False
  return 100_000_000 <= v <= 999_999_999


def _parse_geoglows_return_periods(payload: Any, river_id: int) -> Optional[Dict[str, float]]:
  """Tolerant parser for the GEOGLOWS v2 `/returnperiods` JSON variants."""
  if not isinstance(payload, dict):
    return None
  out: Dict[str, float] = {}

  def _take(period: Any, value: Any) -> None:
    if isinstance(value, dict):
      value = value.get(str(river_id), next(iter(value.values()), None) if value else None)
    if isinstance(value, list):
      value = value[0] if value else None
    try:
      t = int(float(str(period).strip().lower().replace("return_period_", "")))
    except ValueError:
      return
    fv = _safe_float_or_none(value)
    if fv is not None and t in (2, 5, 10, 25, 50, 100):
      out.setdefault(f"return_period_{t}", round(fv, 2))

  candidates = [payload]
  for key in (str(river_id), "return_periods", "returnperiods", "gumbel", "data"):
    if isinstance(payload.get(key), dict):
      candidates.append(payload[key])
  for cand in candidates:
    for key, val in cand.items():
      if str(key).startswith("return_period_") or str(key).strip().isdigit():
        _take(key, val)
  periods = payload.get("return_period")
  values = payload.get("gumbel") if payload.get("gumbel") is not None else payload.get("logpearson3")
  if not out and isinstance(periods, list) and isinstance(values, list):
    for period, value in zip(periods, values):
      _take(period, value)
  if len(out) < 3:
    return None
  if "return_period_20" not in out:
    q20 = _gumbel_quantile_from_return_periods(out, 20)
    if q20 is not None:
      out["return_period_20"] = q20
  return out


def _compute_geoglows_rp_from_retrospective(river_id: int) -> None:
  """Background job: EV1 fit to the reach's GEOGLOWS v2 daily retrospective record."""
  cache_key = f"geoglows_rp_{river_id}"
  try:
    res = _http_get_json(
        f"{GEOGLOWS_BASE_URL}/retrospectivedaily/{river_id}",
        {"format": "json"},
        timeout=GEOGLOWS_RETRO_TIMEOUT_S,
    )
    rps = None
    if isinstance(res, dict):
      times = res.get("datetime") or []
      values = res.get(str(river_id))
      if not isinstance(values, list):
        values = next(
            (v for k, v in res.items() if k not in ("datetime", "metadata") and isinstance(v, list)),
            None,
        )
      if isinstance(values, list) and times:
        annual_max = _annual_maxima(times, values)
        rps = _gumbel_return_periods(annual_max)
    if rps:
      valid = [v for v in (_safe_float_or_none(x) for x in values) if v is not None and v >= 0]
      meta = res.get("metadata") if isinstance(res.get("metadata"), dict) else {}
      result = {
          "provider": "geoglows",
          "status": "computed",
          "river_id": river_id,
          "source": "GEOGLOWS v2 retrospective simulation (EV1 fit computed locally)",
          "method": "EV1 (Gumbel, method of moments) fit to calendar-year maxima of retrospectivedaily",
          "years_of_record": len(annual_max),
          "record_start": meta.get("start_date") or str(times[0])[:10],
          "record_end": meta.get("end_date") or str(times[-1])[:10],
          "mean_flow": round(sum(valid) / len(valid), 3) if valid else None,
          "unit": "m³/s",
          "official_endpoint_error": _GEOGLOWS_RP_ENDPOINT_STATE.get("last_error"),
          **rps,
      }
      set_cached(cache_key, result, ttl_seconds=86400)
      _flood_cache_put(cache_key, result)
    else:
      set_cached(f"geoglows_rp_fail_{river_id}", True, ttl_seconds=1800)
  except Exception as e:
    logger.debug("GEOGLOWS retrospective return-period job failed for %s: %s", river_id, e)
    set_cached(f"geoglows_rp_fail_{river_id}", True, ttl_seconds=1800)
  finally:
    with _RP_JOBS_LOCK:
      _RP_JOBS_IN_FLIGHT.pop(river_id, None)


def _schedule_geoglows_rp_job(river_id: int) -> Optional[threading.Thread]:
  """Starts (or returns the in-flight) background retrospective EV1 job for a reach."""
  if get_cached(f"geoglows_rp_fail_{river_id}"):
    return None
  with _RP_JOBS_LOCK:
    job = _RP_JOBS_IN_FLIGHT.get(river_id)
    if job is None:
      job = threading.Thread(
          target=_compute_geoglows_rp_from_retrospective,
          args=(river_id,),
          name=f"geoglows-rp-{river_id}",
          daemon=True,
      )
      _RP_JOBS_IN_FLIGHT[river_id] = job
      job.start()
  return job


def _geoglows_fallback_return_periods(river_id: Any, reference_flow: Optional[float] = None) -> Dict[str, Any]:
  """Deterministic GEOGLOWS thresholds from an index-flood scaling (never cached)."""
  try:
    rid = int(river_id)
  except (TypeError, ValueError):
    rid = 0
  ref = reference_flow if reference_flow and reference_flow > 0 else float(7.0 + abs(rid) % 20)
  with _RP_JOBS_LOCK:
    pending = rid in _RP_JOBS_IN_FLIGHT
  return {
      "provider": "geoglows",
      "status": "fallback",
      "river_id": rid or None,
      "source": "Index-flood scaling of the GEOGLOWS reach flow (official return periods unavailable)",
      "method": f"Q_T = {_INDEX_FLOOD_MAF_RATIO} Q_ref (1 + {_INDEX_FLOOD_CV} K_T)",
      "pending_computation": pending,
      "mean_flow": round(ref, 3),
      "unit": "m³/s",
      "official_endpoint_error": _GEOGLOWS_RP_ENDPOINT_STATE.get("last_error"),
      **_scaled_return_periods(ref),
  }


def fetch_geoglows_return_periods(
    river_id: Any, mean_flow: Optional[float] = None, wait_s: float = 0.0
) -> Dict[str, Any]:
  """GEOGLOWS v2 return-period thresholds (m³/s) for a river reach.

  Resolution order:
    1. In-memory TTL cache, then the persistent SQLite flood cache (180 days).
    2. The official `/returnperiods/{river_id}?format=json` endpoint (5 s
       timeout). It currently fails upstream (HTTP 500), so a failure marks it
       unhealthy for 30 minutes to protect request latency.
    3. A background EV1 fit to the reach's `/retrospectivedaily` record,
       persisted once complete (optionally awaited for `wait_s` seconds).
    4. A deterministic index-flood scaling of `mean_flow` (`status="fallback"`,
       `pending_computation=True` while step 3 runs).

  Returns:
    Dict with `return_period_{2,5,10,20,25,50,100}`, `status`
    ("live" | "computed" | "fallback"), `source`, `method` and `unit`.
  """
  try:
    rid = int(river_id)
  except (TypeError, ValueError):
    rid = 0
  if _is_geoglows_river_id(rid):
    cache_key = f"geoglows_rp_{rid}"
    cached = get_cached(cache_key)
    if cached is not None:
      return cached
    persisted = _flood_cache_get(cache_key, max_age_s=180 * 86400)
    if persisted:
      set_cached(cache_key, persisted, ttl_seconds=86400)
      return persisted

    now = time.time()
    if now >= float(_GEOGLOWS_RP_ENDPOINT_STATE.get("unhealthy_until") or 0.0):
      payload, err = _http_get_json_with_error(
          f"{GEOGLOWS_BASE_URL}/returnperiods/{rid}",
          {"format": "json"},
          timeout=GEOGLOWS_RP_OFFICIAL_TIMEOUT_S,
      )
      rps = _parse_geoglows_return_periods(payload, rid) if payload is not None else None
      if rps:
        result = {
            "provider": "geoglows",
            "status": "live",
            "river_id": rid,
            "source": "GEOGLOWS v2 official return periods (/returnperiods)",
            "method": "Official GEOGLOWS return-period dataset",
            "unit": "m³/s",
            **rps,
        }
        set_cached(cache_key, result, ttl_seconds=86400)
        _flood_cache_put(cache_key, result)
        return result
      _GEOGLOWS_RP_ENDPOINT_STATE["unhealthy_until"] = now + GEOGLOWS_RP_UNHEALTHY_BACKOFF_S
      _GEOGLOWS_RP_ENDPOINT_STATE["last_error"] = err or "Unparseable /returnperiods payload"

    job = _schedule_geoglows_rp_job(rid)
    if job is not None and wait_s > 0:
      job.join(timeout=wait_s)
      done = get_cached(cache_key)
      if done is not None:
        return done
  return _geoglows_fallback_return_periods(rid, mean_flow)


# ---------------------------------------------------------------------------
# 4c. JAXA Today's Earth (TE-Global: MATSIRO + CaMa-Flood)
# ---------------------------------------------------------------------------

_TE_CATALOG_TOKENS = (
    "todays", "today", "te-global", "te_global", "camaflood", "cama-flood",
    "cama_flood", "rivout", "fldout", "flddph", "fldfrc", "sfcelv",
)
# Linear-reservoir coefficient (1/day) of above-bankfull floodplain storage.
_CAMA_FLOODPLAIN_K = 0.6
# Share of the routed above-bankfull flow conveyed over the floodplain (FLDOUT).
_CAMA_FLDOUT_SHARE = 0.35
TODAYS_EARTH_EMULATION_NOTE = (
    "Emulated: JAXA does not publish TE-Global CaMa-Flood forecasts through a "
    "public machine-readable API (no Today's Earth collection in the JAXA Earth "
    "STAC catalog). Values are a deterministic CaMa-Flood-physics emulation "
    "driven by the GloFAS v4 forecast; set TODAYS_EARTH_API_URL to connect an "
    "operational TE-Global feed."
)


def _snap_cama_cell(lat: float, lon: float, res: float = CAMA_GRID_RES_DEG) -> Tuple[float, float]:
  """Centre of the CaMa-Flood regular grid cell that contains (lat, lon)."""
  lat = min(max(float(lat), -89.9999), 89.9999)
  lon = ((float(lon) + 180.0) % 360.0) - 180.0
  return (
      round(math.floor(lat / res) * res + res / 2.0, 4),
      round(math.floor(lon / res) * res + res / 2.0, 4),
  )


def _cama_cell_id(cell_lat: float, cell_lon: float) -> str:
  return f"cama_025_{cell_lat:.3f}_{cell_lon:.3f}"


def _cama_cell_polygon(
    cell_lat: float, cell_lon: float, res: float = CAMA_GRID_RES_DEG
) -> Tuple[List[List[float]], Dict[str, float]]:
  """Closed GeoJSON ring and bbox of a res x res cell centred at (cell_lat, cell_lon)."""
  half = res / 2.0
  w, e = round(cell_lon - half, 5), round(cell_lon + half, 5)
  s, n = round(cell_lat - half, 5), round(cell_lat + half, 5)
  ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
  return ring, {"min_lon": w, "min_lat": s, "max_lon": e, "max_lat": n}


def _cell_area_km2(cell_lat: float, res: float = CAMA_GRID_RES_DEG) -> float:
  """Exact spherical area (km²) of a res x res degree cell centred at cell_lat."""
  r = 6371.0088
  phi1 = math.radians(max(cell_lat - res / 2.0, -90.0))
  phi2 = math.radians(min(cell_lat + res / 2.0, 90.0))
  return round(r * r * math.radians(res) * abs(math.sin(phi2) - math.sin(phi1)), 2)


def todays_earth_service_status() -> str:
  """'operational' when a TE-Global feed is configured, otherwise 'emulated'."""
  return "operational" if os.environ.get(TODAYS_EARTH_API_URL_ENV, "").strip() else "emulated"


def _probe_todays_earth_catalog() -> Dict[str, Any]:
  """Scans the public JAXA Earth STAC catalog for a Today's Earth collection (cached)."""
  cache_key = "te_stac_probe"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached
  payload, err = _http_get_json_with_error(JAXA_STAC_CATALOG_URL, timeout=TODAYS_EARTH_TIMEOUT_S)
  scanned, hits = 0, []
  if isinstance(payload, dict):
    for link in payload.get("links") or []:
      if not isinstance(link, dict) or link.get("rel") != "child":
        continue
      scanned += 1
      text = f"{link.get('href', '')} {link.get('title', '')}".lower()
      if any(tok in text for tok in _TE_CATALOG_TOKENS):
        hits.append(link.get("href"))
  result = {
      "catalog_url": JAXA_STAC_CATALOG_URL,
      "reachable": isinstance(payload, dict),
      "collections_scanned": scanned,
      "todays_earth_collections": hits,
      "error": err,
      "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  }
  set_cached(cache_key, result, ttl_seconds=21600 if result["reachable"] else 900)
  return result


def _fetch_ground_elevation(lat: float, lon: float) -> Tuple[float, str]:
  """Ground elevation (m a.s.l.) from the Open-Meteo Copernicus GLO-90 DEM API."""
  cache_key = f"elev_{lat:.3f}_{lon:.3f}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached
  res = _http_get_json(OPEN_METEO_ELEVATION_URL, {"latitude": lat, "longitude": lon}, timeout=4)
  out = (10.0, "default (elevation service unreachable)")
  try:
    elev = float(((res or {}).get("elevation") or [None])[0])
    if math.isfinite(elev):
      out = (round(elev, 1), "Copernicus DEM GLO-90 (Open-Meteo)")
  except (TypeError, ValueError):
    pass
  set_cached(cache_key, out, ttl_seconds=30 * 86400 if "GLO-90" in out[1] else 600)
  return out


def _te_series(payload: Dict[str, Any], *names: str) -> Optional[List[Optional[float]]]:
  """First list-valued field among `names` (top level or under `flood_forecast`)."""
  scopes = [payload]
  if isinstance(payload.get("flood_forecast"), dict):
    scopes.append(payload["flood_forecast"])
  for scope in scopes:
    for name in names:
      val = scope.get(name)
      if isinstance(val, list) and val:
        return [_safe_float_or_none(v) for v in val]
  return None


def _fetch_todays_earth_live(
    lat: float, lon: float, reach_id: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Queries an operator-configured TE-Global point-forecast endpoint, if any.

  The endpoint (env `TODAYS_EARTH_API_URL`) receives `lat`, `lon` and optional
  `reach_id`, and must return JSON with `timestamps` plus `rivout` (or `mean`),
  and optionally `fldout`, `p25`, `p75`, `max`, `min`, `flddph`/`flddph_m`,
  `fldfrc` (0-1) or `fldfrc_pct`, and `sfcelv`/`sfcelv_m`, either top-level or
  nested under `flood_forecast`.
  """
  base = os.environ.get(TODAYS_EARTH_API_URL_ENV, "").strip()
  if not base:
    return None, None
  params: Dict[str, Any] = {"lat": lat, "lon": lon}
  if reach_id:
    params["reach_id"] = reach_id
  payload, err = _http_get_json_with_error(base, params, timeout=TODAYS_EARTH_TIMEOUT_S)
  if not isinstance(payload, dict):
    return None, err or "Invalid Today's Earth payload"
  times = payload.get("timestamps") or payload.get("time")
  rivout = _te_series(payload, "rivout", "RIVOUT")
  mean = _te_series(payload, "mean", "outflw", "OUTFLW")
  if not isinstance(times, list) or not times or not (rivout or mean):
    return None, "Today's Earth payload is missing timestamps or RIVOUT"
  n = len(times)

  def _fit(series: Optional[List[Optional[float]]], default: Optional[float] = None) -> List[Optional[float]]:
    return (list(series or []) + [default] * n)[:n]

  fldout = _fit(_te_series(payload, "fldout", "FLDOUT"), 0.0)
  if rivout is None:
    rivout = [None if m is None else max(m - (f or 0.0), 0.0) for m, f in zip(_fit(mean), fldout)]
  rivout = _fit(rivout)
  if mean is None:
    mean = [None if r is None else r + (f or 0.0) for r, f in zip(rivout, fldout)]
  fldfrc_pct = _te_series(payload, "fldfrc_pct")
  if fldfrc_pct is None:
    frac = _te_series(payload, "fldfrc", "FLDFRC")
    fldfrc_pct = [None if v is None else v * 100.0 for v in frac] if frac else None
  return {
      "timestamps": [str(t) for t in times],
      "mean": _fit(mean),
      "rivout": rivout,
      "fldout": fldout,
      "p25": _fit(_te_series(payload, "p25")),
      "p75": _fit(_te_series(payload, "p75")),
      "max": _fit(_te_series(payload, "max")),
      "min": _fit(_te_series(payload, "min")),
      "flddph_m": _fit(_te_series(payload, "flddph_m", "flddph", "FLDDPH"), 0.0),
      "fldfrc_pct": _fit(fldfrc_pct, 0.0),
      "sfcelv_m": _fit(_te_series(payload, "sfcelv_m", "sfcelv", "SFCELV")),
  }, None


def _route_floodplain_excess(
    series: List[float], q_bankfull: float, k: float = _CAMA_FLOODPLAIN_K
) -> List[float]:
  """Linear-reservoir routing of above-bankfull flow (daily explicit scheme)."""
  routed: List[float] = []
  state: Optional[float] = None
  for q in series:
    excess = max(q - q_bankfull, 0.0)
    state = excess if state is None else state + k * (excess - state)
    routed.append(state)
  return routed


def _emulate_camaflood_forecast(lat: float, lon: float) -> Dict[str, Any]:
  """Deterministic CaMa-Flood-style streamflow + inundation emulation.

  Driven by the GloFAS v4 forecast at the probe's 0.05° river cell:
    * Channel geometry follows the CaMa-Flood power laws (Yamazaki et al.,
      2011): W = 0.40 Q^0.75 and H = 0.10 Q^0.5, Q = long-term mean flow.
    * Bankfull discharge Q_bf is the EV1 1.5-year flow of the reanalysis.
    * Above-bankfull excess is stored on the floodplain and released by a
      linear reservoir (k = 0.6/day); 35 % of it is conveyed as FLDOUT.
    * Stage follows Manning (h ∝ Q^0.6) and FLDDPH = max(h - H, 0).
    * FLDFRC saturates with depth; its ceiling grows with river size and
      shrinks with ground elevation, so flat deltas flood most widely.
  """
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    f_fc = ex.submit(fetch_glofas_forecast, lat, lon, 15)
    f_rp = ex.submit(fetch_glofas_return_periods, lat, lon)
    f_el = ex.submit(_fetch_ground_elevation, lat, lon)
    forecast, rps, (elev, elev_source) = f_fc.result(), f_rp.result(), f_el.result()

  # TE-Global publishes a 5-day forecast (today + 5 days = 6 daily points).
  records = (forecast.get("data") or [])[:6]

  def _col(name: str, fallback: str = "discharge_mean") -> List[float]:
    out = []
    for r in records:
      v = _safe_float_or_none(r.get(name))
      if v is None:
        v = _safe_float_or_none(r.get(fallback))
      out.append(max(v or 0.0, 0.0))
    return out

  central = _col("discharge_median")
  q_clim = _safe_float_or_none(rps.get("mean_flow")) or _series_median(central) or 1.0
  q_clim = max(q_clim, 0.05)
  width = max(0.40 * q_clim ** 0.75, 10.0)
  depth = max(0.10 * q_clim ** 0.5, 1.0)
  q_bf = max(_gumbel_quantile_from_return_periods(rps, 1.5) or 0.0, 1.2 * q_clim, 0.5)
  elev_c = min(max(float(elev), 0.0), 1500.0)
  depth_scale = 1.0 + elev_c / 150.0
  f_max = (0.02 + 0.08 * math.log10(1.0 + q_clim / 10.0)) * (1.0 + 1.5 * math.exp(-elev_c / 30.0))
  f_max = min(max(f_max, 0.02), 0.6)

  def _cama(series: List[float]) -> Tuple[List[float], List[float], List[float]]:
    routed = _route_floodplain_excess(series, q_bf)
    total = [min(q, q_bf) + r for q, r in zip(series, routed)]
    fld = [_CAMA_FLDOUT_SHARE * r for r in routed]
    return total, [t - f for t, f in zip(total, fld)], fld

  total, rivout, fldout = _cama(central)
  stage = [depth * (max(r, 0.0) / q_bf) ** 0.6 for r in rivout]
  flddph = [max(h - depth, 0.0) for h in stage]
  fldfrc = [100.0 * f_max * (1.0 - math.exp(-d / depth_scale)) for d in flddph]
  sfcelv = [max(max(float(elev), 0.0) - depth + h, 0.0) for h in stage]
  r2 = lambda xs: [round(x, 2) for x in xs]  # pylint: disable=unnecessary-lambda-assignment
  return {
      "series": {
          "timestamps": [f"{str(r.get('time'))[:10]}T00:00:00Z" for r in records],
          "mean": r2(total),
          "rivout": r2(rivout),
          "fldout": r2(fldout),
          "p25": r2(_cama(_col("discharge_p25"))[0]),
          "p75": r2(_cama(_col("discharge_p75"))[0]),
          "max": r2(_cama(_col("discharge_max"))[0]),
          "min": r2(_cama(_col("discharge_min"))[0]),
          "flddph_m": [round(d, 3) for d in flddph],
          "fldfrc_pct": r2(fldfrc),
          "sfcelv_m": r2(sfcelv),
      },
      "channel_params": {
          "mean_flow_m3s": round(q_clim, 3),
          "bankfull_discharge_m3s": round(q_bf, 2),
          "channel_width_m": round(width, 1),
          "channel_depth_m": round(depth, 2),
          "ground_elevation_m": float(elev),
          "elevation_source": elev_source,
          "max_flooded_fraction_ceiling_pct": round(100.0 * f_max, 1),
      },
      "forcing_status": forecast.get("status"),
      "return_period_status": rps.get("status"),
  }


def fetch_todays_earth_forecast(lat: float, lon: float, reach_id: Optional[str] = None) -> Dict[str, Any]:
  """JAXA Today's Earth (TE-Global) streamflow and flood-inundation forecast.

  Uses an operator-configured TE-Global feed when `TODAYS_EARTH_API_URL` is
  set (`status="live"`); otherwise returns the deterministic CaMa-Flood
  emulation (`status="fallback"`, `emulated=True`). The public JAXA Earth STAC
  catalog is probed and reported under `live_probe`.

  Returns:
    Dict with `source`, `status`, `grid_cell_id` (`cama_025_<lat>_<lon>`),
    `unit`, `timestamps`, `mean` (RIVOUT + FLDOUT), `rivout`, `fldout`, `p25`,
    `p75`, `max`, `min`, and `flood_forecast` (`flddph_m`, `fldfrc_pct`,
    `sfcelv_m`, `max_flood_depth_m`, `max_flooded_fraction_pct`, ...).
  """
  live_configured = todays_earth_service_status() == "operational"
  cell_lat, cell_lon = _snap_cama_cell(lat, lon)
  cell_id = _cama_cell_id(cell_lat, cell_lon)
  forcing_lat, forcing_lon = _glofas_cell_center(lat, lon)
  cache_key = (
      f"te_forecast_{'live' if live_configured else 'emu'}_{cell_id}_{forcing_lat:.3f}_{forcing_lon:.3f}_"
      f"{reach_id if (live_configured and reach_id) else ''}"
  )
  cached = get_cached(cache_key)
  if cached is not None:
    return cached

  probe = _probe_todays_earth_catalog()
  live, live_error = _fetch_todays_earth_live(lat, lon, reach_id)
  channel_params = None
  forcing_status = None
  if live:
    series = live
    status = "live"
    method = "Operational TE-Global point forecast (TODAYS_EARTH_API_URL)"
  else:
    emu = _emulate_camaflood_forecast(lat, lon)
    series = emu["series"]
    channel_params = emu["channel_params"]
    forcing_status = emu["forcing_status"]
    status = "fallback"
    method = (
        "CaMa-Flood physics emulator (Yamazaki et al., 2011 channel geometry, "
        "EV1 bankfull, linear floodplain reservoir) forced by GloFAS v4"
    )

  times = series["timestamps"]
  flddph = series["flddph_m"]
  fldfrc = series["fldfrc_pct"]
  sfcelv = series["sfcelv_m"]
  valid_depth = [(d, i) for i, d in enumerate(flddph) if d is not None]
  peak_depth, peak_idx = max(valid_depth) if valid_depth else (0.0, None)
  valid_frac = [f for f in fldfrc if f is not None]
  valid_elev = [e for e in sfcelv if e is not None]
  data = []
  for i, t in enumerate(times):
    data.append({
        "time": t,
        "discharge_mean": series["mean"][i],
        "rivout": series["rivout"][i],
        "fldout": series["fldout"][i],
        "discharge_p25": series["p25"][i],
        "discharge_p75": series["p75"][i],
        "discharge_max": series["max"][i],
        "discharge_min": series["min"][i],
        "flddph_m": flddph[i],
        "fldfrc_pct": fldfrc[i],
        "sfcelv_m": sfcelv[i],
    })

  result = {
      "model": "jaxa_todays_earth",
      "available": True,
      "source": TODAYS_EARTH_SOURCE,
      "status": status,
      "emulated": not bool(live),
      "method": method,
      "note": None if live else TODAYS_EARTH_EMULATION_NOTE,
      "grid_cell_id": cell_id,
      "grid_resolution_deg": CAMA_GRID_RES_DEG,
      "cell_center_lat": cell_lat,
      "cell_center_lon": cell_lon,
      "cell_area_km2": _cell_area_km2(cell_lat),
      "reach_id": reach_id,
      "unit": "m³/s",
      "timestamps": times,
      "mean": series["mean"],
      "rivout": series["rivout"],
      "fldout": series["fldout"],
      "p25": series["p25"],
      "p75": series["p75"],
      "max": series["max"],
      "min": series["min"],
      "flood_forecast": {
          "flddph_m": flddph,
          "fldfrc_pct": fldfrc,
          "sfcelv_m": sfcelv,
          "max_flood_depth_m": round(peak_depth or 0.0, 3),
          "max_flooded_fraction_pct": round(max(valid_frac), 2) if valid_frac else 0.0,
          "max_sfcelv_m": round(max(valid_elev), 2) if valid_elev else None,
          "peak_depth_time": times[peak_idx] if (peak_idx is not None and peak_depth > 0) else None,
      },
      "data": data,
      "channel_params": channel_params,
      "forcing_status": forcing_status,
      "live_probe": probe,
      "live_error": live_error,
  }
  set_cached(
      cache_key, result,
      ttl_seconds=3600 if live else (1800 if forcing_status == "live" else FALLBACK_CACHE_TTL_S),
  )
  return result


# ---------------------------------------------------------------------------
# 4d. Google FloodHub Flood Status, Thresholds & Inundation Maps
# ---------------------------------------------------------------------------

_FH_SEVERITY_MAP = {
    "NO_FLOODING": "NO_FLOODING",
    "ABOVE_NORMAL": "WARNING",
    "SEVERE": "DANGER",
    "EXTREME": "EXTREME_DANGER",
    "WARNING": "WARNING",
    "DANGER": "DANGER",
    "EXTREME_DANGER": "EXTREME_DANGER",
}
_FH_TREND_MAP = {
    "RISE": "RISING",
    "RISING": "RISING",
    "FALL": "FALLING",
    "FALLING": "FALLING",
    "NO_CHANGE": "STEADY",
    "STEADY": "STEADY",
}
_FH_SEVERITY_RANK = {"UNKNOWN": 0, "NO_FLOODING": 0, "WARNING": 1, "DANGER": 2, "EXTREME_DANGER": 3}
_FH_SEVERITY_LABELS = {
    "NO_FLOODING": "Normal (FloodHub official)",
    "WARNING": "Warning level (FloodHub official)",
    "DANGER": "Danger level (FloodHub official)",
    "EXTREME_DANGER": "Extreme danger (FloodHub official)",
}
_FH_LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_FH_LEVEL_LABELS = {"HIGH": "High likelihood", "MEDIUM": "Medium likelihood", "LOW": "Low likelihood"}
_FH_LEVEL_COLORS = {"HIGH": "#0e7490", "MEDIUM": "#06b6d4", "LOW": "#67e8f9"}
FLOODHUB_GAUGE_SEARCH_RADIUS_KM = 30.0


def _normalize_fh_severity(raw: Any) -> str:
  return _FH_SEVERITY_MAP.get(str(raw or "").strip().upper(), "UNKNOWN")


def _normalize_fh_trend(raw: Any) -> str:
  return _FH_TREND_MAP.get(str(raw or "").strip().upper(), "UNKNOWN")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  r = 6371.0088
  p1, p2 = math.radians(lat1), math.radians(lat2)
  dp, dl = p2 - p1, math.radians(lon2 - lon1)
  a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
  return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _search_floodhub_status_near(
    lat: float, lon: float, target_area_km2: Optional[float] = None
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[float]]:
  """Best real FloodHub flood status within 30 km: (status, api_reachable, distance_km)."""
  area_tag = f"_{int(round(target_area_km2))}" if (target_area_km2 and target_area_km2 > 0) else ""
  cache_key = f"fh_status_near_{lat:.3f}_{lon:.3f}{area_tag}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached
  d = 0.25
  payload = {
      "loop": {
          "vertices": [
              {"latitude": lat - d, "longitude": lon - d},
              {"latitude": lat + d, "longitude": lon - d},
              {"latitude": lat + d, "longitude": lon + d},
              {"latitude": lat - d, "longitude": lon + d},
          ]
      },
      "pageSize": 100,
      "includeNonQualityVerified": True,
  }
  url = f"{FLOODHUB_BASE_URL}/floodStatus:searchLatestFloodStatusByArea?key={_get_floodhub_key()}"
  res = _http_post_json(url, payload, timeout=8)
  snap_fn = None
  if target_area_km2 and target_area_km2 > 0:
    try:
      from frontend.maas_networks import snap_glofas_cell as snap_fn  # pylint: disable=g-import-not-at-top
    except ImportError:
      try:
        from maas_networks import snap_glofas_cell as snap_fn  # pylint: disable=g-import-not-at-top
      except ImportError:
        snap_fn = None
  best: Optional[Tuple[float, float, Dict[str, Any]]] = None
  for st in (res or {}).get("floodStatuses") or []:
    loc = st.get("gaugeLocation") or {}
    try:
      glat, glon = float(loc["latitude"]), float(loc["longitude"])
      dist = _haversine_km(lat, lon, glat, glon)
    except (KeyError, TypeError, ValueError):
      continue
    if dist > FLOODHUB_GAUGE_SEARCH_RADIUS_KM:
      continue
    if snap_fn is not None and target_area_km2 and target_area_km2 > 0:
      g_snap = snap_fn(glat, glon, target_area_km2, radius_cells=1)
      area_cost = (
          abs(math.log(max(float(g_snap["upstream_area_km2"]), 1.0) / target_area_km2))
          if g_snap else 4.0
      )
      is_hybas = str(st.get("gaugeId") or "").startswith("hybas_") or st.get("source") == "HYBAS"
      score = area_cost + 0.35 * (dist / FLOODHUB_GAUGE_SEARCH_RADIUS_KM) + (0.0 if is_hybas else 2.0)
    else:
      score = dist
    if best is None or score < best[0]:
      best = (score, dist, st)
  out = (best[2] if best else None, res is not None, round(best[1], 2) if best else None)
  set_cached(cache_key, out, ttl_seconds=900 if res is not None else 120)
  return out


def _fetch_floodhub_flood_status(
    gauge_id: Optional[str], lat: float, lon: float
) -> Optional[Dict[str, Any]]:
  """Latest FloodHub flood status for a gauge (or the nearest gauge within 30 km)."""
  if not gauge_id:
    return _search_floodhub_status_near(lat, lon)[0]
  cache_key = f"fh_status_{gauge_id}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached or None
  res = _http_get_json(
      f"{FLOODHUB_BASE_URL}/floodStatus:queryLatestFloodStatusByGaugeIds",
      {"key": _get_floodhub_key(), "gaugeIds": gauge_id},
      timeout=8,
  )
  status = next(
      (st for st in (res or {}).get("floodStatuses") or [] if st.get("gaugeId") == gauge_id), None
  )
  set_cached(cache_key, status or {}, ttl_seconds=900 if status else 300)
  return status


def _round_coords(obj: Any, ndigits: int = 5) -> Any:
  if isinstance(obj, (list, tuple)):
    if obj and isinstance(obj[0], (int, float)):
      return [round(float(v), ndigits) for v in obj]
    return [_round_coords(o, ndigits) for o in obj]
  return obj


def _kml_rings(elem: Any) -> List[List[Tuple[float, float]]]:
  """All coordinate rings under a KML element (namespace-agnostic)."""
  rings = []
  for node in elem.iter():
    if node.tag.split("}")[-1] != "coordinates" or not node.text:
      continue
    ring = []
    for tok in node.text.split():
      parts = tok.split(",")
      if len(parts) >= 2:
        try:
          ring.append((float(parts[0]), float(parts[1])))
        except ValueError:
          continue
    if len(ring) >= 4:
      rings.append(ring)
  return rings


def _kml_to_geometry(kml_text: str, tolerance_deg: float = 0.0005) -> Optional[Any]:
  """Parses FloodHub KML polygons into a simplified shapely (Multi)Polygon."""
  from shapely.geometry import MultiPolygon, Polygon  # pylint: disable=g-import-not-at-top
  from shapely.ops import unary_union  # pylint: disable=g-import-not-at-top

  root = ET.fromstring(kml_text)
  polys = []
  for el in root.iter():
    if el.tag.split("}")[-1] != "Polygon":
      continue
    outer: List[Tuple[float, float]] = []
    inners: List[List[Tuple[float, float]]] = []
    for child in el:
      tag = child.tag.split("}")[-1]
      if tag == "outerBoundaryIs":
        rings = _kml_rings(child)
        outer = rings[0] if rings else []
      elif tag == "innerBoundaryIs":
        inners.extend(_kml_rings(child))
    if len(outer) < 4:
      continue
    poly = Polygon(outer, inners)
    if not poly.is_valid:
      poly = poly.buffer(0)
    if not poly.is_empty:
      polys.append(poly)
  if not polys:
    return None
  try:
    geom = unary_union(polys)
  except Exception:  # pylint: disable=broad-except
    geom = MultiPolygon([p for p in polys if p.geom_type == "Polygon"])
  geom = geom.simplify(tolerance_deg, preserve_topology=True)
  if geom.geom_type == "GeometryCollection":
    parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    geom = unary_union(parts) if parts else None
  if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
    return None
  return geom


def _geom_area_km2(geom: Any) -> float:
  """Approximate area (km²) of a lon/lat geometry via local equirectangular scaling."""
  try:
    lat0 = geom.centroid.y
    return round(geom.area * 111.32 * 111.32 * max(math.cos(math.radians(lat0)), 0.01), 2)
  except Exception:  # pylint: disable=broad-except
    return 0.0


def _fetch_floodhub_polygon_geometry(polygon_id: str) -> Optional[Dict[str, Any]]:
  """Fetches + simplifies a FloodHub serialized inundation polygon (cached 6 h)."""
  cache_key = f"fh_poly_{polygon_id}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached or None
  from shapely.geometry import mapping  # pylint: disable=g-import-not-at-top

  res = _http_get_json(
      f"{FLOODHUB_BASE_URL}/serializedPolygons/{urllib.parse.quote(str(polygon_id), safe='')}",
      {"key": _get_floodhub_key()},
      timeout=15,
  )
  out: Dict[str, Any] = {}
  if res and res.get("kml"):
    try:
      geom = _kml_to_geometry(res["kml"])
      if geom is not None:
        out = {
            "geometry": {"type": geom.geom_type, "coordinates": _round_coords(mapping(geom)["coordinates"])},
            "area_km2": _geom_area_km2(geom),
        }
    except Exception as e:  # pylint: disable=broad-except
      logger.debug("FloodHub KML parse failed for %s: %s", polygon_id, e)
  set_cached(cache_key, out, ttl_seconds=21600 if out else 600)
  return out or None


def _future_values(records: List[Dict[str, Any]], key: str) -> List[float]:
  """Values of `key` at or after today's UTC date (all values if none are)."""
  today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
  pairs = [(str(r.get("time", ""))[:10], _safe_float_or_none(r.get(key))) for r in records or []]
  future = [v for t, v in pairs if v is not None and t >= today]
  return future or [v for _, v in pairs if v is not None]


def _derive_fh_severity_from_forecast(forecast: Dict[str, Any]) -> Tuple[str, str]:
  """(severity, trend) from a FloodHub forecast versus its gauge thresholds."""
  vals = _future_values(forecast.get("data") or [], "discharge")
  th = forecast.get("thresholds") or {}
  warn, danger, extreme = (
      _safe_float_or_none(th.get("warning_2yr")),
      _safe_float_or_none(th.get("danger_5yr")),
      _safe_float_or_none(th.get("extreme_20yr")),
  )
  if not vals or warn is None:
    return "UNKNOWN", "UNKNOWN"
  peak = max(vals)
  if extreme is not None and peak >= extreme:
    severity = "EXTREME_DANGER"
  elif danger is not None and peak >= danger:
    severity = "DANGER"
  elif peak >= warn:
    severity = "WARNING"
  else:
    severity = "NO_FLOODING"
  third = max(len(vals) // 3, 1)
  head, tail = sum(vals[:third]) / third, sum(vals[-third:]) / third
  if tail > head * 1.05:
    trend = "RISING"
  elif tail < head * 0.95:
    trend = "FALLING"
  else:
    trend = "STEADY"
  return severity, trend


def fetch_floodhub_inundation(
    gauge_id: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    include_polygons: bool = True,
) -> Dict[str, Any]:
  """Google FloodHub gauge thresholds, severity, trend and inundation polygons.

  Severity is normalized to NO_FLOODING / WARNING / DANGER / EXTREME_DANGER
  and trend to RISING / FALLING / STEADY. Inundation polygons come from the
  gauge's official `inundationMapSet` (HIGH / MEDIUM / LOW likelihood maps),
  parsed from KML and simplified to ~50 m for web rendering.
  """
  lat = float(lat) if lat is not None else None
  lon = float(lon) if lon is not None else None
  loc_key = f"{lat:.3f}_{lon:.3f}" if lat is not None and lon is not None else "none"
  cache_key = f"fh_inundation_{gauge_id or ''}_{loc_key if not gauge_id else ''}_{int(include_polygons)}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached

  flood_status = None
  distance_km = None
  if gauge_id:
    flood_status = _fetch_floodhub_flood_status(gauge_id, lat or 0.0, lon or 0.0)
  elif lat is not None and lon is not None:
    flood_status, _, distance_km = _search_floodhub_status_near(lat, lon)
  gid = gauge_id or (flood_status or {}).get("gaugeId")
  forecast = fetch_floodhub_forecast(gid) if gid else None

  severity_raw = (flood_status or {}).get("severity")
  trend_raw = (flood_status or {}).get("forecastTrend")
  severity = _normalize_fh_severity(severity_raw)
  trend = _normalize_fh_trend(trend_raw)
  severity_source = "floodhub_flood_status" if flood_status else None
  if forecast and forecast.get("status") == "live" and (severity == "UNKNOWN" or trend == "UNKNOWN"):
    d_sev, d_trend = _derive_fh_severity_from_forecast(forecast)
    if severity == "UNKNOWN" and d_sev != "UNKNOWN":
      severity, severity_source = d_sev, "derived_from_forecast"
    if trend == "UNKNOWN":
      trend = d_trend

  map_set = (flood_status or {}).get("inundationMapSet") or {}
  maps = [m for m in map_set.get("inundationMaps") or [] if m.get("serializedPolygonId")]
  maps.sort(key=lambda m: _FH_LEVEL_ORDER.get(str(m.get("level", "")).upper(), -1))
  loc = (flood_status or {}).get("gaugeLocation") or {}
  gauge_lat = _safe_float_or_none(loc.get("latitude"))
  gauge_lon = _safe_float_or_none(loc.get("longitude"))
  if gauge_lat is None and lat is not None:
    gauge_lat, gauge_lon = lat, lon

  polygons: List[Dict[str, Any]] = []
  if include_polygons and maps:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
      geoms = list(ex.map(lambda m: _fetch_floodhub_polygon_geometry(m["serializedPolygonId"]), maps))
    for m, g in zip(maps, geoms):
      if not g:
        continue
      level = str(m.get("level", "")).upper()
      polygons.append({
          "type": "Feature",
          "geometry": g["geometry"],
          "properties": {
              "layer": "floodhub_extent",
              "provider": "Google FloodHub",
              "source": "Google FloodHub inundation map (inundationMapSet)",
              "derived": False,
              "gauge_id": gid,
              "probability_level": level,
              "label": f"FloodHub inundation — {_FH_LEVEL_LABELS.get(level, level.title())}",
              "severity": severity,
              "area_km2": g.get("area_km2"),
              "map_type": map_set.get("inundationMapType"),
              "time_range": map_set.get("inundationMapsTimeRange"),
              "color": _FH_LEVEL_COLORS.get(level, "#06b6d4"),
          },
      })

  thresholds = ((forecast or {}).get("thresholds") or {}) if (forecast or {}).get("status") == "live" else {}
  if flood_status:
    status = "live"
  elif forecast:
    status = forecast.get("status") or "live"
  else:
    status = "unavailable"
  result = {
      "provider": "floodhub",
      "source": "Google FloodHub (FloodForecasting v1 API)",
      "status": status,
      "severity_source": severity_source,
      "gauge_id": gid,
      "gauge_location": {"lat": gauge_lat, "lon": gauge_lon} if gauge_lat is not None else None,
      "distance_km": distance_km,
      "thresholds": {
          "warning_level": thresholds.get("warning_2yr"),
          "danger_level": thresholds.get("danger_5yr"),
          "extreme_danger_level": thresholds.get("extreme_20yr"),
          "unit": (forecast or {}).get("unit"),
      },
      "severity": severity,
      "severity_raw": severity_raw,
      "severity_rank": _FH_SEVERITY_RANK.get(severity, 0),
      "trend": trend,
      "trend_raw": trend_raw,
      "issued_time": (flood_status or {}).get("issuedTime") or (forecast or {}).get("issued_time"),
      "forecast_time_range": (flood_status or {}).get("forecastTimeRange"),
      "forecast_change": (flood_status or {}).get("forecastChange"),
      "quality_verified": (flood_status or {}).get("qualityVerified"),
      "map_inference_type": (flood_status or {}).get("mapInferenceType"),
      "inundation_map_type": map_set.get("inundationMapType"),
      "inundation_maps_time_range": map_set.get("inundationMapsTimeRange"),
      "inundation_map_levels": [str(m.get("level", "")).upper() for m in maps],
      "inundation_maps_available": bool(maps),
      "inundation_polygons": polygons,
  }
  set_cached(cache_key, result, ttl_seconds=900)
  return result


# ---------------------------------------------------------------------------
# 4e. Spatial Flood Inundation Layers (FloodHub, CaMa-Flood, Reach Exceedance)
# ---------------------------------------------------------------------------

_EXCEEDANCE_CLASSES = (
    {"rank": 0, "label": "Normal", "risk_level": "NORMAL", "return_period": "< 2-yr", "color": "#22c55e"},
    {"rank": 1, "label": "2-Yr Warning", "risk_level": "WARNING", "return_period": "≥ 2-yr", "color": "#eab308"},
    {"rank": 2, "label": "5-Yr Severe", "risk_level": "SEVERE", "return_period": "≥ 5-yr", "color": "#f97316"},
    {"rank": 3, "label": "20-Yr+ Extreme", "risk_level": "EXTREME", "return_period": "≥ 20-yr", "color": "#dc2626"},
    {"rank": 4, "label": "100-Yr+ Extreme", "risk_level": "EXTREME", "return_period": "≥ 100-yr", "color": "#7e22ce"},
)
# Synthetic offline fallbacks are displayed but never classified against a climatology.
_UNASSESSED_LABEL = "Not assessed (offline fallback)"
_UNASSESSED_COLOR = "#94a3b8"
_RISK_RANK = {"UNKNOWN": -1, "NORMAL": 0, "WARNING": 1, "SEVERE": 2, "EXTREME": 3}
_FH_SEVERITY_TO_RISK = {
    "NO_FLOODING": "NORMAL",
    "WARNING": "WARNING",
    "DANGER": "SEVERE",
    "EXTREME_DANGER": "EXTREME",
}
# Visual corridor half-width multipliers (x channel half-width).
_CORRIDOR_WIDTH_FACTOR = (1.0, 3.0, 5.0, 8.0, 10.0)  # by exceedance rank
_FH_DERIVED_WIDTH_FACTOR = (0.0, 4.0, 7.0, 10.0)  # by FloodHub severity rank
_MAX_CORRIDOR_SNAP_KM = 15.0


def _classify_exceedance(peak: Optional[float], rps: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  """Return-period exceedance class of a peak flow against model-specific levels."""
  rank = 0
  if peak is not None:
    for r, key in ((4, "return_period_100"), (3, "return_period_20"), (2, "return_period_5"), (1, "return_period_2")):
      thr = _safe_float_or_none((rps or {}).get(key))
      if thr is not None and thr > 0 and peak >= thr:
        rank = r
        break
  return dict(_EXCEEDANCE_CLASSES[rank])


def _depth_color(depth_m: Optional[float]) -> str:
  if depth_m is None or depth_m <= 0.0:
    return "#bae6fd"
  if depth_m < 0.5:
    return "#38bdf8"
  if depth_m < 1.0:
    return "#0284c7"
  if depth_m < 2.0:
    return "#1d4ed8"
  return "#1e3a8a"


def _channel_half_width_m(mean_discharge: Any) -> float:
  """Half the CaMa-Flood power-law channel width (floored at 40 m for visibility)."""
  q = max(_safe_float(mean_discharge), 0.0)
  return max(0.5 * max(0.40 * q ** 0.75, 10.0), 40.0)


def _daily_series(records: List[Dict[str, Any]], key: str, fallback_key: Optional[str] = None) -> Dict[str, float]:
  """Daily means {YYYY-MM-DD: value} of a (sub-)daily record series."""
  sums: Dict[str, float] = {}
  counts: Dict[str, int] = {}
  for r in records or []:
    v = _safe_float_or_none(r.get(key))
    if v is None and fallback_key:
      v = _safe_float_or_none(r.get(fallback_key))
    t = str(r.get("time", ""))[:10]
    if v is None or len(t) != 10:
      continue
    sums[t] = sums.get(t, 0.0) + v
    counts[t] = counts.get(t, 0) + 1
  return {t: sums[t] / counts[t] for t in sorted(sums)}


def _window_peak(daily: Dict[str, float]) -> Tuple[Optional[float], Optional[str]]:
  """Peak (value, date) over today and later (whole series if nothing is current)."""
  today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
  items = [(t, v) for t, v in daily.items() if t >= today] or list(daily.items())
  if not items:
    return None, None
  t, v = max(items, key=lambda kv: kv[1])
  return round(v, 3), t


def _future_result(fut: Optional[concurrent.futures.Future], default: Any = None) -> Any:
  if fut is None:
    return default
  try:
    return fut.result()
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("MaaS provider task failed: %s", e)
    return default


def _geoglows_forecast_and_rp(
    lat: float, lon: float, river_id: Optional[Any] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """GEOGLOWS forecast and return periods, fetched in parallel once the reach is known."""
  rid = int(river_id) if _is_geoglows_river_id(river_id) else fetch_geoglows_river_id(lat, lon)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    f_fc = ex.submit(fetch_geoglows_forecast, lat, lon, rid)
    f_rp = ex.submit(fetch_geoglows_return_periods, rid) if rid else None
    forecast = f_fc.result()
    rps = _future_result(f_rp)
  if rps is None or rps.get("status") == "fallback":
    median = _series_median([r.get("flow_med") for r in forecast.get("data") or []])
    rps = _geoglows_fallback_return_periods(forecast.get("river_id"), median)
  return forecast, rps


def _reach_exceedance_summary(
    glofas_fc: Optional[Dict[str, Any]],
    glofas_rp: Optional[Dict[str, Any]],
    geoglows_fc: Optional[Dict[str, Any]],
    geoglows_rp: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
  """Forecast-peak return-period exceedance of the reach (worst of the live GloFAS / GEOGLOWS forecasts).

  Synthetic offline fallbacks (`status="fallback"`) are listed in `excluded_models` and never
  classified: a generated series compared with the reach climatology can paint a spurious
  100-yr corridor. Without any live forecast the corridor is "Not assessed" (grey).
  """
  per_model = {}
  excluded = []
  for key, fc, rps, value_keys in (
      ("glofas", glofas_fc, glofas_rp, ("discharge_median", "discharge_mean")),
      ("geoglows", geoglows_fc, geoglows_rp, ("flow_med",)),
  ):
    if not fc:
      continue
    if fc.get("status") == "fallback":
      excluded.append(key)
      continue
    peak, when = _window_peak(_daily_series(fc.get("data"), *value_keys))
    cls = _classify_exceedance(peak, rps)
    per_model[key] = {
        "peak_flow": peak, "peak_time": when, "rank": cls["rank"], "label": cls["label"],
        "return_period_yrs": _estimate_return_period_yrs(peak, rps or {}),
        "thresholds_status": (rps or {}).get("status"),
    }
  if not per_model:
    out = dict(_EXCEEDANCE_CLASSES[0])
    out.update({
        "label": _UNASSESSED_LABEL if excluded else "Not assessed (no forecast)",
        "risk_level": "UNKNOWN", "return_period": None, "color": _UNASSESSED_COLOR,
        "governing_model": None, "per_model": {}, "excluded_models": excluded, "unit": "m³/s",
    })
    return out
  gov = max(per_model, key=lambda k: per_model[k]["rank"])
  out = dict(_EXCEEDANCE_CLASSES[per_model[gov]["rank"]])
  out.update({"governing_model": gov, "per_model": per_model, "excluded_models": excluded, "unit": "m³/s"})
  return out


def _hydrorivers_shapefile() -> Optional[Path]:
  try:
    from frontend.config import RIVER_NETWORKS_DIR  # pylint: disable=g-import-not-at-top
  except ImportError:
    try:
      from config import RIVER_NETWORKS_DIR  # pylint: disable=g-import-not-at-top
    except ImportError:
      return None
  shp = Path(RIVER_NETWORKS_DIR) / "hydrorivers" / "HydroRIVERS_v10.shp"
  return shp if shp.exists() else None


def _query_hydrorivers_reaches(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> List[Dict[str, Any]]:
  """HydroRIVERS reaches intersecting a bbox (shapely geometries; cached 1 h)."""
  cache_key = f"hyriv_{min_lon:.3f}_{min_lat:.3f}_{max_lon:.3f}_{max_lat:.3f}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached
  reaches: List[Dict[str, Any]] = []
  shp = _hydrorivers_shapefile()
  if shp is not None:
    try:
      import pyogrio  # pylint: disable=g-import-not-at-top

      df = pyogrio.read_dataframe(
          str(shp),
          bbox=(min_lon, min_lat, max_lon, max_lat),
          columns=["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", "UPLAND_SKM", "DIS_AV_CMS", "ORD_STRA"],
      )
      for rec in df.itertuples(index=False):
        geom = rec.geometry
        if geom is None or geom.is_empty:
          continue
        reaches.append({
            "hyriv_id": int(rec.HYRIV_ID),
            "next_down": int(rec.NEXT_DOWN or 0),
            "main_riv": int(rec.MAIN_RIV or 0),
            "upstream_area_km2": float(rec.UPLAND_SKM or 0.0),
            "mean_discharge_m3s": float(rec.DIS_AV_CMS or 0.0),
            "stream_order": int(rec.ORD_STRA or 0),
            "geometry": geom,
        })
    except Exception as e:  # pylint: disable=broad-except
      logger.debug("HydroRIVERS corridor query failed: %s", e)
  set_cached(cache_key, reaches, ttl_seconds=3600)
  return reaches


def _main_stem_chain(
    reaches: List[Dict[str, Any]], lat: float, lon: float, reach_id: Optional[str] = None, max_steps: int = 40
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[float]]:
  """(start reach, main-stem chain up/downstream, snap distance km) around a probe."""
  if not reaches:
    return None, [], None
  from shapely.geometry import Point  # pylint: disable=g-import-not-at-top

  by_id = {r["hyriv_id"]: r for r in reaches}
  pt = Point(lon, lat)
  start = None
  if reach_id:
    try:
      start = by_id.get(int(str(reach_id).upper().replace("HYRIV_", "")))
    except ValueError:
      start = None
  if start is None:
    scored = sorted((r["geometry"].distance(pt), r["hyriv_id"]) for r in reaches)
    near = [by_id[h] for d, h in scored if d <= scored[0][0] + 0.0135]  # ~1.5 km
    start = max(near, key=lambda r: r["upstream_area_km2"])
  snap_km = round(start["geometry"].distance(pt) * 111.32, 2)
  if snap_km > _MAX_CORRIDOR_SNAP_KM:
    return start, [], snap_km
  chain, seen, cur = [start], {start["hyriv_id"]}, start
  for _ in range(max_steps):
    nxt = by_id.get(cur["next_down"])
    if nxt is None or nxt["hyriv_id"] in seen:
      break
    chain.append(nxt)
    seen.add(nxt["hyriv_id"])
    cur = nxt
  upstream_of: Dict[int, List[Dict[str, Any]]] = {}
  for r in reaches:
    upstream_of.setdefault(r["next_down"], []).append(r)
  cur = start
  for _ in range(max_steps):
    ups = [u for u in upstream_of.get(cur["hyriv_id"], []) if u["hyriv_id"] not in seen]
    if not ups:
      break
    cur = max(ups, key=lambda u: u["upstream_area_km2"])
    chain.append(cur)
    seen.add(cur["hyriv_id"])
  return start, chain, snap_km


def _polygonal(geom: Any) -> Optional[Any]:
  """Keeps only the (Multi)Polygon part of a geometry."""
  if geom is None or geom.is_empty:
    return None
  if geom.geom_type in ("Polygon", "MultiPolygon"):
    return geom
  if geom.geom_type == "GeometryCollection":
    from shapely.ops import unary_union  # pylint: disable=g-import-not-at-top

    parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else None
  return None


def _buffer_reaches(
    reaches: List[Dict[str, Any]], half_width_m: Any, ref_lat: float, clip: Optional[Any] = None
) -> Optional[Any]:
  """Metric buffer of reach lines (local equirectangular scaling), optionally clipped."""
  from shapely import affinity  # pylint: disable=g-import-not-at-top
  from shapely.ops import unary_union  # pylint: disable=g-import-not-at-top

  cos_lat = max(math.cos(math.radians(ref_lat)), 0.05)
  parts = []
  for r in reaches:
    hw = half_width_m(r) if callable(half_width_m) else float(half_width_m)
    if hw <= 0:
      continue
    scaled = affinity.scale(r["geometry"], xfact=cos_lat, yfact=1.0, origin=(0, 0))
    parts.append(scaled.buffer(hw / 111320.0, quad_segs=4))
  if not parts:
    return None
  geom = affinity.scale(unary_union(parts), xfact=1.0 / cos_lat, yfact=1.0, origin=(0, 0))
  if clip is not None:
    geom = geom.intersection(clip)
  geom = _polygonal(geom)
  return _polygonal(geom.simplify(0.0002, preserve_topology=True)) if geom is not None else None


def _chain_length_in_km(reaches: List[Dict[str, Any]], ref_lat: float, clip: Optional[Any] = None) -> float:
  from shapely import affinity  # pylint: disable=g-import-not-at-top
  from shapely.ops import unary_union  # pylint: disable=g-import-not-at-top

  if not reaches:
    return 0.0
  lines = unary_union([r["geometry"] for r in reaches])
  if clip is not None:
    lines = lines.intersection(clip)
  cos_lat = max(math.cos(math.radians(ref_lat)), 0.05)
  return affinity.scale(lines, xfact=cos_lat, yfact=1.0, origin=(0, 0)).length * 111.32


def _geojson_feature(geom: Any, props: Dict[str, Any]) -> Dict[str, Any]:
  from shapely.geometry import mapping  # pylint: disable=g-import-not-at-top

  props = dict(props)
  props.setdefault("area_km2", _geom_area_km2(geom))
  return {
      "type": "Feature",
      "geometry": {"type": geom.geom_type, "coordinates": _round_coords(mapping(geom)["coordinates"])},
      "properties": props,
  }


def get_maas_flood_inundation(
    lat: float,
    lon: float,
    gauge_id: Optional[str] = None,
    reach_id: Optional[str] = None,
    river_id: Optional[Any] = None,
) -> Dict[str, Any]:
  """Unified spatial flood-inundation forecast as a GeoJSON FeatureCollection.

  Layers (`properties.layer`):
    * `floodhub_extent`: official FloodHub inundation maps (HIGH / MEDIUM / LOW
      likelihood); when a gauge is in flood but has no official map, a river
      corridor buffered in proportion to its severity (`derived=True`).
    * `camaflood_depth`: the Today's Earth 0.25° unit cell coloured by forecast
      peak floodplain depth (FLDDPH), plus a floodplain corridor inside the cell
      whose area equals the forecast peak flooded fraction (FLDFRC).
    * `reach_exceedance`: the HydroRIVERS main-stem corridor through the probe,
      styled by the worst GloFAS / GEOGLOWS forecast-peak return-period
      exceedance (Normal, 2-Yr Warning, 5-Yr Severe, 20-Yr+ / 100-Yr+ Extreme).
  """
  lat, lon = float(lat), float(lon)
  cache_key = f"maas_flood_{lat:.4f}_{lon:.4f}_{gauge_id or ''}_{reach_id or ''}_{river_id or ''}"
  cached = get_cached(cache_key)
  if cached is not None:
    return cached
  t0 = time.time()
  cell_lat, cell_lon = _snap_cama_cell(lat, lon)
  cell_ring, cell_bbox = _cama_cell_polygon(cell_lat, cell_lon)
  cell_id = _cama_cell_id(cell_lat, cell_lon)
  pad = 0.03
  q_bbox = (
      min(cell_bbox["min_lon"], lon - 0.1) - pad,
      min(cell_bbox["min_lat"], lat - 0.1) - pad,
      max(cell_bbox["max_lon"], lon + 0.1) + pad,
      max(cell_bbox["max_lat"], lat + 0.1) + pad,
  )
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    f_fh = ex.submit(fetch_floodhub_inundation, gauge_id, lat, lon, True)
    f_riv = ex.submit(_query_hydrorivers_reaches, *q_bbox)
    f_gg = ex.submit(_geoglows_forecast_and_rp, lat, lon, river_id)
    f_gl = ex.submit(fetch_glofas_forecast, lat, lon, 15)
    f_glrp = ex.submit(fetch_glofas_return_periods, lat, lon)
    gl, glrp = _future_result(f_gl), _future_result(f_glrp)
    f_te = ex.submit(fetch_todays_earth_forecast, lat, lon, reach_id)
    fh = _future_result(f_fh, {}) or {}
    reaches = _future_result(f_riv, []) or []
    gg_fc, gg_rp = _future_result(f_gg, (None, None))
    te = _future_result(f_te, {}) or {}

  from shapely.geometry import Polygon  # pylint: disable=g-import-not-at-top

  cell_poly = Polygon(cell_ring)
  start, chain, snap_km = _main_stem_chain(reaches, lat, lon, reach_id)
  exceedance = _reach_exceedance_summary(gl, glrp, gg_fc, gg_rp)
  te_ff = te.get("flood_forecast") or {}
  peak_depth = _safe_float(te_ff.get("max_flood_depth_m"))
  peak_frac = _safe_float(te_ff.get("max_flooded_fraction_pct"))
  cell_area = _cell_area_km2(cell_lat)
  te_source = te.get("source", TODAYS_EARTH_SOURCE) + (" — emulated" if te.get("emulated") else "")
  features: List[Dict[str, Any]] = []

  # (1) Today's Earth 0.25° unit cell, coloured by forecast peak FLDDPH.
  features.append({
      "type": "Feature",
      "geometry": {"type": "Polygon", "coordinates": [cell_ring]},
      "properties": {
          "layer": "camaflood_depth",
          "feature_role": "unit_cell",
          "provider": "JAXA Today's Earth",
          "source": te_source,
          "status": te.get("status"),
          "emulated": te.get("emulated"),
          "grid_cell_id": cell_id,
          "label": f"CaMa-Flood unit cell {cell_id}",
          "peak_flood_depth_m": round(peak_depth, 3),
          "peak_flooded_fraction_pct": round(peak_frac, 2),
          "peak_sfcelv_m": te_ff.get("max_sfcelv_m"),
          "peak_depth_time": te_ff.get("peak_depth_time"),
          "cell_area_km2": cell_area,
          "area_km2": cell_area,
          "flooded_area_km2": round(peak_frac / 100.0 * cell_area, 2),
          "color": _depth_color(peak_depth),
      },
  })

  # (2) GloFAS / GEOGLOWS reach return-period exceedance corridor.
  if chain:
    factor = _CORRIDOR_WIDTH_FACTOR[exceedance["rank"]]
    geom = _buffer_reaches(chain, lambda r: _channel_half_width_m(r["mean_discharge_m3s"]) * factor, lat)
    if geom is not None:
      features.append(_geojson_feature(geom, {
          "layer": "reach_exceedance",
          "provider": "GEOGLOWS / GloFAS",
          "source": "HydroRIVERS main stem styled by forecast-peak return-period exceedance",
          "label": f"Reach exceedance — {exceedance['label']}",
          "exceedance_rank": exceedance["rank"],
          "exceedance_label": exceedance["label"],
          "risk_level": exceedance["risk_level"],
          "return_period": exceedance["return_period"],
          "governing_model": exceedance.get("governing_model"),
          "per_model": exceedance.get("per_model"),
          "reach_count": len(chain),
          "hydrorivers_reach": f"HYRIV_{start['hyriv_id']}" if start else None,
          "color": exceedance["color"],
      }))

  # (3) Today's Earth floodplain corridor: area = peak FLDFRC x cell area.
  if chain and peak_frac >= 0.1:
    length_km = _chain_length_in_km(chain, lat, clip=cell_poly)
    if length_km >= 0.5:
      target_km2 = peak_frac / 100.0 * cell_area
      half_width_km = min(max(target_km2 / (2.0 * length_km), 0.05), 12.0)
      geom = _buffer_reaches(chain, half_width_km * 1000.0, lat, clip=cell_poly)
      if geom is not None:
        features.append(_geojson_feature(geom, {
            "layer": "camaflood_depth",
            "feature_role": "floodplain",
            "provider": "JAXA Today's Earth",
            "source": te_source,
            "status": te.get("status"),
            "emulated": te.get("emulated"),
            "grid_cell_id": cell_id,
            "label": "CaMa-Flood forecast floodplain inundation",
            "peak_flood_depth_m": round(peak_depth, 3),
            "peak_flooded_fraction_pct": round(peak_frac, 2),
            "target_flooded_area_km2": round(target_km2, 2),
            "peak_depth_time": te_ff.get("peak_depth_time"),
            "color": _depth_color(max(peak_depth, 0.01)),
        }))

  # (4) FloodHub official inundation maps (LOW -> HIGH so HIGH renders on top).
  fh_polys = list(fh.get("inundation_polygons") or [])
  features.extend(fh_polys)

  # (5) FloodHub severity-scaled corridor when the gauge floods without a map.
  fh_rank = int(fh.get("severity_rank") or 0)
  fh_derived = False
  if not fh_polys and fh_rank >= 1 and chain:
    factor = _FH_DERIVED_WIDTH_FACTOR[min(fh_rank, 3)]
    geom = _buffer_reaches(chain, lambda r: _channel_half_width_m(r["mean_discharge_m3s"]) * factor, lat)
    if geom is not None:
      fh_derived = True
      features.append(_geojson_feature(geom, {
          "layer": "floodhub_extent",
          "provider": "Google FloodHub",
          "source": "River corridor buffered by FloodHub severity (no official inundation map)",
          "derived": True,
          "gauge_id": fh.get("gauge_id"),
          "probability_level": None,
          "label": f"FloodHub {fh.get('severity')} zone (derived)",
          "severity": fh.get("severity"),
          "color": "#22d3ee",
      }))

  counts: Dict[str, int] = {}
  for f in features:
    layer = f["properties"]["layer"]
    counts[layer] = counts.get(layer, 0) + 1
  fh_summary = {k: v for k, v in fh.items() if k != "inundation_polygons"}
  te_summary = {
      "grid_cell_id": cell_id,
      "status": te.get("status"),
      "emulated": te.get("emulated"),
      "source": te.get("source"),
      "note": te.get("note"),
      "max_flood_depth_m": te_ff.get("max_flood_depth_m"),
      "max_flooded_fraction_pct": te_ff.get("max_flooded_fraction_pct"),
      "max_sfcelv_m": te_ff.get("max_sfcelv_m"),
      "peak_depth_time": te_ff.get("peak_depth_time"),
      "cell_area_km2": cell_area,
  }
  result = {
      "type": "FeatureCollection",
      "features": features,
      "metadata": {
          "lat": lat,
          "lon": lon,
          "gauge_id": fh.get("gauge_id") or gauge_id,
          "river_id": (gg_fc or {}).get("river_id"),
          "reach_id": f"HYRIV_{start['hyriv_id']}" if start else reach_id,
          "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "elapsed_s": round(time.time() - t0, 2),
          "layers": {
              "floodhub_extent": {
                  "count": counts.get("floodhub_extent", 0),
                  "status": fh.get("status"),
                  "severity": fh.get("severity"),
                  "trend": fh.get("trend"),
                  "official_maps": bool(fh_polys),
                  "derived": fh_derived,
              },
              "camaflood_depth": {"count": counts.get("camaflood_depth", 0), **te_summary},
              "reach_exceedance": {"count": counts.get("reach_exceedance", 0), **exceedance},
          },
          "floodhub": fh_summary,
          "todays_earth": te_summary,
          "reach_exceedance": exceedance,
          "corridor": {
              "dataset": "HydroRIVERS v1.0",
              "hydrorivers_reach": f"HYRIV_{start['hyriv_id']}" if start else None,
              "reach_count": len(chain),
              "snap_distance_km": snap_km,
          },
          "legend": {
              "floodhub_extent": [
                  {"label": _FH_LEVEL_LABELS[k], "color": _FH_LEVEL_COLORS[k]} for k in ("HIGH", "MEDIUM", "LOW")
              ],
              "reach_exceedance": [{"label": c["label"], "color": c["color"]} for c in _EXCEEDANCE_CLASSES],
              "camaflood_depth": [
                  {"label": "0 m", "color": _depth_color(0.0)},
                  {"label": "< 0.5 m", "color": _depth_color(0.25)},
                  {"label": "0.5-1 m", "color": _depth_color(0.75)},
                  {"label": "1-2 m", "color": _depth_color(1.5)},
                  {"label": "> 2 m", "color": _depth_color(2.5)},
              ],
          },
      },
  }
  degraded = bool(exceedance.get("excluded_models")) or te.get("forcing_status") == "fallback"
  set_cached(cache_key, result, ttl_seconds=FALLBACK_CACHE_TTL_S if degraded else 900)
  return result


# ---------------------------------------------------------------------------
# 4f. Unified 4-Provider Forecast (Streamflow, Flood Summary & Consensus)
# ---------------------------------------------------------------------------

DEFAULT_MAAS_MODELS = ("floodhub", "glofas", "geoglows", "todays_earth")
MAAS_MODEL_NAMES = {
    "floodhub": "Google FloodHub",
    "glofas": "Copernicus GloFAS v4",
    "geoglows": "GEOGLOWS ECMWF v2",
    "todays_earth": "JAXA Today's Earth (CaMa-Flood)",
}
_MODEL_ALIASES = {
    "floodhub": "floodhub", "google_floodhub": "floodhub",
    "glofas": "glofas", "copernicus_glofas": "glofas",
    "geoglows": "geoglows",
    "todays_earth": "todays_earth", "todaysearth": "todays_earth", "te": "todays_earth",
    "jaxa": "todays_earth", "jaxa_todays_earth": "todays_earth", "camaflood": "todays_earth",
}


def _normalize_requested_models(requested: Optional[List[str]]) -> List[str]:
  out: List[str] = []
  for m in requested or []:
    key = _MODEL_ALIASES.get(str(m).strip().lower().replace("-", "_").replace(" ", "_").replace("'", ""))
    if key and key not in out:
      out.append(key)
  return out or list(DEFAULT_MAAS_MODELS)


def _spread_confidence(
    central: Optional[float], p25: Optional[float], p75: Optional[float], live: bool = True,
    fallback_thresholds: bool = False,
) -> str:
  """High / Medium / Low from the relative inter-quartile spread at the peak."""
  if not live:
    return "Low"
  level = 1
  if central is not None and p25 is not None and p75 is not None and central > 0:
    rel = max(p75 - p25, 0.0) / central
    level = 2 if rel < 0.25 else (1 if rel < 0.6 else 0)
  if fallback_thresholds:
    level = max(level - 1, 0)
  return ("Low", "Medium", "High")[level]


def _resolve_floodhub_gauge(
    lat: float, lon: float, gauge_id: Optional[str], target_area_km2: Optional[float] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[float]]:
  """(gauge_id, gauge metadata, distance km) for the probe's FloodHub gauge."""
  if gauge_id:
    return gauge_id, None, None
  status, reachable, dist = _search_floodhub_status_near(lat, lon, target_area_km2=target_area_km2)
  if status:
    loc = status.get("gaugeLocation") or {}
    return status.get("gaugeId"), {
        "lat": loc.get("latitude"), "lon": loc.get("longitude"),
        "source": status.get("source"), "quality_verified": status.get("qualityVerified"),
    }, dist
  if not reachable:
    # FloodHub unreachable: keep the legacy offline behaviour (nearest sample gauge).
    best = None
    for g in fetch_floodhub_gauges_bbox(lat - 0.25, lon - 0.25, lat + 0.25, lon + 0.25) or []:
      try:
        d = _haversine_km(lat, lon, float(g["lat"]), float(g["lon"]))
      except (KeyError, TypeError, ValueError):
        continue
      if best is None or d < best[0]:
        best = (d, g)
    if best:
      g = best[1]
      return g["gauge_id"], {
          "lat": g.get("lat"), "lon": g.get("lon"), "source": g.get("source"),
          "quality_verified": g.get("quality_verified"),
      }, round(best[0], 2)
  return None, None, None


def _consensus_row(
    model: str, available: bool, status: Optional[str], peak: Optional[float], peak_time: Optional[str],
    rps: Optional[Dict[str, Any]], thresholds_source: Optional[str], confidence: str,
    unit: str = "m³/s", independent: bool = True, **extra: Any,
) -> Dict[str, Any]:
  # A synthetic offline fallback (status="fallback") is displayed but never judged
  # against the model's climatology: its generated peak would be a spurious class.
  synthetic = available and status == "fallback"
  cls = _classify_exceedance(peak, rps) if (available and not synthetic) else None
  row = {
      "model": model,
      "name": MAAS_MODEL_NAMES.get(model, model),
      "available": available,
      "status": status,
      "unit": unit,
      "peak_flow": round(peak, 2) if peak is not None else None,
      "peak_time": peak_time,
      "return_period": cls["return_period"] if cls else None,
      "return_period_yrs": _estimate_return_period_yrs(peak, rps or {}) if (cls and unit == "m³/s") else None,
      "exceedance_rank": cls["rank"] if cls else None,
      "exceedance_label": cls["label"] if cls else (_UNASSESSED_LABEL if synthetic else None),
      "risk_level": cls["risk_level"] if cls else "UNKNOWN",
      "confidence": confidence if available else "N/A",
      "thresholds_source": thresholds_source,
      "independent": independent,
  }
  row.update(extra)
  return row


def _align(daily: Dict[str, float], dates: List[str], ndigits: int = 2) -> List[Optional[float]]:
  return [round(daily[d], ndigits) if d in daily else None for d in dates]


def get_unified_maas_forecast(
    lat: float,
    lon: float,
    gauge_id: Optional[str] = None,
    river_id: Optional[Any] = None,
    requested_models: Optional[List[str]] = None,
    reach_id: Optional[str] = None,
    upstream_area_km2: Optional[Any] = None,
    area_min_km2: Optional[Any] = None,
    network: Optional[str] = None,
) -> Dict[str, Any]:
  """Unified 4-provider MaaS forecast: streamflow, return periods, flood summary.

  A superset of `aggregate_maas_forecast()` (same `location`, `thresholds` and
  `models` contract) adding `models.todays_earth`, `virtual_station`,
  `consensus` (one row per provider, each judged against its own return
  periods), `flood_summary`, a `flood_inundation` summary block, and a
  date-aligned daily `timeline` for charting. Providers are queried in
  parallel; every provider degrades independently to a flagged fallback.
  """
  lat, lon = float(lat), float(lon)
  t0 = time.time()
  models = _normalize_requested_models(requested_models)

  # When the user clicks a river line of known upstream area, snap GloFAS (and the Today's Earth forcing) and
  # GEOGLOWS to the element on that river rather than the nearest small tributary.
  snapped = None
  if upstream_area_km2 not in (None, ""):
    try:
      from frontend.maas_networks import resolve_click  # pylint: disable=g-import-not-at-top
    except ImportError:
      from maas_networks import resolve_click  # pylint: disable=g-import-not-at-top
    try:
      snapped = resolve_click(lat, lon, upstream_area_km2, area_min_km2=area_min_km2, network=network,
                              river_id=river_id or reach_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Upstream-area snapping failed at (%.3f, %.3f): %s", lat, lon, e)
  gl_snap = (snapped or {}).get("glofas")
  gg_snap = (snapped or {}).get("geoglows")
  target_area = (snapped or {}).get("target_area_km2")
  fh_lat = float(gl_snap["lat"]) if gl_snap else lat
  fh_lon = float(gl_snap["lon"]) if gl_snap else lon
  gl_lat = float(gl_snap.get("query_lat", gl_snap["lat"])) if gl_snap else lat
  gl_lon = float(gl_snap.get("query_lon", gl_snap["lon"])) if gl_snap else lon
  gg_river_id = gg_snap["river_id"] if (gg_snap and not _is_geoglows_river_id(river_id)) else river_id

  def _run_floodhub() -> Optional[Dict[str, Any]]:
    gid, meta, dist = _resolve_floodhub_gauge(fh_lat, fh_lon, gauge_id, target_area_km2=target_area)
    if not gid:
      return None
    return {
        "gauge_id": gid, "meta": meta, "distance_km": dist,
        "forecast": fetch_floodhub_forecast(gid),
        "inundation": fetch_floodhub_inundation(gid, fh_lat, fh_lon, include_polygons=False),
    }

  def _run_te_prep() -> None:
    _fetch_ground_elevation(gl_lat, gl_lon)
    _probe_todays_earth_catalog()

  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    f_fh = ex.submit(_run_floodhub) if "floodhub" in models else None
    f_gg = ex.submit(_geoglows_forecast_and_rp, lat, lon, gg_river_id) if "geoglows" in models else None
    f_gl = ex.submit(fetch_glofas_forecast, gl_lat, gl_lon, 15)
    f_glrp = ex.submit(fetch_glofas_return_periods, gl_lat, gl_lon)
    f_prep = ex.submit(_run_te_prep) if "todays_earth" in models else None
    gl, glrp = _future_result(f_gl), _future_result(f_glrp)
    _future_result(f_prep)
    f_te = ex.submit(fetch_todays_earth_forecast, gl_lat, gl_lon, reach_id) if "todays_earth" in models else None
    fh = _future_result(f_fh)
    gg_fc, gg_rp = _future_result(f_gg, (None, None))
    te = _future_result(f_te)

  fh_fc = (fh or {}).get("forecast") or {}
  fh_inund = (fh or {}).get("inundation") or {}
  fh_th = fh_fc.get("thresholds") or {}
  fh_is_q = str(fh_fc.get("unit") or "").upper() == "CUBIC_METERS_PER_SECOND"
  gid = (fh or {}).get("gauge_id") or gauge_id

  # --- Model outputs (existing contract + additive fields) ---
  models_output: Dict[str, Any] = {}
  if "floodhub" in models:
    if fh_fc:
      models_output["floodhub"] = {
          **fh_fc,
          "severity": fh_inund.get("severity"),
          "trend": fh_inund.get("trend"),
          "severity_source": fh_inund.get("severity_source"),
          "gauge_location": fh_inund.get("gauge_location") or (fh or {}).get("meta"),
          "distance_km": (fh or {}).get("distance_km"),
          "inundation_maps_available": fh_inund.get("inundation_maps_available", False),
      }
    else:
      models_output["floodhub"] = {
          "available": False, "status": "unavailable",
          "message": "No FloodHub gauge within 30 km of the probe.",
      }
  if "geoglows" in models and gg_fc:
    models_output["geoglows"] = {**gg_fc, "return_periods": gg_rp}
  if "glofas" in models and gl:
    models_output["glofas"] = {**gl, "return_periods": glrp}
  if "todays_earth" in models and te:
    models_output["todays_earth"] = te

  # --- Master thresholds for the hydrograph (always m³/s) ---
  if fh_fc.get("status") == "live" and fh_is_q and fh_th.get("warning_2yr") is not None:
    thresholds = {
        "warning_2yr": fh_th.get("warning_2yr"), "danger_5yr": fh_th.get("danger_5yr"),
        "extreme_20yr": fh_th.get("extreme_20yr"), "extreme_100yr": None,
        "source": "Google FloodHub gauge model thresholds", "unit": "m³/s",
    }
  elif glrp and glrp.get("status") == "live":
    thresholds = _thresholds_from_return_periods(glrp, "Copernicus GloFAS v4 reanalysis (EV1 fit)")
  elif gg_rp and gg_rp.get("status") in ("live", "computed"):
    thresholds = _thresholds_from_return_periods(gg_rp, gg_rp.get("source") or "GEOGLOWS v2")
  elif glrp:
    thresholds = _thresholds_from_return_periods(glrp, glrp.get("source") or "Index-flood scaling")
  else:
    thresholds = {
        "warning_2yr": 37.5, "danger_5yr": 75.0, "extreme_20yr": 142.5, "extreme_100yr": None,
        "source": "Static fallback", "unit": "m³/s",
    }

  # --- Consensus: each provider against its own climatology ---
  consensus: List[Dict[str, Any]] = []
  gl_daily = _daily_series((gl or {}).get("data"), "discharge_median", "discharge_mean")
  if "floodhub" in models:
    if fh_fc:
      live = fh_fc.get("status") == "live"
      severity = fh_inund.get("severity")
      sev_risk = _FH_SEVERITY_TO_RISK.get(severity or "")
      sev_live = bool(sev_risk) and fh_inund.get("severity_source") == "floodhub_flood_status"
      peak_t, peak = None, None
      if live:
        pts = [(str(r.get("time", "")), _safe_float_or_none(r.get("discharge"))) for r in fh_fc.get("data") or []]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pts = [p for p in pts if p[1] is not None]
        window = [p for p in pts if p[0][:10] >= today] or pts
        if window:
          peak_t, peak = max(window, key=lambda p: p[1])
      fh_rps = {
          "return_period_2": fh_th.get("warning_2yr"),
          "return_period_5": fh_th.get("danger_5yr"),
          "return_period_20": fh_th.get("extreme_20yr"),
      }
      row = _consensus_row(
          "floodhub", live or sev_live,
          fh_fc.get("status") if live else ("severity_only" if sev_live else "fallback"),
          peak, peak_t, fh_rps if live else None,
          "FloodHub gauge model thresholds" if live else ("FloodHub official flood status" if sev_live else None),
          ("High" if fh_inund.get("quality_verified") else "Medium") if (live or sev_live) else "Low",
          unit="m³/s" if fh_is_q else "m",
          value_type="discharge" if fh_is_q else "stage",
          severity=severity, trend=fh_inund.get("trend"),
          forecast_status=fh_fc.get("status"),
          fallback_reason=fh_fc.get("fallback_reason"),
      )
      if sev_risk and row["available"]:
        row["risk_level"], row["risk_source"] = sev_risk, "floodhub_severity"
        if not live and sev_live:
          # No usable forecast series: express the official severity on the exceedance scale.
          rank = _FH_SEVERITY_RANK.get(severity, 0)
          row.update(
              exceedance_rank=rank,
              exceedance_label=_FH_SEVERITY_LABELS.get(severity, severity),
              return_period=None,
              return_period_yrs=None,
          )
      else:
        row["risk_source"] = "forecast_vs_thresholds" if row["available"] else None
      consensus.append(row)
    else:
      consensus.append(_consensus_row("floodhub", False, "unavailable", None, None, None, None, "N/A"))
  if "glofas" in models:
    if gl:
      peak, when = _window_peak(gl_daily)
      p25 = _daily_series(gl.get("data"), "discharge_p25").get(when)
      p75 = _daily_series(gl.get("data"), "discharge_p75").get(when)
      consensus.append(_consensus_row(
          "glofas", True, gl.get("status"), peak, when, glrp, (glrp or {}).get("source"),
          _spread_confidence(peak, p25, p75, gl.get("status") == "live", (glrp or {}).get("status") != "live"),
      ))
    else:
      consensus.append(_consensus_row("glofas", False, "unavailable", None, None, None, None, "N/A"))
  if "geoglows" in models:
    if gg_fc:
      gg_daily = _daily_series(gg_fc.get("data"), "flow_med")
      peak, when = _window_peak(gg_daily)
      p25 = _daily_series(gg_fc.get("data"), "flow_25p").get(when)
      p75 = _daily_series(gg_fc.get("data"), "flow_75p").get(when)
      consensus.append(_consensus_row(
          "geoglows", True, gg_fc.get("status"), peak, when, gg_rp, (gg_rp or {}).get("source"),
          _spread_confidence(
              peak, p25, p75, gg_fc.get("status") == "live",
              (gg_rp or {}).get("status") not in ("live", "computed"),
          ),
          thresholds_status=(gg_rp or {}).get("status"),
      ))
    else:
      consensus.append(_consensus_row("geoglows", False, "unavailable", None, None, None, None, "N/A"))
  te_ff = (te or {}).get("flood_forecast") or {}
  if "todays_earth" in models:
    if te:
      te_daily = _daily_series(te.get("data"), "discharge_mean")
      peak, when = _window_peak(te_daily)
      p25 = _daily_series(te.get("data"), "discharge_p25").get(when)
      p75 = _daily_series(te.get("data"), "discharge_p75").get(when)
      te_status = te.get("status")
      if te.get("emulated"):
        # An emulation forced by a synthetic GloFAS fallback is itself synthetic.
        te_status = "fallback" if te.get("forcing_status") == "fallback" else "emulated"
      consensus.append(_consensus_row(
          "todays_earth", True, te_status, peak, when, glrp,
          "GloFAS v4 reanalysis EV1 (CaMa-Flood emulator climatology)",
          "Low" if te.get("emulated") else _spread_confidence(peak, p25, p75),
          independent=not te.get("emulated"),
          emulated=bool(te.get("emulated")),
          peak_flood_depth_m=te_ff.get("max_flood_depth_m"),
          peak_flood_fraction_pct=te_ff.get("max_flooded_fraction_pct"),
          peak_sfcelv_m=te_ff.get("max_sfcelv_m"),
      ))
    else:
      consensus.append(_consensus_row("todays_earth", False, "unavailable", None, None, None, None, "N/A"))

  # --- Cross-model flood summary ---
  # Evidence = independent providers with a live forecast (or FloodHub's official
  # severity); synthetic offline fallbacks and the TE emulator do not vote.
  independent = [
      r for r in consensus
      if r["available"] and r["independent"] and r.get("status") in ("live", "severity_only")
  ]
  excluded = [
      r["name"] for r in consensus
      if r["available"] and r["independent"] and r.get("status") not in ("live", "severity_only")
  ]
  n_exceed = sum(1 for r in independent if (r.get("exceedance_rank") or 0) >= 1)
  worst = max(independent, key=lambda r: r.get("exceedance_rank") or 0) if independent else None
  risks = [r["risk_level"] for r in independent if r.get("risk_level") in _RISK_RANK]
  overall = max(risks, key=lambda k: _RISK_RANK[k]) if risks else "UNKNOWN"
  agreement = f"{n_exceed} of {len(independent)} independent models forecast ≥ 2-yr exceedance"
  if te and te.get("emulated"):
    agreement += " (Today's Earth is emulated from GloFAS and excluded)"
  if excluded:
    agreement += f"; offline fallback excluded: {', '.join(excluded)}"
  flood_summary = {
      "max_inundation_depth_m": te_ff.get("max_flood_depth_m"),
      "max_flooded_fraction_pct": te_ff.get("max_flooded_fraction_pct"),
      "peak_sfcelv_m": te_ff.get("max_sfcelv_m"),
      "peak_depth_time": te_ff.get("peak_depth_time"),
      "inundation_source": (
          (TODAYS_EARTH_SOURCE + (" — emulated" if te.get("emulated") else "")) if te else None
      ),
      "floodhub_severity": fh_inund.get("severity") if fh_fc else "UNAVAILABLE",
      "floodhub_trend": fh_inund.get("trend") if fh_fc else None,
      "floodhub_severity_source": fh_inund.get("severity_source"),
      "floodhub_inundation_maps_available": bool(fh_inund.get("inundation_maps_available")),
      "floodhub_inundation_map_levels": fh_inund.get("inundation_map_levels") or [],
      "return_period_exceedance": worst["exceedance_label"] if worst else "Unknown",
      "return_period_exceedance_model": worst["name"] if worst else None,
      "max_exceedance_rank": (worst.get("exceedance_rank") or 0) if worst else None,
      "models_exceeding_2yr": n_exceed,
      "independent_models_evaluated": len(independent),
      "overall_risk_level": overall,
      "agreement": agreement,
  }

  # --- Flood-inundation summary block (geometry via /api/maas/flood-inundation) ---
  inund_params: Dict[str, Any] = {"lat": lat, "lon": lon}
  if gid:
    inund_params["gauge_id"] = gid
  if reach_id:
    inund_params["reach_id"] = reach_id
  if gg_fc and _is_geoglows_river_id(gg_fc.get("river_id")):
    inund_params["river_id"] = gg_fc.get("river_id")
  flood_inundation = {
      "endpoint": "/api/maas/flood-inundation?" + urllib.parse.urlencode(inund_params),
      "layers": ["floodhub_extent", "camaflood_depth", "reach_exceedance"],
      "floodhub": {
          k: fh_inund.get(k) for k in (
              "status", "gauge_id", "severity", "trend", "severity_source", "issued_time",
              "inundation_maps_available", "inundation_map_levels", "inundation_maps_time_range",
          )
      },
      "todays_earth": {
          "grid_cell_id": (te or {}).get("grid_cell_id"),
          "status": (te or {}).get("status"),
          "emulated": (te or {}).get("emulated"),
          "max_flood_depth_m": te_ff.get("max_flood_depth_m"),
          "max_flooded_fraction_pct": te_ff.get("max_flooded_fraction_pct"),
          "max_sfcelv_m": te_ff.get("max_sfcelv_m"),
          "peak_depth_time": te_ff.get("peak_depth_time"),
      },
      "reach_exceedance": _reach_exceedance_summary(gl, glrp, gg_fc, gg_rp),
  }

  # --- Virtual station (per-provider snapping of the probe) ---
  gl_center_lat, gl_center_lon = (
      (float(gl_snap["lat"]), float(gl_snap["lon"])) if gl_snap else _glofas_cell_center(lat, lon)
  )
  fh_loc = fh_inund.get("gauge_location") or (fh or {}).get("meta") or {}
  virtual_station = {
      "probe": {
          "lat": lat, "lon": lon,
          "network": network,
          "upstream_area_km2": (snapped or {}).get("target_area_km2"),
      },
      "floodhub_gauge": {
          "gauge_id": gid, "lat": fh_loc.get("lat"), "lon": fh_loc.get("lon"),
          "distance_km": (fh or {}).get("distance_km"), "status": fh_fc.get("status"),
      } if fh_fc else None,
      "geoglows_reach": {
          "river_id": gg_fc.get("river_id"), "status": gg_fc.get("status"),
          "upstream_area_km2": (gg_snap or {}).get("upstream_area_km2"),
          "offset_km": (gg_snap or {}).get("offset_km"),
      } if gg_fc else None,
      "glofas_cell": {
          "cell_center_lat": gl_center_lat, "cell_center_lon": gl_center_lon, "resolution_deg": 0.05,
          "upstream_area_km2": (gl_snap or {}).get("upstream_area_km2"),
          "offset_cells": (gl_snap or {}).get("offset_cells"),
          "status": (gl or {}).get("status"),
      },
      "todays_earth_cell": {
          "grid_cell_id": te.get("grid_cell_id"),
          "cell_center_lat": te.get("cell_center_lat"),
          "cell_center_lon": te.get("cell_center_lon"),
          "resolution_deg": CAMA_GRID_RES_DEG,
          "area_km2": te.get("cell_area_km2"),
          "status": te.get("status"),
          "emulated": te.get("emulated"),
          "label": f"CaMa 0.25° [{te.get('cell_center_lat'):.3f}, {te.get('cell_center_lon'):.3f}]",
          "model_chain": "MATSIRO + CaMa-Flood",
      } if te else None,
      "hydrorivers_reach": reach_id,
  }

  # --- Date-aligned daily timeline for charting ---
  series_daily: Dict[str, Dict[str, Dict[str, float]]] = {}
  if fh_fc and fh_fc.get("status") == "live":
    series_daily["floodhub"] = {"central": _daily_series(fh_fc.get("data"), "discharge")}
  if gl:
    series_daily["glofas"] = {
        "central": gl_daily,
        "p25": _daily_series(gl.get("data"), "discharge_p25"),
        "p75": _daily_series(gl.get("data"), "discharge_p75"),
        "max": _daily_series(gl.get("data"), "discharge_max"),
        "min": _daily_series(gl.get("data"), "discharge_min"),
    }
  if gg_fc:
    series_daily["geoglows"] = {
        "central": _daily_series(gg_fc.get("data"), "flow_med"),
        "p25": _daily_series(gg_fc.get("data"), "flow_25p"),
        "p75": _daily_series(gg_fc.get("data"), "flow_75p"),
        "max": _daily_series(gg_fc.get("data"), "flow_max"),
        "min": _daily_series(gg_fc.get("data"), "flow_min"),
    }
  if te:
    series_daily["todays_earth"] = {
        "central": _daily_series(te.get("data"), "discharge_mean"),
        "p25": _daily_series(te.get("data"), "discharge_p25"),
        "p75": _daily_series(te.get("data"), "discharge_p75"),
        "rivout": _daily_series(te.get("data"), "rivout"),
        "fldout": _daily_series(te.get("data"), "fldout"),
        "flddph_m": _daily_series(te.get("data"), "flddph_m"),
        "fldfrc_pct": _daily_series(te.get("data"), "fldfrc_pct"),
    }
  dates = sorted({d for s in series_daily.values() for col in s.values() for d in col})
  timeline = {"dates": dates, "series": {}}
  for model, cols in series_daily.items():
    timeline["series"][model] = {name: _align(col, dates, 3 if name == "flddph_m" else 2) for name, col in cols.items()}
  if "floodhub" in timeline["series"]:
    timeline["series"]["floodhub"]["unit"] = "m³/s" if fh_is_q else "m"
    timeline["series"]["floodhub"]["axis"] = "discharge" if fh_is_q else "stage"
  timeline["status"] = {
      "floodhub": fh_fc.get("status") if fh_fc else None,
      "glofas": (gl or {}).get("status"),
      "geoglows": (gg_fc or {}).get("status"),
      "todays_earth": ("emulated" if te.get("emulated") else te.get("status")) if te else None,
  }

  return {
      "location": {
          "lat": lat, "lon": lon, "gauge_id": gid,
          "river_id": (gg_fc or {}).get("river_id") or river_id, "reach_id": reach_id,
      },
      "thresholds": thresholds,
      "thresholds_by_model": {
          "floodhub": {**fh_th, "unit": fh_fc.get("unit")} if (fh_fc and fh_fc.get("status") == "live") else None,
          "glofas": _thresholds_from_return_periods(glrp, glrp.get("source")) if glrp else None,
          "geoglows": _thresholds_from_return_periods(gg_rp, gg_rp.get("source")) if gg_rp else None,
          "todays_earth": (
              _thresholds_from_return_periods(glrp, "GloFAS v4 reanalysis EV1 (emulator climatology)")
              if (te and glrp) else None
          ),
      },
      "return_periods": {"glofas": glrp, "geoglows": gg_rp},
      "models": models_output,
      "virtual_station": virtual_station,
      "consensus": consensus,
      "flood_summary": flood_summary,
      "flood_inundation": flood_inundation,
      "timeline": timeline,
      "meta": {
          "models_requested": models,
          "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "elapsed_s": round(time.time() - t0, 2),
          "todays_earth_service": todays_earth_service_status(),
      },
  }


# ---------------------------------------------------------------------------
# 5. Offline Fallback Synthetic Data Generators
# ---------------------------------------------------------------------------

def _generate_sample_floodhub_gauges(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> List[Dict[str, Any]]:
  """Generates synthetic FloodHub gauges across a bbox for testing/offline resilience."""
  gauges = []
  severities = ["NO_FLOODING", "NO_FLOODING", "NO_FLOODING", "ABOVE_NORMAL", "SEVERE"]
  now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")

  for i in range(8):
    glat = min_lat + (max_lat - min_lat) * ((i * 3 + 1) % 10) / 10.0
    glon = min_lon + (max_lon - min_lon) * ((i * 7 + 2) % 10) / 10.0
    gid = f"hybas_71{int(abs(glat * 100)) + i:08d}"
    sev = severities[i % len(severities)]
    gauges.append({
        "gauge_id": gid,
        "lat": round(glat, 4),
        "lon": round(glon, 4),
        "severity": sev,
        "forecast_trend": "RISING" if sev in ("ABOVE_NORMAL", "SEVERE") else "NO_CHANGE",
        "issued_time": now_iso,
        "quality_verified": (i % 2 == 0),
        "source": "HYBAS",
    })
  return gauges


def _generate_sample_floodhub_forecast(gauge_id: str):
  """Generates realistic synthetic 7-day FloodHub daily forecast and return period thresholds."""
  now = datetime.now(timezone.utc)
  base_flow = 8.5 + (hash(gauge_id) % 25)
  data = []

  for day in range(8):
    t_dt = now + timedelta(days=day)
    # Realistic flood hydrograph peak around Day 3-4
    peak_factor = 2.4 * math.exp(-((day - 3.5) / 1.5) ** 2)
    val = round(base_flow + (base_flow * peak_factor), 2)
    data.append({
        "time": t_dt.strftime("%Y-%m-%dT00:00:00Z"),
        "discharge": val,
    })

  thresholds = {
      "warning_2yr": round(base_flow * 1.8, 1),
      "danger_5yr": round(base_flow * 3.2, 1),
      "extreme_20yr": round(base_flow * 5.0, 1),
  }
  return data, thresholds


def _generate_sample_geoglows_forecast(river_id: int) -> List[Dict[str, Any]]:
  """Generates realistic synthetic 15-day hourly GEOGLOWS ensemble timeseries."""
  now = datetime.now(timezone.utc)
  base_flow = 7.0 + (abs(river_id) % 20)
  data = []

  for h in range(0, 360, 3):  # 15 days at 3-hour resolution
    t_dt = now + timedelta(hours=h)
    day_frac = h / 24.0
    storm = 2.8 * math.exp(-((day_frac - 3.8) / 1.8) ** 2)
    med = base_flow * (1.0 + storm)
    data.append({
        "time": t_dt.strftime("%Y-%m-%dT%H:00:00Z"),
        "flow_med": round(med, 2),
        "flow_avg": round(med * 1.05, 2),
        "flow_max": round(med * 1.45, 2),
        "flow_min": round(med * 0.75, 2),
        "flow_25p": round(med * 0.90, 2),
        "flow_75p": round(med * 1.15, 2),
        "high_res": round(med * 1.10, 2),
    })
  return data


def _generate_sample_glofas_forecast(lat: float, lon: float, forecast_days: int) -> List[Dict[str, Any]]:
  """Generates realistic synthetic 15-day daily GloFAS forecast statistics."""
  now = datetime.now(timezone.utc)
  base_flow = 10.0 + (abs(int(lat * 10)) % 15)
  data = []

  for d in range(forecast_days):
    t_dt = now + timedelta(days=d)
    storm = 2.2 * math.exp(-((d - 4.0) / 2.0) ** 2)
    mean_val = round(base_flow * (1.0 + storm), 2)
    data.append({
        "time": t_dt.strftime("%Y-%m-%d"),
        "discharge_mean": mean_val,
        "discharge_median": round(mean_val * 0.98, 2),
        "discharge_max": round(mean_val * 1.35, 2),
        "discharge_min": round(mean_val * 0.70, 2),
        "discharge_p25": round(mean_val * 0.88, 2),
        "discharge_p75": round(mean_val * 1.12, 2),
    })
  return data


# ---------------------------------------------------------------------------
# 6. Multi-Model Watershed Polygon Resolution & Persistent SQLite Caching
# ---------------------------------------------------------------------------

def _init_watershed_cache_db() -> None:
  """Initializes SQLite database for persistent watershed geometry caching."""
  try:
    WATERSHED_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(WATERSHED_CACHE_DB))) as conn:
      conn.execute("""
        CREATE TABLE IF NOT EXISTS watershed_cache (
          cache_key TEXT PRIMARY KEY,
          fabric TEXT,
          basin_id TEXT,
          area_km2 REAL,
          geojson TEXT,
          created_at REAL
        )
      """)
      conn.commit()
  except Exception as e:
    logger.debug("Failed to initialize watershed SQLite database: %s", e)


def _get_cached_watershed(cache_key: str) -> Optional[Dict[str, Any]]:
  """Retrieves a cached GeoJSON watershed feature by cache key."""
  try:
    if not WATERSHED_CACHE_DB.exists():
      return None
    with closing(sqlite3.connect(str(WATERSHED_CACHE_DB))) as conn:
      cur = conn.cursor()
      cur.execute("SELECT geojson FROM watershed_cache WHERE cache_key = ?", (cache_key,))
      row = cur.fetchone()
      if row and row[0]:
        return json.loads(row[0])
  except Exception as e:
    logger.debug("Failed to read from watershed SQLite cache: %s", e)
  return None


def _save_cached_watershed(
    cache_key: str,
    fabric: str,
    basin_id: str,
    area_km2: float,
    geojson_dict: Dict[str, Any],
) -> None:
  """Saves a computed GeoJSON watershed feature to the SQLite database."""
  try:
    _init_watershed_cache_db()
    with closing(sqlite3.connect(str(WATERSHED_CACHE_DB))) as conn:
      conn.execute(
          """
          INSERT OR REPLACE INTO watershed_cache
          (cache_key, fabric, basin_id, area_km2, geojson, created_at)
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (cache_key, fabric, str(basin_id), float(area_km2), json.dumps(geojson_dict), time.time()),
      )
      conn.commit()
  except Exception as e:
    logger.debug("Failed to save to watershed SQLite cache: %s", e)


def get_maas_watershed_polygon(
    lat: float,
    lon: float,
    fabric: str = "hydroatlas_full",
    gauge_id: Optional[str] = None,
    river_id: Optional[int] = None,
    geofabric: Optional[str] = None,
) -> Dict[str, Any]:
  """Resolves and highlights the authentic watershed polygon corresponding to the selected hydrofabric.

  Supported Fabrics:
    1. 'hydroatlas_full': Full upstream contributing drainage area (Google FloodHub operational basin).
    2. 'merit_reach': Unit subcatchment reach polygon (GEOGLOWS ECMWF streamflow routing basin).
    3. 'hydroatlas_unit': Single Level 12 elementary unit catchment polygon (HydroBASINS unit).
    4. 'glofas_cell': 0.05° LISFLOOD regular grid cell extent (Copernicus GloFAS river grid).
    5. 'camaflood_unit': 0.25° CaMa-Flood unit catchment grid cell (JAXA Today's Earth TE-Global).

  Args:
    lat: Probe latitude.
    lon: Probe longitude.
    fabric: Selected geofabric identifier.
    gauge_id: Optional FloodHub gauge identifier.
    river_id: Optional GEOGLOWS 9-digit reach COMID.
    geofabric: Alias of `fabric` (takes precedence when provided).

  Returns:
    GeoJSON Feature dictionary containing the polygon geometry and hydrological attributes.
  """
  fabric = (geofabric or fabric or "hydroatlas_full").strip().lower()

  # -------------------------------------------------------------
  # FABRIC 5: JAXA Today's Earth CaMa-Flood 0.25° Unit Catchment Cell
  # -------------------------------------------------------------
  if fabric == "camaflood_unit":
    cell_lat, cell_lon = _snap_cama_cell(lat, lon)
    ring, bbox = _cama_cell_polygon(cell_lat, cell_lon)
    label = "Today's Earth CaMa-Flood Unit Grid (0.25°)"
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "fabric": "camaflood_unit",
            "fabric_name": label,
            "geofabric": "camaflood_unit",
            "geofabric_label": label,
            "model": "JAXA Today's Earth (MATSIRO + CaMa-Flood)",
            "source": f"{TODAYS_EARTH_SOURCE} unit-catchment grid",
            "service_status": todays_earth_service_status(),
            "grid_cell_id": _cama_cell_id(cell_lat, cell_lon),
            "cell_center_lat": cell_lat,
            "cell_center_lon": cell_lon,
            "area_km2": _cell_area_km2(cell_lat),
            "resolution": "0.25° (~28 km)",
            "bbox": bbox,
        },
    }

  # -------------------------------------------------------------
  # FABRIC 1: Copernicus GloFAS 0.05° River Cell (LISFLOOD Grid)
  # -------------------------------------------------------------
  if fabric == "glofas_cell":
    grid_res = 0.05
    # GloFAS v4 cell centres sit at x.025 / x.075 (cell edges on the 0.05° lattice).
    cell_lat, cell_lon = _glofas_cell_center(lat, lon, grid_res)
    half = grid_res / 2.0
    box_coords = [
        [round(cell_lon - half, 5), round(cell_lat - half, 5)],
        [round(cell_lon + half, 5), round(cell_lat - half, 5)],
        [round(cell_lon + half, 5), round(cell_lat + half, 5)],
        [round(cell_lon - half, 5), round(cell_lat + half, 5)],
        [round(cell_lon - half, 5), round(cell_lat - half, 5)],
    ]
    area_km2 = round((0.05 * 111.0) * (0.05 * 111.0 * math.cos(math.radians(cell_lat))), 1)
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [box_coords],
        },
        "properties": {
            "fabric": "glofas_cell",
            "fabric_name": "Copernicus GloFAS 0.05° River Cell",
            "model": "LISFLOOD Routing Grid",
            "cell_center_lat": round(cell_lat, 4),
            "cell_center_lon": round(cell_lon, 4),
            "area_km2": area_km2,
            "resolution": "0.05° (~5 km)",
            "bbox": {
                "min_lon": round(cell_lon - half, 5),
                "min_lat": round(cell_lat - half, 5),
                "max_lon": round(cell_lon + half, 5),
                "max_lat": round(cell_lat + half, 5),
            },
        },
    }
    return feature

  # -------------------------------------------------------------
  # FABRIC 2: MERIT-Hydro Reach Catchment (GEOGLOWS ECMWF)
  # -------------------------------------------------------------
  if fabric == "merit_reach":
    eff_river_id = river_id
    if not eff_river_id:
      eff_river_id = fetch_geoglows_river_id(lat, lon)

    if eff_river_id:
      cache_key = f"merit_reach_{eff_river_id}"
      cached = _get_cached_watershed(cache_key)
      if cached:
        return cached

      # Fast direct lookup from partitioned Pfafstetter shapefile
      if _find_merit_shp is not None:
        cat_shp = _find_merit_shp(eff_river_id, "cat")
        if cat_shp and cat_shp.exists():
          try:
            import pyogrio
            from shapely.geometry import mapping

            df = pyogrio.read_dataframe(str(cat_shp), where=f"COMID = {eff_river_id}")
            if not df.empty:
              geom = df.geometry.values[0]
              area_col = "unitarea" if "unitarea" in df.columns else ("uparea" if "uparea" in df.columns else None)
              if area_col:
                area_km2 = float(df[area_col].values[0])
              else:
                area_km2 = float(geom.area * 111.0 * 111.0 * math.cos(math.radians(lat)))

              feature = {
                  "type": "Feature",
                  "geometry": mapping(geom),
                  "properties": {
                      "fabric": "merit_reach",
                      "fabric_name": f"MERIT-Hydro Reach Catchment (COMID {eff_river_id})",
                      "model": "GEOGLOWS ECMWF",
                      "comid": eff_river_id,
                      "area_km2": round(area_km2, 1),
                      "resolution": "90m MERIT-Basins",
                  },
              }
              _save_cached_watershed(cache_key, "merit_reach", str(eff_river_id), area_km2, feature)
              return feature
          except Exception as e:
            logger.debug("MERIT shapefile pyogrio query error: %s", e)

    # Fallback to HydroDelineator merit mode
    if HydroDelineator is not None:
      try:
        delin = HydroDelineator("merit-hydro")
        res = delin.delineate_catchment(lat, lon, mode="unit_catchment")
        props = res.get("properties", {})
        area_km2 = props.get("area_km2", 25.0)
        feature = {
            "type": "Feature",
            "geometry": res["geometry"],
            "properties": {
                "fabric": "merit_reach",
                "fabric_name": f"MERIT Reach Catchment ({props.get('catchment_id', 'Reach')})",
                "model": "GEOGLOWS ECMWF",
                "comid": eff_river_id or 0,
                "area_km2": round(area_km2, 1),
                "resolution": "90m MERIT-Basins",
            },
        }
        if eff_river_id:
          _save_cached_watershed(f"merit_reach_{eff_river_id}", "merit_reach", str(eff_river_id), area_km2, feature)
        return feature
      except Exception as e:
        logger.debug("HydroDelineator merit error: %s", e)

  # -------------------------------------------------------------
  # FABRIC 3 & 4: HydroATLAS Full Drainage Area / Level 12 Unit Catchment
  # -------------------------------------------------------------
  cache_key = f"{fabric}_{gauge_id or f'{lat:.4f}_{lon:.4f}'}"
  cached = _get_cached_watershed(cache_key)
  if cached:
    return cached

  if HydroDelineator is not None:
    try:
      delin = HydroDelineator("hydroatlas")
      mode = "unit_catchment" if fabric == "hydroatlas_unit" else "official_ridgeline"
      res = delin.delineate_catchment(lat, lon, mode=mode)
      props = res.get("properties", {})
      r_attrs = props.get("reach_attributes", {})
      hybas_id = r_attrs.get("hydrobasins_unit", gauge_id or "")
      area_km2 = props.get("area_km2", 0.0)

      is_full = (fabric == "hydroatlas_full")
      feat_name = (
          f"HydroATLAS Full Drainage Area ({hybas_id or 'Basin'})"
          if is_full
          else f"HydroATLAS Level 12 Unit Catchment ({hybas_id or 'Unit'})"
      )

      feature = {
          "type": "Feature",
          "geometry": res["geometry"],
          "properties": {
              "fabric": fabric,
              "fabric_name": feat_name,
              "model": "Google FloodHub",
              "hybas_id": hybas_id,
              "area_km2": round(float(area_km2), 1),
              "resolution": "15 arc-sec HydroATLAS",
              "upstream_count": props.get("upstream_reaches_count", 1),
          },
      }

      # If hybas_id was discovered, also index by hybas_id
      _save_cached_watershed(cache_key, fabric, str(hybas_id), area_km2, feature)
      if hybas_id and str(hybas_id) not in cache_key:
        _save_cached_watershed(f"{fabric}_{hybas_id}", fabric, str(hybas_id), area_km2, feature)
      return feature
    except Exception as e:
      logger.debug("HydroDelineator hydroatlas error: %s", e)

  # Fallback approximate polygon if hydrography unavailable
  approx_deg = 0.04
  n_pts = 16
  coords = []
  for i in range(n_pts):
    ang = 2.0 * math.pi * i / n_pts
    r = approx_deg * (0.85 + 0.25 * math.sin(3 * ang))
    coords.append([round(lon + r * math.cos(ang), 5), round(lat + r * math.sin(ang), 5)])
  coords.append(coords[0])

  return {
      "type": "Feature",
      "geometry": {"type": "Polygon", "coordinates": [coords]},
      "properties": {
          "fabric": fabric,
          "fabric_name": "HydroATLAS Watershed (Geometric Estimate)",
          "model": "Google FloodHub",
          "area_km2": 45.0,
          "resolution": "Approximate",
      },
  }
