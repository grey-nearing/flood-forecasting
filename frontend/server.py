"""Zero-dependency HTTP Server for OpenHydroNet using standard library http.server."""

import gzip
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

import sys
import numpy as np

# Ensure google3 is importable when run directly via python3
_ws_root = str(Path(__file__).resolve().parents[1])
if _ws_root not in sys.path:
  sys.path.insert(0, _ws_root)

from frontend.cns_importer import CNSZarrImporter
from frontend.config import (
    ARCHIVES_DIR,
    ATTRIBUTES_DIR,
    FORECAST_DIR,
    HISTORICAL_DIR,
    HYDRO_DATASETS,
    POLYGONS_DIR,
    REALTIME_DIR,
    STATIC_DIR,
    resolve_hydro_dataset_id,
)
from frontend.delineator import HydroDelineator
from frontend.forecast_zarr import ForecastZarrExtractor
from frontend.historical_zarr import HistoricalZarrExtractor
from frontend.jobs import get_job_manager
from frontend.profile_manager import get_profile_manager
from frontend.river_indexer import HydroRiverNetwork
from frontend.static_attributes import ATTRIBUTE_DEFINITIONS, StaticAttributesExtractor
from frontend.weather_sources import get_weather_source, list_weather_sources

try:
  from frontend.weather_engine import (
      generate_raster_tile,
      generate_weather_frame,
      get_catchment_weather_summary,
      get_frame_index,
      get_sync_status,
      get_weather_models_info,
      get_weather_probe,
      get_wind_vectors,
      reload_if_changed as reload_weather_if_changed,
      SUPPORTED_MODELS,
      SUPPORTED_VARIABLES,
  )
except ImportError:
  from weather_engine import (
      generate_raster_tile,
      generate_weather_frame,
      get_catchment_weather_summary,
      get_frame_index,
      get_sync_status,
      get_weather_models_info,
      get_weather_probe,
      get_wind_vectors,
      reload_if_changed as reload_weather_if_changed,
      SUPPORTED_MODELS,
      SUPPORTED_VARIABLES,
  )

# Cached River Network Indexers per dataset
_river_indexers: Dict[str, HydroRiverNetwork] = {
    "merit-hydro": HydroRiverNetwork("merit-hydro"),
    "hydroatlas": HydroRiverNetwork("hydroatlas"),
}

# Cached Delineators per dataset
_delineators: Dict[str, HydroDelineator] = {
    "merit-hydro": HydroDelineator("merit-hydro"),
    "hydroatlas": HydroDelineator("hydroatlas"),
}

# Lazy-loaded Static Attributes Extractor & DEM Tile Server
_attributes_extractor: Optional[StaticAttributesExtractor] = None
_dem_tile_server: Optional[Any] = None


def get_attributes_extractor() -> StaticAttributesExtractor:
  """Returns singleton instance of StaticAttributesExtractor."""
  global _attributes_extractor
  if _attributes_extractor is None:
    _attributes_extractor = StaticAttributesExtractor()
  return _attributes_extractor


def get_dem_tile_server():
  """Returns singleton instance of DemTileServer."""
  global _dem_tile_server
  if _dem_tile_server is None:
    from frontend.dem_tile_server import DemTileServer

    _dem_tile_server = DemTileServer()
  return _dem_tile_server


class EarthkitHydroHandler(BaseHTTPRequestHandler):
  """Custom HTTP Request Handler supporting JSON REST endpoints and static file delivery."""

  def _set_cors_headers(
      self, status: int = 200, content_type: str = "application/json"
  ):
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    self.send_header(
        "Access-Control-Allow-Headers", "Content-Type, Authorization"
    )
    self.end_headers()

  def _send_tile_png(self, tile_bytes: bytes, status: int = 200):
    self.send_response(status)
    self.send_header("Content-Type", "image/png")
    self.send_header("Content-Length", str(len(tile_bytes)))
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Cache-Control", "public, max-age=300")
    self.end_headers()
    self.wfile.write(tile_bytes)

  def _send_json(self, data: Any, status: int = 200):
    try:
      body = json.dumps(data).encode("utf-8")
      accept_enc = self.headers.get("Accept-Encoding", "")

      if "gzip" in accept_enc and len(body) > 1024:
        compressed = gzip.compress(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(compressed)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )
        self.end_headers()
        self.wfile.write(compressed)
      else:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )
        self.end_headers()
        self.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
      # Client closed or navigated away before response finished sending (e.g. rapid map pan/zoom)
      pass

  def _send_error(self, message: str, status: int = 400, code: Optional[str] = None):
    payload = {"error": message, "message": message, "status": "error"}
    if code:
      payload["code"] = code  # Machine-readable reason, e.g. "account_not_found".
    self._send_json(payload, status=status)

  def _get_request_username(
      self, data: Optional[Dict[str, Any]] = None
  ) -> str:
    """Resolves active username from payload, X-User-Profile header, query param, or active profile."""
    pm = get_profile_manager()
    if data and isinstance(data, dict) and data.get("username"):
      u = str(data["username"]).strip().lower()
      if u:
        return u
    header_user = self.headers.get("X-User-Profile")
    if header_user and header_user.strip():
      return header_user.strip().lower()
    parsed = urllib.parse.urlparse(self.path)
    q = urllib.parse.parse_qs(parsed.query)
    if "username" in q and q["username"] and q["username"][0].strip():
      return q["username"][0].strip().lower()
    return pm.active_username

  def do_OPTIONS(self):
    """Handle CORS pre-flight requests."""
    self._set_cors_headers(200, "text/plain")

  def do_GET(self):
    """Handle GET requests."""
    parsed = urllib.parse.urlparse(self.path)
    path = parsed.path
    query = urllib.parse.parse_qs(parsed.query)

    # 1. API: List Datasets
    if path == "/api/datasets":
      self._send_json({"datasets": list(HYDRO_DATASETS.values())})
      return

    # 1b. API: List Weather Sources with Dynamic Date Range Bounds
    if path == "/api/weather/sources":
      self._send_json({"sources": list_weather_sources()})
      return

    # 1c. API: Static Catchment Attributes Schema (HydroATLAS / Caravan)
    if path == "/api/attributes/schema":
      self._send_json({
          "attributes": ATTRIBUTE_DEFINITIONS,
          "extractor_source": "googlehydrology/static_extractor",
      })
      return

    # 1d. API: Raw HydroATLAS Attribute Map Layer (/api/attributes/map-layer)
    if path == "/api/attributes/map-layer":
      attribute = query.get("attribute", ["ele_mt_sav"])[0].strip()
      zoom = int(query.get("zoom", [7])[0])
      level_param = query.get("level", [None])[0]
      level = (
          int(level_param)
          if (level_param and str(level_param).isdigit())
          else None
      )
      bbox_param = query.get("bbox", ["-90.0,35.0,-80.0,45.0"])[0]
      try:
        parts = [float(p.strip()) for p in bbox_param.split(",")]
        bbox = (parts[0], parts[1], parts[2], parts[3])
      except Exception:
        bbox = (-90.0, 35.0, -80.0, 45.0)

      try:
        extractor = get_attributes_extractor()
        layer_geojson = extractor.get_raw_hydroatlas_map_layer(
            bbox=bbox,
            zoom=zoom,
            attribute=attribute,
            level=level,
        )
        self._send_json(layer_geojson)
      except Exception as e:
        self._send_error(
            f"Failed to load raw HydroATLAS map layer: {str(e)}", status=500
        )
      return

    # 1e. API: Get Extracted Attributes for Active User (/api/attributes/extracted)
    if path == "/api/attributes/extracted":
      user = self._get_request_username()
      pm = get_profile_manager()
      attr_dir = pm.get_attributes_dir(user)
      json_cache_path = attr_dir / "extracted_attributes.json"
      results_by_id = {}
      if json_cache_path.exists():
        try:
          cached = json.loads(json_cache_path.read_text(encoding="utf-8"))
          if isinstance(cached, dict):
            results_by_id = cached.get("results_by_id", cached)
        except Exception:
          results_by_id = {}

      results_list = list(results_by_id.values())
      self._send_json({
          "status": "success",
          "username": user,
          "count": len(results_list),
          "results": results_list,
          "results_by_id": results_by_id,
      })
      return

    # 2. API: Get River Vector Features
    if path == "/api/hydro/rivers":
      raw_dataset = query.get("dataset", query.get("dem", ["hydroatlas"]))[0]
      dataset = resolve_hydro_dataset_id(raw_dataset)
      if dataset not in _river_indexers:
        self._send_error(f"Unknown hydrography dataset: {raw_dataset}", status=400)
        return

      zoom = int(query.get("zoom", [7])[0])
      min_order_param = query.get("min_order", [None])[0]
      min_order = int(min_order_param) if min_order_param else None

      bbox_param = query.get("bbox", [None])[0]
      if bbox_param:
        try:
          parts = [float(p.strip()) for p in bbox_param.split(",")]
          min_lon, min_lat, max_lon, max_lat = (
              parts[0],
              parts[1],
              parts[2],
              parts[3],
          )
        except Exception:
          min_lon, min_lat, max_lon, max_lat = -180.0, -90.0, 180.0, 90.0
      else:
        min_lon, min_lat, max_lon, max_lat = -180.0, -90.0, 180.0, 90.0

      network = _river_indexers[dataset]
      geojson = network.get_rivers_in_bbox(
          min_lon=min_lon,
          min_lat=min_lat,
          max_lon=max_lon,
          max_lat=max_lat,
          zoom=zoom,
          min_stream_order=min_order,
          # Optional `lod=fast` (MaaS tab): wide, high-order views come from the in-memory river pyramid.
          prefer_cache=query.get("lod", [""])[0] == "fast",
      )
      self._send_json(geojson)
      return

    # 3. API: List Active / Uploaded Catchments
    if path == "/api/watersheds":
      user = self._get_request_username()
      pm = get_profile_manager()
      features = pm.load_watersheds(username=user)
      self._send_json({
          "type": "FeatureCollection",
          "count": len(features),
          "features": features,
      })
      return

    # API: Job Status Query (/api/jobs/<job_id> or /api/jobs)
    if path.startswith("/api/jobs"):
      parts = [p for p in path.split("/") if p]
      job_mgr = get_job_manager()
      if len(parts) >= 3 and parts[1] == "jobs":
        job_id = parts[2]
        job = job_mgr.get_job(job_id)
        if job is None:
          self._send_error(f"Job '{job_id}' not found", status=404)
          return
        self._send_json(job)
        return
      elif path == "/api/jobs":
        jobs = job_mgr.list_jobs(limit=20)
        self._send_json({"jobs": jobs})
        return

    # API: Current Active Profile (/api/profile/current)
    if path == "/api/profile/current":
      user = self._get_request_username()
      pm = get_profile_manager()
      self._send_json({
          "active_username": user,
          "profile": pm.get_profile_info(user),
          "available_profiles": pm.list_profiles(),
      })
      return

    # API: List All Saved Profiles (/api/profile/list)
    if path == "/api/profile/list":
      pm = get_profile_manager()
      self._send_json({
          "active_username": pm.active_username,
          "profiles": pm.list_profiles(),
      })
      return

    # API: Account Data Inventory Summary (/api/account/summary or /api/profile/account)
    if path in ["/api/account/summary", "/api/profile/account"]:
      user = self._get_request_username()
      pm = get_profile_manager()
      self._send_json(pm.get_account_summary(user))
      return

    # API: Real-Time Weather Forecast Metadata & Latest Issue Time (/api/weather/forecast/info)
    if path in ["/api/weather/forecast/info", "/api/weather/forecast/latest_info"]:
      extractor = ForecastZarrExtractor()
      self._send_json(extractor.get_latest_issue_info())
      return

    # API: Weather Models & Variables Registry (/api/weather/models)
    if path == "/api/weather/models":
      self._send_json({
          "models": get_weather_models_info(),
          "variables": list(SUPPORTED_VARIABLES.values()),
          "sync": get_sync_status(),
      })
      return

    # API: Whole-world animation frames for smooth playback
    # Format: /api/weather/frames/{model}/{variable}/index.json  (which frame each step shows)
    #         /api/weather/frames/{model}/{variable}/{step}.png  (Web Mercator, +-85.05 deg)
    if path.startswith("/api/weather/frames/"):
      parts = path.strip("/").split("/")
      if len(parts) != 6:
        self._send_error("Expected /api/weather/frames/{model}/{variable}/{step}.png", status=400)
        return
      model, variable, leaf = parts[3], parts[4], parts[5]
      try:
        if leaf == "index.json":
          self._send_json(get_frame_index(model, variable))
          return
        step = int(leaf.split(".")[0])
        png = generate_weather_frame(model, variable, step)
      except ValueError as e:
        self._send_error(f"Bad weather frame request: {e}", status=400)
        return
      except Exception as e:  # pylint: disable=broad-except
        self._send_error(f"Failed to render weather frame: {e}", status=500)
        return
      self.send_response(200)
      self.send_header("Content-Type", "image/png")
      self.send_header("Content-Length", str(len(png)))
      self.send_header("Access-Control-Allow-Origin", "*")
      # The viewer puts the run time and frame version in the URL (?v=...).
      self.send_header("Cache-Control", "public, max-age=86400")
      self.end_headers()
      self.wfile.write(png)
      return

    # API: Dynamic Web Mercator Weather Raster Tiles
    # Format: /api/weather/tiles/{model}/{variable}/{step}/{z}/{x}/{y}.png
    if path.startswith("/api/weather/tiles/"):
      try:
        parts = path.strip("/").split("/")
        if len(parts) >= 8:
          model = parts[3]
          variable = parts[4]
          step = int(parts[5])
          z = int(parts[6])
          x = int(parts[7])
          y = int(parts[8].split(".")[0])
          tile_bytes = generate_raster_tile(model, variable, step, z, x, y)
          self._send_tile_png(tile_bytes)
          return
      except Exception as e:
        self._send_error(f"Failed to generate weather tile: {e}", status=400)
        return

    # API: Wind Vector Grid Matrix for Client-Side Streamlines (/api/weather/wind-vectors)
    if path == "/api/weather/wind-vectors":
      try:
        model = query.get("model", ["ecmwf_ifs"])[0]
        step = int(query.get("step", [0])[0])
        subsample = int(query.get("subsample", [2])[0])
        vectors = get_wind_vectors(model, step, subsample)
        self._send_json(vectors)
        return
      except Exception as e:
        self._send_error(f"Failed to fetch wind vectors: {e}", status=400)
        return

    # API: Point Sounding & Comparative Meteogram Probe (/api/weather/probe)
    if path == "/api/weather/probe":
      try:
        lat = float(query.get("lat", [40.0])[0])
        lon = float(query.get("lon", [-86.0])[0])
        probe = get_weather_probe(lat, lon)
        self._send_json(probe)
        return
      except Exception as e:
        self._send_error(f"Failed to probe coordinate: {e}", status=400)
        return

    # API: Active Catchment Zonal Weather Summary (/api/weather/catchment-summary)
    if path == "/api/weather/catchment-summary":
      try:
        step = int(query.get("step", [0])[0])
        watershed_id = query.get("catchment_id", [None])[0]
        user = self._get_request_username()
        pm = get_profile_manager()
        user_watersheds = pm.load_watersheds(user)
        matched_ws = None
        if watershed_id:
          for w in user_watersheds:
            if str(w.get("properties", {}).get("catchment_id") or w.get("id")) == str(watershed_id):
              matched_ws = w
              break
        if not matched_ws and user_watersheds:
          matched_ws = user_watersheds[0]
        if not matched_ws:
          matched_ws = {
              "id": "global_default",
              "properties": {
                  "catchment_id": "global_default",
                  "area_km2": 2500.0,
                  "outlet_latitude": 40.0,
                  "outlet_longitude": -86.0,
              },
              "geometry": {"type": "Point", "coordinates": [-86.0, 40.0]},
          }
        model = query.get("model", ["ecmwf_ifs"])[0]
        summary = get_catchment_weather_summary(matched_ws, step, model)
        self._send_json(summary)
        return
      except Exception as e:
        self._send_error(f"Failed to generate catchment summary: {e}", status=400)
        return

    # API: RainViewer Real-Time Radar Timestamps (/api/weather/radar-times)
    if path == "/api/weather/radar-times":
      try:
        req = urllib.request.Request(
            "https://api.rainviewer.com/public/weather-maps.json",
            headers={"User-Agent": "EarthkitHydro/1.0"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
          data = json.loads(resp.read().decode("utf-8"))
          self._send_json(data)
          return
      except Exception:
        now_ts = int(time.time())
        base_ts = now_ts - (now_ts % 600)
        fallback_data = {
            "radar": {
                "past": [
                    {"time": base_ts - 600 * i, "path": f"/v2/radar/{base_ts - 600 * i}"}
                    for i in range(6, 0, -1)
                ],
                "nowcast": [
                    {"time": base_ts + 600 * i, "path": f"/v2/radar/{base_ts + 600 * i}"}
                    for i in range(1, 4)
                ],
            }
        }
        self._send_json(fallback_data)
        return

    # =========================================================================
    # Models-as-a-Service (MaaS) Endpoints (/api/maas/*)
    # =========================================================================

    # API: MaaS Supported Models & Status (/api/maas/models)
    if path == "/api/maas/models":
      try:
        from frontend.maas_engine import todays_earth_service_status
      except ImportError:
        from maas_engine import todays_earth_service_status

      models_info = {
          "models": [
              {"id": "floodhub", "name": "Google FloodHub", "type": "AI / Physics", "horizon_days": 7, "units": "m³/s", "status": "operational"},
              {"id": "glofas", "name": "Copernicus GloFAS", "type": "30-Day Ensemble (CEMS)", "horizon_days": 15, "units": "m³/s", "status": "operational"},
              {"id": "geoglows", "name": "GEOGLOWS ECMWF", "type": "15-Day 51-Member Ensemble", "horizon_days": 15, "units": "m³/s", "status": "operational"},
              {"id": "todays_earth", "name": "JAXA Today's Earth (CaMa-Flood)", "type": "MATSIRO + CaMa-Flood (streamflow + inundation)", "horizon_days": 15, "units": "m³/s", "status": todays_earth_service_status()},
          ]
      }
      self._send_json(models_info)
      return

    # API: Viewport-Bounded FloodHub Gauges (/api/maas/gauges)
    if path == "/api/maas/gauges":
      try:
        bbox_str = query.get("bbox", [None])[0]
        if bbox_str:
          parts = [float(p.strip()) for p in bbox_str.split(",")]
          min_lat, min_lon, max_lat, max_lon = parts[0], parts[1], parts[2], parts[3]
        else:
          min_lat = float(query.get("min_lat", [32.0])[0])
          min_lon = float(query.get("min_lon", [-118.0])[0])
          max_lat = float(query.get("max_lat", [34.0])[0])
          max_lon = float(query.get("max_lon", [-116.0])[0])

        try:
          from frontend.maas_engine import fetch_floodhub_gauges_bbox
        except ImportError:
          from maas_engine import fetch_floodhub_gauges_bbox

        gauges = fetch_floodhub_gauges_bbox(min_lat, min_lon, max_lat, max_lon)
        self._send_json({"gauges": gauges, "count": len(gauges)})
        return
      except Exception as e:
        self._send_error(f"Failed to query MaaS gauges: {e}", status=400)
        return

    # API: Per-Model River Network (/api/maas/network)
    if path == "/api/maas/network":
      try:
        model = (query.get("model", ["glofas"])[0] or "glofas").strip().lower()
        zoom = int(float(query.get("zoom", [6])[0]))
        bbox_str = query.get("bbox", [None])[0]
        if bbox_str:
          parts = [float(p.strip()) for p in bbox_str.split(",")]
          min_lon, min_lat, max_lon, max_lat = parts[0], parts[1], parts[2], parts[3]
        else:
          min_lon = float(query.get("min_lon", [-180.0])[0])
          min_lat = float(query.get("min_lat", [-60.0])[0])
          max_lon = float(query.get("max_lon", [180.0])[0])
          max_lat = float(query.get("max_lat", [75.0])[0])

        try:
          from frontend.maas_networks import get_model_network
        except ImportError:
          from maas_networks import get_model_network

        self._send_json(get_model_network(model, min_lon, min_lat, max_lon, max_lat, zoom))
        return
      except Exception as e:
        self._send_error(f"Failed to query MaaS river network: {e}", status=400)
        return

    # API: Unified 4-Provider Streamflow & Flood Forecast Probe (/api/maas/forecast)
    if path == "/api/maas/forecast":
      try:
        lat = float(query.get("lat", [32.756])[0])
        lon = float(query.get("lon", [-117.252])[0])
        gauge_id = query.get("gauge_id", [None])[0] or None
        river_id_str = query.get("river_id", [None])[0]
        river_id = int(river_id_str) if river_id_str and river_id_str.isdigit() else None
        reach_id = query.get("reach_id", [None])[0] or None
        if not reach_id and river_id_str and river_id_str.upper().startswith("HYRIV_"):
          reach_id = river_id_str
        upstream_area_km2 = query.get("upstream_area_km2", [None])[0]
        area_min_km2 = query.get("area_min_km2", [None])[0]
        network = query.get("network", [None])[0] or None
        models_str = query.get("models", ["floodhub,glofas,geoglows,todays_earth"])[0]
        requested_models = [m.strip().lower() for m in models_str.split(",") if m.strip()]

        try:
          from frontend.maas_engine import get_unified_maas_forecast
        except ImportError:
          from maas_engine import get_unified_maas_forecast

        data = get_unified_maas_forecast(
            lat, lon, gauge_id=gauge_id, river_id=river_id, requested_models=requested_models, reach_id=reach_id,
            upstream_area_km2=upstream_area_km2, area_min_km2=area_min_km2, network=network,
        )
        self._send_json(data)
        return
      except Exception as e:
        self._send_error(f"Failed to aggregate MaaS forecast: {e}", status=400)
        return

    # API: Spatial Flood Inundation Layers (/api/maas/flood-inundation)
    if path == "/api/maas/flood-inundation":
      try:
        lat = float(query.get("lat", [32.756])[0])
        lon = float(query.get("lon", [-117.252])[0])
        gauge_id = query.get("gauge_id", [None])[0] or None
        reach_id = query.get("reach_id", [None])[0] or None
        river_id_str = query.get("river_id", [None])[0]
        river_id = int(river_id_str) if river_id_str and river_id_str.isdigit() else None
        if not reach_id and river_id_str and river_id_str.upper().startswith("HYRIV_"):
          reach_id = river_id_str

        try:
          from frontend.maas_engine import get_maas_flood_inundation
        except ImportError:
          from maas_engine import get_maas_flood_inundation

        data = get_maas_flood_inundation(lat, lon, gauge_id=gauge_id, reach_id=reach_id, river_id=river_id)
        self._send_json(data)
        return
      except Exception as e:
        self._send_error(f"Failed to query MaaS flood inundation: {e}", status=400)
        return

    # API: Multi-Model Watershed Polygon Probe (/api/maas/watershed)
    if path == "/api/maas/watershed":
      try:
        lat = float(query.get("lat", [32.756])[0])
        lon = float(query.get("lon", [-117.252])[0])
        fabric = query.get("geofabric", query.get("fabric", ["hydroatlas_full"]))[0]
        gauge_id = query.get("gauge_id", [None])[0]
        river_id_str = query.get("river_id", [None])[0]
        river_id = int(river_id_str) if river_id_str and river_id_str.isdigit() else None

        try:
          from frontend.maas_engine import get_maas_watershed_polygon
        except ImportError:
          from maas_engine import get_maas_watershed_polygon

        data = get_maas_watershed_polygon(lat, lon, fabric=fabric, gauge_id=gauge_id, river_id=river_id)
        self._send_json(data)
        return
      except Exception as e:
        self._send_error(f"Failed to query MaaS watershed polygon: {e}", status=400)
        return

    # API: Real-Time Forecast & Model Status (/api/forecast/status)
    if path == "/api/forecast/status":
      user = self._get_request_username()
      pm = get_profile_manager()
      catchment_id = query.get("catchment_id", [None])[0] or query.get("basin_id", [None])[0]
      lookback_days = int(query.get("nowcast_lookback_days", [14])[0])
      query_upstream = query.get("query_upstream", ["1"])[0] not in ("0", "false", "False")

      if not catchment_id:
        ws = pm.load_watersheds(user)
        if ws:
          catchment_id = str(
              ws[0].get("properties", {}).get("catchment_id")
              or ws[0].get("id")
              or ""
          )
      if not catchment_id:
        self._send_json({
            "status": "no_catchment",
            "username": user,
            "basin_id": None,
            "issue_date": None,
            "expected_state_date": None,
            "saved_state": {"exists": False, "state_date": None},
            "is_state_from_previous_day": False,
            "can_run_coldstart": False,
            "can_run_hotstart": False,
            "coldstart_status_message": "Select or delineate a catchment first.",
            "hotstart_status_message": "Select or delineate a catchment first.",
            "nowcast_products": {},
            "forecast_products": {},
        })
        return

      try:
        try:
          from frontend.realtime_forecast_service import inspect_realtime_stores_for_basin
        except ImportError:
          from realtime_forecast_service import inspect_realtime_stores_for_basin

        payload = inspect_realtime_stores_for_basin(
            username=user,
            basin_id=str(catchment_id),
            nowcast_lookback_days=lookback_days,
            query_upstream_if_empty=query_upstream,
        )
        self._send_json(payload)
        return
      except Exception as e:
        self._send_error(f"Failed to inspect forecast status: {e}", status=500)
        return

    # 4. API: List Stored Data Archives across Folders
    if path == "/api/archives":
      user = self._get_request_username()
      pm = get_profile_manager()

      # Every account, including the guest session, only sees its own folders.
      archives = []
      seen_paths = set()

      # 1. Historical Zarr Stores
      hist_dirs = [
          pm.get_historical_dir(user),
          pm.get_dynamics_dir(user),
          pm.get_targets_dir(user),
          pm.get_profile_dir(user) / "archives",
      ]

      for base_dir in hist_dirs:
        if not base_dir.exists():
          continue

        for item in base_dir.rglob("*.zarr"):
          if "forecast" in item.name or str(item) in seen_paths:
            continue
          seen_paths.add(str(item))
          size_bytes = sum(
              f.stat().st_size for f in item.rglob("*") if f.is_file()
          )
          try:
            import xarray as xr

            h_ds = xr.open_zarr(str(item), decode_timedelta=False)
            basin_list = (
                [str(b) for b in h_ds.basin.values]
                if "basin" in h_ds.coords or "basin" in h_ds.dims
                else []
            )
            time_coord = (
                h_ds.coords.get("date")
                if "date" in h_ds.coords
                else (
                    h_ds.coords.get("time") if "time" in h_ds.coords else None
                )
            )
            n_times = len(time_coord.values) if time_coord is not None else 0
          except Exception:
            basin_list = []
            n_times = 0

          is_master = item.name == "historical_training_master.zarr" or len(basin_list) > 1
          archives.append({
              "catchment_id": (
                  f"HISTORICAL_{len(basin_list)}_BASINS"
                  if is_master
                  else (basin_list[0] if basin_list else item.parent.name if item.parent != base_dir else item.stem)
              ),
              "name": item.name,
              "filename": item.name,
              "type": "master_historical" if is_master else "historical",
              "category": "Historical Data",
              "size_bytes": size_bytes,
              "size_mb": round(size_bytes / (1024 * 1024), 3),
              "total_basins": len(basin_list),
              "basins": basin_list,
              "n_timesteps": n_times,
              "path": str(item),
          })

      # 2. Forecast Weather Stores
      fc_dirs = [pm.get_forecast_dir(user)]
      if (pm.get_profile_dir(user) / "realtime").exists():
        fc_dirs.append(pm.get_profile_dir(user) / "realtime")
      for base_dir in fc_dirs:
        if not base_dir.exists():
          continue
        for item in base_dir.rglob("*.zarr"):
          if str(item) not in seen_paths:
            seen_paths.add(str(item))
            size_bytes = sum(
                f.stat().st_size for f in item.rglob("*") if f.is_file()
            )
            try:
              import xarray as xr

              mfc_ds = xr.open_zarr(str(item), decode_timedelta=False)
              mfc_basins = (
                  [str(b) for b in mfc_ds.basin.values]
                  if "basin" in mfc_ds.coords or "basin" in mfc_ds.dims
                  else []
              )
              mfc_lead = (
                  len(mfc_ds.lead_time.values)
                  if "lead_time" in mfc_ds.coords
                  else 0
              )
            except Exception:
              mfc_basins = []
              mfc_lead = 0
            archives.append({
                "catchment_id": (
                    f"FORECAST_{len(mfc_basins)}_BASINS"
                    if len(mfc_basins) > 1
                    else (mfc_basins[0] if mfc_basins else item.stem)
                ),
                "name": item.name,
                "filename": item.name,
                "type": "forecast",
                "category": "Forecast Weather",
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "total_basins": len(mfc_basins),
                "basins": mfc_basins,
                "lead_time_days": mfc_lead,
                "path": str(item),
            })

      # 3. Static Attributes Files
      attr_dirs = [pm.get_attributes_dir(user), pm.get_statics_dir(user)]
      for base_dir in attr_dirs:
        if not base_dir.exists():
          continue
        for ext in ("*.csv", "*.json", "*.parquet", "*.nc"):
          for attr_file in base_dir.rglob(ext):
            if str(attr_file) not in seen_paths and attr_file.is_file():
              seen_paths.add(str(attr_file))
              size_bytes = attr_file.stat().st_size
              archives.append({
                  "catchment_id": attr_file.stem,
                  "name": attr_file.name,
                  "filename": attr_file.name,
                  "type": "attributes",
                  "category": "Static Attributes",
                  "size_bytes": size_bytes,
                  "size_mb": round(size_bytes / (1024 * 1024), 3),
                  "download_url": (
                      f"/api/attributes/csv?username={user}"
                      if attr_file.suffix == ".csv"
                      else None
                  ),
                  "path": str(attr_file),
              })

      # 4. Polygon Boundaries Files
      poly_dirs = [pm.get_polygons_dir(user)]
      has_watersheds = bool(pm.load_watersheds(user))
      for base_dir in poly_dirs:
        if not base_dir.exists():
          continue
        for ext in ("*.geojson", "*.json", "*.shp", "*.gpkg"):
          for poly_file in base_dir.glob(ext):
            if poly_file.name == "watersheds.json" and not has_watersheds:
              continue  # Empty basin registry, not saved data.
            if str(poly_file) not in seen_paths and poly_file.is_file():
              seen_paths.add(str(poly_file))
              size_bytes = poly_file.stat().st_size
              archives.append({
                  "catchment_id": poly_file.stem,
                  "name": poly_file.name,
                  "filename": poly_file.name,
                  "type": "polygons",
                  "category": "Polygons & Boundaries",
                  "size_bytes": size_bytes,
                  "size_mb": round(size_bytes / (1024 * 1024), 3),
                  "path": str(poly_file),
              })

      self._send_json({"archives": archives})
      return

    # API: Download Caravan Attributes CSV (/api/attributes/csv)
    if path in [
        "/api/attributes/csv",
        "/api/archives/caravan_hydroatlas_attributes.csv",
    ]:
      user = self._get_request_username()
      pm = get_profile_manager()
      # Only the requesting account's own CSV (never another user's or legacy shared data).
      csv_candidates = [
          pm.get_attributes_dir(user) / "caravan_hydroatlas_attributes.csv",
          pm.get_profile_dir(user) / "archives" / "caravan_hydroatlas_attributes.csv",
      ]
      csv_path = next((p for p in csv_candidates if p.exists()), None)
      if csv_path:
        content = csv_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header(
            "Content-Disposition", 'attachment; filename="attributes.csv"'
        )
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)
        return
      else:
        self._send_error(
            "Caravan attributes CSV has not been extracted yet.", status=404
        )
        return

    # 5. API: DEM Slippy Map XYZ Raster Tiles (/api/tiles/dem/{z}/{x}/{y}.png)
    if path.startswith("/api/tiles/dem/"):
      try:
        parts = path[len("/api/tiles/dem/") :].split("/")
        if len(parts) == 3:
          z = int(parts[0])
          x = int(parts[1])
          y_str = parts[2]
          if y_str.endswith(".png"):
            y = int(y_str[:-4])
            tile_server = get_dem_tile_server()
            tile_bytes = tile_server.render_tile(z, x, y)
            self._send_tile_png(tile_bytes)
            return
      except Exception as e:
        print(f"DEM tile render error for {path}: {e}")
      self._send_error("Invalid tile coordinates", status=400)
      return

    # 4. Frontend Single Page UI
    if path == "/" or path == "/index.html":
      index_path = STATIC_DIR / "index.html"
      if index_path.exists():
        content = index_path.read_bytes()
        self._set_cors_headers(200, "text/html; charset=utf-8")
        self.wfile.write(content)
        return

    # 5. Static Assets
    if path.startswith("/static/"):
      rel_path = path[len("/static/") :]
      file_path = (STATIC_DIR / rel_path).resolve()
      if (
          file_path.exists()
          and file_path.is_file()
          and str(file_path).startswith(str(STATIC_DIR))
      ):
        mime, _ = mimetypes.guess_type(str(file_path))
        self._set_cors_headers(200, mime or "application/octet-stream")
        self.wfile.write(file_path.read_bytes())
        return

    self._send_error("Not Found", status=404)

  def do_POST(self):
    """Handle POST requests."""
    parsed = urllib.parse.urlparse(self.path)
    path = parsed.path

    content_length = int(self.headers.get("Content-Length", 0))
    if content_length > 0:
      try:
        body_bytes = self.rfile.read(content_length)
        data = json.loads(body_bytes.decode("utf-8"))
      except Exception:
        self._send_error("Invalid JSON body", status=400)
        return
    else:
      data = {}

    # 1. API: Delineate Catchment from Point Click
    if path == "/api/delineate":
      raw_dataset = (
          data.get("dataset")
          or data.get("dem")
          or data.get("dem_id")
          or "hydroatlas"
      )
      dataset = resolve_hydro_dataset_id(raw_dataset)
      lat = data.get("latitude")
      lon = data.get("longitude")
      snap_radius = data.get("snap_radius_km")
      mode = data.get("mode", "dem_flow_direction")

      if lat is None or lon is None:
        self._send_error("Missing latitude or longitude", status=400)
        return

      if dataset not in _delineators:
        self._send_error(
            f"Unknown hydrography dataset: {raw_dataset}", status=400
        )
        return

      delineator = _delineators[dataset]
      user = self._get_request_username(data)
      pm = get_profile_manager()
      try:
        geojson = delineator.delineate_catchment(
            lat=float(lat),
            lon=float(lon),
            snap_radius_km=snap_radius,
            mode=mode,
        )
        catchment_id = geojson["properties"]["catchment_id"]
        geojson["properties"]["source"] = "delineated"

        existing_feats = pm.load_watersheds(user)
        updated = False
        for i, ef in enumerate(existing_feats):
          if str(ef.get("properties", {}).get("catchment_id") or ef.get("id")) == str(catchment_id):
            existing_feats[i] = geojson
            updated = True
            break
        if not updated:
          existing_feats.append(geojson)
        pm.save_watersheds(existing_feats, username=user)

        self._send_json(geojson)
        return
      except Exception as e:
        self._send_error(str(e), status=400)
        return

    # API: Delete Single Watershed from Active Registry
    if path == "/api/watersheds/delete":
      user = self._get_request_username(data)
      pm = get_profile_manager()
      catchment_id = data.get("catchment_id")
      if not catchment_id:
        self._send_error("Missing catchment_id to delete", status=400)
        return
      existing_feats = pm.load_watersheds(user)
      new_feats = [
          f for f in existing_feats
          if str(f.get("properties", {}).get("catchment_id") or f.get("id")) != str(catchment_id)
      ]
      pm.save_watersheds(new_feats, username=user)
      try:
        json_cache_path = pm.get_attributes_dir(user) / "extracted_attributes.json"
        if json_cache_path.exists():
          cached = json.loads(json_cache_path.read_text(encoding="utf-8"))
          r_map = dict(cached.get("results_by_id", cached)) if isinstance(cached, dict) else {}
          if str(catchment_id) in r_map:
            del r_map[str(catchment_id)]
            json_cache_path.write_text(
                json.dumps({"results_by_id": r_map}, indent=2), encoding="utf-8"
            )
      except Exception:
        pass
      self._send_json({
          "status": "deleted" if len(new_feats) < len(existing_feats) else "not_found",
          "catchment_id": str(catchment_id),
          "remaining_count": len(new_feats),
      })
      return

    # API: Clear All Watersheds from Active Registry
    if path == "/api/watersheds/clear":
      user = self._get_request_username(data)
      pm = get_profile_manager()
      pm.save_watersheds([], username=user)
      try:
        json_cache_path = pm.get_attributes_dir(user) / "extracted_attributes.json"
        if json_cache_path.exists():
          json_cache_path.unlink(missing_ok=True)
      except Exception:
        pass
      self._send_json({"status": "cleared", "remaining_count": 0})
      return

    # API: Log In / Switch User Profile (/api/profile/login)
    # Pass "create_if_missing": false to only allow existing accounts (404 if
    # the account does not exist). The web UI does this and creates accounts
    # explicitly through /api/profile/create. The default (true) is kept for
    # backward compatibility with existing callers.
    if path == "/api/profile/login":
      username = data.get("username", "guest")
      email = data.get("email")
      display_name = data.get("display_name")
      settings = data.get("settings")
      pm = get_profile_manager()

      try:
        prof = pm.login_profile(
            username=username,
            email=email,
            display_name=display_name,
            settings=settings,
            create_if_missing=bool(data.get("create_if_missing", True)),
        )
      except LookupError as e:
        self._send_error(str(e), status=404)
        return

      loaded_feats = pm.load_watersheds(prof.get("username"))

      self._send_json({
          "status": "success",
          "message": f"Logged in as '{prof.get('username')}'",
          "profile": prof,
          "watersheds": loaded_feats,
          "count": len(loaded_feats),
      })
      return

    # API: Create a New User Account (/api/profile/create)
    if path == "/api/profile/create":
      pm = get_profile_manager()
      try:
        prof = pm.create_account(
            username=str(data.get("username") or ""),
            email=data.get("email"),
            display_name=data.get("display_name"),
        )
      except ValueError as e:
        self._send_error(str(e), status=400)
        return
      except FileExistsError as e:
        self._send_error(str(e), status=409)
        return

      self._send_json({
          "status": "success",
          "message": f"Created account '{prof.get('username')}'",
          "profile": prof,
          "watersheds": [],
          "count": 0,
      })
      return

    # API: Remove a User Account (/api/profile/delete)
    # Permanently deletes the account and everything saved in it. The guest
    # session can't be removed. If the removed account was active, the guest
    # session becomes active. "profile" in the reply is the active profile
    # afterwards. A missing account is a 404 with code "account_not_found", so
    # clients can tell it apart from a server that lacks this endpoint.
    if path == "/api/profile/delete":
      pm = get_profile_manager()
      try:
        removed = pm.delete_account(str(data.get("username") or ""))
      except ValueError as e:
        self._send_error(str(e), status=400, code="invalid_account")
        return
      except LookupError as e:
        self._send_error(str(e), status=404, code="account_not_found")
        return
      except OSError as e:
        self._send_error(f"Couldn't remove the account: {e}", status=500)
        return

      prof = pm.get_profile_info(pm.active_username)
      loaded_feats = pm.load_watersheds(pm.active_username)
      self._send_json({
          "status": "success",
          "message": f"Removed account '{removed}'",
          "removed": removed,
          "profile": prof,
          "watersheds": loaded_feats,
          "count": len(loaded_feats),
      })
      return

    # API: Log Out / Switch to Guest (/api/profile/logout)
    if path == "/api/profile/logout":
      pm = get_profile_manager()
      prof = pm.logout_profile()
      loaded_feats = pm.load_watersheds("guest")

      self._send_json({
          "status": "success",
          "message": "Switched to Guest profile",
          "profile": prof,
          "watersheds": loaded_feats,
          "count": len(loaded_feats),
      })
      return

    # API: Save Profile Settings (/api/profile/settings)
    if path == "/api/profile/settings":
      settings = data.get("settings", {})
      pm = get_profile_manager()
      prof = pm.save_settings(settings)
      self._send_json({"status": "saved", "profile": prof})
      return

    # API: Explicitly Persist Watersheds (/api/profile/save_watersheds)
    if path == "/api/profile/save_watersheds":
      user = self._get_request_username(data)
      pm = get_profile_manager()
      feats = pm.load_watersheds(user)
      self._send_json({
          "status": "saved",
          "count": len(feats),
          "username": user,
      })
      return

    # 2. API: Upload Arbitrary Custom Polygon (GeoJSON - Single or Multi-Watershed)
    if path == "/api/polygons/upload":
      polygon_data = data.get("polygon") or data.get("geojson") or data
      if not polygon_data:
        self._send_error("No polygon or GeoJSON payload provided", status=400)
        return

      try:
        from shapely.geometry import shape, mapping
        from shapely.ops import unary_union

        raw_features = []
        if (
            isinstance(polygon_data, dict)
            and polygon_data.get("type") == "FeatureCollection"
        ):
          raw_features = polygon_data.get("features", [])
        elif (
            isinstance(polygon_data, dict)
            and polygon_data.get("type") == "Feature"
        ):
          raw_features = [polygon_data]
        elif isinstance(polygon_data, list):
          raw_features = polygon_data
        else:
          raw_features = [{
              "type": "Feature",
              "geometry": polygon_data.get("geometry", polygon_data),
              "properties": polygon_data.get("properties", {}),
          }]

        if not raw_features:
          self._send_error("No features found in GeoJSON", status=400)
          return

        normalized_features = []
        summaries = []
        total_area = 0.0

        for idx, feat in enumerate(raw_features):
          geom_dict = feat.get("geometry", feat)
          if not geom_dict:
            continue
          geom = shape(geom_dict)
          if geom.is_empty:
            continue
          if geom.geom_type not in ("Polygon", "MultiPolygon"):
            if geom.geom_type == "GeometryCollection":
              polys = [
                  g
                  for g in geom.geoms
                  if g.geom_type in ("Polygon", "MultiPolygon")
              ]
              if not polys:
                continue
              geom = unary_union(polys)
            else:
              continue

          cent = geom.centroid
          props = dict(feat.get("properties", {}))
          cid = (
              props.get("catchment_id")
              or props.get("gauge_id")
              or props.get("station_id")
              or props.get("basin_id")
              or props.get("id")
              or props.get("site_no")
              or props.get("huc8")
              or props.get("HUC8")
              or props.get("huc10")
              or props.get("huc12")
              or f"user_basin_{idx + 1}"
          )
          name = (
              props.get("name")
              or props.get("station_name")
              or props.get("basin_name")
              or props.get("SITE_NAME")
              or f"Watershed {idx + 1} ({cid})"
          )

          lat_scale = 111.0
          lon_scale = 111.0 * np.cos(np.radians(cent.y))
          calc_area = float(geom.area * lat_scale * lon_scale)
          area_km2 = round(
              float(props.get("area_km2") or props.get("area") or calc_area), 2
          )
          total_area += area_km2

          norm_f = {
              "type": "Feature",
              "geometry": mapping(geom),
              "properties": {
                  **props,
                  "catchment_id": str(cid),
                  "name": str(name),
                  "area_km2": area_km2,
                  "dataset_name": "Custom User-Uploaded Watershed",
                  "delineation_method": "User GeoJSON Polygon",
                  "upstream_reaches_count": 1,
                  "stream_order": props.get("stream_order", 3),
                  "outlet": {
                      "latitude": round(float(cent.y), 4),
                      "longitude": round(float(cent.x), 4),
                      "snap_distance_m": 0,
                  },
                  "bbox": [round(float(b), 5) for b in geom.bounds],
              },
          }
          normalized_features.append(norm_f)
          summaries.append({
              "catchment_id": str(cid),
              "name": str(name),
              "area_km2": area_km2,
              "latitude": round(float(cent.y), 4),
              "longitude": round(float(cent.x), 4),
              "bbox": [round(float(b), 5) for b in geom.bounds],
          })

        user = self._get_request_username(data)
        pm = get_profile_manager()
        existing_feats = pm.load_watersheds(user)
        existing_ids = {
            str(f.get("properties", {}).get("catchment_id") or f.get("id"))
            for f in existing_feats
        }
        for nf in normalized_features:
          cid = str(nf.get("properties", {}).get("catchment_id") or nf.get("id"))
          if cid in existing_ids:
            for i, ef in enumerate(existing_feats):
              if str(ef.get("properties", {}).get("catchment_id") or ef.get("id")) == cid:
                existing_feats[i] = nf
                break
          else:
            existing_feats.append(nf)
            existing_ids.add(cid)
        pm.save_watersheds(existing_feats, username=user)

        if not normalized_features:
          self._send_error(
              "No valid Polygon / MultiPolygon features could be parsed.",
              status=400,
          )
          return

        all_minx = min(f["properties"]["bbox"][0] for f in normalized_features)
        all_miny = min(f["properties"]["bbox"][1] for f in normalized_features)
        all_maxx = max(f["properties"]["bbox"][2] for f in normalized_features)
        all_maxy = max(f["properties"]["bbox"][3] for f in normalized_features)

        response_payload = {
            "type": "FeatureCollection",
            "count": len(normalized_features),
            "total_area_km2": round(total_area, 2),
            "bbox": [all_minx, all_miny, all_maxx, all_maxy],
            "summary": summaries,
            "features": normalized_features,
        }
        self._send_json(response_payload)
        return
      except Exception as e:
        self._send_error(
            f"Failed to parse custom GeoJSON: {str(e)}", status=400
        )
        return

    # 3. API: Generate Historical Zarr Archive (Single or Multi-Basin Batch)
    if path == "/api/weather/historical":
      user = self._get_request_username(data)
      catchment_id = data.get("catchment_id")
      catchment_ids = data.get("catchment_ids") or data.get("basin_ids")
      extract_all = data.get("all_active", False)
      start_date = data.get("start_date")
      end_date = data.get("end_date")
      weather_source = (
          data.get("weather_source")
          or data.get("source")
          or data.get("dataset")
          or "era5"
      )
      frequency = data.get("frequency", "1D")
      variables = data.get("variables")
      notify_email = (data.get("notify_email") or "").strip()
      run_async = bool(data.get("async") or "notify_email" in data)

      extractor = HistoricalZarrExtractor(
          output_dir=get_profile_manager().get_historical_dir(user)
      )
      job_mgr = get_job_manager()

      host = self.headers.get("Host", "localhost:8080")
      server_url = f"http://{host}"

      # Check if multiple basins requested
      pm = get_profile_manager()
      user_watersheds = pm.load_watersheds(user)
      ws_by_id = {
          str(f.get("properties", {}).get("catchment_id") or f.get("id")): f
          for f in user_watersheds
      }

      if (
          extract_all
          or catchment_ids
          or (not catchment_id and data.get("features"))
      ):
        features_to_extract = []
        if data.get("features"):
          features_to_extract = data["features"]
        elif catchment_ids:
          for cid in catchment_ids:
            if str(cid) in ws_by_id:
              features_to_extract.append(ws_by_id[str(cid)])
        elif extract_all:
          features_to_extract = user_watersheds

        if not features_to_extract:
          self._send_error("No basins found to extract in batch", status=400)
          return

        if run_async:
          job_id = job_mgr.submit_job(
              job_type="historical_weather_batch",
              task_fn=extractor.extract_and_archive_batch,
              features=features_to_extract,
              start_date=start_date,
              end_date=end_date,
              weather_source=weather_source,
              freq=frequency,
              variables=variables,
              notify_email=notify_email,
              server_url=server_url,
              metadata={
                  "weather_source": weather_source,
                  "basin_count": len(features_to_extract),
                  "start_date": start_date,
                  "end_date": end_date,
              },
          )

          self._send_json({
              "status": "queued",
              "job_id": job_id,
              "message": (
                  f"Historical extraction job submitted for {len(features_to_extract)} basins."
                  + (f" Notification email will be sent to {notify_email}." if notify_email else "")
              ),
              "batch": True,
              "basin_count": len(features_to_extract),
              "notify_email": notify_email,
              "weather_source": weather_source,
          })
          return

        try:
          res = extractor.extract_and_archive_batch(
              features=features_to_extract,
              start_date=start_date,
              end_date=end_date,
              weather_source=weather_source,
              freq=frequency,
              variables=variables,
          )
          self._send_json(res)
          return
        except Exception as e:
          self._send_error(
              f"Batch historical extraction failed: {str(e)}", status=500
          )
          return

      # Single Basin Extraction
      polygon_input = data.get("polygon") or data.get("geometry")
      if not polygon_input and str(catchment_id) in ws_by_id:
        polygon_input = ws_by_id[str(catchment_id)]

      if not polygon_input and not catchment_id:
        self._send_error(
            "catchment_id, catchment_ids, or polygon geometry is required",
            status=400,
        )
        return

      if not polygon_input:
        self._send_error(
            f"Catchment '{catchment_id}' not found. Please delineate a basin or"
            " supply polygon first.",
            status=404,
        )
        return

      if run_async:
        job_id = job_mgr.submit_job(
            job_type="historical_weather_single",
            task_fn=extractor.extract_and_archive,
            polygon_input=polygon_input,
            basin_id=catchment_id,
            start_date=start_date,
            end_date=end_date,
            weather_source=weather_source,
            freq=frequency,
            variables=variables,
            notify_email=notify_email,
            server_url=server_url,
            metadata={
                "weather_source": weather_source,
                "basin_id": catchment_id,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        self._send_json({
            "status": "queued",
            "job_id": job_id,
            "message": (
                f"Historical extraction job submitted for basin {catchment_id}."
                + (f" Notification email will be sent to {notify_email}." if notify_email else "")
            ),
            "batch": False,
            "basin_id": catchment_id,
            "notify_email": notify_email,
            "weather_source": weather_source,
        })
        return

      try:
        res = extractor.extract_and_archive(
            polygon_input=polygon_input,
            basin_id=catchment_id,
            start_date=start_date,
            end_date=end_date,
            weather_source=weather_source,
            freq=frequency,
            variables=variables,
        )
        self._send_json(res)
        return
      except Exception as e:
        self._send_error(
            f"Historical extraction failed: {str(e)}", status=500
        )
        return

    # 4. API: Fetch Real-Time Forecast Zarr Store (Single or Batch for all user polygons)
    if path == "/api/weather/forecast":
      user = self._get_request_username(data)
      pm = get_profile_manager()
      job_mgr = get_job_manager()

      catchment_id = data.get("catchment_id")
      catchment_ids = data.get("catchment_ids") or []
      extract_all = bool(data.get("all_active") or data.get("extract_all") or data.get("batch"))
      features_param = data.get("features") or []
      notify_email = (data.get("notify_email") or "").strip()
      run_async = bool(data.get("async") or notify_email)

      raw_horizon = data.get("horizon_days") or data.get("horizon_hours") or 10
      try:
        raw_val = int(raw_horizon)
        horizon_days = raw_val // 24 if raw_val > 15 else raw_val
      except Exception:
        horizon_days = 10
      horizon_days = max(1, min(horizon_days, 15))

      output_dir = pm.get_forecast_dir(user)
      extractor = ForecastZarrExtractor(output_dir=output_dir)

      # Determine if batch extraction is requested
      features_to_extract = []
      if features_param:
        features_to_extract = features_param
      elif extract_all or catchment_ids:
        user_watersheds = pm.load_watersheds(username=user)
        ws_by_id = {
            f["properties"]["catchment_id"]: f
            for f in user_watersheds
            if "catchment_id" in f.get("properties", {})
        }

        if catchment_ids:
          for cid in catchment_ids:
            if cid in ws_by_id:
              features_to_extract.append(ws_by_id[cid])
        elif extract_all:
          features_to_extract = list(ws_by_id.values())

      if extract_all or len(features_to_extract) > 1:
        if not features_to_extract:
          self._send_error(
              "No watershed polygons found in user profile to extract forecast"
              " for.",
              status=400,
          )
          return

        if run_async:
          server_url = f"http://{self.headers.get('Host', 'localhost:8080')}"
          job_id = job_mgr.submit_job(
              job_type="realtime_forecast_batch",
              task_fn=extractor.fetch_and_archive_batch,
              features=features_to_extract,
              horizon_days=horizon_days,
              notify_email=notify_email,
              server_url=server_url,
              metadata={
                  "source": "dynamical.org",
                  "models": ["ECMWF IFS HRES", "GraphCast (ECMWF AIFS Proxy)"],
                  "basin_count": len(features_to_extract),
                  "horizon_days": horizon_days,
                  "username": user,
              },
          )
          self._send_json({
              "status": "queued",
              "job_id": job_id,
              "message": (
                  f"Real-time forecast extraction job queued for {len(features_to_extract)} basins."
                  + (f" Notification email will be sent to {notify_email}." if notify_email else "")
              ),
              "batch": True,
              "basin_count": len(features_to_extract),
              "notify_email": notify_email,
          })
          return
        else:
          try:
            res = extractor.fetch_and_archive_batch(
                features=features_to_extract,
                horizon_days=horizon_days,
            )
            self._send_json(res)
            return
          except Exception as e:
            self._send_error(
                f"Batch forecast extraction failed: {str(e)}", status=500
            )
            return

      # Single Catchment Extraction
      polygon_input = data.get("polygon") or data.get("geometry")
      if not polygon_input and features_to_extract:
        polygon_input = features_to_extract[0]
      elif not polygon_input and catchment_id:
        user_watersheds = pm.load_watersheds(username=user)
        ws_by_id = {
            f["properties"]["catchment_id"]: f
            for f in user_watersheds
            if "catchment_id" in f.get("properties", {})
        }
        if catchment_id in ws_by_id:
          polygon_input = ws_by_id[catchment_id]

      if not polygon_input and not catchment_id:
        self._send_error(
            "catchment_id, polygon, or features list is required", status=400
        )
        return

      if not polygon_input:
        self._send_error(
            f"Catchment '{catchment_id}' not found in active profile.",
            status=404,
        )
        return

      if run_async:
        server_url = f"http://{self.headers.get('Host', 'localhost:8080')}"
        job_id = job_mgr.submit_job(
            job_type="realtime_forecast_single",
            task_fn=extractor.fetch_and_archive,
            catchment_feature=polygon_input,
            basin_id=catchment_id,
            horizon_days=horizon_days,
            notify_email=notify_email,
            server_url=server_url,
            metadata={
                "source": "dynamical.org",
                "catchment_id": catchment_id,
                "horizon_days": horizon_days,
                "username": user,
            },
        )
        self._send_json({
            "status": "queued",
            "job_id": job_id,
            "message": (
                f"Real-time forecast extraction queued for basin {catchment_id}."
                + (f" Notification email will be sent to {notify_email}." if notify_email else "")
            ),
            "batch": False,
            "catchment_id": catchment_id,
            "notify_email": notify_email,
        })
        return

      try:
        result = extractor.fetch_and_archive(
            catchment_feature=polygon_input,
            basin_id=catchment_id,
            horizon_days=horizon_days,
        )
        self._send_json(result)
        return
      except Exception as e:
        self._send_error(f"Failed to fetch forecast: {str(e)}", status=500)
        return

    # API: Direct CNS Zarr Import (Dev Bypass)
    if path in ["/api/archives/import_cns", "/api/weather/historical/import_cns"]:
      user = self._get_request_username(data)
      cns_path = (data.get("cns_path") or data.get("path") or "").strip()
      dest_filename = (
          data.get("dest_filename")
          or data.get("filename")
          or "historical_training_master.zarr"
      )
      notify_email = (data.get("notify_email") or "").strip()

      if not cns_path:
        self._send_error(
            "Missing 'cns_path' parameter (e.g. /cns/...).", status=400
        )
        return

      if not cns_path.startswith("/cns/"):
        self._send_error(
            f"Invalid CNS path '{cns_path}'. Path must begin with /cns/.",
            status=400,
        )
        return

      importer = CNSZarrImporter(
          output_dir=get_profile_manager().get_historical_dir(user)
      )
      job_mgr = get_job_manager()
      host = self.headers.get("Host", "localhost:8080")
      server_url = f"http://{host}"

      job_id = job_mgr.submit_job(
          job_type="cns_zarr_import",
          task_fn=importer.import_zarr_from_cns,
          cns_path=cns_path,
          dest_filename=dest_filename,
          notify_email=notify_email,
          server_url=server_url,
          metadata={
              "cns_path": cns_path,
              "dest_filename": dest_filename,
              "weather_source": "CNS Zarr Import",
          },
      )

      self._send_json({
          "status": "queued",
          "job_id": job_id,
          "cns_path": cns_path,
          "dest_filename": dest_filename,
          "notify_email": notify_email,
          "message": (
              f"CNS Zarr import queued in background from {cns_path} to {dest_filename}."
              + (f" Notification will be sent to {notify_email}." if notify_email else "")
          ),
      })
      return

    # 5. API: Extract Static Catchment Attributes (HydroATLAS / Caravan)
    if path == "/api/attributes/extract":
      user = self._get_request_username(data)
      pm = get_profile_manager()
      is_batch = bool(data.get("all_active", False) or data.get("scope") == "all")
      catchment_ids_input = data.get("catchment_ids") or data.get("basin_ids") or []
      features_input = data.get("features", [])
      catchment_id = data.get("catchment_id")
      polygon_input = data.get("polygon") or data.get("geometry")
      era5_source = data.get("era5_source") or "hybas"

      try:
        extractor = get_attributes_extractor()
      except Exception as e:
        self._send_error(
            f"Failed to initialize Static Attributes Extractor: {str(e)}",
            status=500,
        )
        return

      user_watersheds = pm.load_watersheds(user)
      ws_by_id = {
          str(f.get("properties", {}).get("catchment_id") or f.get("id")): f
          for f in user_watersheds
      }

      attr_dir = pm.get_attributes_dir(user)
      json_cache_path = attr_dir / "extracted_attributes.json"
      existing_by_id: Dict[str, Any] = {}
      if json_cache_path.exists():
        try:
          cached = json.loads(json_cache_path.read_text(encoding="utf-8"))
          if isinstance(cached, dict):
            existing_by_id = dict(cached.get("results_by_id", cached))
        except Exception:
          existing_by_id = {}

      def _persist_extracted_results(new_results: List[Dict[str, Any]]) -> Path:
        for r in new_results:
          cid = str(r.get("catchment_id") or "basin")
          existing_by_id[cid] = dict(r)
        try:
          json_cache_path.parent.mkdir(parents=True, exist_ok=True)
          json_cache_path.write_text(
              json.dumps({"results_by_id": existing_by_id}, indent=2),
              encoding="utf-8",
          )
        except Exception:
          pass

        csv_path = attr_dir / "caravan_hydroatlas_attributes.csv"
        all_results = list(existing_by_id.values()) or new_results
        extractor.export_caravan_csv(all_results, csv_path)

        master_zarr = pm.get_historical_dir(user) / "historical_training_master.zarr"
        if master_zarr.exists():
          for r in new_results:
            extractor.append_attributes_to_zarr(
                master_zarr, r["catchment_id"], r
            )
        return csv_path

      # Multi-Catchment Mode ("all" or "some" via catchment_ids / features)
      if (
          is_batch
          or catchment_ids_input
          or features_input
          or (
              not polygon_input
              and not catchment_id
              and len(user_watersheds) > 0
          )
      ):
        if features_input:
          target_features = features_input
        elif catchment_ids_input:
          target_features = [
              ws_by_id[str(cid)]
              for cid in catchment_ids_input
              if str(cid) in ws_by_id
          ]
        else:
          target_features = user_watersheds

        if not target_features:
          self._send_error(
              "No matching active or uploaded catchments to extract attributes for.",
              status=400,
          )
          return

        try:
          results = extractor.extract_attributes_batch(
              target_features, era5_source=era5_source
          )
          csv_path = _persist_extracted_results(results)
          first = results[0] if results else {}

          self._send_json({
              "status": "success",
              "batch": True,
              "count": len(results),
              "csv_path": str(csv_path),
              "results": results,
              "results_by_id": existing_by_id,
              "catchment_id": first.get("catchment_id"),
              "summary": first.get("summary", {}),
              "categories": first.get("categories", {}),
              "caravan_attributes": first.get("caravan_attributes", {}),
              "processed_attributes": first.get("processed_attributes", {}),
              "intersected_subbasins_count": sum(
                  int(r.get("intersected_subbasins_count", 0)) for r in results
              ),
          })
          return
        except Exception as e:
          self._send_error(
              f"Failed to extract static attributes batch: {str(e)}", status=500
          )
          return

      # Single Catchment Mode ("one")
      if not polygon_input and str(catchment_id) in ws_by_id:
        polygon_input = ws_by_id[str(catchment_id)]

      if not polygon_input and not catchment_id:
        self._send_error(
            "catchment_id, catchment_ids, features, or polygon geometry is required",
            status=400,
        )
        return

      if not polygon_input:
        self._send_error(
            f"Catchment '{catchment_id}' not found. Please delineate a basin or"
            " upload a polygon first.",
            status=404,
        )
        return

      try:
        result = extractor.extract_attributes_for_polygon(
            polygon_input,
            catchment_id=catchment_id,
            era5_source=era5_source,
        )
        single_copy = dict(result)
        csv_path = _persist_extracted_results([single_copy])
        resp_payload = {
            **single_copy,
            "status": "success",
            "batch": False,
            "count": 1,
            "csv_path": str(csv_path),
            "results": [single_copy],
            "results_by_id": existing_by_id,
        }
        self._send_json(resp_payload)
        return
      except Exception as e:
        self._send_error(
            f"Failed to extract static attributes: {str(e)}", status=500
        )
        return

    # 6. API: Fetch Real-Time MultiMet Forcings (/api/forecast/fetch-realtime)
    if path == "/api/forecast/fetch-realtime":
      user = self._get_request_username(data)
      catchment_id = data.get("catchment_id") or data.get("basin_id")
      mode = data.get("mode", "coldstart")
      reference_date = data.get("reference_date", "latest")
      products = data.get("products")
      lookback_days = data.get("lookback_days")
      overwrite = bool(data.get("overwrite", False))

      try:
        import importlib
        try:
          from frontend import realtime_forecast_service as _rfs
        except ImportError:
          import realtime_forecast_service as _rfs
        if Path(_rfs.__file__).stat().st_mtime > getattr(_rfs, "_LOADED_MTIME", 0.0):
          importlib.reload(_rfs)

        payload = _rfs.fetch_realtime_for_basin(
            username=user,
            basin_id=catchment_id,
            mode=mode,
            reference_date=reference_date,
            products=products,
            lookback_days=lookback_days,
            overwrite=overwrite,
        )
        self._send_json(payload)
        return
      except ValueError as e:
        self._send_error(str(e), status=400)
        return
      except Exception as e:
        self._send_error(f"Failed to fetch real-time MultiMet forcings: {e}", status=500)
        return

    # 7. API: Run Hydrological Model (/api/forecast/run-model)
    if path == "/api/forecast/run-model":
      user = self._get_request_username(data)
      catchment_id = data.get("catchment_id") or data.get("basin_id")
      mode = data.get("mode", "coldstart")
      model_run_dir = data.get("model_run_dir")

      try:
        import importlib
        try:
          from frontend import realtime_forecast_service as _rfs
        except ImportError:
          import realtime_forecast_service as _rfs
        if Path(_rfs.__file__).stat().st_mtime > getattr(_rfs, "_LOADED_MTIME", 0.0):
          importlib.reload(_rfs)

        payload = _rfs.run_hydrological_model_for_basin(
            username=user,
            basin_id=catchment_id,
            mode=mode,
            model_run_dir=model_run_dir,
        )
        self._send_json(payload)
        return
      except ValueError as e:
        self._send_error(str(e), status=400)
        return
      except Exception as e:
        self._send_error(f"Failed to run hydrological model: {e}", status=500)
        return

    self._send_error("Not Found", status=404)

  def log_message(self, format, *args):
    """Clean access logging."""
    print(f"[{self.log_date_time_string()}] {format % args}")


def _start_weather_sync():
  """Checks dynamical.org for newer forecast runs once an hour (weather_sync.py).

  After each check the weather engine switches to a newly downloaded run, so the
  Weather Viewer stays current without restarting the server.
  """
  try:
    try:
      from frontend import weather_sync
    except ImportError:
      import weather_sync  # pylint: disable=g-import-not-at-top
    if weather_sync.start_background_sync(on_finished=reload_weather_if_changed):
      print("  🌦️  Weather forecasts: checking dynamical.org for new runs every "
            f"{weather_sync.CHECK_INTERVAL_MINUTES} min "
            f"(data in {weather_sync.weather_data_root()})")
  except Exception as e:  # pylint: disable=broad-except
    print(f"[WeatherSync] Automatic forecast updates are off: {e}")


def run_server(port: int = 8000, host: str = "0.0.0.0"):
  """Starts the multi-threaded HTTP server."""
  # Every server run starts with an empty guest session: all guest data (basins,
  # stores, caches, jobs) is erased here and then kept until the next restart.
  get_profile_manager().clear_guest_data()
  _start_weather_sync()

  ThreadingHTTPServer.allow_reuse_address = True
  server_address = (host, port)
  httpd = ThreadingHTTPServer(server_address, EarthkitHydroHandler)
  print(f"\n=======================================================")
  print(f"  🌊 OpenHydroNet Web Server Running")
  print(f"  🌐 Local Access:   http://localhost:{port}")
  print(f"  🌐 Network Access: http://{host}:{port}")
  print(f"=======================================================\n")
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    print("\nShutting down OpenHydroNet Web Server...")
    httpd.shutdown()


OpenHydroNetHandler = EarthkitHydroHandler


def main() -> None:
  """CLI entry point for the OpenHydroNet Web Server."""
  import argparse
  parser = argparse.ArgumentParser(description="OpenHydroNet Web Server")
  parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to run the server on")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
  args = parser.parse_args()
  run_server(port=args.port, host=args.host)


if __name__ == "__main__":
  main()
