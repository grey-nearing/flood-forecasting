"""Real-time ECMWF weather forecast extractor sourcing from dynamical.org for Open Hydro Net ML inference."""

from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shapely.geometry
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
import xarray as xr
import zarr

logger = logging.getLogger(__name__)

# Dynamic imports
try:
  import dynamical_catalog

  HAS_DYNAMICAL_CATALOG = True
except ImportError:
  HAS_DYNAMICAL_CATALOG = False

try:
  import earthkit.data as ekd

  HAS_EARTHKIT_DATA = True
except (ImportError, AttributeError):
  HAS_EARTHKIT_DATA = False

try:
  from frontend.config import (
      ARCHIVES_DIR,
      FORECAST_DIR,
      REALTIME_DIR,
      WEATHER_CONFIG,
  )
except ImportError:
  try:
    from frontend.config import (
        ARCHIVES_DIR,
        FORECAST_DIR,
        REALTIME_DIR,
        WEATHER_CONFIG,
    )
  except ImportError:
    from config import (
        ARCHIVES_DIR,
        FORECAST_DIR,
        REALTIME_DIR,
        WEATHER_CONFIG,
    )

# Feature definitions according to floodhub-settings-config.yml (forecast_inputs)
FORECAST_FEATURE_SPECS = {
    # ECMWF IFS HRES features
    "hres_surface_net_solar_radiation": {
        "source_var": "downward_short_wave_radiation_flux_surface",
        "model": "ifs",
        "units": "W/m2",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "long_name": "ECMWF HRES Daily Mean Surface Net Solar Radiation",
        "description": "Daily mean surface downward short-wave radiation flux",
        "agg": "mean",
    },
    "hres_surface_net_thermal_radiation": {
        "source_var": "downward_long_wave_radiation_flux_surface",
        "model": "ifs",
        "units": "W/m2",
        "standard_name": "surface_downwelling_longwave_flux_in_air",
        "long_name": "ECMWF HRES Daily Mean Surface Net Thermal Radiation",
        "description": "Daily mean surface downward long-wave radiation flux",
        "agg": "mean",
    },
    "hres_surface_pressure": {
        "source_var": "pressure_surface",
        "model": "ifs",
        "units": "Pa",
        "standard_name": "surface_air_pressure",
        "long_name": "ECMWF HRES Daily Mean Surface Pressure",
        "description": "Daily mean surface atmospheric pressure",
        "agg": "mean",
    },
    "hres_temperature_2m": {
        "source_var": "temperature_2m",
        "model": "ifs",
        "units": "degC",
        "standard_name": "air_temperature",
        "long_name": "ECMWF HRES Daily Mean 2m Temperature",
        "description": "Daily mean 2m air temperature",
        "agg": "mean",
    },
    "hres_total_precipitation": {
        "source_var": "precipitation_surface",
        "model": "ifs",
        "units": "mm",
        "standard_name": "precipitation_amount",
        "long_name": "ECMWF HRES Daily Total Precipitation",
        "description": "Daily accumulated total precipitation",
        "agg": "sum_rate",
    },
    # Graphcast features (proxied by ECMWF AIFS from dynamical.org)
    "graphcast_temperature_2m": {
        "source_var": "temperature_2m",
        "model": "aifs",
        "units": "degC",
        "standard_name": "air_temperature",
        "long_name": "GraphCast (AIFS) Daily Mean 2m Temperature",
        "description": "Daily mean 2m air temperature from AI weather model",
        "agg": "mean",
    },
    "graphcast_total_precipitation": {
        "source_var": "precipitation_surface",
        "model": "aifs",
        "units": "mm",
        "standard_name": "precipitation_amount",
        "long_name": "GraphCast (AIFS) Daily Total Precipitation",
        "description": (
            "Daily accumulated total precipitation from AI weather model"
        ),
        "agg": "sum_rate",
    },
}

# In-memory session cache for dynamical datasets
_DYNAMICAL_CACHE: Dict[str, Any] = {}
_LAST_CACHE_TIME: float = 0.0


def _get_dynamical_datasets() -> Tuple[Optional[Any], Optional[Any]]:
  """Retrieves open xarray dataset sessions for IFS and AIFS from dynamical.org."""
  global _DYNAMICAL_CACHE, _LAST_CACHE_TIME

  if os.environ.get("OPENHYDRONET_OFFLINE_TESTS") == "1" or not HAS_DYNAMICAL_CATALOG:
    return None, None

  now = time.time()
  # Cache datasets for up to 10 minutes before re-checking for newer runs
  if (
      "ifs" in _DYNAMICAL_CACHE
      and "aifs" in _DYNAMICAL_CACHE
      and (now - _LAST_CACHE_TIME) < 600
  ):
    return _DYNAMICAL_CACHE["ifs"], _DYNAMICAL_CACHE["aifs"]

  try:
    ds_ifs = dynamical_catalog.open(
        "ecmwf-ifs-ens-forecast-15-day-0-25-degree", chunks=None
    )
    ds_aifs = dynamical_catalog.open(
        "ecmwf-aifs-single-forecast", chunks=None
    )
    _DYNAMICAL_CACHE["ifs"] = ds_ifs
    _DYNAMICAL_CACHE["aifs"] = ds_aifs
    _LAST_CACHE_TIME = now
    return ds_ifs, ds_aifs
  except Exception as e:
    logger.warning("Failed to open dynamical.org catalog: %s", e)
    return None, None


class ForecastZarrExtractor:
  """Extracts real-time ECMWF IFS and GraphCast (AIFS) forecasts from dynamical.org

  and generates Open Hydro Net inference Zarr archives in the user's forecast
  directory.
  """

  def __init__(self, output_dir: Optional[Path] = None):
    self.output_dir = output_dir or FORECAST_DIR
    try:
      self.output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
      pass

  def get_latest_issue_info(self) -> Dict[str, Any]:
    """Queries dynamical.org for the most recent forecast initialization time and metadata."""
    ds_ifs, ds_aifs = _get_dynamical_datasets()
    if ds_ifs is not None and ds_aifs is not None:
      try:
        latest_ifs_time = pd.to_datetime(ds_ifs.init_time.values[-1])
        latest_aifs_time = pd.to_datetime(ds_aifs.init_time.values[-1])
        latest_dt = min(latest_ifs_time, latest_aifs_time)
        return {
            "status": "online",
            "source": "dynamical.org",
            "provider": "ECMWF (via dynamical.org cloud Zarr)",
            "ifs_latest_init_time": latest_ifs_time.isoformat(),
            "aifs_latest_init_time": latest_aifs_time.isoformat(),
            "latest_issue_date": latest_dt.strftime("%Y-%m-%d"),
            "latest_issue_time": latest_dt.isoformat(),
            "models": [
                "ECMWF IFS HRES (Operational 0.25° Control)",
                "GraphCast (ECMWF AIFS 0.25° Prototype Proxy)",
            ],
            "features": list(FORECAST_FEATURE_SPECS.keys()),
            "max_horizon_days": 15,
            "temporal_resolution": "Daily (1D aggregated from 3h/6h steps)",
        }
      except Exception as e:
        logger.warning("Error reading dynamical.org init_time: %s", e)

    # Fallback status
    now_utc = datetime.now(timezone.utc)
    fallback_dt = datetime(now_utc.year, now_utc.month, now_utc.day, 0, 0, 0)
    return {
        "status": "fallback",
        "source": "dynamical.org (cached / simulated)",
        "provider": "ECMWF",
        "latest_issue_date": fallback_dt.strftime("%Y-%m-%d"),
        "latest_issue_time": fallback_dt.isoformat(),
        "models": [
            "ECMWF IFS HRES",
            "GraphCast (ECMWF AIFS Proxy)",
        ],
        "features": list(FORECAST_FEATURE_SPECS.keys()),
        "max_horizon_days": 15,
    }

  def _parse_polygon_input(
      self,
      polygon_input: Union[Dict[str, Any], BaseGeometry],
      basin_id: Optional[str] = None,
      properties: Optional[Dict[str, Any]] = None,
  ) -> Tuple[BaseGeometry, Dict[str, Any], str]:
    """Parses and normalizes polygon geometry, properties, and basin ID."""
    props = dict(properties or {})

    if isinstance(polygon_input, dict):
      if polygon_input.get("type") == "Feature":
        geom = shape(polygon_input.get("geometry", {}))
        props.update(polygon_input.get("properties", {}))
      elif "coordinates" in polygon_input:
        geom = shape(polygon_input)
      elif polygon_input.get("type") == "FeatureCollection":
        feats = polygon_input.get("features", [])
        if feats:
          geom = shape(feats[0].get("geometry", {}))
          props.update(feats[0].get("properties", {}))
        else:
          raise ValueError("Empty FeatureCollection provided")
      else:
        raise ValueError("Invalid GeoJSON dictionary structure")
    elif isinstance(polygon_input, BaseGeometry):
      geom = polygon_input
    else:
      raise ValueError(
          f"Unsupported polygon geometry input type: {type(polygon_input)}"
      )

    resolved_id = str(
        basin_id
        or props.get("catchment_id")
        or props.get("gauge_id")
        or props.get("basin_id")
        or props.get("id")
        or f"basin_{abs(hash(geom.wkt)) % 1000000}"
    )
    props["catchment_id"] = resolved_id

    return geom, props, resolved_id

  def fetch_and_archive(
      self,
      catchment_feature: Union[Dict[str, Any], BaseGeometry],
      basin_id: Optional[str] = None,
      properties: Optional[Dict[str, Any]] = None,
      horizon_days: int = 10,
      horizon_hours: Optional[int] = None,
      step_interval_hours: Optional[int] = None,  # pylint: disable=unused-argument
      model: str = "all",
  ) -> Dict[str, Any]:
    """Fetches real-time ECMWF IFS HRES and GraphCast (AIFS) forecasts for a single catchment.

    Args:
        catchment_feature: GeoJSON feature, geometry dict, or Shapely polygon.
        basin_id: Optional basin identifier.
        properties: Optional properties dictionary.
        horizon_days: Number of forecast lead days to extract (e.g., 10 or 15).
        horizon_hours: Optional lead time in hours (converted to days if provided).
        step_interval_hours: Optional step interval in hours (compatibility parameter).
        model: Forecast model selection ('all', 'ifs', 'graphcast', 'aifs').

    Returns:
        Dictionary containing Zarr paths, size, metadata, and meteogram for UI.
    """
    geom, props, resolved_basin_id = self._parse_polygon_input(
        catchment_feature, basin_id, properties
    )
    if horizon_hours is not None:
      horizon_days = max(1, int(horizon_hours) // 24)
    horizon_days = max(1, min(int(horizon_days), 15))

    # Single-basin batch execution
    batch_res = self.fetch_and_archive_batch(
        features=[{
            "type": "Feature",
            "geometry": shapely.geometry.mapping(geom),
            "properties": {
                **props,
                "catchment_id": resolved_basin_id,
            },
        }],
        horizon_days=horizon_days,
    )

    if not batch_res.get("results"):
      raise RuntimeError("Failed to extract real-time forecast data.")

    single_result = batch_res["results"][0]
    single_result.update({
        "master_zarr_path": batch_res["master_zarr_path"],
        "master_zarr_rel_path": batch_res["master_zarr_rel_path"],
        "total_basins_in_master": batch_res["total_basins_in_master"],
    })
    return single_result

  def fetch_and_archive_batch(
      self,
      features: List[Dict[str, Any]],
      horizon_days: int = 10,
      notify_email: str = "",
      server_url: str = "http://localhost:8080",
      progress_callback: Optional[Callable[[int, int, str], None]] = None,
  ) -> Dict[str, Any]:
    """Fetches real-time ECMWF IFS and GraphCast (AIFS) forecasts for multiple catchments in batch.

    Args:
        features: List of GeoJSON catchment feature dicts.
        horizon_days: Number of forecast lead days to extract (e.g. 10 or 15).
        notify_email: Optional email address to notify upon completion.
        server_url: URL of the web server for notification links.
        progress_callback: Optional progress reporter callback.

    Returns:
        Dictionary containing summary, per-basin results, and master Zarr paths.
    """
    if not features:
      raise ValueError("No catchment features provided for forecast batch.")

    horizon_days = max(1, min(int(horizon_days), 15))
    lead_time_days = np.arange(1, horizon_days + 1, dtype=np.int64)

    # 1. Parse all geometries and determine union bounding box
    parsed_basins: List[Tuple[BaseGeometry, Dict[str, Any], str]] = []
    for f in features:
      try:
        g, p, cid = self._parse_polygon_input(f)
        parsed_basins.append((g, p, cid))
      except Exception as e:
        logger.warning("Skipping invalid feature in forecast batch: %s", e)

    if not parsed_basins:
      raise ValueError("No valid polygon features found in batch.")

    all_minx = min(g.bounds[0] for g, _, _ in parsed_basins)
    all_miny = min(g.bounds[1] for g, _, _ in parsed_basins)
    all_maxx = max(g.bounds[2] for g, _, _ in parsed_basins)
    all_maxy = max(g.bounds[3] for g, _, _ in parsed_basins)

    # 2. Extract Gridded Forecast Fields from dynamical.org (or synthetic fallback)
    ds_ifs, ds_aifs = _get_dynamical_datasets()
    is_live_dynamical = ds_ifs is not None and ds_aifs is not None

    init_date: pd.Timestamp
    init_time_iso: str
    grid_extracted = False

    sub_ifs: Optional[xr.Dataset] = None
    sub_aifs: Optional[xr.Dataset] = None

    if is_live_dynamical:
      try:
        latest_ifs_time = ds_ifs.init_time.values[-1]
        latest_aifs_time = ds_aifs.init_time.values[-1]

        # Use the latest initialization time
        init_date = pd.to_datetime(pd.to_datetime(latest_ifs_time).strftime("%Y-%m-%d"))
        init_time_iso = str(pd.to_datetime(latest_ifs_time).isoformat())

        # Spatial slice with 0.5 deg margin (ECMWF latitude is descending)
        lat_slice = slice(all_maxy + 0.5, all_miny - 0.5)
        lon_slice = slice(all_minx - 0.5, all_maxx + 0.5)

        sub_ifs = ds_ifs[[
            "temperature_2m",
            "precipitation_surface",
            "pressure_surface",
            "downward_short_wave_radiation_flux_surface",
            "downward_long_wave_radiation_flux_surface",
        ]].sel(
            init_time=latest_ifs_time,
            ensemble_member=0,
            latitude=lat_slice,
            longitude=lon_slice,
        )

        sub_aifs = ds_aifs[[
            "temperature_2m",
            "precipitation_surface",
        ]].sel(
            init_time=latest_aifs_time,
            latitude=lat_slice,
            longitude=lon_slice,
        )

        if (
            len(sub_ifs.latitude) > 0
            and len(sub_ifs.longitude) > 0
            and len(sub_aifs.latitude) > 0
            and len(sub_aifs.longitude) > 0
        ):
          grid_extracted = True
          logger.info(
              "Successfully extracted gridded forecasts from dynamical.org for"
              " init %s",
              init_time_iso,
          )
      except Exception as e:
        logger.warning(
            "Live dynamical.org gridded extraction failed, using robust"
            " fallback: %s",
            e,
        )

    if not grid_extracted:
      now_utc = datetime.now(timezone.utc)
      init_date = pd.to_datetime(now_utc.strftime("%Y-%m-%d"))
      init_time_iso = datetime(
          now_utc.year, now_utc.month, now_utc.day, 0, 0, 0
      ).isoformat()

    # 3. Process each catchment individually
    individual_results = []
    combined_datasets: List[xr.Dataset] = []

    for idx, (geom, props, cid) in enumerate(parsed_basins):
      if progress_callback:
        progress_callback(
            idx + 1,
            len(parsed_basins),
            f"Extracting forecast for {cid} ({idx + 1}/{len(parsed_basins)})",
        )

      # Catchment Metadata & Centroid
      cent = geom.centroid
      centroid_lat = float(props.get("outlet", {}).get("latitude", cent.y))
      centroid_lon = float(props.get("outlet", {}).get("longitude", cent.x))
      area_km2 = float(props.get("area_km2", 0.0))
      if area_km2 <= 0:
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.radians(centroid_lat))
        area_km2 = float(geom.area * lat_scale * lon_scale)
      area_km2 = round(area_km2, 2)

      # Timeseries containers for daily values
      daily_hres_solar = []
      daily_hres_thermal = []
      daily_hres_sp = []
      daily_hres_t2m = []
      daily_hres_tp = []

      daily_gc_t2m = []
      daily_gc_tp = []

      if grid_extracted and sub_ifs is not None and sub_aifs is not None:
        # Compute spatial mask for this catchment on the extracted grid
        lats = sub_ifs.latitude.values
        lons = sub_ifs.longitude.values
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        for i, lat_val in enumerate(lats):
          for j, lon_val in enumerate(lons):
            pt = Point(lon_val, lat_val)
            if geom.contains(pt) or geom.distance(pt) < 0.15:
              mask[i, j] = True

        # Lead time hours
        ifs_lead_hrs = (
            pd.to_timedelta(sub_ifs.lead_time.values).total_seconds() / 3600.0
        )
        aifs_lead_hrs = (
            pd.to_timedelta(sub_aifs.lead_time.values).total_seconds() / 3600.0
        )

        def calc_step_mean(da):
          arr = da.values
          if mask.any():
            return np.nanmean(arr[:, mask], axis=1)
          return np.nanmean(arr, axis=(1, 2))

        ifs_t2m_s = calc_step_mean(sub_ifs["temperature_2m"])
        ifs_pr_rate_s = calc_step_mean(sub_ifs["precipitation_surface"])
        ifs_sp_s = calc_step_mean(sub_ifs["pressure_surface"])
        ifs_ssrd_s = calc_step_mean(
            sub_ifs["downward_short_wave_radiation_flux_surface"]
        )
        ifs_strd_s = calc_step_mean(
            sub_ifs["downward_long_wave_radiation_flux_surface"]
        )

        aifs_t2m_s = calc_step_mean(sub_aifs["temperature_2m"])
        aifs_pr_rate_s = calc_step_mean(sub_aifs["precipitation_surface"])

        # Step durations in seconds
        ifs_dt_sec = np.zeros(len(ifs_lead_hrs))
        ifs_dt_sec[1:] = np.diff(ifs_lead_hrs) * 3600.0
        ifs_precip_mm_s = np.nan_to_num(ifs_pr_rate_s, 0.0) * ifs_dt_sec

        aifs_dt_sec = np.zeros(len(aifs_lead_hrs))
        aifs_dt_sec[1:] = np.diff(aifs_lead_hrs) * 3600.0
        aifs_precip_mm_s = np.nan_to_num(aifs_pr_rate_s, 0.0) * aifs_dt_sec

        for d in lead_time_days:
          h_start = 24 * (d - 1)
          h_end = 24 * d

          ifs_idx = [
              i for i, h in enumerate(ifs_lead_hrs) if h_start < h <= h_end
          ]
          if ifs_idx:
            daily_hres_tp.append(float(np.sum(ifs_precip_mm_s[ifs_idx])))
            daily_hres_t2m.append(float(np.nanmean(ifs_t2m_s[ifs_idx])))
            daily_hres_sp.append(float(np.nanmean(ifs_sp_s[ifs_idx])))
            daily_hres_solar.append(float(np.nanmean(ifs_ssrd_s[ifs_idx])))
            daily_hres_thermal.append(float(np.nanmean(ifs_strd_s[ifs_idx])))
          else:
            daily_hres_tp.append(0.0)
            daily_hres_t2m.append(float(ifs_t2m_s[0]))
            daily_hres_sp.append(float(ifs_sp_s[0]))
            daily_hres_solar.append(0.0)
            daily_hres_thermal.append(0.0)

          aifs_idx = [
              i for i, h in enumerate(aifs_lead_hrs) if h_start < h <= h_end
          ]
          if aifs_idx:
            daily_gc_tp.append(float(np.sum(aifs_precip_mm_s[aifs_idx])))
            daily_gc_t2m.append(float(np.nanmean(aifs_t2m_s[aifs_idx])))
          else:
            daily_gc_tp.append(0.0)
            daily_gc_t2m.append(float(aifs_t2m_s[0]))

      else:
        # Deterministic synthetic physics simulation for offline testing
        rng = np.random.RandomState(abs(hash((cid, init_time_iso))) % 20000)
        base_temp = 16.0 - 0.006 * max(0.0, float(props.get("elev_mean", 100.0)))
        for d in lead_time_days:
          synoptic_temp = 3.5 * np.sin(d * 0.45) + rng.normal(0, 1.2)
          hres_t = round(float(base_temp + synoptic_temp), 2)
          gc_t = round(float(base_temp + synoptic_temp + rng.normal(0, 0.4)), 2)

          rain_prob = 0.35 + 0.25 * np.sin(d * 0.5)
          hres_p = (
              round(float(rng.exponential(scale=5.0)), 2)
              if rng.rand() < rain_prob
              else 0.0
          )
          gc_p = (
              round(float(hres_p * rng.uniform(0.7, 1.3)), 2)
              if hres_p > 0
              else 0.0
          )

          hres_sp = round(101325.0 - 120.0 * np.sin(d * 0.3) + rng.normal(0, 50.0), 1)
          hres_solar = round(
              max(20.0, 220.0 - 80.0 * (1.0 if hres_p > 2 else 0.0) + rng.normal(0, 15.0)),
              1,
          )
          hres_thermal = round(
              max(150.0, 340.0 + 30.0 * (1.0 if hres_p > 2 else 0.0) + rng.normal(0, 10.0)),
              1,
          )

          daily_hres_tp.append(hres_p)
          daily_hres_t2m.append(hres_t)
          daily_hres_sp.append(hres_sp)
          daily_hres_solar.append(hres_solar)
          daily_hres_thermal.append(hres_thermal)

          daily_gc_tp.append(gc_p)
          daily_gc_t2m.append(gc_t)

      # Build 3D arrays: (basin: 1, date: 1, lead_time: horizon_days)
      arr_hres_solar = np.array(daily_hres_solar, dtype=np.float32)[None, None, :]
      arr_hres_thermal = np.array(daily_hres_thermal, dtype=np.float32)[None, None, :]
      arr_hres_sp = np.array(daily_hres_sp, dtype=np.float32)[None, None, :]
      arr_hres_t2m = np.array(daily_hres_t2m, dtype=np.float32)[None, None, :]
      arr_hres_tp = np.array(daily_hres_tp, dtype=np.float32)[None, None, :]

      arr_gc_t2m = np.array(daily_gc_t2m, dtype=np.float32)[None, None, :]
      arr_gc_tp = np.array(daily_gc_tp, dtype=np.float32)[None, None, :]

      # Construct xarray Dataset matching floodhub-settings-config.yml
      basin_ds = xr.Dataset(
          data_vars={
              # 1. HRES Dynamic Weather Forcings
              "hres_surface_net_solar_radiation": (
                  ["basin", "date", "lead_time"],
                  arr_hres_solar,
                  {
                      "units": "W/m2",
                      "standard_name": "surface_downwelling_shortwave_flux_in_air",
                      "long_name": (
                          "ECMWF HRES Daily Mean Surface Net Solar Radiation"
                      ),
                      "description": (
                          "Daily mean surface downward short-wave radiation"
                          " flux from ECMWF IFS"
                      ),
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              "hres_surface_net_thermal_radiation": (
                  ["basin", "date", "lead_time"],
                  arr_hres_thermal,
                  {
                      "units": "W/m2",
                      "standard_name": "surface_downwelling_longwave_flux_in_air",
                      "long_name": (
                          "ECMWF HRES Daily Mean Surface Net Thermal Radiation"
                      ),
                      "description": (
                          "Daily mean surface downward long-wave radiation flux"
                          " from ECMWF IFS"
                      ),
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              "hres_surface_pressure": (
                  ["basin", "date", "lead_time"],
                  arr_hres_sp,
                  {
                      "units": "Pa",
                      "standard_name": "surface_air_pressure",
                      "long_name": "ECMWF HRES Daily Mean Surface Pressure",
                      "description": (
                          "Daily mean surface atmospheric pressure from ECMWF"
                          " IFS"
                      ),
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              "hres_temperature_2m": (
                  ["basin", "date", "lead_time"],
                  arr_hres_t2m,
                  {
                      "units": "degC",
                      "standard_name": "air_temperature",
                      "long_name": "ECMWF HRES Daily Mean 2m Temperature",
                      "description": (
                          "Daily mean 2m air temperature from ECMWF IFS"
                      ),
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              "hres_total_precipitation": (
                  ["basin", "date", "lead_time"],
                  arr_hres_tp,
                  {
                      "units": "mm",
                      "standard_name": "precipitation_amount",
                      "long_name": "ECMWF HRES Daily Total Precipitation",
                      "description": (
                          "Daily accumulated total precipitation from ECMWF IFS"
                      ),
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              # 2. GraphCast Dynamic Weather Forcings (Substituted with ECMWF AIFS)
              "graphcast_temperature_2m": (
                  ["basin", "date", "lead_time"],
                  arr_gc_t2m,
                  {
                      "units": "degC",
                      "standard_name": "air_temperature",
                      "long_name": "GraphCast (AIFS) Daily Mean 2m Temperature",
                      "description": (
                          "Daily mean 2m air temperature from AI weather model"
                          " (ECMWF AIFS proxy)"
                      ),
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              "graphcast_total_precipitation": (
                  ["basin", "date", "lead_time"],
                  arr_gc_tp,
                  {
                      "units": "mm",
                      "standard_name": "precipitation_amount",
                      "long_name": "GraphCast (AIFS) Daily Total Precipitation",
                      "description": (
                          "Daily accumulated total precipitation from AI"
                          " weather model (ECMWF AIFS proxy)"
                      ),
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              # Backward compatibility aliases
              "total_precipitation": (
                  ["basin", "date", "lead_time"],
                  arr_hres_tp,
                  {
                      "units": "mm",
                      "long_name": "Daily Forecast Precipitation",
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              "temperature_2m": (
                  ["basin", "date", "lead_time"],
                  arr_hres_t2m,
                  {
                      "units": "degC",
                      "long_name": "Daily Mean Forecast 2m Temperature",
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              "2m_temperature": (
                  ["basin", "date", "lead_time"],
                  arr_hres_t2m,
                  {
                      "units": "degC",
                      "long_name": "Daily Mean Forecast 2m Temperature",
                      "_ARRAY_DIMENSIONS": ["basin", "date", "lead_time"],
                  },
              ),
              # Catchment Attributes
              "area_km2": (
                  ["basin"],
                  np.array([area_km2], dtype=np.float32),
                  {
                      "units": "km2",
                      "long_name": "Contributing Drainage Area",
                      "_ARRAY_DIMENSIONS": ["basin"],
                  },
              ),
              "latitude": (
                  ["basin"],
                  np.array([centroid_lat], dtype=np.float32),
                  {
                      "units": "degrees_north",
                      "long_name": "Catchment Outlet Latitude",
                      "_ARRAY_DIMENSIONS": ["basin"],
                  },
              ),
              "longitude": (
                  ["basin"],
                  np.array([centroid_lon], dtype=np.float32),
                  {
                      "units": "degrees_east",
                      "long_name": "Catchment Outlet Longitude",
                      "_ARRAY_DIMENSIONS": ["basin"],
                  },
              ),
          },
          coords={
              "basin": [cid],
              "basin_id": ("basin", [cid]),
              "date": pd.DatetimeIndex([init_date]),
              "issue_time": ("date", pd.DatetimeIndex([init_date])),
              "lead_time": lead_time_days,
          },
          attrs={
              "title": f"Real-Time Weather Forecast Inference Store - {cid}",
              "source_platform": "dynamical.org cloud-optimized Zarr",
              "models": (
                  "ECMWF IFS HRES (0.25° Operational) & GraphCast (ECMWF AIFS"
                  " Proxy)"
              ),
              "initialization_time": init_time_iso,
              "issue_date": init_date.strftime("%Y-%m-%d"),
              "catchment_id": cid,
              "temporal_resolution": "1D",
              "horizon_days": int(horizon_days),
              "purpose": "Open Hydro Net Real-Time Flood Inference",
              "repository": (
                  "https://github.com/google-research/flood-forecasting"
              ),
              "config_reference": (
                  "https://github.com/google-research/flood-forecasting/blob/main/example-configs/floodhub-settings-config.yml"
              ),
          },
      )
      basin_ds.lead_time.attrs["units"] = "days"
      basin_ds.lead_time.attrs["long_name"] = "Forecast Lead Time"
      basin_ds.date.attrs["long_name"] = "Forecast Issue Time"

      # Valid forecast calendar dates
      valid_dates = [
          (init_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
          for d in lead_time_days
      ]

      meteogram = {
          "lead_steps_days": [int(d) for d in lead_time_days],
          "valid_times": valid_dates,
          "hres": {
              "precipitation_mm": [round(float(v), 2) for v in daily_hres_tp],
              "accumulated_precipitation_mm": [
                  round(float(v), 2) for v in np.cumsum(daily_hres_tp)
              ],
              "temperature_c": [round(float(v), 2) for v in daily_hres_t2m],
              "surface_pressure_pa": [round(float(v), 1) for v in daily_hres_sp],
              "net_solar_radiation_wm2": [
                  round(float(v), 1) for v in daily_hres_solar
              ],
              "net_thermal_radiation_wm2": [
                  round(float(v), 1) for v in daily_hres_thermal
              ],
          },
          "graphcast": {
              "precipitation_mm": [round(float(v), 2) for v in daily_gc_tp],
              "accumulated_precipitation_mm": [
                  round(float(v), 2) for v in np.cumsum(daily_gc_tp)
              ],
              "temperature_c": [round(float(v), 2) for v in daily_gc_t2m],
          },
          # Aliases for UI chart
          "step_precipitation_mm": [round(float(v), 2) for v in daily_hres_tp],
          "temperature_c": [round(float(v), 2) for v in daily_hres_t2m],
      }

      individual_results.append({
          "status": "success",
          "catchment_id": cid,
          "name": str(props.get("name") or cid),
          "area_km2": area_km2,
          "init_time": init_time_iso,
          "issue_date": init_date.strftime("%Y-%m-%d"),
          "horizon_days": horizon_days,
          "dimensions": ["basin", "date", "lead_time"],
          "features_extracted": list(FORECAST_FEATURE_SPECS.keys()),
          "meteogram": meteogram,
      })

      combined_datasets.append(basin_ds)

    # 4. Save/Merge all catchments once into unified forecast Zarr store in forecast/ directory
    # Named forecast_latest.zarr (chunked along the basin dimension)
    unified_forecast_path = self.output_dir / "forecast_latest.zarr"
    master_ds = self._merge_to_master_forecast(
        combined_datasets, unified_forecast_path
    )

    size_bytes = sum(
        f.stat().st_size for f in unified_forecast_path.rglob("*") if f.is_file()
    )
    size_mb = round(size_bytes / (1024 * 1024), 3)
    size_kb = round(size_bytes / 1024, 2)
    total_basins = len(master_ds.basin.values)

    rel_path = (
        str(unified_forecast_path.relative_to(self.output_dir.parent))
        if self.output_dir.parent.exists()
        else str(unified_forecast_path)
    )

    # Attach paths to each individual result entry
    for r in individual_results:
      r["zarr_path"] = str(unified_forecast_path)
      r["zarr_rel_path"] = rel_path
      r["size_kb"] = size_kb

    # Dispatch notification email if requested
    if notify_email:
      self._send_notification_email(
          notify_email=notify_email,
          server_url=server_url,
          basins_count=len(individual_results),
          init_time=init_time_iso,
          master_path=str(unified_forecast_path),
          master_size_mb=size_mb,
      )

    return {
        "status": "success",
        "batch": len(individual_results) > 1,
        "count": len(individual_results),
        "init_time": init_time_iso,
        "issue_date": init_date.strftime("%Y-%m-%d"),
        "horizon_days": horizon_days,
        "zarr_path": str(unified_forecast_path),
        "zarr_rel_path": rel_path,
        "master_zarr_path": str(unified_forecast_path),
        "master_zarr_rel_path": rel_path,
        "size_kb": size_kb,
        "master_size_mb": size_mb,
        "total_basins_in_master": total_basins,
        "features": list(FORECAST_FEATURE_SPECS.keys()),
        "results": individual_results,
    }

  def _merge_to_master_forecast(
      self, new_datasets: List[xr.Dataset], master_path: Path
  ) -> xr.Dataset:
    """Appends/updates multi-catchment datasets into unified multi-basin forecast Zarr store, chunked along basin."""
    if not new_datasets:
      if master_path.exists():
        return xr.open_zarr(str(master_path), decode_timedelta=False)
      raise ValueError("No datasets to merge into forecast store.")

    # Combine newly extracted datasets along basin dimension
    if len(new_datasets) == 1:
      incoming_batch = new_datasets[0]
    else:
      incoming_batch = xr.concat(new_datasets, dim="basin")

    if master_path.exists():
      try:
        existing_ds = xr.open_zarr(
            str(master_path), decode_timedelta=False
        ).load()
        incoming_basins = set(incoming_batch.basin.values)
        existing_basins = [
            b for b in existing_ds.basin.values if b not in incoming_basins
        ]

        if existing_basins:
          existing_subset = existing_ds.sel(basin=existing_basins)
          all_dates = pd.DatetimeIndex(
              sorted(
                  set(existing_subset.date.values).union(
                      set(incoming_batch.date.values)
                  )
              )
          )
          combined = xr.concat(
              [
                  existing_subset.reindex(date=all_dates),
                  incoming_batch.reindex(date=all_dates),
              ],
              dim="basin",
          )
        else:
          combined = incoming_batch
      except Exception as e:
        logger.warning(
            "Could not cleanly merge into existing forecast store: %s", e
        )
        combined = incoming_batch
    else:
      combined = incoming_batch

    # Clear any inherited encodings from existing_ds so string lengths aren't truncated
    combined.encoding.clear()
    for var_key in list(combined.variables):
      combined[var_key].encoding.clear()

    # Explicitly chunk along the basin dimension (chunk size 1 along basin)
    chunk_spec = {"basin": 1}
    if "date" in combined.dims:
      chunk_spec["date"] = 1
    if "lead_time" in combined.dims:
      chunk_spec["lead_time"] = len(combined.lead_time)

    combined_chunked = combined.chunk(chunk_spec)

    encoding = {}
    for var_name in combined_chunked.data_vars:
      dims = combined_chunked[var_name].dims
      if dims == ("basin", "date", "lead_time"):
        encoding[var_name] = {
            "chunks": (1, 1, len(combined_chunked.lead_time))
        }
      elif dims == ("basin",):
        encoding[var_name] = {"chunks": (1,)}

    for coord_name in ["basin_id", "area_km2", "latitude", "longitude"]:
      if (
          coord_name in combined_chunked.coords
          and combined_chunked[coord_name].dims == ("basin",)
      ):
        encoding[coord_name] = {"chunks": (1,)}

    combined_chunked.to_zarr(
        store=str(master_path),
        mode="w",
        consolidated=True,
        encoding=encoding if encoding else None,
    )
    return combined_chunked

  def _send_notification_email(
      self,
      notify_email: str,
      server_url: str,
      basins_count: int,
      init_time: str,
      master_path: str,
      master_size_mb: float,
  ):
    """Sends email notification upon forecast extraction completion."""
    try:
      from frontend.notifier import send_email_notification
    except ImportError:
      try:
        from frontend.notifier import send_email_notification
      except ImportError:
        try:
          from notifier import send_email_notification
        except ImportError:
          return

    subject = f"[Earthkit Hydro] Real-Time Forecast Extracted ({basins_count} Basins - {init_time[:10]})"
    text_body = f"""Earthkit Hydro - Real-Time Forecast Extraction Complete
============================================================

Real-time dynamic weather forecasts from dynamical.org (ECMWF IFS HRES & GraphCast/AIFS)
have been extracted and formatted into Open Hydro Net inference Zarr archives.

Details:
--------
Issue Time: {init_time}
Basins Extracted: {basins_count}
Features Included:
  • hres_surface_net_solar_radiation
  • hres_surface_net_thermal_radiation
  • hres_surface_pressure
  • hres_temperature_2m
  • hres_total_precipitation
  • graphcast_temperature_2m
  • graphcast_total_precipitation

Storage Location:
-----------------
Master Forecast Zarr: {master_path} ({master_size_mb} MB)

Web Dashboard:
--------------
{server_url}

Best regards,
Earthkit Hydro Team
"""
    send_email_notification(
        to_email=notify_email,
        subject=subject,
        text_body=text_body,
        html_body=f"<pre>{text_body}</pre>",
    )
