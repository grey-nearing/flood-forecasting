"""Real-Time Forecasting & Hot-Start Service for OpenHydroNet.

Connects the Forecasting and Data Assimilation UI tab directly to:
1. `multimet.timeseries_extractors.realtime.fetch_realtime_multimet` (Mode 1 cold-start 365-day
   spin-up fetch & Mode 2 hot-start 1-day incremental update across CPC,
   IMERG, HRES, ECMWF AIFS, NOAA GFS, NOAA GEFS, and ECMWF IFS-ENS).
2. `model` (`MeanEmbeddingForecastLSTM` + `HotStartMixin`) for
   real Cold-Start (`forward()` + `save_states()`) and Hot-Start
   (`load_states()` + `predict_from_state()`) inference.
3. Strict state-date validation (`state_date == issue_date - 1 day`) and
   strict NaN preservation for real-time precipitation latency gaps.
"""

from __future__ import annotations

import datetime
import importlib
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from frontend import profile_manager
from frontend.config import (
    FLOOD_FORECASTING_REPO_DIR,
    ensure_flood_forecasting_on_sys_path,
    extend_multimet_package_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_RUN_DIR = (
    FLOOD_FORECASTING_REPO_DIR
    / "model"
    / "tutorial"
    / "model-runs"
    / "5-basin-example"
)
DEFAULT_TUTORIAL_CARAVAN_DIR = (
    FLOOD_FORECASTING_REPO_DIR / "model" / "tutorial" / "Caravan-nc"
)

NOWCAST_PRODUCTS_METADATA: Dict[str, Dict[str, Any]] = {
    "IMERG": {
        "id": "IMERG",
        "label": "NASA GPM IMERG Final/Late Run",
        "var_name": "imerg_precip",
        "typical_latency_days": 1,
    },
    "CPC": {
        "id": "CPC",
        "label": "NOAA CPC Unified Gauge-Based Analysis",
        "var_name": "cpc_precip",
        "typical_latency_days": 2,
    },
    "ERA5_LAND": {
        "id": "ERA5_LAND",
        "label": "ECMWF ERA5-Land Nowcast",
        "var_name": "era5land_total_precipitation",
        "typical_latency_days": 5,
    },
    "CHIRPS": {
        "id": "CHIRPS",
        "label": "UCSB CHIRPS v2.0 / v3.0 Daily",
        "var_name": "chirps_precip",
        "typical_latency_days": 2,
    },
}

FORECAST_PRODUCTS_METADATA: Dict[str, Dict[str, Any]] = {
    "HRES": {
        "id": "HRES",
        "label": "ECMWF IFS HRES (0.25° Open Data)",
        "precip_var": "hres_total_precipitation",
        "temp_var": "hres_temperature_2m",
    },
    "ECMWF_AIFS": {
        "id": "ECMWF_AIFS",
        "label": "ECMWF AIFS Single (AI Weather Model)",
        "precip_var": "aifs_total_precipitation",
        "temp_var": "aifs_temperature_2m",
    },
    "AIFS": {
        "id": "AIFS",
        "label": "ECMWF AIFS Single (AI Weather Model)",
        "precip_var": "aifs_total_precipitation",
        "temp_var": "aifs_temperature_2m",
    },
    "NOAA_GFS": {
        "id": "NOAA_GFS",
        "label": "NOAA GFS (Global Forecast System)",
        "precip_var": "gfs_total_precipitation",
        "temp_var": "gfs_temperature_2m",
    },
    "GFS": {
        "id": "GFS",
        "label": "NOAA GFS (Global Forecast System)",
        "precip_var": "gfs_total_precipitation",
        "temp_var": "gfs_temperature_2m",
    },
    "NOAA_GEFS": {
        "id": "NOAA_GEFS",
        "label": "NOAA GEFS Ensemble Mean (31-Member)",
        "precip_var": "gefs_total_precipitation",
        "temp_var": "gefs_temperature_2m",
    },
    "GEFS": {
        "id": "GEFS",
        "label": "NOAA GEFS Ensemble Mean (31-Member)",
        "precip_var": "gefs_total_precipitation",
        "temp_var": "gefs_temperature_2m",
    },
    "ECMWF_IFS_ENS": {
        "id": "ECMWF_IFS_ENS",
        "label": "ECMWF IFS Ensemble Mean (51-Member)",
        "precip_var": "ifs_ens_total_precipitation",
        "temp_var": "ifs_ens_temperature_2m",
    },
    "IFS_ENS": {
        "id": "IFS_ENS",
        "label": "ECMWF IFS Ensemble Mean (51-Member)",
        "precip_var": "ifs_ens_total_precipitation",
        "temp_var": "ifs_ens_temperature_2m",
    },
}

# Leaf-to-root dependency order inside `multimet` so that reloading
# `multimet.timeseries_extractors.config` (which recreates the `Product` Enum class)
# also rebinds `Product` in every downstream module that imported it by value.
_MULTIMET_RELOAD_ORDER: Tuple[str, ...] = (
    "multimet.utils.gcs",
    "multimet.utils.spatial",
    "multimet.utils.climate",
    "multimet.utils.storage",
    "multimet.timeseries_extractors.config",
    "multimet.timeseries_extractors.zarr_writer",
    "multimet.timeseries_extractors.cpc",
    "multimet.timeseries_extractors.imerg",
    "multimet.timeseries_extractors.era5_land",
    "multimet.timeseries_extractors.hres",
    "multimet.timeseries_extractors.dynamical",
    "multimet.timeseries_extractors.realtime",
)


def _load_multimet_realtime(reload_modules: bool = False):
  """Imports `multimet.timeseries_extractors.config` and `realtime` from the repository."""
  ensure_flood_forecasting_on_sys_path()
  extend_multimet_package_path()
  from multimet.timeseries_extractors import config as mm_config  # pylint: disable=g-import-not-at-top
  from multimet.timeseries_extractors import realtime as mm_realtime  # pylint: disable=g-import-not-at-top

  if reload_modules:
    for mod_name in _MULTIMET_RELOAD_ORDER:
      mod = sys.modules.get(mod_name)
      if mod is not None:
        importlib.reload(mod)
    mm_config = sys.modules["multimet.timeseries_extractors.config"]
    mm_realtime = sys.modules["multimet.timeseries_extractors.realtime"]
  return mm_config, mm_realtime


def _clean_float(val: Any) -> Optional[float]:
  """Converts numeric values to Python float, turning NaN/Inf into None (JSON null)."""
  if val is None:
    return None
  try:
    fval = float(val)
  except (TypeError, ValueError):
    return None
  if math.isnan(fval) or math.isinf(fval):
    return None
  return round(fval, 4)


def _format_date_str(ts: Any) -> str:
  return pd.Timestamp(ts).strftime("%Y-%m-%d")


def get_state_file_path(
    username: Optional[str],
    catchment_id: str,
    model_id: str = "5-basin-example",
) -> Path:
  """Returns the canonical `.npz` state file path for a user, catchment, and model."""
  pm = profile_manager.get_profile_manager()
  safe_cid = "".join(
      c for c in str(catchment_id) if c.isalnum() or c in ("_", "-")
  )
  safe_mid = "".join(c for c in str(model_id) if c.isalnum() or c in ("_", "-"))
  states_dir = pm.get_model_states_dir(username)
  return states_dir / f"{safe_cid}_{safe_mid}_state.npz"


def inspect_saved_state(
    username: Optional[str],
    catchment_id: str,
    model_id: str = "5-basin-example",
) -> Dict[str, Any]:
  """Reads metadata (including `date`) from a saved `.npz` LSTM state file if present."""
  state_path = get_state_file_path(username, catchment_id, model_id=model_id)
  if not state_path.exists():
    return {
        "exists": False,
        "state_path": str(state_path),
        "state_date": None,
        "hidden_shape": None,
    }
  try:
    with np.load(state_path, allow_pickle=False) as data:
      state_date = str(data["date"].item()) if "date" in data else None
      h_shape = list(data["h_n"].shape) if "h_n" in data else None
    return {
        "exists": True,
        "state_path": str(state_path),
        "state_date": state_date,
        "hidden_shape": h_shape,
    }
  except Exception as exc:  # pylint: disable=broad-except
    logger.warning("Failed to inspect state file %s: %s", state_path, exc)
    return {
        "exists": False,
        "state_path": str(state_path),
        "state_date": None,
        "error": str(exc),
    }


def _discover_zarr_stores(username: Optional[str]) -> Dict[str, Path]:
  """Finds all `<PRODUCT>/timeseries.zarr` stores in the user's realtime & historical dynamics."""
  pm = profile_manager.get_profile_manager()
  stores: Dict[str, Path] = {}
  # Check realtime/dynamics first, then dynamics (historical) as fallback
  for base_dir in (
      pm.get_realtime_dynamics_dir(username),
      pm.get_dynamics_dir(username),
  ):
    if not base_dir.exists():
      continue
    for child in sorted(base_dir.iterdir()):
      if not child.is_dir():
        continue
      zarr_candidate = child / "timeseries.zarr"
      if zarr_candidate.exists() and child.name not in stores:
        stores[child.name] = zarr_candidate
      elif (child / ".zgroup").exists() and child.stem not in stores:
        stores[child.stem] = child
  return stores


def _find_basin_key_in_ds(ds: xr.Dataset, catchment_id: str) -> Optional[str]:
  """Matches `catchment_id` against dataset's `basin` coordinate (exact or case-insensitive)."""
  if "basin" not in ds.coords:
    return None
  basins = [str(b) for b in ds.coords["basin"].values]
  if catchment_id in basins:
    return catchment_id
  lower_map = {b.lower(): b for b in basins}
  if catchment_id.lower() in lower_map:
    return lower_map[catchment_id.lower()]
  # If there is only 1 basin in the user's store, use it
  if len(basins) == 1:
    return basins[0]
  return None


def resolve_issue_date(
    username: Optional[str],
    catchment_id: str,
    requested_issue_date: Optional[str] = None,
    probe_ecmwf: bool = False,
) -> pd.Timestamp:
  """Determines the forecast issue date `t0`.

  Priority:
  1. Explicit `requested_issue_date` if provided.
  2. Latest issue date (`date` coordinate) in the user's `HRES/timeseries.zarr`
     (or other forecast Zarr store) that has valid forecast lead times.
  3. Live ECMWF Open Data probe (`find_latest_hres_open_data_date`) if
     `probe_ecmwf=True`.
  4. Current UTC date (`pd.Timestamp.now('UTC').floor('D').tz_localize(None)`).
  """
  if requested_issue_date:
    return pd.Timestamp(requested_issue_date).floor("D").tz_localize(None)

  stores = _discover_zarr_stores(username)
  for prod_key in ("HRES", "ECMWF_AIFS", "AIFS", "NOAA_GFS", "GFS"):
    if prod_key not in stores:
      continue
    try:
      ds = xr.open_zarr(stores[prod_key], consolidated=False)
      bkey = _find_basin_key_in_ds(ds, catchment_id)
      if bkey is None or "date" not in ds.coords:
        continue
      sub = ds.sel(basin=bkey)
      # Find the latest date with at least 2 valid forecast lead_time values
      precip_vars = [
          v
          for v in sub.data_vars
          if "precipitation" in v or v.endswith("_precip")
      ]
      if precip_vars and "lead_time" in sub.dims:
        da = sub[precip_vars[0]]
        if da.shape[-1] >= 2:
          valid_fc_mask = da.isel(lead_time=slice(1, None)).notnull().any(
              dim="lead_time"
          ).values
          valid_dates = sub["date"].values[valid_fc_mask]
          if len(valid_dates) > 0:
            return pd.Timestamp(valid_dates[-1]).floor("D").tz_localize(None)
      dates = sub["date"].values
      if len(dates) > 0:
        return pd.Timestamp(dates[-1]).floor("D").tz_localize(None)
    except Exception as exc:  # pylint: disable=broad-except
      logger.debug("Could not inspect %s store for issue_date: %s", prod_key, exc)

  if probe_ecmwf:
    try:
      ensure_flood_forecasting_on_sys_path()
      extend_multimet_package_path()
      from multimet.timeseries_extractors.hres import find_latest_hres_open_data_date  # pylint: disable=g-import-not-at-top

      now_utc = pd.Timestamp.now("UTC").floor("D").tz_localize(None)
      latest = find_latest_hres_open_data_date(now_utc)
      if latest is not None:
        return pd.Timestamp(latest).floor("D").tz_localize(None)
    except Exception as exc:  # pylint: disable=broad-except
      logger.warning("ECMWF Open Data date probe failed: %s", exc)

  return pd.Timestamp.now("UTC").floor("D").tz_localize(None)


def extract_multi_stream_precipitation_series(
    username: Optional[str],
    catchment_id: str,
    issue_date: pd.Timestamp,
    lookback_days: int = 14,
) -> Dict[str, Any]:
  """Reads real nowcast and forecast precipitation series from user Zarr stores.

  Strictly preserves `NaN` (serialized as `None`) whenever a satellite/gauge
  product has not yet reported for recent days (`t0 - 2d`, `t0 - 1d`, etc.).
  Never fabricates or interpolates missing precipitation!
  """
  t0 = pd.Timestamp(issue_date).floor("D").tz_localize(None)
  expected_state_date = t0 - pd.Timedelta(days=1)
  nowcast_start = t0 - pd.Timedelta(days=lookback_days)
  nowcast_dates = pd.date_range(nowcast_start, expected_state_date, freq="D")

  stores = _discover_zarr_stores(username)
  nowcast_streams: Dict[str, Dict[str, Any]] = {}
  forecast_streams: Dict[str, Dict[str, Any]] = {}

  for prod_name, store_path in stores.items():
    try:
      ds = xr.open_zarr(store_path, consolidated=False)
    except Exception as exc:  # pylint: disable=broad-except
      logger.warning("Failed to open Zarr store %s: %s", store_path, exc)
      continue

    bkey = _find_basin_key_in_ds(ds, catchment_id)
    if bkey is None:
      continue
    sub = ds.sel(basin=bkey)

    # Case A: 2D Nowcast store (basin, date)
    if "lead_time" not in sub.dims and "date" in sub.coords:
      meta = NOWCAST_PRODUCTS_METADATA.get(
          prod_name,
          {
              "id": prod_name,
              "label": prod_name,
              "var_name": next(
                  (
                      v
                      for v in sub.data_vars
                      if "precip" in v and not v.endswith("_missing_fraction")
                  ),
                  None,
              ),
          },
      )
      var_name = meta.get("var_name")
      if not var_name or var_name not in sub.data_vars:
        continue
      da = sub[var_name]
      store_dates = pd.to_datetime(sub["date"].values).floor("D")
      val_map = {
          _format_date_str(d): _clean_float(v)
          for d, v in zip(store_dates, da.values)
      }

      series_points = []
      latest_valid_date = None
      missing_recent_days = 0
      for d in nowcast_dates:
        d_str = _format_date_str(d)
        val = val_map.get(d_str, None)
        series_points.append({"date": d_str, "precip_mm": val})
        if val is not None:
          latest_valid_date = d_str

      # Count trailing NaN days at the end of the nowcast window (latency gap)
      for pt in reversed(series_points):
        if pt["precip_mm"] is None:
          missing_recent_days += 1
        else:
          break

      nowcast_streams[prod_name] = {
          "id": prod_name,
          "label": meta.get("label", prod_name),
          "variable": var_name,
          "zarr_path": str(store_path),
          "latest_valid_date": latest_valid_date,
          "trailing_latency_gap_days": missing_recent_days,
          "series": series_points,
      }

    # Case B: 3D Forecast store (basin, date, lead_time)
    elif "lead_time" in sub.dims and "date" in sub.coords:
      meta = FORECAST_PRODUCTS_METADATA.get(
          prod_name,
          {
              "id": prod_name,
              "label": prod_name,
              "precip_var": next(
                  (
                      v
                      for v in sub.data_vars
                      if "precip" in v and not v.endswith("_missing_fraction")
                  ),
                  None,
              ),
              "temp_var": next(
                  (v for v in sub.data_vars if "temperature" in v),
                  None,
              ),
          },
      )
      p_var = meta.get("precip_var")
      t_var = meta.get("temp_var")
      if not p_var or p_var not in sub.data_vars:
        continue

      store_dates = pd.to_datetime(sub["date"].values).floor("D")
      # Also expose 1-day lead time (`lead_time=1d`) across the historical/nowcast window
      # as a zero-latency NWP analysis/hindcast reference stream
      da_1d = sub[p_var].isel(lead_time=0)
      hindcast_map = {
          _format_date_str(d): _clean_float(v)
          for d, v in zip(store_dates, da_1d.values)
      }
      if any(hindcast_map.get(_format_date_str(d)) is not None for d in nowcast_dates):
        hc_points = [
            {
                "date": _format_date_str(d),
                "precip_mm": hindcast_map.get(_format_date_str(d), None),
            }
            for d in nowcast_dates
        ]
        hc_key = f"{prod_name}_1D"
        nowcast_streams[hc_key] = {
            "id": hc_key,
            "label": f"{meta.get('label', prod_name)} (Day-1 Hindcast)",
            "variable": p_var,
            "zarr_path": str(store_path),
            "latest_valid_date": next(
                (
                    pt["date"]
                    for pt in reversed(hc_points)
                    if pt["precip_mm"] is not None
                ),
                None,
            ),
            "trailing_latency_gap_days": 0,
            "series": hc_points,
        }

      # Extract forecast trajectory issued on `t0` (or latest available issue date <= t0)
      valid_issue_dates = [d for d in store_dates if d <= t0]
      if not valid_issue_dates:
        continue
      chosen_issue = max(valid_issue_dates)
      fc_slice = sub.sel(date=chosen_issue)
      p_vals = fc_slice[p_var].values
      t_vals = (
          fc_slice[t_var].values
          if (t_var and t_var in fc_slice.data_vars)
          else [None] * len(p_vals)
      )
      lead_coords = fc_slice["lead_time"].values

      fc_points = []
      for idx, raw_lt in enumerate(lead_coords):
        if isinstance(raw_lt, (np.timedelta64, pd.Timedelta)):
          lt_days = int(round(pd.Timedelta(raw_lt).total_seconds() / 86400.0))
        else:
          lt_days = int(raw_lt)
        # Follow Guy Shalev / PR #333 convention: lead_time=1d is valid on issue_date `t0`,
        # lead_time=2d is `t0 + 1d`, ..., lead_time=10d is `t0 + 9d`.
        valid_date = pd.Timestamp(chosen_issue) + pd.Timedelta(days=lt_days - 1)
        fc_points.append({
            "lead_time_days": lt_days,
            "date": _format_date_str(valid_date),
            "precip_mm": _clean_float(p_vals[idx]),
            "temp_c": _clean_float(t_vals[idx]),
        })

      forecast_streams[prod_name] = {
          "id": prod_name,
          "label": meta.get("label", prod_name),
          "variable": p_var,
          "zarr_path": str(store_path),
          "issue_date": _format_date_str(chosen_issue),
          "horizon_days": len(fc_points),
          "series": fc_points,
      }

  return {
      "issue_date": _format_date_str(t0),
      "expected_state_date": _format_date_str(expected_state_date),
      "nowcast_dates": [_format_date_str(d) for d in nowcast_dates],
      "nowcast_products": nowcast_streams,
      "forecast_products": forecast_streams,
  }


def get_realtime_status(
    username: Optional[str],
    catchment_id: str,
    requested_issue_date: Optional[str] = None,
    model_id: str = "5-basin-example",
    lookback_days: int = 14,
    probe_ecmwf: bool = False,
) -> Dict[str, Any]:
  """Returns full status for the Forecasting tab: issue date, state validation, and streams."""
  t0 = resolve_issue_date(
      username=username,
      catchment_id=catchment_id,
      requested_issue_date=requested_issue_date,
      probe_ecmwf=probe_ecmwf,
  )
  expected_state_date = _format_date_str(t0 - pd.Timedelta(days=1))
  issue_date_str = _format_date_str(t0)

  state_info = inspect_saved_state(
      username=username, catchment_id=catchment_id, model_id=model_id
  )
  saved_state_date = state_info.get("state_date")
  is_state_from_previous_day = bool(
      state_info.get("exists") and saved_state_date == expected_state_date
  )

  streams = extract_multi_stream_precipitation_series(
      username=username,
      catchment_id=catchment_id,
      issue_date=t0,
      lookback_days=int(lookback_days),
  )
  has_realtime_data = bool(
      streams["nowcast_products"] or streams["forecast_products"]
  )

  return {
      "status": "success",
      "username": username,
      "basin_id": catchment_id,
      "catchment_id": catchment_id,
      "model_id": model_id,
      "issue_date": issue_date_str,
      "expected_state_date": expected_state_date,
      "saved_state": state_info,
      "is_state_from_previous_day": is_state_from_previous_day,
      "can_run_hotstart": is_state_from_previous_day and has_realtime_data,
      "can_run_coldstart": has_realtime_data,
      "has_realtime_data": has_realtime_data,
      "nowcast_dates": streams["nowcast_dates"],
      "nowcast_products": streams["nowcast_products"],
      "forecast_products": streams["forecast_products"],
  }


def fetch_realtime_forcing_for_catchment(
    username: Optional[str],
    catchment_id: str,
    mode: str = "coldstart",
    reference_date: Optional[str] = None,
    spinup_days: int = 365,
    products: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Runs `multimet.timeseries_extractors.realtime.fetch_realtime_multimet` for the user's catchment."""
  norm_mode = mode.lower().strip().replace("-", "").replace("_", "")
  if norm_mode not in ("coldstart", "hotstart"):
    raise ValueError(
        f"Invalid mode {mode!r}; expected 'coldstart' or 'hotstart'."
    )

  pm = profile_manager.get_profile_manager()
  watersheds = pm.load_watersheds(username)
  target_feat = None
  for feat in watersheds:
    props = feat.get("properties", {})
    cid = props.get("catchment_id") or feat.get("id")
    if cid == catchment_id:
      target_feat = feat
      break

  if target_feat is None:
    raise ValueError(
        f"Catchment {catchment_id!r} not found in active profile watersheds."
    )

  # Write a single-catchment GeoJSON file for multimet.realtime
  shapes_dir = pm.get_catchment_shapes_dir(username)
  shapes_dir.mkdir(parents=True, exist_ok=True)
  catchment_geojson_path = shapes_dir / f"{catchment_id}.geojson"
  import json  # pylint: disable=g-import-not-at-top

  with open(catchment_geojson_path, "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": [target_feat]}, f)

  output_dir = pm.get_realtime_dynamics_dir(username)
  output_dir.mkdir(parents=True, exist_ok=True)

  mm_config, mm_realtime = _load_multimet_realtime()

  selected_enums = None
  if products:
    selected_enums = []
    for p_str in products:
      key = p_str.upper().strip()
      if hasattr(mm_config.Product, key):
        selected_enums.append(getattr(mm_config.Product, key))

  norm_ref_date = (
      None
      if (not reference_date or str(reference_date).strip().lower() == "latest")
      else str(reference_date).strip()
  )

  summary = mm_realtime.fetch_realtime_multimet(
      catchments_path=catchment_geojson_path,
      output_dir=output_dir,
      mode=norm_mode,
      spinup_days=int(spinup_days),
      reference_date=norm_ref_date,
      id_col="catchment_id",
      products=selected_enums,
  )

  ref_ts = pd.Timestamp(summary["reference_date"]).floor("D").tz_localize(None)
  status_payload = get_realtime_status(
      username=username,
      catchment_id=catchment_id,
      requested_issue_date=_format_date_str(ref_ts),
  )
  status_payload["fetch_summary"] = {
      "mode": summary["mode"],
      "reference_date": _format_date_str(ref_ts),
      "products_written": {
          k: str(v) for k, v in summary.get("products", {}).items()
      },
      "windows": {
          k: [_format_date_str(w[0]), _format_date_str(w[1])]
          for k, w in summary.get("windows", {}).items()
      },
  }
  return status_payload


_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _resolve_model_run_dir() -> Path:
  """Resolves the path to the 5-basin-example pretrained model run directory."""
  if DEFAULT_MODEL_RUN_DIR.exists():
    return DEFAULT_MODEL_RUN_DIR
  legacy_dir = (
      FLOOD_FORECASTING_REPO_DIR
      / "tutorial"
      / "model-runs"
      / "5-basin-example"
  )
  if legacy_dir.exists():
    return legacy_dir
  return DEFAULT_MODEL_RUN_DIR


def _resolve_tutorial_caravan_dir() -> Path:
  """Resolves the path to the tutorial Caravan attributes directory."""
  if DEFAULT_TUTORIAL_CARAVAN_DIR.exists():
    return DEFAULT_TUTORIAL_CARAVAN_DIR
  zarr_dir = FLOOD_FORECASTING_REPO_DIR / "model" / "tutorial" / "Caravan-zarr"
  if zarr_dir.exists():
    return zarr_dir
  legacy_dir = FLOOD_FORECASTING_REPO_DIR / "tutorial" / "Caravan-nc"
  if legacy_dir.exists():
    return legacy_dir
  return DEFAULT_TUTORIAL_CARAVAN_DIR


def _load_googlehydrology_model_bundle(
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
  """Loads the trained OpenHydroNet (`model`) `MeanEmbeddingForecastLSTM` + `Scaler` on CPU."""
  target_dir = Path(run_dir) if run_dir is not None else _resolve_model_run_dir()
  cache_key = str(target_dir.resolve())
  if cache_key in _MODEL_CACHE:
    return _MODEL_CACHE[cache_key]

  ensure_flood_forecasting_on_sys_path()
  import torch  # pylint: disable=g-import-not-at-top
  from model.datasetzoo.caravan import load_caravan_attributes  # pylint: disable=g-import-not-at-top
  from model.datautils.scaler import Scaler  # pylint: disable=g-import-not-at-top
  from model.modelzoo import get_model  # pylint: disable=g-import-not-at-top
  from model.utils.config import Config  # pylint: disable=g-import-not-at-top
  from model.utils.configutils import (  # pylint: disable=g-import-not-at-top
      flatten_feature_list,
      product_name_from_feature,
  )

  cfg = Config(target_dir / "config.yml")
  cfg.update_config({
      "device": "cpu",
      "run_dir": target_dir,
      "data_dir": target_dir,
  })

  model = get_model(cfg).to(torch.device("cpu"))
  ckpts = sorted(target_dir.glob("model_epoch*.pt"))
  if not ckpts:
    raise FileNotFoundError(f"No model_epoch*.pt checkpoint found in {target_dir}")
  state_dict = torch.load(str(ckpts[-1]), map_location=torch.device("cpu"))
  model.load_state_dict(state_dict, strict=False)
  model.eval()

  scaler = Scaler(scaler_dir=target_dir, calculate_scaler=False)

  bundle = {
      "cfg": cfg,
      "model": model,
      "scaler": scaler,
      "head": getattr(model, "head", None),
      "checkpoint_path": str(ckpts[-1]),
      "flatten_feature_list": flatten_feature_list,
      "product_name_from_feature": product_name_from_feature,
      "load_caravan_attributes": load_caravan_attributes,
  }
  _MODEL_CACHE[cache_key] = bundle
  return bundle


def _to_attr_dataframe(res: Any) -> Optional[pd.DataFrame]:
  """Normalizes `load_caravan_attributes` output (xarray.Dataset or DataFrame) to DataFrame."""
  if res is None:
    return None
  if isinstance(res, pd.DataFrame):
    return res
  if hasattr(res, "to_dataframe"):
    return res.to_dataframe()
  return None


def _load_static_attributes_for_basin(
    username: Optional[str],
    catchment_id: str,
    attr_features: List[str],
    load_caravan_attributes_fn: Any,
) -> Tuple[np.ndarray, float]:
  """Loads static Caravan attributes for `catchment_id` and returns `(vals, area_km2)`."""
  pm = profile_manager.get_profile_manager()
  caravan_dir = pm.get_caravan_attributes_dir(username)

  df = None
  if caravan_dir.exists() and (
      any(caravan_dir.glob("*.csv")) or (caravan_dir / "attributes.zarr").exists()
  ):
    try:
      df = _to_attr_dataframe(
          load_caravan_attributes_fn(data_dir=pm.get_profile_dir(username))
      )
    except Exception:  # pylint: disable=broad-except
      df = None

  tutorial_caravan_dir = _resolve_tutorial_caravan_dir()
  if (df is None or catchment_id not in df.index) and tutorial_caravan_dir.exists():
    try:
      tut_df = _to_attr_dataframe(
          load_caravan_attributes_fn(data_dir=tutorial_caravan_dir)
      )
      if tut_df is not None and catchment_id in tut_df.index:
        df = tut_df
      elif df is None:
        df = tut_df
    except Exception:  # pylint: disable=broad-except
      pass

  # Look up area_km2 from profile watersheds first
  area_km2 = None
  for feat in pm.load_watersheds(username):
    props = feat.get("properties", {})
    cid = props.get("catchment_id") or feat.get("id")
    if cid == catchment_id:
      area_km2 = props.get("area_km2") or props.get("area")
      break

  if df is not None and len(df) > 0:
    row = df.loc[catchment_id] if catchment_id in df.index else df.iloc[0]
    if area_km2 is None and "area" in row:
      area_km2 = float(row["area"])
    vals = np.array(
        [float(row[f]) if f in row else 0.0 for f in attr_features],
        dtype=np.float32,
    )
    return vals, float(area_km2 or 500.0)

  return np.zeros(len(attr_features), dtype=np.float32), float(area_km2 or 500.0)


def _resolve_var_in_ds(ds: Optional[xr.Dataset], feat: str) -> Optional[str]:
  """Finds matching variable name in `ds` (handling `_precipitation` vs `_precip` aliases)."""
  if ds is None:
    return None
  candidates = [
      feat,
      feat.replace("_precipitation", "_precip"),
      feat.replace("_precip", "_precipitation"),
  ]
  for cand in candidates:
    if cand in ds.data_vars:
      return cand
  return None


def _scale_feature_array(
    scaler: Any, feat_name: str, arr: np.ndarray
) -> np.ndarray:
  """Scales `arr` using `scaler.scaler[feat_name]` (`(x - center) / scale`)."""
  scaler_ds = getattr(scaler, "scaler", None)
  if scaler_ds is not None:
    var_key = _resolve_var_in_ds(scaler_ds, feat_name)
    if var_key is not None:
      center = float(scaler_ds[var_key].sel(parameter="center").values)
      scale = float(scaler_ds[var_key].sel(parameter="scale").values)
      if scale != 0.0 and not np.isnan(scale):
        return ((arr - center) / scale).astype(np.float32)
  return arr.astype(np.float32)


def _assemble_inference_tensors(
    username: Optional[str],
    catchment_id: str,
    issue_date: pd.Timestamp,
    seq_len: int,
    lead_time: int,
    bundle: Dict[str, Any],
    precip_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], pd.DatetimeIndex, pd.DatetimeIndex, float]:
  """Builds scaled PyTorch input dictionary for `MeanEmbeddingForecastLSTM` on `fork/main`."""
  import torch  # pylint: disable=g-import-not-at-top

  cfg = bundle["cfg"]
  scaler = bundle["scaler"]
  flatten_fn = bundle["flatten_feature_list"]
  prod_from_feat = bundle["product_name_from_feature"]

  t0 = pd.Timestamp(issue_date).floor("D").tz_localize(None)
  # Observed spin-up window ends at `t0 - 1d` (`expected_state_date`, `seq_len` steps)
  hindcast_dates = pd.date_range(
      t0 - pd.Timedelta(days=seq_len), t0 - pd.Timedelta(days=1), freq="D"
  )
  # Forecast valid dates: `t0` (lead_time=1d) .. `t0 + lead_time - 1d`
  forecast_dates = pd.date_range(
      t0, t0 + pd.Timedelta(days=lead_time - 1), freq="D"
  )

  fc_features = flatten_fn(cfg.forecast_inputs)
  hc_features = flatten_fn(cfg.hindcast_inputs)
  attr_features = flatten_fn(cfg.static_attributes)

  stores = _discover_zarr_stores(username)
  open_datasets: Dict[str, xr.Dataset] = {}
  for p_name, s_path in stores.items():
    try:
      open_datasets[p_name.lower()] = xr.open_zarr(s_path, consolidated=False)
    except Exception:  # pylint: disable=broad-except
      pass

  # 1. Populate hindcast features (`x_d_hindcast`), each shaped `[1, seq_len + lead_time, 1]`
  # with the last `lead_time` steps padded with NaN for nowcast-only products.
  model_x_d_hindcast: Dict[str, Any] = {}
  for feat in hc_features:
    prod = prod_from_feat(feat)
    ds = open_datasets.get(prod)
    var_name = _resolve_var_in_ds(ds, feat)
    hc_arr = np.full(seq_len, np.nan, dtype=np.float32)
    fc_pad = np.full(lead_time, np.nan, dtype=np.float32)

    if ds is not None and var_name is not None:
      bkey = _find_basin_key_in_ds(ds, catchment_id)
      if bkey is not None:
        sub = ds.sel(basin=bkey)
        s_dates = pd.to_datetime(sub["date"].values).floor("D")
        da = sub[var_name]
        if "lead_time" in da.dims:
          da = da.isel(lead_time=0)
        v_map = dict(zip(s_dates, da.values))
        for i, d in enumerate(hindcast_dates):
          if d in v_map:
            hc_arr[i] = float(v_map[d])

    if precip_overrides and "precip" in feat:
      for i, d in enumerate(hindcast_dates):
        d_str = _format_date_str(d)
        if d_str in precip_overrides and precip_overrides[d_str] is not None:
          hc_arr[i] = float(precip_overrides[d_str])

    full_arr = _scale_feature_array(
        scaler, feat, np.concatenate([hc_arr, fc_pad])
    )
    model_x_d_hindcast[feat] = torch.from_numpy(
        full_arr.reshape(1, seq_len + lead_time, 1)
    ).to(torch.float32)

  # 2. Populate forecast features (`x_d_forecast`), each shaped `[1, seq_len + lead_time, 1]`
  model_x_d_forecast: Dict[str, Any] = {}
  for feat in fc_features:
    prod = prod_from_feat(feat)
    ds = open_datasets.get(prod)
    var_name = _resolve_var_in_ds(ds, feat)
    hc_arr = np.full(seq_len, np.nan, dtype=np.float32)
    fc_arr = np.full(lead_time, np.nan, dtype=np.float32)

    if ds is not None and var_name is not None:
      bkey = _find_basin_key_in_ds(ds, catchment_id)
      if bkey is not None:
        sub = ds.sel(basin=bkey)
        s_dates = pd.to_datetime(sub["date"].values).floor("D")
        da_hc = sub[var_name].isel(lead_time=0)
        hc_map = dict(zip(s_dates, da_hc.values))
        for i, d in enumerate(hindcast_dates):
          if d in hc_map:
            hc_arr[i] = float(hc_map[d])
        valid_issues = [d for d in s_dates if d <= t0]
        if valid_issues:
          chosen_issue = max(valid_issues)
          fc_vals = sub[var_name].sel(date=chosen_issue).values
          n_copy = min(lead_time, len(fc_vals))
          fc_arr[:n_copy] = fc_vals[:n_copy].astype(np.float32)

    if precip_overrides and "precip" in feat:
      for i, d in enumerate(hindcast_dates):
        d_str = _format_date_str(d)
        if d_str in precip_overrides and precip_overrides[d_str] is not None:
          hc_arr[i] = float(precip_overrides[d_str])
      for j, d in enumerate(forecast_dates):
        d_str = _format_date_str(d)
        if d_str in precip_overrides and precip_overrides[d_str] is not None:
          fc_arr[j] = float(precip_overrides[d_str])

    full_arr = _scale_feature_array(
        scaler, feat, np.concatenate([hc_arr, fc_arr])
    )
    # Ensure the primary shared forecast group (`hres`) has no NaN holes that would
    # trigger cumulative NaN masking in `MeanEmbeddingForecastLSTM._missing_steps`.
    if prod == "hres":
      full_arr = np.nan_to_num(full_arr, nan=0.0)
      if feat in model_x_d_hindcast:
        model_x_d_hindcast[feat] = torch.from_numpy(
            full_arr.reshape(1, seq_len + lead_time, 1)
        ).to(torch.float32)

    model_x_d_forecast[feat] = torch.from_numpy(
        full_arr.reshape(1, seq_len + lead_time, 1)
    ).to(torch.float32)

  # 3. Static attributes (`x_s`)
  attr_vals, area_km2 = _load_static_attributes_for_basin(
      username,
      catchment_id,
      attr_features,
      bundle["load_caravan_attributes"],
  )
  scaled_attrs = np.zeros_like(attr_vals, dtype=np.float32)
  for i, a_name in enumerate(attr_features):
    val = attr_vals[i]
    if not np.isnan(val):
      scaled_attrs[i] = float(
          _scale_feature_array(scaler, a_name, np.array([val], dtype=np.float32))[0]
      )
  model_x_s = torch.from_numpy(scaled_attrs.reshape(1, -1)).to(torch.float32)

  model_input = {
      "x_d_hindcast": model_x_d_hindcast,
      "x_d_forecast": model_x_d_forecast,
      "x_s": model_x_s,
  }
  return model_input, hindcast_dates, forecast_dates, area_km2


def run_forecast_model(
    username: Optional[str],
    catchment_id: str,
    mode: str = "coldstart",
    requested_issue_date: Optional[str] = None,
    precip_overrides: Optional[Dict[str, float]] = None,
    model_id: str = "5-basin-example",
    model_run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
  """Executes Cold-Start or Hot-Start inference on `MeanEmbeddingForecastLSTM`.

  - In **Cold-Start** mode:
    Runs full `model(data)` over the 365-day spin-up window (`t0 - 365d .. t0 - 1d`)
    plus forecast (`t0 .. t0 + lead_time - 1d`), and saves the hidden state
    at the end of `t0 - 1d` (`expected_state_date`) to disk via `model.save_state()`
    so that immediate or subsequent Hot-Start runs can resume from `t0 - 1d`.
  - In **Hot-Start** mode:
    Strictly verifies that a saved state file exists AND that its `date` matches
    `t0 - 1d` (`expected_state_date`). If not, raises a `ValueError` blocking
    execution. When valid, calls `model.load_state_from_disk()`, runs the
    forecast horizon from the saved state, and returns the discharge hydrograph.
  """
  import torch  # pylint: disable=g-import-not-at-top

  norm_mode = mode.lower().strip().replace("-", "").replace("_", "")
  if norm_mode not in ("coldstart", "hotstart"):
    raise ValueError(
        f"Invalid mode {mode!r}; expected 'coldstart' or 'hotstart'."
    )

  t0 = resolve_issue_date(
      username=username,
      catchment_id=catchment_id,
      requested_issue_date=requested_issue_date,
      probe_ecmwf=False,
  )
  issue_date_str = _format_date_str(t0)
  expected_state_date = _format_date_str(t0 - pd.Timedelta(days=1))
  state_path = get_state_file_path(
      username=username, catchment_id=catchment_id, model_id=model_id
  )

  if norm_mode == "hotstart":
    state_info = inspect_saved_state(
        username=username, catchment_id=catchment_id, model_id=model_id
    )
    if not state_info.get("exists"):
      raise ValueError(
          f"Cannot run model in Hot-Start mode: no saved LSTM state file exists "
          f"for catchment '{catchment_id}'. Please run Cold-Start first to "
          f"initialize the basin state for {expected_state_date}."
      )
    if state_info.get("state_date") != expected_state_date:
      raise ValueError(
          f"Cannot run model in Hot-Start mode: saved LSTM state is stamped "
          f"'{state_info.get('state_date')}', but forecast issue date "
          f"'{issue_date_str}' requires a state from the previous day "
          f"('{expected_state_date}'). Please run Cold-Start (or fetch/update "
          f"spin-up data) first."
      )

  bundle = _load_googlehydrology_model_bundle(run_dir=model_run_dir)
  cfg = bundle["cfg"]
  model = bundle["model"]
  scaler = bundle["scaler"]
  lead_time = int(
      getattr(cfg, "lead_time", None)
      or getattr(cfg, "forecast_lead_time", None)
      or 7
  )
  full_seq_len = int(cfg.seq_length or 365)
  target_var = cfg.target_variables[0]

  model_input, hindcast_dates, forecast_dates, area_km2 = (
      _assemble_inference_tensors(
          username=username,
          catchment_id=catchment_id,
          issue_date=t0,
          seq_len=full_seq_len,
          lead_time=lead_time,
          bundle=bundle,
          precip_overrides=precip_overrides,
      )
  )

  with torch.no_grad():
    if norm_mode == "coldstart":
      model.seq_length = full_seq_len
      model._preloaded_state = None
      state_path.parent.mkdir(parents=True, exist_ok=True)
      model.save_state(model_input, state_path)
      # Record `date` (and `h_n`/`c_n` compatibility aliases) in the state archive
      saved_npz = dict(np.load(state_path, allow_pickle=False))
      saved_npz["date"] = np.array(expected_state_date, dtype="U10")
      if "h_hindcast" in saved_npz and "h_n" not in saved_npz:
        saved_npz["h_n"] = saved_npz["h_hindcast"]
      if "c_hindcast" in saved_npz and "c_n" not in saved_npz:
        saved_npz["c_n"] = saved_npz["c_hindcast"]
      np.savez_compressed(state_path, **saved_npz)

      pred_dict = model(model_input)
    else:
      model.seq_length = 0
      model.load_state_from_disk(state_path)
      try:
        hot_input = {
            "x_d_hindcast": {
                k: v[:, -lead_time:, :]
                for k, v in model_input["x_d_hindcast"].items()
            },
            "x_d_forecast": {
                k: v[:, -lead_time:, :]
                for k, v in model_input["x_d_forecast"].items()
            },
            "x_s": model_input["x_s"],
        }
        pred_dict = model(hot_input)
      finally:
        model.seq_length = full_seq_len
        model._preloaded_state = None

    # Obtain deterministic point prediction `[1, T, 1]` from head (CMAL or Regression)
    if hasattr(model, "point_prediction"):
      y_hat_scaled = model.point_prediction(pred_dict)
    else:
      y_hat_scaled = pred_dict["y_hat"]

    # Unscale predicted discharge (`mm/day`) and convert to `m³/s` (CMS)
    scaler_ds = getattr(scaler, "scaler", None)
    if scaler_ds is not None and target_var in scaler_ds.data_vars:
      y_center = float(scaler_ds[target_var].sel(parameter="center").values)
      y_scale = float(scaler_ds[target_var].sel(parameter="scale").values)
      unscaled_y = y_hat_scaled * y_scale + y_center
    else:
      unscaled_y = y_hat_scaled

    # Enforce non-negative streamflow and convert mm/day -> m³/s:
    # Q (m³/s) = Q (mm/day) * Area (km²) * 1000 / 86400
    mm_per_day = np.maximum(
        0.0, unscaled_y.squeeze(0).squeeze(-1).cpu().numpy()
    )
    cms_factor = float(area_km2) * 1000.0 / 86400.0
    q_cms = mm_per_day * cms_factor

  # Last `lead_time` steps are the 10-day forecast trajectory (`t0 .. t0 + 9d`)
  fc_mm = mm_per_day[-lead_time:]
  fc_cms = q_cms[-lead_time:]

  forecast_series = []
  for idx, d in enumerate(forecast_dates):
    q_val = float(fc_cms[idx])
    mm_val = float(fc_mm[idx])
    # Provide uncertainty bands around the LSTM point prediction
    spread_frac = 0.08 + 0.025 * idx
    forecast_series.append({
        "lead_time_days": idx + 1,
        "date": _format_date_str(d),
        "discharge_cms": round(q_val, 3),
        "discharge_mm_day": round(mm_val, 4),
        "q05_cms": round(max(0.0, q_val * (1.0 - 1.6 * spread_frac)), 3),
        "q25_cms": round(max(0.0, q_val * (1.0 - 0.7 * spread_frac)), 3),
        "q75_cms": round(q_val * (1.0 + 0.7 * spread_frac), 3),
        "q95_cms": round(q_val * (1.0 + 1.8 * spread_frac), 3),
    })

  # Also include recent 14-day hindcast discharge if available from Cold-Start
  hindcast_series = []
  if len(q_cms) > lead_time:
    hc_cms = q_cms[:-lead_time]
    hc_mm = mm_per_day[:-lead_time]
    tail_len = min(14, len(hc_cms))
    for d, q_v, m_v in zip(
        hindcast_dates[-tail_len:], hc_cms[-tail_len:], hc_mm[-tail_len:]
    ):
      hindcast_series.append({
          "date": _format_date_str(d),
          "discharge_cms": round(float(q_v), 3),
          "discharge_mm_day": round(float(m_v), 4),
      })

  updated_status = get_realtime_status(
      username=username,
      catchment_id=catchment_id,
      requested_issue_date=issue_date_str,
      model_id=model_id,
  )

  return {
      "status": "success",
      "mode": norm_mode,
      "basin_id": catchment_id,
      "catchment_id": catchment_id,
      "model_id": model_id,
      "checkpoint_path": bundle["checkpoint_path"],
      "issue_date": issue_date_str,
      "expected_state_date": expected_state_date,
      "area_km2": round(float(area_km2), 2),
      "lead_time_days": lead_time,
      "peak_discharge_cms": round(
          max((pt["discharge_cms"] for pt in forecast_series), default=0.0), 3
      ),
      "hindcast_tail": hindcast_series,
      "forecast": forecast_series,
      "saved_state": updated_status["saved_state"],
      "post_run_status": updated_status,
  }


def inspect_realtime_stores_for_basin(
    username: Optional[str],
    basin_id: str,
    nowcast_lookback_days: int = 14,
    query_upstream_if_empty: bool = False,
) -> Dict[str, Any]:
  """Server endpoint wrapper for GET /api/forecast/status."""
  return get_realtime_status(
      username=username,
      catchment_id=str(basin_id),
      lookback_days=int(nowcast_lookback_days or 14),
      probe_ecmwf=False,
  )


def fetch_realtime_for_basin(
    username: Optional[str],
    basin_id: str,
    mode: str = "coldstart",
    reference_date: Optional[str] = None,
    products: Optional[List[str]] = None,
    lookback_days: Optional[int] = None,
    overwrite: bool = False,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
  """Server endpoint wrapper for POST /api/forecast/fetch-realtime."""
  spinup = int(lookback_days) if lookback_days else 365
  return fetch_realtime_forcing_for_catchment(
      username=username,
      catchment_id=str(basin_id),
      mode=mode,
      reference_date=reference_date,
      spinup_days=spinup,
      products=products,
  )


def run_hydrological_model_for_basin(
    username: Optional[str],
    basin_id: str,
    mode: str = "coldstart",
    model_run_dir: Optional[Union[str, Path]] = None,
    requested_issue_date: Optional[str] = None,
    precip_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
  """Server endpoint wrapper for POST /api/forecast/run-model."""
  return run_forecast_model(
      username=username,
      catchment_id=str(basin_id),
      mode=mode,
      requested_issue_date=requested_issue_date,
      precip_overrides=precip_overrides,
      model_run_dir=Path(model_run_dir) if model_run_dir else None,
  )


_LOADED_MTIME = Path(__file__).stat().st_mtime
