"""Streamflow CSV Ingestion & Zarr Persistence Engine for Earthkit Hydro Web.

Handles user-uploaded streamflow CSVs for both:
1. Historical Target Time Series (`targets/streamflow.zarr`) + Gumbel/empirical
   return-period flood threshold calculation (`targets/return_periods.json`).
2. Real-Time Data Assimilation Observations (`assimilation/streamflow_realtime.zarr`).

Normalizes physical units between volumetric discharge (m^3/s) and specific
runoff (mm/day) using the catchment drainage area:
    streamflow_mm_day = (Q_cms * 86.4) / Area_km2
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

try:
  from frontend.profile_manager import (
      ProfileManager,
      _extract_basin_id,
      get_profile_manager,
  )
except ImportError:
  try:
    from frontend.profile_manager import (
        ProfileManager,
        _extract_basin_id,
        get_profile_manager,
    )
  except ImportError:
    from profile_manager import (
        ProfileManager,
        _extract_basin_id,
        get_profile_manager,
    )

logger = logging.getLogger(__name__)

# Physical conversion factor: (86,400 s/day * 1,000 mm/m) / (1,000,000 m^2/km^2) = 86.4
CMS_TO_MM_DAY_FACTOR = 86.4

# Euler-Mascheroni constant for Gumbel Type-I method-of-moments fitting
EULER_MASCHERONI = 0.5772156649015329

DATE_COLUMN_CANDIDATES = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "valid_time",
    "obs_time",
    "day",
)

VALUE_COLUMN_CANDIDATES = (
    "streamflow",
    "discharge",
    "discharge_cms",
    "q",
    "q_cms",
    "streamflow_mm",
    "streamflow_mm_day",
    "mm_day",
    "mm/day",
    "mm/d",
    "cms",
    "m3/s",
    "m3s",
    "runoff",
    "flow",
    "value",
)

SENTINEL_VALUES = (-999.0, -9999.0, -99999.0, -99.9, -9999.9)

RETURN_PERIOD_YEARS = (2, 5, 10, 20, 50, 100)


def _read_raw_csv_text(csv_data: Union[str, bytes, Path]) -> str:
  """Resolves CSV input (str content, file Path, or raw bytes) into UTF-8 text."""
  if isinstance(csv_data, bytes):
    return csv_data.decode("utf-8-sig", errors="replace")
  if isinstance(csv_data, Path):
    return csv_data.read_text(encoding="utf-8-sig")
  if isinstance(csv_data, str):
    # Check if string is a path to an existing file on disk
    if "\n" not in csv_data and "\r" not in csv_data and len(csv_data) < 1024:
      candidate = Path(csv_data)
      try:
        if candidate.is_file():
          return candidate.read_text(encoding="utf-8-sig")
      except OSError:
        pass
    return csv_data
  raise TypeError(f"Unsupported csv_data type: {type(csv_data)}")


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
  """Auto-detects the date/timestamp column and streamflow value column."""
  if df.empty and len(df.columns) == 0:
    raise ValueError("Uploaded CSV has no columns.")
  cols = list(df.columns)
  lower_map = {str(c).strip().lower(): c for c in cols}

  date_col = None
  for cand in DATE_COLUMN_CANDIDATES:
    if cand in lower_map:
      date_col = lower_map[cand]
      break
  if date_col is None:
    for c in cols:
      c_low = str(c).strip().lower()
      if any(k in c_low for k in ("date", "time", "timestamp")):
        date_col = c
        break
  if date_col is None:
    date_col = cols[0]

  val_col = None
  for cand in VALUE_COLUMN_CANDIDATES:
    if cand in lower_map and lower_map[cand] != date_col:
      val_col = lower_map[cand]
      break
  if val_col is None:
    for c in cols:
      if c == date_col:
        continue
      c_low = str(c).strip().lower()
      if any(
          k in c_low
          for k in ("streamflow", "discharge", "flow", "runoff", "cms", "mm", "q", "val")
      ):
        val_col = c
        break
  if val_col is None:
    remaining = [c for c in cols if c != date_col]
    if not remaining:
      raise ValueError(
          f"CSV must contain at least a date column and a value column; found {cols}."
      )
    val_col = remaining[0]

  return date_col, val_col


def _normalize_input_units(input_units: Optional[str], val_col_name: str) -> str:
  """Resolves input_units ('auto', 'cms'/'m3/s', 'mm/day'/'mm/d') to canonical 'm3/s' or 'mm/day'."""
  u = (input_units or "auto").strip().lower()
  if u in ("cms", "m3/s", "m^3/s", "m3s", "cubic_meters_per_second", "discharge_cms"):
    return "m3/s"
  if u in ("mm/day", "mm/d", "mm_day", "mm", "streamflow_mm", "specific_runoff"):
    return "mm/day"
  if u != "auto":
    raise ValueError(
        f"Unsupported input_units '{input_units}'. Expected 'auto', 'm3/s'/'cms', or 'mm/day'/'mm/d'."
    )

  col_low = str(val_col_name).strip().lower()
  if any(
      token in col_low
      for token in ("streamflow_mm", "mm_day", "mm/day", "mm/d", "q_mm", "runoff_mm", "_mm")
  ) or col_low in ("mm", "mm_day", "mm/day", "mm/d"):
    return "mm/day"
  return "m3/s"


def resolve_basin_area_km2(
    basin_id: str,
    area_km2: Optional[float] = None,
    username: str = "guest",
    profile_manager: Optional[ProfileManager] = None,
) -> float:
  """Resolves catchment area (km^2) explicitly or from the user's saved watersheds in ProfileManager."""
  if area_km2 is not None and float(area_km2) > 0:
    return float(area_km2)

  pm = profile_manager or get_profile_manager()
  watersheds = pm.load_watersheds(username)
  target_id = str(basin_id).strip()

  for idx, feat in enumerate(watersheds):
    if not isinstance(feat, dict):
      continue
    fid = _extract_basin_id(feat, idx)
    props = feat.get("properties") if isinstance(feat.get("properties"), dict) else {}
    alt_ids = {
        fid,
        str(feat.get("id") or "").strip(),
        str(props.get("id") or "").strip(),
        str(props.get("basin_id") or "").strip(),
        str(props.get("catchment_id") or "").strip(),
        str(props.get("HYBAS_ID") or "").strip(),
        str(props.get("gauge_id") or "").strip(),
    }
    if target_id in alt_ids:
      for area_key in (
          "area_km2",
          "SUB_AREA",
          "UP_AREA",
          "area_sqkm",
          "area",
          "AREA_KM2",
          "sub_area",
          "up_area",
          "catchment_area_km2",
      ):
        val = props.get(area_key)
        if val is None:
          val = feat.get(area_key)
        if val is not None:
          try:
            fval = float(val)
            if fval > 0:
              return fval
          except (TypeError, ValueError):
            pass

      # Geodesic fallback if polygon geometry is present
      geom_dict = feat.get("geometry")
      if isinstance(geom_dict, dict):
        try:
          import pyproj
          import shapely.geometry

          geom_obj = shapely.geometry.shape(geom_dict)
          geod = pyproj.Geod(ellps="WGS84")
          area_m2, _ = geod.geometry_area_perimeter(geom_obj)
          calc_km2 = abs(area_m2) / 1e6
          if calc_km2 > 0:
            return float(calc_km2)
        except Exception:
          pass

  raise ValueError(
      f"Catchment area (area_km2) for basin '{basin_id}' was not provided and could not "
      f"be resolved from saved watersheds for user '{username}'."
  )


def parse_streamflow_csv(
    csv_data: Union[str, bytes, Path],
    basin_id: str,
    area_km2: float,
    input_units: str = "auto",
) -> Dict[str, Any]:
  """Parses a streamflow CSV, cleans sentinels, aggregates sub-daily to daily UTC, and normalizes units.

  Returns a dictionary with:
    - 'daily_df': DataFrame indexed by daily UTC `date` (`datetime64[ns]`) with float32 columns
      `streamflow` (mm/day) and `discharge_cms` (m3/s).
    - 'raw_df': DataFrame with parsed raw UTC `timestamp` (`datetime64[ns]`), `streamflow`, and `discharge_cms`.
    - 'detected_units': 'm3/s' or 'mm/day'
    - 'date_col': str
    - 'value_col': str
    - 'is_subdaily': bool
  """
  if area_km2 is None or float(area_km2) <= 0:
    raise ValueError(f"area_km2 must be positive, got {area_km2}")
  area_val = float(area_km2)

  raw_text = _read_raw_csv_text(csv_data)
  if not raw_text or not raw_text.strip():
    raise ValueError("Uploaded CSV content is empty.")

  df = pd.read_csv(io.StringIO(raw_text))
  if df.empty:
    raise ValueError("Uploaded CSV contains no data rows.")

  date_col, val_col = _detect_columns(df)
  detected_units = _normalize_input_units(input_units, val_col)

  # Parse timestamps to UTC and strip timezone to standard UTC datetime64[ns]
  parsed_ts = pd.to_datetime(df[date_col], utc=True, errors="coerce")
  valid_ts_mask = parsed_ts.notna()
  if not valid_ts_mask.any():
    raise ValueError(f"Could not parse any valid dates from column '{date_col}'.")

  ts_utc_naive = (
      parsed_ts[valid_ts_mask]
      .dt.tz_convert("UTC")
      .dt.tz_localize(None)
      .astype("datetime64[ns]")
  )

  raw_vals = pd.to_numeric(df.loc[valid_ts_mask, val_col], errors="coerce").to_numpy(
      dtype=np.float64
  )

  # Convert negative values and explicit sentinel codes (e.g. -999, -9999) to NaN BEFORE aggregation
  invalid_mask = (~np.isfinite(raw_vals)) | (raw_vals < 0.0)
  for sentinel in SENTINEL_VALUES:
    invalid_mask |= np.isclose(raw_vals, sentinel, atol=1e-4)
  clean_vals = np.where(invalid_mask, np.nan, raw_vals)

  work_df = pd.DataFrame(
      {"timestamp": ts_utc_naive.to_numpy(), "value": clean_vals}
  )
  # Sort chronologically and deduplicate exact timestamps (taking mean if duplicate non-NaN)
  work_df = (
      work_df.sort_values("timestamp", kind="mergesort")
      .groupby("timestamp", as_index=False)["value"]
      .mean()
  )

  work_df["date"] = work_df["timestamp"].dt.floor("D").astype("datetime64[ns]")
  has_subdaily_times = bool(
      (work_df["timestamp"] != work_df["date"]).any()
      or (work_df["date"].duplicated().any())
  )

  # Physical unit conversion on raw series
  if detected_units == "m3/s":
    raw_cms = work_df["value"].to_numpy(dtype=np.float64)
    raw_mm = (raw_cms * CMS_TO_MM_DAY_FACTOR) / area_val
  else:
    raw_mm = work_df["value"].to_numpy(dtype=np.float64)
    raw_cms = (raw_mm * area_val) / CMS_TO_MM_DAY_FACTOR

  raw_df = pd.DataFrame(
      {
          "timestamp": work_df["timestamp"].astype("datetime64[ns]"),
          "date": work_df["date"].astype("datetime64[ns]"),
          "streamflow": raw_mm.astype(np.float32),
          "discharge_cms": raw_cms.astype(np.float32),
      }
  )

  # Aggregate to daily mean (preserving all dates, even those whose observations were all NaN)
  daily_grouped = (
       work_df.groupby("date", sort=True, as_index=True)["value"].mean()
  )
  daily_vals = daily_grouped.to_numpy(dtype=np.float64)

  if detected_units == "m3/s":
    daily_cms = daily_vals
    daily_mm = (daily_cms * CMS_TO_MM_DAY_FACTOR) / area_val
  else:
    daily_mm = daily_vals
    daily_cms = (daily_mm * area_val) / CMS_TO_MM_DAY_FACTOR

  daily_index = pd.DatetimeIndex(daily_grouped.index, name="date").astype(
      "datetime64[ns]"
  )
  daily_df = pd.DataFrame(
      {
          "streamflow": daily_mm.astype(np.float32),
          "discharge_cms": daily_cms.astype(np.float32),
      },
      index=daily_index,
  )

  return {
      "basin_id": str(basin_id).strip(),
      "area_km2": area_val,
      "detected_units": detected_units,
      "date_col": str(date_col),
      "value_col": str(val_col),
      "is_subdaily": has_subdaily_times,
      "daily_df": daily_df,
      "raw_df": raw_df,
  }


def compute_return_periods(
    series_or_df: Union[
        pd.DataFrame, pd.Series, xr.Dataset, np.ndarray, Sequence[float]
    ],
    dates: Optional[Union[pd.DatetimeIndex, Sequence[Any]]] = None,
    area_km2: Optional[float] = None,
    input_units: str = "cms",
    basin_id: Optional[str] = None,
    return_periods: Sequence[int] = RETURN_PERIOD_YEARS,
) -> Dict[str, Any]:
  """Computes flood return period thresholds (Q2, Q5, Q10, Q20, Q50, Q100) in m3/s and mm/day.

  Uses Gumbel Type-I extreme value fitting on annual maxima when >= 3 years of valid
  data are present, falling back to empirical quantile estimation with Gumbel tail
  monotonicity when < 3 years of data are available.
  """
  cms_series: Optional[np.ndarray] = None
  mm_series: Optional[np.ndarray] = None
  dt_index: Optional[pd.DatetimeIndex] = None

  if isinstance(series_or_df, xr.Dataset):
    ds = series_or_df
    if "basin" in ds.dims and ds.sizes["basin"] == 1:
      ds = ds.isel(basin=0)
    elif basin_id and "basin" in ds.coords:
      ds = ds.sel(basin=str(basin_id))
    if "date" in ds.coords:
      dt_index = pd.DatetimeIndex(pd.to_datetime(ds.coords["date"].values))
    if "discharge_cms" in ds.data_vars:
      cms_series = np.asarray(ds["discharge_cms"].values, dtype=np.float64)
    if "streamflow" in ds.data_vars:
      mm_series = np.asarray(ds["streamflow"].values, dtype=np.float64)
  elif isinstance(series_or_df, pd.DataFrame):
    df = series_or_df
    if isinstance(df.index, pd.DatetimeIndex):
      dt_index = df.index
    elif "date" in df.columns:
      dt_index = pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce"))
    if "discharge_cms" in df.columns:
      cms_series = pd.to_numeric(df["discharge_cms"], errors="coerce").to_numpy(
          dtype=np.float64
      )
    if "streamflow" in df.columns:
      mm_series = pd.to_numeric(df["streamflow"], errors="coerce").to_numpy(
          dtype=np.float64
      )
    if cms_series is None and mm_series is None:
      _, vcol = _detect_columns(df.reset_index())
      vals = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=np.float64)
      u = _normalize_input_units(input_units, vcol)
      if u == "m3/s":
        cms_series = vals
      else:
        mm_series = vals
  elif isinstance(series_or_df, pd.Series):
    if isinstance(series_or_df.index, pd.DatetimeIndex):
      dt_index = series_or_df.index
    vals = pd.to_numeric(series_or_df, errors="coerce").to_numpy(dtype=np.float64)
    u = _normalize_input_units(input_units, str(series_or_df.name or ""))
    if u == "m3/s":
      cms_series = vals
    else:
      mm_series = vals
  else:
    vals = np.asarray(series_or_df, dtype=np.float64)
    u = _normalize_input_units(input_units, "")
    if u == "m3/s":
      cms_series = vals
    else:
      mm_series = vals

  if dates is not None and dt_index is None:
    dt_index = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce"))

  if cms_series is None and mm_series is not None:
    if area_km2 is not None and float(area_km2) > 0:
      cms_series = (mm_series * float(area_km2)) / CMS_TO_MM_DAY_FACTOR
    else:
      cms_series = mm_series.copy()
  elif mm_series is None and cms_series is not None:
    if area_km2 is not None and float(area_km2) > 0:
      mm_series = (cms_series * CMS_TO_MM_DAY_FACTOR) / float(area_km2)
    else:
      mm_series = cms_series.copy()

  assert cms_series is not None and mm_series is not None

  valid_mask = np.isfinite(cms_series) & (cms_series >= 0.0)
  valid_cms = cms_series[valid_mask]
  valid_mm = mm_series[valid_mask]
  n_obs = int(valid_cms.size)

  if n_obs == 0:
    raise ValueError("Cannot compute return periods: no valid non-negative observations.")

  annual_max_cms: np.ndarray = np.array([], dtype=np.float64)
  if dt_index is not None and len(dt_index) == len(cms_series):
    valid_dates = dt_index[valid_mask]
    s_cms = pd.Series(valid_cms, index=valid_dates)
    annual_max_cms = s_cms.groupby(s_cms.index.year).max().dropna().to_numpy(dtype=np.float64)

  n_years = int(annual_max_cms.size)
  use_gumbel = n_years >= 3 and float(np.std(annual_max_cms, ddof=1)) > 0.0

  cms_thresholds: Dict[str, float] = {}
  mm_thresholds: Dict[str, float] = {}
  cms_by_period_str: Dict[str, float] = {}
  mm_by_period_str: Dict[str, float] = {}

  if use_gumbel:
    method = "gumbel"
    mean_am = float(np.mean(annual_max_cms))
    std_am = float(np.std(annual_max_cms, ddof=1))
    beta = (math.sqrt(6.0) / math.pi) * std_am
    mu = mean_am - EULER_MASCHERONI * beta

    for rp in return_periods:
      y_t = -math.log(-math.log(1.0 - 1.0 / float(rp)))
      q_cms = max(0.0, mu + beta * y_t)
      if area_km2 is not None and float(area_km2) > 0:
        q_mm = (q_cms * CMS_TO_MM_DAY_FACTOR) / float(area_km2)
      else:
        # Scale proportionally if area_km2 was not supplied
        ratio = float(np.mean(valid_mm) / np.mean(valid_cms)) if np.mean(valid_cms) > 0 else 1.0
        q_mm = q_cms * ratio
      cms_thresholds[f"Q{rp}"] = round(float(q_cms), 4)
      mm_thresholds[f"Q{rp}"] = round(float(q_mm), 4)
      cms_by_period_str[str(rp)] = round(float(q_cms), 4)
      mm_by_period_str[str(rp)] = round(float(q_mm), 4)
  else:
    method = "empirical"
    std_all = float(np.std(valid_cms)) if n_obs > 1 else 0.0
    tail_scale = max(std_all * 0.05, float(np.max(valid_cms)) * 0.01, 1e-4)
    prev_q_cms = -1.0
    for rp in return_periods:
      p = 100.0 * (1.0 - 1.0 / float(rp))
      q_cms = float(np.percentile(valid_cms, p))
      if q_cms <= prev_q_cms:
        y_t = -math.log(-math.log(1.0 - 1.0 / float(rp)))
        q_cms = prev_q_cms + tail_scale * max(0.1, y_t * 0.1)
      prev_q_cms = q_cms
      if area_km2 is not None and float(area_km2) > 0:
        q_mm = (q_cms * CMS_TO_MM_DAY_FACTOR) / float(area_km2)
      else:
        ratio = float(np.mean(valid_mm) / np.mean(valid_cms)) if np.mean(valid_cms) > 0 else 1.0
        q_mm = q_cms * ratio
      cms_thresholds[f"Q{rp}"] = round(float(q_cms), 4)
      mm_thresholds[f"Q{rp}"] = round(float(q_mm), 4)
      cms_by_period_str[str(rp)] = round(float(q_cms), 4)
      mm_by_period_str[str(rp)] = round(float(q_mm), 4)

  result: Dict[str, Any] = {
      "basin_id": str(basin_id) if basin_id is not None else None,
      "method": method,
      "n_years": n_years,
      "n_observations": n_obs,
      "area_km2": float(area_km2) if area_km2 is not None else None,
      "updated_at": datetime.now(timezone.utc).isoformat(),
      "cms": {**cms_thresholds, **cms_by_period_str},
      "m3/s": {**cms_thresholds, **cms_by_period_str},
      "discharge_cms": {**cms_thresholds, **cms_by_period_str},
      "mm_day": {**mm_thresholds, **mm_by_period_str},
      "mm/day": {**mm_thresholds, **mm_by_period_str},
      "streamflow_mm": {**mm_thresholds, **mm_by_period_str},
      "thresholds_cms": cms_thresholds,
      "thresholds_mm_day": mm_thresholds,
  }
  for rp in return_periods:
    k = f"Q{rp}"
    result[k] = cms_thresholds[k]
    result[f"{k}_cms"] = cms_thresholds[k]
    result[f"{k}_mm_day"] = mm_thresholds[k]
    result[f"return_period_{rp}yr"] = cms_thresholds[k]

  return result


def _build_single_basin_dataset(
    basin_id: str,
    daily_df: pd.DataFrame,
    area_km2: float,
    mode: str = "historical",
) -> xr.Dataset:
  """Constructs a googlehydrology-compliant (basin, date) xarray.Dataset with float32 variables."""
  b_id = str(basin_id).strip()
  dates = pd.DatetimeIndex(daily_df.index).astype("datetime64[ns]")
  streamflow_arr = daily_df["streamflow"].to_numpy(dtype=np.float32)[np.newaxis, :]
  discharge_arr = daily_df["discharge_cms"].to_numpy(dtype=np.float32)[np.newaxis, :]

  ds = xr.Dataset(
      data_vars={
          "streamflow": (
              ("basin", "date"),
              streamflow_arr,
              {
                  "units": "mm/day",
                  "long_name": "Specific catchment runoff / streamflow",
              },
          ),
          "discharge_cms": (
              ("basin", "date"),
              discharge_arr,
              {
                  "units": "m3/s",
                  "long_name": "Volumetric river discharge",
              },
          ),
      },
      coords={
          "basin": np.asarray([b_id], dtype=str),
          "date": dates,
      },
      attrs={
          "title": f"Earthkit Hydro Web {mode.title()} Streamflow Store",
          "mode": mode,
          "updated_at": datetime.now(timezone.utc).isoformat(),
      },
  )
  return ds


def _merge_and_write_streamflow_zarr(
    new_ds: xr.Dataset,
    zarr_path: Path,
    basin_id: str,
) -> xr.Dataset:
  """Merges `new_ds` into `zarr_path` along `(basin, date)` and writes atomically in Zarr format."""
  zarr_path = Path(zarr_path)
  zarr_path.parent.mkdir(parents=True, exist_ok=True)
  b_id = str(basin_id).strip()

  existing_ds: Optional[xr.Dataset] = None
  if zarr_path.exists():
    try:
      with xr.open_zarr(str(zarr_path), consolidated=False) as opened:
        existing_ds = opened.load()
    except Exception as e:
      logger.warning(
          "Could not open existing Zarr store at %s (%s); recreating clean store.",
          zarr_path,
          e,
      )
      existing_ds = None

  if (
      existing_ds is not None
      and "basin" in existing_ds.coords
      and "date" in existing_ds.coords
  ):
    existing_basins = [str(b) for b in existing_ds.coords["basin"].values]
    existing_ds = existing_ds.assign_coords(
        basin=np.asarray(existing_basins, dtype=str),
        date=pd.DatetimeIndex(existing_ds.coords["date"].values).astype("datetime64[ns]"),
    )

    all_basins = list(dict.fromkeys(existing_basins + [b_id]))
    all_dates = pd.DatetimeIndex(
        sorted(
            set(pd.DatetimeIndex(existing_ds.coords["date"].values)).union(
                set(pd.DatetimeIndex(new_ds.coords["date"].values))
            )
        ),
        name="date",
    ).astype("datetime64[ns]")

    # Use xr.combine_by_coords when adding a disjoint basin, or align + overwrite matching (basin, date)
    if b_id not in existing_basins:
      ex_reindexed = existing_ds.reindex(date=all_dates)
      new_reindexed = new_ds.reindex(date=all_dates)
      combined = xr.combine_by_coords(
          [ex_reindexed, new_reindexed],
          combine_attrs="override",
      )
      # Preserve deterministic basin ordering
      combined = combined.sel(basin=np.asarray(all_basins, dtype=str))
    else:
      combined = existing_ds.reindex(
          basin=np.asarray(all_basins, dtype=str),
          date=all_dates,
          fill_value=np.float32(np.nan),
      ).copy(deep=True)
      for var_name in ("streamflow", "discharge_cms"):
        if var_name in new_ds.data_vars:
          if var_name not in combined.data_vars:
            combined[var_name] = (
                ("basin", "date"),
                np.full(
                    (len(all_basins), len(all_dates)), np.nan, dtype=np.float32
                ),
            )
          combined[var_name].loc[
              dict(basin=b_id, date=new_ds.coords["date"].values)
          ] = (
              new_ds[var_name].sel(basin=b_id).values.astype(np.float32)
          )
  else:
    combined = new_ds

  # Enforce canonical dimension ordering, dtypes, and attributes
  combined = combined.transpose("basin", "date")
  combined = combined.assign_coords(
      basin=np.asarray([str(b) for b in combined.coords["basin"].values], dtype=str),
      date=pd.DatetimeIndex(combined.coords["date"].values, name="date").astype(
          "datetime64[ns]"
      ),
  )
  for var_name, units_str, long_name in (
      ("streamflow", "mm/day", "Specific catchment runoff / streamflow"),
      ("discharge_cms", "m3/s", "Volumetric river discharge"),
  ):
    if var_name in combined.data_vars:
      combined[var_name] = combined[var_name].astype(np.float32)
      combined[var_name].attrs["units"] = units_str
      combined[var_name].attrs["long_name"] = long_name

  chunked = combined.chunk({"basin": 1, "date": max(1, int(combined.sizes["date"]))})

  if zarr_path.exists():
    shutil.rmtree(zarr_path, ignore_errors=True)

  chunked.to_zarr(store=str(zarr_path), mode="w", consolidated=True)
  return combined


def _update_return_periods_json(
    return_periods_path: Path,
    basin_id: str,
    rp_data: Dict[str, Any],
) -> Dict[str, Any]:
  """Saves/updates targets/return_periods.json keyed by basin_id."""
  return_periods_path = Path(return_periods_path)
  return_periods_path.parent.mkdir(parents=True, exist_ok=True)

  store: Dict[str, Any] = {}
  if return_periods_path.exists():
    try:
      with open(return_periods_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        if isinstance(loaded, dict):
          store = loaded
    except Exception as e:
      logger.warning("Failed to read existing return_periods.json: %s", e)

  store[str(basin_id).strip()] = rp_data
  with open(return_periods_path, "w", encoding="utf-8") as f:
    json.dump(store, f, indent=2)
  return store


def ingest_historical_streamflow_csv(
    csv_data: Union[str, bytes, Path],
    basin_id: str,
    username: str = "guest",
    area_km2: Optional[float] = None,
    input_units: str = "auto",
    profile_manager: Optional[ProfileManager] = None,
) -> Dict[str, Any]:
  """Ingests a user-uploaded historical streamflow CSV into targets/streamflow.zarr and updates return_periods.json."""
  pm = profile_manager or get_profile_manager()
  clean_basin_id = str(basin_id).strip()
  if not clean_basin_id:
    raise ValueError("basin_id must be a non-empty string.")

  resolved_area = resolve_basin_area_km2(
      basin_id=clean_basin_id,
      area_km2=area_km2,
      username=username,
      profile_manager=pm,
  )

  raw_csv_text = _read_raw_csv_text(csv_data)
  parsed = parse_streamflow_csv(
      csv_data=raw_csv_text,
      basin_id=clean_basin_id,
      area_km2=resolved_area,
      input_units=input_units,
  )
  daily_df: pd.DataFrame = parsed["daily_df"]

  # Save raw uploaded CSV to targets/uploads/<basin_id>_historical.csv
  targets_dir = pm.get_targets_dir(username)
  uploads_dir = targets_dir / "uploads"
  uploads_dir.mkdir(parents=True, exist_ok=True)
  raw_csv_path = uploads_dir / f"{clean_basin_id}_historical.csv"
  raw_csv_path.write_text(raw_csv_text, encoding="utf-8")

  # Build and persist Zarr Dataset at targets/streamflow.zarr
  zarr_path = pm.get_targets_zarr_path(username)
  single_ds = _build_single_basin_dataset(
      basin_id=clean_basin_id,
      daily_df=daily_df,
      area_km2=resolved_area,
      mode="historical",
  )
  combined_ds = _merge_and_write_streamflow_zarr(
      new_ds=single_ds,
      zarr_path=zarr_path,
      basin_id=clean_basin_id,
  )

  # Compute return periods and persist targets/return_periods.json
  rp_data = compute_return_periods(
      series_or_df=daily_df,
      area_km2=resolved_area,
      basin_id=clean_basin_id,
  )
  rp_path = pm.get_return_periods_path(username)
  _update_return_periods_json(rp_path, clean_basin_id, rp_data)

  valid_mask = daily_df["streamflow"].notna()
  n_total = int(len(daily_df))
  n_valid = int(valid_mask.sum())
  n_missing = n_total - n_valid

  dates_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in daily_df.index]
  preview_n = min(60, n_total)

  return {
      "status": "success",
      "mode": "historical",
      "username": username,
      "basin_id": clean_basin_id,
      "area_km2": resolved_area,
      "detected_input_units": parsed["detected_units"],
      "is_subdaily_aggregated": parsed["is_subdaily"],
      "n_observations": n_total,
      "n_valid": n_valid,
      "n_missing": n_missing,
      "start_date": dates_str[0] if dates_str else None,
      "end_date": dates_str[-1] if dates_str else None,
      "zarr_path": str(zarr_path),
      "raw_csv_path": str(raw_csv_path),
      "return_periods_path": str(rp_path),
      "return_periods": rp_data,
      "all_basins": [str(b) for b in combined_ds.coords["basin"].values],
      "preview": {
          "dates": dates_str[:preview_n],
          "streamflow_mm_day": [
              None if np.isnan(v) else round(float(v), 4)
              for v in daily_df["streamflow"].to_numpy()[:preview_n]
          ],
          "discharge_cms": [
              None if np.isnan(v) else round(float(v), 4)
              for v in daily_df["discharge_cms"].to_numpy()[:preview_n]
          ],
      },
  }


def ingest_realtime_streamflow_csv(
    csv_data: Union[str, bytes, Path],
    basin_id: str,
    username: str = "guest",
    area_km2: Optional[float] = None,
    input_units: str = "auto",
    profile_manager: Optional[ProfileManager] = None,
) -> Dict[str, Any]:
  """Ingests a user-uploaded real-time gauge CSV into assimilation/streamflow_realtime.zarr."""
  pm = profile_manager or get_profile_manager()
  clean_basin_id = str(basin_id).strip()
  if not clean_basin_id:
    raise ValueError("basin_id must be a non-empty string.")

  resolved_area = resolve_basin_area_km2(
      basin_id=clean_basin_id,
      area_km2=area_km2,
      username=username,
      profile_manager=pm,
  )

  raw_csv_text = _read_raw_csv_text(csv_data)
  parsed = parse_streamflow_csv(
      csv_data=raw_csv_text,
      basin_id=clean_basin_id,
      area_km2=resolved_area,
      input_units=input_units,
  )
  daily_df: pd.DataFrame = parsed["daily_df"]
  raw_df: pd.DataFrame = parsed["raw_df"]

  # Save raw uploaded CSV to assimilation/uploads/<basin_id>_realtime.csv
  assim_dir = pm.get_assimilation_dir(username)
  uploads_dir = assim_dir / "uploads"
  uploads_dir.mkdir(parents=True, exist_ok=True)
  raw_csv_path = uploads_dir / f"{clean_basin_id}_realtime.csv"
  raw_csv_path.write_text(raw_csv_text, encoding="utf-8")

  # Build and persist Zarr Dataset at assimilation/streamflow_realtime.zarr
  zarr_path = pm.get_assimilation_zarr_path(username)
  single_ds = _build_single_basin_dataset(
      basin_id=clean_basin_id,
      daily_df=daily_df,
      area_km2=resolved_area,
      mode="realtime",
  )
  combined_ds = _merge_and_write_streamflow_zarr(
      new_ds=single_ds,
      zarr_path=zarr_path,
      basin_id=clean_basin_id,
  )

  valid_mask = daily_df["streamflow"].notna()
  n_total = int(len(daily_df))
  n_valid = int(valid_mask.sum())
  n_missing = n_total - n_valid

  dates_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in daily_df.index]
  raw_timestamps_str = [
      pd.Timestamp(ts).isoformat() for ts in raw_df["timestamp"].to_numpy()
  ]
  preview_n = min(60, n_total)

  return {
      "status": "success",
      "mode": "realtime",
      "username": username,
      "basin_id": clean_basin_id,
      "area_km2": resolved_area,
      "detected_input_units": parsed["detected_units"],
      "is_subdaily_aggregated": parsed["is_subdaily"],
      "n_observations": n_total,
      "n_valid": n_valid,
      "n_missing": n_missing,
      "start_date": dates_str[0] if dates_str else None,
      "end_date": dates_str[-1] if dates_str else None,
      "zarr_path": str(zarr_path),
      "raw_csv_path": str(raw_csv_path),
      "raw_timestamps": raw_timestamps_str,
      "all_basins": [str(b) for b in combined_ds.coords["basin"].values],
      "preview": {
          "dates": dates_str[:preview_n],
          "streamflow_mm_day": [
              None if np.isnan(v) else round(float(v), 4)
              for v in daily_df["streamflow"].to_numpy()[:preview_n]
          ],
          "discharge_cms": [
              None if np.isnan(v) else round(float(v), 4)
              for v in daily_df["discharge_cms"].to_numpy()[:preview_n]
          ],
      },
  }


def read_basin_streamflow_series(
    basin_id: str,
    username: str = "guest",
    mode: str = "historical",
    profile_manager: Optional[ProfileManager] = None,
) -> Dict[str, Any]:
  """Reads a basin's stored streamflow time series (and return periods if historical) from Zarr."""
  pm = profile_manager or get_profile_manager()
  clean_basin_id = str(basin_id).strip()
  mode_norm = (mode or "historical").strip().lower()

  if mode_norm in ("realtime", "assimilation", "da"):
    zarr_path = pm.get_assimilation_zarr_path(username)
    canonical_mode = "realtime"
  else:
    zarr_path = pm.get_targets_zarr_path(username)
    canonical_mode = "historical"

  if not zarr_path.exists():
    raise FileNotFoundError(
        f"No {canonical_mode} streamflow Zarr store found at {zarr_path} for user '{username}'."
    )

  with xr.open_zarr(str(zarr_path), consolidated=False) as ds:
    ds_loaded = ds.load()

  basins_in_store = [str(b) for b in ds_loaded.coords["basin"].values]
  if clean_basin_id not in basins_in_store:
    raise KeyError(
        f"Basin '{clean_basin_id}' not found in {zarr_path}. Available basins: {basins_in_store}"
    )

  basin_ds = ds_loaded.sel(basin=clean_basin_id)
  dates_arr = pd.DatetimeIndex(basin_ds.coords["date"].values)
  streamflow_vals = np.asarray(basin_ds["streamflow"].values, dtype=np.float32)
  discharge_vals = np.asarray(basin_ds["discharge_cms"].values, dtype=np.float32)

  # Filter out leading/trailing padding NaNs that may exist solely from multi-basin outer date alignment
  # while preserving internal NaNs within the basin's observed date range
  valid_indices = np.where(np.isfinite(streamflow_vals) | np.isfinite(discharge_vals))[0]
  if valid_indices.size > 0:
    first_idx = int(valid_indices[0])
    last_idx = int(valid_indices[-1]) + 1
    dates_arr = dates_arr[first_idx:last_idx]
    streamflow_vals = streamflow_vals[first_idx:last_idx]
    discharge_vals = discharge_vals[first_idx:last_idx]

  dates_list = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates_arr]
  streamflow_list = [
      None if not np.isfinite(v) else float(v) for v in streamflow_vals
  ]
  discharge_list = [
      None if not np.isfinite(v) else float(v) for v in discharge_vals
  ]

  return_periods_info = None
  if canonical_mode == "historical":
    rp_path = pm.get_return_periods_path(username)
    if rp_path.exists():
      try:
        with open(rp_path, "r", encoding="utf-8") as f:
          rp_all = json.load(f)
          if isinstance(rp_all, dict):
            return_periods_info = rp_all.get(clean_basin_id)
      except Exception:
        pass

  return {
      "status": "success",
      "basin_id": clean_basin_id,
      "username": username,
      "mode": canonical_mode,
      "zarr_path": str(zarr_path),
      "dates": dates_list,
      "streamflow": streamflow_list,
      "streamflow_mm_day": streamflow_list,
      "discharge_cms": discharge_list,
      "n_observations": len(dates_list),
      "n_valid": int(np.isfinite(streamflow_vals).sum()),
      "n_missing": int((~np.isfinite(streamflow_vals)).sum()),
      "start_date": dates_list[0] if dates_list else None,
      "end_date": dates_list[-1] if dates_list else None,
      "return_periods": return_periods_info,
  }
