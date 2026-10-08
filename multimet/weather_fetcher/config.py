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

"""Model and variable physical metadata and unit conversions for Weather Data Fetcher."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Grid and timeline specifications (0.25 deg global grid, +90.0 to -90.0 lat)
N_LAT: int = 721
N_LON: int = 1440
GRID_DEG: float = 0.25
NUM_STEPS: int = 81  # 10 days at 3-hour steps (0, 3, ..., 240 h)
NUM_HOURS: int = 81
STEP_HOURS: int = 3
VIEWER_STEP_HOURS: int = 3
MAX_LEAD_HOURS: int = (NUM_STEPS - 1) * STEP_HOURS

# Metadata and synchronization constants
STAC_CATALOG_URL: str = "https://stac.dynamical.org/catalog.json"
RUN_METADATA_FILE: str = "latest_dynamical_meta.json"
SYNC_STATUS_FILE: str = "sync_status.json"
KEEP_PREVIOUS_RUNS: int = 1
DEFAULT_MSLP_OFFSET_HPA: float = 1000.0

MIN_VALID_RESAMPLE_FRACTION: float = 0.8
MSLP_OFFSET_HPA: float = 1000.0

MIN_VALID_RESAMPLE_FRACTION: float = 0.8


# Oldest acceptable last-valid day of a cached NOAA PSL CPC annual NetCDF file
# before it must be downloaded again, and the maximum age of the cached file.
CPC_MAX_PUBLICATION_LAG_DAYS: int = 7
CPC_CACHE_REFRESH_HOURS: float = 6.0

RUN_DATASET_TO_MODEL: Dict[str, str] = {
    "ecmwf_aifs_single_forecast": "ecmwf_aifs",
    "noaa_gfs_forecast": "noaa_gfs",
    "ecmwf_ifs_ens_forecast_15_day_0_25_degree": "ecmwf_ifs",
    "ecmwf_ifs_hres_0_25_degree": "ecmwf_hres",
    "noaa_gefs_forecast_35_day": "noaa_gefs",
    "noaa_hrrr_forecast_48_hour": "noaa_hrrr",
    "nasa_imerg_analysis_early": "nasa_imerg",
    "noaa_cpc_unified_gauge_precip": "noaa_cpc",
}

SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
    "ecmwf_hres": {
        "id": "ecmwf_hres",
        "name": "ECMWF IFS HRES Operational",
        "type": "physics",
        "resolution": "0.25° Global (9km native)",
        "organization": "ECMWF",
    },
    "ecmwf_ifs": {
        "id": "ecmwf_ifs",
        "name": "ECMWF IFS (ensemble control run)",
        "type": "physics",
        "resolution": "0.25° Global (9km native)",
        "organization": "ECMWF",
    },
    "ecmwf_aifs": {
        "id": "ecmwf_aifs",
        "name": "ECMWF AIFS",
        "type": "ai",
        "resolution": "0.25° Global",
        "organization": "ECMWF",
    },
    "noaa_gfs": {
        "id": "noaa_gfs",
        "name": "NOAA GFS",
        "type": "physics",
        "resolution": "0.25° Global",
        "organization": "NOAA / NWS",
    },
    "noaa_gefs": {
        "id": "noaa_gefs",
        "name": "NOAA GEFS (ensemble control run)",
        "type": "physics",
        "resolution": "0.25° Global",
        "organization": "NOAA / NWS",
    },
    "noaa_hrrr": {
        "id": "noaa_hrrr",
        "name": "NOAA HRRR (CONUS 3km)",
        "type": "physics",
        "resolution": "3 km CONUS",
        "organization": "NOAA / NWS",
    },
    "nasa_imerg": {
        "id": "nasa_imerg",
        "name": "NASA IMERG Early",
        "type": "observation",
        "resolution": "0.1° Global",
        "organization": "NASA",
    },
    "noaa_cpc": {
        "id": "noaa_cpc",
        "name": "NOAA CPC Global Unified Gauge",
        "type": "observation",
        "resolution": "0.5° Global",
        "organization": "NOAA",
    },
}

SUPPORTED_VARIABLES: Dict[str, Dict[str, Any]] = {
    "precipitation": {
        "id": "precipitation",
        "name": "Total Precipitation Rate",
        "unit": "mm/h",
        "min": 0.0,
        "max": 25.0,
    },
    "accumulated_precip": {
        "id": "accumulated_precip",
        "name": "Precipitation Accumulated Since Forecast Start",
        "unit": "mm",
        "min": 0.0,
        "max": 250.0,
    },
    "temperature": {
        "id": "temperature",
        "name": "2m Ambient Temperature",
        "unit": "°C",
        "min": -40.0,
        "max": 45.0,
    },
    "wind": {
        "id": "wind",
        "name": "10m Surface Wind Velocity",
        "unit": "m/s",
        "min": 0.0,
        "max": 40.0,
    },
    "pressure": {
        "id": "pressure",
        "name": "Mean Sea Level Pressure",
        "unit": "hPa",
        "min": 960.0,
        "max": 1040.0,
    },
}

# Stream suffix -> dynamical.org variable name
STREAM_VARIABLES: Dict[str, str] = {
    "precip": "precipitation_surface",
    "temp": "temperature_2m",
    "mslp": "pressure_reduced_to_mean_sea_level",
    "u10": "wind_u_10m",
    "v10": "wind_v_10m",
}

# Physical variable -> suffix of the binary stream that holds it
STREAM_SUFFIX: Dict[str, str] = {
    "precipitation": "precip",
    "accumulated_precip": "precip",
    "temperature": "temp",
    "pressure": "mslp",
}

# (stream id, file name, is precipitation)
STREAM_FILES: Tuple[Tuple[str, str, bool], ...] = (
    ("ecmwf_hres_precip", "ecmwf_hres_precip.bin", True),
    ("ecmwf_hres_temp", "ecmwf_hres_temp.bin", False),
    ("ecmwf_hres_mslp", "ecmwf_hres_mslp.bin", False),
    ("ecmwf_hres_u10", "ecmwf_hres_u10.bin", False),
    ("ecmwf_hres_v10", "ecmwf_hres_v10.bin", False),
    ("ecmwf_ifs_precip", "ecmwf_ifs_precip.bin", True),
    ("ecmwf_ifs_temp", "ecmwf_ifs_temp.bin", False),
    ("ecmwf_ifs_mslp", "ecmwf_ifs_mslp.bin", False),
    ("ecmwf_ifs_u10", "ecmwf_ifs_u10.bin", False),
    ("ecmwf_ifs_v10", "ecmwf_ifs_v10.bin", False),
    ("ecmwf_aifs_precip", "ecmwf_aifs_precip.bin", True),
    ("ecmwf_aifs_temp", "ecmwf_aifs_temp.bin", False),
    ("ecmwf_aifs_mslp", "ecmwf_aifs_mslp.bin", False),
    ("ecmwf_aifs_u10", "ecmwf_aifs_u10.bin", False),
    ("ecmwf_aifs_v10", "ecmwf_aifs_v10.bin", False),
    ("noaa_gfs_precip", "noaa_gfs_precip.bin", True),
    ("noaa_gfs_temp", "noaa_gfs_temp.bin", False),
    ("noaa_gfs_mslp", "noaa_gfs_mslp.bin", False),
    ("noaa_gfs_u10", "noaa_gfs_u10.bin", False),
    ("noaa_gfs_v10", "noaa_gfs_v10.bin", False),
    ("noaa_gefs_precip", "noaa_gefs_precip.bin", True),
    ("noaa_gefs_temp", "noaa_gefs_temp.bin", False),
    ("noaa_gefs_mslp", "noaa_gefs_mslp.bin", False),
    ("noaa_gefs_u10", "noaa_gefs_u10.bin", False),
    ("noaa_gefs_v10", "noaa_gefs_v10.bin", False),
    ("noaa_hrrr_precip", "noaa_hrrr_precip.bin", True),
    ("noaa_hrrr_temp", "noaa_hrrr_temp.bin", False),
    ("noaa_hrrr_mslp", "noaa_hrrr_mslp.bin", False),
    ("noaa_hrrr_u10", "noaa_hrrr_u10.bin", False),
    ("noaa_hrrr_v10", "noaa_hrrr_v10.bin", False),
    ("nasa_imerg_precip", "nasa_imerg_precip.bin", True),
    ("noaa_cpc_precip", "noaa_cpc_precip.bin", True),
)

GLOBAL_VARS: List[str] = [
    "ecmwf_hres_precip",
    "ecmwf_hres_temp",
    "ecmwf_hres_u10",
    "ecmwf_hres_v10",
    "ecmwf_ifs_precip",
    "ecmwf_ifs_temp",
    "ecmwf_ifs_u10",
    "ecmwf_ifs_v10",
    "ecmwf_aifs_precip",
    "ecmwf_aifs_temp",
    "noaa_gfs_precip",
    "noaa_gfs_temp",
    "noaa_gefs_precip",
    "noaa_gefs_temp",
    "noaa_hrrr_precip",
    "noaa_hrrr_temp",
    "nasa_imerg_precip",
    "noaa_cpc_precip",
]

# Default operational models synchronized by sync_all_models
DEFAULT_SYNC_MODELS: Tuple[str, ...] = (
    "ecmwf_hres",
    "ecmwf_ifs",
    "ecmwf_aifs",
    "noaa_gfs",
    "noaa_gefs",
    "noaa_hrrr",
    "nasa_imerg",
    "noaa_cpc",
)


# Human-readable upstream source per ``DYNAMICAL_MODELS[...]["source"]``.
SOURCE_LABELS: Dict[str, str] = {
    "dynamical": "dynamical.org",
    "dynamical_analysis": "dynamical.org",
    "ecmwf_open_data": "gs://ecmwf-open-data",
    "noaa_psl_cpc": "NOAA PSL (downloads.psl.noaa.gov)",
}

DYNAMICAL_MODELS: Dict[str, Dict[str, Any]] = {
    "ecmwf_hres": {
        "dataset": "ecmwf-ifs-hres-0-25-degree",
        "title": "ECMWF IFS HRES Operational (0.25°)",
        "source": "ecmwf_open_data",
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "ecmwf_ifs": {
        "dataset": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "title": "ECMWF IFS ENS control member (0.25°)",
        "source": "dynamical",
        "ensemble_member": 0,
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "ecmwf_aifs": {
        "dataset": "ecmwf-aifs-single-forecast",
        "title": "ECMWF AIFS Single (0.25°)",
        "source": "dynamical",
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "noaa_gfs": {
        "dataset": "noaa-gfs-forecast",
        "title": "NOAA GFS (0.25°)",
        "source": "dynamical",
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "noaa_gefs": {
        "dataset": "noaa-gefs-forecast-35-day",
        "title": "NOAA GEFS ENS control member (0.25°)",
        "source": "dynamical",
        "ensemble_member": 0,
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "noaa_hrrr": {
        "dataset": "noaa-hrrr-forecast-48-hour",
        "title": "NOAA HRRR CONUS (3 km)",
        "source": "dynamical",
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "nasa_imerg": {
        "dataset": "nasa-imerg-analysis-early",
        "title": "NASA GPM IMERG Early Analysis (0.10°)",
        "source": "dynamical_analysis",
        "streams": ("precip",),
    },
    "noaa_cpc": {
        "dataset": "noaa-cpc-unified-gauge-precip",
        "title": "NOAA CPC Unified Global Gauge Precip (0.50°)",
        "source": "noaa_psl_cpc",
        "streams": ("precip",),
    },
}

MODEL_NATIVE_LEAD_HOURS: Dict[str, Tuple[int, ...]] = {
    "ecmwf_hres": tuple(list(range(0, 145, 3)) + list(range(150, 241, 6))),
    "ecmwf_ifs": tuple(list(range(0, 145, 3)) + list(range(150, 241, 6))),
    "ecmwf_aifs": tuple(range(0, 241, 6)),
    "noaa_gfs": tuple(list(range(0, 121)) + list(range(123, 241, 3))),
    "noaa_gefs": tuple(range(0, 241, 3)),
    "noaa_hrrr": tuple(range(0, 49)),
    "nasa_imerg": tuple(range(0, 241, 3)),
    "noaa_cpc": tuple(range(0, 241, 24)),
}


def output_lead_hours(
    in_leads: Sequence[int],
    max_lead: int = MAX_LEAD_HOURS,
    step: int = STEP_HOURS,
) -> List[int]:
  """Returns stored lead hours: unique model leads on 3-hourly steps."""
  return sorted({
      int(h) for h in in_leads if 0 <= int(h) <= max_lead and int(h) % step == 0
  })


def run_lead_hours(model_key: str, n_steps: int) -> List[int]:
  """Returns the lead hours of the first ``n_steps`` stored planes of a model.

  Args:
    model_key: Supported model key (see ``SUPPORTED_MODELS``).
    n_steps: Number of planes stored in the archived binary stream.

  Returns:
    Lead hours (one per plane) derived from the model's native step list.

  Raises:
    KeyError: If ``model_key`` is not a supported model.
    ValueError: If ``n_steps`` exceeds the number of storable planes.
  """
  if model_key not in MODEL_NATIVE_LEAD_HOURS:
    raise KeyError(
        f"Unknown weather model {model_key!r}. Supported:"
        f" {list(MODEL_NATIVE_LEAD_HOURS)}"
    )
  leads = output_lead_hours(MODEL_NATIVE_LEAD_HOURS[model_key])
  if n_steps < 1 or n_steps > len(leads):
    raise ValueError(
        f"{model_key}: cannot label {n_steps} stored planes; the model stores"
        f" between 1 and {len(leads)} planes ({leads[0]}..{leads[-1]} h)."
    )
  return leads[:n_steps]


_PRECIP_UNIT_TO_MM_PER_H: Dict[str, float] = {
    "kg m-2 s-1": 3600.0,
    "kg m**-2 s**-1": 3600.0,
    "mm/s": 3600.0,
    "mm s-1": 3600.0,
    "mm/h": 1.0,
    "mm/hr": 1.0,
    "mm h-1": 1.0,
    "mm/day": 1.0 / 24.0,
    "mm/d": 1.0 / 24.0,
    "mm day-1": 1.0 / 24.0,
}
_KELVIN_UNITS: Tuple[str, ...] = ("K", "kelvin", "degK", "deg_K")
_CELSIUS_UNITS: Tuple[str, ...] = (
    "degC",
    "deg_C",
    "C",
    "celsius",
    "degree_Celsius",
    "degrees_Celsius",
)
_PASCAL_UNITS: Tuple[str, ...] = ("Pa", "pa", "pascal")
_HECTOPASCAL_UNITS: Tuple[str, ...] = ("hPa", "hpa", "mbar", "millibar")
_WIND_UNITS: Tuple[str, ...] = ("m/s", "m s-1", "m s**-1", "m/sec")


def to_stored_units(
    stream: str,
    values: Any,
    source_units: Optional[str] = None,
) -> np.ndarray:
  """Converts upstream physical units to stored float16 planes.

  Missing values (``NaN``) are preserved for every stream; negative
  precipitation rates (GRIB packing noise) are clamped to ``0.0``.

  Args:
    stream: Stream suffix ('precip', 'temp', 'mslp', 'u10', 'v10').
    values: Input array-like in ``source_units``.
    source_units: Units of ``values``. Precipitation accepts ``kg m-2 s-1`` /
      ``mm/s`` (default when omitted), ``mm/h`` and ``mm/day``; temperature
      accepts ``K`` and ``degC``; pressure accepts ``Pa`` (default) and
      ``hPa``; wind accepts ``m/s``. When omitted for temperature, Kelvin is
      assumed if the finite mean exceeds 150 (no physical 2 m temperature in
      degC can reach that value).

  Returns:
    Float16 NumPy array in stored units (precip in mm/h, temp in degC,
    MSLP in hPa - ``MSLP_OFFSET_HPA``, wind in m/s).

  Raises:
    ValueError: If ``stream`` or ``source_units`` is not recognised.
  """
  arr = np.asarray(values, dtype=np.float32)
  if stream == "precip":
    units = source_units if source_units is not None else "kg m-2 s-1"
    if units not in _PRECIP_UNIT_TO_MM_PER_H:
      raise ValueError(
          f"Unsupported precipitation units {units!r}; expected one of"
          f" {sorted(_PRECIP_UNIT_TO_MM_PER_H)}"
      )
    arr = np.where(np.isnan(arr), np.nan, np.maximum(arr, 0.0)).astype(
        np.float32
    ) * np.float32(_PRECIP_UNIT_TO_MM_PER_H[units])
  elif stream == "mslp":
    units = source_units if source_units is not None else "Pa"
    if units in _PASCAL_UNITS:
      arr = arr / 100.0 - MSLP_OFFSET_HPA
    elif units in _HECTOPASCAL_UNITS:
      arr = arr - MSLP_OFFSET_HPA
    else:
      raise ValueError(
          f"Unsupported pressure units {units!r}; expected Pa or hPa."
      )
  elif stream == "temp":
    if source_units is None:
      finite = arr[np.isfinite(arr)]
      is_kelvin = finite.size > 0 and float(np.mean(finite)) > 150.0
    elif source_units in _KELVIN_UNITS:
      is_kelvin = True
    elif source_units in _CELSIUS_UNITS:
      is_kelvin = False
    else:
      raise ValueError(
          f"Unsupported temperature units {source_units!r}; expected K or"
          " degC."
      )
    if is_kelvin:
      arr = arr - 273.15
  elif stream in ("u10", "v10"):
    if source_units is not None and source_units not in _WIND_UNITS:
      raise ValueError(
          f"Unsupported wind units {source_units!r}; expected m/s."
      )
  else:
    raise ValueError(
        f"Unknown stream {stream!r}; expected one of {list(STREAM_VARIABLES)}"
    )
  return arr.astype(np.float16)


def from_stored_units(
    stream: str,
    values: Any,
    mslp_offset_hpa: float = DEFAULT_MSLP_OFFSET_HPA,
) -> np.ndarray:
  """Restores physical units from stored float16 planes."""
  arr = np.asarray(values, dtype=np.float32)
  if stream == "mslp":
    return arr + float(mslp_offset_hpa)
  return arr


def parse_zarr_metadata_time_extent(
    zmetadata_json: str,
) -> Optional[Dict[str, Any]]:
  """Parses CF-compliant time extent from Zarr .zmetadata JSON content."""
  if not zmetadata_json:
    return None
  parsed = json.loads(zmetadata_json)
  meta = parsed.get("metadata", {})
  time_arr = meta.get("time/.zarray", {})
  time_attrs = meta.get("time/.zattrs", {})
  shape = time_arr.get("shape", [])
  if not shape:
    return None
  n_steps = int(shape[0])
  units = str(time_attrs.get("units", ""))
  if "days since " not in units:
    return None
  base_str = units.split("days since ")[1].split()[0]
  base_dt = datetime.strptime(base_str, "%Y-%m-%d")
  start_date = base_dt.strftime("%Y-%m-%d")
  end_date = (base_dt + timedelta(days=n_steps - 1)).strftime("%Y-%m-%d")
  total_years = round(n_steps / 365.25, 1)
  return {
      "start_date": start_date,
      "end_date": end_date,
      "n_timesteps": n_steps,
      "total_years": total_years,
      "is_dynamic": True,
  }
