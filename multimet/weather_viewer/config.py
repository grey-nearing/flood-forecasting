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

"""Model & variable catalog, colormaps, and unit conversions for Weather Viewer."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
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
SYNC_INTERVAL_HOURS: int = 6

# Versioning and rendering constants
TILE_VERSION: str = "3"
FRAME_SIZE: int = 1440
FRAME_VERSION: str = "1"
MERCATOR_MAX_LAT: float = 85.0511287798
FRAME_VARIABLES: Tuple[str, ...] = (
    "precipitation",
    "accumulated_precip",
    "temperature",
    "pressure",
)
TEMP_LEVELS: int = 128
PRESSURE_LEVELS: int = 51

# Metadata and synchronization constants
STAC_CATALOG_URL: str = "https://stac.dynamical.org/catalog.json"
RUN_METADATA_FILE: str = "latest_dynamical_meta.json"
SYNC_STATUS_FILE: str = "sync_status.json"
SYNC_LOG_FILE: str = "sync.log"
CHECK_INTERVAL_MINUTES: int = 60
KEEP_PREVIOUS_RUNS: int = 1
DEFAULT_MSLP_OFFSET_HPA: float = 1000.0
MSLP_OFFSET_HPA: float = 1000.0
SUBPROCESS_TIMEOUT_S: int = 45 * 60

RUN_DATASET_TO_MODEL: Dict[str, str] = {
    "ecmwf_aifs_single_forecast": "ecmwf_aifs",
    "noaa_gfs_forecast": "noaa_gfs",
    "ecmwf_ifs_ens_forecast_15_day_0_25_degree": "ecmwf_ifs",
    "noaa_gefs_forecast_35_day": "noaa_gefs",
    "noaa_hrrr_forecast_48_hour": "noaa_hrrr",
}

SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
    "ecmwf_ifs": {
        "id": "ecmwf_ifs",
        "name": "ECMWF IFS (ensemble control run)",
        "badge": "0.25° Physics 10-Day",
        "type": "physics",
        "resolution": "0.25° Global (9km native)",
        "organization": "ECMWF",
    },
    "ecmwf_aifs": {
        "id": "ecmwf_aifs",
        "name": "ECMWF AIFS",
        "badge": "0.25° Global AI 10-Day",
        "type": "ai",
        "resolution": "0.25° Global",
        "organization": "ECMWF",
    },
    "graphcast": {
        "id": "graphcast",
        "name": "Google DeepMind GraphCast",
        "badge": "0.25° AI 10-Day",
        "type": "ai",
        "resolution": "0.25° Global",
        "organization": "Google DeepMind",
    },
    "noaa_gfs": {
        "id": "noaa_gfs",
        "name": "NOAA GFS",
        "badge": "0.25° Physics 10-Day",
        "type": "physics",
        "resolution": "0.25° Global",
        "organization": "NOAA / NWS",
    },
    "noaa_gefs": {
        "id": "noaa_gefs",
        "name": "NOAA GEFS (ensemble control run)",
        "badge": "0.25° Ensemble 10-Day",
        "type": "physics",
        "resolution": "0.25° Global",
        "organization": "NOAA / NWS",
    },
    "noaa_hrrr": {
        "id": "noaa_hrrr",
        "name": "NOAA HRRR (CONUS 3km)",
        "badge": "3km Physics 48-Hour",
        "type": "physics",
        "resolution": "3 km CONUS",
        "organization": "NOAA / NWS",
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

# Tile colour classes: (lower bound, RGBA). Values below the first bound are
# transparent.
RAIN_RATE_CLASSES: Tuple[Tuple[float, Tuple[int, int, int, int]], ...] = (
    (0.1, (125, 211, 252, 150)),
    (0.5, (59, 130, 246, 185)),
    (2.0, (34, 197, 94, 205)),
    (5.0, (234, 179, 8, 220)),
    (10.0, (239, 68, 68, 235)),
    (20.0, (192, 38, 211, 245)),
)

RAIN_ACCUM_CLASSES: Tuple[Tuple[float, Tuple[int, int, int, int]], ...] = (
    (1.0, (125, 211, 252, 140)),
    (5.0, (59, 130, 246, 175)),
    (10.0, (34, 197, 94, 195)),
    (25.0, (234, 179, 8, 215)),
    (50.0, (239, 68, 68, 230)),
    (100.0, (192, 38, 211, 245)),
)

# Viewer stream suffix -> dynamical.org variable name
STREAM_VARIABLES: Dict[str, str] = {
    "precip": "precipitation_surface",
    "temp": "temperature_2m",
    "mslp": "pressure_reduced_to_mean_sea_level",
    "u10": "wind_u_10m",
    "v10": "wind_v_10m",
}

# Viewer variable -> suffix of the binary stream that holds it
STREAM_SUFFIX: Dict[str, str] = {
    "precipitation": "precip",
    "accumulated_precip": "precip",
    "temperature": "temp",
    "pressure": "mslp",
}

# (stream id, file name, is precipitation)
STREAM_FILES: Tuple[Tuple[str, str, bool], ...] = (
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
    ("graphcast_precip", "graphcast_precip.bin", True),
    ("graphcast_temp", "graphcast_temp.bin", False),
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
)

GLOBAL_VARS: List[str] = [
    "ecmwf_ifs_precip",
    "ecmwf_ifs_temp",
    "ecmwf_ifs_u10",
    "ecmwf_ifs_v10",
    "ecmwf_aifs_precip",
    "ecmwf_aifs_temp",
    "graphcast_precip",
    "graphcast_temp",
    "noaa_gfs_precip",
    "noaa_gfs_temp",
]

# Default operational models synchronized from dynamical.org
DEFAULT_SYNC_MODELS: Tuple[str, ...] = ("ecmwf_ifs", "ecmwf_aifs", "noaa_gfs")

DYNAMICAL_MODELS: Dict[str, Dict[str, Any]] = {
    "ecmwf_ifs": {
        "dataset": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "title": "ECMWF IFS ENS control member (0.25°)",
        "ensemble_member": 0,
        "streams": ("precip", "temp"),
    },
    "ecmwf_aifs": {
        "dataset": "ecmwf-aifs-single-forecast",
        "title": "ECMWF AIFS Single (0.25°)",
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "noaa_gfs": {
        "dataset": "noaa-gfs-forecast",
        "title": "NOAA GFS (0.25°)",
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "noaa_gefs": {
        "dataset": "noaa-gefs-forecast-35-day",
        "title": "NOAA GEFS ENS control member (0.25°)",
        "ensemble_member": 0,
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
    "noaa_hrrr": {
        "dataset": "noaa-hrrr-forecast-48-hour",
        "title": "NOAA HRRR CONUS (3 km)",
        "streams": ("precip", "temp", "mslp", "u10", "v10"),
    },
}


def output_lead_hours(
    in_leads: Sequence[int],
    max_lead: int = MAX_LEAD_HOURS,
    step: int = VIEWER_STEP_HOURS,
) -> List[int]:
  """Returns stored lead hours: model leads that fall on viewer steps."""
  return [
      int(h) for h in in_leads if 0 <= int(h) <= max_lead and int(h) % step == 0
  ]


def run_lead_hours(model_key: str, n_steps: int) -> List[int]:
  """Returns default lead hours of each plane in an archived run file."""
  if model_key == "ecmwf_aifs":
    return [6 * i for i in range(n_steps)]
  if model_key == "noaa_gfs":
    hourly = list(range(min(n_steps, 121)))
    return hourly + [120 + 3 * (i + 1) for i in range(n_steps - len(hourly))]
  return [STEP_HOURS * i for i in range(n_steps)]


def to_stored_units(stream: str, values: Any) -> np.ndarray:
  """Converts dynamical.org physical units to stored float16 planes.

  Args:
    stream: Stream suffix ('precip', 'temp', 'mslp', 'u10', 'v10').
    values: Input array-like in native dynamical.org units (precip in kg/m2/s
      or mm/s, temperature in degC or K, MSLP in Pa, wind in m/s).

  Returns:
    Float16 NumPy array in stored units (precip in mm/h, temp in degC,
    MSLP in hPa - 1000.0, wind in m/s).
  """
  arr = np.asarray(values, dtype=np.float32)
  if stream == "precip":
    arr = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, None) * 3600.0
  elif stream == "mslp":
    arr = arr / 100.0 - MSLP_OFFSET_HPA
  elif stream == "temp":
    # If input values are in Kelvin (e.g. mean > 150 K), convert to Celsius
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and float(np.mean(finite)) > 150.0:
      arr = arr - 273.15
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


@dataclasses.dataclass
class WeatherSource:
  """Metadata and dynamic temporal coverage for a gridded weather dataset."""

  id: str
  name: str
  provider: str
  description: str
  resolution: str
  temporal_resolution: str
  available_start: str
  latency_days: int = 5
  fixed_end: Optional[str] = None
  cns_zarr_path: Optional[str] = None
  default_variables: List[str] = dataclasses.field(
      default_factory=lambda: ["total_precipitation", "2m_temperature"]
  )
  citation: str = ""

  def get_time_range(
      self, cns_extent: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Computes the available temporal range for this dataset."""
    if cns_extent is not None:
      return {
          "start_date": cns_extent["start_date"],
          "end_date": cns_extent["end_date"],
          "latency_days": 0,
          "total_years": cns_extent["total_years"],
          "is_dynamic": True,
      }

    if self.fixed_end:
      end_date = self.fixed_end
    else:
      now_utc = datetime.now(timezone.utc)
      latest_dt = now_utc - timedelta(days=self.latency_days)
      end_date = latest_dt.strftime("%Y-%m-%d")

    start_dt = datetime.strptime(self.available_start, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_years = round((end_dt - start_dt).days / 365.25, 1)

    return {
        "start_date": self.available_start,
        "end_date": end_date,
        "latency_days": self.latency_days,
        "total_years": total_years,
        "is_dynamic": self.fixed_end is None,
    }

  def to_dict(
      self, cns_extent: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Serializes source metadata with dynamic time range."""
    time_range = self.get_time_range(cns_extent=cns_extent)
    return {
        "id": self.id,
        "name": self.name,
        "provider": self.provider,
        "description": self.description,
        "resolution": self.resolution,
        "temporal_resolution": self.temporal_resolution,
        "start_date": time_range["start_date"],
        "end_date": time_range["end_date"],
        "latency_days": time_range["latency_days"],
        "total_years": time_range["total_years"],
        "cns_zarr_path": self.cns_zarr_path or "",
        "default_variables": self.default_variables,
        "citation": self.citation,
    }


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


WEATHER_SOURCES: Dict[str, WeatherSource] = {
    "cpc": WeatherSource(
        id="cpc",
        name="NOAA CPC Global Precipitation",
        provider="NOAA Climate Prediction Center",
        description=(
            "Global daily gauge-based precipitation analysis spanning 1979 to"
            " present at 0.5° resolution."
        ),
        resolution="0.50° (~55 km)",
        temporal_resolution="Daily (1D)",
        available_start="1979-01-01",
        cns_zarr_path="/cns/jn-d/home/floods/lsm/loaded/lsm_loader_2024_07_03/cpc/precip.zarr",
        default_variables=["cpc_precipitation"],
        citation="Xie et al. (2007), J. Hydrometeorology",
    ),
    "imerg": WeatherSource(
        id="imerg",
        name="NASA GPM IMERG Final Run",
        provider="NASA Goddard Space Flight Center",
        description=(
            "Integrated Multi-satellitE Retrievals for GPM precipitation"
            " spanning 2000 to present at 0.1° resolution."
        ),
        resolution="0.10° (~10 km)",
        temporal_resolution="Daily (1D) / Half-Hourly",
        available_start="2000-06-01",
        cns_zarr_path="/cns/jn-d/home/floods/lsm/loaded/lsm_loader_2024_07_03/imerg/precip.zarr",
        default_variables=["imerg_precipitation"],
        citation="Huffman et al. (2020), NASA GSFC",
    ),
    "era5": WeatherSource(
        id="era5",
        name="ECMWF ERA5 Reanalysis",
        provider="ECMWF / Copernicus Climate Change Service (C3S)",
        description=(
            "Fifth generation ECMWF global atmospheric reanalysis spanning 1950"
            " to present."
        ),
        resolution="0.25° (~31 km)",
        temporal_resolution="Daily (1D) / Hourly",
        available_start="1950-01-01",
        cns_zarr_path="/cns/jn-d/home/floods/lsm/loaded/lsm_loader_2024_07_03/era5/tp.zarr",
        default_variables=[
            "total_precipitation",
            "2m_temperature",
            "surface_pressure",
            "surface_net_solar_radiation",
            "surface_net_thermal_radiation",
            "dewpoint_temperature",
        ],
        citation="Hersbach et al. (2020), QJRMS",
    ),
    "ifs": WeatherSource(
        id="ifs",
        name="ECMWF IFS HRES (High Resolution)",
        provider="ECMWF Operational Meteorological Archive",
        description=(
            "High-resolution operational deterministic model providing"
            " surface radiation, temperature, pressure, and precipitation."
        ),
        resolution="0.10° (~9 km)",
        temporal_resolution="Daily (1D) / 6-Hourly",
        available_start="2016-01-01",
        latency_days=2,
        default_variables=[
            "hres_total_precipitation",
            "hres_temperature_2m",
            "hres_surface_pressure",
            "hres_surface_net_solar_radiation",
            "hres_surface_net_thermal_radiation",
        ],
        citation="ECMWF IFS Documentation (2024)",
    ),
    "graphcast": WeatherSource(
        id="graphcast",
        name="GraphCast (DeepMind AI Weather Model)",
        provider="Google DeepMind / ECMWF",
        description=(
            "State-of-the-art AI weather forecast model generating fast,"
            " high-accuracy temperature and precipitation forecasts."
        ),
        resolution="0.25° (~28 km)",
        temporal_resolution="Daily (1D) / 6-Hourly",
        available_start="2019-01-01",
        latency_days=2,
        default_variables=[
            "graphcast_total_precipitation",
            "graphcast_temperature_2m",
        ],
        citation="Lam et al. (2023), Science",
    ),
    "era5-land": WeatherSource(
        id="era5-land",
        name="ECMWF ERA5-Land",
        provider="ECMWF / Copernicus Climate Change Service (C3S)",
        description=(
            "Enhanced land-surface reanalysis dataset at 0.1° (~9 km)"
            " resolution spanning 1950 to present."
        ),
        resolution="0.10° (~9 km)",
        temporal_resolution="Daily (1D) / Hourly",
        available_start="1950-01-01",
        latency_days=5,
        default_variables=[
            "total_precipitation",
            "2m_temperature",
            "surface_solar_radiation",
            "dewpoint_temperature",
        ],
        citation="Muñoz-Sabater et al. (2021), Earth System Science Data",
    ),
    "chirps": WeatherSource(
        id="chirps",
        name="CHIRPS v2.0 Global Precipitation",
        provider="UC Santa Barbara / Climate Hazards Center",
        description=(
            "Quasi-global high-resolution precipitation combining satellite"
            " imagery and station data (1981 to present)."
        ),
        resolution="0.05° (~5 km)",
        temporal_resolution="Daily (1D)",
        available_start="1981-01-01",
        latency_days=2,
        default_variables=["total_precipitation"],
        citation="Funk et al. (2015), Scientific Data",
    ),
    "cerra": WeatherSource(
        id="cerra",
        name="Copernicus CERRA Regional",
        provider="Copernicus Climate Change Service (C3S)",
        description=(
            "High-resolution regional reanalysis for the European domain at 5.5"
            " km resolution (1984 to 2021)."
        ),
        resolution="5.5 km",
        temporal_resolution="Daily (1D) / 3-hourly",
        available_start="1984-01-01",
        fixed_end="2021-06-30",
        latency_days=0,
        default_variables=["total_precipitation", "2m_temperature"],
        citation="Schimanke et al. (2024), QJRMS",
    ),
}


def get_weather_source(source_id: str = "era5") -> WeatherSource:
  """Retrieves a weather source by ID, defaulting to ERA5."""
  return WEATHER_SOURCES.get(source_id, WEATHER_SOURCES["era5"])


def list_weather_sources() -> List[Dict[str, Any]]:
  """Returns serialized list of all weather sources."""
  return [source.to_dict() for source in WEATHER_SOURCES.values()]
