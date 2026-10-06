"""Weather data sources registry with dynamic CNS temporal range resolution."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory cache for dynamically queried CNS time extents
_CNS_EXTENT_CACHE: Dict[str, Dict[str, Any]] = {}


def query_cns_zarr_extent(zarr_path: str) -> Optional[Dict[str, Any]]:
  """Queries the real time coordinate extent from a CNS Zarr store metadata."""
  if not zarr_path:
    return None
  if zarr_path in _CNS_EXTENT_CACHE:
    return _CNS_EXTENT_CACHE[zarr_path]

  if os.environ.get("UNITTEST_ON_FORGE") == "1":
    return None

  try:
    cmd = ["fileutil", "cat", f"{zarr_path}/.zmetadata"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if res.returncode == 0 and res.stdout:
      meta = json.loads(res.stdout).get("metadata", {})
      time_arr = meta.get("time/.zarray", {})
      time_attrs = meta.get("time/.zattrs", {})
      shape = time_arr.get("shape", [])
      if shape:
        n_steps = shape[0]
        units = time_attrs.get("units", "")
        if "days since " in units:
          base_str = units.split("days since ")[1].split()[0]
          base_dt = datetime.strptime(base_str, "%Y-%m-%d")
          start_date = base_dt.strftime("%Y-%m-%d")
          end_date = (base_dt + timedelta(days=n_steps - 1)).strftime("%Y-%m-%d")
          total_years = round(n_steps / 365.25, 1)
          extent_info = {
              "start_date": start_date,
              "end_date": end_date,
              "n_timesteps": n_steps,
              "total_years": total_years,
              "is_dynamic": True,
          }
          _CNS_EXTENT_CACHE[zarr_path] = extent_info
          return extent_info
  except Exception as e:
    logger.debug("Could not query CNS Zarr extent for %s: %s", zarr_path, e)

  return None


@dataclass
class WeatherSource:
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
  default_variables: List[str] = field(
      default_factory=lambda: ["total_precipitation", "2m_temperature"]
  )
  citation: str = ""

  def get_time_range(self) -> Dict[str, Any]:
    """Dynamically computes or queries the available temporal range for this dataset."""
    if self.cns_zarr_path:
      cns_extent = query_cns_zarr_extent(self.cns_zarr_path)
      if cns_extent:
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

  def to_dict(self) -> Dict[str, Any]:
    """Serializes source metadata with real-time dynamic time range."""
    time_range = self.get_time_range()
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
  """Returns serialized list of all weather sources with dynamic time bounds."""
  return [source.to_dict() for source in WEATHER_SOURCES.values()]

