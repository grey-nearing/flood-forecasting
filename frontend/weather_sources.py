"""Registry of historical weather datasets served by the OpenHydroNet UI.

This module is frontend-only: it describes the archival datasets that
``frontend.historical_zarr`` can extract from (ERA5, CPC, IMERG, ...), where
they live on CNS, and which date range they cover. It is independent of
``multimet.weather_fetcher``, which only deals with real-time forecast feeds.

Temporal coverage is resolved in two ways:

* Stores with a ``cns_zarr_path`` are probed once per process via
  ``fsspec cat <store>/.zmetadata`` and the consolidated metadata is parsed
  with :func:`parse_zarr_metadata_time_extent`.
* Otherwise the statically declared ``available_start`` / ``fixed_end`` /
  ``latency_days`` values are used.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CNS_ROOT = "gs://open-multimet/gridded-data-archives"
_METADATA_TIMEOUT_S = 5.0

# Successful .zmetadata lookups, keyed by store path (process lifetime).
_CNS_EXTENT_CACHE: Dict[str, Dict[str, Any]] = {}
# Store paths whose lookup failure has already been logged.
_CNS_EXTENT_WARNED: set[str] = set()


def parse_zarr_metadata_time_extent(
    zmetadata_json: str,
) -> Optional[Dict[str, Any]]:
  """Parses the daily time extent from consolidated Zarr ``.zmetadata``.

  Consolidated metadata only exposes the ``time`` array shape and its CF
  ``units`` attribute, not the coordinate values. The extent is therefore
  derived under the convention used by xarray-written daily archives: the
  epoch of ``"days since <YYYY-MM-DD>"`` is the first timestamp and the axis
  is contiguous daily, so the last timestamp is ``epoch + (n - 1)`` days.

  Args:
    zmetadata_json: Raw JSON text of the ``.zmetadata`` document.

  Returns:
    ``{"start_date", "end_date", "n_timesteps", "total_years", "is_dynamic"}``
    or ``None`` when the store has no daily CF time axis (empty document,
    missing ``time`` array, or units that are not ``days since``).

  Raises:
    json.JSONDecodeError: If ``zmetadata_json`` is not valid JSON.
    ValueError: If the epoch date in the units attribute is malformed.
  """
  if not zmetadata_json:
    return None
  parsed = json.loads(zmetadata_json)
  meta = parsed.get("metadata", {})
  time_arr = meta.get("time/.zarray", {})
  time_attrs = meta.get("time/.zattrs", {})
  shape = time_arr.get("shape", [])
  if not shape or int(shape[0]) <= 0:
    return None
  n_steps = int(shape[0])
  units = str(time_attrs.get("units", ""))
  if "days since " not in units:
    return None
  base_str = units.split("days since ", 1)[1].split()[0]
  base_dt = dt.datetime.strptime(base_str[:10], "%Y-%m-%d")
  start_date = base_dt.strftime("%Y-%m-%d")
  end_date = (base_dt + dt.timedelta(days=n_steps - 1)).strftime("%Y-%m-%d")
  return {
      "start_date": start_date,
      "end_date": end_date,
      "n_timesteps": n_steps,
      "total_years": round(n_steps / 365.25, 1),
      "is_dynamic": True,
  }


def query_cns_zarr_extent(zarr_path: str) -> Optional[Dict[str, Any]]:
  """Reads the time extent of a CNS Zarr store from its ``.zmetadata``.

  Successful lookups are cached for the lifetime of the process. Failures
  (``fsspec`` missing, CNS unreachable, unparsable metadata) are logged once
  per store and reported as ``None`` so callers fall back to the declared
  static coverage; they are not cached so a later call can recover.

  Args:
    zarr_path: Absolute ``gs://open-multimet/data`` path of the Zarr store.

  Returns:
    The parsed extent (see :func:`parse_zarr_metadata_time_extent`) or
    ``None`` when it cannot be determined.
  """
  if not zarr_path:
    return None
  cached = _CNS_EXTENT_CACHE.get(zarr_path)
  if cached is not None:
    return cached
  if os.environ.get("OPENHYDRONET_OFFLINE_TESTS") == "1":
    return None

  cmd = ["fsspec", "cat", f"{zarr_path}/.zmetadata"]
  try:
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_METADATA_TIMEOUT_S,
        check=True,
    )
    extent = parse_zarr_metadata_time_extent(res.stdout)
  except (OSError, subprocess.SubprocessError, ValueError) as exc:
    # json.JSONDecodeError is a ValueError subclass.
    if zarr_path not in _CNS_EXTENT_WARNED:
      _CNS_EXTENT_WARNED.add(zarr_path)
      logger.warning(
          "Could not read CNS Zarr time extent for %s (%s: %s); using the"
          " declared static coverage instead.",
          zarr_path,
          type(exc).__name__,
          exc,
      )
    return None

  if extent is not None:
    _CNS_EXTENT_CACHE[zarr_path] = extent
  return extent


@dataclasses.dataclass
class WeatherSource:
  """Metadata and temporal coverage of one archival weather dataset.

  Attributes:
    id: Stable identifier used by the API (``/api/weather/sources``).
    name: Human-readable dataset name.
    provider: Producing institution.
    description: One-sentence summary shown in the UI.
    resolution: Native horizontal resolution as displayed text.
    temporal_resolution: Native/derived temporal resolution as displayed text.
    available_start: First available date (``YYYY-MM-DD``).
    latency_days: Publication delay used to estimate the last available date
      when no live extent is known.
    fixed_end: Last date of a closed archive (``YYYY-MM-DD``), if any.
    cns_zarr_path: CNS Zarr store probed for the live time extent, if any.
    default_variables: Variables extracted when the caller does not pick any.
    citation: Reference to cite for the dataset.
  """

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
  default_variables: List[str] = dataclasses.field(default_factory=list)
  citation: str = ""

  def get_time_range(
      self, cns_extent: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Returns the available date range of this dataset.

    Args:
      cns_extent: Pre-fetched extent (as returned by
        :func:`query_cns_zarr_extent`). When omitted and ``cns_zarr_path`` is
        set, the store is queried; if that fails the declared static coverage
        is used.

    Returns:
      ``{"start_date", "end_date", "latency_days", "total_years",
      "is_dynamic"}``. ``is_dynamic`` is ``True`` when the end date moves
      with time (live extent or latency-based estimate).
    """
    if cns_extent is None and self.cns_zarr_path:
      cns_extent = query_cns_zarr_extent(self.cns_zarr_path)
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
      latest = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
          days=self.latency_days
      )
      end_date = latest.strftime("%Y-%m-%d")
    start_dt = dt.datetime.strptime(self.available_start, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d")
    return {
        "start_date": self.available_start,
        "end_date": end_date,
        "latency_days": self.latency_days,
        "total_years": round((end_dt - start_dt).days / 365.25, 1),
        "is_dynamic": self.fixed_end is None,
    }

  def to_dict(
      self, cns_extent: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Serialises the source metadata together with its time range."""
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
        "is_dynamic": time_range["is_dynamic"],
        "cns_zarr_path": self.cns_zarr_path or "",
        "default_variables": list(self.default_variables),
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
        cns_zarr_path=f"{_CNS_ROOT}/cpc/precip.zarr",
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
        cns_zarr_path=f"{_CNS_ROOT}/imerg/precip.zarr",
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
        cns_zarr_path=f"{_CNS_ROOT}/era5/tp.zarr",
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
            "AI weather forecast model providing archived temperature and"
            " precipitation forecasts."
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
            "High-resolution regional reanalysis for the European domain at"
            " 5.5 km resolution (1984 to 2021)."
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
  """Returns the registered source for ``source_id``.

  Args:
    source_id: Registry key (see :data:`WEATHER_SOURCES`).

  Raises:
    KeyError: If ``source_id`` is not registered. Unknown identifiers are
      never silently remapped to another dataset.
  """
  if source_id not in WEATHER_SOURCES:
    raise KeyError(
        f"Unknown weather source {source_id!r}; registered sources:"
        f" {sorted(WEATHER_SOURCES)}"
    )
  return WEATHER_SOURCES[source_id]


def list_weather_sources() -> List[Dict[str, Any]]:
  """Serialises every registered source with its resolved time range."""
  return [source.to_dict() for source in WEATHER_SOURCES.values()]
