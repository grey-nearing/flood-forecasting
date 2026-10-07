"""Weather data sources registry with dynamic CNS temporal range resolution.

Delegates core catalog definitions and Zarr metadata parsing to
`multimet.weather_fetcher.config`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

from multimet.weather_fetcher.config import (
    parse_zarr_metadata_time_extent,
    WEATHER_SOURCES as _BASE_WEATHER_SOURCES,
    WeatherSource as _BaseWeatherSource,
)

logger = logging.getLogger(__name__)

# In-memory cache for dynamically queried CNS time extents
_CNS_EXTENT_CACHE: Dict[str, Dict[str, Any]] = {}


def query_cns_zarr_extent(zarr_path: str) -> Optional[Dict[str, Any]]:
  """Queries the real time coordinate extent from a CNS Zarr store metadata."""
  if not zarr_path:
    return None
  if zarr_path in _CNS_EXTENT_CACHE:
    return _CNS_EXTENT_CACHE[zarr_path]

  if os.environ.get("OPENHYDRONET_OFFLINE_TESTS") == "1":
    return None

  try:
    cmd = ["fsspec", "cat", f"{zarr_path}/.zmetadata"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    if res.returncode == 0 and res.stdout:
      extent_info = parse_zarr_metadata_time_extent(res.stdout)
      if extent_info is not None:
        _CNS_EXTENT_CACHE[zarr_path] = extent_info
        return extent_info
  except Exception as e:  # pylint: disable=broad-except
    logger.debug("Could not query CNS Zarr extent for %s: %s", zarr_path, e)

  return None


class WeatherSource(_BaseWeatherSource):
  """Frontend adapter for WeatherSource with live CNS Zarr extent lookup."""

  def get_time_range(
      self, cns_extent: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Dynamically computes or queries the available temporal range."""
    if cns_extent is None and self.cns_zarr_path:
      cns_extent = query_cns_zarr_extent(self.cns_zarr_path)
    return super().get_time_range(cns_extent=cns_extent)


WEATHER_SOURCES: Dict[str, WeatherSource] = {
    key: WeatherSource(
        id=src.id,
        name=src.name,
        provider=src.provider,
        description=src.description,
        resolution=src.resolution,
        temporal_resolution=src.temporal_resolution,
        available_start=src.available_start,
        latency_days=src.latency_days,
        fixed_end=src.fixed_end,
        cns_zarr_path=src.cns_zarr_path,
        default_variables=list(src.default_variables),
        citation=src.citation,
    )
    for key, src in _BASE_WEATHER_SOURCES.items()
}


def get_weather_source(source_id: str = "era5") -> WeatherSource:
  """Retrieves a weather source by ID, defaulting to ERA5."""
  return WEATHER_SOURCES.get(source_id, WEATHER_SOURCES["era5"])


def list_weather_sources() -> List[Dict[str, Any]]:
  """Returns serialized list of all weather sources with dynamic time bounds."""
  return [source.to_dict() for source in WEATHER_SOURCES.values()]
