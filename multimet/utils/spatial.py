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

"""Spatial bounding box utilities for MultiMet meteorological forcing extraction.

Provides coordinate-system-aware geographic subsetting (bounding box slicing) for
1D coordinate arrays, Xarray Datasets, and precomputed ZonalWeightMatrix objects.
Supports both [-180, 180] and [0, 360] longitude conventions, ascending and
descending latitude axes, Prime Meridian (0°) crossings, and Antimeridian (180°)
crossings.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import shapely.geometry
import xarray as xr


@dataclasses.dataclass(frozen=True)
class BoundingBox:
  """Represents a geographic bounding box in standard WGS84 [-180, 180] degrees.

  Attributes:
    min_lon: Westernmost longitude in [-180.0, 180.0].
    min_lat: Southernmost latitude in [-90.0, 90.0].
    max_lon: Easternmost longitude in [-180.0, 180.0].
    max_lat: Northernmost latitude in [-90.0, 90.0].
  """

  min_lon: float
  min_lat: float
  max_lon: float
  max_lat: float

  def __post_init__(self) -> None:
    if self.min_lat > self.max_lat:
      raise ValueError(
          f"min_lat ({self.min_lat}) cannot be greater than max_lat ({self.max_lat})."
      )
    if self.min_lat < -90.0 or self.max_lat > 90.0:
      raise ValueError(
          f"Latitude bounds [{self.min_lat}, {self.max_lat}] out of range [-90, 90]."
      )
    if self.min_lon < -180.0 or self.max_lon > 180.0:
      raise ValueError(
          f"Longitude bounds [{self.min_lon}, {self.max_lon}] out of range [-180, 180]."
      )

  @property
  def crosses_antimeridian(self) -> bool:
    """True if box wraps across the 180° / -180° antimeridian (min_lon > max_lon)."""
    return self.min_lon > self.max_lon

  @property
  def crosses_prime_meridian(self) -> bool:
    """True if box spans across 0° Prime Meridian without crossing antimeridian."""
    return (
        not self.crosses_antimeridian
        and self.min_lon < 0.0
        and self.max_lon > 0.0
    )

  @property
  def lon_span(self) -> float:
    """Total longitudinal width in degrees."""
    if self.crosses_antimeridian:
      return (180.0 - self.min_lon) + (self.max_lon + 180.0)
    return self.max_lon - self.min_lon

  @property
  def lat_span(self) -> float:
    """Total latitudinal height in degrees."""
    return self.max_lat - self.min_lat

  def is_global(self, threshold_lon: float = 359.0, threshold_lat: float = 179.0) -> bool:
    """True if bounding box covers virtually the entire globe."""
    return self.lon_span >= threshold_lon and self.lat_span >= threshold_lat

  def to_tuple(self) -> Tuple[float, float, float, float]:
    """Returns (min_lon, min_lat, max_lon, max_lat)."""
    return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

  def as_dict(self) -> Dict[str, float]:
    """Returns dictionary representation of bounds."""
    return {
        "min_lon": self.min_lon,
        "min_lat": self.min_lat,
        "max_lon": self.max_lon,
        "max_lat": self.max_lat,
    }

  def buffer(self, degrees: float) -> BoundingBox:
    """Returns a new BoundingBox expanded by `degrees` in all directions.

    Clamps latitudes to [-90, 90] and longitudes to [-180, 180].

    Args:
      degrees: Angular buffer in degrees (>= 0).

    Returns:
      Buffered BoundingBox instance.
    """
    if degrees == 0.0:
      return self
    new_min_lat = max(-90.0, self.min_lat - degrees)
    new_max_lat = min(90.0, self.max_lat + degrees)

    if self.lon_span + 2.0 * degrees >= 360.0:
      return BoundingBox(-180.0, new_min_lat, 180.0, new_max_lat)

    if self.crosses_antimeridian:
      new_min_lon = self.min_lon - degrees
      new_max_lon = self.max_lon + degrees
      if new_min_lon <= new_max_lon:
        return BoundingBox(-180.0, new_min_lat, 180.0, new_max_lat)
      return BoundingBox(new_min_lon, new_min_lat, new_max_lon, new_max_lat)
    else:
      new_min_lon = max(-180.0, self.min_lon - degrees)
      new_max_lon = min(180.0, self.max_lon + degrees)
      return BoundingBox(new_min_lon, new_min_lat, new_max_lon, new_max_lat)

  def to_0_360_ranges(self) -> List[Tuple[float, float]]:
    """Converts longitude bounds to one or two [start, stop] intervals in [0, 360].

    Returns:
      List of (start_lon_360, stop_lon_360) tuples.
    """
    if self.lon_span >= 360.0 or (self.min_lon <= -180.0 and self.max_lon >= 180.0):
      return [(0.0, 360.0)]

    if self.crosses_antimeridian:
      # e.g. min_lon = 170, max_lon = -170 -> [170, 190] in [0, 360]
      start_360 = self.min_lon % 360.0
      stop_360 = self.max_lon % 360.0
      return [(start_360, stop_360)]

    if self.crosses_prime_meridian:
      # e.g. min_lon = -10, max_lon = 20 -> [350, 360] and [0, 20]
      return [(self.min_lon + 360.0, 360.0), (0.0, self.max_lon)]

    start_360 = self.min_lon + 360.0 if self.min_lon < 0.0 else self.min_lon
    stop_360 = self.max_lon + 360.0 if self.max_lon < 0.0 else self.max_lon
    return [(start_360, stop_360)]

  @classmethod
  def from_geodataframe(
      cls, gdf: gpd.GeoDataFrame, buffer_degrees: float = 0.25
  ) -> BoundingBox:
    """Computes a BoundingBox from a GeoDataFrame in EPSG:4326.

    Args:
      gdf: GeoDataFrame of catchment polygons.
      buffer_degrees: Safety margin in degrees around total bounds.

    Returns:
      BoundingBox instance.
    """
    if gdf.empty:
      raise ValueError("Cannot compute BoundingBox from an empty GeoDataFrame.")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
      gdf = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    box = cls(
        min_lon=max(-180.0, float(minx)),
        min_lat=max(-90.0, float(miny)),
        max_lon=min(180.0, float(maxx)),
        max_lat=min(90.0, float(maxy)),
    )
    return box.buffer(buffer_degrees)

  @classmethod
  def from_geometry(
      cls,
      geom: shapely.geometry.base.BaseGeometry,
      buffer_degrees: float = 0.25,
  ) -> BoundingBox:
    """Computes a BoundingBox from a single Shapely geometry."""
    minx, miny, maxx, maxy = geom.bounds
    box = cls(
        min_lon=max(-180.0, float(minx)),
        min_lat=max(-90.0, float(miny)),
        max_lon=min(180.0, float(maxx)),
        max_lat=min(90.0, float(maxy)),
    )
    return box.buffer(buffer_degrees)

  @classmethod
  def from_tuple(
      cls,
      bounds: Union[Sequence[float], Tuple[float, float, float, float], BoundingBox],
      buffer_degrees: float = 0.0,
  ) -> BoundingBox:
    """Coerces a 4-tuple (min_lon, min_lat, max_lon, max_lat) or BoundingBox."""
    if isinstance(bounds, BoundingBox):
      return bounds.buffer(buffer_degrees)
    if len(bounds) != 4:
      raise ValueError(
          f"Expected 4-element bounds (min_lon, min_lat, max_lon, max_lat), got {bounds}."
      )
    box = cls(
        min_lon=float(bounds[0]),
        min_lat=float(bounds[1]),
        max_lon=float(bounds[2]),
        max_lat=float(bounds[3]),
    )
    return box.buffer(buffer_degrees)


def coerce_bounding_box(
    bounds: Union[
        BoundingBox,
        gpd.GeoDataFrame,
        shapely.geometry.base.BaseGeometry,
        Sequence[float],
        Tuple[float, float, float, float],
    ],
    buffer_degrees: float = 0.0,
) -> BoundingBox:
  """Coerces any supported spatial bounds representation into a BoundingBox."""
  if isinstance(bounds, BoundingBox):
    return bounds.buffer(buffer_degrees)
  if isinstance(bounds, gpd.GeoDataFrame):
    return BoundingBox.from_geodataframe(bounds, buffer_degrees=buffer_degrees)
  if isinstance(bounds, shapely.geometry.base.BaseGeometry):
    return BoundingBox.from_geometry(bounds, buffer_degrees=buffer_degrees)
  return BoundingBox.from_tuple(bounds, buffer_degrees=buffer_degrees)


def find_lat_lon_dims(
    ds: Union[xr.Dataset, xr.DataArray],
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
) -> Tuple[str, str]:
  """Detects latitude and longitude coordinate names in an Xarray object."""
  coords_and_dims = set(ds.coords) | set(ds.dims)
  if lat_dim is None:
    for candidate in ("latitude", "lat", "y"):
      if candidate in coords_and_dims:
        lat_dim = candidate
        break
  if lon_dim is None:
    for candidate in ("longitude", "lon", "x"):
      if candidate in coords_and_dims:
        lon_dim = candidate
        break
  if lat_dim is None or lon_dim is None:
    raise ValueError(
        f"Could not automatically detect lat/lon dimensions in {list(coords_and_dims)}."
    )
  return lat_dim, lon_dim


def slice_coordinates_by_bounds(
    lats: np.ndarray,
    lons: np.ndarray,
    bounds: Union[
        BoundingBox,
        gpd.GeoDataFrame,
        shapely.geometry.base.BaseGeometry,
        Sequence[float],
    ],
    buffer_degrees: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Finds 1D coordinate subsets and integer index arrays matching a BoundingBox.

  Automatically handles ascending/descending latitudes and [-180, 180] or [0, 360]
  longitudes. Only falls back to the nearest grid cell if a sub-grid-cell basin
  actually lies within half a grid cell (`<= dlat/2`, `<= dlon/2`) of that cell
  center; basins outside the grid domain return empty index arrays.

  Args:
    lats: 1D numpy array of latitude coordinates.
    lons: 1D numpy array of longitude coordinates.
    bounds: BoundingBox, GeoDataFrame, Shapely geometry, or 4-tuple.
    buffer_degrees: Additional angular buffer in degrees.

  Returns:
    Tuple of (sub_lats, sub_lons, lat_indices, lon_indices).
  """
  bbox = coerce_bounding_box(bounds, buffer_degrees=buffer_degrees)
  lats = np.asarray(lats)
  lons = np.asarray(lons)

  dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.0
  dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.0

  # 1. Latitude mask
  lat_mask = (lats >= (bbox.min_lat - 1e-6)) & (lats <= (bbox.max_lat + 1e-6))
  lat_indices = np.where(lat_mask)[0]
  if len(lat_indices) == 0 and len(lats) > 0:
    mid_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    nearest_lat = int(np.argmin(np.abs(lats - mid_lat)))
    if abs(float(lats[nearest_lat]) - mid_lat) <= dlat / 2.0 + 1e-6:
      lat_indices = np.array([nearest_lat])

  # 2. Longitude mask (detect [0, 360] vs [-180, 180])
  is_0_360 = bool(np.any(lons > 180.0 + 1e-4))
  if is_0_360:
    lon_mask = np.zeros(len(lons), dtype=bool)
    for start_360, stop_360 in bbox.to_0_360_ranges():
      lon_mask |= (lons >= (start_360 - 1e-6)) & (lons <= (stop_360 + 1e-6))
  else:
    if bbox.crosses_antimeridian:
      lon_mask = (lons >= (bbox.min_lon - 1e-6)) | (lons <= (bbox.max_lon + 1e-6))
    else:
      lon_mask = (lons >= (bbox.min_lon - 1e-6)) & (lons <= (bbox.max_lon + 1e-6))

  lon_indices = np.where(lon_mask)[0]
  if len(lon_indices) == 0 and len(lons) > 0:
    mid_lon = (
        ((bbox.min_lon + bbox.max_lon) / 2.0) % 360.0
        if is_0_360
        else (bbox.min_lon + bbox.max_lon) / 2.0
    )
    nearest_lon = int(np.argmin(np.abs(lons - mid_lon)))
    raw_lon_diff = abs(float(lons[nearest_lon]) - mid_lon)
    lon_diff = min(raw_lon_diff, 360.0 - raw_lon_diff)
    if lon_diff <= dlon / 2.0 + 1e-6:
      lon_indices = np.array([nearest_lon])

  return lats[lat_indices], lons[lon_indices], lat_indices, lon_indices


def slice_dataset_by_bounds(
    ds: Union[xr.Dataset, xr.DataArray],
    bounds: Union[
        BoundingBox,
        gpd.GeoDataFrame,
        shapely.geometry.base.BaseGeometry,
        Sequence[float],
    ],
    buffer_degrees: float = 0.0,
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
    target_lon_range: str = "auto",
    normalize_lons_to_180: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
  """Subsets an Xarray Dataset or DataArray to a geographic BoundingBox."""
  bbox = coerce_bounding_box(bounds, buffer_degrees=buffer_degrees)
  if bbox.is_global() and bbox.min_lat <= -89.0 and bbox.max_lat >= 89.0:
    return ds

  lat_name, lon_name = find_lat_lon_dims(ds, lat_dim=lat_dim, lon_dim=lon_dim)
  lats = ds[lat_name].values
  lons = ds[lon_name].values

  _, _, lat_indices, lon_indices = slice_coordinates_by_bounds(
      lats, lons, bbox, buffer_degrees=0.0
  )

  # Use contiguous slice when possible for optimal Zarr/Dask chunk fusion
  if len(lat_indices) > 1 and np.all(np.diff(lat_indices) == 1):
    lat_indexer: Union[slice, np.ndarray] = slice(
        int(lat_indices[0]), int(lat_indices[-1]) + 1
    )
  else:
    lat_indexer = lat_indices

  if len(lon_indices) > 1 and np.all(np.diff(lon_indices) == 1):
    lon_indexer: Union[slice, np.ndarray] = slice(
        int(lon_indices[0]), int(lon_indices[-1]) + 1
    )
  else:
    lon_indexer = lon_indices

  sliced = ds.isel({lat_name: lat_indexer, lon_name: lon_indexer})

  is_0_360 = bool(np.any(lons > 180.0 + 1e-4))
  if is_0_360 and normalize_lons_to_180 and target_lon_range in ("auto", "minus180_180"):
    sub_lons = sliced[lon_name].values
    lons_180 = np.where(sub_lons > 180.0, sub_lons - 360.0, sub_lons)
    sliced = sliced.assign_coords({lon_name: lons_180})
    if len(lons_180) > 1 and not np.all(np.diff(lons_180) > 0):
      sliced = sliced.sortby(lon_name)
  elif not is_0_360 and target_lon_range == "0_360":
    sub_lons = sliced[lon_name].values
    lons_360 = sub_lons % 360.0
    sliced = sliced.assign_coords({lon_name: lons_360}).sortby(lon_name)

  return sliced
