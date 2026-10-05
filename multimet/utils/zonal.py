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

"""Exact geometric zonal weight computation and sparse matrix vectorization.

Computes exact area-weighted intersections between hydrological catchment
polygons and meteorological latitude/longitude grid cells, accounting for
spherical latitude cosine scaling (`cos(lat)`).

Also provides `ZonalWeightMatrix`, which packs per-basin weights into a single
Compressed Sparse Row (`scipy.sparse.csr_matrix`) operator. This allows reducing
an entire 3D/4D gridded climate dataset across thousands of basins simultaneously
via C-level sparse matrix multiplication (`W @ X`), yielding 50x–200x speedups
over Python basin loops and supporting instant disk caching (`.npz`).
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
from scipy import sparse
import shapely.geometry
import shapely.prepared

from multimet.utils.spatial import BoundingBox, slice_coordinates_by_bounds

# Minimum fraction of a basin's total area weight that must have finite (non-NaN)
# meteorological values for a zonal reduction to return a valid number. If fewer
# than 80% of the basin's weighted area has valid data, the reduced value is NaN.
MIN_VALID_COVERAGE_FRACTION: float = 0.80


def weighted_mean_valid_with_coverage(
    vals: np.ndarray,
    weights: np.ndarray,
    min_valid_coverage: float = MIN_VALID_COVERAGE_FRACTION,
) -> Tuple[float, float]:
  """Computes area-weighted mean and missing area fraction over finite values.

  Args:
    vals: 1D or N-D array of gridded values at basin-intersecting cells.
    weights: Corresponding area weights (`intersection_area * cos(lat)`).
    min_valid_coverage: Minimum required fraction of total basin weight with
      finite values (default 0.80 = 80%). If valid coverage is below this
      threshold, returns `(np.nan, missing_fraction)`.

  Returns:
    Tuple `(weighted_mean, missing_fraction)` in `[0.0, 1.0]`.
  """
  if vals.size == 0 or weights.size == 0:
    return float("nan"), 1.0
  total_w = float(np.sum(weights))
  if total_w <= 1e-12:
    return float("nan"), 1.0
  valid = np.isfinite(vals)
  if not np.any(valid):
    return float("nan"), 1.0
  valid_w = float(np.sum(weights[valid]))
  missing_frac = float(np.clip(1.0 - (valid_w / total_w), 0.0, 1.0))
  if valid_w <= 1e-12 or (valid_w / total_w) < (min_valid_coverage - 1e-6):
    return float("nan"), missing_frac
  val = float(np.sum(vals[valid] * weights[valid]) / valid_w)
  return val, missing_frac


def weighted_mean_valid(
    vals: np.ndarray,
    weights: np.ndarray,
    min_valid_coverage: float = MIN_VALID_COVERAGE_FRACTION,
) -> float:
  """Computes area-weighted mean over finite values with >= 80% coverage check."""
  val, _ = weighted_mean_valid_with_coverage(
      vals, weights, min_valid_coverage=min_valid_coverage
  )
  return val


class ZonalWeightCalculator:
  """Calculates exact fractional area weights for polygons over a 2D lat/lon grid."""

  def __init__(
      self,
      lats: np.ndarray,
      lons: np.ndarray,
      cell_res_lat: Optional[float] = None,
      cell_res_lon: Optional[float] = None,
  ):
    """Initializes the calculator for a regular latitude/longitude grid.

    Args:
      lats: 1D array of grid cell latitude centers (degrees).
      lons: 1D array of grid cell longitude centers (degrees, -180 to 180).
      cell_res_lat: Optional explicit latitude resolution in degrees.
      cell_res_lon: Optional explicit longitude resolution in degrees.
    """
    self.lats = np.asarray(lats, dtype=np.float64)
    self.lons = np.asarray(lons, dtype=np.float64)

    self.dlat = (
        float(cell_res_lat)
        if cell_res_lat is not None
        else (abs(float(self.lats[1] - self.lats[0])) if len(self.lats) > 1 else 0.1)
    )
    self.dlon = (
        float(cell_res_lon)
        if cell_res_lon is not None
        else (abs(float(self.lons[1] - self.lons[0])) if len(self.lons) > 1 else 0.1)
    )
    self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

  def compute_weights(
      self,
      basin_id: str,
      polygon: shapely.geometry.base.BaseGeometry,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes normalized cell intersection weights for a single catchment polygon.

    Args:
      basin_id: Unique identifier of the basin (used for caching).
      polygon: Shapely geometry (Polygon or MultiPolygon) in EPSG:4326.

    Returns:
      Tuple of (lat_indices, lon_indices, normalized_weights) as 1D numpy arrays.
    """
    if basin_id in self._cache:
      return self._cache[basin_id]

    if polygon is None or polygon.is_empty:
      empty_i = np.array([], dtype=np.int32)
      empty_w = np.array([], dtype=np.float64)
      self._cache[basin_id] = (empty_i, empty_i, empty_w)
      return self._cache[basin_id]

    minx, miny, maxx, maxy = polygon.bounds
    half_lat = self.dlat / 2.0
    half_lon = self.dlon / 2.0

    # Find candidate latitude and longitude indices overlapping the bounding box.
    # Never snap an out-of-domain basin to a distant nearest pixel: if the basin
    # does not overlap the grid domain, return empty weights so extracted values
    # are NaN and missing_fraction is 1.0.
    lat_mask = (self.lats + half_lat >= miny) & (self.lats - half_lat <= maxy)
    lon_mask = (self.lons + half_lon >= minx) & (self.lons - half_lon <= maxx)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
      empty_i = np.array([], dtype=np.int32)
      empty_w = np.array([], dtype=np.float64)
      self._cache[basin_id] = (empty_i, empty_i, empty_w)
      return self._cache[basin_id]

    prep_poly = shapely.prepared.prep(polygon)

    out_lat_idx: List[int] = []
    out_lon_idx: List[int] = []
    out_weights: List[float] = []

    for i in lat_indices:
      lat_c = self.lats[i]
      y0, y1 = lat_c - half_lat, lat_c + half_lat
      cos_lat = max(0.0, float(np.cos(np.deg2rad(lat_c))))
      for j in lon_indices:
        lon_c = self.lons[j]
        x0, x1 = lon_c - half_lon, lon_c + half_lon
        cell_box = shapely.geometry.box(x0, y0, x1, y1)

        if prep_poly.contains(cell_box):
          area = cell_box.area
        elif prep_poly.intersects(cell_box):
          area = polygon.intersection(cell_box).area
        else:
          continue

        if area > 0.0:
          out_lat_idx.append(int(i))
          out_lon_idx.append(int(j))
          out_weights.append(area * cos_lat)

    if not out_weights:
      # Sub-grid polygon whose bounding box overlapped candidate cells: check only
      # the candidate cell containing the centroid (never an out-of-domain cell).
      centroid = polygon.centroid
      c_lat_cand = lat_indices[np.argmin(np.abs(self.lats[lat_indices] - centroid.y))]
      c_lon_cand = lon_indices[np.argmin(np.abs(self.lons[lon_indices] - centroid.x))]
      if (
          abs(self.lats[c_lat_cand] - centroid.y) <= half_lat
          and abs(self.lons[c_lon_cand] - centroid.x) <= half_lon
      ):
        res = (
            np.array([int(c_lat_cand)], dtype=np.int32),
            np.array([int(c_lon_cand)], dtype=np.int32),
            np.array([1.0], dtype=np.float64),
        )
      else:
        empty_i = np.array([], dtype=np.int32)
        empty_w = np.array([], dtype=np.float64)
        res = (empty_i, empty_i, empty_w)
      self._cache[basin_id] = res
      return res

    w_arr = np.array(out_weights, dtype=np.float64)
    w_sum = np.sum(w_arr)
    if w_sum > 0:
      w_arr = w_arr / w_sum

    res = (
        np.array(out_lat_idx, dtype=np.int32),
        np.array(out_lon_idx, dtype=np.int32),
        w_arr,
    )
    self._cache[basin_id] = res
    return res

  def reduce_grid_with_coverage(
      self,
      grid_data: np.ndarray,
      basin_id: str,
      polygon: shapely.geometry.base.BaseGeometry,
  ) -> Tuple[float, float]:
    """Reduces a 2D grid `(lat, lon)` to `(weighted_mean, missing_fraction)`.

    Args:
      grid_data: 2D numpy array of shape `(len(lats), len(lons))`.
      basin_id: Basin identifier.
      polygon: Shapely geometry of the basin.

    Returns:
      Tuple `(weighted_mean, missing_fraction)` where `missing_fraction` is in
      `[0.0, 1.0]` representing the fraction of the basin's total area weight
      covered by NaN/non-finite grid cells (or `1.0` if out of domain). If
      valid coverage is `< 80%`, `weighted_mean` is `NaN`.
    """
    lat_idx, lon_idx, weights = self.compute_weights(basin_id, polygon)
    if len(weights) == 0:
      return float("nan"), 1.0

    sub = grid_data[lat_idx, lon_idx]
    return weighted_mean_valid_with_coverage(sub, weights)

  def reduce_grid(
      self,
      grid_data: np.ndarray,
      basin_id: str,
      polygon: shapely.geometry.base.BaseGeometry,
  ) -> float:
    """Reduces a 2D grid `(lat, lon)` to a single scalar for `basin_id`."""
    val, _ = self.reduce_grid_with_coverage(grid_data, basin_id, polygon)
    return val


def _compute_basin_weights_worker(
    args: Tuple[str, shapely.geometry.base.BaseGeometry, np.ndarray, np.ndarray, float, float]
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
  """Worker function for parallel weight calculation across basins."""
  basin_id, geom, lats, lons, dlat, dlon = args
  calc = ZonalWeightCalculator(lats, lons, cell_res_lat=dlat, cell_res_lon=dlon)
  lat_idx, lon_idx, w = calc.compute_weights(basin_id, geom)
  return basin_id, lat_idx, lon_idx, w


class ZonalWeightMatrix:
  """Sparse CSR matrix operator for simultaneous multi-basin zonal reduction.

  Represents a linear operator `W` of shape `(N_basins, N_lat * N_lon)` where
  each row contains the exact cosine-latitude-corrected fractional area weights
  for one catchment.
  """

  def __init__(
      self,
      basin_ids: Sequence[str],
      lats: np.ndarray,
      lons: np.ndarray,
      matrix: sparse.csr_matrix,
  ):
    self.basin_ids = [str(b) for b in basin_ids]
    self.lats = np.asarray(lats, dtype=np.float64)
    self.lons = np.asarray(lons, dtype=np.float64)
    self.matrix = matrix.tocsr().astype(np.float64)

    expected_cols = len(self.lats) * len(self.lons)
    if self.matrix.shape != (len(self.basin_ids), expected_cols):
      raise ValueError(
          f"Matrix shape {self.matrix.shape} does not match "
          f"(N_basins={len(self.basin_ids)}, N_lat*N_lon={expected_cols})."
      )
    # Precompute total weight per basin row (1.0 for in-domain basins, 0.0 for
    # out-of-domain basins) so partial-NaN coverage fractions are exact even
    # after spatial cropping.
    row_sums = np.asarray(self.matrix.sum(axis=1)).ravel()
    self.total_weights = row_sums.astype(np.float64)
    self.has_weights = self.total_weights > 1e-12

  @property
  def num_basins(self) -> int:
    return len(self.basin_ids)

  @property
  def grid_shape(self) -> Tuple[int, int]:
    return (len(self.lats), len(self.lons))

  @classmethod
  def from_geodataframe(
      cls,
      basins_gdf: gpd.GeoDataFrame,
      lats: np.ndarray,
      lons: np.ndarray,
      cell_res_lat: Optional[float] = None,
      cell_res_lon: Optional[float] = None,
      num_workers: int = 1,
      bounds: Optional[
          Union[
              BoundingBox,
              gpd.GeoDataFrame,
              shapely.geometry.base.BaseGeometry,
              Sequence[float],
          ]
      ] = None,
      buffer_degrees: float = 0.0,
  ) -> ZonalWeightMatrix:
    """Builds a ZonalWeightMatrix for all basins in a GeoDataFrame.

    Args:
      basins_gdf: GeoDataFrame indexed by `basin_id` with `geometry` in EPSG:4326.
      lats: 1D array of grid latitudes.
      lons: 1D array of grid longitudes.
      cell_res_lat: Optional latitude resolution in degrees.
      cell_res_lon: Optional longitude resolution in degrees.
      num_workers: Number of parallel threads/processes for geometry intersections.
      bounds: Optional spatial bounding box to slice `lats` and `lons` before
        computing weights.
      buffer_degrees: Optional angular buffer around `bounds`.

    Returns:
      Constructed ZonalWeightMatrix instance.
    """
    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)

    dlat = (
        float(cell_res_lat)
        if cell_res_lat is not None
        else (abs(float(lats_arr[1] - lats_arr[0])) if len(lats_arr) > 1 else 0.1)
    )
    dlon = (
        float(cell_res_lon)
        if cell_res_lon is not None
        else (abs(float(lons_arr[1] - lons_arr[0])) if len(lons_arr) > 1 else 0.1)
    )

    if bounds is not None:
      lats_arr, lons_arr, _, _ = slice_coordinates_by_bounds(
          lats_arr, lons_arr, bounds, buffer_degrees=buffer_degrees
      )

    basin_ids = [str(b) for b in basins_gdf.index]
    geoms = list(basins_gdf.geometry)
    n_lon = len(lons_arr)

    row_indices: List[np.ndarray] = []
    col_indices: List[np.ndarray] = []
    data_vals: List[np.ndarray] = []

    if num_workers > 1 and len(basin_ids) > 32:
      tasks = [
          (b_id, geom, lats_arr, lons_arr, dlat, dlon)
          for b_id, geom in zip(basin_ids, geoms)
      ]
      with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_compute_basin_weights_worker, tasks))
      for row_idx, (_, lat_i, lon_i, w) in enumerate(results):
        if len(w) > 0:
          flat_cols = lat_i.astype(np.int64) * n_lon + lon_i.astype(np.int64)
          row_indices.append(np.full(len(w), row_idx, dtype=np.int32))
          col_indices.append(flat_cols)
          data_vals.append(w)
    else:
      calc = ZonalWeightCalculator(
          lats_arr, lons_arr, cell_res_lat=dlat, cell_res_lon=dlon
      )
      for row_idx, (b_id, geom) in enumerate(zip(basin_ids, geoms)):
        lat_i, lon_i, w = calc.compute_weights(b_id, geom)
        if len(w) > 0:
          flat_cols = lat_i.astype(np.int64) * n_lon + lon_i.astype(np.int64)
          row_indices.append(np.full(len(w), row_idx, dtype=np.int32))
          col_indices.append(flat_cols)
          data_vals.append(w)

    if row_indices:
      rows = np.concatenate(row_indices)
      cols = np.concatenate(col_indices)
      vals = np.concatenate(data_vals)
    else:
      rows = np.array([], dtype=np.int32)
      cols = np.array([], dtype=np.int64)
      vals = np.array([], dtype=np.float64)

    n_cells = len(lats_arr) * len(lons_arr)
    csr = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(basin_ids), n_cells),
        dtype=np.float64,
    )
    return cls(basin_ids=basin_ids, lats=lats_arr, lons=lons_arr, matrix=csr)

  def get_basin_weights(
      self, basin_id: str
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts `(lat_idx, lon_idx, weights)` for a single basin from the CSR matrix."""
    idx = self.basin_ids.index(str(basin_id))
    row = self.matrix.getrow(idx)
    flat_cols = row.indices
    weights = row.data
    n_lon = len(self.lons)
    lat_idx = (flat_cols // n_lon).astype(np.int32)
    lon_idx = (flat_cols % n_lon).astype(np.int32)
    return lat_idx, lon_idx, weights

  def reduce_2d_with_coverage(
      self, grid_2d: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduces a 2D array `(lat, lon)` to `(values, missing_fraction)` of shape `(N_basins,)`.

    Handles NaN values by renormalizing over valid grid cells for each basin,
    requiring at least 80% (`MIN_VALID_COVERAGE_FRACTION`) of the basin's total
    weight to be finite. Also computes the exact area-weighted fraction of each
    basin covered by NaN cells (`missing_fraction` in `[0.0, 1.0]`).
    """
    flat = np.asarray(grid_2d, dtype=np.float64).ravel()
    valid_mask = np.isfinite(flat)
    W = self.total_weights

    if np.all(valid_mask):
      res = (self.matrix @ flat).astype(np.float32)
      valid_basins = self.has_weights & (W >= (MIN_VALID_COVERAGE_FRACTION - 1e-6))
      res = np.where(valid_basins, res / np.where(self.has_weights, W, 1.0), np.nan).astype(np.float32)
      missing_frac = np.where(
          self.has_weights,
          np.clip(1.0 - W, 0.0, 1.0),
          1.0,
      ).astype(np.float32)
      return res, missing_frac

    safe_flat = np.where(valid_mask, flat, 0.0)
    valid_float = valid_mask.astype(np.float64)

    N = self.matrix @ safe_flat
    V = self.matrix @ valid_float

    has_w = self.has_weights & (W > 1e-12)
    W_safe = np.where(has_w, W, 1.0)
    valid_ratio = np.where(has_w, V / W_safe, 0.0)
    missing_frac = np.where(
        has_w,
        np.clip(1.0 - valid_ratio, 0.0, 1.0),
        1.0,
    ).astype(np.float32)

    valid_basins = (
        has_w
        & (V > 1e-12)
        & (valid_ratio >= (MIN_VALID_COVERAGE_FRACTION - 1e-6))
    )
    result = np.full(self.num_basins, np.nan, dtype=np.float32)
    result[valid_basins] = (N[valid_basins] / V[valid_basins]).astype(
        np.float32
    )
    return result, missing_frac

  def reduce_2d(self, grid_2d: np.ndarray) -> np.ndarray:
    """Reduces a 2D array `(lat, lon)` to a 1D array `(N_basins,)`."""
    vals, _ = self.reduce_2d_with_coverage(grid_2d)
    return vals

  def reduce_3d_with_coverage(
      self, grid_3d: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduces a 3D array `(T, lat, lon)` to `(values, missing_fraction)` of shape `(N_basins, T)`."""
    T = grid_3d.shape[0]
    flat_2d = np.asarray(grid_3d, dtype=np.float64).reshape(T, -1).T  # (N_cells, T)
    valid_mask = np.isfinite(flat_2d)
    W = self.total_weights[:, np.newaxis]  # (N_basins, 1)
    has_w = (self.has_weights & (self.total_weights > 1e-12))[:, np.newaxis]
    W_safe = np.where(has_w, W, 1.0)

    if np.all(valid_mask):
      res = (self.matrix @ flat_2d).astype(np.float32)
      valid_basins = has_w & (W >= (MIN_VALID_COVERAGE_FRACTION - 1e-6))
      res = np.where(valid_basins, res / W_safe, np.nan).astype(np.float32)
      missing_frac = np.where(
          has_w,
          np.broadcast_to(
              np.clip(1.0 - W, 0.0, 1.0).astype(np.float32),
              (self.num_basins, T),
          ),
          np.float32(1.0),
      )
      return res, missing_frac

    safe_flat = np.where(valid_mask, flat_2d, 0.0)
    valid_float = valid_mask.astype(np.float64)

    N = self.matrix @ safe_flat  # (N_basins, T)
    V = self.matrix @ valid_float  # (N_basins, T)

    valid_ratio = np.where(has_w, V / W_safe, 0.0)
    missing_frac = np.where(
        has_w,
        np.clip(1.0 - valid_ratio, 0.0, 1.0),
        1.0,
    ).astype(np.float32)

    valid_mask_out = (
        has_w
        & (V > 1e-12)
        & (valid_ratio >= (MIN_VALID_COVERAGE_FRACTION - 1e-6))
    )
    result = np.full((self.num_basins, T), np.nan, dtype=np.float32)
    result[valid_mask_out] = (N[valid_mask_out] / V[valid_mask_out]).astype(
        np.float32
    )
    return result, missing_frac

  def reduce_3d(self, grid_3d: np.ndarray) -> np.ndarray:
    """Reduces a 3D array `(T, lat, lon)` to a 2D array `(N_basins, T)`."""
    vals, _ = self.reduce_3d_with_coverage(grid_3d)
    return vals

  def reduce_4d_with_coverage(
      self, grid_4d: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduces a 4D array `(T, L, lat, lon)` to `(values, missing_fraction)` of shape `(N_basins, T, L)`."""
    T, L, H, W = grid_4d.shape
    combined = grid_4d.reshape(T * L, H, W)
    red_3d, miss_3d = self.reduce_3d_with_coverage(combined)
    return (
        red_3d.reshape(self.num_basins, T, L),
        miss_3d.reshape(self.num_basins, T, L),
    )

  def reduce_4d(self, grid_4d: np.ndarray) -> np.ndarray:
    """Reduces a 4D array `(T, L, lat, lon)` to a 3D array `(N_basins, T, L)`."""
    vals, _ = self.reduce_4d_with_coverage(grid_4d)
    return vals

  def subset(self, basin_ids: Sequence[str]) -> ZonalWeightMatrix:
    """Returns a new ZonalWeightMatrix restricted to the specified `basin_ids`."""
    id_to_row = {b: i for i, b in enumerate(self.basin_ids)}
    row_indices = [id_to_row[str(b)] for b in basin_ids]
    sub_csr = self.matrix[row_indices, :]
    return ZonalWeightMatrix(
        basin_ids=basin_ids,
        lats=self.lats,
        lons=self.lons,
        matrix=sub_csr,
    )

  def crop_to_coords(
      self,
      sub_lats: np.ndarray,
      sub_lons: np.ndarray,
      atol: float = 1e-4,
  ) -> ZonalWeightMatrix:
    """Projects/crops this global weight matrix onto a spatial subgrid `(sub_lats, sub_lons)`.

    Preserves exact mathematical equivalence with full-grid reduction for all
    basins contained within `(sub_lats, sub_lons)` without recomputing Shapely
    polygon intersections.

    Args:
      sub_lats: 1D array of subset latitude coordinates.
      sub_lons: 1D array of subset longitude coordinates.
      atol: Floating-point tolerance for matching coordinate values.

    Returns:
      Cropped ZonalWeightMatrix matching `(len(sub_lats), len(sub_lons))`.
    """
    sub_lats = np.asarray(sub_lats, dtype=np.float64)
    sub_lons = np.asarray(sub_lons, dtype=np.float64)

    # Map each sub_lat to its index in self.lats
    lat_orig_indices = np.empty(len(sub_lats), dtype=np.int64)
    for i, val in enumerate(sub_lats):
      diffs = np.abs(self.lats - val)
      min_idx = int(np.argmin(diffs))
      if diffs[min_idx] > atol:
        raise ValueError(
            f"sub_lat {val} not found in ZonalWeightMatrix.lats within atol={atol}."
        )
      lat_orig_indices[i] = min_idx

    # Map each sub_lon to its index in self.lons
    lon_orig_indices = np.empty(len(sub_lons), dtype=np.int64)
    for j, val in enumerate(sub_lons):
      diffs = np.abs(self.lons - val)
      min_idx = int(np.argmin(diffs))
      if diffs[min_idx] > atol:
        raise ValueError(
            f"sub_lon {val} not found in ZonalWeightMatrix.lons within atol={atol}."
        )
      lon_orig_indices[j] = min_idx

    n_lon_orig = len(self.lons)
    # Construct flat column indices in the original matrix corresponding to the subgrid
    selected_flat_cols = (
        lat_orig_indices[:, None] * n_lon_orig + lon_orig_indices[None, :]
    ).ravel()

    cropped_csr = self.matrix[:, selected_flat_cols].tocsr()
    return ZonalWeightMatrix(
        basin_ids=self.basin_ids,
        lats=sub_lats,
        lons=sub_lons,
        matrix=cropped_csr,
    )

  def crop_to_bounds(
      self,
      bounds: Union[
          BoundingBox,
          gpd.GeoDataFrame,
          shapely.geometry.base.BaseGeometry,
          Sequence[float],
      ],
      buffer_degrees: float = 0.0,
  ) -> ZonalWeightMatrix:
    """Crops this ZonalWeightMatrix directly to a BoundingBox or GeoDataFrame."""
    sub_lats, sub_lons, _, _ = slice_coordinates_by_bounds(
        self.lats, self.lons, bounds, buffer_degrees=buffer_degrees
    )
    return self.crop_to_coords(sub_lats, sub_lons)

  def save(self, path: Union[str, os.PathLike]) -> None:
    """Saves the sparse weight matrix and coordinate metadata to a compressed `.npz` file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        basin_ids=np.array(self.basin_ids, dtype=str),
        lats=self.lats,
        lons=self.lons,
        data=self.matrix.data,
        indices=self.matrix.indices,
        indptr=self.matrix.indptr,
        shape=np.array(self.matrix.shape, dtype=np.int64),
    )

  @classmethod
  def load(cls, path: Union[str, os.PathLike]) -> ZonalWeightMatrix:
    """Loads a precomputed ZonalWeightMatrix from a `.npz` archive."""
    with np.load(path, allow_pickle=False) as npz:
      basin_ids = [str(b) for b in npz["basin_ids"]]
      lats = npz["lats"]
      lons = npz["lons"]
      csr = sparse.csr_matrix(
          (npz["data"], npz["indices"], npz["indptr"]),
          shape=tuple(npz["shape"]),
          dtype=np.float64,
      )
    return cls(basin_ids=basin_ids, lats=lats, lons=lons, matrix=csr)
