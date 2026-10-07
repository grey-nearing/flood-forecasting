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

"""Direct catchment timeseries extractors for dynamical.org Icechunk datasets.

Supports historical and real-time extraction directly from dynamical.org
Icechunk stores without requiring an intermediate global gridded Zarr archive:
  - ``GFSExtractor`` (``noaa-gfs-forecast``, deterministic 0.25-deg, 9 vars)
  - ``GEFSExtractor`` (``noaa-gefs-forecast-35-day``, 31-member 0.25-deg, 9 vars x 7 stats)
  - ``IFSEnsExtractor`` (``ecmwf-ifs-ens-forecast-15-day-0-25-degree``, 51-member 0.25-deg, 8 vars x 7 stats)
  - ``AIFSExtractor`` (``ecmwf-aifs-single-forecast``, deterministic 0.25-deg, 8 vars)
  - ``AIFSEnsExtractor`` (``ecmwf-aifs-ens-forecast``, 51-member 0.25-deg, 8 vars x 7 stats)
  - ``DynamicalIMERGExtractor`` (``nasa-imerg-analysis-early``, 0.1-deg half-hourly analysis)

Uses active-chunk spatial slicing and compressed CSR sparse matrix reduction so
only spatial Zarr chunks intersecting target catchment polygons are fetched over
S3 byte-range requests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
import shapely.geometry
import xarray as xr

from multimet.timeseries_extractors.base import BaseExtractor
from multimet.timeseries_extractors.config import (
    ENSEMBLE_MISSING_FRACTION_VAR,
    ENSEMBLE_STAT_SUFFIXES,
    FORECAST_LEAD_DAYS,
    MISSING_FRACTION_VAR,
    PRODUCT_BANDS,
    PRODUCT_METADATA_ATTRS,
    Product,
)
from multimet.utils.geometry import load_basin_geometries
from multimet.utils.spatial import BoundingBox, slice_coordinates_by_bounds
from multimet.utils.zonal import ZonalWeightMatrix

if importlib.util.find_spec("dynamical_catalog") is not None:
  import dynamical_catalog
else:
  dynamical_catalog = None


def list_catalog_datasets() -> List[str]:
  """Returns all dataset identifiers available in the dynamical.org catalog."""
  if dynamical_catalog is None:
    raise ImportError(
        "dynamical_catalog is required to list catalog datasets. "
        "Install it via: pip install dynamical-catalog icechunk"
    )
  return list(dynamical_catalog.list())


@dataclasses.dataclass(frozen=True)
class DynamicalDatasetInfo:
  """Metadata describing the coordinates and dimensions of a dynamical.org dataset."""

  dataset_id: str
  spatial_dims: Tuple[str, str]
  temporal_dim: str
  has_lead_time: bool
  has_ensemble: bool
  variables: List[str]
  dataset_attrs: Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class DynamicalBandSpec:
  """Specification for converting and aggregating a single dynamical.org variable."""

  base_name: str
  source_var: str
  agg_mode: str  # "rate_sum" | "mean" | "mean_6h_reset" | "trapz" | "min" | "max"
  scale: float = 1.0
  clip_min: Optional[float] = None


ECMWF_DYNAMICAL_BAND_SPECS: Tuple[DynamicalBandSpec, ...] = (
    DynamicalBandSpec(
        base_name="dewpoint_temperature_2m",
        source_var="dew_point_temperature_2m",
        agg_mode="trapz",
    ),
    DynamicalBandSpec(
        base_name="downward_long_wave_radiation",
        source_var="downward_long_wave_radiation_flux_surface",
        agg_mode="mean",
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="downward_short_wave_radiation",
        source_var="downward_short_wave_radiation_flux_surface",
        agg_mode="mean",
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="surface_pressure",
        source_var="pressure_surface",
        agg_mode="trapz",
        scale=1e-3,
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="temperature_2m",
        source_var="temperature_2m",
        agg_mode="trapz",
    ),
    DynamicalBandSpec(
        base_name="total_precipitation",
        source_var="precipitation_surface",
        agg_mode="rate_sum",
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="u_component_of_wind_10m",
        source_var="wind_u_10m",
        agg_mode="trapz",
    ),
    DynamicalBandSpec(
        base_name="v_component_of_wind_10m",
        source_var="wind_v_10m",
        agg_mode="trapz",
    ),
)

NOAA_DYNAMICAL_BAND_SPECS: Tuple[DynamicalBandSpec, ...] = (
    DynamicalBandSpec(
        base_name="downward_long_wave_radiation",
        source_var="downward_long_wave_radiation_flux_surface",
        agg_mode="mean_6h_reset",
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="downward_short_wave_radiation",
        source_var="downward_short_wave_radiation_flux_surface",
        agg_mode="mean_6h_reset",
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="surface_pressure",
        source_var="pressure_surface",
        agg_mode="trapz",
        scale=1e-3,
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="temperature_2m",
        source_var="temperature_2m",
        agg_mode="trapz",
    ),
    DynamicalBandSpec(
        base_name="temperature_2m_max",
        source_var="maximum_temperature_2m",
        agg_mode="max",
    ),
    DynamicalBandSpec(
        base_name="temperature_2m_min",
        source_var="minimum_temperature_2m",
        agg_mode="min",
    ),
    DynamicalBandSpec(
        base_name="total_precipitation",
        source_var="precipitation_surface",
        agg_mode="rate_sum",
        clip_min=0.0,
    ),
    DynamicalBandSpec(
        base_name="u_component_of_wind_10m",
        source_var="wind_u_10m",
        agg_mode="trapz",
    ),
    DynamicalBandSpec(
        base_name="v_component_of_wind_10m",
        source_var="wind_v_10m",
        agg_mode="trapz",
    ),
)

DYNAMICAL_FORECAST_DATASETS: Dict[
    str, Tuple[Product, str, str, bool, Tuple[DynamicalBandSpec, ...]]
] = {
    "AIFS": (
        Product.AIFS,
        "ecmwf-aifs-single-forecast",
        "aifs",
        False,
        ECMWF_DYNAMICAL_BAND_SPECS,
    ),
    "AIFS_ENS": (
        Product.AIFS_ENS,
        "ecmwf-aifs-ens-forecast",
        "aifs_ens",
        True,
        ECMWF_DYNAMICAL_BAND_SPECS,
    ),
    "GFS": (
        Product.GFS,
        "noaa-gfs-forecast",
        "gfs",
        False,
        NOAA_DYNAMICAL_BAND_SPECS,
    ),
    "GEFS": (
        Product.GEFS,
        "noaa-gefs-forecast-35-day",
        "gefs",
        True,
        NOAA_DYNAMICAL_BAND_SPECS,
    ),
    "IFS_ENS": (
        Product.IFS_ENS,
        "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "ifs_ens",
        True,
        ECMWF_DYNAMICAL_BAND_SPECS,
    ),
}


def _build_compressed_weight_matrix(
    matrix: ZonalWeightMatrix,
) -> Tuple[np.ndarray, ZonalWeightMatrix]:
  """Compresses a ZonalWeightMatrix to only columns with non-zero basin weights.

  Returns:
    Tuple of ``(active_cols, compressed_matrix)`` where ``active_cols`` is the
    sorted 1D array of original flat grid indices ``r * n_lon + c`` that have
    non-zero weight in at least one basin.
  """
  active_cols = np.unique(matrix.matrix.indices)
  k_active = len(active_cols)
  if k_active == 0:
    empty_csr = sparse.csr_matrix((matrix.num_basins, 0), dtype=np.float64)
    comp_wm = ZonalWeightMatrix(
        matrix.basin_ids,
        np.array([], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        empty_csr,
    )
    comp_wm.total_weights = matrix.total_weights
    comp_wm.has_weights = matrix.has_weights
    return active_cols, comp_wm

  comp_indices = np.searchsorted(active_cols, matrix.matrix.indices).astype(
      np.int32
  )
  comp_csr = sparse.csr_matrix(
      (matrix.matrix.data, comp_indices, matrix.matrix.indptr),
      shape=(matrix.num_basins, k_active),
      dtype=np.float64,
  )
  comp_wm = ZonalWeightMatrix(
      matrix.basin_ids,
      np.arange(k_active, dtype=np.float64),
      np.array([0.0], dtype=np.float64),
      comp_csr,
  )
  comp_wm.total_weights = matrix.total_weights
  comp_wm.has_weights = matrix.has_weights
  return active_cols, comp_wm


def _align_or_build_weights_matrix(
    sub_ds: xr.Dataset,
    basins_gdf: gpd.GeoDataFrame,
    weights_matrix: Optional[ZonalWeightMatrix],
    lat_coord: str = "latitude",
    lon_coord: str = "longitude",
    default_res: float = 0.25,
) -> Tuple[xr.Dataset, ZonalWeightMatrix]:
  """Aligns ``sub_ds`` and ``weights_matrix`` on identical ``(lat, lon)`` coordinates."""
  sub_lats = np.asarray(sub_ds[lat_coord].values, dtype=np.float64)
  sub_lons = np.asarray(sub_ds[lon_coord].values, dtype=np.float64)
  dlat = (
      abs(float(sub_lats[1] - sub_lats[0]))
      if len(sub_lats) > 1
      else default_res
  )
  dlon = (
      abs(float(sub_lons[1] - sub_lons[0]))
      if len(sub_lons) > 1
      else default_res
  )

  if (
      weights_matrix is not None
      and len(weights_matrix.lats) > 0
      and len(weights_matrix.lons) > 0
  ):
    wm_dlat = (
        abs(float(weights_matrix.lats[1] - weights_matrix.lats[0]))
        if len(weights_matrix.lats) > 1
        else dlat
    )
    wm_dlon = (
        abs(float(weights_matrix.lons[1] - weights_matrix.lons[0]))
        if len(weights_matrix.lons) > 1
        else dlon
    )
    if np.isclose(wm_dlat, dlat, atol=1e-3) and np.isclose(
        wm_dlon, dlon, atol=1e-3
    ):
      wm_lat_min = float(np.min(weights_matrix.lats)) - 1e-3
      wm_lat_max = float(np.max(weights_matrix.lats)) + 1e-3
      wm_lon_min = float(np.min(weights_matrix.lons)) - 1e-3
      wm_lon_max = float(np.max(weights_matrix.lons)) + 1e-3
      lat_mask = (sub_lats >= wm_lat_min) & (sub_lats <= wm_lat_max)
      lon_mask = (sub_lons >= wm_lon_min) & (sub_lons <= wm_lon_max)
      if np.any(lat_mask) and np.any(lon_mask):
        if not np.all(lat_mask) or not np.all(lon_mask):
          sub_ds = sub_ds.isel(
              {
                  lat_coord: np.where(lat_mask)[0],
                  lon_coord: np.where(lon_mask)[0],
              }
          )
          sub_lats = np.asarray(sub_ds[lat_coord].values, dtype=np.float64)
          sub_lons = np.asarray(sub_ds[lon_coord].values, dtype=np.float64)

        if (
            weights_matrix.grid_shape == (len(sub_lats), len(sub_lons))
            and np.allclose(weights_matrix.lats, sub_lats, atol=1e-3)
            and np.allclose(weights_matrix.lons, sub_lons, atol=1e-3)
        ):
          return sub_ds, weights_matrix
        return sub_ds, weights_matrix.crop_to_coords(
            sub_lats, sub_lons, atol=1e-3
        )

  matrix = ZonalWeightMatrix.from_geodataframe(
      basins_gdf,
      sub_lats,
      sub_lons,
      cell_res_lat=dlat,
      cell_res_lon=dlon,
  )
  return sub_ds, matrix


def _gather_active_cells(
    sub_ds: xr.Dataset,
    variables: Sequence[str],
    active_cols: np.ndarray,
    n_lons: int,
    lat_coord: str = "latitude",
    lon_coord: str = "longitude",
) -> Dict[str, np.ndarray]:
  """Loads only spatial chunks intersecting ``active_cols`` and gathers active cells.

  For Dask-backed Icechunk datasets where active catchment polygons cover a
  subset of spatial chunks in ``sub_ds``, groups horizontally adjacent active
  chunks into contiguous slices and computes only those blocks, avoiding S3
  reads and memory allocation for ocean or non-basin chunks.

  Args:
    sub_ds: Spatially/temporally sliced Dataset.
    variables: Variable names to load from ``sub_ds``.
    active_cols: 1D sorted array of flat grid cell indices ``r * n_lons + c``.
    n_lons: Number of longitude columns in ``sub_ds``.
    lat_coord: Latitude coordinate/dimension name.
    lon_coord: Longitude coordinate/dimension name.

  Returns:
    Dictionary mapping each variable name to a float32 array of shape
    ``(*leading_dims, len(active_cols))``.
  """
  k_active = len(active_cols)
  active_r = active_cols // n_lons
  active_c = active_cols % n_lons

  first_da = sub_ds[variables[0]]
  chunks = first_da.chunks
  if (
      chunks is not None
      and lat_coord in first_da.dims
      and lon_coord in first_da.dims
      and k_active > 0
  ):
    lat_axis = first_da.dims.index(lat_coord)
    lon_axis = first_da.dims.index(lon_coord)
    lat_chunks = chunks[lat_axis]
    lon_chunks = chunks[lon_axis]
    total_blocks = len(lat_chunks) * len(lon_chunks)

    lat_edges = np.concatenate([[0], np.cumsum(lat_chunks)])
    lon_edges = np.concatenate([[0], np.cumsum(lon_chunks)])
    b_r = np.searchsorted(lat_edges[1:], active_r, side="right")
    b_c = np.searchsorted(lon_edges[1:], active_c, side="right")
    unique_blocks = sorted(set(zip(b_r.tolist(), b_c.tolist())))

    if len(unique_blocks) < total_blocks:
      # Merge horizontally contiguous chunk blocks in each latitude chunk row
      merged_runs: List[Tuple[int, int, int]] = []
      for br, bc in unique_blocks:
        if (
            merged_runs
            and merged_runs[-1][0] == br
            and merged_runs[-1][2] == bc
        ):
          prev_br, prev_c0, _ = merged_runs[-1]
          merged_runs[-1] = (prev_br, prev_c0, bc + 1)
        else:
          merged_runs.append((br, bc, bc + 1))

      ds_vars = sub_ds[list(variables)]
      run_slices: List[xr.Dataset] = []
      run_masks: List[Tuple[np.ndarray, int, int]] = []
      for br, bc_start, bc_end in merged_runs:
        r0 = int(lat_edges[br])
        r1 = int(lat_edges[br + 1])
        c0 = int(lon_edges[bc_start])
        c1 = int(lon_edges[bc_end])
        mask = (b_r == br) & (b_c >= bc_start) & (b_c < bc_end)
        run_slices.append(
            ds_vars.isel({lat_coord: slice(r0, r1), lon_coord: slice(c0, c1)})
        )
        run_masks.append((mask, r0, c0))

      with dask.config.set(scheduler="threads"):
        computed_runs = dask.compute(*run_slices)

      out: Dict[str, np.ndarray] = {}
      for var_name in variables:
        da = sub_ds[var_name]
        leading_shape = da.shape[:-2]
        gathered = np.empty((*leading_shape, k_active), dtype=np.float32)
        for comp_ds, (mask, r0, c0) in zip(computed_runs, run_masks):
          block_vals = np.asarray(comp_ds[var_name].values, dtype=np.float32)
          gathered[..., mask] = block_vals[
              ..., active_r[mask] - r0, active_c[mask] - c0
          ]
        out[var_name] = gathered
      return out

  with dask.config.set(scheduler="threads"):
    computed_ds = sub_ds[list(variables)].compute()

  out = {}
  for var_name in variables:
    vals = np.asarray(computed_ds[var_name].values, dtype=np.float32)
    out[var_name] = vals[..., active_r, active_c]
  return out


class DynamicalDataLoader:
  """Data loader for 1D geographic dynamical.org Icechunk catalog datasets."""

  def __init__(
      self,
      dataset_id: str,
      auto_open: bool = True,
      ds: Optional[xr.Dataset] = None,
  ):
    if ds is None and dynamical_catalog is None:
      raise ImportError(
          "dynamical_catalog is required. Install via: "
          "pip install dynamical-catalog icechunk"
      )

    self.dataset_id = dataset_id
    self._ds: Optional[xr.Dataset] = ds

    self.spatial_dims: Tuple[str, str] = ("latitude", "longitude")
    self.lat_coord: str = "latitude"
    self.lon_coord: str = "longitude"
    self.lat_descending: bool = True
    self.lon_ascending: bool = True

    self.time_dim: str = "time"
    self.has_lead_time: bool = False
    self.has_ensemble: bool = False

    if self._ds is not None:
      self._analyze_schema()
    elif auto_open:
      self.open()

  @property
  def ds(self) -> xr.Dataset:
    """Returns the opened xarray.Dataset."""
    if self._ds is None:
      self.open()
    return self._ds

  def open(self) -> xr.Dataset:
    """Opens the dataset via dynamical_catalog and inspects its coordinate schema."""
    if self._ds is None:
      if dynamical_catalog is None:
        raise ImportError(
            "dynamical_catalog is required. Install via: "
            "pip install dynamical-catalog icechunk"
        )
      self._ds = dynamical_catalog.open(self.dataset_id)
    self._analyze_schema()
    return self._ds

  def _analyze_schema(self) -> None:
    """Inspects coordinate dimensions and temporal/spatial topologies."""
    ds = self._ds
    assert ds is not None

    if "init_time" in ds.coords:
      self.time_dim = "init_time"
    elif "time" in ds.coords:
      self.time_dim = "time"
    else:
      for c in ds.coords:
        if np.issubdtype(ds[c].dtype, np.datetime64):
          self.time_dim = str(c)
          break

    self.has_lead_time = "lead_time" in ds.coords or "lead_time" in ds.dims
    self.has_ensemble = (
        "ensemble_member" in ds.coords or "ensemble_member" in ds.dims
    )

    coords = set(ds.coords.keys())
    if "latitude" in coords:
      self.lat_coord = "latitude"
    elif "lat" in coords:
      self.lat_coord = "lat"
    else:
      raise ValueError(f"Could not find latitude coordinate in {list(coords)}")

    if "longitude" in coords:
      self.lon_coord = "longitude"
    elif "lon" in coords:
      self.lon_coord = "lon"
    else:
      raise ValueError(f"Could not find longitude coordinate in {list(coords)}")

    self.spatial_dims = (self.lat_coord, self.lon_coord)
    lat_vals = ds[self.lat_coord].values
    lon_vals = ds[self.lon_coord].values
    self.lat_descending = bool(len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1])
    self.lon_ascending = bool(len(lon_vals) > 1 and lon_vals[0] < lon_vals[-1])

  def get_info(self) -> DynamicalDatasetInfo:
    """Returns structured metadata about the dataset."""
    ds = self.ds
    return DynamicalDatasetInfo(
        dataset_id=self.dataset_id,
        spatial_dims=self.spatial_dims,
        temporal_dim=self.time_dim,
        has_lead_time=self.has_lead_time,
        has_ensemble=self.has_ensemble,
        variables=list(ds.data_vars.keys()),
        dataset_attrs=dict(ds.attrs),
    )

  def compute_spatial_slices(
      self,
      watersheds: Union[
          gpd.GeoDataFrame,
          shapely.geometry.base.BaseGeometry,
          Sequence[float],
          Tuple[float, float, float, float],
          str,
          Path,
      ],
      buffer: Optional[float] = None,
  ) -> Dict[str, Union[slice, np.ndarray]]:
    """Computes coordinate index slices bounding the requested watersheds."""
    ds = self.ds
    if isinstance(watersheds, (str, Path)):
      gdf = load_basin_geometries(str(watersheds))
      bbox = BoundingBox.from_geodataframe(gdf, buffer_degrees=0.0)
    elif isinstance(watersheds, gpd.GeoDataFrame):
      bbox = BoundingBox.from_geodataframe(watersheds, buffer_degrees=0.0)
    elif isinstance(watersheds, shapely.geometry.base.BaseGeometry):
      bbox = BoundingBox.from_geometry(watersheds, buffer_degrees=0.0)
    elif isinstance(watersheds, Sequence) and len(watersheds) == 4:
      bbox = BoundingBox.from_tuple(watersheds, buffer_degrees=0.0)
    else:
      raise TypeError(f"Unsupported watersheds type: {type(watersheds)}")

    lat_vals = ds[self.lat_coord].values
    lon_vals = ds[self.lon_coord].values

    dlat = abs(float(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 0.25
    dlon = abs(float(lon_vals[1] - lon_vals[0])) if len(lon_vals) > 1 else 0.25
    buf = buffer if buffer is not None else max(dlat, dlon) * 1.5
    _, _, lat_idx, lon_idx = slice_coordinates_by_bounds(
        lat_vals, lon_vals, bbox, buffer_degrees=buf
    )

    if len(lat_idx) == 0:
      lat_slice: Union[slice, np.ndarray] = slice(0, 0)
    elif len(lat_idx) == 1 or bool(np.all(np.diff(lat_idx) == 1)):
      lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    else:
      lat_slice = lat_idx

    if len(lon_idx) == 0:
      lon_slice: Union[slice, np.ndarray] = slice(0, 0)
    elif len(lon_idx) == 1 or bool(np.all(np.diff(lon_idx) == 1)):
      lon_slice = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    else:
      lon_slice = lon_idx

    return {
        self.lat_coord: lat_slice,
        self.lon_coord: lon_slice,
    }

  def load_spatial_subset(
      self,
      watersheds: Union[
          gpd.GeoDataFrame,
          shapely.geometry.base.BaseGeometry,
          Sequence[float],
          Tuple[float, float, float, float],
          str,
          Path,
      ],
      variables: Optional[Union[str, Sequence[str]]] = None,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      lead_time_slice: Optional[Union[slice, Sequence[int]]] = None,
      ensemble_members: Optional[Union[int, Sequence[int], slice]] = None,
      buffer: Optional[float] = None,
      compute: bool = True,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Loads a spatially and temporally sliced xarray.Dataset from dynamical.org."""
    ds = self.ds

    if variables is not None:
      if isinstance(variables, str):
        variables = [variables]
      missing = [v for v in variables if v not in ds.data_vars]
      if missing:
        raise KeyError(
            f"Variables {missing} not found in {self.dataset_id}. "
            f"Available: {list(ds.data_vars.keys())}"
        )
      sub_ds = ds[list(variables)]
    else:
      sub_ds = ds

    if use_bounding_box:
      spatial_slices = self.compute_spatial_slices(watersheds, buffer=buffer)
      sub_ds = sub_ds.isel(spatial_slices)

    if start_date is not None or end_date is not None:
      s_dt = pd.to_datetime(start_date) if start_date is not None else None
      e_dt = pd.to_datetime(end_date) if end_date is not None else None
      if e_dt is not None:
        is_date_only_str = (
            isinstance(end_date, str) and len(end_date.strip()) <= 10
        )
        is_midnight_ts = (
            isinstance(end_date, pd.Timestamp)
            and e_dt.hour == 0
            and e_dt.minute == 0
            and e_dt.second == 0
            and e_dt.nanosecond == 0
            and (s_dt is None or s_dt.floor("D") == s_dt)
        )
        if is_date_only_str or is_midnight_ts:
          e_dt = e_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
      t_slice = slice(s_dt, e_dt)
      sub_ds = sub_ds.sel({self.time_dim: t_slice})

    if self.has_lead_time and lead_time_slice is not None:
      sub_ds = sub_ds.isel(lead_time=lead_time_slice)

    if self.has_ensemble and ensemble_members is not None:
      sub_ds = sub_ds.isel(ensemble_member=ensemble_members)

    if compute:
      with dask.config.set(scheduler="threads"):
        sub_ds = sub_ds.compute()

    return sub_ds

  def extract_basin_timeseries(
      self,
      watersheds: Union[gpd.GeoDataFrame, str, Path],
      variables: Optional[Union[str, Sequence[str]]] = None,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      lead_time_slice: Optional[Union[slice, Sequence[int]]] = None,
      ensemble_members: Optional[Union[int, Sequence[int], slice]] = None,
      buffer: Optional[float] = None,
      id_column: Optional[str] = None,
      use_bounding_box: bool = True,
  ) -> xr.Dataset:
    """Extracts area-weighted catchment timeseries for arbitrary dynamical.org datasets."""
    if isinstance(watersheds, (str, Path)):
      basins_gdf = load_basin_geometries(str(watersheds), id_column=id_column)
    elif isinstance(watersheds, gpd.GeoDataFrame):
      basins_gdf = watersheds
    else:
      raise TypeError(
          "watersheds must be a GeoDataFrame or path to shapefile/geojson, "
          f"got {type(watersheds)}"
      )

    basin_ids = [str(b) for b in basins_gdf.index]
    sub_ds = self.load_spatial_subset(
        watersheds=basins_gdf,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        lead_time_slice=lead_time_slice,
        ensemble_members=ensemble_members,
        buffer=buffer,
        compute=False,
        use_bounding_box=use_bounding_box,
    )

    sub_ds, matrix = _align_or_build_weights_matrix(
        sub_ds,
        basins_gdf,
        weights_matrix=None,
        lat_coord=self.lat_coord,
        lon_coord=self.lon_coord,
    )
    active_cols, comp_wm = _build_compressed_weight_matrix(matrix)
    target_vars = list(sub_ds.data_vars.keys())
    gathered_vars = _gather_active_cells(
        sub_ds,
        target_vars,
        active_cols,
        n_lons=len(matrix.lons),
        lat_coord=self.lat_coord,
        lon_coord=self.lon_coord,
    )

    out_coords: Dict[str, Any] = {
        "basin": basin_ids,
        "date": sub_ds[self.time_dim].values,
    }
    if self.has_lead_time and "lead_time" in sub_ds.coords:
      out_coords["lead_time"] = sub_ds["lead_time"].values
    if self.has_ensemble and "ensemble_member" in sub_ds.coords:
      out_coords["ensemble_member"] = sub_ds["ensemble_member"].values

    non_negative_vars = {
        "precipitation_surface",
        "downward_short_wave_radiation_flux_surface",
        "downward_long_wave_radiation_flux_surface",
        "pressure_surface",
    }
    out_vars: Dict[str, Any] = {}
    k_active = len(active_cols)
    for var_name in target_vars:
      da = sub_ds[var_name]
      arr = gathered_vars[var_name]
      leading_shape = arr.shape[:-1]
      n_leading = int(np.prod(leading_shape)) if leading_shape else 1
      if k_active == 0 or n_leading == 0:
        red = np.full(
            (len(basin_ids), *leading_shape), np.nan, dtype=np.float32
        )
      else:
        red_flat, _ = comp_wm.reduce_3d_with_coverage(
            arr.reshape(n_leading, k_active, 1)
        )
        red = red_flat.reshape(len(basin_ids), *leading_shape)
        if var_name in non_negative_vars:
          red = np.where(
              np.isnan(red), np.nan, np.maximum(np.float32(0.0), red)
          ).astype(np.float32)

      if red.ndim == 2:
        out_vars[var_name] = (["basin", "date"], red)
      elif red.ndim == 3:
        out_vars[var_name] = (["basin", "date", "lead_time"], red)
      elif red.ndim == 4:
        # Respect whether ensemble_member or lead_time comes first in da.dims
        non_spatial_dims = [
            "date" if d == self.time_dim else str(d) for d in da.dims[:-2]
        ]
        out_vars[var_name] = (["basin", *non_spatial_dims], red)

    result_ds = xr.Dataset(data_vars=out_vars, coords=out_coords)
    result_ds.attrs.update(sub_ds.attrs)
    result_ds.attrs["dynamical_dataset_id"] = self.dataset_id
    return result_ds


class DynamicalIMERGExtractor(BaseExtractor):
  """Extractor for NASA GPM IMERG Early via dynamical.org Icechunk.

  Produces ``dynamical_imerg_precipitation`` (``mm/day``) and
  ``dynamical_imerg_missing_fraction`` (``[0, 1]``) on a 2D ``(basin, date)``
  grid, keeping ``Product.DYNAMICAL_IMERG`` completely distinct from the primary
  NASA GES DISC ``Product.IMERG`` (``imerg_precipitation``).

  Slices the underlying ``(1440, 100, 100)`` Icechunk store in 30-day
  (1,440 half-hourly step) chunk-aligned windows and fetches only spatial chunks
  intersecting target catchment polygons.
  """

  DEFAULT_DATASET_ID: str = "nasa-imerg-analysis-early"
  DEFAULT_BATCH_DAYS: int = 30

  def __init__(
      self,
      product: Optional[Product] = None,
      source: str = "dynamical",
      dataset_id: str = DEFAULT_DATASET_ID,
      loader: Optional[DynamicalDataLoader] = None,
      batch_days: int = DEFAULT_BATCH_DAYS,
      **kwargs,
  ):
    del kwargs
    prod = product if product is not None else Product.DYNAMICAL_IMERG
    super().__init__(prod)
    self.source = source.lower().strip() if source else "dynamical"
    if self.source in ("archive", "gridded_archive", "zarr_archive"):
      raise ValueError(
          "DynamicalIMERGExtractor reads from the dynamical.org Icechunk "
          "catalog and does not support source='archive'; use IMERGExtractor "
          "with source='archive' for gridded Zarr archives."
      )
    self.dataset_id = dataset_id
    self.batch_days = max(1, int(batch_days))
    self.loader = (
        loader if loader is not None else DynamicalDataLoader(dataset_id)
    )
    self.lats: Optional[np.ndarray] = None
    self.lons: Optional[np.ndarray] = None
    self._init_coords()

  def _init_coords(self) -> None:
    ds = self.loader.ds
    if "latitude" in ds.coords:
      self.lats = ds.latitude.values
      self.lons = ds.longitude.values

  _ICECHUNK_EPOCH = pd.Timestamp("1998-01-01")

  def _chunk_anchor(self) -> pd.Timestamp:
    """Returns the 30-day Icechunk time-chunk epoch anchor date."""
    ds = self.loader.ds
    if "time" in ds.coords and len(ds.time) > 0:
      return pd.to_datetime(ds.time.values[0]).floor("D")
    return self._ICECHUNK_EPOCH

  def _iter_chunk_aligned_windows(
      self, start_dt: pd.Timestamp, end_dt: pd.Timestamp
  ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Splits ``[start_dt, end_dt]`` into windows aligned with 30-day Icechunk time chunks."""
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    anchor = self._chunk_anchor()

    cur = start_dt
    step_days = self.batch_days
    while cur <= end_dt:
      days_since_anchor = int((cur - anchor).days)
      if days_since_anchor >= 0 and step_days > 1:
        rem_step = days_since_anchor % step_days
        days_in_win = step_days - 1 - rem_step
        if step_days < 30:
          rem_chunk = days_since_anchor % 30
          days_in_win = min(days_in_win, 29 - rem_chunk)
        win_end = min(end_dt, cur + pd.Timedelta(days=days_in_win))
      else:
        win_end = min(end_dt, cur + pd.Timedelta(days=step_days - 1))
      windows.append((cur, win_end))
      cur = win_end + pd.Timedelta(days=1)
    return windows

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts daily accumulated dynamical.org IMERG precipitation for given basins."""
    del kwargs
    if start_date is None or end_date is None:
      raise ValueError(
          "DynamicalIMERGExtractor.extract_for_basins requires both start_date "
          "and end_date to be explicitly provided."
      )
    basin_ids = [str(b) for b in basins_gdf.index]
    start_dt = pd.to_datetime(start_date).floor("D")
    end_dt = pd.to_datetime(end_date).floor("D")
    if end_dt < start_dt:
      raise ValueError(
          f"end_date ({end_dt}) must be >= start_date ({start_dt})."
      )
    date_idx = pd.date_range(start_dt, end_dt, freq="D")

    precip_matrix = np.full(
        (len(basin_ids), len(date_idx)), np.nan, dtype=np.float32
    )
    missing_matrix = np.ones(
        (len(basin_ids), len(date_idx)), dtype=np.float32
    )

    active_matrix = weights_matrix
    active_cols: Optional[np.ndarray] = None
    comp_wm: Optional[ZonalWeightMatrix] = None

    for w_start, w_end in self._iter_chunk_aligned_windows(start_dt, end_dt):
      sub_ds = self.loader.load_spatial_subset(
          watersheds=basins_gdf,
          variables=["precipitation_surface"],
          start_date=w_start.strftime("%Y-%m-%d"),
          end_date=w_end.strftime("%Y-%m-%d"),
          buffer=0.1,
          compute=False,
          use_bounding_box=use_bounding_box,
      )
      if "time" not in sub_ds.dims or len(sub_ds.time) == 0:
        continue

      sub_ds, active_matrix = _align_or_build_weights_matrix(
          sub_ds,
          basins_gdf,
          weights_matrix=active_matrix,
          lat_coord=self.loader.lat_coord,
          lon_coord=self.loader.lon_coord,
          default_res=0.1,
      )
      if active_cols is None or comp_wm is None:
        active_cols, comp_wm = _build_compressed_weight_matrix(active_matrix)

      k_active = len(active_cols)
      if k_active == 0:
        continue

      gathered = _gather_active_cells(
          sub_ds,
          ["precipitation_surface"],
          active_cols,
          n_lons=len(active_matrix.lons),
          lat_coord=self.loader.lat_coord,
          lon_coord=self.loader.lon_coord,
      )["precipitation_surface"]

      raw_times = pd.DatetimeIndex(
          pd.to_datetime(sub_ds.time.values).tz_localize(None)
      )
      floored_days = raw_times.floor("D")
      unique_days = floored_days.unique()

      valid_days: List[pd.Timestamp] = []
      daily_active_list: List[np.ndarray] = []
      for dt_day in unique_days:
        if dt_day not in date_idx:
          continue
        day_mask = floored_days == dt_day
        if int(np.sum(day_mask)) < 48:
          continue
        day_steps = gathered[day_mask, :]
        # Each 30-minute step rate (kg m-2 s-1 = mm/s) * 1800s = depth in mm;
        # np.sum propagates NaN if any of the 48 half-hour steps is NaN.
        day_sum = np.sum(day_steps, axis=0) * np.float32(1800.0)
        day_sum = np.where(
            np.isnan(day_sum), np.nan, np.maximum(np.float32(0.0), day_sum)
        )
        valid_days.append(dt_day)
        daily_active_list.append(day_sum)

      if not valid_days:
        continue

      daily_active = np.stack(daily_active_list, axis=0)  # (N_days, K)
      reduced, reduced_miss = comp_wm.reduce_3d_with_coverage(
          daily_active[:, :, np.newaxis]
      )
      for idx_in_batch, dt_day in enumerate(valid_days):
        d_pos = int(date_idx.get_loc(dt_day))
        precip_matrix[:, d_pos] = reduced[:, idx_in_batch]
        missing_matrix[:, d_pos] = reduced_miss[:, idx_in_batch]

    ds = xr.Dataset(
        data_vars={
            "dynamical_imerg_precipitation": (
                ["basin", "date"],
                precip_matrix.astype(np.float32),
            ),
            "dynamical_imerg_missing_fraction": (
                ["basin", "date"],
                missing_matrix.astype(np.float32),
            ),
        },
        coords={
            "basin": basin_ids,
            "date": date_idx.values,
        },
    )
    if self.product in PRODUCT_METADATA_ATTRS:
      ds.attrs.update(PRODUCT_METADATA_ATTRS[self.product])
    ds.attrs["dynamical_dataset_id"] = self.dataset_id
    return ds

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      **kwargs,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 day of dynamical.org IMERG precipitation across basins."""
    ds = self.extract_for_basins(
        basins_gdf,
        start_date=dt,
        end_date=dt,
        weights_matrix=weights_matrix,
        **kwargs,
    )
    return {band: ds[band].values[:, 0] for band in ds.data_vars}


def _lead_times_to_seconds(
    lead_coord: Union[xr.DataArray, np.ndarray],
) -> np.ndarray:
  """Converts a lead_time coordinate array into elapsed seconds from initialization."""
  vals = (
      lead_coord.values
      if isinstance(lead_coord, xr.DataArray)
      else np.asarray(lead_coord)
  )
  if np.issubdtype(vals.dtype, np.timedelta64):
    return (vals / np.timedelta64(1, "s")).astype(np.float64)
  return np.asarray(vals, dtype=np.float64) * 3600.0


def find_latest_dynamical_forecast_date(
    dataset_id: str = "ecmwf-aifs-single-forecast",
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    max_lookback_days: int = 7,
    require_full_10d: bool = True,
    lead_days: int = 10,
    loader: Optional[DynamicalDataLoader] = None,
) -> pd.Timestamp:
  """Finds the newest published 00z forecast initialization date in a dynamical.org store.

  Walks backwards from ``reference_date`` (defaulting to current UTC date) up to
  ``max_lookback_days`` and probes the forecast variables at a grid pixel at the
  24h lead step (and at the terminal ``lead_days * 24h`` step when
  ``require_full_10d=True``) to verify that the 00:00:00 UTC initialization has
  been ingested and contains valid (finite) values.
  """
  active_loader = (
      loader if loader is not None else DynamicalDataLoader(dataset_id)
  )
  ds = active_loader.ds
  time_dim = active_loader.time_dim
  if time_dim not in ds.coords:
    raise FileNotFoundError(
        f"Temporal coordinate {time_dim!r} not found in dynamical dataset"
        f" {dataset_id!r}."
    )
  if "lead_time" not in ds.coords:
    raise FileNotFoundError(
        f"Coordinate 'lead_time' not found in dynamical dataset {dataset_id!r}."
    )

  core_vars = (
      "temperature_2m",
      "precipitation_surface",
      "wind_u_10m",
      "wind_v_10m",
  )
  missing_vars = [v for v in core_vars if v not in ds.data_vars]
  if missing_vars:
    raise FileNotFoundError(
        f"Required forecast variable(s) {missing_vars} not found in dynamical "
        f"dataset {dataset_id!r}; available: {list(ds.data_vars.keys())}."
    )
  all_known_source_vars = tuple(
      dict.fromkeys(
          s.source_var
          for s in (*ECMWF_DYNAMICAL_BAND_SPECS, *NOAA_DYNAMICAL_BAND_SPECS)
      )
  )
  probe_vars = [v for v in all_known_source_vars if v in ds.data_vars]

  raw_times = pd.DatetimeIndex(
      pd.to_datetime(ds[time_dim].values).tz_localize(None)
  )
  if len(raw_times) == 0:
    raise FileNotFoundError(
        f"No timestamps found in coordinate {time_dim!r} of dynamical dataset"
        f" {dataset_id!r}."
    )
  if reference_date is None or str(reference_date).strip().lower() == "latest":
    ref_dt = pd.Timestamp(raw_times[-1]).floor("D")
  else:
    ts = pd.to_datetime(reference_date)
    if ts.tzinfo is not None:
      ts = ts.tz_convert("UTC").tz_localize(None)
    ref_dt = ts.floor("D")

  earliest_dt = ref_dt - pd.Timedelta(days=max(0, int(max_lookback_days)))
  is_00z = (raw_times.hour == 0) & (raw_times.minute == 0)
  in_window = (raw_times >= earliest_dt) & (
      raw_times <= ref_dt + pd.Timedelta(hours=23, minutes=59)
  )
  candidate_indices = np.where(is_00z & in_window)[0][::-1]

  if len(candidate_indices) == 0:
    raise FileNotFoundError(
        "No 00z forecast initialization found in dynamical.org dataset "
        f"{dataset_id!r} within {max_lookback_days} days of "
        f"{ref_dt.strftime('%Y-%m-%d')}."
    )

  lead_sec = _lead_times_to_seconds(ds["lead_time"])
  idx_24h = int(np.argmin(np.abs(lead_sec - 86400.0)))
  if not np.isclose(lead_sec[idx_24h], 86400.0, atol=1.0):
    raise FileNotFoundError(
        "24-hour lead step (86400s) not found in dynamical.org dataset "
        f"{dataset_id!r}."
    )
  probe_lead_indices: List[int] = [idx_24h]
  if require_full_10d:
    target_sec = float(lead_days) * 86400.0
    idx_end = int(np.argmin(np.abs(lead_sec - target_sec)))
    if not np.isclose(lead_sec[idx_end], target_sec, atol=1.0):
      raise FileNotFoundError(
          f"Terminal {lead_days}-day lead step ({target_sec:.0f}s) not found in"
          f" dynamical.org dataset {dataset_id!r} (max"
          f" lead={lead_sec.max():.0f}s)."
      )
    if idx_end not in probe_lead_indices:
      probe_lead_indices.append(idx_end)

  sp_dim0, sp_dim1 = active_loader.spatial_dims

  for t_idx in candidate_indices:
    isel_kwargs: Dict[str, Any] = {
        time_dim: int(t_idx),
        "lead_time": probe_lead_indices,
        sp_dim0: 0,
        sp_dim1: 0,
    }
    with dask.config.set(scheduler="threads"):
      probed_ds = ds[probe_vars].isel(**isel_kwargs).compute()
    all_vars_valid = True
    for var_name in probe_vars:
      vals = np.asarray(probed_ds[var_name].values, dtype=np.float32)
      if vals.size == 0 or not bool(np.all(np.isfinite(vals))):
        all_vars_valid = False
        break
    if all_vars_valid:
      return pd.Timestamp(raw_times[t_idx]).floor("D")

  raise FileNotFoundError(
      "No populated 00z forecast run found in dynamical.org dataset "
      f"{dataset_id!r} within {max_lookback_days} days of "
      f"{ref_dt.strftime('%Y-%m-%d')} (require_full_10d={require_full_10d})."
  )


class DynamicalForecastExtractor(BaseExtractor):
  """BaseExtractor for dynamical.org medium-range deterministic and ensemble forecasts.

  Extracts 10-day forecasts initialized at 00:00:00 UTC from dynamical.org
  Icechunk stores, aggregates sub-daily lead steps over each 24-hour window
  into daily lead steps ``1..D``, and applies exact zonal weighting over
  catchment geometries.

  For ensemble products (``IFS_ENS``, ``AIFS_ENS``, ``GEFS``), computes exact
  daily catchment trajectories per ensemble member first, then emits 7 ensemble
  summary statistics across members (``mean``, ``std``, ``min``, ``max``,
  ``p10``, ``p50``, ``p90``) as 3D ``(basin, date, lead_time)`` variables, and
  optionally emits the raw 4D ``(basin, date, ensemble_member, lead_time)``
  member trajectories and ``{prefix}_missing_fraction_ensemble`` when
  ``include_ensemble_members=True``.
  """

  DEFAULT_PRODUCT: Product = Product.AIFS
  DEFAULT_DATASET_ID: str = "ecmwf-aifs-single-forecast"
  DEFAULT_BAND_PREFIX: str = "aifs"
  DEFAULT_IS_ENSEMBLE: bool = False
  DEFAULT_BAND_SPECS: Tuple[DynamicalBandSpec, ...] = ECMWF_DYNAMICAL_BAND_SPECS

  def __init__(
      self,
      product: Optional[Union[Product, str]] = None,
      source: str = "dynamical",
      dataset_id: Optional[str] = None,
      band_prefix: Optional[str] = None,
      lead_days: Optional[int] = None,
      include_ensemble_members: bool = False,
      data_dir: Optional[Union[str, os.PathLike]] = None,
      loader: Optional[DynamicalDataLoader] = None,
      **kwargs,
  ):
    del kwargs
    if product is None:
      prod_enum = self.DEFAULT_PRODUCT
    elif isinstance(product, Product):
      prod_enum = product
    else:
      prod_enum = Product[str(product).strip().upper()]

    super().__init__(prod_enum)
    self.source = source.lower().strip() if source else "dynamical"
    if self.source in ("archive", "gridded_archive", "zarr_archive"):
      raise ValueError(
          f"{self.__class__.__name__} ({prod_enum.value}) reads from the "
          "dynamical.org Icechunk catalog and does not support source='archive'."
      )
    self.data_dir = str(data_dir) if data_dir is not None else None

    if prod_enum.value not in DYNAMICAL_FORECAST_DATASETS:
      raise ValueError(
          f"Unsupported dynamical forecast product: {prod_enum!r}. "
          f"Expected one of {list(DYNAMICAL_FORECAST_DATASETS.keys())}."
      )
    default_meta = DYNAMICAL_FORECAST_DATASETS[prod_enum.value]
    self.dataset_id = dataset_id if dataset_id is not None else default_meta[1]
    self.band_prefix = (
        band_prefix if band_prefix is not None else default_meta[2]
    )
    self.is_ensemble = default_meta[3]
    self.band_specs: Tuple[DynamicalBandSpec, ...] = default_meta[4]
    self.lead_days = int(
        lead_days
        if lead_days is not None
        else FORECAST_LEAD_DAYS[prod_enum]
    )
    self.include_ensemble_members = bool(include_ensemble_members)
    self.loader: DynamicalDataLoader = (
        loader if loader is not None else DynamicalDataLoader(self.dataset_id)
    )
    self.lats: Optional[np.ndarray] = None
    self.lons: Optional[np.ndarray] = None
    self._init_coords()

  def _init_coords(self) -> None:
    ds = self.loader.ds
    if "latitude" in ds.coords:
      self.lats = ds.latitude.values
      self.lons = ds.longitude.values

  def _lead_slice_for_days(self, max_lead_days: int) -> slice:
    """Computes the integer lead_time index slice covering ``0h .. max_lead_days * 24h``."""
    target_sec = float(max_lead_days) * 86400.0
    ds = self.loader.ds
    if "lead_time" not in ds.coords:
      raise ValueError(
          f"Coordinate 'lead_time' not found in dynamical dataset"
          f" {self.dataset_id!r}."
      )
    lead_sec = _lead_times_to_seconds(ds["lead_time"])
    stop_idx = int(np.searchsorted(lead_sec, target_sec, side="right"))
    if stop_idx <= 0:
      raise ValueError(
          f"No lead_time steps <= {target_sec:.0f}s found in dynamical dataset "
          f"{self.dataset_id!r}."
      )
    return slice(0, stop_idx)

  @staticmethod
  def _aggregate_daily_step(
      red_arr: np.ndarray,
      step_idx: np.ndarray,
      w_sec: np.ndarray,
      w_norm: np.ndarray,
      trapz_idx: np.ndarray,
      w_trapz: np.ndarray,
      reset_idx: np.ndarray,
      spec: DynamicalBandSpec,
  ) -> np.ndarray:
    """Aggregates sub-daily basin values along the last axis (``lead_time``) for 1 lead day."""
    if spec.agg_mode == "rate_sum":
      sub = red_arr[..., step_idx]
      day_val = np.sum(sub * w_sec, axis=-1) * np.float32(spec.scale)
    elif spec.agg_mode == "mean":
      sub = red_arr[..., step_idx]
      day_val = np.sum(sub * w_norm, axis=-1) * np.float32(spec.scale)
    elif spec.agg_mode == "mean_6h_reset":
      sub = red_arr[..., reset_idx]
      day_val = np.mean(sub, axis=-1) * np.float32(spec.scale)
    elif spec.agg_mode == "trapz":
      sub = red_arr[..., trapz_idx]
      day_val = np.sum(sub * w_trapz, axis=-1) * np.float32(spec.scale)
    elif spec.agg_mode == "min":
      sub = red_arr[..., step_idx]
      day_val = np.min(sub, axis=-1) * np.float32(spec.scale)
    elif spec.agg_mode == "max":
      sub = red_arr[..., step_idx]
      day_val = np.max(sub, axis=-1) * np.float32(spec.scale)
    else:
      raise ValueError(f"Unsupported agg_mode {spec.agg_mode!r}")

    if spec.clip_min is not None:
      c_min = np.float32(spec.clip_min)
      day_val = np.where(np.isnan(day_val), np.nan, np.maximum(c_min, day_val))
    return day_val.astype(np.float32)

  @staticmethod
  def _compute_ensemble_stats(
      member_vals: np.ndarray,
  ) -> Dict[str, np.ndarray]:
    """Computes the 7 summary statistics across the ensemble member axis (``axis=-1``).

    Drops ``NaN`` ensemble members at each ``(basin, ..., lead_time)`` slice:
    ``mean``, ``min``, ``max``, ``p10``, ``p50``, and ``p90`` are computed across
    all valid members whenever at least 1 member is finite (``NaN`` if 0 valid),
    and sample standard deviation (``std`` with ``ddof=1``) is computed whenever
    at least 2 members are finite (``NaN`` if fewer than 2 valid).

    Args:
      member_vals: Array of shape ``(N_basins, ..., M)`` containing daily
        catchment values per ensemble member.

    Returns:
      Dictionary mapping each suffix in ``ENSEMBLE_STAT_SUFFIXES`` to a float32
      array of shape ``(N_basins, ...)``.
    """
    leading_shape = member_vals.shape[:-1]
    n_members = member_vals.shape[-1]
    flat = member_vals.reshape(-1, n_members).astype(np.float64)
    n_rows = flat.shape[0]

    valid_counts = np.sum(np.isfinite(flat), axis=1)
    has_one = valid_counts >= 1
    has_two = valid_counts >= 2

    out_mean = np.full(n_rows, np.nan, dtype=np.float32)
    out_std = np.full(n_rows, np.nan, dtype=np.float32)
    out_min = np.full(n_rows, np.nan, dtype=np.float32)
    out_max = np.full(n_rows, np.nan, dtype=np.float32)
    out_p10 = np.full(n_rows, np.nan, dtype=np.float32)
    out_p50 = np.full(n_rows, np.nan, dtype=np.float32)
    out_p90 = np.full(n_rows, np.nan, dtype=np.float32)

    if np.any(has_one):
      sub1 = flat[has_one]
      out_mean[has_one] = np.nanmean(sub1, axis=1).astype(np.float32)
      out_min[has_one] = np.nanmin(sub1, axis=1).astype(np.float32)
      out_max[has_one] = np.nanmax(sub1, axis=1).astype(np.float32)
      p10, p50, p90 = np.nanpercentile(sub1, [10.0, 50.0, 90.0], axis=1)
      out_p10[has_one] = np.asarray(p10, dtype=np.float32)
      out_p50[has_one] = np.asarray(p50, dtype=np.float32)
      out_p90[has_one] = np.asarray(p90, dtype=np.float32)

    if np.any(has_two):
      sub2 = flat[has_two]
      out_std[has_two] = np.nanstd(sub2, axis=1, ddof=1).astype(np.float32)

    return {
        "mean": out_mean.reshape(leading_shape),
        "std": out_std.reshape(leading_shape),
        "min": out_min.reshape(leading_shape),
        "max": out_max.reshape(leading_shape),
        "p10": out_p10.reshape(leading_shape),
        "p50": out_p50.reshape(leading_shape),
        "p90": out_p90.reshape(leading_shape),
    }

  def _extract_window(
      self,
      basins_gdf: gpd.GeoDataFrame,
      w_start: pd.Timestamp,
      w_end: pd.Timestamp,
      max_lead_days: int,
      date_idx: pd.DatetimeIndex,
      data_dict: Dict[str, np.ndarray],
      ensemble_dict: Dict[str, np.ndarray],
      missing_fraction: np.ndarray,
      weights_matrix: Optional[ZonalWeightMatrix],
      use_bounding_box: bool,
  ) -> Optional[ZonalWeightMatrix]:
    """Loads and reduces a single temporal/lead window into ``data_dict`` and ``missing_fraction``."""
    ds_full = self.loader.ds
    missing_vars = [
        spec.source_var
        for spec in self.band_specs
        if spec.source_var not in ds_full.data_vars
    ]
    if missing_vars:
      raise KeyError(
          f"Required variables {missing_vars} not found in {self.dataset_id}. "
          f"Available: {list(ds_full.data_vars.keys())}"
      )
    present_specs = list(self.band_specs)
    source_vars = [spec.source_var for spec in present_specs]

    lead_slice = self._lead_slice_for_days(max_lead_days)
    sub_ds = self.loader.load_spatial_subset(
        watersheds=basins_gdf,
        variables=source_vars,
        start_date=w_start.strftime("%Y-%m-%d"),
        end_date=w_end.strftime("%Y-%m-%d"),
        lead_time_slice=lead_slice,
        buffer=0.1,
        compute=False,
        use_bounding_box=use_bounding_box,
    )

    if "init_time" in sub_ds.dims and len(sub_ds.init_time) > 0:
      init_times = pd.to_datetime(sub_ds.init_time.values)
      is_00z = (init_times.hour == 0) & (init_times.minute == 0)
      sub_ds = sub_ds.isel(init_time=is_00z)

    if "init_time" not in sub_ds.dims or len(sub_ds.init_time) == 0:
      return weights_matrix

    if len(sub_ds.lead_time) < 2:
      return weights_matrix

    lead_sec = _lead_times_to_seconds(sub_ds["lead_time"])
    pos_lead_mask = lead_sec > 0.0
    if not np.any(pos_lead_mask):
      return weights_matrix

    prev_sec = np.zeros_like(lead_sec)
    prev_sec[pos_lead_mask] = np.concatenate(
        [[0.0], lead_sec[pos_lead_mask][:-1]]
    )
    dt_sec = lead_sec - prev_sec

    sub_ds, matrix = _align_or_build_weights_matrix(
        sub_ds,
        basins_gdf,
        weights_matrix=weights_matrix,
        lat_coord=self.loader.lat_coord,
        lon_coord=self.loader.lon_coord,
        default_res=0.25,
    )
    active_cols, comp_wm = _build_compressed_weight_matrix(matrix)
    k_active = len(active_cols)
    if k_active == 0:
      return matrix

    gathered_vars = _gather_active_cells(
        sub_ds,
        source_vars,
        active_cols,
        n_lons=len(matrix.lons),
        lat_coord=self.loader.lat_coord,
        lon_coord=self.loader.lon_coord,
    )

    n_basins = len(basins_gdf)
    n_time = len(sub_ds.init_time)
    n_lead = len(sub_ds.lead_time)
    has_ens_dim = "ensemble_member" in sub_ds.dims
    n_ens = int(sub_ds.sizes["ensemble_member"]) if has_ens_dim else 1

    reduced_vars: Dict[str, np.ndarray] = {}
    reduced_miss: Dict[str, np.ndarray] = {}

    for spec in present_specs:
      da = sub_ds[spec.source_var]
      arr = gathered_vars[spec.source_var]
      if has_ens_dim:
        # Standardize leading dims to (init_time, ensemble_member, lead_time, K)
        non_sp_dims = list(da.dims[:-2])
        if non_sp_dims == ["init_time", "lead_time", "ensemble_member"]:
          arr = np.transpose(arr, (0, 2, 1, 3))
        flat_3d = arr.reshape(n_time * n_ens * n_lead, k_active, 1)
        red_flat, miss_flat = comp_wm.reduce_3d_with_coverage(flat_3d)
        red = red_flat.reshape(n_basins, n_time, n_ens, n_lead)
        miss = miss_flat.reshape(n_basins, n_time, n_ens, n_lead)
      else:
        flat_3d = arr.reshape(n_time * n_lead, k_active, 1)
        red_flat, miss_flat = comp_wm.reduce_3d_with_coverage(flat_3d)
        red = red_flat.reshape(n_basins, n_time, n_lead)
        miss = miss_flat.reshape(n_basins, n_time, n_lead)

      reduced_vars[spec.base_name] = red
      reduced_miss[spec.base_name] = miss

    sub_init_times = pd.to_datetime(sub_ds.init_time.values).tz_localize(None)

    valid_t_indices: List[int] = []
    target_d_positions: List[int] = []
    for t_idx, t_val in enumerate(sub_init_times):
      dt_day = pd.to_datetime(t_val.strftime("%Y-%m-%d"))
      if dt_day in date_idx:
        valid_t_indices.append(t_idx)
        target_d_positions.append(int(date_idx.get_loc(dt_day)))

    if not valid_t_indices:
      return matrix

    t_sel = np.asarray(valid_t_indices, dtype=np.int64)
    d_sel = np.asarray(target_d_positions, dtype=np.int64)
    needs_trapz = any(s.agg_mode == "trapz" for s in present_specs)
    needs_6h_reset = any(s.agg_mode == "mean_6h_reset" for s in present_specs)

    for lt_day in range(1, max_lead_days + 1):
      t_start = float(lt_day - 1) * 86400.0
      t_end = float(lt_day) * 86400.0
      step_idx = np.where((lead_sec > t_start + 1.0) & (lead_sec <= t_end + 1.0))[0]
      if len(step_idx) == 0:
        continue
      if not np.isclose(prev_sec[step_idx[0]], t_start, atol=1.0):
        continue
      if not np.isclose(lead_sec[step_idx[-1]], t_end, atol=1.0):
        continue
      w_sec = dt_sec[step_idx].astype(np.float32)
      if not bool(np.all(w_sec > 0.0)):
        continue
      total_dt = float(np.sum(w_sec))
      if not np.isclose(total_dt, 86400.0, atol=1.0):
        continue
      w_norm = (w_sec / total_dt).astype(np.float32)

      trapz_idx = np.where(
          (lead_sec >= t_start - 1.0) & (lead_sec <= t_end + 1.0)
      )[0]
      if needs_trapz:
        if (
            len(trapz_idx) < 2
            or not np.isclose(lead_sec[trapz_idx[0]], t_start, atol=1.0)
            or not np.isclose(lead_sec[trapz_idx[-1]], t_end, atol=1.0)
        ):
          continue
        t_sub = lead_sec[trapz_idx]
        w_trapz_64 = np.empty(len(t_sub), dtype=np.float64)
        w_trapz_64[0] = 0.5 * (t_sub[1] - t_sub[0]) / 86400.0
        w_trapz_64[1:-1] = 0.5 * (t_sub[2:] - t_sub[:-2]) / 86400.0
        w_trapz_64[-1] = 0.5 * (t_sub[-1] - t_sub[-2]) / 86400.0
        w_trapz = w_trapz_64.astype(np.float32)
      else:
        w_trapz = np.array([], dtype=np.float32)

      reset_idx = step_idx[
          np.isclose(lead_sec[step_idx] % 21600.0, 0.0, atol=1.0)
      ]
      if needs_6h_reset and len(reset_idx) != 4:
        continue

      lt_pos = lt_day - 1
      any_nan = np.zeros((n_basins, len(t_sel)), dtype=bool)
      day_miss_ens = np.zeros((n_basins, len(t_sel), n_ens), dtype=np.float32)
      day_miss_det = np.zeros((n_basins, len(t_sel)), dtype=np.float32)

      for spec in present_specs:
        red = reduced_vars[spec.base_name][:, t_sel, ...]
        miss_arr = reduced_miss[spec.base_name][:, t_sel, ...]
        if spec.agg_mode == "trapz":
          idx_used = trapz_idx
        elif spec.agg_mode == "mean_6h_reset":
          idx_used = reset_idx
        else:
          idx_used = step_idx

        spec_miss = np.max(miss_arr[..., idx_used], axis=-1)
        day_val = self._aggregate_daily_step(
            red,
            step_idx,
            w_sec,
            w_norm,
            trapz_idx,
            w_trapz,
            reset_idx,
            spec,
        )
        if self.is_ensemble:
          if day_val.ndim == 2:
            day_val = day_val[:, :, np.newaxis]
          if spec_miss.ndim == 2:
            spec_miss = spec_miss[:, :, np.newaxis]
          spec_miss = np.where(
              np.isnan(day_val), np.float32(1.0), spec_miss
          ).astype(np.float32)
          day_miss_ens = np.maximum(day_miss_ens, spec_miss)

          stats = self._compute_ensemble_stats(day_val)
          for stat_name, stat_arr in stats.items():
            band_name = f"{self.band_prefix}_{spec.base_name}_{stat_name}"
            if band_name in data_dict:
              data_dict[band_name][:, d_sel, lt_pos] = stat_arr
          any_nan |= np.isnan(stats["mean"])
          if self.include_ensemble_members:
            ens_band = f"{self.band_prefix}_{spec.base_name}_ensemble"
            if ens_band in ensemble_dict:
              ensemble_dict[ens_band][:, d_sel, :, lt_pos : lt_pos + 1] = (
                  day_val[..., np.newaxis]
              )
        else:
          if day_val.ndim == 3:
            day_val = np.mean(day_val, axis=-1).astype(np.float32)
          if spec_miss.ndim == 3:
            spec_miss = np.mean(spec_miss, axis=-1).astype(np.float32)
          spec_miss = np.where(
              np.isnan(day_val), np.float32(1.0), spec_miss
          ).astype(np.float32)
          day_miss_det = np.maximum(day_miss_det, spec_miss)

          band_name = f"{self.band_prefix}_{spec.base_name}"
          if band_name in data_dict:
            data_dict[band_name][:, d_sel, lt_pos] = day_val
          any_nan |= np.isnan(day_val)

      if self.is_ensemble:
        if self.include_ensemble_members:
          ens_missing_var = ENSEMBLE_MISSING_FRACTION_VAR[self.product]
          if ens_missing_var in ensemble_dict:
            ensemble_dict[ens_missing_var][
                :, d_sel, :, lt_pos : lt_pos + 1
            ] = day_miss_ens[..., np.newaxis]
        day_miss = np.mean(day_miss_ens, axis=-1)
        day_miss = np.where(any_nan, 1.0, day_miss).astype(np.float32)
      else:
        day_miss = np.where(any_nan, 1.0, day_miss_det).astype(np.float32)

      missing_fraction[:, d_sel, lt_pos] = day_miss

    return matrix

  def extract_for_basins(
      self,
      basins_gdf: gpd.GeoDataFrame,
      start_date: Optional[Union[str, pd.Timestamp]] = None,
      end_date: Optional[Union[str, pd.Timestamp]] = None,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      use_bounding_box: bool = True,
      spinup_only_before: Optional[Union[str, pd.Timestamp]] = None,
      **kwargs,
  ) -> xr.Dataset:
    """Extracts daily forecasts (lead days 1..D) for given basins."""
    del kwargs
    if start_date is None or end_date is None:
      raise ValueError(
          f"{self.__class__.__name__}.extract_for_basins requires both "
          "start_date and end_date to be explicitly provided."
      )
    basin_ids = [str(b) for b in basins_gdf.index]
    start_dt = pd.to_datetime(start_date).floor("D")
    end_dt = pd.to_datetime(end_date).floor("D")
    if end_dt < start_dt:
      raise ValueError(
          f"end_date ({end_dt}) must be >= start_date ({start_dt})."
      )
    date_idx = pd.date_range(start_dt, end_dt, freq="D")
    lead_steps = self.lead_days
    lead_time_idx = pd.to_timedelta(range(1, lead_steps + 1), unit="D")

    shape = (len(basin_ids), len(date_idx), lead_steps)
    expected_bands = PRODUCT_BANDS[self.product]
    data_dict: Dict[str, np.ndarray] = {
        band: np.full(shape, np.nan, dtype=np.float32)
        for band in expected_bands
    }
    missing_fraction = np.ones(shape, dtype=np.float32)

    ensemble_dict: Dict[str, np.ndarray] = {}
    ens_coords: Optional[np.ndarray] = None
    if self.is_ensemble and self.include_ensemble_members:
      ds_full = self.loader.ds
      if "ensemble_member" in ds_full.coords:
        ens_coords = np.asarray(ds_full["ensemble_member"].values)
      elif "ensemble_member" in ds_full.dims:
        ens_coords = np.arange(int(ds_full.sizes["ensemble_member"]))
      else:
        ens_coords = np.arange(1)
      ens_shape = (
          len(basin_ids),
          len(date_idx),
          len(ens_coords),
          lead_steps,
      )
      for spec in self.band_specs:
        ens_band = f"{self.band_prefix}_{spec.base_name}_ensemble"
        ensemble_dict[ens_band] = np.full(ens_shape, np.nan, dtype=np.float32)
      ens_missing_var = ENSEMBLE_MISSING_FRACTION_VAR[self.product]
      ensemble_dict[ens_missing_var] = np.ones(ens_shape, dtype=np.float32)

    windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if spinup_only_before is not None:
      cutoff_dt = pd.to_datetime(spinup_only_before).floor("D")
      if start_dt < cutoff_dt:
        spinup_end = min(end_dt, cutoff_dt - pd.Timedelta(days=1))
        windows.append((start_dt, spinup_end, 1))
      if end_dt >= cutoff_dt:
        forecast_start = max(start_dt, cutoff_dt)
        windows.append((forecast_start, end_dt, lead_steps))
    else:
      windows.append((start_dt, end_dt, lead_steps))

    active_matrix = weights_matrix
    for w_start, w_end, w_max_leads in windows:
      active_matrix = self._extract_window(
          basins_gdf=basins_gdf,
          w_start=w_start,
          w_end=w_end,
          max_lead_days=w_max_leads,
          date_idx=date_idx,
          data_dict=data_dict,
          ensemble_dict=ensemble_dict,
          missing_fraction=missing_fraction,
          weights_matrix=active_matrix,
          use_bounding_box=use_bounding_box,
      )

    data_vars: Dict[str, Any] = {
        band: (
            ["basin", "date", "lead_time"],
            data_dict[band].astype(np.float32),
        )
        for band in expected_bands
    }
    for ens_band, ens_arr in ensemble_dict.items():
      data_vars[ens_band] = (
          ["basin", "date", "ensemble_member", "lead_time"],
          ens_arr.astype(np.float32),
      )

    missing_var = MISSING_FRACTION_VAR[self.product]
    data_vars[missing_var] = (
        ["basin", "date", "lead_time"],
        missing_fraction.astype(np.float32),
    )

    coords: Dict[str, Any] = {
        "basin": basin_ids,
        "date": date_idx.values,
        "lead_time": lead_time_idx.values,
    }
    if ens_coords is not None:
      coords["ensemble_member"] = ens_coords

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    if self.product in PRODUCT_METADATA_ATTRS:
      ds.attrs.update(PRODUCT_METADATA_ATTRS[self.product])
    ds.attrs["dynamical_dataset_id"] = self.dataset_id
    return ds

  def extract_day(
      self,
      dt: pd.Timestamp,
      basins_gdf: gpd.GeoDataFrame,
      weights_matrix: Optional[ZonalWeightMatrix] = None,
      **kwargs,
  ) -> Dict[str, np.ndarray]:
    """Extracts 1 forecast initialization date across lead days."""
    ds = self.extract_for_basins(
        basins_gdf,
        start_date=dt,
        end_date=dt,
        weights_matrix=weights_matrix,
        **kwargs,
    )
    return {band: ds[band].values[:, 0, ...] for band in ds.data_vars}


class AIFSExtractor(DynamicalForecastExtractor):
  """Extractor for ECMWF AIFS single-forecast via dynamical.org (8 surface vars)."""

  DEFAULT_PRODUCT = Product.AIFS
  DEFAULT_DATASET_ID = "ecmwf-aifs-single-forecast"
  DEFAULT_BAND_PREFIX = "aifs"
  DEFAULT_IS_ENSEMBLE = False
  DEFAULT_BAND_SPECS = ECMWF_DYNAMICAL_BAND_SPECS


class AIFSEnsExtractor(DynamicalForecastExtractor):
  """Extractor for ECMWF AIFS 51-member ensemble forecast via dynamical.org."""

  DEFAULT_PRODUCT = Product.AIFS_ENS
  DEFAULT_DATASET_ID = "ecmwf-aifs-ens-forecast"
  DEFAULT_BAND_PREFIX = "aifs_ens"
  DEFAULT_IS_ENSEMBLE = True
  DEFAULT_BAND_SPECS = ECMWF_DYNAMICAL_BAND_SPECS


class GFSExtractor(DynamicalForecastExtractor):
  """Extractor for NOAA GFS operational forecasts via dynamical.org (9 surface vars)."""

  DEFAULT_PRODUCT = Product.GFS
  DEFAULT_DATASET_ID = "noaa-gfs-forecast"
  DEFAULT_BAND_PREFIX = "gfs"
  DEFAULT_IS_ENSEMBLE = False
  DEFAULT_BAND_SPECS = NOAA_DYNAMICAL_BAND_SPECS


class GEFSExtractor(DynamicalForecastExtractor):
  """Extractor for NOAA GEFS 31-member ensemble forecasts via dynamical.org."""

  DEFAULT_PRODUCT = Product.GEFS
  DEFAULT_DATASET_ID = "noaa-gefs-forecast-35-day"
  DEFAULT_BAND_PREFIX = "gefs"
  DEFAULT_IS_ENSEMBLE = True
  DEFAULT_BAND_SPECS = NOAA_DYNAMICAL_BAND_SPECS


class IFSEnsExtractor(DynamicalForecastExtractor):
  """Extractor for ECMWF IFS 51-member ensemble (ENS) forecasts via dynamical.org."""

  DEFAULT_PRODUCT = Product.IFS_ENS
  DEFAULT_DATASET_ID = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
  DEFAULT_BAND_PREFIX = "ifs_ens"
  DEFAULT_IS_ENSEMBLE = True
  DEFAULT_BAND_SPECS = ECMWF_DYNAMICAL_BAND_SPECS


def load_dynamical(
    dataset_id: str,
    watersheds: Union[
        gpd.GeoDataFrame,
        shapely.geometry.base.BaseGeometry,
        Sequence[float],
        Tuple[float, float, float, float],
        str,
        Path,
    ],
    variables: Optional[Union[str, Sequence[str]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    lead_time_slice: Optional[Union[slice, Sequence[int]]] = None,
    buffer: Optional[float] = None,
    mode: str = "cube",
    compute: bool = True,
) -> xr.Dataset:
  """Convenience function to load geographically bounded data from dynamical.org."""
  loader = DynamicalDataLoader(dataset_id)
  if mode == "timeseries":
    return loader.extract_basin_timeseries(
        watersheds=watersheds,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        lead_time_slice=lead_time_slice,
        buffer=buffer,
    )
  return loader.load_spatial_subset(
      watersheds=watersheds,
      variables=variables,
      start_date=start_date,
      end_date=end_date,
      lead_time_slice=lead_time_slice,
      buffer=buffer,
      compute=compute,
  )
