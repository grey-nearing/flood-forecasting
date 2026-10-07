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

"""Benchmark harness for MultiMet catchment-averaged timeseries reconstruction.

Reconstructs catchment-averaged MultiMet timeseries (`CPC`, `IMERG`, `HRES`)
from gridded Zarr archives (`--archive-store PRODUCT=URI`) for a benchmark basin
dataset (`--dataset`) over one or more date windows (`--date-windows START:END`)
and compares the reconstructed timeseries against the canonical MultiMet Zarr
stores (`--canonical-dir`).

Key capabilities:
1. Supports `.parquet` / `.geoparquet` (with `geometry_wkt` or `geometry`) as
   well as `.geojson`, `.shp`, and `.gpkg`.
2. Filters `--dataset` basins to those present in the canonical store's `basin`
   coordinate and reports match counts.
3. Detects and flags polygon-area mismatches where spherical polygon area
   (`geom.area * 111^2 * cos(lat)`) differs from `ref_area_km2` by `> 50%`,
   reporting metrics for both `all_matched` and `unrevised_geometry` basins.
4. Precomputes `ZonalWeightMatrix` once per product grid and reuses it across
   all date windows (`use_bounding_box=False`).
5. Computes overall, per-size-tier, per-dataset, per-date-window, per-HRES-tier,
   per-HRES-lead-time, and per-basin metrics (`NSE`, `KGE`, `pearson_r`, `bias`,
   `mae`, `rmse`, `variance_ratio`, `median_abs_err`, `p95_abs_err`,
   `p99_abs_err`, `max_abs_err`, and valid/NaN confusion matrices).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import logging
import os
import time
from typing import Any

import dask
import geopandas as gpd
from multimet.timeseries_extractors.config import (
    FORECAST_LEAD_DAYS,
    PRODUCT_BANDS,
    PRODUCT_TYPES,
    Product,
    ProductType,
)
from multimet.timeseries_extractors.gridded_archive import (
    extract_from_archive,
    get_archive_spec,
    open_gridded_archive,
)
from multimet.timeseries_extractors.runner import _parse_product_uri_pairs
from multimet.timeseries_extractors.zarr_writer import MultiMetZarrWriter
from multimet.utils.geometry import CANDIDATE_ID_COLUMNS, load_basin_geometries
from multimet.utils.zonal import ZonalWeightMatrix
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.validation
from shapely import wkb, wkt
import xarray as xr

logger = logging.getLogger(__name__)

# Strictly supported gridded archive benchmark products (excludes ERA5_LAND and GRAPHCAST).
SUPPORTED_BENCHMARK_PRODUCTS: tuple[str, ...] = ("CPC", "IMERG", "HRES")

# Maximum number of days per single extraction call to bound peak memory while
# reading remote gridded Zarr archives.
MAX_DAYS_PER_EXTRACTION_CHUNK: dict[str, int] = {
    "CPC": 732,
    "IMERG": 62,
    "HRES": 16,
}

# HRES upstream archive tier boundaries in Caravan MultiMet v1.1.
HRES_TIER1_END = pd.Timestamp("2023-01-10")
HRES_TIER2_START = pd.Timestamp("2023-01-11")
HRES_TIER2_END = pd.Timestamp("2023-07-11")
HRES_TIER3_START = pd.Timestamp("2023-07-12")

AREA_MISMATCH_REL_THRESHOLD = 0.50


def classify_hres_upstream_tier(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> str:
  """Classifies a date window into its HRES upstream source tier."""
  s = pd.Timestamp(start_date).normalize()
  e = pd.Timestamp(end_date).normalize()
  if e <= HRES_TIER1_END:
    return "Tier 1 (WB2: 2016-01-01..2023-01-10)"
  if s >= HRES_TIER2_START and e <= HRES_TIER2_END:
    return "Tier 2 (Internal 0.1 deg: 2023-01-11..2023-07-11)"
  if s >= HRES_TIER3_START:
    return "Tier 3 (ECMWF Open Data: 2023-07-12..2024-09-30)"
  return "Mixed Tiers"


def parse_date_windows(
    window_specs: Sequence[str],
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
  """Parses `START:END` date window strings into validated Timestamp tuples."""
  if not window_specs:
    raise ValueError("At least one --date-windows START:END specification is required.")
  parsed: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
  for spec in window_specs:
    for token in str(spec).split(","):
      tok = token.strip()
      if not tok:
        continue
      if ":" not in tok:
        raise ValueError(
            f"Invalid date window {tok!r}; expected format YYYY-MM-DD:YYYY-MM-DD."
        )
      start_str, end_str = tok.split(":", 1)
      start_dt = pd.Timestamp(start_str.strip()).normalize()
      end_dt = pd.Timestamp(end_str.strip()).normalize()
      if end_dt < start_dt:
        raise ValueError(
            f"Invalid date window {tok!r}: end date {end_str} < start date {start_str}."
        )
      label = f"{start_dt.strftime('%Y-%m-%d')}:{end_dt.strftime('%Y-%m-%d')}"
      parsed.append((start_dt, end_dt, label))
  if not parsed:
    raise ValueError("No valid date windows found in --date-windows.")
  return parsed


def _derive_size_tier(area_km2: float) -> str:
  """Assigns a basin size tier from drainage area in km^2."""
  if not np.isfinite(area_km2) or area_km2 <= 0.0:
    return "unknown"
  if area_km2 < 100.0:
    return "1_micro"
  if area_km2 < 500.0:
    return "2_small"
  if area_km2 < 2500.0:
    return "3_medium"
  if area_km2 < 10000.0:
    return "4_large"
  return "5_macro"


def load_benchmark_dataset(
    dataset_path: str,
    *,
    id_column: str | None = None,
    area_mismatch_threshold: float = AREA_MISMATCH_REL_THRESHOLD,
) -> gpd.GeoDataFrame:
  """Loads a benchmark basin dataset and computes polygon-area diagnostics.

  Supports `.parquet` / `.geoparquet` with `geometry_wkt` or `geometry`, as well
  as `.geojson`, `.shp`, and `.gpkg`.

  Args:
    dataset_path: Path to the benchmark basin dataset file.
    id_column: Optional column name for basin/gauge IDs.
    area_mismatch_threshold: Relative area difference threshold above which a
      basin polygon is flagged as `is_geometry_revised = True` (default 0.50).

  Returns:
    GeoDataFrame indexed by `basin_id` with columns:
    `['geometry', 'dataset', 'size_tier', 'ref_area_km2', 'spherical_area_km2',
      'area_rel_diff', 'is_geometry_revised', 'is_out_of_coverage']`.
  """
  if not dataset_path or not str(dataset_path).strip():
    raise ValueError("--dataset path must be a non-empty string.")
  if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Benchmark dataset does not exist: {dataset_path}")

  ext = os.path.splitext(str(dataset_path))[1].lower()
  if ext in (".parquet", ".geoparquet"):
    raw_df = pd.read_parquet(dataset_path)
    if raw_df.empty:
      raise ValueError(f"Benchmark dataset {dataset_path} is empty.")

    # Resolve ID column
    resolved_id_col: str | None = None
    if id_column is not None:
      if id_column not in raw_df.columns:
        raise KeyError(
            f"Specified id_column {id_column!r} not found in {dataset_path}."
        )
      resolved_id_col = id_column
    else:
      for cand in CANDIDATE_ID_COLUMNS:
        if cand in raw_df.columns:
          resolved_id_col = cand
          break

    if resolved_id_col is not None:
      raw_df["basin_id"] = raw_df[resolved_id_col].astype(str)
    elif not isinstance(raw_df.index, pd.RangeIndex):
      raw_df["basin_id"] = raw_df.index.astype(str)
    else:
      raise KeyError(
          f"Could not identify a basin ID column in {dataset_path}. "
          f"Checked {CANDIDATE_ID_COLUMNS}."
      )

    if "geometry_wkt" in raw_df.columns:
      geoms = []
      for val in raw_df["geometry_wkt"]:
        if val is None or pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() == "none":
          geoms.append(shapely.geometry.Polygon())
        else:
          geoms.append(wkt.loads(str(val)))
    elif "geometry" in raw_df.columns:
      geoms = []
      for val in raw_df["geometry"]:
        if val is None or (not isinstance(val, (bytes, bytearray, memoryview)) and pd.isna(val)):
          geoms.append(shapely.geometry.Polygon())
        elif isinstance(val, (bytes, bytearray, memoryview)):
          geoms.append(wkb.loads(bytes(val)))
        elif isinstance(val, str):
          if val.strip() == "" or val.strip().lower() == "none":
            geoms.append(shapely.geometry.Polygon())
          else:
            geoms.append(wkt.loads(val))
        else:
          geoms.append(val)
    else:
      raise KeyError(
          f"Parquet dataset {dataset_path} must contain 'geometry_wkt' or 'geometry'."
      )

    cleaned_geoms = [
        shapely.geometry.Polygon()
        if (g is None or g.is_empty)
        else (shapely.validation.make_valid(g) if not g.is_valid else g)
        for g in geoms
    ]
    gdf = gpd.GeoDataFrame(
        raw_df, geometry=cleaned_geoms, crs="EPSG:4326"
    ).set_index("basin_id")
  else:
    gdf = load_basin_geometries(dataset_path, id_column=id_column)

  if gdf.empty:
    raise ValueError(f"No valid basin geometries loaded from {dataset_path}.")

  if gdf.index.duplicated().any():
    gdf = gdf[~gdf.index.duplicated(keep="first")].copy()

  # Ensure metadata columns exist
  basin_ids = [str(b) for b in gdf.index]
  if "dataset" not in gdf.columns:
    gdf["dataset"] = [
        b.split("_", 1)[0] if "_" in b else "unknown" for b in basin_ids
    ]
  else:
    gdf["dataset"] = gdf["dataset"].astype(str)

  if "delineation_status" in gdf.columns:
    gdf["is_out_of_coverage"] = [
        str(s).startswith("OUT_OF_COVERAGE")
        for s in gdf["delineation_status"].fillna("")
    ]
  else:
    gdf["is_out_of_coverage"] = False

  # Compute spherical polygon area: geom.area * 111^2 * cos(lat)
  has_nonempty_geom = np.array(
      [g is not None and not g.is_empty for g in gdf.geometry], dtype=bool
  )
  centroids_lat = np.array(
      [0.0 if (g is None or g.is_empty) else float(g.centroid.y) for g in gdf.geometry],
      dtype=np.float64,
  )
  deg2_areas = np.array(
      [0.0 if (g is None or g.is_empty) else float(g.area) for g in gdf.geometry],
      dtype=np.float64,
  )
  spherical_area_km2 = (
      deg2_areas * (111.0**2) * np.cos(np.radians(centroids_lat))
  )
  gdf["spherical_area_km2"] = spherical_area_km2

  if "ref_area_km2" in gdf.columns:
    ref_area = pd.to_numeric(gdf["ref_area_km2"], errors="coerce").to_numpy(
        dtype=np.float64
    )
  elif "area_km2" in gdf.columns:
    ref_area = pd.to_numeric(gdf["area_km2"], errors="coerce").to_numpy(
        dtype=np.float64
    )
  else:
    ref_area = np.full(len(gdf), np.nan, dtype=np.float64)
  gdf["ref_area_km2"] = ref_area

  if "size_tier" not in gdf.columns:
    area_for_tier = np.where(
        np.isfinite(ref_area) & (ref_area > 0), ref_area, spherical_area_km2
    )
    gdf["size_tier"] = [_derive_size_tier(float(a)) for a in area_for_tier]
  else:
    gdf["size_tier"] = gdf["size_tier"].astype(str)

  has_ref_area = np.isfinite(ref_area) & (ref_area > 1e-6)
  area_rel_diff = np.where(
      has_nonempty_geom & has_ref_area,
      np.abs(spherical_area_km2 - ref_area) / np.where(has_ref_area, ref_area, 1.0),
      np.where(~has_nonempty_geom, np.nan, 0.0),
  )
  gdf["area_rel_diff"] = area_rel_diff
  gdf["is_geometry_revised"] = bool(True) & (
      has_nonempty_geom & has_ref_area & (area_rel_diff > float(area_mismatch_threshold))
  )

  keep_cols = [
      "geometry",
      "dataset",
      "size_tier",
      "ref_area_km2",
      "spherical_area_km2",
      "area_rel_diff",
      "is_geometry_revised",
      "is_out_of_coverage",
  ]
  return gdf[keep_cols].copy()


def build_product_weight_matrix(
    product: str,
    archive_uri: str,
    basins_gdf: gpd.GeoDataFrame,
    *,
    num_workers: int = 8,
) -> ZonalWeightMatrix:
  """Precomputes the global `ZonalWeightMatrix` once for a product grid."""
  spec = get_archive_spec(product)
  ds = open_gridded_archive(archive_uri)
  raw_lats = np.asarray(ds["latitude"].values, dtype=np.float64)
  raw_lons = np.asarray(ds["longitude"].values, dtype=np.float64)

  if np.any(raw_lons >= 180.0):
    converted_lons = np.where(raw_lons >= 180.0, raw_lons - 360.0, raw_lons)
    lons = np.sort(converted_lons)
  else:
    lons = raw_lons

  res_lat = (
      abs(float(raw_lats[1] - raw_lats[0]))
      if len(raw_lats) > 1
      else spec.cell_res_lat
  )
  res_lon = (
      abs(float(lons[1] - lons[0])) if len(lons) > 1 else spec.cell_res_lon
  )

  return ZonalWeightMatrix.from_geodataframe(
      basins_gdf,
      raw_lats,
      lons,
      cell_res_lat=res_lat,
      cell_res_lon=res_lon,
      num_workers=num_workers,
  )


def extract_window_with_chunking(
    product: str,
    archive_uri: str,
    basins_gdf: gpd.GeoDataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    weights_matrix: ZonalWeightMatrix,
    max_days_per_chunk: int | None = None,
) -> xr.Dataset:
  """Extracts a date window from a gridded archive in memory-safe sub-chunks."""
  prod_key = product.strip().upper()
  chunk_days = (
      max_days_per_chunk
      if max_days_per_chunk is not None
      else MAX_DAYS_PER_EXTRACTION_CHUNK.get(prod_key, 365)
  )
  all_dates = pd.date_range(start_date, end_date, freq="1D")
  if len(all_dates) <= chunk_days:
    return extract_from_archive(
        prod_key,
        archive_uri,
        basins_gdf,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        weights_matrix=weights_matrix,
        use_bounding_box=False,
    )

  datasets: list[xr.Dataset] = []
  for idx in range(0, len(all_dates), chunk_days):
    sub_dates = all_dates[idx : idx + chunk_days]
    ds_part = extract_from_archive(
        prod_key,
        archive_uri,
        basins_gdf,
        sub_dates[0].strftime("%Y-%m-%d"),
        sub_dates[-1].strftime("%Y-%m-%d"),
        weights_matrix=weights_matrix,
        use_bounding_box=False,
    )
    datasets.append(ds_part)
  return xr.concat(datasets, dim="date")


def compute_per_basin_nse_kge(
    ext_2d: np.ndarray,
    can_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes per-basin `(pearson_r, nse, kge)` for arrays of shape `(N_basins, N_pts)`.

  Args:
    ext_2d: Reconstructed values of shape `(N_basins, N_pts)`.
    can_2d: Canonical reference values of shape `(N_basins, N_pts)`.

  Returns:
    Tuple `(r_arr, nse_arr, kge_arr)` of shape `(N_basins,)`.
  """
  n_basins = ext_2d.shape[0]
  r_arr = np.full(n_basins, np.nan, dtype=np.float64)
  nse_arr = np.full(n_basins, np.nan, dtype=np.float64)
  kge_arr = np.full(n_basins, np.nan, dtype=np.float64)

  for b in range(n_basins):
    e_row = ext_2d[b]
    c_row = can_2d[b]
    valid = np.isfinite(e_row) & np.isfinite(c_row)
    n_val = int(np.sum(valid))
    if n_val < 2:
      continue

    ev = e_row[valid].astype(np.float64)
    cv = c_row[valid].astype(np.float64)
    diff = ev - cv
    sse = float(np.sum(diff * diff))

    c_mean = float(np.mean(cv))
    e_mean = float(np.mean(ev))
    c_cent = cv - c_mean
    e_cent = ev - e_mean
    sst = float(np.sum(c_cent * c_cent))
    sse_ext = float(np.sum(e_cent * e_cent))

    if sst <= 1e-12 and sse <= 1e-12:
      r_arr[b] = 1.0
      nse_arr[b] = 1.0
      kge_arr[b] = 1.0
      continue

    if sst > 1e-12:
      nse_arr[b] = 1.0 - (sse / sst)

    if sst > 1e-12 and sse_ext > 1e-12:
      r_val = float(np.sum(e_cent * c_cent) / np.sqrt(sse_ext * sst))
      r_val = float(np.clip(r_val, -1.0, 1.0))
      r_arr[b] = r_val

      c_std = float(np.sqrt(sst / float(n_val)))
      e_std = float(np.sqrt(sse_ext / float(n_val)))
      alpha = e_std / c_std
      if abs(c_mean) > 1e-12:
        beta = e_mean / c_mean
        kge_arr[b] = 1.0 - float(
            np.sqrt((r_val - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        )

  return r_arr, nse_arr, kge_arr


def compute_array_metrics(
    ext_arr: np.ndarray,
    can_arr: np.ndarray,
) -> dict[str, Any]:
  """Computes pooled and per-basin metrics on aligned arrays with leading axis `basin`.

  Args:
    ext_arr: Reconstructed array of shape `(N_basins, ...)`
    can_arr: Canonical array of shape `(N_basins, ...)`

  Returns:
    Dictionary of evaluation metrics.
  """
  if ext_arr.shape != can_arr.shape:
    raise ValueError(
        f"Shape mismatch between extracted {ext_arr.shape} and canonical {can_arr.shape}."
    )

  n_basins = int(ext_arr.shape[0])
  total_points = int(ext_arr.size)
  nan_metric_fields = {
      "pearson_r": float("nan"),
      "bias": float("nan"),
      "mae": float("nan"),
      "rmse": float("nan"),
      "variance_ratio": float("nan"),
      "median_abs_err": float("nan"),
      "p75_abs_err": float("nan"),
      "p90_abs_err": float("nan"),
      "p95_abs_err": float("nan"),
      "p99_abs_err": float("nan"),
      "max_abs_err": float("nan"),
      "frac_within_1e_3": float("nan"),
      "frac_within_1e_5": float("nan"),
      "min_nse": float("nan"),
      "p1_nse": float("nan"),
      "p5_nse": float("nan"),
      "p10_nse": float("nan"),
      "p25_nse": float("nan"),
      "median_nse": float("nan"),
      "mean_nse": float("nan"),
      "uncond_p10_nse": float("nan"),
      "uncond_median_nse": float("nan"),
      "min_kge": float("nan"),
      "p1_kge": float("nan"),
      "p5_kge": float("nan"),
      "p10_kge": float("nan"),
      "p25_kge": float("nan"),
      "median_kge": float("nan"),
      "mean_kge": float("nan"),
      "uncond_p10_kge": float("nan"),
      "uncond_median_kge": float("nan"),
  }
  if total_points == 0 or n_basins == 0:
    return {
        "n_basins": n_basins,
        "total_points": 0,
        "both_valid_count": 0,
        "both_nan_count": 0,
        "extracted_only_nan_count": 0,
        "canonical_only_nan_count": 0,
        "both_valid_pct": float("nan"),
        "both_nan_pct": float("nan"),
        "extracted_only_nan_pct": float("nan"),
        "canonical_only_nan_pct": float("nan"),
        "has_extracted_only_nan_discrepancy": False,
        **nan_metric_fields,
    }

  ext_valid = np.isfinite(ext_arr)
  can_valid = np.isfinite(can_arr)

  both_valid = ext_valid & can_valid
  both_nan = (~ext_valid) & (~can_valid)
  ext_only_nan = (~ext_valid) & can_valid
  can_only_nan = ext_valid & (~can_valid)

  both_valid_count = int(np.sum(both_valid))
  both_nan_count = int(np.sum(both_nan))
  ext_only_nan_count = int(np.sum(ext_only_nan))
  can_only_nan_count = int(np.sum(can_only_nan))

  inv_tot = 100.0 / float(total_points)
  out: dict[str, Any] = {
      "n_basins": n_basins,
      "total_points": total_points,
      "both_valid_count": both_valid_count,
      "both_nan_count": both_nan_count,
      "extracted_only_nan_count": ext_only_nan_count,
      "canonical_only_nan_count": can_only_nan_count,
      "both_valid_pct": float(both_valid_count * inv_tot),
      "both_nan_pct": float(both_nan_count * inv_tot),
      "extracted_only_nan_pct": float(ext_only_nan_count * inv_tot),
      "canonical_only_nan_pct": float(can_only_nan_count * inv_tot),
      "has_extracted_only_nan_discrepancy": bool(ext_only_nan_count > 0),
  }

  if both_valid_count == 0:
    out.update(nan_metric_fields)
    return out

  ev = ext_arr[both_valid].astype(np.float64)
  cv = can_arr[both_valid].astype(np.float64)
  diff = ev - cv
  abs_err = np.abs(diff)

  bias = float(np.mean(diff))
  mae = float(np.mean(abs_err))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  p50, p75, p90, p95, p99 = np.percentile(
      abs_err, [50.0, 75.0, 90.0, 95.0, 99.0]
  )
  max_abs = float(np.max(abs_err))
  frac_1e3 = float(np.mean(abs_err <= 1e-3))
  frac_1e5 = float(np.mean(abs_err <= 1e-5))

  e_mean = float(np.mean(ev))
  c_mean = float(np.mean(cv))
  e_var = float(np.var(ev))
  c_var = float(np.var(cv))

  if c_var > 1e-12:
    variance_ratio = float(e_var / c_var)
  elif e_var <= 1e-12:
    variance_ratio = 1.0
  else:
    variance_ratio = float("nan")

  if both_valid_count >= 2 and e_var > 1e-12 and c_var > 1e-12:
    e_cent = ev - e_mean
    c_cent = cv - c_mean
    denom = float(np.sqrt(np.sum(e_cent * e_cent) * np.sum(c_cent * c_cent)))
    pearson_r = (
        float(np.sum(e_cent * c_cent) / denom)
        if denom > 1e-12
        else float("nan")
    )
  elif e_var <= 1e-12 and c_var <= 1e-12 and max_abs <= 1e-12:
    pearson_r = 1.0
  else:
    pearson_r = float("nan")

  ext_2d = ext_arr.reshape(n_basins, -1)
  can_2d = can_arr.reshape(n_basins, -1)
  _, nse_arr, kge_arr = compute_per_basin_nse_kge(ext_2d, can_2d)

  valid_nse = nse_arr[np.isfinite(nse_arr)]
  valid_kge = kge_arr[np.isfinite(kge_arr)]

  # Unconditional per-basin NSE/KGE: basins that have extracted-only NaN points
  # and failed NSE/KGE (NaN) are penalized as -1e9 so failed basins are never
  # excluded from lower-tail percentiles.
  basin_has_ext_only_nan = np.any((~np.isfinite(ext_2d)) & np.isfinite(can_2d), axis=1)
  uncond_nse_list: list[float] = []
  uncond_kge_list: list[float] = []
  for b_idx in range(n_basins):
    if np.isfinite(nse_arr[b_idx]):
      uncond_nse_list.append(float(nse_arr[b_idx]))
    elif basin_has_ext_only_nan[b_idx]:
      uncond_nse_list.append(-1e9)
    if np.isfinite(kge_arr[b_idx]):
      uncond_kge_list.append(float(kge_arr[b_idx]))
    elif basin_has_ext_only_nan[b_idx]:
      uncond_kge_list.append(-1e9)
  uncond_nse = np.asarray(uncond_nse_list, dtype=np.float64)
  uncond_kge = np.asarray(uncond_kge_list, dtype=np.float64)

  out.update({
      "pearson_r": pearson_r,
      "bias": bias,
      "mae": mae,
      "rmse": rmse,
      "variance_ratio": variance_ratio,
      "median_abs_err": float(p50),
      "p75_abs_err": float(p75),
      "p90_abs_err": float(p90),
      "p95_abs_err": float(p95),
      "p99_abs_err": float(p99),
      "max_abs_err": max_abs,
      "frac_within_1e_3": frac_1e3,
      "frac_within_1e_5": frac_1e5,
      "min_nse": float(np.min(valid_nse)) if len(valid_nse) > 0 else float("nan"),
      "p1_nse": float(np.percentile(valid_nse, 1.0)) if len(valid_nse) > 0 else float("nan"),
      "p5_nse": float(np.percentile(valid_nse, 5.0)) if len(valid_nse) > 0 else float("nan"),
      "p10_nse": float(np.percentile(valid_nse, 10.0)) if len(valid_nse) > 0 else float("nan"),
      "p25_nse": float(np.percentile(valid_nse, 25.0)) if len(valid_nse) > 0 else float("nan"),
      "median_nse": float(np.median(valid_nse)) if len(valid_nse) > 0 else float("nan"),
      "mean_nse": float(np.mean(valid_nse)) if len(valid_nse) > 0 else float("nan"),
      "uncond_p10_nse": float(np.percentile(uncond_nse, 10.0)) if len(uncond_nse) > 0 else float("nan"),
      "uncond_median_nse": float(np.median(uncond_nse)) if len(uncond_nse) > 0 else float("nan"),
      "min_kge": float(np.min(valid_kge)) if len(valid_kge) > 0 else float("nan"),
      "p1_kge": float(np.percentile(valid_kge, 1.0)) if len(valid_kge) > 0 else float("nan"),
      "p5_kge": float(np.percentile(valid_kge, 5.0)) if len(valid_kge) > 0 else float("nan"),
      "p10_kge": float(np.percentile(valid_kge, 10.0)) if len(valid_kge) > 0 else float("nan"),
      "p25_kge": float(np.percentile(valid_kge, 25.0)) if len(valid_kge) > 0 else float("nan"),
      "median_kge": float(np.median(valid_kge)) if len(valid_kge) > 0 else float("nan"),
      "mean_kge": float(np.mean(valid_kge)) if len(valid_kge) > 0 else float("nan"),
      "uncond_p10_kge": float(np.percentile(uncond_kge, 10.0)) if len(uncond_kge) > 0 else float("nan"),
      "uncond_median_kge": float(np.median(uncond_kge)) if len(uncond_kge) > 0 else float("nan"),
  })
  return out


def compute_per_basin_dataframe(
    *,
    product: str,
    variable: str,
    basins_gdf: gpd.GeoDataFrame,
    ext_arr: np.ndarray,
    can_arr: np.ndarray,
) -> pd.DataFrame:
  """Computes a detailed per-basin metric table for `(product, variable)`."""
  n_basins = len(basins_gdf)
  ext_2d = ext_arr.reshape(n_basins, -1)
  can_2d = can_arr.reshape(n_basins, -1)
  r_arr, nse_arr, kge_arr = compute_per_basin_nse_kge(ext_2d, can_2d)

  rows: list[dict[str, Any]] = []
  for b_idx, (basin_id, meta) in enumerate(basins_gdf.iterrows()):
    e_row = ext_2d[b_idx]
    c_row = can_2d[b_idx]
    n_pts = int(e_row.size)
    e_val = np.isfinite(e_row)
    c_val = np.isfinite(c_row)
    bv = e_val & c_val
    bn = (~e_val) & (~c_val)
    eo = (~e_val) & c_val
    co = e_val & (~c_val)

    bv_cnt = int(np.sum(bv))
    inv_n = 100.0 / float(n_pts) if n_pts > 0 else float("nan")

    if bv_cnt > 0:
      ev = e_row[bv].astype(np.float64)
      cv = c_row[bv].astype(np.float64)
      diff = ev - cv
      abs_err = np.abs(diff)
      bias = float(np.mean(diff))
      mae = float(np.mean(abs_err))
      rmse = float(np.sqrt(np.mean(diff * diff)))
      p50, p75, p90, p95, p99 = np.percentile(
          abs_err, [50.0, 75.0, 90.0, 95.0, 99.0]
      )
      max_abs = float(np.max(abs_err))
      e_var = float(np.var(ev))
      c_var = float(np.var(cv))
      var_ratio = (
          float(e_var / c_var)
          if c_var > 1e-12
          else (1.0 if e_var <= 1e-12 else float("nan"))
      )
    else:
      bias = float("nan")
      mae = float("nan")
      rmse = float("nan")
      p50 = float("nan")
      p75 = float("nan")
      p90 = float("nan")
      p95 = float("nan")
      p99 = float("nan")
      max_abs = float("nan")
      var_ratio = float("nan")

    rows.append({
        "product": product,
        "variable": variable,
        "gauge_id": str(basin_id),
        "dataset": str(meta["dataset"]),
        "size_tier": str(meta["size_tier"]),
        "ref_area_km2": float(meta["ref_area_km2"]),
        "spherical_area_km2": float(meta["spherical_area_km2"]),
        "area_rel_diff": float(meta["area_rel_diff"]),
        "is_geometry_revised": bool(meta["is_geometry_revised"]),
        "total_points": n_pts,
        "both_valid_count": bv_cnt,
        "both_nan_count": int(np.sum(bn)),
        "extracted_only_nan_count": int(np.sum(eo)),
        "canonical_only_nan_count": int(np.sum(co)),
        "both_valid_pct": float(bv_cnt * inv_n),
        "both_nan_pct": float(int(np.sum(bn)) * inv_n),
        "extracted_only_nan_pct": float(int(np.sum(eo)) * inv_n),
        "canonical_only_nan_pct": float(int(np.sum(co)) * inv_n),
        "has_extracted_only_nan_discrepancy": bool(int(np.sum(eo)) > 0),
        "pearson_r": float(r_arr[b_idx]),
        "nse": float(nse_arr[b_idx]),
        "kge": float(kge_arr[b_idx]),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "variance_ratio": var_ratio,
        "median_abs_err": float(p50),
        "p75_abs_err": float(p75),
        "p90_abs_err": float(p90),
        "p95_abs_err": float(p95),
        "p99_abs_err": float(p99),
        "max_abs_err": max_abs,
    })

  return pd.DataFrame(rows)


def _align_canonical_slice(
    ds_canon: xr.Dataset,
    canonical_basin_map: Mapping[str, Any],
    matched_basin_ids: Sequence[str],
    target_dates: pd.DatetimeIndex,
    *,
    is_forecast: bool,
    lead_days: int = 10,
) -> xr.Dataset:
  """Slices and aligns a canonical MultiMet Zarr dataset to `(basin, date[, lead_time])`."""
  raw_basin_keys = [canonical_basin_map[b] for b in matched_basin_ids]
  canon_dates = pd.DatetimeIndex(
      pd.to_datetime(ds_canon["date"].values).floor("D")
  )
  date_to_canon_idx = {d: i for i, d in enumerate(canon_dates)}

  present_target_positions: list[int] = []
  canon_date_indices: list[int] = []
  for pos, d in enumerate(target_dates):
    if d in date_to_canon_idx:
      present_target_positions.append(pos)
      canon_date_indices.append(date_to_canon_idx[d])

  if not canon_date_indices:
    raise ValueError(
        f"None of the requested dates [{target_dates[0].strftime('%Y-%m-%d')} .. "
        f"{target_dates[-1].strftime('%Y-%m-%d')}] exist in the canonical Zarr store."
    )

  ds_sub = ds_canon.sel(basin=raw_basin_keys).isel(date=canon_date_indices)
  ds_sub = ds_sub.assign_coords(
      basin=list(matched_basin_ids),
      date=target_dates[present_target_positions].values,
  )
  if len(present_target_positions) != len(target_dates):
    ds_sub = ds_sub.reindex(date=target_dates.values, fill_value=np.nan)

  if is_forecast:
    raw_leads = ds_sub["lead_time"].values
    if np.issubdtype(raw_leads.dtype, np.timedelta64):
      lead_ints = (
          pd.to_timedelta(raw_leads) / pd.Timedelta(days=1)
      ).astype(int)
    else:
      lead_ints = np.asarray(raw_leads, dtype=int)
    ds_sub = ds_sub.assign_coords(lead_time=lead_ints)
    ds_sub = ds_sub.reindex(
        lead_time=np.arange(1, lead_days + 1, dtype=int), fill_value=np.nan
    )

  return ds_sub


def _write_benchmark_markdown_report(
    output_dir: str,
    *,
    dataset_path: str,
    canonical_dir: str,
    archive_stores: Mapping[str, str],
    total_input_basins: int,
    matched_gdf: gpd.GeoDataFrame,
    summary_df: pd.DataFrame,
    size_tier_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    window_df: pd.DataFrame,
    hres_lead_df: pd.DataFrame,
    timings: Mapping[str, dict[str, float]],
) -> str:
  """Writes `benchmark_report.md` summarizing all reconstruction benchmark tables.

  All primary headline tables (Sections 2 through 6) report `all_matched`
  (100% of canonical-matched basins, unmasked, zero outlier filtering) alongside
  explicit 4-way valid/NaN mask counts, percentages, and missing-data discrepancy
  flags.
  """
  report_path = os.path.join(output_dir, "benchmark_report.md")
  n_matched = len(matched_gdf)
  revised_gdf = matched_gdf[matched_gdf["is_geometry_revised"]]
  n_revised = len(revised_gdf)
  n_unrevised = n_matched - n_revised

  all_matched_summary = summary_df[summary_df["basin_subset"] == "all_matched"]
  if all_matched_summary.empty:
    all_matched_summary = summary_df
  total_ext_only_nan = int(all_matched_summary["extracted_only_nan_count"].sum())
  nan_audit_status = (
      f"DISCREPANCY DETECTED ({total_ext_only_nan:,} extracted-only NaN points across all_matched)"
      if total_ext_only_nan > 0
      else "PASS (0 extracted-only NaN points across all_matched)"
  )

  lines: list[str] = [
      "# MultiMet Catchment Timeseries Reconstruction Benchmark Report",
      "",
      "## 1. Benchmark Configuration, Missing-Data Audit & Basin Matching Summary",
      "",
      f"- **Input Basin Dataset**: `{dataset_path}`",
      f"- **Canonical MultiMet Store Root**: `{canonical_dir}`",
      f"- **Total Input Basins**: `{total_input_basins}`",
      f"- **Canonical Matched Basins (`all_matched` — Primary Unmasked Headline Set)**: `{n_matched} / {total_input_basins}`",
      f"- **Missing-Data Audit (`extracted_only_nan`)**: **{nan_audit_status}**",
      "- **Evaluation Integrity Protocol**:",
      "  1. **Zero Fallback to Canonical Data**: Reconstructed values come strictly from zonal extraction over the gridded Zarr archives.",
      "  2. **Zero Imputation or Interpolation**: Missing values remain `NaN` and are audited in the 4-way confusion matrix.",
      "  3. **Zero Silent `NaN` Masking**: Every metric table below includes exact counts and percentages for `Both Valid`, `Both NaN`, `Ext-Only NaN`, and `Canon-Only NaN`, plus an explicit discrepancy flag whenever `extracted_only_nan_count > 0`.",
      f"  4. **Zero Outlier Basin Filtering in Headline Tables**: All `{n_matched}` canonical-matched basins (`all_matched`) are reported unmasked as the primary table in Sections 2–6 (`{n_unrevised}` unrevised-geometry basins and `{n_revised}` polygon-area-mismatch basins).",
      "",
      "### Gridded Archive Stores & Wall-Clock Performance",
      "",
      "| Product | Gridded Archive URI | Weight Matrix Build (s) | Zonal Extraction (s) | Canonical Read & Eval (s) |",
      "| :--- | :--- | ---: | ---: | ---: |",
  ]
  for prod, uri in archive_stores.items():
    t_info = timings.get(prod, {})
    lines.append(
        f"| `{prod}` | `{uri}` | "
        f"{t_info.get('weights_sec', 0.0):.2f} | "
        f"{t_info.get('extract_sec', 0.0):.2f} | "
        f"{t_info.get('eval_sec', 0.0):.2f} |"
    )

  lines.extend([
      "",
      f"## 2. Primary Headline Summary Metrics (`all_matched`, n={n_matched} Unmasked Basins)",
      "",
      "| Product | Variable | Basins | Total Points | Both Valid (Count / %) | Both NaN (Count / %) | Ext-Only NaN (Count / %) | Canon-Only NaN (Count / %) | NaN Flag | Pearson $r$ | Median NSE | Mean NSE | P10 NSE | Median KGE | Mean KGE | P10 KGE | Bias | MAE | RMSE | Var Ratio | P50 Err | P95 Err | P99 Err | Max Abs Err |",
      "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for _, row in all_matched_summary.iterrows():
    eo_cnt = int(row["extracted_only_nan_count"])
    flag_str = f"DISCREPANCY ({eo_cnt:,})" if eo_cnt > 0 else "OK (0)"
    lines.append(
        f"| `{row['product']}` | `{row['variable']}` | "
        f"{int(row['n_basins'])} | {int(row['total_points']):,} | "
        f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.3f}%) | "
        f"{int(row['both_nan_count']):,} ({row['both_nan_pct']:.3f}%) | "
        f"{eo_cnt:,} ({row['extracted_only_nan_pct']:.4f}%) | "
        f"{int(row['canonical_only_nan_count']):,} ({row['canonical_only_nan_pct']:.4f}%) | "
        f"`{flag_str}` | "
        f"{row['pearson_r']:.6f} | {row['median_nse']:.6f} | "
        f"{row['mean_nse']:.6f} | {row['p10_nse']:.6f} | "
        f"{row['median_kge']:.6f} | {row['mean_kge']:.6f} | "
        f"{row['p10_kge']:.6f} | {row['bias']:.4e} | "
        f"{row['mae']:.4e} | {row['rmse']:.4e} | "
        f"{row['variance_ratio']:.6f} | {row['median_abs_err']:.4e} | "
        f"{row['p95_abs_err']:.4e} | {row['p99_abs_err']:.4e} | "
        f"{row['max_abs_err']:.4e} |"
    )

  lines.extend([
      "",
      f"## 3. Headline Metrics by Date Window & Upstream Source Tier (`all_matched`, n={n_matched} Unmasked Basins)",
      "",
      "| Product | Variable | Date Window | Upstream Tier | Basins | Total Points | Both Valid (Count / %) | Both NaN (Count / %) | Ext-Only NaN (Count / %) | Canon-Only NaN (Count / %) | NaN Flag | Pearson $r$ | Median NSE | Mean NSE | Median KGE | MAE | RMSE | Max Abs Err |",
      "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  win_all = window_df[window_df["basin_subset"] == "all_matched"]
  if win_all.empty:
    win_all = window_df
  for _, row in win_all.iterrows():
    eo_cnt = int(row["extracted_only_nan_count"])
    flag_str = f"DISCREPANCY ({eo_cnt:,})" if eo_cnt > 0 else "OK (0)"
    lines.append(
        f"| `{row['product']}` | `{row['variable']}` | `{row['window']}` | "
        f"{row['upstream_tier']} | {int(row['n_basins'])} | {int(row['total_points']):,} | "
        f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.3f}%) | "
        f"{int(row['both_nan_count']):,} ({row['both_nan_pct']:.3f}%) | "
        f"{eo_cnt:,} ({row['extracted_only_nan_pct']:.4f}%) | "
        f"{int(row['canonical_only_nan_count']):,} ({row['canonical_only_nan_pct']:.4f}%) | "
        f"`{flag_str}` | "
        f"{row['pearson_r']:.6f} | {row['median_nse']:.6f} | "
        f"{row['mean_nse']:.6f} | {row['median_kge']:.6f} | "
        f"{row['mae']:.4e} | {row['rmse']:.4e} | {row['max_abs_err']:.4e} |"
    )

  lines.extend([
      "",
      f"## 4. Headline Metrics by Basin Size Tier (`all_matched`, n={n_matched} Unmasked Basins)",
      "",
      "| Product | Variable | Size Tier | Basins | Total Points | Both Valid (Count / %) | Both NaN (Count / %) | Ext-Only NaN (Count / %) | Canon-Only NaN (Count / %) | NaN Flag | Pearson $r$ | Median NSE | Mean NSE | Median KGE | Bias | MAE | RMSE | P95 Err | Max Abs Err |",
      "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  tier_all = size_tier_df[size_tier_df["basin_subset"] == "all_matched"]
  if tier_all.empty:
    tier_all = size_tier_df
  for _, row in tier_all.iterrows():
    eo_cnt = int(row["extracted_only_nan_count"])
    flag_str = f"DISCREPANCY ({eo_cnt:,})" if eo_cnt > 0 else "OK (0)"
    lines.append(
        f"| `{row['product']}` | `{row['variable']}` | `{row['size_tier']}` | "
        f"{int(row['n_basins'])} | {int(row['total_points']):,} | "
        f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.3f}%) | "
        f"{int(row['both_nan_count']):,} ({row['both_nan_pct']:.3f}%) | "
        f"{eo_cnt:,} ({row['extracted_only_nan_pct']:.4f}%) | "
        f"{int(row['canonical_only_nan_count']):,} ({row['canonical_only_nan_pct']:.4f}%) | "
        f"`{flag_str}` | "
        f"{row['pearson_r']:.6f} | {row['median_nse']:.6f} | "
        f"{row['mean_nse']:.6f} | {row['median_kge']:.6f} | "
        f"{row['bias']:.4e} | {row['mae']:.4e} | {row['rmse']:.4e} | "
        f"{row['p95_abs_err']:.4e} | {row['max_abs_err']:.4e} |"
    )

  lines.extend([
      "",
      f"## 5. Headline Metrics by Source Dataset (`all_matched`, n={n_matched} Unmasked Basins)",
      "",
      "| Product | Variable | Dataset | Basins | Total Points | Both Valid (Count / %) | Both NaN (Count / %) | Ext-Only NaN (Count / %) | Canon-Only NaN (Count / %) | NaN Flag | Pearson $r$ | Median NSE | Mean NSE | Median KGE | Bias | MAE | RMSE | P95 Err | Max Abs Err |",
      "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  ds_all = dataset_df[dataset_df["basin_subset"] == "all_matched"]
  if ds_all.empty:
    ds_all = dataset_df
  for _, row in ds_all.iterrows():
    eo_cnt = int(row["extracted_only_nan_count"])
    flag_str = f"DISCREPANCY ({eo_cnt:,})" if eo_cnt > 0 else "OK (0)"
    lines.append(
        f"| `{row['product']}` | `{row['variable']}` | `{row['dataset']}` | "
        f"{int(row['n_basins'])} | {int(row['total_points']):,} | "
        f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.3f}%) | "
        f"{int(row['both_nan_count']):,} ({row['both_nan_pct']:.3f}%) | "
        f"{eo_cnt:,} ({row['extracted_only_nan_pct']:.4f}%) | "
        f"{int(row['canonical_only_nan_count']):,} ({row['canonical_only_nan_pct']:.4f}%) | "
        f"`{flag_str}` | "
        f"{row['pearson_r']:.6f} | {row['median_nse']:.6f} | "
        f"{row['mean_nse']:.6f} | {row['median_kge']:.6f} | "
        f"{row['bias']:.4e} | {row['mae']:.4e} | {row['rmse']:.4e} | "
        f"{row['p95_abs_err']:.4e} | {row['max_abs_err']:.4e} |"
    )

  if not hres_lead_df.empty:
    lines.extend([
        "",
        f"## 6. Headline HRES Forecast Lead-Time Breakdown (`all_matched`, n={n_matched} Unmasked Basins)",
        "",
        "| Variable | Upstream Tier | Lead Day | Basins | Total Points | Both Valid (Count / %) | Both NaN (Count / %) | Ext-Only NaN (Count / %) | Canon-Only NaN (Count / %) | NaN Flag | Pearson $r$ | Median NSE | Mean NSE | Median KGE | MAE | RMSE | Max Abs Err |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    lead_all = hres_lead_df[hres_lead_df["basin_subset"] == "all_matched"]
    if lead_all.empty:
      lead_all = hres_lead_df
    for _, row in lead_all.iterrows():
      eo_cnt = int(row["extracted_only_nan_count"])
      flag_str = f"DISCREPANCY ({eo_cnt:,})" if eo_cnt > 0 else "OK (0)"
      lines.append(
          f"| `{row['variable']}` | {row['upstream_tier']} | {int(row['lead_time'])} | "
          f"{int(row['n_basins'])} | {int(row['total_points']):,} | "
          f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.3f}%) | "
          f"{int(row['both_nan_count']):,} ({row['both_nan_pct']:.3f}%) | "
          f"{eo_cnt:,} ({row['extracted_only_nan_pct']:.4f}%) | "
          f"{int(row['canonical_only_nan_count']):,} ({row['canonical_only_nan_pct']:.4f}%) | "
          f"`{flag_str}` | "
          f"{row['pearson_r']:.6f} | {row['median_nse']:.6f} | "
          f"{row['mean_nse']:.6f} | {row['median_kge']:.6f} | "
          f"{row['mae']:.4e} | {row['rmse']:.4e} | {row['max_abs_err']:.4e} |"
      )

  if n_revised > 0:
    lines.extend([
        "",
        "## 7. Diagnostic Appendix: Polygon-Area Revision Attribution (Informational Context Only)",
        "",
        "> **Note**: All primary headline tables in Sections 2–6 include 100% of canonical-matched basins (`all_matched`) with zero outlier filtering. The diagnostic breakdown below is provided solely to attribute differences caused by upstream catchment polygon revisions ($|\\Delta A| / A_{\\text{ref}} > 50\\%$).",
        "",
        "| Gauge ID | Dataset | Size Tier | Reference Area ($\\text{km}^2$) | Spherical Polygon Area ($\\text{km}^2$) | Relative Diff ($\\times$) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: |",
    ])
    for gid, r_row in revised_gdf.iterrows():
      lines.append(
          f"| `{gid}` | `{r_row['dataset']}` | `{r_row['size_tier']}` | "
          f"{r_row['ref_area_km2']:.2f} | "
          f"{r_row['spherical_area_km2']:.2f} | "
          f"{r_row['area_rel_diff']:.2f}x |"
      )

    lines.extend([
        "",
        "### Diagnostic Subset Comparison (`all_matched` vs. `unrevised_geometry` vs. `revised_geometry`)",
        "",
        "| Product | Variable | Basin Subset | Basins | Total Points | Both Valid (Count / %) | Ext-Only NaN (Count / %) | Canon-Only NaN (Count / %) | Pearson $r$ | Median NSE | Mean NSE | Median KGE | MAE | RMSE | Max Abs Err |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in summary_df.iterrows():
      lines.append(
          f"| `{row['product']}` | `{row['variable']}` | `{row['basin_subset']}` | "
          f"{int(row['n_basins'])} | {int(row['total_points']):,} | "
          f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.3f}%) | "
          f"{int(row['extracted_only_nan_count']):,} ({row['extracted_only_nan_pct']:.4f}%) | "
          f"{int(row['canonical_only_nan_count']):,} ({row['canonical_only_nan_pct']:.4f}%) | "
          f"{row['pearson_r']:.6f} | {row['median_nse']:.6f} | "
          f"{row['mean_nse']:.6f} | {row['median_kge']:.6f} | "
          f"{row['mae']:.4e} | {row['rmse']:.4e} | {row['max_abs_err']:.4e} |"
      )

  with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return report_path


def run_timeseries_benchmark(
    *,
    dataset_path: str,
    canonical_dir: str,
    archive_stores: Mapping[str, str],
    date_windows: Sequence[str],
    output_dir: str,
    products: Sequence[str] | None = None,
    save_reconstructed_zarr: str | None = None,
    id_column: str | None = None,
    area_mismatch_threshold: float = AREA_MISMATCH_REL_THRESHOLD,
    num_workers: int = 8,
) -> dict[str, pd.DataFrame]:
  """Runs the MultiMet catchment timeseries reconstruction benchmark.

  Args:
    dataset_path: Path to the benchmark basin dataset (`.parquet`, `.geojson`,
      `.shp`).
    canonical_dir: Explicit path or `gs://` URI to the canonical MultiMet Zarr
      root directory (containing `<PRODUCT>/timeseries.zarr`).
    archive_stores: Mapping of product name (`CPC`, `IMERG`, `HRES`) to its
      explicit gridded archive Zarr URI or local path.
    date_windows: Sequence of `START:END` date window strings.
    output_dir: Directory where CSVs, Parquet per-basin table, and Markdown
      report are written.
    products: Optional subset of products to benchmark; defaults to keys of
      `archive_stores`.
    save_reconstructed_zarr: Optional directory to write reconstructed Zarr
      stores (`<dir>/<PRODUCT>/timeseries.zarr`).
    id_column: Optional basin ID column name in `dataset_path`.
    area_mismatch_threshold: Relative difference threshold (`> 0.50`) used to
      flag polygon-area mismatches against `ref_area_km2`.
    num_workers: Number of parallel threads for `ZonalWeightMatrix` building.

  Returns:
    Dictionary of output DataFrames (`summary`, `by_size_tier`, `by_dataset`,
    `by_window`, `hres_by_lead`, `per_basin`).
  """
  if not canonical_dir or not str(canonical_dir).strip():
    raise ValueError("--canonical-dir must be explicitly provided.")
  if not output_dir or not str(output_dir).strip():
    raise ValueError("--output-dir must be explicitly provided.")
  if not archive_stores:
    raise ValueError(
        "At least one --archive-store PRODUCT=URI must be explicitly provided."
    )

  norm_stores: dict[str, str] = {
      str(k).strip().upper(): str(v).strip()
      for k, v in archive_stores.items()
  }
  target_products = (
      [str(p).strip().upper() for p in products]
      if products is not None
      else list(norm_stores.keys())
  )
  if not target_products:
    raise ValueError("No products specified for timeseries benchmark.")

  for prod_key in target_products:
    if prod_key not in SUPPORTED_BENCHMARK_PRODUCTS:
      raise ValueError(
          f"Unsupported benchmark product {prod_key!r}. Supported products: "
          f"{SUPPORTED_BENCHMARK_PRODUCTS} (ERA5_LAND and GRAPHCAST are excluded)."
      )
    if prod_key not in norm_stores:
      raise ValueError(
          f"Missing --archive-store URI for requested product {prod_key!r}."
      )

  parsed_windows = parse_date_windows(date_windows)
  os.makedirs(output_dir, exist_ok=True)

  full_gdf = load_benchmark_dataset(
      dataset_path,
      id_column=id_column,
      area_mismatch_threshold=area_mismatch_threshold,
  )
  total_input_basins = len(full_gdf)

  # Open the first product's canonical store to determine canonical basin list
  first_canon_uri = os.path.join(
      str(canonical_dir).rstrip("/"), target_products[0], "timeseries.zarr"
  )
  first_canon_ds = open_gridded_archive(first_canon_uri)
  if "basin" not in first_canon_ds.coords:
    raise KeyError(
        f"Canonical store {first_canon_uri} is missing 'basin' coordinate."
    )
  canon_basin_map: dict[str, Any] = {
      str(b).rstrip("\x00"): b for b in first_canon_ds["basin"].values
  }

  n_out_of_coverage = int(full_gdf["is_out_of_coverage"].sum())
  in_cov_gdf = full_gdf[~full_gdf["is_out_of_coverage"]].copy()

  matched_ids = [
      str(b) for b in in_cov_gdf.index if str(b) in canon_basin_map
  ]
  if not matched_ids:
    raise ValueError(
        f"Zero in-coverage basins from {dataset_path} (n={len(in_cov_gdf)}, "
        f"out_of_coverage={n_out_of_coverage}) matched "
        f"canonical store {first_canon_uri} (n={len(canon_basin_map)})."
    )

  matched_gdf = in_cov_gdf.loc[matched_ids].copy()
  n_revised = int(matched_gdf["is_geometry_revised"].sum())
  logger.info(
      "Matched %d / %d in-coverage basins against canonical store (%d out of coverage, "
      "%d unrevised geometry, %d revised geometry > %.0f%% area diff).",
      len(matched_gdf),
      len(in_cov_gdf),
      n_out_of_coverage,
      len(matched_gdf) - n_revised,
      n_revised,
      area_mismatch_threshold * 100.0,
  )

  # Define basin subsets to evaluate
  unrevised_mask = ~matched_gdf["is_geometry_revised"].to_numpy(dtype=bool)
  revised_mask = matched_gdf["is_geometry_revised"].to_numpy(dtype=bool)
  basin_subsets: list[tuple[str, np.ndarray]] = [
      ("all_matched", np.ones(len(matched_gdf), dtype=bool)),
  ]
  if int(np.sum(unrevised_mask)) > 0:
    basin_subsets.append(("unrevised_geometry", unrevised_mask))
  if int(np.sum(revised_mask)) > 0:
    basin_subsets.append(("revised_geometry", revised_mask))

  writer = (
      MultiMetZarrWriter(save_reconstructed_zarr)
      if save_reconstructed_zarr is not None
      else None
  )

  summary_rows: list[dict[str, Any]] = []
  size_tier_rows: list[dict[str, Any]] = []
  dataset_rows: list[dict[str, Any]] = []
  window_rows: list[dict[str, Any]] = []
  hres_lead_rows: list[dict[str, Any]] = []
  per_basin_dfs: list[pd.DataFrame] = []
  timings: dict[str, dict[str, float]] = {}

  expected_date_list: list[pd.Timestamp] = []
  for start_dt, end_dt, _ in parsed_windows:
    expected_date_list.extend(pd.date_range(start_dt, end_dt, freq="1D"))
  expected_all_dates = pd.DatetimeIndex(sorted(set(expected_date_list)))

  for prod_key in target_products:
    prod_enum = Product[prod_key]
    is_forecast = PRODUCT_TYPES[prod_enum] == ProductType.FORECAST
    lead_days = FORECAST_LEAD_DAYS.get(prod_enum, 10) if is_forecast else 1
    archive_uri = norm_stores[prod_key]
    canon_uri = os.path.join(
        str(canonical_dir).rstrip("/"), prod_key, "timeseries.zarr"
    )
    ds_canon = open_gridded_archive(canon_uri)
    prod_canon_basin_map = {
        str(b).rstrip("\x00"): b for b in ds_canon["basin"].values
    }

    t0_w = time.time()
    logger.info("Building ZonalWeightMatrix for %s from %s...", prod_key, archive_uri)
    weights_matrix = build_product_weight_matrix(
        prod_key,
        archive_uri,
        matched_gdf,
        num_workers=num_workers,
    )
    weights_sec = time.time() - t0_w

    t0_ext = time.time()
    window_extracted_list: list[tuple[pd.Timestamp, pd.Timestamp, str, xr.Dataset]] = []
    for start_dt, end_dt, win_label in parsed_windows:
      logger.info(
          "Extracting %s window %s (%d basins)...",
          prod_key,
          win_label,
          len(matched_gdf),
      )
      ds_win = extract_window_with_chunking(
          prod_key,
          archive_uri,
          matched_gdf,
          start_dt,
          end_dt,
          weights_matrix=weights_matrix,
      )
      window_extracted_list.append((start_dt, end_dt, win_label, ds_win))
    extract_sec = time.time() - t0_ext

    t0_eval = time.time()
    ds_ext_full = (
        window_extracted_list[0][3]
        if len(window_extracted_list) == 1
        else xr.concat([w[3] for w in window_extracted_list], dim="date")
    )
    # Deduplicate dates if overlapping windows were supplied
    ext_dates_all = pd.DatetimeIndex(
        pd.to_datetime(ds_ext_full["date"].values).floor("D")
    )
    ds_ext_full = ds_ext_full.assign_coords(date=ext_dates_all)
    if ext_dates_all.duplicated().any():
      keep_idx = np.where(~ext_dates_all.duplicated(keep="first"))[0]
      ds_ext_full = ds_ext_full.isel(date=keep_idx)
    ds_ext_full = ds_ext_full.sortby("date")
    ds_ext_full = ds_ext_full.reindex(
        basin=matched_ids, date=expected_all_dates.values, fill_value=np.nan
    )

    if writer is not None:
      writer.write_or_append(
          ds_ext_full, prod_enum, overwrite_existing_basins=True
      )

    full_target_dates = expected_all_dates
    ds_can_full = _align_canonical_slice(
        ds_canon,
        prod_canon_basin_map,
        matched_ids,
        full_target_dates,
        is_forecast=is_forecast,
        lead_days=lead_days,
    )

    eval_vars = [
        b
        for b in PRODUCT_BANDS[prod_enum]
        if b in ds_ext_full.data_vars and b in ds_can_full.data_vars
    ]
    if not eval_vars:
      raise KeyError(
          f"No overlapping data variables found between extracted {prod_key} "
          f"({list(ds_ext_full.data_vars)}) and canonical store {canon_uri} "
          f"({list(ds_can_full.data_vars)})."
      )

    size_tiers_unique = sorted(matched_gdf["size_tier"].unique())
    datasets_unique = sorted(matched_gdf["dataset"].unique())

    for var_name in eval_vars:
      if is_forecast:
        ext_da = ds_ext_full[var_name].transpose("basin", "date", "lead_time")
        can_da = ds_can_full[var_name].transpose("basin", "date", "lead_time")
      else:
        ext_da = ds_ext_full[var_name].transpose("basin", "date")
        can_da = ds_can_full[var_name].transpose("basin", "date")

      with dask.config.set(scheduler="threads", num_workers=16):
        ext_vals = np.asarray(ext_da.compute().values, dtype=np.float32)
        can_vals = np.asarray(can_da.compute().values, dtype=np.float32)

      # Per-basin metrics across all requested dates
      pb_df = compute_per_basin_dataframe(
          product=prod_key,
          variable=var_name,
          basins_gdf=matched_gdf,
          ext_arr=ext_vals,
          can_arr=can_vals,
      )
      per_basin_dfs.append(pb_df)

      for subset_name, b_mask in basin_subsets:
        sub_gdf = matched_gdf.iloc[np.where(b_mask)[0]]
        ext_sub = ext_vals[b_mask]
        can_sub = can_vals[b_mask]

        overall_m = compute_array_metrics(ext_sub, can_sub)
        summary_rows.append({
            "product": prod_key,
            "variable": var_name,
            "basin_subset": subset_name,
            "n_windows": len(parsed_windows),
            "n_days": len(full_target_dates),
            **overall_m,
        })

        # Breakdown by size_tier
        for st in size_tiers_unique:
          st_mask = b_mask & (matched_gdf["size_tier"].to_numpy() == st)
          if not np.any(st_mask):
            continue
          st_m = compute_array_metrics(ext_vals[st_mask], can_vals[st_mask])
          size_tier_rows.append({
              "product": prod_key,
              "variable": var_name,
              "basin_subset": subset_name,
              "size_tier": st,
              **st_m,
          })

        # Breakdown by source dataset
        for dname in datasets_unique:
          ds_mask = b_mask & (matched_gdf["dataset"].to_numpy() == dname)
          if not np.any(ds_mask):
            continue
          ds_m = compute_array_metrics(ext_vals[ds_mask], can_vals[ds_mask])
          dataset_rows.append({
              "product": prod_key,
              "variable": var_name,
              "basin_subset": subset_name,
              "dataset": dname,
              **ds_m,
          })

        # Breakdown by date window and (for HRES) upstream tier
        date_index_map = {d: idx for idx, d in enumerate(full_target_dates)}
        tier_to_date_indices: dict[str, list[int]] = {}
        for start_dt, end_dt, win_label in parsed_windows:
          win_dates = pd.date_range(start_dt, end_dt, freq="1D")
          d_idxs = [date_index_map[d] for d in win_dates if d in date_index_map]
          if not d_idxs:
            continue
          tier_label = (
              classify_hres_upstream_tier(start_dt, end_dt)
              if prod_key == "HRES"
              else "Single Source"
          )
          tier_to_date_indices.setdefault(tier_label, []).extend(d_idxs)

          ext_win = ext_sub[:, d_idxs] if not is_forecast else ext_sub[:, d_idxs, :]
          can_win = can_sub[:, d_idxs] if not is_forecast else can_sub[:, d_idxs, :]
          win_m = compute_array_metrics(ext_win, can_win)
          window_rows.append({
              "product": prod_key,
              "variable": var_name,
              "basin_subset": subset_name,
              "window": win_label,
              "upstream_tier": tier_label,
              "n_days": len(d_idxs),
              **win_m,
          })

        # If HRES has multiple windows per tier, also add pooled tier summary rows
        if prod_key == "HRES" and len(parsed_windows) > 1:
          for tier_label, t_idxs in tier_to_date_indices.items():
            uniq_t_idxs = sorted(set(t_idxs))
            ext_tier = ext_sub[:, uniq_t_idxs, :]
            can_tier = can_sub[:, uniq_t_idxs, :]
            tier_m = compute_array_metrics(ext_tier, can_tier)
            window_rows.append({
                "product": prod_key,
                "variable": var_name,
                "basin_subset": subset_name,
                "window": f"ALL ({tier_label.split(':', 1)[0]})",
                "upstream_tier": tier_label,
                "n_days": len(uniq_t_idxs),
                **tier_m,
            })

        # For HRES: per lead_time (1..10) overall and per upstream_tier
        if is_forecast:
          for l_idx in range(lead_days):
            lead_day = l_idx + 1
            lead_m = compute_array_metrics(
                ext_sub[:, :, l_idx], can_sub[:, :, l_idx]
            )
            hres_lead_rows.append({
                "product": prod_key,
                "variable": var_name,
                "basin_subset": subset_name,
                "upstream_tier": "ALL",
                "lead_time": lead_day,
                "n_days": len(full_target_dates),
                **lead_m,
            })
          if prod_key == "HRES":
            for tier_label, t_idxs in tier_to_date_indices.items():
              uniq_t_idxs = sorted(set(t_idxs))
              for l_idx in range(lead_days):
                lead_day = l_idx + 1
                lead_tier_m = compute_array_metrics(
                    ext_sub[:, uniq_t_idxs, l_idx],
                    can_sub[:, uniq_t_idxs, l_idx],
                )
                hres_lead_rows.append({
                    "product": prod_key,
                    "variable": var_name,
                    "basin_subset": subset_name,
                    "upstream_tier": tier_label,
                    "lead_time": lead_day,
                    "n_days": len(uniq_t_idxs),
                    **lead_tier_m,
                })

      del sub_gdf

    eval_sec = time.time() - t0_eval
    timings[prod_key] = {
        "weights_sec": weights_sec,
        "extract_sec": extract_sec,
        "eval_sec": eval_sec,
    }

  summary_df = pd.DataFrame(summary_rows)
  size_tier_df = pd.DataFrame(size_tier_rows)
  dataset_df = pd.DataFrame(dataset_rows)
  window_df = pd.DataFrame(window_rows)
  hres_lead_df = pd.DataFrame(hres_lead_rows)
  per_basin_df = (
      pd.concat(per_basin_dfs, ignore_index=True)
      if per_basin_dfs
      else pd.DataFrame()
  )

  summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
  size_tier_df.to_csv(
      os.path.join(output_dir, "metrics_by_size_tier.csv"), index=False
  )
  dataset_df.to_csv(
      os.path.join(output_dir, "metrics_by_dataset.csv"), index=False
  )
  window_df.to_csv(
      os.path.join(output_dir, "metrics_by_window.csv"), index=False
  )
  hres_lead_df.to_csv(
      os.path.join(output_dir, "hres_metrics_by_lead.csv"), index=False
  )
  per_basin_df.to_parquet(
      os.path.join(output_dir, "per_basin_metrics.parquet"), index=False
  )
  per_basin_df.to_csv(
      os.path.join(output_dir, "per_basin_metrics.csv"), index=False
  )

  report_path = _write_benchmark_markdown_report(
      output_dir,
      dataset_path=dataset_path,
      canonical_dir=canonical_dir,
      archive_stores={k: norm_stores[k] for k in target_products},
      total_input_basins=total_input_basins,
      matched_gdf=matched_gdf,
      summary_df=summary_df,
      size_tier_df=size_tier_df,
      dataset_df=dataset_df,
      window_df=window_df,
      hres_lead_df=hres_lead_df,
      timings=timings,
  )
  logger.info(
      "Saved MultiMet timeseries benchmark outputs to %s (report: %s)",
      output_dir,
      report_path,
  )
  return {
      "summary": summary_df,
      "by_size_tier": size_tier_df,
      "by_dataset": dataset_df,
      "by_window": window_df,
      "hres_by_lead": hres_lead_df,
      "per_basin": per_basin_df,
  }


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds CLI argument parser for MultiMet timeseries reconstruction benchmark."""
  parser = argparse.ArgumentParser(
      prog="benchmark-multimet-timeseries",
      description=(
          "Reconstruct catchment-averaged MultiMet timeseries (CPC, IMERG, HRES) "
          "from gridded Zarr archives and compare against canonical MultiMet stores."
      ),
  )
  parser.add_argument(
      "--dataset",
      type=str,
      required=True,
      help="Path to benchmark basin dataset (.parquet, .geojson, .shp).",
  )
  parser.add_argument(
      "--canonical-dir",
      "--canonical_dir",
      dest="canonical_dir",
      type=str,
      required=True,
      help="Root directory or gs:// URI containing canonical <PRODUCT>/timeseries.zarr.",
  )
  parser.add_argument(
      "--archive-store",
      "--archive_store",
      dest="archive_stores",
      action="append",
      required=True,
      help="Repeatable PRODUCT=URI mapping for gridded Zarr archives (e.g. CPC=gs://...).",
  )
  parser.add_argument(
      "--date-windows",
      "--date_windows",
      dest="date_windows",
      nargs="+",
      required=True,
      help="One or more START:END date windows (YYYY-MM-DD:YYYY-MM-DD).",
  )
  parser.add_argument(
      "--output-dir",
      "--output_dir",
      dest="output_dir",
      type=str,
      required=True,
      help="Directory where CSVs, per_basin_metrics.parquet, and benchmark_report.md are written.",
  )
  parser.add_argument(
      "--products",
      nargs="+",
      default=None,
      help="Optional subset of products to benchmark (CPC, IMERG, HRES).",
  )
  parser.add_argument(
      "--save-reconstructed-zarr",
      "--save_reconstructed_zarr",
      dest="save_reconstructed_zarr",
      type=str,
      default=None,
      help="Optional directory to write reconstructed <PRODUCT>/timeseries.zarr stores.",
  )
  parser.add_argument(
      "--id-column",
      "--id_column",
      dest="id_column",
      type=str,
      default=None,
      help="Optional column name for basin IDs in --dataset.",
  )
  parser.add_argument(
      "--area-mismatch-threshold",
      "--area_mismatch_threshold",
      dest="area_mismatch_threshold",
      type=float,
      default=AREA_MISMATCH_REL_THRESHOLD,
      help="Relative area difference threshold to flag revised geometries (default: 0.50).",
  )
  parser.add_argument(
      "--num-workers",
      "--num_workers",
      dest="num_workers",
      type=int,
      default=8,
      help="Number of threads for ZonalWeightMatrix construction.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for `multimet/timeseries_extractors/benchmark.py`."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  archive_stores = _parse_product_uri_pairs(args.archive_stores)
  run_timeseries_benchmark(
      dataset_path=args.dataset,
      canonical_dir=args.canonical_dir,
      archive_stores=archive_stores,
      date_windows=args.date_windows,
      output_dir=args.output_dir,
      products=args.products,
      save_reconstructed_zarr=args.save_reconstructed_zarr,
      id_column=args.id_column,
      area_mismatch_threshold=args.area_mismatch_threshold,
      num_workers=args.num_workers,
  )


if __name__ == "__main__":
  main()
