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

"""Regenerates HydroATLAS Level 12 and Caravan ERA5-Land static climate tables.

Provides vectorized 2D (date x basin) computation of all 18 Caravan climate
signatures (1981-2020 baseline, uniformly rounded to 4 decimal places) for:
1. All 9 continental HydroATLAS Level 12 tables
   (``{continent}_climate_indices.txt`` JSONL).
2. All Caravan subdataset ``attributes_caravan_<subdataset>.csv`` and
   ``attributes_<subdataset>.parquet`` tables from pre-extracted
   ``ERA5_LAND/timeseries.zarr`` stores.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import fsspec
import numpy as np
import pandas as pd

from multimet.static_extractor.config import (
    CARAVAN_CLIMATE_COLUMNS as CARAVAN_CLIMATE_KEYS,
)

logger = logging.getLogger(__name__)

HYDROATLAS_CONTINENTS: Tuple[str, ...] = (
    "af",
    "ar",
    "as",
    "au",
    "eu",
    "gr",
    "na",
    "sa",
    "si",
)


def compute_climate_metrics_2d(
    dates: pd.DatetimeIndex,
    prcp: np.ndarray,
    t2m_c: np.ndarray,
    pev_native: np.ndarray,
    pet_fao: np.ndarray,
    baseline_years: Tuple[int, int] = (1981, 2020),
    round_decimals: int = 4,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
  """Computes all 18 Caravan climate metrics vectorized across N basins/polygons.

  Matches ``multimet.utils.climate.compute_caravan_climate_metrics`` across an
  entire ``(T, N)`` matrix simultaneously, rounding all 18 output metrics to
  ``round_decimals`` (default 4) decimal places.

  Args:
    dates: Daily ``DatetimeIndex`` of length ``T``.
    prcp: Daily precipitation in mm/day, shape ``(T, N)``.
    t2m_c: Daily 2m air temperature in °C, shape ``(T, N)``.
    pev_native: Daily native ERA5-Land potential evaporation in mm/day (already
      sign-aligned so positive values represent evaporation), shape ``(T, N)``.
    pet_fao: Daily FAO Penman-Monteith reference ET in mm/day, shape ``(T, N)``.
    baseline_years: Inclusive ``(start_year, end_year)`` tuple.
    round_decimals: Number of decimal places to round each metric to.

  Returns:
    Tuple of ``(metrics_2d_dict, valid_basin_mask)`` where ``metrics_2d_dict``
    maps each of the 18 keys in ``CARAVAN_CLIMATE_KEYS`` to a 1D ``float64``
    array of length ``N``.
  """
  mask_yr = (dates.year >= baseline_years[0]) & (dates.year <= baseline_years[1])
  if not np.any(mask_yr):
    n_basins = prcp.shape[1]
    nan_arr = np.full(n_basins, np.nan, dtype=np.float64)
    return {k: nan_arr.copy() for k in CARAVAN_CLIMATE_KEYS}, np.zeros(
        n_basins, dtype=bool
    )

  dates_b = dates[mask_yr]
  prcp_b = np.asarray(prcp[mask_yr], dtype=np.float64)
  t2m_b = np.asarray(t2m_c[mask_yr], dtype=np.float64)
  pev_b = np.asarray(pev_native[mask_yr], dtype=np.float64)
  fao_b = np.asarray(pet_fao[mask_yr], dtype=np.float64)

  valid_basin = ~(
      np.all(np.isnan(prcp_b), axis=0)
      | np.all(np.isnan(t2m_b), axis=0)
      | np.all(np.isnan(pev_b), axis=0)
  )

  p_mean = np.nanmean(prcp_b, axis=0)
  pet_mean_era5 = np.nanmean(pev_b, axis=0)
  pet_mean_fao = np.nanmean(fao_b, axis=0)

  aridity_era5 = np.where(p_mean > 0, pet_mean_era5 / p_mean, np.nan)
  aridity_fao = np.where(p_mean > 0, pet_mean_fao / p_mean, np.nan)

  n_basins = prcp_b.shape[1]
  months = dates_b.month.values
  m_prcp = np.zeros((12, n_basins), dtype=np.float64)
  m_t2m = np.zeros((12, n_basins), dtype=np.float64)
  m_pev = np.zeros((12, n_basins), dtype=np.float64)
  m_fao = np.zeros((12, n_basins), dtype=np.float64)
  for m in range(1, 13):
    idx_m = months == m
    m_prcp[m - 1] = np.nanmean(prcp_b[idx_m], axis=0)
    m_t2m[m - 1] = np.nanmean(t2m_b[idx_m], axis=0)
    m_pev[m - 1] = np.nanmean(pev_b[idx_m], axis=0)
    m_fao[m - 1] = np.nanmean(fao_b[idx_m], axis=0)

  tot_monthly_p = np.sum(m_prcp, axis=0)
  snow_monthly_p = np.sum(np.where(m_t2m < 0.0, m_prcp, 0.0), axis=0)
  frac_snow = np.where(
      valid_basin & (tot_monthly_p > 0),
      snow_monthly_p / tot_monthly_p,
      np.nan,
  )

  high_thresh = 5.0 * p_mean[None, :]
  high_cond = prcp_b >= high_thresh
  high_prec_freq = np.where(
      valid_basin & (p_mean > 0), np.mean(high_cond, axis=0), np.nan
  )

  low_cond = prcp_b < 1.0
  low_prec_freq = np.where(
      valid_basin & (p_mean > 0), np.mean(low_cond, axis=0), np.nan
  )

  high_prec_dur = np.full(n_basins, np.nan, dtype=np.float64)
  low_prec_dur = np.full(n_basins, np.nan, dtype=np.float64)

  for cond_mat, out_dur in (
      (high_cond, high_prec_dur),
      (low_cond, low_prec_dur),
  ):
    pad = np.vstack([
        np.zeros((1, n_basins), dtype=bool),
        cond_mat,
        np.zeros((1, n_basins), dtype=bool),
    ])
    diff = pad[1:].astype(np.int8) - pad[:-1].astype(np.int8)
    n_runs = np.sum(diff == 1, axis=0)
    total_true = np.sum(cond_mat, axis=0)
    out_dur[:] = np.where(
        valid_basin & (p_mean > 0),
        np.where(n_runs > 0, total_true / np.maximum(n_runs, 1), 0.0),
        np.nan,
    )

  def _mi_and_seas(
      m_p: np.ndarray, m_e: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    p_gt_et = m_p > m_e
    p_lt_et = m_p < m_e
    mi_m = np.zeros((12, n_basins), dtype=np.float64)
    mi_m = np.where(
        p_gt_et,
        1.0 - np.divide(m_e, m_p, out=np.zeros_like(m_e), where=p_gt_et),
        mi_m,
    )
    mi_m = np.where(
        p_lt_et,
        np.divide(m_p, m_e, out=np.zeros_like(m_p), where=p_lt_et) - 1.0,
        mi_m,
    )
    mi = np.where(valid_basin, np.mean(mi_m, axis=0), np.nan)
    seas = np.where(
        valid_basin,
        np.max(mi_m, axis=0) - np.min(mi_m, axis=0),
        np.nan,
    )
    return mi, seas

  mi_era5, seas_era5 = _mi_and_seas(m_prcp, m_pev)
  mi_fao, seas_fao = _mi_and_seas(m_prcp, m_fao)

  raw_metrics = {
      "p_mean": p_mean,
      "pet_mean": pet_mean_fao,
      "pet_mean_FAO_PM": pet_mean_fao,
      "pet_mean_ERA5_LAND": pet_mean_era5,
      "aridity": aridity_fao,
      "aridity_FAO_PM": aridity_fao,
      "aridity_ERA5_LAND": aridity_era5,
      "frac_snow": frac_snow,
      "moisture_index": mi_fao,
      "moisture_index_FAO_PM": mi_fao,
      "moisture_index_ERA5_LAND": mi_era5,
      "seasonality": seas_fao,
      "seasonality_FAO_PM": seas_fao,
      "seasonality_ERA5_LAND": seas_era5,
      "high_prec_freq": high_prec_freq,
      "high_prec_dur": high_prec_dur,
      "low_prec_freq": low_prec_freq,
      "low_prec_dur": low_prec_dur,
  }
  rounded_metrics = {
      k: np.where(np.isnan(arr), np.nan, np.round(arr, round_decimals))
      for k, arr in raw_metrics.items()
  }
  return rounded_metrics, valid_basin


def round_jsonl_climate_records(
    lines: Sequence[str],
    round_decimals: int = 4,
) -> List[str]:
  """Normalizes JSONL climate table records so all float metrics are rounded to ``round_decimals``."""
  out_lines: List[str] = []
  for raw_line in lines:
    stripped = raw_line.strip()
    if not stripped:
      continue
    rec = json.loads(stripped)
    norm_rec: Dict[str, Any] = {}
    if "gauge_id" in rec:
      norm_rec["gauge_id"] = str(rec["gauge_id"])
    if "HYBAS_ID" in rec:
      norm_rec["HYBAS_ID"] = int(rec["HYBAS_ID"])
    if not norm_rec:
      raise KeyError("JSONL record must contain 'gauge_id' or 'HYBAS_ID'.")
    for k in CARAVAN_CLIMATE_KEYS:
      val = rec.get(k)
      if val is None or (isinstance(val, float) and np.isnan(val)):
        norm_rec[k] = None
      else:
        norm_rec[k] = round(float(val), round_decimals)
    out_lines.append(json.dumps(norm_rec))
  return out_lines


def update_subdataset_climate_tables(
    caravan_csv_df: pd.DataFrame,
    parquet_df: pd.DataFrame,
    climate_by_basin: Mapping[str, Mapping[str, float]],
    round_decimals: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Updates ``attributes_caravan_<subdataset>.csv`` and ``attributes_<subdataset>.parquet`` DataFrames."""
  updated_csv = caravan_csv_df.copy()
  updated_pq = parquet_df.copy()

  if "gauge_id" in updated_csv.columns:
    csv_ids = [str(g) for g in updated_csv["gauge_id"].values]
  else:
    csv_ids = [str(g) for g in updated_csv.index.values]

  if "gauge_id" in updated_pq.columns:
    pq_ids = [str(g) for g in updated_pq["gauge_id"].values]
  else:
    pq_ids = [str(g) for g in updated_pq.index.values]

  for k in CARAVAN_CLIMATE_KEYS:
    updated_csv[k] = [
        round(float(climate_by_basin[gid][k]), round_decimals)
        if gid in climate_by_basin
        and not np.isnan(climate_by_basin[gid].get(k, np.nan))
        else np.nan
        for gid in csv_ids
    ]
    updated_pq[k] = [
        round(float(climate_by_basin[gid][k]), round_decimals)
        if gid in climate_by_basin
        and not np.isnan(climate_by_basin[gid].get(k, np.nan))
        else np.nan
        for gid in pq_ids
    ]

  return updated_csv, updated_pq


def normalize_gcs_tables_rounding(
    era5_climate_uri: str = "gs://open-multimet/ancillary-data/hydroatlas/era5_climate",
    caravan_root_uri: str = "gs://open-multimet/caravan-new",
    round_decimals: int = 4,
    skip_continents: bool = False,
) -> None:
  """Ensures all 9 continental JSONL tables and 22 Caravan subdataset tables use uniform rounding."""
  fs, _ = fsspec.core.url_to_fs(era5_climate_uri)

  if not skip_continents:
    for cont in HYDROATLAS_CONTINENTS:
      uri = f"{era5_climate_uri.rstrip('/')}/{cont}_climate_indices.txt"
      _, fs_path = fsspec.core.url_to_fs(uri)
      if not fs.exists(fs_path):
        continue
      with fs.open(fs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
      norm_lines = round_jsonl_climate_records(
          lines, round_decimals=round_decimals
      )
      with fs.open(fs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(norm_lines) + "\n")
      logger.info(
          "Normalized %d records in %s to %d decimals.",
          len(norm_lines),
          uri,
          round_decimals,
      )

  collections = ("google-internal", "caravan-extensions", "caravan-original")
  for coll in collections:
    attr_root = f"{caravan_root_uri.rstrip('/')}/{coll}/attributes"
    _, attr_root_path = fsspec.core.url_to_fs(attr_root)
    if not fs.exists(attr_root_path):
      continue
    for sub_dir in sorted(fs.ls(attr_root_path)):
      sub = os.path.basename(sub_dir.rstrip("/"))
      csv_fs_path = f"{sub_dir.rstrip('/')}/attributes_caravan_{sub}.csv"
      pq_fs_path = f"{sub_dir.rstrip('/')}/attributes_{sub}.parquet"
      if not fs.exists(csv_fs_path) or not fs.exists(pq_fs_path):
        continue
      csv_uri = f"gs://{csv_fs_path}"
      pq_uri = f"gs://{pq_fs_path}"
      df_csv = pd.read_csv(csv_uri)
      df_pq = pd.read_parquet(pq_uri)
      for k in CARAVAN_CLIMATE_KEYS:
        if k in df_csv.columns:
          df_csv[k] = df_csv[k].round(round_decimals)
        if k in df_pq.columns:
          df_pq[k] = df_pq[k].round(round_decimals)
      df_csv.to_csv(csv_uri, index=False)
      df_pq.to_parquet(pq_uri)
      logger.info(
          "Normalized %s/%s climate columns to %d decimals.",
          coll,
          sub,
          round_decimals,
      )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Normalize or regenerate ERA5-Land static climate tables."
  )
  parser.add_argument(
      "--era5-climate-uri",
      default="gs://open-multimet/ancillary-data/hydroatlas/era5_climate",
      help="GCS URI for continental HydroATLAS climate tables.",
  )
  parser.add_argument(
      "--caravan-root-uri",
      default="gs://open-multimet/caravan-new",
      help="GCS root URI for Caravan collections.",
  )
  parser.add_argument(
      "--round-decimals",
      type=int,
      default=4,
      help="Number of decimal places for all 18 float climate metrics.",
  )
  parser.add_argument(
      "--skip-continents",
      action="store_true",
      help="Skip normalizing the 9 continental JSONL tables.",
  )
  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
  )
  normalize_gcs_tables_rounding(
      era5_climate_uri=args.era5_climate_uri,
      caravan_root_uri=args.caravan_root_uri,
      round_decimals=args.round_decimals,
      skip_continents=args.skip_continents,
  )


if __name__ == "__main__":
  main()
