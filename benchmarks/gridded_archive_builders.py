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

"""Benchmark harness for Open-MultiMet gridded archive builders (CPC and IMERG).

Rebuilds a user-specified date window from upstream NOAA PSL (for CPC) or NASA
GES DISC / local granules (for IMERG) into a Zarr store and compares the rebuilt
daily grids against an existing reference gridded archive Zarr store.

Computes:
1. Grid dimension and coordinate (`latitude`, `longitude`, `time`) exact parity.
2. Valid/NaN cell mask confusion matrix (`both_valid_pct`, `both_nan_pct`,
   `rebuilt_only_nan_pct`, `ref_only_nan_pct`).
3. Numerical accuracy across `both_valid` grid cells (`mae`, `rmse`, `bias`,
   `p50_abs_err`, `p95_abs_err`, `p99_abs_err`, `max_abs_err`, `pearson_r`,
   `frac_within_1e_5`, `frac_exact_match`).
4. Saves `summary_metrics.csv`, `daily_metrics.csv`, and `benchmark_report.md`
   to `--output-dir`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import os
import time
from typing import Any

from multimet.gridded_archive_builders.build_cpc_archive import (
    CPC_VARIABLE,
    NOAA_PSL_URL_TEMPLATE,
    build_cpc_archive,
)
from multimet.gridded_archive_builders.build_imerg_archive import (
    DEFAULT_GESDISC_URL,
    IMERG_DAILY_SHORT_NAME,
    IMERG_VARIABLE,
    build_imerg_archive,
)
from multimet.timeseries_extractors.gridded_archive import open_gridded_archive
from multimet.utils.http import (
    EarthdataSession,
    get_earthdata_credentials_from_netrc,
    query_cmr_granules,
)
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_ARCHIVE_PRODUCTS: tuple[str, ...] = ("CPC", "IMERG")

PRODUCT_VARIABLES: dict[str, tuple[str, ...]] = {
    "CPC": (CPC_VARIABLE,),
    "IMERG": (IMERG_VARIABLE,),
}


def verify_gesdisc_access(
    sample_date: pd.Timestamp,
    *,
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
    netrc_path: str | None = None,
    timeout: int = 30,
) -> tuple[bool, int, str]:
  """Checks whether NASA GES DISC is accessible for a given sample date.

  Does not use try/except; inspects credentials and HTTP status codes explicitly.

  Args:
    sample_date: Sample UTC date to query on NASA CMR and GES DISC.
    username: Optional Earthdata username.
    password: Optional Earthdata password.
    token: Optional Earthdata bearer token.
    netrc_path: Optional path to `.netrc` file.
    timeout: Request timeout in seconds.

  Returns:
    Tuple of `(is_accessible, http_status_code, message)`.
  """
  eff_token = token or os.environ.get("EARTHDATA_TOKEN")
  eff_user = username or os.environ.get("EARTHDATA_USERNAME")
  eff_pass = password or os.environ.get("EARTHDATA_PASSWORD")
  if not (eff_user and eff_pass) and not eff_token:
    netrc_user, netrc_pass = get_earthdata_credentials_from_netrc(netrc_path)
    if netrc_user and netrc_pass:
      eff_user, eff_pass = netrc_user, netrc_pass

  if not (eff_user and eff_pass) and not eff_token:
    return (
        False,
        0,
        "No NASA Earthdata credentials found in environment or ~/.netrc.",
    )

  urls = query_cmr_granules(
      IMERG_DAILY_SHORT_NAME, pd.Timestamp(sample_date), timeout=timeout
  )
  if not urls:
    return (
        False,
        404,
        f"No NASA CMR granules found for {IMERG_DAILY_SHORT_NAME} on "
        f"{pd.Timestamp(sample_date).strftime('%Y-%m-%d')}.",
    )

  session = EarthdataSession(
      username=eff_user,
      password=eff_pass,
      token=eff_token,
      netrc_path=netrc_path,
  )
  with session.get(urls[0], stream=True, timeout=timeout) as resp:
    status = int(resp.status_code)
    if 200 <= status < 300:
      return (True, status, f"Authenticated HTTP {status} OK for {urls[0]}")
    return (
        False,
        status,
        f"NASA GES DISC returned HTTP {status} for {urls[0]}",
    )


def _compute_grid_comparison_metrics(
    rebuilt_arr: np.ndarray,
    ref_arr: np.ndarray,
) -> dict[str, Any]:
  """Computes mask confusion matrix and numerical accuracy on 2D or 3D grids."""
  if rebuilt_arr.shape != ref_arr.shape:
    raise ValueError(
        f"Grid shape mismatch between rebuilt {rebuilt_arr.shape} and "
        f"reference {ref_arr.shape}."
    )

  total_cells = int(rebuilt_arr.size)
  if total_cells == 0:
    raise ValueError("Cannot compute grid comparison metrics on empty arrays.")

  reb_valid = np.isfinite(rebuilt_arr)
  ref_valid = np.isfinite(ref_arr)

  both_valid = reb_valid & ref_valid
  both_nan = (~reb_valid) & (~ref_valid)
  rebuilt_only_nan = (~reb_valid) & ref_valid
  ref_only_nan = reb_valid & (~ref_valid)

  both_valid_count = int(np.sum(both_valid))
  both_nan_count = int(np.sum(both_nan))
  rebuilt_only_nan_count = int(np.sum(rebuilt_only_nan))
  ref_only_nan_count = int(np.sum(ref_only_nan))

  inv_total = 100.0 / float(total_cells)
  metrics: dict[str, Any] = {
      "total_grid_cells": total_cells,
      "both_valid_count": both_valid_count,
      "both_nan_count": both_nan_count,
      "rebuilt_only_nan_count": rebuilt_only_nan_count,
      "ref_only_nan_count": ref_only_nan_count,
      "both_valid_pct": float(both_valid_count * inv_total),
      "both_nan_pct": float(both_nan_count * inv_total),
      "rebuilt_only_nan_pct": float(rebuilt_only_nan_count * inv_total),
      "ref_only_nan_pct": float(ref_only_nan_count * inv_total),
      "has_rebuilt_only_nan_discrepancy": bool(rebuilt_only_nan_count > 0),
  }

  if both_valid_count == 0:
    metrics.update({
        "mae": float("nan"),
        "rmse": float("nan"),
        "bias": float("nan"),
        "p50_abs_err": float("nan"),
        "p75_abs_err": float("nan"),
        "p90_abs_err": float("nan"),
        "p95_abs_err": float("nan"),
        "p99_abs_err": float("nan"),
        "max_abs_err": float("nan"),
        "pearson_r": float("nan"),
        "frac_within_1e_5": float("nan"),
        "frac_exact_match": float("nan"),
        "rebuilt_mean": float("nan"),
        "ref_mean": float("nan"),
        "rebuilt_std": float("nan"),
        "ref_std": float("nan"),
    })
    return metrics

  reb_v = rebuilt_arr[both_valid].astype(np.float64)
  ref_v = ref_arr[both_valid].astype(np.float64)
  diff = reb_v - ref_v
  abs_err = np.abs(diff)

  mae = float(np.mean(abs_err))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  bias = float(np.mean(diff))
  p50, p75, p90, p95, p99 = np.percentile(
      abs_err, [50.0, 75.0, 90.0, 95.0, 99.0]
  )
  max_abs = float(np.max(abs_err))
  frac_1e5 = float(np.mean(abs_err <= 1e-5))
  frac_exact = float(np.mean(abs_err == 0.0))

  reb_mean = float(np.mean(reb_v))
  ref_mean = float(np.mean(ref_v))
  reb_std = float(np.std(reb_v))
  ref_std = float(np.std(ref_v))

  if both_valid_count >= 2 and reb_std > 1e-12 and ref_std > 1e-12:
    reb_centered = reb_v - reb_mean
    ref_centered = ref_v - ref_mean
    denom = float(
        np.sqrt(np.sum(reb_centered**2) * np.sum(ref_centered**2))
    )
    pearson_r = (
        float(np.sum(reb_centered * ref_centered) / denom)
        if denom > 1e-12
        else float("nan")
    )
  else:
    pearson_r = float("nan")

  metrics.update({
      "mae": mae,
      "rmse": rmse,
      "bias": bias,
      "p50_abs_err": float(p50),
      "p75_abs_err": float(p75),
      "p90_abs_err": float(p90),
      "p95_abs_err": float(p95),
      "p99_abs_err": float(p99),
      "max_abs_err": max_abs,
      "pearson_r": pearson_r,
      "frac_within_1e_5": frac_1e5,
      "frac_exact_match": frac_exact,
      "rebuilt_mean": reb_mean,
      "ref_mean": ref_mean,
      "rebuilt_std": reb_std,
      "ref_std": ref_std,
  })
  return metrics


def compare_gridded_archives(
    *,
    product: str,
    rebuilt_zarr: str,
    reference_zarr: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compares a rebuilt gridded Zarr store against a reference Zarr store.

  Args:
    product: Product identifier (`"CPC"` or `"IMERG"`).
    rebuilt_zarr: Path or URI to the rebuilt Zarr store.
    reference_zarr: Path or URI to the reference Zarr store.
    start_date: Inclusive start date (`YYYY-MM-DD`).
    end_date: Inclusive end date (`YYYY-MM-DD`).

  Returns:
    Tuple `(summary_df, daily_df)` containing overall per-variable metrics and
    day-by-day metrics.
  """
  prod_key = product.strip().upper()
  if prod_key not in SUPPORTED_ARCHIVE_PRODUCTS:
    raise ValueError(
        f"Unsupported product {product!r}. Expected one of "
        f"{SUPPORTED_ARCHIVE_PRODUCTS}."
    )

  start_ts = pd.Timestamp(start_date).normalize()
  end_ts = pd.Timestamp(end_date).normalize()
  if end_ts < start_ts:
    raise ValueError(
        f"end_date ({end_date}) must be >= start_date ({start_date})."
    )
  expected_dates = pd.date_range(start_ts, end_ts, freq="1D")

  ds_reb = open_gridded_archive(rebuilt_zarr)
  ds_ref = open_gridded_archive(reference_zarr)

  for label, ds in (("rebuilt", ds_reb), ("reference", ds_ref)):
    for req_coord in ("time", "latitude", "longitude"):
      if req_coord not in ds.coords and req_coord not in ds.dims:
        raise KeyError(
            f"Required coordinate {req_coord!r} missing from {label} archive."
        )

  time_slice = slice(
      start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")
  )
  ds_reb_win = ds_reb.sel(time=time_slice)
  ds_ref_win = ds_ref.sel(time=time_slice)

  reb_times = pd.DatetimeIndex(
      pd.to_datetime(ds_reb_win["time"].values).floor("D")
  )
  ref_times = pd.DatetimeIndex(
      pd.to_datetime(ds_ref_win["time"].values).floor("D")
  )

  if len(ref_times) != len(expected_dates) or not (ref_times == expected_dates).all():
    raise ValueError(
        f"Reference archive time coordinate does not match expected window "
        f"[{start_date}, {end_date}]: reference has {len(ref_times)} dates, "
        f"expected {len(expected_dates)}."
    )

  time_exact_match = bool(
      len(reb_times) == len(expected_dates)
      and (reb_times == expected_dates).all()
  )
  ds_ref_win = ds_ref_win.assign_coords(time=ref_times)
  ds_reb_win = ds_reb_win.assign_coords(time=reb_times).reindex(
      time=ref_times, fill_value=np.nan
  )

  reb_lats = np.asarray(ds_reb_win["latitude"].values, dtype=np.float64)
  ref_lats = np.asarray(ds_ref_win["latitude"].values, dtype=np.float64)
  reb_lons = np.asarray(ds_reb_win["longitude"].values, dtype=np.float64)
  ref_lons = np.asarray(ds_ref_win["longitude"].values, dtype=np.float64)

  if reb_lats.shape != ref_lats.shape:
    raise ValueError(
        f"Latitude shape mismatch: rebuilt {reb_lats.shape} vs reference "
        f"{ref_lats.shape}."
    )
  if reb_lons.shape != ref_lons.shape:
    raise ValueError(
        f"Longitude shape mismatch: rebuilt {reb_lons.shape} vs reference "
        f"{ref_lons.shape}."
    )

  max_lat_abs_diff = float(np.max(np.abs(reb_lats - ref_lats)))
  max_lon_abs_diff = float(np.max(np.abs(reb_lons - ref_lons)))
  lat_exact_match = bool(np.allclose(reb_lats, ref_lats, atol=1e-5))
  lon_exact_match = bool(np.allclose(reb_lons, ref_lons, atol=1e-5))

  summary_rows: list[dict[str, Any]] = []
  daily_rows: list[dict[str, Any]] = []

  for var_name in PRODUCT_VARIABLES[prod_key]:
    if var_name not in ds_reb_win.data_vars:
      raise KeyError(
          f"Variable {var_name!r} not found in rebuilt archive {rebuilt_zarr}."
      )
    if var_name not in ds_ref_win.data_vars:
      raise KeyError(
          f"Variable {var_name!r} not found in reference archive "
          f"{reference_zarr}."
      )

    reb_da = ds_reb_win[var_name].transpose("time", "latitude", "longitude")
    ref_da = ds_ref_win[var_name].transpose("time", "latitude", "longitude")

    reb_vals = np.asarray(reb_da.values, dtype=np.float32)
    ref_vals = np.asarray(ref_da.values, dtype=np.float32)

    overall_metrics = _compute_grid_comparison_metrics(reb_vals, ref_vals)
    summary_row: dict[str, Any] = {
        "product": prod_key,
        "variable": var_name,
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "end_date": end_ts.strftime("%Y-%m-%d"),
        "n_days": len(expected_dates),
        "n_lat": len(reb_lats),
        "n_lon": len(reb_lons),
        "time_exact_match": time_exact_match,
        "lat_exact_match": lat_exact_match,
        "lon_exact_match": lon_exact_match,
        "max_lat_abs_diff": max_lat_abs_diff,
        "max_lon_abs_diff": max_lon_abs_diff,
        **overall_metrics,
    }
    summary_rows.append(summary_row)

    for day_idx, dt in enumerate(expected_dates):
      day_metrics = _compute_grid_comparison_metrics(
          reb_vals[day_idx], ref_vals[day_idx]
      )
      daily_rows.append({
          "product": prod_key,
          "variable": var_name,
          "date": dt.strftime("%Y-%m-%d"),
          **day_metrics,
      })

  return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows)


def _write_markdown_report(
    output_dir: str,
    summary_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    rebuilt_zarr: str,
    reference_zarr: str,
    build_elapsed_sec: float,
    compare_elapsed_sec: float,
) -> str:
  """Writes a Markdown report summarizing the gridded archive benchmark."""
  report_path = os.path.join(output_dir, "benchmark_report.md")
  lines: list[str] = [
      "# Gridded Archive Builder Benchmark Report",
      "",
      f"- **Rebuilt Zarr Store**: `{rebuilt_zarr}`",
      f"- **Reference Zarr Store**: `{reference_zarr}`",
      f"- **Archive Build Wall Time**: `{build_elapsed_sec:.2f} s`",
      f"- **Grid Comparison Wall Time**: `{compare_elapsed_sec:.2f} s`",
      "",
      "## 1. Coordinate & Dimension Parity",
      "",
      "| Product | Variable | Date Window | Days | Grid Shape (`lat x lon`) | Time Parity | Lat Parity (`max diff`) | Lon Parity (`max diff`) |",
      "| :--- | :--- | :--- | ---: | :--- | :--- | :--- | :--- |",
  ]
  for _, row in summary_df.iterrows():
    lines.append(
        f"| `{row['product']}` | `{row['variable']}` | "
        f"`{row['start_date']} .. {row['end_date']}` | {int(row['n_days'])} | "
        f"`{int(row['n_lat'])} x {int(row['n_lon'])}` | "
        f"`{bool(row['time_exact_match'])}` | "
        f"`{bool(row['lat_exact_match'])}` (`{row['max_lat_abs_diff']:.2e}`) | "
        f"`{bool(row['lon_exact_match'])}` (`{row['max_lon_abs_diff']:.2e}`) |"
    )

  total_rebuilt_only_nan = int(summary_df["rebuilt_only_nan_count"].sum())
  discrepancy_banner = (
      f"DISCREPANCY DETECTED ({total_rebuilt_only_nan:,} rebuilt-only NaN cells)"
      if total_rebuilt_only_nan > 0
      else "PASS (0 rebuilt-only NaN cells; zero silent NaN masking)"
  )
  lines.extend([
      "",
      "## 2. Valid / NaN Cell Mask Confusion Matrix (Unmasked Full Grid)",
      "",
      f"- **Missing-Data Audit Status**: **{discrepancy_banner}**",
      "- **Protocol**: Zero fallback to reference data, zero imputation/interpolation, all grid cells evaluated.",
      "",
      "| Product | Variable | Total Grid Cells | Both Valid (Count / %) | Both NaN (Count / %) | Rebuilt-Only NaN (Count / %) | Ref-Only NaN (Count / %) | Discrepancy Flag |",
      "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
  ])
  for _, row in summary_df.iterrows():
    reb_nan_cnt = int(row["rebuilt_only_nan_count"])
    flag_str = f"DISCREPANCY ({reb_nan_cnt:,})" if reb_nan_cnt > 0 else "OK (0)"
    lines.append(
        f"| `{row['product']}` | `{row['variable']}` | "
        f"{int(row['total_grid_cells']):,} | "
        f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.4f}%) | "
        f"{int(row['both_nan_count']):,} ({row['both_nan_pct']:.4f}%) | "
        f"{reb_nan_cnt:,} ({row['rebuilt_only_nan_pct']:.6f}%) | "
        f"{int(row['ref_only_nan_count']):,} ({row['ref_only_nan_pct']:.6f}%) | "
        f"`{flag_str}` |"
    )

  lines.extend([
      "",
      "## 3. Numerical Accuracy & Companion Mask Breakdown",
      "",
      "| Product | Variable | Total Cells | Both Valid (Count / %) | Rebuilt-Only NaN (Count / %) | Ref-Only NaN (Count / %) | Pearson $r$ | MAE | RMSE | Bias | P50 Abs Err | P75 Abs Err | P90 Abs Err | P95 Abs Err | P99 Abs Err | Max Abs Err | Within `1e-5` (%) | Exact Match (%) |",
      "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for _, row in summary_df.iterrows():
    lines.append(
        f"| `{row['product']}` | `{row['variable']}` | "
        f"{int(row['total_grid_cells']):,} | "
        f"{int(row['both_valid_count']):,} ({row['both_valid_pct']:.4f}%) | "
        f"{int(row['rebuilt_only_nan_count']):,} ({row['rebuilt_only_nan_pct']:.6f}%) | "
        f"{int(row['ref_only_nan_count']):,} ({row['ref_only_nan_pct']:.6f}%) | "
        f"{row['pearson_r']:.8f} | "
        f"{row['mae']:.6e} | "
        f"{row['rmse']:.6e} | "
        f"{row['bias']:.6e} | "
        f"{row['p50_abs_err']:.6e} | "
        f"{row['p75_abs_err']:.6e} | "
        f"{row['p90_abs_err']:.6e} | "
        f"{row['p95_abs_err']:.6e} | "
        f"{row['p99_abs_err']:.6e} | "
        f"{row['max_abs_err']:.6e} | "
        f"{row['frac_within_1e_5'] * 100.0:.4f}% | "
        f"{row['frac_exact_match'] * 100.0:.4f}% |"
    )

  lines.extend([
      "",
      "## 4. Daily Breakdown (with 4-Way Mask Counts)",
      "",
      "| Date | Variable | Total Cells | Both Valid (Count / %) | Both NaN (Count / %) | Rebuilt-Only NaN (Count / %) | Ref-Only NaN (Count / %) | MAE | RMSE | Max Abs Err | Within `1e-5` (%) |",
      "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for _, drow in daily_df.iterrows():
    lines.append(
        f"| `{drow['date']}` | `{drow['variable']}` | "
        f"{int(drow['total_grid_cells']):,} | "
        f"{int(drow['both_valid_count']):,} ({drow['both_valid_pct']:.4f}%) | "
        f"{int(drow['both_nan_count']):,} ({drow['both_nan_pct']:.4f}%) | "
        f"{int(drow['rebuilt_only_nan_count']):,} ({drow['rebuilt_only_nan_pct']:.6f}%) | "
        f"{int(drow['ref_only_nan_count']):,} ({drow['ref_only_nan_pct']:.6f}%) | "
        f"{drow['mae']:.6e} | "
        f"{drow['rmse']:.6e} | "
        f"{drow['max_abs_err']:.6e} | "
        f"{drow['frac_within_1e_5'] * 100.0:.4f}% |"
    )

  with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return report_path


def run_gridded_archive_benchmark(
    *,
    product: str,
    start_date: str,
    end_date: str,
    reference_zarr: str,
    output_dir: str,
    rebuilt_zarr: str | None = None,
    cache_dir: str | None = None,
    cleanup_cache: bool = False,
    source_url_template: str = NOAA_PSL_URL_TEMPLATE,
    imerg_source: str = "gesdisc",
    imerg_local_dir: str | None = None,
    imerg_local_format: str = "nc4",
    gesdisc_url: str = DEFAULT_GESDISC_URL,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    earthdata_token: str | None = None,
    netrc_path: str | None = None,
    num_workers: int = 4,
    project: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Builds a gridded archive over `[start_date, end_date]` and benchmarks it.

  Args:
    product: `"CPC"` or `"IMERG"`.
    start_date: Inclusive start date (`YYYY-MM-DD`).
    end_date: Inclusive end date (`YYYY-MM-DD`).
    reference_zarr: Explicit path or `gs://` URI to reference Zarr store.
    output_dir: Directory where CSVs, Markdown report, and default rebuilt Zarr
      store are saved.
    rebuilt_zarr: Optional explicit path for the rebuilt Zarr store.
    cache_dir: Optional directory for staging upstream files.
    cleanup_cache: Whether to delete downloaded cache files after building.
    source_url_template: Upstream URL template for CPC (`{year}`).
    imerg_source: `"gesdisc"` or `"local"` for IMERG.
    imerg_local_dir: Optional local directory for IMERG when
      `imerg_source="local"`.
    imerg_local_format: `"nc4"` or `"h5"` when `imerg_source="local"`.
    gesdisc_url: Base URL for NASA GES DISC IMERG daily archive.
    earthdata_username: Optional NASA Earthdata username.
    earthdata_password: Optional NASA Earthdata password.
    earthdata_token: Optional NASA Earthdata bearer token.
    netrc_path: Optional path to `.netrc` file.
    num_workers: Number of worker threads/processes.
    project: Optional GCP project for `gs://` access.

  Returns:
    Tuple of `(summary_df, daily_df)`.
  """
  prod_key = product.strip().upper()
  if prod_key not in SUPPORTED_ARCHIVE_PRODUCTS:
    raise ValueError(
        f"Unsupported product {product!r}. Expected one of "
        f"{SUPPORTED_ARCHIVE_PRODUCTS}."
    )
  if not start_date or not end_date:
    raise ValueError("Both --start-date and --end-date are required.")
  if not reference_zarr or not reference_zarr.strip():
    raise ValueError("--reference-zarr must be explicitly provided.")
  if not output_dir or not output_dir.strip():
    raise ValueError("--output-dir must be explicitly provided.")

  os.makedirs(output_dir, exist_ok=True)
  target_rebuilt_zarr = (
      rebuilt_zarr
      if rebuilt_zarr is not None
      else os.path.join(output_dir, f"rebuilt_{prod_key.lower()}.zarr")
  )

  start_ts = pd.Timestamp(start_date).normalize()
  end_ts = pd.Timestamp(end_date).normalize()
  if end_ts < start_ts:
    raise ValueError(
        f"end_date ({end_date}) must be >= start_date ({start_date})."
    )

  t0_build = time.time()
  if prod_key == "CPC":
    build_cpc_archive(
        target_zarr=target_rebuilt_zarr,
        start_year=int(start_ts.year),
        end_year=int(end_ts.year),
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        project=project,
        cache_dir=cache_dir,
        source_url_template=source_url_template,
        cleanup_cache=cleanup_cache,
        overwrite=True,
        extend_archive=False,
        num_workers=num_workers,
    )
  else:
    if imerg_source == "gesdisc" and imerg_local_dir is None:
      ok, http_status, msg = verify_gesdisc_access(
          start_ts,
          username=earthdata_username,
          password=earthdata_password,
          token=earthdata_token,
          netrc_path=netrc_path,
      )
      logger.info("NASA GES DISC preflight check: %s (status=%d)", msg, http_status)
      if not ok:
        raise PermissionError(
            f"NASA GES DISC access check failed prior to IMERG build: {msg}"
        )

    build_imerg_archive(
        target_zarr=target_rebuilt_zarr,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        source_type=imerg_source,
        local_format=imerg_local_format,
        project=project,
        gesdisc_url=gesdisc_url,
        batch_size=30,
        num_workers=num_workers,
        cache_dir=cache_dir,
        cleanup_cache=cleanup_cache,
        overwrite=True,
        in_place=False,
        extend_archive=False,
        local_dir=imerg_local_dir,
        earthdata_username=earthdata_username,
        earthdata_password=earthdata_password,
        earthdata_token=earthdata_token,
        netrc_path=netrc_path,
    )
  build_elapsed = time.time() - t0_build

  t0_cmp = time.time()
  summary_df, daily_df = compare_gridded_archives(
      product=prod_key,
      rebuilt_zarr=target_rebuilt_zarr,
      reference_zarr=reference_zarr,
      start_date=start_ts.strftime("%Y-%m-%d"),
      end_date=end_ts.strftime("%Y-%m-%d"),
  )
  compare_elapsed = time.time() - t0_cmp

  summary_df["build_wall_sec"] = build_elapsed
  summary_df["compare_wall_sec"] = compare_elapsed

  summary_csv = os.path.join(output_dir, "summary_metrics.csv")
  daily_csv = os.path.join(output_dir, "daily_metrics.csv")
  summary_df.to_csv(summary_csv, index=False)
  daily_df.to_csv(daily_csv, index=False)

  report_path = _write_markdown_report(
      output_dir,
      summary_df,
      daily_df,
      rebuilt_zarr=target_rebuilt_zarr,
      reference_zarr=reference_zarr,
      build_elapsed_sec=build_elapsed,
      compare_elapsed_sec=compare_elapsed,
  )
  logger.info(
      "Saved gridded archive benchmark outputs to %s (report: %s)",
      output_dir,
      report_path,
  )
  return summary_df, daily_df


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds CLI argument parser for gridded archive benchmarking."""
  parser = argparse.ArgumentParser(
      prog="benchmark-gridded-archive",
      description=(
          "Rebuild a date window for CPC or IMERG from upstream and compare "
          "against a reference gridded Zarr archive."
      ),
  )
  parser.add_argument(
      "--product",
      type=str,
      required=True,
      choices=list(SUPPORTED_ARCHIVE_PRODUCTS),
      help="Gridded archive product to benchmark ('CPC' or 'IMERG').",
  )
  parser.add_argument(
      "--start-date",
      "--start_date",
      dest="start_date",
      type=str,
      required=True,
      help="Inclusive start date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--end-date",
      "--end_date",
      dest="end_date",
      type=str,
      required=True,
      help="Inclusive end date (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--reference-zarr",
      "--reference_zarr",
      dest="reference_zarr",
      type=str,
      required=True,
      help="Path or gs:// URI to the reference gridded Zarr store.",
  )
  parser.add_argument(
      "--output-dir",
      "--output_dir",
      dest="output_dir",
      type=str,
      required=True,
      help="Directory where benchmark CSVs, report, and rebuilt Zarr are written.",
  )
  parser.add_argument(
      "--rebuilt-zarr",
      "--rebuilt_zarr",
      dest="rebuilt_zarr",
      type=str,
      default=None,
      help="Optional path for the rebuilt Zarr store (defaults to <output-dir>/rebuilt_<product>.zarr).",
  )
  parser.add_argument(
      "--cache-dir",
      "--cache_dir",
      dest="cache_dir",
      type=str,
      default=None,
      help="Optional directory to stage downloaded upstream NetCDF/HDF5 files.",
  )
  parser.add_argument(
      "--cleanup-cache",
      "--cleanup_cache",
      dest="cleanup_cache",
      action="store_true",
      help="Delete downloaded upstream cache files after building.",
  )
  parser.add_argument(
      "--source-url-template",
      "--source_url_template",
      dest="source_url_template",
      type=str,
      default=NOAA_PSL_URL_TEMPLATE,
      help="Upstream NOAA PSL URL template containing {year} (CPC only).",
  )
  parser.add_argument(
      "--imerg-source",
      "--imerg_source",
      dest="imerg_source",
      type=str,
      default="gesdisc",
      choices=["gesdisc", "local"],
      help="IMERG upstream source mode ('gesdisc' or 'local').",
  )
  parser.add_argument(
      "--imerg-local-dir",
      "--imerg_local_dir",
      dest="imerg_local_dir",
      type=str,
      default=None,
      help="Local directory containing IMERG granules when --imerg-source=local.",
  )
  parser.add_argument(
      "--imerg-local-format",
      "--imerg_local_format",
      dest="imerg_local_format",
      type=str,
      default="nc4",
      choices=["nc4", "h5"],
      help="Local IMERG granule format ('nc4' or 'h5').",
  )
  parser.add_argument(
      "--gesdisc-url",
      "--gesdisc_url",
      dest="gesdisc_url",
      type=str,
      default=DEFAULT_GESDISC_URL,
      help="Base URL for NASA GES DISC IMERG daily archive.",
  )
  parser.add_argument(
      "--earthdata-username",
      "--earthdata_username",
      dest="earthdata_username",
      type=str,
      default=None,
      help="Optional NASA Earthdata Login username.",
  )
  parser.add_argument(
      "--earthdata-password",
      "--earthdata_password",
      dest="earthdata_password",
      type=str,
      default=None,
      help="Optional NASA Earthdata Login password.",
  )
  parser.add_argument(
      "--earthdata-token",
      "--earthdata_token",
      dest="earthdata_token",
      type=str,
      default=None,
      help="Optional NASA Earthdata Bearer token.",
  )
  parser.add_argument(
      "--netrc-path",
      "--netrc_path",
      dest="netrc_path",
      type=str,
      default=None,
      help="Optional custom path to .netrc file.",
  )
  parser.add_argument(
      "--num-workers",
      "--num_workers",
      dest="num_workers",
      type=int,
      default=4,
      help="Number of parallel download/extraction workers.",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="Optional GCP project for gs:// access.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for `multimet/gridded_archive_builders/benchmark.py`."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  args = build_arg_parser().parse_args(argv)
  run_gridded_archive_benchmark(
      product=args.product,
      start_date=args.start_date,
      end_date=args.end_date,
      reference_zarr=args.reference_zarr,
      output_dir=args.output_dir,
      rebuilt_zarr=args.rebuilt_zarr,
      cache_dir=args.cache_dir,
      cleanup_cache=args.cleanup_cache,
      source_url_template=args.source_url_template,
      imerg_source=args.imerg_source,
      imerg_local_dir=args.imerg_local_dir,
      imerg_local_format=args.imerg_local_format,
      gesdisc_url=args.gesdisc_url,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
      num_workers=args.num_workers,
      project=args.project,
  )


if __name__ == "__main__":
  main()
