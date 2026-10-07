# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: C901, E501, FBT003, PLR0912, PLR0913, PLR0915, PLR0917, PLR2004, PTH123, S108, S603, T201

"""End-to-end Caravan benchmark comparing Python `return_periods` vs. USGS R & Fortran.

This benchmark harness does NOT redistribute third-party USGS source code. Instead, it provides
explicit setup instructions (and optional automated compilation) so users can clone the two
official USGS repositories:

1. USGS `peakfqr` (v8.0 Fortran `emafit.f`, `probfun.f`, `imslfake.f`):
   https://code.usgs.gov/water/peakfqr.git
2. USGS CRAN `MGBT` (R package `MGBT17c`, `MGBT17c.verb`, `MGBTcohn2013`, `MGBTcohn2016`, `MGBTnb`):
   https://github.com/cran/MGBT.git

It then compiles `peakfq.so` directly from the unmodified USGS Fortran sources, runs `MGBT` via
`R`, runs the Python `return_periods` package across Caravan basins in both volumetric flow
(`ft^3/s`) and specific discharge (`mm/day`), outputs the master benchmark CSV, and generates all
four diagnostic comparison figures.

Strict Benchmarking Rules Enforced:

- Zero `try/except` blocks anywhere in the benchmark pipeline.
- Zero imputed data or fallback values.
- Zero Python approximations standing in for R or Fortran: every Fortran column comes from the
  compiled `peakfq.so` binary (`emafit_`) and every R column comes from `R` executing the
  official CRAN `MGBT` R functions.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr

from return_periods.generalized_expected_moments_algorithm import GEMAFitter
from return_periods.grubbs_beck_tester import (
    GrubbsBeckTester,
    MultipleGrubbsBeckTester,
)
from return_periods.theoretical_distribution_utilities import (
    SimpleLogPearson3Fitter,
    pearson3_invcdf,
    sample_moments,
)

mpl.use('Agg')
import matplotlib.pyplot as plt

SETUP_INSTRUCTIONS = """
================================================================================
USGS BULLETIN 17C BENCHMARK — EXTERNAL REPOSITORY SETUP INSTRUCTIONS
================================================================================

Because we do not redistribute third-party USGS R or Fortran code in this repository,
follow these steps once to clone and build the two official USGS reference packages
in an external directory (e.g. `/tmp/usgs_src`):

1. Prerequisites
----------------
Ensure `git`, `gfortran` (with LAPACK/BLAS), and `R` (`Rscript`) are installed:
  # Debian / Ubuntu:
  sudo apt-get update && sudo apt-get install -y gfortran liblapack-dev libblas-dev r-base r-base-dev

2. Clone & Compile Official USGS `peakfqr` (Fortran `emafit.f` / `probfun.f`)
-----------------------------------------------------------------------------
  mkdir -p /tmp/usgs_src
  git clone https://code.usgs.gov/water/peakfqr.git /tmp/usgs_src/peakfqr

  # Compile the unmodified Fortran 77/90 source into a shared library `peakfq.so`:
  gfortran -shared -fPIC -O2 \\
      /tmp/usgs_src/peakfqr/src/emafit.f \\
      /tmp/usgs_src/peakfqr/src/probfun.f \\
      /tmp/usgs_src/peakfqr/src/imslfake.f \\
      -llapack -lblas \\
      -o /tmp/usgs_src/peakfqr/src/peakfq.so

3. Clone Official USGS CRAN `MGBT` R Package
--------------------------------------------
  git clone https://github.com/cran/MGBT.git /tmp/usgs_src/MGBT

4. Run the Full Caravan Benchmark (Python vs. Compiled Fortran vs. CRAN R)
--------------------------------------------------------------------------
  python -m benchmarks.return_periods \\
      --caravan-dir /path/to/caravan \\
      --peakfqr-repo /tmp/usgs_src/peakfqr \\
      --mgbt-repo /tmp/usgs_src/MGBT \\
      --output-dir ./benchmark_output \\
      --workers 16
================================================================================
"""

SUBDATASETS = (
    'camels',
    'camelsaus',
    'camelsbr',
    'camelscl',
    'camelsgb',
    'hysets',
    'lamah',
)

# m^3/s per (mm/day * km^2) = 1000 / 86400; 1 m^3/s = 35.31466672148859 ft^3/s.
MM_DAY_KM2_TO_CFS = (1000.0 / 86400.0) * 35.31466672148859


def extract_caravan_peaks(
    caravan_dir: Path,
    peaks_npz_path: Path,
    min_valid_days_per_year: int = 330,
    min_years: int = 10,
) -> dict[str, Any]:
    """Extract October-September Water-Year annual maximums from Caravan NetCDF files."""
    if peaks_npz_path.exists():
        print(f'Loading cached Caravan annual peaks from {peaks_npz_path} ...')
        loaded = np.load(peaks_npz_path, allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}

    print(
        f'Extracting Water-Year annual maximums from Caravan directory {caravan_dir} ...'
    )
    basin_ids: list[str] = []
    subdatasets: list[str] = []
    areas_km2: list[float] = []
    peaks_mm_list: list[np.ndarray] = []
    peaks_cfs_list: list[np.ndarray] = []

    for sub in SUBDATASETS:
        attr_csv = (
            caravan_dir / 'attributes' / sub / f'attributes_other_{sub}.csv'
        )
        nc_dir = caravan_dir / 'timeseries' / 'netcdf' / sub
        if not attr_csv.exists() or not nc_dir.exists():
            continue

        attr_df = pd.read_csv(attr_csv).set_index('gauge_id')
        nc_files = sorted(nc_dir.glob('*.nc'))
        print(f'  Scanning {sub}: {len(nc_files)} NetCDF files ...')

        for nc_file in nc_files:
            gid = nc_file.stem
            if gid not in attr_df.index:
                continue
            area = float(attr_df.loc[gid, 'area'])
            if not np.isfinite(area) or area <= 0.0:
                continue

            ds = xr.open_dataset(nc_file)
            if 'streamflow' not in ds:
                ds.close()
                continue
            s = ds['streamflow'].to_pandas().dropna()
            ds.close()
            if s.empty:
                continue

            s = s[s >= 0.0]
            if s.empty:
                continue

            wy = s.index.year + (s.index.month >= 10).astype(int)
            counts = s.groupby(wy).count()
            valid_wys = counts[counts >= min_valid_days_per_year].index
            if len(valid_wys) < min_years:
                continue

            ann_max_mm = (
                s[wy.isin(valid_wys)]
                .groupby(wy[wy.isin(valid_wys)])
                .max()
                .to_numpy(dtype=np.float64)
            )
            pos_vals = ann_max_mm[ann_max_mm > 0.0]
            if len(ann_max_mm) < min_years or len(np.unique(pos_vals)) < 3:
                continue

            ann_max_cfs = ann_max_mm * area * MM_DAY_KM2_TO_CFS
            basin_ids.append(gid)
            subdatasets.append(sub)
            areas_km2.append(area)
            peaks_mm_list.append(ann_max_mm)
            peaks_cfs_list.append(ann_max_cfs)

    peaks_npz_path.parent.mkdir(parents=True, exist_ok=True)
    data_dict = {
        'basin_ids': np.array(basin_ids, dtype=object),
        'subdatasets': np.array(subdatasets, dtype=object),
        'areas_km2': np.array(areas_km2, dtype=np.float64),
        'peaks_mm': np.array(peaks_mm_list, dtype=object),
        'peaks_cfs': np.array(peaks_cfs_list, dtype=object),
    }
    np.savez_compressed(peaks_npz_path, **data_dict)
    print(f'Saved {len(basin_ids)} valid Caravan basins to {peaks_npz_path}.')
    return data_dict


def ensure_compiled_peakfq_so(peakfqr_repo: Path) -> Path:
    """Verify or compile `peakfq.so` from unmodified USGS Fortran sources."""
    so_path = peakfqr_repo / 'src' / 'peakfq.so'
    if so_path.exists():
        return so_path

    fortran_files = [
        peakfqr_repo / 'src' / 'emafit.f',
        peakfqr_repo / 'src' / 'probfun.f',
        peakfqr_repo / 'src' / 'imslfake.f',
    ]
    for fpath in fortran_files:
        if not fpath.exists():
            raise FileNotFoundError(
                f'Missing USGS Fortran file {fpath}.\n{SETUP_INSTRUCTIONS}'
            )

    gfortran_bin = shutil.which('gfortran')
    if gfortran_bin is None:
        raise RuntimeError(
            f'`gfortran` compiler not found on PATH.\n{SETUP_INSTRUCTIONS}'
        )

    cmd = [
        gfortran_bin,
        '-shared',
        '-fPIC',
        '-O2',
        str(fortran_files[0]),
        str(fortran_files[1]),
        str(fortran_files[2]),
        '-llapack',
        '-lblas',
        '-o',
        str(so_path),
    ]
    print('Compiling USGS Fortran shared library:', ' '.join(cmd))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f'Failed to compile peakfq.so (exit {res.returncode}):\n{res.stderr}'
        )
    return so_path


def supervise_fortran_workers(
    peaks_npz_path: Path,
    peakfqr_repo: Path,
    peakfq_so_path: Path,
    r_bin: str,
    output_dir: Path,
    total_basins: int,
    num_workers: int,
    basin_timeout_sec: float,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Run parallel `peakfqr` (R wrapper + compiled Fortran `peakfq.so`) workers."""
    f_dir = output_dir / 'fortran_workers'
    f_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(peaks_npz_path, allow_pickle=True)
    basin_ids = data['basin_ids'][:total_basins]
    peaks_cfs = data['peaks_cfs'][:total_basins]
    peaks_mm = data['peaks_mm'][:total_basins]

    main_r = (peakfqr_repo / 'R' / 'main.R').resolve()
    wrappers_r = (peakfqr_repo / 'R' / 'fortranWrappers.R').resolve()
    r_script_file = f_dir / 'fortran_worker.R'
    r_code = f"""args <- commandArgs(trailingOnly = TRUE)
in_file <- args[1]
out_file <- args[2]
prog_file <- args[3]
start_line <- as.integer(args[4])

dyn.load("{peakfq_so_path.resolve()}")
source("{main_r}")
source("{wrappers_r}")

lines <- readLines(in_file)
n_lines <- length(lines)
aeps <- c(0.5, 0.2, 0.1, 0.04, 0.02, 0.01, 0.005, 0.002)

if (start_line <= n_lines) {{
  for (idx in start_line:n_lines) {{
    parts <- strsplit(lines[idx], "\\t")[[1]]
    bid <- parts[1]
    writeLines(paste("RUNNING", idx, bid, sep="\\t"), prog_file)

    x_cfs <- as.numeric(strsplit(parts[2], ",")[[1]])
    x_mm  <- as.numeric(strsplit(parts[3], ",")[[1]])
    n <- length(x_cfs)

    QT_cfs <- data.frame(
      ql = pmax(x_cfs, Qmin),
      qu = pmax(x_cfs, Qmin),
      tl = rep(Qmin, n),
      tu = rep(Qmax, n),
      dtype = rep(0L, n),
      peak_WY = seq_len(n)
    )
    ema_cfs <- emafit(QT_cfs, LOthresh = 0, rG = 0, rGmse = -1e99, AEPs = aeps)
    exp_cfs <- ema_cfs[[1]]
    qnt_cfs <- ema_cfs[[2]]
    dat_cfs <- ema_cfs[[3]]
    ncen_cfs <- sum(dat_cfs$ql_ema != dat_cfs$qu_ema)

    QT_mm <- data.frame(
      ql = pmax(x_mm, Qmin),
      qu = pmax(x_mm, Qmin),
      tl = rep(Qmin, n),
      tu = rep(Qmax, n),
      dtype = rep(0L, n),
      peak_WY = seq_len(n)
    )
    ema_mm <- emafit(QT_mm, LOthresh = 0, rG = 0, rGmse = -1e99, AEPs = aeps)
    exp_mm <- ema_mm[[1]]
    qnt_mm <- ema_mm[[2]]
    dat_mm <- ema_mm[[3]]
    ncen_mm <- sum(dat_mm$ql_ema != dat_mm$qu_ema)

    row_str <- paste(
      idx, bid,
      exp_cfs$PILFs, ncen_cfs, sprintf("%.12g", exp_cfs$PILF_Thresh),
      sprintf("%.12g", exp_cfs$Mean), sprintf("%.12g", exp_cfs$StandDev^2), sprintf("%.12g", exp_cfs$Skew),
      sprintf("%.12g", qnt_cfs$Estimate[1]), sprintf("%.12g", qnt_cfs$Estimate[2]),
      sprintf("%.12g", qnt_cfs$Estimate[3]), sprintf("%.12g", qnt_cfs$Estimate[4]),
      sprintf("%.12g", qnt_cfs$Estimate[5]), sprintf("%.12g", qnt_cfs$Estimate[6]),
      sprintf("%.12g", qnt_cfs$Estimate[7]), sprintf("%.12g", qnt_cfs$Estimate[8]),
      exp_mm$PILFs, ncen_mm, sprintf("%.12g", exp_mm$PILF_Thresh),
      sprintf("%.12g", exp_mm$Mean), sprintf("%.12g", exp_mm$StandDev^2), sprintf("%.12g", exp_mm$Skew),
      sprintf("%.12g", qnt_mm$Estimate[1]), sprintf("%.12g", qnt_mm$Estimate[2]),
      sprintf("%.12g", qnt_mm$Estimate[3]), sprintf("%.12g", qnt_mm$Estimate[4]),
      sprintf("%.12g", qnt_mm$Estimate[5]), sprintf("%.12g", qnt_mm$Estimate[6]),
      sprintf("%.12g", qnt_mm$Estimate[7]), sprintf("%.12g", qnt_mm$Estimate[8]),
      sep = "\\t"
    )
    cat(row_str, "\\n", sep = "", file = out_file, append = TRUE)
    writeLines(paste("DONE", idx, bid, sep="\\t"), prog_file)
  }}
}}
"""
    r_script_file.write_text(r_code, encoding='utf-8')

    chunk = (total_basins + num_workers - 1) // num_workers
    workers: list[dict[str, Any]] = []
    for wid in range(num_workers):
        s_idx = wid * chunk
        e_idx = min(total_basins, (wid + 1) * chunk)
        if s_idx >= e_idx:
            break
        in_tsv = f_dir / f'in_{wid}.tsv'
        out_tsv = f_dir / f'out_{wid}.tsv'
        prog_txt = f_dir / f'prog_{wid}.txt'
        crash_tsv = f_dir / f'crashes_{wid}.tsv'
        out_tsv.unlink(missing_ok=True)
        prog_txt.unlink(missing_ok=True)
        crash_tsv.unlink(missing_ok=True)

        with open(in_tsv, 'w', encoding='utf-8') as f:
            for i in range(s_idx, e_idx):
                cfs_s = ','.join(f'{v:.12g}' for v in peaks_cfs[i])
                mm_s = ','.join(f'{v:.12g}' for v in peaks_mm[i])
                f.write(f'{basin_ids[i]}\t{cfs_s}\t{mm_s}\n')

        workers.append(
            {
                'wid': wid,
                'cur': 1,
                'total': e_idx - s_idx,
                'in': in_tsv,
                'out': out_tsv,
                'prog': prog_txt,
                'crash': crash_tsv,
                'proc': None,
                'last_idx': -1,
                'last_time': time.time(),
            }
        )

    def spawn_worker(w: dict[str, Any]) -> None:
        cmd = [
            r_bin,
            '--vanilla',
            '--slave',
            '-f',
            str(r_script_file),
            '--args',
            str(w['in']),
            str(w['out']),
            str(w['prog']),
            str(w['cur']),
        ]
        w['proc'] = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        w['last_idx'] = w['cur']
        w['last_time'] = time.time()

    for w in workers:
        spawn_worker(w)

    active = len(workers)
    while active > 0:
        time.sleep(0.2)
        active = 0
        for w in workers:
            proc = w['proc']
            if proc is None:
                continue
            status_line = (
                w['prog'].read_text(encoding='utf-8').strip()
                if w['prog'].exists()
                else ''
            )
            parts = status_line.split('\t') if status_line else []
            state = parts[0] if len(parts) == 3 else ''
            c_idx = int(parts[1]) if len(parts) == 3 else w['cur']
            c_bid = parts[2] if len(parts) == 3 else f'line_{c_idx}'

            if c_idx != w['last_idx']:
                w['last_idx'] = c_idx
                w['last_time'] = time.time()

            ret = proc.poll()
            if ret is not None:
                if ret == 0 and state == 'DONE' and c_idx == w['total']:
                    w['proc'] = None
                    continue
                _, stderr_txt = proc.communicate()
                err_msg = (
                    stderr_txt.strip().replace('\n', ' | ') or f'EXIT_{ret}'
                )
                with open(w['crash'], 'a', encoding='utf-8') as cf:
                    cf.write(f'{c_idx}\t{c_bid}\t{err_msg}\n')
                w['cur'] = c_idx + 1
                if w['cur'] <= w['total']:
                    spawn_worker(w)
                    active += 1
                else:
                    w['proc'] = None
            elif time.time() - w['last_time'] > basin_timeout_sec:
                proc.kill()
                proc.wait()
                with open(w['crash'], 'a', encoding='utf-8') as cf:
                    cf.write(
                        f'{c_idx}\t{c_bid}\tTIMEOUT_{basin_timeout_sec}s\n'
                    )
                w['cur'] = c_idx + 1
                if w['cur'] <= w['total']:
                    spawn_worker(w)
                    active += 1
                else:
                    w['proc'] = None
            else:
                active += 1

    fortran_rows: dict[str, list[str]] = {}
    fortran_crashes: dict[str, str] = {}
    for w in workers:
        if w['out'].exists():
            for line in w['out'].read_text(encoding='utf-8').splitlines():
                p = line.split('\t')
                if len(p) >= 22:
                    fortran_rows[p[1]] = p
        if w['crash'].exists():
            for line in w['crash'].read_text(encoding='utf-8').splitlines():
                p = line.split('\t')
                if len(p) >= 3:
                    fortran_crashes[p[1]] = p[2]
    return fortran_rows, fortran_crashes


def run_r_mgbt_workers(
    peaks_npz_path: Path,
    r_bin: str,
    mgbt_repo: Path,
    output_dir: Path,
    total_basins: int,
    num_workers: int,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Run parallel `R` workers calling CRAN `MGBT` functions on Caravan basins."""
    r_dir = output_dir / 'r_mgbt_workers'
    r_dir.mkdir(parents=True, exist_ok=True)

    r_files = [
        'CondMoms.R',
        'EMS.R',
        'gtmoms.R',
        'peta.R',
        'RthOrderPValueOrthoT.R',
        'MGBT.R',
    ]
    for rf in r_files:
        rf_path = mgbt_repo / 'R' / rf
        if not rf_path.exists():
            raise FileNotFoundError(
                f'Missing CRAN MGBT R file {rf_path}.\n{SETUP_INSTRUCTIONS}'
            )

    data = np.load(peaks_npz_path, allow_pickle=True)
    basin_ids = data['basin_ids'][:total_basins]
    peaks_cfs = data['peaks_cfs'][:total_basins]
    peaks_mm = data['peaks_mm'][:total_basins]

    r_script_file = r_dir / 'worker.R'
    source_lines = '\n'.join(
        f'source("{(mgbt_repo / "R" / rf).resolve()}")' for rf in r_files
    )
    r_code = (
        source_lines
        + '\n'
        + """args <- commandArgs(trailingOnly = TRUE)
in_file <- args[1]
out_file <- args[2]
prog_file <- args[3]
start_line <- as.integer(args[4])

lines <- readLines(in_file)
n_lines <- length(lines)
if (start_line <= n_lines) {
  for (idx in start_line:n_lines) {
    parts <- strsplit(lines[idx], "\\t")[[1]]
    bid <- parts[1]
    cfs <- as.numeric(strsplit(parts[2], ",")[[1]])
    mm  <- as.numeric(strsplit(parts[3], ",")[[1]])

    r_17c_cfs <- MGBT17c(cfs)
    r_17c_mm  <- MGBT17c(mm)
    r_verb    <- MGBT17c.verb(cfs)
    r_2013    <- MGBTcohn2013(cfs)

    writeLines(paste("STAGE1", idx, bid, r_17c_cfs$klow, r_17c_cfs$lowout,
                     r_17c_mm$klow, r_17c_mm$lowout, r_verb$klow, r_2013$klow, sep="\\t"), prog_file)

    r_2016 <- MGBTcohn2016(cfs)
    r_nb   <- MGBTnb(cfs)

    row_str <- paste(idx, bid,
                     r_17c_cfs$klow, sprintf("%.12g", r_17c_cfs$lowout),
                     r_17c_mm$klow,  sprintf("%.12g", r_17c_mm$lowout),
                     r_verb$klow, r_2013$klow, r_2016$klow, r_nb$klow,
                     sep="\\t")
    cat(row_str, "\\n", sep="", file=out_file, append=TRUE)
    writeLines(paste("DONE", idx, bid, sep="\\t"), prog_file)
  }
}
"""
    )
    r_script_file.write_text(r_code, encoding='utf-8')

    chunk = (total_basins + num_workers - 1) // num_workers
    workers: list[dict[str, Any]] = []
    for wid in range(num_workers):
        s_idx = wid * chunk
        e_idx = min(total_basins, (wid + 1) * chunk)
        if s_idx >= e_idx:
            break
        in_tsv = r_dir / f'in_{wid}.tsv'
        out_tsv = r_dir / f'out_{wid}.tsv'
        prog_txt = r_dir / f'prog_{wid}.txt'
        crash_tsv = r_dir / f'crashes_{wid}.tsv'
        out_tsv.unlink(missing_ok=True)
        prog_txt.unlink(missing_ok=True)
        crash_tsv.unlink(missing_ok=True)

        with open(in_tsv, 'w', encoding='utf-8') as f:
            for i in range(s_idx, e_idx):
                cfs_s = ','.join(f'{v:.12g}' for v in peaks_cfs[i])
                mm_s = ','.join(f'{v:.12g}' for v in peaks_mm[i])
                f.write(f'{basin_ids[i]}\t{cfs_s}\t{mm_s}\n')

        workers.append(
            {
                'wid': wid,
                'cur': 1,
                'total': e_idx - s_idx,
                'in': in_tsv,
                'out': out_tsv,
                'prog': prog_txt,
                'crash': crash_tsv,
                'proc': None,
            }
        )

    def spawn_r(w: dict[str, Any]) -> None:
        cmd = [
            r_bin,
            '--vanilla',
            '--slave',
            '-f',
            str(r_script_file),
            '--args',
            str(w['in']),
            str(w['out']),
            str(w['prog']),
            str(w['cur']),
        ]
        w['proc'] = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    for w in workers:
        spawn_r(w)

    active = len(workers)
    while active > 0:
        time.sleep(0.2)
        active = 0
        for w in workers:
            proc = w['proc']
            if proc is None:
                continue
            ret = proc.poll()
            if ret is not None:
                status_line = (
                    w['prog'].read_text(encoding='utf-8').strip()
                    if w['prog'].exists()
                    else ''
                )
                parts = status_line.split('\t') if status_line else []
                state = parts[0] if parts else ''
                c_line = int(parts[1]) if len(parts) >= 2 else w['cur']
                c_bid = parts[2] if len(parts) >= 3 else f'line_{c_line}'

                if ret == 0 and state == 'DONE' and c_line == w['total']:
                    w['proc'] = None
                    continue
                _, stderr_txt = proc.communicate()
                err_msg = (
                    stderr_txt.strip().replace('\n', ' | ') or f'EXIT_{ret}'
                )
                if state == 'STAGE1' and len(parts) == 9:
                    with open(w['out'], 'a', encoding='utf-8') as of:
                        of.write(
                            f'{c_line}\t{c_bid}\t{parts[3]}\t{parts[4]}\t{parts[5]}\t{parts[6]}\t{parts[7]}\t{parts[8]}\t-1\t-1\n'
                        )
                with open(w['crash'], 'a', encoding='utf-8') as cf:
                    cf.write(f'{c_line}\t{c_bid}\t{err_msg}\n')
                w['cur'] = c_line + 1
                if w['cur'] <= w['total']:
                    spawn_r(w)
                    active += 1
                else:
                    w['proc'] = None
            else:
                active += 1

    r_rows: dict[str, list[str]] = {}
    r_crashes: dict[str, str] = {}
    for w in workers:
        if w['out'].exists():
            for line in w['out'].read_text(encoding='utf-8').splitlines():
                p = line.split('\t')
                if len(p) >= 10:
                    r_rows[p[1]] = p
        if w['crash'].exists():
            for line in w['crash'].read_text(encoding='utf-8').splitlines():
                p = line.split('\t')
                if len(p) >= 3:
                    r_crashes[p[1]] = p[2]
    return r_rows, r_crashes


def assemble_master_csv(
    peaks_npz_path: Path,
    fortran_rows: dict[str, list[str]],
    fortran_crashes: dict[str, str],
    r_rows: dict[str, list[str]],
    r_crashes: dict[str, str],
    output_dir: Path,
    total_basins: int,
) -> pd.DataFrame:
    """Run Python `return_periods` fitters and merge with compiled Fortran & CRAN R outputs."""
    data = np.load(peaks_npz_path, allow_pickle=True)
    basin_ids = data['basin_ids'][:total_basins]
    subdatasets = data['subdatasets'][:total_basins]
    areas_km2 = data['areas_km2'][:total_basins]
    peaks_cfs = data['peaks_cfs'][:total_basins]
    peaks_mm = data['peaks_mm'][:total_basins]

    rps = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0])
    aep = 1.0 / rps

    records: list[dict[str, Any]] = []
    for i in range(total_basins):
        bid = str(basin_ids[i])
        sub = str(subdatasets[i])
        area = float(areas_km2[i])
        cfs = np.asarray(peaks_cfs[i], dtype=np.float64)
        mm = np.asarray(peaks_mm[i], dtype=np.float64)
        n = len(cfs)
        n_zeros = int(np.sum(cfs <= 0.0))

        mgbt_cfs = MultipleGrubbsBeckTester(data=cfs, is_log_transformed=False)
        mgbt_mm = MultipleGrubbsBeckTester(data=mm, is_log_transformed=False)
        pos_cfs = cfs[cfs > 0.0]
        gb17b = GrubbsBeckTester(np.log10(pos_cfs))
        klow_17b = n_zeros + len(gb17b.out_of_population_sample)

        gema_cfs = GEMAFitter(data=cfs, log_transform=True)
        q_py_cfs = gema_cfs.flow_values_from_exceedance_probabilities(aep)

        gema_mm = GEMAFitter(data=mm, log_transform=True)
        q_py_mm_in_cfs = (
            gema_mm.flow_values_from_exceedance_probabilities(aep)
            * area
            * MM_DAY_KM2_TO_CFS
        )

        # Non-iterative LP3 (SimpleLogPearson3Fitter)
        lp3_raw = SimpleLogPearson3Fitter(data=cfs, log_transform=True)
        q_lp3_raw = lp3_raw.flow_values_from_exceedance_probabilities(aep)

        above_cfs = gema_cfs.outlier_tester.in_population_sample
        m_0step = sample_moments(above_cfs)
        if abs(m_0step[2]) < 1e-6:
            m_0step = (
                m_0step[0],
                m_0step[1],
                1e-6 if m_0step[2] >= 0 else -1e-6,
            )
        q_0step = 10.0 ** np.asarray(
            pearson3_invcdf(1.0 - aep, m_0step), dtype=np.float64
        )
        m_1step = gema_cfs._update_moments_from_intervals_and_moments(m_0step)  # noqa: SLF001
        q_1step = 10.0 ** np.asarray(
            pearson3_invcdf(1.0 - aep, m_1step), dtype=np.float64
        )
        m_2step = gema_cfs._update_moments_from_intervals_and_moments(m_1step)  # noqa: SLF001
        q_2step = 10.0 ** np.asarray(
            pearson3_invcdf(1.0 - aep, m_2step), dtype=np.float64
        )

        rec: dict[str, Any] = {
            'basin_id': bid,
            'subdataset': sub,
            'area_km2': area,
            'n_years': n,
            'n_zeros': n_zeros,
            'klow_py_cfs': int(mgbt_cfs.klow),
            'klow_py_mm': int(mgbt_mm.klow),
            'thresh_py_cfs': float(10.0**mgbt_cfs.threshold),
            'klow_17b_cfs': klow_17b,
            'mu_py_cfs': float(gema_cfs.moments[0]),
            'sigma_py_cfs': float(gema_cfs.moments[1]),
            'skew_py_cfs': float(gema_cfs.moments[2]),
        }
        for j, rp_int in enumerate([2, 5, 10, 25, 50, 100, 200, 500]):
            rec[f'q{rp_int}_py_cfs'] = float(q_py_cfs[j])
            rec[f'q{rp_int}_py_mm_in_cfs'] = float(q_py_mm_in_cfs[j])
            rec[f'q{rp_int}_simple_lp3'] = float(q_lp3_raw[j])
            rec[f'q{rp_int}_mgbt_0step'] = float(q_0step[j])
            rec[f'q{rp_int}_mgbt_1step'] = float(q_1step[j])
            rec[f'q{rp_int}_mgbt_2step'] = float(q_2step[j])

        if bid in r_rows:
            r_p = r_rows[bid]
            rec['r_status'] = 'CRASH_COHN2016_NB' if bid in r_crashes else 'OK'
            rec['klow_r_17c_cfs'] = int(r_p[2])
            rec['klow_r_17c_mm'] = int(r_p[4])
            rec['klow_r_verb_cfs'] = int(r_p[6])
            rec['klow_r_2013_cfs'] = int(r_p[7])
            rec['klow_r_2016_cfs'] = int(r_p[8]) if int(r_p[8]) >= 0 else np.nan
            rec['klow_r_nb_cfs'] = int(r_p[9]) if int(r_p[9]) >= 0 else np.nan
        else:
            rec['r_status'] = r_crashes.get(bid, 'MISSING')
            for col in (
                'klow_r_17c_cfs',
                'klow_r_17c_mm',
                'klow_r_verb_cfs',
                'klow_r_2013_cfs',
                'klow_r_2016_cfs',
                'klow_r_nb_cfs',
            ):
                rec[col] = np.nan

        if bid in fortran_rows:
            f_p = fortran_rows[bid]
            rec['fortran_status'] = 'OK'
            rec['nlow_fortran_cfs'] = int(f_p[2])
            rec['nlow_censored_fortran_cfs'] = int(f_p[3])
            rec['mu_fortran_cfs'] = float(f_p[5])
            rec['sigma_fortran_cfs'] = float(np.sqrt(float(f_p[6])))
            rec['skew_fortran_cfs'] = float(f_p[7])
            for j, rp_int in enumerate([2, 5, 10, 25, 50, 100, 200, 500]):
                rec[f'q{rp_int}_fortran_cfs'] = float(f_p[8 + j])
            rec['nlow_fortran_mm'] = int(f_p[16])
            rec['nlow_censored_fortran_mm'] = int(f_p[17])
            for j, rp_int in enumerate([2, 5, 10, 25, 50, 100, 200, 500]):
                rec[f'q{rp_int}_fortran_mm_in_cfs'] = (
                    float(f_p[22 + j]) * area * MM_DAY_KM2_TO_CFS
                )
        else:
            rec['fortran_status'] = fortran_crashes.get(bid, 'CRASH')
            for col in (
                'nlow_fortran_cfs',
                'nlow_censored_fortran_cfs',
                'mu_fortran_cfs',
                'sigma_fortran_cfs',
                'skew_fortran_cfs',
                'nlow_fortran_mm',
                'nlow_censored_fortran_mm',
            ):
                rec[col] = np.nan
            for rp_int in [2, 5, 10, 25, 50, 100, 200, 500]:
                rec[f'q{rp_int}_fortran_cfs'] = np.nan
                rec[f'q{rp_int}_fortran_mm_in_cfs'] = np.nan

        records.append(rec)

    df = pd.DataFrame(records)
    csv_path = output_dir / 'caravan_benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'Wrote {len(df)}-basin master benchmark CSV to {csv_path}.')
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Caravan USGS benchmark runner."""
    parser = argparse.ArgumentParser(
        description=(
            'Run the Caravan USGS Bulletin 17C benchmark comparing Python '
            '`return_periods` against compiled USGS Fortran (`peakfqr`) and '
            'CRAN R (`MGBT`).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SETUP_INSTRUCTIONS,
    )
    parser.add_argument(
        '--print-setup-instructions',
        action='store_true',
        help='Print detailed instructions for cloning and building the USGS R and Fortran repos and exit.',
    )
    parser.add_argument(
        '--caravan-dir',
        type=Path,
        default=None,
        help='Path to the Caravan dataset root directory.',
    )
    parser.add_argument(
        '--peaks-npz',
        type=Path,
        default=None,
        help='Optional path to cached `caravan_annual_peaks.npz` (created automatically if omitted).',
    )
    parser.add_argument(
        '--peakfqr-repo',
        type=Path,
        default=Path('/tmp/usgs_src/peakfqr'),
        help='Path to cloned `https://code.usgs.gov/water/peakfqr.git` repository.',
    )
    parser.add_argument(
        '--peakfq-so',
        type=Path,
        default=None,
        help='Optional explicit path to compiled `peakfq.so` shared library.',
    )
    parser.add_argument(
        '--r-bin',
        '--rscript-bin',
        dest='r_bin',
        type=str,
        default='R',
        help='Path or binary name for `R` executable (default: R).',
    )
    parser.add_argument(
        '--mgbt-repo',
        '--mgbt-r-lib',
        dest='mgbt_repo',
        type=Path,
        default=Path('/tmp/usgs_src/MGBT'),
        help='Path to cloned `https://github.com/cran/MGBT.git` repository.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./benchmark_output'),
        help='Directory to write benchmark results CSV and figures.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of parallel worker processes (default: 16).',
    )
    parser.add_argument(
        '--basin-timeout-sec',
        type=float,
        default=25.0,
        help='Timeout in seconds per basin before killing a hung Fortran worker (default: 25.0).',
    )
    parser.add_argument(
        '--max-basins',
        type=int,
        default=None,
        help='Optional limit on number of basins to run (for smoke testing).',
    )

    return parser.parse_args(argv)


def generate_benchmark_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate the Caravan benchmark comparison figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.titleweight': 'bold',
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 8.5,
            'figure.dpi': 300,
        }
    )

    f_ok = df[df['fortran_status'] == 'OK'].copy()
    if f_ok.empty:
        return

    # Figure 1: Outlier comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 5.0), gridspec_kw={'width_ratios': [1.3, 1.0]}
    )
    fig.subplots_adjust(
        wspace=0.28, left=0.22, right=0.97, top=0.84, bottom=0.14
    )

    r_2016_ok = df.dropna(subset=['klow_r_2016_cfs'])
    r_nb_ok = df.dropna(subset=['klow_r_nb_cfs'])
    methods = [
        'PeakFQ Fortran Censored (cfs)',
        'PeakFQ Fortran Censored (mm/day)',
        'USGS R MGBT17c (cfs)',
        'USGS R MGBT17c (mm/day)',
        'USGS R MGBT17c.verb (cfs)',
        'USGS R MGBTcohn2016 (cfs)*',
        'USGS R MGBTcohn2013 (cfs)',
        'USGS R MGBTnb (cfs)*',
        'Bulletin 17B Single GB (cfs)',
    ]
    exact_rates = [
        100.0
        * np.mean(f_ok['klow_py_cfs'] == f_ok['nlow_censored_fortran_cfs']),
        100.0 * np.mean(f_ok['klow_py_mm'] == f_ok['nlow_censored_fortran_mm']),
        100.0 * np.mean(df['klow_py_cfs'] == df['klow_r_17c_cfs']),
        100.0 * np.mean(df['klow_py_mm'] == df['klow_r_17c_mm']),
        100.0 * np.mean(df['klow_py_cfs'] == df['klow_r_verb_cfs']),
        100.0
        * np.mean(r_2016_ok['klow_py_cfs'] == r_2016_ok['klow_r_2016_cfs'])
        if not r_2016_ok.empty
        else 0.0,
        100.0 * np.mean(df['klow_py_cfs'] == df['klow_r_2013_cfs']),
        100.0 * np.mean(r_nb_ok['klow_py_cfs'] == r_nb_ok['klow_r_nb_cfs'])
        if not r_nb_ok.empty
        else 0.0,
        100.0 * np.mean(df['klow_py_cfs'] == df['klow_17b_cfs']),
    ]
    y_pos = np.arange(len(methods))[::-1]
    ax1.barh(y_pos, exact_rates, height=0.62, color='#1a73e8', alpha=0.9)
    for y, ex in zip(y_pos, exact_rates, strict=True):
        ax1.text(
            ex + 0.5,
            y,
            f'{ex:.2f}%',
            va='center',
            fontsize=8.5,
            fontweight='bold',
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods)
    ax1.set_xlim(50, 109)
    ax1.set_xlabel('Exact PILF Count (klow) Match Rate (%)')
    ax1.set_title('A. PILF Screening Match vs. Python MGBT')
    ax1.grid(axis='x', linestyle=':', alpha=0.5)

    diffs_17c = df['klow_r_17c_cfs'] - df['klow_py_cfs']
    bins = np.arange(-5.5, 6.5, 1.0)
    h17c, _ = np.histogram(np.clip(diffs_17c[diffs_17c != 0], -5, 5), bins=bins)
    x_centers = np.arange(-5, 6)
    ax2.bar(
        x_centers,
        h17c,
        width=0.55,
        color='#1a73e8',
        label='USGS R MGBT17c (cfs)',
    )
    ax2.set_xlabel('PILF Count (USGS R) - PILF Count (Python)')
    ax2.set_ylabel('Number of Discrepant Basins')
    ax2.set_title('B. Direction of PILF Count Differences')
    ax2.legend()
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    fig.savefig(figures_dir / 'fig1-outlier-comparison.png', dpi=300)
    plt.close(fig)

    # Figure 2: Quantile agreement
    f_ok['fortran_scale_err'] = (
        100.0
        * np.abs(f_ok['q100_fortran_cfs'] - f_ok['q100_fortran_mm_in_cfs'])
        / f_ok['q100_fortran_cfs']
    )
    f_clean = f_ok[f_ok['fortran_scale_err'] < 1e-4].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    fig.subplots_adjust(
        wspace=0.25, left=0.07, right=0.97, top=0.84, bottom=0.14
    )
    if not f_clean.empty:
        rel_q100 = np.maximum(
            100.0
            * np.abs(f_clean['q100_py_cfs'] - f_clean['q100_fortran_cfs'])
            / f_clean['q100_fortran_cfs'],
            1e-14,
        )
        s_all = np.sort(rel_q100)
        ax1.plot(
            s_all,
            np.linspace(0, 100, len(s_all)),
            color='#1a73e8',
            lw=2.2,
            label=f'All Unit-Consistent Basins (n = {len(f_clean):,})',
        )
    ax1.set_xscale('log')
    ax1.set_xlim(1e-13, 1e2)
    ax1.set_xlabel(
        'Relative Q100 Difference |Delta Q100| / Q100 (%) [log scale]'
    )
    ax1.set_ylabel('Cumulative Percentage of Basins (%)')
    ax1.set_title('A. Python vs. Compiled Fortran Q100 CDF')
    ax1.legend()
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    sc = ax2.scatter(
        f_ok['q100_fortran_cfs'],
        f_ok['q100_py_cfs'],
        c=(f_ok['fortran_scale_err'] >= 1e-4).astype(int),
        cmap='coolwarm',
        s=7,
        alpha=0.6,
    )
    del sc
    ax2.plot([1e-1, 1e7], [1e-1, 1e7], 'k--', lw=1.0)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Compiled USGS Fortran Q100 (cfs)')
    ax2.set_ylabel('Python GEMAFitter Q100 (cfs)')
    ax2.set_title('B. 100-Year Flood Quantile Parity')
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.savefig(figures_dir / 'fig2-quantile-agreement.png', dpi=300)
    plt.close(fig)

    # Figure 3: Non-iterative switch
    pilf = df[df['klow_py_cfs'] > 0].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    fig.subplots_adjust(
        wspace=0.25, left=0.07, right=0.97, top=0.84, bottom=0.15
    )
    rps_list = [2, 5, 10, 25, 50, 100, 200, 500]
    if not pilf.empty:
        for col_prefix, label, color in [
            (
                'simple_lp3',
                'SimpleLogPearson3Fitter (No MGBT, No EMA Loop)',
                '#d93025',
            ),
            (
                'mgbt_0step',
                'MGBT + 0-Step LP-III (Above T_PILF Only)',
                '#f29900',
            ),
            ('mgbt_1step', 'MGBT + 1-Step EMA (max_iterations=1)', '#1a73e8'),
            ('mgbt_2step', 'MGBT + 2-Step EMA (max_iterations=2)', '#1e8e3e'),
        ]:
            meds = [
                float(
                    np.median(
                        100.0
                        * np.abs(
                            pilf[f'q{r}_{col_prefix}'] - pilf[f'q{r}_py_cfs']
                        )
                        / pilf[f'q{r}_py_cfs']
                    )
                )
                for r in rps_list
            ]
            ax1.plot(
                rps_list, meds, marker='o', lw=2.0, color=color, label=label
            )
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Return Period T (Years)')
    ax1.set_ylabel('Median Relative Error vs. Full Bulletin 17C EMA (%)')
    ax1.set_title(f'A. Median Error on PILF Basins (n = {len(pilf):,})')
    ax1.legend()
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    if not pilf.empty:
        signed_lp3 = (
            100.0
            * (pilf['q100_simple_lp3'] - pilf['q100_py_cfs'])
            / pilf['q100_py_cfs']
        )
        signed_0st = (
            100.0
            * (pilf['q100_mgbt_0step'] - pilf['q100_py_cfs'])
            / pilf['q100_py_cfs']
        )
        bins_err = np.linspace(-60, 40, 51)
        ax2.hist(
            np.clip(signed_lp3, -60, 40),
            bins=bins_err,
            alpha=0.55,
            color='#d93025',
            label='SimpleLogPearson3Fitter',
        )
        ax2.hist(
            np.clip(signed_0st, -60, 40),
            bins=bins_err,
            alpha=0.65,
            color='#1a73e8',
            label='MGBT + 0-Step LP-III',
        )
    ax2.axvline(0, color='k', linestyle='--', lw=1.0)
    ax2.set_xlabel('Signed Relative Error in Q100 (%)')
    ax2.set_ylabel('Number of PILF Basins')
    ax2.set_title('B. Signed Q100 Bias on PILF Basins')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)
    fig.savefig(figures_dir / 'fig3-noniterative-switch.png', dpi=300)
    plt.close(fig)
    print(f'Saved 3 comparison benchmark plots to {figures_dir}.')


def main(argv: list[str] | None = None) -> int:
    """Entry point for running the Caravan USGS Bulletin 17C benchmark."""
    args = parse_args(argv)

    if args.print_setup_instructions:
        print(SETUP_INSTRUCTIONS)
        return 0

    if args.caravan_dir is None:
        raise ValueError(
            '--caravan-dir is required unless --print-setup-instructions is passed.'
        )

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    peaks_npz = (
        args.peaks_npz.resolve()
        if args.peaks_npz is not None
        else output_dir / 'caravan_annual_peaks.npz'
    )

    data = extract_caravan_peaks(args.caravan_dir, peaks_npz)
    total_basins = len(data['basin_ids'])
    if args.max_basins is not None:
        total_basins = min(total_basins, args.max_basins)

    peakfqr_repo = args.peakfqr_repo.resolve()
    peakfq_so = (
        args.peakfq_so.resolve()
        if args.peakfq_so is not None
        else ensure_compiled_peakfq_so(peakfqr_repo)
    )

    print(
        f'Running compiled USGS Fortran (`{peakfq_so}`) across {total_basins} basins '
        f'with {args.workers} workers ...'
    )
    fortran_rows, fortran_crashes = supervise_fortran_workers(
        peaks_npz_path=peaks_npz,
        peakfqr_repo=peakfqr_repo,
        peakfq_so_path=peakfq_so,
        r_bin=args.r_bin,
        output_dir=output_dir,
        total_basins=total_basins,
        num_workers=args.workers,
        basin_timeout_sec=args.basin_timeout_sec,
    )
    print(
        f'Fortran completed: {len(fortran_rows)}/{total_basins} succeeded, '
        f'{len(fortran_crashes)} crashed/timed out.'
    )

    print(
        f'Running CRAN R `MGBT` (`{args.r_bin}`) across {total_basins} basins '
        f'with {args.workers} workers ...'
    )
    r_rows, r_crashes = run_r_mgbt_workers(
        peaks_npz_path=peaks_npz,
        r_bin=args.r_bin,
        mgbt_repo=args.mgbt_repo.resolve(),
        output_dir=output_dir,
        total_basins=total_basins,
        num_workers=args.workers,
    )
    print(
        f'R MGBT completed: {len(r_rows)}/{total_basins} finished Stage 1, '
        f'{len(r_crashes)} crashed in MGBTcohn2016/MGBTnb.'
    )

    df = assemble_master_csv(
        peaks_npz_path=peaks_npz,
        fortran_rows=fortran_rows,
        fortran_crashes=fortran_crashes,
        r_rows=r_rows,
        r_crashes=r_crashes,
        output_dir=output_dir,
        total_basins=total_basins,
    )

    generate_benchmark_plots(df, output_dir / 'figures')

    f_ok = df[df['fortran_status'] == 'OK']
    if not f_ok.empty:
        exact_klow = float(
            np.mean(f_ok['klow_py_cfs'] == f_ok['nlow_censored_fortran_cfs'])
        )
        print(
            f'Python vs. Compiled Fortran MGBT outlier count match: '
            f'{exact_klow * 100:.4f}% ({int(np.sum(f_ok["klow_py_cfs"] == f_ok["nlow_censored_fortran_cfs"]))}/{len(f_ok)})'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
