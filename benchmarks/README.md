# OpenHydroNet Canonical Benchmark Suite (`benchmarks/`)

The `benchmarks/` package provides standalone, reproducible command-line benchmarks for evaluating every core algorithmic component of `flood-forecasting` (`openhydronet`) against published canonical reference datasets (Caravan, Caravan-MultiMet v1.1 `gs://caravan-multimet/v1.1`, HydroATLAS v1.0, and USGS Bulletin 17C `peakfq` / `MGBT`).

> **See Also:** Full benchmark protocol, anti-masking/anti-imputation invariants, lower-tail (`[Min, P1, P5, P10, P25, P50]`) reporting rules, and canonical baseline numbers are documented in [`skills/benchmarking.md`](../skills/benchmarking.md).

---

## 1. Available Benchmark Modules & CLI Entrypoints

Installing the repository (`pip install -e .`) registers all six benchmark CLIs:

| CLI Command | Python Module | Target Package / Submodule | Reference Ground Truth |
| :--- | :--- | :--- | :--- |
| **`benchmark-catchment`** | [`benchmarks.catchment_delineation`](./catchment_delineation.py) | `multimet.catchment_delineation` | `1,200` global gauges (`benchmark_basins_1000.parquet`) & `490` Caravan gauges (`benchmark_basins_500.parquet`) |
| **`benchmark-static-extractor`** | [`benchmarks.static_extractor`](./static_extractor.py) | `multimet.static_extractor` | `210` published Caravan reference attributes (`ref_*`) across `490` Caravan basins |
| **`benchmark-gridded-archive`** | [`benchmarks.gridded_archive_builders`](./gridded_archive_builders.py) | `multimet.gridded_archive_builders` | Canonical `CPC` (`0.5°`) & `IMERG` (`0.1°`) `daily_surface.zarr` archives |
| **`benchmark-timeseries-extractor`** | [`benchmarks.timeseries_extractors`](./timeseries_extractors.py) | `multimet.timeseries_extractors` | Canonical `gs://caravan-multimet/v1.1/{CPC,IMERG,HRES}/timeseries.zarr` |
| **`benchmark-return-periods`** | [`benchmarks.return_periods`](./return_periods.py) | `return_periods` | USGS Fortran `peakfqr` v8.0 (`EMA`) & CRAN R `MGBT` v1.1.6 across `11,213` Caravan basins |
| **`benchmark-model`** | [`benchmarks.model`](./model.py) | `model` | Canonical vs. reconstructed MultiMet forcings, model architectures (`MeanEmbeddingForecastLSTM`, `HandoffForecastLSTM`), and cold-start vs. hot-start state handoff |

 Auxiliary dataset-preparation scripts live in [`benchmarks/tools/`](./tools/):
- [`benchmarks/tools/build_benchmark_dataset.py`](./tools/build_benchmark_dataset.py): Builds a stratified multi-continent reference benchmark Parquet file (`geometry_wkt` and `reference_area_km2`) from reference shapefiles and coordinate tables.

---

## 2. Hard Rules for All Benchmarks

Every benchmark in `benchmarks/` enforces five strict data-integrity rules (detailed in [`skills/benchmarking.md`](../skills/benchmarking.md)):

1. **Zero Fallback to Canonical Data:** Never substitute canonical polygons, `ref_*` attributes, or `union_mapping` fallback variables when a component fails or outputs `NaN`.
2. **Zero Silent `NaN` Masking:** Explicitly count and report every instance where a component produces `NaN` while the reference value is valid (`pred_nan_when_ref_valid` / `extracted_only_nan_count`).
3. **Zero Silent Outlier Dropping:** Retain all input cohort basins in headline summary tables (never drop failed delineations or revised upstream shapefiles from primary metrics).
4. **Zero Imputation or Clamping:** Never interpolate, forward/backward-fill, or clamp missing outputs prior to computing benchmark errors.
5. **Mandatory Lower-Tail & Failure-Rate Reporting:** Report hard failure rates, lower-tail skill percentiles (`[Min, P1, P5, P10, P25, P50]`), upper-tail error percentiles (`[P50, P75, P90, P95, P99, Max]`), and worst-case failure diagnostics alongside median/mean scores.

---

## 3. Quick-Start Examples

```bash
# 1. Catchment Delineation Benchmark (90m HydroSHEDS D8 flow directions)
benchmark-catchment \
  --dataset /path/to/benchmark_basins_1000.parquet \
  --tiles-dir /path/to/tiles_5deg \
  --workers 16 \
  --output /tmp/catchment_results.parquet

# 2. Static Attribute Extractor Benchmark (HydroATLAS Level 12 + ERA5 climate)
benchmark-static-extractor \
  --dataset /path/to/benchmark_basins_500.parquet \
  --gdb-path /path/to/BasinATLAS_v10.gdb \
  --era5-source hybas \
  --era5-cache-dir /path/to/era5_climate \
  --workers 16 \
  -o /tmp/static_benchmark_out

# 3. Gridded Archive Builder Parity Benchmark (CPC / IMERG)
benchmark-gridded-archive \
  --product CPC \
  --start-date 2020-01-01 --end-date 2020-01-31 \
  --reference-zarr gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --output-dir /tmp/cpc_archive_bench

# 4. MultiMet Catchment Timeseries Reconstruction Benchmark (CPC, IMERG, HRES)
benchmark-timeseries-extractor \
  --dataset /path/to/benchmark_basins_500.parquet \
  --canonical-dir gs://caravan-multimet/v1.1 \
  --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --date-windows 2016-01-01:2023-12-31 \
  --output-dir /tmp/timeseries_bench \
  --num-workers 8

# 5. Return Period Calculator Benchmark (USGS Bulletin 17C MGBT & EMA)
benchmark-return-periods \
  --caravan-dir /path/to/Caravan-nc \
  --peakfq-so /path/to/peakfq.so \
  --output-dir /tmp/return_periods_bench

# 6. Core Forecasting Model Benchmark (Forcing Sensitivity, Architectures, Hot-Start)
benchmark-model \
  --mode all \
  --basins-file /path/to/basins_25.txt \
  --statics-dir /path/to/Caravan-zarr \
  --targets-dir /path/to/Caravan-zarr \
  --canonical-dynamics-dir /path/to/canonical_zarr \
  --reconstructed-dynamics-dir /path/to/reconstructed_zarr \
  --output-dir /tmp/model_bench \
  --seq-length 180 \
  --lead-time 7 \
  --epochs 3 \
  --device cpu
```
