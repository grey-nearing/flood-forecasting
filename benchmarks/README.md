# OpenHydroNet Canonical Benchmark Suite (`benchmarks/`)

> **Do you need these tools?**
> Most users training or running flood forecasting models do not need to run this benchmark suite. These command-line tools are intended for contributors and researchers verifying numerical parity against published reference datasets after modifying data extractors, catchment delineation, return period fitting, or model architectures.

The `benchmarks/` package provides standalone command-line benchmarks for evaluating every core component of `flood-forecasting` (`openhydronet`) against published reference datasets (Caravan, Caravan-MultiMet v1.1 `gs://caravan-multimet/v1.1`, HydroATLAS v1.0, and USGS Bulletin 17C `peakfqr` / `MGBT`).

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

## 2. Benchmark Evaluation Principles

Every benchmark in `benchmarks/` reports both **conditional** metrics (evaluated on valid outputs) and **unconditional** metrics (where in-coverage failures or unexpected `NaN` outputs are penalized rather than dropped):

1. **No Fallback to Reference Data:** Benchmarks never substitute reference polygons, `ref_*` attributes, or `union_mapping` fallback variables when a component fails or outputs `NaN`.
2. **Explicit `NaN` Accounting:** Every benchmark counts and reports instances where a component outputs `NaN` while the reference value is valid (`pred_nan_when_ref_valid` / `extracted_only_nan_count`).
3. **Full Cohort Retention:** All input basins in the benchmark cohort are retained in headline summary tables.
4. **No Imputation:** Missing outputs remain `NaN` and are never interpolated or filled before computing error metrics.
5. **Tail & Failure-Rate Reporting:** Reports include hard failure rates, lower-tail skill percentiles (`[Min, P1, P5, P10, P25, P50]`), and upper-tail error percentiles (`[P50, P75, P90, P95, P99, Max]`).

---

## 3. CLI Flags & Quick-Start Examples

### 3.1 `benchmark-catchment`
| Flag | Required | Description |
| :--- | :--- | :--- |
| `--dataset` | Yes | Path to benchmark Parquet dataset (`benchmark_basins_1000.parquet` or `benchmark_basins_500.parquet`). |
| `--tiles-dir` | No | Local directory containing `5° x 5°` `.npy` D8 flow-direction tiles. |
| `--workers` | No | Number of parallel worker processes (default: `8`). |
| `--snap-cells` | No | Maximum search radius in cells when snapping without an area hint (default: `5`). |
| `--no-area-hint` | No | Disable area-guided snapping (`expected_area_km2`). |
| `--output` | No | Optional path to save per-basin benchmark metrics as Parquet or CSV. |
| `--save-geometries` | No | Include `del_geometry_wkt` in `--output`. |
| `--export-redelineated-dataset` | No | Export a copy of `--dataset` with `geometry_wkt` replaced by `del_geometry_wkt` (`None` on failed delineations) for cascaded downstream benchmarks. |

```bash
benchmark-catchment \
  --dataset /path/to/benchmark_basins_500.parquet \
  --tiles-dir /path/to/tiles_5deg \
  --workers 16 \
  --save-geometries \
  --output /tmp/catchment_500_results.parquet \
  --export-redelineated-dataset /tmp/benchmark_basins_500_redelineated.parquet
```

### 3.2 `benchmark-static-extractor`
| Flag | Required | Description |
| :--- | :--- | :--- |
| `--dataset`, `-d` | Yes | Path to `benchmark_basins_500.parquet`. |
| `--gdb-path` | No | Local path to `BasinATLAS_v10.gdb`. |
| `--era5-source` | No | Climate extraction mode: `hybas` (default), `gridded`, or `none` (pure 196 HydroATLAS attributes). |
| `--era5-cache-dir` | No | Local directory containing `{continent}_climate_indices.txt` when `--era5-source hybas`. |
| `--gridded-era5-uri` | No | Path/URI to gridded ERA5-Land Zarr store when `--era5-source gridded`. |
| `--workers` | No | Parallel worker threads (default: `8`). |
| `--output-dir`, `-o` | No | Directory to write `benchmark_extracted_vs_ref.parquet` and `benchmark_variable_metrics.csv`. |

```bash
benchmark-static-extractor \
  --dataset /path/to/benchmark_basins_500.parquet \
  --gdb-path /path/to/BasinATLAS_v10.gdb \
  --era5-source hybas \
  --era5-cache-dir /path/to/era5_climate \
  --workers 16 \
  -o /tmp/static_benchmark_out
```

### 3.3 `benchmark-gridded-archive`
| Flag | Required | Description |
| :--- | :--- | :--- |
| `--product` | Yes | Gridded product to rebuild: `CPC` or `IMERG`. |
| `--start-date`, `--end-date` | Yes | Inclusive date window (`YYYY-MM-DD`). |
| `--reference-zarr` | Yes | Path or GCS URI to reference `daily_surface.zarr`. |
| `--output-dir` | Yes | Directory for CSV metrics and `benchmark_report.md`. |
| `--rebuilt-zarr` | No | Optional path to an existing rebuilt Zarr store (`--skip-rebuild`). |

```bash
benchmark-gridded-archive \
  --product CPC \
  --start-date 2020-01-01 --end-date 2020-01-31 \
  --reference-zarr gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --output-dir /tmp/cpc_archive_bench
```

### 3.4 `benchmark-timeseries-extractor`
| Flag | Required | Description |
| :--- | :--- | :--- |
| `--dataset` | Yes | Path to benchmark basin dataset (`.parquet`, `.geojson`, or `.shp`). |
| `--canonical-dir` | Yes | Root directory or GCS URI of canonical MultiMet Zarr stores (`gs://caravan-multimet/v1.1`). |
| `--archive-store` | Yes | One or more `PRODUCT=URI` mappings (`CPC=...`, `IMERG=...`, `HRES=...`). |
| `--date-windows` | Yes | One or more `START:END` date windows (`YYYY-MM-DD:YYYY-MM-DD`). |
| `--output-dir` | Yes | Output directory for CSV/Parquet tables and `benchmark_report.md`. |
| `--save-reconstructed-zarr` | No | Optional directory to save reconstructed `<PRODUCT>/timeseries.zarr` stores. |
| `--num-workers` | No | Number of parallel workers for zonal weight computation (default: `8`). |

```bash
benchmark-timeseries-extractor \
  --dataset /path/to/benchmark_basins_500.parquet \
  --canonical-dir gs://caravan-multimet/v1.1 \
  --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --date-windows 2016-01-01:2023-12-31 \
  --output-dir /tmp/timeseries_bench \
  --num-workers 8
```

### 3.5 `benchmark-return-periods`
| Flag | Required | Description |
| :--- | :--- | :--- |
| `--mode` | No | `live` (default pure-Python verification on Caravan Zarr) or `external` (full 4-way benchmark against compiled Fortran `peakfqr` and CRAN R `MGBT`). |
| `--caravan-dir` | Yes | Path to Caravan Zarr directory (`--mode live`) or Caravan NetCDF directory (`--mode external`). |
| `--output-dir` | Yes | Directory to write CSV and Markdown benchmark reports. |
| `--dataset` | No | Optional Parquet file to filter evaluated basins in `--mode live`. |
| `--peakfq-so` | Conditional | Path to compiled `peakfq.so` (required when `--mode external`). |
| `--mgbt-repo` | Conditional | Path to CRAN `MGBT` repository (required when `--mode external`). |

```bash
# Pure-Python live verification mode (no Fortran or R required)
benchmark-return-periods \
  --mode live \
  --caravan-dir /path/to/Caravan-zarr \
  --dataset /path/to/benchmark_basins_500.parquet \
  --output-dir /tmp/return_periods_live

# External USGS Fortran peakfqr + CRAN R MGBT parity mode
benchmark-return-periods \
  --mode external \
  --caravan-dir /path/to/Caravan-nc \
  --peakfq-so /path/to/peakfq.so \
  --mgbt-repo /path/to/MGBT \
  --output-dir /tmp/return_periods_external
```

### 3.6 `benchmark-model`
| Flag | Required | Description |
| :--- | :--- | :--- |
| `--mode` | No | `all` (default), `compare-forcings` (alias `forcing-sensitivity`), `architectures`, or `hot-start`. |
| `--canonical-multimet-dir` | Yes | Path to canonical MultiMet Zarr directory (alias `--canonical-dynamics-dir`). |
| `--reconstructed-multimet-dir` | Conditional | Path to reconstructed MultiMet Zarr directory (alias `--reconstructed-dynamics-dir`; required except in `--mode hot-start`). |
| `--caravan-dir` | Yes | Path to Caravan directory containing `streamflow.zarr` and `attributes.zarr` (alias `--targets-dir`). |
| `--basin-file` | Yes | Path to text file listing gauge IDs (alias `--basins-file`). |
| `--output-dir`, `-o` | Yes | Directory to write CSV and JSON benchmark outputs. |

```bash
benchmark-model \
  --mode all \
  --basin-file /path/to/basins_25.txt \
  --caravan-dir /path/to/Caravan-zarr \
  --canonical-multimet-dir /path/to/canonical_zarr \
  --reconstructed-multimet-dir /path/to/reconstructed_zarr \
  --output-dir /tmp/model_bench \
  --seq-length 180 \
  --lead-time 7 \
  --epochs 3 \
  --device cpu
```
