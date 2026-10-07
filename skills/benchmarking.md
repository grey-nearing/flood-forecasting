---
name: benchmarking
description: >-
  End-to-end canonical benchmarking protocol, anti-masking/anti-imputation/zero-fallback
  hard rules, lower-tail and failure-rate reporting standards, CLI runbooks for
  the root benchmarks/ package, and canonical baseline numbers across all core
  flood-forecasting components. Use whenever running, adding, updating, or
  reviewing canonical benchmarks.
---

# Canonical Benchmarking Protocol & Baseline Reference (`skills/benchmarking.md`)

This skill defines the mandatory rules, CLI runbooks, lower-tail/failure-rate reporting standards, and canonical baseline numbers for benchmarking all core components of the `flood-forecasting` (`openhydronet`) repository in the root **`benchmarks/`** package against the canonical Caravan and Caravan-MultiMet v1.1 datasets (`gs://caravan-multimet/v1.1` and `gs://open-multimet/`).

> **Mandatory Requirement for New Submodules (`benchmarks/`):** Every new submodule, subpackage, or major algorithmic component added to the repository must include **both** automated unit/integration tests in `<package>/tests/` **and** a manual, comprehensive canonical benchmark in `benchmarks/<component>.py` (with unit tests in `benchmarks/tests/` and CLI registration in `setup.py`). Features such as user interfaces (UI), interactive frontends, or visual-only helpers where quantitative canonical benchmarking does not make sense or is not possible are exempt from adding a `benchmarks/` module.

---

## 1. Non-Negotiable Anti-Masking, Anti-Imputation & Failure-Reporting Hard Rules

> **CRITICAL HARD RULE:** Under no circumstances may any benchmark script, extractor, delineator, archive builder, return-period fitter, or model evaluation harness mask bad data, impute missing values, drop failed/outlier basins from headline metrics, or fall back to canonical/reference data when a component fails.

Every benchmark run and report MUST strictly enforce the following **five invariants**:

1. **Zero Fallback to Reference / Canonical Data:**
   - Components under test must **never** read or fall back to canonical ground-truth values (`ref_*` attribute columns, canonical reference polygons, canonical Zarr stores, or `union_mapping` to `ERA5_LAND`) when an extraction, delineation, or model prediction fails or outputs `NaN`.
   - In `benchmarks/model.py`, `_assert_no_fallback_or_imputation(cfg)` must verify `not cfg.union_mapping` and `not cfg.tester_skip_obs_all_nan` before running any benchmark.
   - In end-to-end cascaded benchmarks (`Catchment Delineation -> Static Attributes` and `Catchment Delineation -> Zonal Timeseries`), basins where delineation failed (`status != 'SUCCESS'`) must receive `geometry_wkt = None` (empty geometry) rather than substituting the canonical reference polygon.
2. **Zero Silent `NaN` Masking of Predictions (`pred_nan_when_ref_valid` / `extracted_only_nan`):**
   - A pairwise mask such as `mask = ~(np.isnan(y_true) | np.isnan(y_pred))` is **never** sufficient on its own, because it silently drops points where the canonical reference `y_true` is valid (`~np.isnan(y_true)`) and the tool/model failed (`np.isnan(y_pred)`).
   - Every benchmark table MUST explicitly count and prominently report:
     - `pred_nan_when_ref_valid = int((~np.isnan(y_true) & np.isnan(y_pred)).sum())` (or `extracted_only_nan_count` / `extracted_only_nan_pct` in the 4-way mask confusion matrix: `both_valid`, `both_nan`, `extracted_only_nan`, `canonical_only_nan`).
     - Both **Conditional Metrics** (on `both_valid` points) and **Unconditional / Penalized Metrics** (where `pred_nan_when_ref_valid` or in-coverage failures are penalized as `0.0` correlation / `0.0` IoU / `wrong` categorical classification).
3. **Zero Silent Filtering of Outlier or Failed Basins from Headline Tables:**
   - Every basin in the input benchmark cohort (`1,200` in `benchmark_basins_1000.parquet`, `490` in `benchmark_basins_500.parquet`, `445` canonical-matched in `gs://caravan-multimet/v1.1`) must be retained in the primary headline tables.
   - Do **not** filter out basins whose upstream shapefiles were revised (`camelscl_5421001` and `camelscl_4714001`) or basins where `DemDelineator` raises `CatchmentAreaMismatchError` (`AREA_HINT_FAILURE`) from headline metrics. Diagnostic subset tables (`unrevised_geometry` vs. `revised_geometry`) may only be shown as secondary attribution appendices alongside the unmasked `all_matched` primary table.
4. **Zero Imputation, Interpolation, or Clamping:**
   - Missing or `NaN` outputs from any component must remain `NaN` and never be interpolated, forward/backward-filled, or replaced with climatology/defaults prior to metric calculation.
5. **Mandatory Lower-Tail (`[Min, P1, P5, P10, P25, P50]`) & Failure-Rate Threshold Reporting:**
   - Reporting only `Median` or `Mean` metrics hides pipeline failure modes. Every benchmark report MUST include:
     - Hard failure rate (`%` of basins/attributes/grid cells failing or producing unexpected `NaN`s).
     - Lower-tail skill percentiles (`Min`, `P1`, `P5`, `P10`, `P25`, `P50`) for similarity metrics (`IoU`, `Dice`, `Pearson r`, `NSE`, `KGE`, `Categorical Accuracy`).
     - Upper-tail error percentiles (`P50`, `P75`, `P90`, `P95`, `P99`, `Max`) and exceedance rates (`% > 5%`, `% > 10%`, `% > 20%`, `% > 50%`) for error metrics (`Area Error %`, `Attribute Rel Error %`, `MAE`, `RMSE`).
     - Explicit roster of the worst-performing basins/attributes and their physical/data root causes.

---

## 2. Scope, Exclusions & Canonical Data Locations

### 2.1 Explicit Exclusions
1. **No `ERA5_LAND` Gridded Archive Extraction:** Do not run `ERA5_LAND` gridded archive or zonal timeseries extraction benchmarks until `gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr` is built. For static attributes, benchmark both `--era5-source none` (`196` pure HydroATLAS Level 12 attributes) and `--era5-source hybas` (`210` attributes using pre-computed continental sub-basin tables).
2. **No `GRAPHCAST`:** Exclude `GRAPHCAST` from all timeseries extractor and model benchmarks.

### 2.2 Canonical Cloud & Staged Local Paths
- **Canonical Caravan-MultiMet v1.1 Zarr Stores (Public, `token="anon"`):**
   - `gs://caravan-multimet/v1.1/CPC/timeseries.zarr` (`cpc_precipitation`, `1979-01-01 .. 2024-07-31`)
   - `gs://caravan-multimet/v1.1/IMERG/timeseries.zarr` (`imerg_precipitation`, `2000-06-01 .. 2024-10-31`)
   - `gs://caravan-multimet/v1.1/HRES/timeseries.zarr` (7 variables, `lead_time: 1..10`, `2016-01-01 .. 2024-09-30`)
- **Gridded Meteorological Archives (Internal GCS, `token="google_default"`):**
   - `gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr` (`0.5°`, `360 x 720`)
   - `gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr` (`0.1°`, `1800 x 3600`)
   - `gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr` (`0.1°`, `10 x 1801 x 3600`)
- **Staged Benchmark Datasets & Ancillary Files (`gsnearing-large-1` / `gs://open-multimet/ancillary-data/`):**
   - `1,200`-Basin Global Delineation Benchmark: `/usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_1000.parquet` (`200` basins per continent across 6 continents, stratified across 5 size tiers).
   - `490`-Basin Caravan Multi-Component Benchmark: `/usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_500.parquet` (`70` basins per Caravan dataset across `camels`, `camelsaus`, `camelsbr`, `camelscl`, `camelsgb`, `hysets`, `lamah`, with `geometry_wkt`, `ref_area_km2`, and `210` `ref_*` attributes).
   - HydroSHEDS 3-arcsec (`90m`) D8 Flow-Direction Tiles: `/usr/local/google/home/gsnearing/data/DEMs/tiles_5deg/` (`gs://open-multimet/ancillary-data/dems/hydrosheds_dir_3s_tiles_5deg/`).
   - HydroATLAS Level 12 Geodatabase: `<CACHE_DIR>/hydroatlas/BasinATLAS_v10.gdb` (`gs://open-multimet/ancillary-data/hydroatlas/BasinATLAS_v10.gdb/`).
   - HydroATLAS Pre-Aggregated ERA5 Climate Tables: `<CACHE_DIR>/era5_climate/` (`gs://open-multimet/ancillary-data/hydroatlas/era5_climate/`).
   - Caravan Streamflow & Static Zarr Stores: `/usr/local/google/home/gsnearing/Projects/caravan_data/Caravan-zarr/` (`streamflow.zarr`, `attributes.zarr`).

### 2.3 Post-ERA5-Land Archive Checklist & Completed Code Updates (Tracked in Issue #46)

Once `gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr` and the basin-clipped `ERA5_LAND/timeseries.zarr` stores are finalized, complete the post-archive data regeneration steps below (the required code fixes in `multimet/` are already implemented):

1. **Completed Code Fixes (`multimet/utils/climate.py` & `multimet/static_extractor/`):**
   - **Fixed `W/m² -> MJ/m²/day` unit factor in `calculate_fao_pm_pet` (`multimet/utils/climate.py`):** Uses `* 86400.0 / 1e6` (`86,400` seconds/day) by default (`radiation_units="W/m^2"`), reproducing canonical `era5land_potential_evaporation_FAO_PENMAN_MONTEITH` to `Pearson r = 0.999999997` (`max_abs_diff = 0.0014 mm/day`).
   - **Added `era5land_*` column support & on-the-fly `FAO_PM` PET in `StaticAttributesExtractor.extract_attributes_for_polygon` (`multimet/static_extractor/extractor.py`):** Recognizes `era5land_*` columns, applies `np.abs(...)` to PET series, and computes `FAO_PM` PET on-the-fly if `pet_fao` is absent from `timeseries_df`.
   - **Added on-the-fly `FAO_PM` PET fallback in `ERA5GriddedExtractor` (`multimet/static_extractor/climate.py`):** Computes daily `pet_fao` on-the-fly from the 7 extracted daily surface variables (`t2m`, `d2m`, `sp`, `ssr`, `str`, `u10`, `v10`) via `calculate_fao_pm_pet` whenever the gridded archive does not store a pre-baked `FAO_PENMAN_MONTEITH` variable.
2. **Post-Archive Ancillary & Dataset Regeneration Steps (Issue #46):**
   - **Step A — Regenerate the 9 continental HydroATLAS Level 12 climate tables (`gs://open-multimet/ancillary-data/hydroatlas/era5_climate/{continent}_climate_indices.txt`):** Run `ERA5GriddedExtractor.extract_climate_metrics_for_polygons_batch` (`baseline_years=(1981, 2020)`) over all HydroATLAS Level 12 sub-basins (`HYBAS_ID`) for `af`, `ar`, `as`, `au`, `eu`, `gr`, `na`, `sa`, `si`. Writing the full 18-key output of `compute_caravan_climate_metrics` into `{continent}_climate_indices.txt` populates the 4 missing `*_ERA5_LAND` keys (`pet_mean_ERA5_LAND`, `aridity_ERA5_LAND`, `moisture_index_ERA5_LAND`, `seasonality_ERA5_LAND`) and replaces the legacy unclipped Knoben `seasonality`/`moisture_index` values so `--era5-source hybas` works accurately for arbitrary user polygons without requiring the gridded Zarr store.
   - **Step B — Update canonical `attributes.zarr` climate columns from basin-clipped `ERA5_LAND/timeseries.zarr`:** For canonical Caravan-MultiMet basins, compute the 14 climate attributes directly from `ERA5_LAND/timeseries.zarr` (`1981-01-01 .. 2020-12-31`) via `compute_caravan_climate_metrics` (verified to match canonical Caravan v1.1 within `0.0001–0.004`).
   - **Step C — Add `ERA5_LAND` to Phase 2, Phase 3A/3B, and Phase 5 benchmark runs:** Re-run `benchmark-static-extractor` (`--era5-source hybas` and `--era5-source gridded`), `benchmark-gridded-archive --product ERA5_LAND`, and `benchmark-timeseries-extractor --archive-store ERA5_LAND=gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr`.

---

## 3. Component CLI Entrypoints & Standard Runbook Commands (`benchmarks/`)

All six component benchmark harnesses live in the root **`benchmarks/`** package, are registered as console scripts in `setup.py`, and obey the **Zero `try`/`except`** rule (`skills/algorithm-rules-and-norms.md`):

| # | CLI Entrypoint | Module (`benchmarks/`) | Component Under Test |
| :--- | :--- | :--- | :--- |
| **1** | `benchmark-catchment` | `benchmarks.catchment_delineation` | 90m (`3-arcsec`) D8 watershed delineation & pour-point snapping (`DemDelineator`) |
| **2** | `benchmark-static-extractor` | `benchmarks.static_extractor` | HydroATLAS Level 12 & Caravan static catchment attribute extraction (`StaticAttributesExtractor`) |
| **3a** | `benchmark-gridded-archive` | `benchmarks.gridded_archive_builders` | Gridded Zarr archive builders (`CPCArchiveBuilder`, `IMERGArchiveBuilder`) vs. reference archives |
| **3b** | `benchmark-timeseries-extractor` | `benchmarks.timeseries_extractors` | Spherical cosine-latitude `ZonalWeightMatrix` & MultiMet catchment timeseries extractors (`CPC`, `IMERG`, `HRES`) |
| **4** | `benchmark-return-periods` | `benchmarks.return_periods` | USGS Bulletin 17C `MultipleGrubbsBeckTester` (`MGBT`) & `GEMAFitter` (`EMA` LP-III) |
| **5** | `benchmark-model` | `benchmarks.model` | Core deep learning forecasting models (`MeanEmbeddingForecastLSTM`, `HandoffForecastLSTM`), forcing sensitivity, and hot-start state handoff |

### 3.1 Phase 1: Catchment Delineation (`benchmark-catchment`)
```bash
# Run 1A: 1,200 Global Basins WITH Area Hint
python -m benchmarks.catchment_delineation \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_1000.parquet \
  --tiles-dir /usr/local/google/home/gsnearing/data/DEMs/tiles_5deg \
  --workers 16 \
  --output <OUT_DIR>/catchment_1000_with_hint.parquet

# Run 1B: 1,200 Global Basins WITHOUT Area Hint (Blind Coordinate Snapping)
python -m benchmarks.catchment_delineation \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_1000.parquet \
  --tiles-dir /usr/local/google/home/gsnearing/data/DEMs/tiles_5deg \
  --no-area-hint \
  --workers 16 \
  --output <OUT_DIR>/catchment_1000_no_hint.parquet

# Run 1C: 490 Caravan Basins WITH Area Hint & Saved Re-Delineated Geometries (for Cascaded Benchmarks)
python -m benchmarks.catchment_delineation \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_500.parquet \
  --tiles-dir /usr/local/google/home/gsnearing/data/DEMs/tiles_5deg \
  --save-geometries \
  --workers 16 \
  --output <OUT_DIR>/catchment_500_with_geom.parquet
```

### 3.2 Phase 2: Static Attribute Extractor (`benchmark-static-extractor`)
```bash
# Run 2A-HydroATLAS: Pure 196 HydroATLAS Level 12 Attributes on All 490 Canonical Polygons
python -m benchmarks.static_extractor \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_500.parquet \
  --gdb-path <CACHE_DIR>/hydroatlas/BasinATLAS_v10.gdb \
  --era5-source none \
  --workers 16 \
  -o <OUT_DIR>/static_500_canonical_none

# Run 2A: Full 210 Attributes (HydroATLAS + Pre-Aggregated hybas ERA5) on All 490 Canonical Polygons
python -m benchmarks.static_extractor \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_500.parquet \
  --gdb-path <CACHE_DIR>/hydroatlas/BasinATLAS_v10.gdb \
  --era5-source hybas \
  --era5-cache-dir <CACHE_DIR>/era5_climate \
  --workers 16 \
  -o <OUT_DIR>/static_500_canonical_hybas

# Run 2C (Unconditional All-490 Cascade): Re-Delineated Polygons from Run 1C (with None on failed delineations)
python -m benchmarks.static_extractor \
  --dataset <OUT_DIR>/benchmark_basins_500_redelineated_all490.parquet \
  --gdb-path <CACHE_DIR>/hydroatlas/BasinATLAS_v10.gdb \
  --era5-source hybas \
  --era5-cache-dir <CACHE_DIR>/era5_climate \
  --workers 16 \
  -o <OUT_DIR>/static_500_redelineated_all490_hybas
```

### 3.3 Phase 3A & 3B & 3C: Gridded Archive Builders & MultiMet Timeseries Reconstruction
```bash
# Phase 3A: Spot-check rebuilt CPC & IMERG gridded archives against gs://open-multimet/gridded-data-archives/
python -m benchmarks.gridded_archive_builders \
  --product CPC \
  --start-date 2020-01-01 --end-date 2020-01-31 \
  --reference-zarr gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --output-dir <OUT_DIR>/gridded_archive_cpc

python -m benchmarks.gridded_archive_builders \
  --product IMERG \
  --start-date 2024-01-01 --end-date 2024-01-03 \
  --reference-zarr gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
  --output-dir <OUT_DIR>/gridded_archive_imerg

# Phase 3B: Reconstruct Canonical Caravan-MultiMet v1.1 Timeseries (CPC, IMERG, HRES across all 3 upstream HRES tiers)
python -m benchmarks.timeseries_extractors \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_500.parquet \
  --canonical-dir gs://caravan-multimet/v1.1 \
  --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --date-windows 1985-01-01:1985-12-31 1995-01-01:1995-12-31 2005-01-01:2005-12-31 2016-01-01:2023-12-31 \
  --output-dir <OUT_DIR>/3b_cpc \
  --num-workers 8

python -m benchmarks.timeseries_extractors \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_500.parquet \
  --canonical-dir gs://caravan-multimet/v1.1 \
  --archive-store IMERG=gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
  --date-windows 2005-06-01:2005-06-15 2010-01-01:2010-01-15 2015-07-01:2015-07-30 2020-01-01:2020-01-31 2022-05-01:2022-06-15 2024-01-01:2024-01-15 \
  --output-dir <OUT_DIR>/3b_imerg \
  --num-workers 8

python -m benchmarks.timeseries_extractors \
  --dataset /usr/local/google/home/gsnearing/ancillary-data/benchmarks/benchmark_basins_500.parquet \
  --canonical-dir gs://caravan-multimet/v1.1 \
  --archive-store HRES=gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr \
  --date-windows 2018-06-01:2018-06-10 2020-01-01:2020-01-10 2020-06-01:2020-06-10 2022-05-01:2022-05-15 2023-03-01:2023-03-10 2023-06-01:2023-06-10 2023-09-01:2023-09-10 2024-03-01:2024-03-10 \
  --output-dir <OUT_DIR>/3b_hres \
  --num-workers 8

# Phase 3C (Unconditional All-490 Cascade): Re-Delineated Polygons -> Zonal Timeseries
python -m benchmarks.timeseries_extractors \
  --dataset <OUT_DIR>/benchmark_basins_500_redelineated_all490.parquet \
  --canonical-dir gs://caravan-multimet/v1.1 \
  --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --archive-store IMERG=gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
  --archive-store HRES=gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr \
  --date-windows 2020-06-01:2020-06-15 \
  --output-dir <OUT_DIR>/3c_redelineated_cascade_all490 \
  --num-workers 8
```

### 3.4 Phase 4: Return Period Calculator (`benchmark-return-periods`)
```bash
# Full USGS Fortran peakfqr v8.0 & CRAN R MGBT v1.1.6 Parity Benchmark
python -m benchmarks.return_periods \
  --caravan-dir /usr/local/google/home/gsnearing/Projects/caravan_data/Caravan-nc \
  --peakfq-so /tmp/peakfqr/src/peakfq.so \
  --output-dir <OUT_DIR>/return_periods_usgs
```

### 3.5 Phase 5: Core Forecasting Model (`benchmark-model`)
```bash
# Run compare_forcings, benchmark_architectures, and benchmark_hot_start (NO ERA5_LAND, NO GRAPHCAST)
python -m benchmarks.model \
  --mode all \
  --basins-file <STAGED_DIR>/basins_25.txt \
  --statics-dir /usr/local/google/home/gsnearing/Projects/caravan_data/Caravan-zarr \
  --targets-dir /usr/local/google/home/gsnearing/Projects/caravan_data/Caravan-zarr \
  --canonical-dynamics-dir <STAGED_DIR>/canonical_zarr \
  --reconstructed-dynamics-dir <STAGED_DIR>/reconstructed_zarr \
  --output-dir <OUT_DIR>/model_benchmark \
  --seq-length 180 \
  --lead-time 7 \
  --epochs 3 \
  --device cpu
```

---

## 4. Canonical Baseline Metrics, Lower Tails & Failure-Mode Taxonomy

### 4.1 Phase 1 Baseline — Catchment Delineation (`benchmark-catchment`)

| Metric / Lower-Tail Distribution | Run 1A: `1,200` Global (With Area Hint) | Run 1B: `1,200` Global (No Area Hint) | Run 1C: `490` Caravan (With Area Hint) |
| :--- | ---: | ---: | ---: |
| **Total / In-Coverage / Out-of-Coverage (`>60°N`)** | `1,200` / `1,128` / `72` (`6.00%`) | `1,200` / `1,128` / `72` (`6.00%`) | `490` / `489` / `1` (`0.20%`) |
| **In-Coverage Hard Rejection (`AREA_HINT_FAILURE`, `IoU=0.0`)** | `1 / 1,128` (**`0.09%`**) | `0 / 1,128` (`0.00%`) | `8 / 489` (**`1.64%`**) |
| **In-Coverage `IoU < 0.10` / `IoU < 0.50` / `IoU < 0.80` (Uncond.)** | **`0.71%` / `0.89%` / `2.66%`** | **`5.67%` / `6.74%` / `8.07%`** | **`2.45%` / `3.89%` / `12.47%`** |
| **Unconditional In-Coverage IoU `[Min, P1, P5, P10, P25, P50]`** | **`[0.000, 0.594, 0.874, 0.915, 0.953, 0.976]`** | **`[0.000, 0.000, 0.030, 0.873, 0.946, 0.974]`** | **`[0.000, 0.000, 0.628, 0.766, 0.901, 0.956]`** |
| **Unconditional In-Coverage Mean IoU / Mean Dice** | **`0.953` / `0.972`** | `0.902` / `0.921` | **`0.895` / `0.928`** |
| **Conditional `SUCCESS` Area Error `[P50, P75, P90, P95, P99, Max]`** | `[1.8%, 5.3%, 11.8%, 16.8%, 32.6%, 49.9%]` | `[2.1%, 6.9%, 17.7%, 97.2%, 222.7%, 7020.3%]` | `[1.1%, 4.5%, 14.1%, 27.6%, 43.6%, 49.9%]` |

- **Known Failure Modes:**
  1. **HydroSHEDS `60°N` Boundary (`OUT_OF_COVERAGE`):** Gauges north of `60°N` or Scandinavian/Canadian rivers whose headwaters cross `60°N`.
  2. **Micro-Catchments (`<100 km²`, `1_micro`):** `28.6%` (`Run 1A`) to `30.5%` (`Run 1C`) have `IoU < 0.80` (`P10 IoU = 0.280–0.638`) because multiple headwater creeks within the 80-cell (`~7.2 km`) snapping radius have similar drainage areas (`±50%`).
  3. **Canal / Alluvial / Low-Gradient Fenland Gauges (`AREA_HINT_FAILURE`):** `1.64%` of Caravan gauges (e.g., `hysets_09520700` Colorado River canal, `camelsgb_33035` East Anglian Fens) have no D8 channel matching `expected_area_km2` within `80` cells and deliberately raise `CatchmentAreaMismatchError` rather than returning a `99.9%`-wrong floodplain ditch (which occurs in `8.8%` of macro rivers when area hints are disabled in `Run 1B`).

---

### 4.2 Phase 2 Baseline — Static Attribute Extractor (`benchmark-static-extractor`)

| Metric / Lower-Tail Distribution | Run 2A-HydroATLAS (`196` Attrs, `none`) | Run 2A Full (`210` Attrs, `hybas`) | Run 2C Unconditional Cascade (`490` Re-Delineated, `hybas`) |
| :--- | ---: | ---: | ---: |
| **Evaluated Basins / Failed Geometries (`None`)** | `490 / 0` (`0.00%`) | `490 / 0` (`0.00%`) | `481 / 9` (**`1.84%` failed geoms**) |
| **`pred_nan_when_ref_valid` Count (`%` of cells)** | **`0 / 96,040` (`0.00%`)** | **`1,960 / 102,900` (`1.90%`, 4 `*_ERA5_LAND` attrs)** | **`3,724 / 102,900` (`3.62%`, 4 attrs + 9 failed geoms)** |
| **Continuous Attr Pearson $r$ (Valid) `[Min, P1, P5, P10, P25, P50]`** | `[0.7556, 0.9791, 0.9926, 0.9963, 0.9991, 0.9999]` | `[0.0443, 0.4228, 0.9776, 0.9949, 0.9989, 0.9998]` | `[0.0441, 0.4228, 0.9739, 0.9935, 0.9978, 0.9996]` |
| **Continuous Attr Pearson $r$ (Penalized `NaN=0`) `[Min, P1, P5, P10, P25, P50]`** | `[0.0000, 0.6422, 0.9886, 0.9959, 0.9990, 0.9999]` | `[0.0000, 0.0000, 0.9068, 0.9937, 0.9988, 0.9998]` | `[0.0000, 0.0000, 0.8920, 0.9922, 0.9977, 0.9995]` |
| **Continuous Mean Pearson $r$ (Valid / Penalized `NaN=0`)** | **`0.9973` / `0.9865`** | **`0.9825` / `0.9530`** | **`0.9797` / `0.9349` (in-cov)** |
| **Per-Basin Median Attr Rel Error `[P50, P75, P90, P95, P99, Max]`** | `[0.01%, 0.04%, 0.08%, 0.12%, 0.51%, 33.1%]` | `[0.01%, 0.05%, 0.09%, 0.13%, 0.54%, 33.4%]` | `[0.15%, 0.42%, 0.91%, 1.77%, 26.3%, NaN (9 failed)]` |
| **Categorical Majority Accuracy (Conditional / Unconditional `NaN=wrong`)** | **`99.55%` / `99.55%`** | **`99.55%` / `99.55%`** | **`99.23%` / `97.41%` (`97.61%` in-cov)** |

- **Known Failure Modes:**
  1. **`*_ERA5_LAND` PET Columns Without `--gridded-era5-uri`:** `pet_mean_ERA5_LAND`, `aridity_ERA5_LAND`, `moisture_index_ERA5_LAND`, and `seasonality_ERA5_LAND` are `100% NaN` under `--era5-source hybas` because the precomputed `{continent}_climate_indices.txt` JSONL files only store 10 keys (`p_mean`, `pet_mean`, `aridity`, `frac_snow`, `moisture_index`, `seasonality`, `high_prec_freq`, `high_prec_dur`, `low_prec_freq`, `low_prec_dur`) and omit the 4 `*_ERA5_LAND` keys (see Subsection 2.3 Step A).
  2. **`*_FAO_PM` Columns in Pre-Staged `hybas` Tables:** `seasonality_FAO_PM` ($r=0.044$), `moisture_index_FAO_PM` ($r=0.180$), `pet_mean_FAO_PM` ($r=0.441$), and `aridity_FAO_PM` ($r=0.648$) diverge from canonical Caravan because the pre-staged `{continent}_climate_indices.txt` files were generated without clipping negative winter monthly `PET` before computing Knoben et al. (2018) $1 - \text{PET}/P$ (`59.6%` of records in `na_climate_indices.txt` have `seasonality > 2.0`). By contrast, `compute_caravan_climate_metrics` in `multimet/utils/climate.py` matches canonical Caravan v1.1 within `0.0001–0.004` across all 14 climate attributes when fed `1981–2020` `ERA5_LAND` daily series (see Subsection 2.3).
  3. **Sum-Aggregated Anthropogenic Attribute (`gdp_ud_ssu`):** Only HydroATLAS attribute with $r < 0.96$ ($r = 0.7556$, Spearman $\rho = 0.9891$, `Median Rel Error = 0.29%`) due to planar vs. WGS84 geodesic boundary-overlap weighting on a single metropolitan outlier basin.

---

### 4.3 Phase 3A, 3B & 3C Baseline — Gridded Archives & MultiMet Timeseries Reconstruction

- **Phase 3A Gridded Archive Builders (`benchmark-gridded-archive`):**
  - `CPC` (`2020-01-01 .. 2020-01-31`, `8,035,200` cells) and `IMERG` (`2024-01-01 .. 2024-01-03`, `19,440,000` cells): **`0` `Rebuilt-Only NaN` cells (`0.000000%`), `Pearson r = 1.00000000`, `MAE = 0.0`, `Max Abs Error = 0.0` (`100.0000%` exact cell-by-cell match)**.
- **Phase 3B Canonical Polygon Timeseries Reconstruction (`benchmark-timeseries-extractor`, All `445` Matched Basins Unmasked):**

| Product / Variable (`N = 445` Basins) | `Ext-Only NaN` Count (`%`) | Overall Pearson $r$ | Per-Basin Pearson $r$ `[Min, P1, P5, P10, P50]` | Per-Basin NSE `[Min, P1, P5, P10, P50]` | Per-Basin KGE `[Min, P1, P5, P10, P50]` | MAE / RMSE |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: |
| **`CPC` `cpc_precipitation`** (`1.79M` pts) | **`2` (`0.0001%`)** | `0.999812` | `[0.9666, 0.9978, 0.9994, 0.9996, 1.0000]` | `[0.9247, 0.9956, 0.9988, 0.9992, 1.0000]` | `[0.8628, 0.9895, 0.9965, 0.9979, 0.9996]` | `5.69e-03` / `1.14e-01` |
| **`IMERG` `imerg_precipitation`** (`67.6k` pts) | **`0` (`0.0000%`)** | `0.996841` | `[0.8413, 0.9672, 0.9876, 0.9913, 0.9977]` | `[0.5082, 0.9348, 0.9746, 0.9816, 0.9951]` | `[0.2990, 0.9158, 0.9581, 0.9674, 0.9891]` | `1.77e-01` / `6.80e-01` |
| **`HRES` `hres_total_precipitation`** (`356k` pts) | **`0` (`0.0000%`)** | `0.999764` | `[0.8794, 0.9998, 1.0000, 1.0000, 1.0000]` | `[0.7206, 0.9995, 0.9999, 0.9999, 1.0000]` | `[0.5219, 0.9853, 0.9955, 0.9973, 0.9995]` | `1.29e-02` / `1.64e-01` |
| **`HRES` `hres_temperature_2m`** (`356k` pts) | **`0` (`0.0000%`)** | `0.998862` | `[0.9779, 1.0000, 1.0000, 1.0000, 1.0000]` | `[-0.9221, 0.9994, 1.0000, 1.0000, 1.0000]` | `[0.4459, 0.9713, 0.9932, 0.9968, 0.9997]` | `3.99e-02` / `4.79e-01` |
| **`HRES` `hres_surface_pressure`** (`356k` pts) | **`0` (`0.0000%`)** | `0.995869` | `[0.4095, 0.9999, 1.0000, 1.0000, 1.0000]` | `[-2662.7, 0.5087, 0.9724, 0.9921, 0.9999]` | `[0.3709, 0.9943, 0.9984, 0.9991, 0.9999]` | `6.42e-02` / `7.65e-01` |
| **`HRES` `hres_surface_net_solar_radiation`** (`356k` pts) | **`0` (`0.0000%`)** | `0.999733` | `[0.9788, 1.0000, 1.0000, 1.0000, 1.0000]` | `[0.8785, 0.9999, 1.0000, 1.0000, 1.0000]` | `[0.7930, 0.9962, 0.9985, 0.9993, 0.9999]` | `1.64e-01` / `2.05e+00` |
| **`HRES` `hres_surface_net_thermal_radiation`** (`356k` pts) | **`0` (`0.0000%`)** | `0.999532` | `[0.9697, 1.0000, 1.0000, 1.0000, 1.0000]` | `[0.7956, 0.9998, 1.0000, 1.0000, 1.0000]` | `[0.6486, 0.9947, 0.9975, 0.9986, 0.9998]` | `9.76e-02` / `1.01e+00` |

- **Phase 3C End-to-End Cascaded Re-Delineated Polygons (`all490`, `n = 445` matched):**
  - The `8` canonical-matched basins that failed delineation (`1` `>60°N` + `7` `AREA_HINT_FAILURE`) produce `1.7978%` `Ext-Only NaN` points (`120 / 6,675` for CPC/IMERG, `1,200 / 66,750` for HRES).
  - Across the `437` valid delineated basins, `Ext-Only NaN = 0.0000%`, `Median NSE >= 0.9985`, and `Median KGE >= 0.9862` across all products (`CPC`, `IMERG`, `HRES`).

---

### 4.4 Phase 4 Baseline — Return Period Calculator (`benchmark-return-periods`)

- **USGS Fortran `peakfqr` v8.0 & CRAN R `MGBT` v1.1.6 Parity (`11,213` Global Caravan Basins):**
  - Convergence rate: **`100.00%` (`11,213 / 11,213`)**, `0` `NaN` quantiles.
  - `MGBT` `k_outliers` exact match vs. CRAN R `MGBT`: **`99.96%` (`11,208 / 11,213`)** in `mm/day` and **`99.94%` (`11,206 / 11,213`)** in `cfs`.
  - `GEMA` vs. Fortran `R MGBT + Fortran EMA` $Q_{100}$ relative difference: **`Median = 0.00%`, `Mean = 0.08%`, `99.30%` of basins within `< 1%`**, `Pearson r = 1.000000`.
- **Live `benchmark_basins_500.parquet` Verification (`437 / 500` Eligible Basins with $\ge 10$ Water Years):**
  - `converged_basins = 437 / 437` (`100.00%`), `failed_or_nan_quantile_count = 0`.
  - Unit-conversion invariance (`mm/day` vs. `cfs` rescaled): `MGBT` `klow` **`100.00%` match**; deterministic `GEMA` $Q_{100}$ relative difference **`Median = 1.90e-13%`, `Max = 1.14e-08%`**.
  - `SimpleLP3` vs. `GEMA` tail divergence on the `115 / 437` (`26.32%`) PILF basins (`klow > 0`): **`[P50 = 12.81%, P75 = 24.16%, P90 = 38.70%, Max = 111.64%]`** at $Q_{100}$.

---

### 4.5 Phase 5 Baseline — Core Forecasting Model (`benchmark-model`)

- **Completeness (`25` MultiMet Basins, `CPC` + `IMERG` + `HRES`):** `evaluated_basins = 25 / 25`, `failed_or_nan_basins = 0`, `pred_nan_when_obs_valid = 0`.
- **`compare_forcings` (Canonical vs. Reconstructed `CPC`+`IMERG`+`HRES`, `seq_length=180, lead_time=7`):**
  - Deterministic `MeanEmbeddingForecastLSTM` & `HandoffForecastLSTM`: `pred_mean_abs_diff = 1.16e-03 to 1.34e-03 mm/d`, `pred_median_pearson_r = 0.999999`, `median_abs_delta_NSE = 3.0e-04 to 3.6e-04`, `median_abs_delta_KGE = 1.8e-04 to 2.5e-04`.
  - Note on Monte Carlo `head='cmal'` (`handoff_forecast_lstm_cmal`): Until Issue #72 (Upstream #353) seeds the RNG at the start of `BaseTester.evaluate()`, `sample_cmal` draws independent Monte Carlo trajectories across consecutive `evaluate()` calls (`pred_median_pearson_r = 0.9059`, `median_abs_delta_KGE = 0.0241` from sampling variance). Always evaluate forcing sensitivity primarily on deterministic heads (`regression`, `cmal_deterministic`).
- **`benchmark_hot_start` (`seq_length=180, lead_time=7`):**
  - `handoff_forecast_lstm`: **`3.87x` forward speedup**, `max_abs_diff = 1.19e-07 mm/d` (`median_max_abs_diff = 1.49e-08 mm/d`).
  - `mean_embedding_forecast_lstm`: **`2.24x` forward speedup**, `max_abs_diff = 2.38e-07 mm/d` (`median_max_abs_diff = 5.96e-08 mm/d`).

---

## 5. Pass/Fail Regression Thresholds for Future Releases

When re-running this benchmark suite on future branches or PRs, any of the following constitutes a **blocking regression**:

1. **Catchment Delineation (`benchmark-catchment`):**
   - In-coverage `AREA_HINT_FAILURE` rate increases above `0.20%` on `benchmark_basins_1000.parquet` or `2.00%` on `benchmark_basins_500.parquet`.
   - Unconditional in-coverage `Median IoU` drops below `0.970` (`1,000` global) or `0.950` (`500` Caravan), or `P10 IoU` drops below `0.900` (`1,000` global).
2. **Static Attribute Extractor (`benchmark-static-extractor`):**
   - `pred_nan_when_ref_valid > 0` on any of the `196` HydroATLAS attributes (`--era5-source none`) or the `10` `hybas`-supported ERA5 attributes.
   - Median continuous Pearson $r$ drops below `0.9995` or categorical majority accuracy drops below `99.0%`.
3. **Gridded Archive Builders & Timeseries Extractors (`benchmark-gridded-archive`, `benchmark-timeseries-extractor`):**
   - `rebuilt_only_nan_count > 0` or `max_abs_err > 1e-5` on rebuilt `CPC` or `IMERG` grids.
   - `extracted_only_nan_count > 0` on `IMERG` or `HRES` (or `> 2` coastal coverage boundary points on `CPC` across `4,017` days).
   - Unmasked `all_matched` `Median NSE < 0.995` on `IMERG` or `< 0.9999` on `CPC` / `HRES`.
4. **Return Periods (`benchmark-return-periods`):**
   - `failed_or_nan_quantile_count > 0` on any eligible basin ($\ge 10$ water years).
   - Unit-conversion invariance error (`mm/day` vs. `cfs`) exceeds `1e-6%` for `SimpleLP3` or deterministic `GEMA`.
5. **Forecasting Models (`benchmark-model`):**
   - `failed_or_nan_basins > 0` or `pred_nan_when_obs_valid > 0`.
   - Cold-start vs. hot-start `max_abs_diff > 1e-5 mm/d`.
   - Canonical vs. reconstructed forcing `pred_median_pearson_r < 0.9999` or `median_abs_delta_KGE > 1e-3` on deterministic heads.
