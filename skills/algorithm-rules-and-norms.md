---
name: algorithm-rules-and-norms
description: >-
  Non-negotiable algorithmic rules, data-integrity invariants, error-handling
  norms, spatial/temporal alignment assumptions, and cache-safety requirements
  across all flood-forecasting, multimet, catchment_delineation, and
  return_periods code. Use whenever writing, modifying, or reviewing algorithms
  and data pipelines.
---

# Algorithm Rules, Assumptions & Data-Integrity Norms (`flood-forecasting`)

Every model, data ingestion pipeline, spatial extractor, catchment delineator, and statistical estimator in this repository must strictly obey the following algorithmic rules and data-integrity norms.

---

## 1. Missing Data In Always Means `NaN` Out

- **Zero Silent Imputation or Gap-Filling:**
  - **Never** replace missing dates, missing sub-daily granules, missing grid cells, or missing forecast lead times with `0.0`, forward-filled values (`ffill`), backward-filled values (`bfill`), linear interpolation, or climatological averages.
  - **Never** substitute backup or fallback data products when a requested source is unavailable or delayed.
  - **Never** broadcast a single forecast step or analysis value across missing forecast lead times.
- **Explicit Missing-Data Representation:**
  - Missing numerical outputs must always be represented as `np.nan` (`float('nan')`), and missing geometries in batch delineation must be represented as `geometry: None` with an explicit status code (`MISSING_DATA_*` or `out_of_coverage`).
  - When sanitizing features for export, missing required fields (such as drainage area or outlet coordinates) must raise `KeyError`—never default to `0.0`.

---

## 2. Zero `try`/`except`/`finally` Blocks That Mask Failures

- **Do Not Catch Exceptions to Hide Bad Data or Broken Code:**
  - Avoid `try`/`except` blocks across data pipelines (`multimet/`, `catchment_delineation/`, `return_periods/`). Never catch broad exceptions (`Exception`, `KeyError`, `ValueError`, `OSError`) to log a warning and return `NaN`, `None`, or an empty collection.
- **Validate Preconditions Explicitly:**
  - Check local file/directory existence explicitly with `os.path.exists` / `Path.exists()`.
  - Check HTTP status codes explicitly (`response.status_code == 404`) before calling `response.raise_for_status()`.
  - Check dictionary keys, DataFrame columns, and NetCDF/HDF5/Zarr variable names and dimensions explicitly with `in` checks, raising descriptive `ValueError`, `KeyError`, or `FileNotFoundError` immediately when expectations are violated.

---

## 3. Explicit Inputs & Paths Only (No Hidden Defaults)

- **Never Hardcode Fallback Paths or Implicit Buckets:**
  - Tools such as `DemDelineator`, `StaticAttributesExtractor`, `build-cpc-archive`, and `build-imerg-archive` must require callers to pass explicit local paths or `gs://` URIs.
  - When reading remote tiles or datasets that require local staging, require an explicit `--cache-dir` / `cache_dir`.
  - When `--clean-cache` / `--cleanup_cache` is requested, delete **only** the specific files downloaded during the current run—never delete pre-existing user files or directories.

---

## 4. Sub-Daily Accumulation & Dataset Version Strictness

- **Complete Sub-Daily Coverage Required:**
  - When accumulating 48 half-hourly NASA IMERG Early V07 HDF5 granules into a daily UTC total (`mm/day`), verify that all 48 unique 30-minute start tokens (`-S000000` through `-S233000`) are present before reading data.
  - At every `(lat, lon)` grid cell, require all 48 half-hourly observations to be finite (`valid_counts == 48`, scaled by `0.5 hr`). Any grid cell with `< 48` valid half-hourly observations on a day must be set to `NaN` for that day.
- **No Dataset Version Mixing:**
  - In IMERG, accept only V07 (`precipitation` in `Grid/precipitation`) and reject legacy V06 (`precipitationCal`). Never mix incompatible product versions in the same archive.

---

## 5. Spatial & Zonal Aggregation Norms

- **Exact Area-Weighted Intersection:**
  - Zonal extraction (`ZonalEngine`) and HydroATLAS static attribute aggregation must use exact polygon-grid / polygon-subbasin intersection areas.
- **Minimum Valid Coverage Threshold:**
  - Gridded zonal extraction requires at least **80% area-weighted valid (non-`NaN`) grid coverage** over a catchment polygon at a given timestep; if valid coverage is `< 80%`, the catchment value for that timestep must be `NaN`.
- **No Centroid or Nearest-Neighbor Fallbacks for Out-of-Domain Polygons:**
  - If a polygon falls outside the grid domain or does not intersect any HydroATLAS sub-basin (or all intersections fall below `min_overlap_threshold`), return `NaN`—never snap to the nearest grid cell or sub-basin.
- **Catchment Delineation Integrity:**
  - Never truncate upstream watersheds at 5°×5° tile boundaries (load neighbor tiles seamlessly; if a required upstream tile is missing, raise `FileNotFoundError` / `RuntimeError` immediately) or at the `60°N` HydroSHEDS boundary.
  - Exceeding an explicit `max_cells` cap must raise `CatchmentCoverageError` rather than returning a truncated polygon.
  - When an expected drainage area hint (`--expected-area` / `expected_area_km2`) is provided and no channel within the search window matches within `area_tolerance`, log `[AREA HINT FAILURE]` to `stderr` and raise `CatchmentAreaMismatchError` without writing an output polygon.

---

## 6. Temporal Alignment, Archive Continuity & Cache Safety

- **Forecast Lead-Time Alignment:**
  - Maintain strict alignment between forecast issue date, `lead_time`, and valid date across `union_features` and `Multimet` dataset loaders (no off-by-one shifts).
- **Archive Continuity & Tail Publication Lag (`MAX_PUBLICATION_LAG_DAYS = 7`):**
  - Gridded Zarr archives must have strictly contiguous daily time axes (`1D` frequency) with no interior date gaps and no interior all-`NaN` days.
  - When `--end_date` is explicitly passed to `build-cpc-archive` or `build-imerg-archive`, require 100% of dates through `end_date` to exist and contain finite values.
  - When `--end_date` is omitted (defaulting to today UTC), allow at most `MAX_PUBLICATION_LAG_DAYS = 7` days of upstream publication lag at the very end of the archive (including early-January year rollover for CPC when the new annual file is not yet posted on NOAA PSL), while still rejecting any interior date gap prior to the latest published day.
- **Incremental Extension (`--extend_archive`) & Cache Validation:**
  - Never trust a cached annual NetCDF file blindly by filename (`precip.{year}.nc`); always verify its internal timestamps cover the required end date (`_is_cached_cpc_netcdf_usable`).
  - Support `--extend_archive` (`--extend-archive`), which validates an existing target Zarr store's tail date and forces fresh upstream downloads (bypassing pre-cached files) for all newly appended dates.
