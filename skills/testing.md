---
name: testing
description: >-
  Test suite organization, co-located package test directories, native-grid
  synthetic testing requirements, strict mocking boundaries, and data-integrity
  regression testing rules for flood-forecasting. Use whenever writing,
  moving, running, or reviewing unit, integration, or canary tests.
---

# Testing Standards & Organization (`flood-forecasting`)

This skill defines how tests are organized, written, and reviewed across the `flood-forecasting` repository.

---

## 1. Co-Located Package Test Suites (`<package>/tests/`)

1. **Test Directory Layout:**
   - Every top-level package maintains its own co-located `tests/` directory:
     - `multimet/tests/`: Unit, integration, and canary tests for `catchment_delineation`, `gridded_archive_builders`, `timeseries_extractors`, `static_extractor`, and `multimet/utils`.
     - `return_periods/tests/`: Unit and USGS Bulletin 17C verification tests (`moose_river_example_data.csv`, `orestimba_creek_example_data.csv`).
     - `model/tests/`: Core hydrology model, trainer, tester, loss, and datasetzoo tests.
     - `benchmarks/tests/`: Unit and synthetic end-to-end verification tests for the canonical benchmark harnesses in `benchmarks/`.
2. **Pytest & CI Configuration:**
   - Register every test directory in `pyproject.toml` under `[tool.pytest.ini_options] testpaths`.
   - Use standard Pytest markers registered in `pyproject.toml`:
     - `@pytest.mark.unit`: Fast, hermetic unit tests.
     - `@pytest.mark.integration`: End-to-end local pipeline integration tests.
     - `@pytest.mark.slow`: Computationally intensive tests.
     - `@pytest.mark.gpu`: Tests requiring CUDA GPU hardware (skipped in CPU CI via `-m "not gpu"`).
     - `@pytest.mark.canary`: Live checks against external upstream feeds (NOAA PSL, NASA CMR/Earthdata); skipped by default unless `--run-canary` is passed.

---

## 2. Native-Resolution Synthetic Testing (No Toy-Grid Monkeypatching)

- **Always Test at Native Spatial Resolution:**
  - Gridded builders and extractors must be tested at their real production grid dimensions using synthetic NetCDF, HDF5, Zarr, or NumPy files written to `tmp_path`:
    - **NOAA CPC:** `360 × 720` (`0.5°` global grid, latitudes `-89.75 .. 89.75`, longitudes `-179.75 .. 179.75` or `0.25 .. 359.75`).
    - **NASA IMERG:** `1800 × 3600` (`0.1°` global grid, latitudes `-89.95 .. 89.95`, longitudes `-179.95 .. 179.95`).
    - **Catchment Delineation DEM Tiles:** `6000 × 6000` (`5° × 5°` tile at `3-arcsec` / `1/1200°` resolution).
  - **Never monkeypatch** `LAT_COUNT`, `LON_COUNT`, `IMERG_LATS`, `IMERG_LONS`, or `TILE_CELLS` to tiny toy grids (`2 × 4`, `3 × 3`, etc.) in tests. Toy-grid monkeypatching hides coordinate-indexing bugs, transpose errors, and memory/chunking bugs.

---

## 3. Strict Mocking Boundaries

- **Only Mock External Network or Cloud Service Boundaries:**
  - Permitted mocks: `requests.get`, `requests.Session.get`, `download_http_file`, `check_http_url_exists`, `download_tile_from_gcs`, or `gcsfs.GCSFileSystem` (when testing `gs://` URI streaming hermetically).
- **Never Mock Internal Logic Under Test:**
  - Never mock NetCDF/HDF5/Zarr file parsing, coordinate validation, `ZonalEngine` weight calculation, date continuity checks, or Zarr array writing when testing pipeline functions. Write real synthetic files to `tmp_path` and assert on the actual output Zarr stores, GeoParquet/GeoJSON files, or DataFrames.

---

## 4. Required Data-Integrity Regression Tests

Every new module or PR must include explicit negative and edge-case tests verifying our algorithmic norms:

1. **Missing & Corrupt Inputs Fail Loudly:**
   - Missing input files, missing Zarr stores, missing required columns, or missing CLI flags raise `FileNotFoundError`, `KeyError`, `ValueError`, or `SystemExit`.
   - Partial sub-daily granules (e.g., 47 of 48 IMERG half-hours, or duplicate timestamps masking a missing slot) raise `ValueError`.
   - Sub-daily `NaN` cells propagate to daily `NaN` cells (`valid_counts < 48` $\implies$ `NaN`, never partial sum).
2. **No Silent Fallbacks:**
   - Out-of-domain polygons, non-intersecting sub-basins, or `< 80%` valid grid coverage produce `NaN` (never nearest-cell or zero fill).
   - Unmatched `--expected-area` hints in catchment delineation raise `CatchmentAreaMismatchError` and write no output file.
3. **Archive Continuity & Extension (`--extend_archive`):**
   - Interior all-`NaN` days or date gaps raise `ValueError`.
   - Tail publication lag up to `MAX_PUBLICATION_LAG_DAYS = 7` succeeds when `--end_date` is omitted, fails when lag exceeds 7 days, and fails when `--end_date` is explicitly requested.
   - Stale cached NetCDF files are re-downloaded when their internal dates do not cover the required end date, and `--extend_archive` always forces fresh downloads.

---

## 5. Mandatory Companion Canonical Benchmark (`benchmarks/`) for New Submodules

- Automated unit and integration tests (`<package>/tests/`) are necessary but not sufficient for new algorithmic submodules.
- Every new submodule, subpackage, or major algorithmic pipeline must also add a **manual, comprehensive canonical benchmark** in the root `benchmarks/` package (`benchmarks/<component>.py`, with unit tests in `benchmarks/tests/`) following [`skills/benchmarking.md`](./benchmarking.md).
- **Exemption:** Features where quantitative canonical benchmarking does not make sense or is not possible (such as UI components, interactive dashboards, or pure visualization utilities) are exempt from adding a `benchmarks/` harness, provided they still include unit/integration tests in `<package>/tests/`.
