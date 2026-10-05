---
name: repo-organization
description: >-
  Repository architecture, package and subpackage layout, test co-location,
  shared utilities, strict data-integrity rules, and documentation strategy
  (README.md and Sphinx docs) for the flood-forecasting (OpenHydroNet /
  Open-MultiMet) repository. Use whenever adding, refactoring, reviewing, or
  documenting modules, subpackages, CLI entry points, or tests.
---

# Repository Organization & Documentation Strategy (`flood-forecasting`)

This skill defines the architectural conventions, code-quality invariants, test layout, and documentation standards for the `flood-forecasting` repository. Follow these rules whenever creating, refactoring, or reviewing code and documentation.

---

## 1. Top-Level Repository Layout

The repository is organized into self-contained top-level Python packages plus shared documentation, environment specifications, and CI workflows:

```text
flood-forecasting/
├── .github/workflows/           # All GitHub Actions CI workflows (root-only)
├── googlehydrology/             # Core deep-learning hydrological modeling package
│   ├── datasetzoo/              # Dataset loaders (Caravan, MultiMet, CAMELS, ...)
│   ├── datautils/               # Scalers, normalization, and unit utilities
│   ├── evaluation/              # Testers, metrics, uncertainty & assimilation
│   ├── modelzoo/                # Neural network architectures (CudaLSTM, Handoff, ...)
│   ├── training/                # Trainers, loss functions, regularizers, logger
│   └── utils/                   # Config parser, command-line scheduler, errors
├── multimet/                    # Multi-source meteorological & static data pipelines
│   ├── gridded_archive_builders/# Upstream gridded Zarr builders (CPC, IMERG)
│   ├── timeseries_extractors/   # Catchment area-weighted zonal timeseries extractors
│   ├── static_extractor/        # HydroATLAS & Caravan climate static attribute extractor
│   ├── utils/                   # Shared storage, HTTP/Earthdata, Zarr, climate, spatial & zonal utils
│   └── tests/                   # Co-located unit, integration, and canary tests for multimet
├── catchment_delineation/       # Global DEM flow routing & watershed polygon delineation
├── return_periods/              # USGS Bulletin 17C flood frequency (MGBT + EMA) calculator
├── docs/                        # Sphinx ReadTheDocs documentation (usage/ and api/)
├── environments/                # Conda (conda.yml, environment_cpu.yml) & RTD requirements
├── skills/                      # Project-level AI agent skills
├── pyproject.toml               # Ruff linter and Pytest configuration
└── setup.py                     # Package installation and console_scripts entry points
```

> **Ongoing Consolidation (Issues #17, #18, & #20):**
> - Model-specific assets (`pretrained-models/`, `tutorial/`, `example-configs/`, `tools/`) and hydrology model tests in `test/` are being consolidated into `googlehydrology/` (`googlehydrology/tests/`).
> - Package-specific tests (`test/test_static_extractor.py`, `test/test_catchment_delineation.py`, `test/test_return_periods.py`) belong inside their respective `<package>/tests/` directories (`multimet/tests/`, `catchment_delineation/tests/`, `return_periods/tests/`).
> - All packages must be included in `--cov=<package>` in `.github/workflows/pytest-ci.yml`.

---

## 2. `multimet/` Subpackage Boundaries & Shared Utilities

1. **One Subpackage per Distinct Workflow:**
   - `multimet/gridded_archive_builders/`: CLI builders and incremental extenders (`build-cpc-archive`, `build-imerg-archive`) that download native-resolution daily precipitation grids from NOAA PSL and NASA GES DISC and write standardized `(time, latitude, longitude)` Zarr stores.
   - `multimet/timeseries_extractors/`: Catchment-polygon zonal averaging extractors (`CPC`, `IMERG`, `ERA5-Land`, `HRES`, `GraphCast`, real-time Cold-Start/Hot-Start fetchers, and `dynamical.org` forecast catalogs) with serial and Dask runners.
   - `multimet/static_extractor/`: Static catchment attribute extractor (`extract-caravan-static`, `extract-static-attributes`) computing HydroATLAS Level 12 summaries and long-term Caravan climate signatures.

2. **`multimet/utils/` is the Single Source of Truth for Shared & Domain Helpers:**
   - Any utility needed by more than one subpackage—and **all methods belonging to a utility domain** (even if a specific function is only called by one subpackage)—**must** live in `multimet/utils/` so utilities are kept in a single central location.
   - **Never duplicate** helper functions across subpackages, and **never leave thin re-export shim modules** inside subpackages—import directly from `multimet.utils`:
     - `multimet/utils/http.py` & `earthdata.py`: NASA Earthdata session authentication (Bearer token, username/password, `.netrc`), NASA CMR granule queries, and atomic HTTP downloads (`download_http_file`, `check_http_url_exists`).
     - `multimet/utils/cpc.py`: Shared NOAA PSL CPC NetCDF download, cache date validation, and grid standardization (`CPC_LATS`, `CPC_LONS`, `ensure_psl_cpc_netcdf`, `process_cpc_netcdf_to_dataset`).
     - `multimet/utils/storage.py`, `gcs.py`, & `zarr.py`: Local vs. `gs://` URI resolution (`is_remote_target`, `get_zarr_mapper`), Zarr v2/v3 metadata and CF-time coordinate inspection, contiguous archive resume planning (`plan_archive_resume`), and batch/in-place Zarr writers.
     - `multimet/utils/climate.py`: All climate and hydrological signature math (`calculate_fao56_penman_monteith_pet`, `calculate_fao_pm_pet`, `calculate_knoben_moisture_and_seasonality`, `compute_caravan_climate_metrics`, `depth_to_mm`, `temp_to_celsius`).
     - `multimet/utils/geometry.py`, `spatial.py`, & `zonal.py`: Polygon validation/repair (`load_basin_geometries`), bounding-box slicing (`slice_dataset_by_bounds`, `slice_coordinates_by_bounds`), and exact area-weighted polygon-grid intersection weights (`ZonalWeightMatrix`, `ZonalWeightCalculator`, `weighted_mean_valid`, `weighted_mean_valid_with_coverage`).

---

## 3. Strict Data Integrity & Error Handling Rules

Every data ingestion, extraction, and archive-building module must obey these non-negotiable data-integrity rules:

1. **Zero `try`/`except`/`finally` Blocks or Equivalent Error-Suppression Logic:**
   - Avoid `try`/`except` blocks and equivalent silent error-masking constructs (`ignore_errors=True`, `on_error="ignore"`, returning `None` on invalid user inputs, or runtime signature duck-typing) in data pipelines (`multimet/`). Instead, validate preconditions explicitly (check file existence with `os.path.exists`, check HTTP status codes before `raise_for_status()`, validate dictionary keys and NetCDF/HDF5/Zarr variable names and dimensions) and let unexpected errors raise immediately with descriptive messages.

2. **No Default Output Paths, No Hardcoded Archive URIs, and No Unused Archival Paths:**
   - **Never include default output paths** in CLI flags or pipeline functions—require the user to explicitly pass all output paths (`--output_dir`, `--target_zarr`, `--output-parquet`, etc.) as required command-line arguments.
   - **Never hardcode default archive Zarr URIs or fallback buckets**—require explicit user-supplied archive store URIs (`--archive_store`, `--era5_zarr_uri`, etc.) and strip any unused or internal archival paths (e.g., `/cns/...`, `/namespace/...`) and unused extraction code paths from the codebase.

3. **Missing Data In Always Means Missing Data Out:**
   - **Never** silently fill missing dates, missing sub-daily granules, or missing forecast lead times with zeros, forward-filled values, interpolated values, broadcasted lead times, or backup/fallback data sources (never silently switch between upstream providers such as `dynamical` vs. `gesdisc` or `open_data` vs. `wb2`).
   - **Minimum 80% Valid Basin Area Coverage (`MIN_VALID_COVERAGE_FRACTION = 0.80`):** Catchment zonal aggregation requires at least **80%** of a basin's area-weighted grid cells to have valid (non-`NaN`) data on a given timestep (`weighted_mean_valid` / `ZonalWeightMatrix.reduce_*`). If `< 80%` of the basin has valid data, the catchment value **must be `NaN`**, and the exact area-weighted missing pixel fraction `[0.0, 1.0]` must be recorded in the companion `<product>_missing_fraction` variable.
   - **Sub-daily accumulation:** When summing 48 half-hourly IMERG HDF5 granules for a UTC day, require all 48 unique 30-minute start tokens (`-S000000` .. `-S233000`) to exist, and require all 48 observations at a grid cell to be valid (`valid_counts == 48`) for that cell's daily total to be finite; any cell with `< 48` valid half-hours must be `NaN`.
   - **No version mixing:** In IMERG, accept only V07 (`precipitation`) and reject legacy V06 (`precipitationCal`).

4. **Archive Continuity & Safe Incremental Extension (`--extend_archive`):**
   - Reject interior all-`NaN` days and non-contiguous date gaps (`ValueError` / `FileNotFoundError`).
   - When `--end_date` is explicitly provided, require 100% of dates through `end_date` to exist and contain finite values.
   - When `--end_date` is omitted, allow at most `MAX_PUBLICATION_LAG_DAYS = 7` days of upstream publication lag at the tail of the archive (including early-January year rollover for CPC), while still rejecting any interior date gap before the latest published date.
   - Always validate cached files against the required end date (`_is_cached_cpc_netcdf_usable`) or bypass cached files entirely when `--extend_archive` (`--extend-archive`) is passed.

---

## 4. Test Organization & Testing Rigor

1. **Co-Located Package Test Suites (`<package>/tests/`):**
   - Place tests for each top-level package in `<package>/tests/` (e.g., `multimet/tests/`).
   - Register test directories in `pyproject.toml` under `[tool.pytest.ini_options] testpaths` and include `--cov=<package>` in `.github/workflows/pytest-ci.yml`.
   - Mark tests with `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.gpu`, or `@pytest.mark.canary` (live upstream network checks skipped unless `--run-canary` is passed).

2. **No Toy-Grid Monkeypatching or Masking Mocks:**
   - **Always test at native spatial resolution** (e.g., `360 × 720` for CPC, `1800 × 3600` for IMERG, `721 × 1440` for HRES) using local synthetic NetCDF/HDF5/GRIB2/Zarr files in `tmp_path`. Never monkeypatch `LAT_COUNT`, `LON_COUNT`, `IMERG_LATS`, or `IMERG_LONS` to tiny grids in tests.
   - Only mock external network boundaries (`requests.get`, `Session.get`, `download_http_file`, or GCS transport); never mock internal data parsing, binary/GRIB/NetCDF decoding, coordinate validation, zonal reduction, or Zarr read/write logic.

---

## 5. CI Workflows, Packaging & Dependencies

1. **Root-Only GitHub Actions Workflows:**
   - GitHub Actions only discovers workflows in the root `.github/workflows/` directory. Never place `.github/workflows/` inside subpackages.
2. **Keep Packaging Synchronized:**
   - When adding or moving a subpackage or CLI command, update:
     - `setup.py` (`packages=[...]` and `entry_points['console_scripts']`)
     - `environments/conda.yml`, `environments/environment_cpu.yml`, and `environments/rtd_requirements.txt`
     - `pyproject.toml` (`testpaths`)

---

## 6. Documentation Strategy (`README.md` & Sphinx `docs/`)

We maintain a three-tier documentation structure written in clear, accessible English:

1. **Tier 1 — Concise Top-Level `README.md`:**
   - Keep the root `README.md` concise and scannable. Provide a high-level summary of the repository's capabilities, quick links to ReadTheDocs (`https://openhydronet.readthedocs.io/`), and a table pointing directly to each package and subpackage `README.md`.

2. **Tier 2 — Self-Contained Package & Subpackage `README.md`s:**
   - Every top-level package (`multimet/README.md`, `catchment_delineation/README.md`) and major workflow subpackage (`multimet/gridded_archive_builders/README.md`, `multimet/timeseries_extractors/README.md`) must have its own `README.md` containing:
     - A **"Do you need these tools?"** callout at the top explaining when users can simply read pre-built datasets from `gs://open-multimet/` (or `gs://caravan-multimet/v1.1`) instead of running the raw ingestion/extraction pipeline.
     - An overview table of CLI commands / modules, spatial grids, and date coverage.
     - Prerequisites (Conda environment activation, `pip install -e .`, and any external credentials such as NASA Earthdata Login).
     - Copy-pasteable CLI and Python usage examples (including `--extend_archive` and `--cleanup_cache`).
     - Complete reference bullets for all command-line flags.

3. **Tier 3 — Sphinx Documentation (`docs/source/`):**
   - Every user-facing feature or subpackage must be documented in Sphinx:
     - **Usage Guide:** Add or update `docs/source/usage/<feature>.rst` and register it in the `.. toctree::` of `docs/source/index.rst`.
     - **API Reference:** Add or update `docs/source/api/<module>.rst` using `.. automodule::` directives and register it in `docs/source/api/modules.rst` and `docs/Makefile`.
     - Verify that `make -C docs html` builds cleanly without Sphinx warnings.
