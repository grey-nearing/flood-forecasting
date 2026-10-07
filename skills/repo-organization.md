---
name: repo-organization
description: >-
  Repository architecture, top-level package layout, multimet subpackage
  boundaries, harmonized package-scoped tools/ directories, shared utility
  rules (multimet/utils), and packaging/CI synchronization for the
  flood-forecasting (OpenHydroNet / Open-MultiMet) repository. Use whenever
  adding, moving, refactoring, or reviewing modules, subpackages, helper
  scripts/tools, or CLI entry points.
---

# Repository Organization (`flood-forecasting`)

This skill defines the package layout, subpackage boundaries, helper-tool placement, shared-utility rules, and packaging conventions for the `flood-forecasting` repository.

---

## 1. Top-Level Repository Layout

The repository is organized into self-contained top-level Python packages (`model/`, `multimet/`, `return_periods/`, `benchmarks/`) plus shared documentation, environment specifications, agent skills, and CI workflows:

```text
flood-forecasting/
├── .github/workflows/             # All GitHub Actions CI workflows (root-only)
├── benchmarks/                    # Standalone canonical benchmark suite across all core components
│   ├── catchment_delineation.py   # Global 90m D8 watershed polygon accuracy benchmark (benchmark-catchment)
│   ├── static_extractor.py        # HydroATLAS & Caravan static attribute benchmark (benchmark-static-extractor)
│   ├── gridded_archive_builders.py# Gridded Zarr archive parity benchmark (benchmark-gridded-archive)
│   ├── timeseries_extractors.py   # MultiMet catchment timeseries reconstruction benchmark (benchmark-timeseries-extractor)
│   ├── return_periods.py          # Caravan USGS R (MGBT) + Fortran (peakfq) benchmark (benchmark-return-periods)
│   ├── model.py                   # Core forecasting model, forcing & hot-start benchmark (benchmark-model)
│   ├── tools/                     # Benchmark cohort dataset builders (build_benchmark_dataset.py)
│   └── tests/                     # Co-located unit tests for all benchmark harnesses
├── model/                         # Core deep-learning hydrological modeling package
│   ├── datasetzoo/                # Dataset loaders (Caravan, MultiMet, CAMELS, ...)
│   ├── datautils/                 # Scalers, normalization, climate/unit utilities
│   ├── evaluation/                # Testers, metrics, uncertainty & data assimilation
│   ├── modelzoo/                  # Neural network architectures (CudaLSTM, Handoff, MeanEmbedding, ...)
│   ├── training/                  # Trainers, loss functions, regularizers, logger
│   ├── utils/                     # Config parser, command-line scheduler, custom errors
│   ├── example-configs/           # Reference YAML configs (FloodHub, State Handoff, CAMELS)
│   ├── pretrained-models/         # Pre-trained FloodHub model weights, scalers & configs
│   ├── tutorial/                  # Interactive OpenHydroNet Tutorial notebook, configs & sample data
│   └── tests/                     # Co-located hydrology model unit & integration tests
├── multimet/                      # Multi-source meteorological, static & spatial data pipelines
│   ├── catchment_delineation/     # Global DEM flow routing & watershed polygon delineation
│   │   └── tools/                 # DEM tile slicing scripts (slice_continental_dems.py)
│   ├── gridded_archive_builders/  # Upstream gridded Zarr builders (CPC, IMERG)
│   ├── timeseries_extractors/     # Catchment area-weighted zonal timeseries extractors
│   ├── static_extractor/          # HydroATLAS & Caravan climate static attribute extractor
│   ├── utils/                     # Shared storage, HTTP/Earthdata, Zarr, GCS, spatial & zonal helpers
│   └── tests/                     # Co-located unit, integration, and canary tests for all of multimet
├── return_periods/                # USGS Bulletin 17C flood frequency (MGBT + EMA) calculator
│   └── tests/                     # Co-located unit & USGS Bulletin 17C verification tests
├── docs/                          # Sphinx ReadTheDocs documentation (source/usage/ and source/api/)
├── environments/                  # Conda (conda.yml, environment_cpu.yml) & RTD requirements
├── skills/                        # Project-level AI agent skills
├── pyproject.toml                 # Ruff linter and Pytest configuration
└── setup.py                       # Package installation and console_scripts entry points
```

---

## 2. Package-Scoped `tools/` Directories (No Root `scripts/` or `tools/`)

- **Never create top-level `scripts/` or `tools/` folders at the repository root.**
- Any auxiliary scripts or data-preparation utilities must live inside the package or subpackage they belong to, using the harmonized directory name **`tools/`** (e.g., `multimet/catchment_delineation/tools/slice_continental_dems.py` and `benchmarks/tools/build_benchmark_dataset.py`).
- If a script is a primary user-facing workflow or canonical benchmark, expose it as an installed CLI entry point in `setup.py` (`console_scripts`) rather than requiring users to invoke a standalone script path.

---

## 3. `multimet/` Subpackage Boundaries & Shared Utilities

1. **One Subpackage per Distinct Workflow:**
   - `multimet/catchment_delineation/`: High-resolution (`90m` / 3-arcsec) D8 flow-direction watershed delineation (`delineate-catchment`); benchmarked via `benchmarks/catchment_delineation.py` (`benchmark-catchment`).
   - `multimet/gridded_archive_builders/`: CLI builders and incremental extenders (`build-cpc-archive`, `build-imerg-archive`) that download native-resolution daily precipitation grids from NOAA PSL and NASA GES DISC and write standardized `(time, latitude, longitude)` Zarr stores; benchmarked via `benchmarks/gridded_archive_builders.py` (`benchmark-gridded-archive`).
   - `multimet/timeseries_extractors/`: Catchment-polygon zonal averaging extractors (`CPC`, `IMERG`, `ERA5-Land`, and `HRES`) with serial and Dask runners (`extract-multimet`, `extract-multimet-dask`); benchmarked via `benchmarks/timeseries_extractors.py` (`benchmark-timeseries-extractor`).
   - `multimet/static_extractor/`: Static catchment attribute extractor (`extract-caravan-static`, `extract-static-attributes`, `extract-caravan-static-batch`) computing HydroATLAS Level 12 summaries and long-term Caravan climate signatures; benchmarked via `benchmarks/static_extractor.py` (`benchmark-static-extractor`).

2. **`multimet/utils/` is the Single Source of Truth for Shared Helpers:**
   - Any utility needed by more than one subpackage **must** live in `multimet/utils/`.
   - **Never duplicate** helper functions across subpackages, and **never leave thin re-export shim modules** inside subpackages—import directly from `multimet.utils`:
     - `multimet/utils/http.py` & `earthdata.py`: NASA Earthdata session authentication (Bearer token, username/password, `.netrc`), NASA CMR granule queries, and atomic HTTP downloads (`download_http_file`, `check_http_url_exists`).
     - `multimet/utils/storage.py`, `gcs.py`, & `zarr.py`: Local vs. `gs://` URI resolution (`is_gcs_path`, `normalize_gcs_path`, `upload_file_to_gcs`, `is_remote_zarr_target`, `get_zarr_mapper`), Zarr v2/v3 metadata and CF-time coordinate inspection, contiguous archive resume planning (`plan_archive_resume`), and batch/in-place Zarr writers.
     - `multimet/utils/climate.py`: FAO-56 Penman-Monteith PET (`calculate_fao_pm_pet`) and Caravan climate indices (`compute_caravan_climate_metrics`, `calculate_knoben_moisture_and_seasonality`).
     - `multimet/utils/geometry.py`, `spatial.py`, & `zonal.py`: Polygon validation/repair, bounding-box slicing, and exact area-weighted polygon-grid intersection weights (`ZonalEngine`).

---

## 4. Mandatory Comprehensive Benchmark Suite (`benchmarks/`) for All Submodules

1. **Dual Verification Requirement (`<package>/tests/` + `benchmarks/`):**
   - All canonical benchmark harnesses live in the dedicated root-level **`benchmarks/`** package (`benchmarks/<component>.py`, with unit tests in `benchmarks/tests/`). Do not scatter `benchmark.py` files inside individual subpackages.
   - Every new top-level package, subpackage, or major algorithmic component added to the repository **must** include **both**:
     1. Automated unit and integration tests in `<package>/tests/` (run automatically in CI), **and**
     2. A manual, comprehensive canonical benchmark in `benchmarks/<component>.py` (registered as a `benchmark-<component>` CLI entry point in `setup.py`, covered by unit tests in `benchmarks/tests/`, and documented in `benchmarks/README.md` and [`skills/benchmarking.md`](./benchmarking.md)) that evaluates the component at scale against canonical reference data with zero `NaN` masking, zero imputation, zero fallback to reference data, and full lower-tail (`[Min, P1, P5, P10, P25, P50]`) / failure-rate reporting.
2. **Exemption for Non-Benchmarkable Features (e.g., UI / Visualization):**
   - Certain new features or submodules—such as user interfaces (UI), interactive web/notebook frontends, plotting/visualization helpers, or pure configuration/devops utilities—may be added where quantitative canonical benchmarking does not make sense or is not possible. Such features are exempt from adding a `benchmarks/` harness (though they must still include unit/integration tests in `<package>/tests/` and note the exemption in the PR description).

---

## 5. CI Workflows, Packaging & Dependencies

1. **Root-Only GitHub Actions Workflows:**
   - GitHub Actions only discovers workflows in the root `.github/workflows/` directory. Never place `.github/workflows/` inside subpackages.
2. **Keep Packaging Synchronized:**
   - Whenever adding, renaming, or moving a subpackage, benchmark module, CLI command, or test directory, update:
     - `setup.py` (`packages=[...]` and `entry_points['console_scripts']`)
     - `environments/conda.yml`, `environments/environment_cpu.yml`, and `environments/rtd_requirements.txt`
     - `pyproject.toml` (`[tool.pytest.ini_options] testpaths`)
     - `.github/workflows/pytest-ci.yml` (`--cov=<package>` flags)
