---
name: repo-organization
description: >-
  Repository architecture, top-level package layout, multimet subpackage
  boundaries, shared utility rules (multimet/utils), and packaging/CI
  synchronization for the flood-forecasting (OpenHydroNet / Open-MultiMet)
  repository. Use whenever adding, moving, refactoring, or reviewing modules,
  subpackages, or CLI entry points.
---

# Repository Organization (`flood-forecasting`)

This skill defines the package layout, subpackage boundaries, shared-utility rules, and packaging conventions for the `flood-forecasting` repository.

---

## 1. Top-Level Repository Layout

The repository is organized into self-contained top-level Python packages plus shared documentation, environment specifications, agent skills, and CI workflows:

```text
flood-forecasting/
├── .github/workflows/            # All GitHub Actions CI workflows (root-only)
├── googlehydrology/              # Core deep-learning hydrological modeling package
│   ├── datasetzoo/               # Dataset loaders (Caravan, MultiMet, CAMELS, ...)
│   ├── datautils/                # Scalers, normalization, climate/unit utilities
│   ├── evaluation/               # Testers, metrics, uncertainty & data assimilation
│   ├── modelzoo/                 # Neural network architectures (CudaLSTM, Handoff, MeanEmbedding, ...)
│   ├── training/                 # Trainers, loss functions, regularizers, logger
│   ├── utils/                    # Config parser, command-line scheduler, custom errors
│   └── tests/                    # Co-located hydrology model tests (Issue #17)
├── multimet/                     # Multi-source meteorological & static data pipelines
│   ├── gridded_archive_builders/ # Upstream gridded Zarr builders (CPC, IMERG)
│   ├── timeseries_extractors/    # Catchment area-weighted zonal timeseries extractors
│   ├── static_extractor/         # HydroATLAS & Caravan climate static attribute extractor
│   ├── utils/                    # Shared storage, HTTP/Earthdata, Zarr, spatial & zonal helpers
│   └── tests/                    # Co-located unit, integration, and canary tests for multimet
├── catchment_delineation/        # Global DEM flow routing & watershed polygon delineation
│   ├── scripts/                  # DEM tile slicing & benchmark dataset builder scripts
│   └── tests/                    # Co-located unit & integration tests for catchment_delineation
├── return_periods/               # USGS Bulletin 17C flood frequency (MGBT + EMA) calculator
│   ├── benchmark/                # Caravan USGS R + Fortran peakfq benchmark suite & report
│   └── tests/                    # Co-located unit & USGS Bulletin 17C benchmark tests
├── docs/                         # Sphinx ReadTheDocs documentation (source/usage/ and source/api/)
├── environments/                 # Conda (conda.yml, environment_cpu.yml) & RTD requirements
├── skills/                       # Project-level AI agent skills
├── pyproject.toml                # Ruff linter and Pytest configuration
└── setup.py                      # Package installation and console_scripts entry points
```

> **Ongoing Consolidation (Issue #17):**
> Model-specific assets (`pretrained-models/`, `tutorial/`, `example-configs/`, `tools/`) and hydrology model tests in `test/` are being consolidated into `googlehydrology/` (`googlehydrology/tests/`) so every top-level package is completely self-contained.

---

## 2. `multimet/` Subpackage Boundaries & Shared Utilities

1. **One Subpackage per Distinct Workflow:**
   - `multimet/gridded_archive_builders/`: CLI builders and incremental extenders (`build-cpc-archive`, `build-imerg-archive`) that download native-resolution daily precipitation grids from NOAA PSL and NASA GES DISC and write standardized `(time, latitude, longitude)` Zarr stores.
   - `multimet/timeseries_extractors/`: Catchment-polygon zonal averaging extractors (`CPC`, `IMERG`, `ERA5-Land`, `HRES`, `GraphCast`, and `dynamical.org` forecast catalogs) with serial and Dask runners (`extract-multimet`, `extract-multimet-dask`).
   - `multimet/static_extractor/`: Static catchment attribute extractor (`extract-caravan-static`, `extract-static-attributes`, `extract-caravan-static-batch`, `benchmark-static-extractor`) computing HydroATLAS Level 12 summaries and long-term Caravan climate signatures.

2. **`multimet/utils/` is the Single Source of Truth for Shared Helpers:**
   - Any utility needed by more than one subpackage **must** live in `multimet/utils/`.
   - **Never duplicate** helper functions across subpackages, and **never leave thin re-export shim modules** inside subpackages—import directly from `multimet.utils`:
     - `multimet/utils/http.py` & `earthdata.py`: NASA Earthdata session authentication (Bearer token, username/password, `.netrc`), NASA CMR granule queries, and atomic HTTP downloads (`download_http_file`, `check_http_url_exists`).
     - `multimet/utils/storage.py`, `gcs.py`, & `zarr.py`: Local vs. `gs://` URI resolution (`is_remote_zarr_target`, `get_zarr_mapper`), Zarr v2/v3 metadata and CF-time coordinate inspection, contiguous archive resume planning (`plan_archive_resume`), and batch/in-place Zarr writers.
     - `multimet/utils/climate.py`: FAO-56 Penman-Monteith PET (`calculate_fao_pm_pet`) and Caravan climate indices (`compute_caravan_climate_metrics`, `calculate_knoben_moisture_and_seasonality`).
     - `multimet/utils/geometry.py`, `spatial.py`, & `zonal.py`: Polygon validation/repair, bounding-box slicing, and exact area-weighted polygon-grid intersection weights (`ZonalEngine`).

---

## 3. CI Workflows, Packaging & Dependencies

1. **Root-Only GitHub Actions Workflows:**
   - GitHub Actions only discovers workflows in the root `.github/workflows/` directory. Never place `.github/workflows/` inside subpackages.
2. **Keep Packaging Synchronized:**
   - Whenever adding, renaming, or moving a subpackage, CLI command, or test directory, update:
     - `setup.py` (`packages=[...]` and `entry_points['console_scripts']`)
     - `environments/conda.yml`, `environments/environment_cpu.yml`, and `environments/rtd_requirements.txt`
     - `pyproject.toml` (`[tool.pytest.ini_options] testpaths`)
