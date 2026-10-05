---
name: gcs-bucket-organization
description: >-
  Complete architecture, directory hierarchy, dataset schemas, and Python/CLI
  access patterns for the gs://open-multimet/ Google Cloud Storage bucket
  (caravan-new, caravan-old, caravan-multimet, gridded-data-archives,
  ancillary-data, and data/era5_land). Use whenever reading from, writing to,
  extending, or documenting datasets in gs://open-multimet/ or
  gs://caravan-multimet/.
---

# `gs://open-multimet/` GCS Bucket Organization & Runbook

This skill documents the canonical layout, schemas, and access conventions for the **`gs://open-multimet/`** Google Cloud Storage bucket used by the Open-MultiMet and OpenHydroNet (`flood-forecasting`) pipelines.

- **Bucket Root:** `gs://open-multimet/`
- **Bucket Root Documentation:** `gs://open-multimet/README.md`
- **GCP Project:** `global-ungauged-experiments`
- **Storage Class & Region:** Standard (`us-central1`)

---

## 1. Top-Level Partition Summary

`gs://open-multimet/` is divided into six isolated top-level partitions:

| Partition | Purpose | Status |
| :--- | :--- | :--- |
| **`caravan-new/`** | Next-generation Caravan benchmark across **26,708 basins** (master `coordinates.csv`, rederived watershed polygons in `shapefiles-rederived/`, HydroATLAS + Caravan climate `attributes/`, and extracted `<product>/timeseries.zarr` stores). | Active Production Target |
| **`caravan-old/`** | Complete legacy Caravan v1.0, community extensions, and Google-internal extensions (baseline shapefiles, CSV/NetCDF timeseries, and CSV attributes). | 100% Verified Archive |
| **`caravan-multimet/`** | Pre-extracted MultiMet basin Zarr archives (`v1` and `v1.1`, 112.1 GiB total) across standard Caravan basins for 7 meteorological products (`CHIRPS`, `CHIRPS_GEFS`, `CPC`, `ERA5_LAND`, `GRAPHCAST`, `HRES`, `IMERG`). | 100% Complete |
| **`gridded-data-archives/`** | Native-resolution Analysis-Ready Cloud-Optimized (ARCO) global daily surface Zarr stores (`CPC/daily_surface.zarr`, `IMERG/daily_surface.zarr`, `HRES/daily_surface.zarr`). | Active Operational Stores |
| **`ancillary-data/`** | Pipeline dependencies: HydroSHEDS 3-arcsec flow directions/DEMs (`dems/`), HydroATLAS Level 12 geometries & BasinATLAS (`hydroatlas/`), and standard evaluation cohorts (`benchmarks/benchmark_basins_{500,1000}.parquet`). | Complete |
| **`data/`** | ECMWF ERA5-Land 0.1° daily surface reanalysis archive (`data/era5_land/daily_surface.zarr`, 1980-01-01 to 2026-09-08, 17 surface variables + FAO Penman-Monteith PET). | Completed |

---

## 2. Full Directory Hierarchy

```text
gs://open-multimet/
│
├── README.md                                    # Root bucket documentation & dataset guide
│
├── caravan-new/                                 # Next-Generation Caravan Production Benchmark
│   ├── coordinates.csv                          # Unified master coordinate table (all 26,708 basins)
│   ├── all_caravan_coordinates.csv              # Mirror alias of master coordinate table
│   │
│   ├── caravan-original/                        # Caravan v1.0 cohorts (16,299 basins across 7 subdatasets)
│   │   ├── CPC/timeseries.zarr/                 # Extracted daily CPC precipitation Zarr store
│   │   ├── HRES/timeseries.zarr/                # Extracted ECMWF HRES forecast Zarr store
│   │   ├── IMERG/timeseries.zarr/               # Extracted NASA IMERG Early V07 precipitation Zarr store
│   │   ├── attributes/<subdataset>/             # Parquet & CSV attributes (camels, camelsaus, camelsbr, ...)
│   │   ├── shapefiles-rederived/<subdataset>/   # Rederived watershed polygons (.geoparquet, .geojson, .shp)
│   │   └── shapefiles/<subdataset>/             # Baseline shapefiles + pour-point coordinates.csv
│   │
│   ├── caravan-extensions/                      # Community cohorts (8,548 basins across 8 subdatasets)
│   │   ├── CPC/timeseries.zarr/
│   │   ├── HRES/timeseries.zarr/
│   │   ├── IMERG/timeseries.zarr/
│   │   ├── attributes/<subdataset>/             # camelsch, camelscz, camelsde, camelsdk, camelses, grdc, il, lamahice
│   │   ├── shapefiles-rederived/<subdataset>/
│   │   └── shapefiles/<subdataset>/
│   │
│   └── google-internal/                         # Google internal cohorts (7 subdatasets)
│       ├── CPC/timeseries.zarr/
│       ├── HRES/timeseries.zarr/
│       ├── IMERG/timeseries.zarr/
│       ├── attributes/<subdataset>/             # camelscol, camelsfr, camelsind, camelskr, camelslux, camelsnz, camelspe
│       ├── shapefiles-rederived/<subdataset>/
│       └── shapefiles/<subdataset>/
│
├── caravan-old/                                 # Complete Legacy Caravan Benchmark Archive
│   ├── caravan-original/                        # camels, camelsaus, camelsbr, camelscl, camelsgb, hysets, lamah
│   │   ├── README.md, VERSION, licenses/
│   │   ├── shapefiles/<subdataset>/
│   │   ├── attributes/<subdataset>/             # attributes_caravan, attributes_hydroatlas, attributes_other
│   │   └── timeseries/{csv,netcdf}/<subdataset>/
│   ├── caravan-extensions/                      # camelsch, camelscz, camelsde, camelsdk, camelses, grdc, il, lamahice
│   │   ├── licenses/
│   │   ├── shapefiles/<subdataset>/
│   │   ├── attributes/<subdataset>/
│   │   └── timeseries/{csv,netcdf}/<subdataset>/
│   └── google-internal/                         # camelscol, camelsfr, camelsind, camelskr, camelslux, camelsnz, camelspe
│       ├── shapefiles/<subdataset>/
│       ├── attributes/<subdataset>/
│       └── timeseries/csv/<subdataset>/
│
├── caravan-multimet/                            # Pre-extracted MultiMet basin Zarr archives (112.1 GiB)
│   ├── v1.1/                                    # MultiMet v1.1 (56.03 GiB across 7 products)
│   │   ├── CHIRPS/timeseries.zarr/
│   │   ├── CHIRPS_GEFS/timeseries.zarr/
│   │   ├── CPC/timeseries.zarr/
│   │   ├── ERA5_LAND/timeseries.zarr/
│   │   ├── GRAPHCAST/timeseries.zarr/
│   │   ├── HRES/timeseries.zarr/
│   │   └── IMERG/timeseries.zarr/
│   └── v1/                                      # MultiMet v1.0 (56.04 GiB across 7 products)
│       └── <PRODUCT>/timeseries.zarr/
│
├── gridded-data-archives/                       # Native-resolution ARCO global daily surface Zarr stores
│   ├── CPC/daily_surface.zarr/                  # NOAA CPC 0.5° global daily precipitation (1979-present)
│   ├── IMERG/daily_surface.zarr/                # NASA GPM IMERG Early V07 0.1° daily precipitation (2000-06-01-present)
│   └── HRES/daily_surface.zarr/                 # ECMWF HRES 0.1° operational forecast archive
│
├── ancillary-data/                              # Delineation & static extraction inputs
│   ├── benchmarks/
│   │   ├── benchmark_basins_500.parquet         # Standard 500-basin global evaluation cohort
│   │   └── benchmark_basins_1000.parquet        # Standard 1000-basin global evaluation cohort
│   ├── dems/                                    # HydroSHEDS 3-arcsec flow direction grids (*_dir_3s.tif) & 5° tiles
│   └── hydroatlas/                              # hydro_atlas_lev12.parquet, BasinATLAS_v10.gdb/, subpolygons/, era5_climate/
│
└── data/
    └── era5_land/daily_surface.zarr/            # ECMWF ERA5-Land 0.1° daily surface reanalysis (1980-01-01 to 2026-09-08)
```

---

## 3. Dataset Schemas & Path Contracts

### 3.1 Provenance Collections & Subdatasets (`caravan-new/` & `caravan-old/`)

| Collection (`<collection>`) | Subdatasets (`<subdataset>`) | Basins |
| :--- | :--- | :--- |
| **`caravan-original`** | `camels` (671), `camelsaus` (561), `camelsbr` (870), `camelscl` (505), `camelsgb` (671), `hysets` (12,162), `lamah` (859) | 16,299 |
| **`caravan-extensions`** | `camelsch` (296), `camelscz` (249), `camelsde` (1,887), `camelsdk` (308), `camelses` (269), `grdc` (5,356), `il` (95), `lamahice` (88) | 8,548 |
| **`google-internal`** | `camelscol` (347), `camelsfr` (654), `camelsind` (313), `camelskr` (282), `camelslux` (56), `camelsnz` (355), `camelspe` (136) | 2,143 |

### 3.2 Pour-Point Coordinates (`coordinates.csv`)
- **Global Master Table:** `gs://open-multimet/caravan-new/coordinates.csv`
  - Columns: `gauge_id,gauge_lat,gauge_lon,original_id,collection,subdataset`
- **Per-Subdataset Table:** `gs://open-multimet/caravan-new/<collection>/shapefiles/<subdataset>/coordinates.csv`
  - Columns: `gauge_id,gauge_lat,gauge_lon,original_id`

### 3.3 Rederived Watershed Polygons (`shapefiles-rederived/`)
- **Path Pattern:** `gs://open-multimet/caravan-new/<collection>/shapefiles-rederived/<subdataset>/<subdataset>_basin_shapes.<ext>`
- **Extensions:** `.geoparquet` (recommended cloud-native format), `.geojson`, and ESRI Shapefile suite (`.shp`, `.shx`, `.dbf`, `.prj` in `EPSG:4326`, `.cpg` in `UTF-8`).
- **Required Columns:** `gauge_id` (string primary key matching `coordinates.csv`), `area` (`float64`, $\text{km}^2$), `gauge_lat` (`float64`), `gauge_lon` (`float64`), `geometry`.

### 3.4 Static Catchment Attributes (`attributes/`)
- **Path Pattern:** `gs://open-multimet/caravan-new/<collection>/attributes/<subdataset>/`
  - `attributes_<subdataset>.parquet`: Unified outer-joined Parquet table indexed by `gauge_id`.
  - `attributes_hydroatlas_<subdataset>.csv`: ~198 HydroATLAS Level 12 physical, terrain, soil, land-cover, and geology attributes.
  - `attributes_caravan_<subdataset>.csv`: Long-term climatic and hydrologic signatures (`p_mean`, `pet_mean`, `aridity`, `frac_snow`, `moisture_index`, `seasonality`, `high_prec_freq`, `high_prec_dur`, `low_prec_freq`, `low_prec_dur`, `gauge_lat`, `gauge_lon`).
  - `attributes_other_<subdataset>.csv`: Provider metadata (`gauge_name`, `country`, `area`, `elevation`).

### 3.5 Gridded ARCO Surface Zarr Archives (`gridded-data-archives/`)
- **NOAA CPC (`gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr`):**
  - Grid: `0.5°` (`360` latitudes `-89.75 .. 89.75` ascending $\times$ `720` longitudes `-179.75 .. 179.75`), variable `cpc_precipitation` (`mm/day`, `float32`).
  - Built/extended via:
    ```bash
    build-cpc-archive \
      --target_zarr gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
      --project global-ungauged-experiments \
      --extend_archive \
      --cleanup_cache
    ```
- **NASA GPM IMERG Early V07 (`gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr`):**
  - Grid: `0.1°` (`1800` latitudes `-89.95 .. 89.95` ascending $\times$ `3600` longitudes `-179.95 .. 179.95`), variable `imerg_precipitation` (`mm/day`, `float32`).
  - Built/extended via:
    ```bash
    build-imerg-archive \
      --target_zarr gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
      --project global-ungauged-experiments \
      --extend_archive \
      --cleanup_cache
    ```
