# MultiMet Data Workflows (`multimet`)

The `multimet` package provides tools for **watershed boundary delineation** ([`multimet/catchment_delineation`](catchment_delineation/README.md)), **gridded meteorological archives** ([`multimet/gridded_archive_builders`](gridded_archive_builders/README.md)), **static watershed attribute tables** (`multimet/static_extractor`), and **catchment meteorological timeseries** ([`multimet/timeseries_extractors`](timeseries_extractors/README.md)) for OpenHydroNet.

---

## Part 1: Gridded Precipitation Archive Builders (`multimet/gridded_archive_builders`)

The [`multimet/gridded_archive_builders`](gridded_archive_builders/README.md) subpackage includes command-line tools (`build-cpc-archive` and `build-imerg-archive`) to download public gridded precipitation data from NOAA and NASA and save it into standardized daily Zarr archives.

> **Do you need these tools?**
> If you only want to train or evaluate flood-forecasting models using the published MultiMet dataset, **you do not need to run these tools**. Simply point `dynamics_data_dir` in your training configuration file to `gs://caravan-multimet/v1.1`.
>
> Use these tools only if you want to download raw precipitation grids directly from the upstream providers and build or update your own Zarr archives.

### Overview

Two gridded archive command-line tools are installed when you run `pip install -e .` from the root of this repository:

| Command | Dataset | Spatial Grid | Available Dates |
| --- | --- | --- | --- |
| `build-cpc-archive` | NOAA CPC Global Unified Daily Precipitation | 0.5° (`360 × 720`) | 1979 to present |
| `build-imerg-archive` | NASA GPM IMERG Early V07 Daily Precipitation | 0.1° (`1800 × 3600`) | 2000-06-01 to present |

Each tool downloads raw files from the weather agency, validates coordinates and dimensions, converts them onto a consistent daily grid, marks missing values as `NaN`, and saves the result to the `--target_zarr` path you specify.

### Prerequisites

Activate the `googlehydrology` Conda environment and install the repository in editable mode:

```bash
conda activate googlehydrology
pip install -e .
```

#### Additional Requirements by Dataset

* **NOAA CPC (`build-cpc-archive`):** No extra packages or accounts are required.
* **NASA GPM IMERG (`build-imerg-archive`):** Downloading from NASA GES DISC requires a free [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) account. You can provide your credentials in any of three ways:
  1. A `~/.netrc` file on your computer with entries for `urs.earthdata.nasa.gov` and `gpm1.gesdisc.eosdis.nasa.gov`.
  2. Environment variables: `EARTHDATA_TOKEN` (or `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`).
  3. Command-line arguments: `--earthdata_token` (or `--earthdata_username` and `--earthdata_password`).

### 1. NOAA CPC Daily Precipitation (`build-cpc-archive`)

Downloads yearly NetCDF files (`precip.{year}.nc`) from the NOAA Physical Sciences Laboratory, flips latitude so it runs south-to-north (`-89.75` to `+89.75`), shifts longitude to `-179.75` to `+179.75`, and writes the daily variable `cpc_precipitation` (`mm/day`, `float32`).

```bash
# Build a local archive for 2020 to 2022 and delete temporary downloads
build-cpc-archive \
  --target_zarr ./data/cpc_daily.zarr \
  --start_year 2020 \
  --end_year 2022 \
  --cleanup_cache

# Run the same command later to append newly published days
build-cpc-archive \
  --target_zarr ./data/cpc_daily.zarr \
  --cleanup_cache
```

#### Command-Line Arguments (`build-cpc-archive`)

* `--target_zarr` *(required)*: Path where the output Zarr archive is saved. Use a local folder path (e.g., `./data/cpc.zarr`) or a Google Cloud Storage URI starting with `gs://` (e.g., `gs://my-bucket/cpc.zarr`).
* `--start_year`: First year to download (integer, default: `1979`).
* `--end_year`: Last year to download, inclusive (integer, default: current calendar year).
* `--start_date`: Optional start date filter (`YYYY-MM-DD`) if you only want dates on or after a specific day inside `--start_year`.
* `--end_date`: Optional end date filter (`YYYY-MM-DD`) if you only want dates on or before a specific day inside `--end_year`.
* `--cache_dir`: Local folder used to store downloaded NOAA NetCDF files before processing. If omitted, a temporary folder is created and removed automatically when the command finishes.
* `--cleanup_cache`: Deletes each downloaded NetCDF file as soon as it is written to the Zarr archive. Recommended to save disk space.
* `--overwrite`: Deletes the existing Zarr archive at `--target_zarr` and rebuilds it from scratch. If not set, the tool resumes and appends only new dates.
* `--num_workers`: Number of years to download and process in parallel (integer, default: number of CPU cores up to `32`). Set to `1` to run one year at a time.
* `--project`: Google Cloud project ID used for billing and authentication when `--target_zarr` is a `gs://` bucket (default: `None`).
* `--source_url_template`: Custom download URL template containing `{year}` (default: official NOAA PSL URL).

### 2. NASA GPM IMERG Daily Precipitation (`build-imerg-archive`)

Builds a daily `0.1°` global precipitation archive (`1800` latitudes `-89.95 .. 89.95` by `3600` longitudes `-179.95 .. 179.95`) from NASA GPM IMERG Early Run Version 07 (V07), saving the variable `imerg_precipitation` (`mm/day`, `float32`).

```bash
# Download daily V07 files from NASA GES DISC into a local Zarr archive
build-imerg-archive \
  --target_zarr ./data/imerg_daily.zarr \
  --start_date 2024-01-01 \
  --end_date 2024-01-10 \
  --cleanup_cache

# Build from a local directory of pre-downloaded V07 .nc4 files
build-imerg-archive \
  --target_zarr ./data/imerg_daily.zarr \
  --source local \
  --local_format nc4 \
  --local_dir /path/to/local/imerg_files \
  --start_date 2024-01-01 \
  --end_date 2024-01-10
```

#### Command-Line Arguments (`build-imerg-archive`)

* `--target_zarr` *(required)*: Path where the output Zarr archive is saved (local path or `gs://` URI).
* `--start_date`: First date to include in `YYYY-MM-DD` format (default: `2000-06-01`).
* `--end_date`: Last date to include in `YYYY-MM-DD` format (default: yesterday UTC).
* `--source`: Where to read IMERG data from (choices: `gesdisc` or `local`, default: `gesdisc`).
  * `gesdisc`: Discovers the published daily V07 NetCDF-4 granule via NASA CMR and downloads it from NASA GES DISC over HTTPS.
  * `local`: Reads pre-downloaded V07 files from `--local_dir`.
* `--local_format`: Local file format when `--source local` is used (choices: `nc4` or `h5`, default: `nc4`).
* `--local_dir`: Path to a local folder containing pre-downloaded IMERG V07 daily NetCDF-4 files (`.nc4` / `.nc`) or 48 half-hourly HDF5 granules (`.RT-H5` / `.HDF5`) per day. Required when `--source local` is used.
* `--earthdata_token`: NASA Earthdata Bearer token for `--source gesdisc` (can also be set via the `EARTHDATA_TOKEN` environment variable).
* `--earthdata_username`: NASA Earthdata username (can also be set via `EARTHDATA_USERNAME` or `~/.netrc`).
* `--earthdata_password`: NASA Earthdata password (can also be set via `EARTHDATA_PASSWORD` or `~/.netrc`).
* `--netrc_path`: Path to a custom `.netrc` file containing Earthdata credentials (default: `~/.netrc`).
* `--cache_dir` (or `--local_cache`): Local folder used to stage files downloaded from NASA GES DISC. If omitted, a temporary folder is created and cleaned up automatically on exit.
* `--cleanup_cache`: Deletes each downloaded NetCDF file immediately after it is processed and removes `--cache_dir` on exit.
* `--batch_size`: Number of daily grids accumulated before each Zarr write (integer, default: `30`).
* `--num_workers`: Number of dates downloaded or extracted in parallel (integer, default: `4`). Keep between `4` and `8` when downloading from NASA GES DISC to avoid server rate limits.
* `--granule_workers`: Number of parallel threads used to read the 48 half-hourly HDF5 files per day when using `--source local` with `--local_format h5` (integer, default: `8`).
* `--overwrite`: Deletes the existing Zarr archive at `--target_zarr` and rebuilds it from scratch.
* `--in_place`: Overwrites the requested `--start_date` to `--end_date` dates in-place inside an existing Zarr archive.
* `--project`: Google Cloud project ID used when writing to a `gs://` bucket (default: `None`).
* `--gesdisc_url`: Custom base URL for NASA GES DISC IMERG V07 daily files.

---

## Part 2: Extracting Static Attributes for Watersheds (`multimet/static_extractor`)

The `multimet.static_extractor` submodule builds the static watershed attribute tables required by the models.

To predict river flow in a watershed, OpenHydroNet needs a table of unchanging ("static") facts about that watershed—such as its area, elevation, slope, soil type, land cover, and long-term average weather.

If you are working with watersheds from the published [Caravan](https://www.nature.com/articles/s41597-023-01975-w) dataset, those tables are already provided. If you want to run models on **your own watersheds**, this tool takes a map file of your watershed boundaries and creates a Caravan-compatible CSV table for you.

It calculates these attributes using the same community datasets and methods used by Caravan:
- **[HydroATLAS](https://www.hydrosheds.org/hydroatlas) (BasinATLAS v10, Level 12):** Geography, elevation, slope, soil, land cover, lakes, and human footprint.
- **[ERA5-Land](https://cds.climate.copernicus.eu/) (1981–2020):** Long-term climate averages (precipitation, temperature, evaporation, aridity, snow fraction, and wet/dry spell frequency).

---

## Before You Start

### 1. What Your Input File Needs
- **File format:** A map file containing polygon boundaries for one or more watersheds in **GeoJSON** (`.geojson`), **Shapefile** (`.shp`), **GeoPackage** (`.gpkg`), or **GeoParquet** (`.parquet`) format.
- **Coordinates:** A defined coordinate reference system (standard latitude and longitude `EPSG:4326` / WGS84, or any projected coordinate system, which will be converted to `EPSG:4326`).
- **Watershed ID column:** Each polygon must have a unique ID in the `gauge_id` column (or in the column you specify with `--id-column <column_name>`).

### 2. Reference Datasets
You provide the paths where the HydroATLAS and ERA5-Land reference datasets are stored:
- **HydroATLAS (`--gdb-path` or `--gcs-gdb-uri`):** Local path to `BasinATLAS_v10.gdb` (or `BasinATLAS_v10_lev12.shp`). If you do not already have it on disk, pass `--gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas/BasinATLAS_v10.gdb` and the tool will download it into `--gdb-path` (~4.9 GB).
- **ERA5-Land Climate Data:**
  - When using `--era5-source hybas`, pass `--era5-cache-dir /path/to/era5_climate`. If you do not already have the continental climate tables on disk, also pass `--gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate` (~550 MB) to download them into `--era5-cache-dir`.
  - When using `--era5-source gridded`, pass `--gridded-era5-uri gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr` (or a local Zarr path). This streams the daily grid slices directly from the Zarr store without downloading the archive to your machine.
- **Running Without Downloading Cloud Files (`--no-download`):**
  - If you have limited local disk space and do not want to download the HydroATLAS or ERA5 files to your computer, pass `--no-download` along with `--gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas` and `--gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate` (or `--gridded-era5-uri`). The tool will stream the required HydroATLAS and ERA5 data directly from Google Cloud Storage in memory without saving reference files to disk.
  - Alternatively, if you download the files to disk for a run and want them deleted automatically when the run finishes, add `--clean-cache`.

### 3. Choosing `--era5-source` (`hybas` vs. `gridded`)
Every run requires the `--era5-source` flag to tell the tool how to calculate climate statistics:

| Option | How It Works | Speed | When to Use It |
| :--- | :--- | :--- | :--- |
| **`--era5-source hybas`** | Combines pre-calculated climate summaries from the standard HydroATLAS sub-basins that overlap your watershed. *(If you also pass `--gridded-era5-uri`, the four `*_ERA5_LAND` evaporation columns are computed from the daily weather grid).* | Fast | **Recommended for most users.** Works well whenever your watersheds are roughly the size of standard river sub-basins (~100 $\text{km}^2$) or larger. |
| **`--era5-source gridded`** | Reads 40 years (1981–2020) of daily ERA5-Land weather grids over your exact polygon boundary and calculates every climate number from scratch. | Slower (several seconds per basin) | Use this if your watersheds are very small, have custom boundaries that do not follow natural river sub-basins, or if you want to stream climate data directly from a Zarr store without downloading continental climate tables. |

---

## Command-Line Usage (`static_extractor`)

### 1. Extract Attributes for One File (`extract-caravan-static`)

Use `extract-caravan-static` (or its alias `extract-static-attributes`) when you have a single file of watershed polygons and want a single CSV table of attributes.

```bash
# Using local HydroATLAS and pre-calculated sub-basin climate tables
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv \
    --gdb-path /path/to/BasinATLAS_v10.gdb \
    --era5-source hybas \
    --era5-cache-dir /path/to/era5_climate

# Downloading HydroATLAS and climate tables from Google Cloud Storage on first run
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv \
    --gdb-path /path/to/BasinATLAS_v10.gdb \
    --gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas/BasinATLAS_v10.gdb \
    --era5-source hybas \
    --era5-cache-dir /path/to/era5_climate \
    --gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate

# Streaming directly in memory from Google Cloud Storage without downloading to disk
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv \
    --gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas \
    --era5-source hybas \
    --gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate \
    --no-download

# Streaming climate numbers directly from daily ERA5-Land grids on Google Cloud Storage
extract-caravan-static \
    --input /path/to/watershed_polygons.geojson \
    --output /path/to/extracted_caravan_attributes.csv \
    --gdb-path /path/to/BasinATLAS_v10.gdb \
    --era5-source gridded \
    --gridded-era5-uri gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr

# Specifying a custom ID column and running across 8 CPU cores
extract-caravan-static \
    --input /path/to/basins.shp \
    --output /path/to/attributes.csv \
    --gdb-path /path/to/BasinATLAS_v10.gdb \
    --era5-source hybas \
    --era5-cache-dir /path/to/era5_climate \
    --id-column station_id \
    --workers 8
```

#### All Flags for `extract-caravan-static`

| Flag | Required? | Default | What It Does |
| :--- | :--- | :--- | :--- |
| `--input`, `-i` | **Yes** | — | Path to your input watershed boundary file (`.geojson`, `.shp`, `.gpkg`, or `.parquet`). |
| `--output`, `-o` | **Yes** | — | Path where the output CSV file will be saved. |
| `--gdb-path`, `-g` | Required unless `--no-download` | `None` | Local path to `BasinATLAS_v10.gdb`, shapefile, or GeoParquet file. |
| `--era5-source` | **Yes** | — | How to calculate ERA5 climate numbers: `hybas` (from pre-calculated sub-basin tables) or `gridded` (from daily ERA5-Land grids). |
| `--era5-cache-dir` | Required if `hybas` (unless `--no-download`) | `None` | Local folder where continental ERA5 climate tables (`<continent>_climate_indices.txt`) are stored. |
| `--gridded-era5-uri` | Required if `gridded` | `None` | Google Cloud Storage (`gs://...`) URI or local folder path for the daily ERA5-Land Zarr dataset. Can also be passed with `--era5-source hybas` to compute the four `*_ERA5_LAND` columns. |
| `--gcs-gdb-uri` | Required if `--no-download` without `--gdb-path` | `None` | Google Cloud Storage (`gs://...`) URI for HydroATLAS data. Downloaded into `--gdb-path` by default, or streamed directly in memory when `--no-download` is set. |
| `--gcs-era5-climate-uri` | Required if `hybas` with `--no-download` | `None` | Google Cloud Storage (`gs://...`) URI for continental ERA5 climate tables. Downloaded into `--era5-cache-dir` by default, or streamed directly in memory when `--no-download` is set. |
| `--no-download` | No | Disabled | Stream HydroATLAS and ERA5 data directly from Google Cloud Storage in memory without downloading files to local disk. |
| `--id-column` | No | `gauge_id` | Name of the column in your input file that holds the unique watershed ID. |
| `--workers`, `-w` | No | `1` | Number of CPU processes to run in parallel. Increase this (for example, `-w 8`) when extracting many watersheds. |
| `--min-overlap-threshold` | No | `0.0` | Minimum overlap area in $\text{km}^2$ required for a sub-basin fragment to be included (unless the watershed covers more than 50% of that sub-basin). Useful for ignoring tiny border slivers along the edge of a polygon. |
| `--clean-cache` | No | Disabled | Delete the local `--gdb-path` and `--era5-cache-dir` folders after the command finishes. |
| `--verbose`, `-v` | No | Disabled | Print detailed progress and debugging messages while running. |

---

### 2. Extract Attributes for Many Folders at Once (`extract-caravan-static-batch`)

Use `extract-caravan-static-batch` (or its alias `extract-static-attributes-batch`) when you have multiple dataset folders containing watershed boundary files and want to process all of them in a single run.

```bash
# Process all dataset folders inside a parent folder and also save one combined CSV
extract-caravan-static-batch \
    --parent-dir /path/to/watershed_folders/ \
    --output-dir /path/to/output_attributes/ \
    --gdb-path /path/to/BasinATLAS_v10.gdb \
    --era5-source hybas \
    --era5-cache-dir /path/to/era5_climate \
    --workers 16 \
    --combine

# Process a specific list of folders
extract-caravan-static-batch \
    --input-dirs /data/shapes/camels /data/shapes/camelsaus /data/shapes/lamah \
    --output-dir /path/to/output_attributes/ \
    --gdb-path /path/to/BasinATLAS_v10.gdb \
    --era5-source hybas \
    --era5-cache-dir /path/to/era5_climate \
    --workers 16
```

#### All Flags for `extract-caravan-static-batch`

*You must provide at least one of `--parent-dir`, `--input-dirs`, or `--input-files`.*

| Flag | Required? | Default | What It Does |
| :--- | :--- | :--- | :--- |
| `--parent-dir`, `-p` | One input flag required | — | One or more parent folders (local path or `gs://...`) that contain dataset subfolders of watershed files. |
| `--input-dirs`, `-d` | One input flag required | — | Space-separated list of specific dataset folders to process. |
| `--input-files`, `-f` | One input flag required | — | Space-separated list of specific polygon files (`.shp`, `.geojson`, `.gpkg`, `.parquet`) to process. |
| `--output-dir`, `-o` | **Yes** | — | Local folder or Google Cloud Storage (`gs://...`) path where output files will be written. |
| `--gdb-path`, `-g` | Required unless `--no-download` | `None` | Local path to `BasinATLAS_v10.gdb`, shapefile, or GeoParquet file. |
| `--era5-source` | **Yes** | — | How to calculate ERA5 climate numbers: `hybas` or `gridded`. |
| `--era5-cache-dir` | Required if `hybas` (unless `--no-download`) | `None` | Local folder where continental ERA5 climate tables are stored. |
| `--gridded-era5-uri` | Required if `gridded` | `None` | Google Cloud Storage (`gs://...`) URI or local path for the daily ERA5-Land Zarr dataset. |
| `--staging-dir` | Required for `gs://` inputs/outputs | `None` | Local folder used to stage downloaded input shapefiles or output CSVs when reading from or writing to `gs://` paths. |
| `--gcs-gdb-uri` | Required if `--no-download` without `--gdb-path` | `None` | Google Cloud Storage (`gs://...`) URI for HydroATLAS data. Downloaded into `--gdb-path` by default, or streamed directly in memory when `--no-download` is set. |
| `--gcs-era5-climate-uri` | Required if `hybas` with `--no-download` | `None` | Google Cloud Storage (`gs://...`) URI for continental ERA5 climate tables. Downloaded into `--era5-cache-dir` by default, or streamed directly in memory when `--no-download` is set. |
| `--no-download` | No | Disabled | Stream HydroATLAS and ERA5 data directly from Google Cloud Storage in memory without downloading files to local disk. |
| `--id-column` | No | `gauge_id` | Name of the column in your input files that holds the unique watershed ID. |
| `--workers`, `-w` | No | `1` | Number of CPU processes to run in parallel. |
| `--combine` | No | Disabled | In addition to per-dataset files, save a single merged table (`attributes_caravan_combined.csv`) containing all watersheds across all processed folders. |
| `--partition-outputs`, `-P` | No | Disabled | When enabled, writes separate HydroATLAS and Caravan climate tables (`attributes_hydroatlas_<dataset>.csv`, `attributes_caravan_<dataset>.csv`, and `attributes_<dataset>.parquet`) inside each dataset's output subfolder instead of a single flat CSV. |
| `--gcs-output-uri` | No | `None` | Optional Google Cloud Storage (`gs://...`) destination where finished files should be uploaded after saving them locally in `--output-dir`. |
| `--no-resume` | No | Disabled | Re-run and overwrite datasets even if their output files already exist in `--output-dir` (by default, already finished datasets are skipped). |
| `--min-overlap-threshold` | No | `0.0` | Minimum overlap area in $\text{km}^2$ required for a sub-basin fragment to be included. |
| `--clean-staging` | No | Disabled | Delete `--staging-dir` after the batch run finishes. |
| `--clean-cache` | No | Disabled | Delete `--gdb-path`, `--era5-cache-dir`, and `--staging-dir` when the batch run finishes. |
| `--no-progress` | No | Disabled | Turn off interactive progress bars. |
| `--verbose`, `-v` | No | Disabled | Print detailed progress and debugging messages while running. |

---

## Using `static_extractor` in Python

### 1. Extract Attributes from a File to a DataFrame and CSV

```python
from multimet.static_extractor import StaticAttributesExtractor

extractor = StaticAttributesExtractor(
    gdb_path="/path/to/BasinATLAS_v10.gdb",
    era5_source="hybas",
    era5_cache_dir="/path/to/era5_climate",
)
df = extractor.extract_attributes_from_file(
    input_path="basins.geojson",
    output_csv_path="caravan_attributes.csv",
    id_column="gauge_id",
)
print(df.head())
```

### 2. Extract Attributes for a Single Polygon in Python

```python
import shapely.geometry
from multimet.static_extractor import StaticAttributesExtractor

extractor = StaticAttributesExtractor(
    gdb_path="/path/to/BasinATLAS_v10.gdb",
    era5_source="hybas",
    era5_cache_dir="/path/to/era5_climate",
)

# Polygon coordinates in (longitude, latitude)
polygon = shapely.geometry.Polygon([
    [-86.9, 40.4],
    [-86.8, 40.4],
    [-86.8, 40.5],
    [-86.9, 40.5],
    [-86.9, 40.4],
])

result = extractor.extract_attributes_for_polygon(
    polygon,
    catchment_id="my_basin_01",
    era5_source="hybas",
)

attrs = result["caravan_attributes"]
print("Drainage Area (km²):", result["total_area_km2"])
print("Mean Elevation (m):", attrs["ele_mt_sav"])
print("Mean Precipitation (mm/yr):", attrs["pre_mm_syr"])
```

### 3. Add Extracted Attributes to a Zarr Store

```python
extractor.append_attributes_to_zarr(
    master_zarr_path="/data/multimet/caravan.zarr",
    basin_id="my_basin_01",
    attributes=attrs,
)
```

---

## Checking Results Against Published Caravan Data (`benchmark-static-extractor`)

If you want to compare the numbers produced by this package against published Caravan values across reference basins, run `benchmark-static-extractor`:

```bash
# Compare all basins in a reference dataset
benchmark-static-extractor \
    --dataset /path/to/benchmark_basins_500.parquet \
    --gdb-path /path/to/BasinATLAS_v10.gdb \
    --era5-source hybas \
    --era5-cache-dir /path/to/era5_climate \
    --workers 14 \
    -o /path/to/benchmark_results/
```

---

## Part 3: Catchment Meteorological Timeseries Extractors (`multimet/timeseries_extractors`)

The [`multimet/timeseries_extractors`](timeseries_extractors/README.md) subpackage reduces gridded meteorological archives and upstream weather feeds to catchment-averaged daily forcing time series (`extract-multimet` and `extract-multimet-dask`) standardized to the Caravan MultiMet schema. See [`multimet/timeseries_extractors/README.md`](timeseries_extractors/README.md) for full documentation and CLI examples.

---

## Part 4: Watershed Boundary Delineation (`multimet/catchment_delineation`)

The [`multimet/catchment_delineation`](catchment_delineation/README.md) subpackage creates watershed boundary polygons and calculates drainage areas ($\text{km}^2$) from 90-meter flow-direction map tiles (`delineate-catchment` and `benchmark-catchment`). See [`multimet/catchment_delineation/README.md`](catchment_delineation/README.md) for quick-start commands, Python examples, and CLI flags.
