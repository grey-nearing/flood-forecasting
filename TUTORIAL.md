# OpenHydroNet Tutorial

This tutorial shows you how to install and use **OpenHydroNet**. It has three sections:

1. **[Section 1: Setup and Installation](#section-1-setup-and-installation)** — How to install the software, set up credentials, and download data.
2. **[Section 2: Using the Backend (CLI Commands and Python APIs)](#section-2-using-the-backend-cli-commands-and-python-apis)** — How to run every command-line tool and Python library function without the browser.
3. **[Section 3: Using the Interactive Web UI](#section-3-using-the-interactive-web-ui)** — How to use every tab, map tool, and chart in the browser application (`openhydronet-ui`).

---

## Section 1: Setup and Installation

### 1.1 System Requirements

Before you start, make sure your computer has:

- **Operating System**: Linux or macOS (Windows users should use WSL2).
- **Python**: Version `3.12` or newer.
- **Conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/). Conda is the easiest way to install geospatial C libraries like GDAL.
- **Disk Space**: At least `10 GB` of free space for code, sample data, and model weights.
- **GPU (Optional)**: An NVIDIA GPU with CUDA makes model training much faster, but a CPU works for testing, fine-tuning small basins, and running forecasts.

---

### 1.2 Step-by-Step Installation

Run these commands in your terminal to download the code and install OpenHydroNet:

#### Step 1: Clone the repository

```bash
git clone https://github.com/google-research/flood-forecasting.git
cd flood-forecasting
```

#### Step 2: Create and activate the Conda environment

Use the environment file in `environments/conda.yml` to create an environment named `openhydronet`:

```bash
conda env create -f environments/conda.yml
conda activate openhydronet
```

> **Note:** Always run `conda activate openhydronet` when you open a new terminal window.

#### Step 3: Install OpenHydroNet

Install the package in editable mode so all command-line tools are registered in your terminal:

```bash
pip install -e .
```

#### Step 4: Install optional weather forecast dependencies (if needed)

If you plan to download live weather forecast grids with `sync-weather-forecasts` or Dynamical.org Icechunk catalogs (`AIFS`, `GFS`, `GEFS`, `IFS_ENS`), install the optional `weather` or `dynamical` extras:

```bash
pip install -e ".[weather,dynamical]"
```

#### Step 5: Verify your installation

Check that the command-line tools work:

```bash
run --help
openhydronet-ui --help
```

If both commands print help messages, your installation is ready.

---

### 1.3 Credentials and API Keys (Optional)

Most parts of OpenHydroNet use free, public cloud buckets (`gs://open-multimet` and `gs://caravan-multimet/v1.1`) and do not need any login.

You only need credentials for three optional tasks:

| Data Source | When You Need It | How to Set It Up |
| :--- | :--- | :--- |
| **NASA Earthdata** | Downloading NASA GPM IMERG satellite rainfall (`build-imerg-archive`, `extract-multimet`, or `multimet-realtime` with `IMERG`). | Create a free account at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/) and add `machine urs.earthdata.nasa.gov login <USER> password <PASS>` to `~/.netrc` (`chmod 600 ~/.netrc`), or pass `--earthdata_username` and `--earthdata_password`. |
| **Google Earth Engine (GEE)** | Downloading raw MERIT-Hydro 90m flow-direction tiles from Earth Engine (`download-merit-d8-tiles`). | Run `earthengine authenticate` in your terminal and pass `--ee-project your-gcp-project-id`. |
| **Google FloodHub API** | Querying live FloodHub gauge forecasts in `maas` (`fetch-maas-forecast`). | Pass `--floodhub-api-key YOUR_KEY` or set `export FLOODHUB_API_KEY="YOUR_KEY"`. |

---

### 1.4 Downloading Training Data and Pre-Trained Models

#### Caravan Streamflow and Static Attributes
Global models train on the **Caravan** dataset (streamflow in `mm/day` and static catchment attributes) paired with **MultiMet** weather forcings.

1. Download the [Caravan NetCDF archive (`Caravan-nc.tar.gz`) from Zenodo](https://doi.org/10.5281/zenodo.6522634) and unpack it.
2. Convert the Caravan NetCDF and CSV files into fast Zarr stores (`streamflow.zarr` and `attributes.zarr`) using the built-in converter:

```bash
run convert-caravan \
  --caravan-dir ~/data/Caravan-nc \
  --output-dir ~/data/Caravan-zarr
```

#### MultiMet Historical Weather Archives
Pre-extracted MultiMet Zarr archives for global Caravan basins are available in the public Google Cloud Storage bucket `gs://caravan-multimet/v1.1` (and raw global gridded surface archives live in `gs://open-multimet/gridded-data-archives/`):
- `gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr` (NOAA CPC gauge precipitation, `0.5°`)
- `gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr` (NASA IMERG satellite precipitation, `0.1°`)
- `gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr` (ECMWF ERA5-Land reanalysis, `0.1°`)
- `gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr` (ECMWF HRES operational forecasts, `0.25°`)

#### Pre-Trained Model Weights
The repository includes three pre-trained `MeanEmbeddingForecastLSTM` checkpoints inside `model/pretrained-models/`:

1. `model/pretrained-models/google-floodhub-settings-110-epochs/` (`model_epoch110.pt`) — Uses `hres`, `graphcast`, `imerg`, and `cpc`.
2. `model/pretrained-models/google-floodhub-settings-no-graphcast-110-epochs/` (`model_epoch110.pt`) — Uses `hres`, `imerg`, and `cpc` (no GraphCast required).
3. `model/pretrained-models/google-floodhub-settings-aifs-110-epochs/` (`model_epoch110.pt`) — Uses `hres`, `aifs`, `imerg`, and `cpc`.

Each folder contains the model weights (`model_epoch*.pt`), normalization statistics (`scaler.zarr`), and configuration (`config.yml`). These checkpoints replicate the 2025 generation of Google FloodHub (a former operational model that has since been superseded by a newer model release).

---

## Section 2: Using the Backend (CLI Commands and Python APIs)

You can run every step of the flood forecasting workflow from the terminal or from a Python script.

Below is a summary of the command-line tools installed by `pip install -e .`:

| Category | CLI Command | Purpose |
| :--- | :--- | :--- |
| **Catchment Delineation** | `delineate-catchment` | Trace upstream basin boundaries from `(lat, lon)` outlet points. |
| | `download-merit-d8-tiles` | Download `5° × 5°` MERIT-Hydro 90m flow-direction tiles from Earth Engine. |
| **Static Attributes** | `extract-static-attributes` | Extract ~210 Caravan/HydroATLAS static attributes for a polygon file (alias: `extract-caravan-static`). |
| | `extract-static-attributes-batch` | Extract Caravan static attributes across multiple dataset folders or files (alias: `extract-caravan-static-batch`). |
| **Gridded Weather Archives** | `build-cpc-archive` | Build or update a global NOAA CPC daily precipitation Zarr archive. |
| | `build-imerg-archive` | Build or update a global NASA IMERG daily precipitation Zarr archive. |
| **Weather Time Series** | `extract-multimet` | Extract basin-averaged historical weather time series (single process). |
| | `extract-multimet-dask` | Extract basin-averaged historical weather time series in parallel with Dask. |
| | `multimet-realtime` | Fetch live Cold-Start or Hot-Start weather time series for basins. |
| | `sync-weather-forecasts` | Synchronize 10-day gridded weather forecasts into local binary stores. |
| **Return Periods** | *(Python API)* | Fit USGS Bulletin 17C flood return-period thresholds (`return_periods`). |
| **External Flood Forecasts** | `fetch-maas-forecast` | Fetch external flood forecasts from FloodHub, GloFAS v4, and GEOGLOWS v2. |
| **Hydrological Models** | `run` | Train, continue training, fine-tune, evaluate, or run inference with a model. |
| | `schedule-runs` | Run multiple training, evaluation, or fine-tuning jobs across GPUs. |
| **Web Interface** | `openhydronet-ui` | Launch the local interactive web application. |
| **Benchmarks** | `benchmark-catchment`, `benchmark-static-extractor`, `benchmark-gridded-archive`, `benchmark-timeseries-extractor`, `benchmark-return-periods`, `benchmark-model` | Run speed and numerical accuracy benchmarks for any subpackage. |

---

### 2.1 Step 1: Catchment Delineation (`multimet.catchment_delineation`)

Catchment delineation finds the boundary polygon of all land that drains into a river at a specific `(latitude, longitude)` point. All output polygons use WGS84 geographic coordinates (`EPSG:4326`) and include the drainage area in square kilometers (`km²`).

OpenHydroNet supports two 90-meter (`3-arc-second`) Digital Elevation Models (`--dem`):
- `hydrosheds_90m` (default) — HydroSHEDS 90m flow-direction tiles (`gs://open-multimet/ancillary-data/dems/tiles_5deg`).
- `merit_hydro_90m` — MERIT-Hydro 90m flow-direction tiles.

And five delineation modes (`--mode`):
- `dem` (default) — Pure 90m D8 flow-direction pixel tracing.
- `hybrid` — Snaps to a vector river reach (HydroRIVERS or MERIT-Basins) and traces the 90m D8 grid inside the outlet catchment.
- `ridgeline` — Dissolves upstream vector unit catchments (HydroBASINS Level 12 or MERIT-Basins).
- `exact_pour_point` — Combines upstream vector unit catchments with a 90m D8-clipped outlet unit catchment.
- `unit_catchment` — Returns only the local unit catchment polygon containing the point.

#### CLI Commands

**Delineate a single basin from Google Cloud Storage tiles (`delineate-catchment`)**:

```bash
delineate-catchment \
  --lat 38.85 \
  --lon -121.05 \
  --id my_basin_01 \
  --dem hydrosheds_90m \
  --mode dem \
  --gcs-uri gs://open-multimet/ancillary-data/dems/tiles_5deg \
  --cache-dir ~/.cache/openhydronet/data/hydrosheds_dir/tiles_5deg \
  --snap-window 12 \
  --output my_basin_01.geojson
```

**Delineate many basins from a CSV or Parquet file in parallel**:
Create a CSV file with columns `gauge_id,lat,lon`, then run:

```bash
delineate-catchment \
  --csv gauges.csv \
  --tiles-dir ~/.cache/openhydronet/data/hydrosheds_dir/tiles_5deg \
  --workers 8 \
  --output all_basins.geojson
```

**Key CLI flags for `delineate-catchment`**:
- `--lat`, `--lon`: Outlet latitude and longitude in decimal degrees (`EPSG:4326`).
- `--coords`: Space-separated list of `"lat,lon"` strings.
- `--csv`: Path (local or `gs://`) to a CSV or Parquet table of coordinates.
- `--id`: Basin ID for single-point delineation.
- `--dem`: `hydrosheds_90m` (default) or `merit_hydro_90m`.
- `--mode`: `dem` (default), `hybrid`, `ridgeline`, `exact_pour_point`, or `unit_catchment`.
- `--tiles-dir`: Local directory (or `gs://` URI) of `5° × 5°` `.npy` flow-direction tiles.
- `--gcs-uri` and `--cache-dir`: GCS tile URI and local cache folder for on-demand tile downloads.
- `--snap-window`: Search radius in 90m grid cells to snap the point to the highest-accumulation stream cell (default: `12`).
- `--expected-area`: Optional expected drainage area in `km²` to guide stream snapping.
- `-o`, `--output`: Output file path (`.geojson`, `.parquet`, `.geoparquet`, or `.shp`).

**Download MERIT-Hydro 90m D8 tiles from Earth Engine (`download-merit-d8-tiles`)**:

```bash
download-merit-d8-tiles \
  --target-dir ~/.cache/openhydronet/data/merit_hydro_dir/tiles_5deg \
  --ee-project your-gcp-project-id \
  --bbox -125 32 -114 42 \
  --workers 12
```

#### Python API

```python
from pathlib import Path
from multimet.catchment_delineation import (
    DemDelineator,
    RiverNetwork,
    delineate_hybrid,
)

# 1. Pure 90m DEM delineation (downloads needed 5x5 degree tiles from GCS on demand)
delineator = DemDelineator(
    gcs_uri="gs://open-multimet/ancillary-data/dems/tiles_5deg",
    cache_dir=Path("~/.cache/openhydronet/data/hydrosheds_dir/tiles_5deg").expanduser(),
)
feature = delineator.delineate_point(
    lat=38.85,
    lon=-121.05,
    catchment_id="my_basin_01",
    snap_window_cells=12,
)
print("Area (km²):", feature["properties"]["area_km2"])

# 2. Hybrid vector-river + 90m DEM delineation
rivers = RiverNetwork.from_hydrorivers(
    "~/.cache/openhydronet/data/HydroRIVERS_v10_shp/HydroRIVERS_v10.shp"
)
hybrid_feature = delineate_hybrid(
    dem=delineator,
    network=rivers,
    lat=38.85,
    lon=-121.05,
    catchment_id="my_basin_01",
)
```

---

### 2.2 Step 2: Static Catchment Attributes (`multimet.static_extractor`)

Hydrological models use static catchment attributes (such as drainage area, elevation, slope, soil clay fraction, forest cover, and long-term climate indices) to learn how different landscapes respond to rain. OpenHydroNet extracts ~210 Caravan attributes by combining **HydroATLAS Level 12** sub-basin properties (`196` attributes) with **ERA5-Land** climate indices (`14` attributes).

All required HydroATLAS and ERA5 climate tables are hosted in the public bucket `gs://open-multimet/ancillary-data/hydroatlas`.

#### CLI Commands

**Extract static attributes for one vector file (`extract-static-attributes` or `extract-caravan-static`)**:

```bash
# Stream HydroATLAS and ERA5 climate tables directly from GCS (--no-download)
extract-static-attributes \
  --input my_basin_01.geojson \
  --output my_basin_01_static.csv \
  --id-column gauge_id \
  --gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas \
  --gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate \
  --no-download \
  --workers 4
```

Or if you want to cache the HydroATLAS tables locally on disk for repeated runs:

```bash
extract-static-attributes \
  --input my_basin_01.geojson \
  --output my_basin_01_static.csv \
  --id-column gauge_id \
  --gdb-path ~/.cache/openhydronet/data/hydroatlas \
  --era5-cache-dir ~/.cache/openhydronet/data/era5_climate \
  --gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas \
  --gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate \
  --workers 4
```

**Extract static attributes across many dataset folders (`extract-static-attributes-batch` or `extract-caravan-static-batch`)**:

```bash
extract-static-attributes-batch \
  --input-files all_basins.geojson \
  --output-dir ./extracted_attributes \
  --id-column gauge_id \
  --gdb-path ~/.cache/openhydronet/data/hydroatlas \
  --era5-cache-dir ~/.cache/openhydronet/data/era5_climate \
  --gcs-gdb-uri gs://open-multimet/ancillary-data/hydroatlas \
  --gcs-era5-climate-uri gs://open-multimet/ancillary-data/hydroatlas/era5_climate \
  --partition-outputs \
  --combine \
  --workers 8
```

**Key CLI flags**:
- `--input`, `-i` (or `--input-files` / `--input-dirs` / `--parent-dir` in batch mode): Input polygon file(s) (`.geojson`, `.shp`, `.gpkg`, `.parquet`).
- `--output`, `-o` (or `--output-dir` in batch mode): Output CSV path or directory.
- `--id-column`: Column containing the basin ID (default: `gauge_id`).
- `--era5-source`: Climate source priority (`auto`, `timeseries`, `hybas`, or `gridded`).
- `--gdb-path` / `--gcs-gdb-uri`: Local path or GCS URI (`gs://open-multimet/ancillary-data/hydroatlas`) for HydroATLAS Level 12 data.
- `--era5-cache-dir` / `--gcs-era5-climate-uri`: Local path or GCS URI (`gs://open-multimet/ancillary-data/hydroatlas/era5_climate`) for precomputed ERA5 climate indices.
- `--gridded-era5-uri`: Optional path/URI to gridded daily ERA5-Land Zarr (`gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr`).
- `--no-download`: Stream HydroATLAS and ERA5 tables directly from GCS in memory without saving local copies.
- `--workers`, `-w`: Number of parallel worker processes.

#### Python API

```python
from multimet.static_extractor import StaticAttributesExtractor

extractor = StaticAttributesExtractor(
    gcs_gdb_uri="gs://open-multimet/ancillary-data/hydroatlas",
    gcs_era5_climate_uri="gs://open-multimet/ancillary-data/hydroatlas/era5_climate",
    no_download=True,
)

# Extract ~210 Caravan + HydroATLAS attributes to a DataFrame and CSV
df = extractor.extract_attributes_from_file(
    input_path="my_basin_01.geojson",
    output_csv_path="my_basin_01_static.csv",
    id_column="gauge_id",
    workers=4,
)
print(df[["area", "ele_mt_sav", "slp_dg_sav", "p_mean", "aridity_ERA5_LAND"]])
```

---

### 2.3 Step 3: Building Gridded Weather Archives (`multimet.gridded_archive_builders`)

If you want to build or extend your own global gridded daily precipitation Zarr archives from NOAA and NASA upstream servers, use `build-cpc-archive` and `build-imerg-archive`.

#### CLI Commands

**Build or extend the NOAA CPC unified gauge precipitation archive (`build-cpc-archive`)**:

```bash
build-cpc-archive \
  --target_zarr ./cpc_daily_surface.zarr \
  --start_year 1979 \
  --end_year 2025 \
  --num_workers 8
```

To extend an existing archive with the newest days published by NOAA PSL, pass `--extend_archive`.

**Build or extend the NASA IMERG Early V07 daily precipitation archive (`build-imerg-archive`)**:

```bash
# Requires NASA Earthdata credentials in ~/.netrc or via --earthdata_username / --earthdata_password
build-imerg-archive \
  --target_zarr ./imerg_daily_surface.zarr \
  --start_date 2000-06-01 \
  --end_date 2025-12-31 \
  --source gesdisc \
  --batch_size 30 \
  --num_workers 4
```

#### Python API

```python
from multimet.gridded_archive_builders.build_cpc_archive import build_cpc_archive
from multimet.gridded_archive_builders.build_imerg_archive import build_imerg_archive

# 1. Build or extend a NOAA CPC 0.5-degree daily precipitation Zarr archive
build_cpc_archive(
    target_zarr="./cpc_daily_surface.zarr",
    start_year=2023,
    end_year=2024,
    num_workers=4,
)

# 2. Build or extend a NASA IMERG 0.1-degree daily precipitation Zarr archive
build_imerg_archive(
    target_zarr="./imerg_daily_surface.zarr",
    start_date="2024-01-01",
    end_date="2024-01-31",
    source_type="gesdisc",
    num_workers=4,
)
```

---

### 2.4 Step 4: Weather Time-Series Extraction (`multimet.timeseries_extractors` and `multimet.weather_fetcher`)

To train or run a hydrological model on a basin, you need area-weighted weather time series (such as precipitation in `mm/day`, 2m temperature in `°C`, solar and thermal radiation in `W/m²`, and surface pressure in `kPa` or `Pa`) averaged over the basin polygon. Each extracted product is saved in Caravan-MultiMet layout at `<output_dir>/<PRODUCT>/timeseries.zarr`.

#### Historical Extraction (`extract-multimet` and `extract-multimet-dask`)

Use `extract-multimet` (single process) or `extract-multimet-dask` (parallel across CPU cores or a Dask cluster) to extract area-weighted time series for one or more basins.

Supported `--products` values:
- Core products (default): `CPC`, `ERA5_LAND`, `IMERG`, `HRES`
- Additional forecast & ensemble products: `AIFS`, `AIFS_ENS`, `GFS`, `GEFS`, `IFS_ENS`, `DYNAMICAL_IMERG`

**Serial CLI example (`extract-multimet`) reading from the public GCS gridded archives**:

```bash
extract-multimet \
  --basins_path my_basin_01.geojson \
  --output_dir ./multimet_extracted \
  --products CPC,ERA5_LAND,IMERG,HRES \
  --start_date 2020-01-01 \
  --end_date 2023-12-31 \
  --source archive \
  --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --archive-store ERA5_LAND=gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr \
  --archive-store IMERG=gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
  --archive-store HRES=gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr \
  --id_column gauge_id
```

**Parallel Dask CLI example (`extract-multimet-dask`)**:

```bash
extract-multimet-dask \
  --basins_path all_basins.geojson \
  --output_dir ./multimet_extracted \
  --products CPC,ERA5_LAND,IMERG,HRES \
  --start_date 2015-01-01 \
  --end_date 2023-12-31 \
  --source archive \
  --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --archive-store ERA5_LAND=gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr \
  --archive-store IMERG=gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
  --archive-store HRES=gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr \
  --num-workers 8 \
  --memory-limit 8GB \
  --id-column gauge_id
```

#### Real-Time Extraction for Operational Forecasting (`multimet-realtime`)

When running a live flood forecast today (`reference_date`), `multimet-realtime` supports two modes (`--mode`):
1. **`coldstart`** — Fetches a **365-day historical spin-up window** (`[t₀ - 365d, t₀]`) plus the **10-day operational forecast** issued on `t₀`.
2. **`hotstart`** (default) — Inspects your existing `<output_dir>/<PRODUCT>/timeseries.zarr` stores and downloads only the newly elapsed days up to `t₀` plus the new forecast.

```bash
# Cold-Start: fetch 365 days of spin-up + latest 10-day forecast
multimet-realtime \
  --basins_path my_basin_01.geojson \
  --output_dir ./realtime_forcings \
  --mode coldstart \
  --reference_date latest \
  --products HRES,IMERG,CPC \
  --id_column gauge_id

# Hot-Start: incrementally update an existing store with today's new weather
multimet-realtime \
  --basins_path my_basin_01.geojson \
  --output_dir ./realtime_forcings \
  --mode hotstart \
  --reference_date latest \
  --products HRES,IMERG,CPC
```

Supported `--products` for `multimet-realtime`: `HRES`, `IMERG`, `CPC`, `AIFS`, `AIFS_ENS`, `GFS`, `GEFS`, `IFS_ENS`, `DYNAMICAL_IMERG`.

#### Synchronizing Gridded Weather Forecasts for the Map Viewer (`sync-weather-forecasts`)

Use `sync-weather-forecasts` to download the latest operational gridded weather runs (`ecmwf_ifs`, `ecmwf_aifs`, `noaa_gfs`, `noaa_gefs`, `noaa_hrrr`, `nasa_imerg`, `ecmwf_hres`, `noaa_cpc`) into local `float16` binary streams for the Weather Viewer map:

```bash
# Synchronize all 8 gridded weather models
sync-weather-forecasts --data-dir ~/.cache/openhydronet/weather

# Synchronize only ECMWF HRES and NOAA GFS
sync-weather-forecasts \
  --data-dir ~/.cache/openhydronet/weather \
  --models ecmwf_hres,noaa_gfs

# Print current sync status as JSON
sync-weather-forecasts --data-dir ~/.cache/openhydronet/weather --status
```

#### Python API

```python
from pathlib import Path
from multimet.timeseries_extractors import (
    extract_multimet_serial,
    extract_multimet_dask,
    fetch_realtime_multimet,
)
from multimet.weather_fetcher.sync import sync_all_models, read_sync_status

# 1. Extract historical weather time series from public GCS gridded archives
stores = extract_multimet_serial(
    basins="my_basin_01.geojson",
    output_dir="./multimet_extracted",
    products=["CPC", "ERA5_LAND"],
    start_date="2020-01-01",
    end_date="2022-12-31",
    source="archive",
    archive_stores={
        "CPC": "gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr",
        "ERA5_LAND": "gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr",
    },
    id_column="gauge_id",
)
print("Extracted Zarr stores:", stores)

# 2. Fetch live Cold-Start or Hot-Start weather forcings
rt_result = fetch_realtime_multimet(
    basins="my_basin_01.geojson",
    output_dir="./realtime_forcings",
    mode="coldstart",
    reference_date="latest",
    products=["HRES", "CPC"],
    id_column="gauge_id",
)
print(rt_result.summary())

# 3. Synchronize gridded weather forecast maps
sync_status = sync_all_models(
    data_dir=Path("~/.cache/openhydronet/weather").expanduser(),
    models=["ecmwf_hres", "noaa_gfs"],
)
```

---

### 2.5 Step 5: Flood Return Periods (`return_periods`)

A **return period** (such as the 2-year, 5-year, 20-year, or 100-year flood) tells you how severe a river flow value is compared to historical annual peak floods. For example, a 20-year flood has a `1 / 20 = 5%` chance of being equaled or exceeded in any given year.

OpenHydroNet implements the official **USGS Bulletin 17C** methodology (`ReturnPeriodCalculator`):
1. **Multiple Grubbs-Beck Test (`MultipleGrubbsBeckTester`)** — Screens out low drought-year outliers (*Potentially Influential Low Floods*, or PILFs).
2. **Expected Moments Algorithm (`GEMAFitter`, `fitter='gema'`)** — Fits a Log-Pearson Type III (`LP-III`) extreme-value distribution. Also supports `fitter='simple_lp3'` (`SimpleLogPearson3Fitter`) and `fitter='log_linear'` (`LogLogTrendFitter`).

#### Python API

```python
import pandas as pd
from return_periods import ReturnPeriodCalculator, MultipleGrubbsBeckTester

# Option A: Fit return periods directly from an Annual Maximum Series (m³/s)
annual_peaks = pd.Series(
    [2080, 1670, 1480, 2940, 1560, 2380, 2720, 2860, 2620, 1710, 3150, 1890],
    index=range(2012, 2024),
)
rpc = ReturnPeriodCalculator(peaks_series=annual_peaks, fitter="gema")

# Compute discharge thresholds for 2, 5, 10, 20, 50, and 100-year floods
return_periods = [2, 5, 10, 20, 50, 100]
flood_thresholds = rpc.flow_values_from_return_periods(return_periods)
for rp, q_val in zip(return_periods, flood_thresholds):
    print(f"{rp}-year flood: {q_val:.1f} m³/s")

# Convert any flow value back into an estimated return period (in years)
estimated_rps = rpc.return_periods_from_flow_values([2500.0, 3200.0])

# Option B: Fit return periods from a daily streamflow Series (extracts annual peaks)
# Use water_year_start_month=10 for October-September water years
daily_dates = pd.date_range("1990-01-01", "2023-12-31", freq="D")
daily_flow = pd.Series(100.0, index=daily_dates)
rpc_daily = ReturnPeriodCalculator(
    hydrograph_series=daily_flow,
    fitter="gema",
    max_missing_days_in_year=30,
    water_year_start_month=10,
)

# Option C: Run standalone Multiple Grubbs-Beck low-outlier screening
tester = MultipleGrubbsBeckTester(data=annual_peaks.values, is_log_transformed=False)
print("Number of low outliers (klow):", tester.klow)
```

---

### 2.6 Step 6: External Global Flood Models (`maas`)

The **Models-as-a-Service (`maas`)** package fetches operational river discharge forecasts, historical reanalysis, and flood return-period thresholds from three global flood forecasting systems for any `(latitude, longitude)` point:

1. **Google FloodHub** (`floodhub`, `FloodHubClient`) — Gauge- and reach-level streamflow forecasts and severity levels via `https://floodforecasting.googleapis.com/v1`.
2. **Copernicus GloFAS v4** (`glofas`, `GloFASClient`) — ECMWF-forced LISFLOOD global `0.05°` river discharge forecasts (deterministic control run, ensemble mean/median/percentiles, and all 50 ensemble members) via the Open-Meteo Flood API (`https://flood-api.open-meteo.com/v1/flood`) and local Zarr or NetCDF archives.
3. **GEOGLOWS ECMWF v2** (`geoglows`, `GeoGLOWSClient`) — Reach-level streamflow forecasts on ~6.8M global TDX-Hydro reaches (`LINKNO`) via `https://geoglows.ecmwf.int/api/v2` and AWS S3 retrospective Zarr (`s3://geoglows-v2/retrospective.zarr`).

All models return river discharge in cubic meters per second (**`m³/s`**).

#### CLI Command (`fetch-maas-forecast`)

```bash
fetch-maas-forecast \
  --lat 38.6270 \
  --lon -90.1994 \
  --models floodhub,glofas,geoglows \
  --cache-dir ~/.cache/openhydronet/maas \
  --river-networks-dir data/river_networks \
  --output maas_forecast_bundle.json
```

**Key CLI flags for `fetch-maas-forecast`**:
- `--lat`, `--lon`: Target latitude and longitude in decimal degrees (`EPSG:4326`, required).
- `--cache-dir`: Directory for SQLite return-period caches (required).
- `--river-networks-dir`: Directory containing static river network pyramids (defaults to `--cache-dir`).
- `--models`: Comma-separated list of providers (`floodhub,glofas,geoglows`).
- `--gauge-id`: Optional Google FloodHub gauge ID.
- `--river-id`: Optional GEOGLOWS v2 9-digit reach ID (`LINKNO`).
- `--reach-id`: Optional HydroRIVERS reach ID (`HYRIV_ID`).
- `--floodhub-api-key`: Optional Google FloodHub API key (or set `$FLOODHUB_API_KEY`).
- `--timeout`: HTTP request timeout in seconds (default: `12.0`).
- `--output`: Output JSON file path (prints to stdout if omitted).

#### Python API

```python
from pathlib import Path
from maas import (
    MaaSConfig,
    MaaSDataFetcher,
    fetch_forecasts,
    fetch_historical,
    fetch_return_periods,
    resolve_reaches,
)

config = MaaSConfig(
    cache_dir=Path("~/.cache/openhydronet/maas").expanduser(),
    river_networks_dir=Path("data/river_networks"),
    floodhub_api_key="",
)
fetcher = MaaSDataFetcher(config)

# 1. Resolve matching river reaches across GEOGLOWS, GloFAS, and HydroRIVERS
reaches = resolve_reaches(38.6270, -90.1994, config=config)
print("Resolved reaches:", reaches)

# 2. Fetch multi-model operational forecasts & aligned daily series (in m³/s)
# glofas_reference can be 'control', 'median', or 'mean'
forecasts = fetch_forecasts(
    config,
    38.6270,
    -90.1994,
    models=["floodhub", "glofas", "geoglows"],
    glofas_reference="control",
)

# 3. Fetch historical reanalysis and return-period thresholds (q_2yr .. q_100yr)
geoglows_id = reaches["geoglows"]["river_id"]
historical = fetch_historical(config, "geoglows", reach_id=geoglows_id)
thresholds = fetch_return_periods(config, "geoglows", reach_id=geoglows_id)
```

---

### 2.7 Step 7: Hydrological Modeling (`model`)

The `model` package trains, fine-tunes, evaluates, and runs inference with deep learning river forecasting models built in PyTorch.

#### Supported Model Architectures (`model:` in `config.yml`)
- **`mean_embedding_forecast_lstm`** (`MeanEmbeddingForecastLSTM`) — Encodes static attributes, hindcast weather products, and forecast weather products with separate multilayer perceptrons (`statics_embedding`, `hindcast_embedding`, `forecast_embedding`). At each day, it averages the available weather product embeddings so the model keeps working even when one weather source is missing, then runs a sequential hindcast-to-forecast LSTM with a probabilistic CMAL output head (`CountableMixtureAsymmetricLaplace`). Replicates the 2025 generation of Google FloodHub (a former operational model).
- **`handoff_forecast_lstm`** (`HandoffForecastLSTM`) — Runs a hindcast LSTM over lookback weather (`seq_length` days) and passes its final hidden and cell states through a small feed-forward handoff network to initialize a separate forecast LSTM over the future `lead_time` days. Replicates the 2024 generation of Google FloodHub (a former operational model).

#### CLI Commands (`run` and `schedule-runs`)

All modeling actions use a YAML configuration file (`config.yml`). Example configurations are located in `model/example-configs/`:
- `model/example-configs/floodhub-settings-config.yml` — Global `mean_embedding_forecast_lstm` setup.
- `model/example-configs/handoff-forecast-lstm-config.yml` — Global `handoff_forecast_lstm` setup.
- `model/example-configs/template_hot_start_config.yml` — Template for saving (`save_state`) and loading (`hot_start_path`) daily LSTM hidden states.
- `model/example-configs/camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml` — Setup with variational streamflow data assimilation (`assimilation_config`).

**1. Train a new model (`run train`)**:

```bash
run train --config-file model/example-configs/floodhub-settings-config.yml --gpu 0
```

Flags for `--gpu`:
- `--gpu 0`: Use GPU device `0` (`cuda:0`).
- `--gpu -1`: Force CPU execution (`cpu`).
- Omit `--gpu`: Use the `device:` setting inside `config.yml`.

**2. Continue training an interrupted run (`run continue_training`)**:

```bash
run continue_training --run-dir ./runs/my_experiment_run --gpu 0
```

**3. Fine-tune a pre-trained model on local basins (`run finetune`)**:

```bash
run finetune --config-file finetune_config.yml --gpu 0
```

In `finetune_config.yml`, set `base_run_dir` to the path of a pre-trained checkpoint directory (such as `model/pretrained-models/google-floodhub-settings-110-epochs`) and list the neural network modules you want to update under `finetune_modules` (valid `module_parts` for `mean_embedding_forecast_lstm`: `static_embedding_fc`, `local_static_embeddings_fc`, `hindcast_embeddings_fc`, `forecast_embeddings_fc`, `shared_embeddings_fc`, `hindcast_lstm`, `forecast_lstm`, `head`). See [`model/pretrained-models/Pretrained-Models-README.md`](./model/pretrained-models/Pretrained-Models-README.md) for a complete template.

**4. Evaluate a trained model on validation or test data (`run evaluate`)**:

```bash
run evaluate \
  --run-dir ./runs/my_experiment_run \
  --period test \
  --epoch 110 \
  --gpu 0
```

This computes the evaluation metrics listed under `metrics:` in the run configuration (for example `NSE` and `KGE`) on days with observed streamflow and writes predictions and metrics to `<run-dir>/test/model_epoch110/`. Pass `--assimilate` to enable variational data assimilation during evaluation.

**5. Run operational inference on all dates (`run infer`)**:

```bash
# Standard forecast inference (runs even when target streamflow observations are NaN)
run infer \
  --run-dir ./runs/my_experiment_run \
  --epoch 110

# Forecast inference with variational streamflow data assimilation enabled
# (the run's config.yml must contain an `assimilation_config` section)
run infer \
  --run-dir ./runs/my_experiment_run \
  --epoch 110 \
  --assimilate
```

**6. Convert Caravan NetCDF files to Zarr (`run convert-caravan`)**:

```bash
run convert-caravan \
  --caravan-dir ~/data/Caravan-nc \
  --output-dir ~/data/Caravan-zarr
```

**7. Schedule multiple runs across multiple GPUs (`schedule-runs`)**:
If you have a directory containing several `.yml` config files (for `train` or `finetune`) or several run directories (for `evaluate`), run them in parallel across your GPUs:

```bash
schedule-runs train \
  --directory ./my_config_files/ \
  --gpu-ids 0 1 2 3 \
  --runs-per-gpu 2
```

#### Python API

```python
from pathlib import Path
from model.utils.config import Config
from model.run import start_run, continue_run, finetune, eval_run
from model.datautils.convert import convert_caravan_to_zarr

# 1. Convert Caravan NetCDF archive to Zarr from Python
convert_caravan_to_zarr(
    Path("~/data/Caravan-nc").expanduser(),
    Path("~/data/Caravan-zarr").expanduser(),
)

# 2. Load and customize a YAML configuration in Python
cfg = Config(Path("model/example-configs/floodhub-settings-config.yml"))
cfg.update_config({"epochs": 5, "device": "cpu", "run_dir": Path("./runs")})

# 3. Train a model
start_run(config=cfg, gpu=-1)

# 4. Fine-tune a pre-trained model using a fine-tune config file
finetune(config_file=Path("finetune_config.yml"), gpu=-1)

# 5. Evaluate or run inference on a trained run directory
run_dir = Path("./runs/my_experiment_run")
run_cfg = Config(run_dir / "config.yml")
run_cfg.inference_mode = True
run_cfg.tester_skip_obs_all_nan = False
eval_run(
    config=run_cfg,
    run_dir=run_dir,
    period="test",
    epoch=110,
    gpu=-1,
    assimilate=False,
)
```

---

### 2.8 Step 8: Running Benchmarks (`benchmarks`)

You can verify speed and numerical accuracy across every subpackage against reference datasets using the six built-in benchmark CLIs (see [`benchmarks/README.md`](./benchmarks/README.md) for full details):

```bash
# 1. Benchmark catchment delineation against reference Caravan polygons
benchmark-catchment \
  --dataset /path/to/benchmark_basins_500.parquet \
  --tiles-dir ~/.cache/openhydronet/data/hydrosheds_dir/tiles_5deg \
  --workers 8 \
  --output /tmp/catchment_results.parquet

# 2. Benchmark Caravan static attribute extraction against 210 reference attributes
benchmark-static-extractor \
  --dataset /path/to/benchmark_basins_500.parquet \
  --gdb-path ~/.cache/openhydronet/data/hydroatlas \
  --era5-source hybas \
  --era5-cache-dir ~/.cache/openhydronet/data/era5_climate \
  --workers 8 \
  -o /tmp/static_benchmark_out

# 3. Benchmark CPC or IMERG gridded archive builders against reference Zarr stores
benchmark-gridded-archive \
  --product CPC \
  --start-date 2020-01-01 --end-date 2020-01-31 \
  --reference-zarr gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --output-dir /tmp/cpc_archive_bench

# 4. Benchmark weather time-series extraction against canonical Caravan-MultiMet v1.1
benchmark-timeseries-extractor \
  --dataset /path/to/benchmark_basins_500.parquet \
  --canonical-dir gs://caravan-multimet/v1.1 \
  --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
  --date-windows 2016-01-01:2023-12-31 \
  --output-dir /tmp/timeseries_bench

# 5. Benchmark USGS Bulletin 17C return-period calculator
benchmark-return-periods \
  --mode live \
  --caravan-dir ~/data/Caravan-zarr \
  --output-dir /tmp/return_periods_bench

# 6. Benchmark hydrological model architectures and hot-start state handoff
benchmark-model \
  --mode all \
  --basin-file /path/to/basins_25.txt \
  --caravan-dir ~/data/Caravan-zarr \
  --canonical-multimet-dir /path/to/canonical_zarr \
  --reconstructed-multimet-dir /path/to/reconstructed_zarr \
  --output-dir /tmp/model_bench
```

---

## Section 3: Using the Interactive Web UI

OpenHydroNet includes a full interactive web application (`openhydronet-ui`) so you can run the entire workflow visually in your browser.

---

### 3.1 Starting the Web Server

Run this command in your terminal:

```bash
openhydronet-ui --host 127.0.0.1 --port 8080
```

Then open **`http://127.0.0.1:8080`** in your web browser.

> **Tip for Remote Servers / SSH:** If you are running on a remote Linux server, forward port `8080` over SSH (`ssh -L 8080:127.0.0.1:8080 user@remote-server`), then open `http://localhost:8080` on your laptop.

---

### 3.2 Workspace Layout and Navigation

When the page loads, you see three main areas:

1. **Top navigation bar**:
   - It has 8 workflow tabs: `Weather Viewer`, `Models-as-a-Service`, `Basin Delineation`, `Geographical Features`, `Training`, `Fine Tuning`, `Forecasting and Data Assimilation`, and `Account`. The `Account` tab is at the right end of the bar and shows the active account, for example `Account (Guest)`.
   - Drag to reorder: click and drag any tab left or right to change the tab order. Your browser saves the order.
   - Activity spinners: when a background job (such as a weather refresh, attribute extraction, or fine-tuning) is running in a tab, a spinner appears next to the tab name.
   - Stop downloads: while a download is running, a red `Stop Downloads` button appears in the tab bar, just left of the `Account` tab. Click it to stop all active background downloads.
2. **Left control sidebar**:
   - The controls change with the active tab.
   - Hide or show: click `Hide` at the top of the sidebar, click the panel button at the left end of the tab bar, or press the `[` key. When the sidebar is hidden, click the `Controls` button at the left edge of the screen to open it again.
   - Resize: drag the right border of the sidebar to make it wider or narrower.
3. **Main area**:
   - The `Weather Viewer`, `Models-as-a-Service`, `Basin Delineation`, and `Geographical Features` tabs show an interactive Leaflet map.
   - The `Fine Tuning` and `Forecasting and Data Assimilation` tabs show interactive Chart.js charts.
   - The `Account` tab shows the data saved in your account and a table of your basins.

---

### 3.3 Tab 1: Weather Viewer

The Weather Viewer tab shows animated weather forecasts and recent precipitation observations on the map.

#### Left Sidebar Controls
- Model: choose one of 8 weather products. All products are shown on the same global 0.25° grid, for up to 10 days (240 hours) in 3-hour steps.
  - ECMWF HRES (`ecmwf_hres`): 0.25° global deterministic forecast, 10 days.
  - ECMWF IFS ENS (`ecmwf_ifs`): control member of the ECMWF ensemble forecast, 0.25°, first 10 days.
  - ECMWF AIFS (`ecmwf_aifs`): 0.25° AI-driven global forecast, 10 days in 6-hour steps.
  - NOAA GFS (`noaa_gfs`): 0.25° global deterministic forecast, first 10 days.
  - NOAA GEFS (`noaa_gefs`): control member of the NOAA ensemble forecast, 0.25°, first 10 days.
  - NOAA HRRR (`noaa_hrrr`): 3 km forecast for the continental US, 48 hours.
  - NASA IMERG (`nasa_imerg`): 0.1° satellite precipitation (Early Run), past 10 days.
  - NOAA CPC (`noaa_cpc`): 0.5° gauge-based precipitation over land, past 10 days.
- Sync status and refresh:
  - Each model button shows a tag: `run <date>` (a synced forecast run), `obs <date>` (synced observations), or `not synced`. Hold the mouse over a button to see the run time and the forecast length.
  - Click `Refresh All` at the top of the sidebar to sync all weather models, or click the small ↻ icon on a model button to sync only that model.
- Variable (the list depends on the model):
  - `Rain rate` (mm/h)
  - `Rain accumulated since forecast start` (mm). For NASA IMERG and NOAA CPC, this option is `Rain accumulated (past 10 days)`.
  - `Air temperature (2 m)` (°C)
  - `Sea-level pressure` (hPa)
  - NASA IMERG and NOAA CPC have only the two rain options.
- Layers:
  - `Wind particles`: turn the animated wind particles on or off. This switch appears only for models with wind data.
  - `Colour opacity`: drag the slider (10% to 100%, default 80%) to make the weather colours more or less transparent.
  - `Show Catchments`: show or hide your basin outlines on the weather map.

#### Map Controls and Point Forecast Drawer
- Time bar (below the map): click `Play` to animate the forecast. Use the ‹ and › buttons to move one time step back or forward, the ⏮ button to go back to the forecast start, and the `1x` button to change the playback speed (1x, 2x, or 4x).
- Point forecast: click any location on the map to open the `Point forecast` drawer below the time bar. It plots the rain rate of every model at that point, plus the 2 m temperature of the selected forecast model (ECMWF HRES when IMERG or CPC is selected). The time axis runs from 240 hours in the past (observations) to 240 hours ahead (forecasts).

---

### 3.4 Tab 2: Models-as-a-Service

The Models-as-a-Service tab shows live river flood alerts and compares discharge forecasts from three external flood models.

#### Left Sidebar Controls
- River networks (1 to 3 at a time):
  - Click one or more model buttons to show their river reaches on the map. GloFAS is shown by default.
    - `FloodHub`: Google FloodHub (HydroSHEDS HydroRIVERS network)
    - `GloFAS`: Copernicus GloFAS v4 (LISFLOOD 0.05° river grid)
    - `GEOGLOWS`: GEOGLOWS ECMWF v2 (TDX-Hydro stream network)
  - Click `Refresh` to fetch the newest live forecasts.
- Return-period colours: river reaches are coloured by their forecast return period. Each model uses its own colours:
  - GloFAS (default): Blue = Normal (< 2-yr), Yellow = ≥ 2-yr Medium, Red = ≥ 5-yr High, Purple = ≥ 20-yr Severe.
  - FloodHub: Blue = Normal (< 2-yr), Orange = Warning (≥ 2-yr), Red = Danger (≥ 5-yr), Dark red = Extreme (≥ 20-yr), grey dashed line = no forecast.
  - GEOGLOWS: Blue = Normal (< 2-yr), Yellow = ≥ 2-yr, Orange = ≥ 10-yr, Red = ≥ 25-yr, Purple = ≥ 50-yr.
  - When two or three networks are shown at once, normal rivers use each network's own colour, and alerts use Yellow (≥ 2-yr), Red (≥ 5-yr), and Purple (≥ 20-yr).

#### Forecast View
- Click any river reach on the map. The map moves to the left, and the forecast view opens on the right. Clicks on empty map areas do not open a forecast.
- The hydrograph shows discharge in `m³/s`. Use the model buttons above the chart to compare Google FloodHub, Copernicus GloFAS, and GEOGLOWS ECMWF side by side. Dashed horizontal lines show each model's 2-yr, 5-yr, and 20-yr return periods.
- GloFAS reference: switch the main GloFAS line between `Control (Unperturbed)`, `Ensemble Median`, and `Ensemble Mean`. GEOGLOWS has its own list (`Ensemble Median`, `Ensemble Mean`, `High-Res (HRES)`).
- Ensemble options: click `+ Ensemble` to show every ensemble member (50 for GloFAS, 51 for GEOGLOWS), or `+ Range` to shade the Min–Max ensemble range.
- Click × or press `Esc` to close the forecast view.

---

### 3.5 Tab 3: Basin Delineation

The Basin Delineation tab lets you create watershed polygons by clicking on the map or by uploading a GeoJSON file.

#### Step-by-Step Workflow
1. **Select a river network and DEM**:
   - Choose `HydroRIVERS` (HydroSHEDS 90 m D8 DEM) or `MERIT-Basins` (MERIT-Hydro 90 m D8 DEM with vector unit catchments).
2. **Choose a basemap**:
   - Choose `Map`, `Satellite`, `OpenStreetMap`, `HydroSHEDS 90m DEM`, or `OpenTopoMap`. Use the `River Network Opacity` slider (20% to 100%, default 90%) to make the blue river lines lighter or darker.
3. **Turn on Delineate Mode**:
   - Click the `Delineate Mode` button in the sidebar, or press `D` while this tab is open. The mouse cursor changes to a crosshair. Press `Esc` to turn the mode off.
4. **Set the river snap search radius**:
   - Drag the `River Snap Search Radius` slider (0.1 km to 10.0 km, default 1.0 km). When you click near a river, OpenHydroNet moves your click onto a nearby river reach within this radius (it prefers larger rivers). This makes the basin start on the river channel, not on a hillside.
5. **Click the map to delineate**:
   - Click your outlet point on the map. After a few seconds, the upstream watershed boundary appears on the map and is added to the `Active Watersheds` list.
6. **Upload a custom basin (optional)**:
   - If you already have a polygon file, click `Upload Custom Basin (GeoJSON)` to import a `.geojson` or `.json` file (`Feature`, `FeatureCollection`, or a raw `Polygon`/`MultiPolygon` geometry, in `EPSG:4326` longitude/latitude degrees).
7. **Inspect a basin**:
   - Click a basin in the `Active Watersheds` list to select it, or click its zoom icon to zoom to it. The `Selected Basin` card shows the source, upstream sub-units, drainage area (km²), stream order, and outlet or centroid coordinates.
   - To rename or delete basins, use the `Account` tab (Section 3.10).

---

### 3.6 Tab 4: Geographical Features

The Geographical Features tab extracts and maps the ~210 Caravan/HydroATLAS static physical attributes of your delineated basins.

#### Step-by-Step Workflow
1. **Select basins**:
   - In the sidebar list, check the basins you want to process. Use the quick-select buttons: `Unextracted` (basins without saved attributes), `All`, or `None`.
2. **Extract attributes**:
   - Click `Extract Attributes`. Each basin shows `Extracting…` while it runs, and the button shows how many basins are done. When a basin is finished, its row shows `✓ Saved` and the value of the attribute selected under `Map Attribute`.
3. **View and export the attribute table**:
   - Click `Attribute Table` (below the `Extract Attributes` button) to open a full-screen table. Choose a basin in the table's dropdown.
   - Filter attributes by category (`All`, `Topography`, `Climate`, `Soils & Geology`, `Land Cover`, `Hydrology`, `Anthropogenic`) or type in the search box.
   - Click `Export CSV` or `Export JSON` in the table window to download the attributes of that basin. The `CSV` button next to `Attribute Table` downloads one CSV file with all saved attributes of your account.
4. **Map an attribute**:
   - Choose an attribute in the `Map Attribute` dropdown (for example mean elevation, slope, clay content, or forest fraction).
   - Under `Show on map`, choose `My Basins` (colour each of your basins by its value), `HydroATLAS` (colour the HydroATLAS sub-basins in the map view), or `Both`. For `HydroATLAS`, you can also set the `Sub-basin Level` (`Auto (Zoom)`, levels 5–10, or 12) and the opacity (15% to 90%, default 65%).

---

### 3.7 Tab 5: Training

The Training tab extracts multi-year historical weather time series for your basins and adds them to one multi-basin training Zarr archive.

#### Step-by-Step Workflow
1. **Choose a weather source**:
   - Under `Historical Weather Archives`, pick one source in the `Weather Repository` dropdown: NOAA CPC, NASA GPM IMERG, ECMWF ERA5, ECMWF IFS HRES, GraphCast, ECMWF ERA5-Land, CHIRPS v2.0, or Copernicus CERRA. Each extraction uses one source.
2. **Set the dates and scope**:
   - Enter the `Start Date` and `End Date` (`YYYY-MM-DD`), or click one of the `Dataset Quick Presets`.
   - If you have more than one basin, choose the `Extraction Scope`: all your basins (`All N Active Basins`) or only the selected basin (`Selected (<basin>)`).
   - *(Optional)* Enter an email address in `Notification Email (optional)` to get an email when a long extraction finishes.
3. **Extract**:
   - Click `Extract Historical Data`. The job runs in the background, and the result card below the button shows its status. Click `Stop` at any time.
   - Click `Browse Zarr Archives` (at the top of the main area) to see your saved Zarr stores with their variables, number of basins and time steps, and size on disk.

---

### 3.8 Tab 6: Fine Tuning

The Fine Tuning tab adapts a pre-trained global flood model to one local watershed. It uses your own streamflow gauge records, together with global weather data extracted for the basin and/or your own local rain and temperature CSVs.

#### Step-by-Step Workflow (Left Sidebar Steps 1–5)

1. **Step 1 — Base Model Inventory**:
   - Select a pre-trained global model checkpoint from the dropdown (`google-floodhub-settings-110-epochs`, `google-floodhub-settings-no-graphcast-110-epochs`, or `google-floodhub-settings-aifs-110-epochs`). Models that you saved from this tab are also listed. The card shows the model architecture, head and loss, sequence length and lead time, and checkpoint epoch.
2. **Step 2 — Basin Data & Forcings**:
   - Select your basin in the `Clipped Watershed Polygon (Account)` dropdown.
   - The `Global Forcing Archive` card shows which global weather products the selected model needs and whether they are already extracted for your basin. Click `Extract Global Forcings for Catchment` to extract the historical weather time series that the base model needs for your basin. A progress bar shows the extraction. Click `Stop` to cancel it.
   - Choose the `Input Forcing Source`:
     - `Global Data Only (HRES / IMERG / CPC / AIFS)` — uses only the extracted global weather data.
     - `Global Data + Uploaded Local Inputs` — combines the global weather data with your uploaded local precipitation and temperature CSVs.
     - `Uploaded Local Inputs Only` — uses only your uploaded local weather CSVs.
     - If you upload only a streamflow CSV, `Global Data Only` is used automatically.
   - Upload the target streamflow under `Target Streamflow (Required)`: a CSV with a date column and one value column (for example `date,streamflow`). Choose its units in `CSV / Display Units` (`m³/s` or `mm/day`). If you choose `m³/s`, OpenHydroNet converts the discharge to `mm/day` with your basin's drainage area in km² (shown in `Area km² (Auto)`).
   - *(Optional)* Under `Optional Local Input CSVs`, upload `Local Precipitation (mm/day)` and `Local Temperature (°C)` files. Click `Download CSV Template` for an example of the `date,lead_1,…,lead_7` layout (one row per date, one column per forecast lead day).
3. **Step 3 — Train / Validation Split**:
   - Drag the slider (`20%` to `90%`, default `70%`) to split your streamflow dates, in time order, into a Training period and a Validation period. The start and end dates of each period update below the slider.
4. **Step 4 — Tunable Model Components**:
   - Choose which network layers to unfreeze during fine-tuning (frozen layers keep their pre-trained global weights):
     - Use the preset buttons `Local + Head`, `Local + Forecast LSTM + Head`, or `Unfreeze All`, or check single modules (`static_embedding_fc`, `local_static_embeddings_fc`, `hindcast_embeddings_fc`, `forecast_embeddings_fc`, `shared_embeddings_fc`, `hindcast_lstm`, `forecast_lstm`, `head`).
     - The badge at the top of this step shows how many parameters will be trained.
5. **Step 5 — Hyperparameters & Step Control**:
   - Set `Epochs to Run Next` (default `3`), `Learning Rate` (default `0.001`), `Loss`, `LR Schedule` (`ConstantLR`, `ReduceLROnPlateau`, or `StepLR`), and `Batch Size` (default `16`). The `Loss` choices depend on the model head: CMAL-head models, such as the FloodHub checkpoints, use `CMALLoss`; other models offer `MSE`, `NSE`, `RMSE`, and `WMSE`.
   - Click `Run N Epochs & Pause`. The model trains for `N` epochs, saves a checkpoint after each epoch, updates the charts, and pauses so you can inspect the results. Click it again to train for another `N` epochs, or click `Reset` to go back to Epoch 0 (the pre-trained weights).

#### Right-Hand Fine-Tuning Dashboard
- Epoch Inspector: drag the epoch slider at the top (or click a row in the `Epoch Checkpoint History` table at the bottom) to see the hydrograph and metrics of any saved epoch, from Epoch 0 (the pre-trained baseline) to the latest epoch.
- Save a model: click `Save Epoch N to Inventory` (or the save button in a table row), type a name, and click `Save to Inventory`. The saved model then appears in the model lists of the `Fine Tuning` and `Forecasting and Data Assimilation` tabs.
- 5 KPI cards: `Train Loss`, `Validation Loss`, `Train NSE`, `Validation NSE`, and `Validation KGE`.
- Input forcing chart: precipitation bars (mm/day) and 2 m temperature lines (°C) for all global and local forcing products over the fine-tuning period. Click products in the legend to show or hide them on the chart. This does not change the model inputs.
- Streamflow hydrograph: observed streamflow, the pre-trained base model (Epoch 0), and the fine-tuned model at the inspected epoch. Shading and a dashed vertical line separate the Training and Validation periods. Use the `Display Units` dropdown in the chart header to switch between `m³/s` and `mm/day`.

---

### 3.9 Tab 7: Forecasting and Data Assimilation

The Forecasting and Data Assimilation tab makes real-time flood forecasts for your basins, assimilates live gauge observations, and lets you run interactive "What-If" weather scenarios.

#### Step-by-Step Workflow (Left Sidebar Steps 1–5)

1. **Step 1 — Select Custom Catchments & Start Mode**:
   - Check one or more basins in the catchment list (`All`, `None`).
   - Choose the `Start Mode`:
     - `Cold Start` — runs the full 365-day hindcast lookback from scratch.
     - `Hot Start (t₀ − 1d)` — loads a saved LSTM state (normally from yesterday, `t₀ − 1d`) and runs only the days since that state. This button works only when a valid saved state exists for all selected basins.
   - The panel shows the forecast issue date (`t₀`), the required state date (`t₀ − 1d`), and the date of the saved state file, so you can see whether Hot Start is possible.
2. **Step 2 — Fetch Real-Time Weather Data**:
   - Check the real-time weather products you want to download (`ECMWF HRES`, `NASA IMERG`, `NOAA CPC`, `ECMWF AIFS`, `NOAA GFS`, `NOAA GEFS`, `ECMWF IFS-ENS`).
   - Click `Fetch Real-Time Data`. Each product shows a live download progress bar. Click `Stop Download` to cancel.
3. **Step 3 — Realtime Streamflow (Optional)**:
   - If you have recent gauge observations up to today (`t₀`), click `Upload CSV` and choose a two-column CSV (`date,discharge_cms` in `m³/s`, dates as `YYYY-MM-DD`). These observations are plotted on the hydrograph and used by Step 5 (Data Assimilation).
4. **Step 4 — Make Forecasts**:
   - Choose the model in `Pretrained / Fine-Tuned Model Weights` (one of the 3 pre-trained global checkpoints, or a fine-tuned model that you saved in Tab 6). The line below the list shows the forecast and hindcast inputs that the model needs.
   - Click `Run Model` to run the model and plot the forecast on the hydrograph.
5. **Step 5 — Data Assimilation (Optional)**:
   - When you have uploaded real-time streamflow in Step 3, you can correct the forecast with two methods:
   - **Method A: `Nudging (Persistence)`** — computes the error between the model and the observed streamflow at issue time `t₀`, and applies a correction that decays exponentially over the future lead days. Click `Enable Nudging Overlay` to show the nudged forecast on the hydrograph. Drag the control points on the decay curve pad, or click a preset (`Fast · .30`, `Best · .65`, `Slow · .85`, `Full · 1.0`), to update the forecast immediately.
   - **Method B: `Embedding Variational` (4D-Var)** — optimizes latent embedding perturbations (`δe`) via backpropagation through the frozen LSTM so the hindcast trajectory matches observed streamflow over the assimilation window (`Win (Days)`).
     - Choose the loss function (`MSE`, `CMAL`, `NSE`), the active latent components (`Static`, `Hindcast`, `Forecast`), `Steps/Run` (default `5`), `Learn Rate` (default `0.01`), `Win (Days)` (default `14`), and the background weight `Prior λ_bg` (default `0.50`). Or click a preset: `Balanced`, `Tight · 7d`, `Dynamic`, or `Long · 60d`.
     - Click `Start Variational DA (N Steps)` to start from Step 0 (the prior), or `+N Steps` to continue from the inspected step. A progress bar shows the run, and `Stop` ends it early. Use the `Iteration Inspector` slider (or its ◀ ▶ buttons, or a click on its loss chart) to show any optimization step on the hydrograph.

#### Right-Hand Forecast Charts & Interactive "What-If" Weather Editor
- **Top chart — input weather forcings (lookback + forecast)**:
  - Shows the historical lookback and the future forecast weather (`Precipitation` bars and `Temperature` lines). Use the `Catchment:` dropdown above the chart to choose the basin to view.
  - Interactive What-If editing: click and drag any future precipitation bar or temperature point on the top chart to test a different storm (for example, double tomorrow's rain or move a storm one day earlier).
  - Click `Run What-If Forecast` to run the model again with your edited weather. The result appears as a new What-If line on the bottom hydrograph, next to the baseline forecast. Click `Reset Inputs` to restore the original weather data.
- **Bottom chart — streamflow forecast hydrograph (m³/s)**:
  - Plots the hindcast (spin-up) simulation, the baseline forecast, observed streamflow from your CSV, the nudged forecast, the variational DA forecast, each What-If scenario, and horizontal return-period lines (2-yr to 100-yr, when available).
  - Use the `What-If Scenarios` bar to turn each scenario on or off or delete it, or click `Delete All What-Ifs`.

---

### 3.10 Tab 8: Account

The Account tab manages local user accounts, shows how much data each account holds, and has one table to rename or delete basins across the whole application.

#### Features
- **Account switcher**:
  - By default, you work in the `Guest` account. Guest data is deleted every time the server starts. Create an account to keep your data.
  - Click `Create account`, enter a `Username` (and, if you like, a display name and an email), and click `Create`. OpenHydroNet creates the account and switches to it.
  - Each account has its own folder with its own basins, attributes, weather extractions, and fine-tuned models: `<data folder>/profiles/<username>/`. The data folder is `$OPENHYDRONET_DATA_DIR`, or `frontend/data` in the repository by default. Set `$OPENHYDRONET_PROFILES_DIR` to keep accounts in another folder.
  - Click an account in the list to switch to it. Click `Remove an account` to delete an account you no longer need.
- **6 storage KPI cards**:
  - Show how much data the account holds:
    1. `Catchment polygons` (number of basins and total area)
    2. `Static attributes`
    3. `Historical training data`
    4. `Streamflow targets`
    5. `Real-time forecasts`
    6. `Models & disk use` (number of model runs and the total disk space used by the account)
- **Basins table**:
  - Lists every basin in the account with its name and ID, drainage area (`km²`), and badges for its saved data (`Polygon`, `N attrs`, `Historical`, `Targets`, `Forecasts`).
  - Rename: click `Rename` on a row to change a basin's display name everywhere in the UI.
  - Delete: click `Delete` on a row (or `Delete all basins`) to permanently remove the basin and its polygon, static attributes, weather files, streamflow targets, and forecast data from all storage folders.

---

### 3.11 Web UI REST API Reference

When `openhydronet-ui` is running, the web server (`frontend/server.py`, built on the Python standard library `http.server`) exposes JSON endpoints that you can also call from `curl` or Python `requests`. The main endpoints are:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/profile/list` | `GET` | List local user accounts. |
| `/api/profile/create` | `POST` | Create a user account (`{"username": "..."}`). |
| `/api/profile/login` | `POST` | Switch the active user account (`{"username": "..."}`). |
| `/api/account/summary` | `GET` | Return storage usage and the basin inventory of the active account. |
| `/api/watersheds` | `GET` | List the saved basin polygons. |
| `/api/watersheds/rename` | `POST` | Rename a basin (`{"catchment_id": "...", "name": "..."}`). |
| `/api/watersheds/delete` | `POST` | Delete a basin (`{"catchment_id": "..."}`). |
| `/api/polygons/upload` | `POST` | Upload a custom GeoJSON basin polygon. |
| `/api/delineate` | `POST` | Delineate an upstream basin polygon (`{"latitude": ..., "longitude": ..., "dataset": ...}`). |
| `/api/weather/models` | `GET` | List the weather forecast products, map variables, and local sync status. |
| `/api/weather/probe` | `GET` | Return point forecast time series from every synced weather model at (`lat`, `lon`). |
| `/api/weather/sync` | `POST` | Start background weather synchronization (`{"model": "..."}` or `{"models": [...]}`). |
| `/api/maas/models` | `GET` | List the external flood models (`floodhub`, `glofas`, `geoglows`). |
| `/api/maas/reaches` | `GET` | Match a point (`lat`, `lon`) to river reaches in each flood model network. |
| `/api/maas/forecast` | `GET` | Return multi-model discharge forecasts and return periods for (`lat`, `lon`). |
| `/api/attributes/extract` | `POST` | Start static attribute extraction for selected basins. |
| `/api/attributes/extracted` | `GET` | Return the extracted static attributes of the active account. |
| `/api/weather/historical` | `POST` | Start historical weather extraction for selected basins and dates. |
| `/api/finetune/models` | `GET` | List the pre-trained base models available for fine-tuning. |
| `/api/finetune/run` | `POST` | Run `N` fine-tuning epochs and return training and validation metrics. |
| `/api/forecast/fetch-realtime` | `POST` | Fetch real-time lookback and forecast weather for selected basins and products. |
| `/api/forecast/run-model` | `POST` | Run the forecast model (including What-If weather inputs). |
| `/api/forecast/assimilate` | `POST` | Run nudging or variational data assimilation for a basin. |
| `/api/jobs` | `GET` | List background jobs. |
| `/api/jobs/cancel` | `POST` | Cancel one job (`{"job_id": "..."}`) or, without `job_id`, stop all active downloads. |
