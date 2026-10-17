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
2. `model/pretrained-models/google-floodhub-settings-no-graphcast-110-epochs/` (`model_epoch027.pt`) — Uses `hres`, `imerg`, and `cpc` (no GraphCast required).
3. `model/pretrained-models/google-floodhub-settings-aifs-110-epochs/` (`model_epoch026.pt`) — Uses `hres`, `aifs`, `imerg`, and `cpc`.

Each folder contains the model weights (`model_epoch*.pt`), normalization statistics (`scaler.zarr`), and configuration (`config.yml`). These checkpoints replicate the 2025 generation of Google FloodHub (a former operational model that has since been superseded by a newer model release).

---

## Section 2: Using the Backend (CLI Commands and Python APIs)

You can run every step of the flood forecasting workflow from the terminal or from a Python script.

Below is a summary of all 18 command-line tools installed by `pip install -e .`:

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
rivers = RiverNetwork(Path("~/.cache/openhydronet/data/HydroRIVERS_v10_shp/HydroRIVERS_v10.shp").expanduser())
hybrid_feature = delineate_hybrid(
    lat=38.85,
    lon=-121.05,
    delineator=delineator,
    river_network=rivers,
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
from multimet.gridded_archive_builders import (
    build_cpc_archive,
    build_imerg_archive,
)

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
2. **Copernicus GloFAS v4** (`glofas`, `GloFASClient`) — ECMWF-forced LISFLOOD global `0.05°` river discharge forecasts (deterministic control run, ensemble mean/median/percentiles, and all 50 ensemble members) via the Open-Meteo Flood API (`https://flood-api.open-meteo.com/v1/flood`) and local/cloud Zarr archives.
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
- **`handoff_forecast_lstm`** (`HandoffForecastLSTM`) — Runs a hindcast LSTM over lookback weather (`seq_length` days) and passes its final hidden and cell states through a linear handoff network to initialize a separate forecast LSTM over the future `lead_time` days. Replicates the 2024 generation of Google FloodHub (a former operational model).

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

This computes evaluation metrics (`NSE`, `KGE`, `Alpha-NSE`, `Beta-NSE`, `Beta-KGE`, `Pearson-r`) on days with observed streamflow and writes predictions and metrics to `<run-dir>/test/model_epoch110/`. Pass `--assimilate` to enable variational data assimilation during evaluation.

**5. Run operational inference on all dates (`run infer`)**:

```bash
# Standard forecast inference (runs even when target streamflow observations are NaN)
run infer \
  --run-dir model/pretrained-models/google-floodhub-settings-no-graphcast-110-epochs \
  --epoch 27

# Forecast inference with real-time variational streamflow data assimilation enabled
run infer \
  --run-dir model/pretrained-models/google-floodhub-settings-no-graphcast-110-epochs \
  --epoch 27 \
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
convert_caravan_to_zarr("~/data/Caravan-nc", "~/data/Caravan-zarr")

# 2. Load and customize a YAML configuration in Python
cfg = Config(Path("model/example-configs/floodhub-settings-config.yml"))
cfg.update_config({"epochs": 5, "device": "cpu", "run_dir": Path("./runs")})

# 3. Train a model
start_run(config=cfg, gpu=-1)

# 4. Fine-tune a pre-trained model using a fine-tune config file
finetune(config_file=Path("finetune_config.yml"), gpu=-1)

# 5. Evaluate or run inference on a trained run directory
run_dir = Path("model/pretrained-models/google-floodhub-settings-no-graphcast-110-epochs")
run_cfg = Config(run_dir / "config.yml")
run_cfg.inference_mode = True
run_cfg.tester_skip_obs_all_nan = False
eval_run(
    cfg=run_cfg,
    run_dir=run_dir,
    period="test",
    epoch=27,
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

When the page loads, you will see three main areas:

1. **Top Navigation Bar**:
   - Contains **8 workflow tabs**: `Weather Viewer`, `Models-as-a-Service`, `Basin Delineation`, `Geographical Features`, `Training`, `Fine Tuning`, `Forecasting and Data Assimilation`, and `Account`.
   - **Drag to Reorder**: You can click and drag any tab left or right to customize the tab order. Your layout is saved automatically in your browser.
   - **Activity Spinners & Stop Downloads**: When a background job (such as weather syncing, attribute extraction, or fine-tuning) is running in a tab, a spinning indicator appears next to that tab's title. Click the red **`Stop Downloads`** button in the top-right corner to stop active background downloads.
2. **Left Control Sidebar**:
   - Changes automatically depending on which tab is active.
   - **Collapse / Expand**: Click the collapse arrow button on the edge of the sidebar or press the **`[`** key on your keyboard to hide or show the sidebar.
   - **Resize**: Drag the right border of the sidebar to make it wider or narrower.
3. **Main Canvas**:
   - Shows the interactive global **Leaflet Map** on map-based tabs (`Weather Viewer`, `Models-as-a-Service`, `Basin Delineation`, `Geographical Features`).
   - Shows interactive **Chart.js Dashboards** on modeling tabs (`Training`, `Fine Tuning`, `Forecasting and Data Assimilation`).
   - Shows your **Account & Basin Management Dashboard** on the `Account` tab.

---

### 3.3 Tab 1: Weather Viewer

The **Weather Viewer** tab displays animated global weather forecasts and satellite precipitation on the interactive map.

#### Left Sidebar Controls
- **Forecast Model Selector**: Choose from 8 weather products:
  - **ECMWF HRES** (`ecmwf_hres` — `0.25°` global deterministic 10-day forecast)
  - **ECMWF IFS** (`ecmwf_ifs` — `0.25°` global ensemble mean forecast)
  - **ECMWF AIFS** (`ecmwf_aifs` — `0.25°` AI-driven global forecast)
  - **NOAA GFS** (`noaa_gfs` — `0.25°` global deterministic 16-day forecast)
  - **NOAA GEFS** (`noaa_gefs` — `0.5°` 30-member ensemble mean forecast)
  - **NOAA HRRR** (`noaa_hrrr` — `3 km` Continental US rapid refresh forecast)
  - **NASA IMERG** (`nasa_imerg` — `0.1°` global satellite precipitation, 7-day lookback)
  - **NOAA CPC** (`noaa_cpc` — `0.5°` global gauge precipitation, 7-day lookback)
- **Sync Status & Refresh Buttons**:
  - Each model card shows its sync status (`Ready`, `Syncing`, or `Not downloaded`) and forecast time range.
  - Click **`Refresh All`** at the top of the model list to sync all weather models, or click the **`↻`** button on a single model card to sync just that model.
- **Variable Selector**:
  - **Precipitation** (`mm/hr` or `mm/day` depending on model)
  - **Accumulated Precip** (`mm` running total from forecast start)
  - **2m Temperature** (`°C`)
  - **10m Wind Speed** (`m/s`)
  - **Mean Sea Level Pressure** (`hPa`)
- **Layer Overlays**:
  - **Wind Streamline Particles**: Toggle animated moving wind arrows across the map.
  - **Colour Opacity**: Drag the slider (`0%` to `100%`) to make the weather heatmap more transparent or opaque.
  - **Show Catchments**: Toggle your delineated basin polygons on top of the weather map.

#### Map Controls and Point Forecast Drawer
- **Bottom Timeline Scrubber**: Click **`Play`** to animate through forecast lead times, use the **`⏮`** / **`⏭`** step buttons to move one time step back or forward, or click **`1x`** to cycle animation speed.
- **Click-Anywhere Point Meteogram**: Click any location on the map (or click one of your catchment polygons) to open the **Point Forecast Drawer** at the bottom of the screen. It plots the full time series of precipitation, temperature, wind speed, and pressure at that exact `(lat, lon)` coordinate.

---

### 3.4 Tab 2: Models-as-a-Service

The **Models-as-a-Service** tab lets you view live global river flood alerts and compare discharge forecasts from external operational flood models.

#### Left Sidebar Controls
- **River Network Overlays (Multi-Select)**:
  - Check one or more boxes to display river reaches on the map:
    - **Google FloodHub** (HydroRIVERS network)
    - **Copernicus GloFAS v4** (LISFLOOD `0.05°` grid)
    - **GEOGLOWS ECMWF v2** (TDX-Hydro stream network)
  - Click **`Refresh`** to fetch the latest flood severity statuses.
- **Return Period Severity Legend**:
  - River reaches are colored by their forecasted peak return period:
    - **Blue (`Normal < 2-yr`)**: Below the 2-year flood threshold.
    - **Yellow (`≥ 2-yr Medium`)**: Exceeds the 2-year flood threshold.
    - **Orange (`≥ 5-yr High`)**: Exceeds the 5-year flood threshold.
    - **Red (`≥ 20-yr Severe`)**: Exceeds the 20-year flood threshold.
- **Forecast Comparison Panel**:
  - Zoom in on the map until river reaches appear, then **click any river reach** (or click anywhere on the map to snap to the nearest river within `25 km`).
  - The sidebar displays a multi-model hydrograph in **`m³/s`** comparing Google FloodHub, Copernicus GloFAS v4, and GEOGLOWS ECMWF v2 side by side, along with horizontal dashed lines for the `2-yr`, `5-yr`, and `20-yr` flood return periods.
  - **GloFAS Reference**: Switch the main GloFAS line between **`Control`**, **`Median`**, and **`Mean`**.
  - **Uncertainty Controls**: Click **`+ Ensemble`** to show all individual ensemble member trajectories, or **`+ Range`** to shade the `Min–Max` ensemble spread.

---

### 3.5 Tab 3: Basin Delineation

The **Basin Delineation** tab lets you create new watershed polygons by clicking on the map or uploading a GeoJSON file.

#### Step-by-Step Workflow
1. **Select a River Network & DEM**:
   - Choose **`HydroRIVERS`** (HydroSHEDS 90m D8 DEM) or **`MERIT-Basins`** (MERIT-Hydro 90m D8 DEM + vector unit catchments).
2. **Choose a Basemap**:
   - Switch between **`Map`**, **`Satellite`**, **`OpenStreetMap`**, **`HydroSHEDS 90m DEM`**, and **`OpenTopoMap`**. Use the **River Network Opacity** slider to adjust how bright the blue river lines appear.
3. **Turn On Delineate Mode**:
   - Click the **`Delineate Mode`** button in the sidebar (or press **`D`** on your keyboard). Your mouse cursor will change to a crosshair.
4. **Set the River Snap Search Radius**:
   - Adjust the slider (`0.1 km` to `10.0 km`, default `5.0 km`). When you click near a river, OpenHydroNet snaps your click to the largest river reach within this radius so the basin traces from the true river channel instead of a hillside.
5. **Click the Map to Delineate**:
   - Click your desired outlet point on the map. Within a few seconds, the upstream watershed boundary appears on the map and is added to the **Active Watersheds** list.
6. **Upload Custom Basin (Optional)**:
   - If you already have a polygon file on your computer, click **`Upload Custom Basin (GeoJSON)`** to import a `.geojson` or `.json` file (`Feature`, `FeatureCollection`, or raw `Polygon`/`MultiPolygon` in `EPSG:4326`).
7. **Inspect, Rename, or Export**:
   - Click any basin in the **Active Watersheds** list to zoom to it and view its area (`km²`), perimeter (`km`), and outlet coordinates in the **Selected Basin** card.

---

### 3.6 Tab 4: Geographical Features

The **Geographical Features** tab extracts and visualizes the ~210 Caravan/HydroATLAS static physical attributes for your delineated basins.

#### Step-by-Step Workflow
1. **Select Catchments**:
   - In the left sidebar, check the basins you want to process. Use the quick-select buttons (**`Unextracted`**, **`All`**, **`None`**) to select all basins that do not have attributes yet.
2. **Extract Attributes**:
   - Click **`Extract Attributes`**. A progress bar shows the extraction status for each basin.
3. **View and Export the Attribute Table**:
   - Once a basin shows a green **`✓ Extracted`** badge, click **`Attribute Table`** on its card to open a full-screen table modal.
   - Filter attributes by category (**`All`**, **`Topography`**, **`Climate`**, **`Soils & Geology`**, **`Land Cover`**, **`Hydrology`**, **`Anthropogenic`**) or type in the search box.
   - Click **`Export CSV`** or **`Export JSON`** inside the modal (or click the **`CSV`** button directly on the basin card) to download the attributes to your computer.
4. **Visualize Sub-Basin Choropleth Maps**:
   - Click **`Map Attribute`** on an extracted basin card to color its internal HydroATLAS sub-basins on the map by any selected attribute (such as mean elevation, slope, soil clay content, or forest fraction).

---

### 3.7 Tab 5: Training

The **Training** tab lets you extract multi-year historical weather archives for your basins and configure multi-basin deep learning experiments.

#### Step-by-Step Workflow
1. **Extract Historical Weather Archives**:
   - Under **Historical Weather Extraction**, check the weather sources you want to extract (`CPC`, `IMERG`, `ERA5`, `IFS / HRES`, `GraphCast`, `ERA5-Land`, `CHIRPS`, `CERRA`).
   - Select your **Start Date** and **End Date** (`YYYY-MM-DD`).
   - Choose **Scope**: **`All Uploaded Basins`** or **`Selected Basin Only`**.
   - *(Optional)* Enter an email address in **Notification Email** if you want an email when a long extraction finishes.
   - Click **`Extract Historical Data`**. You can watch the progress bar or click **`Stop`** at any time.
   - Click **`Browse Zarr Archives`** to inspect the extracted Zarr variables, date ranges, and storage sizes on disk.
2. **Configure Model Architecture & Hyperparameters**:
   - Choose the **Model Architecture** (`MeanEmbeddingForecastLSTM`, `HandoffForecastLSTM`, `CudaLSTM`, `EALSTM`, `MTSLSTM`, `Transformer`), hidden size, dropout, loss function (`CMAL`, `NSE`, `MSE`), learning rate, batch size, and sequence lengths (`Hindcast Days` and `Forecast Horizon`).
3. **Set Temporal Splits**:
   - Configure the **Training**, **Validation**, and **Testing** date ranges and monitor training and validation loss curves on the right-hand dashboard.

---

### 3.8 Tab 6: Fine Tuning

The **Fine Tuning** tab adapts a pre-trained global flood model to a specific local watershed using your own local streamflow gauge records (and optional local rain/temperature gauge CSVs).

#### Step-by-Step Workflow (Left Sidebar Steps 1–5)

1. **Step 1 — Base Model Inventory**:
   - Select a pre-trained global model checkpoint from the dropdown (`google-floodhub-settings-110-epochs`, `google-floodhub-settings-no-graphcast-110-epochs`, or `google-floodhub-settings-aifs-110-epochs`). The card shows the model architecture, parameter count, and required weather inputs.
2. **Step 2 — Basin Data & Forcings**:
   - Select your **Clipped Watershed Polygon** from the dropdown.
   - Click **`Extract Global Forcings for Catchment`** to extract the historical weather time series needed by the base model for your basin.
   - Choose the **Input Forcing Source**:
     - **`Global Data Only`** — Uses only the cloud-extracted satellite/reanalysis weather.
     - **`Global Data + Uploaded Local Inputs`** — Combines global weather with your uploaded local weather station CSVs.
     - **`Uploaded Local Inputs Only`** — Uses only your uploaded local weather CSVs.
   - **Upload Target Streamflow CSV (Required)**: Upload a two-column CSV file (`date,streamflow`) and choose the physical units (**`m³/s`** or **`mm/day`**). If you choose `m³/s`, OpenHydroNet automatically converts the discharge to `mm/day` using your basin's delineated drainage area in `km²`. Click **`Download CSV Template`** if you need an example file.
   - *(Optional)* Upload **Local Precipitation (`mm/day`)** and **Local Temperature (`°C`)** CSV files.
3. **Step 3 — Train / Validation Split**:
   - Drag the slider (`20%` to `90%`, default `70%`) to divide your overlapping streamflow dates into a chronological **Training** period and **Validation** period. The exact start and end dates for each split update live below the slider.
4. **Step 4 — Tunable Model Components**:
   - Choose which neural network layers to unfreeze during fine-tuning (frozen layers keep their pre-trained global weights):
     - Use the quick-preset buttons: **`Local + Head`**, **`Local + Forecast LSTM + Head`**, or **`Unfreeze All`**, or check individual modules (`static_embedding_fc`, `local_static_embeddings_fc`, `hindcast_embeddings_fc`, `forecast_embeddings_fc`, `shared_embeddings_fc`, `hindcast_lstm`, `forecast_lstm`, `head`).
5. **Step 5 — Hyperparameters & Step Control**:
   - Set **Epochs to Run Next** (e.g., `5`), **Learning Rate** (e.g., `0.0005`), **Loss** (`CMAL`, `NSE`, or `MSE`), **LR Schedule**, and **Batch Size**.
   - Click **`Run N Epochs & Pause`**. The model trains for `N` epochs, streams live metrics to the charts, saves a checkpoint after each epoch, and pauses so you can inspect the results. Click it again to continue training for another `N` epochs, or click **`Reset`** to start over.

#### Right-Hand Fine-Tuning Dashboard
- **Epoch Inspector Scrubber**: Drag the epoch slider at the top (or click any row in the **Epoch Checkpoint History** table at the bottom) to view the hydrograph and metrics from any saved epoch (`Epoch 0 (Pretrained Baseline)` through the latest epoch).
- **5 Performance KPI Cards**: Displays **Train Loss**, **Validation Loss**, **Train NSE**, **Validation NSE**, and **Validation KGE**, along with the change (`Δ`) compared to the un-tuned global baseline (`Epoch 0`).
- **Hydrograph & Hanging Hyetograph**: Plots Observed vs. Simulated Streamflow on the main axis and inverted precipitation bars from the top axis, with a vertical line separating the Training and Validation periods. Use the **`Display Units`** toggle in the chart header to switch between **`m³/s`** and **`mm/day`**.

---

### 3.9 Tab 7: Forecasting and Data Assimilation

The **Forecasting and Data Assimilation** tab generates real-time operational flood forecasts for your basins, assimilates live gauge observations, and lets you run interactive **"What-If"** weather scenarios.

#### Step-by-Step Workflow (Left Sidebar Steps 1–5)

1. **Step 1 — Select Custom Catchments & Start Mode**:
   - Check one or more basins in the catchment list (`All`, `None`).
   - Choose the **Start Mode**:
     - **`Cold Start (365d Lookback)`** — Runs the full 365-day hindcast lookback window from scratch.
     - **`Hot Start (t₀ − 1d)`** — Loads a saved LSTM hidden state from yesterday (`t₀ − 1d`) and steps forward using only 1 day of new weather data. Select the **Forecast Issue Date (`t₀`)** to verify whether a cached state is ready.
2. **Step 2 — Fetch Real-Time Weather Data**:
   - Check the real-time weather products you want to download (`ECMWF HRES`, `NASA IMERG`, `NOAA CPC`, `ECMWF AIFS`, `NOAA GFS`, `NOAA GEFS`, `ECMWF IFS-ENS`).
   - Click **`Fetch Real-Time Data`**. Each product displays a live download progress bar.
3. **Step 3 — Realtime Streamflow (Optional)**:
   - If you have recent gauge observations up to today (`t₀`), upload a two-column CSV (`date,discharge_cms` in `m³/s`). These observations are plotted on the hydrograph and used by Step 5 (Data Assimilation).
4. **Step 4 — Make Forecasts**:
   - Choose your **Model Weights** from the dropdown (any of the 3 pre-trained global checkpoints or a fine-tuned model checkpoint you created in Tab 6).
   - Click **`Run Model`** to generate the probabilistic flood forecast (`Median Q50`, `25th–75th` percentile band, and `5th–95th` percentile band).
5. **Step 5 — Data Assimilation (Optional)**:
   - When you have uploaded real-time streamflow in Step 3, you can correct the forecast using two methods:
   - **Method A: `Nudging (Persistence)`** — Computes the residual error between the model and observed streamflow at issue time `t₀` and applies an exponentially decaying correction across future lead days. Drag the control points on the interactive decay curve pad or click a preset (**`Fast .30`**, **`Best .65`**, **`Slow .85`**, **`Full 1.0`**) to update the assimilated forecast immediately.
   - **Method B: `Embedding Variational` (4D-Var)** — Optimizes latent embedding perturbations (`δe`) via backpropagation through the frozen LSTM so the hindcast trajectory matches observed streamflow over the assimilation window (`Win (Days)`).
     - Choose the loss function (**`MSE`**, **`CMAL`**, **`NSE`**), active latent components (**`Static`**, **`Hindcast`**, **`Forecast`**), **Steps/Run**, **Learn Rate**, **Win (Days)**, and background regularization **Prior `λ_bg`** (or click a preset: **`Balanced`**, **`Tight 7d`**, **`Dynamic`**, **`Long 60d`**).
     - Click **`Start Variational DA`** (or **`+N Steps`**) and use the **Iteration Inspector** slider to scrub through every optimization step.

#### Right-Hand Forecast Charts & Interactive "What-If" Weather Editor
- **Top Chart — Input Weather Forcings (Lookback + Forecast)**:
  - Displays the historical lookback and future forecast weather variables (`Precipitation` bars and `Temperature` lines).
  - **Interactive What-If Editing**: You can **click and drag any future precipitation bar or temperature point** directly on the top chart to test hypothetical storms (for example, doubling tomorrow's rainfall or shifting a storm earlier).
  - Click **`Run What-If Forecast`** to re-run the neural network with your modified weather and plot the resulting **What-If Forecast** curve on the bottom hydrograph alongside the original baseline forecast. Click **`Reset Inputs`** to restore the original weather data.
- **Bottom Chart — Streamflow Forecast Hydrograph**:
  - Plots Historical Hindcast simulation, Observed Streamflow, Forecast Median (`Q50`), `50%` and `90%` probabilistic uncertainty bands, Assimilated Forecast, What-If Forecast, and horizontal Return-Period flood thresholds.

---

### 3.10 Tab 8: Account

The **Account** tab manages local user workspaces, monitors disk usage, and provides a single table to rename or delete basins across the entire application.

#### Features
- **Account Switcher**:
  - By default, you work in the **`Guest`** workspace.
  - Under **Create account**, enter a name (and optional email) and click **`Create & switch`** to create a separate workspace folder (`~/.cache/openhydronet/profiles/<account_id>/`) with its own basins, attributes, weather extractions, and fine-tuned models.
  - Switch between accounts at any time, or click **`Remove an account`** to delete an old workspace.
- **6 Storage KPI Cards**:
  - Shows real-time counts and disk space used across:
    1. **Catchment polygons**
    2. **Static attributes**
    3. **Historical training data**
    4. **Streamflow targets**
    5. **Real-time forecasts**
    6. **Models & disk use**
- **Centralized Basins Table**:
  - Lists every basin in the active account along with its drainage area (`km²`), outlet coordinates, and badges showing whether static attributes, historical weather, streamflow observations, and real-time forcings are present.
  - **Rename**: Click **`Rename`** on any row to change a basin's display name everywhere in the UI.
  - **Delete**: Click **`Delete`** on any row (or **`Delete all basins`**) to permanently remove that basin and clean up its associated polygon, static attributes, weather files, and streamflow records across all storage folders.

---

### 3.11 Web UI REST API Reference

When `openhydronet-ui` is running, the Flask backend (`frontend/server.py`) exposes JSON REST endpoints that you can also call from `curl` or Python `requests`:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/accounts` | `GET`, `POST` | List user accounts or create a new user account. |
| `/api/accounts/switch` | `POST` | Switch the active user account (`{"account_id": "..."}`). |
| `/api/accounts/<account_id>/overview` | `GET` | Return storage usage KPIs and per-basin inventory for an account. |
| `/api/accounts/<account_id>/basins/<basin_id>` | `PATCH`, `DELETE` | Rename (`{"name": "..."}`) or cascade-delete a basin across all stores. |
| `/api/weather/models` | `GET` | List all 8 weather forecast products and their local sync status. |
| `/api/weather/grid` | `GET` | Fetch a 2D gridded weather slice for map rendering (`model`, `variable`, `time_idx`). |
| `/api/weather/point` | `GET` | Fetch a point forecast meteogram time series (`model`, `lat`, `lon`). |
| `/api/weather/sync` | `POST` | Start background weather synchronization (`{"model": "all"}` or specific model ID). |
| `/api/maas/models` | `GET` | List available external global flood models (`floodhub`, `glofas`, `geoglows`). |
| `/api/maas/reaches` | `GET` | Fetch river reaches and return-period severity colors inside a map bounding box. |
| `/api/maas/forecast` | `GET` | Fetch multi-model discharge forecasts and return periods for `(lat, lon)`. |
| `/api/delineate` | `POST` | Delineate an upstream catchment polygon from `{"lat": ..., "lon": ..., "network": ...}`. |
| `/api/catchments` | `GET`, `POST` | List all saved catchments or upload a custom GeoJSON catchment. |
| `/api/static/extract` | `POST` | Start Caravan/HydroATLAS static attribute extraction for selected basins. |
| `/api/static/attributes/<basin_id>` | `GET` | Return the extracted ~210 static attributes for `<basin_id>`. |
| `/api/training/extract` | `POST` | Start historical MultiMet Zarr extraction for selected sources and date range. |
| `/api/finetune/models` | `GET` | List pre-trained base models available for fine-tuning. |
| `/api/finetune/step` | `POST` | Run `N` epochs of local fine-tuning and return training/validation metrics and hydrographs. |
| `/api/forecast/fetch` | `POST` | Start real-time lookback + forecast weather extraction for selected basins and products. |
| `/api/forecast/run` | `POST` | Run hydrological forecast inference (and optional nudging or variational DA / What-If). |
| `/api/forecast/variational_step` | `POST` | Run iterative 4D-Var embedding variational data assimilation steps. |
| `/api/jobs/stop-all` | `POST` | Stop all active background download and extraction jobs. |
| `/api/health` | `GET` | Health check endpoint returning `{"status": "ok"}`. |
