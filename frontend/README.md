# OpenHydroNet Interactive Web Platform (`frontend`)

> **Do you need these tools?**
> If you only want to train or evaluate flood forecasting models on existing basins, you do **not** need to run the interactive web server. Pre-built basin datasets are available in `gs://caravan-multimet/v1.1` and `gs://open-multimet/`, and you can run models directly from the command line with `run` (see the top-level [`README.md`](../README.md)). Use `openhydronet-ui` when you want a browser interface to view weather forecasts, compare external river forecasts, delineate new watershed boundaries on a map, extract basin attributes, or run interactive forecasts.

---

## Overview

The `frontend` package runs a local web server and browser interface for OpenHydroNet. All spatial coordinates use `EPSG:4326` latitude and longitude degrees, and all dates use UTC (`YYYY-MM-DD`).

| Browser Tab | Module | What It Does | Resolution & Units |
| :--- | :--- | :--- | :--- |
| **Weather Viewer** (`#tab-weather`) | `frontend.weather_engine`, `frontend.weather_viewer` | Displays global weather forecast maps, wind particle animations, point values, and basin time series for ECMWF (IFS, AIFS) and NOAA (GFS, GEFS, HRRR). | `0.25°` / `3 km`; precipitation in `mm/day`, temperature in `°C`, wind in `m/s` |
| **Models-as-a-Service (MaaS) Hub** (`#tab-maas`) | `frontend.maas_engine`, `frontend.maas_viewer` | Compares live river discharge forecasts from three global providers: **Google FloodHub**, **Copernicus GloFAS**, and **GEOGLOWS ECMWF**, with USGS Bulletin 17C flood return period thresholds (`return_periods`). | River reaches (`EPSG:4326`); discharge in `m³/s` ($\text{m}^3/\text{s}$) |
| **Basin Delineation** (`#tab-delineation`) | `frontend.delineator`, `frontend.river_indexer` | Traces upstream watershed boundaries (`Polygon` / `MultiPolygon`) from any map click using 90 m D8 flow-direction grids (**HydroSHEDS 90 m DEM ↔ HydroRIVERS** or **MERIT-Hydro 90 m DEM ↔ MERIT-Basins**) or custom GeoJSON uploads. | `3 arc-second` (~`90 m`); drainage area in $\text{km}^2$ |
| **Geographical Features** (`#tab-geo-features`) | `frontend.static_attributes` | Computes and maps 214 Caravan static catchment attributes from HydroATLAS v10 Level-12 sub-basins and ERA5-Land (1981–2020) climate averages. | Level-12 sub-basins; area-weighted averages ($\ge 80\%$ valid coverage) |
| **Training** (`#tab-training`) | `frontend.historical_zarr`, `frontend.jobs` | Extracts historical basin weather time series into Zarr stores and launches background `model` training runs. | Daily (`1D`) UTC steps; precipitation in `mm/day`, radiation in `W/m²`, pressure in `kPa` |
| **Fine-Tuning** (`#tab-finetuning`) | `frontend.finetune_service` | Fine-tunes pretrained or user `BaseModel` checkpoints on local catchment CSV data (`precip`, `temp`, `streamflow`) with interactive module freeze/unfreeze controls and train/validation hydrograph diagnostics. | Daily (`1D`) UTC steps; precipitation in `mm/day`, temperature in `°C`, discharge in `mm/day` or `m³/s` |
| **Forecasting & Data Assimilation** (`#tab-forecasting`) | `frontend.realtime_forecast_service`, `frontend.forecast_zarr` | Runs 365-day Cold-Start spin-up (saving LSTM hidden/cell state via `save_state`) and incremental Hot-Start updates (loading prior state via `load_state_from_disk`), plus interactive What-If weather scaling. | Daily (`1D`) UTC lead steps; discharge in `m³/s` and `mm/day` |
| **Account & Profile Workspace** (`#tab-account`) | `frontend.profile_manager` | Manages isolated per-user workspaces (`catchments/`, `statics/`, `dynamics/`, `realtime/`, `targets/`, `assimilation/`, `models/`, `forecasts/`, `maas/`, `jobs/`). | Local directory tree under `frontend/data/users/<username>/` |

---

## Prerequisites & Optional Environment Variables

Activate the Conda environment and install the package in editable mode:

```bash
conda activate openhydronet
pip install -e .
```

You can configure local dataset caches and API credentials using the following optional environment variables:

| Environment Variable | Default Path / Value | Purpose |
| :--- | :--- | :--- |
| `FLOODHUB_API_KEY` | `~/.cache/openhydronet/floodhub_api_key` | Google FloodHub API key required to query live FloodHub discharge forecasts in the MaaS Hub. |
| `HYDROATLAS_GDB_PATH` | `~/.cache/openhydronet/data/hydroatlas` | Local path to `BasinATLAS_v10.gdb` or directory of `hybas_*_lev12_v1c.geoparquet` files for static attribute extraction. |
| `ERA5_CLIMATE_CACHE_DIR` | `~/.cache/openhydronet/data/era5_climate` | Local cache directory for pre-computed ERA5-Land 1981–2020 catchment climate indices. |
| `OPENHYDRONET_EE_PROJECT` | *(none)* | Google Cloud project ID for Google Earth Engine when downloading missing MERIT-Hydro 90 m D8 tiles on demand. |
| `OPENHYDRONET_PROFILES_DIR` | `frontend/data/users` | Root directory where per-user profile workspaces and model runs are stored. |
| `OPENHYDRONET_WEATHER_DATA_DIR` | `~/.cache/openhydronet/weather` | Local directory where `sync-weather-forecasts` stores global weather forecast grids for the Weather Viewer. |

---

## Quick Start Examples

### 1. Start the Local Web Server

```bash
# Start via the installed console script (listens on http://127.0.0.1:8080 by default):
openhydronet-ui --host 127.0.0.1 --port 8080

# Or run the server module directly:
python -m frontend.server --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` in your browser.

### 2. Build River Network Vector Pyramids (Optional)

To pre-build simplified river network `.gpkg` and `.npz` spatial pyramids for fast map rendering:

```bash
python -m frontend.tools.build_river_pyramids
```

### 3. Python API Usage

```python
from frontend.server import OpenHydroNetHandler, run_server
from frontend.delineator import HydroDelineator, get_delineator
from frontend.static_attributes import get_attributes_extractor
from frontend import realtime_forecast_service

# Initialize the 90 m D8 watershed delineator:
delineator = get_delineator('hydroatlas')

# Start the local HTTP server programmatically:
run_server(host='127.0.0.1', port=8080)
```

---

## What to Watch Out For

- **Localhost binding (`127.0.0.1`):** The server binds to `127.0.0.1` by default because profile and workspace endpoints accept a client-supplied username without password authentication. Do not pass `--host 0.0.0.0` on an untrusted network.
- **80% valid spatial coverage threshold:** Area-weighted catchment aggregation requires at least **80%** of the polygon's area to have finite (non-`NaN`) values. If fewer than 80% of a basin is covered by valid data on a given date, the extracted value is set to `NaN`.
- **Cold-Start before Hot-Start:** Before running a Hot-Start forecast for a basin, run a Cold-Start forecast first so the model executes the 365-day historical spin-up and writes a saved `(h, c)` state file (`save_state`) to `models/hot_start/`.
- **Default forecast model:** The Forecasting tab uses the pre-trained checkpoint in `model/pretrained-models/google-floodhub-settings-110-epochs` unless the request sets `model_run_dir`. Your basin must have every static attribute listed under `static_attributes` in that run's `config.yml`; otherwise the forecast stops with a `KeyError` that names the missing attributes.
- **Local 90 m DEM tiles:** Exact 90 m D8 watershed delineation requires 5° × 5° flow-direction `.npy` tiles in `frontend/data/base_layers/dem/<dem_id>/tiles_5deg` or `~/.cache/openhydronet/data/`. Use `download-merit-d8-tiles` to download MERIT-Hydro tiles before delineating basins offline.

---

## Command-Line Arguments Reference (`openhydronet-ui`)

| Flag / Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `str` | `127.0.0.1` | Host interface address to bind the HTTP server to. Use `127.0.0.1` for local use. |
| `--port` | `int` | `8080` | TCP port number for the HTTP server (overrides the `PORT` environment variable). |
| `PORT` | env var (`int`) | `8080` | Fallback TCP port used when `--port` is not passed on the command line. |

---

## Running Tests

Run the frontend unit and integration test suite with `pytest`:

```bash
pytest frontend/tests
```
