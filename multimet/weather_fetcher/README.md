# Gridded Weather Forecast Fetcher & Synchronizer (`multimet.weather_fetcher`)

This package downloads global weather forecasts from [dynamical.org](https://dynamical.org/) and provides a Python API for querying gridded forecast arrays, wind vector grids, point meteogram time series, and catchment weather summaries.

> **Do you need these tools?**
> If you only want to train or evaluate flood-forecasting models using the published MultiMet dataset, **you do not need to run these tools**. Point `dynamics_data_dir` in your training configuration file to `gs://caravan-multimet/v1.1`.
>
> Use `multimet.weather_fetcher` when you need to download and query live 10-day gridded weather forecasts from operational numerical weather prediction (NWP) and AI weather models.

---

## Overview & Entry Points

Installing this repository with `pip install -e .` provides the `sync-weather-forecasts` command-line tool and the `multimet.weather_fetcher` Python package.

| Component | Entry Point | Purpose |
| :--- | :--- | :--- |
| **CLI Synchronizer** | `sync-weather-forecasts` | Downloads the newest 10-day global forecast runs (`0.25°`, `721 × 1440`) and updates the active run folder. |
| **Python Sync API** | `WeatherSynchronizer`, `sync_all_models` | Checks upstream forecast catalogs, converts units to `float16` binary grids, and swaps the `current` symlink. |
| **Python Data Fetcher** | `WeatherDataFetcher` | Reads synced forecast runs to return 2D physical forecast grids, subsampled 10 m U/V wind arrays, 10-day point meteograms, and catchment-averaged summaries. |

### Supported Weather Models

| Model Key | Model Name | Upstream Dataset ID | Grid Resolution | Variables |
| :--- | :--- | :--- | :--- | :--- |
| `ecmwf_ifs` | ECMWF IFS ENS (Control Member) | `ecmwf-ifs-ens-forecast-15-day-0-25-degree` | `0.25°` (`721 × 1440`) | Precipitation rate, 2 m temperature |
| `ecmwf_aifs` | ECMWF AIFS Single | `ecmwf-aifs-single-forecast` | `0.25°` (`721 × 1440`) | Precipitation rate, 2 m temperature, sea-level pressure, 10 m U/V wind |
| `noaa_gfs` | NOAA GFS | `noaa-gfs-forecast` | `0.25°` (`721 × 1440`) | Precipitation rate, 2 m temperature, sea-level pressure, 10 m U/V wind |
| `noaa_gefs` | NOAA GEFS (Control Member) | `noaa-gefs-forecast-35-day` | `0.25°` (`721 × 1440`) | Precipitation rate, 2 m temperature, sea-level pressure, 10 m U/V wind |
| `noaa_hrrr` | NOAA HRRR CONUS | `noaa-hrrr-forecast-48-hour` | `3 km` CONUS | Precipitation rate, 2 m temperature, sea-level pressure, 10 m U/V wind |

### Supported Weather Variables

| Variable Key | Description | Physical Units | Range |
| :--- | :--- | :--- | :--- |
| `precipitation` | Mean precipitation rate over each 3-hour step | `mm/h` | `0.0` to `25.0` |
| `accumulated_precip` | Cumulative precipitation since forecast start | `mm` | `0.0` to `250.0` |
| `temperature` | Air temperature at 2 meters above ground | `°C` | `-40.0` to `45.0` |
| `wind` | Wind speed and direction at 10 meters above ground | `m/s` | `0.0` to `40.0` |
| `pressure` | Atmospheric pressure reduced to mean sea level | `hPa` | `960.0` to `1040.0` |

---

## Prerequisites

Activate the Conda environment and install the package in editable mode:

```bash
conda activate googlehydrology
pip install -e .
```

No API keys or login credentials are required to download public forecasts from `dynamical.org`.

---

## Quick Start Examples

### 1. Command-Line Usage (`sync-weather-forecasts`)

```bash
# Download the newest runs for default models (ECMWF IFS, ECMWF AIFS, NOAA GFS)
sync-weather-forecasts --data-dir /tmp/weather_cache

# Download only NOAA GFS and ECMWF AIFS
sync-weather-forecasts \
  --data-dir /tmp/weather_cache \
  --models noaa_gfs,ecmwf_aifs

# Force re-download even if the current run timestamp has not changed
sync-weather-forecasts \
  --data-dir /tmp/weather_cache \
  --force

# Print the current synchronization status as JSON
sync-weather-forecasts --data-dir /tmp/weather_cache --status
```

### 2. Python API Usage

```python
from pathlib import Path
from multimet.weather_fetcher import WeatherDataFetcher, WeatherSynchronizer

data_dir = Path("/tmp/weather_cache")

# 1. Synchronize forecast runs to disk
synchronizer = WeatherSynchronizer(
    data_dir=data_dir,
    models=["ecmwf_aifs", "noaa_gfs"],
)
status = synchronizer.sync_all()

# 2. Open the data fetcher on the synced directory
fetcher = WeatherDataFetcher(data_dir=data_dir)

# Fetch a 2D physical forecast grid (step=2 -> +6h)
precip_grid = fetcher.fetch_forecast_grid("ecmwf_aifs", "precipitation", step_idx=2)

# Fetch subsampled global 10m U/V wind vectors
wind = fetcher.fetch_wind_grid("ecmwf_aifs", step_idx=2, subsample=2)

# Query a 10-day multi-model point forecast meteogram
probe = fetcher.fetch_point_timeseries(lat=40.42, lon=-86.92, models=["ecmwf_aifs", "noaa_gfs"])
```

---

## Architecture & What to Watch Out For

* **Directory Layout:** Synchronized runs are written to `<data_dir>/runs/<timestamp>/` as memory-mapped `float16` binary grids (`<model>_<stream>.bin`) alongside `latest_dynamical_meta.json`. Once all streams for a run finish downloading, `<data_dir>/current` is updated to point to the new run folder.
* **Required Explicit Path in Python API:** `WeatherDataFetcher`, `WeatherSynchronizer`, and `sync_all_models` require an explicit `data_dir` argument.
* **Synced Data Required:** `WeatherDataFetcher` reads only real downloaded forecast files from `data_dir`. If you request grids, wind vectors, or point time series for a model that has not been downloaded to `data_dir`, Python raises `FileNotFoundError`.
* **Upstream Publication Delays:** Weather agencies publish forecast lead times progressively. If the newest run in the catalog still has missing values at the end of the 240-hour horizon, the synchronizer skips the incomplete run and keeps the previous complete run active.

---

## Command-Line Arguments Reference (`sync-weather-forecasts`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-dir` / `--data-root` | `str` | `$EARTHKIT_WEATHER_DATA_DIR` or `~/.cache/openhydronet/weather` | Local folder where forecast runs and `sync_status.json` are stored. |
| `--models` | `str` | `ecmwf_ifs,ecmwf_aifs,noaa_gfs` | Comma-separated list of model keys to download. |
| `--force` | `flag` | `False` | Downloads the run again even if the local run has the same issue time. |
| `--status` | `flag` | `False` | Prints the contents of `sync_status.json` and exits without downloading. |

---

## Running Tests

Run the unit and integration test suites with `pytest`:

```bash
pytest multimet/tests/test_weather_fetcher.py multimet/tests/test_weather_sync.py -v
```
