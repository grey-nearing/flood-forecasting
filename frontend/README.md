# OpenHydroNet Interactive Web Platform (`frontend`)

## Purpose

The `frontend` package provides the interactive web UI and local HTTP server for **OpenHydroNet**, unifying all end-to-end operational flood forecasting workflows into a single browser-based platform:

1. **Weather Viewer** (`#tab-weather`): Interactive global meteorological forecast viewer (ECMWF IFS/AIFS, NOAA GFS/GEFS/HRRR) with WebMercator raster tiles, animated wind particles, point probes, and catchment meteograms.
2. **Models-as-a-Service (MaaS) Hub** (`#tab-maas`): Multi-model operational river discharge comparison hub across **Google FloodHub** (HydroSHEDS), **Copernicus GloFAS v4** (LISFLOOD 0.05°), **GEOGLOWS ECMWF v2** (TDX-Hydro), and **JAXA Today's Earth** (CaMa-Flood) with USGS Bulletin 17C return period overlays (`return_periods`).
3. **Basin Delineation** (`#tab-delineation`): Interactive 1:1 DEM and river network selector (**HydroSHEDS 90m DEM ↔ HydroRIVERS** and **MERIT-Hydro 90m DEM ↔ MERIT-Basins**) backed by `multimet.catchment_delineation.DemDelineator` for exact 90m D8 flow-direction watershed delineation and custom GeoJSON basin upload.
4. **Geographical Features** (`#tab-geo-features`): Interactive extraction and choropleth map exploration of 214 Caravan static catchment attributes (HydroATLAS v10 Level 12 + ERA5-Land 1981–2020 climate indices) backed by `multimet.static_extractor.StaticAttributesExtractor`.
5. **Training & Fine-Tuning** (`#tab-training`): Historical forcing Zarr extraction, streamflow CSV ingestion (USGS NWIS / Environment Agency / GRDC / Caravan), and model training workspace management.
6. **Forecasting & Data Assimilation** (`#tab-forecasting`): Operational Cold-Start (365-day spin-up + `save_states()`) and Hot-Start (1-day incremental step + `load_states()` / `predict_from_state()`) inference backed by `multimet.timeseries_extractors.realtime` and `model` (`MeanEmbeddingForecastLSTM`).
7. **Account & Profile Workspace** (`#tab-account`): Isolated 13-folder per-user workspace management (`ProfileManager`) matching the OpenHydroNet `model` data directory specification.

---

## Entry Points

### CLI Server Entry Point

```bash
# Via console script (installed by setup.py):
openhydronet-ui --host 0.0.0.0 --port 8000

# Or directly via Python module:
python -m frontend.server --host 0.0.0.0 --port 8000
```

### MERIT-Hydro 90m D8 Tile Downloader (`frontend/tools/`)

```bash
python -m frontend.tools.download_merit_d8_tiles \
  --target-dir ~/.cache/openhydronet/data/merit_dem/tiles_5deg \
  --workers 12
```

### Python API

```python
from frontend.server import OpenHydroNetHandler, run_server
from frontend.dem_delineator import get_dem_delineator
from frontend.static_attributes import get_attributes_extractor
from frontend import realtime_forecast_service
```

---

## Core Backend Integration (`main`)

The `frontend` package directly consumes the first-class packages on `main`:

| UI Capability | Core OHN Package on `main` | Frontend Integration Module |
| :--- | :--- | :--- |
| 90m D8 Flow-Direction Catchment Delineation | `multimet.catchment_delineation` (`DemDelineator`) | `frontend/dem_delineator.py` |
| 214 Caravan Static Catchment Attributes | `multimet.static_extractor` (`StaticAttributesExtractor`, `ERA5ClimateLoader`, `ERA5GriddedExtractor`) | `frontend/static_attributes.py` |
| Real-Time Forcing Fetch (Cold-Start & Hot-Start) | `multimet.timeseries_extractors` (`realtime.fetch_realtime_multimet`, `hres.find_latest_hres_open_data_date`) | `frontend/realtime_forecast_service.py` |
| Hydrological Model Inference (`MeanEmbeddingForecastLSTM`) | `model` (`model.modelzoo`, `model.datautils.scaler.Scaler`, `model.datasetzoo.caravan`, `model.utils.config.Config`) | `frontend/realtime_forecast_service.py` |
| USGS Bulletin 17C Flood Return Periods | `return_periods` (`compute_return_periods`, `compute_empirical_weibull_return_periods`) | `frontend/maas_engine.py` |

---

## Running Tests

All frontend unit and integration tests are co-located in `frontend/tests/` and registered in `pyproject.toml`:

```bash
pytest frontend/tests
```
