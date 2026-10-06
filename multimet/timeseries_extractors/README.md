# Catchment Timeseries Extractors (`multimet.timeseries_extractors`)

The `multimet.timeseries_extractors` subpackage reduces gridded meteorology to catchment-averaged forcing time series standardized to the **Caravan benchmark specification** ([Kratzert et al., 2023](https://nature.com/articles/s41597-023-01960-w); [Kratzert et al., 2024, arXiv:2411.09459](https://arxiv.org/abs/2411.09459)).

## Supported Meteorological Products

The extractor supports **4 core products**, operating either against user-supplied **Open-MultiMet Gridded Zarr Archives** (`--source archive --archive-store PRODUCT=URI`) or directly against **third-party agency upstream feeds** (`--source public`, for CPC, IMERG, and HRES):

| Product | Type | Native Grid | Forecast Lead | Variables Extracted | Supported Sources |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ERA5-Land** (`ERA5_LAND`) | Daily Reanalysis | $0.1^\circ$ (1801 $\times$ 3600) | N/A | **17 variables** (`era5land_temperature_2m`, `era5land_temperature_2m_min`, `era5land_temperature_2m_max`, `era5land_dewpoint_temperature_2m`, `era5land_surface_pressure`, `era5land_total_precipitation`, solar/thermal radiation, 10m U/V wind, soil moisture layers 1–4, snow depth water equivalent, FAO-56 & ERA5-Land PET) + `era5land_missing_fraction` | **Gridded Zarr archive only** (explicit user-supplied `gs://` or local `.zarr` URI; third-party sources are disabled to prevent 0.25° ERA5 substitution) |
| **CPC Global Precip** (`CPC`) | Daily Gauge | $0.5^\circ$ (360 $\times$ 720) | N/A | **2 variables**: `cpc_precipitation` ($\text{mm/day}$) and `cpc_num_stations` (reporting rain-gauge count per cell) + `cpc_missing_fraction` | User-supplied gridded Zarr archive (`--source archive`), NOAA PSL NetCDF (`https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/`), or local CPC binary grids |
| **IMERG Early V07** (`IMERG`) | Daily / Half-Hourly Satellite | $0.1^\circ$ (1800 $\times$ 3600) | N/A | `imerg_precipitation` ($\text{mm/day}$) + `imerg_missing_fraction` | User-supplied gridded Zarr archive (`--source archive`) or NASA GES DISC (`GPM_3IMERGDE.07`) |
| **ECMWF IFS HRES** (`HRES`) | Operational NWP Forecast | $0.25^\circ$ (721 $\times$ 1440) | 10 days ($1 \dots 10$) | **5 variables** (`hres_total_precipitation`, `hres_temperature_2m`, `hres_surface_pressure`, `hres_surface_net_solar_radiation`, `hres_surface_net_thermal_radiation`) + `hres_missing_fraction` | User-supplied gridded Zarr archive (`--source archive`), ECMWF Open Data (`gs://ecmwf-open-data`, `--source open_data`), or explicit Zarr/GRIB store (`data_dir`) |

---

## Architecture & Data-Quality Invariants

### A. Exact Fractional Zonal Averaging & Coverage Auditing
- **Fractional Polygon Intersection**: Computes exact fractional overlap between basin boundaries (Shapely polygons from GeoJSON or Shapefiles) and raster grid cells, with latitude cosine weighting ($\cos(\phi)$) to account for spherical surface distortion.
- **No Out-of-Domain Nearest-Cell Snapping**: Sub-grid-scale polygons receive weight on a single cell **only** when the polygon genuinely lies inside that grid cell. Basins outside the grid domain receive zero weight and evaluate to `NaN` (`missing_fraction = 1.0`), never snapping to a distant edge pixel.
- **Companion Coverage Variable (`<prefix>_missing_fraction`)**: Every extracted dataset records the area-weighted fraction $[0.0, 1.0]$ of missing (`NaN`) pixels within each catchment at each timestep (`cpc_missing_fraction`, `era5land_missing_fraction`, `imerg_missing_fraction`, `hres_missing_fraction`).
- **No Hardcoded Bucket Paths or Placeholder Dates**: All `gs://` archive URIs, `start_date`, and `end_date` arguments must be explicitly supplied by the user.

### B. Caravan Harmonization & Unit Standardization
- Converts cumulative energy fluxes ($\text{J/m}^2$) to daily-mean rates ($\text{W/m}^2$).
- Converts Kelvin temperatures to Celsius ($^\circ\text{C}$), including daily mean, daily minimum (`era5land_temperature_2m_min`), and daily maximum (`era5land_temperature_2m_max`) from 24 hourly steps.
- Converts surface pressure from Pascals to $\text{kPa}$.
- Implements the **FAO-56 Penman-Monteith** formulation for reference evapotranspiration (PET).
- Converts HRES continuous accumulations into daily increments ($P_d = P_{24d} - P_{24(d-1)}$).

---

## Quickstart: Extracting from Gridded Archives & Upstream Feeds

### Python API — Gridded Zarr Archives (`source="archive"`)

```python
from multimet.timeseries_extractors import extract_multimet_serial

output_stores = extract_multimet_serial(
    basins="multimet/tests/test_data/shapefiles/us/us_basin_shapes.geojson",
    output_dir="/tmp/multimet_extracted",
    products=["CPC", "ERA5_LAND", "IMERG", "HRES"],
    start_date="2022-01-01",
    end_date="2022-01-05",
    source="archive",
    archive_stores={
        "CPC": "gs://<your-bucket>/gridded-data-archives/CPC/daily_surface.zarr",
        "ERA5_LAND": "gs://<your-bucket>/data/era5_land/daily_surface.zarr",
        "IMERG": "gs://<your-bucket>/gridded-data-archives/IMERG/daily_surface.zarr",
        "HRES": "gs://<your-bucket>/gridded-data-archives/HRES/daily_surface.zarr",
    },
)
```

### Command-Line Interface (CLI)

```bash
# Extract from GCS gridded archives (user-supplied URIs required)
extract-multimet \
  --basins_path multimet/tests/test_data/shapefiles/us/us_basin_shapes.geojson \
  --output_dir /tmp/multimet_extracted \
  --products CPC,ERA5_LAND,IMERG,HRES \
  --start_date 2022-01-01 \
  --end_date 2022-01-05 \
  --source archive \
  --archive-store CPC=gs://<your-bucket>/gridded-data-archives/CPC/daily_surface.zarr \
  --archive-store ERA5_LAND=gs://<your-bucket>/data/era5_land/daily_surface.zarr \
  --archive-store IMERG=gs://<your-bucket>/gridded-data-archives/IMERG/daily_surface.zarr \
  --archive-store HRES=gs://<your-bucket>/gridded-data-archives/HRES/daily_surface.zarr

# Extract CPC and IMERG directly from upstream agency HTTP feeds
extract-multimet \
  --basins_path multimet/tests/test_data/shapefiles/us/us_basin_shapes.geojson \
  --output_dir /tmp/multimet_upstream \
  --products CPC,IMERG \
  --start_date 2022-01-01 \
  --end_date 2022-01-02 \
  --source public
```

---

## Real-Time Operational Forcing Fetcher (`multimet-realtime`)

`RealtimeForcingFetcher` and `fetch_realtime_multimet` fetch live operational forecasts and spin-up observations (`HRES` from `gs://ecmwf-open-data`, `IMERG` from NASA GES DISC, and `CPC` from NOAA PSL) and write or incrementally append them into `<output_dir>/<PRODUCT>/timeseries.zarr`:

- **Cold-Start (`mode="coldstart"`)**: Fetches a 365-day historical spin-up window (`[t0 - 365d, t0]`) plus the 10-day operational forecast issued on `t0`. By default, historical spin-up dates prior to the forecast issue window use a 1-day lead-time optimization (`step=24h` only) to cut HRES spin-up download volume by $10\times$.
- **Hot-Start (`mode="hotstart"`)**: Inspects existing Zarr stores (and/or a saved `openhydronet` `.npz` state file or directory) to identify the latest valid date across all bands and basins, automatically re-fetching and healing any trailing `NaN` dates caused by upstream publication latency alongside newly elapsed days up to `t0`.

```python
from multimet.timeseries_extractors import fetch_realtime_multimet

# 1. Cold-Start (365-day spin-up + 10-day forecast)
cold_res = fetch_realtime_multimet(
    basins="multimet/tests/test_data/shapefiles/us/us_basin_shapes.geojson",
    output_dir="/tmp/realtime_forcing",
    mode="coldstart",
    reference_date="latest",
)

# 2. Hot-Start (incremental catch-up + trailing NaN healing)
hot_res = fetch_realtime_multimet(
    basins="multimet/tests/test_data/shapefiles/us/us_basin_shapes.geojson",
    output_dir="/tmp/realtime_forcing",
    mode="hotstart",
    reference_date="latest",
)
```

```bash
# CLI Cold-Start
multimet-realtime \
  --basins_path multimet/tests/test_data/shapefiles/us/us_basin_shapes.geojson \
  --output_dir /tmp/realtime_forcing \
  --mode coldstart \
  --reference_date latest

# CLI Hot-Start
multimet-realtime \
  --basins_path multimet/tests/test_data/shapefiles/us/us_basin_shapes.geojson \
  --output_dir /tmp/realtime_forcing \
  --mode hotstart \
  --reference_date latest
```
