---
name: data-structures
description: >-
  Canonical reference for all data structures, coordinates, dimensions, dtypes,
  chunking, spatial/temporal aggregation protocols, internal gridded archives
  (ERA5-Land, HRES, CPC, IMERG), Caravan and Caravan-MultiMet basin timeseries,
  catchment polygons, static attributes, and model tensors for Cold-Start and
  Hot-Start realtime forecasting. Use whenever reading, writing, transforming,
  validating, or modeling data in model/, multimet/, or return_periods/.
---

# Data Structures, Coordinates & Aggregation Reference (`flood-forecasting`)

This skill defines the schemas, coordinate systems, dimensions, data types, chunk layouts, and spatial/temporal aggregation protocols across `multimet/`, `model/`, and `gs://open-multimet/`.

---

## 1. Canonical External References & Public Archives

Every data structure in this repository is grounded in the **Caravan** and **Caravan-MultiMet** community standards. Consult and cite these primary references whenever modifying schemas or data pipelines:

1. **Caravan GitHub Repository (Frederik Kratzert et al.):**
   - Repository: <https://github.com/kratzert/Caravan>
2. **Caravan-MultiMet Example Python Notebook (`kratzert/Caravan`):**
   - GitHub Notebook: <https://github.com/kratzert/Caravan/blob/main/examples/Caravan_MultiMet_Extending_Caravan_with_Multiple_Weather_Nowcasts_and_Forecasts.ipynb>
   - Google Colab Notebook: <https://colab.research.google.com/github/kratzert/Caravan/blob/main/examples/Caravan_MultiMet_Extending_Caravan_with_Multiple_Weather_Nowcasts_and_Forecasts.ipynb>
3. **Caravan *Nature Scientific Data* Paper (Kratzert et al., 2023):**
   - Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., Gudmundsson, L., Hassidim, A., Klotz, D., Nevo, S., Shalev, G., & Matias, Y. (2023). *"Caravan - A global community dataset for large-sample hydrology."* *Scientific Data*, 10, 61.
   - Paper URL: <https://www.nature.com/articles/s41597-023-01975-w> (DOI: <https://doi.org/10.1038/s41597-023-01975-w>)
4. **Caravan Zenodo Repositories (Header Documentation & Version History):**
   - NetCDF Release (v1.6+): <https://zenodo.org/records/6522634> (DOI: <https://doi.org/10.5281/zenodo.6522634>)
   - CSV Release (v1.6+): <https://zenodo.org/records/15530021> (DOI: <https://doi.org/10.5281/zenodo.15530021>)
   - *Note:* Consult the Zenodo records only for header documentation, variable descriptions, and version changelogs (e.g., the v1.5 `_FAO_PM` vs. `_ERA5_LAND` PET update). The canonical Caravan dataset from Zenodo is already mirrored directly in `gs://open-multimet/caravan-old/` and standardized into cloud-native Zarr, Parquet, and GeoParquet in `gs://open-multimet/caravan-new/`.
5. **Caravan-MultiMet Paper & Zenodo Archives (Guy Shalev & Frederik Kratzert, 2024):**
   - Shalev, G., & Kratzert, F. (2024). *"Caravan MultiMet: Extending Caravan with Multiple Weather Nowcasts and Forecasts."* *arXiv preprint arXiv:2411.09459*.
   - Paper URL: <https://arxiv.org/abs/2411.09459> (DOI: <https://doi.org/10.48550/arXiv.2411.09459>)
   - Zenodo Part 1 (Nowcasts): <https://zenodo.org/records/14161235> (DOI: <https://doi.org/10.5281/zenodo.14161235>)
   - Zenodo Part 2 (Forecasts): <https://zenodo.org/records/14161281> (DOI: <https://doi.org/10.5281/zenodo.14161281>)
   - Public & Internal GCS Mirrors: `gs://caravan-multimet/v1.1/{PRODUCT}/timeseries.zarr/` and `gs://open-multimet/caravan-multimet/v1.1/{PRODUCT}/timeseries.zarr/`.

---

## 2. Unified Coordinate Philosophy & Aggregation Protocols

### 2.1 Unified Gridded Archive Coordinate Philosophy

All global gridded meteorological archives in `gs://open-multimet/` share a unified coordinate philosophy:

| Property | Standard Specification |
| :--- | :--- |
| **3D Nowcast Dimensions** | `("time", "latitude", "longitude")` |
| **4D Forecast Dimensions** | `("time", "lead_time", "latitude", "longitude")` |
| **CRS & Registration** | Geographic WGS84 (`EPSG:4326`), regular lat/lon grid with **cell-center registration** (each `(latitude[i], longitude[j])` is the center of a $\Delta\text{lat} \times \Delta\text{lon}$ box). |
| **`latitude` Coordinate** | `float32`, monotonically **ascending south-to-north** in `[-90.0, 90.0]` (`-89.75 .. 89.75` for `0.5°` CPC; `-89.95 .. 89.95` for `0.1°` IMERG; `-90.0 .. 90.0` for `0.1°` HRES). *(Note: Legacy `data/era5_land/daily_surface.zarr` has descending `90.0 .. -90.0` from native ECMWF GRIB order; all new gridded archives must use ascending south-to-north latitude, and `multimet` readers align either orientation automatically.)* |
| **`longitude` Coordinate** | `float32`, monotonically **ascending west-to-east** in `[-180.0, 180.0)` (`-180.0 .. 179.9` for `0.1°` ECMWF grids; `-179.95 .. 179.95` for `0.1°` IMERG; `-179.75 .. 179.75` for `0.5°` CPC). Upstream `[0, 360)` grids are shifted via `np.where(lon >= 180.0, lon - 360.0, lon)` and sorted ascending. |
| **`time` Coordinate** | `datetime64[ns]`, strictly contiguous daily (`1D`) **UTC+0** midnight timestamps (`YYYY-MM-DDT00:00:00`), **left-labeled** so date $D$ covers $[D\text{ 00:00:00}, D+1\text{ 00:00:00})$. |
| **`lead_time` Coordinate** | **1-indexed daily steps** (`int64` `1 .. L` with `{"units": "days"}` in gridded archives, or `timedelta64[ns]` `1 day .. L days` in basin Zarr stores). On forecast issue date $D$ (`00:00 UTC`), `lead_time = k` (0-based index `k - 1`) covers the valid 24-hour UTC window $[D + (k - 1)\text{ 00:00:00}, D + k\text{ 00:00:00})$. |
| **Data Type & Missing Values** | `float32` for all physical data variables, with `fill_value = NaN` (`np.nan`). |
| **Chunk Layout** | 1 full global spatial slice per daily chunk (`(1, N_lat, N_lon)` for `0.1°` nowcasts, `(30, 360, 720)` for `0.5°` CPC, and `(1, L, N_lat, N_lon)` for forecasts) to support lock-free parallel daily writes and fast time-range slicing. |

### 2.2 Temporal Aggregation Protocol

1. **UTC+0 vs. Gauge Local Standard Time:**
   - **Original Caravan (`caravan-old/`):** Meteorological forcings from ERA5-Land were shifted to each gauge's **local standard time** (ignoring Daylight Saving Time) prior to daily aggregation so that daily meteorological windows match local gauge streamflow reporting windows (Kratzert et al., 2023).
   - **Caravan-MultiMet (`caravan-multimet/` & `caravan-new/`) and Gridded Archives:** All meteorological products—including `ERA5_LAND`—are aggregated in **UTC+0** (Shalev & Kratzert, 2024). Operational weather forecasts (`HRES`, `GRAPHCAST`, `CHIRPS_GEFS`) are initialized globally at `00:00 UTC`, and daily gauge/satellite products (`CPC`, `CHIRPS`) are published as fixed daily grids without sub-daily data.
2. **Left-Labeled Daily Windows & 1-Indexed Forecast Lead Alignment:**
   - Every daily timestamp $D$ is **left-labeled** and represents the 24-hour UTC interval $[D\text{ 00:00:00}, D+1\text{ 00:00:00})$.
   - For a forecast initialized at $D\text{ 00:00:00 UTC}$:
     - `lead_time = 1 day` (`isel(lead_time=0)`) covers $[D\text{ 00:00:00}, D+1\text{ 00:00:00})$—**the exact same 24-hour window as `date = D` in 2D nowcast products**.
     - `lead_time = k days` (`isel(lead_time=k-1)`) covers $[D + (k - 1)\text{ days 00:00:00}, D + k\text{ days 00:00:00})$.
     - The valid calendar date of forecast `(date = D, lead_time = k days)` is therefore $D_{\text{valid}} = D + (k - 1)\text{ days}$.
3. **Sub-Daily State Averaging & Flux De-Accumulation Rules:**
   - **Instantaneous state variables** (`temperature_2m`, `dewpoint_temperature_2m`, `surface_pressure`, `u/v_component_of_wind_10m`, `snow_depth_water_equivalent`, `volumetric_soil_water_layer_1..4`): Aggregated over $[D\text{ 00:00}, D+1\text{ 00:00})$ into daily mean, daily minimum (`_min`), and daily maximum (`_max`). Temperatures are converted from $\text{K}$ to $^\circ\text{C}$ ($-273.15$), pressure from $\text{Pa}$ to $\text{kPa}$ ($\times 10^{-3}$), and SWE from $\text{m}$ to $\text{mm}$ ($\times 1000$).
   - **Accumulated water fluxes** (`total_precipitation`, `potential_evaporation_*`): Summed/de-accumulated over the 24-hour UTC window and converted from $\text{m}$ to $\text{mm/day}$ ($\times 1000$, clipped at $\ge 0.0$ for precipitation, SWE, and FAO-PM PET).
     - *ECMWF ERA5-Land:* Hourly GRIB accumulations reset at `00:00 UTC`, so the `00:00 UTC` step of day $D+1$ holds the full 24-hour accumulation for day $D$.
     - *ECMWF HRES (`00Z`):* Cumulative forecast variables (`tp`, `ssr`, `str`) accumulate from initialization $t_0 = D\text{ 00:00 UTC}$. The daily increment for lead day $k \in \{1,\dots,10\}$ is $\text{val}(t_0 + 24k\text{ h}) - \text{val}(t_0 + 24(k-1)\text{ h})$ (with $\text{val}(t_0) = 0$).
     - *NASA IMERG Early V07 (`GPM_3IMERGHHE.07`):* Requires all 48 half-hourly HDF5 granules (`-S000000` through `-S233000`) to exist for day $D$. At every `(lat, lon)` pixel, all 48 half-hourly rates ($\text{mm/hr}$) must be finite (`valid_counts == 48`, multiplied by $0.5\text{ hr}$); any pixel with $< 48$ valid half-hourly observations is set to `NaN`.
   - **Accumulated radiation fluxes** (`surface_net_solar_radiation`, `surface_net_thermal_radiation`): 24-hour accumulated energy density ($\text{J m}^{-2}$) divided by $86,400\text{ s/day}$ to yield daily mean power density in $\text{W m}^{-2}$ (positive downwards).

### 2.3 Spatial (Zonal) Aggregation Protocol

Catchment-averaged timeseries (`multimet/utils/zonal.py`) are extracted from gridded archives via exact spherical area-weighted intersection:

1. **Cosine-Latitude Area Weighting (`ZonalWeightCalculator` & `ZonalWeightMatrix`):**
   - For a catchment polygon $P$ in `EPSG:4326` and grid cell box $C_{i,j} = [\text{lon}_j - \frac{\Delta\text{lon}}{2}, \text{lon}_j + \frac{\Delta\text{lon}}{2}] \times [\text{lat}_i - \frac{\Delta\text{lat}}{2}, \text{lat}_i + \frac{\Delta\text{lat}}{2}]$, the unnormalized cell weight is:
     $$w_{i,j} = \text{Area}_{\text{deg}^2}(P \cap C_{i,j}) \cdot \max\bigl(0, \cos(\text{lat}_i)\bigr)$$
     normalized over all cells intersecting basin $b$: $\tilde{w}_{b,i,j} = w_{b,i,j} / \sum_{u,v} w_{b,u,v}$.
   - Multi-basin weights are packed into a sparse CSR operator $\mathbf{W} \in \mathbb{R}^{N_{\text{basins}} \times (N_{\text{lat}} N_{\text{lon}})}$ (`scipy.sparse.csr_matrix`), reducing 3D/4D climate arrays across thousands of basins via sparse matrix multiplication ($\mathbf{W}\mathbf{X}$).
2. **$\ge 80\%$ Valid Coverage Threshold & Companion `missing_fraction`:**
   - At each timestep, let $W_b = \sum_{i,j} \tilde{w}_{b,i,j} \in \{0.0, 1.0\}$ be the total weight of basin $b$ and $V_b = \sum_{(i,j)\text{ finite}} \tilde{w}_{b,i,j}$ be the weight covered by finite (non-`NaN`) grid cells.
   - Every extracted dataset records `<prefix>_missing_fraction` $= 1.0 - V_b / W_b \in [0.0, 1.0]$ (`1.0` if out of domain).
   - If valid coverage $V_b / W_b < 0.80$ (`MIN_VALID_COVERAGE_FRACTION = 0.80`), the basin value is set to **`NaN`**. Otherwise, it is renormalized over valid cells:
     $$\bar{x}_b = \frac{\sum_{(i,j)\text{ finite}} \tilde{w}_{b,i,j}\, x_{i,j}}{V_b}$$
   - Out-of-domain polygons (e.g., basins poleward of $\pm 50^\circ$ latitude in `CHIRPS`) are **never** snapped to nearest grid cells; they receive empty weights, `NaN` values, and `missing_fraction = 1.0`.

---

## 3. Internal Historical Gridded Archives (`gs://open-multimet/`)

We maintain four native-resolution Analysis-Ready Cloud-Optimized (ARCO) global daily surface Zarr archives in `gs://open-multimet/` that power `multimet/timeseries_extractors/` and `multimet/static_extractor/`:

### 3.1 Summary of Internal Gridded Archives

| Product | GCS Zarr URI | Dims & Shape | Grid Resolution & Coordinates | Chunk Shape | Time Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ERA5_LAND`** | `gs://open-multimet/data/era5_land/daily_surface.zarr` | `("time", "latitude", "longitude")`<br>`(17053, 1801, 3600)` | `0.1° × 0.1°`<br>`latitude`: `90.0 .. -90.0` (`1801`)<br>`longitude`: `-180.0 .. 179.9` (`3600`) | `(1, 1801, 3600)` | `1980-01-01` to `2026-09-08` |
| **`HRES`** | `gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr` | `("time", "lead_time", "latitude", "longitude")`<br>`(3926, 10, 1801, 3600)` | `0.1° × 0.1°`<br>`latitude`: `-90.0 .. 90.0` (`1801`)<br>`longitude`: `-180.0 .. 179.9` (`3600`)<br>`lead_time`: `1 .. 10` (`int64`, days) | `(1, 10, 1801, 3600)` | `2016-01-01` to `2026-09-30` |
| **`CPC`** | `gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr` | `("time", "latitude", "longitude")`<br>`(17426+, 360, 720)` | `0.5° × 0.5°`<br>`latitude`: `-89.75 .. 89.75` (`360`)<br>`longitude`: `-179.75 .. 179.75` (`720`) | `(30, 360, 720)` | `1979-01-01` to present |
| **`IMERG`** | `gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr` | `("time", "latitude", "longitude")`<br>`(9605+, 1800, 3600)` | `0.1° × 0.1°`<br>`latitude`: `-89.95 .. 89.95` (`1800`)<br>`longitude`: `-179.95 .. 179.95` (`3600`) | `(1, 1800, 3600)` | `2000-06-01` to present |

### 3.2 `ERA5-Land` Internal Gridded Archive (`data/era5_land/daily_surface.zarr`)

- **Provenance & Construction:** Built directly from raw hourly ECMWF ERA5-Land GRIB files (`ERA5_Land_Hourly_{YYYYMMDD}_default_{HH}.grib`, DOI: `10.24381/cds.e2161bac`) stored in Google's internal Earth Engine (`gestalt-ingest`) backend archive (`gs://open-multimet/data/era5_land/daily_surface.zarr`, backing Earth Engine's `ECMWF/ERA5_LAND/HOURLY` catalog), bypassing the Earth Engine API to preserve native `0.1°` (`1801 × 3600`) floating-point precision.
- **39 `float32` Surface Variables:**
  - **12 state & radiation variables with daily mean, `_min`, and `_max` (`36` bands):**
    - `era5land_temperature_2m{,_min,_max}` ($^\circ\text{C}$)
    - `era5land_dewpoint_temperature_2m{,_min,_max}` ($^\circ\text{C}$)
    - `era5land_surface_pressure{,_min,_max}` ($\text{kPa}$)
    - `era5land_u_component_of_wind_10m{,_min,_max}` ($\text{m s}^{-1}$)
    - `era5land_v_component_of_wind_10m{,_min,_max}` ($\text{m s}^{-1}$)
    - `era5land_snow_depth_water_equivalent{,_min,_max}` ($\text{mm}$)
    - `era5land_surface_net_solar_radiation{,_min,_max}` ($\text{W m}^{-2}$)
    - `era5land_surface_net_thermal_radiation{,_min,_max}` ($\text{W m}^{-2}$)
    - `era5land_volumetric_soil_water_layer_1{,_min,_max}` ($\text{m}^3\text{ m}^{-3}$, `0–7 cm`)
    - `era5land_volumetric_soil_water_layer_2{,_min,_max}` ($\text{m}^3\text{ m}^{-3}$, `7–28 cm`)
    - `era5land_volumetric_soil_water_layer_3{,_min,_max}` ($\text{m}^3\text{ m}^{-3}$, `28–100 cm`)
    - `era5land_volumetric_soil_water_layer_4{,_min,_max}` ($\text{m}^3\text{ m}^{-3}$, `100–289 cm`)
  - **3 daily accumulation variables (`3` bands):**
    - `era5land_total_precipitation` ($\text{mm/day}$)
    - `era5land_potential_evaporation_DEPRECATED` ($\text{mm/day}$, native ERA5-Land `pev`)
    - `era5land_potential_evaporation_FAO_PENMAN_MONTEITH` ($\text{mm/day}$, FAO-56 Penman-Monteith reference ET computed from daily ERA5-Land temperature, dewpoint, wind speed, pressure, and net radiation)

### 3.3 `HRES` Internal Gridded Archive (`gridded-data-archives/HRES/daily_surface.zarr`)

- **Provenance & 3-Tier Contiguous Construction (`2016-01-01` to present):**
  1. **`2016-01-01` to `2023-01-10` (WeatherBench 2 HRES Archive):** Ingested from `gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr` (`00Z` initialization, lead days `1..10`). Provides `hres_temperature_2m`, `hres_surface_pressure`, and `hres_total_precipitation`. Because WeatherBench 2 HRES does not archive surface radiation fluxes or daily min/max temperature, `hres_surface_net_solar_radiation`, `hres_surface_net_thermal_radiation`, `hres_temperature_2m_min`, and `hres_temperature_2m_max` are **`NaN`** during `2016-01-01 .. 2023-01-10`.
  2. **`2023-01-11` to `2023-07-11` (Google Flood Forecasting Internal ECMWF HRES Archive):** Ingested from `gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr{YYYY-MM-DD}-tp-2t-sp-ssr-str-sf.nc` (`00Z` initialization, lead days `1..10`). Provides all 5 core MultiMet HRES bands (`hres_temperature_2m`, `hres_surface_pressure`, `hres_total_precipitation`, `hres_surface_net_solar_radiation`, `hres_surface_net_thermal_radiation`).
  3. **`2023-07-12` to present (ECMWF Open Data Operational IFS HRES GRIB2 Archive):** Ingested from `gs://ecmwf-open-data/<YYYYMMDD>/00z/` (`0p4-beta/oper` on `2023-07-12`, `ifs/0p25/oper` from `2023-07-13` onward). Provides all 7 variables including `hres_temperature_2m_min` and `hres_temperature_2m_max`.
- **7 `float32` Forecast Variables (`dims: ("time", "lead_time", "latitude", "longitude")`):**
  - `hres_temperature_2m`, `hres_temperature_2m_min`, `hres_temperature_2m_max` ($^\circ\text{C}$)
  - `hres_total_precipitation` ($\text{mm/day}$)
  - `hres_surface_pressure` ($\text{kPa}$)
  - `hres_surface_net_solar_radiation` ($\text{W m}^{-2}$)
  - `hres_surface_net_thermal_radiation` ($\text{W m}^{-2}$)

### 3.4 `CPC` & `IMERG` Internal Gridded Archives

- **`CPC` (`gridded-data-archives/CPC/daily_surface.zarr`):**
  - Built and incrementally extended via `build-cpc-archive` (`multimet/gridded_archive_builders/build_cpc_archive.py`) from NOAA PSL yearly NetCDF files (`https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc`).
  - Flips raw PSL descending latitude (`89.75 .. -89.75`) to ascending (`-89.75 .. 89.75`), shifts `[0.25 .. 359.75]` longitude to `[-179.75 .. 179.75]`, and writes `cpc_precipitation` (`float32`, $\text{mm/day}$, ocean/missing cells as `NaN`).
- **`IMERG` (`gridded-data-archives/IMERG/daily_surface.zarr`):**
  - Built and incrementally extended via `build-imerg-archive` (`multimet/gridded_archive_builders/build_imerg_archive.py`) from NASA GES DISC `GPM_3IMERGDE.07` daily NetCDF-4 files and Google's internal mirror of `GPM_3IMERGHHE.07` half-hourly HDF5 granules (`gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr`).
  - Writes `imerg_precipitation` (`float32`, $\text{mm/day}$) on `(-89.95 .. 89.95) × (-179.95 .. 179.95)`.

---

## 4. Basin Timeseries Data Structures (`Caravan` & `Caravan-MultiMet`)

### 4.1 Basin Identifier Convention (`basin` / `gauge_id`)

- Every catchment is uniquely identified by a lowercase string `{subdataset}_{original_id}` (e.g., `camels_01013500`, `camelsgb_10002`, `hysets_01AF007`, `grdc_6335020`).
- Named **`basin`** (`<U22` or `object` string coordinate) in all Zarr/NetCDF `xarray.Dataset` stores and **`gauge_id`** in CSV, Parquet, GeoParquet, GeoJSON, and Shapefile tables.

### 4.2 Streamflow Target & Original Caravan Timeseries (`model/datasetzoo/caravan.py`)

- **Target Variable (`streamflow`):** Area-normalized daily surface runoff / specific discharge in **$\text{mm/day}$** (`float32`), aggregated in **gauge local standard time**:
  $$Q_{\text{mm/day}} = \frac{Q_{\text{m}^3/\text{s}} \times 86,400\text{ s/day}}{\text{Area}_{\text{km}^2} \times 10^6\text{ m}^2/\text{km}^2} \times 1000\text{ mm/m} = \frac{86.4 \cdot Q_{\text{m}^3/\text{s}}}{\text{Area}_{\text{km}^2}}$$
- **Supported On-Disk Layouts (`load_caravan_timeseries`):**
  1. **Cloud-Native Zarr Store (Preferred):** `<targets_data_dir>/{streamflow.zarr,targets.zarr,timeseries.zarr}` with dimensions `("basin", "date")`, `date` as `datetime64[ns]`, and `streamflow` as `float32`.
  2. **Legacy Per-Basin NetCDF / CSV (`caravan-old/` & Zenodo v1.6):** `<targets_data_dir>/timeseries/{netcdf,csv}/{subdataset}/{basin}.{nc,csv}` indexed by `date` (`1950-01-01` to `2020-12-31`), containing `streamflow` ($\text{mm/day}$) alongside 39 local-time ERA5-Land variables (`total_precipitation_sum`, `potential_evaporation_sum_ERA5_LAND`, `potential_evaporation_sum_FAO_PENMAN_MONTEITH`, `temperature_2m_mean/min/max`, `dewpoint_temperature_2m_mean/min/max`, `snow_depth_water_equivalent_mean/min/max`, `surface_net_solar_radiation_mean/min/max`, `surface_net_thermal_radiation_mean/min/max`, `surface_pressure_mean/min/max`, `u_component_of_wind_10m_mean/min/max`, `v_component_of_wind_10m_mean/min/max`, `volumetric_soil_water_layer_1..4_mean/min/max`).

### 4.3 Caravan-MultiMet 2D Nowcast Zarr Stores (`dims: ("basin", "date")`)

- **Store Path Pattern:** `<dynamics_data_dir>/{PRODUCT}/timeseries.zarr/`
- **Coordinates:**
  - `basin`: `str` (`<U22`)
  - `date`: `datetime64[ns]` (UTC+0 left-labeled daily timestamps)
- **Zarr Chunking in `multimet` (`MultiMetZarrWriter`):** `{"basin": N_basins, "date": 1}` (`DEFAULT_CHUNKS_NOWCAST = {"basin": -1, "date": 1}`) so distributed workers can write individual daily slices without locks.
- **Nowcast Products, Bands & Units (`float32`):**

| Product | Native Grid | Canonical v1.1 Date Range | Band Name(s) | Units | Notes & Companion Variables |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`CPC`** | `0.5°` global land | `1979-01-01` – `2024-07-31` (`v1.1`)<br>`1979-01-01` – present (`caravan-new`) | `cpc_precipitation`<br>`cpc_num_stations` *(optional)* | $\text{mm/day}$<br>$\text{count}$ | Companion `cpc_missing_fraction` $\in [0, 1]$. Sporadic missing dates/basins are `NaN`. |
| **`IMERG`** | `0.1°` global | `2000-05-01` – `2024-10-31` (`v1.1`)<br>`2000-06-01` – present (`caravan-new`) | `imerg_precipitation` | $\text{mm/day}$ | NASA GPM IMERG V07 Early (~4h latency). Companion `imerg_missing_fraction` $\in [0, 1]$. |
| **`CHIRPS`** | `0.05°` quasi-global (`[-50°, 50°]` lat) | `1981-01-01` – `2024-07-30` (`v1.1`) | `chirps_precipitation` | $\text{mm/day}$ | Only basins wholly within `[-50°, 50°]` latitude are valid (`NaN` poleward). Companion `chirps_missing_fraction`. |
| **`ERA5_LAND`** | `0.1°` global land | `1950-01-01` – `2024-10-31` (`v1.1`)<br>`1980-01-01` – `2026-09-08` (internal archive) | `era5land_dewpoint_temperature_2m{,_min,_max}`<br>`era5land_potential_evaporation_DEPRECATED`<br>`era5land_potential_evaporation_FAO_PENMAN_MONTEITH`<br>`era5land_snow_depth_water_equivalent{,_min,_max}`<br>`era5land_surface_net_solar_radiation{,_min,_max}`<br>`era5land_surface_net_thermal_radiation{,_min,_max}`<br>`era5land_surface_pressure{,_min,_max}`<br>`era5land_temperature_2m{,_min,_max}`<br>`era5land_total_precipitation`<br>`era5land_u_component_of_wind_10m{,_min,_max}`<br>`era5land_v_component_of_wind_10m{,_min,_max}`<br>`era5land_volumetric_soil_water_layer_1..4{,_min,_max}` | $^\circ\text{C}$<br>$\text{mm/day}$<br>$\text{mm/day}$<br>$\text{mm}$<br>$\text{W m}^{-2}$<br>$\text{W m}^{-2}$<br>$\text{kPa}$<br>$^\circ\text{C}$<br>$\text{mm/day}$<br>$\text{m s}^{-1}$<br>$\text{m s}^{-1}$<br>$\text{m}^3\text{ m}^{-3}$ | Public `caravan-multimet/v1.1` includes the 17 core daily bands (means + `temperature_2m_min/max` + accumulations); `multimet` gridded archive extraction supports all 39 bands plus `era5land_missing_fraction`. |

### 4.4 Caravan-MultiMet 3D Forecast Zarr Stores (`dims: ("basin", "date", "lead_time")`)

- **Store Path Pattern:** `<dynamics_data_dir>/{PRODUCT}/timeseries.zarr/`
- **Coordinates:**
  - `basin`: `str` (`<U22`)
  - `date`: `datetime64[ns]` (`00:00 UTC` forecast initialization / issue date)
  - `lead_time`: `timedelta64[ns]` (`1 day .. L days`, stored in Zarr as `int64` `[1, 2, ..., L]` with CF attribute `{"units": "days"}` so `xr.open_zarr(..., decode_timedelta=True)` decodes to `timedelta64[ns]`).
- **Zarr Chunking in `multimet` (`MultiMetZarrWriter`):** `{"basin": N_basins, "date": 1, "lead_time": L}` (`DEFAULT_CHUNKS_FORECAST = {"basin": -1, "date": 1, "lead_time": -1}`).
- **Forecast Products, Lead Horizons, Bands & Units (`float32`):**

| Product | Native Grid | Canonical v1.1 Issue Date Range | `lead_time` Horizon | Band Name(s) | Units | Notes & Companion Variables |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`HRES`** | `0.1°` / `0.25°` global | `2012-07-01` – `2024-09-30` (`v1.1`)<br>`2016-01-01` – present (`caravan-new`) | `1 .. 10 days` | `hres_surface_net_solar_radiation`<br>`hres_surface_net_thermal_radiation`<br>`hres_surface_pressure`<br>`hres_temperature_2m`<br>`hres_total_precipitation` | $\text{W m}^{-2}$<br>$\text{W m}^{-2}$<br>$\text{kPa}$<br>$^\circ\text{C}$<br>$\text{mm/day}$ | ECMWF IFS HRES `00Z` forecast. Companion `hres_missing_fraction`. |
| **`GRAPHCAST`** | `0.25°` / `0.1°` global | `2016-01-02` – `2023-12-21` (`v1.1`) | `1 .. 10 days` | `graphcast_temperature_2m`<br>`graphcast_total_precipitation`<br>`graphcast_u_component_of_wind_10m`<br>`graphcast_v_component_of_wind_10m` | $^\circ\text{C}$<br>$\text{mm/day}$<br>$\text{m s}^{-1}$<br>$\text{m s}^{-1}$ | Initialized from ECMWF HRES `00Z` states (6-hourly steps aggregated to daily). |
| **`CHIRPS_GEFS`** | `0.05°` quasi-global (`[-50°, 50°]` lat) | `2000-01-01` – `2024-01-31` (`v1.1`) | `1 .. 16 days` | `chirpsgefs_precipitation` | $\text{mm/day}$ | CHIRPS-compatible bias-corrected NCEP GEFS precipitation forecast. Companion `chirpsgefs_missing_fraction`. |

---

## 5. Catchment Polygons & Pour-Point Coordinates (`catchment_delineation`)

### 5.1 Pour-Point Coordinate Tables (`coordinates.csv`)

1. **Global Master Coordinate Table (`gs://open-multimet/caravan-new/coordinates.csv`):**
   - Covers all **26,708** Caravan, community-extension, and Google-internal basins.
   - Schema: `gauge_id` (`str`), `gauge_lat` (`float64`, decimal degrees `[-90, 90]`), `gauge_lon` (`float64`, decimal degrees `[-180, 180]`), `original_id` (`str`), `collection` (`caravan-original` | `caravan-extensions` | `google-internal`), `subdataset` (`str`).
2. **Per-Subdataset Coordinate Table (`shapefiles/<subdataset>/coordinates.csv`):**
   - Schema: `gauge_id,gauge_lat,gauge_lon,original_id`.
3. **CLI Coordinate Column Allowlists (`delineate-catchment`):**
   - Latitude column: `latitude`, `lat`, `gauge_lat`, `caravan:gauge_lat`, `outlet_lat`, `pour_point_lat`.
   - Longitude column: `longitude`, `lon`, `long`, `lng`, `gauge_lon`, `caravan:gauge_lon`, `outlet_lon`, `pour_point_lon`.
   - ID column: `gauge_id`, `catchment_id`, `station_id`, `hybas_id`, `id`, `caravan:gauge_id`.

### 5.2 Watershed Polygon Schema (`shapefiles/` & `shapefiles-rederived/`)

- **Supported Formats:** `.geoparquet` (recommended cloud-native format), `.geojson`, `.gpkg`, and ESRI Shapefile suite (`.shp`, `.shx`, `.dbf`, `.prj` in `EPSG:4326`, `.cpg` in `UTF-8`).
- **Coordinate Reference System (CRS):** Strictly **`EPSG:4326`** (WGS84 geographic 2D).
- **Required Columns / Properties:**

| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| **`gauge_id`** | `str` | — | Unique basin identifier (`{subdataset}_{id}`). Indexed as `basins_gdf.index` when loaded by `load_basin_geometries()`. |
| **`area`** | `float64` | $\text{km}^2$ | Geodesic catchment drainage area computed on the WGS84 ellipsoid (`pyproj.Geod(ellps="WGS84")`). Never default missing area to `0.0` (raise `KeyError`). |
| **`gauge_lat`** | `float64` | degrees | Pour-point / gauge latitude in `[-90.0, 90.0]`. |
| **`gauge_lon`** | `float64` | degrees | Pour-point / gauge longitude in `[-180.0, 180.0]`. |
| **`geometry`** | `Polygon` \| `MultiPolygon` | `EPSG:4326` | Topological watershed boundary repaired via `shapely.make_valid`. In batch delineation, out-of-coverage or aborted basins emit `geometry: None` with an explicit `status` code (`MISSING_DATA_*` or `out_of_coverage`) rather than a truncated polygon. |

- **DEM Delineation Grid (`gs://open-multimet/ancillary-data/dems/`):**
  - 3-arcsecond (`1/1200°` $\approx 90\text{ m}$) D8 flow-direction GeoTIFF tiles (`5° × 5°`, `6000 × 6000` `uint8` pixels) using standard ESRI D8 encoding (`1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE`).

---

## 6. Static Catchment Attributes (`static_extractor` & `attributes/`)

### 6.1 Storage Formats & Table Split

Static catchment attributes are indexed by `basin` (`gauge_id`) with no time dimension (`dims: ("basin",)`):

1. **Cloud-Native Zarr (`attributes.zarr` / `statics.zarr`):**
   - 1D `xarray.Dataset` with dimension `("basin",)`, chunked `{"basin": -1}`, all numeric variables stored as `float32`. Loaded first if present by `load_caravan_attributes()` (`model/datasetzoo/caravan.py`).
2. **Unified Parquet (`attributes_<subdataset>.parquet`):**
   - Outer-joined Parquet table in `gs://open-multimet/caravan-new/<collection>/attributes/<subdataset>/attributes_<subdataset>.parquet` indexed by `gauge_id`.
3. **Canonical Caravan 3-CSV Split (`attributes/<subdataset>/`):**
   - `attributes_hydroatlas_<subdataset>.csv`: ~198 HydroATLAS v1.0 Level 12 physical, terrain, soil, land-cover, geology, and anthropogenic attributes.
   - `attributes_caravan_<subdataset>.csv`: Long-term ERA5-Land climate signatures (`1981-01-01` to `2020-12-31`).
   - `attributes_other_<subdataset>.csv`: Provider metadata (`gauge_id`, `gauge_name`, `gauge_lat`, `gauge_lon`, `country`, `area`, `elevation`).

### 6.2 HydroATLAS Level 12 Spatial Aggregation Protocol (`StaticAttributesExtractor`)

Given a catchment polygon $P$ and intersecting HydroATLAS Level 12 (`BasinATLAS_v10`) sub-basins $\{S_k\}$ with geodesic intersection areas $w_k = \text{Area}_{\text{WGS84}}(P \cap S_k)$ ($\text{km}^2$):

1. **Continuous Sub-Basin Areal Properties (`_s*` suffixes, e.g., `ele_mt_sav`, `slp_dg_sav`, `cly_pc_sav`, `for_pc_sse`, `swc_pc_syr`):**
   - Aggregated by exact area-weighted mean: $\bar{v} = \frac{\sum_k w_k v_k}{\sum_k w_k}$.
   - Pre-accumulated HydroATLAS upstream attributes (`_u*` suffixes in `UPSTREAM_PROPERTIES`) are ignored in favor of integrating local sub-basin (`_s*`) properties directly over $P$.
2. **Discrete Categorical Properties (`MAJORITY_PROPERTIES`: `clz_cl_smj`, `cls_cl_smj`, `glc_cl_smj`, `pnv_cl_smj`, `wet_cl_smj`, `tbi_cl_smj`, `tec_cl_smj`, `fmh_cl_smj`, `fec_cl_smj`, `lit_cl_smj`):**
   - Aggregated by area-weighted majority vote across intersecting sub-basins (and expanded into per-class area fraction columns where applicable).
3. **Pour-Point / Terminal Upstream Properties (`POUR_POINT_PROPERTIES`: `dis_m3_pmn`, `dis_m3_pmx`, `dis_m3_pyr`, `lkv_mc_usu`, `rev_mc_usu`, `ria_ha_usu`, `riv_tc_usu`, `pop_ct_usu`, `dor_pc_pva`):**
   - Computed by `compute_pour_point_properties()`, which starts at the sub-basin with maximum fractional overlap ($w_k / \text{SUB\_AREA}_k$), traverses the `NEXT_DOWN` topological graph until exiting the catchment polygon (overlap $< 50\%$) or reaching the ocean (`NEXT_DOWN == 0`), and sums the property across the direct terminal outlet sub-basin(s).

### 6.3 Caravan Climate Indices & v1.5+ `_FAO_PM` vs. `_ERA5_LAND` Variants

Computed over the 40-year reference period (`1981-01-01` to `2020-12-31`). Following Caravan v1.5 (which added FAO-56 Penman-Monteith PET because native ERA5-Land `potential_evaporation` represents open-water evaporation and can unrealistically overestimate PET in arid catchments), all PET-dependent indices are stored in **both** variants:

| Attribute Name(s) | Units | Definition |
| :--- | :--- | :--- |
| **`p_mean`** | $\text{mm/day}$ | Mean daily precipitation (`era5land_total_precipitation`). |
| **`pet_mean_FAO_PM`** / **`pet_mean_ERA5_LAND`** (legacy alias: `pet_mean`) | $\text{mm/day}$ | Mean daily potential evapotranspiration from FAO-56 Penman-Monteith vs. native ERA5-Land `pev`. |
| **`aridity_FAO_PM`** / **`aridity_ERA5_LAND`** (legacy alias: `aridity`) | ratio | $\text{pet\_mean} / \text{p\_mean}$. |
| **`frac_snow`** | fraction $[0, 1]$ | Fraction of total precipitation falling on days with daily mean temperature $< 0^\circ\text{C}$. |
| **`moisture_index_FAO_PM`** / **`moisture_index_ERA5_LAND`** (legacy alias: `moisture_index`) | index $\in [-1, 1]$ | Mean annual Knoben et al. (2018) moisture index computed from monthly $P$ and $\text{PET}$ climatologies. |
| **`seasonality_FAO_PM`** / **`seasonality_ERA5_LAND`** (legacy alias: `seasonality`) | index $\in [0, 2]$ | Knoben et al. (2018) seasonality index ($\max(\text{MI}_m) - \min(\text{MI}_m)$ across 12 calendar months). |
| **`high_prec_freq`** | $\text{days/yr}$ or fraction | Frequency of extreme precipitation days ($\ge 5 \times \text{p\_mean}$). |
| **`high_prec_dur`** | $\text{days}$ | Mean duration of consecutive extreme precipitation days ($\ge 5 \times \text{p\_mean}$). |
| **`low_prec_freq`** | $\text{days/yr}$ or fraction | Frequency of dry days ($< 1\text{ mm/day}$). |
| **`low_prec_dur`** | $\text{days}$ | Mean duration of consecutive dry spells ($< 1\text{ mm/day}$). |

---

## 7. Model Data Structures: Training, Cold-Start & Hot-Start Realtime Forecasting (`model/`)

### 7.1 Lead-Time-Aware Feature Unioning (`model/datautils/union_features.py`)

When `union_mapping: {primary_var: fallback_var}` is configured (e.g., filling historical gaps in `cpc_precipitation`, `imerg_precipitation`, `hres_*`, or `graphcast_*` with `era5land_*`), `union_features()` combines each primary variable with its fallback via `combine_first` while preserving exact valid-date alignment (`_valid_date_offset_days(lead_time) = k - 1`):

1. **2D Nowcast $\leftarrow$ 2D Nowcast (`("basin", "date") ← ("basin", "date")`):** Direct `feature_da.combine_first(mask_feature_da)` on `date`.
2. **3D Forecast $\leftarrow$ 2D Nowcast (`("basin", "date", "lead_time") ← ("basin", "date")`):** For each `lead_time = k days`, shifts the 2D fallback along `date` by `-(k - 1)` days (`da.shift(date=-(k - 1))`) so that a forecast issued on `date = D` at lead `k` is filled with the fallback reanalysis value from its valid date `D + k - 1`.
3. **2D Nowcast $\leftarrow$ 3D Forecast (`("basin", "date") ← ("basin", "date", "lead_time")`):** Selects `min_lead_time` (`1 day`, valid-date offset `0`) from the 3D fallback and combines on `date`.

### 7.2 Normalization Scaler Store (`scaler.zarr`)

- **Path:** `<run_dir>/scaler.zarr` (or loaded from `<base_run_dir>/scaler.zarr` during fine-tuning and evaluation).
- **Schema (`model/datautils/scaler.py`):**
  - `xarray.Dataset` with single coordinate `parameter = ["center", "scale", "mean", "std"]` (`dims: ("parameter",)`).
  - Contains a `float32` DataArray of shape `(4,)` for every loaded static, hindcast, forecast, and target variable, plus `{var}_obs` and `{var}_sim` aliases for target variables.
  - Transformation:
    $$\tilde{x} = \frac{x - \text{scaler}[\text{"center"}]}{\text{scaler}[\text{"scale"}]}, \qquad x = \tilde{x} \cdot \text{scaler}[\text{"scale"}] + \text{scaler}[\text{"center"}]$$
  - Computed **only** on the training date split (`is_train=True, compute_scaler=True`); never recomputed on validation, test, fine-tuning, or realtime inference data.

### 7.3 Cold-Start Sample & Batch Tensor Schema (`Multimet.__getitem__` & `collate_fn`)

In `Multimet` (`model/datasetzoo/multimet.py`), each sample is anchored at a **forecast issue date $D$** (`00:00 UTC`), with hindcast sequence length $S =$ `seq_length` (e.g., `365` days), forecast horizon $L =$ `lead_time` (e.g., `7` days), shortest lead $k_{\min} =$ `min_lead_time` $= 1$ day, and optional overlap $O =$ `forecast_overlap` (e.g., `365` days or `0`):

```text
Timeline for Sample Issued on Date D (00:00 UTC) with seq_length=S, lead_time=L, min_lead_time=1:

  Hindcast Window (S completed days before D 00:00 UTC):
  ├───────────────────────── S days ─────────────────────────┤
  [ D - S,   D - S + 1,   ...,   D - 2,   D - 1 ]
                                                │
                                                ▼ Forecast Issue Time (D 00:00 UTC)
                                                ├────────────── L days ──────────────┤
                                                [ D (lead=1d), D+1 (lead=2d), ..., D+L-1 (lead=Ld) ]
                                                  Forecast Valid Window (L days)

  Target & Date Sequence (S steps ending at D + L - 1):
  ├─────────────────────────────── S steps ───────────────────────────────┤
  [ D + L - S, ..., D - 1, D, D + 1, ..., D + L - 1 ]
                           ├──── predict_last_n ────┤ (e.g. L+1 = 8 supervised steps: D-1 .. D+L-1)
```

| Batch Dictionary Key | Type | Shape (Collated Batch of Size $B$) | Temporal / Coordinate Alignment |
| :--- | :--- | :--- | :--- |
| **`x_s`** | `torch.Tensor` (`float32`) | `(B, F_static)` | Static catchment attributes stacked in `cfg.static_attributes` order for each sample's basin. |
| **`x_d_hindcast`** | `dict[str, torch.Tensor]` (`float32`) | `{feature_name: (B, S, 1)}` | Completed historical days **`[D - S, ..., D - 1]`**.<br>• **2D nowcast features** (`cpc_*`, `imerg_*`, `era5land_*`): sliced on `date = [D - S .. D - 1]`.<br>• **3D forecast features used in hindcast** (`hres_*`, `graphcast_*`): sliced at **`lead_time = 1 day` (`isel(lead_time=0)`)** on `date = [D - S .. D - 1]` (since `lead_time=1d` issued on $t$ covers $[t\text{ 00:00}, t+1\text{ 00:00})$).<br>• If `timestep_counter: True`, includes `"hindcast_counter"` of shape `(B, S, 1)` filled with `0`. *(Note: In pure hindcast mode without `forecast_inputs`, `x_d_hindcast` is renamed to `x_d` and covers `[D - S + 1 .. D]`.)* |
| **`x_d_forecast`** | `dict[str, torch.Tensor]` (`float32`) | `{feature_name: (B, O + L, 1)}` | • **Forecast horizon (`L` steps):** 3D forecast features sliced at issue date `date = D` across `lead_time = [1 day .. L days]`, valid on **`[D, D + 1, ..., D + L - 1]`**.<br>• **Historical overlap (`O` steps, when `forecast_overlap = O > 0`):** Prepends `lead_time = 1 day` (`isel(lead_time=0)`) from issue dates `[D - O .. D - 1]` before the `L` forecast steps, yielding length `O + L`.<br>• If `timestep_counter: True`, includes `"forecast_counter"` of shape `(B, O + L, 1)` containing `[1, ..., 1]` ($O$ times) followed by `[1, 2, ..., L]`. |
| **`y`** | `torch.Tensor` (`float32`) | `(B, S, F_target)` | Target variables (`streamflow`) on valid dates **`[D + L - S, ..., D + L - 1]`**. Loss and evaluation subset the trailing `predict_last_n` steps (`y[:, -predict_last_n:, :]`). |
| **`date`** | `np.ndarray` (`datetime64[ns]`) | `(B, S)` | Valid dates **`[D + L - S, ..., D + L - 1]`** matching `y`. Note that the last date `date[:, -1]` is $D + L - 1$, from which `BaseTester` recovers the forecast issue date $D = \text{date}[:, -1] - (L - 1)\text{ days}$. |
| **`basin_index`** | `torch.Tensor` (`int64`) | `(B,)` | Integer index of each sample's basin in `dataset._basins`. |
| **`per_basin_target_stds`** | `torch.Tensor` (`float32`) | `(B, 1, F_target)` | Present when `loss: nse`; per-basin standard deviation of target variables for basin-normalized NSE loss. |

### 7.4 Cold-Start vs. Hot-Start Realtime Forecast Data & State Persistence

`model/` supports two operational modes for generating forecasts on issue date $D$ (`model/evaluation/tester.py`, `model/modelzoo/handoff_forecast_lstm.py`, and `model/modelzoo/mean_embedding_forecast_lstm.py`):

#### Mode A: Cold-Start Forecasting (Default)
- **Input Data Required on Issue Date $D$:**
  - Full historical hindcast spinup window `[D - S, ..., D - 1]` ($S =$ `seq_length`, typically `365` days) in `x_d_hindcast`.
  - Any historical forecast overlap `[D - O, ..., D - 1]` ($O =$ `forecast_overlap`) plus the new `00Z` forecast issued on date $D$ (`lead_time = 1 .. L days`) in `x_d_forecast`.
- **Execution Flow:**
  - `hindcast_lstm` starts from a zero state $(h_0, c_0) = (\mathbf{0}, \mathbf{0})$ at $D - S$ and unrolls through $D - 1$.
  - In `HandoffForecastLSTM`, the hindcast state at $D - O - 1$ passes through `handoff_net` / `handoff_linear` to initialize `forecast_lstm`, which then unrolls through the $O$ overlap steps (`D - O .. D - 1`) and the $L$ forecast lead steps (`D .. D + L - 1`).
  - In `MeanEmbeddingForecastLSTM`, `hindcast_lstm` unrolls over `D - S .. D - 1` (padded with `NaN` over the $L$ forecast steps, where `torch.nanmean` ignores missing groups), and its output hidden sequence is concatenated into `forecast_lstm` across the full $S + L$ timeline.

#### Mode B: Hot-Start Forecasting (Warm-Started Recurrent State Persistence)
- **Purpose:** Eliminates redundant 365-day historical spinup computations in realtime operational forecasting by persisting each basin's recurrent LSTM cell and hidden states $(h, c)$ at the end of the historical window ($D - 1$) and reloading them on the next forecast cycle.
- **1. Saving Hot-Start States (`save_state: True`):**
  - When `save_state: True` is set in `Config` during evaluation/inference (`period != "train"`), `BaseTester._evaluate` calls `model.save_state(last_data, path)` after processing each basin.
  - Writes a compressed NumPy archive (`.npz`, `allow_pickle=False`) at:
    ```text
    <run_dir>/hot_start_states/state_<basin>.npz
    ```
  - **`.npz` State Archive Schema (`float32` arrays):**

| `.npz` Key | Accepted Alias | Shape | Description |
| :--- | :--- | :--- | :--- |
| **`h_hindcast`** | `h_hind` | `(1, 1, H_hind)` | Hindcast LSTM hidden state after processing through the last completed historical day ($D - 1$). |
| **`c_hindcast`** | `c_hind` | `(1, 1, H_hind)` | Hindcast LSTM cell state after processing through $D - 1$. |
| **`h_forecast`** | `h_fore` | `(1, 1, H_fore)` | Forecast LSTM hidden state at the handoff boundary ($D - 1$, after state handoff and any historical `forecast_overlap` steps). |
| **`c_forecast`** | `c_fore` | `(1, 1, H_fore)` | Forecast LSTM cell state at the handoff boundary ($D - 1$). |

- **2. Loading & Running Hot-Start Inference (`hot_start_path: <path>`):**
  - **Configuration Invariant:** `hot_start_path` strictly requires **`batch_size: 1`** (enforced in `BaseTester.__init__`) so recurrent states are loaded and stepped per basin without cross-basin batch contamination.
  - **State Resolution:** If `hot_start_path` is a directory, `BaseTester` looks up `<hot_start_path>/state_<basin>.npz` (falling back to `<hot_start_path>/<basin>.npz`); if `hot_start_path` is a file, it loads that `.npz` file directly via `model.load_state_from_disk()`.
  - **Input Tensor Shapes & Execution in `HandoffForecastLSTM`:**
    - **Zero-Lookback Pure Hot Start (`seq_length = 0`, `x_d_hindcast` time length $= 0$):**
      - `x_d_hindcast[feat]` has shape `(1, 0, 1)` (empty historical spinup).
      - `x_d_forecast[feat]` requires only the $L$ forecast lead steps `(1, L, 1)` for issue date $D$ (if `forecast_overlap` steps are present, `forward()` slices `forecast_embeddings[:, -lead_time:, :]`).
      - `forward()` **completely bypasses `hindcast_lstm` and `handoff_net`**, initializes `forecast_lstm` directly with `(h_forecast, c_forecast)` from the `.npz` file, and unrolls `forecast_lstm` over only the $L$ forecast lead days (`D .. D + L - 1`).
    - **Incremental Update Hot Start (`seq_length > 0` with preloaded state):**
      - When new completed hindcast day(s) (e.g., 1 new day since the saved state) are passed in `x_d_hindcast` (`shape: (1, seq_length, 1)`), `hindcast_lstm` initializes from `(h_hindcast, c_hindcast)` instead of zeros, advances through the new hindcast step(s), executes `handoff_net`, and unrolls `forecast_lstm` over the $L$ forecast leads.
  - **Execution in `MeanEmbeddingForecastLSTM`:**
    - Passes `(h_hindcast, c_hindcast)` as `hx` to `hindcast_lstm` and `(h_forecast, c_forecast)` as `hx` to `forecast_lstm`.

### 7.5 Evaluation & Realtime Inference Output Zarr (`test_results.zarr`)

When `inference_mode: True` and `save_results: True`, `BaseTester` writes unscaled predictions and observations to `<run_dir>/<period>/model_epoch<NNN>/<period>_results.zarr` (with consolidated Zarr metadata):

- **Coordinates:**
  - `basin` (`str`): Basin ID (`camels_01013500`, etc.)
  - `date` (`datetime64[ns]`): **Forecast issue date $D$** (`00:00 UTC`)
  - `time_step` (`int64`): Relative lead-time index across the `predict_last_n` output steps. For a forecast model with `lead_time = 7` and `predict_last_n = 8`, `time_step = [0, 1, 2, 3, 4, 5, 6, 7]`, where:
    - `time_step = 0` is the last completed hindcast day ($D - 1$),
    - `time_step = 1` is forecast lead day 1 (valid on $[D\text{ 00:00}, D+1\text{ 00:00})$),
    - `time_step = k` ($k \in \{1,\dots,L\}$) is forecast lead day $k$ (valid on $[D + k - 1, D + k)$).
- **Data Variables (`float32`, unscaled to physical units $\text{mm/day}$):**
  - **Regression Head (`head: regression`):**
    - `streamflow_sim`: dims `("basin", "date", "time_step")`
    - `streamflow_obs`: dims `("basin", "date", "time_step")`
  - **Probabilistic / Mixture Head (`head: cmal` / `umal` / `gmm`):**
    - `streamflow_sim`: dims `("basin", "date", "time_step", "samples")`, where `samples` has length `n_samples` (e.g., `7,500` Monte Carlo draws from the Countable Mixture of Asymmetric Laplacians distribution).
    - `streamflow_obs`: dims `("basin", "date", "time_step")`.
