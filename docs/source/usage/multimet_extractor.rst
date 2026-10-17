==========================
MultiMet Forcing Extractor
==========================

The **MultiMet Forcing Extractor** (``multimet.timeseries_extractors``) calculates daily area-weighted weather time series over watershed boundary polygons (``EPSG:4326``). It reads either **gridded Zarr weather archives** (local or on Google Cloud Storage ``gs://``) or **live upstream agency feeds** and writes standardized ``<PRODUCT>/timeseries.zarr`` stores ready for OpenHydroNet model training and inference (`Kratzert et al., 2023 <https://www.nature.com/articles/s41597-023-01975-w>`__).

.. contents:: On this page
   :local:
   :depth: 2

Supported Weather Products
--------------------------

The extractor supports **4 core historical/operational products** plus **6 dynamical.org Icechunk products**. For every product, the extractor also writes a companion ``<prefix>_missing_fraction`` variable that records the area-weighted fraction (from ``0.0`` to ``1.0``) of the watershed that had missing (``NaN``) grid cells on each day:

1. **ERA5-Land (ECMWF)** (``ERA5_LAND``):
   Daily global reanalysis at ``0.1°`` resolution (``1801 × 3600`` grid cells). Extracted from an ERA5-Land gridded Zarr archive (``--source archive``, canonical URI ``gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr``). Produces **39 daily variables** (12 state and radiation variables with daily mean, ``_min``, and ``_max``, plus ``era5land_total_precipitation`` in ``mm/day``, ``era5land_potential_evaporation_FAO_PENMAN_MONTEITH`` in ``mm/day``, and ``era5land_potential_evaporation_DEPRECATED``), plus ``era5land_missing_fraction``.
2. **CPC Global Unified Precipitation (NOAA PSL)** (``CPC``):
   Daily rain-gauge precipitation analysis at ``0.5°`` resolution (``360 × 720`` grid cells). Produces ``cpc_precipitation`` (``mm/day``), ``cpc_num_stations`` (number of reporting rain gauges per grid cell), and ``cpc_missing_fraction``. Reads from a gridded Zarr archive (``--source archive``, canonical URI ``gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr``) or directly from NOAA PSL NetCDF / binary files (``--source public`` or ``--source local``).
3. **IMERG Early V07 (NASA GPM)** (``IMERG``):
   Global satellite precipitation at ``0.1°`` resolution (``1800 × 3600`` grid cells, ``60°N`` to ``60°S``). Produces ``imerg_precipitation`` (``mm/day``) and ``imerg_missing_fraction``. Reads from a gridded Zarr archive (``--source archive``, canonical URI ``gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr``) or NASA GES DISC (``--source public``).
4. **ECMWF IFS HRES** (``HRES``):
   High-resolution 10-day numerical weather forecasts at ``0.25°`` resolution (``721 × 1440`` grid cells). Produces daily forecast precipitation (``mm/day``), 2 m temperature, surface pressure, and solar/thermal radiation fluxes across lead days ``1..10``, plus ``hres_missing_fraction``. Reads from a gridded Zarr archive (``--source archive``, canonical URI ``gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr``) or ECMWF Open Data on GCS (``gs://ecmwf-open-data``).
5. **dynamical.org Deterministic & Ensemble Forecasts** (``GFS``, ``GEFS``, ``IFS_ENS``, ``AIFS``, ``AIFS_ENS``):
   10-day global operational forecasts at ``0.25°`` resolution (``721 × 1440`` grid cells) streamed directly from ``dynamical.org`` Icechunk stores. Deterministic products extract 8 ECMWF surface variables (``AIFS``) or 9 NOAA surface variables including daily min/max 2 m temperature (``GFS``). Ensemble products (``IFS_ENS``, ``AIFS_ENS``, ``GEFS``) compute area-weighted watershed averages for each ensemble member and save 7 summary statistics per variable (``_mean``, ``_std`` with ``ddof=1``, ``_min``, ``_max``, ``_p10``, ``_p50``, ``_p90``), plus optional 4D per-member arrays when ``--include-ensemble-members`` is enabled.
6. **dynamical.org NASA IMERG Early** (``DYNAMICAL_IMERG``):
   Half-hourly ``0.1°`` satellite precipitation from ``nasa-imerg-analysis-early`` aggregated into daily ``dynamical_imerg_precipitation`` (``mm/day``) and ``dynamical_imerg_missing_fraction``.

How Area-Weighted Extraction Works
----------------------------------

1. **Exact Spherical Intersection Weights:** For each watershed polygon, the extractor intersects the polygon with the weather product's latitude/longitude grid and weights each overlapping grid cell by its spherical surface area (:math:`\cos(\text{lat})`). Small watersheds that sit inside a single grid cell receive full weight (``1.0``) on that cell.
2. **Missing Data Transparency:** Missing grid cells or missing dates are never filled with zeros or interpolated. Instead, missing values remain ``NaN`` and the fraction of missing watershed area is recorded in ``<prefix>_missing_fraction`` (from ``0.0`` when 100% of the watershed has valid data to ``1.0`` when the entire watershed is missing).

Quick Start: Historical & Archive Extraction
--------------------------------------------

1. Command-Line: Serial Extraction (``extract-multimet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``extract-multimet`` to extract one or more products over a date range (``YYYY-MM-DD``):

.. code-block:: bash

   extract-multimet \
     --basins_path /path/to/basins.geojson \
     --output_dir /tmp/multimet_extracted \
     --products CPC,ERA5_LAND,IMERG,HRES \
     --start_date 2022-01-01 \
     --end_date 2022-01-05 \
     --source archive \
     --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
     --archive-store ERA5_LAND=gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr \
     --archive-store IMERG=gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
     --archive-store HRES=gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr

2. Command-Line: Parallel Dask Extraction (``extract-multimet-dask``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-year historical backfills across many watersheds, use ``extract-multimet-dask`` to distribute daily extraction tasks across local CPU workers or a remote Dask cluster, with automatic chunk-level resume:

.. code-block:: bash

   extract-multimet-dask \
     --basins_path /path/to/basins.geojson \
     --output-dir /tmp/multimet_extracted \
     --products CPC,ERA5_LAND,IMERG,HRES \
     --start-date 2016-01-01 \
     --end-date 2023-12-31 \
     --num-workers 16 \
     --source archive \
     --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
     --archive-store ERA5_LAND=gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr \
     --archive-store IMERG=gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr \
     --archive-store HRES=gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr

3. Python API (``extract_multimet_serial`` & ``extract_multimet_dask``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from multimet.timeseries_extractors import (
       extract_multimet_dask,
       extract_multimet_serial,
   )

   stores = extract_multimet_serial(
       basins="/path/to/basins.geojson",
       output_dir="/tmp/multimet_extracted",
       products=["CPC", "ERA5_LAND", "IMERG", "HRES"],
       start_date="2022-01-01",
       end_date="2022-01-05",
       source="archive",
       archive_stores={
           "CPC": "gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr",
           "ERA5_LAND": "gs://open-multimet/gridded-data-archives/ERA5_LAND/daily_surface.zarr",
           "IMERG": "gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr",
           "HRES": "gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr",
       },
   )

Real-Time Operational Forcing Fetcher (``multimet-realtime``)
-------------------------------------------------------------

``multimet-realtime`` (and the Python function :py:func:`multimet.timeseries_extractors.fetch_realtime_multimet`) downloads and updates operational weather time series directly from live upstream feeds (``HRES`` from ``gs://ecmwf-open-data``, ``IMERG`` from NASA GES DISC, ``CPC`` from NOAA PSL, and ``GFS``, ``GEFS``, ``IFS_ENS``, ``AIFS``, ``AIFS_ENS``, ``DYNAMICAL_IMERG`` from ``dynamical.org``).

It supports two operational modes:

* **Cold-Start (``--mode coldstart``):** Downloads a 365-day historical spin-up window (``[t0 - 365 days, t0]``) plus the 10-day weather forecast issued on ``t0``. By default, historical spin-up dates before the forecast issue window download only the first lead day (``lead_time = 1 day`` / ``24h``), reducing HRES download volume by 10x.
* **Hot-Start (``--mode hotstart``):** Checks your existing ``<PRODUCT>/timeseries.zarr`` stores (or a saved OpenHydroNet ``.npz`` state file) to find the last valid date, automatically re-fetches any trailing ``NaN`` dates caused by upstream publication delays, and appends new days up to ``t0``.

.. code-block:: bash

   # Cold-Start: 365-day spin-up + 10-day forecast
   multimet-realtime \
     --basins_path /path/to/basins.geojson \
     --output_dir /tmp/realtime_forcing \
     --mode coldstart \
     --reference_date latest

   # Hot-Start: incremental catch-up + trailing NaN healing
   multimet-realtime \
     --basins_path /path/to/basins.geojson \
     --output_dir /tmp/realtime_forcing \
     --mode hotstart \
     --reference_date latest

.. code-block:: python

   from multimet.timeseries_extractors import fetch_realtime_multimet

   result = fetch_realtime_multimet(
       basins="/path/to/basins.geojson",
       output_dir="/tmp/realtime_forcing",
       mode="hotstart",
       reference_date="latest",
   )

Command-Line Reference
----------------------

``extract-multimet`` & ``extract-multimet-dask`` Flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``extract-multimet`` uses the underscore spelling for most multi-word flags (for example ``--output_dir`` and ``--start_date``). ``extract-multimet-dask`` accepts both spellings (for example ``--output_dir`` or ``--output-dir``).

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Flag
     - Default
     - Description
   * - ``--basins_path``
     - *(required)*
     - One or more paths to watershed boundary files (``.geojson``, ``.shp``, ``.gpkg``, ``.parquet``).
   * - ``--output_dir``, ``--output-dir``
     - *(required)*
     - Destination directory or ``gs://`` URI for ``<PRODUCT>/timeseries.zarr`` stores.
   * - ``--products``
     - ``CPC,ERA5_LAND,IMERG,HRES``
     - Comma-separated list of products to extract.
   * - ``--start_date``, ``--start-date``
     - *(required)*
     - Inclusive start date (``YYYY-MM-DD``).
   * - ``--end_date``, ``--end-date``
     - *(required)*
     - Inclusive end date (``YYYY-MM-DD``).
   * - ``--source``
     - ``public``
     - Data source mode: ``archive`` (read from ``--archive-store`` Zarr archives), ``public``/``upstream``, or ``local``.
   * - ``--archive-store``
     - ``None``
     - Repeatable ``PRODUCT=URI`` mapping for gridded Zarr archives (required when ``--source archive``).
   * - ``--id_column``, ``--id-column``
     - ``None``
     - Column name in ``--basins_path`` containing the watershed ID (auto-detected if omitted).
   * - ``--weights_cache``, ``--weights-cache``
     - ``None``
     - Optional path to save or reuse a precomputed ``.npz`` zonal weight matrix.
   * - ``--overwrite``
     - ``False``
     - Delete and rebuild existing output Zarr stores.
   * - ``--append``
     - ``False``
     - Append new dates or new basins to an existing Zarr store.
   * - ``--include-ensemble-members``
     - ``False``
     - Save 4D per-member arrays alongside 3D summary statistics for ``GEFS``, ``IFS_ENS``, and ``AIFS_ENS``.
   * - ``--num-workers`` *(Dask only)*
     - CPU count - 1
     - Number of parallel Dask worker processes.
   * - ``--dask-scheduler`` *(Dask only)*
     - ``None``
     - Optional remote Dask scheduler address (``tcp://host:port``).
   * - ``--batch-days`` *(Dask only)*
     - ``1`` (``30`` for ``DYNAMICAL_IMERG``)
     - Number of consecutive days processed per Dask worker task.
   * - ``--no-resume`` *(Dask only)*
     - ``False``
     - Re-extract all dates instead of skipping already completed Zarr chunks.

``multimet-realtime`` Flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Flag
     - Default
     - Description
   * - ``--basins_path``
     - *(required)*
     - One or more paths to watershed boundary files (``.geojson``, ``.shp``).
   * - ``--output_dir``
     - *(required)*
     - Destination directory for ``<PRODUCT>/timeseries.zarr`` stores.
   * - ``--mode``
     - ``hotstart``
     - ``coldstart`` (365-day spin-up + 10-day forecast) or ``hotstart`` (incremental update + trailing ``NaN`` healing).
   * - ``--reference_date``
     - ``latest``
     - Forecast issue date (``YYYY-MM-DD``) or ``latest`` to auto-detect the newest published ``00z`` run.
   * - ``--lookback_days``
     - ``365`` / ``7``
     - Override the spin-up or catch-up lookback window in days before ``--reference_date``.
   * - ``--products``
     - ``HRES,IMERG,CPC``
     - Comma-separated list of real-time products (supports ``HRES``, ``IMERG``, ``CPC``, ``AIFS``, ``AIFS_ENS``, ``GFS``, ``GEFS``, ``IFS_ENS``, ``DYNAMICAL_IMERG``).
   * - ``--hot_start_state``
     - ``None``
     - Optional path to an OpenHydroNet ``.npz`` hot-start state file or directory.
   * - ``--full_hindcast_leads``
     - ``False``
     - Fetch all 10 forecast lead days for every historical spin-up date instead of only ``lead_time=1D``.
   * - ``--full_forecast_days``
     - ``2``
     - Number of trailing issue dates up to ``--reference_date`` for which all 10 forecast lead days are fetched.
   * - ``--include_ensemble_members``
     - ``False``
     - Also write 4D per-member arrays for ensemble products.
