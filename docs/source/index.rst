.. OpenHydroNet documentation master file

Welcome to OpenHydroNet's documentation!
========================================

**OpenHydroNet** (``google-research/flood-forecasting``) is an open-source Python toolkit for global and local river flood forecasting. It includes deep learning streamflow models (including former operational model generations used in previous releases of Google FloodHub), end-to-end geospatial and meteorological data pipelines, a multi-model flood forecast hub, USGS Bulletin 17C flood frequency analysis, and an interactive web UI.

The source code is available on `GitHub <https://github.com/google-research/flood-forecasting/>`__.

User Guides
-----------

* :doc:`Quick Start <usage/quickstart>` — Install the package, prepare Caravan and MultiMet Zarr datasets, and train, fine-tune, or evaluate a flood forecasting model using ``run`` and ``schedule-runs``.
* :doc:`Modelzoo <usage/models>` — Learn how the ``MeanEmbeddingForecastLSTM`` (former 2025 FloodHub generation) and ``HandoffForecastLSTM`` (former 2024 FloodHub generation) architectures and ``CMAL`` / ``Regression`` output heads work.
* :doc:`Catchment Delineation <usage/catchment_delineation>` — Trace upstream watershed boundary polygons (``EPSG:4326``) from any latitude and longitude using 90 m HydroSHEDS or MERIT-Hydro flow direction grids (``delineate-catchment`` and ``download-merit-d8-tiles``).
* :doc:`Static Attribute Extractor <usage/static_extractor>` — Compute 210 Caravan-compatible static watershed attributes (196 HydroATLAS variables and 14 ERA5-Land climate indices) for custom polygons (``extract-static-attributes`` and ``extract-static-attributes-batch``).
* :doc:`Gridded Weather Archives <usage/gridded_archives>` — Build and update global daily precipitation Zarr archives from NOAA CPC and NASA GPM IMERG (``build-cpc-archive`` and ``build-imerg-archive``).
* :doc:`MultiMet Timeseries Extractor <usage/multimet_extractor>` — Compute area-weighted daily weather time series over watershed polygons from gridded historical and forecast Zarr archives (``extract-multimet``, ``extract-multimet-dask``, and ``multimet-realtime``).
* :doc:`Live Gridded Weather Forecasts <usage/weather_fetcher>` — Synchronize the newest operational weather forecast runs (ECMWF HRES, IFS ENS, and AIFS; NOAA GFS, GEFS, and HRRR) and recent NASA IMERG and NOAA CPC precipitation onto a 0.25° global grid for fast local reading (``sync-weather-forecasts``).
* :doc:`MaaS Flood Forecast Fetcher <usage/maas>` — Fetch and compare live and historical river discharge (``m³/s``) forecasts from Google FloodHub, Copernicus GloFAS, and GEOGLOWS ECMWF (``fetch-maas-forecast``).
* :doc:`Return Periods & Flood Frequency <usage/return_periods>` — Estimate flood return periods (such as 2-, 5-, 20-, and 100-year flood flows in ``m³/s``) using the USGS Bulletin 17C Log-Pearson Type III (``EMA`` and ``MGBT``) algorithm.
* :doc:`Interactive Web UI <usage/frontend>` — Launch the local browser dashboard (``openhydronet-ui``) for multi-model forecast comparison, weather map inspection, and interactive watershed modeling.
* :doc:`Canonical Benchmarks <usage/benchmarks>` — Run the six verification benchmarks (``benchmark-*``) to check numerical accuracy against published Caravan, MultiMet, and USGS reference datasets.
* :doc:`Configuration Arguments <usage/config>` and :doc:`Example Configs <example-configs/example-configs>` — Reference for all YAML configuration settings and pre-configured experiment files.
* :doc:`API Reference <api/modules>` — Detailed class and function reference for all OpenHydroNet subpackages.

You can also visit our `research team's webpage <https://sites.research.google/gr/floodforecasting/>`__.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/quickstart
   usage/models
   usage/catchment_delineation
   usage/static_extractor
   usage/gridded_archives
   usage/multimet_extractor
   usage/weather_fetcher
   usage/maas
   usage/return_periods
   usage/frontend
   usage/benchmarks
   usage/config
   example-configs/example-configs
   api/modules
