===========================================
Interactive Web UI (``openhydronet-ui``)
===========================================

The ``frontend`` package provides a local, browser-based user interface (``openhydronet-ui``) for exploring global weather forecasts, comparing external global flood models, delineating watersheds, extracting static attributes, building historical weather archives, fine-tuning models, and running live flood forecasts with data assimilation.

.. contents:: On this page
   :local:
   :depth: 2

Overview: The 8 Workflow Tabs
-----------------------------

``openhydronet-ui`` starts a local HTTP server with eight interactive tabs in one browser window:

1. **Weather Viewer:** Browse operational weather forecasts and recent precipitation observations synchronized by ``sync-weather-forecasts`` (``ecmwf_hres``, ``ecmwf_ifs``, ``ecmwf_aifs``, ``noaa_gfs``, ``noaa_gefs``, ``noaa_hrrr``, ``nasa_imerg``, and ``noaa_cpc``). All products are shown on one global ``0.25°`` (``EPSG:4326``) grid, for up to 10 days in 3-hour steps. Includes animated wind particles, timeline playback, and click-anywhere point forecasts.
2. **Models-as-a-Service:** Compare live river discharge (``m³/s``) forecasts from Google FloodHub, Copernicus GloFAS v4, and GEOGLOWS ECMWF v2. River reaches are colored by flood return period, using each model's own color scheme (for GloFAS: ``Normal < 2-yr``, ``>= 2-yr Medium``, ``>= 5-yr High``, ``>= 20-yr Severe``). The forecast view has a GloFAS reference selector (``Control``, ``Median``, ``Mean``) and ensemble toggles (``+ Ensemble``, ``+ Range``).
3. **Basin Delineation:** Click any river outlet on the map (using HydroRIVERS / HydroSHEDS 90m DEM or MERIT-Basins / MERIT-Hydro 90m DEM) or upload a custom GeoJSON file to trace upstream watershed boundaries and calculate drainage area (``km²``).
4. **Geographical Features:** Extract, inspect, search, export (CSV/JSON), and map the ~210 Caravan and HydroATLAS static attributes for your delineated watersheds.
5. **Training:** Extract multi-year historical weather time series for your basins into one multi-basin training Zarr archive.
6. **Fine Tuning:** Adapt a pre-trained global model checkpoint (``google-floodhub-settings-110-epochs``, ``google-floodhub-settings-no-graphcast-110-epochs``, or ``google-floodhub-settings-aifs-110-epochs``) to a local watershed. Training uses an uploaded streamflow gauge CSV (in ``m³/s`` or ``mm/day``) together with global forcings extracted for the basin, optional local precipitation/temperature CSVs, or both. Includes an epoch scrubber, KPI cards, an input-forcing chart, and a streamflow hydrograph. You can save any epoch to the model inventory.
7. **Forecasting and Data Assimilation:** Run Cold-Start (365-day lookback) or Hot-Start (``t0 - 1d`` cached state) operational flood forecasts, assimilate live streamflow observations using Nudging (Persistence) or Embedding Variational (4D-Var) data assimilation, and drag future rainfall/temperature values on the chart to run interactive What-If weather scenarios.
8. **Account:** Create or switch local user accounts, see how much data each account holds (six summary cards, including total disk use), and rename or delete basins across all stores from one table. The default ``Guest`` account is emptied every time the server starts.

Starting the Server
-------------------

After installing the package (``pip install -e .``), start the web server from your terminal:

.. code-block:: bash

   openhydronet-ui --host 127.0.0.1 --port 8080

Then open ``http://127.0.0.1:8080`` in your browser. If you leave out ``--port``, the server uses port ``8000``, or the value of the ``PORT`` environment variable if it is set.

Command-Line Options
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Flag
     - Default
     - Description
   * - ``--host``
     - ``127.0.0.1``
     - Network interface to bind. Use ``0.0.0.0`` to accept connections from other machines or a remote forwarding proxy.
   * - ``--port``, ``-p``
     - ``8000`` (or ``$PORT``)
     - TCP port for the HTTP server.

Environment Variables & Data Paths
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Variable
     - Default
     - What it controls
   * - ``OPENHYDRONET_DATA_DIR``
     - ``<repo>/frontend/data``
     - Root folder for the server's local data, including user accounts (``profiles/``).
   * - ``OPENHYDRONET_PROFILES_DIR``
     - ``profiles/`` inside the data folder
     - Folder for user accounts. Each account is stored in ``<folder>/<username>/``.
   * - ``FLOOD_FORECASTING_REPO_DIR``
     - The repository root (the folder that contains ``frontend/``)
     - OpenHydroNet repository used to find the pre-trained models in ``model/pretrained-models/``.
   * - ``OPENHYDRONET_WEATHER_DATA_DIR``
     - ``~/.cache/openhydronet/weather``
     - Weather runs downloaded by :doc:`sync-weather-forecasts <weather_fetcher>`. The **Weather Viewer** tab reads them from here.
   * - ``FLOODHUB_API_KEY``
     - Not set
     - Google Flood Forecasting API key for FloodHub forecasts in the **Models-as-a-Service** tab. If it is not set, the server reads the key from ``~/.cache/openhydronet/floodhub_api_key``. Without a key, FloodHub forecasts are not downloaded from the API.

DEM tiles, river networks, and HydroATLAS tables that the server downloads on demand from ``gs://open-multimet`` are cached in ``~/.cache/openhydronet/data/``. This cache folder is fixed: ``OPENHYDRONET_DATA_DIR`` does not change it.
