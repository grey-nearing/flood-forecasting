===========================================
Interactive Web UI (``openhydronet-ui``)
===========================================

The ``frontend`` package provides a local, browser-based user interface (``openhydronet-ui``) for exploring global weather forecasts, comparing external global flood models, delineating watersheds, extracting static attributes, training and fine-tuning models, and running live flood forecasts with data assimilation.

.. contents:: On this page
   :local:
   :depth: 2

Overview: The 8 Workflow Tabs
-----------------------------

``openhydronet-ui`` starts a local HTTP server with eight interactive tabs in one browser window:

1. **Weather Viewer:** Browse global ``0.25°`` (``EPSG:4326``) operational weather forecasts and precipitation grids synchronized by ``sync-weather-forecasts`` (``ecmwf_hres``, ``ecmwf_ifs``, ``ecmwf_aifs``, ``noaa_gfs``, ``noaa_gefs``, ``noaa_hrrr``, ``nasa_imerg``, and ``noaa_cpc``). Includes animated wind streamlines, timeline playback, and click-anywhere point meteograms.
2. **Models-as-a-Service:** Compare live and historical river discharge (``m³/s``) forecasts from **Google FloodHub**, **Copernicus GloFAS v4**, and **GEOGLOWS ECMWF v2**. River reaches are color-coded by flood return-period severity (``Normal < 2-yr``, ``>= 2-yr Medium``, ``>= 5-yr High``, ``>= 20-yr Severe``), with a **GloFAS Reference** selector (``Control``, ``Median``, ``Mean``) and ensemble spread toggles.
3. **Basin Delineation:** Click any river outlet on the map (using **HydroRIVERS** / HydroSHEDS 90m DEM or **MERIT-Basins** / MERIT-Hydro 90m DEM) or upload a custom GeoJSON file to trace upstream watershed boundaries and calculate drainage area (``km²``).
4. **Geographical Features:** Extract, inspect, search, export (CSV/JSON), and map the ~210 Caravan and HydroATLAS static attributes for your delineated watersheds.
5. **Training:** Extract multi-year historical weather Zarr archives for your basins and configure multi-basin deep learning training runs.
6. **Fine Tuning:** Adapt a pre-trained global model checkpoint (`google-floodhub-settings-110-epochs`, `google-floodhub-settings-no-graphcast-110-epochs`, or `google-floodhub-settings-aifs-110-epochs`) to a local watershed using uploaded streamflow gauge CSVs (in ``m³/s`` or ``mm/day``) and optional local precipitation/temperature CSVs. Includes an interactive epoch scrubber, KPI cards, and a hydrograph/hanging-hyetograph chart.
7. **Forecasting and Data Assimilation:** Run **Cold-Start** (365-day lookback) or **Hot-Start** (``t0 - 1d`` cached state) operational flood forecasts, assimilate live streamflow observations using **Nudging (Persistence)** or **Embedding Variational (4D-Var)** data assimilation, and drag future rainfall/temperature bars on the chart to run interactive **What-If** weather scenarios.
8. **Account:** Create or switch local user workspaces, view disk usage across six storage categories, and rename or cascade-delete basins across all stores from one table.

Starting the Server
-------------------

After installing the package (``pip install -e .``), launch the web server from your terminal:

.. code-block:: bash

   openhydronet-ui --host 127.0.0.1 --port 8080

Then open ``http://127.0.0.1:8080`` in your browser.

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
   * - ``--port``
     - ``8080``
     - TCP port for the HTTP server.
   * - ``--data-dir``
     - ``~/.cache/openhydronet/data``
     - Root directory for local datasets, DEM tiles, and user workspaces.
   * - ``--repo-dir``
     - Repository root
     - Path to the OpenHydroNet repository containing ``model/pretrained-models/``.

Environment Variables & Data Paths
----------------------------------

* **Google FloodHub API key:** Set ``FLOODHUB_API_KEY`` in your environment to fetch operational gauge forecasts from the Google Flood API in the **Models-as-a-Service** tab.
* **Synced weather forecasts:** Set ``OPENHYDRONET_WEATHER_DATA_DIR`` (defaults to ``~/.cache/openhydronet/weather``) to point the **Weather Viewer** tab at the directory managed by :doc:`sync-weather-forecasts <weather_fetcher>`.
* **Local DEM tiles & HydroATLAS:** Set ``OPENHYDRONET_DATA_DIR`` (defaults to ``~/.cache/openhydronet/data``) to customize where on-demand DEM tiles and HydroATLAS tables from ``gs://open-multimet`` are cached.
