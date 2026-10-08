=========================
Gridded Weather Forecasts
=========================

This guide explains how to use :mod:`multimet.weather_fetcher` to download the
newest 10-day weather forecasts from public sources, store them as compact
binary grids on disk, and read them back as maps, wind fields, point time
series, and catchment (river basin) averages.

.. note::

   **Do you need these tools?**
   If you only want to train or evaluate flood-forecasting models with the
   published MultiMet dataset, **you do not need this package**. Point
   ``dynamics_data_dir`` in your training configuration to
   ``gs://caravan-multimet/v1.1`` (see :doc:`quickstart`).

   Use ``multimet.weather_fetcher`` when you need live gridded forecasts from
   operational weather models.

--------
Overview
--------

Installing this repository provides one command-line tool and one Python
package:

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - Component
     - Entry point
     - Purpose
   * - CLI synchronizer
     - ``sync-weather-forecasts``
     - Downloads the newest runs of the selected models into ``<data-dir>``
       and updates the ``current`` run.
   * - Python sync API
     - :class:`~multimet.weather_fetcher.sync.WeatherSynchronizer`,
       :func:`~multimet.weather_fetcher.sync.sync_all_models`,
       :func:`~multimet.weather_fetcher.sync.sync_model`
     - Same as the CLI, from Python.
   * - Python data fetcher
     - :class:`~multimet.weather_fetcher.fetcher.WeatherDataFetcher`
     - Reads a synced ``<data-dir>`` and returns forecast grids, wind vectors,
       point meteograms, and catchment summaries.

Supported Weather Models
^^^^^^^^^^^^^^^^^^^^^^^^

All models are stored on the same global ``0.25°`` grid (``721 × 1440`` cells,
latitude ``+90`` to ``-90``, longitude ``-180`` to ``+179.75``). Each model
keeps its own forecast lead hours; the fetcher never interpolates between them.

.. list-table::
   :header-rows: 1
   :widths: 13 24 16 17 30

   * - Model key
     - Model
     - Source
     - Native grid
     - Stored lead hours
   * - ``ecmwf_hres``
     - ECMWF IFS HRES (deterministic)
     - ``gs://ecmwf-open-data``
     - ``0.25°`` global
     - every 3 h to 144 h, then every 6 h to 240 h
   * - ``ecmwf_ifs``
     - ECMWF IFS ENS (control member)
     - dynamical.org
     - ``0.25°`` global
     - every 3 h to 144 h, then every 6 h to 240 h
   * - ``ecmwf_aifs``
     - ECMWF AIFS (AI model)
     - dynamical.org
     - ``0.25°`` global
     - every 6 h to 240 h
   * - ``noaa_gfs``
     - NOAA GFS
     - dynamical.org
     - ``0.25°`` global
     - every 3 h to 240 h
   * - ``noaa_gefs``
     - NOAA GEFS (control member)
     - dynamical.org
     - ``0.25°`` global
     - every 3 h to 240 h
   * - ``noaa_hrrr``
     - NOAA HRRR (CONUS only)
     - dynamical.org
     - ``3 km`` CONUS, averaged onto ``0.25°``
     - every 3 h to 48 h
   * - ``nasa_imerg``
     - NASA GPM IMERG Early (satellite analysis)
     - dynamical.org
     - ``0.10°`` global, averaged onto ``0.25°``
     - last 10 days, every 3 h
   * - ``noaa_cpc``
     - NOAA CPC Unified (gauge analysis)
     - NOAA PSL
     - ``0.50°`` land only, mapped onto ``0.25°``
     - last 10 days, every 24 h

The forecast models provide precipitation, temperature, pressure, and wind;
the two analyses (``nasa_imerg``, ``noaa_cpc``) provide precipitation only.
Cells with no data (for example outside the HRRR domain, or over the ocean for
CPC) are stored as ``NaN`` and returned as ``NaN`` (``null`` in JSON). They are
never replaced by zeros or by neighbouring cells.

Supported Weather Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - Variable key
     - Description
     - Units
   * - ``precipitation``
     - Mean rain rate over the model interval that ends at the requested time
     - ``mm/h``
   * - ``accumulated_precip``
     - Rain accumulated since the forecast start
     - ``mm``
   * - ``temperature``
     - Air temperature 2 m above ground
     - ``°C``
   * - ``wind``
     - Wind speed and direction 10 m above ground (``u``/``v`` components)
     - ``m/s``
   * - ``pressure``
     - Air pressure reduced to mean sea level
     - ``hPa``

-------------
Prerequisites
-------------

Activate the ``openhydronet`` Conda environment and install the package with
the ``weather`` extra, which adds the download dependencies (``pystac`` and
``icechunk`` for dynamical.org, ``eccodes`` for ECMWF Open Data GRIB2 files):

.. code-block:: bash

   conda activate openhydronet
   pip install -e ".[weather]"

Downloading needs network access; no API keys are required. Reading synced
data uses ``numpy``, ``shapely``, ``geopandas``, ``pyproj``, and ``scipy``,
which the ``openhydronet`` environment already provides.

-----------
Quick Start
-----------

Command line (``sync-weather-forecasts``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--data-dir`` is always required. There is no default location.

.. code-block:: bash

   # Download the newest runs of all eight models
   sync-weather-forecasts --data-dir /tmp/weather_cache

   # Download only NOAA GFS and ECMWF AIFS
   sync-weather-forecasts --data-dir /tmp/weather_cache --models noaa_gfs,ecmwf_aifs

   # Download again even if the newest run is already on disk
   sync-weather-forecasts --data-dir /tmp/weather_cache --force

   # Print the last synchronization status as JSON and exit
   sync-weather-forecasts --data-dir /tmp/weather_cache --status

The command exits with code ``0`` when the data directory is consistent
afterwards (``updated``, ``up_to_date``, or ``busy`` because another process
is already synchronizing) and ``1`` when a selected model failed (``partial``
or ``error``). An unknown model key raises ``ValueError``.

Command-Line Arguments
""""""""""""""""""""""

-  ``--data-dir`` / ``--data-root`` *(required)*: Folder for downloaded runs,
   ``current``, ``sync_status.json``, and ``cpc_cache``.
-  ``--models``: Comma-separated model keys to download (default: all eight
   models).
-  ``--cpc-cache-dir``: Folder caching the NOAA PSL CPC annual NetCDF files
   (default: ``<data-dir>/cpc_cache``).
-  ``--force``: Download again even if the newest run is already on disk.
-  ``--status``: Print ``sync_status.json`` and exit without downloading.

Python API
^^^^^^^^^^

.. code-block:: python

   from pathlib import Path

   from multimet.weather_fetcher import WeatherDataFetcher, WeatherSynchronizer

   data_dir = Path("/tmp/weather_cache")

   # Download runs (same work as the CLI).
   synchronizer = WeatherSynchronizer(
       data_dir=data_dir, models=["ecmwf_aifs", "noaa_gfs"]
   )
   status = synchronizer.sync_all()
   print(status["last_result"])  # "updated", "up_to_date", "partial", "error" or "busy"

   # Read the synced runs. `step_idx` counts 3-hour steps: step 2 = +6 h.
   with WeatherDataFetcher(data_dir=data_dir) as fetcher:
     info = fetcher.get_model_info("ecmwf_aifs")
     print(info["init_time"], info["stored_lead_hours"], info["real_variables"])

     # 2D grid (721 x 1440) in physical units, or None when +6 h is not stored.
     rain = fetcher.fetch_forecast_grid("ecmwf_aifs", "precipitation", step_idx=2)

     # 10 m wind components on a 2-degree grid (subsample=2), optional viewport.
     wind = fetcher.fetch_wind_grid("ecmwf_aifs", step_idx=2, subsample=2)
     viewport = fetcher.fetch_wind_grid(
         "noaa_gfs", step_idx=2, subsample=1, bbox=(-90.0, 35.0, -80.0, 45.0)
     )

     # 10-day meteogram at one point for several models.
     probe = fetcher.fetch_point_timeseries(
         lat=40.42, lon=-86.92, models=["ecmwf_aifs", "noaa_gfs"]
     )
     print(probe["lead_hours"][:3], probe["models"]["noaa_gfs"]["temp_c"][:3])

     # Area-weighted catchment statistics for a GeoJSON Feature.
     basin = {
         "id": "wabash",
         "properties": {"area_km2": 3200.0},
         "geometry": {
             "type": "Polygon",
             "coordinates": [[
                 [-87.0, 40.0], [-86.0, 40.0], [-86.0, 41.0], [-87.0, 41.0],
                 [-87.0, 40.0],
             ]],
         },
     }
     summary = fetcher.fetch_catchment_summary(
         basin, step_idx=2, model_key="noaa_gfs"
     )
     print(summary["basin_mean_precip_mmh"], summary["basin_accumulated_10d_mm"])

     # Pick up a newer run written by the synchronizer (safe while other
     # threads are still reading the previous run).
     fetcher.reload_if_changed()

------------------------
What the Fetcher Returns
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Method
     - Returns
     - Missing data
   * - ``to_xarray(model_key, variables=None)``
     - ``xarray.Dataset`` containing all stored leads on a ``(lead_time, latitude, longitude)`` physical grid, plus ``valid_time``.
     - masked cells are ``NaN``. Coordinates and data arrays have standard CF ``units`` and ``long_name`` attrs.
   * - ``fetch_forecast_grid(model_key, var_key, step_idx=None, lead_hours=None, lats=None, lons=None, bilinear=False)``
     - ``float32`` array, shape ``(721, 1440)`` or ``(len(lats), len(lons))``
     - ``None`` when the lead is beyond the run, or when temperature/pressure
       is not stored at exactly that lead; masked cells are ``NaN``
   * - ``fetch_wind_grid(model_key, step_idx=None, lead_hours=None, resolution_deg=None, subsample=2, bbox=None, bilinear=False)``
     - ``{"header": {...}, "u": [...], "v": [...]}`` with ``nx * ny`` values
       on a ``1° × subsample`` grid (``subsample`` 1 to 4)
     - masked cells are ``None``; ``header["missing_count"]`` counts them;
       ``ValueError`` when wind is not stored at that lead
   * - ``fetch_point_timeseries(lat, lon, models=None, strict=True)``
     - ``lead_hours`` (0 to 240 h, step 3 h) and, per model,
       ``precip_rate_mmh``, ``accum_precip_mm``, ``temp_c``,
       ``wind_speed_mps``, ``wind_direction_deg``, ``pressure_hpa``,
       ``stored_lead_hours``, ``init_time``, ``max_lead_hours``
     - ``None`` entries at leads the model did not store, beyond its horizon,
       or over masked cells
   * - ``fetch_catchment_summary(geojson_feature, model_key, step_idx=None, lead_hours=None)``
     - ``catchment_id``, ``area_km2``, ``area_km2_source``,
       ``basin_mean_precip_mmh``, ``basin_max_precip_mmh``,
       ``basin_accumulated_10d_mm``, ``basin_mean_temp_c``,
       ``missing_area_fraction``, ``grid_cells``, ``centroid``,
       ``valid_time_utc``, ``accumulation_hours``
     - a statistic is ``None`` when less than 80% of the basin area has data
       at that lead
   * - ``get_model_info(model_key)`` / ``get_all_models_info()``
     - ``data_source`` (``"archived_run"`` or ``"unavailable"``),
       ``init_time``, ``stored_lead_hours``, ``max_lead_hours``,
       ``real_variables``, ``missing_variables``
     - empty lists / ``None`` when the model is not synced
   * - ``get_sync_status()``
     - contents of ``sync_status.json`` plus ``sync_status_found`` and the
       loaded ``data_dir``
     - ``sync_status_found`` is ``False`` before the first sync

Catchment statistics weight every grid cell by its intersection area with the
polygon (holes excluded) times ``cos(latitude)``. The feature needs an ``id``
or ``properties.catchment_id`` and a ``Polygon`` or ``MultiPolygon`` geometry
in longitude/latitude degrees. ``properties.area_km2`` is reported when
present; otherwise the WGS84 geodesic area of the polygon is computed
(``area_km2_source`` tells you which).

----------------
Directory Layout
----------------

.. code-block:: text

   <data-dir>/
     current -> runs/<run>/            symlink to the active run (swapped atomically)
     runs/<run>/<model>_<stream>.bin   float16 planes, shape (n_leads, 721, 1440)
     runs/<run>/latest_dynamical_meta.json   init time, lead hours and units per model
     sync_status.json                  result and time of the last synchronization
     cpc_cache/                        cached NOAA PSL CPC annual NetCDF files

Streams are ``precip`` (mm/h), ``temp`` (°C), ``mslp`` (hPa minus the offset
stored in the metadata, 1000 hPa), ``u10`` and ``v10`` (m/s).

---------------------
What to Watch Out For
---------------------

* **``step_idx`` and ``lead_hours``.** ``step_idx`` counts 3-hour steps (e.g., ``step_idx=2`` means +6 h). Or use physical ``lead_hours`` directly. The maximum is
  80 (+240 h). Steps beyond a model's horizon return ``None``.
* **No lead substitution.** For a 6-hourly model (``ecmwf_aifs``) temperature,
  pressure, and wind are ``None`` at +3 h, +9 h, and so on. Rain rate at +3 h
  is the mean rate of the model interval that contains +3 h (0 to 6 h), and
  the accumulation grows only at stored leads.
* **Errors are explicit.** Requesting a model or variable that is not synced
  raises ``FileNotFoundError``; invalid coordinates (latitude outside
  ``[-90, 90]``, longitude outside ``[-180, 360]``), invalid geometries,
  unknown model keys, or bad ``step_idx``/``subsample`` values raise
  ``ValueError``; a catchment feature without an id raises ``KeyError``.
* **Antimeridian.** Wind ``bbox`` values may cross the antimeridian
  (``min_lon > max_lon``); the returned header ``lo1`` continues past 180.
  Catchment polygons spanning more than 180° of longitude are rejected: split
  them at ±180 first.
* **Corrupt or inconsistent runs are refused.** A ``.bin`` file whose size is
  not a whole number of planes, or that does not match the lead hours in
  ``latest_dynamical_meta.json``, raises ``ValueError`` when the fetcher opens
  the run.
* **Hot reload is safe.** ``reload_if_changed()`` swaps in the newer run;
  requests that are still reading the old run keep their arrays until they
  finish. Call ``close()`` (or use ``with WeatherDataFetcher(...)``) when you
  are done so the memory maps are released.
* **Upstream publication delays.** Weather centres publish lead times
  progressively. The synchronizer keeps the previous complete run active until
  the new run is complete for every selected model.

-------------
Running Tests
-------------

.. code-block:: bash

   pytest multimet/tests/test_weather_fetcher.py multimet/tests/test_weather_sync.py -v

   # Also run the live check against the dynamical.org catalog (needs network):
   pytest multimet/tests/test_weather_fetcher.py --run-canary -m canary

See :doc:`../api/weather_fetcher` for the full API reference.
