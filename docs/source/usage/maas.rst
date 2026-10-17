=================================================
Models-as-a-Service (MaaS) Flood Forecast Fetcher
=================================================

The ``maas`` package downloads live and historical river discharge forecasts from three global flood forecasting systems and aligns them on a single UTC timeline:

1. **Google FloodHub** (:py:class:`maas.floodhub.FloodHubClient`): Gauge-level AI river forecasts, return-period thresholds, and operational warning levels (``NORMAL``, ``WARNING``, ``DANGER``, ``EXTREME``).
2. **Copernicus GloFAS v4** (:py:class:`maas.glofas.GloFASClient`): Daily river discharge on a global ``0.05°`` grid (about ``5 km``), read from the Open-Meteo Flood API or from a GloFAS v4 Zarr archive. Each forecast includes the unperturbed control run, the ensemble median, the ensemble mean, the 25th/75th percentiles, the min/max range, and all 50 ensemble members.
3. **GEOGLOWS ECMWF v2** (:py:class:`maas.geoglows.GeoGLOWSClient`): 3-hourly river forecasts across roughly 6.8 million global river reaches (identified by ``LINKNO`` reach IDs).

All river discharge values are in cubic meters per second (``m³/s``), and all timestamps are in UTC (``YYYY-MM-DD``). You can also explore these forecasts visually in the **MaaS Forecast Hub** tab of :doc:`openhydronet-ui <frontend>`.

.. contents:: On this page
   :local:
   :depth: 2

Do You Need This Package?
-------------------------

* **Yes:** Use ``maas`` (or the ``fetch-maas-forecast`` CLI) when you want **live operational flood forecasts** for a river location, or when you want to compare Google FloodHub, Copernicus GloFAS, and GEOGLOWS side by side.
* **No:** You do **not** need ``maas`` to train or evaluate OpenHydroNet deep learning models on the pre-built ``gs://caravan-multimet/v1.1`` datasets.

Prerequisites
-------------

.. code-block:: bash

   conda activate openhydronet
   pip install -e .

* **GloFAS and GEOGLOWS** are public and require no API keys.
* **Google FloodHub** requires an API key passed via ``--floodhub-api-key``, ``MaaSConfig(floodhub_api_key=...)``, or the ``FLOODHUB_API_KEY`` environment variable. If no key is provided, FloodHub is reported as ``unavailable`` while GloFAS and GEOGLOWS still run normally.
* Two local directories must always be provided explicitly: ``cache_dir`` (for local SQLite caches) and ``river_networks_dir`` (for river network spatial index files).

Quick Start (Python)
--------------------

.. code-block:: python

   from pathlib import Path
   from maas import (
       MaaSConfig,
       MaaSDataFetcher,
       fetch_forecasts,
       fetch_historical,
       fetch_return_periods,
       resolve_reaches,
   )

   config = MaaSConfig(
       cache_dir=Path('/tmp/maas_cache'),
       river_networks_dir=Path('/tmp/maas_cache'),
       floodhub_api_key='',
   )

   # 1. Match a latitude and longitude (EPSG:4326) to the nearest river reach
   reaches = resolve_reaches(38.6270, -90.1994, config=config)

   # 2. Fetch forecasts from all three systems on one daily UTC timeline
   bundle = fetch_forecasts(
       config,
       38.6270,
       -90.1994,
       models=['floodhub', 'glofas', 'geoglows'],
       glofas_reference='control',  # 'control' | 'median' | 'mean'
   )
   print(bundle['peaks'])             # Peak flow (m³/s) and peak date per model
   print(bundle['glofas_reference'])  # Which GloFAS summary series was selected

   # 3. Fetch historical flow and return-period thresholds (2-, 5-, 20-, 100-year)
   river_id = reaches['geoglows']['river_id']
   historical = fetch_historical(config, 'geoglows', reach_id=river_id)
   thresholds = fetch_return_periods(config, 'geoglows', reach_id=river_id)

   # 4. Or use the object-oriented MaaSDataFetcher directly
   fetcher = MaaSDataFetcher(config)
   full = fetcher.fetch_forecasts(38.6270, -90.1994)

Choosing the GloFAS Reference Series
------------------------------------

GloFAS publishes three summary curves for each grid cell and day. The ``glofas_reference`` argument (default ``'control'``) selects which curve is used for the primary GloFAS daily series, its peak discharge (`m³/s`), and its return-period threshold comparison:

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Key
     - Record Field
     - Meaning
   * - ``control``
     - ``discharge_deterministic``
     - Control (unperturbed) forecast run forced by the single ECMWF control weather forecast.
   * - ``median``
     - ``discharge_median``
     - Median (`50th percentile`) across the 50 perturbed ensemble members on each day.
   * - ``mean``
     - ``discharge_mean``
     - Arithmetic mean across the 50 perturbed ensemble members on each day.

Every GloFAS record always includes all three fields (`discharge_deterministic`, `discharge_median`, and `discharge_mean`), so you can switch between them without re-downloading. Passing an unrecognized key raises ``ValueError``.

In the web UI (:doc:`openhydronet-ui <frontend>`), the **GloFAS Reference** drop-down in the Forecast View updates both the plotted curve and the multi-model consensus summary:

.. code-block:: text

   GET /api/maas/forecast?lat=38.627&lon=-90.199&glofas_reference=median

Reading Cloud or Local GloFAS v4 Zarr Archives
----------------------------------------------

Instead of querying the Open-Meteo HTTP API, ``fetch_historical(..., zarr_path=...)`` and :py:func:`maas.extract_glofas_zarr_series` can read a GloFAS v4 Zarr or NetCDF archive directly:

* **Canonical Copernicus GloFAS v4 historical reanalysis Zarr store on GCS:** ``gs://open-multimet/glofas_v4/river_discharge_v4_0/`` (daily ``dis24`` in ``m³/s`` on the ``0.05°`` global grid).
* **Local Zarr or NetCDF forecast/reanalysis archives:** Any local Zarr or NetCDF store with ``time``, ``latitude``, and ``longitude`` coordinates and a discharge variable (default ``dis24``). If the archive also has a forecast lead-time dimension (``step``) or an ensemble member dimension (``number``), pass ``step_index`` and ``member_index`` explicitly:

.. code-block:: python

   from pathlib import Path
   from maas import extract_glofas_zarr_series

   records = extract_glofas_zarr_series(
       Path('glofas_forecast.zarr'),
       38.6270,
       -90.1994,
       step_index=0,     # required when the archive has a `step` dimension
       member_index=0,   # required when the archive has a `number` dimension
   )

Missing timesteps are returned as ``NaN`` without interpolation.

Command-Line Usage (``fetch-maas-forecast``)
--------------------------------------------

Installing OpenHydroNet registers the ``fetch-maas-forecast`` command (also runnable via ``python -m maas.cli``):

.. code-block:: bash

   fetch-maas-forecast \
     --lat 38.6270 \
     --lon -90.1994 \
     --models glofas,geoglows \
     --cache-dir ~/.cache/flood_forecasting/maas \
     --river-networks-dir data/river_networks \
     --output forecast_bundle.json

Command-Line Flags
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Flag
     - Default
     - Description
   * - ``--lat``
     - *(required)*
     - Target latitude in decimal degrees (``EPSG:4326``, ``-90`` to ``90``).
   * - ``--lon``
     - *(required)*
     - Target longitude in decimal degrees (``EPSG:4326``, ``-180`` to ``180``).
   * - ``--cache-dir``
     - *(required)*
     - Local directory for SQLite forecast and historical caches.
   * - ``--river-networks-dir``
     - ``--cache-dir``
     - Local directory containing river network spatial index files (defaults to ``--cache-dir`` when omitted).
   * - ``--models``
     - ``floodhub,glofas,geoglows``
     - Comma-separated subset of providers to query.
   * - ``--gauge-id``
     - ``None``
     - Explicit Google FloodHub gauge ID (skips spatial gauge search when set).
   * - ``--river-id``
     - ``None``
     - Explicit GEOGLOWS 9-digit ``LINKNO`` reach ID (skips spatial reach snapping when set).
   * - ``--reach-id``
     - ``None``
     - Explicit HydroRIVERS ``HYRIV_ID`` reach ID.
   * - ``--floodhub-api-key``
     - ``$FLOODHUB_API_KEY``
     - Google Flood API key.
   * - ``--timeout``
     - ``12.0``
     - HTTP request timeout in seconds.
   * - ``--output``
     - stdout
     - Optional file path to write the JSON forecast bundle.

What to Watch Out For
---------------------

* **Daily mean vs. 3-hourly instantaneous flow:** FloodHub and GloFAS (`dis24`) report daily mean river discharge at ``00:00 UTC``, whereas GEOGLOWS reports 3-hourly instantaneous discharge.
* **No silent interpolation:** Missing values remain ``None`` in JSON output and ``NaN`` in ``pandas`` DataFrames.
* **Return period calculation:** When return periods are computed from a provider's historical reanalysis series, ``maas`` uses the USGS Bulletin 17C Expected Moments Algorithm (`EMA`) from :doc:`return_periods`.

Testing
-------

.. code-block:: bash

   # Offline unit tests (no network required)
   pytest maas/tests frontend/tests/test_maas_viewer.py -m "not canary"

   # Opt-in live canary against Open-Meteo GloFAS and GEOGLOWS
   MAAS_LIVE_CANARY=1 pytest maas/tests/test_canary_live.py -m canary -v

API reference: :doc:`../api/maas`.
