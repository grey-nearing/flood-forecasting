Models-as-a-Service (MaaS) Flood Forecast Fetcher
=================================================

The ``maas`` package downloads live river discharge forecasts from three
global flood forecasting systems and puts them on one common timeline:

1. **Google FloodHub** (``maas.floodhub.FloodHubClient``): gauge-level AI
   forecasts and warning levels (``NORMAL``, ``WARNING``, ``DANGER``,
   ``EXTREME``).
2. **Copernicus GloFAS v4** (``maas.glofas.GloFASClient``): daily river
   discharge on a global 0.05° grid (about 5 km), read from the Open-Meteo
   Flood API. Each forecast carries the control run, the ensemble median, the
   ensemble mean, the 25th/75th percentiles, the min/max envelope, and all 50
   ensemble members.
3. **GEOGLOWS ECMWF v2** (``maas.geoglows.GeoGLOWSClient``): 3-hourly
   forecasts on about 6.8 million river reaches (``LINKNO`` identifiers).

All discharge values are in cubic meters per second (``m³/s``). Dates are UTC
(``YYYY-MM-DD``). The interactive map view of the same data is the
*Models-as-a-Service Hub* tab of the web UI (see ``frontend/README.md``).

Do You Need This Package?
-------------------------

- Use ``maas`` when you want **live operational forecasts** for a point on a
  river, or when you want to compare the three systems against each other.
- You do **not** need ``maas`` to train or evaluate OpenHydroNet models on the
  pre-built ``gs://caravan-multimet/v1.1`` datasets.

Prerequisites
-------------

.. code-block:: bash

   conda activate openhydronet
   pip install -e .

- GloFAS and GEOGLOWS need no credentials.
- FloodHub needs an API key, passed as ``MaaSConfig(floodhub_api_key=...)``
  or through the ``FLOODHUB_API_KEY`` environment variable. Without a key the
  FloodHub provider is reported as ``unavailable``.
- Two directories must always be given explicitly: ``cache_dir`` (SQLite
  caches) and ``river_networks_dir`` (river network index files).

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

   # 1. Match the clicked point to a reach in each river network.
   reaches = resolve_reaches(38.6270, -90.1994, config=config)

   # 2. Fetch forecasts from all three systems on one daily timeline.
   bundle = fetch_forecasts(
       config,
       38.6270,
       -90.1994,
       models=['floodhub', 'glofas', 'geoglows'],
       glofas_reference='control',  # 'control' | 'median' | 'mean'
   )
   print(bundle['peaks'])          # peak flow and peak date per model
   print(bundle['glofas_reference'])  # which GloFAS series was used

   # 3. Historical record and return-period thresholds for one provider.
   river_id = reaches['geoglows']['river_id']
   historical = fetch_historical(config, 'geoglows', reach_id=river_id)
   thresholds = fetch_return_periods(config, 'geoglows', reach_id=river_id)

   # 4. Or use the object-oriented fetcher for the full multi-model bundle.
   fetcher = MaaSDataFetcher(config)
   full = fetcher.fetch_forecasts(38.6270, -90.1994)

Choosing the GloFAS Reference Series
------------------------------------

GloFAS publishes three summary series for every grid cell and day. ``maas``
never picks one for you silently: the ``glofas_reference`` argument (default
``'control'``) selects the series used for the GloFAS daily series, its peak
flow, and its return-period exceedance.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Key
     - Record field
     - Meaning
   * - ``control``
     - ``discharge_deterministic``
     - Control (unperturbed) run forced by the ECMWF control forecast.
   * - ``median``
     - ``discharge_median``
     - Median of the 50 perturbed ensemble members for that day.
   * - ``mean``
     - ``discharge_mean``
     - Mean of the 50 perturbed ensemble members for that day.

Every GloFAS record always carries all three fields, so you can switch
without downloading again. An unknown key raises ``ValueError``.

In the web UI, the **GloFAS Reference** drop-down in the Forecast View
(``Control (Unperturbed)`` / ``Ensemble Median`` / ``Ensemble Mean``) changes
both the plotted GloFAS line and the consensus / exceedance summary at the
same time. The same choice is available on the HTTP endpoint:

.. code-block:: text

   GET /api/maas/forecast?lat=38.627&lon=-90.199&glofas_reference=median

Reading Local GloFAS Archives
-----------------------------

``fetch_historical(..., zarr_path=...)`` can read a local GloFAS Zarr or
NetCDF archive instead of calling the API. The archive must have ``time``,
``latitude``/``longitude`` coordinates and a discharge variable (default
``dis24``). If the archive also has a forecast lead-time dimension (``step``)
or an ensemble member dimension (``number``), you must say which index to
read; nothing is averaged or chosen for you:

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

Any other extra dimension raises ``ValueError``. Missing timesteps are
returned as ``NaN``.

Command-Line Usage
------------------

.. code-block:: bash

   python -m maas.cli \
     --lat 38.6270 \
     --lon -90.1994 \
     --models glofas,geoglows \
     --cache-dir ~/.cache/flood_forecasting/maas \
     --river-networks-dir data/river_networks \
     --output forecast_bundle.json

What to Watch Out For
---------------------

- Daily-mean series (FloodHub, GloFAS ``dis24``) are plotted at ``00:00 UTC``.
  GEOGLOWS points are instantaneous 3-hourly values. The web UI tooltips label
  each value as ``daily mean`` or ``3-hourly instantaneous``.
- Forecast records must have an ISO-8601 ``time`` and numeric (or ``null``)
  values. Malformed records raise an error instead of being skipped.
- Missing values stay missing: they appear as ``None`` in aligned series and
  ``NaN`` in ``pandas`` output. Nothing is interpolated or filled.
- Return periods come from the provider's own reanalysis / retrospective
  record (USGS Bulletin 17C EMA by default; see :doc:`return_periods`).

Testing
-------

.. code-block:: bash

   # Offline unit tests (hermetic, no network)
   pytest maas/tests frontend/tests/test_maas_viewer.py -m "not canary"

   # Opt-in live canary against Open-Meteo GloFAS and GEOGLOWS
   MAAS_LIVE_CANARY=1 pytest maas/tests/test_canary_live.py -m canary -v

The live canary checks that Open-Meteo returns all 50 GloFAS ensemble members
and that the published min/max envelope matches those members.

API reference: :doc:`../api/maas`.
