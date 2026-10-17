=====================
Catchment Delineation
=====================

This guide explains how to trace upstream watershed boundary polygons (``EPSG:4326``) and calculate drainage areas (:math:`\text{km}^2`) for river gauges using ``delineate-catchment`` and ``download-merit-d8-tiles``.

.. note::
   **Do you need this tool?**
   If you are using gauges from the published Caravan or Caravan-MultiMet collections, **you do not need to run this tool**. Pre-delineated watershed boundary polygons are already available in ``gs://open-multimet/caravan-new/<collection>/shapefiles-rederived/`` and ``gs://caravan-multimet/v1.1``.

   Use this tool when you want to delineate watershed boundary polygons for new river gauge locations or evaluate delineation accuracy against reference polygons.

.. contents:: On this page
   :local:
   :depth: 2

--------
Overview
--------

To extract weather time series (MultiMet) or static watershed attributes (Caravan) for a new river gauge, you first need the **polygon boundary** of the land area that drains into that gauge (also called the *outlet* or *pour point*).

Given the latitude and longitude of a river gauge, ``delineate-catchment`` produces that polygon in three steps:

1. **Snaps to the river channel:** Gauge coordinates recorded by water agencies are often a few hundred meters away from the center of the river on a digital map. The tool searches nearby 90 m grid cells (about ``1.1 km`` by default) to move the point directly onto the river channel. If you also know the approximate drainage area of the gauge (``--expected-area`` or ``--area-col``), the tool searches up to ``7.2 km`` around the point to find the specific river channel matching that area.
2. **Traces upstream water flow:** Each 90 m pixel in a **D8 flow-direction map** records which of its 8 neighboring pixels water flows into. Starting from the snapped river point, the tool follows water flow backward (upstream) through every pixel that drains into the gauge. If a large river crosses from one 5° × 5° map tile into neighboring tiles, the tool loads those tiles automatically so rivers are never cut off at tile borders.
3. **Saves the polygon:** All upstream pixels are combined into a single boundary polygon (in ``EPSG:4326`` / WGS84 coordinates) and saved as a **GeoJSON**, **GeoParquet**, or **ESRI Shapefile** with the calculated drainage area in square kilometers (:math:`\text{km}^2`).

-----------------
Data Requirements
-----------------

Before running ``delineate-catchment``, you need a folder (local or on Google Cloud Storage ``gs://``) containing 90 m (3-arc-second) D8 flow-direction map tiles saved as 5° × 5° NumPy (``.npy``) files named by their top-left corner coordinate (such as ``n35w090.npy`` or ``n40w090.npy``).

* **Canonical HydroSHEDS 90 m tiles on GCS:** Public 5° × 5° HydroSHEDS 90 m tiles are hosted at ``gs://open-multimet/ancillary-data/dems/tiles_5deg``. Pass ``--gcs-uri gs://open-multimet/ancillary-data/dems/tiles_5deg --cache-dir /tmp/tile_cache`` (or ``--tiles-dir /path/to/local/tiles_5deg``).
* **Downloading MERIT-Hydro 90 m tiles from Google Earth Engine (``download-merit-d8-tiles``):** If you want to use MERIT-Hydro 90 m flow-direction grids (``--dem merit_hydro_90m``), use the ``download-merit-d8-tiles`` CLI tool to download 5° × 5° ``.npy`` tiles from Google Earth Engine:

  .. code-block:: bash

     download-merit-d8-tiles \
       --target-dir /path/to/merit_tiles_5deg \
       --ee-project your-gcp-project-id \
       --bbox -125 24 -66 50 \
       --workers 12

* **Slicing raw continental GeoTIFF files:** If you have raw HydroSHEDS or MERIT GeoTIFF files on disk, use ``multimet/catchment_delineation/tools/slice_continental_dems.py`` to slice them into 5° × 5° ``.npy`` tiles:

  .. code-block:: bash

     python multimet/catchment_delineation/tools/slice_continental_dems.py \
       --input-tifs /path/to/raw_hydrosheds_tifs/*.tif \
       --out-dir /path/to/tiles_5deg

-----------
Quick Start
-----------

1. Delineate a Single Gauge
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run ``delineate-catchment`` with ``--lat``, ``--lon``, ``--tiles-dir``, and an output file ``-o``:

.. code-block:: bash

   delineate-catchment \
     --lat 39.6828 \
     --lon -88.7729 \
     --id USGS_05592500 \
     --expected-area 480.0 \
     --tiles-dir /path/to/tiles_5deg \
     -o basin.geojson \
     --pretty

.. tip::
   Whenever your water agency provides a published drainage area for a gauge, pass it with ``--expected-area`` (in :math:`\text{km}^2`). This helps the tool snap to the main river rather than a small nearby tributary.

2. Delineate a Table of Gauges (CSV or Parquet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have many gauges, list them in a CSV or Parquet file (for example, ``gauges.csv``):

.. code-block:: text

   gauge_id,latitude,longitude,area_km2
   camels_01013500,47.2374,-68.5826,2252.7
   camels_03335500,40.4172,-86.8858,18821.0

Then run ``delineate-catchment`` with ``--csv`` and ``--workers`` to process multiple gauges in parallel:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --area-col area_km2 \
     --tiles-dir /path/to/tiles_5deg \
     --workers 8 \
     -o basins.geoparquet

3. Save in Standard Caravan Folder Layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your ``gauge_id`` values follow the Caravan naming style ``<subdataset>_<id>`` (such as ``camels_01013500`` or ``grdc_6340110``), add ``--preserve-caravan-dirs`` and ``--output-dir``. This writes the polygons into the folder structure expected by Caravan and MultiMet tools:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --area-col area_km2 \
     --tiles-dir /path/to/tiles_5deg \
     --output-dir /path/to/caravan_dataset \
     --preserve-caravan-dirs \
     --format all \
     --workers 8

This creates:

.. code-block:: text

   /path/to/caravan_dataset/
   └── shapefiles/
       └── camels/
           ├── camels_basin_shapes.geoparquet
           ├── camels_basin_shapes.geojson
           ├── camels_basin_shapes.shp
           ├── camels_basin_shapes.shx
           ├── camels_basin_shapes.dbf
           ├── camels_basin_shapes.prj
           └── camels_basin_shapes.cpg

4. Read Map Tiles from Google Cloud Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your 5° × 5° ``.npy`` tiles are stored in a ``gs://`` bucket, provide both ``--gcs-uri`` and a local ``--cache-dir`` where downloaded tiles can be stored temporarily. Add ``--clean-cache`` if you want those downloaded files deleted automatically when the command finishes:

.. code-block:: bash

   delineate-catchment \
     --csv gauges.csv \
     --area-col area_km2 \
     --gcs-uri gs://open-multimet/ancillary-data/dems/tiles_5deg \
     --cache-dir /tmp/dem_tile_cache \
     --clean-cache \
     --workers 8 \
     -o basins.geoparquet

5. Use from Python
^^^^^^^^^^^^^^^^^^

You can also run delineation directly inside a Python script or notebook:

.. code-block:: python

   from multimet.catchment_delineation import DemDelineator

   delineator = DemDelineator(tiles_dir="/path/to/tiles_5deg")

   feature = delineator.delineate(
       lat=39.6828,
       lon=-88.7729,
       catchment_id="USGS_05592500",
       expected_area_km2=480.0,  # optional expected area in km²
   )

   print("Gauge ID:", feature["properties"]["catchment_id"])
   print("Area (km²):", feature["properties"]["area_km2"])

----------------------
What to Watch Out For
----------------------

Please keep these five points in mind when running the tool:

1. **Always provide your tile and output paths**
   Pass the path to your map tiles (``--tiles-dir`` or ``--gcs-uri`` + ``--cache-dir``) and your output destination (``-o`` or ``--output-dir``).

2. **Gauges on wide rivers benefit from an expected area hint (``--expected-area`` or ``--area-col``)**
   On a 90 m grid, a wide river (such as the Danube, Mississippi, or Amazon) spans many pixels. A gauge coordinate near the riverbank can sit closer to a tiny creek on the bank than to the center of the main river.

   * Passing ``--expected-area`` (or ``--area-col`` for CSV tables) tells the tool to find the nearby channel whose drainage area is within ``±50%`` (configurable via ``--area-tolerance``) of the expected area.
   * If you provide an expected area and no river within ``~7.2 km`` matches that area, the tool logs an ``[AREA HINT FAILURE]`` error and raises ``CatchmentAreaMismatchError`` rather than outputting a mismatched polygon.

3. **Latitude coverage**
   * **HydroSHEDS 90 m (``--dem hydrosheds_90m``):** Covers latitudes from ``-56°S`` to ``60°N``. If a gauge is north of ``60°N`` or its upstream river reaches north across ``60°N``, the tool stops rather than cutting the river off at the ``60°N`` border. In batch runs, these basins are recorded with ``status: "MISSING_DATA: ..."`` and ``geometry: null``.
   * **MERIT-Hydro 90 m (``--dem merit_hydro_90m``):** Extends high-latitude coverage up to ``90°N`` when MERIT-Hydro 5° × 5° D8 tiles are used.

4. **All upstream map tiles must be present**
   Large rivers can start hundreds of kilometers away from the gauge and cross several 5° × 5° map tiles (named ``{lat_top}{lon_left}.npy``, such as ``n40w090.npy`` and ``n45w090.npy``). If your ``--tiles-dir`` has the tile for the gauge location but is missing an upstream tile that flows into that river, the tool raises ``FileNotFoundError`` instead of returning a truncated polygon.

5. **Coordinate order and measurement units**

   * **Coordinates:** Always use decimal degrees (``EPSG:4326`` / WGS84) with **latitude first** and **longitude second** (``-180`` to ``180``).
   * **Drainage areas:** All area inputs (``--expected-area``, ``--area-col``) and outputs (``area_km2``, ``area``) are in **square kilometers** (:math:`\text{km}^2`).

----------------------------------------------------
``delineate-catchment`` Command-Line Arguments (CLI)
----------------------------------------------------

All arguments for ``delineate-catchment`` (also runnable as ``python -m multimet.catchment_delineation``) are listed below by group.

Coordinate Input Options
^^^^^^^^^^^^^^^^^^^^^^^^

-  ``--lat`` *(float, default: None)*: Latitude of a single river gauge or outlet point in decimal degrees (e.g., ``39.6828``). Must be used together with ``--lon``.
-  ``--lon`` *(float, default: None)*: Longitude of a single river gauge or outlet point in decimal degrees (e.g., ``-88.7729``). Must be used together with ``--lat``.
-  ``--id`` *(string, default: None)*: Custom name or gauge ID when running a single point with ``--lat`` and ``--lon`` (e.g., ``USGS_05592500``). If omitted, an ID is generated automatically from the coordinates.
-  ``--coords`` *(one or more strings, default: None)*: Space-separated ``"lat,lon"`` coordinate pairs for running a few points without creating a CSV file (e.g., ``--coords "39.68,-88.77" "40.42,-86.89"``).
-  ``--csv`` *(path or gs:// URI, default: None)*: Path to a CSV or Parquet file containing gauge coordinates, or a Caravan directory containing ``attributes/`` tables.
-  ``--lat-col`` *(string, default: auto-detected)*: Name of the latitude column in your ``--csv`` file. Recognized defaults: ``latitude``, ``lat``, ``gauge_lat``, ``caravan:gauge_lat``, ``outlet_lat``, ``pour_point_lat``.
-  ``--lon-col`` *(string, default: auto-detected)*: Name of the longitude column in your ``--csv`` file. Recognized defaults: ``longitude``, ``lon``, ``long``, ``lng``, ``gauge_lon``, ``caravan:gauge_lon``, ``outlet_lon``, ``pour_point_lon``.
-  ``--id-col`` *(string, default: auto-detected)*: Name of the gauge ID column in your ``--csv`` file. Recognized defaults: ``gauge_id``, ``catchment_id``, ``station_id``, ``hybas_id``, ``id``, ``caravan:gauge_id``.
-  ``--workers``, ``-w`` *(int, default: 1)*: Number of CPU processes to run in parallel during batch runs.

Algorithm, DEM & Snapping Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``--dem`` *(string, default: ``hydrosheds_90m``)*: Digital Elevation Model (DEM) dataset identifier (``hydrosheds_90m`` or ``merit_hydro_90m``).
-  ``--mode`` *(choice: ``dem``, ``hybrid``, ``ridgeline``, ``exact_pour_point``, ``unit_catchment``; default: ``dem``)*: Delineation method:

   * ``dem``: Pure 90 m D8 flow-direction grid tracing.
   * ``hybrid``: Vector river reach snapping (HydroRIVERS or MERIT-Basins) combined with 90 m D8 grid tracing.
   * ``ridgeline``: Dissolves all upstream vector unit catchments (HydroBASINS Level 12 or MERIT-Basins).
   * ``exact_pour_point``: Dissolves upstream vector unit catchments and clips the local outlet unit catchment at the pour point.
   * ``unit_catchment``: Returns only the local vector unit catchment containing the pour point.

-  ``--tiles-dir`` *(path or gs:// URI, default: None)*: Folder containing the 5° × 5° ``.npy`` flow-direction tiles (such as ``n35w090.npy``). You must provide either ``--tiles-dir`` or ``--gcs-uri`` (for ``dem`` and ``hybrid`` modes).
-  ``--gcs-uri`` *(gs:// URI, default: None)*: Google Cloud Storage folder containing the 5° × 5° ``.npy`` flow-direction tiles (e.g., ``gs://open-multimet/ancillary-data/dems/tiles_5deg``). Must be paired with ``--cache-dir``.
-  ``--cache-dir`` *(local path, default: None)*: Local folder where tiles downloaded from Google Cloud Storage are saved.
-  ``--hydrorivers-shp`` *(path, default: None)*: Path to ``HydroRIVERS_v10.shp`` (used in ``hybrid`` and ``exact_pour_point`` modes with HydroSHEDS).
-  ``--hydrobasins-dir`` *(path, default: None)*: Directory containing HydroBASINS Level 12 shapefiles (used in ``ridgeline``, ``exact_pour_point``, and ``unit_catchment`` modes with HydroSHEDS).
-  ``--merit-rivers-dir`` *(path, default: None)*: Directory containing MERIT-Basins ``riv_pfaf_*.shp`` files (used in ``hybrid`` and vector modes with MERIT-Hydro).
-  ``--merit-catchments-dir`` *(path, default: None)*: Directory containing MERIT-Basins ``cat_pfaf_*.shp`` files (used in vector modes with MERIT-Hydro).
-  ``--snap-window`` *(int, default: 12)*: Half-width of the search box (in 90 m grid cells) used to snap the gauge coordinate onto the nearest river channel (``12`` cells is roughly ``1.1 km``).
-  ``--expected-area`` *(float, default: None)*: Approximate expected drainage area in :math:`\text{km}^2` for a single gauge (or applied to all points if ``--area-col`` is not set). Searches up to ``80`` grid cells (``~7.2 km``) for a river channel whose upstream area matches ``--expected-area`` within ``--area-tolerance``.
-  ``--area-col`` *(string, default: None)*: Column name in your ``--csv`` or Parquet file containing the expected drainage area (:math:`\text{km}^2`) for each gauge.
-  ``--area-tolerance`` *(float, default: 0.50)*: Allowed relative difference between the delineated area and ``--expected-area`` / ``--area-col`` (``0.50`` = ``±50%``).
-  ``--max-cells`` *(int, default: None)*: Optional upper limit on the number of 90 m grid cells traced upstream (``None`` means no limit).

Output & Utility Options
^^^^^^^^^^^^^^^^^^^^^^^^

-  ``-o``, ``--output`` *(path or gs:// URI, default: stdout)*: File path where the delineated polygons are saved (``.geojson``, ``.geoparquet``, ``.parquet``, or ``.shp``).
-  ``--output-dir`` *(path or gs:// URI, default: None)*: Root folder for saving multi-file or Caravan-structured outputs.
-  ``--preserve-caravan-dirs`` *(flag, default: False)*: Groups gauges by the dataset prefix in ``gauge_id`` (such as ``camels`` in ``camels_01013500``) and saves outputs into ``<output-dir>/shapefiles/<subdataset>/<subdataset>_basin_shapes.*``.
-  ``--format`` *(choice: ``all``, ``geoparquet``, ``parquet``, ``geojson``, ``shp``; default: ``all``)*: File format(s) to write when using ``--output-dir`` or ``--preserve-caravan-dirs``.
-  ``--pretty`` *(flag, default: False)*: Formats JSON/GeoJSON output with indentation.
-  ``--clean-cache`` *(flag, default: False)*: Deletes only the ``.npy`` tile files that were downloaded into ``--cache-dir`` during this run.
-  ``--list-tiles`` *(flag, default: False)*: Lists all ``.npy`` tile files found in ``--tiles-dir`` and exits.

-------------------------------------------------------------
``download-merit-d8-tiles`` Command-Line Arguments (CLI)
-------------------------------------------------------------

Use ``download-merit-d8-tiles`` (``multimet.catchment_delineation.tools.download_merit_d8_tiles``) to download 5° × 5° MERIT-Hydro 90 m D8 flow-direction ``.npy`` tiles from Google Earth Engine:

-  ``--target-dir`` *(path, required)*: Destination directory where ``.npy`` tiles are saved.
-  ``--ee-project`` *(string, required)*: Google Cloud project ID with Google Earth Engine enabled.
-  ``--tiles`` *(one or more strings, mutually exclusive with ``--reference-tiles-dir`` and ``--bbox``)*: Explicit 5° × 5° tile names to download (for example, ``n40w090 n45w090``).
-  ``--reference-tiles-dir`` *(path, mutually exclusive with ``--tiles`` and ``--bbox``)*: Existing directory of ``.npy`` tiles whose tile names should be mirrored.
-  ``--bbox`` *(4 floats ``MIN_LON MIN_LAT MAX_LON MAX_LAT``, mutually exclusive with ``--tiles`` and ``--reference-tiles-dir``)*: Geographic bounding box in decimal degrees (``EPSG:4326``) to cover.
-  ``--workers`` *(int, default: 12)*: Number of parallel download threads.
