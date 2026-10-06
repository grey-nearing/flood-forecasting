========================
Gridded Weather Archives
========================

This guide explains how to use the command-line tools in the
:mod:`multimet.gridded_archive_builders` package to download public gridded
precipitation data and save it in standardized daily Zarr archives.

.. note::

   **Do you need these tools?**
   If you only want to train or evaluate flood-forecasting models using the
   published MultiMet dataset, **you do not need to run these tools**. Simply
   point ``dynamics_data_dir`` in your configuration file to
   ``gs://caravan-multimet/v1.1`` (see :doc:`quickstart`).

   Use these tools only if you want to download raw precipitation grids
   directly from NOAA or NASA and build or update your own Zarr archives.

--------
Overview
--------

Two command-line tools for gridded archive construction are installed when you
run ``pip install -e .`` from the repository root:

.. list-table::
   :header-rows: 1
   :widths: 25 40 15 20

   * - Command
     - Dataset
     - Spatial Grid
     - Available Dates
   * - ``build-cpc-archive``
     - NOAA CPC Global Unified Daily Precipitation
     - 0.5° (``360 × 720``)
     - 1979 to present
   * - ``build-imerg-archive``
     - NASA GPM IMERG Early V07 Daily Precipitation
     - 0.1° (``1800 × 3600``)
     - 2000-06-01 to present

Each tool downloads raw files from the upstream weather agency, validates
coordinates and dimensions, converts them to a consistent daily
``(time, latitude, longitude)`` grid, replaces missing values with ``NaN``, and
saves the result to the ``--target_zarr`` location you provide.

-------------
Prerequisites
-------------

Activate the ``openhydronet`` Conda environment and install the package:

.. code-block:: bash

   conda activate openhydronet
   pip install -e .

Additional Requirements by Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **NOAA CPC (** ``build-cpc-archive`` **):** No extra packages or accounts are
  needed.
* **NASA GPM IMERG (** ``build-imerg-archive`` **):** Downloading directly from
  NASA GES DISC requires a free `NASA Earthdata Login
  <https://urs.earthdata.nasa.gov/>`_ account. You can provide your credentials
  in any of three ways:

  1. A ``~/.netrc`` file on your computer with entries for
     ``urs.earthdata.nasa.gov`` and ``gpm1.gesdisc.eosdis.nasa.gov``.
  2. Environment variables: ``EARTHDATA_TOKEN`` (or ``EARTHDATA_USERNAME`` and
     ``EARTHDATA_PASSWORD``).
  3. Command-line arguments: ``--earthdata_token`` (or ``--earthdata_username``
     and ``--earthdata_password``).

-----------------------------------------
1. NOAA CPC Daily Precipitation
-----------------------------------------

``build-cpc-archive`` downloads yearly NetCDF files (``precip.{year}.nc``) from
the NOAA Physical Sciences Laboratory, flips latitude so it runs south-to-north
(``-89.75`` to ``+89.75``), shifts longitude to ``-179.75`` to ``+179.75``, and
writes the daily precipitation variable ``cpc_precipitation`` (``mm/day``).

Example Usage
^^^^^^^^^^^^^

.. code-block:: bash

   # Build a local archive for 2020 to 2022 and delete temporary downloads
   build-cpc-archive \
     --target_zarr ./data/cpc_daily.zarr \
     --start_year 2020 \
     --end_year 2022 \
     --cleanup_cache

   # Extend an existing archive with newly published days (ignores pre-cached files)
   build-cpc-archive \
     --target_zarr ./data/cpc_daily.zarr \
     --extend_archive \
     --cleanup_cache

Command-Line Arguments (``build-cpc-archive``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``--target_zarr`` *(required)*: Path where the output Zarr archive is saved.
   Use a local folder path (e.g., ``./data/cpc.zarr``) or a Google Cloud
   Storage URI starting with ``gs://`` (e.g., ``gs://my-bucket/cpc.zarr``).
-  ``--start_year``: First year to download (integer, default: ``1979``).
-  ``--end_year``: Last year to download, inclusive (integer, default: current
   calendar year).
-  ``--start_date``: Optional start date filter (``YYYY-MM-DD``) if you only
   want dates on or after a specific day inside ``--start_year``.
-  ``--end_date``: Optional end date filter (``YYYY-MM-DD``) if you only want
   dates on or before a specific day inside ``--end_year``. When omitted, the
   active year is extended up to the latest published day on NOAA PSL (allowing
   up to 7 days of upstream publication lag while rejecting any interior date
   gaps).
-  ``--cache_dir``: Local folder used to store downloaded NOAA NetCDF files
   before processing. If omitted, a temporary folder is created and removed
   automatically when the command finishes.
-  ``--cleanup_cache``: Deletes each downloaded NetCDF file as soon as it is
   written to the Zarr archive. Recommended to save disk space.
-  ``--overwrite``: Deletes the existing Zarr archive at ``--target_zarr`` and
   rebuilds it from scratch. If not set, the tool resumes and appends only new
   dates.
-  ``--extend_archive`` (or ``--extend-archive``): Extends an existing archive
   in place without reusing any pre-cached NetCDF files; raises an error if
   ``--target_zarr`` does not already exist.
-  ``--num_workers``: Number of years to download and process in parallel
   (integer, default: number of CPU cores up to ``32``). Set to ``1`` to run
   one year at a time.
-  ``--project``: Google Cloud project ID used for billing and authentication
   when ``--target_zarr`` is a ``gs://`` bucket (default: ``None``).
-  ``--source_url_template``: Custom download URL template containing
   ``{year}`` (default: official NOAA PSL URL).

------------------------------------------
2. NASA GPM IMERG Daily Precipitation
------------------------------------------

``build-imerg-archive`` builds a daily ``0.1°`` global precipitation archive
(``1800`` latitudes ``-89.95 .. 89.95`` by ``3600`` longitudes
``-179.95 .. 179.95``) from NASA GPM IMERG Early Run Version 07 (V07), saving
the variable ``imerg_precipitation`` (``mm/day``).

Example Usage
^^^^^^^^^^^^^

.. code-block:: bash

   # Download daily V07 files from NASA GES DISC into a local Zarr archive
   build-imerg-archive \
     --target_zarr ./data/imerg_daily.zarr \
     --start_date 2024-01-01 \
     --end_date 2024-01-10 \
     --cleanup_cache

   # Extend an existing archive up to the latest published day on NASA GES DISC
   build-imerg-archive \
     --target_zarr ./data/imerg_daily.zarr \
     --extend_archive \
     --cleanup_cache

   # Build from a local directory of pre-downloaded V07 .nc4 files
   build-imerg-archive \
     --target_zarr ./data/imerg_daily.zarr \
     --source local \
     --local_format nc4 \
     --local_dir /path/to/local/imerg_files \
     --start_date 2024-01-01 \
     --end_date 2024-01-10

Command-Line Arguments (``build-imerg-archive``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``--target_zarr`` *(required)*: Path where the output Zarr archive is saved
   (local path or ``gs://`` URI).
-  ``--start_date``: First date to include in ``YYYY-MM-DD`` format (default:
   ``2000-06-01``).
-  ``--end_date``: Last date to include in ``YYYY-MM-DD`` format (default:
   latest published day within 7 days of yesterday UTC; every date up to that
   day must exist with no interior gaps).
-  ``--source``: Where to read IMERG data from (choices: ``gesdisc`` or
   ``local``, default: ``gesdisc``).

   -  ``gesdisc``: Discovers the published daily V07 NetCDF-4 granule via NASA
      CMR and downloads it from NASA GES DISC over HTTPS.
   -  ``local``: Reads pre-downloaded V07 files from ``--local_dir``.

-  ``--local_format``: Local file format when ``--source local`` is used
   (choices: ``nc4`` or ``h5``, default: ``nc4``).
-  ``--local_dir``: Path to a local folder containing pre-downloaded IMERG V07
   daily NetCDF-4 files (``.nc4`` / ``.nc``) or 48 half-hourly HDF5 granules
   (``.RT-H5`` / ``.HDF5``) per day. Required when ``--source local`` is used.
-  ``--earthdata_token``: NASA Earthdata Bearer token for ``--source gesdisc``
   (can also be set via the ``EARTHDATA_TOKEN`` environment variable).
-  ``--earthdata_username``: NASA Earthdata username (can also be set via
   ``EARTHDATA_USERNAME`` or ``~/.netrc``).
-  ``--earthdata_password``: NASA Earthdata password (can also be set via
   ``EARTHDATA_PASSWORD`` or ``~/.netrc``).
-  ``--netrc_path``: Path to a custom ``.netrc`` file containing Earthdata
   credentials (default: ``~/.netrc``).
-  ``--cache_dir`` (or ``--local_cache``): Local folder used to stage files
   downloaded from NASA GES DISC. If omitted, a temporary folder is created and
   cleaned up automatically on exit.
-  ``--cleanup_cache``: Deletes each downloaded NetCDF file immediately after
   it is processed and removes ``--cache_dir`` on exit.
-  ``--batch_size``: Number of daily grids accumulated before each Zarr write
   (integer, default: ``30``).
-  ``--num_workers``: Number of dates downloaded or extracted in parallel
   (integer, default: ``4``). Keep between ``4`` and ``8`` when downloading
   from NASA GES DISC to avoid server rate limits.
-  ``--granule_workers``: Number of parallel threads used to read the 48
   half-hourly HDF5 files per day when using ``--source local`` with
   ``--local_format h5`` (integer, default: ``8``).
-  ``--overwrite``: Deletes the existing Zarr archive at ``--target_zarr`` and
   rebuilds it from scratch.
-  ``--in_place``: Overwrites the requested ``--start_date`` to ``--end_date``
   dates in-place inside an existing Zarr archive.
-  ``--extend_archive`` (or ``--extend-archive``): Extends an existing archive
   in place without reusing any pre-cached files; raises an error if
   ``--target_zarr`` does not already exist.
-  ``--project``: Google Cloud project ID used when writing to a ``gs://``
   bucket (default: ``None``).
-  ``--gesdisc_url``: Custom base URL for NASA GES DISC IMERG V07 daily files.

----------------------------------------------
What to Watch Out For (Common Questions)
----------------------------------------------

1. **Local Paths vs. Cloud Paths (** ``gs://`` **)**
   Any ``--target_zarr`` path that does not start with a URI scheme (such as
   ``./data/cpc.zarr`` or ``output/cpc.zarr``) is saved on your local computer.
   To write to Google Cloud Storage, always include ``gs://`` at the start of
   the path (for example, ``gs://my-bucket/cpc.zarr``).

2. **Disk Space During Large Downloads**
   Multi-year weather downloads require tens of gigabytes of temporary space.
   Pass ``--cleanup_cache`` when running ``build-cpc-archive`` or
   ``build-imerg-archive`` so temporary NetCDF files are deleted as soon as each
   batch is written to the Zarr store.

3. **Safe Incremental Updates (** ``--extend_archive`` **)**
   Pass ``--extend_archive`` when running ``build-cpc-archive`` or
   ``build-imerg-archive`` against an existing Zarr archive to append newly
   published days. It guarantees that the target store already exists, ignores
   any stale pre-cached files, allows up to 7 days of upstream publication lag
   at the end of the archive when ``--end_date`` is omitted, and enforces strict
   daily continuity with zero interior date gaps.

4. **Strict Data Integrity (No Silent Fallbacks)**

   * **No version mixing in IMERG:** ``build-imerg-archive`` accepts only
     **IMERG Version 07 (V07)** files (variable ``precipitation``). Legacy
     **Version 06 (V06)** files (``precipitationCal``) raise an error immediately
     so different calibration versions are never mixed.
   * **Complete 48-half-hour requirement for local IMERG HDF5 files:** When
     summing 48 half-hourly ``.RT-H5`` files for a day (``--local_format h5``),
     all 48 unique half-hour intervals must be present, and all 48 half-hours
     must be valid at a grid cell for that cell's daily total to be finite. If
     any half-hour is missing at a grid cell, that cell is set to ``NaN`` for
     the day rather than summing an incomplete day.
   * **Network or file errors stop the run:** If a file is missing, corrupted,
     or has unexpected coordinates/dimensions, the builder stops immediately
     with an error rather than writing empty or fallback data.
