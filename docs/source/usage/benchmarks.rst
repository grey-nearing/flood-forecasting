=========================
Canonical Benchmark Suite
=========================

The ``benchmarks`` package provides six command-line tools to verify and benchmark every core component of OpenHydroNet against published reference datasets (Caravan, Caravan-MultiMet v1.1 ``gs://caravan-multimet/v1.1``, HydroATLAS v1.0, and USGS Bulletin 17C ``peakfqr`` / ``MGBT``).

.. note::
   **Do you need these tools?**
   Most users training or running flood forecasting models do not need to run this benchmark suite. These tools are intended for contributors and researchers verifying numerical accuracy after modifying catchment delineation, static or dynamic data extractors, return period fitting, or model architectures.

.. contents:: On this page
   :local:
   :depth: 2

Overview of Benchmark Commands
------------------------------

Installing OpenHydroNet (``pip install -e .``) registers six benchmark commands:

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - CLI Command
     - Python Module
     - What It Verifies
   * - ``benchmark-catchment``
     - :mod:`benchmarks.catchment_delineation`
     - Evaluates 90 m (3-arc-second) D8 watershed delineation and pour-point snapping against published Caravan and GRDC reference polygons (``IoU``, ``Dice``, and relative area error in :math:`\text{km}^2`).
   * - ``benchmark-static-extractor``
     - :mod:`benchmarks.static_extractor`
     - Compares extracted HydroATLAS Level 12 (196 variables) and ERA5-Land climate (14 variables) attributes against the 210 published Caravan reference attributes.
   * - ``benchmark-gridded-archive``
     - :mod:`benchmarks.gridded_archive_builders`
     - Rebuilds a date window of NOAA CPC (``0.5°``) or NASA IMERG (``0.1°``) gridded precipitation archives from upstream files and checks cell-by-cell parity against a reference ``daily_surface.zarr`` store.
   * - ``benchmark-timeseries-extractor``
     - :mod:`benchmarks.timeseries_extractors`
     - Reconstructs catchment-averaged MultiMet weather time series (``CPC``, ``IMERG``, ``HRES``) from gridded Zarr archives and compares them against canonical ``gs://caravan-multimet/v1.1`` stores.
   * - ``benchmark-return-periods``
     - :mod:`benchmarks.return_periods`
     - Verifies ``MultipleGrubbsBeckTester`` (``MGBT``) and ``GEMAFitter`` (``EMA`` Log-Pearson Type III) unit-conversion invariance (``live`` mode) and 4-way numerical parity against USGS Fortran ``peakfqr`` v8.0 and CRAN R ``MGBT`` v1.1.6 (``external-usgs`` mode).
   * - ``benchmark-model``
     - :mod:`benchmarks.model`
     - Compares forecasting model skill (``NSE``, ``KGE``) across canonical vs. reconstructed weather forcings, evaluates model architectures (``MeanEmbeddingForecastLSTM``, ``HandoffForecastLSTM``), and verifies cold-start vs. hot-start LSTM state handoff.

Benchmark Evaluation Principles
-------------------------------

Every benchmark in ``benchmarks`` follows four strict rules so errors are never hidden:

1. **No Fallback to Reference Data:** Benchmarks never substitute reference polygons, reference attributes, or ``union_mapping`` fallback variables when a component fails or outputs ``NaN``.
2. **Explicit Missing-Data (``NaN``) Accounting:** Every benchmark counts and flags any case where the tested component outputs ``NaN`` while the reference value is valid.
3. **Both Conditional and Unconditional Metrics:** Reports show both *conditional* accuracy (scored on valid outputs) and *unconditional* accuracy (where unexpected failures or ``NaN`` outputs are penalized across 100% of in-coverage basins).
4. **Full Percentile Distributions:** Instead of reporting only a single mean or median, reports include lower-tail skill percentiles (``Min, P1, P5, P10, P25, P50``) and upper-tail error percentiles (``P50, P75, P90, P95, P99, Max``).

CLI Reference & Quick-Start Examples
------------------------------------

1. ``benchmark-catchment``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Flag
     - Required
     - Description
   * - ``--dataset``
     - Yes
     - Path to benchmark Parquet dataset (containing ``latitude``, ``longitude``, ``reference_area_km2``, and ``geometry_wkt``).
   * - ``--dem``
     - No
     - DEM dataset identifier (``hydrosheds_90m`` or ``merit_hydro_90m``, default ``hydrosheds_90m``).
   * - ``--mode``
     - No
     - Delineation mode: ``dem`` (default) or ``hybrid`` (requires ``--river-network-path``).
   * - ``--tiles-dir``
     - Conditional
     - Local directory (or ``gs://`` URI) containing 5° × 5° ``.npy`` D8 flow-direction tiles.
   * - ``--gcs-uri`` / ``--cache-dir``
     - Conditional
     - GCS URI containing 5° × 5° ``.npy`` tiles and local download cache directory.
   * - ``--workers``
     - No
     - Number of parallel worker processes (default ``8``).
   * - ``--snap-window``
     - No
     - Snap search window half-width in 90 m grid cells (default ``12``).
   * - ``--no-area-hint``
     - No
     - Disable passing ``reference_area_km2`` as the expected area hint.
   * - ``--area-tolerance``
     - No
     - Allowed relative difference around ``reference_area_km2`` (default ``0.50``).
   * - ``--output``
     - No
     - Output file path (``.csv`` or ``.parquet``) for per-basin benchmark results.
   * - ``--save-geometries``
     - No
     - Include delineated polygon WKT (``del_geometry_wkt``) in ``--output``.
   * - ``--export-redelineated-dataset``
     - No
     - Export a copy of ``--dataset`` with ``geometry_wkt`` replaced by ``del_geometry_wkt`` for downstream cascade benchmarking.

.. code-block:: bash

   benchmark-catchment \
     --dataset /path/to/benchmark_basins_500.parquet \
     --tiles-dir /path/to/tiles_5deg \
     --workers 16 \
     --save-geometries \
     --output /tmp/catchment_500_results.parquet \
     --export-redelineated-dataset /tmp/benchmark_basins_500_redelineated.parquet

2. ``benchmark-static-extractor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Flag
     - Required
     - Description
   * - ``--dataset``
     - Yes
     - Path to benchmark Parquet dataset containing ``geometry_wkt`` and ``ref_*`` Caravan columns.
   * - ``--output-dir``, ``-o``
     - Yes
     - Directory to write ``benchmark_attribute_metrics.csv``, ``benchmark_basin_metrics.csv``, and ``benchmark_report.md``.
   * - ``--gdb-path``
     - Conditional
     - Local path to ``BasinATLAS_v10.gdb`` (or pass ``--gcs-gdb-uri`` with ``--no-download``).
   * - ``--era5-source``
     - Yes
     - Climate extraction mode: ``hybas``, ``gridded``, or ``none`` (196 HydroATLAS attributes only).
   * - ``--era5-cache-dir``
     - Conditional
     - Local directory containing ``{continent}_climate_indices.txt`` tables when ``--era5-source hybas``.
   * - ``--gridded-era5-uri``
     - Conditional
     - Path or ``gs://`` URI to gridded daily ERA5-Land Zarr store when ``--era5-source gridded``.
   * - ``--workers``
     - No
     - Number of parallel worker processes (default ``8``).

.. code-block:: bash

   benchmark-static-extractor \
     --dataset /path/to/benchmark_basins_500.parquet \
     --gdb-path /path/to/BasinATLAS_v10.gdb \
     --era5-source hybas \
     --era5-cache-dir /path/to/era5_climate \
     --workers 16 \
     -o /tmp/static_benchmark_out

3. ``benchmark-gridded-archive``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Flag
     - Required
     - Description
   * - ``--product``
     - Yes
     - Gridded precipitation product to rebuild and verify: ``CPC`` or ``IMERG``.
   * - ``--start-date``, ``--end-date``
     - Yes
     - Inclusive date window (``YYYY-MM-DD``).
   * - ``--reference-zarr``
     - Yes
     - Path or ``gs://`` URI to the reference ``daily_surface.zarr`` store.
   * - ``--output-dir``
     - Yes
     - Directory for summary/daily CSV tables, ``benchmark_report.md``, and the rebuilt Zarr store.
   * - ``--num-workers``
     - No
     - Number of parallel download/extraction workers (default ``4``).

.. code-block:: bash

   benchmark-gridded-archive \
     --product CPC \
     --start-date 2020-01-01 \
     --end-date 2020-01-31 \
     --reference-zarr gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
     --output-dir /tmp/cpc_archive_bench

4. ``benchmark-timeseries-extractor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Flag
     - Required
     - Description
   * - ``--dataset``
     - Yes
     - Path to benchmark basin dataset (``.parquet``, ``.geojson``, or ``.shp``).
   * - ``--canonical-dir``
     - Yes
     - Root directory or ``gs://`` URI containing canonical ``<PRODUCT>/timeseries.zarr`` stores (e.g., ``gs://caravan-multimet/v1.1``).
   * - ``--archive-store``
     - Yes
     - Repeatable ``PRODUCT=URI`` mapping for gridded Zarr archives (e.g., ``CPC=gs://...``).
   * - ``--date-windows``
     - Yes
     - One or more ``START:END`` date windows (``YYYY-MM-DD:YYYY-MM-DD``).
   * - ``--output-dir``
     - Yes
     - Directory where CSV tables, ``per_basin_metrics.parquet``, and ``benchmark_report.md`` are written.
   * - ``--save-reconstructed-zarr``
     - No
     - Optional directory to write reconstructed ``<PRODUCT>/timeseries.zarr`` stores for downstream model benchmarking.
   * - ``--num-workers``
     - No
     - Number of threads for ``ZonalWeightMatrix`` construction (default ``8``).

.. code-block:: bash

   benchmark-timeseries-extractor \
     --dataset /path/to/benchmark_basins_500.parquet \
     --canonical-dir gs://caravan-multimet/v1.1 \
     --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
     --date-windows 2016-01-01:2023-12-31 \
     --output-dir /tmp/timeseries_bench \
     --num-workers 8

5. ``benchmark-return-periods``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Flag
     - Required
     - Description
   * - ``--mode``
     - No
     - ``external-usgs`` (default, 4-way comparison against compiled USGS Fortran ``peakfqr`` and CRAN R ``MGBT``) or ``live`` (pure-Python verification on Caravan Zarr/NetCDF).
   * - ``--caravan-dir``
     - Yes
     - Path to Caravan dataset root directory (Zarr or NetCDF).
   * - ``--output-dir``, ``-o``
     - Yes
     - Directory to write benchmark CSV tables, figures, and Markdown reports.
   * - ``--dataset``
     - No
     - Optional benchmark basin Parquet file to filter basins in ``--mode live``.
   * - ``--peakfqr-repo``
     - Conditional
     - Path to cloned ``https://code.usgs.gov/water/peakfqr.git`` repository (required when ``--mode external-usgs``).
   * - ``--mgbt-repo``
     - Conditional
     - Path to cloned ``https://github.com/cran/MGBT.git`` repository (required when ``--mode external-usgs``).
   * - ``--peakfq-so``
     - No
     - Optional explicit path to compiled ``peakfq.so`` shared library.
   * - ``--print-setup-instructions``
     - No
     - Print step-by-step instructions for cloning and compiling the USGS Fortran and R repositories and exit.

.. code-block:: bash

   # Pure-Python live verification mode (no Fortran or R required)
   benchmark-return-periods \
     --mode live \
     --caravan-dir /path/to/Caravan-zarr \
     --dataset /path/to/benchmark_basins_500.parquet \
     --output-dir /tmp/return_periods_live

   # External USGS Fortran peakfqr + CRAN R MGBT parity mode
   benchmark-return-periods \
     --mode external-usgs \
     --caravan-dir /path/to/Caravan-nc \
     --peakfqr-repo /path/to/peakfqr \
     --mgbt-repo /path/to/MGBT \
     --output-dir /tmp/return_periods_external

6. ``benchmark-model``
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Flag
     - Required
     - Description
   * - ``--mode``
     - No
     - ``all`` (default), ``compare-forcings`` (alias ``forcing-sensitivity``), ``architectures``, or ``hot-start``.
   * - ``--canonical-multimet-dir``
     - Yes
     - Path to canonical MultiMet Zarr directory (alias ``--canonical-dynamics-dir``).
   * - ``--reconstructed-multimet-dir``
     - Conditional
     - Path to reconstructed MultiMet Zarr directory (required except in ``--mode hot-start``).
   * - ``--caravan-dir``
     - Yes
     - Path to Caravan directory containing ``streamflow.zarr`` and ``attributes.zarr`` (alias ``--targets-dir``).
   * - ``--basin-file``
     - Yes
     - Path to text file listing gauge IDs (alias ``--basins-file``).
   * - ``--output-dir``, ``-o``
     - Yes
     - Directory to write CSV and JSON benchmark outputs.

.. code-block:: bash

   benchmark-model \
     --mode all \
     --basin-file /path/to/basins_25.txt \
     --caravan-dir /path/to/Caravan-zarr \
     --canonical-multimet-dir /path/to/canonical_zarr \
     --reconstructed-multimet-dir /path/to/reconstructed_zarr \
     --output-dir /tmp/model_bench \
     --seq-length 180 \
     --lead-time 7 \
     --epochs 3 \
     --device cpu

API reference: :doc:`../api/benchmarks`.
