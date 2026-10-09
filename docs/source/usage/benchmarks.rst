Canonical Benchmark Suite
=========================

The ``benchmarks`` package provides six command-line tools to verify and benchmark every core component of OpenHydroNet against published reference datasets (Caravan, Caravan-MultiMet v1.1, HydroATLAS v1.0, and USGS Bulletin 17C ``peakfq`` / ``MGBT``).

.. note::
   **Do you need these tools?**
   Most users training or running flood forecasting models do not need to run the benchmark suite. These tools are intended for contributors and researchers verifying numerical parity after modifying data extractors, catchment delineation, return period fitting, or model architectures.

Available Benchmark CLIs
------------------------

Installing OpenHydroNet (``pip install -e .``) registers six benchmark commands:

1. ``benchmark-catchment`` (:mod:`benchmarks.catchment_delineation`): Evaluates 90m (3-arcsec) HydroSHEDS D8 watershed delineation and pour-point snapping against reference basin polygons.
2. ``benchmark-static-extractor`` (:mod:`benchmarks.static_extractor`): Evaluates HydroATLAS Level 12 and ERA5 climate static attribute extraction against Caravan reference attributes.
3. ``benchmark-gridded-archive`` (:mod:`benchmarks.gridded_archive_builders`): Rebuilds a date window of NOAA CPC or NASA IMERG gridded precipitation archives and checks cell-by-cell numerical parity against a reference Zarr archive.
4. ``benchmark-timeseries-extractor`` (:mod:`benchmarks.timeseries_extractors`): Reconstructs catchment-averaged MultiMet forcing timeseries (``CPC``, ``IMERG``, ``HRES``) from gridded Zarr archives and compares them against ``gs://caravan-multimet/v1.1``.
5. ``benchmark-return-periods`` (:mod:`benchmarks.return_periods`): Evaluates ``MultipleGrubbsBeckTester`` (``MGBT``) and ``GEMAFitter`` (``EMA`` Log-Pearson Type III) unit-conversion invariance and parity against USGS Fortran ``peakfqr`` and CRAN R ``MGBT``.
6. ``benchmark-model`` (:mod:`benchmarks.model`): Evaluates forecasting model sensitivity to reconstructed vs. canonical forcings, compares model architectures and dataset loading modes, and verifies cold-start vs. hot-start LSTM state handoff.

Example Usage
-------------

.. code-block:: bash

   # 1. Catchment Delineation Benchmark
   benchmark-catchment \
     --dataset /path/to/benchmark_basins_1000.parquet \
     --tiles-dir /path/to/tiles_5deg \
     --workers 16 \
     --output /tmp/catchment_results.parquet

   # 2. Static Attribute Extractor Benchmark
   benchmark-static-extractor \
     --dataset /path/to/benchmark_basins_500.parquet \
     --gdb-path /path/to/BasinATLAS_v10.gdb \
     --era5-source hybas \
     --era5-cache-dir /path/to/era5_climate \
     --workers 16 \
     -o /tmp/static_benchmark_out

   # 3. Gridded Archive Builder Benchmark
   benchmark-gridded-archive \
     --product CPC \
     --start-date 2020-01-01 \
     --end-date 2020-01-31 \
     --reference-zarr gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
     --output-dir /tmp/cpc_archive_bench

   # 4. MultiMet Catchment Timeseries Extractor Benchmark
   benchmark-timeseries-extractor \
     --dataset /path/to/benchmark_basins_500.parquet \
     --canonical-dir gs://caravan-multimet/v1.1 \
     --archive-store CPC=gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr \
     --date-windows 2016-01-01:2023-12-31 \
     --output-dir /tmp/timeseries_bench \
     --num-workers 8

   # 5. Return Period Calculator Benchmark (Pure-Python Live Mode)
   benchmark-return-periods \
     --mode live \
     --caravan-dir /path/to/Caravan-zarr \
     --output-dir /tmp/return_periods_bench

   # 6. Core Forecasting Model Benchmark
   benchmark-model \
     --mode all \
     --basin-file /path/to/basins_25.txt \
     --caravan-dir /path/to/Caravan-zarr \
     --canonical-multimet-dir /path/to/canonical_zarr \
     --reconstructed-multimet-dir /path/to/reconstructed_zarr \
     --output-dir /tmp/model_bench \
     --seq-length 180 \
     --lead-time 7 \
     --epochs 3
