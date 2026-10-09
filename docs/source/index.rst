.. OpenHydroNet documentation master file, created by
   sphinx-quickstart on Mon Aug 17 14:00:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenHydroNet's documentation!
===========================================

This is the documentation for the OpenHydroNet Python package.
The source code is available on `GitHub <https://github.com/google-research/flood-forecasting/>`_.

On this documentation page, you'll find a :doc:`quickstart guide <usage/quickstart>` with step-by-step instructions on installation, required datasets, and command-line usage.
There is also a :doc:`tutorial <tutorial/tutorial>` that walks you through training your first model.
The :doc:`modelzoo <usage/models>` lists the models available in this repository.
The :doc:`catchment delineation guide <usage/catchment_delineation>` explains how to extract watershed boundary polygons directly from DEM flow direction grids.
If you are working with your own watersheds, the :doc:`static attribute extractor guide <usage/static_extractor>` shows how to create Caravan-compatible static attribute tables from watershed boundary files.
The :doc:`gridded weather archives <usage/gridded_archives>` guide explains how to download and build daily gridded precipitation archives from NOAA CPC and NASA GPM IMERG.
The :doc:`gridded weather forecasts <usage/weather_fetcher>` guide explains how to download the latest operational forecast runs and read them as grids, point meteograms, wind fields, and catchment averages.
The :doc:`return periods guide <usage/return_periods>` explains how to compute flood frequency quantiles and return periods using the USGS Bulletin 17C algorithm.
The :doc:`canonical benchmarks guide <usage/benchmarks>` explains how to run the six end-to-end verification benchmarks across all core components.
Finally, the :doc:`API docs <api/model>` show in-depth information on all modules, classes, and functions within OpenHydroNet.

You might also be interested in our `team's webpage <https://sites.research.google/gr/floodforecasting/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/quickstart
   usage/catchment_delineation
   usage/static_extractor
   usage/models
   usage/gridded_archives
   usage/multimet_extractor
   usage/weather_fetcher
   usage/return_periods
   usage/benchmarks
   tutorial/tutorial
   usage/config
   api/modules
   example-configs/example-configs