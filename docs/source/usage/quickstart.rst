===========
Quick Start
===========

This guide shows how to install **OpenHydroNet** (``google-research/flood-forecasting``), set up the required datasets, and train, fine-tune, or evaluate a flood forecasting model.

.. contents:: On this page
   :local:
   :depth: 2

---------------
Obtain the Code
---------------

Clone or download the repository to get the source code, Conda environment files, example configurations (``model/example-configs/``), and pre-trained global checkpoints (``model/pretrained-models/``).

Option A: Clone with Git (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/google-research/flood-forecasting.git
   cd flood-forecasting

Option B: Download a Zip Archive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not use Git, download and unpack the repository archive:

.. code-block:: bash

   curl -L https://github.com/google-research/flood-forecasting/zipball/main -o flood-forecasting.zip
   unzip flood-forecasting.zip
   cd google-research-flood-forecasting-*

----------------------------------
Prerequisites & Environment Setup
----------------------------------

We recommend using **Conda** to install Python, PyTorch, CUDA, and geospatial libraries.

Using Conda (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

Create and activate the ``openhydronet`` environment from ``environments/conda.yml``:

.. code-block:: bash

   conda env create -f environments/conda.yml
   conda activate openhydronet

Manual Setup
^^^^^^^^^^^^

If you do not use Conda, use **Python >= 3.12** and install the dependencies listed in ``setup.py`` (or ``environments/rtd_requirements.txt``).

------------
Installation
------------

With your Python environment active, install the package in editable mode from the repository root:

.. code-block:: bash

   pip install -e .

To also install optional libraries for downloading live gridded weather forecasts (``sync-weather-forecasts``), install the ``weather`` or ``dynamical`` extras:

.. code-block:: bash

   pip install -e ".[weather,dynamical]"

----------
Data Setup
----------

OpenHydroNet trains and evaluates models on two main data sources:

1. **Caravan** (`Kratzert et al., 2023 <https://www.nature.com/articles/s41597-023-01975-w>`__): Provides daily observed river discharge (streamflow, normalized to ``mm/day``) and 210 static watershed attributes (such as drainage area in ``km²``, elevation in ``m``, slope, land cover, and climate indices).
2. **Caravan-MultiMet**: Provides daily catchment-averaged weather time series (hindcast observations and multi-day weather forecasts).

1. Download and Convert Caravan Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Visit the `Caravan Zenodo repository <https://doi.org/10.5281/zenodo.6522634>`__.
2. Download the **NetCDF archive** (``Caravan-nc.tar.gz``).
3. Extract the archive into a local folder:

   .. code-block:: bash

      mkdir -p ~/data/
      tar -xvzf Caravan-nc.tar.gz -C ~/data/

4. Convert the NetCDF files into consolidated Zarr stores (``attributes.zarr`` and ``streamflow.zarr``) for fast loading:

   .. code-block:: bash

      run convert-caravan --caravan-dir ~/data/Caravan-nc --output-dir ~/data/Caravan-zarr

2. Configure MultiMet Dynamic Weather Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can stream the public Caravan-MultiMet v1.1 dataset directly from **Google Cloud Storage** or read it from a local copy. In your YAML configuration file, set ``dynamics_data_dir`` to:

.. code-block:: yaml

   dynamics_data_dir: gs://caravan-multimet/v1.1

3. Custom Watersheds (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are modeling your own river gauges that are not in Caravan:

* Use :doc:`delineate-catchment <catchment_delineation>` to trace upstream watershed boundary polygons (``EPSG:4326``) from latitude and longitude coordinates.
* Use :doc:`extract-static-attributes <static_extractor>` to compute the 210 Caravan-compatible static watershed attributes from your polygon file.
* Use :doc:`extract-multimet <multimet_extractor>` or :doc:`multimet-realtime <multimet_extractor>` to extract daily catchment-averaged weather time series.

--------------------------------------------
Example Configs & Pre-Trained Global Models
--------------------------------------------

Every model run is controlled by a YAML configuration file:

* **Example Configurations (`model/example-configs/`):** Includes configurations reproducing former operational Google FloodHub model generations (`floodhub-settings-config.yml` for the 2025 `MeanEmbeddingForecastLSTM` generation and `handoff-forecast-lstm-config.yml` for the 2024 `HandoffForecastLSTM` generation), 531-basin CAMELS-US benchmarks, stream-gauge data assimilation (`camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml`), and hot-start inference (`template_hot_start_config.yml`). See :doc:`Example Configuration Files <../example-configs/example-configs>` for details.
* **Pre-Trained Global Checkpoints (`model/pretrained-models/`):** Three pre-trained global ``MeanEmbeddingForecastLSTM`` checkpoints (trained on 4,888 global Caravan basins from 1982 to 2022) are included in the repository:

  .. list-table::
     :header-rows: 1
     :widths: 38 22 40

     * - Directory under ``model/pretrained-models/``
       - Checkpoint File
       - Weather Inputs Used
     * - ``google-floodhub-settings-110-epochs``
       - ``model_epoch110.pt``
       - **Hindcast:** HRES, GraphCast, IMERG, CPC (with ERA5-Land fallback). **Forecast:** HRES + GraphCast.
     * - ``google-floodhub-settings-aifs-110-epochs``
       - ``model_epoch026.pt``
       - **Hindcast:** HRES, AIFS, IMERG, CPC (with ERA5-Land fallback). **Forecast:** HRES + AIFS.
     * - ``google-floodhub-settings-no-graphcast-110-epochs``
       - ``model_epoch027.pt``
       - **Hindcast:** HRES, IMERG, CPC (with ERA5-Land fallback). **Forecast:** HRES only (no AI weather model).

  See ``model/pretrained-models/Pretrained-Models-README.md`` for step-by-step instructions on evaluating or fine-tuning these checkpoints.

Basin Lists and Date Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Basin lists:** ``train_basin_file``, ``validation_basin_file``, and ``test_basin_file`` point to plain-text files with one gauge ID per line (for example, ``model/example-configs/multimet-basins-list-without-chirps.txt``). To test how well a model predicts in **ungauged watersheds** (basins without historical streamflow training data), place those gauge IDs only in ``test_basin_file``.
* **Date format:** All start and end dates in YAML configuration files use ``DD/MM/YYYY`` format (for example, ``01/01/2016``). Keep your training, validation, and test periods separate so test metrics reflect true out-of-sample performance.

Required Paths in Your Config File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before starting a run, update these five paths in your YAML configuration file:

=====================  ============================================================================================================
Config Key             What to Set
=====================  ============================================================================================================
**run_dir**            Local folder where model checkpoints, logs, and outputs are saved (for example, ``./runs/``).
**train_basin_file**   Path to a text file with one gauge ID per line (plus ``validation_basin_file`` and ``test_basin_file``).
**targets_data_dir**   Path to the directory containing ``streamflow.zarr`` (or set ``data_dir`` if all stores share one root).
**statics_data_dir**   Path to the directory containing ``attributes.zarr`` (or a list of directories to merge static attributes).
**dynamics_data_dir**  Path or GCS URI containing ``<PRODUCT>/timeseries.zarr`` (for example, ``gs://caravan-multimet/v1.1``).
=====================  ============================================================================================================

------------------
Command-Line Usage
------------------

The ``run`` command executes a single model job, and ``schedule-runs`` runs multiple jobs in parallel across one or more GPUs.

Train a New Model
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   run train --config-file model/example-configs/floodhub-settings-config.yml

Continue an Interrupted Training Run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   run continue_training --run-dir /path/to/your/model_run/

Fine-Tune a Pre-Trained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your fine-tuning YAML file, set ``base_run_dir`` to a pre-trained run directory (such as ``model/pretrained-models/google-floodhub-settings-110-epochs``) and list the layers to update in ``finetune_modules``:

.. code-block:: bash

   run finetune --config-file /path/to/your/finetune-config.yml

Evaluate on a Test Set
^^^^^^^^^^^^^^^^^^^^^^

Compute verification metrics (such as Nash-Sutcliffe Efficiency ``NSE`` and Kling-Gupta Efficiency ``KGE``) on the configured split (default ``--split test``). Add ``--assimilate`` to enable gradient-based streamflow data assimilation when ``assimilation_config`` is present in the run configuration:

.. code-block:: bash

   run evaluate --run-dir /path/to/your/model_run/ --split test --epoch 110

Run Operational Inference
^^^^^^^^^^^^^^^^^^^^^^^^^

Generate forecast trajectories across all dates in the configured split without skipping dates that lack observed streamflow:

.. code-block:: bash

   run infer --run-dir /path/to/your/model_run/ --split test

Schedule Multiple Runs Across GPUs (``schedule-runs``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train, fine-tune, or evaluate a directory of YAML configs (or existing run directories) concurrently across multiple GPUs:

.. code-block:: bash

   schedule-runs train \
     --directory /path/to/config_folder/ \
     --gpu-ids 0 1 2 3 \
     --runs-per-gpu 2
