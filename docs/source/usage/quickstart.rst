===========
Quick Start
===========

This guide shows how to install and run the **OpenHydroNet Flood Forecasting** package (`google-research/flood-forecasting`).

---------------
Obtain the Code
---------------

Clone or download the repository to access the Conda environment files, example configurations (`model/example-configs/`), and pre-trained checkpoints (`model/pretrained-models/`).

Option A: Via GitHub Cloning (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/google-research/flood-forecasting.git
   cd flood-forecasting

Option B: Via Zipball
^^^^^^^^^^^^^^^^^^^^^

If you do not use git, download and extract the source archive:

.. code-block:: bash

   curl -L https://github.com/google-research/flood-forecasting/zipball/main -o flood-forecasting.zip
   unzip flood-forecasting.zip
   cd google-research-flood-forecasting-*

----------------------------------
Prerequisites & Environment Setup
----------------------------------

We recommend using **Conda** to install Python, PyTorch, CUDA, and geospatial dependencies.

Using Conda (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

The environment specification is located in ``environments/conda.yml``:

.. code-block:: bash

   # Create the environment from the file in the repo
   conda env create -f environments/conda.yml

   # Activate the environment (MANDATORY)
   conda activate openhydronet

Manual Setup
^^^^^^^^^^^^

If you prefer not to use Conda, use **Python >= 3.12** and install the dependencies listed in ``environments/rtd_requirements.txt``.

------------
Installation
------------

With the ``openhydronet`` environment active, install the package in editable mode from the repository root:

.. code-block:: bash

   pip install -e .

----------
Data Setup
----------

OpenHydroNet uses the `Caravan <https://www.nature.com/articles/s41597-023-01975-w>`_ dataset for streamflow observations and static catchment attributes.

Download Caravan Data
^^^^^^^^^^^^^^^^^^^^^

To run experiments on the Caravan dataset:

1. Visit the `Caravan Zenodo repository <https://doi.org/10.5281/zenodo.6522634>`_.
2. Download the **NetCDF version** of the dataset (``Caravan-nc.tar.gz``).
3. Unpack the archive into a local directory:

.. code-block:: bash

   mkdir -p ~/data/
   tar -xvzf Caravan-nc.tar.gz -C ~/data/

4. Convert the unpacked Caravan directory into canonical Zarr stores (``attributes.zarr`` and ``streamflow.zarr``) required by the dataset loader:

.. code-block:: bash

   run convert-caravan --caravan-dir ~/data/Caravan-nc --output-dir ~/data/Caravan-zarr

MultiMet Dynamic Data
^^^^^^^^^^^^^^^^^^^^^

The Caravan-MultiMet weather forcing dataset is streamed directly from **Google Cloud Storage** or read from a local directory. Set ``dynamics_data_dir`` in your YAML configuration file to:

.. code-block:: yaml

   dynamics_data_dir: gs://caravan-multimet/v1.1

Static Attributes & Catchment Polygons for Custom Watersheds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- To delineate upstream watershed polygons from gauge latitude and longitude coordinates, see :doc:`Catchment Delineation <catchment_delineation>`.
- To extract Caravan-compatible static attributes from watershed polygons (``.geojson``, ``.shp``, or ``.gpkg``), see :doc:`Extracting Static Attributes for Custom Watersheds <static_extractor>`.

----------------------
Training Configuration
----------------------

Experiments are configured via YAML files:

- **Production & Benchmark Example Configs:** ``model/example-configs/`` (including ``floodhub-settings-config.yml``, ``camels-multimet-mean-embedding-forecast-lstm-config.yml``, ``camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml``, and ``camels-multimet-handoff-forecast-lstm-config.yml``)
- **Pre-Trained Global Checkpoints:** ``model/pretrained-models/google-floodhub-settings-110-epochs/``

Understanding Basin Lists and Time Periods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Basin lists:** ``train_basin_file``, ``validation_basin_file``, and ``test_basin_file`` point to plain-text files with one basin ID per line (for example, ``model/example-configs/multimet-basins-list-without-chirps.txt``). Basins listed only in the test file are evaluated as held-out (ungauged) basins.
- **Time periods:** ``train_start_date`` / ``train_end_date``, ``validation_start_date`` / ``validation_end_date``, and ``test_start_date`` / ``test_end_date`` use the ``DD/MM/YYYY`` format.

In research and production experiments, keep the validation and test periods separate to avoid data leakage.

Local Path Requirements
^^^^^^^^^^^^^^^^^^^^^^^

Copy an example configuration (for example, ``model/example-configs/floodhub-settings-config.yml``) and update these arguments to match your system paths:

=====================  ============================================================================================================
Argument               Description
=====================  ============================================================================================================
**run_dir**            Directory where weights, logs, and config copies are saved (e.g., ``./runs/``).

**train_basin_file**   Path to a plain-text file listing basin IDs (e.g., ``model/example-configs/multimet-basins-list-without-chirps.txt``).

**targets_data_dir**   Path to your converted Caravan Zarr directory (``streamflow.zarr``).

**statics_data_dir**   Path to your converted Caravan Zarr directory (``attributes.zarr``).

**dynamics_data_dir**  Path to the MultiMet forcing dataset (``gs://caravan-multimet/v1.1`` or a local directory).
=====================  ============================================================================================================

-----
Usage
-----

Training a Model
^^^^^^^^^^^^^^^^

.. code-block:: bash

   run train --config-file model/example-configs/floodhub-settings-config.yml

Fine-Tuning a Pre-Trained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``base_run_dir`` in your fine-tuning configuration to a trained run directory such as ``model/pretrained-models/google-floodhub-settings-110-epochs`` (an example configuration is given in ``model/pretrained-models/Pretrained-Models-README.md``):

.. code-block:: bash

   run finetune --config-file /path/to/your/finetune-config.yml

Evaluation
^^^^^^^^^^

To calculate performance metrics (such as NSE and KGE) on the test period (pass ``--assimilate`` to enable variational data assimilation if configured):

.. code-block:: bash

   run evaluate --run-dir /path/to/your/model_run/

Inference
^^^^^^^^^

To generate predictions across all dates without skipping missing observations:

.. code-block:: bash

   run infer --run-dir /path/to/your/model_run/
