===========
Quick Start
===========

This guide shows how to install and run the **OpenHydroNet Flood Forecasting** package (`google-research/flood-forecasting`).

---------------
Obtain the Code
---------------

Clone or download the repository to access the Conda environment files, example configurations (`model/example-configs/`), pre-trained checkpoints (`model/pretrained-models/`), and tutorial (`model/tutorial/`).

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

Download or Use Sample Caravan Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A 5-basin sample dataset is included in ``model/tutorial/Caravan-nc``. This sample is ready to use with ``model/tutorial/configs/train-config.yml`` and ``model/tutorial/OpenHydroNet_Tutorial.ipynb``.

To run experiments on the full global Caravan dataset:

1. Visit the `Caravan Zenodo repository <https://doi.org/10.5281/zenodo.6522634>`_.
2. Download the **NetCDF version** of the dataset (``Caravan-nc.tar.gz``).

   .. note::
      Avoid loading raw CSV files during large-scale training because CSV parsing is much slower than NetCDF or Zarr.

3. Unpack the archive into a local directory:

.. code-block:: bash

   mkdir -p ~/data/
   tar -xvzf Caravan-nc.tar.gz -C ~/data/

4. (Optional) Convert Caravan NetCDF/CSV directories to Zarr stores (``attributes.zarr`` and ``streamflow.zarr``) for faster loading:

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

- **Tutorial Configs:** ``model/tutorial/configs/train-config.yml`` and ``model/tutorial/configs/finetune-config.yml``
- **Production & Benchmark Example Configs:** ``model/example-configs/`` (including ``floodhub-settings-config.yml`` and ``handoff-forecast-lstm-config.yml``)
- **Pre-Trained Global Checkpoints:** ``model/pretrained-models/google-floodhub-settings-110-epochs/``

Understanding the Tutorial Dataset Splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Training Set (5-basin set):** ``model/tutorial/basin-lists/5-basin-train.txt`` — used to optimize model weights.
- **Test Set (8-basin set):** ``model/tutorial/basin-lists/8-basin-test.txt`` — includes the 5 training basins plus 3 held-out basins to test spatial generalization.

Understanding Time Periods
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Training Period:** ``01/01/2000`` to ``31/12/2020``
- **Validation/Test Period:** ``01/01/2022`` to ``31/12/2024``

In full research and production experiments, keep the validation and test periods separate to avoid data leakage.

Local Path Requirements
^^^^^^^^^^^^^^^^^^^^^^^

Update these arguments in ``model/tutorial/configs/train-config.yml`` to match your system paths:

=====================  ============================================================================================================
Argument               Description
=====================  ============================================================================================================
**run_dir**            Directory where weights, logs, and config copies are saved (e.g., ``model/tutorial/model-runs/``).

**train_basin_file**   Path to a plain-text file listing basin IDs (e.g., ``model/tutorial/basin-lists/5-basin-train.txt``).

**targets_data_dir**   Path to the tutorial sample (``model/tutorial/Caravan-nc``) or your unpacked Caravan dataset directory.

**statics_data_dir**   Path to the tutorial sample (``model/tutorial/Caravan-nc``) or your unpacked Caravan dataset directory.

**dynamics_data_dir**  Path to the MultiMet forcing dataset (``gs://caravan-multimet/v1.1`` or a local directory).
=====================  ============================================================================================================

-----
Usage
-----

Training a Model
^^^^^^^^^^^^^^^^

.. code-block:: bash

   run train --config-file model/tutorial/configs/train-config.yml

Fine-Tuning a Pre-Trained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   run finetune --config-file model/tutorial/configs/finetune-config.yml

Evaluation
^^^^^^^^^^

To calculate performance metrics (such as NSE and KGE) on the test period:

.. code-block:: bash

   run evaluate --run-dir /path/to/your/model_run/

Inference
^^^^^^^^^

To generate predictions across all dates without skipping missing observations:

.. code-block:: bash

   run infer --run-dir /path/to/your/model_run/
