Example Configuration Files
===========================

The ``model/example-configs/`` directory contains ready-to-use YAML configuration files. You can use these files to reproduce former operational Google FloodHub model generations, run standard CAMELS-US benchmarks, assimilate recent stream-gauge observations, or run hot-start inference.

Overview
~~~~~~~~

The example configurations fall into three groups:

* **Former Operational FloodHub Generations:** Configurations that reproduce the model architectures and training settings used in previous releases of Google FloodHub (the 2024 state-handoff generation and the 2025 multi-product mean-embedding generation).
* **CAMELS-US Benchmarks & Data Assimilation:** Smaller, fast-running configurations on the 531 CAMELS-US watersheds for testing model changes and evaluating stream-gauge data assimilation.
* **Hot-Start Inference Template:** A template showing how to run fast operational inference by loading saved LSTM hidden and cell states (``seq_length: 0``, ``batch_size: 1``) instead of re-processing a 365-day historical sequence.

Configuration Files
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 38 22 40
   :header-rows: 1

   * - File Name
     - Model Architecture
     - Purpose

   * - ``floodhub-settings-config.yml``
     - :mod:`~model.modelzoo.mean_embedding_forecast_lstm`
     - Reproduces the former (2025 generation) Google FloodHub operational model configuration (`Gauch et al., 2025 <https://hess.copernicus.org/articles/29/6221/2025/>`__).

   * - ``handoff-forecast-lstm-config.yml``
     - :mod:`~model.modelzoo.handoff_forecast_lstm`
     - Reproduces the former (2024 generation) Google FloodHub operational model configuration (`Nearing et al., 2024 <https://www.nature.com/articles/s41586-024-07145-1>`__).

   * - ``camels-multimet-mean-embedding-forecast-lstm-config.yml``
     - :mod:`~model.modelzoo.mean_embedding_forecast_lstm`
     - Standard 531-basin CAMELS-US benchmark configuration for the mean-embedding architecture.

   * - ``camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml``
     - :mod:`~model.modelzoo.mean_embedding_forecast_lstm`
     - CAMELS-US configuration with gradient-based data assimilation (``assimilation_config``) to adjust latent embeddings using recent streamflow observations.

   * - ``camels-multimet-handoff-forecast-lstm-config.yml``
     - :mod:`~model.modelzoo.handoff_forecast_lstm`
     - Standard 531-basin CAMELS-US benchmark configuration for the state-handoff architecture.

   * - ``template_hot_start_config.yml``
     - :mod:`~model.modelzoo.handoff_forecast_lstm`
     - Template for hot-start evaluation and inference (``seq_length: 0``, ``batch_size: 1``, ``hot_start_path``) using pre-computed LSTM states.

Former Operational FloodHub Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2025 Generation: Mean-Embedding Forecast LSTM
---------------------------------------------

``floodhub-settings-config.yml`` defines the settings for the former 2025 operational generation of Google FloodHub (`Gauch et al., 2025 <https://hess.copernicus.org/articles/29/6221/2025/>`__). It uses :mod:`~model.modelzoo.mean_embedding_forecast_lstm` (``hidden_size: 512``, ``seq_length: 365``, ``lead_time: 7``, ``head: cmal``) and combines multiple weather products (ECMWF HRES, GraphCast, NASA IMERG, NOAA CPC, and ERA5-Land) by averaging their per-product embeddings at each time step. Pre-trained checkpoints matching this configuration (along with AIFS and no-GraphCast variants) are provided in ``model/pretrained-models/``.

2024 Generation: State-Handoff Forecast LSTM
--------------------------------------------

``handoff-forecast-lstm-config.yml`` defines the settings for the former 2024 operational generation of Google FloodHub (`Nearing et al., 2024 <https://www.nature.com/articles/s41586-024-07145-1>`__). It uses :mod:`~model.modelzoo.handoff_forecast_lstm` (``hidden_size: 256``, ``seq_length: 365``, ``lead_time: 7``, ``head: cmal``), which runs a hindcast LSTM over the past 365 days and passes its final hidden and cell states through a nonlinear handoff network to initialize the 7-day forecast LSTM.

Benchmarking, Data Assimilation, and Hot-Start Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **CAMELS-US Benchmarks (``camels-multimet-*.yml``):** These configurations train and evaluate on the 531 CAMELS-US basins (``camels-us-531-basins-list.txt``). Use them when developing new features or verifying that code changes preserve forecast accuracy.
* **Streamflow Data Assimilation (``camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml``):** Adds an ``assimilation_config`` section that optimizes the static, hindcast, and forecast embeddings over a 90-day window of observed streamflow immediately before the forecast issue date (``run evaluate --run-dir <run_dir> --assimilate``).
* **Hot-Start Inference (``template_hot_start_config.yml``):** Demonstrates how to resume from saved per-basin LSTM states (``save_state: True`` during a previous run, followed by ``hot_start_path: /path/to/hot_start_states`` and ``seq_length: 0``).
