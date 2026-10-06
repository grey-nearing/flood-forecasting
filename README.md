# OpenHydroNet: Riverine Flood Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/flood-forecasting/blob/main/model/tutorial/OpenHydroNet_Tutorial.ipynb)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://openhydronet.readthedocs.io/)

OpenHydroNet (`google-research/flood-forecasting`) provides open-source implementations of the deep learning streamflow forecasting models and hydrological data pipelines that power [Google FloodHub](https://sites.research.google/floods/). Built on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology), it trains and runs sequence models that combine historical weather (*hindcasts*) and predicted future weather (*forecasts*) from multiple meteorological sources across global watersheds.

> *This is not an officially supported Google product. This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.*

## Quick Links

- **Documentation:** [openhydronet.readthedocs.io](https://openhydronet.readthedocs.io/)
- **Interactive Colab Tutorial:** [`model/tutorial/OpenHydroNet_Tutorial.ipynb`](https://colab.research.google.com/github/google-research/flood-forecasting/blob/main/model/tutorial/OpenHydroNet_Tutorial.ipynb) ([YouTube Video Walkthrough](https://www.youtube.com/watch?v=431Kr3mxidU))
- **Pre-Built Caravan-MultiMet Dataset:** `gs://caravan-multimet/v1.1` ([Shalev et al., 2026](https://arxiv.org/abs/2601.04886))
- **Pre-Trained Global FloodHub Weights:** [`model/pretrained-models/Pretrained-Models-README.md`](./model/pretrained-models/Pretrained-Models-README.md)

---

## Repository Components & Subpackages

| Component | Path | Description & Subpackage Documentation |
| :--- | :--- | :--- |
| **Core Forecasting Models** | [`model/`](./model/) | Deep learning streamflow forecasting models (`MeanEmbeddingForecastLSTM`, `HandoffForecastLSTM`), training, evaluation, inference, and pre-trained global FloodHub checkpoints ([`Pretrained-Models-README.md`](./model/pretrained-models/Pretrained-Models-README.md)). |
| **MultiMet Data Pipelines** | [`multimet/`](./multimet/) | End-to-end meteorological and static data pipelines ([`multimet/README.md`](./multimet/README.md)):<br>• **Gridded Archive Builders** ([`README.md`](./multimet/gridded_archive_builders/README.md)): Apache Beam pipelines for building global Zarr weather archives (ERA5-Land, IMERG, CPC, GraphCast, HRES).<br>• **Timeseries Extractors** ([`README.md`](./multimet/timeseries_extractors/README.md)): Area-weighted catchment timeseries extraction from historical archives, `dynamical.org` forecasts (`AIFS`, `GFS`, `GEFS`, `IFS-ENS`), and real-time operational forecasts (`multimet-realtime`).<br>• **Static Attribute Extractor** ([`multimet/static_extractor/`](./multimet/static_extractor/)): HydroATLAS/Caravan static watershed attribute extraction. |
| **Catchment Delineation** | [`multimet/catchment_delineation/`](./multimet/catchment_delineation/) | Global 90m (`3-arcsec`) flow-direction watershed polygon delineation and pour-point snapping (`delineate-catchment`) ([`README.md`](./multimet/catchment_delineation/README.md)). |
| **Return Periods** | [`return_periods/`](./return_periods/) | USGS Bulletin 17C flood frequency analysis (`MGBT` low-outlier screening and `EMA` Log-Pearson Type III fitting) ([`README.md`](./return_periods/README.md)). |

---

## Installation

Use **Conda** to install Python, PyTorch, geospatial libraries, and command-line tools. Installing the repository in editable mode registers the `openhydronet` package (`model`, `multimet`, and `return_periods` modules) along with the `run`, `delineate-catchment`, and `multimet-realtime` CLI entry points:

```bash
# 1. Create and activate the Conda environment
conda env create -f environments/conda.yml
conda activate openhydronet

# 2. Install the openhydronet package in editable mode from the repository root
pip install -e .
```

---

## Forecasting Models (`model/`)

- **Mean-Embedding-Forecast-LSTM (`mean_embedding_forecast_lstm`):** Embeds each meteorological input product separately and combines available products using a masked mean before running the hindcast (historical) and forecast (future) LSTMs ([`model/modelzoo/mean_embedding_forecast_lstm.py`](./model/modelzoo/mean_embedding_forecast_lstm.py)).
  - **Status:** Current operational model (as of December 2025) for [Google FloodHub](https://sites.research.google/floods/).
  - **Reference:** Gauch, Martin, et al. "[How to deal with missing input data](https://hess.copernicus.org/articles/29/6221/2025/)." *Hydrology and Earth System Sciences* (2025).
- **Handoff-Forecast-LSTM (`handoff_forecast_lstm`):** Runs a hindcast LSTM over historical inputs up to the forecast issue time, then passes the hidden and cell states through a nonlinear handoff network to initialize a forecast LSTM ([`model/modelzoo/handoff_forecast_lstm.py`](./model/modelzoo/handoff_forecast_lstm.py)).
  - **Status:** Former operational model for [Google FloodHub](https://sites.research.google/floods/).
  - **Reference:** Nearing, Grey, et al. "[Global prediction of extreme floods in ungauged watersheds](https://www.nature.com/articles/s41586-024-07145-1)." *Nature* (2024).

---

## Data Setup

1. **Tutorial Sample Data:** A 5-basin Caravan sample dataset is included at `model/tutorial/Caravan-nc` for running the tutorial notebook and quickstart configurations.
2. **Full Caravan Dataset:** Download the Caravan NetCDF dataset from [Zenodo](https://doi.org/10.5281/zenodo.6522634). To convert Caravan NetCDF/CSV folders into Zarr stores (`attributes.zarr` and `streamflow.zarr`), run:
   ```bash
   run convert-caravan --caravan-dir ~/data/Caravan-nc --output-dir ~/data/Caravan-zarr
   ```
3. **MultiMet Dynamic Forcings:** Stream meteorological forcings directly from Google Cloud Storage by setting `dynamics_data_dir: gs://caravan-multimet/v1.1` in your YAML configuration file (or point to a local Zarr directory).

---

## Training, Evaluation, and Inference

Experiments are configured with YAML files (see [`model/tutorial/configs/train-config.yml`](./model/tutorial/configs/train-config.yml)):

```bash
# Train a model
run train --config-file model/tutorial/configs/train-config.yml

# Fine-tune a pre-trained model
run finetune --config-file model/tutorial/configs/finetune-config.yml

# Evaluate performance metrics (NSE, KGE) on the test split
run evaluate --run-dir /path/to/your/model_run/

# Generate predictions across all dates without skipping missing observations
run infer --run-dir /path/to/your/model_run/
```

### Reference Configurations & Pre-Trained Checkpoints

- **Example Configs ([`model/example-configs/`](./model/example-configs/)):**
  - [`model/example-configs/floodhub-settings-config.yml`](./model/example-configs/floodhub-settings-config.yml): Global Caravan-MultiMet training configuration for `mean_embedding_forecast_lstm`.
  - [`model/example-configs/handoff-forecast-lstm-config.yml`](./model/example-configs/handoff-forecast-lstm-config.yml): Global Caravan-MultiMet training configuration for `handoff_forecast_lstm`.
  - [`model/example-configs/camels-multimet-mean-embedding-forecast-lstm-config.yml`](./model/example-configs/camels-multimet-mean-embedding-forecast-lstm-config.yml): CAMELS-US (531 basins) benchmark configuration for `mean_embedding_forecast_lstm`.
  - [`model/example-configs/camels-multimet-handoff-forecast-lstm-config.yml`](./model/example-configs/camels-multimet-handoff-forecast-lstm-config.yml): CAMELS-US (531 basins) benchmark configuration for `handoff_forecast_lstm`.
- **Pre-Trained Checkpoints ([`model/pretrained-models/`](./model/pretrained-models/)):**
  - `model/pretrained-models/google-floodhub-settings-110-epochs/`: Pre-trained global `mean_embedding_forecast_lstm` checkpoint (`model_epoch110.pt`), feature/target normalizer (`scaler.zarr`), and run config (`config.yml`). See [`model/pretrained-models/Pretrained-Models-README.md`](./model/pretrained-models/Pretrained-Models-README.md) for usage and fine-tuning instructions.

---

## Issue Reporting

If you encounter bugs, please open an issue on the [GitHub Issue Tracker](https://github.com/google-research/flood-forecasting/issues) with a clear description, steps to reproduce, and expected behavior.
