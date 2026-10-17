# OpenHydroNet: Riverine Flood Forecasting

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://openhydronet.readthedocs.io/)

OpenHydroNet (`google-research/flood-forecasting`) is an open-source Python library and web application for predicting daily river flow (*streamflow*) and floods around the world.

Built on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology), OpenHydroNet trains and runs deep learning models that combine:
- **Past weather (*hindcasts*):** Recent rain, snow, temperature, solar radiation, and air pressure from satellites, rain gauges, and weather models.
- **Future weather (*forecasts*):** Predicted weather for the next 7 to 10 days.
- **Static basin attributes:** Unchanging facts about each river basin (*watershed* or *catchment*), such as its drainage area ($\text{km}^2$), elevation, slope, soil type, and land cover.

The deep learning architectures in this repository (`MeanEmbeddingForecastLSTM` and `HandoffForecastLSTM`) are **former operational models** that powered earlier generations of [Google FloodHub](https://sites.research.google/floods/). A newer model release has since replaced them in production.

> *This is not an officially supported Google product. This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.*

---

## Quick Links

- **Step-by-Step Tutorial (Setup, Backend APIs/CLIs & Web UI):** [`TUTORIAL.md`](./TUTORIAL.md)
- **Full Documentation (ReadTheDocs):** [openhydronet.readthedocs.io](https://openhydronet.readthedocs.io/)
- **Interactive Web Interface Guide:** [`frontend/README.md`](./frontend/README.md) (`openhydronet-ui`)
- **Pre-Built Caravan-MultiMet Dataset:** `gs://caravan-multimet/v1.1` ([Shalev et al., 2026](https://arxiv.org/abs/2601.04886))
- **Pre-Trained Global Model Checkpoints:** [`model/pretrained-models/Pretrained-Models-README.md`](./model/pretrained-models/Pretrained-Models-README.md)

---

## What Is Inside This Repository

| Package | Path | What It Does | Detailed Guide |
| :--- | :--- | :--- | :--- |
| **Forecasting Models** | [`model/`](./model/) | Trains, fine-tunes, evaluates, and runs deep learning river flow models (`MeanEmbeddingForecastLSTM` and `HandoffForecastLSTM`), including variational data assimilation and pre-trained global checkpoints. | [`Pretrained-Models-README.md`](./model/pretrained-models/Pretrained-Models-README.md) |
| **Watershed Delineation** | [`multimet/catchment_delineation/`](./multimet/catchment_delineation/) | Traces upstream river basin boundaries (`Polygon` / `MultiPolygon`) and calculates drainage area ($\text{km}^2$) from 90-meter (`3-arc-second`) flow-direction maps (`delineate-catchment`). | [`multimet/catchment_delineation/README.md`](./multimet/catchment_delineation/README.md) |
| **Static Basin Attributes** | [`multimet/static_extractor/`](./multimet/static_extractor/) | Computes Caravan-compatible tables of unchanging basin facts (elevation, slope, soil, land cover, and 40-year climate averages) from HydroATLAS and ERA5-Land (`extract-static-attributes`). | [`multimet/README.md`](./multimet/README.md#part-2-extracting-static-attributes-for-watersheds-multimetstatic_extractor) |
| **Gridded Weather Archives** | [`multimet/gridded_archive_builders/`](./multimet/gridded_archive_builders/) | Downloads daily global rain grids from NOAA CPC (`0.5°`) and NASA GPM IMERG (`0.1°`) and saves them as Zarr archives (`build-cpc-archive`, `build-imerg-archive`). | [`multimet/gridded_archive_builders/README.md`](./multimet/gridded_archive_builders/README.md) |
| **Basin Weather Time Series** | [`multimet/timeseries_extractors/`](./multimet/timeseries_extractors/) | Calculates area-weighted daily weather time series over river basin polygons from historical archives (`extract-multimet`) and live weather feeds (`multimet-realtime`). | [`multimet/timeseries_extractors/README.md`](./multimet/timeseries_extractors/README.md) |
| **Live Weather Forecast Grids** | [`multimet/weather_fetcher/`](./multimet/weather_fetcher/) | Downloads the newest 10-day gridded weather forecasts from ECMWF (`HRES`, `IFS ENS`, `AIFS`) and NOAA (`GFS`, `GEFS`, `HRRR`) onto a `0.25°` global grid (`sync-weather-forecasts`). | [`multimet/weather_fetcher/README.md`](./multimet/weather_fetcher/README.md) |
| **External Flood Forecasts (MaaS)** | [`maas/`](./maas/) | Fetches and compares live river flow forecasts and flood return-period thresholds across Google FloodHub, Copernicus GloFAS v4, and GEOGLOWS ECMWF v2 (`fetch-maas-forecast`). | [`maas/README.md`](./maas/README.md) |
| **Flood Return Periods** | [`return_periods/`](./return_periods/) | Calculates flood frequency return periods (such as the 2-year, 5-year, 20-year, and 100-year flood flows) using the official USGS Bulletin 17C method (`MGBT` + `EMA`). | [`return_periods/README.md`](./return_periods/README.md) |
| **Interactive Web UI** | [`frontend/`](./frontend/) | Local web server and browser interface (`openhydronet-ui`) for viewing weather maps, comparing global flood models, delineating basins, extracting attributes, fine-tuning models, and running live forecasts. | [`frontend/README.md`](./frontend/README.md) |
| **Verification Benchmarks** | [`benchmarks/`](./benchmarks/) | Standalone test suite (`benchmark-*`) that checks accuracy across all data pipelines, return period calculators, and forecasting models against reference datasets. | [`benchmarks/README.md`](./benchmarks/README.md) |

---

## Installation

We recommend using **Conda** to install Python ($\ge 3.12$), PyTorch, and map/weather libraries. Running `pip install -e .` installs the `openhydronet` package and all command-line tools:

```bash
# 1. Clone the repository
git clone https://github.com/google-research/flood-forecasting.git
cd flood-forecasting

# 2. Create and activate the Conda environment
conda env create -f environments/conda.yml
conda activate openhydronet

# 3. Install OpenHydroNet in editable mode
pip install -e .
```

For full setup details (including NASA Earthdata login and Google Cloud Storage access), see **Section 1** of [`TUTORIAL.md`](./TUTORIAL.md#section-1-setup-and-installation).

---

## Two Ways to Use OpenHydroNet

### 1. Run the Interactive Web Interface (`openhydronet-ui`)

Start the local web server and open `http://127.0.0.1:8080` in your web browser:

```bash
openhydronet-ui --host 127.0.0.1 --port 8080
```

From the browser, you can view global weather forecasts, compare external flood models, click a map to trace river basin boundaries, extract static attributes, fine-tune pre-trained models on local gauge CSV files, and run Cold-Start or Hot-Start river forecasts with data assimilation. See **Section 3** of [`TUTORIAL.md`](./TUTORIAL.md#section-3-using-the-interactive-web-ui) for a complete walkthrough of every tab.

### 2. Run from the Command Line or Python (`run` and Backend CLIs)

#### Prepare Caravan Streamflow & Static Data

1. Download the Caravan NetCDF archive (`Caravan-nc.tar.gz`) from [Zenodo](https://doi.org/10.5281/zenodo.6522634) and unpack it.
2. Convert the Caravan NetCDF/CSV folders into fast Zarr stores (`attributes.zarr` and `streamflow.zarr`):
   ```bash
   run convert-caravan --caravan-dir ~/data/Caravan-nc --output-dir ~/data/Caravan-zarr
   ```
3. Stream MultiMet weather forcings directly from Google Cloud Storage by setting `dynamics_data_dir: gs://caravan-multimet/v1.1` in your YAML configuration file (or point to a local folder).

#### Train, Fine-Tune, Evaluate, and Run Forecasts

```bash
# Train a model from scratch using a YAML configuration file
run train --config-file model/example-configs/floodhub-settings-config.yml

# Continue an interrupted training run
run continue_training --run-dir /path/to/your/model_run/

# Fine-tune a pre-trained global model on local basins
run finetune --config-file /path/to/your/finetune-config.yml

# Evaluate accuracy metrics (NSE, KGE) on the test period
# (Add --assimilate to turn on variational data assimilation)
run evaluate --run-dir /path/to/your/model_run/

# Generate predictions for all dates (including days without gauge observations)
run infer --run-dir /path/to/your/model_run/
```

See **Section 2** of [`TUTORIAL.md`](./TUTORIAL.md#section-2-using-the-backend-cli-commands-and-python-apis) for examples of every command-line tool and Python API.

---

## Forecasting Models & Pre-Trained Checkpoints (`model/`)

### Model Architectures

- **Mean-Embedding-Forecast-LSTM (`mean_embedding_forecast_lstm`):** Embeds each weather input product separately and averages the available embeddings at each day before running the past (*hindcast*) and future (*forecast*) LSTM networks ([`model/modelzoo/mean_embedding_forecast_lstm.py`](./model/modelzoo/mean_embedding_forecast_lstm.py)). Because it averages available product embeddings, it continues to run even when one weather source is missing on a given day.
  - **FloodHub History:** Former operational model for [Google FloodHub](https://sites.research.google/floods/) (2025 generation; superseded by a newer model release).
  - **Paper:** Gauch, Martin, et al. "[How to deal with missing input data](https://hess.copernicus.org/articles/29/6221/2025/)." *Hydrology and Earth System Sciences* (2025).
- **Handoff-Forecast-LSTM (`handoff_forecast_lstm`):** Runs a past (*hindcast*) LSTM up to the forecast issue date, then passes its memory state through a small neural network (*handoff network*) to start a separate future (*forecast*) LSTM ([`model/modelzoo/handoff_forecast_lstm.py`](./model/modelzoo/handoff_forecast_lstm.py)).
  - **FloodHub History:** Former operational model for [Google FloodHub](https://sites.research.google/floods/) (2024 generation; superseded by later model releases).
  - **Paper:** Nearing, Grey, et al. "[Global prediction of extreme floods in ungauged watersheds](https://www.nature.com/articles/s41586-024-07145-1)." *Nature* (2024).

### Example Configurations & Pre-Trained Weights

- **Example YAML Configurations ([`model/example-configs/`](./model/example-configs/)):**
  - [`floodhub-settings-config.yml`](./model/example-configs/floodhub-settings-config.yml): Global Caravan-MultiMet training setup for `mean_embedding_forecast_lstm` (2025 FloodHub generation settings).
  - [`handoff-forecast-lstm-config.yml`](./model/example-configs/handoff-forecast-lstm-config.yml): Global Caravan-MultiMet training setup for `handoff_forecast_lstm` (2024 FloodHub generation settings).
  - [`template_hot_start_config.yml`](./model/example-configs/template_hot_start_config.yml): Template configuration for saving and loading LSTM states (`save_state` / `hot_start_path`) for fast daily forecasts.
  - [`camels-multimet-mean-embedding-forecast-lstm-config.yml`](./model/example-configs/camels-multimet-mean-embedding-forecast-lstm-config.yml): 531-basin CAMELS-US benchmark setup for `mean_embedding_forecast_lstm`.
  - [`camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml`](./model/example-configs/camels-multimet-mean-embedding-forecast-lstm-assimilation-config.yml): 531-basin CAMELS-US setup with variational data assimilation enabled.
  - [`camels-multimet-handoff-forecast-lstm-config.yml`](./model/example-configs/camels-multimet-handoff-forecast-lstm-config.yml): 531-basin CAMELS-US benchmark setup for `handoff_forecast_lstm`.
- **Pre-Trained Global Checkpoints ([`model/pretrained-models/`](./model/pretrained-models/)):**
  - `google-floodhub-settings-110-epochs/`: Global `mean_embedding_forecast_lstm` trained with `HRES`, `GraphCast`, `IMERG`, and `CPC` (`model_epoch110.pt`, `scaler.zarr`, `config.yml`).
  - `google-floodhub-settings-no-graphcast-110-epochs/`: Global `mean_embedding_forecast_lstm` trained with `HRES`, `IMERG`, and `CPC` (no GraphCast required).
  - `google-floodhub-settings-aifs-110-epochs/`: Global `mean_embedding_forecast_lstm` trained with `HRES`, `ECMWF AIFS`, `IMERG`, and `CPC`.
  - See [`model/pretrained-models/Pretrained-Models-README.md`](./model/pretrained-models/Pretrained-Models-README.md) for usage and fine-tuning instructions.

---

## Issue Reporting

If you find a bug, please open an issue on the [GitHub Issue Tracker](https://github.com/google-research/flood-forecasting/issues) with the steps to reproduce it and the expected result.
