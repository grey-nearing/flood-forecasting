# Google FloodHub: Pre-Trained OpenHydroNet Weights

This directory (`model/pretrained-models/`) contains a pre-trained global model run (`google-floodhub-settings-110-epochs`) using the Google FloodHub architecture (`mean_embedding_forecast_lstm` in `model`). Use these weights to warm-start fine-tuning on local watersheds or to run spatial generalization (Prediction in Ungauged Basins) experiments.

> **IMPORTANT — Methodological Caveat (Read Before Using):**
>
> **This model was trained on the full historical period (`01/01/1982`–`30/09/2023`) without a temporal holdout split.**
>
> Because the model saw the entire `01/01/1982`–`30/09/2023` timeline during training, **do not evaluate temporal forecasting skill on the 1982–2023 period for basins in the training list.** Evaluating on in-sample dates and basins causes data leakage and produces artificially inflated metrics. See **Appropriate & Inappropriate Use Cases** below.

---

## 1. Model Overview

We provide a pre-trained global baseline model trained on the Caravan-MultiMet dataset (excluding the CHIRPS precipitation product):

### Full Basin Baseline (`google-floodhub-settings-110-epochs`)

- **Model Architecture:** `mean_embedding_forecast_lstm` (`model.modelzoo.mean_embedding_forecast_lstm.MeanEmbeddingForecastLSTM`)
- **Reference Configuration:** [`model/example-configs/floodhub-settings-config.yml`](../example-configs/floodhub-settings-config.yml)
- **Training Period:** `01/01/1982` to `30/09/2023`
- **Training Basin List:** `15,955` listed basins (`10,137` evaluable basins with streamflow observations) in [`model/example-configs/multimet-basins-list-without-chirps.txt`](../example-configs/multimet-basins-list-without-chirps.txt)

---

## 2. Contents of the Release

The runtime directory `model/pretrained-models/google-floodhub-settings-110-epochs/` includes all files required by the `openhydronet` (`model`) package:

- **Model Weights (`model_epoch110.pt`):** Trained neural network parameters saved at epoch 110.
- **Pre-Computed Scalers (`scaler.zarr/`):** Feature and target normalization statistics (`center`, `scale`, `mean`, `std`) computed across the global training dataset. When you fine-tune this model, `model` automatically loads `scaler.zarr` so your local inputs are normalized identically to the pre-trained features.
- **Run Configuration (`config.yml`):** Exact hyperparameters, dynamic input products, and static catchment attributes used for the run.
- **In-Sample Evaluation Metrics (`test/model_epoch110/test_metrics.csv`):** Full-dataset (`10,137`-basin) in-sample metrics for verification.

---

## 3. Appropriate & Inappropriate Use Cases

### Inappropriate Uses (Do Not Do This)

- **Historical Benchmarking on Training Basins:** Running evaluation on the `01/01/1982`–`30/09/2023` training period for basins included in `multimet-basins-list-without-chirps.txt` and reporting NSE, KGE, or other skill scores.
- **Unvalidated Operational Deployment:** Using these weights directly for live flood forecasting without local validation and fine-tuning.

### Appropriate Uses (Recommended)

- **Fine-Tuning (Transfer Learning):** Initializing a model from `google-floodhub-settings-110-epochs` and fine-tuning on a local dataset with a dedicated temporal validation and test split.
- **Spatial Generalization (Prediction in Ungauged Basins):** Evaluating the model on basins that were completely excluded from `multimet-basins-list-without-chirps.txt`.
- **Forward Inference After Training Cutoff:** Running inference on new observations and forecasts strictly after the training cutoff (`30/09/2023`).

---

## 4. How to Use for Fine-Tuning

You can warm-start a fine-tuning run from `google-floodhub-settings-110-epochs` by setting `base_run_dir` in a fine-tuning YAML configuration file. A step-by-step walkthrough is also provided in [`model/tutorial/OpenHydroNet_Tutorial.ipynb`](../tutorial/OpenHydroNet_Tutorial.ipynb) and [`model/tutorial/configs/finetune-config.yml`](../tutorial/configs/finetune-config.yml).

### Example Fine-Tuning Configuration

Create a configuration file for your local basins (for example, `finetune_config.yml`) and point `base_run_dir` to `model/pretrained-models/google-floodhub-settings-110-epochs`:

```yaml
# Example fine-tuning configuration (update paths and dates for your dataset)

# --- Experiment & Base Model Setup ---
base_run_dir: model/pretrained-models/google-floodhub-settings-110-epochs
checkpoint_path: model/pretrained-models/google-floodhub-settings-110-epochs/model_epoch110.pt
experiment_name: finetune-local-basins
run_dir: ./runs

# --- Fine-Tuning Module Selection ---
# Valid module_parts for mean_embedding_forecast_lstm:
#   static_embedding_fc, hindcast_embeddings_fc, forecast_embeddings_fc,
#   shared_embeddings_fc, hindcast_lstm, forecast_lstm, head
finetune_modules:
  - static_embedding_fc
  - head

# --- Training Hyperparameters ---
epochs: 30
batch_size: 256
initial_learning_rate: 0.0005
learning_rate_strategy: StepLR
learning_rate_drop_factor: 0.9
learning_rate_epochs_drop: 5

# --- Dataset Paths & Temporal Splits ---
targets_data_dir: /path/to/your/Caravan-zarr
statics_data_dir: /path/to/your/Caravan-zarr
dynamics_data_dir: gs://caravan-multimet/v1.1

train_basin_file: /path/to/your/local_finetune_basins.txt
train_start_date: 01/01/1990
train_end_date: 31/12/2015

validation_basin_file: /path/to/your/local_finetune_basins.txt
validation_start_date: 01/01/2016
validation_end_date: 31/12/2019

test_basin_file: /path/to/your/local_finetune_basins.txt
test_start_date: 01/01/2020
test_end_date: 31/12/2023
```

Run fine-tuning with the `run` CLI (after activating `conda activate openhydronet`):

```bash
run finetune --config-file finetune_config.yml
```

### Fine-Tuning with Additional Local Weather Products (Local QPE / QPF)

You can also add local historical weather observations (such as rain gauge or radar QPE) and local weather forecasts (QPF) alongside the global Caravan-MultiMet inputs:

1. Store each local product in `<local_dynamics_dir>/<PRODUCT>/timeseries.zarr` (using the same `(basin, date)` or `(basin, date, lead_time)` Zarr layout as MultiMet).
2. Pass a list of directories to `dynamics_data_dir` so the loader reads global products from `gs://caravan-multimet/v1.1` and local products from your local directory.
3. Add the new products to `hindcast_inputs` and/or `forecast_inputs`, and use the dictionary syntax in `finetune_modules` to train the new product embeddings alongside `static_embedding_fc` and `head`:

```yaml
dynamics_data_dir:
  - gs://caravan-multimet/v1.1
  - /path/to/your/local_dynamics_zarr

hindcast_inputs:
  era5_land:
    - era5land_total_precipitation
    - era5land_temperature_2m
  cpc:
    - cpc_precipitation
  imerg:
    - imerg_precipitation
  hres:
    - hres_total_precipitation
    - hres_temperature_2m
  daymet:
    - daymet_prcp
    - daymet_tmax
    - daymet_tmin

forecast_inputs:
  hres:
    - hres_total_precipitation
    - hres_temperature_2m
  gefs_reforecast:
    - gefs_reforecast_apcp_sfc
    - gefs_reforecast_tmp_2m

finetune_modules:
  hindcast_embeddings_fc:
    - daymet
  forecast_embeddings_fc:
    - gefs_reforecast
  static_embedding_fc: true
  head: true
```

### Note on Data Scaling

When fine-tuning with `base_run_dir`, the `model` package loads `scaler.zarr` from `base_run_dir` so all pre-trained global dynamic features, static attributes, and target variables use their exact pre-trained normalization statistics. If you add new local dynamic variables in `hindcast_inputs` or `forecast_inputs`, `model` automatically computes `center` and `scale` statistics for the new variables over your fine-tuning training split and saves the combined scaler to `<run_dir>/scaler.zarr`.
