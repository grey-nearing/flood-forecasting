# Google FloodHub: Pre-Trained OpenHydroNet Weights

This directory (`model/pretrained-models/`) contains a pre-trained global model run (`google-floodhub-settings-110-epochs`) using the Google FloodHub architecture (`mean_embedding_forecast_lstm` in `model`). Use these weights to warm-start fine-tuning on local watersheds or to run spatial generalization (Prediction in Ungauged Basins) experiments.

> **IMPORTANT — Methodological Caveat (Read Before Using):**
>
> **This model was trained on the full historical period (1982–2023) without a temporal holdout split.**
>
> Because the model saw the entire 1982–2023 timeline during training, **do not evaluate temporal forecasting skill on the 1982–2023 period for basins in the training list.** Evaluating on in-sample dates and basins causes data leakage and produces artificially inflated metrics. See **Appropriate & Inappropriate Use Cases** below.

---

## 1. Model Overview

We provide a pre-trained global baseline model trained on the Caravan-MultiMet dataset (excluding the CHIRPS precipitation product):

### Full Basin Baseline (`google-floodhub-settings-110-epochs`)

- **Model Architecture:** `mean_embedding_forecast_lstm` (`model.modelzoo.mean_embedding_forecast_lstm.MeanEmbeddingForecastLSTM`)
- **Reference Configuration:** [`model/example-configs/floodhub-settings-config.yml`](../example-configs/floodhub-settings-config.yml)
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

- **Historical Benchmarking on Training Basins:** Running evaluation on the 1982–2023 training period for basins included in `multimet-basins-list-without-chirps.txt` and reporting NSE, KGE, or other skill scores.
- **Unvalidated Operational Deployment:** Using these weights directly for live flood forecasting without local validation and fine-tuning.

### Appropriate Uses (Recommended)

- **Fine-Tuning (Transfer Learning):** Initializing a model from `google-floodhub-settings-110-epochs` and fine-tuning on a local dataset with a dedicated temporal validation and test split.
- **Spatial Generalization (Prediction in Ungauged Basins):** Evaluating the model on basins that were completely excluded from `multimet-basins-list-without-chirps.txt`.
- **Forward Inference After 2023:** Running inference on new observations and forecasts strictly after the training cutoff (`31/12/2023`).

---

## 4. How to Use for Fine-Tuning

You can warm-start a fine-tuning run from `google-floodhub-settings-110-epochs` by setting `base_run_dir` in a fine-tuning YAML configuration file. A step-by-step walkthrough is also provided in [`model/tutorial/OpenHydroNet_Tutorial.ipynb`](../tutorial/OpenHydroNet_Tutorial.ipynb) and [`model/tutorial/configs/finetune-config.yml`](../tutorial/configs/finetune-config.yml).

### Example Fine-Tuning Configuration

Create a configuration file for your local basins (for example, `finetune_config.yml`) and point `base_run_dir` to `model/pretrained-models/google-floodhub-settings-110-epochs`:

```yaml
# Example fine-tuning configuration (update paths and dates for your dataset)

# --- Fine-tuning arguments ---
base_run_dir: model/pretrained-models/google-floodhub-settings-110-epochs
checkpoint_path: model/pretrained-models/google-floodhub-settings-110-epochs/model_epoch110.pt
finetune_modules:
  - statics_embedding

epochs: 30
initial_learning_rate: 0.0001
learning_rate_strategy: ReduceLROnPlateau

# --- Dataset splits and paths ---
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

### Note on Data Scaling

When fine-tuning with `base_run_dir`, the `model` package loads `scaler.zarr` from `base_run_dir`. Your fine-tuning dataset must provide the same dynamic input variables, static attributes, and target variables expected by the pre-trained model.
