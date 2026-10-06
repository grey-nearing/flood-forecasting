# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""5-basin end-to-end golden numerical regression test on CPU."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model.datasetzoo.multimet import _convert_to_tensor
from model.evaluation import get_tester
from model.training.basetrainer import BaseTrainer
from model.utils.config import Config

EXPECTED_BASINS = ['basin_01', 'basin_02', 'basin_03', 'basin_04', 'basin_05']

EXPECTED_NSE = np.array(
    [
        -7.262215614318848,
        -0.2223786115646362,
        -0.3845325708389282,
        0.0677303075790405,
        -0.2977261543273926,
    ],
    dtype=np.float64,
)

EXPECTED_KGE = np.array(
    [
        -0.6679253673048586,
        -0.209148931936286,
        -0.0266082219316117,
        0.0376948033576702,
        -0.4297055936408429,
    ],
    dtype=np.float64,
)

EXPECTED_FIRST_BATCH_Y_HAT = np.array(
    [
        [-0.43689650297164917, -0.43665313720703125],
        [-0.43727242946624756, -0.4370308518409729],
        [-0.4377673864364624, -0.4355616569519043],
        [-0.4376685619354248, -0.43466830253601074],
        [-0.4357684850692749, -0.433330774307251],
    ],
    dtype=np.float32,
)


@pytest.mark.integration
def test_five_basin_golden_numerical_regression(five_basin_dataset, tmp_path):
    """Verify deterministic 5-basin training + evaluation against golden values."""
    embedding = {
        'type': 'fc',
        'hiddens': [8],
        'activation': ['tanh'],
        'dropout': 0.0,
    }
    cfg = Config(
        {
            'experiment_name': 'golden_regression',
            'run_dir': tmp_path / 'runs',
            'dataset': 'multimet',
            'data_dir': Path(five_basin_dataset),
            'train_basin_file': five_basin_dataset.train_basin_file,
            'validation_basin_file': five_basin_dataset.validation_basin_file,
            'test_basin_file': five_basin_dataset.test_basin_file,
            'train_start_date': five_basin_dataset['train_start_date'],
            'train_end_date': five_basin_dataset['train_end_date'],
            'validation_start_date': five_basin_dataset[
                'validation_start_date'
            ],
            'validation_end_date': five_basin_dataset['validation_end_date'],
            'test_start_date': five_basin_dataset['test_start_date'],
            'test_end_date': five_basin_dataset['test_end_date'],
            'model': 'mean_embedding_forecast_lstm',
            'head': 'regression',
            'output_activation': 'linear',
            'output_dropout': 0.0,
            'hidden_size': 8,
            'seq_length': 10,
            'lead_time': 2,
            'forecast_overlap': 10,
            'predict_last_n': 2,
            'timestep_counter': False,
            'static_attributes': ['area', 'p_mean'],
            'hindcast_inputs': [
                'era5land_total_precipitation',
                'graphcast_total_precipitation',
            ],
            'forecast_inputs': ['graphcast_total_precipitation'],
            'target_variables': ['streamflow'],
            'statics_embedding': embedding.copy(),
            'hindcast_embedding': embedding.copy(),
            'forecast_embedding': embedding.copy(),
            'dynamics_embedding': embedding.copy(),
            'state_handoff_network': embedding.copy(),
            'optimizer': 'Adam',
            'loss': 'MSE',
            'initial_learning_rate': 0.01,
            'batch_size': 16,
            'epochs': 2,
            'save_weights_every': 2,
            'validate_every': 0,
            'device': 'cpu',
            'seed': 42,
            'compile': False,
            'num_workers': 0,
            'verbose': 0,
            'log_interval': 1,
            'log_tensorboard': False,
            'log_n_figures': 0,
            'save_git_diff': False,
            'metrics': ['NSE', 'KGE'],
        }
    )

    trainer = BaseTrainer(cfg)
    trainer.initialize_training()
    trainer.train_and_validate()

    tester = get_tester(
        cfg, run_dir=cfg.run_dir, period='test', init_model=True
    )
    tester.evaluate(save_results=False, metrics=['NSE', 'KGE'])

    metrics_csv = cfg.run_dir / 'test' / 'model_epoch002' / 'test_metrics.csv'
    assert metrics_csv.exists()
    metrics_df = pd.read_csv(metrics_csv, index_col='basin').loc[
        EXPECTED_BASINS
    ]

    np.testing.assert_allclose(
        metrics_df['NSE'].to_numpy(dtype=np.float64),
        EXPECTED_NSE,
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        metrics_df['KGE'].to_numpy(dtype=np.float64),
        EXPECTED_KGE,
        atol=1e-4,
        rtol=1e-4,
    )

    samples = [
        {k: _convert_to_tensor(k, v) for k, v in tester.dataset[i].items()}
        for i in range(5)
    ]
    batch0 = tester.dataset.collate_fn(samples)
    with torch.no_grad():
        batch0 = tester.model.pre_model_hook(batch0, is_train=False)
        y_hat0 = (
            tester.model(batch0)['y_hat'][:, -cfg.predict_last_n :, 0]
            .detach()
            .cpu()
            .numpy()
        )

    np.testing.assert_allclose(
        y_hat0,
        EXPECTED_FIRST_BATCH_Y_HAT,
        atol=1e-4,
        rtol=1e-4,
    )
