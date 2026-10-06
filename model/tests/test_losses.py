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

"""Unit tests for model.training.loss and regularization."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from model.training.loss import (
    MaskedCMALLoss,
    MaskedMSELoss,
    MaskedNSELoss,
    MaskedRMSELoss,
)
from model.training.regularization import BaseRegularization


class DummyRegularization(BaseRegularization):
    def __init__(self, cfg=None, weight: float = 0.5):
        super().__init__(cfg=cfg, name='dummy_reg', weight=weight)

    def forward(self, prediction, ground_truth, model_parameters):
        return torch.tensor(2.0)


@pytest.fixture
def dummy_config():
    cfg = MagicMock()
    cfg.predict_last_n = 10
    cfg.target_variables = ['streamflow']
    cfg.target_loss_weights = None
    cfg.n_distributions = 3
    return cfg


@pytest.mark.unit
def test_base_loss_target_weights(dummy_config):
    # Single target default equal weights
    loss = MaskedMSELoss(dummy_config)
    assert torch.allclose(loss._target_weights, torch.tensor([1.0]))

    # Multi-target default equal weights
    dummy_config.target_variables = ['var1', 'var2']
    dummy_config.target_loss_weights = None
    loss = MaskedMSELoss(dummy_config)
    assert torch.allclose(loss._target_weights, torch.tensor([0.5, 0.5]))

    # Multi-target custom weights
    dummy_config.target_loss_weights = [0.8, 0.2]
    loss = MaskedMSELoss(dummy_config)
    assert torch.allclose(loss._target_weights, torch.tensor([0.8, 0.2]))

    # Weight length mismatch error
    dummy_config.target_loss_weights = [0.5]
    with pytest.raises(
        ValueError,
        match='Number of weights must be equal to the number of target',
    ):
        MaskedMSELoss(dummy_config)


@pytest.mark.unit
def test_masked_mse_loss(dummy_config):
    loss_fn = MaskedMSELoss(dummy_config)

    # Batch of 2, sequence of 10, 1 target
    y_hat = torch.tensor([[[2.0], [4.0]], [[6.0], [8.0]]])  # shape [2, 2, 1]
    y = torch.tensor([[[1.0], [np.nan]], [[4.0], [10.0]]])   # shape [2, 2, 1]

    dummy_config.predict_last_n = 2
    loss_fn = MaskedMSELoss(dummy_config)

    prediction = {'y_hat': y_hat}
    data = {'y': y}

    total_loss, all_losses = loss_fn(prediction, data)
    # Valid differences:
    # (2-1)=1 -> 1^2=1; (6-4)=2 -> 2^2=4; (8-10)=-2 -> (-2)^2=4
    # Mean of squared errors = (1 + 4 + 4) / 3 = 3.0
    # MaskedMSE multiplies by 0.5 -> 1.5
    assert np.isclose(total_loss.item(), 1.5)
    assert 'loss' in all_losses
    assert 'total_loss' in all_losses


@pytest.mark.unit
def test_masked_rmse_loss(dummy_config):
    dummy_config.predict_last_n = 2
    loss_fn = MaskedRMSELoss(dummy_config)

    y_hat = torch.tensor([[[2.0], [4.0]]])  # shape [1, 2, 1]
    y = torch.tensor([[[1.0], [np.nan]]])    # shape [1, 2, 1]

    prediction = {'y_hat': y_hat}
    data = {'y': y}

    total_loss, _ = loss_fn(prediction, data)
    # (2-1)^2 = 1.0 * 0.5 = 0.5 -> sqrt(0.5)
    expected = np.sqrt(0.5)
    assert np.isclose(total_loss.item(), expected)


@pytest.mark.unit
def test_masked_nse_loss(dummy_config):
    dummy_config.predict_last_n = 2
    loss_fn = MaskedNSELoss(dummy_config, eps=0.1)

    y_hat = torch.tensor([[[2.0], [4.0]]])  # shape [1, 2, 1]
    y = torch.tensor([[[1.0], [3.0]]])      # shape [1, 2, 1]
    per_basin_target_stds = torch.tensor([[[1.0]]])  # shape [1, 1, 1]

    prediction = {'y_hat': y_hat}
    data = {'y': y, 'per_basin_target_stds': per_basin_target_stds}

    total_loss, _ = loss_fn(prediction, data)
    # Squared errors: (2-1)^2 = 1.0, (4-3)^2 = 1.0 -> Mean = 1.0
    # Weights = 1 / (1.0 + 0.1)^2 = 1 / 1.21 = 0.826446
    # Scaled loss = 1.0 * 0.826446 = 0.826446
    expected = 1.0 / (1.1 ** 2)
    assert np.isclose(total_loss.item(), expected, atol=1e-5)


@pytest.mark.unit
def test_masked_cmal_loss(dummy_config):
    dummy_config.predict_last_n = 2
    dummy_config.n_distributions = 3
    loss_fn = MaskedCMALLoss(dummy_config)

    batch_size = 2
    seq_len = 2
    n_dist = 3

    # Create synthetic CMAL predictions
    mu = torch.zeros(batch_size, seq_len, n_dist)
    b = torch.ones(batch_size, seq_len, n_dist)
    tau = torch.full((batch_size, seq_len, n_dist), 0.5)
    pi = torch.full((batch_size, seq_len, n_dist), 1.0 / n_dist)

    y = torch.zeros(batch_size, seq_len, 1)

    prediction = {'mu': mu, 'b': b, 'tau': tau, 'pi': pi}
    data = {'y': y}

    total_loss, all_losses = loss_fn(prediction, data)
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0


@pytest.mark.unit
def test_loss_with_regularization(dummy_config):
    dummy_config.predict_last_n = 2
    loss_fn = MaskedMSELoss(dummy_config)

    reg = DummyRegularization(weight=0.5)
    loss_fn.set_regularization_terms([reg])

    y_hat = torch.tensor([[[2.0], [2.0]]])
    y = torch.tensor([[[2.0], [2.0]]])

    prediction = {'y_hat': y_hat}
    data = {'y': y}

    total_loss, all_losses = loss_fn(prediction, data)
    # MSE loss = 0.0, reg = 0.5 * 2.0 = 1.0
    assert np.isclose(total_loss.item(), 1.0)
    assert 'dummy_reg' in all_losses
    assert all_losses['dummy_reg'].item() == 2.0


@pytest.mark.unit
def test_masked_rmse_loss_zero_residual_finite_gradients(dummy_config):
    dummy_config.predict_last_n = 2
    loss_fn = MaskedRMSELoss(dummy_config)

    # Exact zero residual with a linear layer
    linear = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(linear.weight)
    x = torch.ones(1, 2, 2)
    y_hat = linear(x)
    y = torch.tensor([[[0.0], [np.nan]]])

    total_loss, _ = loss_fn({'y_hat': y_hat}, {'y': y})
    assert total_loss.item() == 0.0
    total_loss.backward()
    assert linear.weight.grad is not None
    assert torch.all(torch.isfinite(linear.weight.grad))
    assert torch.all(linear.weight.grad == 0.0)

    # Non-zero residuals match sqrt(0.5 * mean(diff ** 2)) to machine precision
    y_hat_nz = torch.tensor(
        [[[1.5], [-2.25]], [[0.75], [3.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    y_nz = torch.tensor(
        [[[0.5], [np.nan]], [[-0.25], [1.1]]],
        dtype=torch.float64,
    )
    loss_nz, _ = loss_fn({'y_hat': y_hat_nz}, {'y': y_nz})
    mask = ~torch.isnan(y_nz)
    diff = y_hat_nz[mask] - y_nz[mask]
    expected_nz = torch.sqrt(0.5 * torch.mean(diff ** 2))
    assert torch.allclose(loss_nz, expected_nz, rtol=1e-15, atol=1e-15)

    # All-NaN ground truth returns differentiable zero
    y_all_nan = torch.tensor([[[np.nan], [np.nan]]])
    loss_nan, _ = loss_fn(
        {'y_hat': torch.zeros(1, 2, 1, requires_grad=True)},
        {'y': y_all_nan},
    )
    assert loss_nan.requires_grad
    assert loss_nan.item() == 0.0


@pytest.mark.unit
def test_cmal_and_masked_cmal_loss_float16_extreme_logits_finite(dummy_config):
    from model.modelzoo.head import CMAL

    dummy_config.predict_last_n = 2
    dummy_config.n_distributions = 2
    loss_fn = MaskedCMALLoss(dummy_config)

    head = CMAL(n_in=4, n_out=4 * dummy_config.n_distributions, n_hidden=8)
    b_slice = slice(dummy_config.n_distributions, 2 * dummy_config.n_distributions)
    t_slice = slice(2 * dummy_config.n_distributions, 3 * dummy_config.n_distributions)
    with torch.no_grad():
        head.fc1.weight.fill_(0.1)
        head.fc1.bias.fill_(0.0)
        head.fc2.weight.fill_(0.0)
        # At t_latent = 12.0, float16 sigmoid saturates to 1.0, whereas float32 stays < 1.0
        head.fc2.bias[b_slice] = -20.0
        head.fc2.bias[t_slice] = 12.0

    x_12 = torch.ones(2, 2, 4)
    with torch.amp.autocast('cpu', dtype=torch.float16):
        pred_12 = head(x_12)
        assert torch.all((pred_12['tau'] > 0.0) & (pred_12['tau'] < 1.0))

    # Even at extreme t_latent = 20.0, CMAL + MaskedCMALLoss stay finite in forward & backward
    with torch.no_grad():
        head.fc2.bias[t_slice] = 20.0

    x = torch.ones(2, 2, 4, requires_grad=True)
    y = torch.ones(2, 2, 1)

    with torch.amp.autocast('cpu', dtype=torch.float16):
        pred = head(x)
        for key in ('mu', 'b', 'tau', 'pi'):
            assert pred[key].dtype == torch.float32
            assert torch.all(torch.isfinite(pred[key]))

        total_loss, _ = loss_fn(pred, {'y': y})

    assert total_loss.dtype == torch.float32
    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    for param in head.parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))

    # Direct float16 inputs where tau saturated to 1.0 in float16
    t_latent_fp16 = torch.full((2, 2, 2), 20.0, dtype=torch.float16, requires_grad=True)
    tau_fp16 = (1.0 - 1e-5) * torch.sigmoid(t_latent_fp16) + 1e-5
    assert torch.all(tau_fp16 == 1.0)  # Verify float16 saturation to 1.0

    b_fp16 = torch.full((2, 2, 2), 0.5, dtype=torch.float16, requires_grad=True)
    mu_fp16 = torch.zeros((2, 2, 2), dtype=torch.float16, requires_grad=True)
    pi_fp16 = torch.full((2, 2, 2), 0.5, dtype=torch.float16, requires_grad=True)
    y_fp16 = torch.full((2, 2, 1), 1.0, dtype=torch.float16)

    direct_loss, _ = loss_fn(
        {'mu': mu_fp16, 'b': b_fp16, 'tau': tau_fp16, 'pi': pi_fp16},
        {'y': y_fp16},
    )
    assert direct_loss.dtype == torch.float32
    assert torch.isfinite(direct_loss)
    direct_loss.backward()
    assert torch.all(torch.isfinite(mu_fp16.grad))
    assert torch.all(torch.isfinite(b_fp16.grad))
    assert torch.all(torch.isfinite(t_latent_fp16.grad))
    assert torch.all(torch.isfinite(pi_fp16.grad))

    # Also verify small b in float16 where error / b = 1e6 > 65504 (float16 max)
    b_small_fp16 = torch.full((2, 2, 2), 1e-5, dtype=torch.float16)
    y_large_fp16 = torch.full((2, 2, 1), 10.0, dtype=torch.float16)
    small_b_loss, _ = loss_fn(
        {'mu': mu_fp16, 'b': b_small_fp16, 'tau': tau_fp16, 'pi': pi_fp16},
        {'y': y_large_fp16},
    )
    assert small_b_loss.dtype == torch.float32
    assert torch.isfinite(small_b_loss)
