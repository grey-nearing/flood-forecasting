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

import pytest
import torch

from model.training.loss import MaskedCMALLoss
from model.utils.config import Config


def _cmal_config(minimal_config: Config) -> Config:
    minimal_config.update_config(
        {
            'predict_last_n': 2,
            'no_loss_frequencies': [],
            'target_variables': ['streamflow'],
            'target_loss_weights': None,
            'n_distributions': 3,
        }
    )
    return minimal_config


@pytest.mark.unit
def test_cmal_loss_masks_individual_missing_timesteps(minimal_config):
    loss_fn = MaskedCMALLoss(_cmal_config(minimal_config))

    mu = torch.zeros(2, 2, 3)
    b = torch.ones(2, 2, 3)
    tau = torch.full((2, 2, 3), 0.5)
    pi = torch.full((2, 2, 3), 1.0 / 3)
    y = torch.tensor([[[0.0], [torch.nan]], [[2.0], [0.0]]])

    total_loss, _ = loss_fn(
        {'mu': mu, 'b': b, 'tau': tau, 'pi': pi}, {'y': y}
    )

    expected = torch.log(torch.tensor(4.0)) + 1.0 / 3.0
    assert torch.allclose(total_loss, expected, atol=1e-6)


@pytest.mark.unit
def test_cmal_loss_all_missing_is_differentiable_zero(minimal_config):
    loss_fn = MaskedCMALLoss(_cmal_config(minimal_config))

    mu = torch.zeros(2, 2, 3, requires_grad=True)
    prediction = {
        'mu': mu,
        'b': torch.ones(2, 2, 3),
        'tau': torch.full((2, 2, 3), 0.5),
        'pi': torch.full((2, 2, 3), 1.0 / 3),
    }
    y = torch.full((2, 2, 1), torch.nan)

    total_loss, _ = loss_fn(prediction, {'y': y})
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert total_loss.item() == 0.0
    assert torch.equal(mu.grad, torch.zeros_like(mu))


@pytest.mark.unit
def test_cmal_loss_is_finite_when_every_sequence_has_a_gap(minimal_config):
    loss_fn = MaskedCMALLoss(_cmal_config(minimal_config))

    mu = torch.zeros(2, 2, 3)
    b = torch.ones(2, 2, 3)
    tau = torch.full((2, 2, 3), 0.5)
    pi = torch.full((2, 2, 3), 1.0 / 3)
    y = torch.tensor([[[0.0], [torch.nan]], [[torch.nan], [0.0]]])

    total_loss, _ = loss_fn(
        {'mu': mu, 'b': b, 'tau': tau, 'pi': pi}, {'y': y}
    )

    assert torch.isfinite(total_loss)
    assert torch.allclose(total_loss, torch.log(torch.tensor(4.0)), atol=1e-6)
