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

"""Unit tests for model.utils.samplingutils."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr

from model.utils import samplingutils


@pytest.fixture
def mock_scaler():
    scaler = MagicMock()
    # Create synthetic scaler xarray Dataset for 'streamflow'
    center = 5.0
    scale = 2.0
    ds = xr.DataArray(
        [center, scale],
        coords={'parameter': ['center', 'scale']},
        dims=['parameter']
    )
    scaler.scaler = {'streamflow': ds}
    return scaler


@pytest.fixture
def mock_model():
    class DummyModel:
        pass

    model = DummyModel()
    model.parameters = lambda: iter([torch.zeros(1)])
    cfg = MagicMock()
    cfg.head = 'cmal'
    cfg.target_variables = ['streamflow']
    cfg.predict_last_n = 3
    cfg.n_distributions = 3
    cfg.negative_sample_handling = 'none'
    cfg.negative_sample_max_retries = 3
    model.cfg = cfg
    return model


@pytest.mark.unit
def test_calc_normalized_zero_thresholds(mock_scaler):
    threshold = samplingutils._calc_normalized_zero_thresholds(
        scaler=mock_scaler,
        targets=['streamflow'],
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    # -center/scale = -5.0 / 2.0 = -2.5
    assert torch.isclose(threshold[0], torch.tensor(-2.5))


@pytest.mark.unit
def test_handle_negative_values_clip():
    cfg = MagicMock(negative_sample_handling='clip')
    values = torch.tensor([-3.0, -1.0, 2.0, 5.0])
    norm_zero = torch.tensor(-2.5)

    result = samplingutils._handle_negative_values(
        cfg=cfg,
        values=values,
        sample_values=lambda ids: torch.zeros_like(ids, dtype=torch.float),
        normalized_zero=norm_zero,
    )
    assert result[0].item() == -2.5
    assert result[1].item() == -1.0
    assert result[2].item() == 2.0


@pytest.mark.unit
def test_handle_negative_values_truncate():
    cfg = MagicMock(
        negative_sample_handling='truncate',
        negative_sample_max_retries=5
    )
    values = torch.tensor([-3.0, 2.0])
    norm_zero = torch.tensor(-2.5)

    # resample function replaces negative values with positive 1.0
    def resample(mask):
        return torch.full((mask.sum(),), 1.0)

    result = samplingutils._handle_negative_values(
        cfg=cfg,
        values=values,
        sample_values=resample,
        normalized_zero=norm_zero,
    )
    assert result[0].item() == 1.0
    assert result[1].item() == 2.0


@pytest.mark.unit
def test_handle_negative_values_invalid_mode():
    cfg = MagicMock(negative_sample_handling='unsupported_mode')
    with pytest.raises(
        NotImplementedError,
        match='not supported for handling negative samples',
    ):
        samplingutils._handle_negative_values(
            cfg=cfg,
            values=torch.tensor([1.0]),
            sample_values=lambda x: x,
            normalized_zero=torch.tensor(0.0),
        )


@pytest.mark.unit
def test_sample_asymmetric_laplacians():
    m = torch.tensor([0.0, 1.0])
    b = torch.tensor([1.0, 2.0])
    t = torch.tensor([0.5, 0.5])
    ids = torch.tensor([True, True])

    sampled = samplingutils._sample_asymmetric_laplacians(ids, m, b, t)
    assert sampled.shape == (2,)
    assert torch.all(torch.isfinite(sampled))


@pytest.mark.unit
def test_sample_cmal(mock_model, mock_scaler):
    data = {
        'x_d': {'ERA5': torch.zeros(2, 10, 3)},
        'y': torch.zeros(2, 10, 1),
    }
    outputs = {
        'mu': torch.zeros(2, 3, 3),
        'b': torch.ones(2, 3, 3),
        'tau': torch.full((2, 3, 3), 0.5),
        'pi': torch.full((2, 3, 3), 1.0 / 3),
    }

    samples = samplingutils.sample_cmal(
        model=mock_model,
        data=data,
        n_samples=5,
        scaler=mock_scaler,
        outputs=outputs,
    )
    assert 'y_hat' in samples
    # Expected shape: [batch, time, target, n_samples] -> [2, 3, 1, 5]
    assert samples['y_hat'].shape == (2, 3, 1, 5)


@pytest.mark.unit
@pytest.mark.parametrize('head', ['regression', 'unsupported_head'])
def test_sample_pointpredictions_dispatch(mock_model, mock_scaler, head):
    mock_model.cfg.head = head
    data = {'y': torch.zeros(2, 5, 1)}
    with pytest.raises(
        NotImplementedError, match='Sampling mode not supported'
    ):
        samplingutils.sample_pointpredictions(
            model=mock_model,
            data=data,
            n_samples=5,
            scaler=mock_scaler,
        )


@pytest.mark.unit
def test_sample_cmal_and_deterministic_negative_sample_handling_none_vs_clip(
    mock_model, mock_scaler
):
    # Normalized zero threshold for mock_scaler is -center / scale = -5.0 / 2.0 = -2.5.
    # Use a strongly negative mu (-10.0) and tiny scale (1e-4) so raw samples/quantiles are < -2.5.
    data = {
        'x_d': {'ERA5': torch.zeros(1, 5, 3)},
        'y': torch.zeros(1, 5, 1),
    }
    outputs = {
        'mu': torch.full((1, 3, 3), -10.0),
        'b': torch.full((1, 3, 3), 1e-4),
        'tau': torch.full((1, 3, 3), 0.5),
        'pi': torch.full((1, 3, 3), 1.0 / 3),
    }

    # 1. sample_cmal with 'none' leaves negative values unclipped (< -2.5)
    mock_model.cfg.head = 'cmal'
    mock_model.cfg.negative_sample_handling = 'none'
    samples_none = samplingutils.sample_cmal(
        model=mock_model,
        data=data,
        n_samples=20,
        scaler=mock_scaler,
        outputs=outputs,
    )['y_hat']
    assert torch.all(samples_none < -2.5)

    # 2. sample_cmal with 'clip' clamps values at normalized_zero (-2.5)
    mock_model.cfg.negative_sample_handling = 'clip'
    samples_clip = samplingutils.sample_cmal(
        model=mock_model,
        data=data,
        n_samples=20,
        scaler=mock_scaler,
        outputs=outputs,
    )['y_hat']
    assert torch.allclose(samples_clip, torch.full_like(samples_clip, -2.5))

    # 3. sample_cmal_deterministic with 'none' vs 'clip'
    mock_model.cfg.head = 'cmal_deterministic'
    mock_model.cfg.negative_sample_handling = 'none'
    det_none = samplingutils.sample_cmal_deterministic(
        model=mock_model,
        data=data,
        scaler=mock_scaler,
        outputs=outputs,
    )['y_hat']
    assert torch.all(det_none < -2.5)

    mock_model.cfg.negative_sample_handling = 'clip'
    det_clip = samplingutils.sample_cmal_deterministic(
        model=mock_model,
        data=data,
        scaler=mock_scaler,
        outputs=outputs,
    )['y_hat']
    assert torch.allclose(det_clip, torch.full_like(det_clip, -2.5))
