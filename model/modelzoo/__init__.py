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

"""Model zoo factory and checkpoint weight loading utilities."""

from pathlib import Path

import torch
from torch import nn

from model.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from model.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from model.utils.config import Config


def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """
    if cfg.model.lower() == 'handoff_forecast_lstm':
        model = HandoffForecastLSTM(cfg=cfg)
    elif cfg.model.lower() == 'mean_embedding_forecast_lstm':
        model = MeanEmbeddingForecastLSTM(cfg=cfg)
    else:
        raise NotImplementedError(
            f'{cfg.model} not implemented or not linked in `get_model()`'
        )

    if cfg.compile:
        return torch.compile(model, mode='max-autotune')
    return model


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device | str,
) -> None:
    """Load a model state_dict while stripping ``_orig_mod.`` prefixes.

    Parameters
    ----------
    model : nn.Module
        Target model instance (compiled or uncompiled) to load weights into.
    checkpoint_path : Path | str
        Filesystem path to the saved ``state_dict`` checkpoint file.
    device : torch.device | str
        Target device for ``torch.load(..., map_location=device)``.
    """
    state_dict = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True
    )
    state_dict = {
        k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()
    }
    target_model = getattr(model, '_orig_mod', model)
    target_model.load_state_dict(state_dict)

