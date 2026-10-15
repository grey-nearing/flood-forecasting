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

import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from model.modelzoo.basemodel import BaseModel
from model.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from model.modelzoo.mean_embedding_forecast_lstm import (
    MeanEmbeddingForecastLSTM,
)
from model.utils.config import Config

LOGGER = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    'handoff_forecast_lstm': HandoffForecastLSTM,
    'mean_embedding_forecast_lstm': MeanEmbeddingForecastLSTM,
}

_DEFAULT_DYNAMIC_EMBEDDING_MODULES: tuple[str, ...] = (
    'local_static_embeddings_fc',
    'hindcast_embeddings_fc',
    'forecast_embeddings_fc',
    'shared_embeddings_fc',
)

_MIN_EMBEDDING_KEY_PARTS = 2


def get_model_class(model_name: str) -> type[BaseModel]:
    """Resolve a model architecture name to its ``BaseModel`` subclass.

    Parameters
    ----------
    model_name : str
        Name of the model architecture (case-insensitive).

    Returns
    -------
    type[BaseModel]
        The registered ``BaseModel`` subclass.
    """
    key = model_name.strip().lower()
    model_cls = MODEL_REGISTRY.get(key)
    if model_cls is None:
        raise NotImplementedError(
            f'{model_name} not implemented or not linked in `get_model_class()`'
        )
    return model_cls


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
    model_cls = get_model_class(cfg.model)
    model = model_cls(cfg=cfg)

    if cfg.compile:
        return torch.compile(model, mode='max-autotune')
    return model


def _extract_embedding_pairs(
    keys: Iterable[str], embedding_prefixes: tuple[str, ...]
) -> set[tuple[str, str]]:
    """Extract ``(module_group, product_key)`` pairs from state_dict keys."""
    pairs: set[tuple[str, str]] = set()
    for key in keys:
        if key.startswith(embedding_prefixes):
            parts = key.split('.', 2)
            if len(parts) >= _MIN_EMBEDDING_KEY_PARTS:
                pairs.add((parts[0], parts[1]))
    return pairs


def _validate_no_prefix_migration(
    ckpt_pairs: set[tuple[str, str]],
    newly_init_pairs: set[tuple[str, str]],
) -> None:
    """Raise ValueError if a checkpoint product migrated between groups."""
    ckpt_group_by_prod = {prod: grp for grp, prod in ckpt_pairs}
    for new_grp, prod in sorted(newly_init_pairs):
        old_grp = ckpt_group_by_prod.get(prod)
        if old_grp is not None and old_grp != new_grp:
            raise ValueError(
                f"Product '{prod}' moved from '{old_grp}' in checkpoint "
                f"to '{new_grp}' in the target model; migrating a pretrained "
                'product between hindcast-only, forecast-only, and shared '
                'embedding groups is not supported.'
            )


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device | str,
    *,
    allow_new_embeddings: bool = False,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Load a model state_dict while stripping ``_orig_mod.`` prefixes.

    Parameters
    ----------
    model : nn.Module
        Target model instance (compiled or uncompiled) to load weights into.
    checkpoint_path : Path | str
        Filesystem path to the saved ``state_dict`` checkpoint file.
    device : torch.device | str
        Target device for ``torch.load(..., map_location=device)``.
    allow_new_embeddings : bool, optional
        If True (used during fine-tuning), allows newly added product keys in
        the model's ``dynamic_embedding_modules`` to remain at their initial
        weights and ignores dropped product embedding keys, while enforcing
        strict matching on all shared core modules.

    Returns
    -------
    tuple[set[tuple[str, str]], set[tuple[str, str]]]
        Sets of ``(module_group, product_key)`` pairs that were newly
        initialized in the target model and dropped from the checkpoint,
        respectively.
    """
    state_dict = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True
    )
    state_dict = {
        k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()
    }
    target_model = getattr(model, '_orig_mod', model)
    if not allow_new_embeddings:
        target_model.load_state_dict(state_dict)
        if isinstance(target_model, BaseModel):
            target_model.set_newly_initialized_embeddings(set())
        return set(), set()

    model_keys = set(target_model.state_dict().keys())
    dynamic_modules = (
        getattr(target_model, 'dynamic_embedding_modules', None)
        or _DEFAULT_DYNAMIC_EMBEDDING_MODULES
    )
    embedding_prefixes = tuple(f'{m}.' for m in dynamic_modules)

    ckpt_pairs = _extract_embedding_pairs(state_dict.keys(), embedding_prefixes)
    model_pairs = _extract_embedding_pairs(model_keys, embedding_prefixes)
    newly_init_pairs = model_pairs - ckpt_pairs
    dropped_pairs = ckpt_pairs - model_pairs

    _validate_no_prefix_migration(ckpt_pairs, newly_init_pairs)

    filtered_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (k in model_keys) or not k.startswith(embedding_prefixes)
    }
    incompatible = target_model.load_state_dict(
        filtered_state_dict, strict=False
    )
    bad_missing = [
        k
        for k in incompatible.missing_keys
        if not k.startswith(embedding_prefixes)
    ]
    if bad_missing or incompatible.unexpected_keys:
        cls_name = target_model.__class__.__name__
        raise RuntimeError(
            f'Error(s) in loading state_dict for {cls_name}:\n'
            f'  Missing non-embedding keys: {bad_missing}\n'
            f'  Unexpected keys: {incompatible.unexpected_keys}'
        )

    if newly_init_pairs:
        LOGGER.info(
            'Newly initialized dynamic embedding modules '
            '(not in checkpoint): %s',
            [f'{grp}.{prod}' for grp, prod in sorted(newly_init_pairs)],
        )
    if dropped_pairs:
        LOGGER.info(
            'Dropped checkpoint dynamic embedding modules '
            '(not in target model): %s',
            [f'{grp}.{prod}' for grp, prod in sorted(dropped_pairs)],
        )

    if isinstance(target_model, BaseModel):
        target_model.set_newly_initialized_embeddings(newly_init_pairs)
    return newly_init_pairs, dropped_pairs
