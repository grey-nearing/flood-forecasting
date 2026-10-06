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

"""Unit tests for model.datasetzoo.datasetregistry."""

from collections.abc import Generator

import pytest
from torch.utils.data import Dataset

from model.datasetzoo import (
    _datasetZooRegistry,
    get_dataset,
    register_dataset,
)
from model.datasetzoo.datasetregistry import DatasetRegistry
from model.datasetzoo.multimet import Multimet
from model.utils.config import Config


@pytest.fixture(autouse=True)
def restore_dataset_zoo_registry() -> Generator[None, None, None]:
    """Restore the global _datasetZooRegistry singleton after each test."""
    registry_dict = _datasetZooRegistry._DatasetRegistry__dataset_class  # noqa: SLF001
    original_registry = registry_dict.copy()
    yield
    registry_dict.clear()
    registry_dict.update(original_registry)


class DummyValidDataset(Dataset):
    """Minimal valid Dataset subclass for registry tests."""

    def __init__(
        self,
        cfg: Config,
        is_train: bool,  # noqa: FBT001
        period: str,
        basins: list[str] | None = None,
        compute_scaler: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize dummy dataset."""
        self.cfg = cfg
        self.is_train = is_train
        self.period = period
        self.basins = basins
        self.compute_scaler = compute_scaler

    def __len__(self) -> int:
        """Return dummy dataset length."""
        return 10

    def __getitem__(self, idx: int) -> int:
        """Return dummy dataset item."""
        return idx


class DummyInvalidClass:
    """Non-Dataset class used to test TypeError on registration."""


@pytest.mark.unit
def test_dataset_registry_registration() -> None:
    """Test registering and instantiating a valid Dataset subclass."""
    registry = DatasetRegistry()
    registry.register_dataset_class('dummy', DummyValidDataset)

    cfg = Config({'dataset': 'DUMMY'})
    instance = registry.instantiate_dataset(
        cfg=cfg,
        is_train=True,
        period='train',
        basins=['basin1'],
        compute_scaler=True,
    )
    assert isinstance(instance, DummyValidDataset)
    assert instance.cfg is cfg
    assert instance.is_train is True
    assert instance.period == 'train'
    assert instance.basins == ['basin1']
    assert instance.compute_scaler is True


@pytest.mark.unit
def test_dataset_registry_invalid_type() -> None:
    """Test that registering a non-Dataset class raises TypeError."""
    registry = DatasetRegistry()
    with pytest.raises(TypeError, match='is not a subclass of Dataset'):
        registry.register_dataset_class('invalid', DummyInvalidClass)


@pytest.mark.unit
def test_dataset_registry_unimplemented_dataset() -> None:
    """Test instantiating an unregistered dataset raises NotImplementedError."""
    registry = DatasetRegistry()
    cfg = Config({'dataset': 'unregistered_dataset'})
    with pytest.raises(
        NotImplementedError, match='No dataset class implemented'
    ):
        registry.instantiate_dataset(
            cfg=cfg,
            is_train=True,
            period='train',
        )


@pytest.mark.unit
def test_module_level_register_and_get_dataset() -> None:
    """Test module-level register_dataset and get_dataset helpers."""
    register_dataset('dummy_module_dataset', DummyValidDataset)
    cfg = Config({'dataset': 'dummy_module_dataset'})
    instance = get_dataset(
        cfg=cfg,
        is_train=False,
        period='test',
        basins=['basin_test'],
    )
    assert isinstance(instance, DummyValidDataset)
    assert instance.cfg is cfg
    assert instance.is_train is False
    assert instance.period == 'test'
    assert instance.basins == ['basin_test']
    assert instance.compute_scaler is False


@pytest.mark.unit
def test_module_level_registry_does_not_leak() -> None:
    """Verify module-level registrations do not leak across tests."""
    cfg = Config({'dataset': 'dummy_module_dataset'})
    with pytest.raises(
        NotImplementedError, match='No dataset class implemented'
    ):
        get_dataset(cfg=cfg, is_train=False, period='test')

    registry_dict = _datasetZooRegistry._DatasetRegistry__dataset_class  # noqa: SLF001
    assert registry_dict.get('multimet') is Multimet
