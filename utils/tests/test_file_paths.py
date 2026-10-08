# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for canonical path constants in utils.file_paths."""

from pathlib import Path

import pytest

import utils
from utils import file_paths


@pytest.mark.unit
def test_file_paths_exports_match_init() -> None:
    """Verify all constants in utils.file_paths are re-exported by utils."""
    for symbol in file_paths.__all__:
        assert hasattr(file_paths, symbol)
        assert hasattr(utils, symbol)
        assert getattr(utils, symbol) == getattr(file_paths, symbol)


@pytest.mark.unit
def test_only_public_uris_and_openhydronet_cache_in_file_paths() -> None:
    """Verify all URIs and paths in file_paths use public domains or openhydronet cache."""
    allowed_uri_prefixes = (
        'gs://open-multimet',
        'gs://weatherbench2/',
        'https://',
        'http://localhost:',
    )
    for symbol in file_paths.__all__:
        value = getattr(file_paths, symbol)
        if isinstance(value, Path):
            assert file_paths.OPENHYDRONET_CACHE_ROOT in (
                value,
                *value.parents,
            ), f'{symbol} must reside under OPENHYDRONET_CACHE_ROOT'
        elif isinstance(value, str) and '://' in value:
            assert value.startswith(allowed_uri_prefixes), (
                f'{symbol} has unexpected URI prefix: {value}'
            )


@pytest.mark.unit
def test_canonical_gcs_and_cache_paths() -> None:
    """Verify canonical GCS URIs and cache directories are well-formed."""
    assert file_paths.OPEN_MULTIMET_BUCKET_URI == 'gs://open-multimet'
    assert file_paths.DEFAULT_GCS_HYDROATLAS_URI.startswith(
        file_paths.OPEN_MULTIMET_BUCKET_URI
    )
    assert file_paths.DEFAULT_GCS_CPC_ARCHIVE_URI.endswith(
        file_paths.DAILY_SURFACE_ZARR_NAME
    )
    assert isinstance(file_paths.OPENHYDRONET_CACHE_ROOT, Path)
    assert file_paths.OPENHYDRONET_CACHE_ROOT.name == 'openhydronet'

