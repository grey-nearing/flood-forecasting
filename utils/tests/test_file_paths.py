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
def test_no_google_internal_paths_in_file_paths() -> None:
    """Verify no /cns/, /google/, or internal hostnames exist in file_paths."""
    forbidden = (
        '/cns/',
        '/google/',
        '/usr/local/google',
        'corp.google.com',
        'googleplex',
        'sendgmr',
        'gestalt-ingest',
        'ecmwf-downloads',
    )
    source_text = Path(file_paths.__file__).read_text(encoding='utf-8')
    for token in forbidden:
        assert token not in source_text, (
            f'file_paths.py source contains {token}'
        )
    home_prefix = str(Path.home())
    for symbol in file_paths.__all__:
        val = str(getattr(file_paths, symbol)).replace(home_prefix, '~')
        for token in forbidden:
            msg = f'{symbol} contains forbidden token {token}'
            assert token not in val, msg


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
