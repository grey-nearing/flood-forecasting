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
"""Tests for the OpenHydroNet Web frontend package."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path


def _cleanup_test_profiles(path: str) -> None:
    """Remove the temporary test profiles directory on interpreter exit."""
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


_EXISTING_PROFILES_DIR = os.environ.get(
    'OPENHYDRONET_PROFILES_DIR',
) or os.environ.get('EARTHKIT_PROFILES_DIR')

if not _EXISTING_PROFILES_DIR:
    _TEST_PROFILES_DIR = tempfile.mkdtemp(prefix='openhydronet_test_profiles_')
    os.environ['OPENHYDRONET_PROFILES_DIR'] = _TEST_PROFILES_DIR
    os.environ['EARTHKIT_PROFILES_DIR'] = _TEST_PROFILES_DIR
    atexit.register(_cleanup_test_profiles, _TEST_PROFILES_DIR)
else:
    os.environ.setdefault('OPENHYDRONET_PROFILES_DIR', _EXISTING_PROFILES_DIR)
    os.environ.setdefault('EARTHKIT_PROFILES_DIR', _EXISTING_PROFILES_DIR)
