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

"""Tests for repository-wide CI coverage configuration."""

import tomllib
from pathlib import Path


def test_pyproject_coverage_addopts() -> None:
    """Verify that pyproject.toml configures pytest coverage for all packages.

    This ensures that when `.github/workflows/pytest-ci.yml` runs, the coverage
    reports include the `maas`, `frontend`, `utils`, and `benchmarks` modules
    in addition to the core model packages.
    """
    pyproject_path = Path(__file__).parent.parent.parent / 'pyproject.toml'
    with pyproject_path.open('rb') as f:
        data = tomllib.load(f)

    addopts = (
        data.get('tool', {})
        .get('pytest', {})
        .get('ini_options', {})
        .get('addopts', [])
    )

    expected_cov_flags = [
        '--cov=maas',
        '--cov=frontend',
        '--cov=utils',
        '--cov=benchmarks',
    ]

    for flag in expected_cov_flags:
        assert flag in addopts, f'Missing {flag} in pyproject.toml addopts'
