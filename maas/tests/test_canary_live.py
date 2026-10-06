# Copyright 2026 Google LLC
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

"""Opt-in live network canary tests for `maas` external flood provider endpoints.

Skipped by default in standard CI/pytest runs. Enable explicitly via:
    MAAS_LIVE_CANARY=1 pytest maas/tests/test_canary_live.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from maas.config import MaaSConfig
from maas.engine import MaaSEngine

LIVE_CANARY_ENABLED = os.environ.get('MAAS_LIVE_CANARY', '').strip() == '1'


@pytest.mark.skipif(
    not LIVE_CANARY_ENABLED,
    reason='Set MAAS_LIVE_CANARY=1 to run live network canary tests against external flood APIs.',
)
class TestMaaSLiveCanary:
    """Live canary tests against Open-Meteo GloFAS v4 and GEOGLOWS ECMWF v2."""

    def test_live_glofas_forecast_st_louis(self, tmp_path: Path) -> None:
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
        )
        engine = MaaSEngine(config)
        fc = engine.glofas.fetch_forecast(38.6270, -90.1994, forecast_days=7)
        assert fc['status'] == 'live'
        assert len(fc['data']) >= 5

    def test_live_geoglows_forecast_mississippi_reach(
        self, tmp_path: Path
    ) -> None:
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
        )
        engine = MaaSEngine(config)
        fc = engine.geoglows.fetch_forecast(720010511)
        assert fc['status'] == 'live'
        assert len(fc['data']) >= 5
