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

"""Opt-in live network canary tests for `maas` external flood endpoints.

Skipped by default in standard CI/pytest runs (`-m "not canary"`). Enable
explicitly via:
    MAAS_LIVE_CANARY=1 pytest maas/tests/test_canary_live.py -m canary -v
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import pytest

from maas.config import MaaSConfig
from maas.fetcher import MaaSDataFetcher

if TYPE_CHECKING:
    from pathlib import Path

LIVE_CANARY_ENABLED = os.environ.get('MAAS_LIVE_CANARY', '').strip() == '1'
_MIN_CANARY_RECORDS = 5
_MISSISSIPPI_ST_LOUIS_RIVER_ID = 760684821
_ST_LOUIS_LAT = 38.6270
_ST_LOUIS_LON = -90.1994
_GLOFAS_ENSEMBLE_SIZE = 50
_GLOFAS_EXPECTED_MEMBERS = frozenset(
    f'member{i:02d}' for i in range(1, _GLOFAS_ENSEMBLE_SIZE + 1)
)
# Open-Meteo rounds each series independently (and we round to 2 decimals), so
# envelope comparisons allow a small absolute tolerance in m³/s.
_ENVELOPE_TOL_M3S = 0.05


@pytest.mark.canary
@pytest.mark.skipif(
    not LIVE_CANARY_ENABLED,
    reason=(
        'Set MAAS_LIVE_CANARY=1 to run live network canary tests against '
        'external flood APIs.'
    ),
)
class TestMaaSLiveCanary:
    """Live canary tests against Open-Meteo GloFAS v4 and GEOGLOWS ECMWF v2."""

    def test_live_glofas_forecast_st_louis(self, tmp_path: Path) -> None:
        """Verify live GloFAS v4 forecast at St. Louis, MO."""
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
        )
        fetcher = MaaSDataFetcher(config)
        fc = fetcher.glofas.fetch_forecast(
            _ST_LOUIS_LAT, _ST_LOUIS_LON, forecast_days=7
        )
        assert fc['status'] == 'live'
        assert len(fc['data']) >= _MIN_CANARY_RECORDS

    def test_live_glofas_50_member_ensemble_and_envelope(
        self, tmp_path: Path
    ) -> None:
        """Verify `ensemble=true` returns 50 members and a consistent envelope.

        Checks, for the live GloFAS v4 feed at St. Louis, MO:
        - exactly 50 member series `member01`..`member50`, each with one
          value per forecast day;
        - all three native reference series are present on every record
          (control `discharge_deterministic`, `discharge_median`,
          `discharge_mean`);
        - `p25 <= median <= p75` and `min <= p25`, `p75 <= max` for every day
          where the statistics are finite;
        - `river_discharge_min` / `river_discharge_max` match the 50-member
          envelope within rounding tolerance on days where every member is
          finite, and every member lies inside `[min, max]`.
        """
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
        )
        fetcher = MaaSDataFetcher(config)
        fc = fetcher.glofas.fetch_forecast(
            _ST_LOUIS_LAT, _ST_LOUIS_LON, forecast_days=15
        )
        assert fc['status'] == 'live'
        records = fc['data']
        n_days = len(records)
        assert n_days >= _MIN_CANARY_RECORDS

        members = fc['ensemble_members']
        assert len(members) == _GLOFAS_ENSEMBLE_SIZE
        assert set(members) == _GLOFAS_EXPECTED_MEMBERS
        for key, series in members.items():
            assert len(series) == n_days, key

        for day_idx, rec in enumerate(records):
            for field in (
                'discharge_deterministic',
                'discharge_median',
                'discharge_mean',
                'discharge_p25',
                'discharge_p75',
                'discharge_min',
                'discharge_max',
            ):
                assert field in rec, (day_idx, field)
            stats = (
                rec['discharge_min'],
                rec['discharge_p25'],
                rec['discharge_median'],
                rec['discharge_p75'],
                rec['discharge_max'],
            )
            if any(v is None for v in stats):
                continue
            q_min, q_p25, q_med, q_p75, q_max = (float(v) for v in stats)
            assert all(
                math.isfinite(v) for v in (q_min, q_p25, q_med, q_p75, q_max)
            )
            assert q_min <= q_p25 + _ENVELOPE_TOL_M3S, day_idx
            assert q_p25 <= q_med + _ENVELOPE_TOL_M3S, day_idx
            assert q_med <= q_p75 + _ENVELOPE_TOL_M3S, day_idx
            assert q_p75 <= q_max + _ENVELOPE_TOL_M3S, day_idx

            member_vals = [members[k][day_idx] for k in sorted(members)]
            if any(v is None for v in member_vals):
                continue
            finite_members = [float(v) for v in member_vals]
            assert len(finite_members) == _GLOFAS_ENSEMBLE_SIZE
            assert min(finite_members) == pytest.approx(
                q_min, abs=_ENVELOPE_TOL_M3S
            ), day_idx
            assert max(finite_members) == pytest.approx(
                q_max, abs=_ENVELOPE_TOL_M3S
            ), day_idx
            for member_val in finite_members:
                assert q_min - _ENVELOPE_TOL_M3S <= member_val, day_idx
                assert member_val <= q_max + _ENVELOPE_TOL_M3S, day_idx

    def test_live_geoglows_forecast_mississippi_reach(
        self, tmp_path: Path
    ) -> None:
        """Verify live GEOGLOWS v2 forecast on the Mississippi at St. Louis."""
        config = MaaSConfig(
            cache_dir=tmp_path / 'cache',
            river_networks_dir=tmp_path / 'river_networks',
        )
        fetcher = MaaSDataFetcher(config)
        fc = fetcher.geoglows.fetch_forecast(_MISSISSIPPI_ST_LOUIS_RIVER_ID)
        assert fc['status'] == 'live'
        assert len(fc['data']) >= _MIN_CANARY_RECORDS
