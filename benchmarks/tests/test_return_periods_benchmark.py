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

"""Unit tests for ``benchmarks.return_periods``."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
import xarray as xr

from benchmarks import return_periods as rp_bench

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_caravan_zarr_dir(tmp_path: Path) -> Path:
    """Create a synthetic Caravan Zarr directory with streamflow.zarr and attributes.zarr."""
    caravan_dir = tmp_path / 'caravan_zarr'
    caravan_dir.mkdir(parents=True, exist_ok=True)

    basins = ['camels_01', 'camels_02', 'camels_short']
    dates = pd.date_range('1990-10-01', '2005-09-30', freq='1D')
    rng = np.random.default_rng(42)

    sf_data = np.full((len(basins), len(dates)), 1.0, dtype=np.float32)
    # Populate annual maxima for camels_01 and camels_02 across 15 water years
    for wy_idx in range(15):
        peak_day = wy_idx * 365 + 150
        if peak_day < len(dates):
            sf_data[0, peak_day] = float(
                10.0
                ** stats.pearson3.rvs(
                    skew=0.2, loc=1.2, scale=0.25, random_state=rng
                )
            )
            sf_data[1, peak_day] = float(
                10.0
                ** stats.pearson3.rvs(
                    skew=-0.3, loc=1.5, scale=0.2, random_state=rng
                )
            )
    # Introduce a low outlier (PILF) in camels_01
    sf_data[0, 150] = 0.05
    # Make camels_short have mostly NaNs (< 10 valid water years)
    sf_data[2, 365 * 5 :] = np.nan

    ds_sf = xr.Dataset(
        data_vars={'streamflow': (('basin', 'date'), sf_data)},
        coords={'basin': basins, 'date': dates},
    )
    ds_sf.to_zarr(caravan_dir / 'streamflow.zarr')

    ds_attr = xr.Dataset(
        data_vars={
            'area': (
                ('basin',),
                np.array([250.0, 1200.0, 500.0], dtype=np.float32),
            )
        },
        coords={'basin': basins},
    )
    ds_attr.to_zarr(caravan_dir / 'attributes.zarr')
    return caravan_dir


def test_main_requires_explicit_paths() -> None:
    """Verify main() requires --caravan-dir and --output-dir without hardcoded defaults."""
    with pytest.raises(ValueError, match='--caravan-dir is required'):
        rp_bench.main([])

    with pytest.raises(ValueError, match='--peakfqr-repo is required'):
        rp_bench.main(
            [
                '--mode',
                'external-usgs',
                '--caravan-dir',
                '/some/caravan',
                '--output-dir',
                '/some/out',
            ]
        )


def test_return_periods_live_mode_end_to_end(
    synthetic_caravan_zarr_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --mode live runs pure-Python verification without Fortran or R."""
    out_dir = tmp_path / 'rp_live_out'
    rc = rp_bench.main(
        [
            '--mode',
            'live',
            '--caravan-dir',
            str(synthetic_caravan_zarr_dir),
            '--output-dir',
            str(out_dir),
        ]
    )
    assert rc == 0
    csv_path = out_dir / 'return_periods_live_benchmark.csv'
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    # All 3 basins are recorded in df (no silent dropping of camels_short)
    assert len(df) == 3
    valid_df = df[df['valid_10yr']]
    assert len(valid_df) == 2
    assert set(valid_df['basin_id']) == {'camels_01', 'camels_02'}
    assert (~valid_df['fit_failed']).all()
    assert (valid_df['klow_exact_match']).all()
    assert (valid_df['q100_unit_rel_diff_pct'] < 1e-3).all()

    captured = capsys.readouterr().out
    assert 'MGBT klow unit-invariance match: conditional=100.0000%, unconditional=100.0000%' in captured
    assert 'Q100 unit-invariance rel diff (%) [P50, P75, P90, P95, P99, Max]' in captured
