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

"""Unit tests for ``benchmarks.model``."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from benchmarks.model import (
    DEFAULT_STATIC_ATTRIBUTES,
    _assert_no_fallback_or_imputation,
    _build_base_config_dict,
    benchmark_architectures,
    benchmark_hot_start,
    compare_forcings,
    main,
)
from model.utils.config import Config

pytestmark = pytest.mark.unit


def _write_zarr_v2(ds: xr.Dataset, path: Path) -> None:
    ds = ds.copy()
    for var_name in list(ds.data_vars):
        ds[var_name] = ds[var_name].astype(np.float32)
    ds.coords['basin'] = ds.coords['basin'].astype(str)
    for k in ds.variables:
        ds[k].encoding.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode='w', zarr_format=2, consolidated=True)


@pytest.fixture(scope='module')
def synthetic_benchmark_env(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    """Create a small deterministic local Caravan + MultiMet Zarr environment."""
    root = tmp_path_factory.mktemp('model_benchmark_fixture')
    basins = ['hysets_01010000', 'hysets_01020000']
    dates = pd.date_range('2018-04-01', '2018-04-25', freq='1D')
    lead_times = np.array(
        [np.timedelta64(i, 'D') for i in range(1, 5)], dtype='timedelta64[ns]'
    )

    rng = np.random.default_rng(12345)
    n_b = len(basins)
    n_d = len(dates)
    n_l = len(lead_times)

    basin_file = root / 'basins.txt'
    basin_file.write_text('\n'.join(basins) + '\n')

    # Caravan streamflow + static attributes
    caravan_dir = root / 'caravan'
    sf_vals = (
        2.0
        + np.sin(np.linspace(0.0, 4.0 * np.pi, n_d))[None, :]
        + rng.uniform(0.1, 0.5, size=(n_b, n_d))
    ).astype(np.float32)
    ds_sf = xr.Dataset(
        data_vars={'streamflow': (('basin', 'date'), sf_vals)},
        coords={'basin': basins, 'date': dates},
    )
    _write_zarr_v2(ds_sf, caravan_dir / 'streamflow.zarr')

    attr_vars = {
        attr: (
            ('basin',),
            np.array([10.0 + i, 20.0 + i], dtype=np.float32),
        )
        for i, attr in enumerate(DEFAULT_STATIC_ATTRIBUTES)
    }
    ds_attr = xr.Dataset(data_vars=attr_vars, coords={'basin': basins})
    _write_zarr_v2(ds_attr, caravan_dir / 'attributes.zarr')

    # Canonical and perturbed reconstructed MultiMet stores (CPC, IMERG, HRES)
    can_dir = root / 'canonical_multimet'
    rec_dir = root / 'reconstructed_multimet'

    cpc_can = rng.uniform(0.5, 10.0, size=(n_b, n_d)).astype(np.float32)
    imerg_can = rng.uniform(0.5, 12.0, size=(n_b, n_d)).astype(np.float32)

    for base_dir, delta in [(can_dir, 0.0), (rec_dir, 0.15)]:
        ds_cpc = xr.Dataset(
            data_vars={
                'cpc_precipitation': (
                    ('basin', 'date'),
                    (cpc_can + delta).astype(np.float32),
                )
            },
            coords={'basin': basins, 'date': dates},
        )
        _write_zarr_v2(ds_cpc, base_dir / 'CPC' / 'timeseries.zarr')

        ds_imerg = xr.Dataset(
            data_vars={
                'imerg_precipitation': (
                    ('basin', 'date'),
                    (imerg_can + delta * 1.5).astype(np.float32),
                )
            },
            coords={'basin': basins, 'date': dates},
        )
        _write_zarr_v2(ds_imerg, base_dir / 'IMERG' / 'timeseries.zarr')

        hres_vars = {}
        for j, vname in enumerate(
            [
                'hres_total_precipitation',
                'hres_temperature_2m',
                'hres_surface_pressure',
                'hres_surface_net_solar_radiation',
                'hres_surface_net_thermal_radiation',
            ]
        ):
            base_arr = (
                (j + 1) * 5.0
                + np.cos(np.linspace(0.0, 3.0 * np.pi, n_d))[None, :, None]
                + np.arange(n_l, dtype=np.float32)[None, None, :] * 0.2
                + np.arange(n_b, dtype=np.float32)[:, None, None] * 0.5
            ).astype(np.float32)
            hres_vars[vname] = (
                ('basin', 'date', 'lead_time'),
                (base_arr + delta).astype(np.float32),
            )
        ds_hres = xr.Dataset(
            data_vars=hres_vars,
            coords={'basin': basins, 'date': dates, 'lead_time': lead_times},
        )
        _write_zarr_v2(ds_hres, base_dir / 'HRES' / 'timeseries.zarr')

    return {
        'basin_file': basin_file,
        'caravan_dir': caravan_dir,
        'can_dir': can_dir,
        'rec_dir': rec_dir,
    }


def test_assert_no_fallback_or_imputation_rejects_violations(
    synthetic_benchmark_env: dict[str, Path], tmp_path: Path
) -> None:
    """Verify union_mapping fallback and silent basin skipping are rejected."""
    cfg_dict = _build_base_config_dict(
        experiment_name='test_guard',
        run_dir=tmp_path,
        dynamics_data_dir=synthetic_benchmark_env['can_dir'],
        caravan_dir=synthetic_benchmark_env['caravan_dir'],
        basin_file=synthetic_benchmark_env['basin_file'],
    )
    cfg_union = Config(
        {
            **cfg_dict,
            'union_mapping': {'cpc_precipitation': 'imerg_precipitation'},
        }
    )
    with pytest.raises(ValueError, match='union_mapping fallback'):
        _assert_no_fallback_or_imputation(cfg_union)

    cfg_skip = Config({**cfg_dict, 'tester_skip_obs_all_nan': True})
    with pytest.raises(ValueError, match='tester_skip_obs_all_nan'):
        _assert_no_fallback_or_imputation(cfg_skip)


def test_compare_forcings_identical_and_perturbed(
    synthetic_benchmark_env: dict[str, Path], tmp_path: Path
) -> None:
    """Verify compare_forcings reports zero diff on identical stores and finite diff on perturbed."""
    common_kwargs = {
        'caravan_dir': synthetic_benchmark_env['caravan_dir'],
        'basin_file': synthetic_benchmark_env['basin_file'],
        'seq_length': 4,
        'lead_time': 2,
        'forecast_overlap': 4,
        'predict_last_n': 3,
        'hidden_size': 8,
        'batch_size': 8,
        'epochs': 1,
        'seed': 42,
        'train_start_date': '06/04/2018',
        'train_end_date': '15/04/2018',
        'test_start_date': '06/04/2018',
        'test_end_date': '20/04/2018',
    }

    res_ident = compare_forcings(
        canonical_multimet_dir=synthetic_benchmark_env['can_dir'],
        reconstructed_multimet_dir=synthetic_benchmark_env['can_dir'],
        output_dir=tmp_path / 'ident',
        **common_kwargs,
    )
    summary_ident = res_ident['summary']
    assert summary_ident['total_basins'] == 2
    assert summary_ident['evaluated_basins'] == 2
    assert summary_ident['missing_basins_count'] == 0
    assert summary_ident['failed_or_nan_basins'] == 0
    assert summary_ident['pred_nan_when_obs_valid_can'] == 0
    assert summary_ident['pred_nan_when_obs_valid_rec'] == 0
    assert summary_ident['pred_max_abs_diff'] == pytest.approx(0.0, abs=1e-6)
    assert summary_ident['max_abs_delta_NSE'] == pytest.approx(0.0, abs=1e-6)

    res_pert = compare_forcings(
        canonical_multimet_dir=synthetic_benchmark_env['can_dir'],
        reconstructed_multimet_dir=synthetic_benchmark_env['rec_dir'],
        output_dir=tmp_path / 'pert',
        trained_run_dir=res_ident['trained_run_dir'],
        epoch=1,
        **common_kwargs,
    )
    summary_pert = res_pert['summary']
    assert summary_pert['evaluated_basins'] == 2
    assert summary_pert['failed_or_nan_basins'] == 0
    assert summary_pert['pred_max_abs_diff'] > 0.0
    assert np.isfinite(summary_pert['median_NSE_canonical'])
    assert np.isfinite(summary_pert['median_NSE_reconstructed'])
    assert Path(summary_pert['per_basin_csv']).is_file()
    assert Path(summary_pert['per_lead_csv']).is_file()


def test_benchmark_architectures_and_hot_start_and_cli(
    synthetic_benchmark_env: dict[str, Path], tmp_path: Path
) -> None:
    """Verify benchmark_architectures, benchmark_hot_start, and CLI main."""
    arch_res = benchmark_architectures(
        canonical_multimet_dir=synthetic_benchmark_env['can_dir'],
        reconstructed_multimet_dir=synthetic_benchmark_env['rec_dir'],
        caravan_dir=synthetic_benchmark_env['caravan_dir'],
        basin_file=synthetic_benchmark_env['basin_file'],
        output_dir=tmp_path / 'arch_out',
        architectures=[
            'mean_embedding_forecast_lstm',
            'handoff_forecast_lstm',
            'cudalstm',
        ],
        seq_length=4,
        lead_time=2,
        hidden_size=8,
        batch_size=8,
        epochs=1,
        seed=42,
        train_start_date='06/04/2018',
        train_end_date='15/04/2018',
        test_start_date='06/04/2018',
        test_end_date='20/04/2018',
    )
    arch_df = arch_res['architecture_df']
    assert len(arch_df) == 3
    assert set(arch_df['architecture']) == {
        'mean_embedding_forecast_lstm',
        'handoff_forecast_lstm',
        'cudalstm',
    }
    assert (arch_df['evaluated_basins'] == 2).all()
    assert (arch_df['failed_or_nan_basins'] == 0).all()
    assert len(arch_res['memory_df']) == 4

    hs_res = benchmark_hot_start(
        canonical_multimet_dir=synthetic_benchmark_env['can_dir'],
        caravan_dir=synthetic_benchmark_env['caravan_dir'],
        basin_file=synthetic_benchmark_env['basin_file'],
        output_dir=tmp_path / 'hs_out',
        seq_length=4,
        lead_time=2,
        hidden_size=8,
        batch_size=8,
        epochs=1,
        seed=42,
        train_start_date='06/04/2018',
        train_end_date='15/04/2018',
        test_start_date='06/04/2018',
        test_end_date='20/04/2018',
    )
    state_df = hs_res['state_df']
    assert len(state_df) == 2
    assert (state_df['max_abs_diff'] < 1e-5).all()
    assert (state_df['failed_or_nan_basins'] == 0).all()

    cli_out = tmp_path / 'cli_out'
    cli_res = main(
        [
            '--mode',
            'compare-forcings',
            '--canonical-multimet-dir',
            str(synthetic_benchmark_env['can_dir']),
            '--reconstructed-multimet-dir',
            str(synthetic_benchmark_env['rec_dir']),
            '--caravan-dir',
            str(synthetic_benchmark_env['caravan_dir']),
            '--basin-file',
            str(synthetic_benchmark_env['basin_file']),
            '--output-dir',
            str(cli_out),
            '--seq-length',
            '4',
            '--lead-time',
            '2',
            '--hidden-size',
            '8',
            '--batch-size',
            '8',
            '--epochs',
            '1',
            '--train-start-date',
            '06/04/2018',
            '--train-end-date',
            '15/04/2018',
            '--test-start-date',
            '06/04/2018',
            '--test-end-date',
            '20/04/2018',
        ]
    )
    assert 'compare_forcings' in cli_res
    assert (cli_out / 'benchmark_summary.json').is_file()
