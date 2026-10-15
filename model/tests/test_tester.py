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

import shutil
from collections.abc import Callable
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import zarr
from torch import nn

from model.datasetzoo.multimet import SampleIndexer
from model.evaluation import get_tester
from model.evaluation.tester import RegressionTester
from model.evaluation.utils import BasinBatchSampler
from model.modelzoo import load_model_weights
from model.training.basetrainer import BaseTrainer
from model.utils.config import Config, TesterSamplesReduction


@pytest.fixture
def fixture():
    sample_index = SampleIndexer(  # sample index -> basin metadata
        (
            (
                'basin',
                np.array(
                    [
                        # basin 101: 7 samples, 3 batches
                        *[101, 101, 101, 101, 101, 101, 101],
                        # basin 102: 6 samples, 2 batches
                        *[102, 102, 102, 102, 102, 102],
                        # basin 103: 2 samples, 1 batch
                        *[103, 103],
                    ]
                ),
            ),
            (
                'date',
                np.array(
                    [
                        # basin 101: 7 samples, 3 batches
                        *[1, 2, 3, 4, 5, 6, 7],
                        # basin 102: 6 samples, 2 batches
                        *[1, 2, 3, 4, 5, 6],
                        # basin 103: 2 samples, 1 batch
                        *[1, 2],
                    ]
                ),
            ),
        )
    )

    expected_groups = {  # basin id -> expected sample indexes
        101: [0, 1, 2, 3, 4, 5, 6],
        102: [7, 8, 9, 10, 11, 12],
        103: [13, 14],
    }

    return {
        'sample_index': sample_index,
        'expected_groups': expected_groups,
        'total_basins': 3,
        'total_samples': 15,
    }


def test_init_groups_basins(fixture):
    """Test grouping all sample indices by their basin id."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=None
    )

    indices = np.concatenate(list(sampler))

    groups = fixture['expected_groups']
    expected_indices = groups[101] + groups[102] + groups[103]
    np.testing.assert_array_equal(indices, expected_indices)


def test_init_groups_basins_subset(fixture):
    """Test grouping all sample indices by their basin id."""
    sampler = BasinBatchSampler(
        fixture['sample_index'],
        batch_size=3,
        basins_indexes=np.array([102, 103]),
    )

    indices = np.concatenate(list(sampler))

    groups = fixture['expected_groups']
    expected_indices = groups[102] + groups[103]
    np.testing.assert_array_equal(indices, expected_indices)


def test_num_batches(fixture):
    """Test _num_batches is total num batches for an epoc (accounting for partial batch)."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=None
    )

    expected_num_batches = ceil(7 / 3) + ceil(6 / 3) + ceil(2 / 3)

    assert sampler._num_batches == expected_num_batches


def test_len_returns_num_batches(fixture):
    """Test __len__ returns total num batches for an epoc (accounting for partial batch)."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=None
    )

    expected_num_batches = ceil(7 / 3) + ceil(6 / 3) + ceil(2 / 3)

    assert len(sampler) == expected_num_batches


def test_iter_yields_all_samples_once(fixture):
    """Test iterating results in all sample indices once per epoc."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=None
    )

    indices = {i for batch in sampler for i in batch}

    assert indices == set(fixture['sample_index'].keys())


def test_one_basin_per_batch(fixture):
    """Test every batch contains samples belonging to only one basin."""
    sampler = BasinBatchSampler(
        fixture['sample_index'], batch_size=3, basins_indexes=None
    )

    basinss = [
        {fixture['sample_index'][i]['basin'] for i in batch}
        for batch in sampler
    ]
    assert all(len(basins) == 1 for basins in basinss)


def test_sampler_with_single_basin(fixture):
    """Test that the sampler works with one basin."""
    sample_index = SampleIndexer(
        (
            ('basin', np.array([101, 101, 101, 101, 101, 101, 101])),
            ('date', np.array([1, 2, 3, 4, 5, 6, 7])),
        ),
    )
    sampler = BasinBatchSampler(
        sample_index, batch_size=3, basins_indexes=None
    )

    indices = {idx for batch in sampler for idx in batch}

    assert len(sampler) == ceil(7 / 3)
    assert indices == set(sample_index.keys())


def test_sampler_with_batch_size_larger_than_samples():
    """Test behavior when a basin has fewer samples than the batch size."""
    sample_index = SampleIndexer((('basin', np.array([201, 201])),))
    sampler = BasinBatchSampler(
        sample_index, batch_size=5, basins_indexes=None
    )
    batches = list(sampler)

    assert len(sampler) == 1
    assert len(batches) == 1
    assert batches[0] == (0, 1)


def test_empty_basins_indexes_yields_zero_batches(fixture):
    """Test behavior when basins_indexes is empty, yielding 0 batches."""
    sampler = BasinBatchSampler(
        fixture['sample_index'],
        batch_size=3,
        basins_indexes=np.array([], dtype=int),
    )
    assert len(sampler) == 0
    assert list(sampler) == []


def test_evaluate_synchronizes_configured_cuda_device():
    """_evaluate must synchronize `self.device`, not only the default GPU."""
    tester = object.__new__(RegressionTester)
    tester.device = torch.device('cuda:2')
    tester.cfg = MagicMock(
        predict_last_n=2,
        hot_start_path=None,
        save_state=False,
    )
    tester.period = 'test'
    tester.loss_obj = lambda preds, data: (
        torch.tensor(0.25),
        {'loss': torch.tensor(0.25)},
    )

    class _FakeDataset:
        _basins = ['basin_A']

    batch = {
        'basin_index': torch.tensor([0, 0]),
        'date': np.array(
            [
                ['2020-01-01', '2020-01-02'],
                ['2020-01-02', '2020-01-03'],
            ],
            dtype='datetime64[D]',
        ),
        'x_d': {'f1': torch.ones((2, 2, 1))},
        'y': torch.full((2, 2, 1), 3.0),
    }

    class _FakeLoader:
        dataset = _FakeDataset()

        def __iter__(self):
            return iter([batch])

    model = MagicMock()
    model.pre_model_hook.side_effect = lambda d, is_train: d
    model.return_value = {'y_hat': torch.full((2, 2, 1), 4.0)}

    with (
        patch.object(torch.Tensor, 'to', lambda self, *a, **kw: self),
        patch('model.evaluation.tester.autocast'),
        patch('torch.cuda.synchronize') as mock_sync,
    ):
        results = list(
            tester._evaluate(
                model=model,
                loader=_FakeLoader(),
                basins=['basin_A'],
            )
        )

    mock_sync.assert_called_once_with(torch.device('cuda:2'))
    assert len(results) == 1
    assert results[0]['basin'] == 'basin_A'
    torch.testing.assert_close(
        results[0]['preds'], torch.full((2, 2, 1), 4.0)
    )
    torch.testing.assert_close(
        results[0]['obs'], torch.full((2, 2, 1), 3.0)
    )


def test_metrics_to_dataframe():
    """metrics_to_dataframe extracts single- and multi-target metrics per basin."""
    from model.evaluation.utils import metrics_to_dataframe

    single_results = {
        'basin_A': {'NSE': 0.85, 'RMSE': 1.2},
        'basin_B': {'NSE': 0.40},
    }
    df_single = metrics_to_dataframe(
        single_results, metrics=['NSE', 'RMSE'], targets=['streamflow']
    )
    assert df_single.index.name == 'basin'
    assert list(df_single.index) == ['basin_A', 'basin_B']
    assert df_single.loc['basin_A', 'NSE'] == pytest.approx(0.85)
    assert df_single.loc['basin_A', 'RMSE'] == pytest.approx(1.2)
    assert np.isnan(df_single.loc['basin_B', 'RMSE'])

    multi_results = {
        'basin_A': {'q1_NSE': 0.9, 'q2_NSE': 0.7},
    }
    df_multi = metrics_to_dataframe(
        multi_results, metrics=['NSE'], targets=['q1', 'q2']
    )
    assert df_multi.loc['basin_A', 'q1_NSE'] == pytest.approx(0.9)
    assert df_multi.loc['basin_A', 'q2_NSE'] == pytest.approx(0.7)


def test_get_tester_raises_not_implemented_for_unsupported_head(
    tmp_path: Path,
) -> None:
    """Unsupported head in get_tester raises NotImplementedError."""
    cfg = Config(
        {'head': 'unsupported_head_type'},
        dev_mode=True,
    )

    with pytest.raises(
        NotImplementedError,
        match='No evaluation method implemented for unsupported_head_type head',
    ):
        get_tester(cfg=cfg, run_dir=tmp_path, period='test', init_model=False)


class _CompiledWrapper(nn.Module):
    """Wrapper mimicking torch.compile OptimizedModule's _orig_mod attribute."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner


def test_load_model_weights_strips_orig_mod_prefix(tmp_path: Path) -> None:
    """load_model_weights loads compiled and uncompiled checkpoints cleanly."""
    source_linear = nn.Linear(3, 2, bias=True)
    compiled_state_dict = {
        f'_orig_mod.{k}': torch.full_like(v, 2.5)
        for k, v in source_linear.state_dict().items()
    }
    compiled_ckpt_path = tmp_path / 'model_epoch001.pt'
    torch.save(compiled_state_dict, compiled_ckpt_path)

    # 1. Load compiled checkpoint (_orig_mod.* keys) into uncompiled model.
    uncompiled_target = nn.Linear(3, 2, bias=True)
    load_model_weights(uncompiled_target, compiled_ckpt_path, device='cpu')
    for param in uncompiled_target.parameters():
        torch.testing.assert_close(param.data, torch.full_like(param.data, 2.5))

    # 2. Load compiled checkpoint (_orig_mod.* keys) into compiled wrapper.
    wrapped_inner = nn.Linear(3, 2, bias=True)
    wrapped_target = _CompiledWrapper(wrapped_inner)
    load_model_weights(wrapped_target, compiled_ckpt_path, device='cpu')
    for param in wrapped_inner.parameters():
        torch.testing.assert_close(param.data, torch.full_like(param.data, 2.5))

    # 3. Load uncompiled checkpoint into compiled wrapper.
    uncompiled_state_dict = {
        k: torch.full_like(v, -1.25)
        for k, v in source_linear.state_dict().items()
    }
    uncompiled_ckpt_path = tmp_path / 'model_epoch002.pt'
    torch.save(uncompiled_state_dict, uncompiled_ckpt_path)
    load_model_weights(wrapped_target, uncompiled_ckpt_path, device='cpu')
    for param in wrapped_inner.parameters():
        torch.testing.assert_close(
            param.data, torch.full_like(param.data, -1.25)
        )


def test_evaluate_seeds_rng_for_reproducible_cmal_sampling(
    make_minimal_config, tmp_path: Path
) -> None:
    """Consecutive UncertaintyTester.evaluate() calls produce bit-identical CMAL samples."""
    run_dir = tmp_path / 'cmal_seed_run'
    cfg = make_minimal_config(
        {
            'experiment_name': 'cmal_seed_test',
            'run_dir': run_dir,
            'base_run_dir': run_dir,
            'head': 'cmal',
            'loss': 'cmalloss',
            'n_distributions': 2,
            'n_samples': 25,
            'tester_sample_reduction': 'mean',
            'negative_sample_handling': 'clip',
            'epochs': 1,
            'save_weights_every': 1,
            'validate_every': 0,
            'metrics': ['NSE', 'KGE'],
            'seed': 123,
            'verbose': 0,
        }
    )

    trainer = BaseTrainer(cfg)
    trainer.initialize_training()
    trainer.train_and_validate()

    tester = get_tester(
        cfg=trainer.cfg,
        run_dir=trainer.cfg.run_dir,
        period='test',
        init_model=True,
    )
    tester.evaluate(save_results=True, metrics=['NSE', 'KGE'])

    eval_dir = trainer.cfg.run_dir / 'test' / 'model_epoch001'
    metrics_1 = pd.read_csv(eval_dir / 'test_metrics.csv', index_col='basin')
    ds_1 = xr.open_zarr(
        eval_dir / 'test_results.zarr', consolidated=False
    ).load()

    # Remove saved evaluation outputs and perturb global RNG states between evaluations
    shutil.rmtree(trainer.cfg.run_dir / 'test')
    torch.manual_seed(99999)
    np.random.seed(99999)

    tester.evaluate(save_results=True, metrics=['NSE', 'KGE'])

    metrics_2 = pd.read_csv(eval_dir / 'test_metrics.csv', index_col='basin')
    ds_2 = xr.open_zarr(
        eval_dir / 'test_results.zarr', consolidated=False
    ).load()

    np.testing.assert_array_equal(
        ds_1['streamflow_sim'].values,
        ds_2['streamflow_sim'].values,
    )
    pd.testing.assert_frame_equal(metrics_1, metrics_2)


@pytest.mark.integration
def test_evaluate_scores_every_remaining_basin_after_all_nan_exclusion(
    make_minimal_config: Callable[..., Config],
    five_basin_dataset: Path,
    tmp_path: Path,
) -> None:
    """Excluding all-NaN basins per period must not drop valid basins (#76, #77, #80).

    Verifies without mocks or stubs that:
    1. `get_tester` excludes all-NaN basins using the period-specific date
       window (`train`, `validation`, `test`) (#77).
    2. After `basin_01` (position 0 in `dataset._basins`) is excluded in the
       `test` period, `evaluate()` resolves `basins_indexes` against
       `dataset._basins` (`[1, 2, 3, 4]`) and scores all four remaining valid
       basins without dropping `basin_05` (#76).
    3. `evaluate(save_results=True)` writes consolidated Zarr metadata (#80).
    """
    data_dir = tmp_path / 'data'
    shutil.copytree(
        five_basin_dataset,
        data_dir,
        ignore=shutil.ignore_patterns('streamflow.zarr'),
    )
    targets = xr.open_zarr(five_basin_dataset / 'streamflow.zarr').load()
    # basin_01 is all-NaN over the test window (04/02/2020 - 13/02/2020) only.
    targets['streamflow'].loc[
        {'basin': 'basin_01', 'date': slice('2020-02-04', '2020-02-13')}
    ] = np.nan
    targets.to_zarr(data_dir / 'streamflow.zarr', mode='w')

    run_dir = tmp_path / 'skip_all_nan_run'
    cfg = make_minimal_config(
        {
            'experiment_name': 'skip_all_nan_test',
            'run_dir': run_dir,
            'base_run_dir': run_dir,
            'data_dir': data_dir,
            'train_basin_file': data_dir / 'basins.txt',
            'validation_basin_file': data_dir / 'basins.txt',
            'test_basin_file': data_dir / 'basins.txt',
            'tester_skip_obs_all_nan': True,
            'metrics': ['NSE'],
        }
    )
    trainer = BaseTrainer(cfg)
    trainer.initialize_training()
    trainer.train_and_validate()

    tester = get_tester(
        cfg=trainer.cfg,
        run_dir=trainer.cfg.run_dir,
        period='test',
        init_model=True,
    )
    assert tester.basins == ['basin_02', 'basin_03', 'basin_04', 'basin_05']
    assert tester.dataset._basins == [f'basin_0{i}' for i in range(1, 6)]

    tester.evaluate(save_results=True, metrics=['NSE'])

    eval_dir = trainer.cfg.run_dir / 'test' / 'model_epoch001'
    metrics = pd.read_csv(eval_dir / 'test_metrics.csv', index_col='basin')
    assert list(metrics.index) == [
        'basin_02',
        'basin_03',
        'basin_04',
        'basin_05',
    ]
    assert np.isfinite(metrics['NSE']).all()

    # Verify consolidated Zarr metadata is written (#80).
    result_file = eval_dir / 'test_results.zarr'
    group = zarr.open_group(str(result_file), mode='r', use_consolidated=True)
    consolidated_arrays = set(group.metadata.consolidated_metadata.metadata)
    assert {'streamflow_obs', 'streamflow_sim'} <= consolidated_arrays

    # Now blank out basin_02 over the validation window (25/01/2020 - 03/02/2020)
    # and basin_03 over the train window (15/01/2020 - 24/01/2020) on disk, and
    # verify that `get_tester` excludes only the period-matching basin (#77).
    targets['streamflow'].loc[
        {'basin': 'basin_02', 'date': slice('2020-01-25', '2020-02-03')}
    ] = np.nan
    targets['streamflow'].loc[
        {'basin': 'basin_03', 'date': slice('2020-01-15', '2020-01-24')}
    ] = np.nan
    shutil.rmtree(data_dir / 'streamflow.zarr')
    targets.to_zarr(data_dir / 'streamflow.zarr', mode='w')

    train_tester = get_tester(
        cfg=trainer.cfg,
        run_dir=trainer.cfg.run_dir,
        period='train',
        init_model=False,
    )
    assert train_tester.basins == [
        'basin_01',
        'basin_02',
        'basin_04',
        'basin_05',
    ]

    val_tester = get_tester(
        cfg=trainer.cfg,
        run_dir=trainer.cfg.run_dir,
        period='validation',
        init_model=False,
    )
    assert val_tester.basins == [
        'basin_01',
        'basin_03',
        'basin_04',
        'basin_05',
    ]

    # Across a multi-window period spanning both validation and test dates,
    # basin_01 (valid in window 1) and basin_02 (valid in window 2) both have
    # valid observations and are retained; only a basin that is all-NaN across
    # BOTH windows (basin_04) is excluded.
    targets['streamflow'].loc[
        {'basin': 'basin_04', 'date': slice('2020-01-25', '2020-02-13')}
    ] = np.nan
    shutil.rmtree(data_dir / 'streamflow.zarr')
    targets.to_zarr(data_dir / 'streamflow.zarr', mode='w')

    trainer.cfg.update_config(
        {
            'test_start_date': ['25/01/2020', '04/02/2020'],
            'test_end_date': ['03/02/2020', '13/02/2020'],
        }
    )
    multi_window_tester = get_tester(
        cfg=trainer.cfg,
        run_dir=trainer.cfg.run_dir,
        period='test',
        init_model=False,
    )
    assert multi_window_tester.basins == [
        'basin_01',
        'basin_02',
        'basin_03',
        'basin_05',
    ]


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Compute the reference NSE over the valid pairs of flat arrays."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[valid], sim[valid]
    return float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def _make_metric_tester(**cfg_overrides: object) -> RegressionTester:
    """Build a tester whose config only drives the metric computation."""
    tester = object.__new__(RegressionTester)
    cfg = {
        'target_variables': ['q'],
        'clip_targets_to_zero': [],
        'head': 'regression',
        'tester_sample_reduction': TesterSamplesReduction.MEAN,
    }
    cfg.update(cfg_overrides)
    tester.cfg = SimpleNamespace(**cfg)
    return tester


def _make_step_results(
    obs_vals: np.ndarray, sim_vals: np.ndarray, time_steps: list[int]
) -> tuple[xr.Dataset, pd.DatetimeIndex]:
    """Wrap `(date, time_step[, samples])` arrays like `evaluate()` does."""
    dates = pd.date_range('2020-01-01', periods=obs_vals.shape[0], freq='1D')
    sim_dims = ('date', 'time_step', 'samples')[: sim_vals.ndim]
    ds = xr.Dataset(
        data_vars={
            'q_obs': (('date', 'time_step'), obs_vals),
            'q_sim': (sim_dims, sim_vals),
        },
        coords={'date': dates, 'time_step': time_steps},
    )
    return ds, dates


@pytest.mark.unit
def test_evaluate_target_metrics_pools_all_steps_including_nowcast() -> None:
    """`NSE` pools every predicted step (nowcast included); steps get suffixes.

    The per-step errors grow with the step, so the pooled value differs from
    the first step (`time_step == 0`) and the last step alone.
    """
    rng = np.random.default_rng(0)
    base = rng.gamma(shape=2.0, scale=5.0, size=90)
    obs_vals = base[:, None] + np.array([0.0, 0.5, 1.0])
    sim_vals = obs_vals + np.array([0.3, 2.0, 6.0]) * rng.normal(size=(90, 3))
    # Missing pairs must be masked out, not propagated.
    obs_vals[3, 1] = np.nan
    sim_vals[7, 2] = np.nan
    ds, dates = _make_step_results(obs_vals, sim_vals, time_steps=[0, 1, 2])

    results = _make_metric_tester()._evaluate_target_metrics(  # noqa: SLF001
        xr=ds,
        basin='basin_a',
        target_variable='q',
        var_metrics=['NSE', 'MSE'],
        date_range=dates,
    )

    assert set(results) == {
        'NSE',
        'MSE',
        'NSE_hindcast',
        'NSE_lead1',
        'NSE_lead2',
        'MSE_hindcast',
        'MSE_lead1',
        'MSE_lead2',
    }
    obs_flat = obs_vals.T.reshape(-1)
    sim_flat = sim_vals.T.reshape(-1)
    assert results['NSE'] == pytest.approx(_nse(obs_flat, sim_flat))
    assert results['MSE'] == pytest.approx(
        np.nanmean((sim_flat - obs_flat) ** 2)
    )
    for step_idx, suffix in enumerate(['hindcast', 'lead1', 'lead2']):
        assert results[f'NSE_{suffix}'] == pytest.approx(
            _nse(obs_vals[:, step_idx], sim_vals[:, step_idx])
        )
    assert results['NSE'] != pytest.approx(results['NSE_hindcast'])
    assert results['NSE'] != pytest.approx(results['NSE_lead2'])


@pytest.mark.unit
def test_evaluate_target_metrics_single_step_has_no_step_suffix() -> None:
    """With one predicted step the aggregate is that step; no suffixed keys."""
    obs_vals = np.linspace(1.0, 20.0, 20)[:, None]
    sim_vals = obs_vals * 1.1
    ds, dates = _make_step_results(obs_vals, sim_vals, time_steps=[2])

    results = _make_metric_tester()._evaluate_target_metrics(  # noqa: SLF001
        xr=ds,
        basin='basin_a',
        target_variable='q',
        var_metrics=['NSE'],
        date_range=dates,
    )

    assert set(results) == {'NSE'}
    assert results['NSE'] == pytest.approx(_nse(obs_vals[:, 0], sim_vals[:, 0]))


@pytest.mark.unit
def test_evaluate_target_metrics_reduces_samples_and_clips() -> None:
    """Uncertainty samples are reduced and clipped before pooling."""
    obs_vals = np.tile(np.linspace(1.0, 10.0, 10)[:, None], (1, 2))
    samples = np.stack(
        [obs_vals - 3.0, obs_vals + 1.0, obs_vals + 2.0, obs_vals + 4.0],
        axis=-1,
    )
    ds, dates = _make_step_results(obs_vals, samples, time_steps=[1, 2])
    tester = _make_metric_tester(
        head='cmal',
        tester_sample_reduction=TesterSamplesReduction.MEDIAN,
        clip_targets_to_zero=['q'],
    )

    results = tester._evaluate_target_metrics(  # noqa: SLF001
        xr=ds,
        basin='basin_a',
        target_variable='q',
        var_metrics=['NSE'],
        date_range=dates,
    )

    reduced = np.clip(np.median(samples, axis=-1), 0, None)
    assert results['NSE'] == pytest.approx(
        _nse(obs_vals.T.reshape(-1), reduced.T.reshape(-1))
    )
    assert results['NSE_lead1'] == pytest.approx(
        _nse(obs_vals[:, 0], reduced[:, 0])
    )


@pytest.mark.unit
@pytest.mark.parametrize('all_nan_variable', ['obs', 'sim'])
def test_evaluate_target_metrics_all_nan_sets_every_metric_to_nan(
    all_nan_variable: str, caplog: pytest.LogCaptureFixture
) -> None:
    """All-NaN obs or sim yields NaN metrics explicitly, without raising."""
    obs_vals = np.linspace(1.0, 10.0, 10)[:, None].repeat(2, axis=1)
    sim_vals = obs_vals.copy()
    if all_nan_variable == 'obs':
        obs_vals[:] = np.nan
    else:
        sim_vals[:] = np.nan
    ds, dates = _make_step_results(obs_vals, sim_vals, time_steps=[1, 2])

    with caplog.at_level('WARNING', logger='model.evaluation.tester'):
        results = _make_metric_tester()._evaluate_target_metrics(  # noqa: SLF001
            xr=ds,
            basin='basin_a',
            target_variable='q',
            var_metrics=['NSE', 'KGE'],
            date_range=dates,
        )

    assert set(results) == {
        'NSE',
        'KGE',
        'NSE_lead1',
        'NSE_lead2',
        'KGE_lead1',
        'KGE_lead2',
    }
    assert all(np.isnan(value) for value in results.values())
    assert 'Basin basin_a target q has all-NaN obs or sim' in caplog.text


@pytest.mark.integration
def test_evaluate_reports_pooled_and_per_lead_metrics_in_both_periods(
    make_minimal_config: Callable[..., Config], tmp_path: Path
) -> None:
    """Validation logs and test CSVs carry pooled and per-lead metrics.

    With `lead_time == 2` and `predict_last_n == 3` the predicted steps are
    `time_step = 0, 1, 2`. The un-suffixed `NSE` must equal the NSE of the
    pairs pooled over all three steps of the saved results, and each suffixed
    key the NSE of its own step.
    """
    run_dir = tmp_path / 'per_lead_run'
    cfg = make_minimal_config(
        {
            'experiment_name': 'per_lead_test',
            'run_dir': run_dir,
            'base_run_dir': run_dir,
            'predict_last_n': 3,
            'validate_every': 1,
            'validate_n_random_basins': 0,
            'metrics': ['NSE'],
        }
    )
    trainer = BaseTrainer(cfg)
    trainer.initialize_training()

    logged_steps: list[dict] = []
    real_log_step = trainer.experiment_logger.log_step

    def _spy_log_step(**kwargs: object) -> None:
        logged_steps.append(kwargs)
        real_log_step(**kwargs)

    trainer.experiment_logger.log_step = _spy_log_step
    trainer.train_and_validate()

    per_lead_keys = {'NSE', 'NSE_hindcast', 'NSE_lead1', 'NSE_lead2'}
    metric_steps = [step for step in logged_steps if 'NSE' in step]
    assert metric_steps, 'validation logged no metrics'
    assert all(set(step) == per_lead_keys for step in metric_steps)
    validation_csv = (
        trainer.cfg.run_dir
        / 'validation'
        / 'model_epoch001'
        / 'validation_metrics.csv'
    )
    validation_metrics = pd.read_csv(validation_csv, index_col='basin')
    assert set(validation_metrics.columns) == per_lead_keys

    tester = get_tester(
        cfg=trainer.cfg,
        run_dir=trainer.cfg.run_dir,
        period='test',
        init_model=True,
    )
    tester.evaluate(save_results=True, metrics=['NSE'])

    eval_dir = trainer.cfg.run_dir / 'test' / 'model_epoch001'
    test_metrics = pd.read_csv(eval_dir / 'test_metrics.csv', index_col='basin')
    assert set(test_metrics.columns) == per_lead_keys
    saved = xr.open_zarr(eval_dir / 'test_results.zarr', consolidated=False)
    assert list(saved['time_step'].to_numpy()) == [0, 1, 2]
    for basin in test_metrics.index:
        obs = saved['streamflow_obs'].sel(basin=basin).to_numpy()
        sim = saved['streamflow_sim'].sel(basin=basin).to_numpy()
        assert test_metrics.loc[basin, 'NSE'] == pytest.approx(
            _nse(obs.reshape(-1), sim.reshape(-1)), rel=1e-5, abs=1e-5
        )
        for step_idx, suffix in enumerate(['hindcast', 'lead1', 'lead2']):
            assert test_metrics.loc[basin, f'NSE_{suffix}'] == pytest.approx(
                _nse(obs[:, step_idx], sim[:, step_idx]), rel=1e-5, abs=1e-5
            )
