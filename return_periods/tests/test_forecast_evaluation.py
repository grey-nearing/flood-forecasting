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

"""Unit tests for return-period forecast evaluation and PR #110 evaluation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from model.datasetzoo.multimet import (
    Multimet,
    MultimetDataLoader,
    _convert_to_tensor,
)
from model.evaluation.metrics import _check_all_nan, _get_fdc
from model.evaluation.tester import BaseTester
from model.evaluation.utils import metrics_to_dataframe
from model.tests.test_multimet import (
    _day_offset_dataset,
    _get_default_config_attributes,
    _write_multimet_stores,
)
from model.utils import config as config_mod
from model.utils.errors import AllNaNError
from return_periods import (
    EXCEEDANCE_WEIGHT_SUM,
    EXCEEDANCE_WEIGHTS,
    PRIMARY_WINDOW_DAYS,
    RETURN_PERIODS,
    compute_precision_recall_f1_single_rp,
    compute_series_return_period_thresholds,
    evaluate_forecast_return_periods,
    extract_peaks_utilities,
    get_up_crossing_indices,
)
from return_periods.exceptions import InvalidFlowValueError
from return_periods.return_period_calculator import ReturnPeriodCalculator

_EXPECTED_CROSSING_COUNT = 2
_SAMPLE_RNG_SEED = 42
_SYNTHETIC_YEARS = 12
_HIGH_RP_TEST_THRESH = 5.0
_ZERO_EVENT_THRESH = 1000.0


def test_get_up_crossing_indices_handles_edges_and_interior_nans() -> None:
    """Verify upward crossings require a finite below-threshold predecessor."""
    series = np.array([1.0, 5.0, 6.0, 2.0, 5.5, np.nan, 7.0])
    crossings = get_up_crossing_indices(series, _HIGH_RP_TEST_THRESH)
    np.testing.assert_array_equal(crossings, np.array([1, 4]))

    index_zero_series = np.array([6.0, 7.0, 1.0, 5.0])
    np.testing.assert_array_equal(
        get_up_crossing_indices(index_zero_series, _HIGH_RP_TEST_THRESH),
        np.array([0, 3]),
    )

    assert len(get_up_crossing_indices(series, np.nan)) == 0
    assert len(get_up_crossing_indices(np.array([]), _HIGH_RP_TEST_THRESH)) == 0


def test_compute_precision_recall_f1_single_rp_matches_and_zero_events() -> (
    None
):
    """Verify event matching within +/- window_days and zero-event NaN rules."""
    obs = np.zeros(20)
    sim = np.zeros(20)
    obs[5] = 10.0
    obs[15] = 10.0
    sim[6] = 8.0
    prec, rec, f1_val, n_obs, n_sim = compute_precision_recall_f1_single_rp(
        obs,
        sim,
        obs_thresh=_HIGH_RP_TEST_THRESH,
        sim_thresh=_HIGH_RP_TEST_THRESH,
        window_days=PRIMARY_WINDOW_DAYS,
    )
    assert n_obs == _EXPECTED_CROSSING_COUNT
    assert n_sim == 1
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(0.5)
    assert f1_val == pytest.approx(2.0 / 3.0)

    # Zero events in both obs and sim -> all metrics are NaN (not 1.0).
    p_zero, r_zero, f1_zero, n_o0, n_s0 = compute_precision_recall_f1_single_rp(
        np.zeros(20),
        np.zeros(20),
        obs_thresh=_HIGH_RP_TEST_THRESH,
        sim_thresh=_HIGH_RP_TEST_THRESH,
    )
    assert (n_o0, n_s0) == (0, 0)
    assert np.isnan(p_zero)
    assert np.isnan(r_zero)
    assert np.isnan(f1_zero)

    # Real observed flood but zero simulated crossings -> recall=0, f1=0, P=NaN.
    p_miss, r_miss, f1_miss, n_om, n_sm = compute_precision_recall_f1_single_rp(
        obs,
        np.zeros(20),
        obs_thresh=_HIGH_RP_TEST_THRESH,
        sim_thresh=_HIGH_RP_TEST_THRESH,
    )
    assert (n_om, n_sm) == (_EXPECTED_CROSSING_COUNT, 0)
    assert np.isnan(p_miss)
    assert r_miss == pytest.approx(0.0)
    assert f1_miss == pytest.approx(0.0)


def test_compute_precision_recall_f1_nan_windows_and_validation() -> None:
    """Verify NaN reference windows and input validation in single-RP eval."""
    obs = np.zeros(15)
    sim = np.zeros(15)
    obs[4] = 10.0
    obs[11] = np.nan
    sim[4] = 10.0
    sim[10] = 10.0
    prec, rec, f1_val, n_obs, n_sim = compute_precision_recall_f1_single_rp(
        obs,
        sim,
        obs_thresh=_HIGH_RP_TEST_THRESH,
        sim_thresh=_HIGH_RP_TEST_THRESH,
        window_days=1,
    )
    # Sim crossing at index 10 has NaN in obs[9:12], so it is dropped from n_sim
    assert n_obs == 1
    assert n_sim == 1
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
    assert f1_val == pytest.approx(1.0)

    # If all sim events are dropped due to NaNs in obs, F1 is NaN (not 0.0).
    obs_gap = np.zeros(10)
    sim_gap = np.zeros(10)
    obs_gap[2] = 10.0
    obs_gap[7] = np.nan
    sim_gap[7] = 10.0
    p_gap, r_gap, f1_gap, n_og, n_sg = compute_precision_recall_f1_single_rp(
        obs_gap,
        sim_gap,
        obs_thresh=_HIGH_RP_TEST_THRESH,
        sim_thresh=_HIGH_RP_TEST_THRESH,
        window_days=1,
    )
    assert (n_og, n_sg) == (1, 0)
    assert np.isnan(p_gap)
    assert r_gap == pytest.approx(0.0)
    assert np.isnan(f1_gap)

    with pytest.raises(ValueError, match='equal length'):
        compute_precision_recall_f1_single_rp(
            np.zeros(5), np.zeros(6), _HIGH_RP_TEST_THRESH, _HIGH_RP_TEST_THRESH
        )
    with pytest.raises(ValueError, match='window_days'):
        compute_precision_recall_f1_single_rp(
            np.zeros(5),
            np.zeros(5),
            _HIGH_RP_TEST_THRESH,
            _HIGH_RP_TEST_THRESH,
            window_days=-1,
        )


def test_compute_series_return_period_thresholds_matches_calculator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify threshold fitting matches ReturnPeriodCalculator and validates."""
    rng = np.random.default_rng(_SAMPLE_RNG_SEED)
    dates = pd.date_range('2000-01-01', '2011-12-31', freq='1D')
    flow = rng.lognormal(mean=1.2, sigma=0.7, size=len(dates))

    thresholds = compute_series_return_period_thresholds(
        flow, dates, fitter='gema'
    )
    peaks = extract_peaks_utilities.extract_annual_maximums(
        hydrograph_series=pd.Series(flow, index=dates),
        max_missing_days_in_year=100,
        water_year_start_month=1,
    )
    expected_q = ReturnPeriodCalculator(
        peaks_series=peaks, fitter='gema'
    ).flow_values_from_return_periods(RETURN_PERIODS)
    for rp, exp_val in zip(RETURN_PERIODS, expected_q, strict=True):
        assert thresholds[rp] == pytest.approx(float(exp_val))

    q_list = [thresholds[rp] for rp in RETURN_PERIODS]
    assert all(q_list[i] <= q_list[i + 1] for i in range(len(q_list) - 1))

    # Negative flow raises InvalidFlowValueError rather than silently clipping.
    bad_flow = flow.copy()
    bad_flow[10] = -1.0
    with pytest.raises(InvalidFlowValueError):
        compute_series_return_period_thresholds(bad_flow, dates)

    # Unknown fitter raises ValueError.
    with pytest.raises(ValueError, match='Unknown fitter'):
        compute_series_return_period_thresholds(flow, dates, fitter='invalid')

    # Fewer than 5 positive annual peaks returns NaNs and logs a warning.
    short_dates = pd.date_range('2000-01-01', '2002-12-31', freq='1D')
    short_flow = rng.lognormal(mean=1.0, sigma=0.5, size=len(short_dates))
    nan_thresholds = compute_series_return_period_thresholds(
        short_flow, short_dates
    )
    assert all(np.isnan(val) for val in nan_thresholds.values())
    assert 'Insufficient positive annual peaks' in caplog.text


def test_evaluate_forecast_return_periods_renormalizes_undefined_rps() -> None:
    """Verify multi-lead evaluation and renormalization for empty high RPs."""
    rng = np.random.default_rng(_SAMPLE_RNG_SEED)
    dates = pd.date_range('2000-01-01', '2011-12-31', freq='1D')
    obs_1d = rng.lognormal(mean=1.0, sigma=0.8, size=len(dates))
    sim_1d = obs_1d * rng.uniform(0.95, 1.05, size=len(dates))
    obs_2d = np.stack([obs_1d, obs_1d], axis=1)
    sim_2d = np.stack([sim_1d, sim_1d], axis=1)

    thresholds = compute_series_return_period_thresholds(obs_1d, dates)
    metrics = evaluate_forecast_return_periods(
        obs_2d,
        sim_2d,
        dates,
        obs_thresholds_gema=thresholds,
        lead_indices=[0, 1],
        windows=[0, PRIMARY_WINDOW_DAYS],
        compute_lp3_lead0=True,
    )
    assert np.isclose(EXCEEDANCE_WEIGHT_SUM, sum(EXCEEDANCE_WEIGHTS.values()))
    assert 0.0 <= float(metrics['RP_ExpRecallNorm']) <= 1.0
    assert 0.0 <= float(metrics['RP_ExpF1Norm_lead1']) <= 1.0
    assert 0.0 <= float(metrics['RP_ExpF1Norm_w0_lead2']) <= 1.0
    assert 'RP_ExpF1Norm_lp3' in metrics

    # If no events exist at any RP (threshold above all flows), summary is NaN.
    high_thresh = dict.fromkeys(RETURN_PERIODS, _ZERO_EVENT_THRESH)
    zero_metrics = evaluate_forecast_return_periods(
        np.ones((len(dates), 1)),
        np.ones((len(dates), 1)),
        dates,
        obs_thresholds_gema=high_thresh,
        compute_lp3_lead0=False,
    )
    assert np.isnan(float(zero_metrics['RP_ExpRecallNorm']))
    assert np.isnan(float(zero_metrics['RP_ExpPrecisionNorm']))
    assert np.isnan(float(zero_metrics['RP_ExpF1Norm']))

    with pytest.raises(ValueError, match='lead_indices'):
        evaluate_forecast_return_periods(
            obs_2d, sim_2d, dates, lead_indices=[5]
        )


def _build_multimet_config(
    tmp_path: Path,
    case_spec: tuple[int, bool, bool, str],
) -> config_mod.Config:
    """Create a synthetic Multimet store and Config in `tmp_path`."""
    forecast_overlap, timestep_counter, include_forecast, loss_name = case_spec
    basins = ['basin_01', 'basin_02']
    dates = pd.date_range('1999-12-01', '2000-02-15', freq='D')
    lead_times = [np.timedelta64(k, 'D') for k in range(1, 5)]
    ds_xr = _day_offset_dataset(basins, dates, lead_times)

    cfg_path = tmp_path / 'cfg.yml'
    cfg_path.write_text('dataset: dummy\n')
    cfg = config_mod.Config(cfg_path)
    attrs = _get_default_config_attributes(
        base_run_dir=tmp_path / 'run_base',
        run_dir=tmp_path / 'run',
        train_basin_file=tmp_path / 'train_basins.txt',
        validation_basin_file=tmp_path / 'val_basins.txt',
        test_basin_file=tmp_path / 'test_basins.txt',
    )
    cfg.update_config(attrs)
    cfg.base_run_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    _write_multimet_stores(tmp_path / 'stores', cfg, ds_xr, basins)

    identity_norm = {'centering': 'none', 'scaling': 'none'}
    updates: dict[str, object] = {
        'seq_length': 5,
        'lead_time': 4 if include_forecast else 0,
        'predict_last_n': 5 if include_forecast else 1,
        'forecast_overlap': forecast_overlap if include_forecast else 0,
        'timestep_counter': timestep_counter,
        'loss': loss_name,
        'hindcast_inputs': (
            ['era5land_2d', 'hres_3d'] if include_forecast else ['era5land_2d']
        ),
        'forecast_inputs': ['hres_3d'] if include_forecast else [],
        'custom_normalization': {
            'era5land_2d': identity_norm,
            'hres_3d': identity_norm,
            'target_v1': identity_norm,
        },
        'train_start_date': ['01/01/2000'],
        'train_end_date': ['31/01/2000'],
        'lazy_load': False,
    }
    cfg.update_config(updates)
    return cfg


@pytest.mark.parametrize(
    'case_spec',
    [
        (2, True, True, 'mse'),
        (0, True, True, 'mse'),
        (0, False, True, 'mse'),
        (0, True, False, 'mse'),
        (1, False, True, 'nse'),
    ],
)
def test_multimet_get_batch_matches_getitem_collation(
    tmp_path: Path,
    case_spec: tuple[int, bool, bool, str],
) -> None:
    """Verify `Multimet.get_batch` and `MultimetDataLoader` match collation."""
    cfg = _build_multimet_config(tmp_path, case_spec)
    ds = Multimet(cfg=cfg, is_train=True, period='train', compute_scaler=True)

    sample_indices = list(range(min(6, len(ds))))
    fast_batch = ds.get_batch(np.asarray(sample_indices, dtype=np.int64))
    ref_samples = [
        {key: _convert_to_tensor(key, val) for key, val in ds[idx].items()}
        for idx in sample_indices
    ]
    ref_batch = ds.collate_fn(ref_samples)

    assert set(fast_batch.keys()) == set(ref_batch.keys())
    for key, ref_val in ref_batch.items():
        fast_val = fast_batch[key]
        if isinstance(ref_val, dict):
            assert isinstance(fast_val, dict)
            assert set(fast_val.keys()) == set(ref_val.keys())
            for sub_key, ref_tensor in ref_val.items():
                torch.testing.assert_close(fast_val[sub_key], ref_tensor)
        elif isinstance(ref_val, torch.Tensor):
            assert isinstance(fast_val, torch.Tensor)
            torch.testing.assert_close(fast_val, ref_val)
        else:
            np.testing.assert_array_equal(fast_val, ref_val)

    # Verify MultimetDataLoader and tuple/empty batch handling.
    assert ds.get_batch([]) == {}
    assert set(ds.get_batch(tuple(sample_indices)).keys()) == set(
        ref_batch.keys()
    )
    loader = MultimetDataLoader(
        ds,
        lazy_load=False,
        logging_level=30,
        batch_size=3,
        shuffle=False,
        collate_fn=ds.collate_fn,
    )
    batches = list(loader)
    assert len(batches) == len(loader)


class _StubTester(BaseTester):
    """Minimal `BaseTester` subclass for testing per-step metric evaluation."""

    def __init__(self) -> None:
        """Initialize stub configuration without loading datasets or weights."""
        self.cfg = SimpleNamespace(
            target_variables=['q'],
            clip_targets_to_zero=['q'],
            head='cmal',
            tester_sample_reduction=config_mod.TesterSamplesReduction.MEAN,
        )

    def _get_loss_obj(self) -> None:
        """Return None for stub loss object."""
        return

    def _load_weights(self, epoch: int | None = None) -> None:
        """No-op weight loader for stub tester."""

    def _get_plots(
        self,
        obs: np.ndarray,
        sim: np.ndarray,
        title: str,
    ) -> tuple[object, str]:
        """Return placeholder plot tuple for stub tester."""
        return (obs, sim), title

    def evaluate_steps(
        self,
        ds: xr.Dataset,
        target_variable: str,
        var_metrics: list[str],
        date_range: pd.DatetimeIndex,
    ) -> dict[str, float]:
        """Invoke `_evaluate_per_step_metrics` on `ds`."""
        return self._evaluate_per_step_metrics(
            xr=ds,
            target_variable=target_variable,
            var_metrics=var_metrics,
            date_range=date_range,
        )


def test_tester_per_step_metrics_and_dataframe_consistency() -> None:
    """Verify per-step metric naming, sample reduction, and CSV consistency."""
    dates = pd.date_range('2020-01-01', periods=10, freq='1D')
    obs_vals = np.broadcast_to(
        np.linspace(1.0, 10.0, 10)[:, None], (10, 3)
    ).copy()
    # Step 2 has all-NaN observations to exercise the warning precondition path.
    obs_vals[:, 2] = np.nan
    sim_vals = np.broadcast_to(
        obs_vals[:, :2, None].repeat(4, axis=2), (10, 2, 4)
    )
    sim_full = np.concatenate([sim_vals, np.ones((10, 1, 4))], axis=1)

    ds = xr.Dataset(
        data_vars={
            'q_obs': (('date', 'time_step'), obs_vals),
            'q_sim': (('date', 'time_step', 'samples'), sim_full),
        },
        coords={'date': dates, 'time_step': [0, 1, 2]},
    )

    stub_tester = _StubTester()
    step_metrics = stub_tester.evaluate_steps(
        ds=ds,
        target_variable='q',
        var_metrics=['NSE'],
        date_range=dates,
    )
    assert 'NSE_hindcast' in step_metrics
    assert 'NSE_lead1' in step_metrics
    assert 'NSE_lead2' in step_metrics
    assert step_metrics['NSE_hindcast'] == pytest.approx(1.0)
    assert step_metrics['NSE_lead1'] == pytest.approx(1.0)
    assert np.isnan(step_metrics['NSE_lead2'])

    # Verify metrics_to_dataframe keeps uniform columns across basins when one
    # basin has all-NaN obs and lacks extra per-lead keys.
    results = {
        'basin_empty': {'xr': ds},
        'basin_valid': {
            'xr': ds,
            'NSE': 1.0,
            **step_metrics,
        },
    }
    df = metrics_to_dataframe(results, metrics=['NSE'], targets=['q'])
    assert list(df.columns) == [
        'NSE',
        'NSE_hindcast',
        'NSE_lead1',
        'NSE_lead2',
    ]
    assert np.isnan(df.loc['basin_empty', 'NSE_lead1'])
    assert df.loc['basin_valid', 'NSE_lead1'] == pytest.approx(1.0)


def test_metrics_fdc_and_check_all_nan() -> None:
    """Verify `_get_fdc` and `_check_all_nan` handle NaNs accurately."""
    da = xr.DataArray([3.0, np.nan, 1.0, 5.0])
    fdc = _get_fdc(da)
    np.testing.assert_array_equal(fdc[:3], np.array([5.0, 3.0, 1.0]))
    assert np.isnan(fdc[3])

    with pytest.raises(AllNaNError):
        _check_all_nan(xr.DataArray([np.nan, np.nan]), xr.DataArray([1.0, 2.0]))
