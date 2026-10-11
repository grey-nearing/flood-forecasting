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

"""Tests for the Real-Time Forecasting & Data Assimilation Tab and Service."""

from __future__ import annotations

import http.client
import json
import shutil
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from frontend import (
    profile_manager,
    realtime_forecast_service,
)
from frontend import (
    tests as _isolated_profiles,  # noqa: F401
)
from frontend.config import FLOOD_FORECASTING_REPO_DIR
from frontend.server import EarthkitHydroHandler
from model.datautils.scaler import Scaler
from model.modelzoo import get_model
from model.utils.config import Config
from multimet.utils import cpc as cpc_utils

TEST_USER = 'test_fc_realtime_user'
TEST_BASIN = 'camels_01022500'
TEST_BASIN_2 = 'camels_01031500'
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
EXPECTED_NOWCAST_LEN = 14
EXPECTED_FORECAST_LEN = 10
SPINUP_DAYS = 365
PRETRAINED_MODEL_ID = 'google-floodhub-settings-110-epochs'
PRETRAINED_CHECKPOINT_NAME = 'model_epoch110.pt'
PRETRAINED_LEAD_TIME_DAYS = 7

SYNTHETIC_MODEL_ID = 'synthetic-mean-embedding-forecast-lstm'
SYNTHETIC_LEAD_TIME_DAYS = 7
SYNTHETIC_HIDDEN_SIZE = 8
SYNTHETIC_SEED = 42
SYNTHETIC_SCALER_BASINS = 16
SYNTHETIC_SCALER_DAYS = 120
SYNTHETIC_HINDCAST_INPUTS: dict[str, list[str]] = {
    'hres': ['hres_temperature_2m', 'hres_total_precipitation'],
    'imerg': ['imerg_precipitation'],
    'cpc': ['cpc_precipitation'],
}
SYNTHETIC_FORECAST_INPUTS: dict[str, list[str]] = {
    'hres': ['hres_temperature_2m', 'hres_total_precipitation'],
    'graphcast': ['graphcast_temperature_2m', 'graphcast_total_precipitation'],
}
# Static attribute name -> (min, max) of the synthetic scaler training sample.
SYNTHETIC_STATIC_ATTRIBUTE_RANGES: dict[str, tuple[float, float]] = {
    'area': (50.0, 5000.0),
    'aridity_ERA5_LAND': (0.3, 2.0),
    'aridity_FAO_PM': (0.3, 2.0),
    'frac_snow': (0.0, 0.6),
    'moisture_index_ERA5_LAND': (-0.8, 0.8),
    'moisture_index_FAO_PM': (-0.8, 0.8),
    'p_mean': (1.0, 6.0),
    'pet_mean_ERA5_LAND': (1.0, 5.0),
    'pet_mean_FAO_PM': (1.0, 5.0),
}

TEST_WATERSHED_FEATURE = {
    'type': 'Feature',
    'id': TEST_BASIN,
    'properties': {
        'catchment_id': TEST_BASIN,
        'name': 'Narraguagus River at Cherryfield, ME',
        'area_km2': 573.6,
        'source': 'delineated',
    },
    'geometry': {
        'type': 'Polygon',
        'coordinates': [
            [
                [-68.05, 44.60],
                [-67.85, 44.60],
                [-67.85, 44.80],
                [-68.05, 44.80],
                [-68.05, 44.60],
            ]
        ],
    },
}

TEST_WATERSHED_FEATURE_2 = {
    'type': 'Feature',
    'id': TEST_BASIN_2,
    'properties': {
        'catchment_id': TEST_BASIN_2,
        'name': 'Piscataquis River near Dover-Foxcroft, ME',
        'area_km2': 771.8,
        'source': 'delineated',
    },
    'geometry': {
        'type': 'Polygon',
        'coordinates': [
            [
                [-69.30, 45.10],
                [-69.10, 45.10],
                [-69.10, 45.25],
                [-69.30, 45.25],
                [-69.30, 45.10],
            ]
        ],
    },
}


def _populate_synthetic_static_attributes(
    pm: profile_manager.ProfileManager,
    username: str = TEST_USER,
) -> None:
    """Write explicit Caravan static attributes CSV for test basins."""
    attr_dir = pm.get_caravan_attributes_dir(username)
    attr_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                'gauge_id': TEST_BASIN,
                'area': 573.6,
                'aridity_ERA5_LAND': 0.55,
                'aridity_FAO_PM': 0.62,
                'frac_snow': 0.24,
                'moisture_index_ERA5_LAND': 0.45,
                'moisture_index_FAO_PM': 0.39,
                'p_mean': 3.45,
                'pet_mean_ERA5_LAND': 1.85,
                'pet_mean_FAO_PM': 2.10,
            },
            {
                'gauge_id': TEST_BASIN_2,
                'area': 771.8,
                'aridity_ERA5_LAND': 0.51,
                'aridity_FAO_PM': 0.58,
                'frac_snow': 0.29,
                'moisture_index_ERA5_LAND': 0.49,
                'moisture_index_FAO_PM': 0.42,
                'p_mean': 3.62,
                'pet_mean_ERA5_LAND': 1.75,
                'pet_mean_FAO_PM': 2.00,
            },
        ]
    )
    df.to_csv(attr_dir / 'caravan_hydroatlas_attributes.csv', index=False)


def _populate_pretrained_static_attributes(
    pm: profile_manager.ProfileManager,
    username: str,
) -> None:
    """Write every static attribute required by the shipped FloodHub run.

    Each attribute takes the training-period ``mean`` stored in the run's
    ``scaler.zarr`` (an average training basin), so ``TEST_BASIN`` has a
    complete, finite attribute row without copying a real Caravan table.
    """
    run_dir = realtime_forecast_service.DEFAULT_MODEL_RUN_DIR
    attr_features = list(Config(run_dir / 'config.yml').static_attributes)
    scaler_ds = xr.open_zarr(run_dir / 'scaler.zarr').load()
    row: dict[str, object] = {'gauge_id': TEST_BASIN, 'area': 573.6}
    for attr in attr_features:
        row[attr] = float(scaler_ds[attr].sel(parameter='mean'))
    attr_dir = pm.get_caravan_attributes_dir(username)
    attr_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        attr_dir / 'caravan_hydroatlas_attributes.csv', index=False
    )


def _write_synthetic_model_run_dir(run_dir: Path) -> Path:
    """Write a hermetic ``MeanEmbeddingForecastLSTM`` run directory.

    The directory holds what ``run train`` writes: ``scaler.zarr`` computed by
    the real ``Scaler`` from a seeded synthetic training sample, a seeded
    randomly initialised ``model_epoch001.pt`` checkpoint, and ``config.yml``.
    No training is performed because these tests exercise forecast-service
    mechanics (state gating, Cold/Hot-Start parity, ``NaN`` propagation,
    What-If overrides) rather than forecast skill.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SYNTHETIC_SEED)
    basins = [f'basin_{i:02d}' for i in range(SYNTHETIC_SCALER_BASINS)]
    dates = pd.date_range('2020-01-01', periods=SYNTHETIC_SCALER_DAYS, freq='D')
    dyn_shape = (len(basins), len(dates))
    dyn_dims = ('basin', 'date')

    dynamic_features = sorted(
        {
            *(f for feats in SYNTHETIC_HINDCAST_INPUTS.values() for f in feats),
            *(f for feats in SYNTHETIC_FORECAST_INPUTS.values() for f in feats),
        }
    )
    sample: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for feat in dynamic_features:
        if 'temperature' in feat:
            values = rng.normal(10.0, 9.0, size=dyn_shape)  # deg C
        else:
            values = rng.gamma(0.8, 5.0, size=dyn_shape)  # mm/day
        sample[feat] = (dyn_dims, values.astype(np.float32))
    sample['streamflow'] = (
        dyn_dims,
        rng.gamma(4.0, 0.5, size=dyn_shape).astype(np.float32),  # mm/day
    )
    for attr, (low, high) in SYNTHETIC_STATIC_ATTRIBUTE_RANGES.items():
        sample[attr] = (
            ('basin',),
            rng.uniform(low, high, size=len(basins)).astype(np.float32),
        )
    sample_ds = xr.Dataset(sample, coords={'basin': basins, 'date': dates})

    scaler = Scaler(
        scaler_dir=run_dir, calculate_scaler=True, dataset=sample_ds
    )
    scaler.scaler = scaler.scaler.compute()
    scaler.check_zero_scale()
    scaler.save()

    embedding = {
        'type': 'fc',
        'hiddens': [SYNTHETIC_HIDDEN_SIZE],
        'activation': ['tanh'],
        'dropout': 0.0,
    }
    cfg = Config(
        {
            'experiment_name': SYNTHETIC_MODEL_ID,
            'run_dir': run_dir,
            'dataset': 'multimet',
            'model': 'mean_embedding_forecast_lstm',
            'head': 'regression',
            'output_activation': 'linear',
            'output_dropout': 0.0,
            'hidden_size': SYNTHETIC_HIDDEN_SIZE,
            'seq_length': SPINUP_DAYS,
            'lead_time': SYNTHETIC_LEAD_TIME_DAYS,
            'forecast_overlap': SPINUP_DAYS,
            'predict_last_n': SYNTHETIC_LEAD_TIME_DAYS + 1,
            'timestep_counter': True,
            'static_attributes': list(SYNTHETIC_STATIC_ATTRIBUTE_RANGES),
            'hindcast_inputs': SYNTHETIC_HINDCAST_INPUTS,
            'forecast_inputs': SYNTHETIC_FORECAST_INPUTS,
            'target_variables': ['streamflow'],
            'statics_embedding': dict(embedding),
            'hindcast_embedding': dict(embedding),
            'forecast_embedding': dict(embedding),
            'device': 'cpu',
            'seed': SYNTHETIC_SEED,
            'compile': False,
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SYNTHETIC_SEED)
        model = get_model(cfg)
    model.eval()
    torch.save(model.state_dict(), run_dir / 'model_epoch001.pt')
    cfg.dump_config(run_dir)
    return run_dir


def _write_nowcast_zarr_stores(
    rt_dyn_dir: Path,
    basin_ids: list[str],
    nowcast_dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> None:
    """Write synthetic CPC and IMERG nowcast Zarr stores with trailing NaNs."""
    n_basins = len(basin_ids)
    cpc_precip = rng.uniform(
        0.0, 18.0, size=(n_basins, len(nowcast_dates))
    ).astype(np.float32)
    cpc_precip[:, -2:] = np.nan
    cpc_ds = xr.Dataset(
        {
            'cpc_precipitation': (('basin', 'date'), cpc_precip),
            'cpc_missing_fraction': (
                ('basin', 'date'),
                np.where(np.isnan(cpc_precip), 1.0, 0.0).astype(np.float32),
            ),
        },
        coords={'basin': basin_ids, 'date': nowcast_dates.to_numpy()},
    )
    cpc_path = rt_dyn_dir / 'CPC' / 'timeseries.zarr'
    if cpc_path.exists():
        shutil.rmtree(cpc_path)
    cpc_path.parent.mkdir(parents=True, exist_ok=True)
    cpc_ds.to_zarr(cpc_path, mode='w', zarr_format=2, consolidated=True)

    imerg_precip = rng.uniform(
        0.0, 22.0, size=(n_basins, len(nowcast_dates))
    ).astype(np.float32)
    imerg_precip[:, -1] = np.nan
    imerg_ds = xr.Dataset(
        {
            'imerg_precipitation': (('basin', 'date'), imerg_precip),
            'imerg_missing_fraction': (
                ('basin', 'date'),
                np.where(np.isnan(imerg_precip), 1.0, 0.0).astype(np.float32),
            ),
        },
        coords={'basin': basin_ids, 'date': nowcast_dates.to_numpy()},
    )
    imerg_path = rt_dyn_dir / 'IMERG' / 'timeseries.zarr'
    if imerg_path.exists():
        shutil.rmtree(imerg_path)
    imerg_path.parent.mkdir(parents=True, exist_ok=True)
    imerg_ds.to_zarr(imerg_path, mode='w', zarr_format=2, consolidated=True)


def _write_hres_zarr_store(
    rt_dyn_dir: Path,
    basin_ids: list[str],
    all_dates: pd.DatetimeIndex,
    t0: pd.Timestamp,
    rng: np.random.Generator,
) -> None:
    """Write synthetic HRES 3D Zarr store with kPa surface pressure."""
    n_basins = len(basin_ids)
    lead_times = pd.to_timedelta(
        np.arange(1, EXPECTED_FORECAST_LEN + 1), unit='D'
    )
    hres_vars = [
        'hres_temperature_2m',
        'hres_total_precipitation',
        'hres_surface_net_solar_radiation',
        'hres_surface_net_thermal_radiation',
        'hres_surface_pressure',
        'hres_specific_humidity',
        'hres_u_component_of_wind_10m',
        'hres_v_component_of_wind_10m',
    ]
    hres_data = {}
    t0_idx = int(np.where(all_dates == t0)[0][0])
    for v in hres_vars:
        arr = np.full(
            (n_basins, len(all_dates), EXPECTED_FORECAST_LEN),
            np.nan,
            dtype=np.float32,
        )
        if 'temperature' in v:
            arr[:, : t0_idx + 1, 0] = rng.uniform(
                2.0, 16.0, size=(n_basins, t0_idx + 1)
            )
            arr[:, t0_idx, :] = np.linspace(8.0, 14.5, EXPECTED_FORECAST_LEN)
        elif 'precipitation' in v:
            arr[:, : t0_idx + 1, 0] = rng.uniform(
                0.0, 15.0, size=(n_basins, t0_idx + 1)
            )
            arr[:, t0_idx, :] = np.array(
                [3.5, 12.0, 25.0, 8.0, 1.5, 0.0, 4.0, 9.5, 2.0, 0.5],
                dtype=np.float32,
            )
        elif 'pressure' in v:
            arr[:, : t0_idx + 1, 0] = 101.325
            arr[:, t0_idx, :] = 101.2
        else:
            arr[:, : t0_idx + 1, 0] = rng.uniform(
                0.5, 5.0, size=(n_basins, t0_idx + 1)
            )
            arr[:, t0_idx, :] = rng.uniform(
                0.5, 5.0, size=(n_basins, EXPECTED_FORECAST_LEN)
            )
        hres_data[v] = (('basin', 'date', 'lead_time'), arr)

    hres_ds = xr.Dataset(
        hres_data,
        coords={
            'basin': basin_ids,
            'date': all_dates.to_numpy(),
            'lead_time': lead_times.to_numpy(),
        },
    )
    hres_path = rt_dyn_dir / 'HRES' / 'timeseries.zarr'
    if hres_path.exists():
        shutil.rmtree(hres_path)
    hres_path.parent.mkdir(parents=True, exist_ok=True)
    hres_ds.to_zarr(hres_path, mode='w', zarr_format=2, consolidated=True)


def _populate_synthetic_realtime_zarr_stores(
    profile_dir: Path,
    basin_id: str | list[str] = TEST_BASIN,
    issue_date_str: str = '2026-03-30',
    *,
    include_extra_forecast_streams: bool = True,
) -> None:
    """Populate real-time Zarr stores with canonical variable names."""
    basin_ids = [basin_id] if isinstance(basin_id, str) else list(basin_id)
    n_basins = len(basin_ids)
    t0 = pd.Timestamp(issue_date_str)
    start_date = t0 - pd.Timedelta(days=SPINUP_DAYS)
    end_date = t0 + pd.Timedelta(days=EXPECTED_FORECAST_LEN - 1)
    nowcast_dates = pd.date_range(
        start_date, t0 - pd.Timedelta(days=1), freq='D'
    )
    all_dates = pd.date_range(start_date, end_date, freq='D')

    rt_dyn_dir = profile_dir / 'realtime' / 'dynamics'
    rt_dyn_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    _write_nowcast_zarr_stores(rt_dyn_dir, basin_ids, nowcast_dates, rng)
    _write_hres_zarr_store(rt_dyn_dir, basin_ids, all_dates, t0, rng)

    if not include_extra_forecast_streams:
        return

    lead_times = pd.to_timedelta(
        np.arange(1, EXPECTED_FORECAST_LEN + 1), unit='D'
    )
    t0_idx = int(np.where(all_dates == t0)[0][0])
    for prod_name, prefix in [
        ('ECMWF_AIFS', 'aifs'),
        ('NOAA_GFS', 'gfs'),
        ('GRAPHCAST', 'graphcast'),
    ]:
        p_arr = np.full(
            (n_basins, len(all_dates), EXPECTED_FORECAST_LEN),
            np.nan,
            dtype=np.float32,
        )
        t_arr = np.full(
            (n_basins, len(all_dates), EXPECTED_FORECAST_LEN),
            np.nan,
            dtype=np.float32,
        )
        p_arr[:, : t0_idx + 1, 0] = rng.uniform(
            0.0, 12.0, size=(n_basins, t0_idx + 1)
        )
        t_arr[:, : t0_idx + 1, 0] = rng.uniform(
            3.0, 15.0, size=(n_basins, t0_idx + 1)
        )
        p_arr[:, t0_idx, :] = np.linspace(1.0, 18.0, EXPECTED_FORECAST_LEN)
        t_arr[:, t0_idx, :] = np.linspace(6.0, 15.0, EXPECTED_FORECAST_LEN)
        if prod_name in ('ECMWF_AIFS', 'NOAA_GFS'):
            p_arr[:, t0_idx, 8] = np.nan
        extra_ds = xr.Dataset(
            {
                f'{prefix}_total_precipitation': (
                    ('basin', 'date', 'lead_time'),
                    p_arr,
                ),
                f'{prefix}_temperature_2m': (
                    ('basin', 'date', 'lead_time'),
                    t_arr,
                ),
            },
            coords={
                'basin': basin_ids,
                'date': all_dates.to_numpy(),
                'lead_time': lead_times.to_numpy(),
            },
        )
        extra_path = rt_dyn_dir / prod_name / 'timeseries.zarr'
        if extra_path.exists():
            shutil.rmtree(extra_path)
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        extra_ds.to_zarr(extra_path, mode='w', zarr_format=2, consolidated=True)


@pytest.mark.unit
class RealtimeForecastTabTest(unittest.TestCase):
    """End-to-end unit tests for the Real-Time Forecasting Tab & Service."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start an isolated HTTP server and write the synthetic model run."""
        cls.model_run_tmp = tempfile.mkdtemp(prefix='openhydronet_test_model_')
        cls.model_run_dir = _write_synthetic_model_run_dir(
            Path(cls.model_run_tmp) / SYNTHETIC_MODEL_ID
        )
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), EarthkitHydroHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down the isolated HTTP server and remove the model run."""
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.model_run_tmp)

    def setUp(self) -> None:
        """Initialize an isolated user profile with watersheds and statics."""
        self.pm = profile_manager.get_profile_manager()
        self.test_user = f'{TEST_USER}_{self._testMethodName}'
        self.profile_dir = self.pm.get_profile_dir(self.test_user)
        if self.profile_dir.exists():
            shutil.rmtree(self.profile_dir)
        self.profile_dir = self.pm.get_profile_dir(self.test_user)
        self.pm.save_watersheds(
            [TEST_WATERSHED_FEATURE, TEST_WATERSHED_FEATURE_2],
            username=self.test_user,
        )
        _populate_synthetic_static_attributes(self.pm, username=self.test_user)

    def tearDown(self) -> None:
        """Clean up the isolated user profile directory."""
        if self.profile_dir.exists():
            shutil.rmtree(self.profile_dir)

    def _request(
        self, method: str, path: str, body: dict | None = None
    ) -> tuple[int, dict]:
        """Issue an HTTP request against the local test server."""
        headers = {'X-User-Profile': self.test_user}
        data = None
        if body is not None:
            data = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        conn = http.client.HTTPConnection('127.0.0.1', self.port)
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode('utf-8')
        status = resp.status
        conn.close()
        return status, json.loads(raw)

    def _post_run_model(self, body: dict) -> tuple[int, dict]:
        """POST ``/api/forecast/run-model`` against the synthetic model run."""
        return self._request(
            'POST',
            '/api/forecast/run-model',
            {**body, 'model_run_dir': str(self.model_run_dir)},
        )

    def test_ui_html_contains_simplified_forecast_controls_and_legacy_ids(
        self,
    ) -> None:
        """Verify #tab-forecasting has multi-catchment and legacy IDs."""
        conn = http.client.HTTPConnection('127.0.0.1', self.port)
        conn.request('GET', '/')
        resp = conn.getresponse()
        html = resp.read().decode('utf-8')
        conn.close()

        required_ids = [
            'tab-forecasting',
            'fcCatchmentSelectorList',
            'fcSelectAllBasinsBtn',
            'fcSelectNoneBasinsBtn',
            'fcSelectedCatchmentCountBadge',
            'fcInputProductsChecklist',
            'fcModeColdBtn',
            'fcModeHotBtn',
            'fcIssueDateText',
            'fcSavedStateDateText',
            'fcExpectedStateDateText',
            'fcStateValidationBanner',
            'fcFetchRealtimeBtn',
            'fcRunModelBtn',
            'fcRunBlockedReason',
            'fcNowcastProductSelect',
            'fcForecastProductSelect',
            'fcForecastVariableSelect',
            'fcLeadTimeSummaryTable',
            'fcHydroCatchmentTabs',
            'whatIfHyetographCanvas',
            'masterHydrographCanvas',
            'fcHydroPeakBadge',
            'fetchForecastBtn',
            'executionModeSelect',
            'modelCheckpointSelect',
            'fcBasinSelect',
        ]
        for dom_id in required_ids:
            assert f'id="{dom_id}"' in html, f'Missing DOM ID: {dom_id}'

        for expected_token in [
            'for="fcBasinSelect"',
            'fcIssueDateBoundaryPlugin',
            'buildSharedForecastTimeline',
            'bridgeHandoffAtT0',
            'LEFT_AXIS_WIDTH',
            'RIGHT_AXIS_WIDTH',
            'yRightAlign',
        ]:
            assert expected_token in html, (
                f'Missing expected UI token: {expected_token}'
            )

        sidebar_end = html.index(
            '</aside>', html.index('id="forecastingSidebar"')
        )
        basin_select_pos = html.index('id="fcBasinSelect"')
        assert basin_select_pos > sidebar_end

        default_model_id = realtime_forecast_service.DEFAULT_MODEL_RUN_DIR.name
        assert (
            f'<option value="{default_model_id}" selected>{default_model_id}'
            '</option>' in html
        )
        assert '5-basin-example' not in html

    def test_default_model_is_shipped_pretrained_floodhub_checkpoint(
        self,
    ) -> None:
        """Verify the default run dir is the pretrained FloodHub checkpoint."""
        run_dir = realtime_forecast_service.DEFAULT_MODEL_RUN_DIR
        assert run_dir == (
            FLOOD_FORECASTING_REPO_DIR
            / 'model'
            / 'pretrained-models'
            / PRETRAINED_MODEL_ID
        )
        assert (run_dir / 'config.yml').is_file()
        assert (run_dir / 'scaler.zarr').is_dir()
        assert [p.name for p in sorted(run_dir.glob('model_epoch*.pt'))] == [
            PRETRAINED_CHECKPOINT_NAME
        ]

        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )
        _populate_pretrained_static_attributes(self.pm, username=self.test_user)

        # No model_run_dir in the request: the service must load the shipped
        # checkpoint, its scaler, and its 84 static attributes end-to-end.
        code, res = self._request(
            'POST',
            '/api/forecast/run-model',
            {'catchment_id': TEST_BASIN, 'mode': 'coldstart'},
        )
        assert code == HTTP_OK, f'Default-model cold-start failed: {res}'
        assert res['status'] == 'success'
        assert res['model_id'] == PRETRAINED_MODEL_ID
        assert res['checkpoint_path'] == str(
            run_dir / PRETRAINED_CHECKPOINT_NAME
        )
        assert res['lead_time_days'] == PRETRAINED_LEAD_TIME_DAYS
        assert len(res['forecast']) == PRETRAINED_LEAD_TIME_DAYS
        assert res['saved_state']['state_date'] == '2026-03-29'
        discharge = np.array(
            [pt['discharge_cms'] for pt in res['forecast']], dtype=np.float64
        )
        assert np.all(np.isfinite(discharge))
        assert np.all(discharge >= 0.0)

    def test_multi_stream_discovery_and_strict_no_fake_data_nan_preservation(
        self,
    ) -> None:
        """Verify multi-stream discovery, variable metadata, and NaNs."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )

        status_code, payload = self._request(
            'GET', f'/api/forecast/status?catchment_id={TEST_BASIN}'
        )
        assert status_code == HTTP_OK
        assert payload['status'] == 'success'
        assert payload['issue_date'] == '2026-03-30'
        assert payload['expected_state_date'] == '2026-03-29'

        nowcast_products = payload['nowcast_products']
        for prod_key in ('CPC', 'IMERG', 'HRES_1D'):
            assert prod_key in nowcast_products

        cpc_series = nowcast_products['CPC']['series']
        assert len(cpc_series) == EXPECTED_NOWCAST_LEN
        assert cpc_series[-1]['date'] == '2026-03-29'
        assert cpc_series[-1]['precip_mm'] is None
        assert cpc_series[-2]['precip_mm'] is None
        assert cpc_series[-3]['precip_mm'] is not None

        imerg_series = nowcast_products['IMERG']['series']
        assert len(imerg_series) == EXPECTED_NOWCAST_LEN
        assert imerg_series[-1]['precip_mm'] is None
        assert imerg_series[-2]['precip_mm'] is not None

        forecast_products = payload['forecast_products']
        for fc_key in ('HRES', 'ECMWF_AIFS', 'NOAA_GFS'):
            assert fc_key in forecast_products

        hres_prod = forecast_products['HRES']
        assert hres_prod['default_variable_mode'] == 'precip_and_temp'
        assert hres_prod['has_precip'] is True
        assert hres_prod['has_temp'] is True
        hres_var_names = {v['name'] for v in hres_prod['variables']}
        assert {
            'hres_total_precipitation',
            'hres_temperature_2m',
            'hres_surface_net_solar_radiation',
            'hres_surface_pressure',
            'hres_u_component_of_wind_10m',
        }.issubset(hres_var_names)

        hres_fc = hres_prod['series']
        assert len(hres_fc) == EXPECTED_FORECAST_LEN
        assert hres_fc[0]['lead_time_days'] == 1
        assert hres_fc[0]['date'] == '2026-03-30'
        assert hres_fc[0]['is_issue_date'] is True
        assert hres_fc[0]['precip_mm'] == pytest.approx(3.5, abs=1e-3)
        assert hres_fc[0]['temperature_c'] == pytest.approx(8.0, abs=1e-3)
        assert hres_fc[0]['values']['hres_surface_pressure'] == pytest.approx(
            101.2, abs=1e-2
        )
        assert hres_fc[-1]['lead_time_days'] == EXPECTED_FORECAST_LEN
        assert hres_fc[-1]['date'] == '2026-04-08'
        assert hres_fc[-1]['is_issue_date'] is False
        assert hres_fc[-1]['temperature_c'] == pytest.approx(14.5, abs=1e-3)

        assert len(hres_prod['hindcast_series']) == EXPECTED_NOWCAST_LEN
        assert (
            'hres_surface_net_solar_radiation'
            in hres_prod['hindcast_series'][-1]['values']
        )

        aifs_fc = forecast_products['ECMWF_AIFS']['series']
        assert aifs_fc[8]['precip_mm'] is None
        assert aifs_fc[8]['temperature_c'] is not None

    def test_state_date_validation_and_cold_to_hot_start_model_execution(
        self,
    ) -> None:
        """Verify Hot-Start blocking when state is missing/stale and parity."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )

        # 1. With no saved state file, Hot-Start must be blocked
        code, status_payload = self._request(
            'GET', f'/api/forecast/status?catchment_id={TEST_BASIN}'
        )
        assert code == HTTP_OK
        assert status_payload['saved_state']['exists'] is False
        assert status_payload['is_state_from_previous_day'] is False
        assert status_payload['can_run_hotstart'] is False
        assert status_payload['can_run_coldstart'] is True

        code, blocked_res = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'hotstart'},
        )
        assert code == HTTP_BAD_REQUEST
        assert (
            'cannot run model in hot-start mode'
            in blocked_res['message'].lower()
        )

        # 2. If a saved state file is outside the 365-day hindcast window, block
        state_path = realtime_forecast_service.get_state_file_path(
            self.test_user, TEST_BASIN
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            state_path,
            h_n=np.zeros((1, 1, 64), dtype=np.float32),
            c_n=np.zeros((1, 1, 64), dtype=np.float32),
            date=np.array('2024-01-01', dtype='U10'),
        )
        code, stale_status = self._request(
            'GET', f'/api/forecast/status?catchment_id={TEST_BASIN}'
        )
        assert code == HTTP_OK
        assert stale_status['saved_state']['exists'] is True
        assert stale_status['saved_state']['state_date'] == '2024-01-01'
        assert stale_status['is_state_from_previous_day'] is False
        assert stale_status['can_run_hotstart'] is False

        code, stale_run_res = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'hotstart'},
        )
        assert code == HTTP_BAD_REQUEST
        assert '2024-01-01' in stale_run_res['message']
        assert '2026-03-29' in stale_run_res['message']

        # 3. Run Cold-Start model: saves state stamped 2026-03-29
        code, cold_res = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'coldstart'},
        )
        assert code == HTTP_OK, f'Cold-start failed: {cold_res}'
        assert cold_res['status'] == 'success'
        assert cold_res['mode'] == 'coldstart'
        assert cold_res['issue_date'] == '2026-03-30'
        assert len(cold_res['forecast']) == cold_res['lead_time_days']
        assert len(cold_res['hindcast_tail']) > 0
        assert cold_res['saved_state']['state_date'] == '2026-03-29'
        assert cold_res['post_run_status']['is_state_from_previous_day'] is True
        assert cold_res['post_run_status']['can_run_hotstart'] is True

        # Deterministic regression head must NOT fabricate quantile spreads
        for pt in cold_res['forecast']:
            assert pt['q05_cms'] is None
            assert pt['q25_cms'] is None
            assert pt['q75_cms'] is None
            assert pt['q95_cms'] is None

        # 4. Run Hot-Start model: should succeed and match Cold-Start forecast
        code, hot_res = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'hotstart'},
        )
        assert code == HTTP_OK, f'Hot-start failed: {hot_res}'
        assert hot_res['status'] == 'success'
        assert hot_res['mode'] == 'hotstart'
        assert len(hot_res['forecast']) == hot_res['lead_time_days']

        cold_q = [pt['discharge_cms'] for pt in cold_res['forecast']]
        hot_q = [pt['discharge_cms'] for pt in hot_res['forecast']]
        np.testing.assert_allclose(cold_q, hot_q, rtol=1e-4, atol=1e-4)

    def test_hotstart_advances_lagging_state_from_t0_minus_2_to_t0_minus_1(
        self,
    ) -> None:
        """Verify Hot-Start advances a state from t0-2 days to t0-1 day."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )

        # First run Cold-Start to obtain a valid state array structure
        code, _ = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'coldstart'},
        )
        assert code == HTTP_OK

        # Re-stamp the state file on disk as t0 - 2 days (2026-03-28)
        state_path = realtime_forecast_service.get_state_file_path(
            self.test_user, TEST_BASIN
        )
        with np.load(state_path, allow_pickle=False) as st_npz:
            payload = {k: st_npz[k] for k in st_npz.files}
        payload['date'] = np.array('2026-03-28', dtype='U10')
        np.savez_compressed(state_path, **payload)

        # Status should allow Hot-Start because 2026-03-28 is within window
        code, status_res = self._request(
            'GET', f'/api/forecast/status?catchment_id={TEST_BASIN}'
        )
        assert code == HTTP_OK
        assert status_res['saved_state']['state_date'] == '2026-03-28'
        assert status_res['is_state_from_previous_day'] is False
        assert status_res['can_run_hotstart'] is True

        # Running Hot-Start should advance state through 2026-03-29 and save it
        code, hot_res = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'hotstart'},
        )
        assert code == HTTP_OK, f'Incremental Hot-Start failed: {hot_res}'
        assert hot_res['saved_state']['state_date'] == '2026-03-29'
        assert len(hot_res['forecast']) == hot_res['lead_time_days']
        assert all(
            pt['discharge_cms'] is not None for pt in hot_res['forecast']
        )

        inspected = realtime_forecast_service.inspect_saved_state(
            self.test_user, TEST_BASIN
        )
        assert inspected['state_date'] == '2026-03-29'

    def test_all_nan_hres_forecast_inputs_yield_nan_discharge_predictions(
        self,
    ) -> None:
        """Verify all-NaN forecast inputs produce NaN discharge, not 0-fill."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
            include_extra_forecast_streams=False,
        )

        # Overwrite HRES forecast on t0 (2026-03-30) with all NaNs
        hres_path = (
            self.profile_dir
            / 'realtime'
            / 'dynamics'
            / 'HRES'
            / 'timeseries.zarr'
        )
        ds = xr.open_zarr(hres_path, consolidated=False).load()
        t0 = pd.Timestamp('2026-03-30')
        for var_name in ds.data_vars:
            ds[var_name].loc[{'date': t0}] = np.nan
        shutil.rmtree(hres_path)
        ds.to_zarr(hres_path, mode='w', zarr_format=2, consolidated=True)

        res = realtime_forecast_service.run_forecast_model(
            username=self.test_user,
            catchment_id=TEST_BASIN,
            mode='coldstart',
            requested_issue_date='2026-03-30',
            model_run_dir=self.model_run_dir,
        )
        assert res['status'] == 'success'
        assert res['peak_discharge_cms'] is None
        for pt in res['forecast']:
            assert pt['discharge_cms'] is None
            assert pt['discharge_mm_day'] is None

    def test_missing_basin_static_attributes_raises_key_error(self) -> None:
        """Verify missing basin static attributes raise KeyError."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )
        # Remove the profile static attributes CSV so TEST_BASIN is missing
        attr_dir = self.pm.get_caravan_attributes_dir(self.test_user)
        if attr_dir.exists():
            shutil.rmtree(attr_dir)

        with pytest.raises(KeyError, match=TEST_BASIN):
            realtime_forecast_service.run_forecast_model(
                username=self.test_user,
                catchment_id=TEST_BASIN,
                mode='coldstart',
                requested_issue_date='2026-03-30',
                model_run_dir=self.model_run_dir,
            )

    def test_multi_catchment_selection_and_strict_yesterday_state_gating(
        self,
    ) -> None:
        """Verify multi-catchment batch status, fetch, and forecast gating."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=[TEST_BASIN, TEST_BASIN_2],
            issue_date_str='2026-03-30',
        )

        # Give TEST_BASIN 2026-03-29 state, TEST_BASIN_2 stale 2024-01-01 state
        state_path_1 = realtime_forecast_service.get_state_file_path(
            self.test_user, TEST_BASIN
        )
        state_path_2 = realtime_forecast_service.get_state_file_path(
            self.test_user, TEST_BASIN_2
        )
        state_path_1.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            state_path_1,
            h_n=np.zeros((1, 1, 64), dtype=np.float32),
            c_n=np.zeros((1, 1, 64), dtype=np.float32),
            date=np.array('2026-03-29', dtype='U10'),
        )
        np.savez(
            state_path_2,
            h_n=np.zeros((1, 1, 64), dtype=np.float32),
            c_n=np.zeros((1, 1, 64), dtype=np.float32),
            date=np.array('2024-01-01', dtype='U10'),
        )

        code, multi_status = self._request(
            'GET',
            f'/api/forecast/status?catchment_id={TEST_BASIN}'
            f'&catchment_ids={TEST_BASIN},{TEST_BASIN_2}',
        )
        assert code == HTTP_OK
        assert multi_status['all_selected_have_previous_day_state'] is False
        assert multi_status['can_run_hotstart'] is False
        assert multi_status['can_fetch_hotstart'] is False
        missing_ids = [
            item['catchment_id']
            for item in multi_status['missing_or_stale_state_catchments']
        ]
        assert TEST_BASIN_2 in missing_ids
        assert TEST_BASIN not in missing_ids
        assert (
            multi_status['missing_or_stale_state_catchments'][0]['state_date']
            == '2024-01-01'
        )

        code, blocked_fetch = self._request(
            'POST',
            '/api/forecast/fetch-realtime',
            {
                'catchment_id': TEST_BASIN,
                'catchment_ids': [TEST_BASIN, TEST_BASIN_2],
                'mode': 'hotstart',
            },
        )
        assert code == HTTP_BAD_REQUEST
        assert TEST_BASIN_2 in blocked_fetch['message']
        assert '2024-01-01' in blocked_fetch['message']

        code, blocked_run = self._post_run_model(
            {
                'catchment_id': TEST_BASIN,
                'catchment_ids': [TEST_BASIN, TEST_BASIN_2],
                'mode': 'hotstart',
            },
        )
        assert code == HTTP_BAD_REQUEST
        assert TEST_BASIN_2 in blocked_run['message']
        assert '2024-01-01' in blocked_run['message']

        # Run Cold-Start across both catchments simultaneously
        code, batch_cold = self._post_run_model(
            {
                'catchment_id': TEST_BASIN,
                'catchment_ids': [TEST_BASIN, TEST_BASIN_2],
                'mode': 'coldstart',
            },
        )
        assert code == HTTP_OK, f'Batch cold-start failed: {batch_cold}'
        assert batch_cold['catchment_ids'] == [TEST_BASIN, TEST_BASIN_2]
        assert TEST_BASIN in batch_cold['results_by_id']
        assert TEST_BASIN_2 in batch_cold['results_by_id']
        assert (
            batch_cold['results_by_id'][TEST_BASIN_2]['saved_state'][
                'state_date'
            ]
            == '2026-03-29'
        )

        # Now Hot-Start across both catchments must succeed
        code, batch_hot = self._post_run_model(
            {
                'catchment_id': TEST_BASIN,
                'catchment_ids': [TEST_BASIN, TEST_BASIN_2],
                'mode': 'hotstart',
            },
        )
        assert code == HTTP_OK, f'Batch hot-start failed: {batch_hot}'
        assert batch_hot['mode'] == 'hotstart'
        for cid in (TEST_BASIN, TEST_BASIN_2):
            cold_q = [
                pt['discharge_cms']
                for pt in batch_cold['results_by_id'][cid]['forecast']
            ]
            hot_q = [
                pt['discharge_cms']
                for pt in batch_hot['results_by_id'][cid]['forecast']
            ]
            np.testing.assert_allclose(cold_q, hot_q, rtol=1e-4, atol=1e-4)

    def test_fetch_realtime_endpoint_calls_multimet_realtime_with_mode(
        self,
    ) -> None:
        """Verify POST /api/forecast/fetch-realtime with HTTP boundary mock."""

        def _fake_cpc_http_download(
            url: str, dest_path: str, **_kwargs: object
        ) -> str:
            del url
            dates = pd.date_range('2026-03-25', '2026-03-30', freq='D')
            precip = np.full((len(dates), 360, 720), 4.2, dtype=np.float32)
            ds = xr.Dataset(
                {'precip': (('time', 'lat', 'lon'), precip)},
                coords={
                    'time': dates,
                    'lat': cpc_utils.EXPECTED_PSL_LATS,
                    'lon': cpc_utils.EXPECTED_PSL_LONS,
                },
            )
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(dest_path)
            return str(dest_path)

        with (
            mock.patch(
                'multimet.utils.cpc._is_cached_cpc_netcdf_usable',
                return_value=False,
            ),
            mock.patch(
                'multimet.utils.cpc.download_http_file',
                side_effect=_fake_cpc_http_download,
            ) as mock_http_dl,
        ):
            code, fetch_res = self._request(
                'POST',
                '/api/forecast/fetch-realtime',
                {
                    'catchment_id': TEST_BASIN,
                    'catchment_ids': [TEST_BASIN, TEST_BASIN_2],
                    'mode': 'coldstart',
                    'reference_date': '2026-03-30',
                    'products': ['CPC'],
                    'lookback_days': 3,
                    'overwrite': True,
                },
            )
            assert code == HTTP_OK, f'Fetch endpoint failed: {fetch_res}'
            assert fetch_res['status'] == 'success'
            assert fetch_res['fetch_summary']['mode'] == 'coldstart'
            assert fetch_res['fetch_summary']['catchment_ids'] == [
                TEST_BASIN,
                TEST_BASIN_2,
            ]
            assert 'CPC' in fetch_res['fetch_summary']['products_written']
            mock_http_dl.assert_called_once()

    def test_profile_watershed_hydration_across_tabs(self) -> None:
        """Verify all profile basins (delineated + uploaded with id) hydrate."""
        uploaded_feature_only_id = {
            'type': 'Feature',
            'properties': {
                'id': 'us_03338780',
                'source': 'uploaded',
                'AREA': 43.2,
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-86.9, 40.3],
                        [-86.8, 40.3],
                        [-86.8, 40.4],
                        [-86.9, 40.4],
                        [-86.9, 40.3],
                    ]
                ],
            },
        }
        self.pm.save_watersheds(
            [
                TEST_WATERSHED_FEATURE,
                TEST_WATERSHED_FEATURE_2,
                uploaded_feature_only_id,
            ],
            username=self.test_user,
        )
        loaded = self.pm.load_watersheds(username=self.test_user)
        assert len(loaded) == 3  # noqa: PLR2004
        cids = [f['properties'].get('catchment_id') for f in loaded]
        assert cids == [TEST_BASIN, TEST_BASIN_2, 'us_03338780']

        code, ws_resp = self._request(
            'GET', f'/api/watersheds?username={self.test_user}'
        )
        assert code == HTTP_OK
        assert [
            f['properties'].get('catchment_id') for f in ws_resp['features']
        ] == [TEST_BASIN, TEST_BASIN_2, 'us_03338780']

        html_path = (
            Path(__file__).resolve().parent.parent / 'static' / 'index.html'
        )
        html = html_path.read_text(encoding='utf-8')
        assert 'updateWatershedListUI' not in html
        assert 'await fetchCurrentProfile();' in html
        assert (
            "if (typeof renderGeoCatchmentSelector === 'function') "
            'renderGeoCatchmentSelector();' in html
        )
        assert (
            "if (typeof loadPersistedExtractedAttributes === 'function') "
            'await loadPersistedExtractedAttributes();' in html
        )
        assert (
            "if (typeof syncForecastTabBasin === 'function') "
            'syncForecastTabBasin();' in html
        )

    def test_what_if_scenario_precip_and_temp_overrides_and_ui_controls(
        self,
    ) -> None:
        """Verify What-If overrides modify trajectories without saving state."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )

        # 1. Run baseline unmodified Cold-Start forecast
        code, base_res = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'coldstart'},
        )
        assert code == HTTP_OK, f'Baseline run failed: {base_res}'
        base_q = np.array([pt['discharge_cms'] for pt in base_res['forecast']])

        state_path = realtime_forecast_service.get_state_file_path(
            self.test_user, TEST_BASIN
        )
        mtime_before = state_path.stat().st_mtime_ns

        # 2. Run What-If Scenario #1 with increased precip and temp
        code, whatif_1 = self._post_run_model(
            {
                'catchment_id': TEST_BASIN,
                'mode': 'coldstart',
                'precip_overrides': {
                    '2026-03-30': 75.0,
                    '2026-03-31': 90.0,
                    '__temp_overrides__': {
                        '2026-03-30': 24.5,
                        '2026-03-31': 25.0,
                    },
                    '__is_what_if__': True,
                },
                'temp_overrides': {'2026-03-30': 24.5, '2026-03-31': 25.0},
                'is_what_if': True,
            },
        )
        assert code == HTTP_OK, f'What-If #1 failed: {whatif_1}'
        assert whatif_1.get('is_what_if') is True
        q_whatif_1 = np.array(
            [pt['discharge_cms'] for pt in whatif_1['forecast']]
        )
        assert not np.allclose(base_q, q_whatif_1)
        assert state_path.stat().st_mtime_ns == mtime_before

        # 3. Run What-If Scenario #2 with per-product overrides
        code, whatif_2 = self._post_run_model(
            {
                'catchment_id': TEST_BASIN,
                'mode': 'coldstart',
                'product_overrides': {
                    'HRES': {
                        'precip': {'2026-04-01': 120.0, '2026-04-02': 110.0},
                        'temp': {'2026-04-01': 5.0},
                    },
                },
                'precip_overrides': {
                    '2026-04-01': 120.0,
                    '2026-04-02': 110.0,
                    '__product_overrides__': {
                        'HRES': {
                            'precip': {
                                '2026-04-01': 120.0,
                                '2026-04-02': 110.0,
                            },
                            'temp': {'2026-04-01': 5.0},
                        },
                    },
                    '__is_what_if__': True,
                },
                'is_what_if': True,
            },
        )
        assert code == HTTP_OK, f'What-If #2 failed: {whatif_2}'
        q_whatif_2 = np.array(
            [pt['discharge_cms'] for pt in whatif_2['forecast']]
        )
        assert not np.allclose(base_q, q_whatif_2)
        assert not np.allclose(q_whatif_1, q_whatif_2)

        # 4. Verify frontend HTML/JS structure for What-If controls
        html_path = (
            Path(__file__).resolve().parent.parent / 'static' / 'index.html'
        )
        html = html_path.read_text(encoding='utf-8')
        for token in (
            'FC_NOWCAST_PRECIP_PALETTE',
            'FC_FORECAST_PRECIP_PALETTE',
            'FC_FORECAST_TEMP_PALETTE',
            'minBarLength: 7',
            'fcWhatIfDragHandlePlugin',
            'initWhatIfChartDragging',
            'enabled: false',
            'id="fcRunWhatIfBtn"',
            'id="fcResetWhatIfInputsBtn"',
            'id="fcWhatIfScenariosBar"',
            'id="fcWhatIfScenariosList"',
            'fc-whatif-toggle-cb',
            'fc-whatif-delete-btn',
        ):
            assert token in html


if __name__ == '__main__':
    unittest.main()
