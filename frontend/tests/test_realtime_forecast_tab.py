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
            'fcHydroCatchmentTabs',
            'whatIfHyetographCanvas',
            'masterHydrographCanvas',
            'fcHydroPeakBadge',
            'fetchForecastBtn',
            'executionModeSelect',
            'modelCheckpointSelect',
            'fcBasinSelect',
            'fcForcingLegendContainer',
            'fcForcingLegendGrid',
        ]
        for dom_id in required_ids:
            assert f'id="{dom_id}"' in html, f'Missing DOM ID: {dom_id}'

        assert 'id="fcLeadTimeInspectorContainer"' not in html
        assert 'id="fcLeadTimeSummaryTable"' not in html
        assert 'Drag forecast bars/points to edit' not in html

        for expected_token in [
            'for="fcBasinSelect"',
            'fcIssueDateBoundaryPlugin',
            'buildSharedForecastTimeline',
            'bridgeHandoffAtT0',
            'LEFT_AXIS_WIDTH',
            'RIGHT_AXIS_WIDTH',
            'yRightAlign',
            'renderForcingLegend',
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
            'getCrossStitchedBarPattern(pal.bg, pal.border)',
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
            'hindcast_precip: 1',
            'forecast_precip: 2',
            'hindcast_temp: 3',
            'forecast_temp: 4',
        ):
            assert token in html

    def test_upload_realtime_streamflow_csv_and_return_periods(self) -> None:
        """Verify uploading a streamflow CSV, preserving NaNs, and plotting."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )

        # 1. Upload 2-column CSV with valid values, missing, and sentinel -999
        csv_text = (
            'date,discharge_cms\n'
            '2026-03-27,14.2\n'
            '2026-03-28,\n'
            '2026-03-29,-999\n'
            '2026-03-30,28.5\n'
        )
        code, upload_res = self._request(
            'POST',
            '/api/forecast/upload-streamflow',
            {
                'catchment_id': TEST_BASIN,
                'filename': 'gauge_obs.csv',
                'csv_content': csv_text,
            },
        )
        assert code == HTTP_OK, f'CSV upload failed: {upload_res}'
        assert upload_res['status'] == 'success'
        obs_series = upload_res['observed_streamflow']
        assert len(obs_series) == 4  # noqa: PLR2004
        by_date = {pt['date']: pt['discharge_cms'] for pt in obs_series}
        assert by_date['2026-03-27'] == pytest.approx(14.2)
        assert by_date['2026-03-28'] is None
        assert by_date['2026-03-29'] is None
        assert by_date['2026-03-30'] == pytest.approx(28.5)

        # Verify assimilation/streamflow_realtime.zarr preserves NaNs
        zarr_path = (
            self.profile_dir / 'assimilation' / 'streamflow_realtime.zarr'
        )
        assert zarr_path.exists()
        ds_obs = xr.open_zarr(zarr_path, consolidated=False)
        assert TEST_BASIN in ds_obs['basin'].to_numpy()
        q_vals = ds_obs['streamflow'].sel(basin=TEST_BASIN).to_numpy()
        assert np.isnan(q_vals[1])
        assert np.isnan(q_vals[2])
        assert np.isclose(q_vals[0], 14.2)
        assert np.isclose(q_vals[3], 28.5)

        # 2. Verify /api/forecast/status returns observed_streamflow
        code, status_res = self._request(
            'GET', f'/api/forecast/status?catchment_id={TEST_BASIN}'
        )
        assert code == HTTP_OK
        assert status_res['has_observed_streamflow'] is True
        assert len(status_res['observed_streamflow']) == 4  # noqa: PLR2004
        assert 'return_periods' in status_res

        # 3. Verify running model includes observed_streamflow without nudging
        code, run_res = self._post_run_model(
            {'catchment_id': TEST_BASIN, 'mode': 'coldstart'}
        )
        assert code == HTTP_OK
        assert run_res['has_observed_streamflow'] is True
        assert len(run_res['observed_streamflow']) == 4  # noqa: PLR2004
        assert 'return_periods' in run_res

        # 4. Clear uploaded CSV and confirm status resets
        code, clear_res = self._request(
            'POST',
            '/api/forecast/clear-streamflow',
            {'catchment_id': TEST_BASIN},
        )
        assert code == HTTP_OK
        assert clear_res['status'] == 'success'
        assert clear_res['observed_streamflow'] == []

        code, status_after = self._request(
            'GET', f'/api/forecast/status?catchment_id={TEST_BASIN}'
        )
        assert code == HTTP_OK
        assert status_after['has_observed_streamflow'] is False
        assert status_after['observed_streamflow'] == []

        # 5. Verify frontend HTML/JS tokens for CSV upload & return periods
        html_path = (
            Path(__file__).resolve().parent.parent / 'static' / 'index.html'
        )
        html = html_path.read_text(encoding='utf-8')
        for token in (
            'id="fcUploadStreamflowCsvBtn"',
            'id="fcStreamflowCsvInput"',
            'id="fcClearStreamflowCsvBtn"',
            'id="fcStreamflowCsvStatus"',
            'OBSERVED STREAMFLOW (CSV)',
            '/api/forecast/upload-streamflow',
            '/api/forecast/clear-streamflow',
            'fcReturnPeriodLinesPlugin',
            'if (t.value > yMax || t.value < yMin) return;',
        ):
            assert token in html

    def test_csv_validation_collision_and_issue_date_regressions(self) -> None:
        """Verify CSV validation, collision-free IDs, and tail fallback."""
        # 1. Invalid CSV payloads must return HTTP 400 with clear errors
        bad_payloads = [
            'col_a,col_b\n2026-03-27,10.0\n',
            'date,discharge_cms\nnot-a-date,10.0\n',
            'date,discharge_cms\n2026-03-27,"12,5"\n',
            'date,discharge_cms\n2026-03-27,10.0\n2026-03-27,15.0\n',
        ]
        for bad_csv in bad_payloads:
            code, err_res = self._request(
                'POST',
                '/api/forecast/upload-streamflow',
                {
                    'catchment_id': TEST_BASIN,
                    'csv_content': bad_csv,
                },
            )
            assert code == HTTP_BAD_REQUEST, f'Expected 400 for {bad_csv!r}'
            assert err_res['status'] == 'error'

        # 2. Multi-basin CSV upload returns observed_by_id for every basin
        multi_csv = (
            'basin,date,streamflow\n'
            f'{TEST_BASIN},2026-03-28,11.0\n'
            f'{TEST_BASIN_2},2026-03-28,22.5\n'
        )
        code, multi_res = self._request(
            'POST',
            '/api/forecast/upload-streamflow',
            {
                'catchment_id': TEST_BASIN,
                'csv_content': multi_csv,
            },
        )
        assert code == HTTP_OK
        assert TEST_BASIN in multi_res['observed_by_id']
        assert TEST_BASIN_2 in multi_res['observed_by_id']
        assert multi_res['observed_by_id'][TEST_BASIN_2][0][
            'discharge_cms'
        ] == pytest.approx(22.5)

        # 3. Disjoint date window on load_realtime_streamflow_series returns []
        disjoint = realtime_forecast_service.load_realtime_streamflow_series(
            TEST_USER,
            TEST_BASIN,
            start_date='2025-01-01',
            end_date='2025-01-10',
        )
        assert disjoint == []

        # 4. Basin IDs that sanitize to similar ASCII stems never collide
        realtime_forecast_service.upload_realtime_streamflow_csv(
            TEST_USER,
            catchment_id='a.b',
            csv_content='date,q\n2026-03-28,5.0\n',
        )
        realtime_forecast_service.upload_realtime_streamflow_csv(
            TEST_USER,
            catchment_id='ab',
            csv_content='date,q\n2026-03-28,99.0\n',
        )
        s_dot = realtime_forecast_service.load_realtime_streamflow_series(
            TEST_USER, 'a.b'
        )
        s_plain = realtime_forecast_service.load_realtime_streamflow_series(
            TEST_USER, 'ab'
        )
        assert s_dot[0]['discharge_cms'] == pytest.approx(5.0)
        assert s_plain[0]['discharge_cms'] == pytest.approx(99.0)

        # 5. Invalid requested_issue_date returns HTTP 400 / raises ValueError
        code, bad_status = self._request(
            'GET',
            f'/api/forecast/status?catchment_id={TEST_BASIN}&issue_date=bad-date',
        )
        assert code == HTTP_BAD_REQUEST
        assert bad_status['status'] == 'error'
        with pytest.raises(ValueError, match='Invalid issue_date'):
            realtime_forecast_service.get_realtime_status(
                TEST_USER,
                TEST_BASIN,
                requested_issue_date='bad-date',
            )

        # 6. 40-date HRES store where the last 14 dates are all-NaN resolves
        # the last valid date (2026-01-20) instead of the final store date
        dates_40 = pd.date_range('2026-01-01', periods=40, freq='D')
        leads = pd.timedelta_range('1D', periods=10, freq='D')
        arr = np.full((1, 40, 10), np.nan, dtype=np.float32)
        arr[:, :20, :] = 3.0
        hres_40_path = self.profile_dir / 'test_hres_40.zarr'
        ds_40 = xr.Dataset(
            data_vars={
                'hres_total_precipitation': (
                    ['basin', 'date', 'lead_time'],
                    arr,
                ),
            },
            coords={
                'basin': [TEST_BASIN],
                'date': dates_40,
                'lead_time': leads,
            },
        )
        ds_40.to_zarr(hres_40_path, mode='w', consolidated=True)
        latest_valid = (
            realtime_forecast_service._extract_latest_issue_from_store(  # noqa: SLF001
                hres_40_path,
                TEST_BASIN,
            )
        )
        assert latest_valid is not None
        assert latest_valid.strftime('%Y-%m-%d') == '2026-01-20'

        # 7. Return period normalization handles rp_2, 25yr, nested dicts,
        # and ignores bare integer keys and booleans
        pm = profile_manager.get_profile_manager()
        ws = [
            {
                **TEST_WATERSHED_FEATURE,
                'properties': {
                    **TEST_WATERSHED_FEATURE['properties'],
                    'rp_2': 45.0,
                    '2': 9999.0,
                    'return_period_5': True,
                    'return_periods_cms': {'25yr': 180.5, '10': 95.0},
                },
            }
        ]
        pm.save_watersheds(ws, username=TEST_USER)
        rps = realtime_forecast_service.load_basin_return_periods(
            TEST_USER, TEST_BASIN
        )
        assert rps['return_period_2'] == pytest.approx(45.0)
        assert rps['return_period_10'] == pytest.approx(95.0)
        assert rps['return_period_25'] == pytest.approx(180.5)
        assert 'return_period_5' not in rps

    def test_realtime_fetch_progress_endpoint_and_ui_indicators(self) -> None:
        """Verify live per-product fetch progress tracking and UI bindings."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir,
            basin_id=TEST_BASIN,
            issue_date_str='2026-03-30',
        )
        realtime_forecast_service._init_user_fetch_progress(  # noqa: SLF001
            user_key=self.test_user,
            target_ids=[TEST_BASIN],
            norm_mode='coldstart',
            requested_products=['HRES', 'IMERG', 'CPC'],
            spinup_days=365,
        )
        realtime_forecast_service._handle_fetch_progress_event(  # noqa: SLF001
            self.test_user,
            {
                'event': 'product_start',
                'product': 'HRES',
                'product_index': 0,
                'total_products': 3,
                'start_date': '2025-03-30',
                'end_date': '2026-03-30',
                'window_days': 366,
            },
        )
        realtime_forecast_service._handle_fetch_progress_event(  # noqa: SLF001
            self.test_user,
            {
                'event': 'bar_step',
                'source': 'tqdm',
                'current': 183,
                'total': 366,
                'unit': 'init_date',
                'desc': 'HRES OpenData',
                'rate': 4.5,
                'eta_seconds': 40.7,
                'bar_id': 101,
            },
        )

        code, prog_res = self._request(
            'GET',
            f'/api/forecast/fetch-progress?catchment_id={TEST_BASIN}&username={self.test_user}',
        )
        assert code == HTTP_OK
        assert prog_res['active'] is True
        assert prog_res['current_product'] == 'HRES'
        assert prog_res['products']['HRES']['status'] == 'downloading'
        assert prog_res['products']['HRES']['current'] == 183  # noqa: PLR2004
        assert prog_res['products']['HRES']['total'] == 366  # noqa: PLR2004
        assert prog_res['products']['HRES']['percent'] == pytest.approx(50.0)
        assert prog_res['products']['IMERG']['status'] == 'queued'
        assert prog_res['products_disk_status']['HRES']['ready_on_disk'] is True

        realtime_forecast_service._handle_fetch_progress_event(  # noqa: SLF001
            self.test_user,
            {
                'event': 'product_complete',
                'product': 'HRES',
                'product_index': 0,
                'total_products': 3,
            },
        )
        realtime_forecast_service._handle_fetch_progress_event(  # noqa: SLF001
            self.test_user,
            {'event': 'complete'},
        )
        code, done_res = self._request(
            'GET',
            f'/api/forecast/fetch-progress?catchment_id={TEST_BASIN}&username={self.test_user}',
        )
        assert code == HTTP_OK
        assert done_res['active'] is False
        assert done_res['status'] == 'completed'
        assert done_res['overall_percent'] == pytest.approx(100.0)

        html_path = (
            Path(__file__).resolve().parent.parent / 'static' / 'index.html'
        )
        html = html_path.read_text(encoding='utf-8')
        for token in (
            'id="fcProdStatusBadge_HRES"',
            'id="fcProdProgressBar_IMERG"',
            'id="fcProdProgressBar_CPC"',
            'id="fcLiveFetchBanner"',
            'renderForecastFetchProgressUI',
            'pollForecastFetchProgressOnce',
            'id="fcRunProgressBox"',
            'id="fcRunOverallBarFill"',
            'id="fcRunBasinsProgressList"',
            'renderForecastRunProgressUI',
            'pollForecastRunProgressOnce',
        ):
            assert token in html

        # Verify per-basin model run progress tracking and endpoint
        realtime_forecast_service._init_user_run_progress(  # noqa: SLF001
            user_key=self.test_user,
            target_ids=[TEST_BASIN, TEST_BASIN_2],
            norm_mode='coldstart',
            model_id=SYNTHETIC_MODEL_ID,
            issue_date_str='2026-03-30',
        )
        realtime_forecast_service._update_user_run_progress_step(  # noqa: SLF001
            self.test_user,
            event='basin_start',
            basin_id=TEST_BASIN,
            basin_index=0,
        )
        code, run_prog = self._request(
            'GET',
            f'/api/forecast/run-progress?catchment_id={TEST_BASIN}&username={self.test_user}',
        )
        assert code == HTTP_OK
        assert run_prog['active'] is True
        assert run_prog['current_basin'] == TEST_BASIN
        assert run_prog['basins'][TEST_BASIN]['status'] == 'running'
        assert run_prog['basins'][TEST_BASIN_2]['status'] == 'queued'

        realtime_forecast_service._update_user_run_progress_step(  # noqa: SLF001
            self.test_user,
            event='basin_complete',
            basin_id=TEST_BASIN,
            basin_index=0,
            basin_elapsed=0.45,
            peak_discharge_cms=42.5,
            peak_date='2026-04-01',
        )
        code, run_prog_mid = self._request(
            'GET',
            f'/api/forecast/run-progress?catchment_id={TEST_BASIN}&username={self.test_user}',
        )
        assert code == HTTP_OK
        assert run_prog_mid['completed_basins_count'] == 1
        assert run_prog_mid['overall_percent'] == pytest.approx(50.0)
        assert run_prog_mid['basins'][TEST_BASIN]['status'] == 'completed'

    def test_data_assimilation_nudging_postprocessing_and_ui_controls(
        self,
    ) -> None:
        """Verify AR(1) nudging post-processing and expandable DA UI."""
        # 1. Direct mathematical verification of Karan's Eq. (6):
        # Q_hat_PP(t + k) = max(0, Q_hat_base(t + k) + alpha * rho^k * e_anchor)
        issue_date = '2026-03-30'
        hindcast_tail = [
            {'date': '2026-03-28', 'discharge_cms': 10.0},
            {'date': '2026-03-29', 'discharge_cms': 12.0},
            {'date': '2026-03-30', 'discharge_cms': 15.0},
        ]
        forecast_series = [
            {'date': '2026-03-30', 'lead_day': 0, 'discharge_cms': 15.0},
            {'date': '2026-03-31', 'lead_day': 1, 'discharge_cms': 20.0},
            {'date': '2026-04-01', 'lead_day': 2, 'discharge_cms': 18.0},
            {'date': '2026-04-02', 'lead_day': 3, 'discharge_cms': 14.0},
        ]
        obs_series = [
            {'date': '2026-03-28', 'discharge_cms': 20.0},
            {'date': '2026-03-29', 'discharge_cms': 22.0},
            {'date': '2026-03-30', 'discharge_cms': 35.0},  # residual +20.0
        ]

        res = realtime_forecast_service.compute_nudging_postprocessing(
            forecast_series=forecast_series,
            hindcast_series=hindcast_tail,
            observed_series=obs_series,
            issue_date=issue_date,
            rho=0.65,
            alpha=1.0,
            window_days=1,
        )
        assert res['applied'] is True
        assert res['anchor_date'] == '2026-03-30'
        assert res['residual_cms'] == pytest.approx(20.0)
        nudged = {pt['date']: pt for pt in res['nudged_series']}
        # k=0: 15.0 + 1.0 * (0.65^0) * 20.0 = 35.0 (matches gauge when alpha=1)
        assert nudged['2026-03-30']['discharge_cms'] == pytest.approx(35.0)
        # k=1: 20.0 + 1.0 * (0.65^1) * 20.0 = 33.0
        assert nudged['2026-03-31']['discharge_cms'] == pytest.approx(33.0)
        # k=2: 18.0 + 1.0 * (0.65^2) * 20.0 = 18.0 + 8.45 = 26.45
        assert nudged['2026-04-01']['discharge_cms'] == pytest.approx(26.45)

        # Verify window_days=3 averages the 3 pre-issue residuals: 40 / 3
        res_w3 = realtime_forecast_service.compute_nudging_postprocessing(
            forecast_series=forecast_series,
            hindcast_series=hindcast_tail,
            observed_series=obs_series,
            issue_date=issue_date,
            rho=0.5,
            alpha=0.5,
            window_days=3,
        )
        assert res_w3['residual_cms'] == pytest.approx(40.0 / 3.0, abs=1e-2)
        nudged_w3 = {pt['date']: pt for pt in res_w3['nudged_series']}
        assert nudged_w3['2026-03-30']['discharge_cms'] == pytest.approx(
            15.0 + 0.5 * (40.0 / 3.0), abs=1e-2
        )

        # Verify non-negativity clamp when residual is strongly negative
        obs_low = [{'date': '2026-03-30', 'discharge_cms': 0.5}]
        res_clamp = realtime_forecast_service.compute_nudging_postprocessing(
            forecast_series=[
                {'date': '2026-03-30', 'lead_day': 0, 'discharge_cms': 25.0},
                {'date': '2026-03-31', 'lead_day': 1, 'discharge_cms': 5.0},
            ],
            hindcast_series=hindcast_tail,
            observed_series=obs_low,
            issue_date=issue_date,
            rho=0.9,
            alpha=1.0,
            window_days=1,
        )
        nudged_clamp = {pt['date']: pt for pt in res_clamp['nudged_series']}
        assert nudged_clamp['2026-03-31']['discharge_cms'] == 0.0

        # 2. Verify /api/forecast/assimilate HTTP endpoint with uploaded obs
        csv_text = (
            'date,streamflow_cms\n'
            '2026-03-28,25.0\n'
            '2026-03-29,30.0\n'
            '2026-03-30,27.0\n'
        )
        code, up_res = self._request(
            'POST',
            '/api/forecast/upload-streamflow',
            {
                'username': self.test_user,
                'catchment_id': TEST_BASIN,
                'filename': 'gauge_obs.csv',
                'csv_content': csv_text,
            },
        )
        assert code == HTTP_OK
        assert up_res['status'] == 'success'

        code, da_res = self._request(
            'POST',
            '/api/forecast/assimilate',
            {
                'username': self.test_user,
                'catchment_id': TEST_BASIN,
                'method': 'nudging',
                'rho': 0.65,
                'alpha': 1.0,
                'window_days': 1,
                'issue_date': '2026-03-30',
                'forecast_series': forecast_series,
                'hindcast_series': hindcast_tail,
            },
        )
        assert code == HTTP_OK
        assert da_res['status'] == 'success'
        assert da_res['method'] == 'nudging'
        assert da_res['applied'] is True
        assert da_res['anchor_date'] == '2026-03-30'
        assert da_res['residual_cms'] == pytest.approx(12.0, abs=1e-2)
        nudged_api = {pt['date']: pt for pt in da_res['nudged_series']}
        assert nudged_api['2026-03-30']['discharge_cms'] == pytest.approx(
            27.0, abs=1e-2
        )

    def test_variational_data_assimilation_and_ui_tokens(self) -> None:
        """Verify 4D-Var Embedding Data Assimilation and DA UI controls."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir, TEST_BASIN, '2026-03-30'
        )
        expected_obs_days = 10
        var_csv_lines = [
            'date,streamflow_cms',
            *[
                f'2026-03-{day:02d},{35.0 + (day - 20) * 1.5:.1f}'
                for day in range(20, 20 + expected_obs_days)
            ],
        ]
        code, up_var_res = self._request(
            'POST',
            '/api/forecast/upload-streamflow',
            {
                'username': self.test_user,
                'catchment_id': TEST_BASIN,
                'filename': 'gauge_obs_window.csv',
                'csv_content': '\n'.join(var_csv_lines) + '\n',
            },
        )
        assert code == HTTP_OK
        assert up_var_res['status'] == 'success'

        code, var_res = self._request(
            'POST',
            '/api/forecast/assimilate',
            {
                'username': self.test_user,
                'catchment_id': TEST_BASIN,
                'method': 'variational',
                'issue_date': '2026-03-30',
                'window_days': expected_obs_days,
                'regularization_weight': 0.15,
                'initial_learning_rate': 0.03,
                'epochs': 20,
                'loss': 'MSE',
                'assimilation_components': [
                    'static_embedding',
                    'hindcast_embedding',
                ],
                'model_run_dir': str(self.model_run_dir),
            },
        )
        assert code == HTTP_OK
        assert var_res['status'] == 'success'
        assert var_res['method'] == 'variational'
        assert var_res['applied'] is True
        assert var_res['n_valid_obs'] == expected_obs_days
        assert var_res['prior_window_rmse_cms'] is not None
        assert var_res['assimilated_window_rmse_cms'] is not None
        assert (
            var_res['assimilated_window_rmse_cms']
            < var_res['prior_window_rmse_cms']
        )
        assert var_res['component_shifts']['static_embedding'] > 0.0
        assert var_res['component_shifts']['hindcast_embedding'] > 0.0
        assert len(var_res['forecast']) == SYNTHETIC_LEAD_TIME_DAYS
        assert len(var_res['hindcast_tail']) > 0
        assert len(var_res['nudged_series']) == len(
            var_res['hindcast_tail']
        ) + len(var_res['forecast'])

        # Verify expandable DA menu and interactive visual controls
        html_path = (
            Path(__file__).resolve().parent.parent / 'static' / 'index.html'
        )
        html = html_path.read_text(encoding='utf-8')
        for token in (
            'id="fcDaSection"',
            'id="fcDaToggleHeaderBtn"',
            'id="fcDaCollapsibleBody"',
            'id="fcDaMethodNudgingBtn"',
            'id="fcDaMethodVariationalBtn"',
            'id="fcDaCurveCanvas"',
            'id="fcDaVariationalPanel"',
            'id="fcDaVarCanvas"',
            'fc-da-var-comp-chip',
            'fc-da-var-loss-chip',
            'id="fcDaVarRunBtn"',
            'id="fcDaVarResetBtn"',
            'id="fcDaActiveToggleBtn"',
            'updateDaControlsUI',
            'initDaCurveCanvasDragging',
            'renderDaCurveCanvas',
            'renderDaVariationalCanvas',
            'runVariationalAssimilationForBasin',
            'computeBasinNudgedHydrograph',
            'initHydrographNudgingDrag',
            'NUDGED FORECAST',
            'VARIATIONAL DA (4D-VAR',
        ):
            assert token in html

    def test_daily_forecast_persistence_and_ttl_expiration(self) -> None:
        """Daily forecasts persist across refresh and auto-expire after TTL."""
        _populate_synthetic_realtime_zarr_stores(
            self.profile_dir, TEST_BASIN, '2026-03-30'
        )
        # 1. Before running the model, cached_forecast is None
        code, st_initial = self._request(
            'GET',
            f'/api/forecast/status?catchment_id={TEST_BASIN}&username={self.test_user}',
        )
        assert code == HTTP_OK
        assert st_initial['cached_forecast'] is None
        assert st_initial['forecast_cache_ttl_seconds'] == 4 * 3600

        # 2. Run Cold-Start daily forecast
        code, run_res = self._request(
            'POST',
            '/api/forecast/run-model',
            {
                'username': self.test_user,
                'catchment_id': TEST_BASIN,
                'mode': 'coldstart',
                'model_run_dir': str(self.model_run_dir),
            },
        )
        assert code == HTTP_OK
        assert run_res['status'] == 'success'
        assert len(run_res['forecast']) == SYNTHETIC_LEAD_TIME_DAYS
        assert len(run_res['hindcast_tail']) > 0

        # 3. Simulate refresh: GET /api/forecast/status returns cached_forecast
        code, st_after = self._request(
            'GET',
            f'/api/forecast/status?catchment_id={TEST_BASIN}&username={self.test_user}',
        )
        assert code == HTTP_OK
        cached = st_after['cached_forecast']
        assert cached is not None
        assert cached['catchment_id'] == TEST_BASIN
        assert len(cached['forecast']) == SYNTHETIC_LEAD_TIME_DAYS
        assert len(cached['hindcast_tail']) == len(run_res['hindcast_tail'])
        assert cached['cached_at_unix'] > 0
        assert cached['expires_at_unix'] > cached['cached_at_unix']
        assert st_after['cached_forecasts_by_id'][TEST_BASIN] is not None

        # 4. Run Hot-Start and verify hindcast_tail from Cold-Start is kept
        code, hot_res = self._request(
            'POST',
            '/api/forecast/run-model',
            {
                'username': self.test_user,
                'catchment_id': TEST_BASIN,
                'mode': 'hotstart',
                'model_run_dir': str(self.model_run_dir),
            },
        )
        assert code == HTTP_OK
        assert hot_res['status'] == 'success'
        code, st_hot = self._request(
            'GET',
            f'/api/forecast/status?catchment_id={TEST_BASIN}&username={self.test_user}',
        )
        assert code == HTTP_OK
        assert st_hot['cached_forecast'] is not None
        assert len(st_hot['cached_forecast']['hindcast_tail']) == len(
            run_res['hindcast_tail']
        )

        # 5. Simulate TTL expiration after 4+ hours: cache file is auto-deleted
        cache_path = realtime_forecast_service.get_daily_forecast_cache_path(
            self.test_user, TEST_BASIN, run_res['model_id']
        )
        assert cache_path.exists()
        raw_cache = json.loads(cache_path.read_text(encoding='utf-8'))
        raw_cache['cached_at_unix'] = time.time() - (4 * 3600 + 60)
        cache_path.write_text(json.dumps(raw_cache), encoding='utf-8')

        expired_load = realtime_forecast_service.load_cached_daily_forecast(
            self.test_user, TEST_BASIN, run_res['model_id']
        )
        assert expired_load is None
        assert not cache_path.exists()

        # 6. Verify frontend localStorage persistence & 4-hour TTL helpers
        html_path = (
            Path(__file__).resolve().parent.parent / 'static' / 'index.html'
        )
        html = html_path.read_text(encoding='utf-8')
        for token in (
            'FC_DAILY_FORECAST_TTL_MS = 4 * 60 * 60 * 1000',
            'savePersistedForecastsToStorage',
            'restorePersistedForecastsFromStorage',
            'pruneExpiredDailyForecasts',
            'cached_forecasts_by_id',
            'let fcDaStateByBasin = {};',
        ):
            assert token in html
        assert 'const fcDaStateByBasin' not in html


if __name__ == '__main__':
    unittest.main()
