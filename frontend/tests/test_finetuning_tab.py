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

"""Tests for the Fine Tuning Tab, separated UI tabs, and fine-tuning service."""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from frontend import finetune_service, profile_manager
from frontend import tests as _isolated_profiles  # noqa: F401
from frontend.config import STATIC_DIR
from frontend.server import EarthkitHydroHandler

TEST_USER = 'test_ft_user'
TEST_BASIN = 'camels_01022500'
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
EXPECTED_NUM_DAYS = 10
EXPECTED_LEAD_TIME = 7
EXPECTED_STEP1_EPOCH = 2
EXPECTED_STEP1_HISTORY_LEN = 3
EXPECTED_STEP2_EPOCH = 3
EXPECTED_STEP2_HISTORY_LEN = 4
MIN_EXPECTED_MODELS = 2

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


class _DOMInspector(HTMLParser):
    """Lightweight HTML parser that records elements by id and ancestor ids."""

    _VOID_TAGS: ClassVar[set[str]] = {
        'area',
        'base',
        'br',
        'col',
        'embed',
        'hr',
        'img',
        'input',
        'link',
        'meta',
        'param',
        'source',
        'track',
        'wbr',
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, dict[str, str | None]]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.tab_buttons: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attr_map = dict(attrs)
        el_id = attr_map.get('id')
        ancestor_ids = [
            a.get('id') for _, a in self.stack if a.get('id') is not None
        ]
        if el_id:
            self.by_id[el_id] = {
                'tag': tag,
                'attrs': attr_map,
                'ancestor_ids': ancestor_ids,
            }
        if tag == 'button' and attr_map.get('data-tab'):
            self.tab_buttons.append(attr_map)
        if tag not in self._VOID_TAGS:
            self.stack.append((tag, attr_map))

    def handle_endtag(self, tag: str) -> None:
        for idx in range(len(self.stack) - 1, -1, -1):
            if self.stack[idx][0] == tag:
                del self.stack[idx:]
                break


def _populate_test_watershed_and_attrs(username: str = TEST_USER) -> None:
    """Populate a test watershed and Caravan static attributes in profile."""
    pm = profile_manager.get_profile_manager()
    pm.save_watersheds([TEST_WATERSHED_FEATURE], username=username)
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
            }
        ]
    )
    df.to_csv(attr_dir / 'attributes_caravan_camels.csv', index=False)


class FineTuningDOMTests(unittest.TestCase):
    """Verify Training and Fine Tuning tabs are separated and complete."""

    def test_training_and_finetuning_tabs_are_separated(self) -> None:
        """Verify #tab-training and #tab-finetuning exist with controls."""
        html_text = (STATIC_DIR / 'index.html').read_text(encoding='utf-8')
        inspector = _DOMInspector()
        inspector.feed(html_text)

        tab_targets = [btn.get('data-tab') for btn in inspector.tab_buttons]
        assert 'tab-training' in tab_targets
        assert 'tab-finetuning' in tab_targets
        assert 'tab-training' in inspector.by_id
        assert 'tab-finetuning' in inspector.by_id

        required_ft_ids = [
            'ftModelSelect',
            'ftBasinSelect',
            'ftPrecipFileInput',
            'ftUploadPrecipBtn',
            'ftTempFileInput',
            'ftUploadTempBtn',
            'ftStreamflowFileInput',
            'ftUploadStreamflowBtn',
            'ftStreamflowUnitsSelect',
            'ftBasinAreaInput',
            'ftDownloadTemplateBtn',
            'ftLoadSampleDataBtn',
            'ftClearDataBtn',
            'ftSplitRatioSlider',
            'ftSplitRatioLabel',
            'ftTrainDateRangeText',
            'ftValDateRangeText',
            'ftEpochsPerStepInput',
            'ftLearningRateInput',
            'ftLossSelect',
            'ftLrStrategySelect',
            'ftBatchSizeInput',
            'ftTunableComponentsList',
            'ftPresetLocalHeadBtn',
            'ftPresetDecoderBtn',
            'ftPresetAllBtn',
            'ftRunEpochsBtn',
            'ftResetSessionBtn',
            'ftEpochScrubber',
            'ftHydrographCanvas',
            'ftLossCurveCanvas',
            'ftEpochHistoryTableBody',
        ]
        for el_id in required_ft_ids:
            assert el_id in inspector.by_id, f'Missing #{el_id} in HTML'
            assert (
                'tab-finetuning' in inspector.by_id[el_id]['ancestor_ids']
            ), f'#{el_id} must be inside #tab-finetuning'

        for train_id in (
            'trainingSidebar',
            'extractHistBtn',
            'trainBaseCheckpoint',
        ):
            assert train_id in inspector.by_id
            assert 'tab-training' in inspector.by_id[train_id]['ancestor_ids']


class FineTuneServiceTests(unittest.TestCase):
    """Unit tests for CSV parsing, model introspection, and fine-tuning."""

    def setUp(self) -> None:
        """Prepare test basin and clear prior fine-tuning artifacts."""
        _populate_test_watershed_and_attrs(TEST_USER)
        finetune_service.clear_finetune_data(TEST_USER, TEST_BASIN)

    def test_parse_time_by_lead_csv_matrix_and_daily(self) -> None:
        """Verify parsing both time x lead_time matrices and daily series."""
        dates = pd.date_range('2018-01-01', periods=EXPECTED_NUM_DAYS, freq='D')
        rows = ['date,lead_1,lead_2,lead_3,lead_4,lead_5,lead_6,lead_7']
        for i, dt in enumerate(dates):
            vals = [f'{float(i + lead):.2f}' for lead in range(1, 8)]
            rows.append(f'{dt.strftime("%Y-%m-%d")},' + ','.join(vals))
        matrix_csv = '\n'.join(rows)

        parsed_dates, matrix, parsed_leads = (
            finetune_service.parse_time_by_lead_csv(
                matrix_csv, target_lead_time=EXPECTED_LEAD_TIME
            )
        )
        assert len(parsed_dates) == EXPECTED_NUM_DAYS
        assert len(parsed_leads) == EXPECTED_LEAD_TIME
        assert matrix.shape == (EXPECTED_NUM_DAYS, EXPECTED_LEAD_TIME)
        np.testing.assert_allclose(matrix[0, 0], 1.0)
        np.testing.assert_allclose(matrix[0, 6], 7.0)

        daily_rows = ['date,streamflow']
        for i, dt in enumerate(dates):
            daily_rows.append(f'{dt.strftime("%Y-%m-%d")},{float(i) * 0.5:.2f}')
        d_dates, d_mat, d_leads = finetune_service.parse_time_by_lead_csv(
            '\n'.join(daily_rows), target_lead_time=EXPECTED_LEAD_TIME
        )
        assert len(d_dates) == EXPECTED_NUM_DAYS
        assert d_leads == [1]
        assert d_mat.shape == (EXPECTED_NUM_DAYS, EXPECTED_LEAD_TIME)
        np.testing.assert_allclose(d_mat[0, 0], 0.0)
        np.testing.assert_allclose(d_mat[0, 1], 0.5)

    def test_parse_time_by_lead_csv_rejects_invalid_or_empty(self) -> None:
        """Verify empty CSVs and all-NaN uploads raise ValueError."""
        with pytest.raises(ValueError, match='empty'):
            finetune_service.parse_time_by_lead_csv('')

        bad_nan_csv = 'date,lead_1\n2020-01-01,NaN\n2020-01-02,NaN\n'
        with pytest.raises(ValueError, match='no finite values'):
            finetune_service.upload_finetune_csv(
                username=TEST_USER,
                basin_id=TEST_BASIN,
                data_type='streamflow',
                csv_content=bad_nan_csv,
            )

    def test_list_finetune_models_includes_dynamic_components(self) -> None:
        """Verify pretrained models expose dynamic tunable components."""
        models = finetune_service.list_finetune_models(TEST_USER)
        model_ids = {m['model_id'] for m in models}
        assert 'google-floodhub-settings-110-epochs' in model_ids
        assert '5-basin-example' in model_ids

        for m in models:
            assert 'tunable_component_names' in m
            comp_names = {c['name'] for c in m['tunable_components']}
            assert 'head' in comp_names
            assert 'forecast_lstm' in comp_names
            assert 'hindcast_lstm' in comp_names
            sub_ids = {
                sub['id']
                for c in m['tunable_components']
                for sub in c.get('subcomponents', [])
            }
            assert 'shared_embeddings_fc.local_forcing' in sub_ids

    def test_upload_csv_and_m3s_unit_conversion(self) -> None:
        """Verify m3/s streamflow CSV upload converts to mm/day."""
        dates = pd.date_range('2015-01-01', periods=60, freq='D')
        q_m3s = 10.0
        rows = ['date,lead_1'] + [
            f'{d.strftime("%Y-%m-%d")},{q_m3s}' for d in dates
        ]
        res = finetune_service.upload_finetune_csv(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            data_type='streamflow',
            csv_content='\n'.join(rows),
            filename='q_m3s.csv',
            units='m3/s',
            area_km2=573.6,
        )
        assert res['units'] == 'mm/day'
        expected_mm_d = q_m3s * 86.4 / 573.6
        assert res['mean_val'] == pytest.approx(expected_mm_d, abs=1e-3)

    def test_interactive_pause_and_resume_finetuning_workflow(self) -> None:
        """Verify interactive step-by-step pause-and-resume fine-tuning."""
        finetune_service.generate_sample_basin_csvs(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            num_days=180,
        )

        status_70 = finetune_service.get_finetune_workspace_status(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            model_id='5-basin-example',
            split_ratio=0.70,
        )
        status_85 = finetune_service.get_finetune_workspace_status(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            model_id='5-basin-example',
            split_ratio=0.85,
        )
        assert status_70['data_ready']
        assert (
            status_70['partition']['split_date']
            < status_85['partition']['split_date']
        )
        assert status_70['timeseries'] is not None
        assert len(status_70['timeseries']['dates']) == len(
            status_70['timeseries']['precip']
        )

        step1 = finetune_service.run_finetune_epochs(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            model_id='5-basin-example',
            epochs=EXPECTED_STEP1_EPOCH,
            learning_rate=1e-3,
            loss='MSE',
            learning_rate_strategy='constant',
            batch_size=64,
            finetune_modules=[
                'head',
                'shared_embeddings_fc.local_forcing',
            ],
            split_ratio=0.75,
        )
        assert step1['session']['status'] == 'paused'
        assert step1['session']['current_epoch'] == EXPECTED_STEP1_EPOCH
        history1 = step1['session']['epoch_history']
        assert len(history1) == EXPECTED_STEP1_HISTORY_LEN
        assert [e['epoch'] for e in history1] == [0, 1, 2]
        assert 'streamflow_sim_lead1' in history1[-1]

        step2 = finetune_service.run_finetune_epochs(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            model_id='5-basin-example',
            epochs=1,
            learning_rate=5e-4,
            loss='NSE',
            learning_rate_strategy='StepLR',
            batch_size=32,
            finetune_modules=['head', 'forecast_lstm'],
            split_ratio=0.75,
        )
        assert step2['session']['status'] == 'paused'
        assert step2['session']['current_epoch'] == EXPECTED_STEP2_EPOCH
        history2 = step2['session']['epoch_history']
        assert len(history2) == EXPECTED_STEP2_HISTORY_LEN
        assert [e['epoch'] for e in history2] == [0, 1, 2, 3]
        assert history2[-1]['learning_rate'] == pytest.approx(5e-4)

        reset_res = finetune_service.reset_finetune_session(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            model_id='5-basin-example',
        )
        assert reset_res['session']['current_epoch'] == 0
        assert reset_res['session']['epoch_history'] == []
        assert reset_res['data_ready']


class FineTuningHTTPServerTests(unittest.TestCase):
    """End-to-end HTTP API tests for `/api/finetune/*` endpoints."""

    server: ClassVar[ThreadingHTTPServer]
    port: ClassVar[int]
    server_thread: ClassVar[threading.Thread]

    @classmethod
    def setUpClass(cls) -> None:
        """Start local HTTP server for fine-tuning endpoint tests."""
        _populate_test_watershed_and_attrs(TEST_USER)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), EarthkitHydroHandler)
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down local HTTP server."""
        cls.server.shutdown()
        cls.server.server_close()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=60)
        headers = {'X-Earthkit-User': TEST_USER}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode('utf-8')
        conn.close()
        return resp.status, json.loads(raw)

    def test_http_endpoints_lifecycle(self) -> None:
        """Test /api/finetune/* HTTP routes from sample data to reset."""
        status, data = self._request('GET', '/api/finetune/models')
        assert status == HTTP_OK
        assert len(data.get('models', [])) >= MIN_EXPECTED_MODELS

        status, sample_data = self._request(
            'POST',
            '/api/finetune/sample-data',
            {
                'basin_id': TEST_BASIN,
                'model_id': 'google-floodhub-settings-110-epochs',
                'num_days': 160,
            },
        )
        assert status == HTTP_OK
        assert sample_data['workspace']['data_ready']

        status, ws = self._request(
            'GET',
            f'/api/finetune/status?basin_id={TEST_BASIN}'
            '&model_id=google-floodhub-settings-110-epochs&split_ratio=0.80',
        )
        assert status == HTTP_OK
        assert ws['partition']['split_ratio'] == pytest.approx(0.80, abs=0.02)

        status, run_res = self._request(
            'POST',
            '/api/finetune/run',
            {
                'basin_id': TEST_BASIN,
                'model_id': 'google-floodhub-settings-110-epochs',
                'epochs': 1,
                'learning_rate': 0.0005,
                'loss': 'CMALLoss',
                'learning_rate_strategy': 'constant',
                'batch_size': 64,
                'finetune_modules': [
                    'head',
                    'shared_embeddings_fc.local_forcing',
                ],
                'split_ratio': 0.80,
            },
        )
        assert status == HTTP_OK
        assert run_res['session']['current_epoch'] == 1
        assert len(run_res['session']['epoch_history']) == EXPECTED_STEP1_EPOCH

        status, reset_res = self._request(
            'POST',
            '/api/finetune/reset',
            {
                'basin_id': TEST_BASIN,
                'model_id': 'google-floodhub-settings-110-epochs',
            },
        )
        assert status == HTTP_OK
        assert reset_res['session']['current_epoch'] == 0

        status, clear_res = self._request(
            'POST',
            '/api/finetune/clear-data',
            {
                'basin_id': TEST_BASIN,
                'model_id': 'google-floodhub-settings-110-epochs',
            },
        )
        assert status == HTTP_OK
        assert not clear_res['workspace']['data_ready']


if __name__ == '__main__':
    unittest.main()
