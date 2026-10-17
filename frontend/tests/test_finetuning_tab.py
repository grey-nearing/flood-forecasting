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
import shutil
import threading
import unittest
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import frontend.tests as _isolated_profiles
from frontend import (
    finetune_service,
    profile_manager,
    realtime_forecast_service,
)
from frontend.config import STATIC_DIR
from frontend.server import EarthkitHydroHandler
from model.datautils.scaler import Scaler
from model.modelzoo import get_model
from model.utils.config import Config

if TYPE_CHECKING:
    from pathlib import Path

TEST_USER = 'test_ft_user'
_ISOLATED_PROFILES_PKG = _isolated_profiles.__name__
TEST_BASIN = 'camels_01022500'
SYNTH_MODEL_RUN = 'synthetic_reg'
SYNTH_MODEL_ID = f'user_{SYNTH_MODEL_RUN}'
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


def _seed_hermetic_user_model(username: str = TEST_USER) -> Path:
    """Create a lightweight hermetic pretrained user model in profile."""
    pm = profile_manager.get_profile_manager()
    run_dir = pm.get_model_runs_dir(username) / SYNTH_MODEL_RUN
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dummy_ds = xr.Dataset(
        {
            'era5_precip': (
                ('sample',),
                np.array([1.0, 3.0, 5.0], dtype=np.float32),
            ),
            'area': (
                ('sample',),
                np.array([100.0, 500.0, 900.0], dtype=np.float32),
            ),
            'p_mean': (
                ('sample',),
                np.array([2.0, 3.5, 5.0], dtype=np.float32),
            ),
            'streamflow': (
                ('sample',),
                np.array([0.5, 2.5, 6.0], dtype=np.float32),
            ),
        }
    )
    scaler = Scaler(
        scaler_dir=run_dir,
        calculate_scaler=True,
        custom_normalization={},
        dataset=dummy_ds,
    )
    if scaler.scaler is not None:
        scaler.scaler = scaler.scaler.compute().astype(np.float32)
    scaler.save()

    cfg_dict: dict[str, object] = {
        'experiment_name': SYNTH_MODEL_RUN,
        'run_dir': str(run_dir),
        'base_run_dir': str(run_dir),
        'dataset': 'multimet',
        'model': 'mean_embedding_forecast_lstm',
        'head': 'regression',
        'output_activation': 'linear',
        'loss': 'MSE',
        'optimizer': 'Adam',
        'initial_learning_rate': 1e-3,
        'learning_rate_strategy': 'ConstantLR',
        'batch_size': 32,
        'epochs': 1,
        'hidden_size': 8,
        'seq_length': 14,
        'forecast_overlap': 14,
        'lead_time': EXPECTED_LEAD_TIME,
        'predict_last_n': EXPECTED_LEAD_TIME,
        'hindcast_inputs': {'era5': ['era5_precip']},
        'forecast_inputs': {'era5': ['era5_precip']},
        'static_attributes': ['area', 'p_mean'],
        'target_variables': ['streamflow'],
        'statics_embedding': {'type': 'fc', 'hiddens': [8]},
        'dynamics_embedding': {'type': 'fc', 'hiddens': [8]},
        'nan_handling_method': 'masked_mean',
        'nan_handling_pos_encoding': True,
        'initial_forget_bias': 3,
        'dropout': 0.0,
        'device': 'cpu',
        'compile': False,
        'seed': 42,
    }
    cfg = Config(cfg_dict, dev_mode=True)
    cfg.dump_config(folder=run_dir)
    torch.manual_seed(42)
    model = get_model(cfg).to(torch.device('cpu'))
    torch.save(model.state_dict(), run_dir / 'model_epoch001.pt')
    return run_dir


def _build_test_csv_strings(
    num_days: int = 80, lead_time: int = EXPECTED_LEAD_TIME
) -> dict[str, str]:
    """Build deterministic test CSV strings for precip, temp, and streamflow."""
    dates = pd.date_range('2021-01-01', periods=num_days, freq='D')
    header = 'date,' + ','.join(f'lead_{k}' for k in range(1, lead_time + 1))
    p_rows = [header]
    t_rows = [header]
    q_rows = [header]
    for idx, dt in enumerate(dates):
        d_str = dt.strftime('%Y-%m-%d')
        p_vals = [
            f'{max(0.0, 3.0 * np.sin(0.2 * (idx + k)) + 2.0):.3f}'
            for k in range(lead_time)
        ]
        t_vals = [
            f'{12.0 + 6.0 * np.cos(0.1 * (idx + k)):.3f}'
            for k in range(lead_time)
        ]
        q_vals = [
            f'{1.5 + 0.8 * np.sin(0.2 * (idx + k - 1)) + 1.0:.3f}'
            for k in range(lead_time)
        ]
        p_rows.append(f'{d_str},' + ','.join(p_vals))
        t_rows.append(f'{d_str},' + ','.join(t_vals))
        q_rows.append(f'{d_str},' + ','.join(q_vals))
    return {
        'precip': '\n'.join(p_rows),
        'temp': '\n'.join(t_rows),
        'streamflow': '\n'.join(q_rows),
    }


def _upload_all_test_csvs(
    username: str = TEST_USER,
    basin_id: str = TEST_BASIN,
    num_days: int = 80,
    lead_time: int = EXPECTED_LEAD_TIME,
) -> dict[str, str]:
    """Upload precip, temp, and streamflow CSVs for a test basin."""
    csv_map = _build_test_csv_strings(num_days=num_days, lead_time=lead_time)
    for dtype, content in csv_map.items():
        units = 'degC' if dtype == 'temp' else 'mm/day'
        finetune_service.upload_finetune_csv(
            username=username,
            basin_id=basin_id,
            data_type=dtype,
            csv_content=content,
            filename=f'{dtype}.csv',
            units=units,
            target_lead_time=lead_time,
        )
    return csv_map


@pytest.mark.unit
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
        assert 'ftLoadSampleDataBtn' not in inspector.by_id
        assert 'ftBasinIdInput' not in inspector.by_id

        required_ft_ids = [
            'ftModelSelect',
            'ftBasinSelect',
            'ftBasinLoadingBadge',
            'ftGlobalForcingCard',
            'ftGlobalArchiveBadge',
            'ftGlobalCoverageSummaryText',
            'ftGlobalProductBadgesList',
            'ftExtractGlobalForcingBtn',
            'ftCancelGlobalExtractBtn',
            'ftGlobalExtractProgressBox',
            'ftForcingModeSelect',
            'ftDisplayUnitsSelect',
            'ftPrecipFileInput',
            'ftUploadPrecipBtn',
            'ftTempFileInput',
            'ftUploadTempBtn',
            'ftStreamflowFileInput',
            'ftUploadStreamflowBtn',
            'ftStreamflowUnitsSelect',
            'ftBasinAreaInput',
            'ftDownloadTemplateBtn',
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
            'ftSaveEpochModelBtn',
            'ftSaveModelPromptBox',
            'ftSaveModelNameInput',
            'ftConfirmSaveModelBtn',
            'ftCancelSaveModelBtn',
            'ftGlobalProductsStatusStrip',
            'ftForcingLegendContainer',
            'ftForcingLegendGrid',
            'ftInputForcingCanvas',
            'ftHydrographCanvas',
            'ftHydrographLoadingOverlay',
            'ftHydrographLoadingTitle',
            'ftHydrographLoadingSubtitle',
            'ftHydrographLoadingEpochCounter',
            'ftHydrographLoadingBarFill',
            'ftLossCurveCanvas',
            'ftEpochHistoryTableBody',
        ]
        for el_id in required_ft_ids:
            assert el_id in inspector.by_id, f'Missing #{el_id} in HTML'
            assert 'tab-finetuning' in inspector.by_id[el_id]['ancestor_ids'], (
                f'#{el_id} must be inside #tab-finetuning'
            )

        for train_id in (
            'trainingSidebar',
            'extractHistBtn',
            'trainBaseCheckpoint',
        ):
            assert train_id in inspector.by_id
            assert 'tab-training' in inspector.by_id[train_id]['ancestor_ids']


@pytest.mark.unit
class FineTuneServiceUnitTests(unittest.TestCase):
    """Unit tests for CSV validation, model inventory, and error handling."""

    def setUp(self) -> None:
        """Prepare test basin, hermetic model, and clean workspace."""
        _populate_test_watershed_and_attrs(TEST_USER)
        _seed_hermetic_user_model(TEST_USER)
        finetune_service.clear_finetune_data(TEST_USER, TEST_BASIN)

    def test_parse_time_by_lead_csv_matrix_and_daily(self) -> None:
        """Verify 7-lead CSV and 1-column CSV (with NaNs at leads 2..7)."""
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
        np.testing.assert_allclose(d_mat[1, 0], 0.5)
        assert np.all(np.isnan(d_mat[:, 1:]))

    def test_parse_time_by_lead_csv_strict_validation(self) -> None:
        """Verify malformed CSVs raise explicit ValueError without fallback."""
        with pytest.raises(ValueError, match='empty'):
            finetune_service.parse_time_by_lead_csv('')

        with pytest.raises(ValueError, match='explicit date column'):
            finetune_service.parse_time_by_lead_csv(
                'station_id,lead_1\n01022500,1.2\n'
            )

        with pytest.raises(ValueError, match='unparseable date'):
            finetune_service.parse_time_by_lead_csv(
                'date,lead_1\nnot-a-date,1.0\n2020-01-02,2.0\n'
            )

        with pytest.raises(ValueError, match='duplicate dates'):
            finetune_service.parse_time_by_lead_csv(
                'date,lead_1\n2020-01-01,1.0\n2020-01-01,2.0\n'
            )

        with pytest.raises(ValueError, match='non-numeric'):
            finetune_service.parse_time_by_lead_csv(
                'date,lead_1\n2020-01-01,1.0\n2020-01-02,ERR\n'
            )

        with pytest.raises(ValueError, match='Unrecognized lead column'):
            finetune_service.parse_time_by_lead_csv(
                'date,lead_1,foo_col\n2020-01-01,1.0,2.0\n'
            )

        with pytest.raises(ValueError, match='Invalid lead column'):
            finetune_service.parse_time_by_lead_csv(
                'date,lead_0\n2020-01-01,1.0\n'
            )

    def test_upload_csv_physical_and_unit_validation(self) -> None:
        """Verify negative precip/flow, bad units, and missing area fail."""
        neg_csv = 'date,lead_1\n2020-01-01,-1.5\n2020-01-02,2.0\n'
        with pytest.raises(ValueError, match='cannot be negative'):
            finetune_service.upload_finetune_csv(
                TEST_USER, TEST_BASIN, 'precip', neg_csv
            )
        with pytest.raises(ValueError, match='cannot be negative'):
            finetune_service.upload_finetune_csv(
                TEST_USER, TEST_BASIN, 'streamflow', neg_csv
            )

        valid_csv = 'date,lead_1\n2020-01-01,10.0\n2020-01-02,12.0\n'
        with pytest.raises(ValueError, match='Unsupported streamflow units'):
            finetune_service.upload_finetune_csv(
                TEST_USER,
                TEST_BASIN,
                'streamflow',
                valid_csv,
                units='ft3/s',
            )
        with pytest.raises(ValueError, match='area_km2'):
            finetune_service.upload_finetune_csv(
                TEST_USER,
                'unknown_basin_without_polygon_area',
                'streamflow',
                valid_csv,
                units='m3/s',
            )

        res_auto_area = finetune_service.upload_finetune_csv(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            data_type='streamflow',
            csv_content=valid_csv,
            filename='q_m3s_auto.csv',
            units='m3/s',
        )
        assert res_auto_area['units'] == 'mm/day'
        assert res_auto_area['area_km2'] == pytest.approx(573.6)
        expected_mm_d = 11.0 * 86.4 / 573.6
        assert res_auto_area['mean_val'] == pytest.approx(
            expected_mm_d, abs=1e-3
        )

    def test_clipped_user_basins_clean_names_and_auto_area(self) -> None:
        """Verify clipped basins strip catchment_hydroatlas_* and have area."""
        pm = profile_manager.get_profile_manager()
        hydroatlas_cid = 'catchment_hydroatlas_HYRIV_70689198_35212_81698'
        named_feat = {
            'type': 'Feature',
            'id': hydroatlas_cid,
            'properties': {
                'catchment_id': hydroatlas_cid,
                'name': 'Broad River near Boiling Springs, NC (USGS 02151500)',
                'area_km2': 2249.8,
                'source': 'delineated',
            },
            'geometry': TEST_WATERSHED_FEATURE['geometry'],
        }
        pm.save_watersheds(
            [named_feat, TEST_WATERSHED_FEATURE], username=TEST_USER
        )

        clipped = finetune_service.list_user_clipped_basins(TEST_USER)
        assert len(clipped) == MIN_EXPECTED_MODELS
        assert clipped[0]['basin_id'] == hydroatlas_cid
        assert (
            clipped[0]['name']
            == 'Broad River near Boiling Springs, NC (USGS 02151500)'
        )
        assert 'catchment_hydroatlas_' not in str(clipped[0]['name'])
        assert clipped[0]['area_km2'] == pytest.approx(2249.8)

        ws = finetune_service.get_finetune_workspace_status(
            username=TEST_USER,
            basin_id='',
            model_id=SYNTH_MODEL_ID,
        )
        assert ws['basin_id'] == hydroatlas_cid
        assert (
            ws['basin_name']
            == 'Broad River near Boiling Springs, NC (USGS 02151500)'
        )
        assert ws['basin_area_km2'] == pytest.approx(2249.8)
        assert len(ws['user_basins']) == MIN_EXPECTED_MODELS

    def test_list_finetune_models_and_strict_model_id_resolution(self) -> None:
        """Verify model inventory and strict FileNotFoundError on unknown id."""
        models = finetune_service.list_finetune_models(TEST_USER)
        model_ids = {m['model_id'] for m in models}
        assert 'google-floodhub-settings-110-epochs' in model_ids
        assert SYNTH_MODEL_ID in model_ids
        assert '5-basin-example' not in model_ids

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

        with pytest.raises(FileNotFoundError, match='Unknown model_id'):
            finetune_service.get_finetune_workspace_status(
                TEST_USER, TEST_BASIN, model_id='non_existent_model_id'
            )

        with pytest.raises(ValueError, match='Invalid user model_id'):
            finetune_service.get_finetune_workspace_status(
                TEST_USER, TEST_BASIN, model_id='user_../escape'
            )

    def test_missing_static_attributes_raises_instead_of_scaler_fallback(
        self,
    ) -> None:
        """Verify missing basin static attributes raise FileNotFoundError."""
        unlisted_basin = 'unlisted_basin_999'
        _upload_all_test_csvs(TEST_USER, unlisted_basin, num_days=60)
        with pytest.raises(
            FileNotFoundError, match='Static attributes for basin'
        ):
            finetune_service.run_finetune_epochs(
                username=TEST_USER,
                basin_id=unlisted_basin,
                model_id=SYNTH_MODEL_ID,
                epochs=1,
            )

    def test_upload_streamflow_csv_with_basin_column_and_multi_basin(
        self,
    ) -> None:
        """Verify basin,date,streamflow CSV auto-switches or multi-uploads."""
        pm = profile_manager.get_profile_manager()
        basin_b = 'catchment_hydroatlas_HYRIV_70582588_39475_80998'
        feat_b = {
            'type': 'Feature',
            'id': basin_b,
            'properties': {
                'catchment_id': basin_b,
                'name': 'Georges Creek at Franklin, MD (USGS 01599000)',
                'area_km2': 187.5,
                'source': 'delineated',
            },
            'geometry': TEST_WATERSHED_FEATURE['geometry'],
        }
        pm.save_watersheds([TEST_WATERSHED_FEATURE, feat_b], username=TEST_USER)
        finetune_service.clear_finetune_data(TEST_USER, basin_b)

        single_basin_csv = (
            'basin,date,streamflow\n'
            f'{basin_b},2025-04-01,10.0\n'
            f'{basin_b},2025-04-02,20.0\n'
        )
        res_single = finetune_service.upload_finetune_csv(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            data_type='streamflow',
            csv_content=single_basin_csv,
            filename='usgs_01599000_streamflow.csv',
            units='m3/s',
        )
        assert res_single['basin_id'] == basin_b
        assert res_single['area_km2'] == pytest.approx(187.5)
        expected_mean = 15.0 * 86.4 / 187.5
        assert res_single['mean_val'] == pytest.approx(expected_mean, abs=1e-3)

        multi_basin_csv = (
            'basin,date,streamflow\n'
            f'{TEST_BASIN},2025-04-01,10.0\n'
            f'{TEST_BASIN},2025-04-02,12.0\n'
            f'{basin_b},2025-04-01,5.0\n'
            f'{basin_b},2025-04-02,7.0\n'
        )
        res_multi = finetune_service.upload_finetune_csv(
            username=TEST_USER,
            basin_id=TEST_BASIN,
            data_type='streamflow',
            csv_content=multi_basin_csv,
            filename='usgs_all_basins_streamflow.csv',
            units='m3/s',
        )
        assert set(res_multi['basins_uploaded']) == {TEST_BASIN, basin_b}
        assert res_multi['basin_id'] == TEST_BASIN

    def test_streamflow_only_upload_falls_back_to_realtime_dynamics_forcing(
        self,
    ) -> None:
        """Verify streamflow-only CSV uses realtime/dynamics/HRES forcing."""
        pm = profile_manager.get_profile_manager()
        rt_dyn_dir = pm.get_realtime_dir(TEST_USER) / 'dynamics'
        rt_hres_zarr = rt_dyn_dir / 'HRES' / 'timeseries.zarr'
        rt_hres_zarr.parent.mkdir(parents=True, exist_ok=True)
        try:
            dates = pd.date_range('2025-01-01', periods=70, freq='D')
            leads = np.arange(1, EXPECTED_LEAD_TIME + 1, dtype=np.int64)
            t_idx = np.arange(len(dates), dtype=np.float32)[
                np.newaxis, :, np.newaxis
            ]
            p_mm = np.full(
                (1, len(dates), EXPECTED_LEAD_TIME), 5.0, dtype=np.float32
            ) + 0.008 * np.sin(0.3 * t_idx)
            t_degc = np.full(
                (1, len(dates), EXPECTED_LEAD_TIME), 15.0, dtype=np.float32
            ) + 2.0 * np.cos(0.2 * t_idx)
            ds_rt = xr.Dataset(
                {
                    'hres_total_precipitation': (
                        ('basin', 'date', 'lead_time'),
                        p_mm,
                    ),
                    'hres_temperature_2m': (
                        ('basin', 'date', 'lead_time'),
                        t_degc,
                    ),
                },
                coords={
                    'basin': np.array([TEST_BASIN], dtype=object),
                    'date': dates.to_numpy().astype('datetime64[ns]'),
                    'lead_time': leads,
                },
            )
            ds_rt.to_zarr(rt_hres_zarr, mode='w', consolidated=True)

            # First run a local_only evaluation so global_dyn_placeholders is
            # populated with all-NaN placeholders and cached in _open_zarr, then
            # switch to global_only and a shifted date range to verify that
            # _open_zarr cache invalidation prevents stale NoTrainDataError.
            _upload_all_test_csvs(
                TEST_USER,
                TEST_BASIN,
                num_days=65,
            )
            ws_local = finetune_service.get_finetune_workspace_status(
                username=TEST_USER,
                basin_id=TEST_BASIN,
                model_id=SYNTH_MODEL_ID,
                forcing_mode='local_only',
            )
            assert ws_local['data_ready'] is True
            finetune_service.clear_finetune_data(TEST_USER, TEST_BASIN)

            q_rows = ['basin,date,streamflow']
            for i, dt in enumerate(dates):
                d_str = dt.strftime('%Y-%m-%d')
                q_rows.append(f'{TEST_BASIN},{d_str},{10.0 + 0.2 * i:.2f}')
            finetune_service.upload_finetune_csv(
                username=TEST_USER,
                basin_id=TEST_BASIN,
                data_type='streamflow',
                csv_content='\n'.join(q_rows),
                filename='usgs_streamflow.csv',
                units='m3/s',
            )

            ws = finetune_service.get_finetune_workspace_status(
                username=TEST_USER,
                basin_id=TEST_BASIN,
                model_id=SYNTH_MODEL_ID,
            )
            assert ws['data_ready'] is True
            assert ws['forcing_mode'] == 'global_only'
            assert ws['global_forcing_status']['has_basin_archive'] is True
            assert (
                ws['global_forcing_status']['fully_covers_streamflow'] is True
            )
            assert (
                'HRES'
                in ws['global_forcing_status']['available_global_products']
            )
            assert ws['timeseries']['input_products']['HRES']['has_precip']
            assert ws['timeseries']['input_products']['HRES']['has_temp']
            assert len(ws['timeseries']['dates']) == len(dates)
            assert np.allclose(ws['timeseries']['precip'], 5.0, atol=1e-2)
            assert np.allclose(
                ws['timeseries']['precip_global'], 5.0, atol=1e-2
            )
            assert ws['timeseries']['precip_local'] == []
            assert len(ws['timeseries']['streamflow_sim_base']) == len(dates)
            assert ws['session']['epoch_history'] == []
            assert ws['partition']['total_days'] == len(dates)

            step_global = finetune_service.run_finetune_epochs(
                username=TEST_USER,
                basin_id=TEST_BASIN,
                model_id=SYNTH_MODEL_ID,
                epochs=1,
                forcing_mode='global_only',
            )
            assert step_global['session']['current_epoch'] == 1
            assert len(step_global['session']['epoch_history']) == (
                EXPECTED_STEP1_EPOCH
            )
            prog = finetune_service.get_finetune_run_progress(
                TEST_USER, TEST_BASIN
            )
            assert (prog['status'], prog['step_epoch'], prog['percent']) == (
                'completed',
                1,
                pytest.approx(100.0),
            )
        finally:
            shutil.rmtree(rt_dyn_dir, ignore_errors=True)


@pytest.mark.integration
class FineTuneWorkflowIntegrationTests(unittest.TestCase):
    """Integration tests for interactive fine-tuning, freezing, and HTTP API."""

    _USER = 'test_ft_workflow_user'

    def setUp(self) -> None:
        """Prepare test basin, hermetic model, and clean workspace."""
        _populate_test_watershed_and_attrs(self._USER)
        _seed_hermetic_user_model(self._USER)
        finetune_service.clear_finetune_data(self._USER, TEST_BASIN)

    def test_interactive_pause_resume_and_module_freezing(self) -> None:
        """Verify frozen weights stay identical while unfrozen weights train."""
        _upload_all_test_csvs(self._USER, TEST_BASIN, num_days=75)

        status_70 = finetune_service.get_finetune_workspace_status(
            username=self._USER,
            basin_id=TEST_BASIN,
            model_id=SYNTH_MODEL_ID,
            split_ratio=0.70,
        )
        status_85 = finetune_service.get_finetune_workspace_status(
            username=self._USER,
            basin_id=TEST_BASIN,
            model_id=SYNTH_MODEL_ID,
            split_ratio=0.85,
        )
        assert status_70['data_ready']
        part_70 = status_70['partition']
        assert part_70['split_date'] < status_85['partition']['split_date']
        gap_days = (
            pd.Timestamp(part_70['validation_start_date'])
            - pd.Timestamp(part_70['train_end_date'])
        ).days
        assert gap_days == EXPECTED_LEAD_TIME

        step1 = finetune_service.run_finetune_epochs(
            username=self._USER,
            basin_id=TEST_BASIN,
            model_id=SYNTH_MODEL_ID,
            epochs=EXPECTED_STEP1_EPOCH,
            learning_rate=5e-3,
            loss='MSE',
            learning_rate_strategy='ConstantLR',
            batch_size=16,
            finetune_modules=['head', 'shared_embeddings_fc.local_forcing'],
            split_ratio=0.70,
        )
        assert step1['session']['status'] == 'paused'
        assert step1['session']['current_epoch'] == EXPECTED_STEP1_EPOCH
        history1 = step1['session']['epoch_history']
        assert len(history1) == EXPECTED_STEP1_HISTORY_LEN
        assert [e['epoch'] for e in history1] == [0, 1, 2]

        pm = profile_manager.get_profile_manager()
        base_ckpt = (
            pm.get_model_runs_dir(self._USER)
            / SYNTH_MODEL_RUN
            / 'model_epoch001.pt'
        )
        sess_run_dir = (
            pm.get_profile_dir(self._USER)
            / 'finetune_workspace'
            / TEST_BASIN
            / SYNTH_MODEL_ID
            / 'run'
        )
        ft_ckpt_2 = sess_run_dir / 'model_epoch002.pt'
        opt_ckpt_2 = sess_run_dir / 'optimizer_state_epoch002.pt'
        assert ft_ckpt_2.is_file()
        assert opt_ckpt_2.is_file()

        base_sd = torch.load(base_ckpt, map_location='cpu', weights_only=True)
        ft_sd = torch.load(ft_ckpt_2, map_location='cpu', weights_only=True)
        for key, base_tensor in base_sd.items():
            if key.startswith(('hindcast_lstm.', 'forecast_lstm.')):
                torch.testing.assert_close(
                    ft_sd[key], base_tensor, rtol=0.0, atol=0.0
                )
        assert not torch.equal(
            ft_sd['head.net.0.weight'], base_sd['head.net.0.weight']
        )

        step2 = finetune_service.run_finetune_epochs(
            username=self._USER,
            basin_id=TEST_BASIN,
            model_id=SYNTH_MODEL_ID,
            epochs=1,
            learning_rate=5e-4,
            loss='NSE',
            learning_rate_strategy='StepLR',
            batch_size=16,
            finetune_modules=['head', 'forecast_lstm'],
            split_ratio=0.70,
        )
        assert step2['session']['status'] == 'paused'
        assert step2['session']['current_epoch'] == EXPECTED_STEP2_EPOCH
        history2 = step2['session']['epoch_history']
        assert len(history2) == EXPECTED_STEP2_HISTORY_LEN
        assert [e['epoch'] for e in history2] == [0, 1, 2, 3]
        assert history2[-1]['learning_rate'] == pytest.approx(5e-4)

        with pytest.raises(KeyError, match='Unknown finetune module'):
            finetune_service.run_finetune_epochs(
                username=self._USER,
                basin_id=TEST_BASIN,
                model_id=SYNTH_MODEL_ID,
                epochs=1,
                finetune_modules=['non_existent_layer'],
            )

        reset_res = finetune_service.reset_finetune_session(
            username=self._USER,
            basin_id=TEST_BASIN,
            model_id=SYNTH_MODEL_ID,
        )
        assert reset_res['session']['current_epoch'] == 0
        assert reset_res['session']['epoch_history'] == []
        assert reset_res['data_ready']

    def test_save_finetune_checkpoint_to_inventory(self) -> None:
        """Verify saving an epoch checkpoint adds it to both inventories."""
        _upload_all_test_csvs(self._USER, TEST_BASIN, num_days=75)
        finetune_service.run_finetune_epochs(
            username=self._USER,
            basin_id=TEST_BASIN,
            model_id=SYNTH_MODEL_ID,
            epochs=2,
            learning_rate=5e-3,
            loss='MSE',
            learning_rate_strategy='ConstantLR',
            batch_size=16,
            finetune_modules=['head', 'shared_embeddings_fc.local_forcing'],
            split_ratio=0.70,
        )
        pm = profile_manager.get_profile_manager()
        ft_ckpt_2 = (
            pm.get_profile_dir(self._USER)
            / 'finetune_workspace'
            / TEST_BASIN
            / SYNTH_MODEL_ID
            / 'run'
            / 'model_epoch002.pt'
        )
        ft_sd = torch.load(ft_ckpt_2, map_location='cpu', weights_only=True)

        saved_info = finetune_service.save_finetune_checkpoint_to_inventory(
            username=self._USER,
            basin_id=TEST_BASIN,
            model_id=SYNTH_MODEL_ID,
            epoch=2,
            model_name='My Saved Basin Model',
        )
        assert saved_info['status'] == 'saved'
        saved_id = str(saved_info['saved_model_id'])
        assert saved_id == 'user_my_saved_basin_model'

        ft_models = finetune_service.list_finetune_models(self._USER)
        ft_saved = next(
            (m for m in ft_models if m['model_id'] == saved_id), None
        )
        assert ft_saved is not None
        assert ft_saved['label'] == 'My Saved Basin Model (Epoch 2)'

        fc_models = realtime_forecast_service.list_forecast_models(self._USER)
        fc_saved = next((m for m in fc_models if m['id'] == saved_id), None)
        assert fc_saved is not None
        assert fc_saved['label'] == 'My Saved Basin Model (Epoch 2)'
        assert fc_saved['available'] is True

        bundle = realtime_forecast_service._load_forecast_model_bundle(  # noqa: SLF001
            model_id=saved_id,
            username=self._USER,
        )
        assert 'model_epoch002.pt' in bundle['checkpoint_path']
        saved_head_w = bundle['model'].state_dict()['head.net.0.weight']
        torch.testing.assert_close(
            saved_head_w.cpu(),
            ft_sd['head.net.0.weight'].cpu(),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.integration
class FineTuningHTTPServerTests(unittest.TestCase):
    """End-to-end HTTP API tests for `/api/finetune/*` endpoints."""

    _USER = 'test_ft_http_user'
    server: ClassVar[ThreadingHTTPServer]
    port: ClassVar[int]
    server_thread: ClassVar[threading.Thread]

    @classmethod
    def setUpClass(cls) -> None:
        """Start local HTTP server for fine-tuning endpoint tests."""
        _populate_test_watershed_and_attrs(cls._USER)
        _seed_hermetic_user_model(cls._USER)
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
        headers = {'X-User-Profile': self._USER}
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
        """Test /api/finetune/* HTTP routes from CSV upload to reset."""
        finetune_service.clear_finetune_data(self._USER, TEST_BASIN)

        status, data = self._request('GET', '/api/finetune/models')
        assert status == HTTP_OK
        assert len(data.get('models', [])) >= MIN_EXPECTED_MODELS

        csv_map = _build_test_csv_strings(num_days=75)
        for dtype, content in csv_map.items():
            st_up, up_res = self._request(
                'POST',
                '/api/finetune/upload-csv',
                {
                    'basin_id': TEST_BASIN,
                    'model_id': SYNTH_MODEL_ID,
                    'data_type': dtype,
                    'filename': f'{dtype}.csv',
                    'csv_content': content,
                    'units': 'degC' if dtype == 'temp' else 'mm/day',
                },
            )
            assert st_up == HTTP_OK
            assert up_res['status'] == 'uploaded'

        status, ws = self._request(
            'GET',
            f'/api/finetune/status?basin_id={TEST_BASIN}'
            f'&model_id={SYNTH_MODEL_ID}&split_ratio=0.80',
        )
        assert status == HTTP_OK
        assert ws['data_ready']
        assert ws['partition']['split_ratio'] == pytest.approx(0.80, abs=0.03)

        status, run_res = self._request(
            'POST',
            '/api/finetune/run',
            {
                'basin_id': TEST_BASIN,
                'model_id': SYNTH_MODEL_ID,
                'epochs': 1,
                'learning_rate': 0.001,
                'loss': 'MSE',
                'learning_rate_strategy': 'ConstantLR',
                'batch_size': 16,
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

        st_prog, prog_res = self._request(
            'GET',
            f'/api/finetune/run-progress?basin_id={TEST_BASIN}',
        )
        assert st_prog == HTTP_OK
        assert prog_res['status'] == 'completed'
        assert prog_res['step_epoch'] == 1
        assert prog_res['total_step_epochs'] == 1
        assert prog_res['percent'] == pytest.approx(100.0)

        st_save, save_res = self._request(
            'POST',
            '/api/finetune/save-model',
            {
                'basin_id': TEST_BASIN,
                'model_id': SYNTH_MODEL_ID,
                'epoch': 1,
                'model_name': 'Narraguagus Tuned Ep 1',
            },
        )
        assert st_save == HTTP_OK
        assert save_res['status'] == 'saved'
        assert save_res['saved_model_id'] == 'user_narraguagus_tuned_ep_1'
        ft_ids = [m['model_id'] for m in save_res.get('models', [])]
        assert 'user_narraguagus_tuned_ep_1' in ft_ids
        fc_ids = [m['id'] for m in save_res.get('forecast_models', [])]
        assert 'user_narraguagus_tuned_ep_1' in fc_ids
        fc_entry = next(
            m
            for m in save_res['forecast_models']
            if m['id'] == 'user_narraguagus_tuned_ep_1'
        )
        assert fc_entry['label'] == 'Narraguagus Tuned Ep 1 (Epoch 1)'
        assert fc_entry['available'] is True

        status, reset_res = self._request(
            'POST',
            '/api/finetune/reset',
            {
                'basin_id': TEST_BASIN,
                'model_id': SYNTH_MODEL_ID,
            },
        )
        assert status == HTTP_OK
        assert reset_res['session']['current_epoch'] == 0

        status, clear_res = self._request(
            'POST',
            '/api/finetune/clear-data',
            {
                'basin_id': TEST_BASIN,
                'model_id': SYNTH_MODEL_ID,
            },
        )
        assert status == HTTP_OK
        assert not clear_res['workspace']['data_ready']


if __name__ == '__main__':
    unittest.main()
