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
"""Hermetic tests for automatic forecast synchronization and frames."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from frontend import weather_engine as we
from frontend import weather_sync as ws
from multimet.weather_fetcher.config import N_LAT, N_LON
from multimet.weather_fetcher.sync import swap_current_symlink

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_MAX_LEAD_HOUR = 240
_EXPECTED_STEPS = 81
_EXPECTED_MSLP_HPA = 1013.2
_EXPECTED_TEMP_C = 12.5
_MAX_AIFS_LEAD_H = 24
_EXPECTED_PRESSURE_POINT_HPA = 1013.0


def _fake_forecast(
    init_times: Sequence[np.datetime64],
    lead_hours: Sequence[int],
    *,
    members: bool = False,
) -> xr.Dataset:
    """Build a 721 x 1440 dynamical.org-schema forecast on the 0.25 deg grid."""
    shape = (len(init_times), len(lead_hours), N_LAT, N_LON)
    precip = np.full(shape, 1.0 / 3600.0, np.float32)
    if int(lead_hours[0]) == 0:
        precip[:, 0] = np.nan
    data: dict[str, tuple[np.ndarray, str]] = {
        'precipitation_surface': (precip, 'kg m-2 s-1'),
        'temperature_2m': (np.full(shape, 12.5, np.float32), 'degree_Celsius'),
        'pressure_reduced_to_mean_sea_level': (
            np.full(shape, 101325.0, np.float32),
            'Pa',
        ),
        'wind_u_10m': (np.full(shape, 3.0, np.float32), 'm s-1'),
        'wind_v_10m': (np.full(shape, -4.0, np.float32), 'm s-1'),
    }
    coords: dict[str, Any] = {
        'init_time': np.array(init_times, dtype='datetime64[ns]'),
        'lead_time': np.array(lead_hours, dtype='timedelta64[h]').astype(
            'timedelta64[ns]'
        ),
        'latitude': np.linspace(90.0, -90.0, N_LAT, dtype=np.float32),
        'longitude': np.linspace(
            -180.0, 180.0, N_LON, endpoint=False, dtype=np.float32
        ),
    }
    dims: tuple[str, ...] = (
        'init_time',
        'lead_time',
        'latitude',
        'longitude',
    )
    if members:
        coords['ensemble_member'] = np.arange(3)
        dims = (
            'init_time',
            'lead_time',
            'ensemble_member',
            'latitude',
            'longitude',
        )
        data = {
            k: (np.repeat(v[:, :, None], 3, axis=2), u)
            for k, (v, u) in data.items()
        }
    return xr.Dataset(
        {k: (dims, v, {'units': u}) for k, (v, u) in data.items()},
        coords=coords,
    )


@pytest.mark.unit
class WeatherSyncTest(unittest.TestCase):
    """Tests model synchronization, atomic directory swapping, and locking."""

    def setUp(self) -> None:
        """Create a temporary sync root directory."""
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory(prefix='ohn_weather_sync_')
        self.root = self.temp_dir.name

    def tearDown(self) -> None:
        """Clean up the temporary sync root directory."""
        self.temp_dir.cleanup()
        super().tearDown()

    def _sync(
        self,
        models: Sequence[str],
        open_ds: Callable[[Any, str], Any],
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        return ws.sync_latest(
            root=self.root,
            models=models,
            catalog=object(),
            open_dataset=open_ds,
            log=lambda _m: None,
            **kwargs,
        )

    def _partial_dirs(self) -> list[str]:
        return sorted(p.name for p in Path(self.root, 'runs').glob('*.partial'))

    def test_output_lead_hours_keeps_viewer_steps_up_to_240h(self) -> None:
        """Verify 3-hourly viewer step filtering up to 240h."""
        gfs = list(range(121)) + list(range(123, 385, 3))
        out = ws.output_lead_hours(gfs)
        assert out[:4] == [0, 3, 6, 9]
        assert out[-1] == _MAX_LEAD_HOUR
        assert len(out) == _EXPECTED_STEPS
        assert ws.output_lead_hours([0, 6, 12, 246]) == [0, 6, 12]

    def test_aggregate_rates_averages_hourly_rain_over_each_step(self) -> None:
        """Verify sub-step rate aggregation and NaN propagation."""
        leads = [0, 1, 2, 3, 4, 5, 6]
        rates = np.array([np.nan, 1, 1, 1, 4, 4, 4], np.float32)[:, None, None]
        out = ws.aggregate_rates(rates, leads, [0, 3, 6])
        np.testing.assert_allclose(out[:, 0, 0], [0.0, 1.0, 4.0])

        out = ws.aggregate_rates(
            np.array([0, 2, 2, 1], np.float32)[:, None, None],
            [0, 1, 2, 5],
            [0, 2, 5],
        )
        np.testing.assert_allclose(out[:, 0, 0], [0.0, 2.0, 1.0])

        rates[4, 0, 0] = np.nan
        out = ws.aggregate_rates(rates, leads, [0, 3, 6])
        assert float(out[1, 0, 0]) == 1.0
        assert np.isnan(out[2, 0, 0])

    def test_units(self) -> None:
        """Verify unit conversion helpers for precipitation and pressure."""
        np.testing.assert_allclose(
            ws.to_stored_units('precip', [1 / 3600.0], 'kg m-2 s-1').astype(
                float
            ),
            [1.0],
            rtol=1e-3,
        )
        mslp_val = (
            float(ws.to_stored_units('mslp', [101325.0], 'Pa')[0])
            + ws.MSLP_OFFSET_HPA
        )
        assert round(mslp_val, 1) == _EXPECTED_MSLP_HPA

    def test_sync_downloads_new_runs_only_and_swaps_atomically(self) -> None:
        """Verify atomic run directory publishing and incremental updates."""
        t0 = np.datetime64('2026-09-29T00:00')
        t1 = np.datetime64('2026-09-29T06:00')
        hourly = list(range(7))
        catalogs = {
            'noaa-gfs-forecast': _fake_forecast([t0], hourly),
            'ecmwf-aifs-single-forecast': _fake_forecast([t0], [0, 6]),
        }

        def open_ds(_cat: object, dataset_id: str) -> xr.Dataset:
            return catalogs[dataset_id]

        models = ['noaa_gfs', 'ecmwf_aifs']

        status = self._sync(models, open_ds)
        assert status['last_result'] == 'updated'
        assert status['errors'] == {}
        assert status['sources']['noaa_gfs'] == 'dynamical.org'
        run1 = ws.current_run_dir(self.root)
        assert run1 is not None
        meta = json.loads(
            Path(run1, ws.RUN_METADATA_FILE).read_text(encoding='utf-8')
        )
        gfs = meta['datasets']['noaa_gfs_forecast']
        assert gfs['model'] == 'noaa_gfs'
        assert gfs['lead_hours'] == [0, 3, 6]
        precip = np.fromfile(Path(run1, 'noaa_gfs_precip.bin'), np.float16)
        precip = precip.reshape(3, N_LAT, N_LON)
        assert np.isfinite(precip).all()
        assert round(float(precip[1, 0, 0]), 2) == 1.0
        assert round(float(precip[2, 400, 900]), 2) == 1.0
        assert float(precip[0, 0, 0]) == 0.0
        assert self._partial_dirs() == []

        status = self._sync(models, open_ds)
        assert status['last_result'] == 'up_to_date'
        assert ws.current_run_dir(self.root) == run1

        catalogs['noaa-gfs-forecast'] = _fake_forecast([t0, t1], hourly)
        status = self._sync(models, open_ds)
        assert status['last_result'] == 'updated'
        assert status['updated_models'] == ['noaa_gfs']
        run2 = ws.current_run_dir(self.root)
        assert run2 is not None
        assert run2 != run1
        assert Path(run2, 'ecmwf_aifs_precip.bin').exists()
        assert (
            status['models']['noaa_gfs']['init_time'] == '2026-09-29T06:00:00'
        )
        assert (
            status['models']['ecmwf_aifs']['init_time'] == '2026-09-29T00:00:00'
        )
        assert (Path(self.root) / 'current').is_symlink()
        assert ws.read_sync_status(self.root)['last_result'] == 'updated'

    def test_incomplete_run_keeps_previous(self) -> None:
        """Verify an incomplete upstream run does not replace a valid run."""
        t0 = np.datetime64('2026-09-29T00:00')
        t1 = np.datetime64('2026-09-29T12:00')
        good = _fake_forecast([t0], [0, 6, 12])
        self._sync(['ecmwf_aifs'], lambda _cat, _id: good)
        run1 = ws.current_run_dir(self.root)

        partial = _fake_forecast([t0, t1], [0, 6, 12])
        for var in partial.data_vars:
            partial[var].to_numpy()[1, -1] = np.nan
        status = self._sync(['ecmwf_aifs'], lambda _cat, _id: partial)
        assert status['last_result'] == 'error'
        assert 'not published' in status['errors']['ecmwf_aifs']
        assert ws.current_run_dir(self.root) == run1
        assert self._partial_dirs() == []

    def test_ensemble_uses_control_member(self) -> None:
        """Verify ensemble datasets extract control member 0."""
        t0 = np.datetime64('2026-09-29T00:00')
        ds = _fake_forecast([t0], [0, 3, 6], members=True)
        ds['temperature_2m'].to_numpy()[:, :, 1:] = 99.0
        status = self._sync(['ecmwf_ifs'], lambda _c, _i: ds)
        assert status['last_result'] == 'updated'
        run_dir = ws.current_run_dir(self.root)
        assert run_dir is not None
        temp = np.fromfile(Path(run_dir, 'ecmwf_ifs_temp.bin'), np.float16)
        assert temp.shape == (3 * N_LAT * N_LON,)
        assert np.all(temp.astype(float) == _EXPECTED_TEMP_C)

    def test_provider_failure_is_recorded_and_next_sync_recovers(self) -> None:
        """Verify provider errors are recorded and subsequent syncs recover."""
        t0 = np.datetime64('2026-09-29T00:00')
        ds = _fake_forecast([t0], [0, 6])
        attempts: list[str] = []

        def flaky_open(_cat: object, dataset_id: str) -> xr.Dataset:
            attempts.append(dataset_id)
            if len(attempts) == 1:
                msg = 'catalog unreachable'
                raise RuntimeError(msg)
            return ds

        status = self._sync(['ecmwf_aifs'], flaky_open)
        assert status['last_result'] == 'error'
        assert status['errors'] == {
            'ecmwf_aifs': 'RuntimeError: catalog unreachable'
        }
        assert status['updated_models'] == []
        assert ws.current_run_dir(self.root) is None
        assert self._partial_dirs() == []

        status = self._sync(['ecmwf_aifs'], flaky_open)
        assert status['last_result'] == 'updated'
        assert status['errors'] == {}
        assert ws.current_run_dir(self.root) is not None

    def test_unknown_model_raises(self) -> None:
        """Verify unknown model keys raise ValueError."""
        with pytest.raises(ValueError, match='Unsupported weather model'):
            self._sync(['graphcast'], lambda _c, _i: None)

    def test_sync_reports_busy_while_lock_is_held(self) -> None:
        """Verify concurrent sync returns busy when lock is held."""
        runs_dir = Path(self.root) / 'runs'
        runs_dir.mkdir(parents=True, exist_ok=True)
        lock_path = Path(self.root) / 'sync.lock'
        with lock_path.open('a+', encoding='utf-8') as holder:
            assert ws._try_acquire_lock(holder.fileno())  # noqa: SLF001
            calls: list[str] = []
            status = self._sync(['ecmwf_aifs'], lambda _c, i: calls.append(i))
        assert status['last_result'] == 'busy'
        assert calls == []
        assert ws.current_run_dir(self.root) is None

    def test_per_model_isolation_within_same_provider(self) -> None:
        """Verify one failing dynamical model does not abort sibling models."""
        t0 = np.datetime64('2026-09-29T00:00')
        aifs_ds = _fake_forecast([t0], [0, 6])

        def open_ds(_cat: object, dataset_id: str) -> xr.Dataset:
            if dataset_id == 'noaa-gfs-forecast':
                msg = 'gfs upstream corrupted'
                raise RuntimeError(msg)
            return aifs_ds

        status = self._sync(['noaa_gfs', 'ecmwf_aifs'], open_ds)
        assert status['last_result'] == 'partial'
        assert status['updated_models'] == ['ecmwf_aifs']
        assert status['errors'] == {
            'noaa_gfs': 'RuntimeError: gfs upstream corrupted'
        }
        run_dir = ws.current_run_dir(self.root)
        assert run_dir is not None
        assert Path(run_dir, 'ecmwf_aifs_precip.bin').exists()

    def test_open_dynamical_catalog_opened_once_and_cached(self) -> None:
        """Verify STAC catalog is opened lazily once across dynamical groups."""
        fake_cat = object()

        with (
            mock.patch.object(
                ws, 'open_dynamical_catalog', return_value=fake_cat
            ) as mock_open_cat,
            mock.patch.object(
                ws,
                'sync_all_models',
                return_value={
                    'last_result': 'updated',
                    'updated_models': ['noaa_gfs'],
                },
            ) as mock_sync_all,
        ):
            status = ws.sync_latest(
                root=self.root,
                models=['ecmwf_hres', 'noaa_gfs', 'ecmwf_aifs'],
                log=lambda _m: None,
            )
            assert status['last_result'] == 'updated'
            assert mock_open_cat.call_count == 1
            seen_catalogs = [
                c.kwargs['catalog'] for c in mock_sync_all.call_args_list
            ]
            assert seen_catalogs == [None, fake_cat, fake_cat]

        with (
            mock.patch.object(
                ws,
                'open_dynamical_catalog',
                side_effect=RuntimeError('stac offline'),
            ) as mock_fail_cat,
            mock.patch.object(ws, 'sync_all_models'),
        ):
            status_err = ws.sync_latest(
                root=self.root,
                models=['noaa_gfs', 'ecmwf_aifs'],
                log=lambda _m: None,
            )
            assert mock_fail_cat.call_count == 1
            assert status_err['last_result'] == 'error'
            assert status_err['errors'] == {
                'noaa_gfs': 'RuntimeError: stac offline',
                'ecmwf_aifs': 'RuntimeError: stac offline',
            }


@pytest.mark.unit
class WeatherEngineReloadAndFramesTest(unittest.TestCase):
    """Tests hot reload of a new run and world animation frames."""

    def setUp(self) -> None:
        """Set up temporary run directory and isolate WeatherEngine."""
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix='ohn_weather_engine_'
        )
        self.root = Path(self.temp_dir.name)
        self._saved = (
            list(we.CANDIDATE_DATA_DIRS),
            os.environ.get('OPENHYDRONET_WEATHER_WARM_CACHE'),
        )
        os.environ['OPENHYDRONET_WEATHER_WARM_CACHE'] = '0'
        we.CANDIDATE_DATA_DIRS[:] = [self.root / 'current']
        we._INITIALIZED = False  # noqa: SLF001
        we.STREAM_INFO = {}
        for thread in threading.enumerate():
            if thread.name == 'weather-cache-warmup':
                thread.join(timeout=60)
        we._INITIALIZED = False  # noqa: SLF001

    def tearDown(self) -> None:
        """Restore WeatherEngine state and remove temporary directory."""
        we._install_streams(None)  # noqa: SLF001
        we.CANDIDATE_DATA_DIRS[:] = self._saved[0]
        if self._saved[1] is None:
            os.environ.pop('OPENHYDRONET_WEATHER_WARM_CACHE', None)
        else:
            os.environ['OPENHYDRONET_WEATHER_WARM_CACHE'] = self._saved[1]
        we._INITIALIZED = False  # noqa: SLF001
        self.temp_dir.cleanup()
        super().tearDown()

    def _write_run(self, name: str, init_time: str, rain_mmh: float) -> None:
        run = self.root / 'runs' / name
        run.mkdir(parents=True)
        leads = [0, 6, 12, 18, 24]
        plane = (we.N_LAT, we.N_LON)
        precip = np.zeros((len(leads), *plane), np.float16)
        precip[1:, 300:400, 500:700] = rain_mmh
        precip.tofile(run / 'ecmwf_aifs_precip.bin')
        temp = np.full((len(leads), *plane), 15.0, np.float16)
        temp.tofile(run / 'ecmwf_aifs_temp.bin')
        mslp = np.full((len(leads), *plane), 13.0, np.float16)
        mslp.tofile(run / 'ecmwf_aifs_mslp.bin')
        meta = {
            'datasets': {
                'ecmwf_aifs_single_forecast': {
                    'model': 'ecmwf_aifs',
                    'init_time': init_time,
                    'lead_steps': len(leads),
                    'lead_hours': leads,
                    'streams': ['precip', 'temp', 'mslp'],
                    'downloaded_utc': '2026-09-29T20:00:00Z',
                }
            }
        }
        (run / we.RUN_METADATA_FILE).write_text(
            json.dumps(meta), encoding='utf-8'
        )
        swap_current_symlink(self.root, name)

    def test_frames_skip_duplicate_steps_and_reload_picks_up_new_run(
        self,
    ) -> None:
        """Verify frame deduplication and hot reload of new runs."""
        self._write_run('r1', '2026-09-29T00:00:00', 3.0)
        info = we.get_model_data_info('ecmwf_aifs')
        assert info['init_time'] == '2026-09-29T00:00:00Z'
        assert info['max_lead_hours'] == _MAX_AIFS_LEAD_H
        assert 'pressure' in info['real_variables']
        assert info['downloaded_utc'] == '2026-09-29T20:00:00Z'

        index = we.get_frame_index('ecmwf_aifs', 'precipitation')
        assert index['frame_steps'] == [2, 4, 6, 8]
        assert index['step_frames'][:9] == [2, 2, 2, 4, 4, 6, 6, 8, 8]
        assert index['step_frames'][9] is None
        temp_index = we.get_frame_index('ecmwf_aifs', 'temperature')
        assert temp_index['frame_steps'] == [0, 2, 4, 6, 8]

        png = we.generate_weather_frame('ecmwf_aifs', 'precipitation', 3)
        assert png.startswith(b'\x89PNG\r\n\x1a\n')
        width, height, depth, color_type = struct.unpack('!IIBB', png[16:26])
        assert (width, height, depth, color_type) == (
            we.FRAME_SIZE,
            we.FRAME_SIZE,
            8,
            3,
        )
        assert (
            we.generate_weather_frame('ecmwf_aifs', 'precipitation', 4) is png
        )
        pressure_png = we.generate_weather_frame('ecmwf_aifs', 'pressure', 2)
        assert pressure_png.startswith(b'\x89PNG')
        point = we._point_value(  # noqa: SLF001
            'ecmwf_aifs', 'pressure', 6, 0.0, 0.0
        )
        assert point is not None
        assert round(point, 1) == _EXPECTED_PRESSURE_POINT_HPA

        assert not we.reload_if_changed()
        self._write_run('r2', '2026-09-29T12:00:00', 8.0)
        assert we.reload_if_changed()
        assert (
            we.get_model_data_info('ecmwf_aifs')['init_time']
            == '2026-09-29T12:00:00Z'
        )
        assert (
            we.generate_weather_frame('ecmwf_aifs', 'precipitation', 4)
            is not png
        )

    def test_frame_index_rejects_unknown_layers(self) -> None:
        """Verify unknown layers or models raise ValueError."""
        with pytest.raises(ValueError, match='no animation frames'):
            we.get_frame_index('ecmwf_aifs', 'radar')
        with pytest.raises(ValueError, match='Unknown weather model'):
            we.get_frame_index('nope', 'precipitation')


@pytest.mark.unit
class ManualRefreshSubprocessTest(unittest.TestCase):
    """Tests subprocess CLI invocation and sync status defaults."""

    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_args(self, mock_popen: mock.MagicMock) -> None:
        """Verify CLI arguments and cwd passed to the sync subprocess."""
        with tempfile.TemporaryDirectory() as tmp_root:
            mock_proc = mock.MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.communicate.return_value = ('ok', '')
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc
            ws.run_sync_subprocess(
                python='fake_python',
                models=['ecmwf_hres', 'noaa_gfs'],
                force=True,
                root=tmp_root,
            )
            assert mock_popen.called
            cmd = mock_popen.call_args[0][0]
            assert cmd[:3] == ['fake_python', '-m', 'frontend.weather_sync']
            assert '--data-root' in cmd
            assert str(Path(tmp_root).resolve()) in cmd
            assert '--models' in cmd
            assert 'ecmwf_hres,noaa_gfs' in cmd
            assert '--force' in cmd
            expected_repo_root = str(Path(ws.__file__).resolve().parent.parent)
            assert mock_popen.call_args.kwargs['cwd'] == expected_repo_root

    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_records_error_on_crash(
        self, mock_popen: mock.MagicMock
    ) -> None:
        """Verify non-zero subprocess exit without fresh status writes error."""
        with tempfile.TemporaryDirectory() as tmp_root:

            def side_effect1(*_a, **kw):  # noqa: ANN002, ANN003, ANN202
                if 'stdout' in kw and hasattr(kw['stdout'], 'write'):
                    kw['stdout'].write('[ecmwf_hres] starting\n')
                if 'stderr' in kw and hasattr(kw['stderr'], 'write'):
                    kw['stderr'].write('Traceback: foo\n')
                m = mock.MagicMock()
                m.poll.return_value = 1
                m.returncode = 1
                return m

            mock_popen.side_effect = side_effect1
            status = ws.run_sync_subprocess(python='fake_python', root=tmp_root)
            assert status['last_result'] == 'error'
            assert status['message'] == 'Traceback: foo'
            persisted = ws.read_sync_status(tmp_root)
            assert persisted['last_result'] == 'error'
            assert persisted['message'] == status['message']

            # Empty stderr falls back to exit code
            def side_effect2(*_a, **_kw):  # noqa: ANN002, ANN003, ANN202
                m = mock.MagicMock()
                m.poll.return_value = 2
                m.returncode = 2
                return m

            mock_popen.side_effect = side_effect2
            Path(tmp_root, 'sync_status.json').unlink()
            status2 = ws.run_sync_subprocess(python='fake', root=tmp_root)
            assert status2['last_result'] == 'error'
            assert status2['message'] == 'Sync subprocess exited with code 2'

    @mock.patch('subprocess.Popen')
    def test_run_sync_subprocess_preserves_fresh_child_status(
        self, mock_popen: mock.MagicMock
    ) -> None:
        """Verify child-written partial status is preserved on exit code 1."""
        with tempfile.TemporaryDirectory() as tmp_root:

            def side_effect(*_a, **kw):  # noqa: ANN002, ANN003, ANN202
                time.sleep(0.01)
                ws._write_json_atomic(  # noqa: SLF001
                    str(Path(tmp_root) / 'sync_status.json'),
                    {
                        'last_check_utc': ws._utc_now_str(),  # noqa: SLF001
                        'last_result': 'error',
                        'message': 'Failed midway',
                        'models': {'ecmwf_hres': {'status': 'updated'}},
                    },
                )
                if 'stderr' in kw and hasattr(kw['stderr'], 'write'):
                    kw['stderr'].write('Traceback: ...')
                m = mock.MagicMock()
                m.poll.return_value = 1
                m.returncode = 1
                return m

            mock_popen.side_effect = side_effect
            status = ws.run_sync_subprocess(python='fake_python', root=tmp_root)
            assert status['last_result'] == 'error'
            assert status['message'] == 'Failed midway'
            assert 'ecmwf_hres' in status.get('models', {})
