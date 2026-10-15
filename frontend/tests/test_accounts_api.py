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
"""HTTP tests for the guest session and explicit account creation endpoints."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import pytest

from frontend import profile_manager
from frontend.server import OpenHydroNetHandler, ThreadingHTTPServer


@pytest.mark.unit
class AccountsApiTest(unittest.TestCase):
    """Verify `/api/profile/*` and `/api/account/summary` HTTP endpoints."""

    server: ThreadingHTTPServer
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        """Start the multi-threaded HTTP server on an ephemeral local port."""
        super().setUpClass()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), OpenHydroNetHandler)
        cls.port = cls.server.server_port
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down the test HTTP server."""
        cls.server.shutdown()
        cls.server.server_close()
        super().tearDownClass()

    def setUp(self) -> None:
        """Isolate each test with a fresh temporary `ProfileManager` root."""
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.profiles_root = Path(temp_dir.name) / 'profiles'
        saved_pm = profile_manager._GLOBAL_PROFILE_MANAGER  # noqa: SLF001
        profile_manager._GLOBAL_PROFILE_MANAGER = (  # noqa: SLF001
            profile_manager.ProfileManager(profiles_dir=self.profiles_root)
        )
        self.addCleanup(
            setattr,
            profile_manager,
            '_GLOBAL_PROFILE_MANAGER',
            saved_pm,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        body = (
            json.dumps(payload).encode('utf-8') if payload is not None else None
        )
        req_headers = {'Content-Type': 'application/json', **(headers or {})}
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        conn.request(method, path, body=body, headers=req_headers)
        resp = conn.getresponse()
        status = resp.status
        data = json.loads(resp.read().decode('utf-8'))
        conn.close()
        return status, data

    def _usernames(self) -> list[str]:
        _, data = self._request('GET', '/api/profile/list')
        return sorted(p['username'] for p in data['profiles'])

    def test_account_lifecycle(self) -> None:
        """Verify strict login, explicit account creation, and logout."""
        assert self._usernames() == ['guest']

        status, data = self._request(
            'POST',
            '/api/profile/login',
            {'username': 'alice', 'create_if_missing': False},
        )
        assert status == 404  # noqa: PLR2004
        assert 'Create Account' in data['error']
        assert self._usernames() == ['guest']

        status, data = self._request(
            'POST',
            '/api/profile/create',
            {
                'username': 'Alice',
                'display_name': 'Alice A.',
                'email': 'alice@example.com',
            },
        )
        assert status == 200  # noqa: PLR2004
        assert data['profile']['username'] == 'alice'
        assert data['profile']['display_name'] == 'Alice A.'
        assert not data['profile']['is_guest']
        assert data['count'] == 0
        assert self._usernames() == ['alice', 'guest']
        _, current = self._request('GET', '/api/profile/current')
        assert current['active_username'] == 'alice'

        status, data = self._request(
            'POST',
            '/api/profile/create',
            {'username': 'alice'},
        )
        assert status == 409  # noqa: PLR2004
        assert 'already exists' in data['error']
        for bad in ('', 'guest', '..'):
            status, _ = self._request(
                'POST',
                '/api/profile/create',
                {'username': bad},
            )
            assert status == 400, bad  # noqa: PLR2004

        status, data = self._request('POST', '/api/profile/logout', {})
        assert status == 200  # noqa: PLR2004
        assert data['profile']['username'] == 'guest'
        status, data = self._request(
            'POST',
            '/api/profile/login',
            {'username': 'alice', 'create_if_missing': False},
        )
        assert status == 200  # noqa: PLR2004
        assert data['profile']['username'] == 'alice'
        self._request('POST', '/api/profile/logout', {})

    def test_login_without_flag_keeps_legacy_create_behavior(self) -> None:
        """Verify callers omitting `create_if_missing` create the account."""
        status, data = self._request(
            'POST',
            '/api/profile/login',
            {'username': 'carol'},
        )
        assert status == 200  # noqa: PLR2004
        assert data['profile']['username'] == 'carol'
        assert data['profile']['email'] == ''
        assert self._usernames() == ['carol', 'guest']
        self._request('POST', '/api/profile/logout', {})

    def test_guest_keeps_data_until_restart_reset(self) -> None:
        """Verify guest data persists across account switches until reset."""
        guest = {'X-User-Profile': 'guest'}
        feature = {
            'type': 'Feature',
            'properties': {'catchment_id': 'guest_basin_1', 'area_km2': 12.0},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]
                ],
            },
        }
        status, _ = self._request(
            'POST',
            '/api/polygons/upload',
            feature,
            headers=guest,
        )
        assert status == 200  # noqa: PLR2004
        _, summary = self._request('GET', '/api/account/summary', headers=guest)
        assert summary['totals']['watersheds_count'] == 1
        _, archives = self._request('GET', '/api/archives', headers=guest)
        assert any(a['name'] == 'watersheds.json' for a in archives['archives'])

        status, _ = self._request(
            'POST',
            '/api/profile/create',
            {'username': 'bob'},
        )
        assert status == 200  # noqa: PLR2004
        self._request('POST', '/api/profile/logout', {})
        _, summary = self._request('GET', '/api/account/summary', headers=guest)
        assert summary['totals']['watersheds_count'] == 1

        profile_manager.get_profile_manager().clear_guest_data()
        _, summary = self._request('GET', '/api/account/summary', headers=guest)
        assert summary['totals']['watersheds_count'] == 0
        _, archives = self._request('GET', '/api/archives', headers=guest)
        assert archives['archives'] == []
        assert 'bob' in self._usernames()

    def test_delete_account(self) -> None:
        """Verify `/api/profile/delete` removes accounts and switches active."""
        self._request('POST', '/api/profile/create', {'username': 'alice'})
        self._request('POST', '/api/profile/create', {'username': 'bob'})

        status, data = self._request(
            'POST',
            '/api/profile/delete',
            {'username': 'Alice'},
        )
        assert status == 200  # noqa: PLR2004
        assert data['removed'] == 'alice'
        assert data['profile']['username'] == 'bob'
        assert self._usernames() == ['bob', 'guest']

        status, data = self._request(
            'POST',
            '/api/profile/delete',
            {'username': 'bob'},
        )
        assert status == 200  # noqa: PLR2004
        assert data['profile']['username'] == 'guest'
        assert data['profile']['is_guest']
        _, current = self._request('GET', '/api/profile/current')
        assert current['active_username'] == 'guest'
        assert self._usernames() == ['guest']

        status, data = self._request(
            'POST',
            '/api/profile/delete',
            {'username': 'bob'},
        )
        assert status == 404  # noqa: PLR2004
        assert data['code'] == 'account_not_found'
        for bad in ('', 'guest'):
            status, data = self._request(
                'POST',
                '/api/profile/delete',
                {'username': bad},
            )
            assert status == 400, bad  # noqa: PLR2004
        status, data = self._request(
            'POST',
            '/api/profile/no_such_endpoint',
            {},
        )
        assert status == 404  # noqa: PLR2004
        assert 'code' not in data

    def _profile_settings_on_disk(self, username: str) -> dict[str, Any]:
        """Read ``settings`` directly from the user's ``profile.json``."""
        meta_file = self.profiles_root / username / 'profile.json'
        return json.loads(meta_file.read_text(encoding='utf-8'))['settings']

    def _basin_list_on_disk(self, username: str) -> list[str]:
        """Read ``catchments/basin_lists/all_basins.txt`` for ``username``."""
        list_file = (
            self.profiles_root
            / username
            / 'catchments'
            / 'basin_lists'
            / 'all_basins.txt'
        )
        return list_file.read_text(encoding='utf-8').split()

    def test_profile_settings_persist_only_to_requested_profile(self) -> None:
        """Verify `/api/profile/settings` writes the requested user's file.

        Regression for Issue #119: the handler used to call
        ``save_settings`` without the request-scoped username, so it rewrote
        whichever profile the process-global singleton happened to have active.
        """
        self._request('POST', '/api/profile/create', {'username': 'alice'})
        self._request('POST', '/api/profile/create', {'username': 'bob'})
        _, current = self._request('GET', '/api/profile/current')
        assert current['active_username'] == 'bob'

        status, data = self._request(
            'POST',
            '/api/profile/settings',
            {
                'username': 'alice',
                'settings': {'default_hydro_dataset': 'merit-hydro'},
            },
        )
        assert status == 200  # noqa: PLR2004
        assert (
            self._profile_settings_on_disk('alice')['default_hydro_dataset']
            == 'merit-hydro'
        ), 'settings must be written to the requested profile, not the active'
        assert (
            self._profile_settings_on_disk('bob')['default_hydro_dataset']
            == 'hydroatlas'
        ), 'the active profile must be left untouched'
        assert data['status'] == 'saved'
        assert data['username'] == 'alice'
        assert data['profile']['settings']['default_hydro_dataset'] == (
            'merit-hydro'
        )
        _, current = self._request('GET', '/api/profile/current')
        assert current['active_username'] == 'bob'

        status, data = self._request(
            'POST',
            '/api/profile/settings',
            {'settings': {'default_weather_source': 'imerg'}},
            headers={'X-User-Profile': 'alice'},
        )
        assert status == 200  # noqa: PLR2004
        alice_settings = self._profile_settings_on_disk('alice')
        assert alice_settings['default_weather_source'] == 'imerg'
        assert alice_settings['default_hydro_dataset'] == 'merit-hydro'
        assert (
            self._profile_settings_on_disk('bob')['default_weather_source']
            == 'cpc'
        )

        status, data = self._request(
            'POST',
            '/api/profile/settings',
            {'username': 'alice', 'settings': 'merit-hydro'},
        )
        assert status == 400  # noqa: PLR2004
        status, data = self._request(
            'POST',
            '/api/profile/settings',
            {'username': 'nobody', 'settings': {'default_hydro_dataset': 'x'}},
        )
        assert status == 404  # noqa: PLR2004
        assert data['code'] == 'account_not_found'
        assert not (self.profiles_root / 'nobody').exists()
        assert self._usernames() == ['alice', 'bob', 'guest']

    def test_save_watersheds_persists_payload_and_resyncs_artifacts(
        self,
    ) -> None:
        """Verify `/api/profile/save_watersheds` really writes to disk.

        Regression for Issue #119: the handler used to load the stored
        watersheds and answer ``{"status": "saved"}`` without persisting
        anything.
        """

        def _feature(cid: str) -> dict[str, Any]:
            return {
                'type': 'Feature',
                'id': cid,
                'properties': {'catchment_id': cid, 'area_km2': 1.0},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]
                    ],
                },
            }

        carol = {'X-User-Profile': 'carol'}
        self._request('POST', '/api/profile/create', {'username': 'carol'})
        self._request('POST', '/api/profile/logout', {})
        carol_dir = self.profiles_root / 'carol'
        ws_json = carol_dir / 'polygons' / 'watersheds.json'
        shapes_dir = carol_dir / 'catchments' / 'shapes'
        all_basins = carol_dir / 'catchments' / 'basin_lists' / 'all_basins.txt'

        status, data = self._request(
            'POST',
            '/api/profile/save_watersheds',
            {'watersheds': [_feature('ws_a'), _feature('ws_b')]},
            headers=carol,
        )
        assert status == 200  # noqa: PLR2004
        assert data['status'] == 'saved'
        assert data['username'] == 'carol'
        assert data['count'] == 2  # noqa: PLR2004
        assert data['catchment_ids'] == ['ws_a', 'ws_b']
        stored = json.loads(ws_json.read_text(encoding='utf-8'))
        assert [f['id'] for f in stored['features']] == ['ws_a', 'ws_b']
        assert (shapes_dir / 'ws_a.geojson').is_file()
        assert (shapes_dir / 'ws_b.geojson').is_file()
        assert self._basin_list_on_disk('carol') == ['ws_a', 'ws_b']
        _, listed = self._request('GET', '/api/watersheds', headers=carol)
        assert listed['count'] == 2  # noqa: PLR2004
        _, guest_listed = self._request(
            'GET',
            '/api/watersheds',
            headers={'X-User-Profile': 'guest'},
        )
        assert guest_listed['count'] == 0

        (shapes_dir / 'ws_a.geojson').unlink()
        all_basins.unlink()
        status, data = self._request(
            'POST',
            '/api/profile/save_watersheds',
            {'username': 'carol'},
            headers=carol,
        )
        assert status == 200  # noqa: PLR2004
        assert data['count'] == 2  # noqa: PLR2004
        assert (shapes_dir / 'ws_a.geojson').is_file()
        assert self._basin_list_on_disk('carol') == ['ws_a', 'ws_b']

        status, data = self._request(
            'POST',
            '/api/profile/save_watersheds',
            {
                'username': 'carol',
                'watersheds': {
                    'type': 'FeatureCollection',
                    'features': [_feature('ws_b')],
                },
            },
        )
        assert status == 200  # noqa: PLR2004
        assert data['catchment_ids'] == ['ws_b']
        assert not (shapes_dir / 'ws_a.geojson').exists()
        assert self._basin_list_on_disk('carol') == ['ws_b']

        for bad_payload in (
            'ws_b',
            [{'type': 'Point', 'coordinates': [0, 0]}],
            [{'type': 'Feature', 'properties': {}}],
            {'type': 'Feature', 'geometry': None, 'properties': {}},
        ):
            status, data = self._request(
                'POST',
                '/api/profile/save_watersheds',
                {'username': 'carol', 'watersheds': bad_payload},
            )
            assert status == 400, bad_payload  # noqa: PLR2004
            assert data['status'] == 'error'
        _, listed = self._request('GET', '/api/watersheds', headers=carol)
        assert [f['id'] for f in listed['features']] == ['ws_b']


if __name__ == '__main__':
    unittest.main()
