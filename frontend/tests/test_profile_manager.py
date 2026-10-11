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
"""Unit tests for `ProfileManager`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

from frontend.profile_manager import ProfileManager


@pytest.mark.unit
class ProfileManagerTest(unittest.TestCase):
    """Verify profile creation, isolation, guest reset, and watersheds."""

    def setUp(self) -> None:
        """Create a temporary profiles directory and `ProfileManager`."""
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.profiles_root = Path(self.temp_dir.name) / 'profiles'
        self.pm = ProfileManager(profiles_dir=self.profiles_root)

    def tearDown(self) -> None:
        """Clean up the temporary profiles directory."""
        self.temp_dir.cleanup()
        super().tearDown()

    def test_guest_default_state(self) -> None:
        """Verify the default session starts as `guest` with no watersheds."""
        assert self.pm.active_username == 'guest'
        info = self.pm.get_profile_info()
        assert info['username'] == 'guest'
        assert info['is_guest']
        assert self.pm.load_watersheds() == []

    def test_login_and_persistence(self) -> None:
        """Verify profile login, default empty email, and watershed saving."""
        info = self.pm.login_profile(
            username='test_user',
            email='test_user@example.com',
            display_name='Test Hydrologist',
        )
        assert self.pm.active_username == 'test_user'
        assert info['username'] == 'test_user'
        assert info['email'] == 'test_user@example.com'
        assert info['display_name'] == 'Test Hydrologist'
        assert not info['is_guest']

        sample_features = [
            {
                'type': 'Feature',
                'id': 'us_03338780',
                'properties': {
                    'catchment_id': 'us_03338780',
                    'name': 'Wabash River at Lafayette',
                    'area_sqkm': 18780.0,
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            [-87.0, 40.0],
                            [-86.5, 40.0],
                            [-86.5, 40.5],
                            [-87.0, 40.0],
                        ]
                    ],
                },
            }
        ]
        assert self.pm.save_watersheds(sample_features)

        loaded = self.pm.load_watersheds('test_user')
        assert len(loaded) == 1
        assert loaded[0]['properties']['catchment_id'] == 'us_03338780'

        profiles = self.pm.list_profiles()
        assert any(p['username'] == 'test_user' for p in profiles)
        gs_prof = next(p for p in profiles if p['username'] == 'test_user')
        assert gs_prof['watersheds_count'] == 1

    def test_logout_and_switch_profile(self) -> None:
        """Verify per-user watershed isolation and logout to `guest`."""
        self.pm.login_profile(username='user1', email='user1@example.com')
        self.pm.save_watersheds([{'type': 'Feature', 'id': 'feat1'}])

        self.pm.login_profile(username='user2', email='user2@example.com')
        self.pm.save_watersheds(
            [
                {'type': 'Feature', 'id': 'feat2_a'},
                {'type': 'Feature', 'id': 'feat2_b'},
            ]
        )

        assert len(self.pm.load_watersheds('user1')) == 1
        assert len(self.pm.load_watersheds('user2')) == 2  # noqa: PLR2004

        guest_info = self.pm.logout_profile()
        assert guest_info['username'] == 'guest'
        assert self.pm.active_username == 'guest'
        assert self.pm.load_watersheds('guest') == []

    def test_organized_directories(self) -> None:
        """Verify standard profile subdirectories are provisioned on login."""
        info = self.pm.login_profile(username='test_user')
        assert info['email'] == ''
        poly_dir = self.pm.get_polygons_dir('test_user')
        attr_dir = self.pm.get_attributes_dir('test_user')
        hist_dir = self.pm.get_historical_dir('test_user')
        fc_dir = self.pm.get_forecast_dir('test_user')

        assert poly_dir.exists()
        assert attr_dir.exists()
        assert hist_dir.exists()
        assert fc_dir.exists()
        assert poly_dir.name == 'polygons'
        assert attr_dir.name == 'attributes'
        assert fc_dir.name == 'forecast'

    def test_guest_data_kept_during_session_and_cleared_on_restart(
        self,
    ) -> None:
        """Verify guest data persists in-session and is wiped on restart."""
        guest_poly = self.pm.get_polygons_dir('guest')
        dummy_file = guest_poly / 'temp.geojson'
        dummy_file.write_text('dummy', encoding='utf-8')
        cache_file = (
            self.pm.get_profile_dir('guest') / 'some_cache' / 'cached.json'
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('{}', encoding='utf-8')
        self.pm.save_watersheds(
            [{'type': 'Feature', 'id': 'guest_feat'}],
            username='guest',
        )
        assert len(self.pm.load_watersheds('guest')) == 1

        self.pm.login_profile(username='test_user')
        assert len(self.pm.load_watersheds('guest')) == 1
        self.pm.logout_profile()
        assert self.pm.active_username == 'guest'
        assert len(self.pm.load_watersheds('guest')) == 1
        assert dummy_file.exists()

        other_pm = ProfileManager(profiles_dir=self.profiles_root)
        assert len(other_pm.load_watersheds('guest')) == 1

        new_pm = ProfileManager(profiles_dir=self.profiles_root)
        new_pm.clear_guest_data()
        assert new_pm.load_watersheds('guest') == []
        assert not dummy_file.exists()
        assert not cache_file.exists()
        assert new_pm.get_polygons_dir('guest').exists()
        assert new_pm.account_exists('test_user')

    def test_create_account(self) -> None:
        """Verify explicit account creation and duplicate/invalid rejection."""
        info = self.pm.create_account(
            'New_User',
            email='new@example.com',
            display_name='New User',
        )
        assert info['username'] == 'new_user'
        assert info['display_name'] == 'New User'
        assert info['email'] == 'new@example.com'
        assert not info['is_guest']
        assert self.pm.active_username == 'new_user'
        assert self.pm.account_exists('new_user')
        assert self.pm.load_watersheds('new_user') == []

        with pytest.raises(FileExistsError):
            self.pm.create_account('new_user')
        for bad_name in ('', '   ', 'guest', 'GUEST', '..', '%%%'):
            with pytest.raises(ValueError, match=r'.+'):
                self.pm.create_account(bad_name)

    def test_create_account_preserves_preexisting_folder(self) -> None:
        """Verify `create_account` never deletes a pre-existing user folder."""
        leftover = self.pm.get_polygons_dir('reused_name') / 'old.geojson'
        leftover.write_text('old', encoding='utf-8')
        assert not self.pm.account_exists('reused_name')
        self.pm.create_account('reused_name')
        assert leftover.exists()

    def test_strict_login_requires_existing_account(self) -> None:
        """Verify `create_if_missing=False` raises `LookupError` if absent."""
        with pytest.raises(LookupError):
            self.pm.login_profile('nobody', create_if_missing=False)
        assert self.pm.active_username == 'guest'
        assert not (self.profiles_root / 'nobody').exists()

        self.pm.create_account('somebody')
        self.pm.logout_profile()
        info = self.pm.login_profile('somebody', create_if_missing=False)
        assert info['username'] == 'somebody'
        self.pm.login_profile('guest', create_if_missing=False)
        assert self.pm.active_username == 'guest'

    def test_list_profiles_only_lists_guest_and_accounts(self) -> None:
        """Verify scratch folders without `profile.json` are excluded."""
        assert [p['username'] for p in self.pm.list_profiles()] == ['guest']
        self.pm.get_profile_dir('scratch_folder')
        self.pm.create_account('real_user')
        names = sorted(p['username'] for p in self.pm.list_profiles())
        assert names == ['guest', 'real_user']

    def test_username_cannot_escape_profiles_folder(self) -> None:
        """Verify path-traversal characters in usernames stay under root."""
        for name in ('..', '.', '../outside', '.hidden'):
            p_dir = self.pm.get_profile_dir(name)
            assert p_dir.parent.resolve() == self.profiles_root.resolve()

    def test_delete_account(self) -> None:
        """Verify `delete_account` removes the account folder and resets."""
        self.pm.create_account('keeper')
        self.pm.create_account('doomed')
        self.pm.save_watersheds(
            [{'type': 'Feature', 'id': 'f1'}],
            username='doomed',
        )
        model_file = (
            self.pm.get_model_runs_dir('doomed') / 'run1' / 'weights.pt'
        )
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text('weights', encoding='utf-8')

        assert self.pm.delete_account('Doomed') == 'doomed'
        assert self.pm.active_username == 'guest'
        assert not (self.profiles_root / 'doomed').exists()
        assert not self.pm.account_exists('doomed')
        names = sorted(p['username'] for p in self.pm.list_profiles())
        assert names == ['guest', 'keeper']

        self.pm.create_account('other')
        self.pm.login_profile('keeper', create_if_missing=False)
        self.pm.delete_account('other')
        assert self.pm.active_username == 'keeper'
        assert self.pm.account_exists('keeper')

        self.pm.create_account('doomed')
        assert self.pm.load_watersheds('doomed') == []
        assert not model_file.exists()

    def test_delete_account_rejects_guest_and_unknown_names(self) -> None:
        """Verify `delete_account` rejects `guest` and non-existent accounts."""
        self.pm.save_watersheds(
            [{'type': 'Feature', 'id': 'g1'}],
            username='guest',
        )
        for bad_name in ('', '   ', 'guest', 'GUEST', '..'):
            with pytest.raises(ValueError, match=r'.+'):
                self.pm.delete_account(bad_name)
        with pytest.raises(LookupError):
            self.pm.delete_account('nobody')
        self.pm.get_profile_dir('scratch_folder')
        with pytest.raises(LookupError):
            self.pm.delete_account('scratch_folder')
        assert len(self.pm.load_watersheds('guest')) == 1
        assert self.pm.active_username == 'guest'


if __name__ == '__main__':
    unittest.main()
