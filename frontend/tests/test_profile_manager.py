"""Unit tests for User Profile Manager."""

import json
from pathlib import Path
import tempfile

try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest

from frontend.profile_manager import ProfileManager


class ProfileManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.profiles_root = Path(self.temp_dir.name) / "profiles"
    self.pm = ProfileManager(profiles_dir=self.profiles_root)

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def test_guest_default_state(self):
    self.assertEqual(self.pm.active_username, "guest")
    info = self.pm.get_profile_info()
    self.assertEqual(info["username"], "guest")
    self.assertTrue(info["is_guest"])
    self.assertEqual(self.pm.load_watersheds(), [])

  def test_login_and_persistence(self):
    # Login as gsnearing
    info = self.pm.login_profile(
        username="gsnearing",
        email="gsnearing@google.com",
        display_name="Grey Nearing",
    )
    self.assertEqual(self.pm.active_username, "gsnearing")
    self.assertEqual(info["username"], "gsnearing")
    self.assertEqual(info["email"], "gsnearing@google.com")
    self.assertEqual(info["display_name"], "Grey Nearing")
    self.assertFalse(info["is_guest"])

    # Save watersheds
    sample_features = [
        {
            "type": "Feature",
            "id": "us_03338780",
            "properties": {
                "catchment_id": "us_03338780",
                "name": "Wabash River at Lafayette",
                "area_sqkm": 18780.0,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-87.0, 40.0], [-86.5, 40.0], [-86.5, 40.5], [-87.0, 40.0]]],
            },
        }
    ]
    self.assertTrue(self.pm.save_watersheds(sample_features))

    # Verify loaded watersheds
    loaded = self.pm.load_watersheds("gsnearing")
    self.assertEqual(len(loaded), 1)
    self.assertEqual(loaded[0]["properties"]["catchment_id"], "us_03338780")

    # Check profile listing
    profiles = self.pm.list_profiles()
    self.assertTrue(any(p["username"] == "gsnearing" for p in profiles))
    gs_prof = next(p for p in profiles if p["username"] == "gsnearing")
    self.assertEqual(gs_prof["watersheds_count"], 1)

  def test_logout_and_switch_profile(self):
    self.pm.login_profile(username="user1", email="user1@example.com")
    self.pm.save_watersheds([{"type": "Feature", "id": "feat1"}])

    self.pm.login_profile(username="user2", email="user2@example.com")
    self.pm.save_watersheds([
        {"type": "Feature", "id": "feat2_a"},
        {"type": "Feature", "id": "feat2_b"},
    ])

    # Check isolation
    self.assertEqual(len(self.pm.load_watersheds("user1")), 1)
    self.assertEqual(len(self.pm.load_watersheds("user2")), 2)

    # Logout to guest
    guest_info = self.pm.logout_profile()
    self.assertEqual(guest_info["username"], "guest")
    self.assertEqual(self.pm.active_username, "guest")
    self.assertEqual(self.pm.load_watersheds("guest"), [])

  def test_organized_directories(self):
    self.pm.login_profile(username="gsnearing")
    poly_dir = self.pm.get_polygons_dir("gsnearing")
    attr_dir = self.pm.get_attributes_dir("gsnearing")
    hist_dir = self.pm.get_historical_dir("gsnearing")
    fc_dir = self.pm.get_forecast_dir("gsnearing")

    self.assertTrue(poly_dir.exists())
    self.assertTrue(attr_dir.exists())
    self.assertTrue(hist_dir.exists())
    self.assertTrue(fc_dir.exists())
    self.assertEqual(poly_dir.name, "polygons")
    self.assertEqual(attr_dir.name, "attributes")
    self.assertEqual(fc_dir.name, "forecast")

  def test_guest_data_kept_during_session_and_cleared_on_restart(self):
    # Simulate guest adding data (including a cache-like file in an unknown folder)
    guest_poly = self.pm.get_polygons_dir("guest")
    dummy_file = guest_poly / "temp.geojson"
    dummy_file.write_text("dummy", encoding="utf-8")
    cache_file = self.pm.get_profile_dir("guest") / "some_cache" / "cached.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{}", encoding="utf-8")
    self.pm.save_watersheds([{"type": "Feature", "id": "guest_feat"}], username="guest")
    self.assertEqual(len(self.pm.load_watersheds("guest")), 1)

    # Switching to an account and back keeps the guest's data for this session
    self.pm.login_profile(username="gsnearing")
    self.assertEqual(len(self.pm.load_watersheds("guest")), 1)
    self.pm.logout_profile()
    self.assertEqual(self.pm.active_username, "guest")
    self.assertEqual(len(self.pm.load_watersheds("guest")), 1)
    self.assertTrue(dummy_file.exists())

    # Creating a ProfileManager (e.g. a script or test) must not wipe the guest
    other_pm = ProfileManager(profiles_dir=self.profiles_root)
    self.assertEqual(len(other_pm.load_watersheds("guest")), 1)

    # Server startup reset erases everything the guest stored, caches included
    new_pm = ProfileManager(profiles_dir=self.profiles_root)
    new_pm.clear_guest_data()
    self.assertEqual(new_pm.load_watersheds("guest"), [])
    self.assertFalse(dummy_file.exists())
    self.assertFalse(cache_file.exists())
    self.assertTrue(new_pm.get_polygons_dir("guest").exists())

    # Accounts are untouched by the guest reset
    self.assertTrue(new_pm.account_exists("gsnearing"))

  def test_create_account(self):
    info = self.pm.create_account(
        "New_User", email="new@example.com", display_name="New User"
    )
    self.assertEqual(info["username"], "new_user")
    self.assertEqual(info["display_name"], "New User")
    self.assertEqual(info["email"], "new@example.com")
    self.assertFalse(info["is_guest"])
    self.assertEqual(self.pm.active_username, "new_user")
    self.assertTrue(self.pm.account_exists("new_user"))
    self.assertEqual(self.pm.load_watersheds("new_user"), [])

    with self.assertRaises(FileExistsError):
      self.pm.create_account("new_user")
    for bad_name in ("", "   ", "guest", "GUEST", "..", "%%%"):
      with self.assertRaises(ValueError, msg=bad_name):
        self.pm.create_account(bad_name)

  def test_create_account_discards_leftover_folder(self):
    leftover = self.pm.get_polygons_dir("reused_name") / "old.geojson"
    leftover.write_text("old", encoding="utf-8")
    self.assertFalse(self.pm.account_exists("reused_name"))
    self.pm.create_account("reused_name")
    self.assertFalse(leftover.exists())

  def test_strict_login_requires_existing_account(self):
    with self.assertRaises(LookupError):
      self.pm.login_profile("nobody", create_if_missing=False)
    self.assertEqual(self.pm.active_username, "guest")
    self.assertFalse((self.profiles_root / "nobody").exists())

    self.pm.create_account("somebody")
    self.pm.logout_profile()
    info = self.pm.login_profile("somebody", create_if_missing=False)
    self.assertEqual(info["username"], "somebody")
    # The guest session can always be entered
    self.pm.login_profile("guest", create_if_missing=False)
    self.assertEqual(self.pm.active_username, "guest")

  def test_list_profiles_only_lists_guest_and_accounts(self):
    self.assertEqual([p["username"] for p in self.pm.list_profiles()], ["guest"])
    # Folders created by data access without an account are not accounts
    self.pm.get_profile_dir("scratch_folder")
    self.pm.create_account("real_user")
    names = sorted(p["username"] for p in self.pm.list_profiles())
    self.assertEqual(names, ["guest", "real_user"])

  def test_username_cannot_escape_profiles_folder(self):
    for name in ("..", ".", "../outside", ".hidden"):
      p_dir = self.pm.get_profile_dir(name)
      self.assertEqual(p_dir.parent.resolve(), self.profiles_root.resolve())

  def test_delete_account(self):
    self.pm.create_account("keeper")
    self.pm.create_account("doomed")  # Now the active account.
    self.pm.save_watersheds([{"type": "Feature", "id": "f1"}], username="doomed")
    model_file = self.pm.get_model_runs_dir("doomed") / "run1" / "weights.pt"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("weights", encoding="utf-8")

    # Deleting the active account removes all its data and falls back to guest.
    self.assertEqual(self.pm.delete_account("Doomed"), "doomed")
    self.assertEqual(self.pm.active_username, "guest")
    self.assertFalse((self.profiles_root / "doomed").exists())
    self.assertFalse(self.pm.account_exists("doomed"))
    names = sorted(p["username"] for p in self.pm.list_profiles())
    self.assertEqual(names, ["guest", "keeper"])

    # Deleting another account leaves the active account alone.
    self.pm.create_account("other")
    self.pm.login_profile("keeper", create_if_missing=False)
    self.pm.delete_account("other")
    self.assertEqual(self.pm.active_username, "keeper")
    self.assertTrue(self.pm.account_exists("keeper"))

    # The name can be used again, and the new account starts empty.
    self.pm.create_account("doomed")
    self.assertEqual(self.pm.load_watersheds("doomed"), [])
    self.assertFalse(model_file.exists())

  def test_delete_account_rejects_guest_and_unknown_names(self):
    self.pm.save_watersheds([{"type": "Feature", "id": "g1"}], username="guest")
    for bad_name in ("", "   ", "guest", "GUEST", ".."):
      with self.assertRaises(ValueError, msg=bad_name):
        self.pm.delete_account(bad_name)
    with self.assertRaises(LookupError):
      self.pm.delete_account("nobody")
    # A data folder without profile.json is not an account.
    self.pm.get_profile_dir("scratch_folder")
    with self.assertRaises(LookupError):
      self.pm.delete_account("scratch_folder")
    # The guest session and its data are untouched.
    self.assertEqual(len(self.pm.load_watersheds("guest")), 1)
    self.assertEqual(self.pm.active_username, "guest")


if __name__ == "__main__":
  absltest.main()
