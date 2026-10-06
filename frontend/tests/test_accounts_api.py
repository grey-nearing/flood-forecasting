"""HTTP tests for the guest session and explicit account creation endpoints."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from frontend import profile_manager
from frontend.server import (
    EarthkitHydroHandler,
    ThreadingHTTPServer,
)


class AccountsApiTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EarthkitHydroHandler)
    cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
    threading.Thread(target=cls.server.serve_forever, daemon=True).start()

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    # A private, empty profile store per test, so tests never see or touch
    # real accounts (or each other's).
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    saved_pm = profile_manager._GLOBAL_PROFILE_MANAGER  # pylint: disable=protected-access
    profile_manager._GLOBAL_PROFILE_MANAGER = profile_manager.ProfileManager(  # pylint: disable=protected-access
        profiles_dir=Path(temp_dir.name) / "profiles"
    )
    self.addCleanup(
        setattr, profile_manager, "_GLOBAL_PROFILE_MANAGER", saved_pm
    )

  def _request(self, method, path, payload=None, headers=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{self.base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
      with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
      return e.code, json.loads(e.read().decode("utf-8"))

  def _usernames(self):
    _, data = self._request("GET", "/api/profile/list")
    return sorted(p["username"] for p in data["profiles"])

  def test_account_lifecycle(self):
    # Only the guest session exists to begin with.
    self.assertEqual(self._usernames(), ["guest"])

    # A strict login (what the web UI sends) to a missing account fails and
    # creates nothing.
    status, data = self._request(
        "POST",
        "/api/profile/login",
        {"username": "alice", "create_if_missing": False},
    )
    self.assertEqual(status, 404)
    self.assertIn("Create Account", data["error"])
    self.assertEqual(self._usernames(), ["guest"])

    # Create the account explicitly; it becomes the active, empty profile.
    status, data = self._request(
        "POST",
        "/api/profile/create",
        {"username": "Alice", "display_name": "Alice A.", "email": "alice@example.com"},
    )
    self.assertEqual(status, 200)
    self.assertEqual(data["profile"]["username"], "alice")
    self.assertEqual(data["profile"]["display_name"], "Alice A.")
    self.assertFalse(data["profile"]["is_guest"])
    self.assertEqual(data["count"], 0)
    self.assertEqual(self._usernames(), ["alice", "guest"])
    _, current = self._request("GET", "/api/profile/current")
    self.assertEqual(current["active_username"], "alice")

    # Duplicate and invalid names are rejected.
    status, data = self._request("POST", "/api/profile/create", {"username": "alice"})
    self.assertEqual(status, 409)
    self.assertIn("already exists", data["error"])
    for bad in ("", "guest", ".."):
      status, _ = self._request("POST", "/api/profile/create", {"username": bad})
      self.assertEqual(status, 400, bad)

    # Back to guest, then a strict login to the existing account works.
    status, data = self._request("POST", "/api/profile/logout", {})
    self.assertEqual(status, 200)
    self.assertEqual(data["profile"]["username"], "guest")
    status, data = self._request(
        "POST",
        "/api/profile/login",
        {"username": "alice", "create_if_missing": False},
    )
    self.assertEqual(status, 200)
    self.assertEqual(data["profile"]["username"], "alice")
    self._request("POST", "/api/profile/logout", {})

  def test_login_without_flag_keeps_legacy_create_behavior(self):
    # Backward-compatible contract: callers that omit "create_if_missing"
    # still get the account created on first login.
    status, data = self._request("POST", "/api/profile/login", {"username": "carol"})
    self.assertEqual(status, 200)
    self.assertEqual(data["profile"]["username"], "carol")
    self.assertEqual(self._usernames(), ["carol", "guest"])
    self._request("POST", "/api/profile/logout", {})

  def test_guest_keeps_data_until_restart_reset(self):
    guest = {"X-User-Profile": "guest"}
    feature = {
        "type": "Feature",
        "properties": {"catchment_id": "guest_basin_1", "area_km2": 12.0},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]],
        },
    }
    status, _ = self._request("POST", "/api/polygons/upload", feature, headers=guest)
    self.assertEqual(status, 200)
    _, summary = self._request("GET", "/api/account/summary", headers=guest)
    self.assertEqual(summary["totals"]["watersheds_count"], 1)
    _, archives = self._request("GET", "/api/archives", headers=guest)
    self.assertTrue(any(a["name"] == "watersheds.json" for a in archives["archives"]))

    # Switching to an account and back keeps the guest's basins for this session.
    status, _ = self._request("POST", "/api/profile/create", {"username": "bob"})
    self.assertEqual(status, 200)
    self._request("POST", "/api/profile/logout", {})
    _, summary = self._request("GET", "/api/account/summary", headers=guest)
    self.assertEqual(summary["totals"]["watersheds_count"], 1)

    # The startup reset (run_server) empties the guest completely.
    profile_manager.get_profile_manager().clear_guest_data()
    _, summary = self._request("GET", "/api/account/summary", headers=guest)
    self.assertEqual(summary["totals"]["watersheds_count"], 0)
    _, archives = self._request("GET", "/api/archives", headers=guest)
    self.assertEqual(archives["archives"], [])
    # Accounts are not affected by the guest reset.
    self.assertIn("bob", self._usernames())

  def test_delete_account(self):
    self._request("POST", "/api/profile/create", {"username": "alice"})
    self._request("POST", "/api/profile/create", {"username": "bob"})  # Active.

    # Removing another account keeps the active one.
    status, data = self._request("POST", "/api/profile/delete", {"username": "Alice"})
    self.assertEqual(status, 200)
    self.assertEqual(data["removed"], "alice")
    self.assertEqual(data["profile"]["username"], "bob")
    self.assertEqual(self._usernames(), ["bob", "guest"])

    # Removing the active account switches to the guest session.
    status, data = self._request("POST", "/api/profile/delete", {"username": "bob"})
    self.assertEqual(status, 200)
    self.assertEqual(data["profile"]["username"], "guest")
    self.assertTrue(data["profile"]["is_guest"])
    _, current = self._request("GET", "/api/profile/current")
    self.assertEqual(current["active_username"], "guest")
    self.assertEqual(self._usernames(), ["guest"])

    # A missing account is a 404 with a code; the guest session can't be removed.
    status, data = self._request("POST", "/api/profile/delete", {"username": "bob"})
    self.assertEqual(status, 404)
    self.assertEqual(data["code"], "account_not_found")
    for bad in ("", "guest"):
      status, data = self._request("POST", "/api/profile/delete", {"username": bad})
      self.assertEqual(status, 400, bad)
    # Unknown endpoints have no code, so clients can tell the two 404s apart.
    status, data = self._request("POST", "/api/profile/no_such_endpoint", {})
    self.assertEqual(status, 404)
    self.assertNotIn("code", data)


if __name__ == "__main__":
  unittest.main()
