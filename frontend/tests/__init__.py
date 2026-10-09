"""Tests for Earthkit Hydro Web package."""

import atexit
import os
import shutil
import tempfile

# Keep the test suite away from real accounts: servers started by tests store
# profiles in a throwaway folder instead of data/profiles (see
# profile_manager.get_profile_manager). Without this, tests create accounts such
# as "test_geo_user" that show up in the app.
if not os.environ.get("EARTHKIT_PROFILES_DIR"):
  _TEST_PROFILES_DIR = tempfile.mkdtemp(prefix="earthkit_test_profiles_")
  os.environ["EARTHKIT_PROFILES_DIR"] = _TEST_PROFILES_DIR
  atexit.register(shutil.rmtree, _TEST_PROFILES_DIR, ignore_errors=True)
