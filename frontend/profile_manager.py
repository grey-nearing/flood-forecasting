"""User Profile Manager for Earthkit Hydro Web.

Manages local user profiles, allowing users to log in or work as guests.
Persists active watersheds, Zarr archives, and settings per profile so data
instantly populates upon logging in.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional

try:
  from frontend.config import (
      ARCHIVES_DIR,
      ATTRIBUTES_DIR,
      DATA_DIR,
      HISTORICAL_DIR,
      POLYGONS_DIR,
      REALTIME_DIR,
  )
except ImportError:
  try:
    from frontend.config import (
        ARCHIVES_DIR,
        ATTRIBUTES_DIR,
        DATA_DIR,
        HISTORICAL_DIR,
        POLYGONS_DIR,
        REALTIME_DIR,
    )
  except ImportError:
    from config import (
        ARCHIVES_DIR,
        ATTRIBUTES_DIR,
        DATA_DIR,
        HISTORICAL_DIR,
        POLYGONS_DIR,
        REALTIME_DIR,
    )

logger = logging.getLogger(__name__)

# Profiles root directory
PROFILES_DIR = DATA_DIR / "profiles"
try:
  PROFILES_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
  pass


def _sanitize_username(username: str) -> str:
  """Sanitizes username for safe filesystem directory naming."""
  clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", (username or "").strip().lower())
  # Leading dots would allow "." / ".." (escaping the profiles folder) or hidden folders.
  clean = clean.lstrip(".")
  return clean or "guest"


LEGACY_PROFILE_SUBDIRS = (
    "polygons",
    "attributes",
    "historical",
    "forecast",
    "archives",
)
DYNAMICS_PRODUCTS = ("ERA5_LAND", "CPC", "IMERG", "HRES", "GRAPHCAST")
V2_PROFILE_TOP_DIRS = (
    "catchments",
    "statics",
    "dynamics",
    "realtime",
    "targets",
    "assimilation",
    "models",
    "forecasts",
    "maas",
    "jobs",
)


def _extract_basin_id(feature: Dict[str, Any], idx: int = 0) -> str:
  """Extracts a canonical basin_id string from a GeoJSON Feature."""
  props = (
      feature.get("properties")
      if isinstance(feature.get("properties"), dict)
      else {}
  )
  for candidate in (
      feature.get("id"),
      props.get("id"),
      props.get("basin_id"),
      props.get("HYBAS_ID"),
      props.get("catchment_id"),
      props.get("hybas_id"),
      props.get("gauge_id"),
  ):
    if candidate is not None and str(candidate).strip():
      return str(candidate).strip()
  return f"basin_{idx}"


class ProfileManager:
  """Manages user profiles, state persistence, and directory sandboxing.

  Accounts:
    * "guest" always exists. Its data is kept for as long as the server runs and
      is erased (clear_guest_data) every time the server starts.
    * User accounts are created explicitly (create_account) and are identified by
      a profiles/<username>/profile.json file. Their data persists across restarts
      until the account is deleted (delete_account).
  """

  def __init__(self, profiles_dir: Optional[Path] = None):
    self.profiles_dir = Path(profiles_dir or PROFILES_DIR)
    try:
      self.profiles_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
      pass
    self._active_username: str = "guest"
    # Guest data is NOT wiped here: the server calls clear_guest_data() once at
    # startup (run_server). Wiping in the constructor would let any script or test
    # that creates a ProfileManager erase a running server's guest session.

  @property
  def active_username(self) -> str:
    return self._active_username

  def clear_guest_data(self) -> None:
    """Erases everything stored for the guest and recreates an empty workspace.

    Removes the whole profiles/guest folder (basins, Zarr stores, attributes,
    models, jobs, MaaS snapshots, cached results, settings, ...), so no guest
    data survives. Called once when the server starts.
    """
    guest_dir = self.profiles_dir / "guest"
    if guest_dir.exists():
      shutil.rmtree(guest_dir, ignore_errors=True)
    if guest_dir.exists():
      # Fall back to clearing known folders if the folder itself can't be removed.
      for sub in LEGACY_PROFILE_SUBDIRS + V2_PROFILE_TOP_DIRS:
        shutil.rmtree(guest_dir / sub, ignore_errors=True)
      for name in ("watersheds.json", "profile.json"):
        (guest_dir / name).unlink(missing_ok=True)
    # Ensure clean empty subdirectories exist
    self.get_profile_dir("guest")
    # Save clean empty feature collection
    self.save_watersheds([], username="guest")

  def account_exists(self, username: Optional[str]) -> bool:
    """True for the built-in guest and for user accounts that have been created."""
    user = _sanitize_username(username)
    if user == "guest":
      return True
    return (self.profiles_dir / user / "profile.json").is_file()

  def create_account(
      self,
      username: str,
      email: Optional[str] = None,
      display_name: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Creates a new, empty user account and makes it the active profile.

    Raises:
      ValueError: If the username is empty or reserved ("guest").
      FileExistsError: If an account with this username already exists.
    """
    if not (username or "").strip():
      raise ValueError("Please enter a username.")
    if username.strip().lower() == "guest":
      raise ValueError("'guest' is reserved for the temporary guest session.")
    clean_user = _sanitize_username(username)
    if clean_user == "guest" or not re.search(r"[a-z0-9]", clean_user):
      raise ValueError("Please choose a username containing letters or numbers.")
    if self.account_exists(clean_user):
      raise FileExistsError(
          f"An account named '{clean_user}' already exists. Log in instead."
      )
    # A folder without profile.json is not an account (e.g. left behind by a
    # script); remove it so the new account starts empty.
    leftover = self.profiles_dir / clean_user
    if leftover.exists():
      shutil.rmtree(leftover, ignore_errors=True)
    info = self.login_profile(clean_user, email=email, display_name=display_name)
    logger.info("Created account '%s'", clean_user)
    return info

  def delete_account(self, username: str) -> str:
    """Permanently deletes a user account and everything saved in it.

    Removes the whole profiles/<username> folder (basins, Zarr stores,
    attributes, models, jobs, cached results, settings, ...). If the account is
    the active profile, the guest session becomes active.

    Args:
      username: The account to delete.

    Returns:
      The username of the deleted account, as stored on disk.

    Raises:
      ValueError: If the username is empty or names the guest session.
      LookupError: If no account with this username exists.
      OSError: If the account could not be deleted.
    """
    if not (username or "").strip():
      raise ValueError("Please choose an account to remove.")
    clean_user = _sanitize_username(username)
    if clean_user == "guest":
      raise ValueError(
          "The guest session can't be removed. Its data is erased when the"
          " server restarts."
      )
    if not self.account_exists(clean_user):
      raise LookupError(f"No account named '{clean_user}'.")
    account_dir = self.profiles_dir / clean_user
    if (
        account_dir.is_symlink()
        or account_dir.resolve().parent != self.profiles_dir.resolve()
    ):
      raise ValueError(f"'{clean_user}' is not stored in the profiles folder.")
    if self._active_username == clean_user:
      self._active_username = "guest"
    shutil.rmtree(account_dir, ignore_errors=True)
    # Anything that could not be deleted (e.g. a file still in use) must no
    # longer count as an account.
    (account_dir / "profile.json").unlink(missing_ok=True)
    if self.account_exists(clean_user):
      raise OSError(f"Couldn't delete {account_dir}.")
    logger.info("Deleted account '%s'", clean_user)
    return clean_user

  def get_profile_dir(self, username: Optional[str] = None) -> Path:
    """Returns the dedicated filesystem directory for a given profile, provisioning both legacy and 13-category subdirs."""
    user = _sanitize_username(username or self._active_username)
    p_dir = self.profiles_dir / user
    try:
      p_dir.mkdir(parents=True, exist_ok=True)
      for sub in LEGACY_PROFILE_SUBDIRS:
        (p_dir / sub).mkdir(parents=True, exist_ok=True)

      # 13-category googlehydrology-native directory structure
      for sub in ("shapes", "zonal_weights", "basin_lists"):
        (p_dir / "catchments" / sub).mkdir(parents=True, exist_ok=True)
      (p_dir / "statics" / "caravan_csv").mkdir(parents=True, exist_ok=True)
      for prod in DYNAMICS_PRODUCTS:
        (p_dir / "dynamics" / prod).mkdir(parents=True, exist_ok=True)
        (p_dir / "realtime" / "dynamics" / prod).mkdir(
            parents=True, exist_ok=True
        )
      (p_dir / "realtime" / "scenarios").mkdir(parents=True, exist_ok=True)
      (p_dir / "targets" / "uploads").mkdir(parents=True, exist_ok=True)
      (p_dir / "assimilation" / "uploads").mkdir(parents=True, exist_ok=True)
      (p_dir / "assimilation" / "da_states").mkdir(parents=True, exist_ok=True)
      (p_dir / "models" / "hot_start").mkdir(parents=True, exist_ok=True)
      (p_dir / "models" / "runs").mkdir(parents=True, exist_ok=True)
      (p_dir / "forecasts").mkdir(parents=True, exist_ok=True)
      (p_dir / "maas" / "snapshots").mkdir(parents=True, exist_ok=True)
      (p_dir / "jobs").mkdir(parents=True, exist_ok=True)
    except Exception:
      pass
    return p_dir

  # ---------------------------------------------------------------------------
  # Legacy Directory Accessors (100% Backwards-Compatible)
  # ---------------------------------------------------------------------------

  def get_polygons_dir(self, username: Optional[str] = None) -> Path:
    """Returns the polygons directory for the specified or active profile."""
    user = _sanitize_username(username or self._active_username)
    d = self.get_profile_dir(user) / "polygons"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_attributes_dir(self, username: Optional[str] = None) -> Path:
    """Returns the attributes directory for the specified or active profile."""
    user = _sanitize_username(username or self._active_username)
    d = self.get_profile_dir(user) / "attributes"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_historical_dir(self, username: Optional[str] = None) -> Path:
    """Returns the historical weather data directory for the specified or active profile."""
    user = _sanitize_username(username or self._active_username)
    d = self.get_profile_dir(user) / "historical"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_forecast_dir(self, username: Optional[str] = None) -> Path:
    """Returns the forecast weather data directory for the specified or active profile."""
    user = _sanitize_username(username or self._active_username)
    d = self.get_profile_dir(user) / "forecast"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_realtime_dir(self, username: Optional[str] = None) -> Path:
    """Backwards compatibility alias for get_forecast_dir."""
    return self.get_forecast_dir(username)

  def get_archives_dir(self, username: Optional[str] = None) -> Path:
    """Returns the primary archives directory (historical) for backwards compatibility."""
    return self.get_historical_dir(username)

  # ---------------------------------------------------------------------------
  # 13-Category googlehydrology-Native Directory Accessors
  # ---------------------------------------------------------------------------

  def get_catchments_dir(self, username: Optional[str] = None) -> Path:
    """Returns the catchments directory for the specified or active profile."""
    d = self.get_profile_dir(username) / "catchments"
    for sub in ("shapes", "zonal_weights", "basin_lists"):
      (d / sub).mkdir(parents=True, exist_ok=True)
    return d

  def get_catchment_shapes_dir(self, username: Optional[str] = None) -> Path:
    """Returns the catchments/shapes directory for per-basin GeoJSON files."""
    d = self.get_catchments_dir(username) / "shapes"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_zonal_weights_dir(self, username: Optional[str] = None) -> Path:
    """Returns the catchments/zonal_weights directory for cached ZonalWeightMatrix (.npz) files."""
    d = self.get_catchments_dir(username) / "zonal_weights"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_basin_lists_dir(self, username: Optional[str] = None) -> Path:
    """Returns the catchments/basin_lists directory for googlehydrology basin list files."""
    d = self.get_catchments_dir(username) / "basin_lists"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_statics_dir(self, username: Optional[str] = None) -> Path:
    """Returns the statics directory for Caravan static attributes."""
    d = self.get_profile_dir(username) / "statics"
    (d / "caravan_csv").mkdir(parents=True, exist_ok=True)
    return d

  def get_caravan_attributes_dir(self, username: Optional[str] = None) -> Path:
    """Returns the statics/caravan_csv directory for Caravan static attribute CSVs."""
    d = self.get_statics_dir(username) / "caravan_csv"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_statics_zarr_path(self, username: Optional[str] = None) -> Path:
    """Returns the consolidated static attributes Zarr store path (statics/attributes.zarr)."""
    return self.get_statics_dir(username) / "attributes.zarr"

  def get_dynamics_dir(
      self, username: Optional[str] = None, product: Optional[str] = None
  ) -> Path:
    """Returns the historical dynamics directory, optionally scoped to a specific meteorological product."""
    d = self.get_profile_dir(username) / "dynamics"
    d.mkdir(parents=True, exist_ok=True)
    if product:
      prod_norm = product.strip().upper().replace("-", "_")
      if prod_norm == "ERA5LAND":
        prod_norm = "ERA5_LAND"
      prod_dir = d / prod_norm
      prod_dir.mkdir(parents=True, exist_ok=True)
      return prod_dir
    return d

  def get_dynamics_zarr_path(
      self, username: Optional[str] = None, product: str = "ERA5_LAND"
  ) -> Path:
    """Returns the historical dynamics Zarr path for a meteorological product (dynamics/<PRODUCT>/timeseries.zarr)."""
    return self.get_dynamics_dir(username, product.upper()) / "timeseries.zarr"

  def get_realtime_dir_v2(self, username: Optional[str] = None) -> Path:
    """Returns the googlehydrology v2 realtime directory (realtime/)."""
    d = self.get_profile_dir(username) / "realtime"
    (d / "dynamics").mkdir(parents=True, exist_ok=True)
    (d / "scenarios").mkdir(parents=True, exist_ok=True)
    return d

  def get_realtime_dynamics_dir(self, username: Optional[str] = None) -> Path:
    """Returns the realtime/dynamics directory for operational forcing stores."""
    d = self.get_realtime_dir_v2(username) / "dynamics"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_realtime_dynamics_zarr_path(
      self, username: Optional[str] = None, product: str = "HRES"
  ) -> Path:
    """Returns the realtime operational dynamics Zarr path for a product (realtime/dynamics/<PRODUCT>/timeseries.zarr)."""
    prod_norm = (product or "HRES").strip().upper().replace("-", "_")
    if prod_norm == "ERA5LAND":
      prod_norm = "ERA5_LAND"
    prod_dir = self.get_realtime_dir_v2(username) / "dynamics" / prod_norm
    prod_dir.mkdir(parents=True, exist_ok=True)
    return prod_dir / "timeseries.zarr"

  def get_targets_dir(self, username: Optional[str] = None) -> Path:
    """Returns the targets directory for historical streamflow observations."""
    d = self.get_profile_dir(username) / "targets"
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    return d

  def get_targets_zarr_path(self, username: Optional[str] = None) -> Path:
    """Returns the historical target streamflow Zarr store path (targets/streamflow.zarr)."""
    return self.get_targets_dir(username) / "streamflow.zarr"

  def get_return_periods_path(self, username: Optional[str] = None) -> Path:
    """Returns the return period flood thresholds JSON path (targets/return_periods.json)."""
    return self.get_targets_dir(username) / "return_periods.json"

  def get_assimilation_dir(self, username: Optional[str] = None) -> Path:
    """Returns the assimilation directory for real-time DA streamflow and DA state checkpoints."""
    d = self.get_profile_dir(username) / "assimilation"
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    (d / "da_states").mkdir(parents=True, exist_ok=True)
    return d

  def get_assimilation_zarr_path(self, username: Optional[str] = None) -> Path:
    """Returns the real-time assimilation streamflow Zarr store path (assimilation/streamflow_realtime.zarr)."""
    return self.get_assimilation_dir(username) / "streamflow_realtime.zarr"

  def get_da_states_dir(self, username: Optional[str] = None) -> Path:
    """Returns the assimilation/da_states directory for runtime Variational DA outputs."""
    d = self.get_assimilation_dir(username) / "da_states"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_models_dir(self, username: Optional[str] = None) -> Path:
    """Returns the models directory for user model runs and hot-start states."""
    d = self.get_profile_dir(username) / "models"
    (d / "hot_start").mkdir(parents=True, exist_ok=True)
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d

  def get_hot_start_dir(self, username: Optional[str] = None) -> Path:
    """Returns the models/hot_start directory for recurrent LSTM (h_t, c_t) states."""
    d = self.get_models_dir(username) / "hot_start"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_model_states_dir(self, username: Optional[str] = None) -> Path:
    """Alias for get_hot_start_dir."""
    return self.get_hot_start_dir(username)

  def get_model_runs_dir(self, username: Optional[str] = None) -> Path:
    """Returns the models/runs directory for trained and fine-tuned googlehydrology run directories."""
    d = self.get_models_dir(username) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_forecasts_dir(
      self, username: Optional[str] = None, basin_id: Optional[str] = None
  ) -> Path:
    """Returns the forecasts directory, optionally scoped to a specific basin_id."""
    d = self.get_profile_dir(username) / "forecasts"
    d.mkdir(parents=True, exist_ok=True)
    if basin_id:
      b_dir = d / str(basin_id).strip()
      (b_dir / "history").mkdir(parents=True, exist_ok=True)
      return b_dir
    return d

  def get_jobs_dir(self, username: Optional[str] = None) -> Path:
    """Returns the jobs directory for persistent async background job states."""
    d = self.get_profile_dir(username) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def get_profile_info(self, username: Optional[str] = None) -> Dict[str, Any]:
    """Reads profile metadata from profile.json."""
    user = _sanitize_username(username or self._active_username)
    p_dir = self.get_profile_dir(user)
    meta_file = p_dir / "profile.json"

    if meta_file.exists():
      try:
        with open(meta_file, "r", encoding="utf-8") as f:
          return json.load(f)
      except Exception as e:
        logger.warning("Failed to read profile.json for %s: %s", user, e)

    # Default metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "username": user,
        "display_name": user.title() if user != "guest" else "Guest User",
        "email": f"{user}@google.com" if user != "guest" else "",
        "created_at": now_iso,
        "last_login": now_iso,
        "is_guest": user == "guest",
        "settings": {
            "default_weather_source": "cpc",
            "default_hydro_dataset": "hydroatlas",
        },
    }

  def login_profile(
      self,
      username: str,
      email: Optional[str] = None,
      display_name: Optional[str] = None,
      settings: Optional[Dict[str, Any]] = None,
      create_if_missing: bool = True,
  ) -> Dict[str, Any]:
    """Logs in to a user profile and sets it as active.

    Args:
      username: Account to log in to.
      email: Optional email to store on the profile.
      display_name: Optional display name to store on the profile.
      settings: Optional settings to merge into the profile.
      create_if_missing: If False, raise LookupError instead of creating an
        account that does not exist yet (use create_account to create one).

    Returns:
      The profile metadata.
    """
    clean_user = _sanitize_username(username)
    if not create_if_missing and not self.account_exists(clean_user):
      raise LookupError(
          f"No account named '{clean_user}'. Use Create Account to make one."
      )
    # Guest data is intentionally kept here: it lasts until the server restarts.
    self._active_username = clean_user
    p_dir = self.get_profile_dir(clean_user)
    meta_file = p_dir / "profile.json"

    info = self.get_profile_info(clean_user)
    info["last_login"] = datetime.now(timezone.utc).isoformat()
    info["is_guest"] = clean_user == "guest"

    if email:
      info["email"] = email.strip()
    if display_name:
      info["display_name"] = display_name.strip()
    if settings:
      info["settings"] = {**info.get("settings", {}), **settings}

    try:
      with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    except Exception as e:
      logger.warning("Failed to save profile.json for %s: %s", clean_user, e)

    logger.info("Active profile set to '%s'", clean_user)
    return info

  def logout_profile(self) -> Dict[str, Any]:
    """Switches back to the guest session (guest data is kept until the server restarts)."""
    self._active_username = "guest"
    return self.get_profile_info("guest")

  def list_profiles(self) -> List[Dict[str, Any]]:
    """Lists the guest session plus every user account stored locally."""
    profiles = []
    if not self.profiles_dir.exists():
      return profiles
    self.get_profile_dir("guest")  # The guest session always exists.

    for entry in self.profiles_dir.iterdir():
      if entry.is_dir():
        if entry.name != "guest" and not (entry / "profile.json").is_file():
          continue  # Not an account (e.g. a folder left behind by a script).
        info = self.get_profile_info(entry.name)
        watersheds = self.load_watersheds(entry.name)
        archives = (
            list((entry / "historical").glob("*.zarr"))
            + list((entry / "forecast").glob("*.zarr"))
            + list((entry / "archives").glob("*.zarr"))
        )
        info["watersheds_count"] = len(watersheds)
        info["archives_count"] = len(set(a.name for a in archives))
        info["is_active"] = entry.name == self._active_username
        profiles.append(info)

    profiles.sort(key=lambda x: x.get("last_login", ""), reverse=True)
    return profiles

  def load_watersheds(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads saved GeoJSON watershed features for a profile from polygons/ or catchments/."""
    user = _sanitize_username(username or self._active_username)
    poly_dir = self.get_polygons_dir(user)
    ws_file = poly_dir / "watersheds.json"

    # Migrate legacy root watersheds.json if present
    p_dir = self.get_profile_dir(user)
    root_ws_file = p_dir / "watersheds.json"
    if not ws_file.exists() and root_ws_file.exists():
      try:
        root_ws_file.replace(ws_file)
      except Exception:
        pass
    elif root_ws_file.exists():
      try:
        root_ws_file.unlink(missing_ok=True)
      except Exception:
        pass

    for candidate in (ws_file, self.get_catchments_dir(user) / "watersheds.geojson"):
      if candidate.exists():
        try:
          with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data.get("type") == "FeatureCollection":
              return data.get("features", [])
            elif isinstance(data, list):
              return data
        except Exception as e:
          logger.warning("Failed to load %s for %s: %s", candidate, user, e)
    return []

  def sync_catchment_artifacts(
      self,
      features: List[Dict[str, Any]],
      username: Optional[str] = None,
  ) -> List[str]:
    """Synchronizes catchments/watersheds.geojson, catchments/shapes/<basin_id>.geojson, and basin_lists/*.txt."""
    user = _sanitize_username(username or self._active_username)
    catchments_dir = self.get_catchments_dir(user)
    shapes_dir = self.get_catchment_shapes_dir(user)
    basin_lists_dir = self.get_basin_lists_dir(user)

    fc = {
        "type": "FeatureCollection",
        "count": len(features),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }

    # 1. Write catchments/watersheds.geojson
    try:
      with open(
          catchments_dir / "watersheds.geojson", "w", encoding="utf-8"
      ) as f:
        json.dump(fc, f, indent=2)
    except Exception as e:
      logger.warning(
          "Failed to write catchments/watersheds.geojson for %s: %s", user, e
      )

    # 2. Write individual per-basin GeoJSON files in catchments/shapes/<basin_id>.geojson
    basin_ids: List[str] = []
    expected_shape_files = set()
    for idx, feat in enumerate(features):
      if not isinstance(feat, dict):
        continue
      basin_id = _extract_basin_id(feat, idx)
      if basin_id not in basin_ids:
        basin_ids.append(basin_id)

      props = (
          dict(feat.get("properties"))
          if isinstance(feat.get("properties"), dict)
          else {}
      )
      props.setdefault("id", basin_id)
      props.setdefault("basin_id", basin_id)

      single_feat = {
          **feat,
          "type": "Feature",
          "id": basin_id,
          "properties": props,
          "geometry": feat.get("geometry"),
      }
      shape_file = shapes_dir / f"{basin_id}.geojson"
      expected_shape_files.add(shape_file.name)
      try:
        with open(shape_file, "w", encoding="utf-8") as f:
          json.dump(single_feat, f, indent=2)
      except Exception as e:
        logger.warning("Failed to write shape file %s: %s", shape_file, e)

    # Remove stale shape files that are no longer in features
    try:
      for existing_file in shapes_dir.glob("*.geojson"):
        if existing_file.name not in expected_shape_files:
          existing_file.unlink(missing_ok=True)
    except Exception:
      pass

    # 3. Write plaintext basin list files in catchments/basin_lists/
    list_content = ("\n".join(basin_ids) + "\n") if basin_ids else ""
    for list_name in (
        "all_basins.txt",
        "train_basins.txt",
        "val_basins.txt",
        "test_basins.txt",
    ):
      try:
        (basin_lists_dir / list_name).write_text(list_content, encoding="utf-8")
      except Exception as e:
        logger.warning("Failed to write basin list %s: %s", list_name, e)

    return basin_ids

  def save_watersheds(
      self,
      features: List[Dict[str, Any]],
      username: Optional[str] = None,
  ) -> bool:
    """Persists GeoJSON watershed features in polygons/watersheds.json and syncs catchments/ artifacts."""
    user = _sanitize_username(username or self._active_username)
    poly_dir = self.get_polygons_dir(user)
    poly_dir.mkdir(parents=True, exist_ok=True)
    ws_file = poly_dir / "watersheds.json"

    # Remove any stale root watersheds.json
    p_dir = self.get_profile_dir(user)
    root_ws_file = p_dir / "watersheds.json"
    if root_ws_file.exists():
      try:
        root_ws_file.unlink(missing_ok=True)
      except Exception:
        pass

    fc = {
        "type": "FeatureCollection",
        "count": len(features),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }

    try:
      with open(ws_file, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
      self.sync_catchment_artifacts(features, username=user)
      return True
    except Exception as e:
      logger.warning("Failed to write watersheds.json for %s: %s", user, e)
      return False

  def save_settings(
      self, settings: Dict[str, Any], username: Optional[str] = None
  ) -> Dict[str, Any]:
    """Updates user preferences and settings for a profile."""
    user = _sanitize_username(username or self._active_username)
    info = self.get_profile_info(user)
    info["settings"] = {**info.get("settings", {}), **settings}
    p_dir = self.get_profile_dir(user)
    meta_file = p_dir / "profile.json"

    try:
      with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    except Exception as e:
      logger.warning("Failed to save settings for %s: %s", user, e)
    return info

  def _inspect_zarr_store(
      self, zarr_path: Path, profile_dir: Path, default_category: str = "historical"
  ) -> Dict[str, Any]:
    """Inspects a Zarr store on disk and returns rich metadata without loading full arrays."""
    size_bytes = sum(
        f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
    )
    try:
      mtime_iso = datetime.fromtimestamp(
          zarr_path.stat().st_mtime, tz=timezone.utc
      ).isoformat()
    except Exception:
      mtime_iso = ""

    try:
      rel_path = str(zarr_path.relative_to(profile_dir))
    except ValueError:
      rel_path = zarr_path.name

    meta: Dict[str, Any] = {
        "filename": zarr_path.name,
        "rel_path": rel_path,
        "path": str(zarr_path),
        "category": default_category,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "modified_at": mtime_iso,
        "basins": [],
        "total_basins": 0,
        "n_timesteps": 0,
        "lead_time_days": 0,
        "start_date": None,
        "end_date": None,
        "dynamic_variables": [],
        "static_variables": [],
        "all_variables": [],
        "dimensions": {},
        "source_dataset": None,
        "temporal_resolution": None,
        "title": None,
        "issue_date": None,
    }

    try:
      import xarray as xr

      ds = xr.open_zarr(str(zarr_path), decode_timedelta=False)
      meta["dimensions"] = {str(k): int(v) for k, v in ds.sizes.items()}

      # Extract basin IDs
      basin_coord = None
      for b_key in ("basin", "basin_id", "catchment_id", "gauge_id"):
        if b_key in ds.coords:
          basin_coord = ds.coords[b_key]
          break
      if basin_coord is not None:
        meta["basins"] = [str(b) for b in basin_coord.values]
      meta["total_basins"] = len(meta["basins"])

      # Extract time / date range
      time_coord = None
      for t_key in ("date", "time", "valid_time", "issue_time"):
        if t_key in ds.coords and ds.coords[t_key].ndim >= 1:
          time_coord = ds.coords[t_key]
          break
      if time_coord is not None and time_coord.size > 0:
        meta["n_timesteps"] = int(time_coord.size)
        t_vals = time_coord.values
        meta["start_date"] = str(t_vals[0])[:10]
        meta["end_date"] = str(t_vals[-1])[:10]

      if "lead_time" in ds.sizes:
        meta["lead_time_days"] = int(ds.sizes["lead_time"])

      all_vars = sorted([str(v) for v in ds.data_vars])
      static_vars = [
          v
          for v in all_vars
          if v.startswith("static_")
          or v in ("basin_area", "area_km2", "latitude", "longitude")
      ]
      dyn_vars = [v for v in all_vars if v not in static_vars]
      meta["all_variables"] = all_vars
      meta["dynamic_variables"] = dyn_vars
      meta["static_variables"] = static_vars

      attrs = dict(ds.attrs or {})
      meta["title"] = attrs.get("title")
      meta["source_dataset"] = (
          attrs.get("source_dataset")
          or attrs.get("models")
          or attrs.get("source_platform")
          or (
              zarr_path.parent.name
              if zarr_path.parent.name in DYNAMICS_PRODUCTS
              else None
          )
      )
      meta["temporal_resolution"] = attrs.get("temporal_resolution", "1D")
      meta["issue_date"] = (
          attrs.get("issue_date")
          or attrs.get("initialization_time")
          or meta["start_date"]
      )
    except Exception as e:
      logger.debug("Could not inspect Zarr store %s: %s", zarr_path, e)

    return meta

  def get_account_summary(
      self, username: Optional[str] = None
  ) -> Dict[str, Any]:
    """Returns a comprehensive inventory of all data stored under a user's account."""
    user = _sanitize_username(username or self._active_username)
    p_dir = self.get_profile_dir(user)
    profile_info = self.get_profile_info(user)

    # 1. Polygons & Catchment Boundaries
    features = self.load_watersheds(user)
    if (
        features
        and not (self.get_catchments_dir(user) / "watersheds.geojson").exists()
    ):
      self.sync_catchment_artifacts(features, username=user)

    catchments_list: List[Dict[str, Any]] = []
    total_area_km2 = 0.0
    for idx, feat in enumerate(features):
      if not isinstance(feat, dict):
        continue
      cid = _extract_basin_id(feat, idx)
      props = (
          feat.get("properties")
          if isinstance(feat.get("properties"), dict)
          else {}
      )
      area = float(
          props.get("area_km2")
          or props.get("SUB_AREA")
          or props.get("UP_AREA")
          or 0.0
      )
      total_area_km2 += area
      source = props.get("source") or (
          "delineated"
          if props.get("delineation_method") != "User GeoJSON Polygon"
          else "uploaded"
      )
      outlet = (
          props.get("outlet") if isinstance(props.get("outlet"), dict) else {}
      )
      catchments_list.append({
          "catchment_id": cid,
          "name": props.get("name") or props.get("station_name") or cid,
          "area_km2": round(area, 2),
          "source": source,
          "dataset": props.get("dataset") or props.get("dem_id") or "hydroatlas",
          "dem_id": props.get("dem_id") or "hydrosheds_90m",
          "delineation_method": props.get("delineation_method")
          or (
              "Auto-Delineated from DEM"
              if source == "delineated"
              else "Uploaded GeoJSON"
          ),
          "stream_order": props.get("stream_order"),
          "outlet_lat": outlet.get("latitude"),
          "outlet_lon": outlet.get("longitude"),
          "reach_id": outlet.get("reach_id"),
          "geometry_type": (feat.get("geometry") or {}).get("type", "Polygon"),
      })

    polygon_files: List[Dict[str, Any]] = []
    seen_poly_files = set()
    for base_d in (self.get_polygons_dir(user), self.get_catchments_dir(user)):
      if not base_d.exists():
        continue
      for ext in ("*.json", "*.geojson", "*.shp", "*.gpkg", "*.npz", "*.txt"):
        for pf in sorted(base_d.rglob(ext)):
          if pf.is_file() and str(pf) not in seen_poly_files:
            seen_poly_files.add(str(pf))
            sz = pf.stat().st_size
            if sz == 0 and pf.suffix == ".txt":
              continue
            polygon_files.append({
                "filename": pf.name,
                "rel_path": str(pf.relative_to(p_dir)),
                "path": str(pf),
                "size_kb": round(sz / 1024.0, 2),
                "size_mb": round(sz / (1024.0 * 1024.0), 4),
            })

    # 2. Static Catchment Attributes
    attr_files: List[Dict[str, Any]] = []
    attr_basins: List[str] = []
    attr_variables: List[str] = []
    attr_basin_previews: List[Dict[str, Any]] = []
    seen_attr_files = set()

    for base_d in (self.get_attributes_dir(user), self.get_statics_dir(user)):
      if not base_d.exists():
        continue
      for ext in ("*.csv", "*.json", "*.parquet", "*.nc"):
        for af in sorted(base_d.rglob(ext)):
          if af.is_file() and str(af) not in seen_attr_files:
            seen_attr_files.add(str(af))
            sz = af.stat().st_size
            attr_files.append({
                "filename": af.name,
                "rel_path": str(af.relative_to(p_dir)),
                "path": str(af),
                "size_kb": round(sz / 1024.0, 2),
                "size_mb": round(sz / (1024.0 * 1024.0), 3),
                "download_url": (
                    f"/api/attributes/csv?username={user}"
                    if af.suffix == ".csv"
                    else None
                ),
            })
            if af.suffix == ".csv" and not attr_basin_previews:
              try:
                import pandas as pd

                df = pd.read_csv(af)
                if not df.empty:
                  id_col = next(
                      (
                          c
                          for c in (
                              "gauge_id",
                              "basin_id",
                              "catchment_id",
                              "id",
                          )
                          if c in df.columns
                      ),
                      df.columns[0],
                  )
                  attr_basins = [str(v) for v in df[id_col].dropna().unique()]
                  attr_variables = [
                      str(c) for c in df.columns if c != id_col
                  ]
                  key_cols = [
                      c
                      for c in (
                          "ele_mt_sav",
                          "slp_dg_sav",
                          "pre_mm_syr",
                          "run_mm_syr",
                          "for_pc_use",
                          "snd_pc_sav",
                          "cly_pc_sav",
                          "ari_ix_sav",
                          "tmp_dc_syr",
                          "inu_pc_slt",
                          "lka_pc_sse",
                          "gwt_cm_sav",
                      )
                      if c in df.columns
                  ]
                  if len(key_cols) < 5:
                    for c in attr_variables:
                      if c not in key_cols and len(key_cols) < 6:
                        key_cols.append(c)
                  for _, row in df.head(50).iterrows():
                    b_id = str(row[id_col])
                    metrics = {}
                    for kc in key_cols[:6]:
                      val = row[kc]
                      if pd.notna(val):
                        try:
                          metrics[kc] = round(float(val), 2)
                        except (ValueError, TypeError):
                          metrics[kc] = str(val)
                    attr_basin_previews.append({
                        "basin_id": b_id,
                        "variable_count": int(row.drop(labels=[id_col]).notna().sum()),
                        "highlights": metrics,
                    })
              except Exception as e:
                logger.debug("Failed to parse attributes CSV %s: %s", af, e)

    statics_zarr = self.get_statics_zarr_path(user)
    if statics_zarr.exists():
      z_info = self._inspect_zarr_store(statics_zarr, p_dir, "static_attributes")
      attr_files.append({
          "filename": statics_zarr.name,
          "rel_path": z_info["rel_path"],
          "path": str(statics_zarr),
          "size_kb": round(z_info["size_bytes"] / 1024.0, 2),
          "size_mb": z_info["size_mb"],
          "download_url": None,
      })
      for b in z_info["basins"]:
        if b not in attr_basins:
          attr_basins.append(b)
      if not attr_variables and z_info["all_variables"]:
        attr_variables = z_info["all_variables"]

    # 3. Historical Training & Meteorological Forcing Data
    historical_stores: List[Dict[str, Any]] = []
    seen_zarrs = set()
    for base_d in (
        self.get_historical_dir(user),
        self.get_dynamics_dir(user),
        p_dir / "archives",
    ):
      if not base_d.exists():
        continue
      for z_item in sorted(base_d.rglob("*.zarr")):
        if "forecast" in z_item.name or str(z_item) in seen_zarrs:
          continue
        seen_zarrs.add(str(z_item))
        z_meta = self._inspect_zarr_store(
            z_item, p_dir, default_category="Historical Training Data"
        )
        historical_stores.append(z_meta)

    # 4. Historical Streamflow Targets & Return Periods
    targets_stores: List[Dict[str, Any]] = []
    targets_zarr = self.get_targets_zarr_path(user)
    if targets_zarr.exists() and str(targets_zarr) not in seen_zarrs:
      seen_zarrs.add(str(targets_zarr))
      targets_stores.append(
          self._inspect_zarr_store(
              targets_zarr, p_dir, default_category="Historical Streamflow Targets"
          )
      )
    return_periods_data: Dict[str, Any] = {}
    rp_file = self.get_return_periods_path(user)
    if rp_file.exists():
      try:
        return_periods_data = json.loads(rp_file.read_text(encoding="utf-8"))
      except Exception:
        return_periods_data = {}
    target_uploads = []
    uploads_dir = self.get_targets_dir(user) / "uploads"
    if uploads_dir.exists():
      for uf in sorted(uploads_dir.glob("*.csv")):
        if uf.is_file():
          target_uploads.append({
              "filename": uf.name,
              "rel_path": str(uf.relative_to(p_dir)),
              "size_kb": round(uf.stat().st_size / 1024.0, 2),
          })

    # 5. Real-Time Weather Forecast Inputs & Assimilation
    forecast_stores: List[Dict[str, Any]] = []
    for base_d in (self.get_forecast_dir(user), self.get_realtime_dir_v2(user)):
      if not base_d.exists():
        continue
      for z_item in sorted(base_d.rglob("*.zarr")):
        if str(z_item) not in seen_zarrs:
          seen_zarrs.add(str(z_item))
          forecast_stores.append(
              self._inspect_zarr_store(
                  z_item, p_dir, default_category="Real-Time Weather Forecast"
              )
          )

    assimilation_stores: List[Dict[str, Any]] = []
    assim_zarr = self.get_assimilation_zarr_path(user)
    if assim_zarr.exists() and str(assim_zarr) not in seen_zarrs:
      seen_zarrs.add(str(assim_zarr))
      assimilation_stores.append(
          self._inspect_zarr_store(
              assim_zarr, p_dir, default_category="Real-Time Streamflow Assimilation"
          )
      )
    da_state_files = []
    da_dir = self.get_da_states_dir(user)
    if da_dir.exists():
      for df_item in sorted(da_dir.rglob("*")):
        if df_item.is_file():
          da_state_files.append({
              "filename": df_item.name,
              "rel_path": str(df_item.relative_to(p_dir)),
              "size_kb": round(df_item.stat().st_size / 1024.0, 2),
          })

    # 6. Models, Fine-Tuned Runs & Hot-Start Checkpoints
    model_runs: List[Dict[str, Any]] = []
    runs_dir = self.get_model_runs_dir(user)
    if runs_dir.exists():
      for run_entry in sorted(runs_dir.iterdir()):
        if run_entry.is_dir():
          r_bytes = sum(
              f.stat().st_size for f in run_entry.rglob("*") if f.is_file()
          )
          ckpt_files = [f.name for f in run_entry.rglob("*.pt")]
          model_runs.append({
              "run_id": run_entry.name,
              "rel_path": str(run_entry.relative_to(p_dir)),
              "size_mb": round(r_bytes / (1024.0 * 1024.0), 3),
              "checkpoints": ckpt_files,
              "has_config": (run_entry / "config.yml").exists(),
          })
    hot_start_files = []
    hs_dir = self.get_hot_start_dir(user)
    if hs_dir.exists():
      for hf in sorted(hs_dir.rglob("*")):
        if hf.is_file():
          hot_start_files.append({
              "filename": hf.name,
              "rel_path": str(hf.relative_to(p_dir)),
              "size_kb": round(hf.stat().st_size / 1024.0, 2),
          })

    # 7. Per-Basin Data Coverage Matrix
    all_basin_ids: List[str] = []
    poly_by_basin = {c["catchment_id"]: c for c in catchments_list}
    for cid in poly_by_basin:
      if cid not in all_basin_ids:
        all_basin_ids.append(cid)
    for b in attr_basins:
      if b not in all_basin_ids:
        all_basin_ids.append(b)
    for st in historical_stores + targets_stores + forecast_stores + assimilation_stores:
      for b in st.get("basins", []):
        if b not in all_basin_ids:
          all_basin_ids.append(b)

    attr_preview_by_basin = {
        p["basin_id"]: p for p in attr_basin_previews
    }

    basin_coverage: List[Dict[str, Any]] = []
    for bid in all_basin_ids:
      p_info = poly_by_basin.get(bid)
      a_info = attr_preview_by_basin.get(bid)
      has_attr = bid in attr_basins

      matching_hist = [s for s in historical_stores if bid in s.get("basins", [])]
      matching_targ = [s for s in targets_stores if bid in s.get("basins", [])]
      matching_fcst = [s for s in forecast_stores if bid in s.get("basins", [])]
      matching_assim = [
          s for s in assimilation_stores if bid in s.get("basins", [])
      ]

      hist_summary_str = None
      if matching_hist:
        h0 = matching_hist[0]
        src = h0.get("source_dataset") or h0.get("filename")
        dates = (
            f"{h0['start_date']} → {h0['end_date']}"
            if h0.get("start_date") and h0.get("end_date")
            else f"{h0.get('n_timesteps', 0)} steps"
        )
        hist_summary_str = f"{src} ({dates}, {len(h0.get('dynamic_variables', []))} vars)"

      fcst_summary_str = None
      if matching_fcst:
        f0 = matching_fcst[0]
        iss = f0.get("issue_date") or "Latest"
        lead = f0.get("lead_time_days") or f0.get("n_timesteps") or 0
        fcst_summary_str = f"Issued {iss} ({lead}d horizon, {len(f0.get('dynamic_variables', []))} vars)"

      targ_summary_str = None
      if matching_targ:
        t0 = matching_targ[0]
        targ_summary_str = (
            f"{t0.get('start_date')} → {t0.get('end_date')} ({t0.get('n_timesteps', 0)}d)"
        )

      basin_coverage.append({
          "basin_id": bid,
          "name": p_info["name"] if p_info else bid,
          "area_km2": p_info["area_km2"] if p_info else None,
          "has_polygon": p_info is not None,
          "polygon_source": p_info["source"] if p_info else None,
          "dataset": p_info["dataset"] if p_info else None,
          "has_attributes": has_attr,
          "attributes_count": (
              a_info["variable_count"]
              if a_info
              else (len(attr_variables) if has_attr else 0)
          ),
          "has_historical_weather": len(matching_hist) > 0,
          "has_historical": len(matching_hist) > 0,
          "historical_weather_info": hist_summary_str,
          "has_streamflow_targets": len(matching_targ) > 0,
          "streamflow_targets_info": targ_summary_str,
          "has_return_periods": bid in return_periods_data,
          "has_realtime_forecast": len(matching_fcst) > 0,
          "realtime_forecast_info": fcst_summary_str,
          "has_assimilation": len(matching_assim) > 0,
      })

    total_bytes = sum(
        f.stat().st_size for f in p_dir.rglob("*") if f.is_file()
    )

    hist_basins_set = set()
    for s in historical_stores:
      hist_basins_set.update(s.get("basins", []))
    fcst_basins_set = set()
    for s in forecast_stores:
      fcst_basins_set.update(s.get("basins", []))
    targ_basins_set = set()
    for s in targets_stores:
      targ_basins_set.update(s.get("basins", []))
    assim_basins_set = set()
    for s in assimilation_stores:
      assim_basins_set.update(s.get("basins", []))

    return {
        "username": user,
        "is_guest": bool(profile_info.get("is_guest", user == "guest")),
        "profile": profile_info,
        "profile_dir": str(p_dir),
        "available_profiles": self.list_profiles(),
        "totals": {
            "watersheds_count": len(catchments_list),
            "polygon_count": len(catchments_list),
            "total_area_km2": round(total_area_km2, 2),
            "attributes_basins_count": len(attr_basins),
            "attribute_basin_count": len(attr_basins),
            "attributes_variables_count": len(attr_variables),
            "attribute_variable_count": len(attr_variables),
            "attributes_files_count": len(attr_files),
            "historical_stores_count": len(historical_stores),
            "historical_store_count": len(historical_stores),
            "historical_basins_count": len(hist_basins_set),
            "targets_stores_count": len(targets_stores),
            "targets_basins_count": len(targ_basins_set),
            "forecast_stores_count": len(forecast_stores),
            "forecast_basins_count": len(fcst_basins_set),
            "assimilation_stores_count": len(assimilation_stores),
            "assimilation_basins_count": len(assim_basins_set),
            "models_runs_count": len(model_runs),
            "hot_start_count": len(hot_start_files),
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024.0 * 1024.0), 3),
        },
        "basin_coverage": basin_coverage,
        "polygons": {
            "count": len(catchments_list),
            "total_area_km2": round(total_area_km2, 2),
            "catchments": catchments_list,
            "files": polygon_files,
        },
        "attributes": {
            "basin_count": len(attr_basins),
            "variable_count": len(attr_variables),
            "basins": attr_basins,
            "variable_names": attr_variables,
            "basin_previews": attr_basin_previews,
            "files": attr_files,
        },
        "historical_training": {
            "store_count": len(historical_stores),
            "basin_count": len(hist_basins_set),
            "stores": historical_stores,
        },
        "streamflow_targets": {
            "store_count": len(targets_stores),
            "basin_count": len(targ_basins_set),
            "stores": targets_stores,
            "return_periods": return_periods_data,
            "uploads": target_uploads,
        },
        "realtime_forecasts": {
            "store_count": len(forecast_stores),
            "basin_count": len(fcst_basins_set),
            "stores": forecast_stores,
        },
        "assimilation": {
            "store_count": len(assimilation_stores),
            "basin_count": len(assim_basins_set),
            "stores": assimilation_stores,
            "da_states": da_state_files,
        },
        "models": {
            "runs_count": len(model_runs),
            "runs": model_runs,
            "hot_start_files": hot_start_files,
        },
    }


# Global Singleton Instance
_GLOBAL_PROFILE_MANAGER: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
  """Returns the global ProfileManager singleton.

  Set EARTHKIT_PROFILES_DIR to keep profiles somewhere other than data/profiles
  (the test suite uses this so it never creates or modifies real accounts).
  """
  global _GLOBAL_PROFILE_MANAGER
  if _GLOBAL_PROFILE_MANAGER is None:
    _GLOBAL_PROFILE_MANAGER = ProfileManager(
        profiles_dir=os.environ.get("EARTHKIT_PROFILES_DIR") or None
    )
  return _GLOBAL_PROFILE_MANAGER

