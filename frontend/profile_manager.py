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
"""User profile manager for OpenHydroNet Web.

Manages local user profiles, allowing users to log in or work as guests.
Persists active watersheds, Zarr archives, and settings per profile so data
populates upon logging in.
"""

import copy
import json
import logging
import os
import re
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from frontend.config import DATA_DIR
from utils.file_paths import (
    ATTRIBUTES_ZARR_NAME,
    STREAMFLOW_REALTIME_ZARR_NAME,
    STREAMFLOW_ZARR_NAME,
    TIMESERIES_ZARR_NAME,
)

logger = logging.getLogger(__name__)

PROFILES_DIR = DATA_DIR / 'profiles'

_MAX_ZARR_INSPECT_CACHE = 128
_ZARR_INSPECT_LOCK = threading.Lock()
_ZARR_INSPECT_CACHE: dict[
    str,
    tuple[tuple[str, int, int, int], dict[str, Any]],
] = {}


def find_zarr_dirs(base_d: Path) -> list[Path]:
    """Return sorted ``*.zarr`` directories under ``base_d``.

    Args:
        base_d: Base directory to scan.

    Returns:
        Sorted list of discovered ``*.zarr`` directory paths.
    """
    if not base_d.exists():
        return []
    found: list[Path] = []
    for root, dirs, _files in os.walk(base_d, followlinks=True):
        zarr_dirs = [d for d in dirs if d.endswith('.zarr')]
        found.extend(Path(root) / zd for zd in zarr_dirs)
        dirs[:] = [d for d in dirs if not d.endswith('.zarr')]
    return sorted(found)


_find_zarr_dirs = find_zarr_dirs


def zarr_cache_signature(zarr_path: Path) -> tuple[str, int, int, int]:
    """Build a fast mtime signature for ``zarr_path``.

    Args:
        zarr_path: Path to the Zarr store directory.

    Returns:
        4-tuple ``(resolved_path, zarr_mtime_ns, zmeta_mtime_ns, child_mtime)``.
    """
    resolved = str(zarr_path.resolve())
    if not zarr_path.is_dir():
        return (resolved, 0, 0, 0)
    zarr_mtime_ns = zarr_path.stat().st_mtime_ns
    zmeta_mtime_ns = 0
    for meta_name in ('.zmetadata', 'zarr.json', '.zgroup', '.zattrs'):
        mpath = zarr_path / meta_name
        if mpath.is_file():
            zmeta_mtime_ns = max(zmeta_mtime_ns, mpath.stat().st_mtime_ns)
    max_child_mtime_ns = max(
        (
            entry.stat().st_mtime_ns
            for entry in zarr_path.iterdir()
            if entry.exists()
        ),
        default=0,
    )
    return (resolved, zarr_mtime_ns, zmeta_mtime_ns, max_child_mtime_ns)


_zarr_cache_signature = zarr_cache_signature


def _sum_non_zarr_bytes(p_dir: Path, seen_zarrs: set[str]) -> int:
    """Sum file sizes under ``p_dir`` outside ``seen_zarrs`` directories."""
    if not p_dir.exists():
        return 0
    non_zarr_bytes = 0
    for root, dirs, files in os.walk(p_dir):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if str(root_path / d) not in seen_zarrs]
        for fname in files:
            fpath = root_path / fname
            if fpath.is_file():
                non_zarr_bytes += fpath.stat().st_size
    return non_zarr_bytes


LEGACY_PROFILE_SUBDIRS = (
    'polygons',
    'attributes',
    'historical',
    'forecast',
    'archives',
)
DYNAMICS_PRODUCTS = ('ERA5_LAND', 'CPC', 'IMERG', 'HRES', 'GRAPHCAST')
V2_PROFILE_TOP_DIRS = (
    'catchments',
    'statics',
    'dynamics',
    'realtime',
    'targets',
    'assimilation',
    'models',
    'forecasts',
    'maas',
    'jobs',
)

_HIGHLIGHT_ATTRIBUTE_COLS = (
    'ele_mt_sav',
    'slp_dg_sav',
    'pre_mm_syr',
    'run_mm_syr',
    'for_pc_use',
    'snd_pc_sav',
    'cly_pc_sav',
    'ari_ix_sav',
    'tmp_dc_syr',
    'inu_pc_slt',
    'lka_pc_sse',
    'gwt_cm_sav',
)
_MIN_KEY_COLS = 5
_MAX_KEY_COLS = 6
_TWO_BASINS = 2


def _sanitize_username(username: str | None) -> str:
    """Sanitize a username for safe filesystem directory naming.

    Args:
        username: Raw username string.

    Returns:
        Sanitized lowercase directory name, defaulting to ``'guest'``.
    """
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', (username or '').strip().lower())
    clean = clean.lstrip('.')
    return clean or 'guest'


def _extract_basin_id(feature: dict[str, Any], idx: int = 0) -> str:
    """Extract a canonical basin identifier from a GeoJSON Feature.

    Args:
        feature: GeoJSON Feature dictionary.
        idx: Fallback index if no explicit identifier is present.

    Returns:
        Basin identifier string.
    """
    raw_props = feature.get('properties')
    props = raw_props if isinstance(raw_props, dict) else {}
    for candidate in (
        feature.get('id'),
        props.get('id'),
        props.get('basin_id'),
        props.get('HYBAS_ID'),
        props.get('catchment_id'),
        props.get('hybas_id'),
        props.get('gauge_id'),
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return f'basin_{idx}'


def _partition_basin_ids(
    basin_ids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Partition basin IDs into disjoint train, validation, and test splits.

    Args:
        basin_ids: Ordered list of unique basin IDs.

    Returns:
        Tuple of ``(train_ids, val_ids, test_ids)`` with zero overlap.
    """
    n_basins = len(basin_ids)
    if n_basins == 0:
        return [], [], []
    if n_basins == 1:
        return [basin_ids[0]], [], []
    if n_basins == _TWO_BASINS:
        return [basin_ids[0]], [], [basin_ids[1]]
    n_test = max(1, n_basins // 5)
    n_val = max(1, (n_basins - n_test) // 4)
    n_train = n_basins - n_val - n_test
    return (
        basin_ids[:n_train],
        basin_ids[n_train : n_train + n_val],
        basin_ids[n_train + n_val :],
    )


def _is_valid_zarr_dir(zarr_path: Path) -> bool:
    """Return True if ``zarr_path`` contains Zarr v2 or v3 root metadata."""
    if not zarr_path.is_dir():
        return False
    return any(
        (zarr_path / marker).exists()
        for marker in ('.zmetadata', '.zgroup', 'zarr.json')
    )


def _normalize_loaded_features(
    data: dict[str, Any] | list[Any],
) -> list[dict[str, Any]] | None:
    """Normalize loaded JSON data into a list of GeoJSON Feature dicts."""
    raw_feats: list[Any] | None = None
    if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
        raw_feats = data.get('features', [])
    elif isinstance(data, list):
        raw_feats = data
    if not isinstance(raw_feats, list):
        return None
    for idx, feat in enumerate(raw_feats):
        if isinstance(feat, dict):
            bid = _extract_basin_id(feat, idx)
            props = feat.get('properties')
            if not isinstance(props, dict):
                props = {}
                feat['properties'] = props
            props.setdefault('catchment_id', bid)
    return raw_feats


def _format_hist_summary(matching_hist: list[dict[str, Any]]) -> str | None:
    """Format a short summary string for a basin's historical Zarr store."""
    if not matching_hist:
        return None
    h0 = matching_hist[0]
    src = h0.get('source_dataset') or h0.get('filename')
    dates = (
        f'{h0["start_date"]} -> {h0["end_date"]}'
        if h0.get('start_date') and h0.get('end_date')
        else f'{h0.get("n_timesteps", 0)} steps'
    )
    n_vars = len(h0.get('dynamic_variables', []))
    return f'{src} ({dates}, {n_vars} vars)'


def _format_fcst_summary(matching_fcst: list[dict[str, Any]]) -> str | None:
    """Format a short summary string for a basin's forecast Zarr store."""
    if not matching_fcst:
        return None
    f0 = matching_fcst[0]
    iss = f0.get('issue_date') or 'Latest'
    lead = f0.get('lead_time_days') or f0.get('n_timesteps') or 0
    n_vars = len(f0.get('dynamic_variables', []))
    return f'Issued {iss} ({lead}d horizon, {n_vars} vars)'


class ProfileManager:
    """Manage user profiles, state persistence, and directory sandboxing.

    Accounts:
      * ``guest`` always exists. Its data is kept while the server runs and is
        erased (``clear_guest_data``) when the server starts.
      * User accounts are created explicitly (``create_account``) and are
        identified by ``profiles/<username>/profile.json``. Their data persists
        across restarts until the account is deleted (``delete_account``).
    """

    def __init__(self, profiles_dir: Path | str | None = None) -> None:
        """Initialize the profile manager rooted at ``profiles_dir``.

        Args:
            profiles_dir: Optional directory for storing user profiles.
        """
        self.profiles_dir = Path(profiles_dir or PROFILES_DIR)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._active_username: str = 'guest'

    @property
    def active_username(self) -> str:
        """Return the currently active profile username."""
        return self._active_username

    def clear_guest_data(self) -> None:
        """Erase everything stored for guest and recreate an empty workspace."""
        guest_dir = self.profiles_dir / 'guest'
        if guest_dir.exists():
            shutil.rmtree(guest_dir)
        self.get_profile_dir('guest')
        self.save_watersheds([], username='guest')

    def account_exists(self, username: str | None) -> bool:
        """Return True for guest or any created user account."""
        user = _sanitize_username(username)
        if user == 'guest':
            return True
        return (self.profiles_dir / user / 'profile.json').is_file()

    def create_account(
        self,
        username: str,
        email: str | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a new user account and make it the active profile.

        Args:
            username: Desired account username.
            email: Optional contact email address.
            display_name: Optional human-readable display name.

        Returns:
            Newly created profile metadata dictionary.

        Raises:
            ValueError: If the username is empty or reserved (``'guest'``).
            FileExistsError: If an account with this username already exists.
        """
        if not (username or '').strip():
            raise ValueError('Please enter a username.')
        if username.strip().lower() == 'guest':
            raise ValueError(
                "'guest' is reserved for the temporary guest session.",
            )
        clean_user = _sanitize_username(username)
        if clean_user == 'guest' or not re.search(r'[a-z0-9]', clean_user):
            raise ValueError(
                'Please choose a username containing letters or numbers.',
            )
        if self.account_exists(clean_user):
            raise FileExistsError(
                f"An account named '{clean_user}' already exists."
                ' Log in instead.',
            )
        info = self.login_profile(
            clean_user,
            email=email,
            display_name=display_name,
        )
        logger.info("Created account '%s'", clean_user)
        return info

    def delete_account(self, username: str) -> str:
        """Permanently delete a user account and all of its stored files.

        Args:
            username: The account to delete.

        Returns:
            The sanitized username of the deleted account.

        Raises:
            ValueError: If the username is empty, reserved, or outside
                ``profiles_dir``.
            LookupError: If no account with this username exists.
            OSError: If the account directory could not be deleted.
        """
        if not (username or '').strip():
            raise ValueError('Please choose an account to remove.')
        clean_user = _sanitize_username(username)
        if clean_user == 'guest':
            raise ValueError(
                "The guest session can't be removed. Its data is erased when"
                ' the server restarts.',
            )
        if not self.account_exists(clean_user):
            raise LookupError(f"No account named '{clean_user}'.")
        account_dir = self.profiles_dir / clean_user
        if (
            account_dir.is_symlink()
            or account_dir.resolve().parent != self.profiles_dir.resolve()
        ):
            raise ValueError(
                f"'{clean_user}' is not stored in the profiles folder.",
            )
        if self._active_username == clean_user:
            self._active_username = 'guest'
        if account_dir.exists():
            shutil.rmtree(account_dir)
        if self.account_exists(clean_user):
            raise OSError(f"Couldn't delete {account_dir}.")
        logger.info("Deleted account '%s'", clean_user)
        return clean_user

    def get_profile_dir(self, username: str | None = None) -> Path:
        """Return and provision the filesystem directory for a profile.

        Args:
            username: Target username, or active profile if None.

        Returns:
            Path to the user's profile directory.
        """
        user = _sanitize_username(username or self._active_username)
        p_dir = self.profiles_dir / user
        p_dir.mkdir(parents=True, exist_ok=True)
        for sub in LEGACY_PROFILE_SUBDIRS:
            (p_dir / sub).mkdir(parents=True, exist_ok=True)

        # 13-category OpenHydroNet directory structure
        for sub in ('shapes', 'zonal_weights', 'basin_lists'):
            (p_dir / 'catchments' / sub).mkdir(parents=True, exist_ok=True)
        (p_dir / 'statics' / 'caravan_csv').mkdir(parents=True, exist_ok=True)
        for prod in DYNAMICS_PRODUCTS:
            (p_dir / 'dynamics' / prod).mkdir(parents=True, exist_ok=True)
            (p_dir / 'realtime' / 'dynamics' / prod).mkdir(
                parents=True,
                exist_ok=True,
            )
        (p_dir / 'realtime' / 'scenarios').mkdir(parents=True, exist_ok=True)
        (p_dir / 'targets' / 'uploads').mkdir(parents=True, exist_ok=True)
        (p_dir / 'assimilation' / 'uploads').mkdir(parents=True, exist_ok=True)
        (p_dir / 'assimilation' / 'da_states').mkdir(
            parents=True,
            exist_ok=True,
        )
        (p_dir / 'models' / 'hot_start').mkdir(parents=True, exist_ok=True)
        (p_dir / 'models' / 'runs').mkdir(parents=True, exist_ok=True)
        (p_dir / 'forecasts').mkdir(parents=True, exist_ok=True)
        (p_dir / 'maas' / 'snapshots').mkdir(parents=True, exist_ok=True)
        (p_dir / 'jobs').mkdir(parents=True, exist_ok=True)
        return p_dir

    # -------------------------------------------------------------------------
    # Legacy Directory Accessors
    # -------------------------------------------------------------------------

    def get_polygons_dir(self, username: str | None = None) -> Path:
        """Return the polygons directory for the specified or active profile."""
        user = _sanitize_username(username or self._active_username)
        d = self.get_profile_dir(user) / 'polygons'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_attributes_dir(self, username: str | None = None) -> Path:
        """Return the attributes directory for the specified profile."""
        user = _sanitize_username(username or self._active_username)
        d = self.get_profile_dir(user) / 'attributes'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_historical_dir(self, username: str | None = None) -> Path:
        """Return the historical weather directory for a profile."""
        user = _sanitize_username(username or self._active_username)
        d = self.get_profile_dir(user) / 'historical'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_forecast_dir(self, username: str | None = None) -> Path:
        """Return the forecast weather directory for a profile."""
        user = _sanitize_username(username or self._active_username)
        d = self.get_profile_dir(user) / 'forecast'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_realtime_dir(self, username: str | None = None) -> Path:
        """Return the forecast directory (backwards-compatibility alias)."""
        return self.get_forecast_dir(username)

    def get_archives_dir(self, username: str | None = None) -> Path:
        """Return the primary historical archives directory."""
        return self.get_historical_dir(username)

    # -------------------------------------------------------------------------
    # 13-Category OpenHydroNet Directory Accessors
    # -------------------------------------------------------------------------

    def get_catchments_dir(self, username: str | None = None) -> Path:
        """Return the catchments directory for a profile."""
        d = self.get_profile_dir(username) / 'catchments'
        for sub in ('shapes', 'zonal_weights', 'basin_lists'):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def get_catchment_shapes_dir(self, username: str | None = None) -> Path:
        """Return the ``catchments/shapes`` directory for per-basin GeoJSONs."""
        d = self.get_catchments_dir(username) / 'shapes'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_zonal_weights_dir(self, username: str | None = None) -> Path:
        """Return the ``catchments/zonal_weights`` directory."""
        d = self.get_catchments_dir(username) / 'zonal_weights'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_basin_lists_dir(self, username: str | None = None) -> Path:
        """Return the ``catchments/basin_lists`` directory for basin lists."""
        d = self.get_catchments_dir(username) / 'basin_lists'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_statics_dir(self, username: str | None = None) -> Path:
        """Return the statics directory for Caravan static attributes."""
        d = self.get_profile_dir(username) / 'statics'
        (d / 'caravan_csv').mkdir(parents=True, exist_ok=True)
        return d

    def get_caravan_attributes_dir(self, username: str | None = None) -> Path:
        """Return the ``statics/caravan_csv`` directory."""
        d = self.get_statics_dir(username) / 'caravan_csv'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_statics_zarr_path(self, username: str | None = None) -> Path:
        """Return the consolidated static attributes Zarr store path."""
        return self.get_statics_dir(username) / ATTRIBUTES_ZARR_NAME

    def get_dynamics_dir(
        self,
        username: str | None = None,
        product: str | None = None,
    ) -> Path:
        """Return the historical dynamics directory, optionally by product."""
        d = self.get_profile_dir(username) / 'dynamics'
        d.mkdir(parents=True, exist_ok=True)
        if product:
            prod_norm = product.strip().upper().replace('-', '_')
            if prod_norm == 'ERA5LAND':
                prod_norm = 'ERA5_LAND'
            prod_dir = d / prod_norm
            prod_dir.mkdir(parents=True, exist_ok=True)
            return prod_dir
        return d

    def get_dynamics_zarr_path(
        self,
        username: str | None = None,
        product: str = 'ERA5_LAND',
    ) -> Path:
        """Return the historical dynamics Zarr path for a product."""
        return (
            self.get_dynamics_dir(username, product.upper())
            / TIMESERIES_ZARR_NAME
        )

    def get_realtime_dir_v2(self, username: str | None = None) -> Path:
        """Return the OpenHydroNet v2 ``realtime/`` directory."""
        d = self.get_profile_dir(username) / 'realtime'
        (d / 'dynamics').mkdir(parents=True, exist_ok=True)
        (d / 'scenarios').mkdir(parents=True, exist_ok=True)
        return d

    def get_realtime_dynamics_dir(self, username: str | None = None) -> Path:
        """Return the ``realtime/dynamics`` directory."""
        d = self.get_realtime_dir_v2(username) / 'dynamics'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_realtime_dynamics_zarr_path(
        self,
        username: str | None = None,
        product: str = 'HRES',
    ) -> Path:
        """Return the real-time operational dynamics Zarr path for a product."""
        prod_norm = (product or 'HRES').strip().upper().replace('-', '_')
        if prod_norm == 'ERA5LAND':
            prod_norm = 'ERA5_LAND'
        prod_dir = self.get_realtime_dir_v2(username) / 'dynamics' / prod_norm
        prod_dir.mkdir(parents=True, exist_ok=True)
        return prod_dir / TIMESERIES_ZARR_NAME

    def get_targets_dir(self, username: str | None = None) -> Path:
        """Return the targets directory for historical streamflow data."""
        d = self.get_profile_dir(username) / 'targets'
        (d / 'uploads').mkdir(parents=True, exist_ok=True)
        return d

    def get_targets_zarr_path(self, username: str | None = None) -> Path:
        """Return the historical target streamflow Zarr store path."""
        return self.get_targets_dir(username) / STREAMFLOW_ZARR_NAME

    def get_return_periods_path(self, username: str | None = None) -> Path:
        """Return the return period flood thresholds JSON path."""
        return self.get_targets_dir(username) / 'return_periods.json'

    def get_assimilation_dir(self, username: str | None = None) -> Path:
        """Return the assimilation directory for real-time DA states."""
        d = self.get_profile_dir(username) / 'assimilation'
        (d / 'uploads').mkdir(parents=True, exist_ok=True)
        (d / 'da_states').mkdir(parents=True, exist_ok=True)
        return d

    def get_assimilation_zarr_path(self, username: str | None = None) -> Path:
        """Return the real-time assimilation streamflow Zarr store path."""
        return (
            self.get_assimilation_dir(username) / STREAMFLOW_REALTIME_ZARR_NAME
        )

    def get_da_states_dir(self, username: str | None = None) -> Path:
        """Return the ``assimilation/da_states`` directory."""
        d = self.get_assimilation_dir(username) / 'da_states'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_models_dir(self, username: str | None = None) -> Path:
        """Return the models directory for model runs and hot-start states."""
        d = self.get_profile_dir(username) / 'models'
        (d / 'hot_start').mkdir(parents=True, exist_ok=True)
        (d / 'runs').mkdir(parents=True, exist_ok=True)
        return d

    def get_hot_start_dir(self, username: str | None = None) -> Path:
        """Return the ``models/hot_start`` directory for LSTM states."""
        d = self.get_models_dir(username) / 'hot_start'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_model_states_dir(self, username: str | None = None) -> Path:
        """Return the ``models/hot_start`` directory (alias)."""
        return self.get_hot_start_dir(username)

    def get_model_runs_dir(self, username: str | None = None) -> Path:
        """Return the ``models/runs`` directory for OpenHydroNet run folders."""
        d = self.get_models_dir(username) / 'runs'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_forecasts_dir(
        self,
        username: str | None = None,
        basin_id: str | None = None,
    ) -> Path:
        """Return the forecasts directory, optionally scoped to ``basin_id``."""
        d = self.get_profile_dir(username) / 'forecasts'
        d.mkdir(parents=True, exist_ok=True)
        if basin_id:
            b_dir = d / str(basin_id).strip()
            (b_dir / 'history').mkdir(parents=True, exist_ok=True)
            return b_dir
        return d

    def get_jobs_dir(self, username: str | None = None) -> Path:
        """Return the jobs directory for persistent background job states."""
        d = self.get_profile_dir(username) / 'jobs'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_profile_info(self, username: str | None = None) -> dict[str, Any]:
        """Read profile metadata from ``profile.json`` or return defaults.

        Args:
            username: Target username, or active profile if None.

        Returns:
            Profile metadata dictionary.
        """
        user = _sanitize_username(username or self._active_username)
        p_dir = self.get_profile_dir(user)
        meta_file = p_dir / 'profile.json'

        if meta_file.is_file() and meta_file.stat().st_size > 0:
            return json.loads(meta_file.read_text(encoding='utf-8'))

        now_iso = datetime.now(UTC).isoformat()
        return {
            'username': user,
            'display_name': user.title() if user != 'guest' else 'Guest User',
            'email': '',
            'created_at': now_iso,
            'last_login': now_iso,
            'is_guest': user == 'guest',
            'settings': {
                'default_weather_source': 'cpc',
                'default_hydro_dataset': 'hydroatlas',
            },
        }

    def login_profile(
        self,
        username: str,
        email: str | None = None,
        display_name: str | None = None,
        settings: dict[str, Any] | None = None,
        *,
        create_if_missing: bool = True,
    ) -> dict[str, Any]:
        """Log in to a user profile and set it as active.

        Args:
            username: Account to log in to.
            email: Optional email to store on the profile.
            display_name: Optional display name to store on the profile.
            settings: Optional settings to merge into the profile.
            create_if_missing: If False, raise LookupError instead of creating
                an account that does not exist yet.

        Returns:
            The active profile metadata dictionary.

        Raises:
            LookupError: If ``create_if_missing`` is False and the account does
                not exist.
        """
        clean_user = _sanitize_username(username)
        if not create_if_missing and not self.account_exists(clean_user):
            raise LookupError(
                f"No account named '{clean_user}'."
                ' Use Create Account to make one.',
            )
        self._active_username = clean_user
        p_dir = self.get_profile_dir(clean_user)
        meta_file = p_dir / 'profile.json'

        info = self.get_profile_info(clean_user)
        info['last_login'] = datetime.now(UTC).isoformat()
        info['is_guest'] = clean_user == 'guest'

        if email:
            info['email'] = email.strip()
        if display_name:
            info['display_name'] = display_name.strip()
        if settings:
            info['settings'] = {**info.get('settings', {}), **settings}

        meta_file.write_text(json.dumps(info, indent=2), encoding='utf-8')
        logger.info("Active profile set to '%s'", clean_user)
        return info

    def logout_profile(self) -> dict[str, Any]:
        """Switch back to the guest session."""
        self._active_username = 'guest'
        return self.get_profile_info('guest')

    def list_profiles(self) -> list[dict[str, Any]]:
        """List the guest session plus every user account stored locally."""
        profiles: list[dict[str, Any]] = []
        if not self.profiles_dir.exists():
            return profiles
        self.get_profile_dir('guest')

        for entry in self.profiles_dir.iterdir():
            if not entry.is_dir():
                continue
            if entry.name != 'guest' and not (entry / 'profile.json').is_file():
                continue
            info = self.get_profile_info(entry.name)
            watersheds = self.load_watersheds(entry.name)
            archives = (
                list((entry / 'historical').glob('*.zarr'))
                + list((entry / 'forecast').glob('*.zarr'))
                + list((entry / 'archives').glob('*.zarr'))
            )
            info['watersheds_count'] = len(watersheds)
            info['archives_count'] = len({a.name for a in archives})
            info['is_active'] = entry.name == self._active_username
            profiles.append(info)

        profiles.sort(key=lambda x: x.get('last_login', ''), reverse=True)
        return profiles

    def load_watersheds(
        self,
        username: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load saved GeoJSON watershed features for a profile.

        Args:
            username: Target username, or active profile if None.

        Returns:
            List of GeoJSON Feature dictionaries.
        """
        user = _sanitize_username(username or self._active_username)
        poly_dir = self.get_polygons_dir(user)
        ws_file = poly_dir / 'watersheds.json'

        p_dir = self.get_profile_dir(user)
        root_ws_file = p_dir / 'watersheds.json'
        if not ws_file.exists() and root_ws_file.exists():
            root_ws_file.replace(ws_file)
        elif root_ws_file.exists():
            root_ws_file.unlink(missing_ok=True)

        candidates = (
            ws_file,
            self.get_catchments_dir(user) / 'watersheds.geojson',
        )
        for candidate in candidates:
            if not candidate.is_file() or candidate.stat().st_size == 0:
                continue
            data = json.loads(candidate.read_text(encoding='utf-8'))
            normalized = _normalize_loaded_features(data)
            if normalized is not None:
                return normalized
        return []

    def sync_catchment_artifacts(
        self,
        features: list[dict[str, Any]],
        username: str | None = None,
    ) -> list[str]:
        """Synchronize ``catchments/`` GeoJSONs and disjoint basin split lists.

        Args:
            features: List of GeoJSON Feature dictionaries.
            username: Target username, or active profile if None.

        Returns:
            Ordered list of unique basin IDs written.
        """
        user = _sanitize_username(username or self._active_username)
        catchments_dir = self.get_catchments_dir(user)
        shapes_dir = self.get_catchment_shapes_dir(user)
        basin_lists_dir = self.get_basin_lists_dir(user)

        fc = {
            'type': 'FeatureCollection',
            'count': len(features),
            'updated_at': datetime.now(UTC).isoformat(),
            'features': features,
        }
        (catchments_dir / 'watersheds.geojson').write_text(
            json.dumps(fc, indent=2),
            encoding='utf-8',
        )

        basin_ids: list[str] = []
        expected_shape_files: set[str] = set()
        for idx, feat in enumerate(features):
            if not isinstance(feat, dict):
                continue
            basin_id = _extract_basin_id(feat, idx)
            if basin_id not in basin_ids:
                basin_ids.append(basin_id)

            raw_props = feat.get('properties')
            props = dict(raw_props) if isinstance(raw_props, dict) else {}
            props.setdefault('id', basin_id)
            props.setdefault('basin_id', basin_id)
            props.setdefault('catchment_id', basin_id)

            single_feat = {
                **feat,
                'type': 'Feature',
                'id': basin_id,
                'properties': props,
                'geometry': feat.get('geometry'),
            }
            shape_file = shapes_dir / f'{basin_id}.geojson'
            expected_shape_files.add(shape_file.name)
            shape_file.write_text(
                json.dumps(single_feat, indent=2),
                encoding='utf-8',
            )

        for existing_file in shapes_dir.glob('*.geojson'):
            if existing_file.name not in expected_shape_files:
                existing_file.unlink(missing_ok=True)

        train_ids, val_ids, test_ids = _partition_basin_ids(basin_ids)
        splits = {
            'all_basins.txt': basin_ids,
            'basins.txt': basin_ids,
            'train_basins.txt': train_ids,
            'val_basins.txt': val_ids,
            'test_basins.txt': test_ids,
        }
        for list_name, split_ids in splits.items():
            content = ('\n'.join(split_ids) + '\n') if split_ids else ''
            (basin_lists_dir / list_name).write_text(content, encoding='utf-8')

        return basin_ids

    def save_watersheds(
        self,
        features: list[dict[str, Any]],
        username: str | None = None,
    ) -> bool:
        """Persist GeoJSON watershed features and sync catchment artifacts.

        Args:
            features: List of GeoJSON Feature dictionaries.
            username: Target username, or active profile if None.

        Returns:
            True once saved.
        """
        user = _sanitize_username(username or self._active_username)
        poly_dir = self.get_polygons_dir(user)
        poly_dir.mkdir(parents=True, exist_ok=True)
        ws_file = poly_dir / 'watersheds.json'

        p_dir = self.get_profile_dir(user)
        root_ws_file = p_dir / 'watersheds.json'
        if root_ws_file.exists():
            root_ws_file.unlink(missing_ok=True)

        fc = {
            'type': 'FeatureCollection',
            'count': len(features),
            'updated_at': datetime.now(UTC).isoformat(),
            'features': features,
        }
        ws_file.write_text(json.dumps(fc, indent=2), encoding='utf-8')
        self.sync_catchment_artifacts(features, username=user)
        return True

    def save_settings(
        self,
        settings: dict[str, Any],
        username: str | None = None,
    ) -> dict[str, Any]:
        """Update user preferences and settings for a profile.

        Args:
            settings: Settings key-value dictionary to merge.
            username: Target username, or active profile if None.

        Returns:
            Updated profile metadata dictionary.
        """
        user = _sanitize_username(username or self._active_username)
        info = self.get_profile_info(user)
        info['settings'] = {**info.get('settings', {}), **settings}
        p_dir = self.get_profile_dir(user)
        meta_file = p_dir / 'profile.json'
        meta_file.write_text(json.dumps(info, indent=2), encoding='utf-8')
        return info

    def inspect_zarr_store(
        self,
        zarr_path: Path,
        profile_dir: Path,
        default_category: str = 'historical',
    ) -> dict[str, Any]:
        """Inspect a Zarr store on disk and return metadata.

        Args:
            zarr_path: Path to the Zarr store directory.
            profile_dir: Root directory of the active user profile.
            default_category: Category label assigned to the returned dict.

        Returns:
            Dictionary describing store dimensions, variables, basins, and size.
        """
        return self._inspect_zarr_store(
            zarr_path,
            profile_dir,
            default_category=default_category,
        )

    @staticmethod
    def _populate_zarr_ds_meta(
        ds: xr.Dataset,
        zarr_path: Path,
        meta: dict[str, Any],
    ) -> None:
        """Populate dataset dimensions, coords, and attributes onto ``meta``."""
        meta['dimensions'] = {str(k): int(v) for k, v in ds.sizes.items()}
        b_key = next(
            (
                k
                for k in ('basin', 'basin_id', 'catchment_id', 'gauge_id')
                if k in ds.coords
            ),
            None,
        )
        if b_key is not None:
            meta['basins'] = [str(b) for b in ds.coords[b_key].to_numpy()]
        meta['total_basins'] = len(meta['basins'])

        t_key = next(
            (
                k
                for k in ('date', 'time', 'valid_time', 'issue_time')
                if k in ds.coords and ds.coords[k].ndim >= 1
            ),
            None,
        )
        if t_key is not None and ds.coords[t_key].size > 0:
            t_coord = ds.coords[t_key]
            meta['n_timesteps'] = int(t_coord.size)
            t_vals = t_coord.to_numpy()
            meta['start_date'] = str(t_vals[0])[:10]
            meta['end_date'] = str(t_vals[-1])[:10]

        if 'lead_time' in ds.sizes:
            meta['lead_time_days'] = int(ds.sizes['lead_time'])

        all_vars = sorted(str(v) for v in ds.data_vars)
        static_vars = [
            v
            for v in all_vars
            if v.startswith('static_')
            or v in ('basin_area', 'area_km2', 'latitude', 'longitude')
        ]
        meta['all_variables'] = all_vars
        meta['dynamic_variables'] = [
            v for v in all_vars if v not in static_vars
        ]
        meta['static_variables'] = static_vars

        attrs = dict(ds.attrs or {})
        meta['title'] = attrs.get('title')
        meta['source_dataset'] = (
            attrs.get('source_dataset')
            or attrs.get('models')
            or attrs.get('source_platform')
            or (
                zarr_path.parent.name
                if zarr_path.parent.name in DYNAMICS_PRODUCTS
                else None
            )
        )
        meta['temporal_resolution'] = attrs.get('temporal_resolution', '1D')
        meta['issue_date'] = (
            attrs.get('issue_date')
            or attrs.get('initialization_time')
            or meta['start_date']
        )

    def _inspect_zarr_store(
        self,
        zarr_path: Path,
        profile_dir: Path,
        default_category: str = 'historical',
    ) -> dict[str, Any]:
        """Inspect a Zarr store on disk and return metadata."""
        rel_path = (
            str(zarr_path.relative_to(profile_dir))
            if zarr_path.is_relative_to(profile_dir)
            else zarr_path.name
        )
        cache_sig = zarr_cache_signature(zarr_path)
        resolved_key = cache_sig[0]
        if cache_sig[1] > 0:
            with _ZARR_INSPECT_LOCK:
                entry = _ZARR_INSPECT_CACHE.get(resolved_key)
                if entry is not None and entry[0] == cache_sig:
                    cached = copy.deepcopy(entry[1])
                    cached['rel_path'] = rel_path
                    cached['path'] = str(zarr_path)
                    cached['category'] = default_category
                    return cached

        is_valid = _is_valid_zarr_dir(zarr_path)
        size_bytes = (
            sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
            if zarr_path.is_dir()
            else 0
        )
        mtime_iso = (
            datetime.fromtimestamp(
                zarr_path.stat().st_mtime,
                tz=UTC,
            ).isoformat()
            if zarr_path.exists()
            else ''
        )

        meta: dict[str, Any] = {
            'filename': zarr_path.name,
            'rel_path': rel_path,
            'path': str(zarr_path),
            'category': default_category,
            'size_bytes': size_bytes,
            'size_mb': round(size_bytes / (1024 * 1024), 3),
            'modified_at': mtime_iso,
            'basins': [],
            'total_basins': 0,
            'n_timesteps': 0,
            'lead_time_days': 0,
            'start_date': None,
            'end_date': None,
            'dynamic_variables': [],
            'static_variables': [],
            'all_variables': [],
            'dimensions': {},
            'source_dataset': None,
            'temporal_resolution': None,
            'title': None,
            'issue_date': None,
        }

        if not is_valid:
            return meta

        with xr.open_zarr(str(zarr_path), decode_timedelta=False) as ds:
            self._populate_zarr_ds_meta(ds, zarr_path, meta)

        if cache_sig[1] > 0:
            with _ZARR_INSPECT_LOCK:
                if (
                    resolved_key not in _ZARR_INSPECT_CACHE
                    and len(_ZARR_INSPECT_CACHE) >= _MAX_ZARR_INSPECT_CACHE
                ):
                    _ZARR_INSPECT_CACHE.pop(next(iter(_ZARR_INSPECT_CACHE)))
                _ZARR_INSPECT_CACHE[resolved_key] = (
                    cache_sig,
                    copy.deepcopy(meta),
                )
        return meta

    def _summarize_polygons(
        self,
        user: str,
        p_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """Collect catchment summaries and polygon files for ``user``."""
        features = self.load_watersheds(user)
        catch_ws_path = self.get_catchments_dir(user) / 'watersheds.geojson'
        if features and not catch_ws_path.exists():
            self.sync_catchment_artifacts(features, username=user)

        catchments_list: list[dict[str, Any]] = []
        total_area_km2 = 0.0
        for idx, feat in enumerate(features):
            if not isinstance(feat, dict):
                continue
            cid = _extract_basin_id(feat, idx)
            raw_props = feat.get('properties')
            props = raw_props if isinstance(raw_props, dict) else {}
            area = float(
                props.get('area_km2')
                or props.get('SUB_AREA')
                or props.get('UP_AREA')
                or 0.0,
            )
            total_area_km2 += area
            source = props.get('source') or (
                'delineated'
                if props.get('delineation_method') != 'User GeoJSON Polygon'
                else 'uploaded'
            )
            raw_outlet = props.get('outlet')
            outlet = raw_outlet if isinstance(raw_outlet, dict) else {}
            catchments_list.append(
                {
                    'catchment_id': cid,
                    'name': props.get('name')
                    or props.get('station_name')
                    or cid,
                    'area_km2': round(area, 2),
                    'source': source,
                    'dataset': (
                        props.get('dataset')
                        or props.get('dem_id')
                        or 'hydroatlas'
                    ),
                    'dem_id': props.get('dem_id') or 'hydrosheds_90m',
                    'delineation_method': props.get('delineation_method')
                    or (
                        'Auto-Delineated from DEM'
                        if source == 'delineated'
                        else 'Uploaded GeoJSON'
                    ),
                    'stream_order': props.get('stream_order'),
                    'outlet_lat': outlet.get('latitude'),
                    'outlet_lon': outlet.get('longitude'),
                    'reach_id': outlet.get('reach_id'),
                    'geometry_type': (feat.get('geometry') or {}).get(
                        'type',
                        'Polygon',
                    ),
                }
            )

        polygon_files: list[dict[str, Any]] = []
        seen_poly_files: set[str] = set()
        exts = ('*.json', '*.geojson', '*.shp', '*.gpkg', '*.npz', '*.txt')
        poly_dirs = (
            self.get_polygons_dir(user),
            self.get_catchments_dir(user),
        )
        for base_d in poly_dirs:
            if not base_d.exists():
                continue
            for ext in exts:
                for pf in sorted(base_d.rglob(ext)):
                    if not pf.is_file() or str(pf) in seen_poly_files:
                        continue
                    seen_poly_files.add(str(pf))
                    sz = pf.stat().st_size
                    if sz == 0 and pf.suffix == '.txt':
                        continue
                    polygon_files.append(
                        {
                            'filename': pf.name,
                            'rel_path': str(pf.relative_to(p_dir)),
                            'path': str(pf),
                            'size_kb': round(sz / 1024.0, 2),
                            'size_mb': round(sz / (1024.0 * 1024.0), 4),
                        }
                    )
        return catchments_list, polygon_files, total_area_km2

    def _parse_attributes_csv_preview(
        self,
        csv_file: Path,
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Extract basin IDs, variable names, and highlights from a CSV file."""
        df = pd.read_csv(csv_file)
        if df.empty:
            return [], [], []
        id_col = next(
            (
                c
                for c in ('gauge_id', 'basin_id', 'catchment_id', 'id')
                if c in df.columns
            ),
            df.columns[0],
        )
        attr_basins = [str(v) for v in df[id_col].dropna().unique()]
        attr_variables = [str(c) for c in df.columns if c != id_col]
        key_cols = [c for c in _HIGHLIGHT_ATTRIBUTE_COLS if c in df.columns]
        if len(key_cols) < _MIN_KEY_COLS:
            for c in attr_variables:
                if c not in key_cols and len(key_cols) < _MAX_KEY_COLS:
                    key_cols.append(c)
        previews: list[dict[str, Any]] = []
        for _, row in df.head(50).iterrows():
            b_id = str(row[id_col])
            metrics: dict[str, Any] = {}
            for kc in key_cols[:_MAX_KEY_COLS]:
                val = row[kc]
                if pd.notna(val):
                    is_num = isinstance(
                        val,
                        (int, float, np.number),
                    ) and not isinstance(val, bool)
                    metrics[kc] = round(float(val), 2) if is_num else str(val)
            previews.append(
                {
                    'basin_id': b_id,
                    'variable_count': int(
                        row.drop(labels=[id_col]).notna().sum()
                    ),
                    'highlights': metrics,
                }
            )
        return attr_basins, attr_variables, previews

    def _summarize_attributes(
        self,
        user: str,
        p_dir: Path,
    ) -> dict[str, Any]:
        """Collect static attribute files, basins, variables, and previews."""
        attr_files: list[dict[str, Any]] = []
        attr_basins: list[str] = []
        attr_variables: list[str] = []
        attr_basin_previews: list[dict[str, Any]] = []
        seen_attr_files: set[str] = set()
        attr_dirs = (
            self.get_attributes_dir(user),
            self.get_statics_dir(user),
        )

        for base_d in attr_dirs:
            for ext in ('*.csv', '*.json', '*.parquet', '*.nc'):
                for af in sorted(base_d.rglob(ext)):
                    if not af.is_file() or str(af) in seen_attr_files:
                        continue
                    seen_attr_files.add(str(af))
                    sz = af.stat().st_size
                    attr_files.append(
                        {
                            'filename': af.name,
                            'rel_path': str(af.relative_to(p_dir)),
                            'path': str(af),
                            'size_kb': round(sz / 1024.0, 2),
                            'size_mb': round(sz / (1024.0 * 1024.0), 3),
                            'download_url': (
                                f'/api/attributes/csv?username={user}'
                                if af.suffix == '.csv'
                                else None
                            ),
                        }
                    )
                    if (
                        af.suffix == '.csv'
                        and not attr_basin_previews
                        and sz > 0
                    ):
                        (
                            attr_basins,
                            attr_variables,
                            attr_basin_previews,
                        ) = self._parse_attributes_csv_preview(af)

        statics_zarr = self.get_statics_zarr_path(user)
        if statics_zarr.exists():
            z_info = self._inspect_zarr_store(
                statics_zarr,
                p_dir,
                'static_attributes',
            )
            attr_files.append(
                {
                    'filename': statics_zarr.name,
                    'rel_path': z_info['rel_path'],
                    'path': str(statics_zarr),
                    'size_kb': round(z_info['size_bytes'] / 1024.0, 2),
                    'size_mb': z_info['size_mb'],
                    'download_url': None,
                }
            )
            for b in z_info['basins']:
                if b not in attr_basins:
                    attr_basins.append(b)
            if not attr_variables and z_info['all_variables']:
                attr_variables = z_info['all_variables']

        return {
            'basin_count': len(attr_basins),
            'variable_count': len(attr_variables),
            'basins': attr_basins,
            'variable_names': attr_variables,
            'basin_previews': attr_basin_previews,
            'files': attr_files,
        }

    def _collect_zarr_stores(
        self,
        dirs: tuple[Path, ...],
        p_dir: Path,
        category: str,
        seen_zarrs: set[str],
        *,
        exclude_forecast: bool = False,
    ) -> list[dict[str, Any]]:
        """Scan ``dirs`` for ``*.zarr`` stores and return inspected metadata."""
        stores: list[dict[str, Any]] = []
        for base_d in dirs:
            if not base_d.exists():
                continue
            for z_item in _find_zarr_dirs(base_d):
                if exclude_forecast and 'forecast' in z_item.name:
                    continue
                if str(z_item) in seen_zarrs:
                    continue
                seen_zarrs.add(str(z_item))
                stores.append(
                    self._inspect_zarr_store(
                        z_item,
                        p_dir,
                        default_category=category,
                    ),
                )
        return stores

    def _summarize_models(
        self,
        user: str,
        p_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collect model runs and hot-start checkpoint files for ``user``."""
        runs_dir = self.get_model_runs_dir(user)
        model_runs: list[dict[str, Any]] = []
        for run_entry in sorted(runs_dir.iterdir()):
            if run_entry.is_dir():
                r_bytes = sum(
                    f.stat().st_size
                    for f in run_entry.rglob('*')
                    if f.is_file()
                )
                model_runs.append(
                    {
                        'run_id': run_entry.name,
                        'rel_path': str(run_entry.relative_to(p_dir)),
                        'size_mb': round(r_bytes / (1024.0 * 1024.0), 3),
                        'checkpoints': [
                            f.name for f in run_entry.rglob('*.pt')
                        ],
                        'has_config': (run_entry / 'config.yml').exists(),
                    }
                )
        hs_dir = self.get_hot_start_dir(user)
        hot_start_files = [
            {
                'filename': hf.name,
                'rel_path': str(hf.relative_to(p_dir)),
                'size_kb': round(hf.stat().st_size / 1024.0, 2),
            }
            for hf in sorted(hs_dir.rglob('*'))
            if hf.is_file()
        ]
        return model_runs, hot_start_files

    def _build_basin_coverage_matrix(
        self,
        catchments_list: list[dict[str, Any]],
        attributes_summary: dict[str, Any],
        stores_by_kind: dict[str, list[dict[str, Any]]],
        return_periods_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Assemble the per-basin data coverage matrix across all categories."""
        attr_basins = attributes_summary['basins']
        attr_variables = attributes_summary['variable_names']
        attr_basin_previews = attributes_summary['basin_previews']
        historical_stores = stores_by_kind['historical']
        targets_stores = stores_by_kind['targets']
        forecast_stores = stores_by_kind['forecast']
        assimilation_stores = stores_by_kind['assimilation']

        all_basin_ids: list[str] = []
        poly_by_basin = {c['catchment_id']: c for c in catchments_list}
        for cid in list(poly_by_basin) + list(attr_basins):
            if cid not in all_basin_ids:
                all_basin_ids.append(cid)
        for st in (
            historical_stores
            + targets_stores
            + forecast_stores
            + assimilation_stores
        ):
            for b in st.get('basins', []):
                if b not in all_basin_ids:
                    all_basin_ids.append(b)

        attr_preview_by_basin = {p['basin_id']: p for p in attr_basin_previews}
        basin_coverage: list[dict[str, Any]] = []
        for bid in all_basin_ids:
            p_info = poly_by_basin.get(bid)
            a_info = attr_preview_by_basin.get(bid)
            has_attr = bid in attr_basins

            matching_hist = [
                s for s in historical_stores if bid in s.get('basins', [])
            ]
            matching_targ = [
                s for s in targets_stores if bid in s.get('basins', [])
            ]
            matching_fcst = [
                s for s in forecast_stores if bid in s.get('basins', [])
            ]
            matching_assim = [
                s for s in assimilation_stores if bid in s.get('basins', [])
            ]

            targ_summary_str = None
            if matching_targ:
                t0 = matching_targ[0]
                targ_summary_str = (
                    f'{t0.get("start_date")} -> {t0.get("end_date")}'
                    f' ({t0.get("n_timesteps", 0)}d)'
                )

            basin_coverage.append(
                {
                    'basin_id': bid,
                    'name': p_info['name'] if p_info else bid,
                    'area_km2': p_info['area_km2'] if p_info else None,
                    'has_polygon': p_info is not None,
                    'polygon_source': p_info['source'] if p_info else None,
                    'dataset': p_info['dataset'] if p_info else None,
                    'has_attributes': has_attr,
                    'attributes_count': (
                        a_info['variable_count']
                        if a_info
                        else (len(attr_variables) if has_attr else 0)
                    ),
                    'has_historical_weather': len(matching_hist) > 0,
                    'has_historical': len(matching_hist) > 0,
                    'historical_weather_info': _format_hist_summary(
                        matching_hist
                    ),
                    'has_streamflow_targets': len(matching_targ) > 0,
                    'streamflow_targets_info': targ_summary_str,
                    'has_return_periods': bid in return_periods_data,
                    'has_realtime_forecast': len(matching_fcst) > 0,
                    'realtime_forecast_info': _format_fcst_summary(
                        matching_fcst
                    ),
                    'has_assimilation': len(matching_assim) > 0,
                }
            )
        return basin_coverage

    def get_account_summary(
        self,
        username: str | None = None,
    ) -> dict[str, Any]:
        """Return a comprehensive inventory of all data stored for an account.

        Args:
            username: Target username, or active profile if None.

        Returns:
            Account inventory summary dictionary.
        """
        user = _sanitize_username(username or self._active_username)
        p_dir = self.get_profile_dir(user)
        profile_info = self.get_profile_info(user)

        catchments_list, polygon_files, total_area_km2 = (
            self._summarize_polygons(user, p_dir)
        )
        attributes_summary = self._summarize_attributes(user, p_dir)
        attr_basins = attributes_summary['basins']
        attr_variables = attributes_summary['variable_names']
        attr_files = attributes_summary['files']

        seen_zarrs: set[str] = set()
        historical_stores = self._collect_zarr_stores(
            (
                self.get_historical_dir(user),
                self.get_dynamics_dir(user),
                p_dir / 'archives',
            ),
            p_dir,
            'Historical Training Data',
            seen_zarrs,
            exclude_forecast=True,
        )

        targets_stores: list[dict[str, Any]] = []
        targets_zarr = self.get_targets_zarr_path(user)
        if targets_zarr.exists() and str(targets_zarr) not in seen_zarrs:
            seen_zarrs.add(str(targets_zarr))
            targets_stores.append(
                self._inspect_zarr_store(
                    targets_zarr,
                    p_dir,
                    default_category='Historical Streamflow Targets',
                ),
            )
        return_periods_data: dict[str, Any] = {}
        rp_file = self.get_return_periods_path(user)
        if rp_file.is_file() and rp_file.stat().st_size > 0:
            return_periods_data = json.loads(
                rp_file.read_text(encoding='utf-8'),
            )
        uploads_dir = self.get_targets_dir(user) / 'uploads'
        target_uploads = [
            {
                'filename': uf.name,
                'rel_path': str(uf.relative_to(p_dir)),
                'size_kb': round(uf.stat().st_size / 1024.0, 2),
            }
            for uf in sorted(uploads_dir.glob('*.csv'))
            if uf.is_file()
        ]

        forecast_stores = self._collect_zarr_stores(
            (self.get_forecast_dir(user), self.get_realtime_dir_v2(user)),
            p_dir,
            'Real-Time Weather Forecast',
            seen_zarrs,
        )

        assimilation_stores: list[dict[str, Any]] = []
        assim_zarr = self.get_assimilation_zarr_path(user)
        if assim_zarr.exists() and str(assim_zarr) not in seen_zarrs:
            seen_zarrs.add(str(assim_zarr))
            assimilation_stores.append(
                self._inspect_zarr_store(
                    assim_zarr,
                    p_dir,
                    default_category='Real-Time Streamflow Assimilation',
                ),
            )
        da_dir = self.get_da_states_dir(user)
        da_state_files = [
            {
                'filename': df_item.name,
                'rel_path': str(df_item.relative_to(p_dir)),
                'size_kb': round(df_item.stat().st_size / 1024.0, 2),
            }
            for df_item in sorted(da_dir.rglob('*'))
            if df_item.is_file()
        ]

        model_runs, hot_start_files = self._summarize_models(user, p_dir)
        stores_by_kind = {
            'historical': historical_stores,
            'targets': targets_stores,
            'forecast': forecast_stores,
            'assimilation': assimilation_stores,
        }
        basin_coverage = self._build_basin_coverage_matrix(
            catchments_list=catchments_list,
            attributes_summary=attributes_summary,
            stores_by_kind=stores_by_kind,
            return_periods_data=return_periods_data,
        )

        inspected_zarr_bytes = sum(
            int(st.get('size_bytes', 0))
            for st in (
                historical_stores
                + targets_stores
                + forecast_stores
                + assimilation_stores
            )
        )
        non_zarr_bytes = _sum_non_zarr_bytes(p_dir, seen_zarrs)
        total_bytes = inspected_zarr_bytes + non_zarr_bytes
        hist_basins_set = {
            b for s in historical_stores for b in s.get('basins', [])
        }
        fcst_basins_set = {
            b for s in forecast_stores for b in s.get('basins', [])
        }
        targ_basins_set = {
            b for s in targets_stores for b in s.get('basins', [])
        }
        assim_basins_set = {
            b for s in assimilation_stores for b in s.get('basins', [])
        }

        return {
            'username': user,
            'is_guest': bool(profile_info.get('is_guest', user == 'guest')),
            'profile': profile_info,
            'profile_dir': str(p_dir),
            'available_profiles': self.list_profiles(),
            'totals': {
                'watersheds_count': len(catchments_list),
                'polygon_count': len(catchments_list),
                'total_area_km2': round(total_area_km2, 2),
                'attributes_basins_count': len(attr_basins),
                'attribute_basin_count': len(attr_basins),
                'attributes_variables_count': len(attr_variables),
                'attribute_variable_count': len(attr_variables),
                'attributes_files_count': len(attr_files),
                'historical_stores_count': len(historical_stores),
                'historical_store_count': len(historical_stores),
                'historical_basins_count': len(hist_basins_set),
                'targets_stores_count': len(targets_stores),
                'targets_basins_count': len(targ_basins_set),
                'forecast_stores_count': len(forecast_stores),
                'forecast_basins_count': len(fcst_basins_set),
                'assimilation_stores_count': len(assimilation_stores),
                'assimilation_basins_count': len(assim_basins_set),
                'models_runs_count': len(model_runs),
                'hot_start_count': len(hot_start_files),
                'total_size_bytes': total_bytes,
                'total_size_mb': round(total_bytes / (1024.0 * 1024.0), 3),
            },
            'basin_coverage': basin_coverage,
            'polygons': {
                'count': len(catchments_list),
                'total_area_km2': round(total_area_km2, 2),
                'catchments': catchments_list,
                'files': polygon_files,
            },
            'attributes': attributes_summary,
            'historical_training': {
                'store_count': len(historical_stores),
                'basin_count': len(hist_basins_set),
                'stores': historical_stores,
            },
            'streamflow_targets': {
                'store_count': len(targets_stores),
                'basin_count': len(targ_basins_set),
                'stores': targets_stores,
                'return_periods': return_periods_data,
                'uploads': target_uploads,
            },
            'realtime_forecasts': {
                'store_count': len(forecast_stores),
                'basin_count': len(fcst_basins_set),
                'stores': forecast_stores,
            },
            'assimilation': {
                'store_count': len(assimilation_stores),
                'basin_count': len(assim_basins_set),
                'stores': assimilation_stores,
                'da_states': da_state_files,
            },
            'models': {
                'runs_count': len(model_runs),
                'runs': model_runs,
                'hot_start_files': hot_start_files,
            },
        }


_GLOBAL_PROFILE_MANAGER: ProfileManager | None = None


def get_profile_manager() -> ProfileManager:
    """Return the global ProfileManager singleton.

    Set ``OPENHYDRONET_PROFILES_DIR`` (or legacy ``EARTHKIT_PROFILES_DIR``) to
    store profiles outside ``data/profiles``.
    """
    global _GLOBAL_PROFILE_MANAGER  # noqa: PLW0603
    if _GLOBAL_PROFILE_MANAGER is None:
        env_dir = os.environ.get('OPENHYDRONET_PROFILES_DIR') or os.environ.get(
            'EARTHKIT_PROFILES_DIR',
        )
        _GLOBAL_PROFILE_MANAGER = ProfileManager(profiles_dir=env_dir or None)
    return _GLOBAL_PROFILE_MANAGER
