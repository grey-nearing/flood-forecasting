# Copyright 2025 Google LLC
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

"""Earth Engine MERIT-Hydro 90m D8 flow-direction tile fetcher."""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import google.auth
import google.auth.transport.requests
import numpy as np
import requests

from multimet.catchment_delineation.config import RES_DEG, TILE_CELLS
from multimet.catchment_delineation.datasets import MERIT_HYDRO_90M
from multimet.catchment_delineation.tiles import (
    is_tile_in_coverage,
    tile_key_to_filename,
)
from utils.file_paths import EE_MERIT_GET_PIXELS_URL, MERIT_HYDRO_EE_ASSET

__all__ = [
    'EE_MERIT_GET_PIXELS_URL',
    'FULL_COLS',
    'HALF_ROWS',
    'MERIT_HYDRO_EE_ASSET',
    'VALID_D8_CODES',
    'download_merit_d8_tile',
    'fetch_merit_d8_half_tile',
]

HALF_ROWS: int = TILE_CELLS // 2
FULL_COLS: int = TILE_CELLS
VALID_D8_CODES: np.ndarray = np.array(
    [1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int16
)
_MIN_TILE_FILE_BYTES: int = 36_000_000
_EE_SCOPES: list[str] = [
    'https://www.googleapis.com/auth/earthengine.readonly',
    'https://www.googleapis.com/auth/cloud-platform',
]

_CRED_LOCK = threading.Lock()
_CRED_CACHE: dict[str, object] = {}


def _get_access_token(
    credentials: object = None, *, force_refresh: bool = False
) -> str:
    """Obtain a valid Google Cloud OAuth2 access token via google.auth."""
    with _CRED_LOCK:
        creds = credentials
        if creds is None:
            if 'default' not in _CRED_CACHE:
                default_creds, _ = google.auth.default(scopes=_EE_SCOPES)
                _CRED_CACHE['default'] = default_creds
            creds = _CRED_CACHE['default']
        if force_refresh or not creds.valid or not creds.token:  # type: ignore[union-attr]
            creds.refresh(google.auth.transport.requests.Request())  # type: ignore[union-attr]
        token = creds.token  # type: ignore[union-attr]
        if not token:
            raise RuntimeError(
                'Failed to obtain Google Cloud Application Default Credentials '
                'access token for Earth Engine.'
            )
        return str(token)


def fetch_merit_d8_half_tile(
    lat_top: float,
    lon_left: float,
    *,
    ee_project: str,
    credentials: object = None,
    retries: int = 3,
) -> np.ndarray:
    """Fetch a (3000, 6000) half-tile of MERIT/Hydro/v1_0_1 'dir' band."""
    if not ee_project or not str(ee_project).strip():
        raise ValueError(
            'An explicit ee_project (Google Cloud project ID) is required to '
            'fetch MERIT-Hydro tiles from Earth Engine.'
        )
    if retries < 1:
        raise ValueError(f'retries must be >= 1; got {retries}')

    payload = json.dumps({
        'fileFormat': 'NPY',
        'bandIds': ['dir'],
        'grid': {
            'dimensions': {'width': FULL_COLS, 'height': HALF_ROWS},
            'affineTransform': {
                'scaleX': RES_DEG,
                'shearX': 0.0,
                'translateX': float(lon_left),
                'shearY': 0.0,
                'scaleY': -RES_DEG,
                'translateY': float(lat_top),
            },
            'crsCode': 'EPSG:4326',
        },
    }).encode('utf-8')

    token = _get_access_token(credentials=credentials, force_refresh=False)
    resp = requests.post(
        EE_MERIT_GET_PIXELS_URL,
        data=payload,
        headers={
            'Authorization': f'Bearer {token}',
            'x-goog-user-project': str(ee_project).strip(),
            'Content-Type': 'application/json',
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = np.load(io.BytesIO(resp.content))
    return np.asarray(raw['dir'])


def download_merit_d8_tile(
    lat_top: int,
    lon_left: int,
    target_dir: str | Path,
    *,
    ee_project: str,
    credentials: object = None,
) -> Path:
    """Download a 5x5 deg (6000, 6000) uint8 MERIT-Hydro D8 tile atomically."""
    if not target_dir:
        raise ValueError('An explicit target_dir must be provided.')
    if not ee_project or not str(ee_project).strip():
        raise ValueError('An explicit ee_project must be provided.')

    tile_lat = int(lat_top)
    tile_lon = int(lon_left)
    tile_name = tile_key_to_filename(tile_lat, tile_lon)
    if not is_tile_in_coverage(tile_lat, tile_lon, dataset=MERIT_HYDRO_90M):
        raise ValueError(
            f'Tile {tile_name} is outside the MERIT-Hydro coverage domain '
            f'({MERIT_HYDRO_90M.min_lat}° to {MERIT_HYDRO_90M.max_lat}° '
            'latitude).'
        )

    directory = Path(target_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / tile_name
    if out_path.is_file() and out_path.stat().st_size >= _MIN_TILE_FILE_BYTES:
        return out_path

    top_arr = fetch_merit_d8_half_tile(
        float(tile_lat),
        float(tile_lon),
        ee_project=ee_project,
        credentials=credentials,
    )
    bot_arr = fetch_merit_d8_half_tile(
        float(tile_lat) - 2.5,
        float(tile_lon),
        ee_project=ee_project,
        credentials=credentials,
    )
    full_raw = np.vstack([top_arr, bot_arr])
    valid_mask = np.isin(full_raw, VALID_D8_CODES)
    d8_uint8 = np.where(valid_mask, full_raw, 0).astype(np.uint8)

    tmp_path = (
        directory
        / f'.{tile_name}.tmp.{os.getpid()}.{threading.get_ident()}.npy'
    )
    np.save(tmp_path, d8_uint8)
    tmp_path.replace(out_path)
    return out_path


