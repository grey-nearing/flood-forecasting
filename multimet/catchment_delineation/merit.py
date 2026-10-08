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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import google.auth
import google.auth.transport.requests
import numpy as np

from multimet.catchment_delineation.config import RES_DEG, TILE_CELLS
from multimet.catchment_delineation.datasets import MERIT_HYDRO_90M
from multimet.catchment_delineation.tiles import (
    is_tile_in_coverage,
    tile_key_to_filename,
)

MERIT_HYDRO_EE_ASSET: str = 'MERIT/Hydro/v1_0_1'
EE_MERIT_GET_PIXELS_URL: str = (
    'https://earthengine-highvolume.googleapis.com/v1/'
    f'projects/earthengine-public/assets/{MERIT_HYDRO_EE_ASSET}:getPixels'
)
HALF_ROWS: int = TILE_CELLS // 2
FULL_COLS: int = TILE_CELLS
VALID_D8_CODES: np.ndarray = np.array(
    [1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int16
)
_MIN_TILE_FILE_BYTES: int = 36_000_000
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_EE_SCOPES: list[str] = [
    'https://www.googleapis.com/auth/earthengine.readonly',
    'https://www.googleapis.com/auth/cloud-platform',
]

_CRED_LOCK = threading.Lock()
_CACHED_CREDENTIALS: Any = None


def _get_access_token(
    credentials: Any = None, *, force_refresh: bool = False
) -> str:
    """Obtain a valid Google Cloud OAuth2 access token via google.auth."""
    global _CACHED_CREDENTIALS
    with _CRED_LOCK:
        creds = credentials
        if creds is None:
            if _CACHED_CREDENTIALS is None:
                _CACHED_CREDENTIALS, _ = google.auth.default(scopes=_EE_SCOPES)
            creds = _CACHED_CREDENTIALS
        if force_refresh or not creds.valid or not creds.token:
            creds.refresh(google.auth.transport.requests.Request())
        token = creds.token
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
    credentials: Any = None,
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

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            token = _get_access_token(
                credentials=credentials, force_refresh=(attempt > 0)
            )
            req = urllib.request.Request(
                EE_MERIT_GET_PIXELS_URL,
                data=payload,
                headers={
                    'Authorization': f'Bearer {token}',
                    'x-goog-user-project': str(ee_project).strip(),
                    'Content-Type': 'application/json',
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = np.load(io.BytesIO(resp.read()))
            return np.asarray(raw['dir'])
        except urllib.error.HTTPError as err:
            last_err = err
            if err.code not in _RETRYABLE_HTTP_CODES or attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
        except urllib.error.URLError as err:
            last_err = err
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(
        f'Failed to fetch MERIT-Hydro dir half-tile ({lat_top}, {lon_left}): '
        f'{last_err}'
    ) from last_err


def download_merit_d8_tile(
    lat_top: int,
    lon_left: int,
    target_dir: str | Path,
    *,
    ee_project: str,
    credentials: Any = None,
    created_files: set[Path] | None = None,
) -> Path:
    """Download a single 5x5 degree (6000, 6000) uint8 MERIT-Hydro D8 tile atomically."""
    if not target_dir:
        raise ValueError('An explicit target_dir must be provided.')
    if not ee_project or not str(ee_project).strip():
        raise ValueError('An explicit ee_project must be provided.')

    tile_lat = int(round(lat_top))
    tile_lon = int(round(lon_left))
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
    try:
        np.save(tmp_path, d8_uint8)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    if created_files is not None:
        created_files.add(out_path)
    return out_path
