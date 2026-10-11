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
"""WebMercator XYZ PNG raster tile and full-world animation frame renderer."""

from __future__ import annotations

import collections
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from frontend.weather_viewer.colormaps import (
    colorize_indexed,
    colorize_rgba,
    encode_indexed_png,
    encode_rgba_png,
)
from multimet.weather_fetcher.config import (
    NUM_STEPS,
    STEP_HOURS,
    STREAM_SUFFIX,
    SUPPORTED_MODELS,
)
from multimet.weather_fetcher.fetcher import (
    fetch_forecast_grid,
    file_step_for_lead,
    get_model_data_info_from_streams,
    rate_file_steps,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

TILE_VERSION: str = '3'
FRAME_SIZE: int = 1440
FRAME_VERSION: str = '1'
MERCATOR_MAX_LAT: float = 85.0511287798
FRAME_VARIABLES: tuple[str, ...] = (
    'precipitation',
    'accumulated_precip',
    'temperature',
    'pressure',
)

_TRANSPARENT_TILE: bytes | None = None
_EMPTY_FRAME: bytes | None = None
_FRAME_COORDS: tuple[np.ndarray, np.ndarray] | None = None

_FRAME_CACHE: collections.OrderedDict[tuple[Any, ...], bytes] = (
    collections.OrderedDict()
)
_FRAME_CACHE_LIMIT_BYTES: int = 384 << 20
_FRAME_CACHE_BYTES: int = 0
_FRAME_LOCK = threading.Lock()


def clear_frame_cache() -> None:
    """Clear the in-memory rendered animation frame cache."""
    global _FRAME_CACHE_BYTES  # noqa: PLW0603
    with _FRAME_LOCK:
        _FRAME_CACHE.clear()
        _FRAME_CACHE_BYTES = 0


def tile_coordinates(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return latitudes (rows) and longitudes (columns) of a Mercator tile."""
    n = 2.0**z
    frac = (np.arange(tile_size, dtype=np.float64) + 0.5) / float(tile_size)
    lons = (x + frac) / n * 360.0 - 180.0
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + frac) / n))))
    return lats, lons


def frame_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Return latitudes (rows) and longitudes (columns) of a world frame."""
    global _FRAME_COORDS  # noqa: PLW0603
    if _FRAME_COORDS is None:
        frac = (np.arange(FRAME_SIZE, dtype=np.float64) + 0.5) / FRAME_SIZE
        lons = frac * 360.0 - 180.0
        lats = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * frac))))
        _FRAME_COORDS = (lats, lons)
    return _FRAME_COORDS


def transparent_tile() -> bytes:
    """Return a cached 256x256 fully transparent RGBA PNG tile."""
    global _TRANSPARENT_TILE  # noqa: PLW0603
    if _TRANSPARENT_TILE is None:
        _TRANSPARENT_TILE = encode_rgba_png(np.zeros((256, 256, 4), np.uint8))
    return _TRANSPARENT_TILE


def empty_frame() -> bytes:
    """Return a cached 1x1 transparent indexed PNG frame."""
    global _EMPTY_FRAME  # noqa: PLW0603
    if _EMPTY_FRAME is None:
        _EMPTY_FRAME = encode_indexed_png(
            np.zeros((1, 1), np.uint8), np.zeros((1, 4), np.uint8)
        )
    return _EMPTY_FRAME


def _frame_signature_from_streams(  # noqa: C901, PLR0911
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step: int,
    *,
    strict: bool = True,
) -> tuple[tuple[Any, ...], float] | None:
    """Return `(signature_tuple, natural_lead_hours)` for a viewer step."""
    lead_h = step * STEP_HOURS
    if var_key == 'accumulated_precip':
        info = stream_info.get(f'{model_key}_precip')
        if not info:
            if strict:
                msg = (
                    f"Precipitation stream '{model_key}_precip' not found in"
                    ' synced data.'
                )
                raise FileNotFoundError(msg)
            return None
        leads = info['lead_hours']
        if lead_h > leads[-1]:
            return None
        k = max(i for i, lead in enumerate(leads) if lead <= lead_h)
        return ('accumulated', k), float(leads[k])

    suffix = STREAM_SUFFIX.get(var_key)
    if suffix is None:
        return None
    stream_id = f'{model_key}_{suffix}'
    info = stream_info.get(stream_id)
    if not info:
        if strict:
            msg = f"Weather stream '{stream_id}' not found in synced data."
            raise FileNotFoundError(msg)
        return None
    if suffix == 'precip':
        rate_steps = rate_file_steps(info, lead_h)
        if not rate_steps:
            return None
        return ('rate', *rate_steps), float(info['lead_hours'][rate_steps[-1]])
    file_step = file_step_for_lead(info, lead_h, is_rate=False)
    if file_step is None:
        return None
    return ('plane', file_step), float(info['lead_hours'][file_step])


def compute_frame_index(
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Compute the frame deduplication index for animation frames."""
    if model_key not in SUPPORTED_MODELS:
        msg = f"Unknown weather model '{model_key}'"
        raise ValueError(msg)
    if var_key not in FRAME_VARIABLES:
        msg = f"Layer '{var_key}' has no animation frames"
        raise ValueError(msg)

    data_info = get_model_data_info_from_streams(stream_info, model_key)
    if not data_info['real_variables']:
        if strict:
            msg = f"No synced forecast run found for model '{model_key}'."
            raise FileNotFoundError(msg)
        return {
            'model': model_key,
            'variable': var_key,
            'init_time': None,
            'data_source': 'unavailable',
            'frame_version': FRAME_VERSION,
            'tile_version': TILE_VERSION,
            'projection': 'EPSG:3857',
            'width': FRAME_SIZE,
            'height': FRAME_SIZE,
            'bounds': [[-MERCATOR_MAX_LAT, -180.0], [MERCATOR_MAX_LAT, 180.0]],
            'step_hours': STEP_HOURS,
            'max_step': 0,
            'step_frames': [None] * NUM_STEPS,
            'frame_steps': [],
        }

    max_step = min(
        NUM_STEPS - 1, int(data_info['max_lead_hours'] // STEP_HOURS)
    )
    sigs = [
        _frame_signature_from_streams(
            stream_info, model_key, var_key, s, strict=strict
        )
        if s <= max_step
        else None
        for s in range(NUM_STEPS)
    ]
    groups: dict[tuple[Any, ...], list[int]] = {}
    natural: dict[tuple[Any, ...], float] = {}
    for s, sig in enumerate(sigs):
        if sig is not None:
            groups.setdefault(sig[0], []).append(s)
            natural[sig[0]] = sig[1]
    rep_of: dict[tuple[Any, ...], int] = {}
    for key, steps in groups.items():
        exact = [s for s in steps if s * STEP_HOURS == natural[key]]
        rep_of[key] = exact[0] if exact else steps[0]
    step_frames = [None if sig is None else rep_of[sig[0]] for sig in sigs]
    return {
        'model': model_key,
        'variable': var_key,
        'init_time': data_info['init_time'],
        'data_source': (
            'archived_run'
            if var_key in data_info['real_variables']
            else 'unavailable'
        ),
        'frame_version': FRAME_VERSION,
        'tile_version': TILE_VERSION,
        'projection': 'EPSG:3857',
        'width': FRAME_SIZE,
        'height': FRAME_SIZE,
        'bounds': [[-MERCATOR_MAX_LAT, -180.0], [MERCATOR_MAX_LAT, 180.0]],
        'step_hours': STEP_HOURS,
        'max_step': max_step,
        'step_frames': step_frames,
        'frame_steps': sorted(set(rep_of.values())),
    }


def evaluate_tile_field(  # noqa: PLR0913, PLR0917
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step_idx: int,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    bilinear: bool = False,
    strict: bool = True,
) -> np.ndarray | None:
    """Evaluate physical field values on `(lats, lons)` from synced streams."""
    suffix = STREAM_SUFFIX.get(var_key)
    if suffix is None:
        if strict:
            msg = f"Unsupported scalar forecast grid variable '{var_key}'."
            raise ValueError(msg)
        return None
    stream_id = f'{model_key}_{suffix}'
    if not strict and (stream_id not in stream_info or stream_id not in arrays):
        return None
    return fetch_forecast_grid(
        arrays,
        stream_info,
        model_key=model_key,
        var_key=var_key,
        step_idx=step_idx,
        lats=lats,
        lons=lons,
        bilinear=bilinear,
    )


def render_raster_tile(  # noqa: PLR0913, PLR0917
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step_idx: int,
    z: int,
    x: int,
    y: int,
    *,
    bilinear: bool = True,
    strict: bool = True,
) -> bytes:
    """Render a 256x256 Web Mercator PNG tile from synced model streams."""
    step_idx = max(0, int(step_idx))
    lats, lons = tile_coordinates(z, x, y)
    values = evaluate_tile_field(
        arrays,
        stream_info,
        model_key,
        var_key,
        step_idx,
        lats,
        lons,
        bilinear=bilinear,
        strict=strict,
    )
    if values is None:
        return transparent_tile()
    rgba = colorize_rgba(var_key, values)
    rgba[np.abs(lats) > MERCATOR_MAX_LAT] = 0
    return encode_rgba_png(rgba)


def render_weather_frame(  # noqa: PLR0913
    arrays: Mapping[str, np.ndarray],
    stream_info: Mapping[str, Mapping[str, Any]],
    model_key: str,
    var_key: str,
    step_idx: int,
    *,
    strict: bool = True,
) -> bytes:
    """Render a whole-world indexed PNG animation frame for one viewer step."""
    global _FRAME_CACHE_BYTES  # noqa: PLW0603
    index = compute_frame_index(stream_info, model_key, var_key, strict=strict)
    step = max(0, min(NUM_STEPS - 1, int(step_idx)))
    rep = index['step_frames'][step]
    if rep is None:
        return empty_frame()

    suffix = STREAM_SUFFIX.get(var_key, '')
    stream_id = f'{model_key}_{suffix}'
    stream_file = str((stream_info.get(stream_id) or {}).get('file') or '')
    arr_id = id(arrays.get(stream_id)) if stream_id in arrays else 0
    key = (arr_id, stream_file, model_key, var_key, rep, index['init_time'])

    with _FRAME_LOCK:
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            _FRAME_CACHE.move_to_end(key)
            return cached

    lats, lons = frame_coordinates()
    values = evaluate_tile_field(
        arrays,
        stream_info,
        model_key,
        var_key,
        rep,
        lats,
        lons,
        bilinear=False,
        strict=strict,
    )
    png = (
        empty_frame()
        if values is None
        else encode_indexed_png(*colorize_indexed(var_key, values))
    )
    with _FRAME_LOCK:
        if key not in _FRAME_CACHE:
            _FRAME_CACHE[key] = png
            _FRAME_CACHE_BYTES += len(png)
            while (
                _FRAME_CACHE_BYTES > _FRAME_CACHE_LIMIT_BYTES
                and len(_FRAME_CACHE) > 1
            ):
                _, old = _FRAME_CACHE.popitem(last=False)
                _FRAME_CACHE_BYTES -= len(old)
    return png
