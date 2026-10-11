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
"""OpenHydroNet weather visualization UI adapter for the Weather Viewer tab.

Delegates all backend forecast stream loading, grid slicing, wind vector
extraction, point meteograms, and catchment weather summaries to
`multimet.weather_fetcher`, and all Web Mercator tile rendering, indexed PNG
animation frame encoding, and colormaps to `frontend.weather_viewer`.

When a model stream has not yet been synced to local disk, tile and animation
frame endpoints pass `strict=False` so the UI receives transparent PNGs and
`data_source='unavailable'` metadata rather than HTTP 500 errors, while synced
models render their native archived streams without substitution.

The loaded forecast run is an immutable `_LoadedRun`. `reload_if_changed()`
builds the new run first and then swaps it in atomically, so HTTP handler
threads that started on the previous run keep a consistent set of arrays and
metadata until they finish; the previous run's memory maps are closed once no
request references them any more.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frontend.weather_sync import (
    CHECK_INTERVAL_MINUTES,
)
from frontend.weather_sync import (
    weather_data_root as _weather_data_root,
)
from frontend.weather_viewer.tiles import (
    FRAME_SIZE,
    FRAME_VERSION,
    TILE_VERSION,
    compute_frame_index,
    render_raster_tile,
    render_weather_frame,
)
from frontend.weather_viewer.tiles import (
    clear_frame_cache as _clear_frame_cache,
)
from frontend.weather_viewer.wind import extract_wind_vectors
from multimet.weather_fetcher.config import (
    N_LAT,
    N_LON,
    RUN_METADATA_FILE,
    SUPPORTED_MODELS,
    SUPPORTED_VARIABLES,
)
from multimet.weather_fetcher.fetcher import (
    StreamHandle,
    extract_point_value,
    fetch_catchment_summary,
    fetch_point_timeseries,
    get_model_data_info_from_streams,
)
from multimet.weather_fetcher.fetcher import (
    clear_accum_grid_cache as _clear_accum_grid_cache,
)
from multimet.weather_fetcher.fetcher import (
    close_unreferenced_mmaps as _close_unreferenced_mmaps,
)
from multimet.weather_fetcher.fetcher import (
    file_step_for_lead as _file_step_for_lead,
)
from multimet.weather_fetcher.fetcher import (
    rate_file_steps as _rate_file_steps,
)
from multimet.weather_fetcher.fetcher import (
    scan_streams as _scan_streams,
)
from multimet.weather_fetcher.sync import read_sync_status as _read_sync_status
from utils.file_paths import OPENHYDRONET_WEATHER_PREWARM_FRAMES_ENV

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

VIEWER_STEP_HOURS = 3
NUM_HOURS = 81

MODEL_BADGES: dict[str, str] = {
    'ecmwf_hres': 'HRES',
    'ecmwf_ifs': 'ENS Control',
    'ecmwf_aifs': 'AI 15-Day',
    'noaa_gfs': 'GFS Physics',
    'noaa_gefs': 'GEFS Control',
    'noaa_hrrr': '3km Physics 48-Hour',
    'nasa_imerg': 'Satellite Obs',
    'noaa_cpc': 'Gauge Obs',
}

__all__ = [
    'CANDIDATE_DATA_DIRS',
    'FRAME_SIZE',
    'N_LAT',
    'N_LON',
    'RUN_METADATA_FILE',
    'SUPPORTED_MODELS',
    'SUPPORTED_VARIABLES',
    '_rate_file_steps',
    'generate_raster_tile',
    'generate_weather_frame',
    'get_catchment_weather_summary',
    'get_frame_index',
    'get_model_data_info',
    'get_sync_status',
    'get_weather_models_info',
    'get_weather_probe',
    'get_wind_vectors',
    'init_weather_streams',
    'reload_if_changed',
]

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
# Streams copied next to the frontend (used when no run has been synced).
DATA_DIR = BASE_DIR / 'data' / 'forecasts'

# Local-disk root written by frontend.weather_sync
# (`$OPENHYDRONET_WEATHER_DATA_DIR` or `~/.cache/openhydronet/weather`).
WEATHER_DATA_ROOT: Path = Path(_weather_data_root())
LOCAL_CURRENT_DIR = WEATHER_DATA_ROOT / 'current'

# Searched in order; the first directory holding `.bin` streams is loaded.
CANDIDATE_DATA_DIRS: list[Path] = [LOCAL_CURRENT_DIR, DATA_DIR]


@dataclasses.dataclass(frozen=True)
class _LoadedRun:
    """One loaded forecast run. Replaced as a whole on reload, never mutated."""

    handles: dict[str, StreamHandle]
    stream_info: dict[str, dict[str, Any]]
    arrays: dict[str, np.ndarray]
    signature: tuple[str, int] | None
    directory: Path | None


_EMPTY_RUN = _LoadedRun({}, {}, {}, None, None)
_RUN: _LoadedRun = _EMPTY_RUN
# Aliases of the loaded run's members; rebound together with `_RUN`.
MMAP_HANDLES: dict[str, StreamHandle] = _RUN.handles
STREAM_INFO: dict[str, dict[str, Any]] = _RUN.stream_info
_ARRAYS: dict[str, np.ndarray] = _RUN.arrays
_INITIALIZED = False
_INIT_LOCK = threading.Lock()


def _find_data_dir() -> Path | None:
    """Return the first candidate directory that holds `.bin` streams."""
    for c_dir in CANDIDATE_DATA_DIRS:
        if c_dir.is_dir() and any(c_dir.glob('*.bin')):
            return c_dir
    return None


def _dir_signature(target_dir: Path) -> tuple[str, int]:
    """Return the identity of a run directory: resolved path and mtime."""
    resolved = target_dir.resolve()
    meta = resolved / RUN_METADATA_FILE
    return (str(resolved), meta.stat().st_mtime_ns if meta.is_file() else 0)


def _swap_run(new_run: _LoadedRun) -> _LoadedRun:
    """Install `new_run` and return the run it replaced."""
    global _RUN, MMAP_HANDLES, STREAM_INFO, _ARRAYS  # noqa: PLW0603
    old = _RUN
    _RUN = new_run
    MMAP_HANDLES, STREAM_INFO, _ARRAYS = (
        new_run.handles,
        new_run.stream_info,
        new_run.arrays,
    )
    return old


def _retire_run(run: _LoadedRun) -> int:
    """Drop caches built from a replaced run and close idle memory maps.

    `run` must be the only remaining reference to the replaced run (callers pass
    the result of `_swap_run` directly). Maps that in-flight requests still read
    are left open and unmapped by the garbage collector when they finish.

    Returns:
        Number of memory maps closed now.
    """
    handles = run.handles
    del run
    _clear_accum_grid_cache()
    _clear_frame_cache()
    return _close_unreferenced_mmaps(handles)


def _install_streams(target_dir: Path | None) -> None:
    """Load the streams of `target_dir` (`None`: no run)."""
    if target_dir is None:
        new_run = _EMPTY_RUN
    else:
        handles, infos, arrays = _scan_streams(target_dir)
        new_run = _LoadedRun(
            handles=handles,
            stream_info=infos,
            arrays=arrays,
            signature=_dir_signature(target_dir),
            directory=target_dir.resolve(),
        )
        del handles, infos, arrays
    closed = _retire_run(_swap_run(new_run))
    if closed:
        logger.info('[WeatherEngine] Closed %d retired stream maps', closed)
    warm_flag = os.environ.get('OPENHYDRONET_WEATHER_WARM_CACHE', '1')
    if new_run.stream_info and warm_flag == '1':
        threading.Thread(
            target=_warm_caches,
            args=(new_run,),
            name='weather-cache-warmup',
            daemon=True,
        ).start()


def init_weather_streams() -> None:
    """Memory-map the forecast streams found in the candidate directories."""
    global _INITIALIZED  # noqa: PLW0603
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        target_dir = _find_data_dir()
        if target_dir is None:
            logger.info(
                '[WeatherEngine] No synced forecast streams in %s',
                [str(d) for d in CANDIDATE_DATA_DIRS],
            )
        _install_streams(target_dir)
        _INITIALIZED = True


def _current_run() -> _LoadedRun:
    """Return the loaded run, initialising the engine on first use."""
    init_weather_streams()
    return _RUN


def reload_if_changed() -> bool:
    """Switch to a newer downloaded run if there is one; `True` if it did."""
    if not _INITIALIZED:
        init_weather_streams()
        return True
    target_dir = _find_data_dir()
    if target_dir is None or _dir_signature(target_dir) == _RUN.signature:
        return False
    with _INIT_LOCK:
        if _dir_signature(target_dir) == _RUN.signature:
            return False
        logger.info(
            '[WeatherEngine] Loading new forecast run from %s',
            target_dir.resolve(),
        )
        _install_streams(target_dir)
    return True


def get_sync_status() -> dict[str, Any]:
    """Return last automatic check status for new runs and what is loaded.

    `sync_status_found` is `False` when no synchronizer has written
    `sync_status.json` yet; `last_result` is then `'never'` and the other status
    fields are `None`.
    """
    status = _read_sync_status(WEATHER_DATA_ROOT)
    found = bool(status)
    loaded = _RUN.directory
    return {
        'last_check_utc': status.get('last_check_utc'),
        'last_success_utc': status.get('last_success_utc'),
        'last_result': status.get('last_result') if found else 'never',
        'message': (
            status.get('message')
            if found
            else 'No automatic update has run yet.'
        ),
        'check_interval_minutes': (
            status.get('check_interval_minutes') or CHECK_INTERVAL_MINUTES
        ),
        'sync_status_found': found,
        'auto_update': False,
        'data_dir': str(loaded) if loaded else None,
    }


def _warm_caches(run: _LoadedRun) -> None:
    _warm_page_cache([info['file'] for info in run.stream_info.values()])
    _prewarm_frames(run)


def _warm_page_cache(paths: Sequence[str]) -> None:
    """Read each file once so later random access hits the OS page cache."""
    for path in paths:
        file_path = Path(path)
        if not file_path.is_file():
            continue
        with file_path.open('rb') as f_handle:
            while f_handle.read(8 << 20):
                pass


def _prewarm_frames(run: _LoadedRun) -> None:
    """Render the rain frames of every model in `run`, so Play starts fast.

    Stops as soon as `run` is no longer the loaded run.
    """
    if os.environ.get(OPENHYDRONET_WEATHER_PREWARM_FRAMES_ENV, '1') != '1':
        return
    for model_key in SUPPORTED_MODELS:
        if f'{model_key}_precip' not in run.stream_info:
            continue
        index = compute_frame_index(
            run.stream_info,
            model_key=model_key,
            var_key='precipitation',
            strict=False,
        )
        for step in index['frame_steps']:
            if _RUN is not run or STREAM_INFO is not run.stream_info:
                return
            render_weather_frame(
                run.arrays,
                run.stream_info,
                model_key=model_key,
                var_key='precipitation',
                step_idx=step,
                strict=False,
            )


def _lead_hours_for_step(step_idx: int) -> int:
    """Return lead hours of a 3-hourly viewer step."""
    if isinstance(step_idx, bool) or int(step_idx) != step_idx or step_idx < 0:
        msg = f'step must be a non-negative integer, got {step_idx!r}.'
        raise ValueError(msg)
    return int(step_idx) * VIEWER_STEP_HOURS


def _valid_time_iso(run: _LoadedRun, model_key: str, lead_h: int) -> str | None:
    """Return valid time of `lead_h` for the model's loaded run."""
    init_time = get_model_data_info_from_streams(run.stream_info, model_key)[
        'init_time'
    ]
    if not init_time:
        return None
    base = datetime.datetime.fromisoformat(init_time)
    valid = base + datetime.timedelta(hours=lead_h)
    return valid.strftime('%Y-%m-%dT%H:%M:%SZ')


def _model_info(run: _LoadedRun, model_key: str) -> dict[str, Any]:
    info = get_model_data_info_from_streams(run.stream_info, model_key)
    return {
        **info,
        'tile_version': TILE_VERSION,
        'frame_version': FRAME_VERSION,
    }


def get_model_data_info(model_key: str) -> dict[str, Any]:
    """Return metadata and availability status for `model_key`.

    `data_source` is `'archived_run'` when the model has synced streams and
    `'unavailable'` otherwise (then `init_time` is `None` and `max_lead_hours`
    is 0).

    Raises:
        ValueError: If `model_key` is not a supported model.
    """
    return _model_info(_current_run(), model_key)


def get_weather_models_info() -> list[dict[str, Any]]:
    """Return `SUPPORTED_MODELS` entries plus stream availability metadata."""
    reload_if_changed()
    run = _current_run()
    models_info = []
    for key, spec in SUPPORTED_MODELS.items():
        info = {**spec, **_model_info(run, key)}
        info['badge'] = MODEL_BADGES.get(key, '')
        models_info.append(info)
    return models_info


def generate_raster_tile(  # noqa: PLR0913, PLR0917
    model_key: str, var_key: str, step_idx: int, z: int, x: int, y: int
) -> bytes:
    """Render a 256x256 Web Mercator PNG tile (transparent when unsynced)."""
    run = _current_run()
    return render_raster_tile(
        run.arrays,
        run.stream_info,
        model_key=model_key,
        var_key=var_key,
        step_idx=step_idx,
        z=z,
        x=x,
        y=y,
        bilinear=False,
        strict=False,
    )


def get_frame_index(model_key: str, var_key: str) -> dict[str, Any]:
    """Return which frame each 3-hourly viewer step shows.

    Serves `/api/weather/frames/{model}/{variable}/index.json`.
    """
    run = _current_run()
    return compute_frame_index(
        run.stream_info, model_key=model_key, var_key=var_key, strict=False
    )


def generate_weather_frame(
    model_key: str, var_key: str, step_idx: int
) -> bytes:
    """Render a whole-world PNG frame for one viewer step."""
    run = _current_run()
    return render_weather_frame(
        run.arrays,
        run.stream_info,
        model_key=model_key,
        var_key=var_key,
        step_idx=step_idx,
        strict=False,
    )


def _unavailable_wind(
    run: _LoadedRun, model_key: str, lead_h: int, reason: str
) -> dict[str, Any]:
    """Return an empty wind payload with `nx = ny = 0` and `reason`."""
    return {
        'header': {
            'model': model_key,
            'step_hours': lead_h,
            'valid_time': _valid_time_iso(run, model_key, lead_h),
            'nx': 0,
            'ny': 0,
            'lo1': None,
            'la1': None,
            'dx': None,
            'dy': None,
            'data_source': 'unavailable',
            'missing_count': 0,
            'reason': reason,
        },
        'u': [],
        'v': [],
    }


def get_wind_vectors(
    model_key: str, step_idx: int = 0, subsample: int = 2
) -> dict[str, Any]:
    """Return downsampled global 10 m U/V wind matrices for streamlines.

    When the model has no synced wind streams, or did not store the requested
    lead, the payload carries no vectors (`u` and `v` are empty, `nx` and `ny`
    are 0, `header.data_source` is `'unavailable'` and `header.reason` says why)
    so the viewer stops drawing particles for that step. Masked cells of a real
    grid are `None`.

    Raises:
        ValueError: If `model_key`, `step_idx`, or `subsample` is invalid.
    """
    run = _current_run()
    if model_key not in SUPPORTED_MODELS:
        msg = f"Unknown weather model '{model_key}'"
        raise ValueError(msg)
    lead_h = _lead_hours_for_step(step_idx)
    u_stream = f'{model_key}_u10'
    v_stream = f'{model_key}_v10'
    if u_stream not in run.arrays or v_stream not in run.arrays:
        return _unavailable_wind(
            run,
            model_key,
            lead_h,
            f"No synced wind streams for '{model_key}'.",
        )
    fu = _file_step_for_lead(run.stream_info[u_stream], lead_h, is_rate=False)
    fv = _file_step_for_lead(run.stream_info[v_stream], lead_h, is_rate=False)
    if fu is None or fv is None:
        return _unavailable_wind(
            run,
            model_key,
            lead_h,
            (
                f"'{model_key}' did not store wind at lead {lead_h} h"
                f' (stored leads: {run.stream_info[u_stream]["lead_hours"]}).'
            ),
        )
    return extract_wind_vectors(
        run.arrays,
        run.stream_info,
        model_key=model_key,
        step_idx=step_idx,
        subsample=subsample,
    )


def _point_value(
    model_key: str, var_key: str, lead_h: float, lat: float, lon: float
) -> float | None:
    """Return nearest-cell value of one variable at `(lat, lon)`.

    Raises:
        FileNotFoundError: If the model's stream for `var_key` is not synced.
    """
    run = _current_run()
    return extract_point_value(
        run.arrays, run.stream_info, model_key, var_key, lead_h, lat, lon
    )


def get_weather_probe(lat: float, lon: float) -> dict[str, Any]:
    """Return comparative multi-model meteograms at `(lat, lon)`.

    Every supported model is listed; models without synced data report
    `data_source='unavailable'` and `None` curves.

    Raises:
        FileNotFoundError: If no model is synced at all.
        ValueError: If the coordinates are invalid.
    """
    run = _current_run()
    probe = fetch_point_timeseries(
        run.arrays,
        run.stream_info,
        lat=lat,
        lon=lon,
        models=list(SUPPORTED_MODELS.keys()),
        strict=False,
    )
    for m_key, m_data in probe['models'].items():
        m_data['tile_version'] = TILE_VERSION
        m_data['frame_version'] = FRAME_VERSION
        if 'badge' not in m_data:
            m_data['badge'] = MODEL_BADGES.get(m_key, '')
    return probe


def get_catchment_weather_summary(
    geojson_feature: dict[str, Any],
    step_idx: int = 0,
    model_key: str = 'ecmwf_ifs',
) -> dict[str, Any]:
    """Compute area-weighted basin precipitation and temperature for one step.

    Args:
        geojson_feature: GeoJSON Feature with a Polygon/MultiPolygon geometry
          and an `id` or `properties.catchment_id`.
        step_idx: 3-hourly viewer step.
        model_key: Supported model key.

    Returns:
        Dictionary of catchment-averaged precipitation and temperature metrics.

    Raises:
        FileNotFoundError: If the model's precipitation stream is not synced.
        KeyError: If the feature has no identifier.
        ValueError: If `model_key`, `step_idx`, or the geometry is invalid.
    """
    run = _current_run()
    return fetch_catchment_summary(
        run.arrays,
        run.stream_info,
        geojson_feature=geojson_feature,
        step_idx=step_idx,
        model_key=model_key,
    )
