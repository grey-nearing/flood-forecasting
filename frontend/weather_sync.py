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
"""Server-side driver around ``multimet.weather_fetcher.sync``.

The library never picks directories, never swallows exceptions and never
fabricates data. This module adds what the long-running OpenHydroNet server
needs on top of it:

* the frontend data-directory convention (``$OPENHYDRONET_WEATHER_DATA_DIR`` or
  ``~/.cache/openhydronet/weather``);
* an advisory ``sync.lock`` so two server instances on one host never run the
import tempfile
  same synchronisation twice;
* failure isolation between weather models (dynamical.org, ECMWF Open Data,
  NOAA PSL): a failed model is recorded in ``sync_status.json`` and does not
  stop the remaining models from refreshing;
* a subprocess runner plus a daemon thread so the periodic check keeps large
  arrays out of the HTTP server process.

Run ``python -m frontend.weather_sync --data-root <dir>`` for one manual sync.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

from multimet.weather_fetcher.config import (
    DEFAULT_SYNC_MODELS,
    DYNAMICAL_MODELS,
    MSLP_OFFSET_HPA,
    RUN_METADATA_FILE,
    SOURCE_LABELS,
    SYNC_STATUS_FILE,
    output_lead_hours,
    to_stored_units,
)
from multimet.weather_fetcher.sync import (
    IncompleteRunError,
    aggregate_rates,
    current_models_metadata,
    open_dynamical_catalog,
    sync_all_models,
)
from multimet.weather_fetcher.sync import (
    current_run_dir as _current_run_dir_core,
)
from multimet.weather_fetcher.sync import (
    read_sync_status as _read_sync_status_core,
)
from multimet.weather_fetcher.sync import (
    utc_now_str as _utc_now_str,
)
from multimet.weather_fetcher.sync import (
    write_json_atomic as _write_json_atomic,
)
from utils.file_paths import (
    OPENHYDRONET_WEATHER_CACHE_DIR,
    OPENHYDRONET_WEATHER_DATA_DIR_ENV,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CHECK_INTERVAL_MINUTES = 60
SYNC_INTERVAL_HOURS = 6
SUBPROCESS_TIMEOUT_S = 45 * 60
SYNC_LOG_FILE = 'sync.log'

__all__ = [
    'CHECK_INTERVAL_MINUTES',
    'MSLP_OFFSET_HPA',
    'RUN_METADATA_FILE',
    'IncompleteRunError',
    'aggregate_rates',
    'current_run_dir',
    'find_sync_python',
    'output_lead_hours',
    'read_sync_status',
    'run_sync_subprocess',
    'start_background_sync',
    'sync_latest',
    'to_stored_units',
    'weather_data_root',
]

logger = logging.getLogger(__name__)

_LOCK_FILE = 'sync.lock'
_LOG_ROTATE_BYTES = 2 << 20
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
_SYNC_PYTHON_PROBE = 'import icechunk, numpy, pystac, scipy, xarray'
_SYNC_PYTHON_PROBE_TIMEOUT_S = 120


def weather_data_root() -> str:
    """Return local-disk root of downloaded runs used by the frontend server.

    ``$OPENHYDRONET_WEATHER_DATA_DIR`` overrides the default of
    ``~/.cache/openhydronet/weather``. The library itself takes the
    directory explicitly; this convention exists only for the server process.
    """
    env = os.environ.get(OPENHYDRONET_WEATHER_DATA_DIR_ENV, '').strip()
    if env:
        return str(Path(env).expanduser().resolve())
    return str(OPENHYDRONET_WEATHER_CACHE_DIR)


def current_run_dir(root: str | None = None) -> str | None:
    """Return resolved directory the ``current`` symlink points at."""
    resolved = _current_run_dir_core(root or weather_data_root())
    return str(resolved) if resolved is not None else None


def read_sync_status(root: str | None = None) -> dict[str, Any]:
    """Read ``sync_status.json`` from ``root`` (default: weather_data_root)."""
    return _read_sync_status_core(root or weather_data_root())


def _provider_of(model_key: str) -> str:
    """Return upstream provider label of a model."""
    if model_key not in DYNAMICAL_MODELS:
        msg = (
            f'Unsupported weather model {model_key!r}. Supported:'
            f' {list(DYNAMICAL_MODELS)}'
        )
        raise ValueError(msg)
    return SOURCE_LABELS[DYNAMICAL_MODELS[model_key]['source']]


def _model_groups(models: Sequence[str]) -> list[list[str]]:
    """Validate and split models into one isolated single-model group each."""
    groups: list[list[str]] = []
    for model_key in models:
        _provider_of(model_key)
        groups.append([model_key])
    return groups


def _group_needs_catalog(
    group: Sequence[str],
    *,
    has_catalog: bool,
    has_open_dataset: bool,
) -> bool:
    """Return True when a group requires opening the dynamical STAC catalog."""
    return (
        not has_catalog
        and not has_open_dataset
        and any(
            DYNAMICAL_MODELS[m]['source'] in ('dynamical', 'dynamical_analysis')
            for m in group
        )
    )


def _remove_own_partial_dirs(root: str, log: Callable[[str], None]) -> None:
    """Delete staging directories left by a failed sync of this process.

    ``sync_all_models`` names its staging directory ``<run>.<pid>.partial`` and
    cannot clean it up when an extraction raises (the library contains no
    exception handling by design). Removing our own leftovers here keeps later
    checks from treating them as a live synchronisation of this process.
    """
    suffix = f'.{os.getpid()}.partial'
    runs_dir = Path(root) / 'runs'
    if not runs_dir.is_dir():
        return
    for item in runs_dir.iterdir():
        if item.is_dir() and item.name.endswith(suffix) and item.exists():
            log(f'Removing staging directory {item.name} after a failed sync')
            shutil.rmtree(item)


def _models_summary(root: str) -> dict[str, dict[str, Any]]:
    """Return per-model summary of the published run."""
    _, current = current_models_metadata(root)
    return {
        key: {
            'init_time': entry.get('init_time'),
            'downloaded_utc': entry.get('downloaded_utc'),
            'source': entry.get('source'),
        }
        for key, entry in current.items()
    }


def _merge_outcome(
    status: dict[str, Any],
    updated: list[str],
    errors: dict[str, str],
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Set ``last_result`` / ``message`` from merged per-model outcome."""
    downloaded = ', '.join(
        f'{key} {(models.get(key) or {}).get("init_time")}' for key in updated
    )
    failed = '; '.join(f'{key}: {msg}' for key, msg in errors.items())
    if updated and not errors:
        status['last_result'] = 'updated'
        status['message'] = f'Downloaded new run: {downloaded}'
    elif updated:
        status['last_result'] = 'partial'
        status['message'] = (
            f'Downloaded new run: {downloaded}; failed: {failed}'
        )
    elif errors:
        status['last_result'] = 'error'
        status['message'] = failed
    else:
        status['last_result'] = 'up_to_date'
        status['message'] = 'All models already have the newest published run.'
    return status


def _try_acquire_lock(lock_fd: int) -> bool:
    """Attempt a non-blocking exclusive lock on ``lock_fd``."""
    if os.name == 'nt':
        os.lseek(lock_fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def sync_latest(  # noqa: PLR0913
    root: str | None = None,
    models: Sequence[str] | None = None,
    *,
    force: bool = False,
    log: Callable[[str], None] = logger.info,
    catalog: Any = None,  # noqa: ANN401
    open_dataset: Callable[[Any, str], Any] | None = None,
    cpc_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Refresh every selected model whose newest upstream run changed.

    Models are synchronised one by one through
    ``multimet.weather_fetcher.sync.sync_all_models``. An exception raised while
    synchronising one model (catalog unreachable, GRIB download failed, ...) is
    recorded for that model in ``errors`` and the remaining models are still
    processed; nothing is retried with substitute data.

    Args:
        root: Data directory (default ``weather_data_root()``).
        models: Model keys (default ``DEFAULT_SYNC_MODELS``).
        force: Re-download even when the newest run is already published.
        log: Line logger callback.
        catalog: Optional pre-opened STAC catalog (or test stub).
        open_dataset: Optional ``(catalog, dataset_id) -> xr.Dataset`` opener.
        cpc_cache_dir: NOAA PSL CPC NetCDF cache (default ``<root>/cpc_cache``).

    Returns:
        The merged status written to ``<root>/sync_status.json``;
        ``last_result`` is ``updated``, ``partial``, ``error``, ``up_to_date``
        or ``busy``.

    Raises:
        ValueError: If ``models`` contains an unsupported key.
    """
    resolved_root = str(root or weather_data_root())
    runs_path = Path(resolved_root) / 'runs'
    runs_path.mkdir(parents=True, exist_ok=True)
    selected = list(models) if models is not None else list(DEFAULT_SYNC_MODELS)
    groups = _model_groups(selected)
    status_path = str(Path(resolved_root) / SYNC_STATUS_FILE)
    lock_path = Path(resolved_root) / _LOCK_FILE

    with lock_path.open('a+', encoding='utf-8') as lock_file:
        if not _try_acquire_lock(lock_file.fileno()):
            status = read_sync_status(resolved_root)
            status.update(
                {
                    'last_check_utc': _utc_now_str(),
                    'last_result': 'busy',
                    'message': (
                        f'Another weather sync holds {lock_path}; skipping.'
                    ),
                }
            )
            log(status['message'])
            return status

        updated: list[str] = []
        errors: dict[str, str] = {}
        active_catalog = catalog
        catalog_error_msg: str | None = None
        for group in groups:
            needs_catalog = _group_needs_catalog(
                group,
                has_catalog=active_catalog is not None,
                has_open_dataset=open_dataset is not None,
            )
            if needs_catalog and catalog_error_msg is not None:
                log(f'[{", ".join(group)}] sync failed: {catalog_error_msg}')
                for model_key in group:
                    errors[model_key] = catalog_error_msg
                continue
            try:
                if needs_catalog:
                    active_catalog = open_dynamical_catalog()
                result = sync_all_models(
                    data_dir=resolved_root,
                    models=group,
                    force=force,
                    log=log,
                    catalog=active_catalog,
                    open_dataset=open_dataset,
                    cpc_cache_dir=cpc_cache_dir,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                # Model isolation: record failure per model in status.
                _remove_own_partial_dirs(resolved_root, log)
                message = f'{type(exc).__name__}: {exc}'
                if needs_catalog and active_catalog is None:
                    catalog_error_msg = message
                log(f'[{", ".join(group)}] sync failed: {message}')
                for model_key in group:
                    errors[model_key] = message
                continue
            if result.get('last_result') == 'busy':
                return result
            updated.extend(result.get('updated_models', []))
            errors.update(result.get('errors', {}))

        status = read_sync_status(resolved_root)
        sources = {key: _provider_of(key) for key in selected}
        status.update(
            {
                'last_check_utc': _utc_now_str(),
                'check_interval_minutes': CHECK_INTERVAL_MINUTES,
                'source': ', '.join(dict.fromkeys(sources.values())),
                'sources': sources,
                'updated_models': updated,
                'errors': errors,
                'models': _models_summary(resolved_root),
            }
        )
        _merge_outcome(status, updated, errors, status['models'])
        _write_json_atomic(status_path, status)
        log(f'{status["last_result"]}: {status["message"]}')
        return status


def find_sync_python() -> str | None:
    """Return interpreter able to import the sync dependencies, or ``None``.

    Candidates, in order: ``$OPENHYDRONET_WEATHER_SYNC_PYTHON``, the running
    interpreter, ``/usr/bin/python3`` and ``python3`` on ``PATH``.
    """
    candidates = [
        os.environ.get('OPENHYDRONET_WEATHER_SYNC_PYTHON'),
        sys.executable,
        '/usr/bin/python3',
        shutil.which('python3'),
    ]
    seen: set[str] = set()
    for python in candidates:
        if not python or python in seen or not Path(python).is_file():
            continue
        if not os.access(python, os.X_OK):
            continue
        seen.add(python)
        probe = subprocess.run(  # noqa: S603
            [python, '-c', _SYNC_PYTHON_PROBE],
            capture_output=True,
            timeout=_SYNC_PYTHON_PROBE_TIMEOUT_S,
            check=False,
        )
        if probe.returncode == 0:
            return python
    return None


def _append_log(root: str, text: str) -> None:
    """Append to ``<root>/<SYNC_LOG_FILE>`` (rotated once past 2 MiB)."""
    path = Path(root) / SYNC_LOG_FILE
    if path.is_file() and path.stat().st_size > _LOG_ROTATE_BYTES:
        path.replace(Path(f'{path}.1'))
    with path.open('a', encoding='utf-8') as f_log:
        f_log.write(text)


def run_sync_subprocess(
    root: str | None = None,
    python: str | None = None,
    models: Sequence[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any] | bool:
    """Run one sync in a separate process and return the resulting status."""
    resolved_root = str(Path(root or weather_data_root()).resolve())
    Path(resolved_root).mkdir(parents=True, exist_ok=True)
    python = python or find_sync_python()
    if not python:
        status = read_sync_status(resolved_root)
        status.update(
            {
                'last_check_utc': _utc_now_str(),
                'check_interval_minutes': CHECK_INTERVAL_MINUTES,
                'last_result': 'error',
                'message': (
                    'No Python interpreter able to import icechunk, pystac,'
                    ' scipy and xarray was found (set'
                    ' OPENHYDRONET_WEATHER_SYNC_PYTHON).'
                ),
            }
        )
        _write_json_atomic(str(Path(resolved_root) / SYNC_STATUS_FILE), status)
        return status
    cmd = [python, '-m', 'frontend.weather_sync', '--data-root', resolved_root]
    if models is not None:
        cmd.extend(['--models', ','.join(models)])
    if force:
        cmd.append('--force')
    started = _utc_now_str()

    with (
        tempfile.TemporaryFile(mode='w+', encoding='utf-8') as out_f,
        tempfile.TemporaryFile(mode='w+', encoding='utf-8') as err_f,
    ):
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=out_f,
            stderr=err_f,
            cwd=_REPO_ROOT,
        )
        start_time = time.monotonic()
        while proc.poll() is None:
            if time.monotonic() - start_time > SUBPROCESS_TIMEOUT_S:
                proc.kill()
                proc.wait()
                logger.error(
                    'Weather sync subprocess timed out after %s seconds',
                    SUBPROCESS_TIMEOUT_S,
                )
                return False
            time.sleep(0.5)

        out_f.seek(0)
        stdout = out_f.read()
        err_f.seek(0)
        stderr = err_f.read()
    _append_log(
        resolved_root,
        f'\n=== {started} {" ".join(cmd)} (exit {proc.returncode})\n'
        f'{stdout}{stderr}',
    )
    status = read_sync_status(resolved_root)
    if (status.get('last_check_utc') or '') < started:
        if proc.returncode == 0:
            status.update(
                {
                    'last_check_utc': _utc_now_str(),
                    'check_interval_minutes': CHECK_INTERVAL_MINUTES,
                    'last_result': 'busy',
                    'message': 'Another weather sync is in progress.',
                }
            )
        else:
            stderr_lines = (stderr or '').strip().splitlines()
            detail = (
                stderr_lines[-1]
                if stderr_lines
                else f'Sync subprocess exited with code {proc.returncode}'
            )
            status.update(
                {
                    'last_check_utc': _utc_now_str(),
                    'check_interval_minutes': CHECK_INTERVAL_MINUTES,
                    'last_result': 'error',
                    'message': detail,
                }
            )
            _write_json_atomic(
                str(Path(resolved_root) / SYNC_STATUS_FILE), status
            )
    return status


_SYNC_THREAD: threading.Thread | None = None


def start_background_sync(
    on_finished: Callable[[], None] | None = None,
    interval_minutes: float | None = None,
    first_delay_s: float = 15,
) -> threading.Thread | None:
    """Start the periodic check in a daemon thread (once per process).

    ``OPENHYDRONET_WEATHER_SYNC=0`` disables it;
    ``OPENHYDRONET_WEATHER_SYNC_MINUTES`` or ``interval_minutes`` override
    ``CHECK_INTERVAL_MINUTES``.
    """
    global _SYNC_THREAD  # noqa: PLW0603
    if _SYNC_THREAD is not None:
        return _SYNC_THREAD
    sync_enabled = os.environ.get('OPENHYDRONET_WEATHER_SYNC', '1')
    if sync_enabled == '0':
        return None
    env_minutes = os.environ.get('OPENHYDRONET_WEATHER_SYNC_MINUTES')
    interval_s = 60.0 * float(
        interval_minutes or env_minutes or CHECK_INTERVAL_MINUTES
    )

    def loop() -> None:
        time.sleep(first_delay_s)
        python = find_sync_python()
        while True:
            status = run_sync_subprocess(python=python)
            logger.info(
                '[WeatherSync] %s: %s',
                status.get('last_result'),
                status.get('message'),
            )
            if on_finished is not None:
                on_finished()
            time.sleep(interval_s)

    _SYNC_THREAD = threading.Thread(
        target=loop, name='weather-sync', daemon=True
    )
    _SYNC_THREAD.start()
    return _SYNC_THREAD


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Synchronise the newest weather model runs into the frontend data'
            ' directory (one pass, then exit).'
        )
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help=(
            'Root directory for downloaded runs (default:'
            ' $OPENHYDRONET_WEATHER_DATA_DIR or'
            ' ~/.cache/openhydronet/weather).'
        ),
    )
    parser.add_argument(
        '--cpc-cache-dir',
        type=str,
        default=None,
        help='NOAA PSL CPC NetCDF cache (default: <data-root>/cpc_cache).',
    )
    parser.add_argument(
        '--models',
        type=str,
        default='',
        help='Comma-separated subset of: ' + ', '.join(DYNAMICAL_MODELS),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Download even if the newest run is already published.',
    )
    parser.add_argument(
        '--fetch-latest',
        action='store_true',
        help=(
            'Accepted for backwards compatibility; a sync is always performed.'
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one synchronisation; exit code 0 unless a model failed."""
    args = _build_parser().parse_args(argv)
    result = sync_latest(
        root=args.data_root,
        models=[m.strip() for m in args.models.split(',') if m.strip()] or None,
        force=args.force,
        log=lambda msg: sys.stdout.write(f'{msg}\n'),
        cpc_cache_dir=args.cpc_cache_dir,
    )
    ok_results = ('updated', 'up_to_date', 'busy')
    return 0 if result.get('last_result') in ok_results else 1


if __name__ == '__main__':
    sys.exit(main())
