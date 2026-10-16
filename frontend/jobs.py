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
"""Background job management and asynchronous task queue for OpenHydroNet Web.

Supports asynchronous background task submission, polling, progress tracking,
and automated email notifications upon job completion.
"""

import concurrent.futures
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from frontend.notifier import send_email_notification
from utils.file_paths import DEFAULT_LOCAL_SERVER_URL

logger = logging.getLogger(__name__)


_JOB_THREAD_LOCAL = threading.local()


class JobCancelledError(RuntimeError):
    """Raised when a background job is cancelled by the user."""


def check_current_job_cancelled() -> None:
    """Raise ``JobCancelledError`` if the active thread's job was cancelled."""
    cancel_ev: threading.Event | None = getattr(
        _JOB_THREAD_LOCAL, 'cancel_event', None
    )
    if cancel_ev is not None and cancel_ev.is_set():
        raise JobCancelledError('Background job cancelled by user.')


class JobManager:
    """Manage background processing jobs and completion notifications."""

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize the background thread pool and job registry.

        Args:
            max_workers: Maximum number of worker threads in the pool.
        """
        self.max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, concurrent.futures.Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit_job(
        self,
        job_type: str,
        task_fn: Callable[..., dict[str, Any]],
        *args: Any,  # noqa: ANN401
        notify_email: str | None = None,
        metadata: dict[str, Any] | None = None,
        server_url: str = DEFAULT_LOCAL_SERVER_URL,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Submit a function to run asynchronously on a background thread.

        Args:
            job_type: Identifier describing the kind of job.
            task_fn: Callable returning a result dictionary.
            *args: Positional arguments forwarded to ``task_fn``.
            notify_email: Optional email address to notify upon completion.
            metadata: Optional metadata dictionary recorded on the job.
            server_url: Base URL of the web server for notification links.
            **kwargs: Keyword arguments forwarded to ``task_fn``.

        Returns:
            Generated job identifier string.
        """
        job_id = f'job_{uuid.uuid4().hex[:10]}'
        now_iso = datetime.now(UTC).isoformat()

        job_record: dict[str, Any] = {
            'job_id': job_id,
            'job_type': job_type,
            'status': 'queued',
            'submitted_at': now_iso,
            'started_at': None,
            'completed_at': None,
            'notify_email': (notify_email or '').strip(),
            'server_url': server_url,
            'metadata': metadata or {},
            'result': None,
            'error': None,
            'email_sent': False,
        }
        cancel_ev = threading.Event()

        with self._lock:
            self._jobs[job_id] = job_record
            self._cancel_events[job_id] = cancel_ev

        fut = self._executor.submit(
            self._execute_job,
            job_id,
            task_fn,
            args,
            kwargs,
            notify_email,
        )
        with self._lock:
            self._futures[job_id] = fut
        logger.info('Submitted background job %s (type=%s)', job_id, job_type)
        return job_id

    def _mark_job_started(
        self, job_id: str
    ) -> tuple[bool, threading.Event | None]:
        """Mark job as running if not already cancelled."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.get('status') == 'cancelled':
                return False, None
            cancel_ev = self._cancel_events.get(job_id)
            if cancel_ev is not None and cancel_ev.is_set():
                job['status'] = 'cancelled'
                job['completed_at'] = datetime.now(UTC).isoformat()
                return False, cancel_ev
            job['status'] = 'running'
            job['started_at'] = datetime.now(UTC).isoformat()
            return True, cancel_ev

    def _execute_job(
        self,
        job_id: str,
        task_fn: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        notify_email: str | None,
    ) -> None:
        """Execute the task and notify the recipient on completion."""
        can_run, cancel_ev = self._mark_job_started(job_id)
        if not can_run:
            return

        _JOB_THREAD_LOCAL.cancel_event = cancel_ev
        try:
            result = task_fn(*args, **kwargs)
        except JobCancelledError as exc:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]['status'] = 'cancelled'
                    self._jobs[job_id]['completed_at'] = datetime.now(
                        UTC
                    ).isoformat()
                    self._jobs[job_id]['error'] = str(exc)
            return
        except Exception as exc:
            with self._lock:
                if job_id in self._jobs:
                    if self._jobs[job_id].get('status') != 'cancelled':
                        self._jobs[job_id]['status'] = 'failed'
                        self._jobs[job_id]['error'] = str(exc)
                    self._jobs[job_id]['completed_at'] = datetime.now(
                        UTC
                    ).isoformat()
            raise
        finally:
            _JOB_THREAD_LOCAL.cancel_event = None

        completed_at = datetime.now(UTC).isoformat()

        with self._lock:
            if self._jobs[job_id].get('status') == 'cancelled':
                return
            self._jobs[job_id]['status'] = 'completed'
            self._jobs[job_id]['completed_at'] = completed_at
            self._jobs[job_id]['result'] = result

        if notify_email:
            job_info = dict(self._jobs[job_id])
            email_sent = send_email_notification(
                recipient_email=notify_email,
                job_info=job_info,
                result=result,
            )
            with self._lock:
                self._jobs[job_id]['email_sent'] = email_sent

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """Cancel a queued or running background job.

        Args:
            job_id: Job identifier to cancel.

        Returns:
            Updated job record dictionary, or None if not found.
        """
        now_iso = datetime.now(UTC).isoformat()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            cancel_ev = self._cancel_events.get(job_id)
            if cancel_ev is not None:
                cancel_ev.set()
            fut = self._futures.get(job_id)
            if fut is not None and not fut.done():
                fut.cancel()
            if job.get('status') in ('queued', 'running'):
                job['status'] = 'cancelled'
                job['completed_at'] = now_iso
                job['error'] = 'Cancelled by user'
            return dict(job)

    def cancel_all_active_jobs(self) -> list[str]:
        """Cancel all currently queued or running background jobs.

        Returns:
            List of cancelled job IDs.
        """
        now_iso = datetime.now(UTC).isoformat()
        cancelled_ids: list[str] = []
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.get('status') in ('queued', 'running'):
                    cancel_ev = self._cancel_events.get(job_id)
                    if cancel_ev is not None:
                        cancel_ev.set()
                    fut = self._futures.get(job_id)
                    if fut is not None and not fut.done():
                        fut.cancel()
                    job['status'] = 'cancelled'
                    job['completed_at'] = now_iso
                    job['error'] = 'Cancelled by user'
                    cancelled_ids.append(job_id)
        return cancelled_ids

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve status and results for a job.

        Args:
            job_id: Job identifier to look up.

        Returns:
            Copy of the job record dictionary, or None if not found.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return dict(job)
        return None

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent jobs ordered from newest to oldest.

        Args:
            limit: Maximum number of jobs to return.

        Returns:
            List of job dictionaries.
        """
        with self._lock:
            all_jobs = [dict(j) for j in self._jobs.values()]
        all_jobs.sort(key=lambda x: x['submitted_at'], reverse=True)
        return all_jobs[:limit]


_GLOBAL_JOB_MANAGER: JobManager | None = None


def get_job_manager() -> JobManager:
    """Return the singleton JobManager instance."""
    global _GLOBAL_JOB_MANAGER  # noqa: PLW0603
    if _GLOBAL_JOB_MANAGER is None:
        _GLOBAL_JOB_MANAGER = JobManager()
    return _GLOBAL_JOB_MANAGER
