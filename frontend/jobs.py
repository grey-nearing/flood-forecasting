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

        with self._lock:
            self._jobs[job_id] = job_record

        self._executor.submit(
            self._execute_job,
            job_id,
            task_fn,
            args,
            kwargs,
            notify_email,
        )
        logger.info('Submitted background job %s (type=%s)', job_id, job_type)
        return job_id

    def _execute_job(
        self,
        job_id: str,
        task_fn: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        notify_email: str | None,
    ) -> None:
        """Execute the task and notify the recipient on completion."""
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id]['status'] = 'running'
            self._jobs[job_id]['started_at'] = datetime.now(UTC).isoformat()

        result = task_fn(*args, **kwargs)
        completed_at = datetime.now(UTC).isoformat()

        with self._lock:
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
