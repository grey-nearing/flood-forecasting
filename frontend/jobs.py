"""Background Job Management and Asynchronous Task Queue for Earthkit Hydro Web.

Supports asynchronous background task submission, polling, progress tracking,
and automated email notifications upon job completion.
"""

import concurrent.futures
from datetime import datetime, timezone
import logging
import threading
from typing import Any, Callable, Dict, List, Optional
import uuid

try:
  from frontend.notifier import send_email_notification
except ImportError:
  try:
    from frontend.notifier import send_email_notification
  except ImportError:
    from notifier import send_email_notification

logger = logging.getLogger(__name__)


class JobManager:
  """Manages background processing jobs and dispatches completion notifications."""

  def __init__(self, max_workers: int = 4):
    self.max_workers = max_workers
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    )
    self._jobs: Dict[str, Dict[str, Any]] = {}
    self._lock = threading.Lock()

  def submit_job(
      self,
      job_type: str,
      task_fn: Callable[..., Dict[str, Any]],
      *args,
      notify_email: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None,
      server_url: str = "http://localhost:8080",
      **kwargs,
  ) -> str:
    """Submits a long-running function to run asynchronously on a background worker thread."""
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    job_record = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",  # queued, running, completed, failed
        "submitted_at": now_iso,
        "started_at": None,
        "completed_at": None,
        "notify_email": (notify_email or "").strip(),
        "server_url": server_url,
        "metadata": metadata or {},
        "result": None,
        "error": None,
        "email_sent": False,
    }

    with self._lock:
      self._jobs[job_id] = job_record

    self._executor.submit(
        self._execute_job, job_id, task_fn, args, kwargs, notify_email
    )
    logger.info("Submitted background job %s (type=%s)", job_id, job_type)
    return job_id

  def _execute_job(
      self,
      job_id: str,
      task_fn: Callable[..., Dict[str, Any]],
      args: tuple,
      kwargs: dict,
      notify_email: Optional[str],
  ):
    """Executes the task and notifies the recipient on completion."""
    with self._lock:
      if job_id not in self._jobs:
        return
      self._jobs[job_id]["status"] = "running"
      self._jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
      result = task_fn(*args, **kwargs)
      completed_at = datetime.now(timezone.utc).isoformat()

      with self._lock:
        self._jobs[job_id]["status"] = "completed"
        self._jobs[job_id]["completed_at"] = completed_at
        self._jobs[job_id]["result"] = result

      # Dispatch Email Notification if recipient email was provided
      if notify_email:
        job_info = dict(self._jobs[job_id])
        email_sent = send_email_notification(
            recipient_email=notify_email,
            job_info=job_info,
            result=result,
        )
        with self._lock:
          self._jobs[job_id]["email_sent"] = email_sent

    except Exception as e:
      logger.exception("Job %s encountered error: %s", job_id, e)
      failed_at = datetime.now(timezone.utc).isoformat()
      with self._lock:
        self._jobs[job_id]["status"] = "failed"
        self._jobs[job_id]["completed_at"] = failed_at
        self._jobs[job_id]["error"] = str(e)

      if notify_email:
        error_result = {
            "status": "failed",
            "error": str(e),
            "weather_source": self._jobs[job_id]["metadata"].get(
                "weather_source", "Historical Weather"
            ),
        }
        send_email_notification(
            recipient_email=notify_email,
            job_info=dict(self._jobs[job_id]),
            result=error_result,
        )

  def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves status and results for a job."""
    with self._lock:
      job = self._jobs.get(job_id)
      if job is not None:
        return dict(job)
    return None

  def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
    """Returns recent jobs."""
    with self._lock:
      all_jobs = [dict(j) for j in self._jobs.values()]
    all_jobs.sort(key=lambda x: x["submitted_at"], reverse=True)
    return all_jobs[:limit]


# Global singleton instance
_GLOBAL_JOB_MANAGER: Optional[JobManager] = None


def get_job_manager() -> JobManager:
  """Returns the global JobManager instance."""
  global _GLOBAL_JOB_MANAGER
  if _GLOBAL_JOB_MANAGER is None:
    _GLOBAL_JOB_MANAGER = JobManager()
  return _GLOBAL_JOB_MANAGER
