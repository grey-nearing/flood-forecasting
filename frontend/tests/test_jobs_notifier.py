"""Unit tests for background JobManager and email notification formatting."""

from unittest import mock
try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest

from frontend.jobs import JobManager, get_job_manager
from frontend.notifier import find_sendgmr_binary, format_extraction_email, send_email_notification


class JobsAndNotifierTest(absltest.TestCase):

  def test_format_extraction_email(self):
    """Verifies that plain-text and HTML emails contain job details, basin counts, and local paths."""
    job_info = {
        "job_id": "job_test_12345",
        "server_url": "http://localhost:8080",
    }
    result = {
        "status": "success",
        "weather_source": "cpc",
        "weather_source_name": "NOAA CPC Global Precipitation",
        "batch": True,
        "basins_extracted_count": 5,
        "total_basins_in_master": 5,
        "n_timesteps": 16608,
        "start_date": "1979-01-01",
        "end_date": "2024-06-20",
        "master_zarr_path": "/google/src/test/historical_training_master.zarr",
        "master_size_mb": 12.45,
    }

    subject, text_body, html_body = format_extraction_email(job_info, result)

    self.assertIn("Historical Extraction SUCCESS", subject)
    self.assertIn("NOAA CPC Global Precipitation", subject)
    self.assertIn("job_test_12345", text_body)
    self.assertIn("/google/src/test/historical_training_master.zarr", text_body)
    self.assertIn("1979-01-01 to 2024-06-20", text_body)
    self.assertIn("12.45 MB", text_body)
    self.assertIn("job_test_12345", html_body)
    self.assertIn("http://localhost:8080", html_body)

  def test_send_email_notification_invalid_email(self):
    """Verifies that invalid or empty email addresses safely return False."""
    self.assertFalse(send_email_notification("", {}, {}))
    self.assertFalse(send_email_notification("not-an-email", {}, {}))

  @mock.patch("frontend.notifier.find_sendgmr_binary")
  @mock.patch("subprocess.run")
  def test_send_email_notification_via_sendgmr(self, mock_run, mock_find):
    """Verifies send_email_notification invokes sendgmr with expected flags."""
    mock_find.return_value = "/google/bin/releases/gws-sre/files/sendgmr/sendgmr"
    mock_run.return_value = mock.MagicMock(returncode=0, stderr="")

    job_info = {"job_id": "job_123"}
    result = {"status": "success", "weather_source": "era5"}
    success = send_email_notification("gsnearing@google.com", job_info, result)

    self.assertTrue(success)
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    self.assertEqual(args[0], "/google/bin/releases/gws-sre/files/sendgmr/sendgmr")
    self.assertIn("-to=gsnearing@google.com", args)

  def test_job_manager_async_execution(self):
    """Verifies JobManager submits, executes asynchronously, and tracks completion."""
    manager = JobManager(max_workers=2)

    def sample_task(val):
      return {
          "status": "success",
          "val": val * 2,
          "weather_source": "test_src",
      }

    job_id = manager.submit_job(
        job_type="test_job",
        task_fn=sample_task,
        val=21,
        notify_email=None,
    )

    self.assertTrue(job_id.startswith("job_"))
    job_before = manager.get_job(job_id)
    self.assertIsNotNone(job_before)
    self.assertIn(job_before["status"], ["queued", "running", "completed"])

    # Wait for completion
    import time
    for _ in range(20):
      job_cur = manager.get_job(job_id)
      if job_cur["status"] == "completed":
        break
      time.sleep(0.1)

    job_after = manager.get_job(job_id)
    self.assertEqual(job_after["status"], "completed")
    self.assertEqual(job_after["result"]["val"], 42)


if __name__ == "__main__":
  absltest.main()
