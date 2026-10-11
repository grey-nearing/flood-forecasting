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
"""Unit tests for background JobManager and email notification formatting."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pytest

from frontend.jobs import JobManager
from frontend.notifier import format_extraction_email, send_email_notification
from utils.file_paths import (
    DEFAULT_LOCAL_SERVER_URL,
    HISTORICAL_MASTER_ZARR_NAME,
)


@pytest.mark.unit
class JobsAndNotifierTest(unittest.TestCase):
    """Verify background job execution and SMTP notification formatting."""

    def test_format_extraction_email(self) -> None:
        """Verify plain-text and HTML emails contain job details and paths."""
        zarr_path = str(
            Path(tempfile.gettempdir())
            / 'openhydronet_test'
            / HISTORICAL_MASTER_ZARR_NAME
        )
        job_info = {
            'job_id': 'job_test_12345',
            'server_url': DEFAULT_LOCAL_SERVER_URL,
        }
        result = {
            'status': 'success',
            'weather_source': 'cpc',
            'weather_source_name': 'NOAA CPC Global Precipitation',
            'batch': True,
            'basins_extracted_count': 5,
            'total_basins_in_master': 5,
            'n_timesteps': 16608,
            'start_date': '1979-01-01',
            'end_date': '2024-06-20',
            'master_zarr_path': zarr_path,
            'master_size_mb': 12.45,
        }

        subject, text_body, html_body = format_extraction_email(
            job_info,
            result,
        )

        assert 'Historical Extraction SUCCESS' in subject
        assert 'NOAA CPC Global Precipitation' in subject
        assert 'job_test_12345' in text_body
        assert zarr_path in text_body
        assert '1979-01-01 to 2024-06-20' in text_body
        assert '12.45 MB' in text_body
        assert 'job_test_12345' in html_body
        assert DEFAULT_LOCAL_SERVER_URL in html_body

    def test_send_email_notification_invalid_email(self) -> None:
        """Verify invalid or empty email addresses return False."""
        assert not send_email_notification('', {}, {})
        assert not send_email_notification('not-an-email', {}, {})

    @mock.patch('smtplib.SMTP')
    @mock.patch('os.environ.get')
    def test_send_email_notification_via_smtp(
        self,
        mock_env: mock.MagicMock,
        mock_smtp: mock.MagicMock,
    ) -> None:
        """Verify send_email_notification invokes SMTP with expected fields."""
        mock_env.return_value = 'smtp.example.com'
        mock_smtp_instance = mock.MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

        job_info = {'job_id': 'job_123'}
        result = {'status': 'success', 'weather_source': 'era5'}
        success = send_email_notification('user@example.com', job_info, result)

        assert success
        mock_smtp_instance.sendmail.assert_called_once()
        args = mock_smtp_instance.sendmail.call_args[0]
        assert args[1] == ['user@example.com']

    def test_job_manager_async_execution(self) -> None:
        """Verify JobManager submits and tracks asynchronous completion."""
        manager = JobManager(max_workers=2)
        expected_val = 42

        def sample_task(val: int) -> dict[str, object]:
            return {
                'status': 'success',
                'val': val * 2,
                'weather_source': 'test_src',
            }

        job_id = manager.submit_job(
            job_type='test_job',
            task_fn=sample_task,
            val=21,
            notify_email=None,
        )

        assert job_id.startswith('job_')
        job_before = manager.get_job(job_id)
        assert job_before is not None
        assert job_before['status'] in ('queued', 'running', 'completed')

        for _ in range(20):
            job_cur = manager.get_job(job_id)
            if job_cur is not None and job_cur['status'] == 'completed':
                break
            time.sleep(0.1)

        job_after = manager.get_job(job_id)
        assert job_after is not None
        assert job_after['status'] == 'completed'
        assert job_after['result']['val'] == expected_val


if __name__ == '__main__':
    unittest.main()
