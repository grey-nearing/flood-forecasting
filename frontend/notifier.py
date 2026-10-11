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
"""Email notification service for OpenHydroNet Web.

Dispatches job completion notifications via SMTP to user-supplied email
addresses.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from utils.file_paths import (
    DEFAULT_LOCAL_SERVER_URL,
    HISTORICAL_MASTER_ZARR_NAME,
)

logger = logging.getLogger(__name__)


def format_extraction_email(
    job_info: dict[str, Any],
    result: dict[str, Any],
) -> tuple[str, str, str]:
    """Format plain-text and HTML emails for historical weather extraction.

    Args:
        job_info: Metadata dictionary describing the submitted job.
        result: Result dictionary returned by the historical extractor.

    Returns:
        A tuple of ``(subject, text_body, html_body)``.
    """
    source_name = result.get('weather_source_name') or result.get(
        'weather_source',
        'Weather Data',
    )
    status = str(result.get('status', 'completed')).upper()
    basin_count = result.get('basins_extracted_count') or (
        len(result.get('extracted_basins', [])) if result.get('batch') else 1
    )
    basin_id = result.get('basin_id', 'Multiple Basins')
    n_days = result.get('n_timesteps', 'Full Extent')
    master_zarr = result.get('master_zarr_path') or result.get(
        'master_zarr_rel_path',
        HISTORICAL_MASTER_ZARR_NAME,
    )
    master_size = result.get('master_size_mb') or result.get('size_mb', 'N/A')
    total_in_master = result.get('total_basins_in_master', basin_count)
    date_start = result.get('start_date', 'N/A')
    date_end = result.get('end_date', 'N/A')
    job_id = job_info.get('job_id', 'N/A')
    server_url = job_info.get('server_url', DEFAULT_LOCAL_SERVER_URL)

    subject = (
        f'[OpenHydroNet] Historical Extraction {status} - {source_name}'
        f' ({basin_count} Basins)'
    )

    text_body = f"""OpenHydroNet - Historical Weather Extraction Notification
============================================================

Your historical meteorological data extraction has completed on the backend.

Job Details:
------------
Job ID: {job_id}
Status: {status}
Weather Source: {source_name}
Date Range: {date_start} to {date_end} ({n_days} days)
Basins Processed: {basin_count} basin(s) (Target: {basin_id})
Total Basins in Master Archive: {total_in_master}
Master Archive Size: {master_size} MB

Data Storage & Output Locations:
--------------------------------
Master Zarr Store (Local Path):
{master_zarr}

Web UI Dashboard:
{server_url}

The extracted time-series data has been appended to the unified multi-basin
Zarr store compliant with the OpenHydroNet flood forecasting schema.

Best regards,
OpenHydroNet Team
"""

    html_body = (
        '<!DOCTYPE html>\n'
        '<html>\n'
        '<head>\n'
        '  <meta charset="utf-8">\n'
        '  <style>\n'
        '    body { font-family: Roboto, -apple-system, sans-serif;'
        ' line-height: 1.5; color: #202124; background-color: #f8f9fa;'
        ' margin: 0; padding: 24px; }\n'
        '    .card { background: #ffffff; border: 1px solid #dadce0;'
        ' border-radius: 12px; max-width: 640px; margin: 0 auto;'
        ' overflow: hidden; }\n'
        '    .header { background: #1a73e8; color: #ffffff; padding: 24px; }\n'
        '    .badge { display: inline-block; padding: 4px 10px;'
        ' border-radius: 9999px; font-size: 11px; font-weight: 700;'
        ' text-transform: uppercase; background: #1e8e3e; color: #ffffff; }\n'
        '    .title { font-size: 20px; font-weight: 700;'
        ' margin: 12px 0 4px 0; color: #ffffff; }\n'
        '    .subtitle { font-size: 13px; color: #d2e3fc; }\n'
        '    .content { padding: 24px; }\n'
        '    .stats-grid { display: grid; grid-template-columns: 1fr 1fr;'
        ' gap: 12px; margin-bottom: 20px; }\n'
        '    .stat-box { background: #f8f9fa; border: 1px solid #e8eaed;'
        ' padding: 12px 16px; border-radius: 8px; }\n'
        '    .stat-label { font-size: 11px; color: #5f6368;'
        ' font-weight: 600; text-transform: uppercase; }\n'
        '    .stat-val { font-size: 15px; font-weight: 700;'
        ' color: #202124; margin-top: 2px; }\n'
        '    .section-title { font-size: 13px; font-weight: 700;'
        ' color: #202124; text-transform: uppercase;'
        ' margin: 20px 0 8px 0; border-bottom: 1px solid #dadce0;'
        ' padding-bottom: 4px; }\n'
        '    .path-box { background: #f1f3f4; color: #174ea6;'
        ' border: 1px solid #dadce0; padding: 12px 16px;'
        ' border-radius: 8px; font-family: monospace; font-size: 12px;'
        ' word-break: break-all; margin-top: 6px; }\n'
        '    .footer { background: #f8f9fa; border-top: 1px solid #dadce0;'
        ' padding: 16px 24px; font-size: 12px; color: #5f6368;'
        ' text-align: center; }\n'
        '    .btn { display: inline-block; background: #1a73e8;'
        ' color: #ffffff; text-decoration: none; padding: 10px 20px;'
        ' border-radius: 8px; font-weight: 600; font-size: 13px;'
        ' margin-top: 12px; }\n'
        '  </style>\n'
        '</head>\n'
        '<body>\n'
        '  <div class="card">\n'
        '    <div class="header">\n'
        '      <span class="badge">Job Completed</span>\n'
        '      <div class="title">Historical Weather Extraction Finished'
        '</div>\n'
        f'      <div class="subtitle">{source_name} &bull; Job ID:'
        f' <code>{job_id}</code></div>\n'
        '    </div>\n'
        '    <div class="content">\n'
        '      <div class="stats-grid">\n'
        '        <div class="stat-box">\n'
        '          <div class="stat-label">Weather Source</div>\n'
        f'          <div class="stat-val">{source_name}</div>\n'
        '        </div>\n'
        '        <div class="stat-box">\n'
        '          <div class="stat-label">Basins Processed</div>\n'
        f'          <div class="stat-val">{basin_count} Basins</div>\n'
        '        </div>\n'
        '        <div class="stat-box">\n'
        '          <div class="stat-label">Date Range</div>\n'
        f'          <div class="stat-val">{date_start} to {date_end}</div>\n'
        '        </div>\n'
        '        <div class="stat-box">\n'
        '          <div class="stat-label">Master Archive Size</div>\n'
        f'          <div class="stat-val">{master_size} MB'
        f' ({total_in_master} basins)</div>\n'
        '        </div>\n'
        '      </div>\n'
        '      <div class="section-title">Master Zarr Storage Location'
        '</div>\n'
        f'      <div class="path-box">{master_zarr}</div>\n'
        '      <div style="text-align: center; margin-top: 24px;">\n'
        f'        <a href="{server_url}" class="btn">Open Web Platform</a>\n'
        '      </div>\n'
        '    </div>\n'
        '    <div class="footer">\n'
        '      Generated automatically by OpenHydroNet Web Platform\n'
        '    </div>\n'
        '  </div>\n'
        '</body>\n'
        '</html>'
    )

    return subject, text_body, html_body


def send_email_notification(
    recipient_email: str,
    job_info: dict[str, Any],
    result: dict[str, Any],
    sender_email: str | None = None,
) -> bool:
    """Send an email notification for a completed extraction job.

    Args:
        recipient_email: Target email address.
        job_info: Dictionary containing metadata about the job.
        result: Result payload from the extraction task.
        sender_email: Optional sender address.

    Returns:
        True if the email was sent, False if the recipient email is invalid.
    """
    clean_recipient = (recipient_email or '').strip()
    if not clean_recipient or '@' not in clean_recipient:
        logger.warning(
            'Invalid or empty recipient email %r, skipping notification.',
            clean_recipient,
        )
        return False

    subject, text_body, html_body = format_extraction_email(job_info, result)

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email or 'openhydronet-noreply@example.com'
    msg['To'] = clean_recipient

    msg.attach(MIMEText(text_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    host = os.environ.get('SMTP_HOST', 'localhost')
    with smtplib.SMTP(host, 25, timeout=5) as smtp_client:
        smtp_client.sendmail(
            msg['From'],
            [clean_recipient],
            msg.as_string(),
        )
    logger.info(
        'Successfully sent job completion email via SMTP (%s) to %s',
        host,
        clean_recipient,
    )
    return True
