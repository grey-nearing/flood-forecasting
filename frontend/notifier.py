"""Email notification service for Earthkit Hydro Web.

Dispatches job completion notifications via Google Mail Relay (sendgmr)
or fallback SMTP to user-supplied email addresses.
"""

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Search paths for Google Mail Relay (sendgmr) binary
_SENDGMR_CANDIDATES = [
    "/google/bin/releases/gws-sre/files/sendgmr/sendgmr",
    "/google/bin/releases/sre/sendgmr",
    "/usr/local/bin/sendgmr",
    "sendgmr",
]


def find_sendgmr_binary() -> Optional[str]:
  """Locates the sendgmr binary on the system."""
  for candidate in _SENDGMR_CANDIDATES:
    if "/" in candidate:
      if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    else:
      found = shutil.which(candidate)
      if found:
        return found
  return None


def format_extraction_email(
    job_info: Dict[str, Any], result: Dict[str, Any]
) -> tuple[str, str, str]:
  """Formats plain-text and HTML notification emails for historical weather extraction."""
  source_name = result.get("weather_source_name") or result.get(
      "weather_source", "Weather Data"
  )
  status = result.get("status", "completed").upper()
  basin_count = result.get("basins_extracted_count") or (
      len(result.get("extracted_basins", [])) if result.get("batch") else 1
  )
  basin_id = result.get("basin_id", "Multiple Basins")
  n_days = result.get("n_timesteps", "Full Extent")
  master_zarr = result.get("master_zarr_path") or result.get(
      "master_zarr_rel_path", "historical_training_master.zarr"
  )
  master_size = result.get("master_size_mb") or result.get("size_mb", "N/A")
  total_in_master = result.get("total_basins_in_master", basin_count)
  date_start = result.get("start_date", "N/A")
  date_end = result.get("end_date", "N/A")
  job_id = job_info.get("job_id", "N/A")
  server_url = job_info.get("server_url", "http://localhost:8080")

  subject = (
      f"[Earthkit Hydro] Historical Extraction {status} - {source_name}"
      f" ({basin_count} Basins)"
  )

  # Plain text body
  text_body = f"""Earthkit Hydro - Historical Weather Extraction Notification
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
Zarr store along the (basin, date, lead_time) dimensions compliant with the
Google Research flood forecasting schema.

Best regards,
Earthkit Hydro Team
"""

  # HTML body
  html_body = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #1e293b; background-color: #f8fafc; margin: 0; padding: 24px; }}
    .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; max-width: 640px; margin: 0 auto; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }}
    .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #ffffff; padding: 24px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; background: #059669; color: #ffffff; }}
    .title {{ font-size: 20px; font-weight: 700; margin: 12px 0 4px 0; color: #ffffff; }}
    .subtitle {{ font-size: 13px; color: #94a3b8; }}
    .content {{ padding: 24px; }}
    .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
    .stat-box {{ background: #f1f5f9; padding: 12px 16px; border-radius: 8px; }}
    .stat-label {{ font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; }}
    .stat-val {{ font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 2px; }}
    .section-title {{ font-size: 13px; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 8px 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }}
    .path-box {{ background: #0f172a; color: #38bdf8; padding: 12px 16px; border-radius: 8px; font-family: monospace; font-size: 12px; word-break: break-all; margin-top: 6px; }}
    .footer {{ background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 16px 24px; font-size: 12px; color: #64748b; text-align: center; }}
    .btn {{ display: inline-block; background: #d97706; color: #ffffff; text-decoration: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; font-size: 13px; margin-top: 12px; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="badge">Job Completed</span>
      <div class="title">Historical Weather Extraction Finished</div>
      <div class="subtitle">{source_name} &bull; Job ID: <code>{job_id}</code></div>
    </div>
    <div class="content">
      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-label">Weather Source</div>
          <div class="stat-val">{source_name}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Basins Processed</div>
          <div class="stat-val">{basin_count} Basins</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Date Range</div>
          <div class="stat-val">{date_start} to {date_end}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Master Archive Size</div>
          <div class="stat-val">{master_size} MB ({total_in_master} basins)</div>
        </div>
      </div>

      <div class="section-title">Master Zarr Storage Location</div>
      <div class="path-box">{master_zarr}</div>

      <div style="text-align: center; margin-top: 24px;">
        <a href="{server_url}" class="btn">Open Web Platform</a>
      </div>
    </div>
    <div class="footer">
      Generated automatically by Earthkit Hydro Web Platform &bull; Google Research
    </div>
  </div>
</body>
</html>"""

  return subject, text_body, html_body


def send_email_notification(
    recipient_email: str,
    job_info: Dict[str, Any],
    result: Dict[str, Any],
    sender_email: Optional[str] = None,
) -> bool:
  """Sends an email notification for a completed extraction job.

  Args:
      recipient_email: Target email address.
      job_info: Dictionary containing metadata about the job.
      result: Result payload from HistoricalZarrExtractor.
      sender_email: Optional sender address.

  Returns:
      True if the email was successfully sent, False otherwise.
  """
  recipient_email = (recipient_email or "").strip()
  if not recipient_email or "@" not in recipient_email:
    logger.warning(
        "Invalid or empty recipient email '%s', skipping notification.",
        recipient_email,
    )
    return False

  subject, text_body, html_body = format_extraction_email(job_info, result)

  # 1. Try Google Mail Relay (sendgmr)
  sendgmr_bin = find_sendgmr_binary()
  if sendgmr_bin:
    try:
      with (
          tempfile.NamedTemporaryFile(
              mode="w+", encoding="utf-8", delete=False
          ) as txt_f,
          tempfile.NamedTemporaryFile(
              mode="w+", encoding="utf-8", delete=False
          ) as html_f,
      ):
        txt_f.write(text_body)
        txt_f.flush()
        html_f.write(html_body)
        html_f.flush()

        cmd = [
            sendgmr_bin,
            f"-to={recipient_email}",
            f"-subject={subject}",
            f"-body_file={txt_f.name}",
            f"-html_file={html_f.name}",
        ]
        if sender_email:
          cmd.append(f"-from={sender_email}")

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )

        try:
          os.unlink(txt_f.name)
          os.unlink(html_f.name)
        except Exception:
          pass

        if proc.returncode == 0:
          logger.info(
              "Successfully sent job completion email via sendgmr to %s",
              recipient_email,
          )
          return True
        else:
          logger.warning(
              "sendgmr exited with code %s: %s", proc.returncode, proc.stderr
          )
    except Exception as e:
      logger.warning("Error running sendgmr binary: %s", e)

  # 2. Fallback: SMTP localhost or mail.corp.google.com
  try:
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email or "earthkit-hydro-noreply@google.com"
    msg["To"] = recipient_email

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    for host in ["localhost", "mail.corp.google.com"]:
      try:
        with smtplib.SMTP(host, 25, timeout=5) as s:
          s.sendmail(msg["From"], [recipient_email], msg.as_string())
        logger.info(
            "Successfully sent job completion email via SMTP (%s) to %s",
            host,
            recipient_email,
        )
        return True
      except Exception:
        continue
  except Exception as e:
    logger.warning("SMTP fallback failed: %s", e)

  return False
