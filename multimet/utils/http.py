# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared HTTP download, NASA Earthdata auth, and CMR granule utilities."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import netrc
import os
from pathlib import Path
import time
import urllib.parse

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

DEFAULT_CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def download_http_file(
    url: str,
    dest_path: str | Path,
    *,
    session: requests.Session | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 180,
    min_bytes: int = 1024,
    resource_label: str = "HTTP resource",
) -> str:
  """Atomically downloads an HTTP resource to ``dest_path``.

  Streams the response to a PID- and timestamp-scoped temporary file in the
  destination directory and atomically renames it into ``dest_path`` once size
  checks succeed.

  Args:
    url: Source HTTP/HTTPS URL.
    dest_path: Local destination file path.
    session: Optional ``requests.Session`` (e.g. ``EarthdataSession``).
    headers: Optional HTTP request headers.
    timeout: Request timeout in seconds.
    min_bytes: Minimum required file size in bytes.
    resource_label: Human-readable label included in error messages.

  Returns:
    String path to the downloaded local file.

  Raises:
    PermissionError: If the server returns HTTP 401 or 403.
    FileNotFoundError: If the server returns HTTP 404.
    requests.HTTPError: If the server returns another error status code.
    ValueError: If the downloaded file is smaller than ``min_bytes``.
  """
  dest_str = str(dest_path)
  os.makedirs(os.path.dirname(os.path.abspath(dest_str)), exist_ok=True)
  temp_path = f"{dest_str}.tmp.{os.getpid()}.{time.time_ns()}"

  req_fn = session.get if session is not None else requests.get
  req_kwargs: dict[str, object] = {"stream": True, "timeout": timeout}
  if headers is not None:
    req_kwargs["headers"] = dict(headers)

  with req_fn(url, **req_kwargs) as resp:  # type: ignore[arg-type]
    if resp.status_code in (401, 403):
      raise PermissionError(
          f"{resource_label} returned HTTP {resp.status_code} Unauthorized "
          f"for URL:\n  {url}\nAccess requires authentication."
      )
    if resp.status_code == 404:
      raise FileNotFoundError(
          f"{resource_label} not published (HTTP 404): {url}"
      )
    resp.raise_for_status()
    with open(temp_path, "wb") as out_f:
      for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
          out_f.write(chunk)

  size = os.path.getsize(temp_path)
  if size <= min_bytes:
    os.remove(temp_path)
    raise ValueError(
        f"Downloaded {resource_label} is unexpectedly small "
        f"({size} bytes): {url}"
    )

  os.replace(temp_path, dest_str)
  return dest_str


def check_http_url_exists(
    url: str,
    *,
    session: requests.Session | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 30,
    resource_label: str = "HTTP resource",
) -> bool:
  """Checks whether an HTTP resource exists without downloading its body.

  Args:
    url: Target HTTP/HTTPS URL.
    session: Optional ``requests.Session``.
    headers: Optional HTTP request headers.
    timeout: Request timeout in seconds.
    resource_label: Human-readable label included in error messages.

  Returns:
    ``True`` if the server responds with HTTP 2xx, ``False`` if the server
    responds with HTTP 404.

  Raises:
    PermissionError: If the server returns HTTP 401 or 403.
    requests.HTTPError: If the server returns any other error status code.
  """
  req_fn = session.get if session is not None else requests.get
  req_kwargs: dict[str, object] = {"stream": True, "timeout": timeout}
  if headers is not None:
    req_kwargs["headers"] = dict(headers)

  with req_fn(url, **req_kwargs) as resp:  # type: ignore[arg-type]
    if resp.status_code in (401, 403):
      raise PermissionError(
          f"{resource_label} returned HTTP {resp.status_code} Unauthorized "
          f"for URL:\n  {url}\nAccess requires authentication."
      )
    if resp.status_code == 404:
      return False
    resp.raise_for_status()
    return True



def get_earthdata_credentials_from_netrc(
    netrc_path: str | None = None,
) -> tuple[str | None, str | None]:
  """Reads NASA Earthdata credentials from ``.netrc`` if present.

  Raises if ``netrc_path`` was explicitly provided by the user and cannot be
  parsed or does not exist.
  """
  if netrc_path is not None:
    if not os.path.exists(netrc_path):
      raise FileNotFoundError(
          f"Specified netrc_path does not exist: {netrc_path}"
      )
    parsed = netrc.netrc(netrc_path)
  else:
    default_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(default_path):
      return None, None
    parsed = netrc.netrc(default_path)

  for host in ("urs.earthdata.nasa.gov", "gpm1.gesdisc.eosdis.nasa.gov"):
    auth_info = parsed.authenticators(host)
    if auth_info:
      return auth_info[0], auth_info[2]
  return None, None


class EarthdataSession(requests.Session):
  """Custom ``requests.Session`` that preserves auth across NASA URS redirects."""

  AUTH_HOST = "urs.earthdata.nasa.gov"

  def __init__(
      self,
      username: str | None = None,
      password: str | None = None,
      token: str | None = None,
      netrc_path: str | None = None,
  ):
    super().__init__()
    token = token or os.environ.get("EARTHDATA_TOKEN")
    username = username or os.environ.get("EARTHDATA_USERNAME")
    password = password or os.environ.get("EARTHDATA_PASSWORD")

    if not (username and password) and not token:
      netrc_user, netrc_pass = get_earthdata_credentials_from_netrc(netrc_path)
      if netrc_user and netrc_pass:
        username, password = netrc_user, netrc_pass

    self.token = token
    self.username = username
    self.password = password

    if token:
      self.headers.update({"Authorization": f"Bearer {token}"})
    elif username and password:
      self.auth = (username, password)

  def rebuild_auth(
      self,
      prepared_request: requests.PreparedRequest,
      response: requests.Response,
  ) -> None:
    """Preserves Authorization header across redirects to/from NASA URS."""
    headers = prepared_request.headers
    url = prepared_request.url

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.hostname == self.AUTH_HOST:
      if self.token:
        headers["Authorization"] = f"Bearer {self.token}"
      elif self.username and self.password:
        prepared_request.prepare_auth((self.username, self.password))
      return

    if "Authorization" in headers:
      original_parsed = urllib.parse.urlparse(response.request.url)
      redirect_parsed = urllib.parse.urlparse(url)
      if (
          original_parsed.hostname != redirect_parsed.hostname
          and redirect_parsed.hostname != self.AUTH_HOST
          and original_parsed.hostname != self.AUTH_HOST
      ):
        del headers["Authorization"]

    super().rebuild_auth(prepared_request, response)


def query_cmr_granules(
    short_name: str,
    date: pd.Timestamp,
    version: str = "07",
    cmr_url: str = DEFAULT_CMR_GRANULES_URL,
    timeout: int = 30,
) -> list[str]:
  """Queries NASA's Common Metadata Repository (CMR) for IMERG granule URLs."""
  dt = pd.Timestamp(date)
  start_iso = dt.strftime("%Y-%m-%dT00:00:00Z")
  end_iso = dt.strftime("%Y-%m-%dT23:59:59Z")
  params = {
      "short_name": short_name,
      "version": version,
      "temporal": f"{start_iso},{end_iso}",
      "page_size": 200,
  }
  resp = requests.get(cmr_url, params=params, timeout=timeout)
  resp.raise_for_status()
  entries = resp.json().get("feed", {}).get("entry", [])
  urls: list[str] = []
  for entry in entries:
    for link in entry.get("links", []):
      href = link.get("href", "")
      rel = link.get("rel", "")
      if (
          href.startswith("https://")
          and "data#" in rel
          and not href.endswith((".xml", ".dmrpp", ".s3"))
          and href.endswith((".RT-H5", ".HDF5", ".h5", ".nc4", ".nc"))
      ):
        urls.append(href)
        break
  return sorted(set(urls))
