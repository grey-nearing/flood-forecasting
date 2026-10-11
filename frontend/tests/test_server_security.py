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
"""Security tests for static asset serving and path traversal prevention."""

from __future__ import annotations

import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

import pytest

from frontend import server as server_mod
from frontend.server import EarthkitHydroHandler, OpenHydroNetHandler


@pytest.mark.unit
class ServerSecurityTest(unittest.TestCase):
    """Verify static file serving blocks directory traversal."""

    def setUp(self) -> None:
        """Create a temporary static directory and start the HTTP server."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.static_dir = self.root_dir / 'static'
        self.static_dir.mkdir(parents=True, exist_ok=True)
        (self.static_dir / 'public.txt').write_text(
            'public_content',
            encoding='utf-8',
        )
        self.secret_file = self.root_dir / 'static_secret.txt'
        self.secret_file.write_text('secret_content', encoding='utf-8')

        self.patcher = mock.patch.object(
            server_mod,
            'STATIC_DIR',
            self.static_dir,
        )
        self.patcher.start()

        self.server = HTTPServer(('127.0.0.1', 0), OpenHydroNetHandler)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()

    def tearDown(self) -> None:
        """Stop the test HTTP server and clean up the temporary directory."""
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_static_path_traversal(self) -> None:
        """Ensure requests escaping `STATIC_DIR` return HTTP 404."""
        assert EarthkitHydroHandler is OpenHydroNetHandler

        public_url = f'http://127.0.0.1:{self.port}/static/public.txt'
        with urllib.request.urlopen(public_url) as resp:
            assert resp.status == 200  # noqa: PLR2004
            assert resp.read().decode('utf-8') == 'public_content'

        traversal_url = (
            f'http://127.0.0.1:{self.port}/static/../static_secret.txt'
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(traversal_url)
        assert exc_info.value.code == 404  # noqa: PLR2004


if __name__ == '__main__':
    unittest.main()
