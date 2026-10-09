import os
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path

from frontend.server import EarthkitHydroHandler

try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest
import urllib.request
import urllib.error

class ServerSecurityTest(absltest.TestCase):
    def setUp(self):
        # We need to start the server
        self.server = HTTPServer(("localhost", 0), EarthkitHydroHandler)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()

    def test_static_path_traversal(self):
        # The STATIC_DIR is frontend/static
        # We'll write a secret file to frontend/static_secret.txt
        # and attempt to read it via /static/../static_secret.txt
        # Wait, if STATIC_DIR is frontend/static, its sibling is frontend/
        ws_root = Path(__file__).resolve().parents[2]
        frontend_dir = ws_root / "frontend"
        secret_file = frontend_dir / "static_secret.txt"
        secret_file.write_text("secret_content")
        
        try:
            url = f"http://localhost:{self.port}/static/../static_secret.txt"
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(url)
            self.assertEqual(cm.exception.code, 404)
        finally:
            secret_file.unlink()

if __name__ == "__main__":
    absltest.main()
