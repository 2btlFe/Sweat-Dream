"""The UI must reject unsupported local runs and persist valid choices."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import app
import simulator_settings as settings


class SettingsTests(unittest.TestCase):
    def test_catalog_and_validation(self):
        self.assertEqual(len(settings.ROBOTS), 9)
        self.assertEqual(len(settings.ROOMS), 6)  # Five Sandbox rooms plus local tabletop.
        self.assertTrue(settings.local_supported(settings.normalize({})))
        self.assertTrue(settings.local_supported(settings.normalize({"room": "kitchen"})))
        self.assertFalse(settings.local_supported(settings.normalize({"robot": "piper", "room": "kitchen"})))
        for bad in ({"robot": "unknown"}, {"room": []}, {"seed": -1}, {"seed": True}):
            with self.assertRaises(ValueError):
                settings.normalize(bad)

    def test_http_settings_and_run_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = app.RUNS, app.SETTINGS
            app.RUNS = Path(temporary)
            app.SETTINGS = app.RUNS / "simulator_settings.json"
            server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def request(path, body=None):
                data = None if body is None else json.dumps(body).encode("utf-8")
                req = urllib.request.Request(base + path, data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req) as response:
                    return json.load(response)

            try:
                selected = {"robot": "piper", "room": "kitchen", "layout": "varied",
                            "seed": 42, "lighting": "warm"}
                self.assertEqual(request("/api/simulator-settings", selected)["settings"], selected)
                self.assertEqual(request("/api/simulator-settings")["settings"], selected)
                with self.assertRaises(urllib.error.HTTPError) as result:
                    request("/api/run", {"task": "put apple into the bowl", "seconds": 10})
                self.assertEqual(result.exception.code, 400)
                self.assertEqual(list(app.RUNS.glob("*/meta.json")), [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                app.RUNS, app.SETTINGS = previous


if __name__ == "__main__":
    unittest.main()
