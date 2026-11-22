"""The UI must reject unsupported local runs and persist valid choices."""

import json
import tempfile
import threading
import sys
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # the code is in src/

import app
import simulator_settings as settings


class SettingsTests(unittest.TestCase):
    def test_catalog_and_validation(self):
        self.assertEqual(len(settings.ROBOTS), 9)
        self.assertEqual(len(settings.ROOMS), 6)  # Five Sandbox rooms plus local tabletop.
        self.assertTrue(settings.local_supported(settings.normalize({})))
        self.assertTrue(settings.local_supported(settings.normalize({"room": "kitchen"})))
        self.assertTrue(settings.local_supported(settings.normalize({"robot": "openyam", "room": "kitchen"})))
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


class BatchSeedTests(unittest.TestCase):
    def test_one_seed_is_the_whole_batch(self):
        ten = [app.seed_web.episode_seed(10, i) for i in range(100)]
        self.assertEqual(ten, [app.seed_web.episode_seed(10, i) for i in range(100)])  # same seed, same episodes
        self.assertEqual(len(set(ten)), 100)                                          # each episode its own layout
        eleven = {app.seed_web.episode_seed(11, i) for i in range(100)}
        self.assertFalse(set(ten) & eleven)                                           # batch 11 shares none
        self.assertTrue(all(0 <= s < 1_000_000 for s in ten))                         # train seeds


class KeyTests(unittest.TestCase):
    def test_only_dream_keys_are_taken(self):
        self.assertIsNone(app.key_problem("dl_live_" + "a1B2_c3-D4e5F6g7h8"))
        for bad in ["--default-scene", "default-scene", "sk-abc123456789012345678", "dl_short",
                    "dl_live_abc def ghi jkl mno", "dl_live_abc$%^&*()1234567890", "put apple into the bowl"]:
            self.assertIsNotNone(app.key_problem(bad), bad)


if __name__ == "__main__":
    unittest.main()
