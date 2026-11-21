"""Check that selected episodes include synchronized training signals."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import collect_success
import judge


class CollectionTests(unittest.TestCase):
    def test_pick_instruction_is_judged(self):
        self.assertEqual(judge.parse_task("pick apple into the bowl"), ("apple", "bowl"))
        self.assertEqual(judge.parse_task("pick bottle into the bowl"), ("water_bottle", "bowl"))

    def test_success_export_requires_actions_and_frames(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "frames").mkdir()
            (folder / "frames" / "00000.jpg").write_bytes(b"frame")
            (folder / "frames.json").write_text(json.dumps([{"file": "00000.jpg", "time_s": 0}]))
            trace = {"t": [0, 0.05], "actions": {"left": [[0], [1]], "right": [[0], [1]]}}
            collect_success.validate_training_trace(trace, folder)
            trace["actions"]["right"].pop()
            with self.assertRaises(ValueError):
                collect_success.validate_training_trace(trace, folder)

    def test_collect_exports_success_and_resume_skips_it(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            (home / "collection_rules.json").write_text(json.dumps(judge.DEFAULT_RULES))

            def fake_run(_command, **kwargs):
                folder = Path(kwargs["env"]["RUN_DIR"])
                folder.mkdir(parents=True)
                times = [0, .5, 1, 1.5, 2]
                positions = [[.02, 0, .78] for _ in times]
                trace = {"t": times, "positions": {"apple": positions,
                         "bowl": [[0, 0, .77] for _ in times]},
                         "touching": {"apple": [0] * len(times)},
                         "actions": {"left": [[0] for _ in times], "right": [[0] for _ in times]}}
                meta = {"task": "pick apple into the bowl", "model_responses": 1, "model_actions": 5}
                (folder / "trajectory.json").write_text(json.dumps(trace))
                (folder / "meta.json").write_text(json.dumps(meta))
                (folder / "frames").mkdir()
                (folder / "frames" / "00000.jpg").write_bytes(b"frame")
                (folder / "frames.json").write_text(json.dumps([{"file": "00000.jpg", "time_s": 0}]))
                return SimpleNamespace(returncode=0)

            with patch.object(collect_success, "HERE", home), patch.object(collect_success, "RUNS", home / "runs"), \
                 patch.object(collect_success, "BATCHES", home / "batches"), \
                 patch.dict("os.environ", {"DREAM_API_KEY": "test-only"}), \
                 patch.object(collect_success.subprocess, "run", side_effect=fake_run) as runner:
                arguments = ["--member", "1", "--object", "apple", "--attempts", "1", "--seconds", "2"]
                collect_success.main(arguments)
                collect_success.main(arguments)
                self.assertEqual(runner.call_count, 1)
                collection = home / "batches" / "member-1-apple"
                manifest = json.loads((collection / "manifest.json").read_text())
                self.assertEqual(manifest["attempts"][0]["label"], "success")
                self.assertTrue((home / manifest["attempts"][0]["export_path"] / "trajectory.json").is_file())


if __name__ == "__main__":
    unittest.main()
