"""Local Dream Labs format and teammate dataset aggregation checks."""

import json
import tempfile
import unittest
from pathlib import Path

import finetune_pipeline as pipeline


class FineTunePipelineTests(unittest.TestCase):
    def test_dummy_passes_official_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dummy"
            pipeline.make_dummy(output)
            self.assertEqual(pipeline.validate(output)["episodes"], 1)

    def test_combine_only_successful_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "member1"
            episode = root / "episodes" / "example"
            pipeline.sample_recording(episode)
            for run_id, label in (("example", "success"), ("failed", "failure")):
                run = root / "runs" / run_id
                run.mkdir(parents=True)
                (run / "meta.json").write_text(json.dumps({"episode": "example", "task": "put apple into the bowl"}))
                (run / "label.json").write_text(json.dumps({"label": label}))
            output = Path(tmp) / "combined"
            pipeline.combine(output, [root])
            self.assertEqual(pipeline.validate(output)["episodes"], 1)
            self.assertEqual(len(pipeline.selected_recordings([root])), 1)


if __name__ == "__main__":
    unittest.main()
