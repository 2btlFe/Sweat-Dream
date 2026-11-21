"""Local Dream Labs validator checks for generated and converted datasets."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import finetune_pipeline as pipeline


class FineTunePipelineTests(unittest.TestCase):
    def test_dummy_dataset_passes_official_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dummy"
            pipeline.build(output, [pipeline.synthetic_episode()])
            self.assertEqual(pipeline.validate(output).episodes, 1)

    def test_labeled_recording_is_converted_and_failure_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "member1"
            episode = root / "episodes" / "example"
            run = root / "runs" / "example"
            episode.mkdir(parents=True)
            run.mkdir(parents=True)
            dummy = pipeline.synthetic_episode()
            times = dummy["t"]
            np.savez(episode / "steps.npz", t=times, state=dummy["state"],
                     action=dummy["action"], frame_t=dummy["frame_t"])
            (episode / "meta.json").write_text(json.dumps({"task": dummy["task"], "robot": "yam"}))
            (run / "meta.json").write_text(json.dumps({"episode": "example"}))
            (run / "label.json").write_text(json.dumps({"label": "success"}))
            for camera, role in pipeline.CAMERAS.items():
                pipeline.write_video(episode / f"{camera}.mp4", dummy["frames"][role])
            chosen = pipeline.selected_recordings([root])
            self.assertEqual(len(chosen), 1)
            output = Path(tmp) / "converted"
            pipeline.build(output, [folder for folder, _ in chosen])
            self.assertEqual(pipeline.validate(output).frames, 12)
            (run / "label.json").write_text(json.dumps({"label": "failure"}))
            self.assertEqual(pipeline.selected_recordings([root]), [])


if __name__ == "__main__":
    unittest.main()
