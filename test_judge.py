"""Behavioral checks for saved-trajectory labeling."""

import unittest

import judge


def episode(distance_cm=3, touching=0, final_x=None):
    times = [0, 0.5, 1.0, 1.5, 2.0]
    x = distance_cm / 100
    object_positions = [[x, 0, 0.78] for _ in times]
    if final_x is not None:
        object_positions[-1][0] = final_x
    return {"t": times, "positions": {"apple": object_positions,
            "bowl": [[0, 0, 0.77] for _ in times]},
            "touching": {"apple": [touching for _ in times]}}


META = {"task": "put apple into the bowl", "model_responses": 1, "model_actions": 5}


class JudgeTests(unittest.TestCase):
    def test_rule_changes_change_the_verdict(self):
        trace = episode(distance_cm=8)
        strict = judge.evaluate(META, trace, judge.DEFAULT_RULES)
        loose = judge.evaluate(META, trace, {"distance_cm": 10, "release": True,
                                            "still_seconds": 1})
        self.assertEqual(strict["label"], "failure")
        self.assertEqual(loose["label"], "success")

    def test_release_toggle(self):
        trace = episode(touching=1)
        required = judge.evaluate(META, trace, judge.DEFAULT_RULES)
        optional = judge.evaluate(META, trace, {"distance_cm": 6.5, "release": False,
                                               "still_seconds": 1})
        self.assertEqual(required["reason"], "still_holding")
        self.assertEqual(optional["label"], "success")

    def test_final_window_and_speed(self):
        trace = episode(final_x=0.055)
        verdict = judge.evaluate(META, trace, judge.DEFAULT_RULES)
        self.assertEqual(verdict["label"], "failure")
        self.assertEqual(verdict["reason"], "not_stable")
        self.assertGreaterEqual(verdict["metrics"]["max_final_speed_m_s"], 0.05)

    def test_early_success_then_final_failure(self):
        trace = episode(final_x=0.2)
        self.assertEqual(judge.evaluate(META, trace, judge.DEFAULT_RULES)["label"], "failure")

    def test_still_duration(self):
        trace = episode()
        trace["positions"]["apple"][0][0] = 0.2
        trace["positions"]["apple"][1][0] = 0.2
        verdict = judge.evaluate(META, trace, judge.DEFAULT_RULES)
        self.assertEqual(verdict["label"], "success")
        longer = judge.evaluate(META, trace, {"distance_cm": 6.5, "release": True,
                                              "still_seconds": 1.5})
        self.assertEqual(longer["label"], "failure")

    def test_execution_error_is_invalid(self):
        verdict = judge.evaluate({**META, "error": "UNAUTHORIZED"}, episode(), judge.DEFAULT_RULES)
        self.assertEqual(verdict["label"], "invalid")


if __name__ == "__main__":
    unittest.main()
