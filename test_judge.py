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

    def test_custom_prompt_is_judged_on_the_chosen_items(self):
        meta = {**META, "task": "grab the red fruit and drop it somewhere round",
                "judge": {"object": "apple", "target": "bowl"}}
        self.assertEqual(judge.evaluate(meta, episode(), judge.DEFAULT_RULES)["label"], "success")
        unparsed = {**META, "task": "grab the red fruit and drop it somewhere round"}
        self.assertEqual(judge.evaluate(unparsed, episode(), judge.DEFAULT_RULES)["reason"], "unsupported_task")

    def test_two_objects_all_must_be_placed(self):
        trace = episode()
        trace["positions"]["banana"] = [[0.4, 0, 0.78] for _ in trace["t"]]  # far from the plate
        trace["positions"]["plate"] = [[0, 0.3, 0.77] for _ in trace["t"]]
        trace["touching"]["banana"] = [0 for _ in trace["t"]]
        meta = {**META, "task": "put apple in the bowl and banana on the plate",
                "judge": {"pairs": [{"object": "apple", "target": "bowl"}, {"object": "banana", "target": "plate"}]}}
        verdict = judge.evaluate(meta, trace, judge.DEFAULT_RULES)
        self.assertEqual(verdict["label"], "failure")
        self.assertEqual(verdict["reason"], "1/2 placed; banana: object_not_moved")
        self.assertEqual([p["label"] for p in verdict["pairs"]], ["success", "failure"])
        trace["positions"]["banana"] = [[0.02, 0.3, 0.78] for _ in trace["t"]]  # now on the plate
        self.assertEqual(judge.evaluate(meta, trace, judge.DEFAULT_RULES)["label"], "success")

    def test_same_object_twice_is_rejected(self):
        meta = {**META, "judge": {"pairs": [{"object": "apple", "target": "bowl"}, {"object": "apple", "target": "plate"}]}}
        self.assertEqual(judge.evaluate(meta, episode(), judge.DEFAULT_RULES)["reason"], "unsupported_task")

    def test_execution_error_is_invalid(self):
        verdict = judge.evaluate({**META, "error": "UNAUTHORIZED"}, episode(), judge.DEFAULT_RULES)
        self.assertEqual(verdict["label"], "invalid")


def track(xs, zs=None, touching=None, dt=0.5):
    """A trace of the apple moving along x (and z), with the other objects resting far away."""
    n = len(xs)
    zs = zs or [0.79] * n
    times = [i * dt for i in range(n)]
    still = lambda x, y: [[x, y, 0.79] for _ in times]
    return {"t": times,
            "positions": {"apple": [[x, -0.02, z] for x, z in zip(xs, zs)], "orange": still(0.25, 0.15),
                          "banana": still(-0.25, 0.15), "water_bottle": still(0.25, -0.2),
                          "bowl": still(0, -0.2), "plate": still(-0.2, -0.2)},
            "touching": {name: (touching or [0] * n) if name == "apple" else [0] * n
                         for name in ("apple", "orange", "banana", "water_bottle")}}


class TaskTypeTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(judge.classify("put apple in the bowl and banana on the plate"),
                         {"type": "pick_place", "pairs": [{"object": "apple", "target": "bowl"},
                                                          {"object": "banana", "target": "plate"}]})
        self.assertEqual(judge.classify("Move an object to the center of the workspace"),
                         {"type": "move_to_area", "object": "any", "area": "center"})
        self.assertEqual(judge.classify("push the apple to the left side"),
                         {"type": "move_to_area", "object": "apple", "area": "left"})
        self.assertEqual(judge.classify("lift the orange"), {"type": "lift", "object": "orange"})
        self.assertEqual(judge.classify("please pick the bottle up"), {"type": "lift", "object": "water_bottle"})
        self.assertEqual(judge.classify("push the banana"), {"type": "push", "object": "banana"})
        self.assertIsNone(judge.classify("wave hello"))

    def test_move_to_area_any_object(self):
        meta = {**META, "task": "Move an object to the left side of the workspace"}
        moved = judge.evaluate(meta, track([0.0, 0.05, 0.15, 0.15, 0.15]), judge.DEFAULT_RULES)
        self.assertEqual((moved["label"], moved["reason"]), ("success", "moved_to_area (apple)"))
        short = judge.evaluate(meta, track([0.0, 0.03, 0.06, 0.06, 0.06]), judge.DEFAULT_RULES)
        self.assertEqual(short["label"], "failure")
        wider = judge.evaluate(meta, track([0.0, 0.03, 0.06, 0.06, 0.06]),
                               {"move_to_area": {"zone_cm": 5, "release": True, "still_seconds": 1}})
        self.assertEqual(wider["label"], "success")

    def test_already_in_the_area_is_not_a_success(self):
        meta = {**META, "task": "move the apple to the center"}
        verdict = judge.evaluate(meta, track([0.0] * 5), judge.DEFAULT_RULES)
        self.assertEqual(verdict["reason"], "already_there")

    def test_lift_height_and_hold(self):
        meta = {**META, "task": "lift the apple"}
        up = track([0.0] * 6, zs=[0.79, 0.85, 0.86, 0.86, 0.86, 0.80])
        self.assertEqual(judge.evaluate(meta, up, judge.DEFAULT_RULES)["label"], "success")
        longer = {"lift": {"height_cm": 5, "hold_seconds": 2}}
        self.assertEqual(judge.evaluate(meta, up, longer)["reason"], "not_held_long_enough")
        higher = {"lift": {"height_cm": 10, "hold_seconds": 1}}
        self.assertEqual(judge.evaluate(meta, up, higher)["reason"], "not_high_enough")

    def test_push_distance_and_lifting(self):
        meta = {**META, "task": "push the apple"}
        slide = track([0.0, 0.02, 0.04, 0.07, 0.07])
        self.assertEqual(judge.evaluate(meta, slide, judge.DEFAULT_RULES)["label"], "success")
        carried = track([0.0, 0.02, 0.04, 0.07, 0.07], zs=[0.79, 0.86, 0.86, 0.79, 0.79])
        self.assertEqual(judge.evaluate(meta, carried, judge.DEFAULT_RULES)["reason"], "lifted_not_pushed")
        allowed = {"push": {"distance_cm": 5, "no_lift": False}}
        self.assertEqual(judge.evaluate(meta, carried, allowed)["label"], "success")

    def test_other_table_frame(self):
        """OpenYAM's Sandbox table: z near 0, heights relative to the bowl, and the camera's left along -y."""
        times = [0, 0.5, 1.0, 1.5, 2.0]
        trace = {"t": times, "positions": {"apple": [[0.4, 0.2, 0.03]] + [[0.48, 0.1, 0.03]] * 4,
                                           "bowl": [[0.48, 0.1, 0.0]] * 5},
                 "touching": {"apple": [0] * 5}}
        meta = {**META, "relative_height_bands": {"bowl": [0.01, 0.12], "plate": [0.0, 0.09]}, "off_table_z": -0.1}
        self.assertEqual(judge.evaluate(meta, trace, judge.DEFAULT_RULES)["label"], "success")
        self.assertEqual(judge.evaluate({k: v for k, v in meta.items() if k != "relative_height_bands"},
                                        trace, judge.DEFAULT_RULES)["reason"], "outside_target")
        area = {**meta, "task": "move the apple to the left",
                "workspace": {"center": [0.4, 0.0], "left": [0.0, -1.0], "front": [1.0, 0.0]}}
        moved_left = {**trace, "positions": {**trace["positions"],
                      "apple": [[0.4, 0.0, 0.03]] + [[0.4, -0.15, 0.03]] * 4}}
        self.assertEqual(judge.evaluate(area, moved_left, judge.DEFAULT_RULES)["label"], "success")

    def test_rules_old_flat_form_is_pick_place(self):
        rules = judge.normalize_rules({"distance_cm": 9, "release": False, "still_seconds": 2})
        self.assertEqual(rules["pick_place"], {"distance_cm": 9.0, "release": False, "still_seconds": 2.0})
        self.assertEqual(rules["lift"], judge.DEFAULT_RULES["lift"])
        with self.assertRaises(ValueError):
            judge.normalize_rules({"lift": {"height_cm": 99}})


if __name__ == "__main__":
    unittest.main()
