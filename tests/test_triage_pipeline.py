"""Deterministic unit tests for the triage pipeline. Run: python3 -m unittest discover tests"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from failure_modes import classify, posterior
from regression import (baseline_policy, naive_regrasp_policy,
                        regression_check)
from telemetry import PickEpisode, generate_episodes
from triage import (novelty_scores, rank_queue, uncertainty_scores)


def make_ep(**kw):
    base = dict(episode_id=0, sku="SKU-0001", clutter=0.4, occlusion=0.2,
                sku_novelty=0.3, approach_speed=0.35, pregrasp_offset=0.4,
                plan_branch=1, force_baseline=2.0, force_peak=12.0,
                slip_events=0, post_lift_var=0.4, drop_detected=False,
                outcome="success", escalated=False,
                ground_truth_mode="success")
    base.update(kw)
    return PickEpisode(**base)


class TestClassifier(unittest.TestCase):
    def test_posterior_is_distribution(self):
        for ep in generate_episodes(200, seed=3):
            p = posterior(ep)
            self.assertAlmostEqual(sum(p.values()), 1.0, places=9)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in p.values()))

    def test_clear_cases_classify_correctly(self):
        # missed grasp: gripper closed on air
        ep = make_ep(outcome="mispick", force_peak=2.5, pregrasp_offset=3.0)
        self.assertEqual(classify(ep)[0], "missed_grasp")
        # grasp slip: slip transients + unstable load
        ep = make_ep(outcome="mispick", slip_events=4, post_lift_var=3.0)
        self.assertEqual(classify(ep)[0], "grasp_slip")
        # multi-pick: force way too high for one item
        ep = make_ep(outcome="mispick", force_peak=21.0, clutter=0.8)
        self.assertEqual(classify(ep)[0], "multi_pick")
        # collision bump: spike + fast approach
        ep = make_ep(outcome="mispick", force_peak=24.0, approach_speed=0.6)
        self.assertEqual(classify(ep)[0], "collision_bump")


class TestTriage(unittest.TestCase):
    def test_scores_bounded(self):
        hist = generate_episodes(500, seed=11)
        fails = [e for e in generate_episodes(300, seed=12)
                 if e.outcome == "mispick"]
        self.assertGreater(len(fails), 0)
        for v in novelty_scores(fails, hist).values():
            self.assertTrue(0.0 <= v <= 1.0, v)
        for v in uncertainty_scores(fails).values():
            self.assertTrue(0.0 <= v <= 1.0, v)

    def test_queue_sorted_and_escalation_boosts_rank(self):
        hist = generate_episodes(500, seed=21)
        a = make_ep(episode_id=1, outcome="mispick", slip_events=4,
                    post_lift_var=3.0, escalated=False,
                    ground_truth_mode="grasp_slip")
        b = make_ep(episode_id=2, outcome="mispick", slip_events=4,
                    post_lift_var=3.0, escalated=True,
                    ground_truth_mode="grasp_slip")
        q = rank_queue([a, b], hist)
        pri = {r["episode_id"]: r["priority"] for r in q}
        self.assertEqual([r["episode_id"] for r in q], [2, 1])
        self.assertGreater(pri[2], pri[1])

    def test_empty_failures(self):
        self.assertEqual(rank_queue([], generate_episodes(50, seed=31)), [])


class TestRegression(unittest.TestCase):
    def test_naive_regrasp_fails_compounding(self):
        fails = [e for e in generate_episodes(400, seed=41)
                 if e.outcome == "mispick"]
        base = regression_check(baseline_policy, fails,
                                baseline_esc_rate=float("inf"))
        res = regression_check(naive_regrasp_policy, fails,
                               baseline_esc_rate=base["escalation_rate"])
        self.assertEqual(res["check_no_compounding"], "FAIL")
        self.assertGreater(res["compounding_events"], 0)
        self.assertEqual(res["verdict"], "FAIL")

    def test_baseline_passes(self):
        fails = [e for e in generate_episodes(400, seed=42)
                 if e.outcome == "mispick"]
        res = regression_check(baseline_policy, fails,
                               baseline_esc_rate=float("inf"))
        self.assertEqual(res["check_no_compounding"], "PASS")
        self.assertGreater(res["resolution_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
