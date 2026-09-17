"""Rule-based failure-mode classification for pick episodes.

Taxonomy covers the mispick modes a bin-picking cell actually sees. It is a
hand-built heuristic classifier over the telemetry fields Sereact documents
(observations, robot state, gripper force feedback, outcome) - NOT a learned
model. The posterior is a softmax over hand-set log-scores so that uncertainty
(entropy over modes) is computable downstream.

Honesty note: the synthetic generator was built with these same modes in mind,
so agreement with the hidden ground truth here is circular by construction.
The value is the pipeline shape (features -> posterior -> uncertainty), not
the accuracy number.
"""
from __future__ import annotations

import math

from telemetry import PickEpisode

TAXONOMY = {
    "success": "Pick completed, item placed in target tote.",
    "missed_grasp": "Gripper closed on air: low peak force, high pregrasp offset.",
    "grasp_slip": "Item grabbed then slipped: slip transients, high post-lift variance.",
    "multi_pick": "More than one item lifted: abnormally high peak force in clutter.",
    "collision_bump": "Arm bumped bin/shelf: force spike with fast approach.",
    "place_miss": "Grasp fine but placement failed: clean force trace, wrong outcome.",
    "regrasp_loop": "Repeated grasp attempts without progress: low force, some slips.",
}


def log_scores(ep: PickEpisode) -> dict[str, float]:
    """Hand-set log-scores for each mode from telemetry features."""
    s: dict[str, float] = {}
    force_ratio = ep.force_peak / max(ep.force_baseline, 0.5)
    s["success"] = 2.0 if ep.outcome == "success" else -3.0
    s["missed_grasp"] = (1.6 if ep.force_peak < 5.0 else -2.0) + \
        (1.2 if ep.pregrasp_offset > 1.5 else -0.5)
    s["grasp_slip"] = (1.4 if ep.slip_events >= 2 else -2.0) + \
        (1.0 if ep.post_lift_var > 1.5 else -0.5)
    s["multi_pick"] = (1.6 if ep.force_peak > 16.0 else -2.0) + \
        (0.8 if ep.clutter > 0.55 else -0.5)
    s["collision_bump"] = (1.6 if ep.force_peak > 18.0 else -2.0) + \
        (0.8 if ep.approach_speed > 0.45 else -0.5)
    s["place_miss"] = (1.0 if ep.outcome == "mispick" and ep.slip_events == 0
                       and ep.force_peak >= 8.0 else -2.0)
    s["regrasp_loop"] = (1.2 if ep.outcome == "mispick" and ep.slip_events >= 1
                         and ep.force_peak < 10.0 else -2.0)
    return s


def posterior(ep: PickEpisode) -> dict[str, float]:
    """Softmax over log-scores: a proper distribution over failure modes."""
    scores = log_scores(ep)
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def classify(ep: PickEpisode) -> tuple[str, dict[str, float]]:
    """Return (best mode, full posterior)."""
    post = posterior(ep)
    return max(post, key=lambda k: post[k]), post
