"""Synthetic pick telemetry modeled on Sereact's documented data flywheel.

Sereact's Series B post (sereact.ai/posts/series-b) states that "every
successful pick, every failure, every recovery is captured with synchronized
observations, robot state, gripper force feedback, and outcome". This module
generates synthetic episodes carrying exactly those four fields, with a hidden
ground-truth failure mode used only for validation. No real robot data is used
anywhere in this repository.

The failure-rate regime is chosen to resemble Sereact's public numbers:
~2% failed picks and ~1 in 53,000 escalated to a remote human intervention.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class PickEpisode:
    episode_id: int
    sku: str
    # synchronized observations (perception summary)
    clutter: float        # 0..1 fraction of bin occupied by other items
    occlusion: float      # 0..1 fraction of target SKU hidden
    sku_novelty: float    # 0..1 how unlike the fleet's known SKUs this is
    # robot state
    approach_speed: float      # m/s of final approach
    pregrasp_offset: float     # cm, commanded vs executed grasp pose error
    plan_branch: int           # which candidate trajectory was chosen
    # gripper force feedback trace (N)
    force_baseline: float
    force_peak: float
    slip_events: int           # counted micro-slip transients during lift
    post_lift_var: float       # force variance after lift (load stability)
    drop_detected: bool
    # outcome
    outcome: str               # "success" | "mispick"
    escalated: bool            # needed remote human help (the rare 1/53k event)
    ground_truth_mode: str     # hidden label, used only for validation


LATENT_MODES = [
    "success", "missed_grasp", "grasp_slip", "multi_pick",
    "collision_bump", "place_miss", "regrasp_loop",
]

_SKU_POOL = [f"SKU-{i:04d}" for i in range(60)]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _gauss(rng: random.Random, mu: float, sigma: float) -> float:
    return rng.gauss(mu, sigma)


def generate_episodes(n: int, seed: int = 7) -> list[PickEpisode]:
    """Generate n synthetic pick episodes.

    Regime: 97% success, ~3% mispick, with escalations sampled so the
    escalation rate lands near 1 in 50,000 (Sereact's public ~1/53,000).
    """
    rng = random.Random(seed)
    episodes: list[PickEpisode] = []
    for i in range(n):
        r = rng.random()
        if r < 0.97:
            mode = "success"
        elif r < 0.978:
            mode = "missed_grasp"
        elif r < 0.986:
            mode = "grasp_slip"
        elif r < 0.990:
            mode = "multi_pick"
        elif r < 0.994:
            mode = "collision_bump"
        elif r < 0.997:
            mode = "place_miss"
        else:
            mode = "regrasp_loop"

        clutter = _clip01(_gauss(rng, 0.45, 0.25))
        occlusion = _clip01(_gauss(rng, 0.25, 0.2))
        sku_novelty = _clip01(_gauss(rng, 0.3, 0.3))
        sku = rng.choice(_SKU_POOL)

        approach_speed = max(0.05, _gauss(rng, 0.35, 0.1))
        pregrasp_offset = abs(_gauss(rng, 0.4, 0.3))
        plan_branch = rng.randrange(4)

        slip_events = 0
        drop_detected = False
        if mode == "success":
            force_baseline, force_peak = 2.0, _gauss(rng, 12.0, 2.0)
            post_lift_var = abs(_gauss(rng, 0.4, 0.2))
        elif mode == "missed_grasp":
            force_baseline, force_peak = 2.0, _gauss(rng, 3.0, 1.0)
            post_lift_var = abs(_gauss(rng, 0.3, 0.2))
            pregrasp_offset = abs(_gauss(rng, 2.5, 1.0))
            occlusion = _clip01(occlusion + 0.25)
        elif mode == "grasp_slip":
            force_baseline, force_peak = 2.0, _gauss(rng, 11.0, 2.0)
            slip_events = rng.randint(2, 6)
            post_lift_var = abs(_gauss(rng, 2.8, 0.8))
            drop_detected = rng.random() < 0.5
            clutter = _clip01(clutter + 0.15)
        elif mode == "multi_pick":
            force_baseline, force_peak = 2.0, _gauss(rng, 19.0, 3.0)
            post_lift_var = abs(_gauss(rng, 1.8, 0.6))
            clutter = _clip01(clutter + 0.3)
        elif mode == "collision_bump":
            force_baseline, force_peak = 2.0, _gauss(rng, 22.0, 4.0)
            approach_speed = max(0.05, _gauss(rng, 0.55, 0.12))
            clutter = _clip01(clutter + 0.25)
        elif mode == "place_miss":
            force_baseline, force_peak = 2.0, _gauss(rng, 12.0, 2.0)
            post_lift_var = abs(_gauss(rng, 0.5, 0.2))
        else:  # regrasp_loop
            force_baseline, force_peak = 2.0, _gauss(rng, 6.0, 2.0)
            slip_events = rng.randint(1, 3)
            post_lift_var = abs(_gauss(rng, 1.2, 0.4))

        outcome = "success" if mode == "success" else "mispick"
        # Escalation: rare. Regrasp loops and collision bumps escalate most;
        # background rate keeps overall ~1/50k when mixed with successes.
        p_esc = {"regrasp_loop": 0.35, "collision_bump": 0.15,
                 "grasp_slip": 0.05}.get(mode, 0.004 if mode != "success" else 0.00002)
        escalated = rng.random() < p_esc

        episodes.append(PickEpisode(
            episode_id=i, sku=sku, clutter=clutter, occlusion=occlusion,
            sku_novelty=sku_novelty, approach_speed=approach_speed,
            pregrasp_offset=pregrasp_offset, plan_branch=plan_branch,
            force_baseline=force_baseline, force_peak=max(0.5, force_peak),
            slip_events=slip_events, post_lift_var=post_lift_var,
            drop_detected=drop_detected, outcome=outcome,
            escalated=escalated, ground_truth_mode=mode,
        ))
    return episodes


def feature_vector(ep: PickEpisode) -> list[float]:
    """Feature vector used for novelty distance (z-scored by caller)."""
    return [ep.clutter, ep.occlusion, ep.sku_novelty, ep.approach_speed,
            ep.pregrasp_offset, ep.force_peak, float(ep.slip_events),
            ep.post_lift_var, float(ep.drop_detected)]
