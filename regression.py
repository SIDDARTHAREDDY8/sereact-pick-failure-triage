"""Automated regression checks on policy candidates, replayed over failures.

Sereact's doctrine (sereact.ai/posts/series-b): "Updated policies pass
automated regression checks and roll out to the fleet." And: "Today's reactive
policies, when they miss, tend to repeat the same motion and compound the
failure."

This harness encodes exactly that as a checkable rule:

  CHECK 1 - no compounding: a policy may not repeat an identical failing
  grasp motion (same plan branch + same approach) twice in a row on one
  episode. Repeating the same motion that just failed is the compounding
  behavior Cortex 2.0 is built to avoid.
  CHECK 2 - escalation budget: on the failure replay set, the candidate's
  escalation rate must not exceed the baseline's.

A policy is a function episode -> ("regrasp" | "replan" | "escalate").
The replay is synthetic: the harness simulates what each action would do based
on the episode's hidden ground truth, with the key documented assumption that
'replan' (pick a different plan branch) succeeds on the modes where the first
branch failed. This is a toy, but the check logic is real and would run the
same way against logged real episodes.
"""
from __future__ import annotations

from telemetry import PickEpisode

ACTIONS = ("regrasp", "replan", "escalate")


def simulate(ep: PickEpisode, action: str) -> tuple[str, bool]:
    """Return (final outcome, escalated) for a policy action on an episode.

    Synthetic dynamics, documented honestly:
    * regrasp (same motion): never fixes the latent mode -> still a mispick.
    * replan (new branch): fixes everything except collision_bump and
      regrasp_loop, which need a human (escalation).
    * escalate: mispick resolved by a human; counts against the escalation
      budget but never fails.
    """
    mode = ep.ground_truth_mode
    if mode == "success":
        return "success", False
    if action == "regrasp":
        return "mispick", False
    if action == "replan":
        if mode in ("collision_bump", "regrasp_loop"):
            return "mispick", True
        return "success", False
    if action == "escalate":
        return "success", True
    raise ValueError(f"unknown action {action}")


def baseline_policy(ep: PickEpisode, attempt: int) -> str:
    """Sane baseline: try a new plan branch once, then escalate."""
    return "replan" if attempt == 0 else "escalate"


def naive_regrasp_policy(ep: PickEpisode, attempt: int) -> str:
    """The bad candidate: keep repeating the same grasp motion (compounds)."""
    return "regrasp"


def regression_check(policy, episodes: list[PickEpisode],
                     baseline_esc_rate: float) -> dict:
    """Replay the failure set. Returns per-check pass/fail and details."""
    compounds = 0
    esc = 0
    resolved = 0
    for ep in episodes:
        last_action = None
        for attempt in range(3):
            action = policy(ep, attempt)
            if (action == last_action == "regrasp"):
                compounds += 1
                break
            last_action = action
            outcome, escalated = simulate(ep, action)
            if outcome == "success":
                resolved += 1
                if escalated:
                    esc += 1
                break
        else:
            esc += 1
    esc_rate = esc / max(len(episodes), 1)
    check_compound = compounds == 0
    check_budget = esc_rate <= baseline_esc_rate + 1e-9
    return {
        "n_episodes": len(episodes),
        "resolved": resolved,
        "resolution_rate": round(resolved / max(len(episodes), 1), 3),
        "escalation_rate": round(esc_rate, 4),
        "baseline_escalation_rate": round(baseline_esc_rate, 4),
        "compounding_events": compounds,
        "check_no_compounding": "PASS" if check_compound else "FAIL",
        "check_escalation_budget": "PASS" if check_budget else "FAIL",
        "verdict": "PASS" if (check_compound and check_budget) else "FAIL",
    }
