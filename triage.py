"""Intervention triage: which failures deserve scarce human review?

Sereact's Series B post says fleet failures are "filtered, prioritized by
novelty and uncertainty" before they go back into the model. This module does
exactly that, transparently:

* novelty: mean distance to the k nearest neighbors in the historical pick
  feature cloud (a pick unlike anything the fleet has seen is worth a look).
* uncertainty: normalized entropy of the failure-mode posterior (a pick the
  classifier can't pin down is worth a look).
* severity: escalated-to-human picks and compounding regrasp loops rank higher.

priority = severity_weight(mode) * (0.5 * novelty + 0.5 * uncertainty)

All inputs are plain floats; the queue is fully explainable: every ranked
item reports its novelty, uncertainty, and severity components.
"""
from __future__ import annotations

import math

import numpy as np

from failure_modes import classify
from telemetry import PickEpisode, feature_vector

K = 5  # neighbors for novelty


def _zscore_matrix(eps: list[PickEpisode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([feature_vector(e) for e in eps], dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    return (X - mu) / sigma, mu, sigma


def novelty_scores(episodes: list[PickEpisode],
                   history: list[PickEpisode],
                   k: int = K) -> dict[int, float]:
    """Mean distance to k nearest history neighbors, 0..1 via tanh squash."""
    hist_z, mu, sigma = _zscore_matrix(history)
    out: dict[int, float] = {}
    for ep in episodes:
        x = (np.array(feature_vector(ep), dtype=float) - mu) / sigma
        d = np.linalg.norm(hist_z - x, axis=1)
        nearest = np.partition(d, min(k, len(d) - 1))[:k]
        out[ep.episode_id] = float(math.tanh(nearest.mean() / 3.0))
    return out


def uncertainty_scores(episodes: list[PickEpisode]) -> dict[int, float]:
    """Normalized Shannon entropy of the failure-mode posterior."""
    n_modes = 7
    out: dict[int, float] = {}
    for ep in episodes:
        _, post = classify(ep)
        ent = -sum(p * math.log(p) for p in post.values() if p > 0)
        out[ep.episode_id] = ent / math.log(n_modes)
    return out


SEVERITY = {
    "success": 0.0, "missed_grasp": 0.4, "grasp_slip": 0.6,
    "multi_pick": 0.6, "collision_bump": 0.9, "place_miss": 0.5,
    "regrasp_loop": 0.8,
}


def rank_queue(failures: list[PickEpisode],
               history: list[PickEpisode]) -> list[dict]:
    """Rank failure episodes for human review. Highest priority first."""
    nov = novelty_scores(failures, history)
    unc = uncertainty_scores(failures)
    rows = []
    for ep in failures:
        mode, _ = classify(ep)
        severity = SEVERITY[mode] + (0.2 if ep.escalated else 0.0)
        priority = severity * (0.5 * nov[ep.episode_id] + 0.5 * unc[ep.episode_id])
        rows.append({
            "episode_id": ep.episode_id,
            "sku": ep.sku,
            "mode": mode,
            "escalated": ep.escalated,
            "novelty": round(nov[ep.episode_id], 3),
            "uncertainty": round(unc[ep.episode_id], 3),
            "severity": round(severity, 3),
            "priority": round(priority, 4),
        })
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows
