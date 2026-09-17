"""End-to-end demo: fleet telemetry -> failure triage -> regression checks.

Usage: python3 run_demo.py [--n N] [--seed S] [--out reports]

Writes reports/report.md and reports/queue.csv, prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from failure_modes import classify
from regression import (baseline_policy, naive_regrasp_policy,
                        regression_check)
from telemetry import generate_episodes
from triage import rank_queue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    history = generate_episodes(20000, seed=1000 + args.seed)
    window = generate_episodes(args.n, seed=args.seed)
    failures = [e for e in window if e.outcome == "mispick"]
    escalated = [e for e in failures if e.escalated]

    queue = rank_queue(failures, history)

    # Baseline run gives the escalation budget the candidates must respect.
    base_res = regression_check(baseline_policy, failures,
                                baseline_esc_rate=float("inf"))
    budget = base_res["escalation_rate"]
    naive_res = regression_check(naive_regrasp_policy, failures,
                                 baseline_esc_rate=budget)

    # Classifier sanity on the synthetic ground truth (circular by design,
    # see failure_modes.py honesty note).
    agree = sum(1 for e in failures
                if classify(e)[0] == e.ground_truth_mode)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "queue.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(queue[0].keys()))
        w.writeheader()
        w.writerows(queue[:50])

    lines = [
        "# Pick-failure triage report (synthetic demo)",
        "",
        f"Window: {args.n} picks (seed {args.seed}); history: 20000 picks.",
        f"Failures: {len(failures)} ({100*len(failures)/args.n:.2f}%), "
        f"escalated to human: {len(escalated)}.",
        f"Classifier agreement with synthetic ground truth: "
        f"{agree}/{len(failures)} (circular by construction - pipeline shape "
        "demo, not an accuracy claim).",
        "",
        "## Top 10 intervention queue (priority = severity x (novelty+uncertainty)/2)",
        "",
        "| rank | episode | sku | mode | escalated | novelty | uncertainty | "
        "severity | priority |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(queue[:10], 1):
        lines.append(
            f"| {i} | {r['episode_id']} | {r['sku']} | {r['mode']} | "
            f"{r['escalated']} | {r['novelty']} | {r['uncertainty']} | "
            f"{r['severity']} | {r['priority']} |")
    lines += [
        "",
        "## Regression checks on failure replay set",
        "",
        f"Baseline (replan once, then escalate): {base_res['verdict']} "
        f"(resolved {base_res['resolution_rate']}, escalation "
        f"{base_res['escalation_rate']}).",
        f"Naive regrasp-loop policy: {naive_res['verdict']} "
        f"(no-compounding check: {naive_res['check_no_compounding']}, "
        f"escalation-budget check: {naive_res['check_escalation_budget']}, "
        f"compounding events: {naive_res['compounding_events']}).",
        "",
        "The naive candidate FAILS because it repeats the identical failing "
        "grasp motion - exactly the compounding behavior Sereact's Cortex 2.0 "
        "doctrine rules out before the arm moves.",
    ]
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"picks={args.n} failures={len(failures)} escalated={len(escalated)}")
    print(f"baseline: {base_res['verdict']} "
          f"(res={base_res['resolution_rate']}, esc={base_res['escalation_rate']})")
    print(f"naive-regrasp: {naive_res['verdict']} "
          f"(no_compounding={naive_res['check_no_compounding']}, "
          f"budget={naive_res['check_escalation_budget']}, "
          f"compounds={naive_res['compounding_events']})")
    print(f"wrote {args.out}/report.md and {args.out}/queue.csv")


if __name__ == "__main__":
    main()
