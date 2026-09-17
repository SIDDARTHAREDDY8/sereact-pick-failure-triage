# sereact-pick-failure-triage

A pick-failure triage harness built around Sereact's published data-flywheel
doctrine: every fleet failure gets captured, classified, and ranked by
**novelty and uncertainty** so scarce human review bandwidth goes to the
episodes that teach the model the most, and every candidate policy update
passes **automated regression checks** on failure replays before rollout.

## The problem (citable, not invented)

From Sereact's own Series B post (<https://sereact.ai/posts/series-b>):

> "Every successful pick, every failure, every recovery is captured with
> synchronized observations, robot state, gripper force feedback, and outcome -
> then **filtered, prioritized by novelty and uncertainty**, and used to update
> the model. **Updated policies pass automated regression checks** and roll out
> to the fleet."

And the failure behavior Cortex 2.0 exists to kill:

> "Today's reactive policies, when they miss, **tend to repeat the same motion
> and compound the failure**."

Public scale numbers: 200+ systems, 1B+ real picks, ~1 in 53,000 picks needing
remote human help (<https://sereact.ai>). The 1/53,000 interventions (and the
far more numerous recoveries) are exactly what this triage queue ranks.

Context: Sereact is opening its first US office in Boston and hiring robotics
engineers there (Core Robotics Engineer, Robotics Application Engineer -
<https://discourse.ros.org/t/were-hiring-sereact-ai-robotics-stuttgart-germany-boston-usa/57455>).

## What this builds

Three small modules, one pipeline:

1. `telemetry.py` - synthetic pick episodes carrying the four fields Sereact
   documents per pick: synchronized observations (clutter, occlusion, SKU
   novelty), robot state (approach speed, pregrasp offset, chosen plan
   branch), gripper force feedback trace, and outcome. Includes a hidden
   ground-truth failure mode used only for validation.
2. `triage.py` - failure-mode posterior (entropy = **uncertainty**), k-NN
   distance to the historical fleet feature cloud (**novelty**), severity
   weighting (escalated picks and compounding regrasp loops rank higher) -
   combined into an explainable priority queue.
3. `regression.py` - failure-replay harness with two hard checks: no
   compounding (a policy may not repeat an identical failing grasp motion),
   and an escalation budget (candidate escalation rate <= baseline).
   Demonstrates a naive regrasp-loop policy FAILING both gates.

Run it:

```bash
pip install -r requirements.txt
python3 run_demo.py            # writes reports/report.md + reports/queue.csv
python3 -m unittest discover -s tests
```

## Honest verification notes (read before citing numbers)

- **All data is synthetic** (seeded `random.Random`; numpy only for math).
  No real robot, no Sereact data, no GPU. CPU-only; numpy 1.26.4.
- Escalation probability is **boosted ~100x** vs Sereact's published 1/53,000
  so escalations actually appear inside a 2,000-pick demo window. The ranking
  math is rate-independent.
- The classifier is **hand-built heuristics, not a learned model**, and the
  synthetic modes were authored with the same taxonomy in mind - so
  classifier-vs-ground-truth agreement (48/55 in the demo) is **circular by
  construction**. It validates the pipeline shape (telemetry -> posterior ->
  uncertainty -> queue), not an accuracy claim.
- Replay dynamics in `regression.py` are a documented toy: "replan fixes
  everything except collision_bump/regrasp_loop" is an assumption, stated in
  the docstring. The check logic (no-compounding, escalation budget) is real
  and would run unchanged against logged real episodes.
- What was actually validated: 7 deterministic unit tests pass (classifier
  posterior is a proper distribution, hand-checkable cases classify
  correctly, scores bounded in [0,1], escalation strictly boosts rank, naive
  policy fails the compounding check, baseline passes). Demo reproduces
  bit-identically under the same seed.

## Files

| file | purpose |
| --- | --- |
| `telemetry.py` | seeded synthetic pick-episode generator |
| `failure_modes.py` | rule-based failure taxonomy + posterior |
| `triage.py` | novelty/uncertainty/severity ranking |
| `regression.py` | failure-replay regression checks |
| `run_demo.py` | end-to-end driver, writes `reports/` |
| `tests/test_triage_pipeline.py` | 7 deterministic unit tests |

Built by Siddartha Reddy Chinthala (AI Software Engineer, AirTrek Robotics;
CV+LiDAR perception for autonomous wingwalking robots) as problem-first
outreach for Sereact's Boston robotics roles.
