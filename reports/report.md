# Pick-failure triage report (synthetic demo)

Window: 2000 picks (seed 7); history: 20000 picks.
Failures: 55 (2.75%), escalated to human: 4.
Classifier agreement with synthetic ground truth: 48/55 (circular by construction - pipeline shape demo, not an accuracy claim).

## Top 10 intervention queue (priority = severity x (novelty+uncertainty)/2)

| rank | episode | sku | mode | escalated | novelty | uncertainty | severity | priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 787 | SKU-0048 | regrasp_loop | False | 0.806 | 0.649 | 0.8 | 0.5819 |
| 2 | 698 | SKU-0042 | regrasp_loop | False | 0.659 | 0.549 | 0.8 | 0.4833 |
| 3 | 1782 | SKU-0000 | multi_pick | False | 0.947 | 0.439 | 0.6 | 0.4159 |
| 4 | 1686 | SKU-0019 | grasp_slip | True | 0.823 | 0.144 | 0.8 | 0.3869 |
| 5 | 1078 | SKU-0009 | grasp_slip | True | 0.774 | 0.181 | 0.8 | 0.3822 |
| 6 | 142 | SKU-0009 | grasp_slip | True | 0.56 | 0.361 | 0.8 | 0.3685 |
| 7 | 1336 | SKU-0015 | grasp_slip | False | 0.828 | 0.361 | 0.6 | 0.3568 |
| 8 | 174 | SKU-0024 | place_miss | False | 0.447 | 0.975 | 0.5 | 0.3554 |
| 9 | 1963 | SKU-0053 | grasp_slip | False | 0.776 | 0.389 | 0.6 | 0.3495 |
| 10 | 110 | SKU-0000 | grasp_slip | False | 0.756 | 0.389 | 0.6 | 0.3435 |

## Regression checks on failure replay set

Baseline (replan once, then escalate): PASS (resolved 1.0, escalation 0.1091).
Naive regrasp-loop policy: FAIL (no-compounding check: FAIL, escalation-budget check: PASS, compounding events: 55).

The naive candidate FAILS because it repeats the identical failing grasp motion - exactly the compounding behavior Sereact's Cortex 2.0 doctrine rules out before the arm moves.
