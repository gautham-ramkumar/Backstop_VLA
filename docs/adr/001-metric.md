# ADR-001: Headline metric is detection at a 5% false-stop budget

- Status: Accepted
- Date: 2026-09-01
- Week: 1

## Decision

The Backstop headline is **detection rate at a 5% false-stop budget** (`detection_at_false_stop`), not raw accuracy and not policy success rate.

- **False-stop:** the guard halts an episode that would have succeeded.
- **Budget:** at most 5% of successful episodes may be halted.
- **Headline:** of the episodes that actually fail, what fraction does the guard catch at that budget.
- **Secondary:** mean lead time in steps (caught failures only), and the full detection/false-stop curve.

`configs/metric.yaml` is the machine-readable lock. `write_guard_report` raises if a caller tries to headline `accuracy`, `raw_accuracy`, or `success_rate`. Changing the budget requires a new ADR.

## Why

A guard that stops every episode catches every failure and is useless. The operator question is: at a false-stop rate someone would tolerate, how many failures are caught, and how early.

Policy success rate is a **setup check** (week 1 baseline reproduction). It is logged in `eval.json` as `policy_success_rate` and is explicitly not the guard headline.

## Consequences

- Week 1 writes the contract before any guard number exists.
- Week 3–4 report detection @ 5% false-stop plus the curve.
- CI fails if `configs/metric.yaml` drifts off 0.05 without updating this file and the schema constant together.
