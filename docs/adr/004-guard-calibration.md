# ADR-004: How the headline metric is actually calibrated and measured

- Status: Accepted. Superseded in part by [ADR-005](005-headline-vs-stopwatch.md): rules are now set from good runs only with no search, learned scores are cross-fit, and results are reported over 20 splits, not one. The stratified split itself is unchanged.
- Date: 2026-09-17
- Week: 3

## Decision

ADR-001 fixed *what* the headline metric is (detection at a 5% false-stop
budget). It did not fix *how* a threshold gets set from data, or which
episodes are allowed to touch which number. Four decisions, landed together
because a threshold calibrated and measured on the same data is close to
tautological, and this is the ADR that keeps that from happening silently.

1. **Calibrate on one half, measure on the other, always.** Every episode is
   assigned to exactly one of a calibration split or an eval split
   (`backstop.guard.split.stratified_split`). The threshold is set from the
   calibration split's successes only (`calibrate_threshold`); false-stop
   rate and detection rate are measured on the eval split only (`evaluate`).
   Reporting false-stop rate on the same data the threshold was tuned against
   would make ~5% true by construction, not by measurement.

2. **The split is stratified by `(shard, success)`, not flat-random.**
   Corpus A's per-family success rates are deliberately different — each
   perturbation family was calibrated in Week 2 to land near 50%, and the
   clean arm sits near 70% — and per-family reporting is mandatory
   (Week2_plan.md's risk table: "Week 3 must also report per-family
   detection, never only pooled"). A flat split risks starving one family's
   already-small failure count from one half by chance. Stratifying keeps
   every `(shard, success)` stratum represented in both halves at roughly
   its overall proportion, with a `seed` for reproducibility
   (`tests/unit/test_guard_split.py` covers proportion-preservation,
   reproducibility under a fixed seed, and the size-1-stratum edge case).

3. **Threshold is an order statistic, not a percentile interpolation.** With
   `n` calibration successes and budget `b`, at most `k = ceil(b * n)` may
   cross the line; the threshold is exactly the `k`-th highest calibration
   score (`calibrate_threshold`). This is the same arithmetic
   Week2_plan.md's "corpus design" section worked through by hand before any
   code existed ("5% of ~75 successes lands on the 4th-worst one, so it
   moves in ~1.3pp quanta") — implemented directly rather than re-derived,
   and `tests/unit/test_guard_calibrate.py::test_matches_week2_plans_own_worked_example`
   checks the two agree. Consequence worth stating plainly: `ceil` of any
   positive number is never 0, so there is no calibration-set size or
   positive budget at which the threshold flags zero calibration successes —
   at least one always can cross, on any finite calibration set.

4. **`language_absent` is quarantined out of calibration and out of the
   pooled headline number**, per Week 2's own design (its perturbation flag
   is 0% success by construction, so it is perfectly predictive of the
   outcome — pooling it in would inflate detection with a case that needs no
   real signal to catch). `scripts/evaluate_guard.py` still runs it through
   the threshold calibrated from everything else and reports its detection
   rate on its own, labelled as the degenerate case it is, rather than
   omitting the number entirely.

## What this first run actually measured

First signal built: action-chunk disagreement (`backstop.guard.signals`,
mean absolute deviation of the K−1 extra flow-noise samples from the
executed one, reduced to one number per episode by taking the max over
steps — order doesn't matter for a yes/no "was this episode ever flagged"
question, only for lead time, which is deferred below). Run against all of
corpus A except `language_absent`, `eval_fraction=0.3`, `seed=0`:

| | value |
|---|---|
| Threshold | 0.2127 (calibrated on 162 successes) |
| Detection rate | 78.0% (32/41 eval failures) |
| False-stop rate | 5.8% (4/69 eval successes) |
| `language_absent` (quarantined) | 100% detection (30/30) |

Full per-family breakdown, the detection curve across budgets, and the
threshold are in `artifacts/guard/action_chunk_disagreement/guard_eval.json`.

**Read the per-family numbers with the counts, not instead of them.** A 30%
eval split of a family with 50 episodes leaves single-digit failure counts
per family (4-9 here) — `lighting`'s 50% detection is 2/4, and
`language_paraphrase`'s 100% false-stop is 1/1, a coin flip, not a rate. This
is the same thin-corpus caveat Week2_plan.md's "corpus design" section
already named before any of this existed: `Week2_plan.md` recommended
bootstrap CIs and cross-validation for Week 4, not a single point estimate
per family. Reporting the pooled number as the headline and the per-family
table as directional, not each cell as a precise measurement, until that
lands.

## What is deliberately not in this report

**`mean_lead_time_steps`** (ADR-001's secondary metric) is not computed.
Lead time needs a ground-truth onset step — the first step after which the
episode was no longer recoverable — and that only exists once the Week-2
taxonomy's onset annotations are produced (still `pending`, per
`Week2_plan.md`). `values["mean_lead_time_steps"]` is written as `null` with
an explanatory note rather than silently omitted or estimated from a guess.

**Three of README's four signals** (temporal self-consistency, visual
precondition check, progress monitoring) are not built. This ADR and
`scripts/evaluate_guard.py` cover the first one, proving the calibration
methodology end-to-end; the other three reuse the same `split`/`calibrate`
machinery unchanged; only a new `backstop.guard.signals` function and an
extraction path for its input are per-signal work.

## Consequences

- `backstop.guard.split.EpisodeKey`/`Split` is the shared unit for keeping
  calibration and eval disjoint; any future signal's evaluation should
  construct its split the same way rather than inventing a second
  train/eval convention.
- `scripts/evaluate_guard.py --seed` changes which episodes land in which
  half — expect the exact threshold and per-family cells to move a little
  between seeds given how thin some family cells are (see above); the
  pooled detection/false-stop rates are far more stable, and multi-seed
  variance is exactly what Week 4's planned bootstrap CIs should quantify.
- `write_guard_report` (`src/backstop/eval/report.py`, built in Week 1) is
  used as designed: this ADR is the first thing to actually call it with a
  real `values` payload rather than a test fixture.
