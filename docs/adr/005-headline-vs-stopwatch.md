# ADR-005: Headline is time saved over a stopwatch, at the same 5% budget

- Status: Accepted (2026-10-08)
- Date: 2026-10-08
- Week: 6
- Amends: ADR-001 (headline metric), ADR-004 (how a rule is set and measured)

## Context

ADR-001 made the headline *detection at a 5% false-stop budget*: of the runs
that fail, what fraction does the guard catch while stopping at most 5% of
good runs. It was written in week 1, before anyone knew what failures in this
corpus look like.

They all look the same in one respect. Every failure in corpus A runs to the
280-step cap, and good runs finish by about step 130. So the simplest
possible guard, a clock that stops any run still going at step 132, catches
41 of 41 held-out failures inside the budget. Our best signal-based guard,
the trust head, catches 34. Under ADR-001's headline a timer looks better
than the project, and the number says nothing about what the signals add.

Two further problems showed up when the same rules were rerun over 20
random splits instead of ADR-004's one (`scripts/evaluate_stopwatch.py`,
"Against a stopwatch" in `docs/evaluation.md`):

- **The budget did not hold.** Rules chosen by searching a grid of lines and
  holds up to the budget edge averaged 6.3–6.8% held-out false stops and went
  over 5% on 10–13 of 20 splits. The trust head also leaked: it is trained on
  the same good runs that then set its line. ADR-004's single seed-0 split
  happened to be favourable (2.9% when it was written, against 6.3%
  averaged over 20 splits for the same rule today).
- **One split cannot show any of this.** With 69 held-out good runs, a single
  run moves the false-stop rate by 1.4 points.

The 5% budget itself is not in question. What is in question is what the
headline counts and how it is measured.

## Decision

1. **Headline: time saved over a stopwatch, at a 5% false-stop budget**
   (`stop_time_saved_at_false_stop`). For every failed run, record the step
   at which the guard stops it. A failure the guard never stops counts at its
   natural end, which is how long an operator would wait for it. Average over
   failures. Do the same for a stopwatch alone at the same budget. The
   headline is the stopwatch's mean stop step minus the guard's, in steps and
   in seconds at 20 Hz. Positive means the guard ends failing runs sooner.

   The guard being measured is the signal OR the stopwatch, under one shared
   budget (`backstop.guard.stopwatch`). A deployed supervisor would run the
   clock anyway, so the honest question is what the signals add on top of it.
   The definition does not assume failures run to the cap. If a future suite
   has failures that end early (a collision, a drop), the stopwatch catches
   few of them, and the guard's early stops count for more.

2. **Every rule is set from good runs only, with no search.** The signal line
   is the k-th highest peak among calibration successes. The clock stops any
   run longer than the k-th longest calibration success. With
   k = floor(budget / 2 × (n + 1)), each trigger stops a new good run with
   probability at most k / (n + 1), so the pair stays under the budget on
   average, by construction rather than by luck. This is a conformal bound.
   It assumes new good runs are exchangeable with the calibration ones,
   which holds within a suite and fails across suites (see 4).

3. **Report over repeated splits, never one.** 20 stratified 70/30 splits
   (ADR-004's split, seeds 0–19). Report the mean, the 10th–90th percentile
   range, and how many splits went over 5%. Under the bound in (2), roughly
   a third of splits landing over 5% is expected noise at 69 held-out good
   runs. Most of them landing over is not.

4. **A new suite is recalibrated from its own good runs before it is
   scored.** Signals carry over between suites. Levels do not: object-suite
   good runs are slower and score higher. The frozen spatial rule stops 17%
   of object good runs. Re-set from n good object runs, the measured rate
   tracks the bound 2/(n + 1), which reaches 5% at n = 39. The number of good
   runs used is reported next to the result. No failure from the new suite is
   used.

5. **Any learned score is cross-fit before it sets a line.** Each calibration
   run is scored by a model trained without it (5 folds). This applies to the
   trust head and to anything like it.

6. **Secondary metrics.** Kept, and always printed beside the headline:
   - share of failures the signal alone catches before the clock
     ("signal first");
   - held-out false-stop rate, mean and spread over splits;
   - lead time against the reviewed onset range. Onsets are one reviewer's
     judgement, and the auto-proposals agree with that reviewer on 118 of 169
     failures, so lead time stays secondary;
   - ADR-001's detection at 5% for the signal alone, with no clock, so the
     week-3 tables can still be compared.

## What this headline says today

From `artifacts/guard/headline/guard_eval.json` (written by
`scripts/evaluate_stopwatch.py`). Spatial held-out half, 20 splits. The
saving is paired: guard and stopwatch on the same split, then the
difference, so the spread is the spread of the saving itself.

| guard | saved over stopwatch, mean | 10th–90th percentile | splits with no gain | false-stop |
|---|---:|---:|---:|---:|
| **disagreement OR stopwatch (headline)** | **13.4 steps (0.67 s)** | −1.4 to 25.9 | 3/20 | 0.043 |
| temporal consistency OR stopwatch | 14.5 steps (0.73 s) | 2.5 to 28.8 | 2/20 | 0.044 |
| trust head (cross-fit) OR stopwatch | 9.3 steps (0.47 s) | −1.0 to 20.5 | 3/20 | 0.042 |

The stopwatch alone stops failures at step 135 on average (6.7 s).

**The headline is reported with its spread and its no-gain count, always.**
The saving is real on most splits and small, and on 3 of 20 the guard does
no better than the clock. A mean on its own would hide that.

**Which guard is the headline.** Disagreement OR stopwatch: one signal,
nothing learned, and the first signal built (ADR-004). Picking the best of
the three after seeing them would be a selection effect; temporal
consistency is within a step of it either way, and all three are printed.

This is a much smaller-sounding number than "85% detection", and it should
be. It is what the signals actually add on this corpus. The signal fires
before the clock on about half the failures. On the other half nothing in
the signals separates the failure from a good run before step 130
(separability table in `docs/evaluation.md`).

**Why the stopwatch is so strong here, and where it would not be.** Every
failure in corpus A is a stall: the policy retries until the 280-step cap
(the week-1 finding, and true of all 169 corpus-A failures and all 17
object-suite failures). That makes
"still running" nearly as informative as any signal. A suite whose
failures end early (a collision, a dropped object, an arm leaving the
workspace) would give the clock little to catch, and the signals would
carry more of the headline. The metric is defined so that this shows up
in the number rather than needing a different metric.

## Alternatives considered

- **Keep ADR-001's headline and add the stopwatch as a footnote.** Rejected:
  the headline would then be a number that a timer matches, and the footnote
  would be the real result.
- **Lead time against the onset as the headline.** Rejected: it rests on a
  single reviewer's onset marks, and every rule including the privileged cup
  position is negative against them, so it ranks rules by how late they are
  rather than by what they save.
- **Keep the searched persistence rules, with a hand-picked safety margin.**
  An internal 2.5% budget brings the searched rules back to about 5% held
  out. Rejected as the method because the margin was itself tuned on these
  splits. The search-free rule needs no margin and comes with a stated bound.
- **Change the 5% budget.** Not proposed. ADR-001's reasoning for 5% still
  holds.

## Consequences

Changes made on acceptance:

- `configs/metric.yaml`: `headline_metric: stop_time_saved_at_false_stop`.
  `detection_at_false_stop` moves to `secondary_metrics`, beside
  `signal_first_fraction`, `false_stop_rate_over_splits` and
  `mean_lead_time_steps`. `false_stop_budget` stays 0.05.
- `src/backstop/schema.py`: `HEADLINE_METRIC` and the `Literal` on the
  contract's `headline_metric` change to match. The CI metric-lock step keeps
  checking the budget.
- `src/backstop/eval/report.py`: `write_guard_report` accepts the new
  headline. The forbidden list (`accuracy`, `raw_accuracy`, `success_rate`)
  is unchanged.
- `src/backstop/eval/report.py`: new `write_secondary_report` for
  ADR-001's detection rate, which names the headline in its payload.
  `scripts/evaluate_guard.py` uses it for every per-signal report.
- `scripts/evaluate_stopwatch.py` computes the headline and writes it with
  `write_guard_report` to `artifacts/guard/headline/guard_eval.json`.
- `scripts/evaluate_guard.py` now rewrites only the block between its
  markers in `docs/evaluation.md` and refuses a report without them, so
  the hand-written sections survive a rerun.
- README's status and headline are rewritten around this number.
- ADR-001 and ADR-004 get a "Superseded in part by ADR-005" line. Their
  text stays as the record of what was decided at the time.

Still to do: the viewer's guard-fire marker should show which trigger
fired, the signal or the clock.
