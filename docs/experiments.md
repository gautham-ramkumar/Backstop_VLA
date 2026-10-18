# Guard experiments

This is the record of every way we tried to tell, while a run was still going, that the
policy was about to fail. One idea became the guard, and a few others are still
maintained. **The rest were removed from `main` once they had lost**, and their code is
kept at the git tag `research-archive-2026-10` (see [How to rerun an archived
experiment](#how-to-rerun-an-archived-experiment)).

[`evaluation.md`](evaluation.md) reports the live guard. This file explains how it got
there.

## Common setup

Every experiment below shares this setup unless it says otherwise.

- **Corpus A:** 400 recorded LIBERO-Spatial episodes, SmolVLA, K=4 sampled action
  chunks per step, 20 Hz, capped at 280 steps.
  - The clean arm has 100 episodes at 70% success.
  - Six perturbation families of 50 episodes each are tuned near 50% success: action
    noise, camera, distractor, lighting, object pose, and instruction paraphrase.
  - `language_absent` (30 runs) names an object that isn't on the table. It is reported
    on its own, never pooled (ADR-004).
- **Unseen suite:** 100 LIBERO-Object episodes (`object_clean`), 83 successes and 17
  failures. Nothing is ever fit on these.
- **Split:** stratified by (shard, outcome), 70% calibration / 30% held out
  (ADR-004). On seed 0 that's 162 calibration successes, and 69 successes plus 41
  failures held out. Results marked "20 splits" repeat this over 20 random seeds.
- **Budget:** at most 5% of good runs may be stopped (`configs/metric.yaml`). Every
  line is set from calibration *successes* only. Failures are never used to place a
  line.
- **Onset:** a human-reviewed step range where each failure becomes visible. It is used
  only to measure lead time, never to fit anything. Lead time is onset minus stop step,
  so negative means the stop came after the onset.
- **The shape of the failures:** every corpus-A failure is a stall that runs to the
  280-step cap, and successes finish by about step 130. That one fact explains most of
  the results below.

## Summary

| # | Experiment | Question | Best result | Verdict | Status |
|---|---|---|---|---|---|
| 1 | Action-chunk disagreement | Do the K sampled plans disagree before a failure? | 32/41 caught, 4/69 false stops (one split) | Works, simplest | **live: the guard's signal** |
| 2 | Stopwatch | Is "still running at step N" already a guard? | 41/41 caught, mean stop step 135 | Strong baseline | **live: the guard's clock** |
| 3 | Disagreement OR stopwatch | Can the signal end failures sooner than the clock? | 13.4 steps (0.67 s) sooner, 4.3% false stops, 20 splits | Headline (ADR-005) | **live** |
| 4 | Temporal self-consistency | Does the plan change its mind step to step? | 14.5 steps sooner (20 splits) | Tie with #3, not simpler | archived |
| 5 | Progress monitoring | Is the arm commanding motion it doesn't make? | 20/41 caught | Too weak | archived |
| 6a | Visual check, one line (CLIP, SmolVLM2, detector) | Is the instructed object visible? | Worse than chance | Failed | archived |
| 6b | Per-object detector gate | Same, with a line per object | 28/30 missing objects stopped at step 0 | Works for that one job | **live, separate from the guard** |
| 7 | Trust head | Do the signals combine into a better score? | 34/41 one split; 9.3 steps sooner (20 splits) | Worse than #3, leaks | archived |
| 8 | Persistence stop | Does "high for N steps" let the line drop? | 0.3 s earlier for disagreement | Over budget on held-out splits | archived |
| 9 | Paper signals and success-path line | Do signals from recent papers stop earlier? | Speed-adjusted overlap 34/41, 1.8 s late | No earlier than #1 | archived |
| 10 | Inside the model | Is the failure visible in SmolVLA's own activations? | 18/41, 5.1 s late | Later and weaker | archived |
| 11 | Per-step model | Does a model of every step, with all inputs, do better? | 41/41 on one split, 5/69 false stops | Over budget, doesn't transfer | archived |
| 12 | Cup motion (privileged ceiling) | How early could *any* guard be, if it saw the cup? | 34/36, 1.7 s late | Ceiling is late too | archived |
| 13 | Object-suite cutoffs per signal | Do the signals work on an unseen suite with its own line? | Disagreement 14/17, 1/42 | Superseded by #3's recalibration | archived |
| 14 | Re-selection pre-test | Would executing another sampled chunk help recovery? | Chunk 0 is not an outlier (ratio ≈1.0) | No headroom | **live, recovery** |

## What the experiments taught us

1. **The failures here are stalls, so a clock is hard to beat on detection.** Any guard
   catches every failure by step ~135 once you add a clock. The only thing left to win
   is time (#2, #3).
2. **The signals are late because the failure is hard to see early, not because the line
   is badly placed.**
   - At the reviewed onset (around step 40), the best signals separate failures from
     still-running successes with AUROC about 0.65. A 5% line needs far better.
   - The privileged cup position is just as late (#12).
3. **Searched rules and learned heads overfit the budget.**
   - A line and hold picked from a grid on the calibration half, or a head trained on
     the same successes that set its line, look inside 5% on one split but drift over
     it on new splits (#7, #8, #11).
   - Rules set from good runs only, by an order statistic with no search, hold the
     budget (#3).
4. **Visual features learned on one scene layout don't transfer.**
   - A visual input to the trust head or the per-step model looked helpful on spatial
     runs, then made object-suite results look far better than they were (#7, #11).
   - One detector line for all objects fails. A line per object, with reference frames
     from the same setting, works (#6).
5. **Levels don't carry over to a new suite, but the signals do.** A new setting needs
   about 40 good runs to set its lines inside budget (#3, #13).

---

## 1. Action-chunk disagreement (live)

**Question.** SmolVLA draws its plan with flow-matching noise. If we draw K plans from the
same observation, do they disagree more when the run is about to fail?

**Method.**
- At every step, sample K=4 chunks (50 steps × 7 dims each) with seeded noise. Index 0 is
  the chunk actually executed.
- The score is the mean absolute deviation of the other three from chunk 0, averaged over
  the whole chunk.
- The episode score is the max over steps.
- Code: `backstop.guard.signals.action_chunk_disagreement`.

**Results.**
- One split (seed 0), with the threshold at the 5% line of calibration successes (0.213):
  - **32/41 failures caught, 4/69 good runs stopped** (5.8%).
  - The 95% bootstrap interval for detection is 0.66–0.90; the 5-fold mean is 0.73.
  - `language_absent` alone: 30/30 caught.
- Per perturbation family, perturbed *successes* score like clean successes (AUROC
  0.32–0.48 against clean), and failures score higher than successes in the same family
  (0.88–0.93). So it reacts to failing, not to the perturbation. Paraphrase is the
  exception: all 3 of its successes score high.

**Verdict.** It works, it learns nothing, and it is one function. It is the signal half of
the guard.

## 2. Stopwatch (live)

**Question.** Every failure runs to 280 steps and good runs end by about 130. Is "stop
anything still running at step N" already a guard?

**Method.** N is the k-th longest calibration success, with k chosen for the budget. A run
is stopped only if it is strictly longer than N, so runs tied with it don't count. Code:
`backstop.guard.stopwatch`.

**Results.**
- N = 132 on seed 0. It catches **41/41** held-out failures and stops 6/69 good runs.
- Over 20 splits: false stops 5.1% on average, over 5% on 10 of 20 splits, mean stop
  step 135 (6.7 s).

**Verdict.** On this corpus the clock alone catches every failure. It became the
baseline any guard has to beat (ADR-005), and the clock half of the guard.

## 3. Disagreement OR stopwatch: the headline (live)

**Question.** With both triggers set from good runs only, how much sooner than the clock
does the guard end a failing run?

**Method.**
- The signal line and the clock each get half the budget:
  k = floor(0.05/2 × (n+1)), so with 162 successes, k = 4.
- Each trigger then stops a new good run with probability at most k/(n+1), with no
  search and no failure used.
- The guard and the clock are compared on the same 20 random splits.
- Code: `backstop.guard.stopwatch`, `scripts/evaluate_stopwatch.py`.

**Results (20 splits).**
- **13.4 steps (0.67 s) sooner than the clock on average** (10th–90th percentile −1.4 to
  25.9). There was no gain on 3/20 splits.
- False stops 4.3%, over 5% on 8/20 splits. All failures are caught.
- The signal fires before the clock on about half the failures. On the other half
  nothing separates the failure from a good run before the clock runs out.
- **Unseen suite.** The frozen spatial rule stops 17% of object successes, because object
  successes are slower (median 153 steps). Re-setting both triggers from n good object
  runs (200 draws each):

  | good object runs | false stops | bound 2/(n+1) | mean stop step |
  |---:|---:|---:|---:|
  | 10 | 0.172 | 0.182 | 123 |
  | 20 | 0.093 | 0.095 | 143 |
  | 40 | 0.041 | 0.049 | 177 |
  | 50 | 0.035 | 0.039 | 188 |

  The measured rate tracks the bound at every n. A new setting needs about 40 good runs.

**Verdict.** The headline guard (ADR-005). It is frozen for live use by
`scripts/freeze_guard.py` and run step by step by `backstop.guard.online.OnlineGuard`.

## 4. Temporal self-consistency (archived)

**Question.** The chunk at step t and the chunk at step t+1 plan the same future, shifted
by one step. Does a failing policy keep changing its mind?

**Method.** The mean absolute difference over the overlap of successive chunks. It's
lagged by one step: the score for t is known at t+1.

**Results.**
- One split: 32/41 caught, but 7/69 false stops (10.1%) at the single-crossing line.
- OR stopwatch over 20 splits: **14.5 steps (0.73 s) sooner**, 2.5 to 28.8; no gain on
  2/20 splits; false stops 4.4%.
- On the unseen suite with its own cutoff: 16/17 caught, 2/42 false stops.
- Separation at step t (AUROC, running max):

  | step | 20 | 40 | 60 | 80 | 100 |
  |---|---:|---:|---:|---:|---:|
  | disagreement | 0.61 | 0.65 | 0.68 | 0.75 | 0.77 |
  | temporal | 0.56 | 0.65 | 0.72 | 0.79 | 0.80 |

**Verdict.** A statistical tie with disagreement: about a step better on average, well
inside the split-to-split spread. It isn't simpler and it carries a one-step lag, so it
wasn't kept.

## 5. Progress monitoring (archived)

**Question.** Is the arm commanding motion it doesn't make, or closing the gripper on
nothing?

**Method.** Two stall scores, summed:
1. The cosine mismatch between the commanded translation and the observed hand step.
2. The gripper closing while the hand stays put over 5 steps, scaled by the open width.

It uses only the recorded state and action (no object pose), and looks 5 steps ahead,
so it isn't causal as written.

**Results.**
- One split: 20/41 caught, 2/69 false stops (detection 0.49).
- Unseen suite: 3/17 at the spatial line.
- A persistence rule can bring its stop nearer the onset only by missing more failures
  (14/41).

**Verdict.** Too weak. Stalls here don't look like "commanded but not moving". The arm
moves; it just doesn't accomplish anything.

## 6. Visual precondition

**Question.** Before the arm moves, is the object the instruction names in view?

### 6a. One line for every object (archived)

**Method.** Score the first agentview frame for the instructed object, and set one line
from spatial successes. Tried in order:
- CLIP whole-frame similarity
- CLIP minus generic scene prompts
- CLIP, best of 13 crops
- SmolVLM2-500M asked "Is there a …?"
- the OWLv2 detector score

**Results.** Using the 30 `language_absent` runs as positives and the other 470 as
negatives:

| score | missing caught | spatial good runs stopped | object suite stopped | AUROC |
|---|---:|---:|---:|---:|
| CLIP, whole frame | 1/30 | 5/69 | 47/100 | 0.27 |
| CLIP, minus scene prompts | 1/30 | 5/69 | 66/100 | 0.24 |
| CLIP, best of 13 crops | 0/30 | 4/69 | 50/100 | 0.29 |
| SmolVLM2-500M | 0/30 | 4/69 | 4/100 | 0.30 |

- As a guard signal (detector score, one line): 1/41 caught on spatial, and it stops
  **every** good object-suite run (83/83).
- SmolVLM2 answered "no" for 362 of 370 spatial scenes with the bowl in plain view.

**Verdict.** All worse than chance. One line can't serve objects whose scores differ by
30×: a real tomato-sauce can scores 0.01, while a moka-pot lookalike scores 0.3.

### 6b. Per-object detector gate (live, not part of the guard)

**Method.** OWLv2 (`google/owlv2-base-patch16-ensemble`) with **a line per object**, set
only from frames that show that object. Below the k-th lowest, the object is called
missing. That bounds false stops at k/(n+1) without ever needing a missing-object
example. An object with no reference frames is reported as unverified. References:
- the black bowl, from 162 spatial successes
- the moka pot, from 40 rendered LIBERO-10 scenes
- each grocery item, from half of its object-suite scenes, split by scene

Code: `backstop.guard.presence`, `scripts/evaluate_presence.py`,
`scripts/render_reference_frames.py`.

**Results.**
- **28/30 missing-moka-pot runs stopped at step 0.** The headline guard catches the same
  runs only at a median of step 86 (4.3 s).
- 2/69 spatial good runs stopped, and 8/60 object-suite runs.
- The 2 misses are a lookalike on the table that scores just above the line.
- It tells *rooms* apart better than objects. It can't see alphabet soup, cream cheese or
  tomato sauce at all. Grocery lines don't carry over to new layouts.

**Verdict.** It works for one job: catching a missing object before the arm moves. It
stays outside the guard. Adding it to a deployed guard would mean giving it a share of
the 5% budget.

## 7. Trust head (archived)

**Question.** Does a learned combination of the signals beat any single one?

**Method.**
- Logistic regression (IRLS) on the episode scores of disagreement, temporal consistency
  and progress, fit on the calibration half only.
- The eval half is scored with the frozen head and judged by the same 5% rule.
- A "running" variant applies the head to each signal's running max, step by step.
- "Cross-fit" scores each calibration run with a head trained without it.
- Code was `backstop.guard.combine` and `guard/summarize.py`.

**Results.**
- One split: **34/41 caught, 3/69 false stops**, the best single-split detection of any
  episode-level score.
- Over 20 splits, single crossing: false stops 6.3%, over 5% on 10/20.
- Cross-fit OR stopwatch: **9.3 steps (0.47 s) sooner** than the clock, worse than
  disagreement's 13.4.
- Unseen suite: 34/83 good runs stopped at the spatial line. With its own cutoff (0.907),
  15/17 caught and 2/42 stopped.

**The visual leak.** With the detector score as a fourth input, the head learned a small
negative weight for it on spatial runs, where the bowl is always in view. Object-suite
groceries score 6.7 standard deviations from anything it was fit on, and that weight made
the object row look far better than it was: 6 of 83 good runs stopped instead of 34. The
visual input was removed before the result above.

**Verdict.** Not better than one unlearned signal once measured fairly. It also leaks:
trained on the same successes that set its line, those successes look safer than new
ones.

## 8. Persistence stop (archived)

**Question.** If a stop requires the score to stay up for several steps, can the line drop
and stop earlier?

**Method.** Search line × hold on the calibration half, within the 5% false-stop count,
giving up at most 5 points of calibration detection. Onset is not used. Code was
`backstop.guard.persist`, `scripts/evaluate_persistence.py`.

**Results (one split):**

| signal | rule | line | hold | caught | false stops | median lead (onset start / end) |
|---|---|---:|---:|---:|---:|---:|
| disagreement | single | 0.213 | 1 | 32/41 | 4/69 | −44 / −24 steps |
| disagreement | persistence | 0.170 | 2 | 31/41 | 3/69 | −37 / −17 steps |
| trust head | single | 0.524 | 1 | 34/41 | 3/69 | −61.5 / −41 steps |
| trust head | persistence | 0.513 | 1 | 35/41 | 3/69 | −63 / −42 steps |

- For disagreement, a hold of 2 stops about 0.3 s earlier, still after the onset.
- **Over 20 splits:** the searched disagreement rule's held-out false stops average 6.6%,
  over 5% on 13/20 splits. The trust head's average 6.8%, also 13/20.

**Verdict.** The gain is small, and the search fits the calibration half: the held-out
rate drifts over budget. This result is what led to the order-statistic rules in #3.

## 9. Signals from the papers, and a line that follows a success (archived)

**Question.** Do signals from recent failure-prediction papers, or a line that tracks a
typical success over time, stop earlier?

**Method.** Two new scores from the stored chunks:
- **Action-chunk magnitude:** the mean size of the planned actions (ActProbe).
- **Speed-adjusted overlap:** temporal consistency divided by the size of the planned
  motion (VLA-FAIL).

Separately, a **success-path line**: subtract the typical calibration-success score at
each step (median or 90th percentile), then apply the persistence rule to the excess.
Code was `guard/timevary.py`, `scripts/evaluate_early.py`.

**Results (one split):**

| signal | caught | false stops | median lead (onset start / end) |
|---|---:|---:|---:|
| chunk magnitude | 1/41 | 7/69 | +63 / +83 steps (one catch) |
| speed-adjusted overlap | 34/41 | 2/69 | −36 / −20 steps (−1.8 s) |
| trust head above the success path | 36/41 | 2/69 | −62 / −42 steps (−3.1 s) |

- Speed-adjusted overlap catches about as many failures as the trust head and stops
  about a second earlier.
- The success-path line catches more failures but no earlier.
- An apparent gain (3.2 s → 2.7 s late) came from the CLIP input, not from the action
  stream, and disappeared without it.

**Verdict.** Nothing here beats disagreement OR stopwatch on time saved, so nothing was
promoted.

## 10. Inside the model (archived)

**Question.** Is the failure visible in SmolVLA's own activations earlier than in its
actions?

**Method.**
- Replay all 89,293 saved steps through SmolVLA without the simulator, one prefix pass
  per step (cameras, instruction and arm state in), taking 75 minutes.
- Keep one 960-number last-layer summary per step. The first step is checked against the
  stored action.
- Score each step by its diagonal-Gaussian distance from calibration-success steps.
- Code was `guard/internal.py`, `scripts/extract_internal.py`,
  `scripts/evaluate_internal.py`.

**Results (one split, 5% persistence rule):** **18/41 caught, 0/69 false stops**, median
stop 101 steps (5.1 s) after the onset start.

**Verdict.** Later and weaker than the action signals. The model's internal state shows
nothing earlier than its outputs do.

## 11. Per-step model (archived)

**Question.** Does a model that sees every step, with every input, do better than
episode-level scores?

**Method.**
- Logistic regression on the calibration half, one row per step. The label is the
  episode outcome on every step, and each episode gets one vote split across its steps.
- At step t: the value now, the mean so far, and the max so far, for the four original
  signals, chunk magnitude, speed-adjusted overlap, and the internal distance (#10).
- The line is the persistence rule on calibration successes.
- Code was `guard/step_model.py`, `scripts/train_step_model.py`.

**Results (one split):**

| visual input | line | hold | caught | false stops | median lead (onset start / end) |
|---|---:|---:|---:|---:|---:|
| none | 0.762 | 3 | 41/41 | 5/69 | −26 / −6 steps (−1.3 s) |
| detector | 0.787 | 2 | 41/41 | 6/69 | −18 / +2 steps |
| CLIP | 0.907 | 1 | 41/41 | 6/69 | −34 / −14 steps |

- It catches every held-out failure, 12 of them before the onset range. But it stops
  5/69 good runs, over budget on its one split.
- The rows differ by about 8 steps, less than the −11 to +33 split-to-split swing of the
  headline, so none of those differences is a measured effect.
- Frozen on the unseen suite, it stops **83/83** good runs. With the detector input it
  had stopped 40, which was the same visual artifact as #7.

**Verdict.** The earliest stops of any experiment, but over budget, never repeated over
splits, and no transfer. Not deployable as measured.

## 12. Cup motion: a privileged ceiling (archived)

**Question.** If a guard could see the cup move, which ADR-003 forbids, how early could it
stop? This bounds what any camera-based estimate of that motion could achieve.

**Method.**
- From the privileged sim-state sidecar: how far the cup sits behind the median
  successful cup at each step, fit on the calibration half, with the 5% persistence
  rule.
- Distractor and object-suite runs have a different layout and are left out (36 failures
  and 59 successes held out).
- Code was `scripts/evaluate_target_motion.py`.

**Results:** **34/36 caught, 2/59 false stops**, median 1.7 s after the onset start.
- A good cup sits still until the pickup, so the 5% line has to wait until a failing cup
  is 15 cm behind a normal one.

**Verdict.** Even perfect knowledge of the object is as late as the best action signal.
The lateness comes from how this task fails, not from a weak signal.

## 13. Object-suite cutoffs per signal (archived)

**Question.** On the unseen suite, do the signals work if their line is set from that
suite's own good runs?

**Method.** 41 good object runs set each signal's 5% line, the other 42 check false stops,
and all 17 failures are scored. The trust head stays the spatial one. Code was
`scripts/evaluate_object_line.py`.

**Results:**

| signal | object cutoff | caught | false stops | spatial cutoff on the same 42 |
|---|---:|---:|---:|---:|
| disagreement | 0.263 | 14/17 | 1/42 | 16/17, 17/42 |
| temporal consistency | 0.302 | 16/17 | 2/42 | 17/17, 20/42 |
| progress | 1.992 | 2/17 | 5/42 | 3/17, 9/42 |
| visual, one line | −0.023 | 3/17 | 4/42 | 17/17, 42/42 |
| trust head | 0.907 | 15/17 | 2/42 | 17/17, 17/42 |

**Verdict.** The signals transfer; their levels don't. That lesson is kept, done properly
for the headline rule, in #3's recalibration table, which superseded this script.

## 14. Re-selection pre-test, for recovery (live)

**Question.** When the guard fires, would executing a different one of the K sampled chunks
help?

**Method.** In each reviewed failure-onset window, compare against good runs at the same
steps:
- chunk 0's distance to the others, relative to their distance from each other (the
  outlier ratio)
- the largest pairwise distance between samples (spread)
- how often the samples disagree on opening vs closing the gripper

Code: `backstop.recover.reselect`, `scripts/pretest_reselect.py`.

**Results.**
- Outlier ratio at failure onsets is 1.03, against 1.01 in good runs. Chunk 0 is just one
  more sample.
- Spread is 0.148 against 0.107.
- The samples split on the gripper decision 25% of the time at onsets, against 9% in
  good runs.

**Verdict.** The policy is undecided, but it redraws a fresh chunk every step, so
swapping one chunk once changes nothing. Recovery uses lift-and-back-off instead
(`docs/recovery.md`).

---

## How to rerun an archived experiment

The tag `research-archive-2026-10` is the last commit where every experiment above still
runs. Check it out in a separate worktree so `main` is untouched:

```bash
git worktree add ../backstop-archive research-archive-2026-10
```

Then, from `../backstop-archive` with this repo's data and artifacts linked in, run the
script named in the experiment, for example `uv run python scripts/evaluate_early.py`.
Each script's docstring says what it reads. All of them read
`artifacts/guard/traces.jsonl`, which at that tag still carries every signal's per-step
series (`scripts/evaluate_guard.py` writes it).

| Experiment | Code at the tag |
|---|---|
| 4 temporal consistency, 5 progress | `src/backstop/guard/signals.py`, `scripts/evaluate_guard.py`, `scripts/evaluate_stopwatch.py` |
| 6a CLIP / SmolVLM2 / one-line visual | `src/backstop/guard/vision.py`, `scripts/evaluate_presence.py --baselines`, `scripts/evaluate_guard.py` |
| 7 trust head | `src/backstop/guard/combine.py`, `src/backstop/guard/summarize.py`, `scripts/evaluate_guard.py`, `scripts/evaluate_stopwatch.py` |
| 8 persistence | `src/backstop/guard/persist.py`, `scripts/evaluate_persistence.py` |
| 9 paper signals, success-path line | `src/backstop/guard/timevary.py`, `scripts/evaluate_early.py` |
| 10 inside the model | `src/backstop/guard/internal.py`, `scripts/extract_internal.py`, `scripts/evaluate_internal.py` |
| 11 per-step model | `src/backstop/guard/step_model.py`, `scripts/train_step_model.py` |
| 12 cup motion | `scripts/evaluate_target_motion.py` |
| 13 object-suite cutoffs | `scripts/evaluate_object_line.py` |
