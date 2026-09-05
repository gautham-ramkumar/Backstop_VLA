# Recovery — design note

**Status: deferred, not in scope for weeks 1–6.** This note exists so the option stays open
at a cost of one Week-2 change (full simulator state capture) rather than a re-record. It is
an argument and a set of open questions, not a commitment.

The scope line in the README stands: *the guard halts and flags, it does not replan.*

---

## The question

A guard that only halts cannot raise the success rate. It can't. It converts a failure into
a different kind of failure — one that announced itself:

```
today          70% success ·  30% silent failure
guard, halt    70% success ·   5% silent failure ·  25% declared failure
guard, recover  ??% success ·   ...
```

The first row is what Week 1 measured. The second is what weeks 3–5 are trying to build, and
it is genuinely valuable — a robot that says "I am stuck" is operationally very different
from one that grinds away until a human notices — but the headline 70% does not move.

Only recovery moves it. This note is about whether that is reachable with frozen weights.

---

## Why "try again" does not work

`n_action_steps: 1` ([configs/record/spatial_baseline.yaml](../configs/record/spatial_baseline.yaml)).
SmolVLA regenerates a full 50-step chunk from scratch at every control step and executes only
the first action. The policy is a stateless function of the current observation and the
instruction.

Two consequences:

1. **Replanning is not a recovery action.** It already happens 20 times a second. All 30
   Week-1 failures are policies that replanned ~280 times and stalled anyway.
2. **Pausing is not a recovery action.** Hold still, resume, and the policy sees the same
   observation and emits the same action. A frozen reactive policy in the same state
   produces the same failure.

Recovery must therefore change the policy's **input**. That leaves three families:

| Action | Mechanism | Notes |
|---|---|---|
| **Reposition** — lift, back off, return to a neutral pose, re-approach | changes the observation | The strongest candidate. See below. |
| **Re-condition** — paraphrase the instruction, or decompose into a sub-goal | changes the language input | Week 2 records paraphrase pairs already; their effect on behaviour is measurable for free. |
| **Re-select** — execute a different one of the K sampled chunks | changes which plan is executed | Free to screen offline. Worthless if all K agree and all K are wrong — which is the case to check first. |

---

## Why this policy's failures are unusually favourable

Every one of the 30 natural Week-1 failures ran the full 280 steps to the cap. Not one
terminated early. There are no crushed grasps, no knocked-over objects, no unrecoverable
scene changes. The failures are **stalls**: the arm works itself into a configuration the
policy handles badly and dithers there until the clock runs out.

Stalls are the friendliest possible failure for reposition-and-retry. The task is still
physically achievable, the objects are where they started, and the only thing wrong is
*where the arm is standing*. Lifting to a neutral pose and re-approaching puts the policy
back into a state distribution it demonstrably handles — it succeeds from the start pose 70%
of the time.

Had the failures been catastrophic, this note would not be worth writing.

**Caveat.** This is a claim about LIBERO-Spatial with this checkpoint. The Week-2 taxonomy
will say whether it survives contact with the perturbed corpus; `layout` and `action_noise`
in particular may produce failures with real scene damage. Re-read this section after the
taxonomy exists.

---

## The strongest argument: recovery relaxes the binding constraint

The headline metric is detection at a **5% false-stop budget** ([ADR-001](adr/001-metric.md)).
That budget is tight because a false positive is maximally expensive: the guard killed an
attempt that was going to succeed. Tightness forces conservatism, and conservatism is what
makes the guard miss real failures.

**Recovery makes false positives cheap.** If a spurious trigger means the arm lifts, returns
to neutral, and re-approaches, the likely outcome is that it still succeeds — a couple of
seconds later. The cost of being wrong drops from "lost a success" to "lost two seconds."

So a recovery-enabled guard could run at a 20% intervention rate instead of 5%, and catch far
more of the failures that matter. The two halves are not additive features; each makes the
other work. This is the real case for building it, and it is stronger than "recovery is a
nice extension."

It also reframes the metric: with recovery, the honest headline is a **paired success-rate
delta** (guard-on minus guard-off, same seeds, same init states), not a detection rate.

---

## Why it cannot be folded into weeks 3–4

The economics are opposite.

The guard is a **classifier over a fixed corpus**. Record once, iterate offline in seconds,
try a hundred signal variants against the same 400 episodes. That property is the reason the
project fits in six weeks on one laptop GPU.

Recovery is an **intervention**. The moment it fires the trajectory diverges, and there is no
recorded ground truth for what happens next. Every recovery variant needs fresh rollouts:
~2.5 h per 100 episodes at the measured Week-1 rate. Five strategies × two thresholds ≈ 25 h
— an entire Week-2 GPU budget for one experiment.

Merging the two means the recovery half sets the pace for everything.

### The metric conflict is a feature

`success_rate` is a forbidden headline, enforced in [`schema.py`](../src/backstop/schema.py)
and in CI. Recovery would legitimately headline exactly that. This is not an obstacle to
route around — it is the contract correctly reporting that recovery is a **different
experiment with a different metric**. Doing it properly means amending ADR-001 or writing
ADR-00N with its own contract, deliberately, in the open. Quietly relaxing the forbidden-headline
check to accommodate a recovery number would undo the main methodological safeguard in the
project.

---

## What Week 2 does about it now

One change, ~half a day, and it is not retrofittable: **capture the full flattened MuJoCo
state every step**, not a curated pose dictionary.

LIBERO already exposes the matched pair, and exercises it on every reset:

```python
env.get_sim_state()                    # -> np.ndarray, float64
env.regenerate_obs_from_state(state)   # restores sim AND returns a consistent observation
```

([`libero/envs/env_wrapper.py:118`](../.venv/lib/python3.12/site-packages/libero/libero/envs/env_wrapper.py),
`set_init_state` is an alias of `regenerate_obs_from_state`, so this is the same mechanism
that seeds every episode — not a new code path to trust.)

**Size.** LIBERO-Spatial states are 92 float64 = 736 B. At 280 steps that is ~206 KB per
episode; ~82 MB for a 400-episode corpus, less compressed. Negligible.

**What it buys.** The ability to restore to any step and branch. Recovery evaluation then
costs *from the intervention point forward* rather than from step 0. With onset typically
around step 150 of 280, that is roughly 45% of a full rollout per variant. It also makes the
comparison exact: identical simulator state, guard-on vs guard-off, nothing else different —
a true paired test rather than a seed-matched approximation.

Without it, every recovery experiment re-records the prefix it already had.

**Prohibition unchanged.** Sim state is privileged. It lives in `privileged/{shard}/{ep}.npz`,
outside the LeRobot dataset, permitted for ground-truth labelling, onset annotation, and
counterfactual branching — **forbidden as guard input**. Recovery may use it to *set up* an
experiment; the guard may never read it at inference. See ADR-003.

---

## The cheap pre-test (not a Week-2 todo)

Before building anything, one offline check on data Week 2 already produces. At the annotated
onset of each failed episode, look at the K=4 sampled chunks:

- If chunk 0 (the executed one) is systematically an outlier among the four → **re-selection
  has headroom**, and the cheapest recovery action is worth building.
- If all four agree and all four are wrong → **re-selection is dead**, and only reposition and
  re-condition remain.

Zero GPU hours. It answers the question the whole re-selection branch rests on. Run it once
the corpus and onset annotations exist — start of Week 3 at the earliest, and it does not
belong on the Week-2 critical path.

---

## Sequencing and kill criteria

**Do not start before the Week-4 number exists.** If the guard cannot separate good attempts
from bad ones, recovery has nothing to fire on and would be a second floor on an unbuilt
first.

Proposed gate, evaluated after Week 4:

| Condition | Action |
|---|---|
| Detection at 5% false-stop is weak (< ~30%) | Do not build recovery. Report the negative result and the taxonomy. |
| Detection is reasonable, and the taxonomy says most failures are recoverable stalls | Build it — reposition first, ~1 week. |
| Detection is reasonable but failures involve scene damage | Re-scope. Reposition will not help; consider re-conditioning only. |

If it goes ahead, the minimum honest experiment is: 100 episodes, matched init states,
guard-off vs guard-on-with-reposition, paired test, intervention rate reported alongside the
success-rate delta. Roughly two days of GPU with branching, a week without.

---

## Open questions

- **Where is neutral?** A fixed home pose, the episode's own start pose, or N steps rewound
  along the executed trajectory? The last is the most likely to work and the least likely to
  transfer to hardware.
- **How many attempts?** Unbounded retry converts "fails" into "never terminates". Needs a
  budget, and the budget is part of the metric.
- **Does the guard re-fire immediately after recovery?** If the signal is a function of the
  observation and the observation is now similar, plausibly yes. Needs a refractory period,
  which is another hyperparameter fit on a thin corpus.
- **Does it generalise past stalls?** The Week-2 taxonomy is the input to this question.
- **Hardware.** Rewinding is free in simulation; a physical arm has to drive back, and the
  scene may have changed underneath it. Real-world recovery is a strictly harder problem.
  Out of scope here, worth stating.

---

## Related

- [ADR-001](adr/001-metric.md) — the false-stop budget this argument leans on
- [Week2_plan.md](../Week2_plan.md) — sim-state capture, privileged-state prohibition, taxonomy
- [README](../README.md) — scope, signals, roadmap
