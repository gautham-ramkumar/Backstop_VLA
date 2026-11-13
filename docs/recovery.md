# Recovery — design note

**Status: built and piloted (2026-11-04). Lift-and-back-off does not help this policy:
on 20 guarded episodes it rescued 1 failure and spoiled 1 good run, a change of 0.**
The kill criterion below is met, so the full run was not made. A second motion, sending
the arm back to its start pose (2026-11-12), rescued 3 and spoiled 1 on the same 20
episodes: +0.10, not significant (p = 0.625). The original design note follows the results.

The scope line in the README stands: *the guard halts and flags, it does not replan.*

## Pilot result

**Setup.** The frozen 5% rule (`artifacts/guard/frozen/rule_05.json`, calibrated without
the clean shard) fires on 32 of the 100 held-out clean episodes: 30 failures and 2 good
runs. The other 68 run exactly as recorded with the guard on. The pilot ran the first 20
triggered episodes (`scripts/run_recovery.py --limit 20`), each restored at its trigger
step from the recorded sim state, then lift and back off, then the policy again: at most 2
recoveries, a 20-step cooldown, the stopwatch restarted after each recovery, the 280-step
cap unchanged (`configs/recovery/recovery.yaml`). Results in `artifacts/recovery/pilot/`.

| | count |
|---|---:|
| episodes paired | 20 (19 failures, 1 good run with the guard off) |
| rescued (failure → success) | **1** (episode 1, after its second recovery) |
| spoiled (success → failure) | **1** (episode 63: a slow good run the clock stopped at step 140) |
| paired success-rate change | **0.0** (McNemar p = 1.0) |
| recoveries per episode | 1.7 on average |
| extra steps per episode | −10.8 (halts after the second recovery end some runs early) |

ADR-006's report: `artifacts/recovery/pilot/recovery_eval.json`.

**Why it fails.** The policy goes straight back to the same stall. After a recovery the
guard fires again a median of 33 steps later, just past the cooldown, on the signal in 13
of 14 second attempts. 15 of the 20 triggers were the clock at step 140, by which point
the run has already been stuck for most of its length. Of the 5 runs the signal caught
early (steps 48–101), recovery rescued 1.

**Checked along the way.**
- *Branching is the live run.* Three episodes run from their reset with the guard live
  (`--live`) first fired at exactly the offline trigger steps (48, 51, 80).
- *Branching is repeatable.* The same episodes branched twice gave the same outcomes,
  step for step.
- *But a recovery is chaotic.* Episode 1 succeeded branched and failed live: the
  restored image differs from the live one by under 0.1 grey level, and after a
  recovery that is enough to take a different path. A guard-on outcome is one sample of
  what recovery does from that state, not a fixed property of it.
- *The motion looks right.* GIFs of the first five (`artifacts/recovery/pilot/gifs/`):
  the arm opens, lifts clear and backs off. In episode 2 the guard fired with a bowl in
  the gripper, and opening dropped it back on the table, the near-grasp case flagged
  below.

## Second motion: send the arm home (2026-11-12)

The policy succeeds on 70% of clean runs from the start pose, so the second motion
sends the arm back there and lets the policy start over: open the gripper, lift 10 cm,
then drive the hand to its recorded start position and rotation (within 1 cm and 0.1 rad),
settle (`reposition.mode: home`, `configs/recovery/return_home.yaml`). One retry per
episode, the step cap raised to 400 so a retry has as long as a fresh run.

**Is there anything to win? The headroom test.** Before trusting a retry, each of the 30
clean failures was re-run from its own start state twice under fresh flow noise, no guard
(`scripts/retry_headroom.py`, cap 160): 25 of 60 tries succeeded (42%), and 17 of the 30
failed scenes succeeded on at least one try. Task 2 and task 5 succeeded every time;
tasks 0, 6 and 8 never did. So a fresh start rescues a lot of failures, in principle.

**The result.** On the same 20 triggered episodes as the first pilot:

| | cooldown 20 | cooldown 140 (clock only) |
|---|---:|---:|
| rescued (failure → success) | 3 (episodes 17, 29, 31) | 3 (same) |
| spoiled (success → failure) | 1 (episode 63) | 1 (same) |
| paired success-rate change | +0.10 (McNemar p = 0.625) | +0.10 (p = 0.625) |
| extra steps per episode | 15.5 | 46.7 |

The motion itself works: in the sim the hand reaches within about 1 cm and 0.02 rad of
its start pose in 50–75 steps, and the camera frames match the real start view. The
second run, with the guard's signal silenced for 140 steps after the retry, gave
identical outcomes, so the guard firing early on the retry is not what holds it back.

**Why a retry from home is worse than a fresh try.**
- *The scene has moved.* When the arm reaches home, the bowl sits more than 5 cm from
  where the episode started in 7 of 20 episodes (31, 50, 56, 58, 61, 63, 68). In four of
  them it moved about 25 cm: the bowl was raised about 1.3 cm when the guard fired, so the
  arm was carrying it, and opening the gripper dropped it somewhere new.
- *Even untouched scenes do worse.* On the 13 scenes where the bowl moved under 4 cm,
  fresh tries succeeded 10 of 26 times (38%) but retries from home 2 of 13. The joint
  angles at home are within 0.05 rad of the start on 12 of them (episode 29 is 0.38 rad
  off and succeeded anyway), so the arm's posture does not explain it. What remains:
  chance (about 5 rescues expected at 38%, 2 seen, on 13 episodes), the different noise
  draws, and the plate shifting 1–2 cm in several episodes. Episodes 23 and 36 succeeded
  on both fresh tries and failed after returning home.

**Where it stands.** +10 points on 20 episodes is not distinguishable from zero. The
carried-bowl case is a fixable design problem (don't open the gripper while holding the
object); the gap on untouched scenes needs more episodes and more retry seeds per
episode before it means anything. Results in `artifacts/recovery/return_home/`,
`artifacts/recovery/return_home_clock/` and `artifacts/recovery/retry_headroom/`.

**What would be next, if anything.** Re-selection has no headroom (the pre-test below).
Re-wording the instruction is the remaining cheap recovery. A recovery that changes the
policy's situation more than a 10 cm move, such as a different approach pose, is a design
question, not a parameter sweep on this pilot.

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

**Verified, with one caveat (`scripts/verify_sim_state_roundtrip.py`, `just
verify-sim-roundtrip`).** Restoring the flattened state (time + qpos + qvel) and replaying the
same actions reproduces the original continuation to float noise (~1e-6) under smooth motion —
the case the recorded corpus actually contains. It does **not** hold under discontinuous
actions (rapid gripper open/close chatter, large uncorrelated joint jumps): the same test with
i.i.d.-noise actions showed real, decaying divergence for several steps after a restore before
re-converging. Root cause: robosuite's flattened state
(`MjSimState.flatten` in `robosuite/utils/binding_utils.py`) is time + qpos + qvel only — it
excludes `qacc_warmstart`, so MuJoCo's contact solver starts a branch from a different warm
start than the original run had whenever contacts are active. Ruled out by elimination first:
this model's `sim.data.act` is empty (no actuator dynamics to miss), and forcing the OSC
controller to re-sync (`controller.update(force=True)` + `reset_goal()`) after restore changed
nothing, because `goal_pos` is recomputed fresh from current `ee_pos` on every `set_goal()`
call rather than carried over. Practical effect: branching from a state reached by smooth
policy motion (the corpus's actual content) is safe; branching mid-chatter (e.g. exactly at a
gripper toggle) may show a few steps of transient divergence before the trajectory
re-converges. Recovery experiments that branch near a grasp attempt should be aware of this and
sanity-check a few branches by eye rather than assume bit-exact replay.

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
