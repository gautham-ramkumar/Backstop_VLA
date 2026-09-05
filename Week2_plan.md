---
name: Week 2 Failure Generation and Taxonomy
overview: Inject perturbations into LIBERO-Spatial, record a labelled corpus of roughly 400 episodes at K=4 with privileged state held outside the dataset, watch every failure, and publish a written failure taxonomy with per-episode onset annotations. No guard code this week.
todos:
  - id: preserve-week1
    content: "DONE. Week-1 corpus copied read-only to data/lerobot/week1_clean_k1_REFERENCE (229 MB, verified byte-identical via diff -r) and its 100 per-episode outcomes frozen into git-tracked docs/week1_reference.csv (sha256 c1067060..., 70 successes, per-task 8,8,8,4,8,7,8,6,9,4, all 30 failures at step 280). The 70/100 gate now compares against a file, not a memory."
    status: completed
  - id: throughput
    content: "Measure the record loop per stage (policy predict / OSMesa render / env step / frame write). Batch the K-1 extra chunk samples into one forward pass, leaving select_action untouched so the executed trajectory stays bit-identical. Decision gate: corpus size from measured s/episode."
    status: pending
  - id: shard-safety
    content: "Recording safety: shard datasets per perturbation family, stop create_dataset from rmtree-ing an existing root, resume from a partial shard, and move verify_corpus invariants into a per-episode in-loop assertion."
    status: pending
  - id: clean-rerecord
    content: "Re-record the clean 100 episodes at K=4 with privileged state. Hard gate: must reproduce 70/100 episode-for-episode. If it does not, the week-1 seeding contract is broken and everything downstream stops."
    status: pending
  - id: adr-003
    content: "ADR-003: action_policy vs action split, privileged sim state as an out-of-dataset sidecar with a stated prohibition on guard use, label sidecar format, K=4 as the corpus standard. Update schema.py and verify_corpus.py together."
    status: pending
  - id: sim-state
    content: "Capture the full flattened MuJoCo state every step via env.get_sim_state() (92 float64, ~206 KB/episode), not a curated pose dict. Assert round-trip restore through regenerate_obs_from_state in a contract test. Not retrofittable; see docs/recovery.md."
    status: pending
  - id: perturb-module
    content: "src/backstop/perturb/: lighting, camera, action noise (robosuite modders + loop hook), then layout (object pose jitter with validity check) and language (absent-object and paraphrase). CPU contract tests for each."
    status: pending
  - id: calibration
    content: "Strength sweep at K=1 on 3 tasks spanning difficulty (8 easy, 0 mid, 3 hard), 5 episodes x 3 strengths x 5 families. Pick the strength whose success rate lands in 30-70%. Publish the table."
    status: pending
  - id: distractor-spike
    content: "Timeboxed one-day spike on distractor-object injection via BDDL/MuJoCo model editing. If it does not land, substitute DynamicsModder (friction/mass) and state the perceptual-distractor gap as a limitation."
    status: pending
  - id: corpus-a
    content: "Record corpus A: clean arm + 5 families x 50 + absent-object slice + paraphrase control, sharded, every shard verified. ~16 h across overnight runs."
    status: pending
  - id: libero-plus
    content: "Timeboxed half-day spike: install the LIBERO-plus fork, check it against MuJoCo 3.3.2 and this lerobot, record a 60-80 episode external-validity slice (corpus B). Hard abort to week 6 if it is not recording by the end of the timebox."
    status: pending
  - id: review-tool
    content: "Auto proto-labeller from privileged state plus a local HTML review page (video, proposal, category buttons, onset scrubber). Same data contract as the week-5 viewer, deliberately."
    status: pending
  - id: taxonomy
    content: "Two-stage coding: open-code 25 failures, freeze a codebook, code all failures against it, re-code a blind 20% for self-consistency. Annotate an onset range on every failed episode. Write docs/taxonomy.md."
    status: pending
isProject: false
---

# Week 2 — Failure generation and taxonomy

Week 1 produced a working rollout loop and 100 clean episodes at 70.0% success. It also
produced the observation that shapes this week: **all 30 natural failures ran the full
280 steps.** None terminated early. The policy does not do something catastrophic and
stop — it stalls, retries, and runs out the clock. One failure mode is not a taxonomy.

Week 2's job is to produce failures that are *genuinely different from each other*, with
labels attached by construction, and then to look at every one of them.

## Decisions taken before writing this plan

| Decision | Choice | Consequence |
|---|---|---|
| Perturbation source | **Hybrid.** Own module with continuous strength knobs (corpus A) + LIBERO-plus as a published external-validity slice (corpus B) | Corpus A is balanceable, corpus B is comparable. B is timeboxed and abortable. |
| Second suite | **Spatial only.** `libero_object` deferred to week 4 | Week 4 gets a genuinely blind holdout, and re-opens the recording path under time pressure. Accepted. |
| Recording budget | **~20 h**, mostly overnight, sharded | ~400 episodes at K=4. Sized properly on day 1 from measurement, not from this estimate. |

### On the hybrid choice

LeRobot 0.6 already ships `LiberoPlusEnv` (`is_libero_plus=True`), wired to the published
LIBERO-plus robustness benchmark — ~10k variants across the seven perturbation dimensions
the brief names. Using it is strictly better for comparability. It is not sufficient on
its own, because its variants are discrete and the brief's own risk mitigation requires a
knob: *"Perturbation strength is a tunable knob. Sweep it in week 2 until the corpus is
balanced."* You cannot sweep a fixed variant set to a 50% failure rate.

So: own module drives the balanced corpus, LIBERO-plus validates externally. If the
install fights the MuJoCo 3.3.2 pin, corpus B moves to week 6 and the week does not slip.

---

## Day 1 gate: the K=4 throughput problem

The week-1 corpus was recorded at **K=1**, which means `sample_chunks` short-circuits and
costs nothing. Action-chunk disagreement — the first guard signal — cannot be computed
from it at all. The week-2 corpus must be K=4, and that is not free.

Measured week-1 baseline: 15,880 frames, corpus file timestamps spanning 11:25→13:52,
so **~0.57 s/step** end to end. (The README claims ~1.5 h for that run; the timestamps say
~2.5 h. Day 1 measures it properly rather than arguing about it.)

At `n_action_steps=1` every step is a full flow-matching chunk prediction with
`num_steps=10`. Naively, K=4 adds three more:

| | s/step | 400 episodes × ~210 steps |
|---|---|---|
| K=1 (week 1, measured) | 0.57 | 13 h |
| K=4 unbatched (projected) | ~1.5 | ~35 h |
| K=4 batched (target) | ~0.75 | ~17 h |

**The work:** batch the K−1 extra samples into a single forward pass with noise of shape
`(K-1, chunk_size, max_action_dim)`, tiling the observation across the batch dimension.
Leave `select_action` completely alone — the executed action stays on the unbatched path
with index-0 noise, so the executed trajectory is bit-identical to week 1 and there is no
reproducibility risk to argue about. Index 0 of the stored samples remains the executed
chunk by construction.

**The benchmark must produce a per-stage breakdown**, not one number. If OSMesa rendering
of two 360×360 cameras dominates the 0.57 s, then K=4 is cheaper than feared and the real
lever is elsewhere. Guessing which stage dominates is how you optimise the wrong thing.

**Decision gate, end of day 1:** measured s/episode at K=4 → corpus size. If batching does
not pay off, the fallbacks in order are K=3, then trim the paraphrase control, then trim
one family. Do not trim the clean arm and do not drop below K=3 — K=2 gives a spread
estimate from two samples, which is not a spread estimate.

---

## Recording safety

The question that sized this week's budget was *"what happens if the recording is bad?"*

Things you can fix offline, from data already on disk: disagreement metrics, temporal
consistency, any feature derived from stored chunks, taxonomy re-coding, thresholds.

Things that need the GPU again: **K samples, per-step chunks, the intended-vs-executed
action split, privileged sim state.** Week 1 shipped four defects in this class — zeroed
chunks, rotation-matrix state, upside-down video, chunks left in policy space — and every
one produced a plausible-looking dataset that raised no exception.

**Defense 0 — preserve the week-1 corpus. Done.** It was on disk at
`data/lerobot/backstop_spatial_baseline` (229 MB, 100 episodes, 15,880 frames, fps 20) with
`data/` gitignored and no second copy, at the exact path `create_dataset` `rmtree`s on start —
so the *first* week-2 record run would have destroyed it, including the clean re-record whose
whole purpose is to be compared against it.

Now copied read-only to `data/lerobot/week1_clean_k1_REFERENCE` and verified byte-identical
(`diff -r`, 7 files, 229 MB). The per-episode outcomes are frozen in git-tracked
[`docs/week1_reference.csv`](docs/week1_reference.csv) — `episode_index, suite, task_id, seed,
success, n_steps` for all 100, sha256 `c1067060…`. Cross-checks at write time: 70 successes,
per-task `8,8,8,4,8,7,8,6,9,4` matching the README, and all 30 failures at exactly step 280.
Recorded at `git_sha 779b913`, `k_samples=1`, `num_steps=10`, `seed=0`, MuJoCo 3.3.2.

The 70/100 gate is an *episode-for-episode* claim. A remembered aggregate is not the same test,
and a 229 MB blob on one laptop is not a reference.

Four further defenses, all landing before the first long run:

1. **Shard per family.** One dataset root per perturbation family. A bad family costs one
   shard, not the corpus. Today `create_dataset` does `shutil.rmtree(root)` on start, so a
   re-run silently destroys the previous corpus — fix that first.
2. **Resume.** A crash at episode 40 of 50 should resume, not restart.
3. **Verify during, not after.** The `verify_corpus.py` invariants become a per-episode
   in-loop assertion. A broken run dies in minute two, not hour six. `verify_corpus.py`
   stays as the post-run gate.
4. **Over-record the cheap insurance.** Privileged sim state costs almost nothing and is
   what makes onset annotation tractable. If it is not in the corpus, you re-record to get it.

Worst realistic case with these in place: one lost shard, 2–4 h.

---

## Perturbation families (corpus A)

`src/backstop/perturb/`, config-driven, each family a strength knob in `[0, 1]` mapped to
family-specific units. `robosuite 1.4.0` is installed and ships `LightingModder`,
`CameraModder`, `TextureModder` and `DynamicsModder`, which makes half of these cheap.

Access route: `make_libero_envs` builds a **Sync** vector env, so the real objects are
in-process — `vec_env.envs[i].unwrapped._env.env.sim` reaches the robosuite sim. Perturbations
that must land between `set_init_state()` and LIBERO's 10 settle steps get a `gym.Wrapper`
around each sub-env rather than a post-hoc mutation, so the returned observation is never stale.

| Family | Knob | Mechanism | Risk |
|---|---|---|---|
| `lighting` | diffuse/ambient scale, light position offset | `LightingModder` at reset, static per episode | low |
| `camera` | agentview position / rotation / fov jitter | `CameraModder` at reset. Wrist camera is rigidly mounted — do not jitter it | low |
| `action_noise` | Gaussian σ as a fraction of action range | injected in the loop between `env_post` and `vec_env.step` | low, but breaks a schema invariant (below) |
| `layout` | object xy σ (m) and yaw σ (rad) | write object free-joint qpos after `set_init_state`, `sim.forward()`, then let the settle steps run | **medium** — needs a validity check |
| `language` | n/a (categorical) | override `task_description`; store original and perturbed | low |
| `distractor` | n/a | **stretch** — BDDL/MuJoCo model edit before compile | **high**, timeboxed |

**`layout` validity check.** Jittered objects can end up interpenetrating, off the table,
or floating. After the settle steps: assert the object is within table bounds, is resting
(low velocity), and is not interpenetrating. Reject and resample with a capped retry count,
and **log the rejection rate** — a high rejection rate means the knob range is wrong, not
that the check is annoying.

**`language` has two sub-modes and both matter.**
- *absent object* — swap in a target noun from a different task in the suite, so the named
  object is not in the scene. This is the visual-precondition signal's target case.
- *paraphrase* — a benign rewording that should **not** change the outcome.

The paraphrase arm is the control that shows a guard is not simply detecting "the
instruction looks unusual". Without it, a language-sensitive guard is uninterpretable.

**`distractor` is the honest weak point.** Adding an object to a LIBERO scene means editing
the BDDL problem and recompiling the MuJoCo model — real surgery, and the brief names it
explicitly. It gets one timeboxed day. If it does not land, substitute `dynamics`
(`DynamicsModder`: friction, mass, damping on the target object), and **say so in the
taxonomy**: a friction shift tests physical robustness, not perceptual robustness, and
substituting one for the other leaves a perceptual-distractor gap in the corpus.

Optionally cheap seventh family if there is slack: `robot_init` (jitter the arm's initial
joint configuration). One of LIBERO-plus's seven dimensions and near-free to implement.

### Why every family targets ~50% success, not "always fails"

The failure mode that would quietly ruin weeks 3–4: **the guard learns to detect the
perturbation instead of the impending failure.** A lighting-shifted image is trivially
distinguishable from a clean one, and if lighting-shifted episodes always fail, "detect the
lighting shift" scores perfectly and generalises to nothing.

The defense is structural: tune each family's strength so roughly half its episodes still
succeed. Then perturbation presence carries almost no information about the outcome, and a
guard that scores well has to be reading something else. Week 3 must also report per-family
detection, never only pooled.

This is why the absent-object slice is quarantined: it is 0% success by construction, so its
perturbation flag *is* perfectly predictive. It stays out of the balanced arm and is
reported separately as the precondition-signal test case.

---

## Corpus design

### Matched pairs

LIBERO ties the initial state to reset ordering, not to the seed: `init_state_id =
episode_index`, incremented per reset. So episode *j* of a task always gets init state *j*,
run after run. That makes clean and perturbed episodes **exactly pairable** — same starting
scene, one knob different — so "did the perturbation cause this failure" is answerable per
pair rather than only in aggregate.

This is conditional on the reproducibility gate below. If the re-record does not reproduce
episode-for-episode, the paired design is void and we fall back to aggregate comparison.

### The reproducibility gate

Re-recording the clean 100 episodes at K=4 **must reproduce 70/100, episode for episode.**
The week-1 adapter already seeds flow noise per `(episode, step, sample)` precisely so that
K does not perturb the executed trajectory — its own docstring says a K-dependent divergence
means "a success-rate shift cannot be attributed to the perturbation rather than the
sampling." This is the test of that claim. If it fails, stop; nothing else recorded this week
would mean anything.

### Strength calibration

3 tasks spanning measured difficulty (task 8 at 9/10, task 0 at 8/10, task 3 at 4/10)
× 5 episodes × 3 candidate strengths × 5 families = 225 episodes at K=1, ~4 h.

15 episodes per cell gives roughly ±13pp on a 50% estimate. That resolves "barely any
effect" from "destroys the task" and nothing finer, which is all the target band (30–70%)
requires. Publish the table with the counts visible so nobody reads more precision into it
than it has.

### Shard table (~16 h, revised on day 1 from measurement)

| Shard | Episodes | K | Purpose | Est. |
|---|---|---|---|---|
| `clean` | 100 | 4 | reproducibility gate + matched-pair baseline + the K=4 clean arm | ~3 h |
| `lighting` | 50 | 4 | | ~2.1 h |
| `camera` | 50 | 4 | | ~2.1 h |
| `layout` | 50 | 4 | | ~2.1 h |
| `action_noise` | 50 | 4 | | ~2.1 h |
| `distractor` / `dynamics` | 50 | 4 | whichever the spike lands | ~2.1 h |
| `language_absent` | 30 | 4 | quarantined slice, 0% by construction | ~1.7 h |
| `language_paraphrase` | 20 | 4 | control arm | ~0.8 h |
| **total** | **400** | | | **~16 h** |

Plus ~4 h calibration ≈ 20 h. That is at the budget ceiling with no slack, so the trim
levers are named in advance: paraphrase 20→15, then one family 50→40. Not the clean arm.

### What this yields, stated honestly

~125–150 perturbed failures plus 30 natural ≈ **155–180 failures against ~250 successes.**

That is a thin corpus for the week-4 headline. At a 30% held-out split, the 5% false-stop
threshold is set from ~75 successes — it lands on the 4th-worst one, so it moves in ~1.3pp
quanta and the resulting detection rate carries a wide confidence interval. Week 4 should
report bootstrap CIs and consider episode-level cross-validation to use all the data, with
the genuinely-blind claim resting on `libero_object`.

Better to write that down now than to discover it while writing the week-4 report.

---

## Schema extension (ADR-003)

The week-1 schema is frozen; changing it takes an ADR. Four changes:

1. **`action_policy` (7,) alongside `action`.** The postprocessed action the policy asked
   for, before injected noise. Under `action_noise` these differ, which breaks the current
   `verify_corpus` invariant `action == action_chunk[0]`. New invariants:
   `action_policy == action_chunk[0]` always, and `action == action_policy` iff the family
   is not `action_noise`. Weakening the check without replacing it would silently retire the
   gate that caught the policy-space chunk bug.

2. **Privileged sim state lives outside the dataset.** Object poses, contact flags, grasp
   events, **and the full flattened MuJoCo state** go to `privileged/{shard}/{episode_index}.npz`,
   *not* into a LeRobot column. Two reasons: object count varies per task so a fixed column
   shape would be padding, and more importantly week-3 signal code physically cannot stumble
   into privileged state that is not in the dataset it loads. The prohibition is written into
   the ADR: **privileged state is permitted for ground-truth labels, onset annotation, and
   counterfactual branching, and forbidden as guard input.**

   Store the *whole* state, not a curated pose dict. LIBERO exposes the matched pair and
   exercises it on every reset — `env.get_sim_state()` and `env.regenerate_obs_from_state()`,
   the latter being what `set_init_state` already calls, so this is not a new code path to
   trust. LIBERO-Spatial states are 92 float64 (736 B), ~206 KB per episode, ~82 MB across
   corpus A. That buys the ability to restore to any step and branch, which is the difference
   between a later recovery experiment costing 45% of a rollout per variant and costing 100%.
   It is not retrofittable. Rationale in [docs/recovery.md](docs/recovery.md); recovery itself
   stays out of scope.

3. **Labels are a sidecar, not a column.** `artifacts/corpus/labels.jsonl`, keyed by
   `(shard, episode_index)`: perturbation family, strength, params, outcome, failure
   category, onset range, review status. Manual review rewrites labels repeatedly; rewriting
   a parquet dataset every time you change your mind about a category is not a workflow.

4. **K=4 is the corpus standard.** Recorded in provenance, asserted by the verify gate.

---

## Labelling and taxonomy

The brief is specific: *"Manually watch fifty failed episodes before designing any signal.
The taxonomy should come from observed failures, not from theory about what failure ought
to look like."* Meet that literally — ≥50 watched in full, not skimmed.

**Two-stage coding.**
- *Stage 1, open coding:* watch 25 failures spanning natural and every family. Write down
  what actually happens. No predetermined codebook. Then freeze one.
- *Stage 2:* code every failure against the frozen codebook. `other` is allowed. The
  codebook may be reopened **once**, and the reopening is logged in the taxonomy doc.
- *Self-consistency:* blind re-code of a random 20% two days later; report agreement. One
  rater is a limitation of this project — state it rather than implying inter-rater reliability.

**Auto proto-labels.** From privileged state, compute per episode: was the object ever
grasped (contact + lift), ever displaced, did the gripper close near it, terminal
object-to-goal distance, eef path length, gripper open/close cycle count. Simple rules turn
these into a *proposed* category. The human confirms or overrides. Report the auto-vs-human
agreement rate — it is a free number, and it tells week 4 whether onset detection can be
automated at all.

**Onset annotation — the deliverable that is easiest to forget.** Week 4 measures detection
lead time and week 5's viewer draws "a second marker where the failure actually occurred".
Both need ground truth for *when*, and only this week's review can produce it.

It is genuinely ambiguous, so record it as a **range, not a point**: the earliest and latest
defensible step at which the episode became unrecoverable. Operational definition: the first
step after which no subsequent step shows progress toward the success condition, with the
auto-proposal seeded from the last grasp event or the last approach to the target. Week 4
reports lead time against both ends of the range.

**Review tool.** A local HTML page per shard: episode video, auto-proposal, category buttons,
onset scrubber, writing to `labels.jsonl` through a tiny local server. Built against the same
episode assets and data contract the week-5 viewer will use — this is a week-5 down payment,
not a throwaway.

---

## Corpus B — LIBERO-plus

Half-day timebox. Install is a git clone of the fork plus `PYTHONPATH` (it ships as a
namespace package, not a pip extra). Compatibility checks: MuJoCo 3.3.2, this lerobot's
`get_task_init_states` (which reads a `libero_newobj/` init-states path for the plus fork),
and the asset download.

If it lands: 60–80 episodes across their variant dimensions, same schema, same verify gate,
reported separately as external validity — "here is our balanced corpus, and here is the
policy on a published perturbation benchmark."

**Abort rule:** if it is not recording by the end of the timebox, it moves to week 6 buffer
and this week does not slip. Written down now so the decision is not made at 2am.

---

## Day plan

Human work overlaps the recording; the GPU runs overnight.

| Day | Foreground | Overnight |
|---|---|---|
| 1 | **Preserve + freeze the week-1 corpus (first, 30 s).** Per-stage throughput benchmark, batched K sampling, shard/resume/no-rmtree, corpus-size gate, **full sim-state capture + restore round-trip test** | `clean` re-record at K=4 |
| 2 | **Verify 70/100 reproduced.** ADR-003, schema, rest of privileged capture, verify gates, contract tests. Cheap families: lighting, camera, action noise | calibration sweep |
| 3 | `layout` with validity check, `language` both sub-modes | calibration sweep continues |
| 4 | Pick strengths, write shard configs, 20-episode canary fully verified. `distractor` spike. LIBERO-plus spike | shards 1–2 |
| 5 | Review tool + auto proto-labeller against completed shards. Stage-1 open coding on 25 failures | shards 3–5 |
| 6 | Code all failures, onset annotation, `docs/taxonomy.md`, README update | corpus B or deferral note |

---

## Risks

| Risk | Severity | Response |
|---|---|---|
| Week-1 corpus destroyed by the first week-2 record run | **highest, and live today** | It is unbacked, gitignored, and sits at the path `create_dataset` `rmtree`s. Copy + `chmod -R a-w` + freeze outcomes to `docs/week1_reference.csv` before anything else runs. Without it the 70/100 gate has nothing to compare against. |
| Clean K=4 re-record does not reproduce 70/100 | **highest** | Stop everything. The week-1 seeding contract is wrong and no week-2 number would be attributable. |
| Batched sampling does not pay off | high | Day-1 gate. Fall back K=3 → trim paraphrase → trim one family. Never below K=3. |
| Every perturbation produces the same stall-and-timeout mode | high | A real finding, and a weak deliverable. The families are chosen to hit different subsystems (perception / grounding / control / precondition) precisely to avoid it. If they still collapse, report it — it also predicts week 3's signals will struggle. |
| Guard could learn perturbation, not failure | high | Structural: ~50% success per family, paraphrase control, absent-object quarantined, per-family reporting mandatory in week 3. |
| `layout` produces invalid scenes | medium | Reject-and-resample with logged rejection rate. |
| Corpus too thin for a tight week-4 interval | medium | Named above with numbers. Bootstrap CIs and cross-validation in week 4; do not pretend the interval is narrow. |
| `distractor` surgery fails | medium | Timeboxed, substitute `dynamics`, declare the perceptual gap. |
| LIBERO-plus install fights the MuJoCo pin | low | Timeboxed, deferred to week 6. Corpus A does not depend on it. |

| Sim-state capture misses the `clean` shard | medium | It is day-1 foreground work precisely because the clean re-record runs on day-1 overnight. If it slips, hold the re-record — the clean arm is the shard a later branching experiment most needs. |

Disk: ~400 episodes × ~5 MB ≈ 2 GB, plus privileged sidecars (~82 MB of sim state). Not a constraint.

---

## Explicitly out of week 2

Guard signals of any kind, the trust head, thresholds, `libero_object`, the hosted viewer,
OpenVLA, policy training, recovery behaviour. The review tool is a labelling instrument, not
the week-5 viewer.

Recovery stays out. Week 2 pays only the one cost that cannot be paid later — full sim-state
capture — and nothing else. The argument, the evaluation design, and the kill criteria are
written down in [docs/recovery.md](docs/recovery.md) so the decision can be made after the
week-4 number exists rather than on enthusiasm now.

## Done when

- Week-1 corpus copied read-only and its per-episode outcomes frozen into a git-tracked
  `docs/week1_reference.csv`
- ADR-003 committed; `schema.py` and `verify_corpus.py` updated together; CI green on CPU
- Clean K=4 re-record reproduces **70/100, episode for episode**, checked against that file
- `src/backstop/perturb/` with CPU contract tests for every family
- Calibration table published, with per-cell counts visible
- Corpus A recorded and sharded; **every shard passes verification**
- Privileged state captured for every episode, outside the dataset — including the full
  flattened sim state, with a contract test proving a mid-episode state restores and
  round-trips through `regenerate_obs_from_state`
- ≥50 failures watched in full; every failure coded against a frozen codebook; blind 20%
  re-code agreement reported
- Onset range annotated on **every** failed episode
- `docs/taxonomy.md`: categories, counts, two representative episodes each, and an explicit
  statement of which categories are too thin to evaluate against
- Corpus B recorded, or deferred with the reason written down
