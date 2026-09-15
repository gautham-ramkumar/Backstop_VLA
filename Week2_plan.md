---
name: Week 2 Failure Generation and Taxonomy
overview: Inject perturbations into LIBERO-Spatial, record a labelled corpus of roughly 400 episodes at K=4 with privileged state held outside the dataset, watch every failure, and publish a written failure taxonomy with per-episode onset annotations. No guard code this week.
todos:
  - id: preserve-week1
    content: "DONE. Week-1 corpus copied read-only to data/lerobot/week1_clean_k1_REFERENCE (229 MB, verified byte-identical via diff -r) and its 100 per-episode outcomes frozen into git-tracked docs/week1_reference.csv (sha256 c1067060..., 70 successes, per-task 8,8,8,4,8,7,8,6,9,4, all 30 failures at step 280). The 70/100 gate now compares against a file, not a memory."
    status: completed
  - id: throughput
    content: "MEASURED AND BATCHED. Per-stage numbers in docs/throughput.md (`just benchmark`). The step is policy-bound, not render-bound: at K=1 the forward pass is 56% and OSMesa render 29%, so sampling is the lever. K=1 0.775 s/step, K=4 unbatched 1.594 s/step (2.06x); the three extra samples cost 328 ms each against 337 ms for the executed one. Batched K-1 sampling landed in policy/adapter.py, checked against the real policy (not just fakes): batched vs sequential diverges up to 0.64 relative on individual chunk values (batch-size-dependent GEMM kernels), but the chunk *spread* -- the actual guard signal -- agrees to 0.4%, and select_action never touches the batched path so the executed trajectory is unaffected. OOM falls back to sequential automatically. Also found: env_reset 6.8 s/episode (hard_reset rebuilds model + renderer) and save_episode 4.9 s/episode, ~1.3 h across the corpus and independent of K. Episode length is 158.8 steps at week-1's 70%, not the 210 assumed here. Remaining: re-run the corpus-size projection with the batched number now that it exists."
    status: completed
  - id: shard-safety
    content: "DONE. config.py record.on_existing (fail/resume/overwrite) replaces the unconditional rmtree; record.shard field; LeRobotDataset.resume() skips already-recorded episodes. Covered by tests/unit/test_record_safety.py (264 lines). verify_corpus.py remains the post-run gate; it has not yet been made an in-loop per-episode assertion, so a bad shard is still caught after the fact, not during recording."
    status: completed
  - id: clean-rerecord
    content: "DONE. Ran overnight (2026-09-10 01:00-05:41, no crashes), configs/record/spatial_clean_k4.yaml. success_rate=0.700 (70/100), matching week 1's aggregate. THE GATE: compared all 100 episodes against docs/week1_reference.csv on (task_id, seed, success, n_steps) -- 100/100 match exactly, including n_steps, not just pass/fail. scripts/verify_corpus.py: 20/20 checks passed, K samples now genuinely spread (max spread=2.112, was 0 at week-1's K=1), action==action_policy holds for the clean shard (ADR-003). The batched K=4 sampling does not perturb the executed trajectory -- seeding contract holds, paired clean/perturbed design for corpus A is sound. Full log in artifacts/logs/overnight_session_notes.md."
    status: completed
  - id: adr-003
    content: "DONE. docs/adr/003-perturbation-schema.md written. action_policy field landed in schema.py + lerobot_features; record/loop.py now actually calls perturbation.perturb_action() and backstop.perturb.attach() (previously built but unwired -- config.py gained record.perturbation/perturbation_strength/perturbation_kwargs to select a family per run); verify_corpus.py checks action_policy==action_chunk[0] unconditionally and action==action_policy iff not action_noise, with a fallback path for pre-ADR-003 datasets with no action_policy column. Covered end-to-end in tests/unit/test_rollout_loop.py against fakes, and confirmed for real: a live 1-episode recording with action_noise at strength 1.0 (real simulator, real policy) passed 18/19 verify_corpus.py checks -- action_policy==action_chunk[0] held and action correctly differed from action_policy, i.e. the perturbation actually reached the executed action, not just the label. (The 19th check, 'at least one episode ended before the step cap', is a single-episode sample-size artifact, not a real failure -- week 1 already established every natural failure hits the 280-step cap.) Label sidecar format (labels.jsonl) is decided in the ADR but not yet built -- that's the review-tool/taxonomy work later in the week."
    status: completed
  - id: sim-state
    content: "DONE. Capture via PrivilegedWriter + env.get_sim_state() (92 float64, ~206 KB/episode), measured at 0.08ms/step (free). Round-trip restore through regenerate_obs_from_state verified in scripts/verify_sim_state_roundtrip.py (`just verify-sim-roundtrip`), against the real simulator, not a fake -- passes to float noise (~1e-6) under smooth motion. Caveat found and written up in docs/recovery.md: branching mid-chatter (rapid gripper toggling, large uncorrelated joint jumps) shows real transient divergence for a few steps, traced to qacc_warmstart not being part of robosuite's flattened state. Does not block the clean re-record; does matter for recovery experiments that branch near a grasp attempt."
    status: completed
  - id: perturb-module
    content: |
      DONE. src/backstop/perturb/: lighting, camera, object_pose (layout jitter,
      reject-and-resample validity check -- table bounds, resting velocity,
      MuJoCo contact penetration, capped at 5 attempts, counted, verified
      against the real simulator), action_noise, distractor (real BDDL/MuJoCo
      surgery, not the DynamicsModder fallback -- see `distractor-spike`), and
      both language sub-modes (`absent_target`, `language_paraphrase`,
      `task_description` override wired through `PerturbedEnv.reset()`) --
      all six families implemented, registered in `FAMILIES`, and wired end to
      end. ADR-003 landed record/loop.py's actual calls to
      `perturbation.perturb_action()` and `backstop.perturb.attach()` plus
      config.py's `record.perturbation`/`perturbation_strength`/
      `perturbation_kwargs`, so any family is selectable for a real run --
      exercised by the calibration sweep (72 cells across five families) and by
      five real corpus-A shards recorded and verified at K=4.

      `absent_target`/`language_paraphrase` confirmed against the real simulator
      with the exact kwargs corpus A now records with (`replacement`/
      `paraphrase` strings, checked via `PerturbedEnv.task_description`), not
      just the fakes in test_perturb.py. One discovery worth recording: every
      one of libero_spatial's 10 tasks shares the identical object roster
      (checked directly against all 10 bddl files) -- so `absent_target`'s
      "swap in a noun from a different task in the suite" had to borrow from
      LIBERO's broader benchmark instead (`moka pot`, from libero_90), since
      libero_spatial itself has no internal object diversity to swap from.
    status: completed
  - id: calibration
    content: |
      DONE, including the lighting fix and a fourth pass added once `distractor` was built
      (see `distractor-spike`). scripts/calibration_sweep.py, 4 passes: 36 cells at strengths
      [0.3, 0.6, 1.0] (the 4 continuous-strength families that existed at the time --
      language is categorical, distractor not yet built); an 18-cell refinement at
      [0.1, 0.15, 0.2] for object_pose and action_noise once the first pass showed their
      whole tested range was too aggressive; a 9-cell lighting-only retest after fixing a
      real bug the first pass surfaced (see families.py Lighting docstring and below); a
      9-cell `distractor` pass at the original [0.3, 0.6, 1.0] strengths once the family
      landed. 72 cells run across all four passes, 63 live in
      artifacts/calibration/sweep_results.csv (9 pre-fix lighting cells superseded and
      backed up separately, see below), ~340 episodes, K=1, eval-only, zero crashes across
      all four passes (~14h wall clock combined). Baselines: task 8 = 90%, task 0 = 80%,
      task 3 = 40%, pooled = 70%.

      Lighting bug: the first pass found `lighting` flat at 60-67% across its *entire*
      tested range including strength 1.0 (the ceiling) -- not a calibration problem, a
      real one. Diagnosed against the real simulator (mean agentview pixel value, not
      guessed): dimming the two named scene lights to 20% only moved mean brightness
      120 -> 87, because MuJoCo's camera headlight (`model.vis.headlight`) is a separate,
      always-on light source `LightingModder` does not touch. Scaling the headlight the
      same way took it to 11 and produced a real gradient. Fixed in `Lighting.on_reset`;
      old lighting rows in artifacts/calibration/sweep_results.csv backed up
      (.bak_pre_lighting_fix) and re-run at the same three strengths.

      | family        | 0.10 | 0.15 | 0.20 | 0.30 | 0.60 | 1.00 |
      |---------------|------|------|------|------|------|------|
      | lighting      |  -   |  -   |  -   | 73%  | 60%  | 47%  |
      | camera        |  -   |  -   |  -   | 60%  | 33%  | 27%  |
      | object_pose   | 60%  | 53%  | 53%  | 33%  | 13%  | 7%   |
      | action_noise  | 60%  | 53%  | 40%  | 33%  | 7%   | 0%   |
      | distractor    |  -   |  -   |  -   | 67%  | 67%  | 53%  |

      `distractor` needed no refinement pass -- all three tested strengths already land in
      the 30-70% band. Flagged, not smoothed over: 0.30 and 0.60 gave *identical* per-task
      results (task 0/3/8 each 3/5, 3/5, 4/5 both times), the same "flat across a strength
      range" shape that exposed the lighting bug -- plausibly n=5 coincidence rather than a
      second bug, since 1.00 does differ from both, but worth a second look before fully
      trusting the gradient.

      Chosen strengths for corpus A: lighting 0.6 (pooled 60%, headlight fix applied),
      camera 0.6 (pooled 33%, real monotonic gradient), object_pose 0.15 (pooled 53%,
      balanced 40/40/80 across task 0/3/8), action_noise 0.15 (pooled 53%, same 40/40/80
      spread), distractor 1.0 (pooled 53%, most balanced of the three tested strengths at
      40/60/60 -- 0.3 and 0.6 both pooled 67% with task 8 running hot at 80%). All five
      families now have a defensible strength; nothing blocking corpus A on this front.
    status: completed
  - id: distractor-spike
    content: |
      DONE, and it landed the real distractor-object route, not the DynamicsModder
      fallback this todo pre-approved. `Distractor` lives in
      src/backstop/perturb/families.py alongside every other family, registered in
      perturb/__init__.py's FAMILIES the same way, selected via
      `make_perturbation("distractor", strength)` -- not a separate module. The one
      structural addition is a new `prepare_env` hook on `PerturbedEnv`
      (wrapper.py), because `distractor` is the only family that has to act
      *before* the env's first reset (BDDL edit, so MuJoCo compiles a different
      model) rather than after it like every other family's `on_reset`; the hook
      is opt-in via `getattr` and a no-op for every other family.

      Mechanism: all 10 libero_spatial task BDDL files share one byte-identical
      `:regions`/`:objects` block, so one fixed distractor object (`milk_1`) and
      region (`(0.30, -0.25, 0.36, -0.19)`, main_table-relative) generalizes
      across the suite -- checked clear of every declared region with margin, not
      assumed. Injected via paren-depth text splicing, not fixed-offset
      replacement, so it survives whitespace differences between task files.

      The pairing problem this todo was named "high risk" over: a `.pruned_init`
      row is a flattened `time+qpos+qvel` state sized for the original model (92
      floats for libero_spatial); adding a free-jointed object changes that to
      105. Solved by building the augmented model once at attach time to read
      back where MuJoCo actually placed the new joint's qpos/qvel addresses, then
      splicing every row of the original init-state array at that address, with a
      per-episode xy/yaw jitter seeded through the existing `episode_rng`
      convention. Falls back to an explicit unpaired mode (`params()["paired"] =
      False`) only if `init_states=False`, which is not the recording default.

      Verified against the real simulator, independently of the implementing
      agent's own report: reset a clean episode and a distractor episode at the
      same episode index, stripped the distractor's 7 qpos values out of the
      perturbed model's qpos vector, and diffed against the clean model directly
      -- max abs difference 9e-18 (float noise) for the robot arm and every
      pre-existing object. 146 unit tests pass (CPU-fakeable contract tests in
      test_perturb.py + test_perturb_wrapper.py), ruff/pyright clean.

      Named limitation, in the class's own docstring: strength currently scales
      placement *variety* within one fixed off-workspace region, not
      proximity/salience to the workspace -- a closer, verified-collision-free
      anchor point was out of scope for the timebox. Calibration should read
      `distractor` strength as "placement spread," not "distraction intensity."

      Not done, and not this todo's scope: no
      configs/record/spatial_distractor_k4.yaml yet -- `distractor` still needs
      its own 3-task x 3-strength calibration pass (same as the other four
      families) before it can record a real corpus-A shard.
    status: completed
  - id: corpus-a
    content: |
      DONE. All 400/400 episodes recorded and verified. Original scope: clean
      arm + 5 families x 50 + absent-object slice + paraphrase control,
      sharded, every shard verified.

      The clean arm (100 episodes, K=4, see `clean-rerecord`), all four
      continuous-strength families recorded overnight 2026-09-11 00:01-08:51
      (~8h50m) via scripts/record_corpus_a_continuous_families.sh -- lighting
      70.0%, camera 58.0%, object_pose 56.0%, action_noise 66.0%, all
      `verify_corpus.py` 20/20 and inside [0.30, 0.70] -- and `distractor`:
      9-cell calibration pass (tasks 8/0/3 x strengths [0.3, 0.6, 1.0], pooled
      66.7%/66.7%/53.3%, see `calibration`) picked strength 1.0, recorded
      13:42-15:40 the same day, 50/50 episodes, success_rate=0.660,
      `in_tolerance=True`, `verify_corpus.py` 20/20. Zero crashes, zero
      out-of-tolerance shards across these five families.

      `language_absent` (30 episodes, quarantined, 0% by construction):
      recorded 2026-09-11 18:48, `verify_corpus.py` 19/20 (the one expected
      "failure" is that every episode hits the step cap, which is the point of
      this arm).

      `language_paraphrase` (20 episodes, control arm on task 0 only, since
      `record/loop.py` takes one `Perturbation` instance per run): took two
      attempts, and the result is a real finding, not a data-pipeline defect.

      First attempt (2026-09-11 19:12-19:48) reached only 10/20 episodes before
      the laptop was powered off mid-run (`journalctl`: `systemd-logind`
      "system will power off now" at 19:49:13, ~27s after episode 10 finished)
      -- the parquet file was left without its footer (valid `PAR1` header,
      no trailing magic, exactly what a writer killed before `close()` leaves
      behind), not corrupted by anything in the recording code. All 10
      completed episodes had already failed (0/10), which at the time looked
      alarming enough to investigate before just resuming.

      Investigation ruled out a pipeline bug: the initial frame of paraphrase
      episode 0 is pixel-identical to clean-arm episode 0 (same task 0, same
      seed 0), confirming the seeding/pairing held; the code path
      (`PerturbedEnv.reset()` in wrapper.py overriding `task_description`,
      re-read into `observation["task"]` every step in `record/loop.py`)
      correctly delivers the reworded instruction to the policy for the whole
      episode, not just at reset. Watching 5 of the 10 recorded videos showed
      the same behavior every time: the arm engages the correct bowl but never
      completes the transport-and-place step before the 280-step cap -- the
      same "stalls and runs out the clock" signature week 1 found in natural
      failures, just happening far more often.

      Second attempt (2026-09-15, fresh re-record, same wording, same
      task_ids/seeds, prior incomplete shard moved aside rather than deleted)
      completed cleanly: 20/20 episodes, `success_rate=0.150` (3/20, at episode
      indices 15/17/19), `in_tolerance=False` against the configured [0.55,
      1.0] band, `verify_corpus.py` 20/20, all 20 privileged sidecars present.
      This reproduces the first attempt's rate almost exactly (0/10 vs. this
      run's first 10) and rules out "unlucky truncated sample" -- two
      independent recordings of the same wording both collapse task 0's clean
      80% baseline, landing at 15% pooled.

      **Finding that changes the plan's own assumption:** `language_paraphrase`
      was designed as a control that "must not move the outcome" (see
      `language has two sub-modes` below) -- a benign reword that preserves the
      spatial disambiguator (task 0's "between the plate and the ramekin")
      should leave success near the 80% clean baseline. It does not. SmolVLA
      here appears to be reading something much closer to the exact training
      phrasing than the semantic content, which is itself a real language-
      robustness result worth reporting in the taxonomy, not a bug to fix
      before recording. It does mean this arm can no longer be read as a
      neutral control in week 3/4 analysis -- it has to be reported as its own
      (surprisingly destructive) perturbation, alongside a note that the
      "detect perturbation vs. detect impending failure" concern this control
      was meant to rule out is now unresolved for the language family.

      Also found in passing: `record/cli.py`'s exit code (2, whenever the
      final rate lands outside `cfg.tolerance`'s band) fires regardless of
      `cfg.tolerance.enforce`, which every other shard's own convention
      ("`enforce: false` ... record and report `in_tolerance=False` rather
      than abort") describes as governing this. Every other shard this week
      happened to land in-tolerance, so this is the first time the exit path
      was exercised -- it silently broke the `&&`-chained `verify_corpus.py`
      step, which had to be run manually afterward (and passed 20/20 once
      run). Not a defect in the recorded data, but worth knowing before
      scripting future out-of-tolerance-prone shards the same way.

      Corpus A total: 100 (clean) + 50x5 (lighting/camera/object_pose/
      action_noise/distractor) + 30 (language_absent) + 20
      (language_paraphrase) = 400/400 episodes recorded and verified.
    status: completed
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
- *paraphrase* — designed as a benign rewording that should **not** change the outcome.

The paraphrase arm was meant to be the control that shows a guard is not simply detecting
"the instruction looks unusual". Without it, a language-sensitive guard is uninterpretable.

**This assumption did not survive contact with the recording.** `language_paraphrase` was
recorded twice (see `corpus-a`) and both times collapsed task 0's 80% clean baseline —
final pooled result 15% (3/20), reproduced across two independent runs, despite the
paraphrase keeping the exact spatial disambiguator the original instruction used. SmolVLA
appears to be reading something much closer to the exact training-set phrasing than the
semantic content. That is a real language-robustness finding for the taxonomy, not grounds
to discard the arm — but it means this shard has to be reported as its own (surprisingly
destructive) perturbation in week 3/4, not treated as a neutral control, and the "guard
detects perturbation vs. detects impending failure" risk this arm was meant to rule out is
still open for the language family.

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
