# Refactor plan: seams, typed corpus, smaller code

**Status:** every stage done (2026-10-25). See "Stages 5–8, 1, 2, 4: done" below. The Week 3–6 work is committed, and the experiments
that lost were archived on 2026-10-18 (tag `research-archive-2026-10`,
[experiments.md](experiments.md)). This plan starts from that smaller base.

## Why

The 2026-10 review found the research logic sound and the structure hard to change:

- **Adding things is expensive.** A new policy needs about 7 edits, and there is no
  seam for a second simulator.
- **The rollout loop does everything in one place.** `_rollout_one` is 146 lines with 18
  parameters and about a dozen jobs. The recovery runner has to slot into the middle of it.
- **Third-party internals are scattered.** Private lerobot/LIBERO attributes
  (`unwrapped._env`, `_format_raw_obs`, `_task_bddl_file`, `_get_action_chunk`, ...) are
  reached from four packages, so a dependency upgrade can break any of them.
- **One "episode" has several shapes.** `EpisodeSignal`, trace-row dicts, label-row
  dicts, privileged npz and loop stats rows, plus three kinds of key. The untyped dicts
  are why `as_float`/`as_int` and the remaining `type: ignore`s exist.
- **`guard` depends on `label`** just to find and read datasets.

The goal is the recovery runner as a small, testable addition, and new policies and
perturbation families as one- or two-file changes.

## Ground rules (every stage)

1. **Behaviour is proven unchanged, not assumed.**
   - Analysis code must reproduce every report, `traces.jsonl`, `labels.jsonl` and
     frozen rule byte for byte.
   - Recording code must reproduce the golden rollout digests (Stage 0).
2. **One stage, one commit (or a short series).** Each commit leaves CI green: ruff,
   pyright and the strict test suite.
3. **No on-disk format changes.** Parquet columns, npz keys, the fields and key order
   of traces/labels/run.json, and timer stage names all stay as they are.
4. **No speculative abstraction.** A protocol is added only when a second
   implementation or a test fake needs it. Each stage says which.
5. **Behaviour fixes found along the way get their own commits**, never folded into a
   refactor. One is already known: `perturbation_strength=None` is coerced in
   `guard/extract.py` but passed through in `label/extract.py`.
6. **The library and scripts must get smaller.** Every commit message reports the net
   change in code lines (comments, docstrings and blank lines excluded) for `src/` plus
   `scripts/`. Tests may grow. A stage that ends net-positive by more than its estimate
   below must say why in its commit message, or be cut back.

## Size: where we are, and the target

Code lines only, comments, docstrings and blank lines excluded:

| | `src/` + `scripts/` | `tests/` |
|---|---|---|
| Archive tag (all experiments live) | 7,450 | 2,849 |
| **After archiving the losing experiments** | **5,405 (−27%)** | 2,605 |
| Target after this plan | ~5,230 (−30% overall) | ~3,250 |
| **After this plan (measured)** | **5,392 (−28%)** | 3,372 |

Most of the shrinking has already happened, by deleting experiments rather than by
refactoring. The archive removed 12 files and trimmed 9 others, and every live number
stayed byte-identical. The stages below mostly add structure, which costs a few lines,
so the remaining saving comes from Stage C.

## Order

```
Stage 0 safety net ── C delete ──┬── 5 simulator seam ── 6 policy seam ── 7 split rollout ── 8 thin pipeline
                                 │                                        └─ recovery runner can start here
                                 └── 1 corpus reader ── 2 typed episodes ── 4 config + constants
```

There was a Stage 3, a guard signal registry. It was dropped: with one live signal there
is nothing to register.

The two tracks share only `schema.py` and `config.py`, so they can interleave.
**Recommended order for one person:** 0 → C → 5 → 6 → 7, build the recovery runner,
then 1 → 2 → 4 → 8. The recording track comes first because recovery depends on it.

"Touched" is how many lines a stage moves or edits, which is the review size. "Net"
is the estimated change in code lines for `src/` plus `scripts/` once the stage is done.

| Stage | What | Effort | Touched | Net (src+scripts) | Risk |
|---|---|---|---|---|---|
| 0 | Golden digests, snapshot, import-layer test — **done** | S | 580 | +224 measured (tests +354) | low |
| C | Delete obsolete scripts and the last duplication — **done** | S | 900 | **−207 measured** | low |
| 5–6 | Simulator and policy seams — **done** | M | 500 | −60 est., **−1 measured** | medium |
| 7 | Split `_rollout_one` into obs / decide / sink — **done** | L | 350 | +40 est., **+51 measured** | **high** |
| 8 | Thin `run_pipeline` — **done** | S | 120 | −10 est., **−2 measured** | low |
| 1 | `backstop/corpus/` reader — **done** | M | 250 | −50 est., **−21 measured** | low–medium |
| 2 | Typed `EpisodeKey` + `TraceRow` — **done** | S–M | 200 | −30 est., **+3 measured** | medium |
| 4 | Shared constants and defaults, plus unused re-exports removed — **done** | S | 150 | −40 est., **−63 measured** | low |
| | **Total after Stage C** | | | **5,425 → 5,392 (−33)**; tests 3,100 → 3,372 | |

Stage 7 is the one stage expected to grow. Replacing 18 keyword arguments with two small
dataclasses, and one long function with four named pieces, costs a few lines. That's the
price of making the recovery runner a reuse instead of a copy.

## Stages 5–8, 1, 2, 4: done

Proved unchanged at every stage: the golden rollout test, the analysis snapshot
(identical to the Stage C baseline after Stages 1, 2 and 4, and at the end), and the
real-simulator recording (`just compare-recordings`, identical to a baseline taken
before Stage 5 after Stages 5, 5–6, 7–8–1 and at the end, with the same timing
stages and call counts). `just verify-sim-roundtrip` passes 8/8. The import-layer
allowlist is empty.

Where the code differs from the plan below, and why:

- **Stage 5. `restore_observation` matches the run's observation closely, not
  pixel for pixel.** After `mj_step`, MuJoCo's body and camera poses still
  describe the start of the last 2 ms substep, and a restore recomputes them for
  the state itself. Measured right after a reset: images differ by under 0.1 grey
  level on average (0.08% and 3.5% of pixels on the two cameras), proprioception
  by under 3e-4. The sim state itself restores exactly. The verify script checks
  that tolerance, and that two restores of one state are identical.
- **Stage 5. One `Perturbation` base class, not a protocol plus `PerturbationBase`.**
  The defaults (no-op hooks, identity `task_description`, `from_config`) live on
  the base, so the three `getattr(..., True)` guesses and the `_CATEGORICAL` table
  are gone. `swap_task_files` takes a rewrite function, so the "simulator not yet
  built" check sits beside the private attributes it guards.
- **Stage 5. New `tests/unit/test_private_attributes.py`:** each LeRobot/LIBERO
  private attribute may be used only in `env/libero.py` or `policy/smolvla.py`.
- **Stage 6. The fake is `tests/fakes/policy.py`**, next to Stage 0's fakes.
  Policy loading moved out of `eval`, so its tests (`test_policy_loading.py`, was
  `test_eval_pipeline.py`) now run in the CPU job.
- **Stage 7. `Decider` is a small class, not a function.** It holds the policy,
  processors, K and timer, and takes the perturbation as a callable: the
  `action_post` timer stage has always included applying it, and the stage counts
  must not change.
- **Stage 8. The record CLI merged into `eval/cli.py`** (`record_main`); the two
  were near copies. `docs/failures.md` is now updated by `just publish-failures`,
  never by a run.
- **Stage 1. `ShardLayout.provenance` returns the schema's own `RunProvenance`.**
  Every `run.json` on disk validates, and every stored strength was already a
  float, so the `perturbation_strength=None` question (ground rule 5) no longer
  arises. `load_shard_signals` already read one episode at a time.
- **Stage 2. `EpisodeSignal` is gone:** its two perturbation fields were never
  read, so the extractor returns the same `TraceRow` the scripts read back, one
  episode shape instead of two. No `EpisodeMeta`: nothing else would use it.
  `TraceRow` compares by identity, since value equality over arrays has no single
  answer.
- **Stage 4. No `CorpusPaths`, `GuardEvalConfig` or `guard/cli.py`.** The roots'
  defaults are `ShardLayout`'s (one source), `ShardLayout.add_args` adds the four
  root flags to every script that reads shards, `TRACES`/`LABELS` sit beside it in
  `corpus/layout.py`, and `EVAL_FRACTION` beside the split. Five scripts gained
  root flags they do not use; every default is unchanged (checked by parsing each
  script's arguments). Budgets come from the metric contract.
- **Also removed:** the unused package-level re-exports (`backstop`, `eval`,
  `record`), the `dataset_features` wrapper, a lazy-numpy helper (numpy is a core
  dependency now) and a second `as_numpy`.

## Stage C: delete the rest — done

| What | Done | Code lines |
|---|---|---|
| `scripts/calibration_sweep.py`, `scripts/record_corpus_a_continuous_families.sh` | Deleted; referenced from nowhere | −170 |
| `guard_viewer.py` ↔ `review_server.py` | One shared base, `backstop.label.web`: video clip cache, JSON replies, traversal-safe assets, Range video. The guard viewer no longer borrows the review server's private methods | −60 net |
| `verify_corpus.py` ↔ `verify_sim_state_roundtrip.py` | One PASS/FAIL harness, `backstop.checks`. `verify_corpus` also dropped its private copy of `parquet_io.as_array` | −25 |
| History-diary prose | About 20 comments and docstrings reworded to say why, not when. Kept: design-document citations, real file names, and any text written into outputs | (prose) |
| Leftovers | The unused `CONTROL_FPS`, two orphaned comments left by the archive deletion | −2 |

- **Fixed along the way:** the guard viewer crashed on its main page, because it still read the `trust` field the archive removed from traces. It had no tests. It now has route-level tests against a real local server, as does the shared base.
- **Verified:**
  - the analysis snapshot is identical to the Stage 0 baseline;
  - both verify scripts print byte-identical output before and after (19/19, 6/6);
  - 334 tests pass with no skips allowed;
  - the import-layer test's script exception is gone (scripts import no other script).
- **Size:** `src/` + `scripts/` −207 (`src/` +127 for the shared modules, `scripts/` −334), tests +141.

---

## Stage 0: safety net — done

Before refactoring anything, make "unchanged" something we can check. Four checks,
each proven to fail when it should (a deliberate break was caught, then reverted):

- **Golden rollout test** (`tests/unit/test_rollout_golden.py`, sim CI job).
  - Runs the real `run_episodes` on the CPU fakes with every recording feature on:
    K=4, two episodes, `action_noise`, privileged sim-state capture.
  - Compares a sha256 of every frame and every sim-state sidecar, plus the readable
    per-episode rows and timer stage counts, with `tests/unit/golden/rollout.json`.
  - Caught: executing a different noise sample (`frames[0] changed`), and capturing
    sim state after the step instead of before (`privileged[0] changed`).
  - The fakes it shares with `test_rollout_loop.py` live in `tests/fakes/rollout.py`.
    The fake env exposes a fake `ControlEnv` through the real `unwrapped._env` seam.
  - Regenerate only for an intended change: `BACKSTOP_UPDATE_GOLDEN=1 uv run pytest
    tests/unit/test_rollout_golden.py`, then review the diff.
- **Real-simulator comparison** (`just compare-recordings OUT [EARLIER]`, local, GPU,
  about 4 minutes). Records `configs/record/golden_smoke.yaml` into `OUT` and compares
  every parquet column, metadata file and sim-state array with an earlier recording.
  - Two recordings at the same commit were **identical**, videos included, on the
    RTX 4070 laptop. So on this machine the comparison is a real gate, not advice.
- **Analysis snapshot** (`just snapshot OUT [MANIFEST]`, local, about 20 seconds).
  Reruns `evaluate_guard`, `evaluate_stopwatch`, `freeze_guard`, `pretest_reselect` and
  `auto_label` (on a copy of the labels) into `OUT`, and writes a sha256 manifest.
  - The `OUT` path and the git commit stamped on frozen rules are normalised first.
  - Two snapshots in a row are identical. No script needed a new flag.
- **Import-layer test** (`tests/unit/test_import_layers.py`, CPU).
  - Enforces the package layering below, with today's three violations listed and
    tied to the stage that removes them. The list can only shrink.
  - Scripts may not import other scripts (one listed exception, Stage C).
  - Importing `guard`, `label` and `recover` must not load torch, transformers,
    lerobot, gymnasium, mujoco, robosuite or libero.

**A finding along the way:** re-running `auto_label` today adds 100 `object_clean`
rows that the committed `labels.jsonl` lacks. All 400 existing rows are unchanged. The
object suite was recorded after the labels were made, and nothing reads object-suite
onsets, so this is a gap in the data, not a bug.

**Size:** `src/` + `scripts/` +224 code lines (the two comparison tools), tests +354.
The estimate below had not counted the tools as scripts.

## Stage 5: simulator seam — done

**Plain version:** every reach into LIBERO's or lerobot's private parts goes through one
file, so an upgrade breaks one place, loudly.

- `env/libero.py` becomes the only module touching env internals. It gains small named
  functions:
  - `format_raw_obs`
  - `refresh_raw_obs`
  - `swap_task_files` (for the Distractor family)
  - `task_description(vec_env, fallback)` (today duplicated twice inside `_rollout_one`)
  - `max_episode_steps`
  - `restore_observation(vec_env, state)`, which returns a batched lerobot observation
    the same way `reset` does. The recovery runner needs exactly this.
- `perturb/wrapper.py` and `perturb/families.py` call those functions instead of
  private attributes. The robosuite modders are public API and stay as they are.
- **`PerturbationBase`** gives every family the defaults `needs_env_hook=True`,
  `prepare_env` no-op and `task_description` identity. That removes the three
  `getattr(..., True)` guesses.
- **`from_config` on each family** replaces the special-cased `_CATEGORICAL` table.
- **No `Env` protocol:** there is no second simulator yet.
- **Tests:**
  - A test listing which files may mention each private attribute.
  - A sim test checking that `restore_observation(sim_state)` right after a reset equals
    the reset observation pixel for pixel.
- **Verify:** golden digests, perturb tests, sim smoke.

## Stage 6: policy seam — done

**Plain version:** the policy interface describes how the loop *actually* calls the
policy, and everything lerobot-specific lives in one module.

- Move `SmolVLAAdapter`, its helpers, and `load_lerobot_policy`/`load_pinned_config`
  (now in `eval/pipeline.py`) into `policy/smolvla.py`.
- `Processors` (a frozen dataclass of the four lerobot processors) replaces
  `processors: dict[str, Any]`.
- `load_policy_runtime(cfg, env_cfg) -> (policy, Processors)` is the single entry point.
- `policy/base.py: ChunkPolicy` protocol with the real surface:
  - `reset`
  - `select_action(obs, *, noise)`
  - `noise_for`
  - `last_action_chunk`
  - `sample_chunks(obs, k, *, episode_seed, step)`

  It is **justified by** a numpy-only `FakeChunkPolicy` in `tests/fakes/policy.py`, which
  lets Stage 7 and the recovery runner be tested in the CPU job.
- Delete the unused `OpenVLAAdapter` stub and `make_adapter`. `UnsupportedPolicyError`
  stays, raised for `policy.type: openvla`.
- **Verify:** golden digests, `test_eval_pipeline`, `test_rollout_loop`.

## Stage 7: split `_rollout_one` (highest risk) — done

**Plain version:** the one long function becomes four small pieces with one job each.
The recovery runner can then reuse the pieces instead of copying the loop.

- `record/obs.py`: `ObsPipeline.prepare(raw_obs, task) -> PreparedObs(policy_obs, frame_obs)`,
  then `.normalize()`. These are kept separate because the sim state must be captured
  *between* the two.
- `record/step.py`: `decide(policy, obs, k, *, episode_seed, step, processors) -> Decision(action, action_policy, chunk, samples)`.
  It owns `inference_mode` and the unnormalize step.
- `record/sink.py`: `EpisodeSink(dataset, privileged)` with `begin`, `add_sim_state`,
  `add_frame` and `end`. `end` keeps the frames-equal-sim-states check.
- `record/loop.py`: `rollout_episode(ctx, spec) -> EpisodeOutcome`, under 50 lines, with
  `RolloutContext` and `EpisodeSpec` replacing the 18 keyword arguments.
- **Must preserve exactly:**
  - timer stage names and nesting
  - random draw order: `episode_rng` once per episode, `noise_for(index=0)` before
    `sample_chunks`
  - the success and done rules
  - the order of sim-state capture, frame write and step
- **Tests:**
  - The golden test, unchanged.
  - New CPU tests of `decide` and `EpisodeSink` using the Stage 6 fakes.
- **Verify:** golden digests, sim smoke, and `timing.json` with the same keys and counts.
- **Stop rule:** if a digest moves and the cause isn't found within an hour, revert and
  split the stage smaller.

## Stage 8: thin `run_pipeline` — done

- Split it into `build_runtime(cfg)`, `record_run(runtime, cfg)` and
  `finalize_report(cfg, stats)`.
- The write into the repo's `docs/failures.md` moves out of runtime code, into
  `scripts/` or a flag.
- **Verify:** `test_cli`, `test_eval_pipeline`, and `run.json`/eval report identical
  apart from timing.

## Stage 1: `backstop/corpus/` reader — done

**Plain version:** one package knows where recordings live and how to read them, so
`guard` and `label` stop borrowing from each other.

- `corpus/layout.py`: `ShardLayout(lerobot_root, privileged_root, eval_root, configs_root)`
  with these methods:
  - `discover_shards`
  - `run_stem`, the one place that knows `spatial_<shard>_k4`
  - `provenance(shard) -> RunProvenance | None`, typed rather than a raw dict
  - `dataset_root`
  - `perturbation_params`
- `corpus/read.py`:
  - `load_episode_meta`
  - `load_episode_video_range`
  - `load_dataset_frames`
  - `feature_shape`
  - `iter_episode_columns(dataset_root, columns)`, one episode at a time, with a
    contiguity check
- `corpus/privileged.py`: the npz *reader* moves here. The writer stays in `record/`.
- **Callers:** `guard/extract.py`, `guard/vision.py`, `label/extract.py` and three
  scripts (`auto_label`, `evaluate_presence`, `review_server`) switch over, with no re-export shims. `label/extract.py` keeps only row
  building and merging.
- **Byte risk:** `perturbation_strength` comes back as a float from the typed model.
  First confirm every `run.json` on disk already stores floats (`build_provenance`
  writes them), or `labels.jsonl` bytes would change.
- Also make `load_shard_signals` stream one episode at a time, which is what its
  docstring already claims.
- **Verify:** snapshot identical, and the `guard -> label` entries leave the import
  allowlist.

## Stage 2: typed episodes — done

- `corpus/episode.py`: `EpisodeKey(NamedTuple)`. It hashes and compares like
  `tuple[str, int]`, so it replaces `traces.Key`, bare tuples and `split.EpisodeKey`
  without breaking dict lookups. The frozen-dataclass key it replaces hashes the same
  tuple, so split order shouldn't move, and the snapshot checks that.
- `EpisodeMeta(key, task, n_steps, success, perturbation, perturbation_strength)`.
  `EpisodeSignal` is built from it.
- `guard/traces.py`:
  - `TraceRow` frozen dataclass with `from_json`/`to_json` (field order identical to
    today's writer), `key` and `series(name)`. `per_step` stays as lists.
  - `write_traces` moves here from `scripts/evaluate_guard.py`, so the format's reader
    and writer sit in one file, with a round-trip test.
- **Labels:** typed for reading only (`LabelView`). The write path stays raw, so
  reviewed rows round-trip byte for byte.
- **Migration:** one script per commit. `as_float`/`as_int` and every
  `type: ignore[arg-type]` go away, with a grep test that the count is 0.
- **Stop rule:** if a script needs more than about 30 lines of adapters, leave it on raw
  rows with a TODO.

## Stage 4: paths/guard config and shared constants — done

- `config.py` gains `CorpusPaths` (dataset roots, traces, labels) and
  `GuardEvalConfig(eval_fraction=0.3, seed=0)`.
  - The defaults live in these pydantic models, the one source of truth. No new YAML
    file is added.
  - The budget always comes from `configs/metric.yaml` via `load_metric_contract`,
    never a literal.
- `guard/cli.py: add_guard_args(parser)` sets the shared argparse defaults. Any
  script-specific override stays visible in that script.
- `schema.py` gains `CONTROL_FPS`, `GRIPPER_INDEX`, `EEF_POS` and `GRIPPER_QPOS`. These
  replace the copies in `guard/traces.py`, `guard/signals.py`, `label/signals.py`,
  `recover/*`, `review_server.py` and `verify_corpus.py`.
- `create_dataset` asserts that lerobot's fps equals `CONTROL_FPS`.
- **Verify:** snapshot identical, and a test that every script's `--help` defaults are
  unchanged.

---

## Target layout and import rules

```
L0  schema, config, provenance, parquet_io
L1  corpus                       -> L0
L2  label, guard                 -> L0, corpus       (neither imports the other)
    recover                      -> L0, guard.online
    env, policy                  -> L0               (the only homes of third-party internals)
    perturb                      -> L0, env
L3  record                       -> env, policy, perturb, corpus.privileged, guard.online, recover
L4  eval                         -> record, env, policy
scripts                          -> any package module; never another script
```

These rules are enforced by `test_import_layers.py` (Stage 0), whose allowlist is
now empty. The private-attribute test from Stage 5 backs them up.

## The recovery runner, afterwards

It lives in `record/recovery.py`, about 150 lines. Each step:

1. `decide(...)`, then `OnlineGuard.check_samples(decision.samples, step)`.
2. When the guard fires, drive `RepositionMotion.next_action(frame_obs["state"])`
   through the env until it returns None, recording through the same `EpisodeSink`.
   Then `policy.reset()` and `guard.rearm(step)`.
3. To branch, use `sim_state` and `restore_observation(state)`, then the same
   `ObsPipeline.prepare`.

It is tested in the CPU job with `FakeChunkPolicy` and a fake env. It needs
Stages 5–7, plus the paths config from Stage 4 for locating the frozen rule
(until then, a CLI flag).

## Not doing

- Changing any on-disk format, or regenerating artifacts in a refactor commit.
- Typing the labels write path, or turning `per_step` into arrays.
- Reordering signal tuples, or renaming timer stages.
- An `Env` ABC, plugin systems, a DI framework, async envs, or `import-linter` as a
  new dependency.
- Folding guard config into `PipelineConfig`, or touching the metric lock.
- Leaving re-export shims beyond the commit that moves the code.
