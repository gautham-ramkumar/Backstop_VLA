# Refactor plan: seams, typed corpus, smaller code

**Status:** proposed, not started. The Week 3–6 work is committed, and the experiments
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
| **Target after this plan** | **~5,000 (−33% overall)** | ~2,900 |

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
| 0 | Golden digests, snapshot, import-layer test | S | 250 | ~0 (tests +250) | low |
| C | Delete obsolete scripts and the last duplication | S | 350 | **about −250** | low |
| 5 | Simulator seam + `PerturbationBase` | M | 250 | −20 | medium |
| 6 | Policy seam: `ChunkPolicy`, `Processors`, `policy/smolvla.py` | M | 250 | −40 | medium |
| 7 | Split `_rollout_one` into obs / decide / sink | L | 350 | **+40** | **high** |
| 8 | Thin `run_pipeline` | S | 120 | −10 | low |
| 1 | `backstop/corpus/` reader | M | 250 | −50 | low–medium |
| 2 | Typed `EpisodeKey` + `TraceRow` | S–M | 200 | −30 | medium |
| 4 | Paths/guard config + shared constants | S | 150 | −40 | low |
| | **Total** | | ~2,200 | **about −400** | |

Stage 7 is the one stage expected to grow. Replacing 18 keyword arguments with two small
dataclasses, and one long function with four named pieces, costs a few lines. That's the
price of making the recovery runner a reuse instead of a copy.

## Stage C: delete the rest

Each item was checked for callers on 2026-10-18. Git history keeps everything.

| What | Why it can go | Code lines |
|---|---|---|
| `scripts/calibration_sweep.py`, `scripts/record_corpus_a_continuous_families.sh` | One-off Week-2 tools, referenced from nowhere (README, docs, justfile, tests, CI) | ~170 |
| `guard_viewer.py` ↔ `review_server.py` | A shared HTTP handler base replaces the borrowed private methods and the duplicated JSON helpers | ~50 |
| `verify_corpus.py` ↔ `verify_sim_state_roundtrip.py` | The same PASS/FAIL check harness, twice | ~20 |
| History-diary prose | "Week N / day 1" references in code. The history is already in ADRs and git; keep the *why*, cut the *when*. Prose isn't counted in the net figure, but files get shorter | (prose) |
| Small leftovers | The unused `CONTROL_FPS` in `review_server.py`, orphan comments | ~10 |

**Verify:** the analysis snapshot is byte-identical. Deleting the two obsolete scripts
needs your explicit OK in the stage's review.

---

## Stage 0: safety net (no `src/` changes)

Before refactoring anything, make "unchanged" something we can check.

- **Golden rollout test** (`tests/unit/test_rollout_golden.py`, sim job):
  - Reuses the CPU fakes in `test_rollout_loop.py` with K=4, two episodes,
    `action_noise` and `privileged: true`.
  - Gives `FakeVecEnv` a fake `unwrapped._env` with `get_sim_state()` and
    `regenerate_obs_from_state()`, so the privileged path runs through the real seam.
  - Hashes every frame (sorted keys, bytes plus dtype and shape), every npz array and
    the per-episode stats rows. The digests are committed to
    `tests/unit/golden/rollout.json`.
- **Real-sim smoke** (advisory, local):
  - `configs/record/golden_smoke.yaml` records task 0, 2 episodes, K=4, privileged.
  - `scripts/compare_recordings.py` checks parquet columns, `meta/episodes` and npz
    arrays for exact equality.
  - Record twice at HEAD first. If the two baselines differ (GPU nondeterminism), the
    fake golden test is the gate and this smoke run only advises.
- **Analysis snapshot** (`just snapshot OUT=...`, local, needs `data/`):
  - Runs `evaluate_guard` (into `OUT`), every `evaluate_*`, `freeze_guard`, and
    `auto_label` on a *copy* of `labels.jsonl`.
  - Writes a sha256 manifest of every output and of stdout.
  - Where a script has no output flag, add one with the same default. This is the only
    script edit in Stage 0.
- **Import-layer test** (`tests/unit/test_import_layers.py`, CPU, AST-based):
  - Asserts the allowed package edges (see *Target layout*).
  - Starts with an allowlist of today's violations, which later stages must shrink.
  - Also asserts that importing `backstop.{guard,label,recover}` pulls in no `torch`,
    `lerobot` or `gymnasium`.

**Done when:** CI is green and the digests and snapshot reproduce on a second run.

## Stage 5: simulator seam

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

## Stage 6: policy seam

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

  It is **justified by** a numpy-only `FakeChunkPolicy` in `tests/unit/fakes.py`, which
  lets Stage 7 and the recovery runner be tested in the CPU job.
- Delete the unused `OpenVLAAdapter` stub and `make_adapter`. `UnsupportedPolicyError`
  stays, raised for `policy.type: openvla`.
- **Verify:** golden digests, `test_eval_pipeline`, `test_rollout_loop`.

## Stage 7: split `_rollout_one` (highest risk)

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

## Stage 8: thin `run_pipeline`

- Split it into `build_runtime(cfg)`, `record_run(runtime, cfg)` and
  `finalize_report(cfg, stats)`.
- The write into the repo's `docs/failures.md` moves out of runtime code, into
  `scripts/` or a flag.
- **Verify:** `test_cli`, `test_eval_pipeline`, and `run.json`/eval report identical
  apart from timing.

## Stage 1: `backstop/corpus/` reader

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

## Stage 2: typed episodes

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

## Stage 4: paths/guard config and shared constants

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

These rules are enforced by `test_import_layers.py` (Stage 0), whose allowlist must be
empty by the end of Stage 5. The private-attribute test from Stage 5 backs them up.

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
