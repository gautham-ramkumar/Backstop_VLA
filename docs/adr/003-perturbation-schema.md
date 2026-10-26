# ADR-003: `action_policy`/`action` split, privileged sim state as a sidecar, K=4 as the corpus standard

- Status: Accepted
- Date: 2026-09-09
- Week: 2

## Decision

Four changes to the frozen week-1 schema, landed together because they are all
"the corpus must carry more than it did in week 1, on purpose, and the extra
fields have a stated contract."

1. **`action_policy` alongside `action`.** `action_policy` is the postprocessed
   action the policy asked for; `action` is what was actually sent to the env.
   They are equal for every family except `action_noise`, where they are
   required to differ. `action_policy == action_chunk[0]` holds unconditionally,
   regardless of family — it is what makes `action_noise` distinguishable from a
   wiring bug that silently records the wrong array.

2. **Privileged sim state lives outside the dataset.** `privileged/{shard}/{episode_index:06d}.npz`,
   not a LeRobot column: object count varies per task (a fixed column would be
   padding), and — the reason that actually matters — week-3 signal code loads
   the LeRobot dataset and physically cannot reach state that was never written
   into it. **Permitted** for ground-truth labels, onset annotation, and
   counterfactual branching. **Forbidden as guard input.** See
   [`docs/recovery.md`](../recovery.md) for what "counterfactual branching"
   actually buys, including the one caveat found while verifying the round trip
   (`scripts/verify_sim_state_roundtrip.py`): branching mid-chatter shows a
   transient divergence for a few steps, traced to `qacc_warmstart` not being
   part of robosuite's flattened state. Branching from smooth motion — what the
   corpus actually contains — round-trips to float noise.

3. **Labels are a sidecar, not a column.** `artifacts/corpus/labels.jsonl`, keyed
   by `(shard, episode_index)`: perturbation family, strength, params, outcome,
   failure category, onset range, review status. Manual review rewrites labels
   repeatedly; rewriting a parquet dataset every time a category changes is not
   a workflow. **Not yet built** — this is the review-tool/taxonomy work later
   in the week, recorded here so the format is decided once rather than
   invented under deadline.

4. **K=4 is the corpus standard.** Recorded in `run.json` provenance
   (`k_samples`), asserted by `verify_corpus.py`.

## Why not weaken the existing invariant instead

The pre-ADR-003 invariant was `action == action_chunk[0]`, unconditional. The
tempting shortcut once `action_noise` exists is to just delete that check for
noised shards. That would silently retire the exact assertion that caught
week 1's "chunks left in policy space" bug — the whole reason invariants like
this exist is that they fire on a mistake nobody thought to look for.

The actual fix keeps two unconditional checks instead of one conditional one:
`action_policy == action_chunk[0]` always, and `action == action_policy` iff
the family is not `action_noise`. Both directions of the second check matter —
`verify_corpus.py` now also fails if a shard is *labelled* `action_noise` but
`action` and `action_policy` come back identical, which is the case that would
mean the perturbation was configured but never reached the loop.

## Where the family is selected, and what still isn't wired

`record.perturbation` (name), `record.perturbation_strength`, and
`record.perturbation_kwargs` (for the categorical families — `absent_target`'s
`replacement`, `language_paraphrase`'s `paraphrase`) select a
`backstop.perturb` family for a run. `run_episodes` builds it once via
`make_perturbation` and `rollout_episode` attaches it to the vec_env
(`backstop.perturb.attach`) before every reset, unless the family declares
`needs_env_hook = False` — `action_noise` is the only family so far that
applies purely in the loop (`perturb_action`, between the policy's postprocessed
action and `vec_env.step`) and has no reason to force `PerturbedEnv` onto every
sub-env.

`record.shard` (the directory label for the dataset root and the privileged
sidecar) and `record.perturbation` (the actual family driving the run) are
**independent config fields.** Nothing enforces that a shard named `lighting`
was actually recorded with `perturbation: lighting` — that consistency is the
recording operator's responsibility when writing each shard's config, the same
way `configs/record/spatial_clean_k4.yaml`'s header comment asks a human to
diff it against `spatial_baseline.yaml` before running. Worth a
`verify_corpus.py` check later if a mismatched shard ever actually happens;
not worth a guard today against a mistake nobody has made yet.

Not wired: `object_pose` has no reject-and-resample validity check for
interpenetrating/off-table/unsettled objects (still on-plan, not done). The
`distractor` family (BDDL/MuJoCo model surgery) remains unbuilt and timeboxed,
per [`Week2_plan.md`](../../Week2_plan.md).

## Consequences

- `scripts/verify_corpus.py` reads `run.json`'s `perturbation` field and checks
  the family-appropriate invariant; datasets recorded before this ADR (no
  `action_policy` column) fall back to the pre-ADR-003 check so old artifacts
  still verify.
- `tests/unit/test_rollout_loop.py` proves the loop actually calls
  `perturb_action` end-to-end (not just in `test_perturb.py`'s per-family unit
  tests) and that the per-episode noise generator advances step to step rather
  than being redrawn.
- The clean K=4 re-record (`configs/record/spatial_clean_k4.yaml`) is
  unaffected: `record.perturbation` defaults to `"clean"`, so `action_policy`
  and `action` are identical by construction and the 70/100 reproducibility
  gate is exactly the week-1 comparison, unchanged.
