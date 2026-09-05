# Throughput: what a control step costs, and what that buys

Week 2's day-1 gate. The corpus must be recorded at K=4 — at K=1 `sample_chunks`
short-circuits and stores four copies of the executed chunk, so chunk disagreement,
the first guard signal, is identically zero — and K=4 is not free. This is the
measurement that sizes the corpus, replacing the plan's estimate.

Reproduce with:

```bash
just benchmark
```

Measured 2026-09-05 on an RTX 4070 Laptop (8 GB), MuJoCo 3.3.2, OSMesa, SmolVLA at
`n_action_steps=1`, `num_steps=10`, two 360×360 cameras. `libero_spatial` task 0,
three episodes per K (660 steps), after a discarded warm-up episode, with the CUDA
queue drained at every stage boundary. Raw JSON in `artifacts/benchmark/` (gitignored).

## Per-stage breakdown

| stage | K=1 ms/step | K=4 ms/step | K=1 % | K=4 % |
|---|---|---|---|---|
| `obs_prep` | 5.7 | 4.8 | 0.7% | 0.3% |
| `privileged` | 0.08 | — | 0.0% | — |
| `normalize` | 1.5 | 1.5 | 0.2% | 0.1% |
| **`policy_select`** | **433.2** | **337.1** | **55.9%** | **21.1%** |
| **`policy_samples`** | 0.1 | **985.2** | 0.0% | **61.8%** |
| `action_post` | 0.3 | 0.3 | 0.0% | 0.0% |
| `env_step` | 251.4 | 197.5 | 32.5% | 12.4% |
| ├ `env_render` | 221.7 | 174.3 | 28.6% | 10.9% |
| └ `env_physics` (derived) | 29.7 | 23.2 | 3.8% | 1.5% |
| `frame_write` | 28.8 | 22.0 | 3.7% | 1.4% |
| `env_reset` | 6792 /episode | 5952 /episode | 4.0% | 1.7% |
| `save_episode` | 4945 /episode | 4130 /episode | 2.9% | 1.2% |
| unaccounted | — | — | 0.0% | 0.0% |
| **total** | **0.775 s/step** | **1.594 s/step** | | |

`unaccounted` is 0.1 s out of 511 s and 1052 s respectively, so the breakdown
explains the wall clock rather than leaving a gap to interpret.

## What it says

**The step is policy-bound, not render-bound.** At K=1 the single flow-matching
forward pass is 56% of the step and OSMesa rendering is 29%. This was the question
worth settling before optimising: a render-bound step would have made K=4 nearly
free and pointed the day at image size or camera count instead. It is not, so
sampling is the lever and shrinking the corpus is the fallback.

**K=4 unbatched costs 2.06×, and the extra samples are full price.** 985.2 ms for
three extra samples is 328.4 ms each, against 337.1 ms for the executed one — the
loop in `SmolVLAAdapter.sample_chunks` runs them one at a time, so they cost exactly
what they look like. This is the case for batching the K−1 extras into a single
forward pass.

**Two stages nobody budgeted.** `env_reset` is 6.8 s per episode because
`hard_reset=True` rebuilds the MuJoCo model and the renderer on every reset, and
`save_episode` is 4.9 s of video encoding. Together they are ~12 s per episode —
about 1.3 h across a 400-episode corpus, independent of K.

**Privileged sim-state capture is free.** 0.08 ms per step, 0.0% of the wall clock,
measured in a separate single-episode run (`--k 1 --episodes 1`, the K=1/K=4 runs
above predate the benchmark enabling it). That run is also the first end-to-end
exercise of `sim_state()` against the real simulator rather than a test fake: 100
steps captured, and the loop's per-episode assertion that the sidecar has exactly
one state per recorded frame passed. There is no throughput argument for leaving it
off any shard.

**Episode length is not a constant, and the plan's number was too high.** Week 2's
plan projected ~210 steps/episode; `docs/week1_reference.csv` says 158.8. Successes
averaged 106.9 steps and all 30 failures ran to the 280-step cap, so length is a
function of the outcome mix: `steps(p) = p·106.9 + (1−p)·280`. A perturbed shard
tuned to ~50% success is 193.4 steps/episode, a third longer than the 70%-success
clean arm. Projections below use this rather than one average.

## Corpus projection

The plan's shard table: 100 clean episodes at the observed 70%, 300 perturbed at
the ~50% target ⇒ 73,918 steps.

| | s/step | 400-episode corpus |
|---|---|---|
| K=1 (week-1 configuration) | 0.775 | 15.9 h |
| K=4 unbatched (measured) | 1.594 | **32.7 h** |
| K=4 batched (target, not yet measured) | — | — |

The plan guessed ~35 h for K=4 unbatched against a ~20 h budget. The measurement
says 32.7 h. **The budget does not survive unbatched sampling**, which is the day-1
gate firing exactly as it was written to.

## Caveats

- **Stage costs are not independent of run load.** Every CPU-side stage got *faster*
  in the K=4 run than in the K=1 run — `env_render` 221.7 → 174.3 ms, `env_reset`
  6792 → 5952 ms, even `policy_select` 433.2 → 337.1 ms — despite doing identical
  work. The K=4 run spends most of its time blocked on the GPU, which plausibly
  leaves more thermal and frequency headroom for the CPU-bound stages on a laptop.
  Treat the **2.06× ratio and the totals** as the reliable numbers and individual
  cross-K stage deltas as suggestive only.
- **One task, three episodes.** Task 0 only, and its episodes ran 100/280/280 steps
  — matching `docs/week1_reference.csv` episodes 0–2 exactly, which is a useful
  incidental check that the week-2 loop still reproduces week 1. A full-suite shard
  mixes easier and harder tasks, so these are a sizing input, not a schedule.
- **CUDA sync inflates the total slightly.** Draining the queue at each stage
  boundary serialises any genuine CPU/GPU overlap. It is on because the per-stage
  split is the point; the overlap in this loop is small, since the adapter calls
  `.cpu()` on every captured chunk anyway.

## Decision

Batching `sample_chunks` is required, not optional. Even if it lands, the fallback
levers named in the plan (K=3 → trim the paraphrase control → trim one family, never
below K=3 and never the clean arm) should be expected to come into play; the
corpus size is not re-fixed until the batched number exists.

The clean K=4 re-record is unaffected either way — it is the reproducibility gate and
runs regardless.
