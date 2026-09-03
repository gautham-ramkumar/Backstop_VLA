# ADR-002: SmolVLA on LIBERO-Spatial, not OpenVLA-7B

- Status: Accepted
- Date: 2026-09-01
- Week: 1

## Decision

Week 1 uses the frozen Hub checkpoint `HuggingFaceVLA/smolvla_libero` (pinned revision in YAML) on **LIBERO-Spatial**. OpenVLA-7B is an unimplemented adapter stub until this record/eval loop is proven.

## Why not OpenVLA-7B

This host is a Dell G16 with an **RTX 4070 Laptop GPU (8188 MiB)**.

| | SmolVLA | OpenVLA-7B |
|---|---|---|
| Params | ~0.45–0.6B | 7B |
| Inference VRAM | ~1–2 GB | ~14 GB fp16; ~6–7 GB INT4 before EGL |
| Stack | First-class LeRobot policy | Separate OpenVLA eval scripts, four suite-specific checkpoints, bitsandbytes |
| Repro target | Paper Spatial 90%; Hub band 80–100%; pinned independent repro **81.5%** | Paper Spatial 84.7%; maintainers warn drift off A100 |
| 8 GB + LIBERO | Fits if render is OSMesa and `batch_size=1` | INT4 + EGL + decode activations is not a safe margin |

The guard is policy-agnostic: it consumes observations and action chunks. Swapping OpenVLA later is a new adapter, not a rewrite.

## Locked recipe

| Knob | Value |
|---|---|
| GPU | RTX 4070 Laptop, 8 GB |
| Python | 3.12 (LeRobot 0.6.x requires `>=3.12`; 3.11 will not install it) |
| Envs | `batch_size=1`, `max_parallel_tasks=1` |
| Render | `MUJOCO_GL=osmesa` (EGL only if VRAM allows) |
| MuJoCo | **3.3.2** (3.4+ breaks Spatial task-5 init-state settling) |
| Control | `relative` |
| Policy | `HuggingFaceVLA/smolvla_libero` |
| `n_action_steps` | `1` (checkpoint `config.json`) |
| `num_steps` | **`1` (locked)** | Checkpoint default is 10; 81.5% Spatial repro uses 1 |
| Init | `init_states=true` |
| Seed | `0` |
| Suite | `libero_spatial`, 10 tasks × 10 episodes = 100 |

**In-tolerance:** policy success rate in **[0.70, 0.90]**. That covers the Hub 80–100% band and the 81.5% OSMesa/MuJoCo 3.3.2 repro. The paper 90% is not this checkpoint.

**Fallback that is now locked:** `policy.num_steps: 1`. Smoke (task 0 × 2 episodes, 2026-09-01) at `num_steps=10` was 0% and too slow for a 100-episode corpus. Do not revert to 10 without a new ADR.

Do not load OpenVLA this week, even to compare.

## K samples

SmolVLA is flow-matching. “Action distribution” means K independent chunks with different flow noise — not logits.

- Smoke (2 episodes): `k_samples=4`
- Full Spatial record: `k_samples=1` (executed chunk only)
- Week 2 perturbation corpus: `k_samples=4`

**K must not perturb the trajectory.** Flow noise is drawn from a dedicated
`torch.Generator` seeded on `(episode_seed, step, sample_index)`, not from the
global RNG (`SmolVLAAdapter.noise_for`). Sample 0 is the executed chunk, so a
K=1 run and a K=4 run at the same seed produce a bit-identical trajectory.

Without this the week-2 K=4 corpus and the week-1 K=1 baseline diverge at the
same seed, and any success-rate gap is unattributable — you cannot tell a
perturbation effect from a sampling artifact. Pinned by
`test_executed_trajectory_is_independent_of_k`.

## Recording contract

`n_action_steps` is provenance-only. The loop re-predicts a full chunk every
control step and executes its first action, which is what `n_action_steps=1`
means operationally.

Frames are captured **after** the env preprocessor and **before** the policy
preprocessor. `LiberoProcessorStep` assembles `observation.state` as
eef_pos(3) + eef_axisangle(3) + gripper_qpos(2) and rotates images 180° for the
checkpoint's camera convention; the policy preprocessor then normalizes. The
corpus stores what the policy saw, in real units. Reconstructing either field
from the raw env dict is how week 1 first shipped rotation-matrix entries as
"state" and upside-down video.

Dataset `fps` is the env control rate (**20 Hz**), one row per control step —
not 30, and not `metadata["render_fps"]` (80).
