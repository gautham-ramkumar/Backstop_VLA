# Backstop runbook (week 1)

## Machine

Dell G16, RTX 4070 Laptop 8 GB. OSMesa rendering. `batch_size=1`.

## One-time setup

```bash
uv python install 3.12
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra sim
uv run pre-commit install
export MUJOCO_GL=osmesa
```

System package already needed: `libosmesa6`. CUDA driver must be live (`nvidia-smi`).

Python is **3.12**, not 3.11: LeRobot 0.6.x requires `>=3.12`. MuJoCo is pinned at **3.3.2**. Frozen recipe uses `num_steps=1` (ADR-002).

## Commands

| Intent | Command |
|---|---|
| Install | `just sync` |
| Lint | `just lint` |
| Unit + contract tests (CPU) | `just test` |
| Smoke (task 0, 2 episodes, K=4) | `just smoke` |
| Spatial baseline (10×10, no dataset) | `just eval-spatial` |
| Record Spatial corpus (K=1) | `just record-spatial` |

Equivalent without `just`:

```bash
uv run backstop-eval --config configs/eval/spatial_smoke.yaml
uv run backstop-eval --config configs/eval/spatial_baseline.yaml
uv run backstop-record --config configs/record/spatial_baseline.yaml
```

## Outputs

- `artifacts/eval/<run>/eval.json` — policy success rate + provenance
- `artifacts/eval/<run>/run.json` — pins (git, MuJoCo, policy revision, seed)
- `artifacts/eval/<run>/failures.md` — natural failures to watch
- `data/lerobot/backstop_spatial_smoke/` or `.../backstop_spatial_baseline/` — LeRobot v3 dataset
- W&B: project `backstop`, default mode **offline** (no API key required)

## Tolerance

Spatial success rate must land in **70–90%**. Below 70%, set `policy.num_steps: 1` in the YAML (ADR-002 fallback) and re-run before recording.

## What week 1 does not do

Guard signals, trust head, perturbation injection, LIBERO-Object, OpenVLA, viewer, policy training.
