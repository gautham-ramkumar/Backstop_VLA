set dotenv-load := false
set shell := ["bash", "-cu"]

export MUJOCO_GL := "osmesa"
export PYOPENGL_PLATFORM := "osmesa"
export CMAKE_POLICY_VERSION_MINIMUM := "3.5"

sync:
	uv sync --locked --extra dev --extra sim

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run pyright

fmt:
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

# The whole suite with no skips allowed, as the CI `sim` job runs it.
test-strict:
	BACKSTOP_ALLOWED_SKIPS= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -rs

smoke:
	uv run backstop-eval --config configs/eval/spatial_smoke.yaml

eval-spatial:
	uv run backstop-eval --config configs/eval/spatial_baseline.yaml

record-spatial:
	uv run backstop-record --config configs/record/spatial_baseline.yaml

# Copy a complete, unsharded clean run's natural failures to docs/failures.md, the list review reads.
publish-failures run="artifacts/eval/spatial_baseline":
	cp {{run}}/failures.md docs/failures.md

benchmark:
	uv run python scripts/benchmark_throughput.py --k 1,4 --episodes 3

verify-sim-roundtrip:
	uv run python scripts/verify_sim_state_roundtrip.py

verify-smoke:
	uv run python scripts/verify_corpus.py \
	  --artifacts artifacts/eval/spatial_smoke \
	  --dataset data/lerobot/backstop_spatial_smoke

verify-baseline:
	uv run python scripts/verify_corpus.py \
	  --artifacts artifacts/eval/spatial_baseline \
	  --dataset data/lerobot/backstop_spatial_baseline

# The guard's reports, in order: traces and the single-split report, then the
# headline against the stopwatch. evaluate_presence.py (GPU detector) is separate.
guard-reports:
	uv run python scripts/evaluate_guard.py
	uv run python scripts/evaluate_stopwatch.py

# Keep baselines under artifacts/snapshots (git-ignored, not cleared like /tmp):
# `just snapshot artifacts/snapshots/after artifacts/snapshots/before/manifest.json`.
# Fingerprint every analysis output into OUT (~20 s), optionally comparing with a manifest.
snapshot out compare="":
	uv run python scripts/snapshot_outputs.py --out {{out}} {{ if compare != "" { "--compare " + compare } else { "" } }}

# Example: `just compare-recordings artifacts/snapshots/rec_after artifacts/snapshots/rec_before`.
# Record a short fixed real-simulator run (GPU), optionally comparing with an earlier one.
compare-recordings out compare="":
	uv run python scripts/compare_recordings.py --out {{out}} {{ if compare != "" { "--compare " + compare } else { "" } }}

# Recovery: freeze the headline guard (5% and 20%), held out from the clean shard.
freeze-guard:
	uv run python scripts/freeze_guard.py

# Recovery: does re-selecting a different sampled chunk have any headroom?
pretest-reselect:
	uv run python scripts/pretest_reselect.py
