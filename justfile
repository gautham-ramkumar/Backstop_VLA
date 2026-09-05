set dotenv-load := false
set shell := ["bash", "-cu"]

export MUJOCO_GL := "osmesa"
export PYOPENGL_PLATFORM := "osmesa"
export CMAKE_POLICY_VERSION_MINIMUM := "3.5"

sync:
	uv sync --extra dev --extra sim

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run pyright

fmt:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

smoke:
	uv run backstop-eval --config configs/eval/spatial_smoke.yaml

eval-spatial:
	uv run backstop-eval --config configs/eval/spatial_baseline.yaml

record-spatial:
	uv run backstop-record --config configs/record/spatial_baseline.yaml

benchmark:
	uv run python scripts/benchmark_throughput.py --k 1,4 --episodes 3

verify-smoke:
	uv run python scripts/verify_corpus.py \
	  --artifacts artifacts/eval/spatial_smoke \
	  --dataset data/lerobot/backstop_spatial_smoke

verify-baseline:
	uv run python scripts/verify_corpus.py \
	  --artifacts artifacts/eval/spatial_baseline \
	  --dataset data/lerobot/backstop_spatial_baseline
