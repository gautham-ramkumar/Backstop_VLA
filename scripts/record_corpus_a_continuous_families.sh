#!/usr/bin/env bash
# Corpus A: the four continuous-strength perturbation shards, run sequentially
# (not concurrently -- one GPU, and each shard already saturates it). Stops on
# the first failure rather than silently continuing past a broken shard;
# resume any individual shard afterward via its `on_existing: resume` variant
# once the cause is fixed.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

for shard in lighting camera object_pose action_noise; do
    echo "=== corpus A: $shard shard starting $(date -Iseconds) ==="
    uv run backstop-record --config "configs/record/spatial_${shard}_k4.yaml"
    echo "=== corpus A: $shard shard done $(date -Iseconds) ==="
    uv run python scripts/verify_corpus.py \
        --artifacts "artifacts/eval/spatial_${shard}_k4" \
        --dataset "data/lerobot/backstop_spatial_${shard}_k4"
done

echo "=== corpus A: all four families done $(date -Iseconds) ==="
