#!/usr/bin/env python
"""Does a recorded run actually contain what the recording contract promises?

Every check here corresponds to a bug that shipped silently once. Run this after
every GPU recording run, before trusting the corpus or moving to the next phase.

    uv run python scripts/verify_corpus.py \
        --artifacts artifacts/eval/spatial_smoke \
        --dataset data/lerobot/backstop_spatial_smoke

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.checks import Checks
from backstop.parquet_io import as_array
from backstop.schema import CONTROL_FPS, GRIPPER_QPOS

LIBERO_SPATIAL_MAX_STEPS = 280
GRIPPER_QPOS_ABS_MAX = 0.06
ACTION_ABS_MAX = 1.5

CHECKS = Checks()
check = CHECKS.check


def verify_artifacts(artifacts: Path) -> dict:
    """Returns the parsed `run.json` (or `{}`), so `main` can pass the recorded
    perturbation family into `verify_dataset` -- which invariant applies to
    `action` vs `action_policy` depends on it (ADR-003)."""
    eval_path, run_path = artifacts / "eval.json", artifacts / "run.json"
    check(eval_path.is_file(), "eval.json written", str(eval_path))
    check(run_path.is_file(), "run.json written", str(run_path))
    if not run_path.is_file():
        return {}
    run = json.loads(run_path.read_text())
    check(bool(run.get("git_sha")), "run.json git_sha is set", str(run.get("git_sha"))[:12])
    check(
        run.get("mujoco_version") == "3.3.2",
        "MuJoCo pinned at 3.3.2 (ADR-002)",
        str(run.get("mujoco_version")),
    )
    check(run.get("mujoco_gl") == "osmesa", "render backend is OSMesa", str(run.get("mujoco_gl")))
    if eval_path.is_file():
        payload = json.loads(eval_path.read_text())
        rate = payload.get("policy_success_rate")
        check(rate is not None, "policy_success_rate recorded", f"{rate}")
        lengths = [row.get("n_steps") for row in payload.get("episodes", [])]
        if any(x is not None for x in lengths):
            short = [x for x in lengths if x is not None and x < LIBERO_SPATIAL_MAX_STEPS]
            check(
                bool(short),
                "at least one episode ended before the step cap",
                f"{len(short)}/{len(lengths)} under {LIBERO_SPATIAL_MAX_STEPS}",
            )
    return run


def verify_dataset(dataset: Path, perturbation: str = "clean") -> None:
    import pyarrow.parquet as pq

    info_path = dataset / "meta" / "info.json"
    check(info_path.is_file(), "dataset info.json exists", str(info_path))
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text())
    check(
        info.get("fps") == CONTROL_FPS,
        f"dataset fps == {CONTROL_FPS} (control rate, not 30 or 80)",
        str(info.get("fps")),
    )
    n_eps = info.get("total_episodes", 0)
    check(n_eps > 0, "dataset has at least one finalized episode", f"total_episodes={n_eps}")

    files = sorted((dataset / "data").rglob("*.parquet"))
    check(bool(files), "parquet shards present", f"{len(files)} file(s)")
    if not files:
        return
    try:
        table = pq.read_table(files[0])
    except Exception as exc:  # noqa: BLE001 - a corrupt shard is the finding
        check(False, "parquet opens cleanly", f"{type(exc).__name__}: {exc}")
        return
    check(True, "parquet opens cleanly", f"{table.num_rows} rows")

    data = table.to_pydict()
    k = info["features"]["action_chunk_samples"]["shape"][0]
    chunk_hw = tuple(info["features"]["action_chunk"]["shape"])
    chunk = as_array(data["action_chunk"], chunk_hw)
    samples = as_array(data["action_chunk_samples"], (k, *chunk_hw))
    state = as_array(data["observation.state"], (8,))
    action = as_array(data["action"], (7,))

    check(
        float(np.abs(chunk).max()) > 0.0,
        "action_chunk is populated, not zeros",
        f"absmax={np.abs(chunk).max():.4g}",
    )
    check(
        bool(np.allclose(samples[:, 0], chunk, rtol=1e-4, atol=1e-5)),
        "action_chunk_samples[0] is the executed chunk",
    )
    if k > 1:
        spread = float(np.abs(samples[:, 1:] - samples[:, :1]).max())
        check(spread > 0.0, "K samples differ from the executed chunk", f"max spread={spread:.4g}")

    # ADR-003: `action_policy` (what the policy asked for) always equals the
    # stored chunk's first row -- that invariant does not depend on which family
    # recorded this shard. `action` (what was actually executed) equals
    # `action_policy` for every family except `action_noise`, which is the one
    # place they are allowed -- and required -- to differ. Checking both
    # unconditionally is what would have caught the original bug (chunks left in
    # policy space) and would also catch a shard silently recording the wrong one.
    if "action_policy" in data:
        action_policy = as_array(data["action_policy"], (7,))
        check(
            bool(np.allclose(action_policy, chunk[:, 0], rtol=1e-3, atol=1e-4)),
            "action_policy == action_chunk[0] (same units)",
        )
        if perturbation == "action_noise":
            check(
                not np.allclose(action, action_policy, rtol=1e-3, atol=1e-4),
                "action_noise: action differs from action_policy",
                "shard is labelled action_noise but nothing was perturbed",
            )
        else:
            check(
                bool(np.allclose(action, action_policy, rtol=1e-3, atol=1e-4)),
                f"action == action_policy (perturbation={perturbation!r}, not action_noise)",
            )
    else:
        check(
            bool(np.allclose(action, chunk[:, 0], rtol=1e-3, atol=1e-4)),
            "executed action == action_chunk[0] -- pre-ADR-003, no action_policy column",
        )
    grip = float(np.abs(state[:, GRIPPER_QPOS]).max())
    check(
        0.0 < grip <= GRIPPER_QPOS_ABS_MAX,
        "observation.state[6:8] looks like gripper qpos",
        f"absmax={grip:.4g}",
    )
    check(
        not np.allclose(state[:, :3], state[0, :3]),
        "observation.state eef position varies over the episode",
    )
    amax = float(np.abs(action).max())
    check(
        amax < ACTION_ABS_MAX,
        "executed actions are in LIBERO range (unnormalize applied)",
        f"absmax={amax:.4g}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)

    run = verify_artifacts(args.artifacts)
    verify_dataset(args.dataset, perturbation=run.get("perturbation", "clean"))

    return CHECKS.report(on_failure="Do NOT proceed to the next phase until these are green.")


if __name__ == "__main__":
    raise SystemExit(main())
