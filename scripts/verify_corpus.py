#!/usr/bin/env python
"""Phase-2 gate: does a recorded run actually contain what week 1 promised?

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
import sys
from pathlib import Path

import numpy as np

CONTROL_FPS = 20
LIBERO_SPATIAL_MAX_STEPS = 280
GRIPPER_QPOS_ABS_MAX = 0.06
ACTION_ABS_MAX = 1.5

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((ok, name, detail))


def _as_array(column: list, trailing: tuple[int, ...]) -> np.ndarray:
    """Parquet list columns come back flat or nested; normalise to (N, *trailing)."""
    flat = np.array([np.asarray(row, dtype=np.float32).reshape(-1) for row in column])
    return flat.reshape((flat.shape[0], *trailing))


def verify_artifacts(artifacts: Path) -> None:
    eval_path, run_path = artifacts / "eval.json", artifacts / "run.json"
    check(eval_path.is_file(), "eval.json written", str(eval_path))
    check(run_path.is_file(), "run.json written", str(run_path))
    if not run_path.is_file():
        return
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


def verify_dataset(dataset: Path) -> None:
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
    chunk = _as_array(data["action_chunk"], chunk_hw)
    samples = _as_array(data["action_chunk_samples"], (k, *chunk_hw))
    state = _as_array(data["observation.state"], (8,))
    action = _as_array(data["action"], (7,))

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
    check(
        bool(np.allclose(action, chunk[:, 0], rtol=1e-3, atol=1e-4)),
        "executed action == action_chunk[0] (same units)",
    )
    grip = float(np.abs(state[:, 6:8]).max())
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

    verify_artifacts(args.artifacts)
    verify_dataset(args.dataset)

    width = max(len(name) for _, name, _ in _results)
    failed = 0
    for ok, name, detail in _results:
        if not ok:
            failed += 1
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    if failed:
        print("\nDo NOT proceed to the next phase until these are green.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
