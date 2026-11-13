#!/usr/bin/env python
"""How often does a failed scene succeed if the policy simply tries again?

Each failed episode of the clean shard is re-run from its own recorded start
state, with no guard and no recovery, under fresh flow noise. The same noise
reproduces the recorded failure (branching is repeatable, docs/recovery.md),
so a different seed is a genuinely new try at the same scene. The success
rate over those re-runs is the ceiling for any recovery that sends the arm
home and lets the policy start over: if failed scenes never succeed on a
fresh try, no retry from the start pose will rescue them.

Runs are capped at `--max-steps` (good clean runs finish by about step 130),
saved as they finish, and a re-run into the same `--out` resumes.

    uv run python scripts/retry_headroom.py --tries 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backstop.config import PipelineConfig
from backstop.corpus.layout import ShardLayout
from backstop.corpus.privileged import load_sim_state
from backstop.env.libero import close_libero
from backstop.eval.pipeline import build_runtime
from backstop.guard.stopwatch import StopwatchRule
from backstop.guard.traces import load_jsonl
from backstop.record.loop import EpisodeSpec
from backstop.record.obs import ObsPipeline
from backstop.record.recovery import Branch, run_guarded_episode
from backstop.record.step import Decider
from backstop.record.timing import StageTimer
from backstop.recover.controller import RecoveryConfig, RecoveryController

NEVER = StopwatchRule(line=float("inf"), stop_step=10**9, k=0, n_calibration=0, bound=0.0)
NOISE_SEED_STRIDE = 1000  # try i draws its noise from seed + i * stride


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/record/spatial_clean_k4.yaml"))
    parser.add_argument("--shard", default="clean")
    parser.add_argument("--tries", type=int, default=2, help="fresh tries per failed episode")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--out", type=Path, default=Path("artifacts/recovery/retry_headroom"))
    ShardLayout.add_args(parser)
    args = parser.parse_args(argv)

    cfg = PipelineConfig.from_yaml(args.config)
    cfg.record.enabled = False
    cfg.record.k_samples = 1  # no guard here, so one sample per step
    layout = ShardLayout.from_args(args)
    eval_json = layout.run_stem(layout.eval_root, args.shard) / "eval.json"
    episodes = json.loads(eval_json.read_text())["episodes"]
    failures = [ep for ep in episodes if not ep["success"]]

    runtime = build_runtime(cfg)
    obs = ObsPipeline(runtime.processors)
    decide = Decider(runtime.policy, runtime.processors, 1, StageTimer())
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / "tries.jsonl"
    done = {(r["episode_index"], r["try"]) for r in load_jsonl(saved)} if saved.is_file() else set()
    try:
        for episode in failures:
            index, suite, task_id = episode["episode_index"], episode["suite"], episode["task_id"]
            states = load_sim_state(layout.privileged_root, args.shard, index)
            assert states is not None, f"{args.shard} episode {index} has no sim state"
            for attempt in range(1, args.tries + 1):
                if (index, attempt) in done:
                    continue
                spec = EpisodeSpec(
                    runtime.envs[suite][task_id], index, episode["seed"], suite, task_id
                )
                outcome = run_guarded_episode(
                    obs,
                    decide,
                    RecoveryController(NEVER, RecoveryConfig(max_attempts=0)),
                    spec,
                    branch=Branch(0, states[0], recover=False),
                    noise_seed=episode["seed"] + attempt * NOISE_SEED_STRIDE,
                    max_steps=args.max_steps,
                )
                row = {
                    "episode_index": index,
                    "task_id": task_id,
                    "try": attempt,
                    "success": outcome.success,
                    "n_steps": outcome.n_steps,
                }
                with saved.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    finally:
        close_libero(runtime.envs)

    rows = load_jsonl(saved)
    scenes = {r["episode_index"] for r in rows}
    rescued = {r["episode_index"] for r in rows if r["success"]}
    summary = {
        "failed_scenes": len(scenes),
        "tries": len(rows),
        "try_success_rate": sum(r["success"] for r in rows) / len(rows) if rows else 0.0,
        "scenes_rescued_by_any_try": len(rescued),
        "max_steps": args.max_steps,
        "by_task": {
            str(t): f"{sum(r['success'] for r in rows if r['task_id'] == t)}/"
            f"{sum(r['task_id'] == t for r in rows)}"
            for t in sorted({r["task_id"] for r in rows})
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
