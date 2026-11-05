#!/usr/bin/env python
"""Guard-off against guard-on-with-recovery, paired on the same recorded episodes.

The guard-off arm is the recording itself. With the guard on, an episode where
the frozen rule never fires runs exactly as recorded, so only the episodes it
fires on are run: each one restored at its trigger step (`record.recovery`),
then lift and back off, then the policy again, under the attempt budget,
cooldown and stopwatch restart of `configs/recovery/recovery.yaml`.

`--live` runs those episodes from their reset instead, with the guard watching
from step 0. Its first trigger must land where the offline rule said; that
checks the branching shortcut.

    uv run python scripts/run_recovery.py --limit 20 --gifs 5   # the pilot
    uv run python scripts/run_recovery.py                       # every trigger
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from backstop.config import PipelineConfig
from backstop.corpus.layout import ShardLayout
from backstop.corpus.privileged import load_sim_state
from backstop.env.libero import close_libero
from backstop.eval.pipeline import build_runtime
from backstop.eval.report import write_recovery_report
from backstop.guard.online import load_rule
from backstop.guard.traces import load_jsonl
from backstop.record.loop import EpisodeSpec
from backstop.record.obs import ObsPipeline
from backstop.record.recovery import Branch, RecoveryOutcome, run_guarded_episode
from backstop.record.step import Decider
from backstop.record.timing import StageTimer
from backstop.recover.controller import RecoveryConfig, RecoveryController
from backstop.recover.paired import paired_result
from backstop.schema import RECOVERY_HEADLINE

FROZEN = Path("artifacts/guard/frozen")


def save_gif(frames: list[np.ndarray], path: Path, every: int = 2) -> None:
    from PIL import Image  # noqa: PLC0415

    images = [Image.fromarray(frame).resize((180, 180)) for frame in frames[::every]]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)


def _row(episode: dict[str, Any], trigger: int, outcome: RecoveryOutcome) -> dict[str, Any]:
    return {
        **episode,
        "offline_trigger_step": trigger,
        "guard_off_success": episode["success"],
        "guard_off_n_steps": episode["n_steps"],
        "success": outcome.success,
        "n_steps": outcome.n_steps,
        "halted_at": outcome.halted_at,
        "attempts": [asdict(attempt) for attempt in outcome.attempts],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/record/spatial_clean_k4.yaml"))
    parser.add_argument("--recovery", type=Path, default=Path("configs/recovery/recovery.yaml"))
    parser.add_argument("--rule", type=Path, default=FROZEN / "rule_05.json")
    parser.add_argument("--shard", default="clean")
    parser.add_argument("--limit", type=int, help="run only the first N triggered episodes")
    parser.add_argument("--live", action="store_true", help="run from reset, not from the trigger")
    parser.add_argument("--gifs", type=int, default=0, help="save GIFs of the first N episodes")
    parser.add_argument("--out", type=Path, default=Path("artifacts/recovery/run"))
    ShardLayout.add_args(parser)
    args = parser.parse_args(argv)

    cfg = PipelineConfig.from_yaml(args.config)
    if cfg.record.perturbation != "clean":
        # Lighting and camera changes are not in the sim state, so a restore drops them.
        raise SystemExit("branching restores sim state only; use a clean shard")
    cfg.record.enabled = False
    layout = ShardLayout.from_args(args)
    rule = load_rule(args.rule)
    recovery = RecoveryConfig.from_yaml(args.recovery)
    triggers_path = args.rule.with_name(args.rule.stem + "_triggers.json")
    triggers = {
        row["episode_index"]: row["trigger_step"] for row in json.loads(triggers_path.read_text())
    }
    eval_json = layout.run_stem(layout.eval_root, args.shard) / "eval.json"
    episodes = json.loads(eval_json.read_text())["episodes"]
    fired = [ep for ep in episodes if triggers.get(ep["episode_index"]) is not None]
    selected = fired[: args.limit] if args.limit is not None else fired

    runtime = build_runtime(cfg)
    obs = ObsPipeline(runtime.processors)
    decide = Decider(runtime.policy, runtime.processors, cfg.record.k_samples, StageTimer())
    args.out.mkdir(parents=True, exist_ok=True)
    # Each episode is saved as it finishes, and a re-run into the same --out
    # skips the ones already there: a full run is hours of GPU.
    saved = args.out / "episodes.jsonl"
    done = {row["episode_index"]: row for row in load_jsonl(saved)} if saved.is_file() else {}
    rows: list[dict[str, Any]] = []
    resets: dict[int, int] = {}
    try:
        for i, episode in enumerate(selected):
            index, trigger = episode["episode_index"], triggers[episode["episode_index"]]
            if index in done:
                rows.append(done[index])
                continue
            suite, task_id = episode["suite"], episode["task_id"]
            spec = EpisodeSpec(runtime.envs[suite][task_id], index, episode["seed"], suite, task_id)
            branch = None
            if args.live:
                # LIBERO hands out init states in reset order, so replay the resets
                # the recording made on this task before this episode.
                position = sum(
                    ep["task_id"] == task_id for ep in episodes[: episodes.index(episode)]
                )
                for _ in range(position - resets.get(task_id, 0)):
                    spec.vec_env.reset(seed=[0])
                resets[task_id] = position + 1
            else:
                states = load_sim_state(layout.privileged_root, args.shard, index)
                assert states is not None, f"{args.shard} episode {index} has no sim state"
                branch = Branch(trigger, states[trigger])
            outcome = run_guarded_episode(
                obs,
                decide,
                RecoveryController(rule, recovery),
                spec,
                branch=branch,
                keep_frames=i < args.gifs,
            )
            if outcome.frames:
                save_gif(outcome.frames, args.out / "gifs" / f"episode_{index:03d}.gif")
            rows.append(_row(episode, trigger, outcome))
            with saved.open("a") as handle:
                handle.write(json.dumps(rows[-1]) + "\n")
            first = outcome.attempts[0].start if outcome.attempts else outcome.halted_at
            print(
                f"episode {index}: off {episode['success']} -> on {outcome.success} "
                f"in {outcome.n_steps} steps, {len(outcome.attempts)} recoveries, "
                f"first trigger {first} (offline {trigger})",
                flush=True,
            )
    finally:
        close_libero(runtime.envs)

    # The episodes the guard never fires on are the recording, both ways, so the
    # shard-wide comparison needs every triggered episode run.
    complete = len(rows) == len(fired)
    quiet = [bool(ep["success"]) for ep in episodes if triggers.get(ep["episode_index"]) is None]
    base = quiet if complete else []
    paired = paired_result(
        base + [bool(r["guard_off_success"]) for r in rows],
        base + [bool(r["success"]) for r in rows],
    )
    values: dict[str, Any] = {
        RECOVERY_HEADLINE: paired.success_rate_change,
        "scope": "shard" if complete else f"the first {len(rows)} triggered episodes only",
        "n_paired": paired.n,
        "off_success": paired.off_success,
        "on_success": paired.on_success,
        "rescued": paired.rescued,
        "spoiled": paired.spoiled,
        "p_value": paired.p_value,
        "intervention_rate": len(fired) / len(episodes),
        "max_attempts": recovery.max_attempts,
        "cooldown_steps": recovery.cooldown_steps,
        "mean_extra_steps": float(np.mean([r["n_steps"] - r["guard_off_n_steps"] for r in rows])),
        "mean_recoveries": float(np.mean([len(r["attempts"]) for r in rows])),
        "mode": "live" if args.live else "branch",
        "rule": str(args.rule),
    }
    if args.live:
        first = [r["attempts"][0]["start"] if r["attempts"] else r["halted_at"] for r in rows]
        values["live_trigger_matches_offline"] = sum(
            f == r["offline_trigger_step"] for f, r in zip(first, rows, strict=True)
        )
    path = write_recovery_report(output_dir=args.out, values=values)
    print(json.dumps(values, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
