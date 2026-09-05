#!/usr/bin/env python
"""Day-1 week-2 gate: what does a control step actually cost, and where?

The week-2 corpus must be recorded at K=4 -- at K=1 `sample_chunks` short-circuits
and stores four copies of the executed chunk, so chunk disagreement, the first
guard signal, is identically zero. K=4 is not free, and the plan's ~20 h recording
budget is only defensible against a measured number.

So this measures, per stage, on the real record loop rather than a copy of it:

    uv run python scripts/benchmark_throughput.py --k 1,4 --episodes 3

and prints the breakdown plus a corpus projection. Two things it is built to
stop you doing:

- **Optimising the wrong stage.** If OSMesa rendering of two 360x360 cameras
  dominates the step, then the extra chunk samples are cheap and shrinking the
  corpus would be the wrong response. `env_render` is measured separately from
  the physics it is interleaved with, so that question has an answer.
- **Projecting from the wrong episode length.** Seconds per *episode* is not a
  constant: week 1's successes averaged 107 steps and every failure ran the full
  280-step cap, so a perturbed shard tuned to ~50% success is a third longer per
  episode than the 70%-success clean arm. Projections here scale s/step by an
  episode length derived from `docs/week1_reference.csv` at the target success
  rate, not by one average carried over from a run with a different outcome mix.

The first episode of a process pays CUDA and LIBERO asset warm-up, which would
land almost entirely on `policy_select`. A warm-up run absorbs it and is thrown
away; `--no-warmup` skips it and the report says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reserved subtree. The benchmark records real frames -- `frame_write` is one of
# the stages -- so it needs somewhere to put them that is not a corpus.
BENCH_ROOT = Path("data/lerobot/_benchmark")
REFERENCE_CSV = Path("docs/week1_reference.csv")


def main(argv: list[str] | None = None) -> int:
    # `run_pipeline` logs per-episode progress at INFO but configures no handler of
    # its own; without this a 40-minute benchmark prints nothing until it finishes.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    out_dir = Path(args.out) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    lengths = episode_length_model(REFERENCE_CSV)
    ks = [int(value) for value in args.k.split(",")]

    runs: dict[str, Any] = {}
    for k in ks:
        print(f"\n=== K={k} " + "=" * 56)
        runs[str(k)] = _measure(k, args=args, out_dir=out_dir)
        _print_stage_table(runs[str(k)]["timing"])

    report = {
        "created": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "episodes_per_k": args.episodes,
        "warmup": bool(args.warmup),
        "episode_length_model": lengths,
        "runs": runs,
        "projection": _project(runs, lengths, args),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    _print_k_comparison(runs)
    _print_projection(report["projection"], lengths, args)
    print(f"\nwrote {path}")
    if not args.keep_datasets:
        _clean(BENCH_ROOT)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/eval/spatial_smoke.yaml"))
    parser.add_argument("--k", default="1,4", help="comma-separated k_samples values to measure")
    parser.add_argument("--episodes", type=int, default=3, help="measured episodes per K")
    parser.add_argument("--task-ids", default=None, help="comma-separated; default: config's")
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument(
        "--corpus-episodes", type=int, default=400, help="episodes to project a corpus for"
    )
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=0.5,
        help="perturbed shards are tuned to this; it sets projected episode length",
    )
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument(
        "--no-sync",
        dest="sync",
        action="store_false",
        help="do not drain the CUDA queue at stage boundaries (faster, blurrier)",
    )
    parser.add_argument("--keep-datasets", action="store_true", help=f"keep {BENCH_ROOT}")
    parser.set_defaults(warmup=True, sync=True)
    return parser.parse_args(argv)


def episode_length_model(reference: Path) -> dict[str, Any]:
    """Mean episode length as a function of success rate, from week-1 outcomes.

    Successes and failures have very different lengths -- a success stops when the
    predicate fires, a failure runs to the cap -- so one overall mean only applies
    to a run with week 1's exact 70% success rate. Splitting them gives a length
    for any target rate: `steps(p) = p * mean_success + (1 - p) * mean_failure`.
    """
    if not reference.is_file():
        raise FileNotFoundError(
            f"{reference} is missing. It is the frozen week-1 per-episode outcome table "
            "and the only measured basis for projecting episode length."
        )
    rows = list(csv.DictReader(reference.open()))
    successes = [int(r["n_steps"]) for r in rows if r["success"].strip().lower() == "true"]
    failures = [int(r["n_steps"]) for r in rows if r["success"].strip().lower() != "true"]
    if not successes or not failures:
        raise ValueError(f"{reference} has no {'failures' if successes else 'successes'} to model")
    return {
        "source": str(reference),
        "n_episodes": len(rows),
        "observed_success_rate": round(len(successes) / len(rows), 4),
        "mean_steps_success": round(statistics.mean(successes), 1),
        "mean_steps_failure": round(statistics.mean(failures), 1),
        "mean_steps_overall": round(statistics.mean(successes + failures), 1),
    }


def steps_at_success_rate(lengths: dict[str, Any], rate: float) -> float:
    return rate * lengths["mean_steps_success"] + (1.0 - rate) * lengths["mean_steps_failure"]


def _measure(k: int, *, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    from backstop.eval.pipeline import run_pipeline  # noqa: PLC0415
    from backstop.record.timing import StageTimer  # noqa: PLC0415

    if args.warmup:
        print(f"warm-up episode (discarded) at K={k} ...")
        run_pipeline(_config(k, args, out_dir, episodes=1, tag="warmup"))

    print(f"measuring {args.episodes} episode(s) at K={k} ...")
    timer = StageTimer(sync=args.sync)
    stats = run_pipeline(_config(k, args, out_dir, episodes=args.episodes, tag="measure"), timer)
    return {
        "k_samples": k,
        "n_episodes": stats["n_total"],
        "success_rate": stats["success_rate"],
        "timing": stats["timing"],
    }


def _config(k: int, args: argparse.Namespace, out_dir: Path, *, episodes: int, tag: str) -> Any:
    """A benchmark copy of the real config: same policy, env, cameras and K path.

    Only the run's bookkeeping is changed. Recording stays *on* -- `frame_write`
    and `save_episode` are stages, and a benchmark that skipped them would
    under-report the very cost that decides whether a shard fits overnight.
    """
    from backstop.config import PipelineConfig  # noqa: PLC0415

    cfg = PipelineConfig.from_yaml(args.config)
    cfg.record.k_samples = k
    cfg.eval.n_episodes = episodes
    if args.task_ids is not None:
        cfg.env.task_ids = [int(value) for value in args.task_ids.split(",")]
    cfg.record.enabled = True
    cfg.record.root = str(BENCH_ROOT / f"k{k}_{tag}")
    cfg.record.repo_id = f"local/backstop_benchmark_k{k}_{tag}"
    cfg.record.on_existing = "overwrite"
    # Named shard, so a benchmark run can never overwrite docs/failures.md.
    cfg.record.shard = f"benchmark_k{k}"
    # Every week-2 corpus shard captures sim state, so a benchmark that skipped it
    # would price a run nobody is going to make. Under the bench root, so the
    # sidecars are cleaned up with the datasets rather than landing in data/privileged.
    cfg.record.privileged = True
    cfg.record.privileged_root = str(BENCH_ROOT / "privileged")
    cfg.output_dir = str(out_dir / f"k{k}_{tag}")
    cfg.wandb.enabled = False
    cfg.tolerance.enforce = False
    return cfg


def _print_stage_table(timing: dict[str, Any]) -> None:
    steps, episodes = timing["n_steps"], timing["n_episodes"]
    print(
        f"{episodes} episodes, {steps} steps, {timing['wall_s']:.1f} s wall "
        f"-> {timing['s_per_step']:.3f} s/step, {timing['s_per_episode']:.1f} s/episode "
        f"(cuda_sync={timing['cuda_sync']})"
    )
    print(f"  {'stage':<16}{'total s':>10}{'ms/step':>10}{'calls':>8}{'ms/call':>10}{'% wall':>9}")
    for name, entry in timing["stages"].items():
        label = f"  {name}" if "nested_in" in entry else name
        per_step = entry.get("ms_per_step", entry.get("ms_per_episode", 0.0))
        unit = "" if "ms_per_step" in entry else "*"
        print(
            f"  {label:<16}{entry['total_s']:>10.2f}{per_step:>9.1f}{unit:<1}"
            f"{entry['calls']:>8}{entry['ms_per_call']:>10.2f}{entry['pc_of_wall']:>8.1f}%"
        )
    for name, entry in timing.get("derived", {}).items():
        print(
            f"  {name:<14}~ {entry['total_s']:>10.2f}{entry['ms_per_step']:>9.1f} "
            f"{'':>7}{'':>10}{entry['pc_of_wall']:>8.1f}%"
        )
    print(
        f"  {'unaccounted':<16}{timing['unaccounted_s']:>10.2f}{'':>9} {'':>8}{'':>10}"
        f"{timing['unaccounted_pc']:>8.1f}%"
    )
    if timing["episode_wall_s"]:
        per = ", ".join(
            f"{w:.0f}s/{s}st"
            for w, s in zip(timing["episode_wall_s"], timing["episode_steps"], strict=True)
        )
        print(f"  per episode: {per}")
    print("  * ms/episode, not ms/step;  ~ derived by subtraction, not measured")


def _print_k_comparison(runs: dict[str, Any]) -> None:
    if len(runs) < 2:
        return
    base_key = min(runs, key=lambda key: int(key))
    base = runs[base_key]["timing"]
    print(f"\n=== K scaling (vs K={base_key}) " + "=" * 40)
    print(f"  {'K':<4}{'s/step':>10}{'x base':>9}{'samples ms/step':>18}{'select ms/step':>17}")
    for key in sorted(runs, key=lambda value: int(value)):
        timing = runs[key]["timing"]
        stages = timing["stages"]
        ratio = timing["s_per_step"] / base["s_per_step"] if base["s_per_step"] else 0.0
        print(
            f"  {key:<4}{timing['s_per_step']:>10.3f}{ratio:>9.2f}"
            f"{stages.get('policy_samples', {}).get('ms_per_step', 0.0):>18.1f}"
            f"{stages.get('policy_select', {}).get('ms_per_step', 0.0):>17.1f}"
        )


def _project(runs: dict[str, Any], lengths: dict[str, Any], args: argparse.Namespace) -> dict:
    """Hours for a corpus, at the clean arm's outcome mix and at the perturbed target."""
    rates = {
        "clean_arm": lengths["observed_success_rate"],
        "perturbed_target": args.target_success_rate,
    }
    out: dict[str, Any] = {"corpus_episodes": args.corpus_episodes, "scenarios": {}}
    for name, rate in rates.items():
        steps = steps_at_success_rate(lengths, rate)
        out["scenarios"][name] = {
            "success_rate": round(rate, 4),
            "steps_per_episode": round(steps, 1),
            "hours_by_k": {
                key: round(
                    runs[key]["timing"]["s_per_step"] * steps * args.corpus_episodes / 3600.0, 2
                )
                for key in sorted(runs, key=lambda value: int(value))
            },
        }
    return out


def _print_projection(
    projection: dict[str, Any], lengths: dict[str, Any], args: argparse.Namespace
) -> None:
    print(f"\n=== Projection: {projection['corpus_episodes']} episodes " + "=" * 32)
    print(
        f"  episode length from {lengths['source']}: "
        f"success {lengths['mean_steps_success']} steps, "
        f"failure {lengths['mean_steps_failure']} steps"
    )
    for name, scenario in projection["scenarios"].items():
        hours = "  ".join(f"K={key}: {value} h" for key, value in scenario["hours_by_k"].items())
        print(
            f"  {name:<18} p(success)={scenario['success_rate']:.2f} "
            f"{scenario['steps_per_episode']:>6.1f} steps/ep   {hours}"
        )
    print(
        "  Measured on the benchmark's task subset. A full-suite shard mixes easier and "
        "harder tasks,\n  so treat these as the sizing input to the day-1 gate, not as a "
        "schedule."
    )


def _clean(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
        print(f"removed benchmark datasets at {root} (--keep-datasets to retain)")


if __name__ == "__main__":
    raise SystemExit(main())
