"""Shared eval/record pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backstop.config import PipelineConfig
from backstop.env.libero import (
    apply_render_backend,
    build_libero_env_config,
    close_libero,
    make_libero_envs,
)
from backstop.eval.report import write_eval_report, write_failures_markdown, write_timing_report
from backstop.eval.tolerance import tolerance_verdict
from backstop.policy.base import ChunkPolicy, Processors
from backstop.policy.smolvla import load_policy_runtime
from backstop.provenance import build_provenance
from backstop.record.loop import create_dataset, run_episodes
from backstop.record.timing import StageTimer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Runtime:
    """The simulator and the policy, built once per run."""

    envs: dict[str, dict[int, Any]]
    env_cfg: Any
    policy: ChunkPolicy
    processors: Processors


def run_pipeline(cfg: PipelineConfig, timer: StageTimer | None = None) -> dict[str, Any]:
    """Run the eval/record pipeline. `timer` is for benchmarks that want CUDA sync;
    every run is timed either way and writes `timing.json`."""
    runtime = build_runtime(cfg)
    try:
        stats = record_run(runtime, cfg, timer)
    finally:
        close_libero(runtime.envs)
    finalize_report(cfg, stats)
    return stats


def build_runtime(cfg: PipelineConfig) -> Runtime:
    apply_render_backend(cfg.env.mujoco_gl)
    from lerobot.utils.random_utils import set_seed  # noqa: PLC0415

    set_seed(cfg.seed)
    envs = make_libero_envs(cfg.env, cfg.eval)
    env_cfg = build_libero_env_config(cfg.env, cfg.eval)
    policy, processors = load_policy_runtime(cfg.policy, env_cfg)
    return Runtime(envs, env_cfg, policy, processors)


def record_run(
    runtime: Runtime, cfg: PipelineConfig, timer: StageTimer | None = None
) -> dict[str, Any]:
    dataset, n_done = (
        create_dataset(cfg, fps=int(runtime.env_cfg.fps)) if cfg.record.enabled else (None, 0)
    )
    return run_episodes(
        envs=runtime.envs,
        policy=runtime.policy,
        processors=runtime.processors,
        cfg=cfg,
        dataset=dataset,
        start_seed=cfg.seed,
        skip_episodes=n_done,
        timer=timer,
    )


def finalize_report(cfg: PipelineConfig, stats: dict[str, Any]) -> None:
    """Judge tolerance, then write run.json, timing.json and failures.md.

    Sets `stats["in_tolerance"]`, which the CLIs turn into their exit code.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_tol = tolerance_verdict(
        stats["success_rate"], partial=bool(stats.get("partial")), tol=cfg.tolerance
    )
    stats["in_tolerance"] = in_tol
    provenance = build_provenance(
        cfg,
        output_dir=output_dir,
        dataset_root=Path(cfg.record.root) if cfg.record.enabled else None,
    )
    provenance.success_rate = stats["success_rate"]
    provenance.n_success = stats["n_success"]
    provenance.n_failure = stats["n_failure"]
    provenance.in_tolerance = in_tol
    if in_tol is None:
        provenance.notes = (
            f"RESUMED run: skipped {stats['n_skipped']} already-recorded episodes, so "
            f"success_rate {stats['success_rate']:.3f} covers only the tail "
            f"({stats['n_total']} episodes). Tolerance not evaluated. Verify the completed "
            "shard with scripts/verify_corpus.py."
        )
    elif not in_tol:
        provenance.notes = (
            f"success_rate {stats['success_rate']:.3f} outside "
            f"[{cfg.tolerance.min_success_rate}, {cfg.tolerance.max_success_rate}]. "
            f"If below min, retry with num_steps={cfg.tolerance.fallback_num_steps} "
            "(see docs/adr/002-checkpoint.md)."
        )

    write_eval_report(output_dir=output_dir, provenance=provenance, stats=stats)
    write_timing_report(output_dir=output_dir, timing=stats["timing"])
    write_failures_markdown(output_dir / "failures.md", stats["episodes"])
    _log_wandb(cfg, stats, provenance)
    logger.info("success_rate=%.3f in_tolerance=%s", stats["success_rate"], in_tol)


def _log_wandb(cfg: PipelineConfig, stats: dict[str, Any], provenance: Any) -> None:
    if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
        return
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        logger.warning("wandb not installed; skip logging")
        return
    os.environ.setdefault("WANDB_MODE", cfg.wandb.mode)
    run = wandb.init(
        project=cfg.wandb.project,
        config=provenance.model_dump(),
        resume="never",
    )
    run.log(
        {
            "pc_success": stats["success_rate"] * 100.0,
            "n_episodes": stats["n_total"],
            "n_success": stats["n_success"],
            "n_failure": stats["n_failure"],
            "seed": cfg.seed,
            "num_steps": cfg.policy.num_steps,
            "n_action_steps": cfg.policy.n_action_steps,
        }
    )
    run.finish()
