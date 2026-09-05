"""Shared eval/record pipeline."""

from __future__ import annotations

import logging
import os
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
from backstop.policy.adapter import SmolVLAAdapter
from backstop.provenance import build_provenance
from backstop.record.loop import create_dataset, run_episodes
from backstop.record.timing import StageTimer

logger = logging.getLogger(__name__)


def load_lerobot_policy(cfg: PipelineConfig, env_cfg: Any) -> Any:
    from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies import make_policy, make_pre_post_processors  # noqa: PLC0415

    load_kwargs: dict[str, Any] = {}
    if cfg.policy.revision:
        load_kwargs["revision"] = cfg.policy.revision
    try:
        policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy.path, **load_kwargs)
    except TypeError:
        policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy.path)
    # Declared as `Path | None`; lerobot only truth-tests it and forwards it as an
    # HF `pretrained_name_or_path`, so a Hub repo id round-trips through Path fine.
    policy_cfg.pretrained_path = Path(cfg.policy.path)
    policy_cfg.device = cfg.policy.device
    # n_action_steps/num_steps live on the concrete policy config (SmolVLAConfig),
    # not the PreTrainedConfig base -- the hasattr guard is the runtime contract.
    if hasattr(policy_cfg, "n_action_steps"):
        policy_cfg.n_action_steps = cfg.policy.n_action_steps  # type: ignore[attr-defined]
    if hasattr(policy_cfg, "num_steps"):
        policy_cfg.num_steps = cfg.policy.num_steps  # type: ignore[attr-defined]
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor_overrides = {"device_processor": {"device": str(policy.config.device)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    return policy, preprocessor, postprocessor, policy_cfg


def run_pipeline(cfg: PipelineConfig, timer: StageTimer | None = None) -> dict[str, Any]:
    """Run the eval/record pipeline. `timer` is for benchmarks that want CUDA sync;
    every run is timed either way and writes `timing.json`."""
    apply_render_backend(cfg.env.mujoco_gl)
    from lerobot.envs import make_env_pre_post_processors  # noqa: PLC0415
    from lerobot.utils.random_utils import set_seed  # noqa: PLC0415

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = make_libero_envs(cfg.env, cfg.eval)
    lerobot_env_cfg = build_libero_env_config(cfg.env, cfg.eval)

    policy, preprocessor, postprocessor, policy_cfg = load_lerobot_policy(cfg, lerobot_env_cfg)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=lerobot_env_cfg, policy_cfg=policy_cfg)
    adapter = SmolVLAAdapter(
        policy,
        n_action_steps=cfg.policy.n_action_steps,
        num_steps=cfg.policy.num_steps,
    )

    dataset, n_done = (
        create_dataset(cfg, fps=int(lerobot_env_cfg.fps)) if cfg.record.enabled else (None, 0)
    )
    try:
        stats = run_episodes(
            envs=envs,
            adapter=adapter,
            processors={
                "env_preprocessor": env_pre,
                "env_postprocessor": env_post,
                "preprocessor": preprocessor,
                "postprocessor": postprocessor,
            },
            cfg=cfg,
            dataset=dataset,
            start_seed=cfg.seed,
            skip_episodes=n_done,
            timer=timer,
        )
    finally:
        close_libero(envs)

    # A resumed run measured only the tail of the shard. Judging that partial rate
    # against the band would either pass a broken run or fail a good one; verify the
    # completed shard with scripts/verify_corpus.py instead.
    in_tol: bool | None = (
        None
        if stats.get("partial")
        else cfg.tolerance.min_success_rate
        <= stats["success_rate"]
        <= cfg.tolerance.max_success_rate
    )
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
    # `docs/failures.md` is the week-1 natural-failure list that week-2 review reads.
    # Mirroring every record run into it means the first perturbed shard silently
    # replaces it. Only an unsharded, complete run may write there.
    if cfg.record.enabled and cfg.record.shard is None and not stats.get("partial"):
        dest = Path("docs") / "failures.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text((output_dir / "failures.md").read_text())

    _log_wandb(cfg, stats, provenance)
    logger.info("success_rate=%.3f in_tolerance=%s", stats["success_rate"], in_tol)
    return stats


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
