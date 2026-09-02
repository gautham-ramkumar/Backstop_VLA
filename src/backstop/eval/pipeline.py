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
from backstop.eval.report import write_eval_report, write_failures_markdown
from backstop.policy.adapter import SmolVLAAdapter
from backstop.provenance import build_provenance
from backstop.record.loop import create_dataset, run_episodes

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
    policy_cfg.pretrained_path = cfg.policy.path
    policy_cfg.device = cfg.policy.device
    if hasattr(policy_cfg, "n_action_steps"):
        policy_cfg.n_action_steps = cfg.policy.n_action_steps
    if hasattr(policy_cfg, "num_steps"):
        policy_cfg.num_steps = cfg.policy.num_steps
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor_overrides = {"device_processor": {"device": str(policy.config.device)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    return policy, preprocessor, postprocessor, policy_cfg


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
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

    dataset = create_dataset(cfg) if cfg.record.enabled else None
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
        )
    finally:
        close_libero(envs)

    in_tol = (
        cfg.tolerance.min_success_rate <= stats["success_rate"] <= cfg.tolerance.max_success_rate
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
    if not in_tol:
        provenance.notes = (
            f"success_rate {stats['success_rate']:.3f} outside "
            f"[{cfg.tolerance.min_success_rate}, {cfg.tolerance.max_success_rate}]. "
            f"If below min, retry with num_steps={cfg.tolerance.fallback_num_steps} "
            "(see docs/adr/002-checkpoint.md)."
        )

    write_eval_report(output_dir=output_dir, provenance=provenance, stats=stats)
    write_failures_markdown(output_dir / "failures.md", stats["episodes"])
    if cfg.record.enabled:
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
