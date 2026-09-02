"""Run provenance: git SHA, MuJoCo version, policy pin, seed."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from backstop.schema import RunProvenance


def git_sha(repo_root: Path | None = None) -> str | None:
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def mujoco_version() -> str | None:
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return None
    return getattr(mujoco, "__version__", None)


def build_provenance(
    cfg: Any, *, output_dir: Path, dataset_root: Path | None = None
) -> RunProvenance:
    task_ids = cfg.env.task_ids
    return RunProvenance(
        git_sha=git_sha(),
        policy_path=cfg.policy.path,
        policy_revision=cfg.policy.revision,
        policy_type=cfg.policy.type,
        device=cfg.policy.device,
        mujoco_version=mujoco_version(),
        mujoco_gl=cfg.env.mujoco_gl,
        seed=cfg.seed,
        suite=cfg.env.task,
        task_ids=list(task_ids) if task_ids is not None else None,
        n_episodes=cfg.eval.n_episodes,
        n_action_steps=cfg.policy.n_action_steps,
        num_steps=cfg.policy.num_steps,
        k_samples=cfg.record.k_samples,
        batch_size=cfg.eval.batch_size,
        dataset_root=str(dataset_root) if dataset_root is not None else cfg.record.root,
        output_dir=str(output_dir),
    )
