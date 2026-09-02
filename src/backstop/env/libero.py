"""LIBERO environment factory. OSMesa is the default on 8 GB GPUs."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from backstop.config import EnvYamlConfig, EvalYamlConfig


def apply_render_backend(mujoco_gl: str) -> None:
    os.environ["MUJOCO_GL"] = mujoco_gl
    if mujoco_gl == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def ensure_libero_config() -> Path:
    """Write ~/.libero/config.yaml so hf-libero does not prompt on first import."""
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return config_file
    root = _installed_libero_root()
    payload = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root.parent / "datasets"),
        "assets": str(root / "assets"),
    }
    config_file.write_text(yaml.safe_dump(payload))
    return config_file


def _installed_libero_root() -> Path:
    for entry in sys.path:
        candidate = Path(entry) / "libero" / "libero"
        if (candidate / "bddl_files").is_dir():
            return candidate
    raise FileNotFoundError("hf-libero is not installed; uv sync --extra sim")


def build_libero_env_config(env_cfg: EnvYamlConfig, eval_cfg: EvalYamlConfig) -> Any:
    ensure_libero_config()
    from lerobot.envs.configs import LiberoEnv  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "task": env_cfg.task,
        "init_states": env_cfg.init_states,
        "control_mode": env_cfg.control_mode,
        "max_parallel_tasks": eval_cfg.max_parallel_tasks,
    }
    if env_cfg.task_ids is not None:
        kwargs["task_ids"] = list(env_cfg.task_ids)
    params = inspect.signature(LiberoEnv).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return LiberoEnv(**kwargs)


def make_libero_envs(env_cfg: EnvYamlConfig, eval_cfg: EvalYamlConfig) -> dict[str, dict[int, Any]]:
    """Build LeRobot LIBERO vector envs. Imports lerobot only when called."""
    apply_render_backend(env_cfg.mujoco_gl)
    ensure_libero_config()
    from lerobot.envs import make_env  # noqa: PLC0415

    lerobot_env = build_libero_env_config(env_cfg, eval_cfg)
    return make_env(
        lerobot_env,
        n_envs=eval_cfg.batch_size,
        use_async_envs=False,
    )


def close_libero(envs: dict[str, dict[int, Any]]) -> None:
    from lerobot.envs import close_envs  # noqa: PLC0415

    close_envs(envs)
