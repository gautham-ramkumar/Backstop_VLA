"""LIBERO environment factory. OSMesa is the default on 8 GB GPUs."""

from __future__ import annotations

import inspect
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from backstop.config import EnvYamlConfig, EvalYamlConfig


def apply_render_backend(mujoco_gl: str) -> None:
    os.environ["MUJOCO_GL"] = mujoco_gl
    if mujoco_gl == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def ensure_libero_config() -> Path:
    """Write ~/.libero/config.yaml so hf-libero does not prompt on first import.

    The assets path must be resolved, never guessed. The `libero` wheel ships
    bddl_files/benchmark/envs/init_files/utils but **no assets** -- object meshes,
    textures and scenes come from the Hub via `libero.libero.get_assets_path()`.
    LIBERO only auto-downloads them when no config file exists, so writing a
    config that names a non-existent assets directory permanently suppresses the
    download. The sim still runs and still renders; the objects are just missing,
    the policy flails, and every episode times out at 0% success with no error.

    This repairs a stale config as well as creating a new one -- the bad path is
    written to ``$HOME``, outside the repo, so it survives a clean checkout.
    """
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    root = _installed_libero_root()

    payload: dict[str, Any] = {}
    if config_file.exists():
        try:
            payload = yaml.safe_load(config_file.read_text()) or {}
        except yaml.YAMLError:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}

    payload.setdefault("benchmark_root", str(root))
    payload.setdefault("bddl_files", str(root / "bddl_files"))
    payload.setdefault("init_states", str(root / "init_files"))
    payload.setdefault("datasets", str(root.parent / "datasets"))
    payload.setdefault("assets", str(root / "assets"))
    # Bootstrap first: importing libero with no config triggers an interactive prompt.
    config_file.write_text(yaml.safe_dump(payload))

    resolved = _resolve_assets_path()
    if resolved is not None and payload.get("assets") != str(resolved):
        payload["assets"] = str(resolved)
        config_file.write_text(yaml.safe_dump(payload))

    _validate_libero_paths(payload, config_file)
    return config_file


def _resolve_assets_path() -> Path | None:
    """Ask LIBERO where assets actually are, downloading them if needed."""
    try:
        from libero.libero import get_assets_path  # noqa: PLC0415

        return Path(get_assets_path())
    except (ImportError, OSError, RuntimeError):
        return None


def _validate_libero_paths(payload: dict[str, Any], config_file: Path) -> None:
    """Fail loudly on a broken benchmark install. A silent 0% is far more expensive."""
    required = ("bddl_files", "init_states", "assets")
    missing = [key for key in required if not Path(str(payload.get(key, ""))).is_dir()]
    if missing:
        detail = "\n".join(f"  {key}: {payload.get(key)}" for key in missing)
        raise FileNotFoundError(
            f"LIBERO paths in {config_file} do not exist:\n{detail}\n"
            "Delete that file and re-run to let LIBERO resolve them from the Hub."
        )


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


def lerobot_env(vec_env: Any, index: int = 0) -> Any:
    """LeRobot's gym `LiberoEnv` for sub-env `index`, under any wrappers."""
    try:
        return vec_env.envs[index].unwrapped
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot reach the LIBERO env behind {type(vec_env).__name__}[{index}]. "
            "This path assumes in-process sub-envs (use_async_envs=False); an async "
            "vector env would need the call routed through `vec_env.call` instead."
        ) from exc


def control_env(env: Any) -> Any:
    """LIBERO's `ControlEnv` inside one LeRobot `LiberoEnv`.

    Raises rather than returning None: every caller is either capturing data that
    cannot be recovered without re-running the episode, or perturbing a scene
    that would otherwise be recorded unperturbed.
    """
    inner = getattr(env, "_env", None)
    if inner is None:
        raise RuntimeError(
            f"{type(env).__name__}._env is None -- the simulator is built lazily on "
            "first reset(). Reach for it after reset, not before."
        )
    return inner


def libero_control_env(vec_env: Any, index: int = 0) -> Any:
    """The LIBERO `ControlEnv` behind a LeRobot vector env.

    `make_libero_envs` builds a SyncVectorEnv, so sub-envs live in this process and
    the simulator is directly reachable: sub-env -> `.unwrapped` (LeRobot's gym
    `LiberoEnv`) -> `._env` (LIBERO's `OffScreenRenderEnv`, an `ControlEnv`
    subclass carrying `get_sim_state` / `regenerate_obs_from_state`).
    """
    inner = control_env(lerobot_env(vec_env, index))
    for method in ("get_sim_state", "regenerate_obs_from_state"):
        if not callable(getattr(inner, method, None)):
            raise RuntimeError(f"{type(inner).__name__} has no {method}(); LIBERO API changed.")
    return inner


def refresh_raw_obs(inner: Any) -> Any:
    """A fresh raw LIBERO observation of the simulator as it is now."""
    return inner.env._get_observations(force_update=True)


def format_raw_obs(env: Any, raw: Any) -> Any:
    """A raw LIBERO observation in the per-env LeRobot layout `reset` returns."""
    return env._format_raw_obs(raw)


def task_description(vec_env: Any, fallback: str) -> str:
    """The instruction the policy sees, as any perturbation wrapper left it."""
    try:
        return str(list(vec_env.call("task_description"))[0])
    except (AttributeError, NotImplementedError, TypeError):
        return fallback


def max_episode_steps(vec_env: Any) -> int:
    return int(vec_env.call("_max_episode_steps")[0])


def swap_task_files(env: Any, rewrite: Callable[[str, Any], tuple[str, Any]]) -> None:
    """Replace a `LiberoEnv`'s BDDL file and init states before its simulator exists.

    `rewrite(bddl_path, init_states)` returns the replacements. LeRobot builds the
    MuJoCo model from these on the first reset, so swapping them later would
    silently change nothing.
    """
    if getattr(env, "_env", None) is not None:
        raise RuntimeError(
            "Task files must be swapped before the LIBERO simulator is built; "
            "this sub-env's model already exists."
        )
    env._task_bddl_file, env._init_states = rewrite(env._task_bddl_file, env._init_states)


def sim_state(vec_env: Any, index: int = 0) -> Any:
    """Flattened MuJoCo state (time + qpos + qvel) as float64.

    Privileged. Permitted for ground-truth labels, onset annotation, and
    counterfactual branching; forbidden as guard input (ADR-003). It is kept out of
    the LeRobot dataset so that guard code cannot reach it by accident.
    """
    import numpy as np  # noqa: PLC0415

    return np.asarray(libero_control_env(vec_env, index).get_sim_state(), dtype=np.float64)


_RENDER_TIMED = "_backstop_render_timed"


def instrument_render(vec_env: Any, timer: Any, index: int = 0) -> bool:
    """Time the OSMesa camera renders that happen inside `vec_env.step`.

    Rendering is not a separable call from the loop's point of view. Robosuite's
    `MujocoEnv.step` runs the physics substeps and samples the camera observables
    in the same loop, so a stopwatch around `vec_env.step` charges both to one
    number -- and the two point corpus sizing in opposite directions, because a
    render-bound step makes K=4 nearly free while a physics-bound one does not.

    The hook wraps the bound `MjSim.render`, which is the exact call each camera
    observable makes, so `env_render` is camera time and nothing else. Physics
    then falls out as `env_step - env_render`.

    Call once per episode, not once per run: `hard_reset` is on by default, so
    LIBERO rebuilds the model and the simulator on every reset and a hook
    installed at startup is discarded by the first one. Idempotent via a flag on
    the sim object, since re-wrapping an already-wrapped render would nest the
    stopwatch and count the same microseconds once per layer.

    Returns whether the hook is in place. Unlike privileged capture, this one
    degrades rather than raises: a missing stage in a diagnostic table is not
    worth killing a six-hour recording run over, and the table reports the gap
    as unaccounted time anyway.
    """
    try:
        sim = getattr(libero_control_env(vec_env, index).env, "sim", None)
        if sim is None or not callable(getattr(sim, "render", None)):
            return False
        if getattr(sim, _RENDER_TIMED, False):
            return True
        inner = sim.render

        def timed(*args: Any, **kwargs: Any) -> Any:
            with timer("env_render"):
                return inner(*args, **kwargs)

        sim.render = timed
        setattr(sim, _RENDER_TIMED, True)
    except (RuntimeError, AttributeError, TypeError):
        return False
    return True


def restore_sim_state(vec_env: Any, state: Any, index: int = 0) -> Any:
    """Set the simulator to `state` and return the observation consistent with it.

    The same call LIBERO uses to apply an init state on every reset
    (`set_init_state` is an alias of `regenerate_obs_from_state`), so this is not a
    new code path. Returns the raw LIBERO observation dict, not a LeRobot one.
    """
    return libero_control_env(vec_env, index).regenerate_obs_from_state(state)


def restore_observation(vec_env: Any, state: Any) -> Any:
    """Restore a one-env vector env to `state`; return the observation batched like `reset`.

    The branch point of a counterfactual run: the result goes through the same
    preprocessing as any other observation. Call it after a reset, so the vector
    env holds no pending autoreset from an earlier episode.

    Close to, not identical with, the observation the run saw at `state`: after
    `mj_step`, MuJoCo's body and camera poses still describe the start of the last
    2 ms substep, and a restore recomputes them for `state` itself. Images differ
    by under 0.1 grey level on average (`scripts/verify_sim_state_roundtrip.py`).
    """
    from gymnasium.vector.utils import concatenate, create_empty_array  # noqa: PLC0415

    if len(vec_env.envs) != 1:
        raise ValueError(f"restore_observation needs a one-env vector env, got {len(vec_env.envs)}")
    observation = format_raw_obs(lerobot_env(vec_env), restore_sim_state(vec_env, state))
    space = vec_env.single_observation_space
    return concatenate(space, [observation], create_empty_array(space, n=1))


# BDDL class names whose plain name is not the snake case spelled out.
OBJECT_NAMES: dict[str, str] = {
    "akita_black_bowl": "black bowl",
    "glazed_rim_porcelain_ramekin": "ramekin",
    "cookies": "cookie box",
    "chefmate_8_frypan": "frying pan",
    "new_salad_dressing": "salad dressing",
}


def scene_objects(suite: str, task_id: int) -> tuple[str, list[str]]:
    """Task language and the plain names of its BDDL objects."""
    import libero.libero as libero_pkg  # noqa: PLC0415
    from libero.libero.benchmark import get_benchmark  # noqa: PLC0415

    task = get_benchmark(suite)().get_task(task_id)
    package_file = libero_pkg.__file__
    if package_file is None:
        raise FileNotFoundError("the libero package has no __file__; cannot find its BDDL files")
    root = Path(package_file).parent / "bddl_files" / task.problem_folder
    text = (root / task.bddl_file).read_text()
    block = re.search(r"\(:objects(.*?)\)", text, re.S)
    if block is None:
        raise ValueError(f"no :objects block in {task.bddl_file}")
    names: list[str] = sorted(set(re.findall(r"- (\w+)", block.group(1))))
    plain = [
        OBJECT_NAMES[name] if name in OBJECT_NAMES else name.replace("_", " ") for name in names
    ]
    return str(task.language), plain
