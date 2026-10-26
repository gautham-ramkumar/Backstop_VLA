"""The simulator seam in `backstop.env.libero`, on fakes shaped like LeRobot's env.

`restore_observation` is the branch point of a counterfactual run, so it must
return what `vec_env.reset` would: the per-env LeRobot observation, batched by
gymnasium's own `concatenate` against the vector env's observation space. The
real-simulator check (restore right after a reset equals the reset observation,
pixel for pixel) is in `scripts/verify_sim_state_roundtrip.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402

from backstop.env.libero import (  # noqa: E402
    max_episode_steps,
    restore_observation,
    swap_task_files,
    task_description,
)

SPACE = spaces.Dict(
    {
        "pixels": spaces.Dict({"image": spaces.Box(0, 255, (2, 2, 3), np.uint8)}),
        "robot_state": spaces.Dict({"pos": spaces.Box(-1.0, 1.0, (3,), np.float64)}),
    }
)


class FakeControlEnv:
    def __init__(self) -> None:
        self.restored: Any = None

    def get_sim_state(self) -> np.ndarray:
        return np.zeros(3)

    def regenerate_obs_from_state(self, state: np.ndarray) -> dict[str, Any]:
        self.restored = state
        return {"image": np.full((2, 2, 3), 7, np.uint8), "pos": np.asarray(state, np.float64)}


class FakeLiberoEnv:
    def __init__(self) -> None:
        self._env = FakeControlEnv()

    def _format_raw_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {"pixels": {"image": raw["image"]}, "robot_state": {"pos": raw["pos"]}}


def _vec_env(n_envs: int = 1) -> Any:
    envs = [SimpleNamespace(unwrapped=FakeLiberoEnv()) for _ in range(n_envs)]
    return SimpleNamespace(envs=envs, single_observation_space=SPACE)


def test_restore_observation_is_batched_like_a_reset() -> None:
    vec_env = _vec_env()
    state = np.array([0.1, 0.2, 0.3])
    observation = restore_observation(vec_env, state)
    assert vec_env.envs[0].unwrapped._env.restored is state
    assert observation["pixels"]["image"].shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(observation["pixels"]["image"][0], 7)
    np.testing.assert_array_equal(observation["robot_state"]["pos"], state[None])


def test_restore_observation_refuses_more_than_one_env() -> None:
    with pytest.raises(ValueError, match="one-env"):
        restore_observation(_vec_env(n_envs=2), np.zeros(3))


def test_restore_observation_before_the_first_reset_raises() -> None:
    vec_env = _vec_env()
    vec_env.envs[0].unwrapped._env = None
    with pytest.raises(RuntimeError, match="built lazily"):
        restore_observation(vec_env, np.zeros(3))


def test_task_description_and_episode_cap_come_from_the_vector_env() -> None:
    def call(name: str) -> tuple[Any, ...]:
        return {"task_description": ("pick up the bowl",), "_max_episode_steps": (280,)}[name]

    assert task_description(SimpleNamespace(call=call), fallback="x") == "pick up the bowl"
    assert max_episode_steps(SimpleNamespace(call=call)) == 280
    assert task_description(SimpleNamespace(), fallback="libero_spatial/task_0") == (
        "libero_spatial/task_0"
    )


def test_swap_task_files_replaces_both_before_the_simulator_exists() -> None:
    env = SimpleNamespace(_env=None, _task_bddl_file="a.bddl", _init_states=np.zeros((2, 5)))
    swap_task_files(env, lambda path, states: (path + ".new", states[:1]))
    assert env._task_bddl_file == "a.bddl.new"
    assert env._init_states.shape == (1, 5)
    env._env = object()
    with pytest.raises(RuntimeError, match="before the LIBERO simulator is built"):
        swap_task_files(env, lambda path, states: (path, states))
