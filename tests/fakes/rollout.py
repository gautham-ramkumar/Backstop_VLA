"""A CPU stand-in for the policy and simulator, for running the real rollout loop.

The loop (`backstop.record.loop.run_episodes`) runs unmodified. Only the
checkpoint and LIBERO are faked:

- `FakePolicy` duck-types SmolVLA, including the private `_get_action_chunk`
  path that hides the chunk from a naive wiretap.
- `FakeVecEnv` is a one-env vector env with a fixed scene. With `with_sim=True`
  it also exposes a fake LIBERO `ControlEnv` at `envs[0].unwrapped._env`, the
  real seam `backstop.env.libero.sim_state` reaches through, so privileged
  capture runs end to end.

The environment preprocessor is the REAL `LiberoProcessorStep`, so the state
layout and the 180-degree image flip are production behaviour, not a mock.

Needs torch and lerobot (the `sim` extra). Import it only from a test module
that has already called `pytest.importorskip` for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from lerobot.envs.configs import LiberoEnv

from backstop.config import PipelineConfig
from backstop.policy.adapter import SmolVLAAdapter
from backstop.record.loop import run_episodes
from backstop.schema import ACTION_DIM, CHUNK_SIZE

IMAGE_HW = 16
DONE_AT = 4
POST_SCALE = 10.0
TASK = "pick up the black bowl"

EEF_POS = np.array([[0.11, 0.22, 0.33]], dtype=np.float64)
EEF_QUAT = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)  # identity
GRIPPER_QPOS = np.array([[0.037, -0.037]], dtype=np.float64)


class FakePolicy:
    """Duck-types SmolVLAPolicy, including the trap that hides the chunk.

    IMPORTANT: `select_action` must NOT call `predict_action_chunk`. Real SmolVLA
    routes through `self._get_action_chunk`, which is exactly why an adapter that
    wiretaps `predict_action_chunk` never sees a chunk. Do not "simplify" this
    fake -- doing so silently un-tests the bug.
    """

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=32,
            n_action_steps=1,
            num_steps=10,
            output_features={"action": SimpleNamespace(shape=(ACTION_DIM,))},
        )

    def reset(self) -> None:
        return None

    def chunk_for(self, noise: Any = None) -> Any:
        """Deterministic given `noise`, exactly like a real flow-matching policy.

        Deliberately carries NO dependence on call order. Real SmolVLA's output is
        a function of (observation, noise) alone, so a fake whose output drifts
        with a call counter would fail the K-independence test for a reason that
        does not exist in production.
        """
        ramp = torch.arange(CHUNK_SIZE, dtype=torch.float32).view(1, CHUNK_SIZE, 1)
        offset = torch.arange(ACTION_DIM, dtype=torch.float32).view(1, 1, ACTION_DIM)
        base = ramp * 0.01 + offset * 0.1
        if noise is None:
            noise = torch.randn(1, CHUNK_SIZE, ACTION_DIM)  # global RNG, non-reproducible
        return base + noise[:, :, :ACTION_DIM]

    def _get_action_chunk(self, batch: Any, noise: Any = None) -> Any:
        return self.chunk_for(noise)

    def predict_action_chunk(self, batch: Any, noise: Any = None) -> Any:
        return self._get_action_chunk(batch, noise)

    def select_action(self, batch: Any, noise: Any = None) -> Any:
        return self._get_action_chunk(batch, noise)[:, 0]


class SpyPostprocessor:
    """Stands in for the unnormalize step. Scaling makes a skipped call visible."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, action: Any) -> Any:
        self.calls += 1
        return action * POST_SCALE


class FakeControlEnv:
    """The two LIBERO `ControlEnv` methods the privileged path calls.

    The state is the step count followed by the running sum of the actions
    applied, so it changes whenever the executed actions do.
    """

    def __init__(self) -> None:
        self.state = np.zeros(1 + ACTION_DIM, dtype=np.float64)

    def get_sim_state(self) -> np.ndarray:
        return self.state.copy()

    def regenerate_obs_from_state(self, state: np.ndarray) -> dict[str, Any]:
        # Never called by recording, but `libero_control_env` refuses an env without it.
        self.state = np.asarray(state, dtype=np.float64).copy()
        return {}

    def advance(self, action: np.ndarray) -> None:
        self.state[0] += 1
        self.state[1:] += np.asarray(action, dtype=np.float64).reshape(-1)[:ACTION_DIM]


class FakeVecEnv:
    def __init__(self, image: np.ndarray, *, with_sim: bool = False) -> None:
        self._image = image
        self.t = 0
        self.control = FakeControlEnv() if with_sim else None
        if self.control is not None:
            self.envs = [SimpleNamespace(unwrapped=SimpleNamespace(_env=self.control))]

    def _obs(self) -> dict[str, Any]:
        return {
            "pixels": {
                "image": self._image[None, ...].copy(),
                "image2": self._image[None, ::-1].copy(),
            },
            "robot_state": {
                # Key order matches gymnasium's `spaces.Dict`, which sorts alphabetically.
                # That ordering is why a "grab the first leaf" fallback lands on `mat`.
                "eef": {
                    "mat": np.eye(3, dtype=np.float64)[None, ...].copy(),
                    "pos": EEF_POS.copy(),
                    "quat": EEF_QUAT.copy(),
                },
                "gripper": {
                    "qpos": GRIPPER_QPOS.copy(),
                    "qvel": np.zeros((1, 2), dtype=np.float64),
                },
                "joints": {
                    "pos": np.zeros((1, 7), dtype=np.float64),
                    "vel": np.zeros((1, 7), dtype=np.float64),
                },
            },
        }

    def reset(self, seed: Any = None, options: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.t = 0
        if self.control is not None:
            self.control.state[:] = 0.0
        return self._obs(), {"is_success": np.array([False])}

    def call(self, name: str) -> tuple[Any, ...]:
        if name == "task_description":
            return (TASK,)
        if name == "_max_episode_steps":
            return (DONE_AT * 4,)
        raise AttributeError(name)

    def step(self, action: np.ndarray) -> tuple[Any, ...]:
        self.t += 1
        if self.control is not None:
            self.control.advance(action)
        done = self.t >= DONE_AT
        return (
            self._obs(),
            np.array([0.0]),
            np.array([done]),
            np.array([False]),
            {"is_success": np.array([done])},
        )

    def close(self) -> None:
        return None


class CaptureDataset:
    """Implements the whole surface the loop uses: add_frame/save_episode/finalize."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.episodes = 0
        self.finalized = False

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.episodes += 1

    def finalize(self) -> None:
        self.finalized = True


@dataclass
class Rollout:
    frames: list[dict[str, Any]]
    stats: dict[str, Any]
    post: SpyPostprocessor
    policy: FakePolicy
    adapter: SmolVLAAdapter
    image: np.ndarray


def fake_image() -> np.ndarray:
    pixels = np.arange(IMAGE_HW * IMAGE_HW * 3) % 251
    return pixels.reshape(IMAGE_HW, IMAGE_HW, 3).astype(np.uint8)


def run_fake_rollout(
    tmp_path: Path,
    k_samples: int,
    *,
    n_episodes: int = 1,
    skip_episodes: int = 0,
    record_overrides: dict[str, Any] | None = None,
    with_sim: bool = False,
) -> Rollout:
    """`run_episodes` on the fakes, with a config built the way a YAML would be."""
    image = fake_image()
    policy = FakePolicy()
    adapter = SmolVLAAdapter(policy, n_action_steps=1, num_steps=10)
    post = SpyPostprocessor()
    env_pre, _unused_env_post = LiberoEnv(task="libero_spatial").get_env_processors()
    dataset = CaptureDataset()

    record: dict[str, Any] = {
        "enabled": True,
        "k_samples": k_samples,
        "root": str(tmp_path / "ds"),
    }
    record.update(record_overrides or {})
    cfg = PipelineConfig.model_validate(
        {
            "seed": 0,
            "output_dir": str(tmp_path),
            "policy": {"type": "smolvla", "path": "fake", "device": "cpu"},
            "env": {"type": "libero", "task": "libero_spatial", "task_ids": [0]},
            "eval": {"n_episodes": n_episodes, "batch_size": 1},
            "record": record,
        }
    )
    stats = run_episodes(
        envs={"libero_spatial": {0: FakeVecEnv(image, with_sim=with_sim)}},
        adapter=adapter,
        processors={
            "env_preprocessor": env_pre,
            "env_postprocessor": lambda transition: transition,
            "preprocessor": lambda observation: observation,
            "postprocessor": post,
        },
        cfg=cfg,
        dataset=dataset,
        start_seed=0,
        skip_episodes=skip_episodes,
    )
    return Rollout(
        frames=dataset.frames,
        stats=stats,
        post=post,
        policy=policy,
        adapter=adapter,
        image=image,
    )
