"""End-to-end rollout-loop contract, on CPU, with no simulator and no weights.

Every bug this file guards against is *silent*: the loop has a fallback for each
one (zeros, first-item, skipped step), so a broken run still writes a plausible
dataset and still passes lint and the schema tests. These assertions are the only
thing standing between a wiring regression and a wasted GPU run.

Fakes cover the policy and the vector env. The environment preprocessor is the
REAL `LiberoProcessorStep`, so the state layout and the 180-degree image flip are
checked against production behaviour rather than against a mock of it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# Everything below the guards, numpy included: it arrives via the `sim` extra
# (through lerobot), so the CPU-only CI job must skip this module rather than
# fail to collect it.
torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

import numpy as np  # noqa: E402
from lerobot.envs.configs import LiberoEnv  # noqa: E402

from backstop.config import PipelineConfig  # noqa: E402
from backstop.policy.adapter import SmolVLAAdapter  # noqa: E402
from backstop.record.loop import run_episodes  # noqa: E402
from backstop.schema import ACTION_DIM, CHUNK_SIZE, STATE_DIM  # noqa: E402

IMAGE_HW = 16
DONE_AT = 4
POST_SCALE = 10.0

EEF_POS = np.array([[0.11, 0.22, 0.33]], dtype=np.float64)
EEF_QUAT = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)  # identity
GRIPPER_QPOS = np.array([[0.037, -0.037]], dtype=np.float64)


class FakePolicy:
    """Duck-types SmolVLAPolicy, including the trap that hides the chunk.

    IMPORTANT: `select_action` must NOT call `predict_action_chunk`. Real SmolVLA
    routes through `self._get_action_chunk` (modeling_smolvla.py:263), which is
    exactly why an adapter that wiretaps `predict_action_chunk` never sees a
    chunk. Do not "simplify" this fake -- doing so silently un-tests the bug.
    """

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=32,
            n_action_steps=1,
            num_steps=10,
            output_features={"action": SimpleNamespace(shape=(ACTION_DIM,))},
        )
        self.chunk_calls = 0

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
        self.chunk_calls += 1
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


class FakeVecEnv:
    def __init__(self, image: np.ndarray) -> None:
        self._image = image
        self.t = 0

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
        return self._obs(), {"is_success": np.array([False])}

    def call(self, name: str) -> tuple[Any, ...]:
        if name == "task_description":
            return ("pick up the black bowl",)
        if name == "_max_episode_steps":
            return (DONE_AT * 4,)
        raise AttributeError(name)

    def step(self, action: np.ndarray) -> tuple[Any, ...]:
        self.t += 1
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


def _run(tmp_path, k_samples: int):
    image = (np.arange(IMAGE_HW * IMAGE_HW * 3) % 251).reshape(IMAGE_HW, IMAGE_HW, 3)
    image = image.astype(np.uint8)

    policy = FakePolicy()
    adapter = SmolVLAAdapter(policy, n_action_steps=1, num_steps=10)
    post = SpyPostprocessor()
    env_pre, _unused_env_post = LiberoEnv(task="libero_spatial").get_env_processors()
    dataset = CaptureDataset()

    cfg = PipelineConfig.model_validate(
        {
            "seed": 0,
            "output_dir": str(tmp_path),
            "policy": {"type": "smolvla", "path": "fake", "device": "cpu"},
            "env": {"type": "libero", "task": "libero_spatial", "task_ids": [0]},
            "eval": {"n_episodes": 1, "batch_size": 1},
            "record": {"enabled": True, "k_samples": k_samples, "root": str(tmp_path / "ds")},
        }
    )

    stats = run_episodes(
        envs={"libero_spatial": {0: FakeVecEnv(image)}},
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
    )
    return SimpleNamespace(
        frames=dataset.frames,
        stats=stats,
        post=post,
        policy=policy,
        adapter=adapter,
        image=image,
    )


@pytest.fixture
def rollout(tmp_path):
    return _run(tmp_path, k_samples=4)


def test_loop_records_one_frame_per_step(rollout) -> None:
    assert len(rollout.frames) == DONE_AT


def test_action_chunk_is_recorded_not_zeroed(rollout) -> None:
    """Guards the monkey-patch bypass: select_action never hits predict_action_chunk."""
    chunk = rollout.frames[0]["action_chunk"]
    assert chunk.shape == (CHUNK_SIZE, ACTION_DIM)
    assert np.abs(chunk).max() > 0.0, "action_chunk is all zeros -- the chunk was never captured"


def test_sample_zero_is_the_executed_chunk(rollout) -> None:
    frame = rollout.frames[0]
    np.testing.assert_allclose(frame["action_chunk_samples"][0], frame["action_chunk"])


def test_policy_postprocessor_is_applied_every_step(rollout) -> None:
    """Guards the missing unnormalize: LIBERO's env_post is empty, so post is the only one."""
    assert rollout.post.calls >= DONE_AT, (
        f"postprocessor called {rollout.post.calls}x for {DONE_AT} steps -- "
        "actions reach MuJoCo un-unnormalized"
    )
    noise = rollout.adapter.noise_for(episode_seed=0, step=0, index=0)
    raw_first_row = rollout.policy.chunk_for(noise)[0, 0].numpy()
    np.testing.assert_allclose(rollout.frames[0]["action"], raw_first_row * POST_SCALE, rtol=1e-5)


def test_recorded_chunk_is_in_the_same_units_as_the_executed_action(rollout) -> None:
    """Executed action must be row 0 of the stored chunk -- both post-unnormalize."""
    frame = rollout.frames[0]
    np.testing.assert_allclose(frame["action"], frame["action_chunk"][0], rtol=1e-5)


def test_observation_state_carries_eef_pos_and_gripper(rollout) -> None:
    """Guards the raw_obs walk that silently grabs eef['mat'] instead."""
    state = rollout.frames[0]["observation.state"]
    assert state.shape == (STATE_DIM,)
    np.testing.assert_allclose(state[:3], EEF_POS[0], rtol=1e-5)
    np.testing.assert_allclose(state[6:8], GRIPPER_QPOS[0], rtol=1e-5)


def test_recorded_image_matches_what_the_policy_saw(rollout) -> None:
    """LiberoProcessorStep flips H and W; the corpus must store the flipped view."""
    recorded = rollout.frames[0]["observation.images.image"]
    assert recorded.shape == (IMAGE_HW, IMAGE_HW, 3)
    np.testing.assert_array_equal(recorded, rollout.image[::-1, ::-1, :])


def test_executed_trajectory_is_independent_of_k(tmp_path) -> None:
    """The whole point of seeded noise: K is an observer, not a participant.

    Without this, a K=4 week-2 corpus and a K=1 week-1 baseline diverge at the
    same seed, and any success-rate gap is unattributable.
    """
    k1 = _run(tmp_path / "k1", k_samples=1)
    k4 = _run(tmp_path / "k4", k_samples=4)
    assert len(k1.frames) == len(k4.frames)
    for step, (a, b) in enumerate(zip(k1.frames, k4.frames, strict=True)):
        np.testing.assert_allclose(
            a["action"], b["action"], rtol=1e-6, err_msg=f"trajectory diverged at step {step}"
        )
        np.testing.assert_allclose(a["action_chunk"], b["action_chunk"], rtol=1e-6)


def test_k_samples_are_distinct_from_each_other(rollout) -> None:
    """K identical chunks would make chunk-disagreement identically zero."""
    samples = rollout.frames[0]["action_chunk_samples"]
    assert samples.shape[0] == 4
    for index in range(1, 4):
        assert not np.allclose(samples[index], samples[0]), f"sample {index} equals the executed"


def test_sampling_leaves_no_trace_on_the_executed_chunk(rollout) -> None:
    """`sample_chunks` runs through the same wrapped hook that captures the chunk."""
    for step, frame in enumerate(rollout.frames):
        np.testing.assert_allclose(
            frame["action"],
            frame["action_chunk"][0],
            rtol=1e-5,
            err_msg=f"stored chunk is not the executed one at step {step}",
        )
