"""The rollout loop's pieces on their own: `Decider` and `EpisodeSink`.

The golden test (`test_rollout_golden.py`) proves the assembled loop records
exactly what it did before the split. These check each piece's own contract
with the numpy-only policy fake, so the recovery runner can rely on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

import numpy as np  # noqa: E402

from backstop.corpus.privileged import load_episode  # noqa: E402
from backstop.policy.base import Processors  # noqa: E402
from backstop.record.privileged import PrivilegedWriter  # noqa: E402
from backstop.record.sink import EpisodeSink  # noqa: E402
from backstop.record.step import Decider, Decision  # noqa: E402
from backstop.record.timing import StageTimer  # noqa: E402
from backstop.schema import ACTION_DIM, CHUNK_SIZE  # noqa: E402
from fakes.policy import FakeChunkPolicy  # noqa: E402
from fakes.rollout import CaptureDataset, FakeVecEnv, fake_image  # noqa: E402

K = 4
SCALE = 10.0


def _processors() -> Processors:
    return Processors(
        env_pre=lambda observation: observation,
        pre=lambda observation: observation,
        post=lambda action: action * SCALE,
        env_post=lambda transition: transition,
    )


def _decide(perturb: Any = lambda action: action) -> tuple[Decision, FakeChunkPolicy, StageTimer]:
    policy, timer = FakeChunkPolicy(), StageTimer()
    decision = Decider(policy, _processors(), K, timer)({}, episode_seed=7, step=3, perturb=perturb)
    return decision, policy, timer


def test_decision_is_the_executed_chunk_and_k_samples_in_env_space() -> None:
    decision, policy, _ = _decide()
    executed = policy.last_action_chunk() * SCALE
    np.testing.assert_allclose(decision.chunk, executed)
    np.testing.assert_allclose(decision.action_policy, executed[:1])
    assert decision.samples.shape == (K, CHUNK_SIZE, ACTION_DIM)
    np.testing.assert_allclose(decision.samples[0], executed)
    assert decision.action is decision.action_policy


def test_only_the_executed_action_is_perturbed() -> None:
    decision, _, _ = _decide(perturb=lambda action: action + 1.0)
    np.testing.assert_allclose(decision.action, decision.action_policy + 1.0)
    np.testing.assert_allclose(decision.chunk[0], decision.action_policy[0])


def test_decide_times_the_three_policy_stages_once_each() -> None:
    _, _, timer = _decide()
    stages = timer.summary()["stages"]
    assert {name: stages[name]["calls"] for name in stages} == {
        "policy_select": 1,
        "policy_samples": 1,
        "action_post": 1,
    }


def _decision() -> Decision:
    decision, _, _ = _decide()
    return decision


def _frame_obs() -> dict[str, np.ndarray]:
    return {"image": fake_image(), "image2": fake_image(), "state": np.zeros(8, np.float32)}


def test_sink_writes_one_frame_and_one_sim_state_per_step(tmp_path: Path) -> None:
    dataset, vec_env = CaptureDataset(), FakeVecEnv(fake_image(), with_sim=True)
    sink = EpisodeSink(dataset, PrivilegedWriter(tmp_path, "shard"), StageTimer(), K)
    sink.begin(5)
    for step in range(3):
        sink.add_sim_state(vec_env)
        sink.add_frame(_frame_obs(), _decision(), success=step == 2, task="pick up the bowl")
    sink.end()
    assert dataset.episodes == 1
    assert [bool(frame["next.success"][0]) for frame in dataset.frames] == [False, False, True]
    assert dataset.frames[0]["action_chunk_samples"].shape == (K, CHUNK_SIZE, ACTION_DIM)
    assert load_episode(tmp_path, "shard", 5)["sim_state"].shape[0] == 3


def test_sink_refuses_a_sidecar_that_does_not_line_up_with_the_frames(tmp_path: Path) -> None:
    sink = EpisodeSink(None, PrivilegedWriter(tmp_path, "shard"), StageTimer(), K)
    sink.begin(0)
    sink.add_sim_state(FakeVecEnv(fake_image(), with_sim=True))
    sink.add_frame(_frame_obs(), _decision(), success=False, task="x")
    sink.add_frame(_frame_obs(), _decision(), success=False, task="x")
    with pytest.raises(RuntimeError, match="1 sim states for 2 recorded frames"):
        sink.end()
