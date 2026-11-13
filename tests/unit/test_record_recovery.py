"""The guarded runner on fakes: it acts, recovers and halts as the controller says."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.envs.configs import LiberoEnv  # noqa: E402

from backstop.guard.stopwatch import StopwatchRule  # noqa: E402
from backstop.policy.base import Processors  # noqa: E402
from backstop.record.loop import EpisodeSpec  # noqa: E402
from backstop.record.obs import ObsPipeline  # noqa: E402
from backstop.record.recovery import run_guarded_episode  # noqa: E402
from backstop.record.step import Decider  # noqa: E402
from backstop.record.timing import StageTimer  # noqa: E402
from backstop.recover.controller import RecoveryConfig, RecoveryController  # noqa: E402
from backstop.recover.reposition import GRIPPER_OPEN  # noqa: E402
from backstop.schema import GRIPPER_INDEX  # noqa: E402
from fakes.policy import FakeChunkPolicy  # noqa: E402
from fakes.rollout import DONE_AT, FakeVecEnv, fake_image  # noqa: E402


class RecordingVecEnv(FakeVecEnv):
    def __init__(self) -> None:
        super().__init__(fake_image())
        self.actions: list[np.ndarray] = []

    def step(self, action: np.ndarray) -> tuple[Any, ...]:
        self.actions.append(np.asarray(action))
        return super().step(action)


def _run(
    line: float, config: RecoveryConfig, **kwargs: Any
) -> tuple[Any, RecordingVecEnv, FakeChunkPolicy]:
    env_pre, _ = LiberoEnv(task="libero_spatial").get_env_processors()
    processors = Processors(env_pre, lambda o: o, lambda a: a, lambda t: t)
    policy, env = FakeChunkPolicy(), RecordingVecEnv()
    rule = StopwatchRule(line=line, stop_step=1000, k=1, n_calibration=10, bound=0.1)
    outcome = run_guarded_episode(
        ObsPipeline(processors),
        Decider(policy, processors, 4, StageTimer()),
        RecoveryController(rule, config),
        EpisodeSpec(env, 0, 0, "libero_spatial", 0),
        keep_frames=True,
        **kwargs,
    )
    return outcome, env, policy


def test_a_quiet_guard_leaves_the_run_as_it_was() -> None:
    outcome, env, policy = _run(np.inf, RecoveryConfig())
    assert (outcome.success, outcome.n_steps, outcome.attempts) == (True, DONE_AT, [])
    assert len(policy.observations) == DONE_AT
    assert len(outcome.frames) == DONE_AT


def test_a_firing_guard_hands_the_arm_to_the_recovery_motion() -> None:
    outcome, env, policy = _run(0.0, RecoveryConfig())
    assert [a.start for a in outcome.attempts] == [0]
    assert len(policy.observations) == 1  # the policy step the guard fired on, then the motion
    assert all(action[0, GRIPPER_INDEX] == GRIPPER_OPEN for action in env.actions)
    assert outcome.n_steps == DONE_AT


def test_with_no_attempts_left_the_guard_halts_the_run() -> None:
    outcome, env, _ = _run(0.0, RecoveryConfig(max_attempts=0))
    assert (outcome.success, outcome.n_steps, outcome.halted_at) == (False, 0, 0)
    assert env.actions == []


def test_max_steps_replaces_the_step_cap() -> None:
    outcome, env, _ = _run(np.inf, RecoveryConfig(), max_steps=2)
    assert (outcome.success, outcome.n_steps, len(env.actions)) == (False, 2, 2)
