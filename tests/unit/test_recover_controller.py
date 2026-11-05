"""The rules of a guarded run: attempt budget, cooldown, stopwatch restart."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backstop.guard.online import step_disagreement
from backstop.guard.stopwatch import StopwatchRule
from backstop.recover.controller import RecoveryConfig, RecoveryController, Verdict
from backstop.recover.reposition import METRES_PER_UNIT, TRANSLATION, RepositionConfig
from backstop.schema import ACTION_DIM, CHUNK_SIZE, EEF_POS

CALM = np.zeros((4, CHUNK_SIZE, ACTION_DIM))
SPREAD = np.random.default_rng(0).normal(size=(4, CHUNK_SIZE, ACTION_DIM))
RULE = StopwatchRule(
    line=step_disagreement(SPREAD) / 2, stop_step=50, k=1, n_calibration=10, bound=0.1
)
START = np.array([-0.21, -0.01, 1.17, 3.14, 0.0, -0.09, 0.039, -0.039])


def _recover(controller: RecoveryController, step: int) -> int:
    """Drive a started recovery on a toy arm; returns the step it ended on."""
    state = START.copy()
    while (action := controller.motion_action(state, step)) is not None:
        state[EEF_POS] += 0.25 * action[TRANSLATION] * METRES_PER_UNIT
        step += 1
    return step


def test_a_quiet_guard_lets_the_policy_act() -> None:
    controller = RecoveryController(RULE, RecoveryConfig())
    assert all(controller.after_policy(CALM, step, START) is Verdict.ACT for step in range(50))
    assert controller.attempts == []


def test_a_firing_guard_starts_a_recovery_that_ends_and_re_arms() -> None:
    controller = RecoveryController(RULE, RecoveryConfig())
    assert controller.after_policy(SPREAD, 10, START) is Verdict.RECOVER
    assert controller.recovering
    end = _recover(controller, 10)
    assert not controller.recovering
    assert end > 10
    assert [(a.start, a.end, a.reason) for a in controller.attempts] == [(10, end, "signal")]


def test_the_signal_is_muted_for_the_cooldown_after_a_recovery() -> None:
    controller = RecoveryController(RULE, RecoveryConfig(cooldown_steps=20))
    controller.after_policy(SPREAD, 10, START)
    end = _recover(controller, 10)
    assert controller.after_policy(SPREAD, end + 19, START) is Verdict.ACT
    assert controller.after_policy(SPREAD, end + 20, START) is Verdict.RECOVER


def test_the_stopwatch_counts_from_the_end_of_the_latest_recovery() -> None:
    controller = RecoveryController(RULE, RecoveryConfig())
    assert controller.after_policy(CALM, 50, START) is Verdict.RECOVER  # the clock, from 0
    assert controller.attempts[0].reason == "clock"
    end = _recover(controller, 50)
    assert controller.after_policy(CALM, end + 49, START) is Verdict.ACT
    assert controller.after_policy(CALM, end + 50, START) is Verdict.RECOVER


def test_the_guard_halts_the_run_once_the_attempts_are_spent() -> None:
    controller = RecoveryController(RULE, RecoveryConfig(max_attempts=1, cooldown_steps=0))
    assert controller.fired(5, START, "offline") is Verdict.RECOVER
    end = _recover(controller, 5)
    assert controller.after_policy(SPREAD, end, START) is Verdict.HALT
    assert controller.halted_at == end
    assert len(controller.attempts) == 1


def test_zero_attempts_is_the_plain_guard() -> None:
    controller = RecoveryController(RULE, RecoveryConfig(max_attempts=0))
    assert controller.after_policy(SPREAD, 3, START) is Verdict.HALT


def test_shipped_config_loads() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "recovery" / "recovery.yaml"
    config = RecoveryConfig.from_yaml(path)
    assert config == RecoveryConfig()
    assert config.reposition == RepositionConfig()
