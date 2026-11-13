"""Lift-and-back-off motion, driven against a toy arm that moves as asked."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from backstop.recover.reposition import (
    GRIPPER_OPEN,
    METRES_PER_UNIT,
    RADIANS_PER_UNIT,
    ROTATION,
    TRANSLATION,
    Phase,
    RepositionConfig,
    RepositionMotion,
)
from backstop.schema import ACTION_DIM, EEF_POS, GRIPPER_INDEX

START = np.array([-0.21, -0.01, 1.17, 3.14, 0.0, -0.09, 0.039, -0.039])


def _run(config: RepositionConfig, lag: float = 0.25) -> tuple[list[np.ndarray], np.ndarray]:
    """Toy arm: each step moves `lag` of the requested displacement."""
    motion = RepositionMotion(config, START)
    state = START.copy()
    actions = []
    for _ in range(config.max_total_steps + 1):
        action = motion.next_action(state)
        if action is None:
            return actions, state
        actions.append(action)
        state[EEF_POS] += lag * action[TRANSLATION] * METRES_PER_UNIT
    raise AssertionError("motion did not finish within max_total_steps")


def test_ends_lifted_and_backed_off() -> None:
    config = RepositionConfig()
    _, final = _run(config)
    moved = final[EEF_POS] - START[EEF_POS]
    assert moved[2] == pytest.approx(config.lift_m, abs=2 * config.tolerance_m)
    assert moved[0] == pytest.approx(-config.back_m, abs=2 * config.tolerance_m)
    assert abs(moved[1]) < config.tolerance_m


def test_every_action_is_valid_and_keeps_the_gripper_open() -> None:
    config = RepositionConfig(max_action=0.3)
    actions, _ = _run(config)
    assert actions
    for action in actions:
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.abs(action[EEF_POS]) <= config.max_action + 1e-12)
        assert action[GRIPPER_INDEX] == GRIPPER_OPEN
        assert np.all(action[3:6] == 0.0)  # never rotates the hand


def test_holds_still_while_opening_then_lifts_before_backing_off() -> None:
    config = RepositionConfig(open_steps=3)
    actions, _ = _run(config)
    assert all(np.all(a[EEF_POS] == 0.0) for a in actions[:3])
    first_move = actions[3]
    assert first_move[2] > 0.0 and first_move[0] == pytest.approx(0.0)


def test_a_stuck_arm_still_finishes_within_the_step_cap() -> None:
    config = RepositionConfig(max_steps_per_phase=7)
    actions, _ = _run(config, lag=0.0)
    assert len(actions) == config.max_total_steps


def test_phases_run_in_order() -> None:
    motion = RepositionMotion(RepositionConfig(open_steps=1, settle_steps=1), START)
    state = START.copy()
    seen = []
    while (action := motion.next_action(state)) is not None:
        if motion.phase not in seen:
            seen.append(motion.phase)
        state[EEF_POS] += action[EEF_POS] * METRES_PER_UNIT
    assert seen == list(Phase)
    assert motion.phase is None


def test_config_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        RepositionConfig(max_action=1.5)
    with pytest.raises(ValidationError):
        RepositionConfig(lift_m=-0.1)


HOME = np.array([-0.30, 0.05, 1.20, 3.0, 0.2, -0.3, 0.039, -0.039])


def test_home_mode_returns_the_hand_to_its_start_position_and_rotation() -> None:
    """Toy arm whose hand also turns by `lag` of the requested world-frame rotation."""
    config = RepositionConfig(mode="home", max_steps_per_phase=200)
    motion = RepositionMotion(config, START, home_state=HOME)
    state = START.copy()
    for _ in range(4 * 200):
        action = motion.next_action(state)
        if action is None:
            break
        state[EEF_POS] += 0.25 * action[TRANSLATION] * METRES_PER_UNIT
        turn = Rotation.from_rotvec(0.25 * action[ROTATION] * RADIANS_PER_UNIT)
        state[3:6] = (turn * Rotation.from_rotvec(state[3:6])).as_rotvec()
    else:
        raise AssertionError("return home did not finish")
    assert np.linalg.norm(state[EEF_POS] - HOME[EEF_POS]) <= config.tolerance_m
    error = Rotation.from_rotvec(HOME[3:6]) * Rotation.from_rotvec(state[3:6]).inv()
    assert np.linalg.norm(error.as_rotvec()) <= config.rotation_tolerance_rad


def test_home_mode_needs_the_start_pose() -> None:
    with pytest.raises(ValueError, match="start pose"):
        RepositionMotion(RepositionConfig(mode="home"), START)
