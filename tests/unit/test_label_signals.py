"""Contract for the auto proto-labeller's signal heuristics (no GPU, no corpus).

Every threshold and slice in `backstop.label.signals` was calibrated against one
real recorded episode (`backstop_spatial_clean_k4` episode 0 -- see the module
docstring and `Week2_plan.md`'s `review-tool` entry for the actual numbers: bowl
moved 0.125 m on a real pick, terminal bowl-to-plate distance 0.033 m on a real
success). These tests exercise the arithmetic against synthetic arrays built
from the module's own slice constants, not against that recording -- the
real-data check already happened once, by hand; what these tests must keep
catching is a regression in the arithmetic itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from backstop.label.signals import (
    BOWL_1_POS_SLICE,
    GRASP_DISTANCE_M,
    LIBERO_SPATIAL_SIM_STATE_WIDTH,
    PLATE_1_POS_SLICE,
    eef_path_length,
    eef_positions,
    gripper_close_events,
    gripper_width,
    motion_shortfall,
    object_signals,
    propose_category,
    propose_onset_range,
    target_displacement,
)


def _state(
    gripper_widths: list[float], eef_xy: list[tuple[float, float]] | None = None
) -> np.ndarray:
    """(T, 8) synthetic observation.state: eef_pos(3) + axisangle(3) + gripper(2).

    `gripper_widths[t]` becomes finger1=+w/2, finger2=-w/2 so `gripper_width`
    recovers exactly `w`.
    """
    t = len(gripper_widths)
    state = np.zeros((t, 8), dtype=np.float32)
    if eef_xy is not None:
        for i, (x, y) in enumerate(eef_xy):
            state[i, 0] = x
            state[i, 1] = y
    for i, w in enumerate(gripper_widths):
        state[i, 6] = w / 2
        state[i, 7] = -w / 2
    return state


def _sim_state(
    bowl_positions: list[tuple[float, float, float]],
    plate_position: tuple[float, float, float],
) -> np.ndarray:
    t = len(bowl_positions)
    sim = np.zeros((t, LIBERO_SPATIAL_SIM_STATE_WIDTH), dtype=np.float64)
    for i, pos in enumerate(bowl_positions):
        sim[i, BOWL_1_POS_SLICE] = pos
    sim[:, PLATE_1_POS_SLICE] = plate_position
    return sim


def test_gripper_width_open_and_closed() -> None:
    state = _state([0.077, 0.077, 0.01, 0.01])
    width = gripper_width(state)
    assert np.allclose(width, [0.077, 0.077, 0.01, 0.01])


def test_gripper_close_events_counts_transitions_and_last_close() -> None:
    # open, open, close, close, open, close
    state = _state([0.08, 0.08, 0.01, 0.01, 0.08, 0.005])
    cycles = gripper_close_events(state)
    # transitions: open->close (t=2), close->open (t=4), open->close (t=5) = 3
    assert cycles.n_cycles == 3
    assert cycles.last_close_step == 5


def test_gripper_close_events_starts_closed() -> None:
    state = _state([0.01, 0.01, 0.01])
    cycles = gripper_close_events(state)
    assert cycles.n_cycles == 0
    assert cycles.last_close_step == 0


def test_gripper_close_events_never_closes() -> None:
    state = _state([0.08, 0.08, 0.08])
    cycles = gripper_close_events(state)
    assert cycles.n_cycles == 0
    assert cycles.last_close_step is None


def test_eef_path_length_sums_step_displacement() -> None:
    state = _state([0.08, 0.08, 0.08], eef_xy=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    assert eef_path_length(state) == pytest.approx(2.0)


def test_eef_path_length_single_step_is_zero() -> None:
    state = _state([0.08])
    assert eef_path_length(state) == 0.0


def test_object_signals_none_when_sim_state_missing_or_wrong_width() -> None:
    state = _state([0.08, 0.01])
    closed = gripper_width(state) <= 0.03
    assert object_signals(None, eef_positions(state), closed) is None
    wrong_width = np.zeros((2, 105), dtype=np.float64)  # distractor's layout
    assert object_signals(wrong_width, eef_positions(state), closed) is None


def test_object_signals_detects_grasp_and_displacement() -> None:
    # eef sits at the bowl's initial position throughout, gripper closes at t=1,
    # bowl is carried 0.15 m and ends 0.02 m from the plate.
    bowl_start = (0.0, 0.0, 0.9)
    plate_pos = (0.15, 0.0, 0.9)
    bowl_positions = [bowl_start, bowl_start, (0.05, 0.0, 0.9), (0.13, 0.0, 0.9)]
    state = _state([0.08, 0.01, 0.01, 0.01], eef_xy=[(0.0, 0.0)] * 4)
    state[:, 2] = 0.9  # eef z matches the bowl's, so eef-to-bowl distance is planar
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, plate_pos)
    result = object_signals(sim_state, eef_positions(state), closed)
    assert result is not None
    assert result.grasped_ever is True
    assert result.grasped_step == 1
    assert result.bowl_displacement_m == pytest.approx(0.13, abs=1e-6)
    assert result.terminal_bowl_to_plate_m == pytest.approx(0.02, abs=1e-6)


def test_object_signals_not_grasped_when_eef_never_close_enough() -> None:
    bowl_positions = [(0.0, 0.0, 0.9)] * 3
    state = _state([0.01, 0.01, 0.01], eef_xy=[(10.0, 10.0)] * 3)  # far away, but closed
    state[:, 2] = 0.9
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, (0.2, 0.0, 0.9))
    result = object_signals(sim_state, eef_positions(state), closed)
    assert result is not None
    assert result.grasped_ever is False
    assert result.grasped_step is None


def test_propose_category_success_short_circuits() -> None:
    assert propose_category(objects=None, success=True) == "success"


def test_propose_category_no_object_signal() -> None:
    assert propose_category(objects=None, success=False) == "unclassified_no_object_signal"


def test_propose_category_no_grasp_attempt() -> None:
    bowl_positions = [(0.0, 0.0, 0.9)] * 3
    state = _state([0.08, 0.08, 0.08], eef_xy=[(5.0, 5.0)] * 3)
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, (0.2, 0.0, 0.9))
    objects = object_signals(sim_state, eef_positions(state), closed)
    assert propose_category(objects, success=False) == "no_grasp_attempt"


def test_propose_category_grasp_failed_when_barely_displaced() -> None:
    bowl_positions = [(0.0, 0.0, 0.9), (0.0, 0.0, 0.9), (0.005, 0.0, 0.9)]
    state = _state([0.08, 0.01, 0.01], eef_xy=[(0.0, 0.0)] * 3)
    state[:, 2] = 0.9
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, (0.2, 0.0, 0.9))
    objects = object_signals(sim_state, eef_positions(state), closed)
    assert objects is not None and objects.grasped_ever
    assert propose_category(objects, success=False) == "grasp_failed"


def test_propose_category_misplaced_when_far_from_plate() -> None:
    bowl_positions = [(0.0, 0.0, 0.9), (0.0, 0.0, 0.9), (0.3, 0.3, 0.9)]
    state = _state([0.08, 0.01, 0.01], eef_xy=[(0.0, 0.0)] * 3)
    state[:, 2] = 0.9
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, (2.0, 2.0, 0.9))  # plate is far away
    objects = object_signals(sim_state, eef_positions(state), closed)
    assert objects is not None and objects.grasped_ever
    assert propose_category(objects, success=False) == "misplaced"


def test_propose_category_near_miss_when_close_to_plate() -> None:
    bowl_positions = [(0.0, 0.0, 0.9), (0.0, 0.0, 0.9), (0.18, 0.0, 0.9)]
    plate_pos = (0.2, 0.0, 0.9)
    state = _state([0.08, 0.01, 0.01], eef_xy=[(0.0, 0.0)] * 3)
    state[:, 2] = 0.9
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, plate_pos)
    objects = object_signals(sim_state, eef_positions(state), closed)
    assert objects is not None and objects.grasped_ever
    assert propose_category(objects, success=False) == "near_miss"


def test_propose_onset_range_none_without_objects() -> None:
    state = _state([0.08])
    assert propose_onset_range(None, eef_positions(state), n_steps=1) is None


def test_propose_onset_range_brackets_the_last_improvement() -> None:
    # Bowl approaches the plate monotonically until t=10, then stalls.
    n = 20
    plate_pos = (0.0, 0.0, 0.9)
    bowl_positions = [(1.0 - 0.1 * min(t, 10), 0.0, 0.9) for t in range(n)]
    state = _state([0.01] * n, eef_xy=[(0.0, 0.0)] * n)
    state[:, 2] = 0.9
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, plate_pos)
    objects = object_signals(sim_state, eef_positions(state), closed)
    assert objects is not None and objects.grasped_ever
    onset = propose_onset_range(objects, eef_positions(state), n_steps=n, pad_before=2, pad_after=3)
    assert onset == (8, 13)


def test_propose_onset_range_clips_to_episode_bounds() -> None:
    n = 5
    plate_pos = (0.0, 0.0, 0.9)
    bowl_positions = [(1.0 - 0.5 * t, 0.0, 0.9) for t in range(n)]  # improves every step
    state = _state([0.01] * n, eef_xy=[(0.0, 0.0)] * n)
    state[:, 2] = 0.9
    closed = gripper_width(state) <= 0.03
    sim_state = _sim_state(bowl_positions, plate_pos)
    objects = object_signals(sim_state, eef_positions(state), closed)
    onset = propose_onset_range(
        objects, eef_positions(state), n_steps=n, pad_before=5, pad_after=15
    )
    assert onset == (0, n - 1)


def test_target_displacement_is_distance_from_the_start() -> None:
    sim = _sim_state([(0.0, 0.0, 0.9), (0.1, 0.0, 0.9)], (0.2, 0.0, 0.9))
    moved = target_displacement(sim)
    assert moved[0] == pytest.approx(0.0)
    assert moved[1] == pytest.approx(0.1)


def test_target_displacement_rejects_a_shifted_layout() -> None:
    with pytest.raises(ValueError, match="spatial layout"):
        target_displacement(np.zeros((3, 105)))


def test_motion_shortfall_is_zero_once_the_target_catches_up() -> None:
    profile = np.array([0.0, 0.02, 0.05])
    behind = motion_shortfall(np.array([0.0, 0.0, 0.05, 0.05]), profile)
    assert behind.tolist() == pytest.approx([0.0, 0.02, 0.0, 0.0])


def test_grasp_distance_constant_is_positive() -> None:
    # Sanity: a zero/negative threshold would make every object_signals call
    # report grasped_ever=False regardless of the scene.
    assert GRASP_DISTANCE_M > 0
