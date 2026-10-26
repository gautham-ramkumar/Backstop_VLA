"""Pure signal computation for the auto proto-labeller (Week 2 review tool).

Every function here takes plain numpy arrays and returns plain values -- no I/O,
no simulator, no dataset paths -- so the heuristics are testable against
synthetic arrays without a GPU or a recorded corpus (`tests/unit/test_label_signals.py`).

Two data sources, never mixed inside one function:

- `observation.state` (T, 8) -- already in every dataset row, every family,
  every shard. `eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)`
  (`record/obs.py::capture_obs`). Gripper open ~ +-0.039 per finger
  (width ~0.077); closed ~0-0.02 (width ~0.01-0.02) -- measured directly off a
  real successful clean episode, not assumed.
- privileged `sim_state` (T, 92) -- the flattened MuJoCo `time + qpos + qvel`
  (`record/privileged.py`). Only meaningful for the object-tracking signals
  below, and only for the 92-wide layout every family except `distractor`
  shares (`distractor` adds one free-jointed object, D=105 -- see
  `Week2_plan.md`'s `distractor-spike` entry). Object slices were read off a
  live MuJoCo model (`sim.model.jnt_qposadr`/`joint_id2name`) for
  libero_spatial task 0, not guessed, and confirmed constant across all ten
  tasks separately (`Week2_plan.md`'s `distractor-spike`: "every one of
  libero_spatial's 10 tasks shares the identical object roster ... byte-
  identical `:regions`/`:objects` block"). Every one of libero_spatial's ten
  task BDDLs also shares one goal predicate, `(On akita_black_bowl_1 plate_1)`
  -- checked directly against all ten `.bddl` files -- so the target object
  and the goal object are the same two names regardless of task_id, and no
  per-task lookup is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- observation.state layout (STATE_DIM = 8, schema.py) ---
EEF_POS_SLICE = slice(0, 3)
GRIPPER_QPOS_SLICE = slice(6, 8)

# Measured off backstop_spatial_clean_k4 episode 0 (a real success): finger
# width (finger1 - finger2) starts ~0.077 (open) and ends ~0.01-0.02 (closed
# around a held bowl). The threshold sits between the two with margin on both
# sides, not at their midpoint -- false "closed" reads are the costlier
# mistake, since they would fabricate a grasp event that never happened.
GRIPPER_CLOSED_WIDTH_M = 0.03

# Live MuJoCo introspection (task 0, `sim.model`): qpos layout is
# [7 arm joints][2 gripper fingers][akita_black_bowl_1: 7][akita_black_bowl_2: 7]
# [cookies_1: 7][ramekin_1: 7][plate_1: 7][3 cabinet-drawer hinges][stove button].
# sim_state = [time(1), qpos(48), qvel(43)], so a qpos address `a` lands at
# sim_state index `1 + a`. Free-joint pos is the first 3 of that object's 7.
_QPOS_TIME_OFFSET = 1
BOWL_1_POS_SLICE = slice(_QPOS_TIME_OFFSET + 9, _QPOS_TIME_OFFSET + 12)
PLATE_1_POS_SLICE = slice(_QPOS_TIME_OFFSET + 37, _QPOS_TIME_OFFSET + 40)
LIBERO_SPATIAL_SIM_STATE_WIDTH = 92

# eef-to-object distance under which "the gripper is at the object", and the
# displacement/proximity bands the category heuristic reads. All three are
# grounded against the one real measurement taken while building this (clean
# episode 0: bowl moved 0.125 m end to end, terminal bowl-to-plate distance
# 0.033 m on a success) -- not tuned against a labelled set, because one
# doesn't exist yet. Expect the human reviewer to override liberally; that
# disagreement rate is itself the deliverable (Week2_plan.md's "review-tool"
# entry), not a bug in the threshold.
GRASP_DISTANCE_M = 0.08
DISPLACEMENT_THRESHOLD_M = 0.03
NEAR_PLATE_THRESHOLD_M = 0.08

# Onset-range padding (steps), applied around the proposed onset step. Wide
# enough to give the human scrubber room without proposing the whole episode;
# Week2_plan.md asks for a range "not a point" precisely because the true
# onset is ambiguous.
ONSET_PAD_BEFORE = 5
ONSET_PAD_AFTER = 15


def gripper_width(state: np.ndarray) -> np.ndarray:
    """(T,) finger1 - finger2. Open ~0.077 m, closed ~0.01-0.02 m."""
    qpos = state[:, GRIPPER_QPOS_SLICE]
    return qpos[:, 0] - qpos[:, 1]


def eef_positions(state: np.ndarray) -> np.ndarray:
    """(T, 3) end-effector position, world frame."""
    return state[:, EEF_POS_SLICE]


def eef_path_length(state: np.ndarray) -> float:
    """Sum of step-to-step eef displacement. Cheap proxy for 'how much did the
    arm move', independent of whether it moved productively."""
    pos = eef_positions(state)
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


@dataclass(frozen=True)
class GripperCycles:
    n_cycles: int
    last_close_step: int | None


def gripper_close_events(
    state: np.ndarray, closed_width_m: float = GRIPPER_CLOSED_WIDTH_M
) -> GripperCycles:
    """Count open<->closed threshold crossings and the last step the gripper
    became closed. `n_cycles` counts both directions (a close and a re-open are
    two crossings), matching what a human watching the video would call "how
    many times did it fumble with the gripper"."""
    width = gripper_width(state)
    closed = width <= closed_width_m
    if len(closed) == 0:
        return GripperCycles(n_cycles=0, last_close_step=None)
    transitions = np.diff(closed.astype(np.int8))
    n_cycles = int(np.count_nonzero(transitions))
    close_steps = np.nonzero(transitions == 1)[0] + 1
    if close_steps.size:
        last_close_step: int | None = int(close_steps[-1])
    elif closed[0]:
        last_close_step = 0
    else:
        last_close_step = None
    return GripperCycles(n_cycles=n_cycles, last_close_step=last_close_step)


@dataclass(frozen=True)
class ObjectSignals:
    """Object-tracking signals. `None` fields mean 'not computed', not zero --
    callers must not treat a missing value as a real measurement."""

    bowl_positions: np.ndarray  # (T, 3)
    plate_positions: np.ndarray  # (T, 3)
    bowl_displacement_m: float
    terminal_bowl_to_plate_m: float
    grasped_ever: bool
    grasped_step: int | None  # first step the grasp proxy fired, or None


def target_displacement(sim_state: np.ndarray) -> np.ndarray:
    """(T,) distance of the target bowl from where it was at step 0.

    This reads the simulator. ADR-003 forbids it as a guard input. It is a
    ceiling: how early the cup's own motion can mark a miss.
    """
    state = np.asarray(sim_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != LIBERO_SPATIAL_SIM_STATE_WIDTH:
        raise ValueError(
            f"target displacement needs the {LIBERO_SPATIAL_SIM_STATE_WIDTH}-wide "
            f"spatial layout, got shape {getattr(state, 'shape', None)}"
        )
    if state.shape[0] == 0:
        raise ValueError("sim state has no steps")
    bowl = state[:, BOWL_1_POS_SLICE]
    return np.linalg.norm(bowl - bowl[0], axis=1)


def motion_shortfall(displacement: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """How far the target is behind a normal success at each step.

    Past the end of the profile, the last profile value is reused. A target
    that has already moved at least that far scores 0.
    """
    moved = np.asarray(displacement, dtype=np.float64).reshape(-1)
    path = np.asarray(profile, dtype=np.float64).reshape(-1)
    if moved.size == 0 or path.size == 0:
        raise ValueError("displacement and profile need at least one step")
    if moved.size <= path.size:
        expected = path[: moved.size]
    else:
        tail = np.full(moved.size - path.size, path[-1], dtype=np.float64)
        expected = np.concatenate([path, tail])
    return np.maximum(expected - moved, 0.0)


def object_signals(
    sim_state: np.ndarray | None, eef_pos: np.ndarray, gripper_closed: np.ndarray
) -> ObjectSignals | None:
    """Bowl/plate tracking from privileged sim_state. Returns None when the
    layout is not the 92-wide one this module's slices were measured against
    (today: only the `distractor` shard, which adds a free-jointed object) --
    silently guessing at a shifted layout would produce a confident-looking
    number that is not measuring the thing it claims to."""
    if sim_state is None:
        return None
    if sim_state.ndim != 2 or sim_state.shape[1] != LIBERO_SPATIAL_SIM_STATE_WIDTH:
        return None
    bowl = sim_state[:, BOWL_1_POS_SLICE]
    plate = sim_state[:, PLATE_1_POS_SLICE]
    n = min(len(bowl), len(eef_pos))
    bowl, plate, eef_pos, gripper_closed = bowl[:n], plate[:n], eef_pos[:n], gripper_closed[:n]

    displacement = np.linalg.norm(bowl - bowl[0], axis=1)
    eef_to_bowl = np.linalg.norm(eef_pos - bowl, axis=1)
    grasp_mask = gripper_closed & (eef_to_bowl <= GRASP_DISTANCE_M)
    grasped_ever = bool(grasp_mask.any())
    grasped_step = int(np.argmax(grasp_mask)) if grasped_ever else None

    return ObjectSignals(
        bowl_positions=bowl,
        plate_positions=plate,
        bowl_displacement_m=float(displacement.max()),
        terminal_bowl_to_plate_m=float(np.linalg.norm(bowl[-1] - plate[-1])),
        grasped_ever=grasped_ever,
        grasped_step=grasped_step,
    )


def propose_category(objects: ObjectSignals | None, success: bool) -> str:
    """Rule-based proposal, confirmed or overridden by the human reviewer.

    `objects is None` means the object-tracking signals could not be computed
    for this shard (see `object_signals`) -- proposes a category that says so
    rather than a guess, so a reviewer isn't misled into thinking the tool
    looked and found nothing.
    """
    if success:
        return "success"
    if objects is None:
        return "unclassified_no_object_signal"
    if not objects.grasped_ever:
        return "no_grasp_attempt"
    if objects.bowl_displacement_m < DISPLACEMENT_THRESHOLD_M:
        return "grasp_failed"
    if objects.terminal_bowl_to_plate_m > NEAR_PLATE_THRESHOLD_M:
        return "misplaced"
    return "near_miss"


def propose_onset_range(
    objects: ObjectSignals | None,
    eef_pos: np.ndarray,
    n_steps: int,
    pad_before: int = ONSET_PAD_BEFORE,
    pad_after: int = ONSET_PAD_AFTER,
) -> tuple[int, int] | None:
    """Earliest/latest defensible onset step, per Week2_plan.md's operational
    definition: the first step after which no subsequent step shows progress
    toward the success condition. Seeded from the last grasp event (bowl
    approaching the plate) if one happened, else the last approach to the
    target (eef approaching the bowl) -- matching the plan's own wording.

    Returns None when there is nothing to seed from (no object signal at all).
    """
    if objects is None or n_steps <= 0:
        return None
    if objects.grasped_ever:
        progress = -np.linalg.norm(objects.bowl_positions - objects.plate_positions, axis=1)
    else:
        n = len(objects.bowl_positions)
        progress = -np.linalg.norm(eef_pos[:n] - objects.bowl_positions, axis=1)
    if progress.size == 0:
        return None
    # The last step that set a *new* running best, not the last step tied with
    # it -- once the bowl reaches the plate, distance floors at ~0 and every
    # later step is "tied" with the best without being a further improvement.
    # `np.diff(running_max) > 0` is exactly "did this step raise the ceiling".
    running_max = np.maximum.accumulate(progress)
    improved = np.diff(running_max) > 0
    last_best_step = int(np.nonzero(improved)[0][-1]) + 1 if improved.any() else 0
    start = max(0, last_best_step - pad_before)
    end = min(n_steps - 1, last_best_step + pad_after)
    return start, end
