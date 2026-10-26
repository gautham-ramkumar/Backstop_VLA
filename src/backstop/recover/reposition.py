"""Lift and back off: the scripted half of a recovery.

When the guard fires, the arm is usually dithering near the bowl (every
corpus failure is a stall, docs/recovery.md). The policy is a stateless
function of what it sees, so holding still changes nothing. This moves the
arm to where it sees the scene differently, then hands control back:

1. OPEN: open the gripper and hold position, so a half-held bowl is set
   down where it is, not dragged.
2. LIFT: straight up until the hand is `lift_m` above where it started.
3. BACK: towards the robot base (-x) until it is `back_m` behind the start,
   holding the lifted height.
4. SETTLE: hold still so the arm stops swinging before the policy looks.

A phase ends when the hand is within `tolerance_m` of its goal, or after
`max_steps_per_phase`, whichever comes first. Feedback is the hand position
in `observation.state`, which the policy already sees. It never uses sim
state (ADR-003), so the same motion would run on a real arm.

Action format, checked in the simulator (LIBERO relative OSC control):
7 numbers in [-1, 1]; position terms ask for up to 5 cm this step; +z is up
and -x is towards the robot base; the last number is the gripper, -1 open,
+1 closed. The arm lags the request (0.5 moves it ~6 mm per step), which is
why each phase closes the loop on the measured position instead of
counting steps.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from backstop.schema import ACTION_DIM, EEF_POS, GRIPPER_INDEX

GRIPPER_OPEN = -1.0
# The action's first three dimensions move the hand in xyz.
TRANSLATION = slice(0, 3)
METRES_PER_UNIT = 0.05  # OSC_POSE output_max for position


class Phase(Enum):
    OPEN = "open"
    LIFT = "lift"
    BACK = "back"
    SETTLE = "settle"


class RepositionConfig(BaseModel):
    lift_m: float = Field(default=0.10, ge=0.0)
    back_m: float = Field(default=0.08, ge=0.0)
    open_steps: int = Field(default=5, ge=0)
    settle_steps: int = Field(default=5, ge=0)
    max_steps_per_phase: int = Field(default=30, ge=1)
    max_action: float = Field(default=0.5, gt=0.0, le=1.0)
    tolerance_m: float = Field(default=0.01, gt=0.0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RepositionConfig:
        with Path(path).open() as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})

    @property
    def max_total_steps(self) -> int:
        """Upper bound on one repositioning, for budgeting the episode."""
        per_phase = self.max_steps_per_phase
        return min(self.open_steps, per_phase) + 2 * per_phase + min(self.settle_steps, per_phase)


class RepositionMotion:
    """Call `next_action(state)` once per control step until it returns None."""

    def __init__(self, config: RepositionConfig, start_state: np.ndarray) -> None:
        self.config = config
        start = np.asarray(start_state, dtype=np.float64)[EEF_POS]
        lifted = start + np.array([0.0, 0.0, config.lift_m])
        self._goals = {
            Phase.LIFT: lifted,
            Phase.BACK: lifted + np.array([-config.back_m, 0.0, 0.0]),
        }
        self._hold_steps = {Phase.OPEN: config.open_steps, Phase.SETTLE: config.settle_steps}
        self._phases = iter(Phase)
        self.phase: Phase | None = next(self._phases)
        self._steps_in_phase = 0

    def next_action(self, state: np.ndarray) -> np.ndarray | None:
        position = np.asarray(state, dtype=np.float64)[EEF_POS]
        while self.phase is not None and self._phase_done(position):
            self.phase = next(self._phases, None)
            self._steps_in_phase = 0
        if self.phase is None:
            return None
        self._steps_in_phase += 1
        action = np.zeros(ACTION_DIM)
        action[GRIPPER_INDEX] = GRIPPER_OPEN
        if self.phase in self._goals:
            error = self._goals[self.phase] - position
            limit = self.config.max_action
            action[TRANSLATION] = np.clip(error / METRES_PER_UNIT, -limit, limit)
        return action

    def _phase_done(self, position: np.ndarray) -> bool:
        assert self.phase is not None
        if self._steps_in_phase >= self.config.max_steps_per_phase:
            return True
        if self.phase in self._hold_steps:
            return self._steps_in_phase >= self._hold_steps[self.phase]
        distance = float(np.linalg.norm(self._goals[self.phase] - position))
        return distance <= self.config.tolerance_m
