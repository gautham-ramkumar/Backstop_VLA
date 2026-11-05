"""When to recover, and when to give up: the rules of a guarded run.

The guard watches every policy step. When it fires, the run either recovers
(the lift-and-back-off motion, then the policy again) or, once the attempt
budget is spent, halts as the plain guard would. Three rules, all here:

- **Attempt budget.** At most `max_attempts` recoveries per episode. Unbounded
  retry turns "fails" into "never ends" (docs/recovery.md).
- **Cooldown.** After a recovery the signal is ignored for `cooldown_steps`, so
  the guard does not fire straight back on a view much like the one it just
  fired on.
- **Stopwatch restart.** The clock counts from the end of the latest recovery,
  not from step 0: each attempt gets the time a fresh run would.

No simulator here. `backstop.record.recovery` steps the env and asks this what
to do, so these rules are tested on fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from backstop.guard.online import OnlineGuard
from backstop.guard.stopwatch import StopwatchRule
from backstop.recover.reposition import RepositionConfig, RepositionMotion


class RecoveryConfig(BaseModel):
    max_attempts: int = Field(default=2, ge=0)
    cooldown_steps: int = Field(default=20, ge=0)
    reposition: RepositionConfig = Field(default_factory=RepositionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecoveryConfig:
        with Path(path).open() as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})


class Verdict(Enum):
    ACT = "act"  # execute the policy's action
    RECOVER = "recover"  # a recovery starts at this step
    HALT = "halt"  # the guard fired with no attempts left: stop the run


@dataclass
class Attempt:
    start: int
    reason: str
    end: int | None = None


class RecoveryController:
    def __init__(self, rule: StopwatchRule, config: RecoveryConfig) -> None:
        self.config = config
        self.guard = OnlineGuard(rule, cooldown=config.cooldown_steps)
        self.attempts: list[Attempt] = []
        self.halted_at: int | None = None
        self._motion: RepositionMotion | None = None

    @property
    def recovering(self) -> bool:
        return self._motion is not None

    def after_policy(self, samples: np.ndarray, step: int, state: np.ndarray) -> Verdict:
        """The guard's call on this policy step's (K, chunk, dim) samples."""
        if not self.guard.check_samples(samples, step):
            return Verdict.ACT
        assert self.guard.fired_reason is not None
        return self.fired(step, state, self.guard.fired_reason)

    def fired(self, step: int, state: np.ndarray, reason: str) -> Verdict:
        """The guard fired at `step` with the hand at `state`: recover, or halt."""
        if len(self.attempts) >= self.config.max_attempts:
            self.halted_at = step
            return Verdict.HALT
        self.attempts.append(Attempt(step, reason))
        self._motion = RepositionMotion(self.config.reposition, state)
        return Verdict.RECOVER

    def motion_action(self, state: np.ndarray, step: int) -> np.ndarray | None:
        """The recovery's next action, or None once it is over.

        On None the guard is re-armed at `step`, so the policy, reset by the
        caller, starts a fresh attempt there.
        """
        assert self._motion is not None, "motion_action outside a recovery"
        action = self._motion.next_action(state)
        if action is None:
            self._motion = None
            self.attempts[-1].end = step
            self.guard.rearm(step)
        return action
