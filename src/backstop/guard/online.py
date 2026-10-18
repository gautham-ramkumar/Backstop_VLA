"""The headline guard, one step at a time, for use inside a rollout.

`stopwatch.stop_step` scores a whole recorded episode. A recovery loop needs
the same decision as the run unfolds: here is this step, stop or not.
`OnlineGuard` is that, with no new signal logic. The score is
`action_chunk_disagreement` on one step, and the rule is a frozen
`StopwatchRule` (see `scripts/freeze_guard.py`). Fed a recorded episode step
by step it stops where `stop_step` does; `tests/unit/test_guard_online.py`
checks that.

Two things the offline rule never needed, both for recovery:

- `rearm(step)` starts a fresh attempt: the clock counts from `step` again,
  and the signal is ignored for `cooldown` steps so the guard does not fire
  straight back on a view much like the one it just fired on.
- `fired_reason` says which trigger fired.

Reads only the K action samples and the step count. No sim state (ADR-003).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from backstop.guard.signals import action_chunk_disagreement
from backstop.guard.stopwatch import StopwatchRule

Trigger = Literal["signal", "clock"]


def save_rule(rule: StopwatchRule, path: Path, **provenance: object) -> None:
    """Write `rule` plus whatever says where it came from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rule": asdict(rule), **provenance}, indent=2) + "\n")


def load_rule(path: Path) -> StopwatchRule:
    return StopwatchRule(**json.loads(Path(path).read_text())["rule"])


def step_disagreement(samples: np.ndarray) -> float:
    """(K, chunk, dim) for one step -> that step's disagreement score."""
    return float(action_chunk_disagreement(np.asarray(samples)[None])[0])


@dataclass
class OnlineGuard:
    rule: StopwatchRule
    cooldown: int = 0
    fired_reason: Trigger | None = field(default=None, init=False)
    _attempt_start: int = field(default=0, init=False, repr=False)
    _signal_muted_until: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {self.cooldown}")

    def check(self, score: float, step: int) -> bool:
        """True when the guard stops the run at `step` (0-based, absolute).

        Same order as `stop_step`: the signal first, then the clock. The
        clock fires once `rule.stop_step` steps have passed since the attempt
        began, i.e. at the step index `stop_step` would return.
        """
        self.fired_reason = None
        if step >= self._signal_muted_until and score >= self.rule.line:
            self.fired_reason = "signal"
        elif step - self._attempt_start >= self.rule.stop_step:
            self.fired_reason = "clock"
        return self.fired_reason is not None

    def check_samples(self, samples: np.ndarray, step: int) -> bool:
        """`check` on this step's (K, chunk, dim) action samples."""
        return self.check(step_disagreement(samples), step)

    def rearm(self, step: int) -> None:
        """Start a fresh attempt at `step`: restart the clock, mute the signal."""
        self._attempt_start = step
        self._signal_muted_until = step + self.cooldown
        self.fired_reason = None
