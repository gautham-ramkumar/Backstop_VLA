"""Visual precondition as a per-object gate: is the instructed object there?

An open-vocabulary detector (`vision.detector_scores`) gives each first frame
a score for the instructed object. That score is not comparable across object
names: a real tomato-sauce can scores 0.01 while a lookalike of a moka pot
scores 0.3. So no single line works, and every object gets its own, set from
frames where that object is known to be present.

The line is the k-th lowest present score, with k = floor(budget * (n + 1)).
A new frame that shows the object, exchangeable with the reference frames,
falls below it with probability at most k / (n + 1). So the gate's false-stop
rate is bounded by the reference set alone, without ever seeing the object
missing. With k = 1 the bound reaches 5% at n = 19 reference frames.

An object with no reference frames is `unverified`: the gate does not guess
in either direction, because a line borrowed from another object means
nothing (see the module docstring's tomato-sauce example).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Verdict = Literal["present", "absent", "unverified"]


@dataclass(frozen=True)
class PresenceLine:
    line: float
    k: int
    n_reference: int
    bound: float


def fit_presence_line(present_scores: np.ndarray, budget: float) -> PresenceLine:
    """Line below which the object is called absent, from present frames only."""
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    scores = np.sort(np.asarray(present_scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        raise ValueError("no reference frames to set a presence line from")
    k = min(max(1, math.floor(budget * (n + 1))), n)
    return PresenceLine(line=float(scores[k - 1]), k=k, n_reference=n, bound=k / (n + 1))


@dataclass(frozen=True)
class PresenceGate:
    lines: dict[str, PresenceLine]

    @classmethod
    def fit(cls, references: dict[str, np.ndarray], budget: float) -> PresenceGate:
        return cls({name: fit_presence_line(scores, budget) for name, scores in references.items()})

    def check(self, phrase: str, score: float) -> Verdict:
        """`absent` strictly below the line, so a reference frame never fails itself."""
        line = self.lines.get(phrase)
        if line is None:
            return "unverified"
        return "absent" if score < line.line else "present"
