"""Guard-off against guard-on on the same episodes: the recovery experiment's numbers.

Each episode is run both ways from the same start state, so the comparison is
paired. What moves the success rate is only the discordant episodes: failures
the recovery turned into successes, and successes the guard spoiled. McNemar's
exact test asks whether that split could be chance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class PairedResult:
    n: int
    off_success: int
    on_success: int
    rescued: int  # failed with the guard off, succeeded with it on
    spoiled: int  # succeeded with the guard off, failed with it on
    p_value: float

    @property
    def success_rate_change(self) -> float:
        return (self.on_success - self.off_success) / self.n if self.n else 0.0


def mcnemar_exact(rescued: int, spoiled: int) -> float:
    """Two-sided exact p-value: the discordant pairs as fair coin flips."""
    n = rescued + spoiled
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(rescued, spoiled) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_result(off: Sequence[bool], on: Sequence[bool]) -> PairedResult:
    if len(off) != len(on):
        raise ValueError(f"{len(off)} guard-off outcomes for {len(on)} guard-on outcomes")
    rescued = sum(not a and b for a, b in zip(off, on, strict=True))
    spoiled = sum(a and not b for a, b in zip(off, on, strict=True))
    return PairedResult(
        n=len(off),
        off_success=sum(off),
        on_success=sum(on),
        rescued=rescued,
        spoiled=spoiled,
        p_value=mcnemar_exact(rescued, spoiled),
    )
