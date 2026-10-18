"""Is a run's success rate inside the expected band, and what should the CLI exit with?

One rule for `run_pipeline`, `backstop-eval` and `backstop-record`. They used
to compute it three ways: the pipeline skipped resumed runs, `backstop-eval`
checked only the lower bound, and `backstop-record` checked both bounds but
ignored `tolerance.enforce`.

- A resumed run measured only the tail of a shard, so it is not judged
  (verdict None). Verify the completed shard with scripts/verify_corpus.py.
- Otherwise the verdict is min <= rate <= max.
- Out of band exits 2 when `enforce` is on, else 0 with the message printed.
"""

from __future__ import annotations

from dataclasses import dataclass

from backstop.config import ToleranceConfig

EXIT_OUT_OF_TOLERANCE = 2


def tolerance_verdict(success_rate: float, *, partial: bool, tol: ToleranceConfig) -> bool | None:
    if partial:
        return None
    return tol.min_success_rate <= success_rate <= tol.max_success_rate


@dataclass(frozen=True)
class CliOutcome:
    exit_code: int
    message: str | None


def cli_outcome(success_rate: float, verdict: bool | None, tol: ToleranceConfig) -> CliOutcome:
    if verdict is None:
        return CliOutcome(0, "Resumed run: tolerance not judged; verify the completed shard.")
    if verdict:
        return CliOutcome(0, None)
    message = f"OUT OF TOLERANCE [{tol.min_success_rate}, {tol.max_success_rate}]."
    if success_rate < tol.min_success_rate:
        message += (
            f" Retry with policy.num_steps={tol.fallback_num_steps} (docs/adr/002-checkpoint.md)."
        )
    return CliOutcome(EXIT_OUT_OF_TOLERANCE if tol.enforce else 0, message)
