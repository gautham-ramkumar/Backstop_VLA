"""A named list of pass/fail checks, printed as one table: the verify scripts' harness.

`scripts/verify_corpus.py` and `scripts/verify_sim_state_roundtrip.py` each run
dozens of independent checks and must report all of them, not stop at the first
failure, then exit non-zero if any failed. Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Check:
    ok: bool
    name: str
    detail: str = ""


@dataclass
class Checks:
    results: list[Check] = field(default_factory=list)

    def check(self, ok: bool, name: str, detail: str = "") -> None:
        self.results.append(Check(bool(ok), name, detail))

    @property
    def n_failed(self) -> int:
        return sum(not result.ok for result in self.results)

    def report(self, *, on_failure: str | None = None) -> int:
        """Print every check, then a total. Returns the exit code: 1 if any failed."""
        width = max((len(result.name) for result in self.results), default=0)
        for result in self.results:
            print(f"{'PASS' if result.ok else 'FAIL'}  {result.name.ljust(width)}  {result.detail}")
        print(f"\n{len(self.results) - self.n_failed}/{len(self.results)} checks passed")
        if self.n_failed and on_failure:
            print(f"\n{on_failure}")
        return 1 if self.n_failed else 0
