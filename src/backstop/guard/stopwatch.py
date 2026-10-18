"""Guard OR stopwatch: one stop rule, two triggers, one false-stop budget.

Every corpus-A failure runs to the 280-step cap, and successes finish by
about step 130. So "stop if the run is still going at step N" is a guard on
its own, and it catches every failure. The signals earn their place only by
stopping sooner than that clock.

Both triggers are set from good runs alone. The line is the k-th highest
peak score among calibration successes, moved up past any tie there
(`calibrate.upper_line`); the clock stops any run that goes longer than the
k-th longest calibration success. Strictly longer, so runs tied on a
whole-step length never push the rate above the bound either.
With k = floor(budget / 2 * (n + 1)), a new good run that is exchangeable
with the calibration ones crosses each
trigger with probability at most k / (n + 1), so the pair stays inside the
budget in expectation. No failure is used, no grid is searched, and so the
held-out rate does not drift above the budget the way a searched rule does.
The same arithmetic says how many good runs a new setting needs: with
k = 1 per trigger the bound is 2 / (n + 1), which reaches 5% at n = 39.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backstop.guard.calibrate import upper_line


@dataclass(frozen=True)
class StopwatchRule:
    line: float
    stop_step: int
    k: int
    n_calibration: int
    bound: float


def fit_stopwatch_rule(
    success_peaks: np.ndarray, success_lengths: np.ndarray, budget: float
) -> StopwatchRule:
    """Line and stop step from calibration successes only.

    `success_peaks` is each good run's highest per-step score;
    `success_lengths` is each good run's step count. A run of length L is
    stopped by the clock when L > stop_step.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    peaks = np.asarray(success_peaks, dtype=np.float64)
    lengths = np.asarray(success_lengths, dtype=np.int64)
    if peaks.shape != lengths.shape:
        raise ValueError(f"{peaks.shape} peaks and {lengths.shape} lengths")
    n = len(peaks)
    if n == 0:
        raise ValueError("no calibration successes to set a stopwatch rule from")
    k = max(1, math.floor(budget / 2 * (n + 1)))
    k = min(k, n)
    line = upper_line(peaks, k)
    stop_step = int(np.sort(lengths)[::-1][k - 1])
    return StopwatchRule(
        line=line,
        stop_step=stop_step,
        k=k,
        n_calibration=n,
        bound=2 * k / (n + 1),
    )


def stop_step(scores: np.ndarray, n_steps: int, rule: StopwatchRule) -> int | None:
    """First step the rule stops this run at, or None if it runs to the end.

    The signal fires on `score >= line`, matching `calibrate_threshold`. The
    clock fires at `rule.stop_step` if the run is still going there.
    """
    hits = np.flatnonzero(np.asarray(scores, dtype=np.float64) >= rule.line)
    signal = int(hits[0]) if hits.size else None
    clock = rule.stop_step if n_steps > rule.stop_step else None
    candidates = [step for step in (signal, clock) if step is not None]
    return min(candidates) if candidates else None
