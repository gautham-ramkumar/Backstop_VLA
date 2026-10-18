"""Stop only when a score stays high.

A single crossing of a high line fires after a failure is already stuck:
successes spike briefly, so the 5% false-stop line sits above those spikes,
and failures reach it during the stall. Requiring the score to remain at or
above a lower line for several consecutive steps ignores the spike.

The rule is chosen on the calibration split only. Onset ranges are not an
input. The reference is the old stop: hold of one step, line set by
`calibrate_threshold` on each success's peak. With `n` calibration successes,
at most `k = ceil(budget * n)` may be flagged. A longer hold may replace that
line only when it stays inside the same count and gives up at most 5
percentage points of calibration detection. Among those, the earliest median
stop step wins. If none is earlier, the single crossing stands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backstop.guard.calibrate import calibrate_threshold

HOLDS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40)
DETECTION_TOLERANCE = 0.05


def first_persistent_step(scores: np.ndarray, threshold: float, hold: int) -> int | None:
    """First step at which `hold` consecutive scores are all >= threshold.

    The returned index is the step the hold completes, which is when a live
    guard would stop. `hold=1` is the old rule: the first crossing.
    """
    if hold < 1:
        raise ValueError(f"hold must be >= 1, got {hold}")
    above = np.asarray(scores, dtype=np.float64) >= threshold
    if hold == 1:
        hits = np.flatnonzero(above)
        return int(hits[0]) if hits.size else None
    if above.size < hold:
        return None
    window = np.convolve(above.astype(np.int8), np.ones(hold, dtype=np.int8), mode="valid")
    hits = np.flatnonzero(window >= hold)
    if hits.size == 0:
        return None
    return int(hits[0] + hold - 1)


@dataclass(frozen=True)
class PersistenceRule:
    threshold: float
    hold: int
    calibration_detection: float
    calibration_false_stop: float
    calibration_median_flag_step: float


def _flag_steps(
    episodes: list[tuple[np.ndarray, bool]], threshold: float, hold: int
) -> tuple[np.ndarray, np.ndarray]:
    """(success_flagged bool array, failure_flag_steps with -1 if uncaught)."""
    success_flags: list[bool] = []
    failure_steps: list[int] = []
    for scores, success in episodes:
        step = first_persistent_step(scores, threshold, hold)
        if success:
            success_flags.append(step is not None)
        else:
            failure_steps.append(-1 if step is None else step)
    return np.asarray(success_flags, dtype=bool), np.asarray(failure_steps, dtype=np.int32)


def select_persistence_rule(
    episodes: list[tuple[np.ndarray, bool]],
    *,
    budget: float = 0.05,
    holds: tuple[int, ...] = HOLDS,
    thresholds: np.ndarray | None = None,
) -> PersistenceRule:
    """Pick `(threshold, hold)` from calibration episodes only.

    `episodes` is `(per_step_scores, success)`. Failures are used only to
    count how many are caught and how early the stop falls, never an onset.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    successes = [scores for scores, success in episodes if success]
    failures = [scores for scores, success in episodes if not success]
    if not successes:
        raise ValueError("no calibration successes to set a persistence rule from")
    if not failures:
        raise ValueError("no calibration failures to choose among persistence rules")
    k = math.ceil(budget * len(successes))
    peaks = np.array(
        [float(np.max(scores)) if len(scores) else 0.0 for scores in successes],
        dtype=np.float64,
    )
    reference = float(calibrate_threshold(peaks, budget))
    if thresholds is None:
        grid = _threshold_grid(successes)
    else:
        grid = np.asarray(thresholds, dtype=np.float64)
    grid = np.unique(np.concatenate([grid, [reference]]))

    def _metrics(threshold: float, hold: int) -> tuple[int, int, float]:
        success_flags, failure_steps = _flag_steps(episodes, threshold, hold)
        caught = failure_steps[failure_steps >= 0]
        median_step = float(np.median(caught)) if caught.size else math.inf
        return int(success_flags.sum()), int(caught.size), median_step

    ref_false, ref_caught, ref_median = _metrics(reference, 1)
    min_caught = (ref_caught / len(failures) - DETECTION_TOLERANCE) * len(failures)
    min_caught = max(math.floor(min_caught), 0)
    # median step, then higher detection, then shorter hold, then the line.
    # Ties at the reference line can flag more than k successes; that line is
    # then not a legal start, and a longer hold has to win on its own.
    legal_reference = (ref_median, -ref_caught, 1, reference, ref_false)
    best: tuple[float, int, int, float, int] | None
    best = legal_reference if ref_false <= k else None
    for hold in holds:
        for threshold in grid:
            n_false, n_caught, median_step = _metrics(float(threshold), hold)
            if n_false > k or n_caught < min_caught:
                continue
            candidate = (median_step, -n_caught, hold, float(threshold), n_false)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("no persistence rule stayed inside the false-stop budget")
    median_step, neg_caught, hold, threshold, n_false = best
    return PersistenceRule(
        threshold=threshold,
        hold=hold,
        calibration_detection=float((-neg_caught) / len(failures)),
        calibration_false_stop=n_false / len(successes),
        calibration_median_flag_step=median_step,
    )


def _threshold_grid(successes: list[np.ndarray]) -> np.ndarray:
    """Lines from the middle of successful steps up to the highest success peak.

    Quantiles of the peaks alone sit on the old high line, so a lower line
    that only works together with a longer hold never enters the search.
    """
    steps = np.concatenate([np.asarray(scores, dtype=np.float64).ravel() for scores in successes])
    peaks = np.array([float(np.max(scores)) if len(scores) else 0.0 for scores in successes])
    lo = float(np.quantile(steps, 0.5))
    hi = float(np.max(peaks))
    if hi <= lo:
        hi = lo + 1e-3
    levels = np.linspace(max(lo, 0.0), hi, 61)
    return np.unique(np.round(levels, decimals=5))
