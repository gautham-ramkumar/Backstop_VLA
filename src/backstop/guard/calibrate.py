"""Threshold calibration and the headline metric itself (ADR-001, ADR-004).

Calibrate on one half, measure on the other -- never both on the same
episodes. Calibrating and reporting false-stop rate on the *same* data would
be close to tautological (the threshold is chosen to hit ~5% there by
construction); measuring on a held-out half is what makes the reported
false-stop rate and detection rate honest numbers rather than a restatement
of the calibration target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def upper_line(scores: np.ndarray, k: int) -> float:
    """Lowest line with at most `k` of `scores` at or above it (flag on `>=`).

    Normally the `k`-th highest score. When scores tie at that value, `>=`
    would flag every tied run and break the "at most k" promise, so the line
    moves up to the next distinct score -- or just above the maximum when the
    tie reaches the top. Every false-stop guarantee here rests on this.
    """
    values = np.asarray(scores, dtype=np.float64)
    if not 1 <= k <= len(values):
        raise ValueError(f"k must be in [1, {len(values)}], got {k}")
    line = float(np.sort(values)[::-1][k - 1])
    if int((values >= line).sum()) <= k:
        return line
    above = values[values > line]
    return float(above.min()) if above.size else float(np.nextafter(line, np.inf))


def calibrate_threshold(calibration_success_scores: np.ndarray, budget: float) -> float:
    """Line such that at most `k = ceil(budget * n)` of the `n` calibration
    successes would be flagged (score >= line).

    Order statistic, not percentile interpolation (`upper_line`). `k` is
    rounded up, so when `budget * n` is not a whole number the calibration
    rate can sit just above `budget` (4 of 70 is 5.7% at a 5% budget).
    `stopwatch.fit_stopwatch_rule` rounds down instead and is the rule the
    headline uses. `k` is always >= 1, so at least one calibration success
    may cross: no positive budget gives a literal 0% false-stop guarantee on a
    finite calibration set, and this function does not pretend otherwise.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    n = len(calibration_success_scores)
    if n == 0:
        raise ValueError("no calibration successes to set a threshold from")
    return upper_line(np.asarray(calibration_success_scores), math.ceil(budget * n))


@dataclass(frozen=True)
class GuardMetrics:
    threshold: float
    false_stop_rate: float
    detection_rate: float
    n_eval_success: int
    n_eval_failure: int
    n_false_stops: int
    n_detected: int


def evaluate(
    eval_success_scores: np.ndarray, eval_failure_scores: np.ndarray, threshold: float
) -> GuardMetrics:
    """Apply a threshold already fixed by `calibrate_threshold` (on the
    *other* half) to this half's scores. Flags on `score >= threshold`,
    matching the calibration side's own boundary convention."""
    success = np.asarray(eval_success_scores, dtype=np.float64)
    failure = np.asarray(eval_failure_scores, dtype=np.float64)
    n_false_stops = int(np.count_nonzero(success >= threshold))
    n_detected = int(np.count_nonzero(failure >= threshold))
    return GuardMetrics(
        threshold=threshold,
        false_stop_rate=n_false_stops / len(success) if len(success) else 0.0,
        detection_rate=n_detected / len(failure) if len(failure) else 0.0,
        n_eval_success=len(success),
        n_eval_failure=len(failure),
        n_false_stops=n_false_stops,
        n_detected=n_detected,
    )


def bootstrap_rates(
    eval_success_scores: np.ndarray,
    eval_failure_scores: np.ndarray,
    threshold: float,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile interval for detection and false-stop on the eval half.

    The threshold stays fixed. It was set on the calibration half; resampling
    it would mix the two questions "how stable is the threshold" and "how
    stable is the rate". Successes and failures are resampled separately so a
    draw keeps the eval set's class counts. With an empty class the interval
    is 0, matching `evaluate`.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    success = np.asarray(eval_success_scores, dtype=np.float64)
    failure = np.asarray(eval_failure_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    detection = np.empty(n_resamples, dtype=np.float64)
    false_stop = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        drawn_success = (
            rng.choice(success, size=success.size, replace=True) if success.size else success
        )
        drawn_failure = (
            rng.choice(failure, size=failure.size, replace=True) if failure.size else failure
        )
        metrics = evaluate(drawn_success, drawn_failure, threshold)
        detection[i] = metrics.detection_rate
        false_stop[i] = metrics.false_stop_rate
    return {
        "n_resamples": float(n_resamples),
        "detection_rate_mean": float(detection.mean()),
        "detection_rate_low": float(np.percentile(detection, 2.5)),
        "detection_rate_high": float(np.percentile(detection, 97.5)),
        "false_stop_rate_low": float(np.percentile(false_stop, 2.5)),
        "false_stop_rate_high": float(np.percentile(false_stop, 97.5)),
    }


def detection_curve(
    calibration_success_scores: np.ndarray,
    eval_success_scores: np.ndarray,
    eval_failure_scores: np.ndarray,
    budgets: list[float],
) -> list[dict[str, float]]:
    """Detection rate at a sweep of false-stop budgets, each threshold
    calibrated independently on the calibration half and measured on the
    eval half -- `configs/metric.yaml`'s `detection_curve` secondary metric.
    """
    curve = []
    for budget in budgets:
        threshold = calibrate_threshold(calibration_success_scores, budget)
        metrics = evaluate(eval_success_scores, eval_failure_scores, threshold)
        curve.append(
            {
                "budget": budget,
                "threshold": metrics.threshold,
                "false_stop_rate": metrics.false_stop_rate,
                "detection_rate": metrics.detection_rate,
            }
        )
    return curve
