"""Contract for threshold calibration and the headline metric computation."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.calibrate import (
    bootstrap_rates,
    calibrate_threshold,
    detection_curve,
    evaluate,
    upper_line,
)


def test_threshold_admits_exactly_the_budgeted_fraction() -> None:
    # 100 calibration successes, scores 1..100. At a 5% budget, ceil(5)=5
    # may cross the line -- the threshold is the 5th-highest score, 96.
    scores = np.arange(1, 101, dtype=np.float64)
    threshold = calibrate_threshold(scores, budget=0.05)
    assert threshold == 96.0
    assert int(np.count_nonzero(scores >= threshold)) == 5


def test_matches_week2_plans_own_worked_example() -> None:
    # Week2_plan.md: "5% of ~75 successes lands on the 4th-worst one."
    # ceil(0.05 * 75) == 4.
    scores = np.arange(1, 76, dtype=np.float64)
    threshold = calibrate_threshold(scores, budget=0.05)
    assert int(np.count_nonzero(scores >= threshold)) == 4


def test_ceil_means_at_least_one_success_can_always_cross() -> None:
    # ceil() of any positive number is >= 1, so even a very tight budget on a
    # tiny calibration set still allows exactly one success through -- there
    # is no calibration set size or positive budget that flags nothing.
    scores = np.array([1.0, 2.0, 3.0])
    threshold = calibrate_threshold(scores, budget=0.01)
    assert int(np.count_nonzero(scores >= threshold)) == 1


def test_rejects_empty_calibration_set() -> None:
    with pytest.raises(ValueError, match="no calibration"):
        calibrate_threshold(np.array([]), budget=0.05)


def test_rejects_out_of_range_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        calibrate_threshold(np.array([1.0]), budget=1.5)


def test_evaluate_computes_rates_on_the_held_out_half() -> None:
    success = np.array([0.1, 0.2, 0.3, 0.4, 0.9])  # one (0.9) crosses
    failure = np.array([0.5, 0.6, 0.05])  # two (0.5, 0.6) cross
    metrics = evaluate(success, failure, threshold=0.5)
    assert metrics.n_false_stops == 1
    assert metrics.false_stop_rate == pytest.approx(0.2)
    assert metrics.n_detected == 2
    assert metrics.detection_rate == pytest.approx(2 / 3)


def test_evaluate_boundary_is_inclusive() -> None:
    metrics = evaluate(np.array([0.5]), np.array([0.5]), threshold=0.5)
    assert metrics.n_false_stops == 1
    assert metrics.n_detected == 1


def test_evaluate_handles_empty_half() -> None:
    metrics = evaluate(np.array([]), np.array([1.0, 2.0]), threshold=0.5)
    assert metrics.false_stop_rate == 0.0
    assert metrics.n_eval_success == 0


def test_detection_curve_is_monotonic_nondecreasing_in_budget() -> None:
    rng = np.random.default_rng(0)
    calib_success = rng.normal(0, 1, 200)
    eval_success = rng.normal(0, 1, 100)
    eval_failure = rng.normal(2, 1, 100)  # separable from success by construction
    curve = detection_curve(
        calib_success, eval_success, eval_failure, budgets=[0.01, 0.05, 0.1, 0.2, 0.5]
    )
    rates = [point["detection_rate"] for point in curve]
    assert rates == sorted(rates)


def test_detection_curve_reports_the_budget_alongside_each_point() -> None:
    calib_success = np.arange(1, 21, dtype=np.float64)
    eval_success = np.arange(1, 21, dtype=np.float64)
    eval_failure = np.arange(15, 35, dtype=np.float64)
    curve = detection_curve(calib_success, eval_success, eval_failure, budgets=[0.05, 0.2])
    assert [p["budget"] for p in curve] == [0.05, 0.2]
    assert all("threshold" in p and "false_stop_rate" in p for p in curve)


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    success = np.array([0.1, 0.2, 0.3, 0.9])
    failure = np.array([0.0, 0.8, 0.7, 0.6])
    metrics = evaluate(success, failure, threshold=0.5)
    interval = bootstrap_rates(success, failure, threshold=0.5, n_resamples=200, seed=0)
    low = interval["detection_rate_low"]
    high = interval["detection_rate_high"]
    assert low <= metrics.detection_rate <= high
    fs_low = interval["false_stop_rate_low"]
    fs_high = interval["false_stop_rate_high"]
    assert fs_low <= metrics.false_stop_rate <= fs_high


def test_upper_line_is_the_kth_highest_when_there_is_no_tie() -> None:
    assert upper_line(np.array([0.1, 0.9, 0.5, 0.7]), k=2) == 0.7


def test_upper_line_moves_past_a_tie_at_the_cutoff() -> None:
    # Three runs tie at 0.5; flagging on >= 0.5 would stop 4 runs when k=2.
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    line = upper_line(scores, k=2)
    assert line == 0.9
    assert (scores >= line).sum() <= 2


def test_upper_line_goes_above_the_max_when_the_tie_reaches_the_top() -> None:
    scores = np.zeros(10)
    line = upper_line(scores, k=1)
    assert line > 0.0
    assert (scores >= line).sum() == 0


def test_calibrate_threshold_never_flags_more_than_k_with_ties() -> None:
    scores = np.array([0.2] * 30 + [0.1] * 70)  # 100 successes, 30 tied at the top
    line = calibrate_threshold(scores, budget=0.05)  # k = 5
    assert (scores >= line).sum() <= 5


@pytest.mark.parametrize("k", [0, 4])
def test_upper_line_rejects_k_out_of_range(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        upper_line(np.array([0.1, 0.2, 0.3]), k=k)
