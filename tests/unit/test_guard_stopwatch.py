"""Guard OR stopwatch, both triggers set from good runs only."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.stopwatch import fit_stopwatch_rule, stop_step


def test_k_is_half_the_budget_on_each_trigger() -> None:
    peaks = np.arange(162, dtype=np.float64)
    lengths = np.arange(100, 262)
    rule = fit_stopwatch_rule(peaks, lengths, budget=0.05)
    # floor(0.025 * 163) = 4
    assert rule.k == 4
    assert rule.line == 158.0
    # 4th longest is 258; only runs longer than that are stopped.
    assert rule.stop_step == 258
    assert rule.bound == pytest.approx(8 / 163)


def test_small_calibration_sets_still_get_one_per_trigger() -> None:
    rule = fit_stopwatch_rule(np.array([0.1, 0.5, 0.3]), np.array([90, 120, 100]), budget=0.05)
    assert rule.k == 1
    assert rule.line == 0.5
    assert rule.stop_step == 120
    assert rule.bound == pytest.approx(0.5)


def test_the_earlier_trigger_wins() -> None:
    rule = fit_stopwatch_rule(np.array([0.5]), np.array([101]), budget=0.05)
    scores = np.zeros(280)
    assert stop_step(scores, 280, rule) == 101
    scores[40] = 0.6
    assert stop_step(scores, 280, rule) == 40


def test_a_run_that_finishes_before_the_clock_and_stays_low_is_not_stopped() -> None:
    rule = fit_stopwatch_rule(np.array([0.5]), np.array([101]), budget=0.05)
    assert stop_step(np.zeros(101), 101, rule) is None
    assert stop_step(np.zeros(102), 102, rule) == 101


def test_ties_on_the_cutoff_length_are_not_stopped() -> None:
    lengths = np.array([100, 120, 120, 120, 90])
    rule = fit_stopwatch_rule(np.zeros(5), lengths, budget=0.05)
    assert rule.stop_step == 120
    assert all(stop_step(np.full(n, -1.0), int(n), rule) is None for n in lengths)


def test_held_out_false_stop_stays_inside_the_bound_on_average() -> None:
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(400):
        peaks = rng.normal(size=200)
        lengths = rng.integers(80, 140, size=200)
        rule = fit_stopwatch_rule(peaks[:100], lengths[:100], budget=0.05)
        stopped = [
            stop_step(np.array([p]), int(n), rule) is not None
            for p, n in zip(peaks[100:], lengths[100:], strict=True)
        ]
        rates.append(np.mean(stopped))
    assert np.mean(rates) <= 0.05


def test_rejects_empty_and_mismatched_input() -> None:
    with pytest.raises(ValueError):
        fit_stopwatch_rule(np.array([]), np.array([], dtype=int), budget=0.05)
    with pytest.raises(ValueError):
        fit_stopwatch_rule(np.array([0.1, 0.2]), np.array([100]), budget=0.05)
    with pytest.raises(ValueError):
        fit_stopwatch_rule(np.array([0.1]), np.array([100]), budget=0.0)


def test_tied_peaks_at_the_line_do_not_push_false_stops_past_k() -> None:
    peaks = np.array([0.4] * 10 + [0.1] * 50)  # k = floor(0.025 * 61) = 1, ten tied at the top
    lengths = np.full(len(peaks), 100)
    rule = fit_stopwatch_rule(peaks, lengths, budget=0.05)
    assert rule.k == 1
    assert (peaks >= rule.line).sum() <= rule.k
