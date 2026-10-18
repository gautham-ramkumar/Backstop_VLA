"""A stop requires the score to stay high, not merely to spike."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.persist import first_persistent_step, select_persistence_rule


def test_hold_one_is_the_first_crossing() -> None:
    scores = np.array([0.0, 0.2, 0.0, 0.9])
    assert first_persistent_step(scores, threshold=0.5, hold=1) == 3


def test_a_one_step_spike_does_not_satisfy_a_longer_hold() -> None:
    scores = np.array([0.0, 0.9, 0.0, 0.2])
    assert first_persistent_step(scores, threshold=0.5, hold=2) is None


def test_stop_is_declared_when_the_hold_completes() -> None:
    scores = np.array([0.0, 0.6, 0.7, 0.8, 0.1])
    # Two consecutive steps at or above 0.5 complete at index 2.
    assert first_persistent_step(scores, threshold=0.5, hold=2) == 2
    assert first_persistent_step(scores, threshold=0.5, hold=3) == 3


def test_selection_stays_inside_the_false_stop_budget() -> None:
    rng = np.random.default_rng(0)
    episodes: list[tuple[np.ndarray, bool]] = []
    for _ in range(40):
        # Successes spike once, then sit low.
        scores = np.zeros(30)
        scores[rng.integers(0, 30)] = 1.0
        episodes.append((scores, True))
    for _ in range(20):
        # Failures rise at step 10 and stay up.
        scores = np.zeros(30)
        scores[10:] = 0.4
        episodes.append((scores, False))
    rule = select_persistence_rule(episodes, budget=0.05, thresholds=np.array([0.3, 0.9]))
    assert rule.calibration_false_stop <= 0.05 + 1e-9
    assert rule.hold > 1
    assert rule.threshold == pytest.approx(0.3)
