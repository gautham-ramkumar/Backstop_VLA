"""Per-object presence gate, every line set from frames that show the object."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.presence import PresenceGate, fit_presence_line


def test_line_is_the_kth_lowest_present_score() -> None:
    scores = np.linspace(0.1, 0.5, 39)
    line = fit_presence_line(scores, budget=0.05)
    # floor(0.05 * 40) = 2
    assert line.k == 2
    assert line.line == pytest.approx(scores[1])
    assert line.bound == pytest.approx(2 / 40)


def test_small_reference_sets_still_get_one() -> None:
    line = fit_presence_line(np.array([0.4, 0.3, 0.5]), budget=0.05)
    assert line.k == 1
    assert line.line == 0.3
    assert line.bound == pytest.approx(0.25)


def test_each_object_is_judged_against_its_own_line() -> None:
    gate = PresenceGate.fit(
        {"black bowl": np.array([0.40, 0.45, 0.50]), "tomato sauce": np.array([0.01, 0.02])},
        budget=0.05,
    )
    # 0.02 is a confident tomato sauce and a missing bowl.
    assert gate.check("tomato sauce", 0.02) == "present"
    assert gate.check("black bowl", 0.02) == "absent"


def test_reference_frames_never_fail_their_own_line() -> None:
    scores = np.array([0.33, 0.41, 0.47])
    gate = PresenceGate.fit({"moka pot": scores}, budget=0.05)
    assert all(gate.check("moka pot", float(s)) == "present" for s in scores)


def test_an_object_without_references_is_unverified_not_guessed() -> None:
    gate = PresenceGate.fit({"black bowl": np.array([0.4])}, budget=0.05)
    assert gate.check("moka pot", 0.0) == "unverified"
    assert gate.check("moka pot", 0.9) == "unverified"


def test_held_out_present_frames_fail_at_the_bound_on_average() -> None:
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(2000):
        scores = rng.beta(5, 3, size=60)
        line = fit_presence_line(scores[:19], budget=0.05)
        rates.append(np.mean(scores[19:] < line.line))
    # With no ties the rate is exactly k / (n + 1) = 1/20 in expectation, so
    # allow Monte Carlo noise around it, and check it is not far below either.
    assert 0.04 <= np.mean(rates) <= 0.055


def test_rejects_empty_references_and_bad_budget() -> None:
    with pytest.raises(ValueError):
        fit_presence_line(np.array([]), budget=0.05)
    with pytest.raises(ValueError):
        fit_presence_line(np.array([0.3]), budget=1.0)
