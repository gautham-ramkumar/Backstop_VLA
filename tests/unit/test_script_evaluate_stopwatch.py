"""scripts/evaluate_stopwatch.py: split summaries count against the budget given."""

from __future__ import annotations

import evaluate_stopwatch


def _runs(rates: list[float]) -> list[dict[str, float]]:
    return [{"false_stop_rate": rate, "detection_rate": 1.0} for rate in rates]


def test_splits_over_budget_uses_the_budget_passed_in() -> None:
    runs = _runs([0.03, 0.08, 0.15, 0.25])
    assert evaluate_stopwatch._summary(runs, budget=0.05)["splits_over_budget"] == 3
    # Was hard-coded to 5%: a 20% run reported 3 splits over budget, not 1.
    assert evaluate_stopwatch._summary(runs, budget=0.20)["splits_over_budget"] == 1


def test_a_rate_exactly_at_the_budget_is_not_over_it() -> None:
    assert evaluate_stopwatch._summary(_runs([0.05, 0.05]), budget=0.05)["splits_over_budget"] == 0
