"""Rule scoring shared by the evaluation scripts."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.scoring import (
    auroc,
    evaluate_persistence_rule,
    fit_and_evaluate_persistence,
    fit_stopwatch_from_rows,
    format_rule_row,
    lead_vs_onset,
)


def test_auroc_counts_ties_as_half_and_handles_empty() -> None:
    assert auroc([2.0, 3.0], [1.0]) == 1.0
    assert auroc([1.0], [2.0, 3.0]) == 0.0
    assert auroc([1.0], [1.0]) == 0.5
    assert auroc([], [1.0]) is None
    assert auroc(np.array([1.0]), np.array([])) is None


@pytest.mark.parametrize(
    ("step", "place", "lead_start"),
    [(20, "before", 10.0), (30, "inside", 0.0), (40, "inside", -10.0), (41, "after", -11.0)],
)
def test_lead_vs_onset(step: int, place: str, lead_start: float) -> None:
    lead = lead_vs_onset(step, (30, 40))
    assert lead["place"] == place
    assert lead["versus_onset_start"] == lead_start
    assert lead["versus_onset_end"] == 40.0 - step


def _episode(
    index: int, success: bool, peak_at: int | None, n: int = 60
) -> tuple[dict, np.ndarray]:
    row = {"shard": "clean", "episode_index": index, "success": success, "n_steps": n}
    scores = np.zeros(n)
    if peak_at is not None:
        scores[peak_at:] = 1.0
    return row, scores


def test_evaluate_persistence_rule_counts_and_leads() -> None:
    episodes = [
        _episode(0, True, None),
        _episode(1, True, 10),  # a false stop
        _episode(2, False, 20),  # caught, onset known
        _episode(3, False, 25),  # caught, onset missing
        _episode(4, False, None),  # missed
    ]
    rows = [row for row, _ in episodes]
    series = {("clean", row["episode_index"]): s for row, s in episodes}
    labels: dict[tuple[str, int], dict[str, object]] = {
        ("clean", 2): {"review": {"onset_range": [30, 40]}}
    }
    out = evaluate_persistence_rule(rows, series, labels, threshold=0.5, hold=3)
    assert (out["n_eval_success"], out["false_stops"]) == (2, 1)
    assert (out["n_eval_failure"], out["caught"], out["missing_onset"]) == (3, 2, 1)
    # Flags on the 3rd consecutive step above the line: step 22 for a rise at 20.
    assert out["median_lead_vs_onset_start"] == 8.0
    assert out["before_onset"] == 1
    assert "hold= 3" in format_rule_row("x", out)


def test_fit_and_evaluate_reports_the_rule_it_fit() -> None:
    episodes = [_episode(i, True, None) for i in range(10)] + [_episode(10, False, 5)]
    rows = [row for row, _ in episodes]
    series = {("clean", row["episode_index"]): s for row, s in episodes}
    out = fit_and_evaluate_persistence(rows, rows, series, {}, budget=0.05)
    assert out["false_stops"] == 0 and out["caught"] == 1
    assert out["calibration_false_stop"] == 0.0


def test_fit_stopwatch_from_rows_ignores_failures() -> None:
    good = [_episode(i, True, None, n=100 + i) for i in range(5)]
    bad = [_episode(9, False, 0, n=280)]
    rows = [row for row, _ in good + bad]
    series = {("clean", row["episode_index"]): s for row, s in good + bad}
    rule = fit_stopwatch_from_rows(rows, series, budget=0.05)
    assert rule.n_calibration == 5
    assert rule.stop_step == 104
    # All five good peaks tie at 0, so the line sits just above them:
    # flagging on >= 0 would stop every good run, not k = 1.
    assert rule.line > 0.0
    assert all(series[("clean", i)].max() < rule.line for i in range(5))
