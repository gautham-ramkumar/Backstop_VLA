"""Rule scoring shared by the evaluation scripts."""

from __future__ import annotations

import numpy as np

from backstop.corpus.episode import EpisodeKey
from backstop.guard.scoring import auroc, fit_stopwatch_from_rows
from backstop.guard.traces import TraceRow, series_map

SIGNAL = "action_chunk_disagreement"


def test_auroc_counts_ties_as_half_and_handles_empty() -> None:
    assert auroc([2.0, 3.0], [1.0]) == 1.0
    assert auroc([1.0], [2.0, 3.0]) == 0.0
    assert auroc([1.0], [1.0]) == 0.5
    assert auroc([], [1.0]) is None
    assert auroc(np.array([1.0]), np.array([])) is None


def _episode(index: int, success: bool, peak_at: int | None, n: int = 60) -> TraceRow:
    scores = np.zeros(n)
    if peak_at is not None:
        scores[peak_at:] = 1.0
    return TraceRow("clean", index, success, n, "task", {}, {SIGNAL: scores})


def test_fit_stopwatch_from_rows_ignores_failures() -> None:
    good = [_episode(i, True, None, n=100 + i) for i in range(5)]
    bad = [_episode(9, False, 0, n=280)]
    series = series_map(good + bad, SIGNAL)
    rule = fit_stopwatch_from_rows(good + bad, series, budget=0.05)
    assert rule.n_calibration == 5
    assert rule.stop_step == 104
    # All five good peaks tie at 0, so the line sits just above them:
    # flagging on >= 0 would stop every good run, not k = 1.
    assert rule.line > 0.0
    assert all(series[EpisodeKey("clean", i)].max() < rule.line for i in range(5))
