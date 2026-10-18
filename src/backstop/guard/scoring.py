"""Scoring helpers shared by the evaluation scripts: AUROC, and fitting the
headline stopwatch rule from trace rows. Reading and splitting rows is in
`traces.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from backstop.guard.stopwatch import StopwatchRule, fit_stopwatch_rule
from backstop.guard.traces import Key, Row, episode_key


def auroc(
    positive: Sequence[float] | np.ndarray, negative: Sequence[float] | np.ndarray
) -> float | None:
    """P(positive scores above negative), ties count half. None if either is empty."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not pos.size or not neg.size:
        return None
    pos, neg = pos[:, None], neg[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def fit_stopwatch_from_rows(
    rows: list[Row], series_of: dict[Key, np.ndarray], budget: float
) -> StopwatchRule:
    """`fit_stopwatch_rule` from the successes among `rows`; failures are ignored."""
    good = [row for row in rows if row["success"]]
    peaks = np.array([float(series_of[episode_key(row)].max()) for row in good])
    lengths = np.array([int(row["n_steps"]) for row in good])  # type: ignore[arg-type]
    return fit_stopwatch_rule(peaks, lengths, budget)
