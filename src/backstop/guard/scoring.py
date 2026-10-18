"""Scoring a stop rule on trace rows: the numbers every evaluation reports.

Separate from `traces.py` (reading and splitting rows) so the metric code can
be read on its own. Moved here unchanged from the evaluation scripts, where
each kept a private copy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from backstop.guard.persist import PersistenceRule, first_persistent_step, select_persistence_rule
from backstop.guard.stopwatch import StopwatchRule, fit_stopwatch_rule
from backstop.guard.traces import Key, Row, episode_key, onset_range


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


def lead_vs_onset(flag_step: int, onset: tuple[int, int]) -> dict[str, float | str]:
    """How far ahead of the reviewed onset window a stop came, and where it fell."""
    start, end = onset
    if flag_step < start:
        place = "before"
    elif flag_step <= end:
        place = "inside"
    else:
        place = "after"
    return {
        "versus_onset_start": float(start - flag_step),
        "versus_onset_end": float(end - flag_step),
        "place": place,
    }


def evaluate_persistence_rule(
    rows: list[Row],
    series_of: dict[Key, np.ndarray],
    labels: dict[Key, Row],
    threshold: float,
    hold: int,
) -> dict[str, object]:
    """False stops on successes, detection and lead time on failures."""
    successes = false_stops = failures = caught = missing_onset = 0
    leads: list[dict[str, float | str]] = []
    for row in rows:
        key = episode_key(row)
        flag = first_persistent_step(series_of[key], threshold, hold)
        if bool(row["success"]):
            successes += 1
            false_stops += int(flag is not None)
            continue
        failures += 1
        if flag is None:
            continue
        caught += 1
        onset = onset_range(labels, *key)
        if onset is None:
            missing_onset += 1
            continue
        leads.append(lead_vs_onset(flag, onset))

    def _stat(field: str, reduce: str) -> float | None:
        if not leads:
            return None
        values = np.array([float(item[field]) for item in leads])
        return float(np.mean(values) if reduce == "mean" else np.median(values))

    places = [str(item["place"]) for item in leads]
    return {
        "threshold": threshold,
        "hold": hold,
        "n_eval_success": successes,
        "n_eval_failure": failures,
        "false_stops": false_stops,
        "false_stop_rate": false_stops / successes if successes else None,
        "caught": caught,
        "detection_rate": caught / failures if failures else None,
        "caught_with_onset": len(leads),
        "missing_onset": missing_onset,
        "mean_lead_vs_onset_start": _stat("versus_onset_start", "mean"),
        "median_lead_vs_onset_start": _stat("versus_onset_start", "median"),
        "mean_lead_vs_onset_end": _stat("versus_onset_end", "mean"),
        "median_lead_vs_onset_end": _stat("versus_onset_end", "median"),
        "before_onset": places.count("before"),
        "inside_onset": places.count("inside"),
        "after_onset": places.count("after"),
    }


def fit_persistence_rule(
    calibration: list[Row], series_of: dict[Key, np.ndarray], budget: float
) -> PersistenceRule:
    """`select_persistence_rule` on the calibration rows' series."""
    pairs = [(series_of[episode_key(row)], bool(row["success"])) for row in calibration]
    return select_persistence_rule(pairs, budget=budget)


def fit_and_evaluate_persistence(
    calibration: list[Row],
    held_out: list[Row],
    series_of: dict[Key, np.ndarray],
    labels: dict[Key, Row],
    budget: float,
) -> dict[str, object]:
    """Fit on calibration, score on held-out: the rule's calibration numbers
    followed by `evaluate_persistence_rule`'s held-out ones."""
    rule = fit_persistence_rule(calibration, series_of, budget)
    return {
        "threshold": rule.threshold,
        "hold": rule.hold,
        "calibration_detection": rule.calibration_detection,
        "calibration_false_stop": rule.calibration_false_stop,
        **evaluate_persistence_rule(held_out, series_of, labels, rule.threshold, rule.hold),
    }


def fit_stopwatch_from_rows(
    rows: list[Row], series_of: dict[Key, np.ndarray], budget: float
) -> StopwatchRule:
    """`fit_stopwatch_rule` from the successes among `rows`; failures are ignored."""
    good = [row for row in rows if row["success"]]
    peaks = np.array([float(series_of[episode_key(row)].max()) for row in good])
    lengths = np.array([int(row["n_steps"]) for row in good])  # type: ignore[arg-type]
    return fit_stopwatch_rule(peaks, lengths, budget)


def format_rule_row(name: str, row: dict[str, object], *, width: int = 44) -> str:
    """One printed line for an `evaluate_persistence_rule` result."""
    lead_start = row["median_lead_vs_onset_start"]
    lead_end = row["median_lead_vs_onset_end"]
    start = "n/a" if lead_start is None else f"{float(lead_start):+.0f}"  # type: ignore[arg-type]
    end = "n/a" if lead_end is None else f"{float(lead_end):+.0f}"  # type: ignore[arg-type]
    return (
        f"{name:{width}} hold={int(row['hold']):2d} line={float(row['threshold']):7.3f} "  # type: ignore[arg-type]
        f"det={int(row['caught']):2d}/{int(row['n_eval_failure']):2d} "  # type: ignore[arg-type]
        f"fs={int(row['false_stops']):2d}/{int(row['n_eval_success']):2d} "  # type: ignore[arg-type]
        f"lead {start:>5}/{end:>5}  "
        f"before/in/after {row['before_onset']}/{row['inside_onset']}/{row['after_onset']}"
    )
