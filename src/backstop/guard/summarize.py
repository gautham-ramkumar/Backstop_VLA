"""Turn per-episode scores into a guard report.

One holdout split, a bootstrap interval on that split, and stratified
k-fold cross-validation. The threshold for a fold is fit on that fold's
calibration successes only. Lead time is left unset: it needs onset ranges
from the taxonomy, which this module does not read.
"""

from __future__ import annotations

from collections.abc import Sequence, Set

import numpy as np

from backstop.corpus.episode import EpisodeKey
from backstop.guard.calibrate import (
    bootstrap_rates,
    calibrate_threshold,
    detection_curve,
    evaluate,
)
from backstop.guard.split import stratified_kfold, stratified_split
from backstop.guard.traces import TraceRow
from backstop.schema import MetricContract

CURVE_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
LEAD_TIME_NOTE = (
    "mean_lead_time_steps is not computed. It needs an onset range on each "
    "failure, and that review is still in progress."
)


def _score_array(
    episodes: Sequence[TraceRow],
    name: str,
    *,
    keys: Set[EpisodeKey] | None,
    success: bool,
) -> np.ndarray:
    selected = [
        episode.scores[name]
        for episode in episodes
        if episode.success == success and (keys is None or episode.key in keys)
    ]
    return np.asarray(selected, dtype=np.float64)


def _per_family(
    episodes: Sequence[TraceRow],
    name: str,
    eval_keys: Set[EpisodeKey],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    table: dict[str, dict[str, float | int]] = {}
    for shard in sorted({episode.shard for episode in episodes}):
        shard_eval = [
            episode for episode in episodes if episode.shard == shard and episode.key in eval_keys
        ]
        metrics = evaluate(
            _score_array(shard_eval, name, keys=None, success=True),
            _score_array(shard_eval, name, keys=None, success=False),
            threshold,
        )
        table[shard] = {
            "n_eval_success": metrics.n_eval_success,
            "n_eval_failure": metrics.n_eval_failure,
            "false_stop_rate": metrics.false_stop_rate,
            "detection_rate": metrics.detection_rate,
        }
    return table


def _cross_validation(
    episodes: Sequence[TraceRow],
    name: str,
    *,
    budget: float,
    n_folds: int,
    seed: int,
) -> dict[str, float | list[float]]:
    splits = stratified_kfold(
        [(episode.shard, episode.episode_index, episode.success) for episode in episodes],
        n_folds=n_folds,
        seed=seed,
    )
    rates: list[float] = []
    for split in splits:
        calib_keys, eval_keys = split.calibration, split.eval
        calib_success = _score_array(episodes, name, keys=calib_keys, success=True)
        if calib_success.size == 0 or not split.eval:
            continue
        threshold = calibrate_threshold(calib_success, budget)
        metrics = evaluate(
            _score_array(episodes, name, keys=eval_keys, success=True),
            _score_array(episodes, name, keys=eval_keys, success=False),
            threshold,
        )
        rates.append(metrics.detection_rate)
    if not rates:
        return {"n_folds": 0, "detection_rate_mean": 0.0, "detection_rate_std": 0.0, "per_fold": []}
    array = np.asarray(rates, dtype=np.float64)
    return {
        "n_folds": len(rates),
        "detection_rate_mean": float(array.mean()),
        "detection_rate_std": float(array.std()),
        "per_fold": [float(rate) for rate in rates],
    }


def report_signal(
    episodes: Sequence[TraceRow],
    name: str,
    *,
    metric: MetricContract,
    eval_fraction: float,
    seed: int,
    n_folds: int = 5,
    n_resamples: int = 1000,
) -> dict[str, object]:
    """Headline numbers for one signal. `episodes` excludes quarantined shards."""
    split = stratified_split(
        [(episode.shard, episode.episode_index, episode.success) for episode in episodes],
        eval_fraction=eval_fraction,
        seed=seed,
    )
    calib_keys, eval_keys = split.calibration, split.eval
    calib_success = _score_array(episodes, name, keys=calib_keys, success=True)
    eval_success = _score_array(episodes, name, keys=eval_keys, success=True)
    eval_failure = _score_array(episodes, name, keys=eval_keys, success=False)
    threshold = calibrate_threshold(calib_success, budget=metric.false_stop_budget)
    metrics = evaluate(eval_success, eval_failure, threshold)
    return {
        "signal": name,
        "false_stop_budget": metric.false_stop_budget,
        "threshold": metrics.threshold,
        "detection_rate": metrics.detection_rate,
        "false_stop_rate": metrics.false_stop_rate,
        "n_eval_success": metrics.n_eval_success,
        "n_eval_failure": metrics.n_eval_failure,
        "n_detected": metrics.n_detected,
        "n_false_stops": metrics.n_false_stops,
        "n_calibration_success": int(calib_success.size),
        "split_seed": seed,
        "eval_fraction": eval_fraction,
        "per_family": _per_family(episodes, name, eval_keys, threshold),
        "detection_curve": detection_curve(
            calib_success, eval_success, eval_failure, CURVE_BUDGETS
        ),
        "bootstrap": bootstrap_rates(
            eval_success, eval_failure, threshold, n_resamples=n_resamples, seed=seed
        ),
        "cross_validation": _cross_validation(
            episodes, name, budget=metric.false_stop_budget, n_folds=n_folds, seed=seed
        ),
        "mean_lead_time_steps": None,
        "notes": LEAD_TIME_NOTE,
    }
