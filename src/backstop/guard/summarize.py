"""Turn per-episode scores into a guard report.

One holdout split, a bootstrap interval on that split, and stratified
k-fold cross-validation. The threshold for a fold is fit on that fold's
calibration successes only. Lead time is left unset: it needs onset ranges
from the taxonomy, which this module does not read.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from backstop.guard.calibrate import (
    bootstrap_rates,
    calibrate_threshold,
    detection_curve,
    evaluate,
)
from backstop.guard.combine import TrustHead, apply_trust_head, fit_trust_head
from backstop.guard.extract import EpisodeSignal
from backstop.guard.split import stratified_kfold, stratified_split
from backstop.schema import MetricContract

CURVE_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
LEAD_TIME_NOTE = (
    "mean_lead_time_steps is not computed. It needs an onset range on each "
    "failure, and that review is still in progress."
)


def _score_array(
    episodes: Sequence[EpisodeSignal],
    name: str,
    *,
    keys: set[tuple[str, int]] | None,
    success: bool,
) -> np.ndarray:
    selected = [
        episode.scores[name]
        for episode in episodes
        if episode.success == success
        and (keys is None or (episode.shard, episode.episode_index) in keys)
    ]
    return np.asarray(selected, dtype=np.float64)


def _per_family(
    episodes: Sequence[EpisodeSignal],
    name: str,
    eval_keys: set[tuple[str, int]],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    table: dict[str, dict[str, float | int]] = {}
    for shard in sorted({episode.shard for episode in episodes}):
        shard_eval = [
            episode
            for episode in episodes
            if episode.shard == shard and (episode.shard, episode.episode_index) in eval_keys
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
    episodes: Sequence[EpisodeSignal],
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
        calib_keys = {(key.shard, key.episode_index) for key in split.calibration}
        eval_keys = {(key.shard, key.episode_index) for key in split.eval}
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
    episodes: Sequence[EpisodeSignal],
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
    calib_keys = {(key.shard, key.episode_index) for key in split.calibration}
    eval_keys = {(key.shard, key.episode_index) for key in split.eval}
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


def _feature_matrix(episodes: Sequence[EpisodeSignal], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[episode.scores[name] for name in names] for episode in episodes],
        dtype=np.float64,
    )


def _cross_validate_trust(
    episodes: list[EpisodeSignal],
    feature_names: tuple[str, ...],
    *,
    budget: float,
    n_folds: int,
    seed: int,
) -> dict[str, float | list[float]]:
    """Refit the head on each fold. The eval fold never trains the head."""
    splits = stratified_kfold(
        [(episode.shard, episode.episode_index, episode.success) for episode in episodes],
        n_folds=n_folds,
        seed=seed,
    )
    rates: list[float] = []
    for split in splits:
        calib_keys = {(key.shard, key.episode_index) for key in split.calibration}
        eval_keys = {(key.shard, key.episode_index) for key in split.eval}
        calibration = [
            episode for episode in episodes if (episode.shard, episode.episode_index) in calib_keys
        ]
        evaluation = [
            episode for episode in episodes if (episode.shard, episode.episode_index) in eval_keys
        ]
        calib_failures = np.asarray([not episode.success for episode in calibration])
        if not calibration or not evaluation or not calib_failures.any():
            continue
        if not any(episode.success for episode in calibration):
            continue
        head = fit_trust_head(
            _feature_matrix(calibration, feature_names),
            calib_failures.astype(np.float64),
            feature_names,
        )
        calib_scores = apply_trust_head(head, _feature_matrix(calibration, feature_names))
        eval_scores = apply_trust_head(head, _feature_matrix(evaluation, feature_names))
        threshold = calibrate_threshold(
            calib_scores[np.asarray([episode.success for episode in calibration])],
            budget,
        )
        metrics = evaluate(
            eval_scores[np.asarray([episode.success for episode in evaluation])],
            eval_scores[np.asarray([not episode.success for episode in evaluation])],
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


def report_trust_head(
    episodes: Sequence[EpisodeSignal],
    feature_names: tuple[str, ...],
    *,
    metric: MetricContract,
    eval_fraction: float,
    seed: int,
    n_resamples: int = 1000,
) -> tuple[dict[str, object], TrustHead]:
    """Fit the head on the calibration half, measure it on the eval half."""
    split = stratified_split(
        [(episode.shard, episode.episode_index, episode.success) for episode in episodes],
        eval_fraction=eval_fraction,
        seed=seed,
    )
    calib_keys = {(key.shard, key.episode_index) for key in split.calibration}
    eval_keys = {(key.shard, key.episode_index) for key in split.eval}
    calibration = [
        episode for episode in episodes if (episode.shard, episode.episode_index) in calib_keys
    ]
    evaluation = [
        episode for episode in episodes if (episode.shard, episode.episode_index) in eval_keys
    ]
    head = fit_trust_head(
        _feature_matrix(calibration, feature_names),
        np.asarray([not episode.success for episode in calibration], dtype=np.float64),
        feature_names,
    )
    # Applying the frozen head to every episode does not use eval labels.
    # The threshold below is still set from calibration successes only.
    all_scores = apply_trust_head(head, _feature_matrix(list(episodes), feature_names))
    score_of = {
        (episode.shard, episode.episode_index): float(all_scores[i])
        for i, episode in enumerate(episodes)
    }
    calib_success = np.asarray(
        [
            score_of[(episode.shard, episode.episode_index)]
            for episode in calibration
            if episode.success
        ],
        dtype=np.float64,
    )
    eval_success = np.asarray(
        [
            score_of[(episode.shard, episode.episode_index)]
            for episode in evaluation
            if episode.success
        ],
        dtype=np.float64,
    )
    eval_failure = np.asarray(
        [
            score_of[(episode.shard, episode.episode_index)]
            for episode in evaluation
            if not episode.success
        ],
        dtype=np.float64,
    )
    threshold = calibrate_threshold(calib_success, budget=metric.false_stop_budget)
    metrics = evaluate(eval_success, eval_failure, threshold)
    cross_validation = _cross_validate_trust(
        list(episodes),
        feature_names,
        budget=metric.false_stop_budget,
        n_folds=5,
        seed=seed,
    )

    named = "trust_head"
    wrapped = [
        EpisodeSignal(
            shard=episode.shard,
            episode_index=episode.episode_index,
            success=episode.success,
            n_steps=episode.n_steps,
            task=episode.task,
            scores={named: score_of[(episode.shard, episode.episode_index)]},
            per_step={},
            perturbation=episode.perturbation,
            perturbation_strength=episode.perturbation_strength,
        )
        for episode in episodes
    ]
    values: dict[str, object] = {
        "signal": named,
        "features": list(feature_names),
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
        "per_family": _per_family(wrapped, named, eval_keys, threshold),
        "detection_curve": detection_curve(
            calib_success, eval_success, eval_failure, CURVE_BUDGETS
        ),
        "bootstrap": bootstrap_rates(
            eval_success, eval_failure, threshold, n_resamples=n_resamples, seed=seed
        ),
        "cross_validation": cross_validation,
        "mean_lead_time_steps": None,
        "notes": (
            "Logistic regression on the calibration half only, then the same "
            "5% false-stop threshold rule as a single signal. " + LEAD_TIME_NOTE
        ),
        "weights": {
            "bias": float(head.weights[-1]),
            **{feature_names[i]: float(head.weights[i]) for i in range(len(feature_names))},
        },
    }
    return values, head
