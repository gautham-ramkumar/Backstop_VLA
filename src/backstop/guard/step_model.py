"""Per-step failure model, fit on the calibration half only.

One logistic regression, the same kind as the episode-level trust head. Each
step is a row. The label on that row is the episode's outcome, copied onto
every step: a failed episode is marked failed from the first frame. Onset
ranges are not an input.

Two weights keep the fit from being a vote of the longest timeouts. Each
step of an episode shares that episode's one vote, so a 280-step failure
does not outweigh a short success. Then the failure votes and the success
votes are scaled to the same total, so the larger class does not win by
count.

The columns are causal. At step t the model sees the signal now, the mean
so far, and the max so far. Nothing from t+1 onward.
"""

from __future__ import annotations

import numpy as np

from backstop.guard.combine import TrustHead, apply_trust_head, fit_trust_head

STEP_SIGNALS = (
    "action_chunk_disagreement",
    "temporal_self_consistency",
    "progress_monitoring",
    # No visual_precondition, for the reason given at `combine.TRUST_FEATURES`.
    "action_chunk_magnitude",
    "velocity_normalized_consistency",
    "internal_distance",
)


def causal_columns(signal: np.ndarray) -> np.ndarray:
    """(T,) -> (T, 3) of value, running mean, running max."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("signal has no steps")
    count = np.arange(1, values.size + 1, dtype=np.float64)
    running_mean = np.cumsum(values) / count
    running_max = np.maximum.accumulate(values)
    return np.stack([values, running_mean, running_max], axis=1)


def step_feature_names(signals: tuple[str, ...] = STEP_SIGNALS) -> tuple[str, ...]:
    names: list[str] = []
    for signal in signals:
        names.extend((f"{signal}_now", f"{signal}_mean", f"{signal}_max"))
    return tuple(names)


def step_features(
    series: dict[str, np.ndarray], signals: tuple[str, ...] = STEP_SIGNALS
) -> np.ndarray:
    """Stack the causal columns of each signal. All series must share a length."""
    columns = []
    length = None
    for signal in signals:
        if signal not in series:
            raise KeyError(signal)
        block = causal_columns(series[signal])
        if length is None:
            length = block.shape[0]
        elif block.shape[0] != length:
            raise ValueError(f"{signal} has length {block.shape[0]}, expected {length}")
        columns.append(block)
    return np.concatenate(columns, axis=1)


def episode_step_weights(lengths: list[int], failure: list[bool]) -> list[np.ndarray]:
    """One weight array per episode. Each episode sums to the same class total.

    Inside an episode every step gets an equal share of that episode's vote.
    Failure votes are then scaled so their sum matches the success votes.
    """
    if len(lengths) != len(failure):
        raise ValueError(f"{len(lengths)} lengths for {len(failure)} labels")
    if any(length < 1 for length in lengths):
        raise ValueError("every episode needs at least one step")
    n_success = sum(1 for failed in failure if not failed)
    n_failure = sum(1 for failed in failure if failed)
    if n_success == 0 or n_failure == 0:
        raise ValueError("step-model weights need both successes and failures")
    failure_scale = n_success / n_failure
    weights: list[np.ndarray] = []
    for length, failed in zip(lengths, failure, strict=True):
        scale = failure_scale if failed else 1.0
        weights.append(np.full(length, scale / length, dtype=np.float64))
    return weights


def fit_step_model(
    episodes: list[tuple[np.ndarray, bool]],
    feature_names: tuple[str, ...],
) -> TrustHead:
    """`episodes` is `(step_features (T, D), failure)` from the calibration half."""
    if not episodes:
        raise ValueError("no calibration episodes to fit a step model")
    lengths = [int(features.shape[0]) for features, _failed in episodes]
    labels = [bool(failed) for _features, failed in episodes]
    row_weights = episode_step_weights(lengths, labels)
    matrix = np.concatenate([features for features, _failed in episodes], axis=0)
    target = np.concatenate(
        [
            np.full(length, 1.0 if failed else 0.0)
            for length, failed in zip(lengths, labels, strict=True)
        ]
    )
    sample_weight = np.concatenate(row_weights)
    return fit_trust_head(
        matrix,
        target,
        feature_names,
        sample_weight=sample_weight,
    )


def score_episode(head: TrustHead, features: np.ndarray) -> np.ndarray:
    """P(failure) at each step."""
    return apply_trust_head(head, features)
