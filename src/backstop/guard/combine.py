"""Trust-score head: one number from the per-signal episode scores.

Fit only on the calibration split. The four signals are computed without
labels; the head is the only place success and failure enter, and they enter
as the calibration half alone. The eval half is scored with the frozen head
and then judged by the same 5% false-stop threshold rule as a single signal
(ADR-004). No taxonomy categories and no onset ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The head's inputs: the three signals read from the action stream. The
# visual precondition is evaluated on its own and as a per-object gate
# (`backstop.guard.presence`), but it is not a head input. In corpus A the
# instructed object is in view in every run the head is fit on, failing or
# not, so the only weight the head can learn for it is a coincidence, and on
# a suite with other objects that weight extrapolates. With the detector
# score as input it did exactly that: the object-suite false-stop rate fell
# from 36/83 to 6/83 for no real reason (docs/evaluation.md).
TRUST_FEATURES = (
    "action_chunk_disagreement",
    "temporal_self_consistency",
    "progress_monitoring",
)


@dataclass(frozen=True)
class TrustHead:
    """Standardize with calibration mean/std, then logistic regression.

    `weights` has length D+1. The last entry is the bias. A high
    `apply_trust_head` output is P(failure): the guard should trust the
    policy less.
    """

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    feature_names: tuple[str, ...]


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_trust_head(
    features: np.ndarray,
    failure: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    l2: float = 1.0,
    steps: int = 50,
    sample_weight: np.ndarray | None = None,
) -> TrustHead:
    """IRLS logistic regression. `failure` is 1 for a failed episode.

    L2 is applied to every coefficient except the bias, so a separated
    calibration set cannot send the coefficients to infinity. Features are
    standardized from this same calibration matrix; the eval half must be
    passed through `apply_trust_head`, which reuses these mean and std.
    `sample_weight` gives each row a share of the fit. Without it, every row
    counts equally.
    """
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(failure, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected features (N, D), got shape {matrix.shape}")
    if labels.shape != (matrix.shape[0],):
        raise ValueError(
            f"failure labels length {labels.shape} does not match {matrix.shape[0]} rows"
        )
    if len(feature_names) != matrix.shape[1]:
        raise ValueError(f"{len(feature_names)} feature names for {matrix.shape[1]} columns")
    if matrix.shape[0] == 0:
        raise ValueError("no calibration episodes to fit a trust head")
    if sample_weight is None:
        row_weight = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        row_weight = np.asarray(sample_weight, dtype=np.float64)
        if row_weight.shape != (matrix.shape[0],):
            raise ValueError(
                f"sample_weight length {row_weight.shape} does not match {matrix.shape[0]} rows"
            )
        if np.any(row_weight < 0) or float(row_weight.sum()) <= 0:
            raise ValueError("sample_weight must be non-negative and not all zero")

    share = row_weight / row_weight.sum()
    mean = share @ matrix
    centered = matrix - mean
    std = np.sqrt(share @ (centered * centered))
    std = np.where(std < 1e-8, 1.0, std)
    scaled = (matrix - mean) / std
    count, width = scaled.shape
    design = np.concatenate([scaled, np.ones((count, 1))], axis=1)
    coefficients = np.zeros(width + 1, dtype=np.float64)
    penalty = np.eye(width + 1) * l2
    penalty[-1, -1] = 0.0

    for _ in range(steps):
        probability = _sigmoid(design @ coefficients)
        hessian_weight = row_weight * probability * (1.0 - probability)
        hessian = design.T @ (design * hessian_weight[:, None]) + penalty
        gradient = design.T @ (row_weight * (probability - labels)) + penalty @ coefficients
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        coefficients = coefficients - step

    return TrustHead(
        mean=mean,
        std=std,
        weights=coefficients,
        feature_names=tuple(feature_names),
    )


def apply_trust_head(head: TrustHead, features: np.ndarray) -> np.ndarray:
    """P(failure) for each row, using the calibration standardizer."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != head.mean.shape[0]:
        raise ValueError(f"expected features (N, {head.mean.shape[0]}), got shape {matrix.shape}")
    scaled = (matrix - head.mean) / head.std
    design = np.concatenate([scaled, np.ones((matrix.shape[0], 1))], axis=1)
    return _sigmoid(design @ head.weights)
