"""A compact summary of the policy's last layer.

The vision-language stack turns the cameras, the instruction, and the arm
state into one sequence of vectors. The guard keeps the average of the
positions that are not padding. That is one vector per step, small enough to
store, and it moves when the scene the model is looking at moves. The flow
matching steps that turn the vector into an action are not part of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def diagonal_distance(values: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """How unusual each step is, one number per step.

    Each dimension is compared with the successes on its own, then the squared
    gaps are averaged. A full covariance in 960 dimensions would treat noise
    as structure. The per-dimension version does not.
    """
    matrix = np.asarray(values, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(variance, dtype=np.float64)
    if matrix.ndim != 2 or center.shape != (matrix.shape[1],) or scale.shape != center.shape:
        raise ValueError(
            f"expected values (steps, dim) with mean and variance of that dim, got {matrix.shape}"
        )
    gap = (matrix - center) / np.sqrt(np.maximum(scale, 1e-8))
    return np.mean(gap * gap, axis=1)


def fit_diagonal(success_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of successful steps. Rows are steps, columns are dimensions."""
    matrix = np.asarray(success_steps, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError(f"need at least two successful steps, got shape {matrix.shape}")
    return matrix.mean(axis=0), matrix.var(axis=0)


def pool_prefix(hidden: np.ndarray, pad_mask: np.ndarray) -> np.ndarray:
    """Mean of the prefix tokens that are not padding.

    `hidden` is (sequence, dim) and `pad_mask` is (sequence,) True where a
    token is real. An all-pad mask falls back to the mean of the whole
    sequence so a bad mask cannot drop the step.
    """
    values = np.asarray(hidden, dtype=np.float64)
    mask = np.asarray(pad_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"expected hidden (sequence, dim), got shape {values.shape}")
    if mask.shape != (values.shape[0],):
        raise ValueError(f"pad mask shape {mask.shape} does not match sequence {values.shape[0]}")
    if not mask.any():
        return values.mean(axis=0)
    return values[mask].mean(axis=0)


def internal_path(root: Path, shard: str, episode: int) -> Path:
    """Where `scripts/extract_internal.py` writes one episode's (T, D) vectors."""
    return Path(root) / shard / f"{episode:04d}.npy"


def load_internal(root: Path, shard: str, episode: int) -> np.ndarray:
    path = internal_path(root, shard, episode)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)
