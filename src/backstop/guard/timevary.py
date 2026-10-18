"""A stop line that follows a normal successful episode.

Successes are loud at some steps and quiet at others. One line for the whole
episode has to clear the loudest success step, so a failure that is only
slightly high, early, never crosses it. Subtracting the typical success score
at each step leaves the excess. The usual 5% rule is then applied to that
excess, on the calibration half only.
"""

from __future__ import annotations

import numpy as np


def success_profile(series: list[np.ndarray], quantile: float = 0.5) -> np.ndarray:
    """Per-step quantile of successful episodes. Later steps use whoever reached them."""
    if not series:
        raise ValueError("no successful episodes to build a time-varying profile from")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    length = max(len(scores) for scores in series)
    if length == 0:
        raise ValueError("successful episodes have no steps")
    profile = np.empty(length, dtype=np.float64)
    for step in range(length):
        values = [scores[step] for scores in series if step < len(scores)]
        profile[step] = float(np.quantile(np.asarray(values, dtype=np.float64), quantile))
    return profile


def subtract_profile(scores: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """How far each step sits above the success profile.

    Past the end of the profile, the last profile value is reused so a longer
    episode is not scored against zero.
    """
    values = np.asarray(scores, dtype=np.float64).copy()
    if profile.size == 0:
        raise ValueError("profile is empty")
    shared = min(values.size, profile.size)
    values[:shared] = values[:shared] - profile[:shared]
    if values.size > profile.size:
        values[profile.size :] -= profile[-1]
    return values
