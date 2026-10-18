"""The stored internal feature is the mean of the real prefix tokens."""

from __future__ import annotations

import numpy as np

from backstop.guard.internal import diagonal_distance, fit_diagonal, pool_prefix


def test_padding_is_left_out_of_the_average() -> None:
    hidden = np.array([[1.0, 1.0], [3.0, 5.0], [100.0, 100.0]])
    mask = np.array([True, True, False])
    pooled = pool_prefix(hidden, mask)
    assert pooled.shape == (2,)
    assert np.allclose(pooled, [2.0, 3.0])


def test_a_step_far_from_successes_scores_higher() -> None:
    successes = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    mean, variance = fit_diagonal(successes)
    ordinary = diagonal_distance(np.array([[0.5, 0.5]]), mean, variance)
    unusual = diagonal_distance(np.array([[10.0, 10.0]]), mean, variance)
    assert unusual[0] > ordinary[0]
