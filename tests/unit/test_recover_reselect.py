"""Re-selection pre-test metrics."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.recover.reselect import GRIPPER_DIM, reselect_metrics


def test_exchangeable_samples_give_an_outlier_ratio_near_one() -> None:
    rng = np.random.default_rng(0)
    metrics = reselect_metrics(rng.normal(size=(500, 4, 50, 7)))
    assert metrics["outlier_ratio"].mean() == pytest.approx(1.0, abs=0.02)


def test_an_odd_chunk_zero_stands_out() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(scale=0.01, size=(5, 4, 50, 7))
    samples[:, 0] += 1.0
    assert np.all(reselect_metrics(samples)["outlier_ratio"] > 10)


def test_gripper_split_flags_a_disagreement_on_open_vs_close() -> None:
    samples = np.zeros((2, 4, 50, 7))
    samples[:, :, :, GRIPPER_DIM] = 1.0
    samples[1, 2, :, GRIPPER_DIM] = -1.0
    np.testing.assert_array_equal(reselect_metrics(samples)["gripper_split"], [0.0, 1.0])


def test_identical_samples_have_no_spread() -> None:
    samples = np.ones((3, 4, 50, 7))
    np.testing.assert_array_equal(reselect_metrics(samples)["spread"], np.zeros(3))


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="T, K"):
        reselect_metrics(np.zeros((4, 50, 7)))
    with pytest.raises(ValueError, match="K >= 3"):
        reselect_metrics(np.zeros((3, 2, 50, 7)))
