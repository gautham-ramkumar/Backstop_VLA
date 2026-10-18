"""Per-step model: causal features, episode-equal weights, early bump is usable."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.combine import fit_trust_head
from backstop.guard.step_model import (
    causal_columns,
    episode_step_weights,
    fit_step_model,
    score_episode,
    step_feature_names,
    step_features,
)


def test_causal_columns_do_not_see_the_future() -> None:
    early = causal_columns(np.array([0.0, 0.0, 1.0]))
    later = causal_columns(np.array([0.0, 0.0, 9.0]))
    assert early[0].tolist() == [0.0, 0.0, 0.0]
    assert np.allclose(early[:2], later[:2])
    assert early[2, 1] == pytest.approx(1.0 / 3.0)
    assert early[2, 2] == pytest.approx(1.0)


def test_episode_step_weights_balance_the_two_classes() -> None:
    weights = episode_step_weights([2, 4, 10], [False, False, True])
    success_total = float(weights[0].sum() + weights[1].sum())
    failure_total = float(weights[2].sum())
    assert success_total == pytest.approx(failure_total)
    assert weights[0].sum() == pytest.approx(1.0)
    assert weights[2][0] == pytest.approx(weights[2][-1])


def test_step_model_scores_an_early_bump_above_a_success() -> None:
    names = step_feature_names(("bump",))
    episodes: list[tuple[np.ndarray, bool]] = []
    for index in range(30):
        values = np.zeros(12)
        failed = index >= 15
        if failed:
            values[2] = 1.0
        episodes.append((step_features({"bump": values}, ("bump",)), failed))
    head = fit_step_model(episodes, names)
    failure_score = score_episode(head, episodes[-1][0])
    success_score = score_episode(head, episodes[0][0])
    assert failure_score[2] > failure_score[0]
    assert failure_score[2] > float(success_score.max()) + 0.2


def test_sample_weight_pulls_the_fit_toward_the_heavy_rows() -> None:
    features = np.array([[0.0], [0.0], [1.0], [-1.0]])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    toward_high = fit_trust_head(
        features, labels, ("x",), sample_weight=np.array([1.0, 1.0, 50.0, 1.0])
    )
    toward_low = fit_trust_head(
        features, labels, ("x",), sample_weight=np.array([1.0, 1.0, 1.0, 50.0])
    )
    assert toward_high.weights[0] > 0
    assert toward_low.weights[0] < 0
