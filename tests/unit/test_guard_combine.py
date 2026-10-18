"""Trust head: fit on labeled rows, rank failures above successes."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.combine import TRUST_FEATURES, apply_trust_head, fit_trust_head
from backstop.guard.step_model import STEP_SIGNALS


def test_trust_head_ranks_failures_above_successes() -> None:
    rng = np.random.default_rng(0)
    successes = rng.normal(0.0, 0.2, size=(40, 2))
    failures = rng.normal(3.0, 0.2, size=(40, 2))
    features = np.vstack([successes, failures])
    labels = np.array([False] * 40 + [True] * 40)
    head = fit_trust_head(features, labels, ("a", "b"))
    probability = apply_trust_head(head, features)
    assert probability[40:].mean() > probability[:40].mean() + 0.4


def test_trust_head_rejects_a_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="feature names"):
        fit_trust_head(np.zeros((4, 2)), np.zeros(4), ("only_one",))


def test_visual_precondition_feeds_neither_learned_head() -> None:
    # It is a per-object gate (backstop.guard.presence), not a head input: in
    # corpus A the object is in view in every training run, so any weight the
    # head learns for it is a coincidence that extrapolates on new objects.
    assert "visual_precondition" not in TRUST_FEATURES
    assert "visual_precondition" not in STEP_SIGNALS
    assert set(TRUST_FEATURES) <= set(STEP_SIGNALS)
