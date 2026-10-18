"""The time-varying line is the path of a normal success, subtracted off."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.timevary import subtract_profile, success_profile


def test_an_early_spike_survives_when_successes_are_only_loud_at_the_end() -> None:
    successes = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.1, 1.0])]
    profile = success_profile(successes, quantile=0.5)
    excess = subtract_profile(np.array([1.0, 0.0, 1.0]), profile)
    assert excess[0] == pytest.approx(1.0)
    assert excess[2] == pytest.approx(0.0)


def test_a_longer_episode_reuses_the_last_profile_value() -> None:
    profile = success_profile([np.array([0.2, 0.4])], quantile=0.5)
    excess = subtract_profile(np.array([0.2, 0.4, 0.4]), profile)
    assert np.allclose(excess, 0.0)
