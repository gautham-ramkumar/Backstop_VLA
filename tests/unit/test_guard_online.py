"""OnlineGuard must stop exactly where the offline rule does."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backstop.guard.online import OnlineGuard, load_rule, save_rule
from backstop.guard.signals import action_chunk_disagreement
from backstop.guard.stopwatch import StopwatchRule, stop_step

CALM = np.zeros((4, 50, 7))
LOUD = np.concatenate([CALM[:1], np.ones((3, 50, 7))])


def _rule(line: float, clock: int) -> StopwatchRule:
    return StopwatchRule(line=line, stop_step=clock, k=1, n_calibration=10, bound=0.18)


def _first_stop(guard: OnlineGuard, samples: np.ndarray) -> int | None:
    for t in range(samples.shape[0]):
        if guard.check_samples(samples[t], t):
            return t
    return None


@pytest.mark.parametrize("seed", range(30))
def test_matches_offline_stop_step(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_steps = int(rng.integers(60, 200))
    samples = rng.normal(scale=rng.uniform(0.05, 0.3), size=(n_steps, 4, 50, 7))
    rule = _rule(line=float(rng.uniform(0.1, 0.4)), clock=int(rng.integers(50, 150)))
    offline = stop_step(action_chunk_disagreement(samples), n_steps, rule)
    assert _first_stop(OnlineGuard(rule), samples) == offline


def test_reports_which_trigger_fired() -> None:
    guard = OnlineGuard(_rule(line=0.5, clock=10))
    assert guard.check_samples(LOUD, 3) and guard.fired_reason == "signal"
    assert not guard.check_samples(CALM, 4) and guard.fired_reason is None
    assert guard.check_samples(CALM, 10) and guard.fired_reason == "clock"


def test_rearm_restarts_the_clock_and_mutes_the_signal() -> None:
    guard = OnlineGuard(_rule(line=0.5, clock=10), cooldown=5)
    guard.rearm(40)
    assert not guard.check_samples(LOUD, 44)  # inside the cooldown
    assert guard.check_samples(LOUD, 45) and guard.fired_reason == "signal"
    assert not guard.check_samples(CALM, 49)
    assert guard.check_samples(CALM, 50) and guard.fired_reason == "clock"


def test_negative_cooldown_is_rejected() -> None:
    with pytest.raises(ValueError, match="cooldown"):
        OnlineGuard(_rule(line=0.5, clock=10), cooldown=-1)


def test_rule_round_trips_through_json(tmp_path: Path) -> None:
    rule = _rule(line=0.25, clock=140)
    save_rule(rule, tmp_path / "rule.json", budget=0.05)
    assert load_rule(tmp_path / "rule.json") == rule
