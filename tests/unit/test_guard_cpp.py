"""The C++ guard (cpp/) must stop exactly where OnlineGuard does."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from backstop.guard.online import save_rule
from backstop.guard.stopwatch import StopwatchRule
from verify_guard_cpp import SCORE_TOLERANCE, build, cpp_replay, python_replay


@pytest.fixture(scope="module")
def binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("cpp"))


def _rule_file(tmp_path: Path, line: float, clock: int) -> Path:
    rule = StopwatchRule(line=line, stop_step=clock, k=1, n_calibration=10, bound=0.18)
    save_rule(rule, tmp_path / "rule.json", budget=0.05, signal="action_chunk_disagreement")
    return tmp_path / "rule.json"


@pytest.mark.parametrize("seed", range(30))
def test_matches_python_guard(seed: int, binary: Path, tmp_path: Path) -> None:
    rng = np.random.default_rng(seed)
    n_steps = int(rng.integers(60, 200))
    # float32, like the recorded corpus; both sides widen to float64 first.
    samples = rng.normal(scale=rng.uniform(0.05, 0.3), size=(n_steps, 4, 50, 7)).astype(np.float32)
    rule = _rule_file(tmp_path, float(rng.uniform(0.1, 0.4)), int(rng.integers(50, 150)))
    cooldown = int(rng.integers(0, 10))
    rearms = tuple(sorted(int(s) for s in rng.choice(n_steps, size=2, replace=False)))

    cpp_scores, cpp_reasons = cpp_replay(binary, rule, samples, cooldown, rearms)
    py_scores, py_reasons = python_replay(rule, samples, cooldown, rearms)
    assert cpp_reasons == py_reasons
    np.testing.assert_allclose(cpp_scores, py_scores, rtol=0, atol=SCORE_TOLERANCE)


def test_one_sample_scores_zero(binary: Path, tmp_path: Path) -> None:
    scores, reasons = cpp_replay(binary, _rule_file(tmp_path, 0.5, 3), np.ones((4, 1, 50, 7)))
    assert scores.tolist() == [0.0] * 4
    assert reasons == ["-", "-", "-", "clock"]


def test_negative_cooldown_is_rejected(binary: Path, tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        cpp_replay(binary, _rule_file(tmp_path, 0.5, 10), np.zeros((2, 4, 50, 7)), cooldown=-1)
