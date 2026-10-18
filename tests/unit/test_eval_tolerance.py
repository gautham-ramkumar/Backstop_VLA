"""One tolerance rule for the pipeline and both CLIs."""

from __future__ import annotations

import pytest

from backstop.config import ToleranceConfig
from backstop.eval.tolerance import EXIT_OUT_OF_TOLERANCE, cli_outcome, tolerance_verdict

TOL = ToleranceConfig(min_success_rate=0.70, max_success_rate=0.90, enforce=True)


@pytest.mark.parametrize(
    ("rate", "expected"), [(0.69, False), (0.70, True), (0.80, True), (0.90, True), (0.91, False)]
)
def test_verdict_checks_both_bounds_inclusive(rate: float, expected: bool) -> None:
    assert tolerance_verdict(rate, partial=False, tol=TOL) is expected


def test_a_resumed_run_is_not_judged() -> None:
    assert tolerance_verdict(0.10, partial=True, tol=TOL) is None
    assert cli_outcome(0.10, None, TOL).exit_code == 0


def test_in_band_exits_zero_quietly() -> None:
    assert cli_outcome(0.80, True, TOL).exit_code == 0
    assert cli_outcome(0.80, True, TOL).message is None


def test_below_band_fails_and_suggests_the_fallback() -> None:
    outcome = cli_outcome(0.50, False, TOL)
    assert outcome.exit_code == EXIT_OUT_OF_TOLERANCE
    assert outcome.message is not None and "num_steps" in outcome.message


def test_above_band_fails_without_the_fallback_hint() -> None:
    outcome = cli_outcome(0.95, False, TOL)
    assert outcome.exit_code == EXIT_OUT_OF_TOLERANCE
    assert outcome.message is not None and "num_steps" not in outcome.message


def test_enforce_off_reports_but_exits_zero() -> None:
    relaxed = TOL.model_copy(update={"enforce": False})
    outcome = cli_outcome(0.50, False, relaxed)
    assert outcome.exit_code == 0
    assert outcome.message is not None and "OUT OF TOLERANCE" in outcome.message
