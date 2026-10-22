"""The verify scripts' PASS/FAIL harness."""

from __future__ import annotations

import pytest

from backstop.checks import Checks


def test_all_passing_reports_every_check_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    checks = Checks()
    checks.check(True, "eval.json written", "artifacts/eval.json")
    checks.check(1 == 1, "a longer check name")
    assert checks.report(on_failure="never shown") == 0
    out = capsys.readouterr().out
    assert "PASS  eval.json written    artifacts/eval.json" in out
    assert "2/2 checks passed" in out
    assert "never shown" not in out


def test_any_failure_is_reported_with_the_hint_and_exits_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checks = Checks()
    checks.check(True, "first")
    checks.check(False, "second", "max abs diff=0.3")
    checks.check(True, "third")
    assert checks.n_failed == 1
    assert checks.report(on_failure="Do not proceed.") == 1
    out = capsys.readouterr().out
    assert "FAIL  second  max abs diff=0.3" in out
    assert "2/3 checks passed" in out
    assert out.rstrip().endswith("Do not proceed.")


def test_truthy_values_are_stored_as_bools() -> None:
    checks = Checks()
    checks.check([1], "non-empty list")  # type: ignore[arg-type]
    assert checks.results[0].ok is True
