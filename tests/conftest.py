"""Shared test setup: the repo root, and strict skips in CI.

Strict skips. Most test modules start with `pytest.importorskip(...)`, so a
missing dependency turns them into skips, not failures. That is how CI once
ran 31 of 299 tests and stayed green. When `BACKSTOP_ALLOWED_SKIPS` is set,
every skip outside that comma-separated list of test-module stems fails the
run. CI sets it per job (empty in the job that installs everything). Unset,
as it is locally, skips behave as usual.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SKIPS_ENV = "BACKSTOP_ALLOWED_SKIPS"

_unexpected_skips: list[str] = []


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def _allowed_skips() -> frozenset[str] | None:
    raw = os.environ.get(ALLOWED_SKIPS_ENV)
    if raw is None:
        return None
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def _note_skip(report: pytest.CollectReport | pytest.TestReport) -> None:
    allowed = _allowed_skips()
    if allowed is None or not report.skipped:
        return
    module = Path(report.nodeid.split("::")[0]).stem
    if module not in allowed:
        _unexpected_skips.append(report.nodeid)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    _note_skip(report)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _note_skip(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _unexpected_skips:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"\n{ALLOWED_SKIPS_ENV} is set and these were skipped:", red=True)
        for nodeid in _unexpected_skips:
            reporter.write_line(f"  {nodeid}", red=True)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
