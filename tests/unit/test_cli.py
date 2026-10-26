"""Both CLIs exit on the pipeline's own tolerance verdict, the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# The CLIs import the pipeline, which imports the rollout loop and its
# gymnasium-based wrapper (`sim` extra). The sim CI job runs this module.
pytest.importorskip("gymnasium")

from backstop.eval import cli
from backstop.eval.tolerance import EXIT_OUT_OF_TOLERANCE

CONFIG = "configs/record/spatial_baseline.yaml"


def _fake_pipeline(rate: float, verdict: bool | None):
    def run(cfg: Any) -> dict[str, Any]:
        return {"success_rate": rate, "n_total": 10, "in_tolerance": verdict}

    return run


@pytest.mark.parametrize("main", [cli.main, cli.record_main])
@pytest.mark.parametrize(
    ("rate", "verdict", "code"),
    [
        (0.80, True, 0),
        (0.50, False, EXIT_OUT_OF_TOLERANCE),
        (0.95, False, EXIT_OUT_OF_TOLERANCE),  # eval used to pass this
        (0.10, None, 0),  # resumed: record used to fail this
    ],
)
def test_exit_code_follows_the_pipeline_verdict(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    main: Any,
    rate: float,
    verdict: bool | None,
    code: int,
) -> None:
    monkeypatch.setattr(cli, "run_pipeline", _fake_pipeline(rate, verdict))
    assert main(["--config", str(repo_root / CONFIG)]) == code
