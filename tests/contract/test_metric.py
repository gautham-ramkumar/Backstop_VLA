from __future__ import annotations

from pathlib import Path

import pytest

from backstop.eval.report import HeadlineMetricError, write_eval_report, write_guard_report
from backstop.schema import (
    DEFAULT_FALSE_STOP_BUDGET,
    HEADLINE_METRIC,
    RunProvenance,
    load_metric_contract,
)


def _provenance() -> RunProvenance:
    return RunProvenance(
        git_sha="deadbeef",
        policy_path="HuggingFaceVLA/smolvla_libero",
        policy_revision=None,
        policy_type="smolvla",
        device="cuda",
        mujoco_version="3.3.2",
        mujoco_gl="osmesa",
        seed=0,
        suite="libero_spatial",
        task_ids=[0],
        n_episodes=2,
        n_action_steps=1,
        num_steps=10,
        k_samples=4,
        batch_size=1,
        success_rate=0.5,
        n_success=1,
        n_failure=1,
        dataset_root="data/lerobot/backstop_spatial_smoke",
        output_dir="artifacts/eval/spatial_smoke",
        in_tolerance=False,
    )


def test_budget_is_locked_at_five_percent() -> None:
    contract = load_metric_contract(Path("configs/metric.yaml"))
    assert contract.false_stop_budget == pytest.approx(DEFAULT_FALSE_STOP_BUDGET)
    assert contract.headline_metric == HEADLINE_METRIC


def test_guard_report_rejects_raw_accuracy(tmp_path: Path) -> None:
    with pytest.raises(HeadlineMetricError):
        write_guard_report(
            output_dir=tmp_path,
            headline_metric="accuracy",
            values={"accuracy": 0.99},
        )


def test_guard_report_rejects_success_rate(tmp_path: Path) -> None:
    with pytest.raises(HeadlineMetricError):
        write_guard_report(
            output_dir=tmp_path,
            headline_metric="success_rate",
            values={"success_rate": 0.9},
        )


def test_guard_report_accepts_detection_at_false_stop(tmp_path: Path) -> None:
    path = write_guard_report(
        output_dir=tmp_path,
        headline_metric=HEADLINE_METRIC,
        values={"detection_rate": 0.4, "false_stop_budget": 0.05},
    )
    assert path.exists()
    text = path.read_text()
    assert HEADLINE_METRIC in text
    assert "0.05" in text


def test_policy_baseline_report_does_not_claim_guard_headline(tmp_path: Path) -> None:
    path = write_eval_report(
        output_dir=tmp_path,
        provenance=_provenance(),
        stats={
            "success_rate": 0.5,
            "n_total": 2,
            "n_success": 1,
            "n_failure": 1,
            "episodes": [],
        },
    )
    payload = path.read_text()
    assert "policy_success_rate" in payload
    assert "locked_not_yet_measured" in payload
    assert '"headline_metric": "detection_at_false_stop"' in payload
