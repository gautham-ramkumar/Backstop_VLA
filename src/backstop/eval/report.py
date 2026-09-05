"""Eval report writer. Guard headline cannot be raw accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backstop.schema import (
    FORBIDDEN_HEADLINES,
    HEADLINE_METRIC,
    MetricContract,
    RunProvenance,
    load_metric_contract,
)


class HeadlineMetricError(ValueError):
    """Raised when a report tries to headline a forbidden metric."""


def write_eval_report(
    *,
    output_dir: Path,
    provenance: RunProvenance,
    stats: dict[str, Any],
    metric: MetricContract | None = None,
) -> Path:
    """Write policy-baseline eval.json. Does not headline guard accuracy."""
    contract = metric or load_metric_contract()
    payload = {
        "kind": "policy_baseline",
        "policy_success_rate": stats["success_rate"],
        "n_total": stats["n_total"],
        "n_success": stats["n_success"],
        "n_failure": stats["n_failure"],
        "in_tolerance": provenance.in_tolerance,
        "episodes": stats["episodes"],
        "provenance": provenance.model_dump(),
        "metric_contract": {
            "false_stop_budget": contract.false_stop_budget,
            "headline_metric": contract.headline_metric,
            "status": "locked_not_yet_measured",
            "note": (
                "Week-1 reports policy success rate as a setup check only. "
                "The guard headline remains detection_at_false_stop."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eval.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "run.json").write_text(provenance.model_dump_json(indent=2) + "\n")
    return path


def write_timing_report(*, output_dir: Path, timing: dict[str, Any]) -> Path:
    """Write the per-stage throughput breakdown beside the eval report.

    Its own file rather than a key in `eval.json`: that payload is a frozen
    week-1 contract, and how long a run took is not a result about the policy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "timing.json"
    path.write_text(json.dumps(timing, indent=2) + "\n")
    return path


def write_guard_report(
    *,
    output_dir: Path,
    headline_metric: str,
    values: dict[str, Any],
    metric: MetricContract | None = None,
) -> Path:
    """Guard evaluation writer. Refuses raw accuracy as the headline."""
    if headline_metric in FORBIDDEN_HEADLINES:
        raise HeadlineMetricError(
            f"Cannot headline {headline_metric!r}. Locked headline is {HEADLINE_METRIC} "
            f"at a {DEFAULT_BUDGET_HINT}. See docs/adr/001-metric.md."
        )
    contract = metric or load_metric_contract()
    if headline_metric != contract.headline_metric:
        raise HeadlineMetricError(
            f"headline_metric must be {contract.headline_metric!r}, got {headline_metric!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "guard_eval.json"
    path.write_text(
        json.dumps(
            {
                "headline_metric": headline_metric,
                "false_stop_budget": contract.false_stop_budget,
                "values": values,
            },
            indent=2,
        )
        + "\n"
    )
    return path


DEFAULT_BUDGET_HINT = "5% false-stop budget"


def write_failures_markdown(path: Path, episodes: list[dict[str, Any]]) -> Path:
    failures = [row for row in episodes if not row.get("success")]
    lines = [
        "# Natural failures (week 1)",
        "",
        "Generated from the recorded/eval run. Notes are placeholders until watched.",
        "",
        f"Count: {len(failures)}",
        "",
        "| episode | suite | task_id | seed | note |",
        "|---|---|---|---|---|",
    ]
    for row in failures:
        lines.append(
            f"| {row['episode_index']} | {row['suite']} | {row['task_id']} "
            f"| {row['seed']} | unreviewed |"
        )
    if not failures:
        lines.append("| — | — | — | — | no failures in this run |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
