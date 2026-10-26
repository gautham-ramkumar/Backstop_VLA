"""How early can the guard stop, and does it beat a stopwatch?

Every corpus-A failure runs to the 280-step cap, so "stop if the run is still
going at step N" already catches all of them. This script measures the guard
against that clock, and checks four things the earlier reports took on
trust:

1. Separability over time: at step t, can the running max of each signal
   tell failures from the successes still running at t? This decides whether
   an earlier stop is possible at all, before any threshold is chosen.
2. Perturbation, not failure: do perturbed runs that succeed score like
   clean successes, while failures in the same family score higher?
3. The clock alone, over 20 random splits: how often does its held-out
   false-stop rate go above the budget?
4. Signal OR stopwatch (`backstop.guard.stopwatch`), set from good runs
   only, over the same 20 splits, and on the object suite recalibrated from
   n good object runs with no object failure used.

The comparison rules this script used to report beside the headline
(temporal consistency, the trust head, searched persistence rules) are in
docs/experiments.md.

Onset ranges are read only to report lead time. Nothing is fit on them.

    uv run python scripts/evaluate_stopwatch.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from backstop.corpus.episode import EpisodeKey
from backstop.corpus.layout import LABELS, TRACES
from backstop.eval.report import write_guard_report
from backstop.guard.scoring import auroc, fit_stopwatch_from_rows
from backstop.guard.split import EVAL_FRACTION
from backstop.guard.stopwatch import StopwatchRule, stop_step
from backstop.guard.traces import (
    OBJECT_SHARD,
    LabelView,
    TraceRow,
    load_labels,
    load_traces,
    onset_range,
    series_map,
    spatial_pool,
    split_halves,
)
from backstop.schema import CONTROL_FPS, HEADLINE_METRIC, load_metric_contract

# ADR-005: the headline guard. The simplest one: a single signal, nothing
# learned, and the first signal built (ADR-004).
SIGNAL = "action_chunk_disagreement"
HEADLINE_GUARD = f"{SIGNAL} OR stopwatch"
SIGNALS = (SIGNAL,)


def separability(spatial: list[TraceRow]) -> list[dict[str, float]]:
    failures = [row for row in spatial if not row.success]
    successes = [row for row in spatial if row.success]
    table = []
    for t in (10, 20, 30, 40, 50, 60, 80, 100, 120):
        alive = [row for row in successes if row.n_steps > t]
        entry: dict[str, float | None] = {"step": t, "n_success_running": len(alive)}
        for name in SIGNALS:
            entry[name] = auroc(
                [float(row.series(name)[:t].max()) for row in failures],
                [float(row.series(name)[:t].max()) for row in alive],
            )
        table.append(entry)
    return table


def perturbation_check(spatial: list[TraceRow]) -> dict[str, dict[str, object]]:
    clean = [row for row in spatial if row.shard == "clean" and row.success]
    families = sorted({row.shard for row in spatial} - {"clean"})
    out: dict[str, dict[str, object]] = {}
    for family in families:
        good = [row for row in spatial if row.shard == family and row.success]
        bad = [row for row in spatial if row.shard == family and not row.success]
        entry: dict[str, object] = {"n_success": len(good), "n_failure": len(bad)}
        for name in SIGNALS:
            entry[name] = {
                "perturbed_success_vs_clean_success": auroc(
                    [row.scores[name] for row in good], [row.scores[name] for row in clean]
                ),
                "failure_vs_success": auroc(
                    [row.scores[name] for row in bad], [row.scores[name] for row in good]
                ),
            }
        out[family] = entry
    return out


def _apply(
    rows: list[TraceRow],
    series: dict[EpisodeKey, np.ndarray],
    rule: StopwatchRule,
    labels: dict[EpisodeKey, LabelView],
) -> dict[str, float]:
    false_stops = 0
    successes = 0
    stops: list[int] = []
    signal_first = 0
    leads: list[float] = []
    for row in rows:
        step = stop_step(series[row.key], row.n_steps, rule)
        if row.success:
            successes += 1
            false_stops += int(step is not None)
            continue
        assert step is not None, "a 280-step failure always outlasts the clock"
        stops.append(step)
        signal_first += int(step < rule.stop_step)
        onset = onset_range(labels, row.key)
        if onset is not None:
            leads.append(float(onset[0] - step))
    return {
        "false_stop_rate": false_stops / successes,
        "detection_rate": 1.0,
        "mean_stop_step": float(np.mean(stops)),
        "median_stop_step": float(np.median(stops)),
        "signal_first_fraction": signal_first / len(stops),
        "median_lead_vs_onset_start": float(np.median(leads)) if leads else float("nan"),
        "stop_step": rule.stop_step,
        "line": rule.line,
    }


def _summary(runs: list[dict[str, float]], budget: float) -> dict[str, float]:
    keys = runs[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([run[key] for run in runs], dtype=np.float64)
        with warnings.catch_warnings():
            # Object-suite runs carry no onset labels, so their lead is all NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"{key}_mean"] = float(np.nanmean(values))
            out[f"{key}_p10"] = float(np.nanpercentile(values, 10))
            out[f"{key}_p90"] = float(np.nanpercentile(values, 90))
    if "saved_over_stopwatch" in runs[0]:
        saved = np.array([run["saved_over_stopwatch"] for run in runs])
        out["saved_over_stopwatch_min"] = float(saved.min())
        out["saved_over_stopwatch_max"] = float(saved.max())
        out["splits_not_helped"] = int((saved <= 0).sum())
    fs = np.array([run["false_stop_rate"] for run in runs])
    out["splits_over_budget"] = int((fs > budget + 1e-12).sum())
    out["n_splits"] = len(runs)
    return out


def budget_drift(
    pooled: list[TraceRow],
    labels: dict[EpisodeKey, LabelView],
    seeds: int,
    eval_fraction: float,
    budget: float,
) -> dict[str, dict[str, float]]:
    """The clock alone, and signal OR stopwatch, over many paired splits."""
    runs: dict[str, list[dict[str, float]]] = {}
    for seed in range(seeds):
        calibration, eval_eps = split_halves(pooled, eval_fraction=eval_fraction, seed=seed)
        series = series_map(calibration + eval_eps, SIGNAL)
        clock_only = {row.key: np.full(row.n_steps, -np.inf) for row in calibration + eval_eps}
        good = [row for row in calibration if row.success]
        lengths = np.array([row.n_steps for row in good])
        k = int(np.ceil(budget * len(good)))
        clock_rule = StopwatchRule(
            line=np.inf,
            stop_step=int(np.sort(lengths)[::-1][k - 1]),
            k=k,
            n_calibration=len(good),
            bound=k / (len(good) + 1),
        )
        clock_run = _apply(eval_eps, clock_only, clock_rule, labels)
        runs.setdefault("stopwatch only", []).append(clock_run)
        rule = fit_stopwatch_from_rows(calibration, series, budget)
        run = _apply(eval_eps, series, rule, labels)
        # ADR-005 headline: paired with the clock on the same split.
        run["saved_over_stopwatch"] = clock_run["mean_stop_step"] - run["mean_stop_step"]
        runs.setdefault(HEADLINE_GUARD, []).append(run)
    return {name: _summary(values, budget) for name, values in runs.items()}


def object_recalibration(
    traces: list[TraceRow],
    pooled: list[TraceRow],
    labels: dict[EpisodeKey, LabelView],
    budget: float,
    draws: int,
) -> dict[str, object]:
    obj = [row for row in traces if row.shard == OBJECT_SHARD]
    good = [row for row in obj if row.success]
    bad = [row for row in obj if not row.success]
    out: dict[str, object] = {}
    rng = np.random.default_rng(0)
    for name in SIGNALS:
        spatial_series = series_map(pooled, name)
        frozen = fit_stopwatch_from_rows(pooled, spatial_series, budget)
        series = series_map(obj, name)
        entry: dict[str, object] = {"frozen_spatial_rule": _apply(obj, series, frozen, labels)}
        for n in (10, 20, 30, 40, 50):
            runs = []
            bounds = []
            for _ in range(draws):
                order = rng.permutation(len(good))
                calibration = [good[i] for i in order[:n]]
                rest = [good[i] for i in order[n:]] + bad
                rule = fit_stopwatch_from_rows(calibration, series, budget)
                bounds.append(rule.bound)
                runs.append(_apply(rest, series, rule, labels))
            summary = _summary(runs, budget)
            summary["bound"] = float(np.mean(bounds))
            entry[f"recalibrated_n{n}"] = summary
        out[name] = entry
    return out


def _print(report: dict[str, object]) -> None:
    print("1. Separability over time (AUROC, running max, failures vs successes still running)")
    for entry in report["separability"]:  # type: ignore[union-attr]
        cells = "  ".join(f"{name[:12]} {entry[name]:.2f}" for name in SIGNALS)
        print(f"   step {entry['step']:>3} ({entry['n_success_running']:>3} running)  {cells}")
    print(
        "\n2. Perturbation check (disagreement): "
        "perturbed-success vs clean-success | failure vs success"
    )
    for family, entry in report["perturbation_check"].items():  # type: ignore[union-attr]
        cell = entry["action_chunk_disagreement"]
        print(
            f"   {family:20s} n_succ={entry['n_success']:>2}  "
            f"{cell['perturbed_success_vs_clean_success']:.2f} | {cell['failure_vs_success']:.2f}"
        )
    print("\n3-4. Spatial, held-out half, over random splits")
    for name, row in report["spatial_splits"].items():  # type: ignore[union-attr]
        line = (
            f"   {name:48s} false-stop {row['false_stop_rate_mean']:.3f} "
            f"[{row['false_stop_rate_p10']:.3f}-{row['false_stop_rate_p90']:.3f}] "
            f"over {report['budget']:.0%} on {row['splits_over_budget']}/{row['n_splits']}  "
            f"detection {row['detection_rate_mean']:.3f}"
        )
        if "mean_stop_step_mean" in row:
            line += (
                f"  mean stop step {row['mean_stop_step_mean']:.0f} "
                f"({row['mean_stop_step_mean'] / CONTROL_FPS:.1f} s), signal first "
                f"{row['signal_first_fraction_mean']:.0%}"
            )
        if "saved_over_stopwatch_mean" in row:
            line += (
                f"  saved {row['saved_over_stopwatch_mean']:.1f} "
                f"[{row['saved_over_stopwatch_min']:.1f}, {row['saved_over_stopwatch_max']:.1f}]"
            )
        print(line)
    print("\n4. Object suite")
    for name, entry in report["object_suite"].items():  # type: ignore[union-attr]
        frozen = entry["frozen_spatial_rule"]
        print(f"   {name}: frozen spatial rule false-stop {frozen['false_stop_rate']:.3f}")
        for key, row in entry.items():
            if not key.startswith("recalibrated"):
                continue
            print(
                f"     {key}: false-stop {row['false_stop_rate_mean']:.3f} "
                f"(bound {row['bound']:.3f}), "
                f"mean stop step {row['mean_stop_step_mean']:.0f}, "
                f"signal first {row['signal_first_fraction_mean']:.0%}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=TRACES)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/guard/stopwatch/report.json")
    )
    parser.add_argument("--budget", type=float, default=load_metric_contract().false_stop_budget)
    parser.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--draws", type=int, default=200)
    args = parser.parse_args()

    traces = load_traces(args.traces)
    labels = load_labels(args.labels)
    pooled = spatial_pool(traces)

    report: dict[str, object] = {
        "budget": args.budget,
        "eval_fraction": args.eval_fraction,
        "fps": CONTROL_FPS,
        "note": (
            "Stopwatch rules are set from calibration successes only. Every "
            "failure outlasts the clock, so detection is 1.0 by construction; "
            "the comparison is false-stop rate and stop step."
        ),
        "separability": separability(pooled),
        "perturbation_check": perturbation_check(pooled),
        "spatial_splits": budget_drift(
            pooled, labels, args.splits, args.eval_fraction, args.budget
        ),
        "object_suite": object_recalibration(traces, pooled, labels, args.budget, args.draws),
    }
    _print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")

    splits = report["spatial_splits"]
    assert isinstance(splits, dict)
    guards = {name: row for name, row in splits.items() if name.endswith("OR stopwatch")}
    headline = guards[HEADLINE_GUARD]
    values = {
        "guard": HEADLINE_GUARD,
        "saved_steps_mean": headline["saved_over_stopwatch_mean"],
        "saved_seconds_mean": headline["saved_over_stopwatch_mean"] / CONTROL_FPS,
        "saved_steps_p10": headline["saved_over_stopwatch_p10"],
        "saved_steps_p90": headline["saved_over_stopwatch_p90"],
        "saved_steps_min": headline["saved_over_stopwatch_min"],
        "saved_steps_max": headline["saved_over_stopwatch_max"],
        "splits_not_helped": headline["splits_not_helped"],
        "false_stop_rate_mean": headline["false_stop_rate_mean"],
        "splits_over_budget": headline["splits_over_budget"],
        "signal_first_fraction_mean": headline["signal_first_fraction_mean"],
        "n_splits": headline["n_splits"],
        "fps": CONTROL_FPS,
        "stopwatch_only": splits["stopwatch only"],
        "all_or_rules": guards,
        "source": str(args.output),
    }
    path = write_guard_report(
        output_dir=args.output.parent.parent / "headline",
        headline_metric=HEADLINE_METRIC,
        values=values,
        metric=load_metric_contract(),
    )
    print(
        f"\nHEADLINE ({HEADLINE_METRIC}): {HEADLINE_GUARD} saves "
        f"{values['saved_steps_mean']:.1f} steps ({values['saved_seconds_mean']:.2f} s) "
        f"over the stopwatch, 10-90% [{values['saved_steps_p10']:.1f}, "
        f"{values['saved_steps_p90']:.1f}], no help on {values['splits_not_helped']}/"
        f"{values['n_splits']} splits, false-stop {values['false_stop_rate_mean']:.3f}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
