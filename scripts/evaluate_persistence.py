"""Fit a persistence stop rule and measure lead time on the held-out half.

The rule is chosen from calibration episodes only: how high a score must be,
and for how many consecutive steps, so that at most 5% of calibration
successes are stopped. Onset ranges are read afterwards, and only to measure
how early the chosen rule stops on the eval half. They never enter the fit.

Usage:
    PYTHONPATH=src python scripts/evaluate_persistence.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.guard.calibrate import calibrate_threshold
from backstop.guard.combine import TRUST_FEATURES, fit_trust_head
from backstop.guard.persist import select_persistence_rule
from backstop.guard.scoring import evaluate_persistence_rule
from backstop.guard.traces import (
    FPS,
    OBJECT_SHARD,
    TRACE_SIGNALS,
    episode_key,
    labels_by_key,
    load_jsonl,
    series_map,
    spatial_pool,
    split_halves,
)


def _fit_one(
    calibration: list[dict[str, object]],
    eval_eps: list[dict[str, object]],
    labels: dict[tuple[str, int], dict[str, object]],
    name: str,
    head: object | None,
    budget: float,
) -> dict[str, object]:
    everyone = calibration + eval_eps
    series = series_map(everyone, name, head)
    calib_pairs = [(series[episode_key(ep)], bool(ep["success"])) for ep in calibration]
    peaks = np.array(
        [
            float(np.max(scores)) if len(scores) else 0.0
            for scores, success in calib_pairs
            if success
        ]
    )
    baseline_threshold = calibrate_threshold(peaks, budget)
    rule = select_persistence_rule(calib_pairs, budget=budget)
    baseline = evaluate_persistence_rule(eval_eps, series, labels, baseline_threshold, hold=1)
    return {
        "baseline_single_crossing": baseline,
        "persistence": {
            "calibration_detection": rule.calibration_detection,
            "calibration_false_stop": rule.calibration_false_stop,
            "calibration_median_flag_step": rule.calibration_median_flag_step,
            **evaluate_persistence_rule(eval_eps, series, labels, rule.threshold, rule.hold),
        },
    }


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


def _seconds(steps: float | None) -> str:
    if steps is None:
        return "n/a"
    return f"{steps / FPS:+.2f} s"


def _print_block(name: str, block: dict[str, object]) -> None:
    print(f"\n{name}")
    pairs = (
        ("single crossing", "baseline_single_crossing"),
        ("persistence", "persistence"),
    )
    for label, key in pairs:
        row = block[key]
        assert isinstance(row, dict)
        det = row["detection_rate"]
        fs = row["false_stop_rate"]
        print(
            f"  {label}: hold={row['hold']} line={float(row['threshold']):.4f} "
            f"det={row['caught']}/{row['n_eval_failure']} ({float(det):.3f}) "
            f"false-stop={row['false_stops']}/{row['n_eval_success']} ({float(fs):.3f})"
        )
        print(
            "    lead vs onset start/end: "
            f"mean {row['mean_lead_vs_onset_start']} / {row['mean_lead_vs_onset_end']} steps "
            f"({_seconds(_as_float(row['mean_lead_vs_onset_start']))} / "
            f"{_seconds(_as_float(row['mean_lead_vs_onset_end']))}), "
            f"median {row['median_lead_vs_onset_start']} / {row['median_lead_vs_onset_end']}; "
            f"before {row['before_onset']}, inside {row['inside_onset']}, "
            f"after {row['after_onset']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/guard/persistence/report.json"),
    )
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = load_jsonl(args.traces)
    labels = labels_by_key(args.labels)
    calibration, eval_eps = split_halves(
        spatial_pool(traces), eval_fraction=args.eval_fraction, seed=args.seed
    )

    features = np.array(
        [[float(row["scores"][name]) for name in TRUST_FEATURES] for row in calibration]  # type: ignore[index]
    )
    failure = np.array([0.0 if row["success"] else 1.0 for row in calibration])
    head = fit_trust_head(features, failure, TRUST_FEATURES)

    report: dict[str, object] = {
        "budget": args.budget,
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "fps": FPS,
        "note": (
            "Positive lead time means the stop landed before that end of the "
            "onset range. Onset was not used to choose the rule."
        ),
        "signals": {},
    }
    signals = report["signals"]
    assert isinstance(signals, dict)
    # Every signal gets its own stop rule; only TRUST_FEATURES feed the head.
    names = [*TRACE_SIGNALS, "trust_running"]
    for name in names:
        block = _fit_one(calibration, eval_eps, labels, name, head, args.budget)
        signals[name] = block
        _print_block(name, block)

    object_eps = [row for row in traces if str(row["shard"]) == OBJECT_SHARD]
    frozen: dict[str, object] = {}
    for name in names:
        block = signals[name]
        assert isinstance(block, dict)
        chosen = block["persistence"]
        assert isinstance(chosen, dict)
        series = series_map(object_eps, name, head)
        frozen[name] = evaluate_persistence_rule(
            object_eps,
            series,
            labels,
            float(chosen["threshold"]),
            int(chosen["hold"]),
        )
        row = frozen[name]
        assert isinstance(row, dict)
        print(
            f"\nobject_clean frozen {name}: det={row['caught']}/{row['n_eval_failure']} "
            f"false-stop={row['false_stops']}/{row['n_eval_success']}"
        )
    report["object_clean_frozen"] = frozen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
