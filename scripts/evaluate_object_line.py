"""Set an object-suite cutoff from its own good episodes.

The scores stay the ones already used on spatial. Each signal is the episode
peak in the traces. The trust head is fit on the spatial calibration half
and then frozen. Only the cutoff changes: it is the 5% line of one half of
the object successes. False stops are counted on the other half. Every
object failure is scored, because failures do not enter the cutoff.

Spatial episodes are not rescored.

Usage:
    PYTHONPATH=src python scripts/evaluate_object_line.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.guard.calibrate import bootstrap_rates, calibrate_threshold, evaluate
from backstop.guard.combine import TRUST_FEATURES, apply_trust_head, fit_trust_head
from backstop.guard.traces import (
    OBJECT_SHARD,
    TRACE_SIGNALS,
    as_float,
    as_int,
    episode_key,
    load_jsonl,
    spatial_pool,
    split_halves,
)


# Every signal gets its own cutoff row; only TRUST_FEATURES feed the head.
def _peak(episode: dict[str, object], name: str) -> float:
    scores = episode["scores"]
    if not isinstance(scores, dict) or name not in scores:
        raise KeyError(f"{episode_key(episode)} is missing {name}")
    return float(scores[name])


def _score_trust(
    spatial_calibration: list[dict[str, object]], episodes: list[dict[str, object]]
) -> dict[tuple[str, int], float]:
    matrix = np.asarray(
        [[_peak(row, name) for name in TRUST_FEATURES] for row in spatial_calibration],
        dtype=np.float64,
    )
    labels = np.asarray([0.0 if row["success"] else 1.0 for row in spatial_calibration])
    head = fit_trust_head(matrix, labels, TRUST_FEATURES)
    scored = apply_trust_head(
        head,
        np.asarray(
            [[_peak(row, name) for name in TRUST_FEATURES] for row in episodes],
            dtype=np.float64,
        ),
    )
    return {episode_key(row): float(scored[i]) for i, row in enumerate(episodes)}


def _measure(
    name: str,
    scores: dict[tuple[str, int], float],
    line_episodes: list[dict[str, object]],
    check_episodes: list[dict[str, object]],
    failures: list[dict[str, object]],
    *,
    budget: float,
    seed: int,
) -> dict[str, object]:
    line_scores = np.asarray([scores[episode_key(row)] for row in line_episodes], dtype=np.float64)
    check_scores = np.asarray(
        [scores[episode_key(row)] for row in check_episodes], dtype=np.float64
    )
    failure_scores = np.asarray([scores[episode_key(row)] for row in failures], dtype=np.float64)
    threshold = calibrate_threshold(line_scores, budget)
    metrics = evaluate(check_scores, failure_scores, threshold)
    interval = bootstrap_rates(check_scores, failure_scores, threshold, seed=seed)
    return {
        "signal": name,
        "threshold": threshold,
        "n_line_success": len(line_episodes),
        "n_check_success": metrics.n_eval_success,
        "n_failure": metrics.n_eval_failure,
        "false_stops": metrics.n_false_stops,
        "false_stop_rate": metrics.false_stop_rate,
        "caught": metrics.n_detected,
        "detection_rate": metrics.detection_rate,
        "detection_low": interval["detection_rate_low"],
        "detection_high": interval["detection_rate_high"],
        "false_stop_low": interval["false_stop_rate_low"],
        "false_stop_high": interval["false_stop_rate_high"],
    }


def _line(row: dict[str, object]) -> str:
    return (
        f"{str(row['signal']):28} line={as_float(row['threshold']):7.3f} "
        f"det={as_int(row['caught']):2d}/{as_int(row['n_failure']):2d} "
        f"fs={as_int(row['false_stops']):2d}/{as_int(row['n_check_success']):2d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/guard/holdout/object_line.json"),
    )
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--eval-fraction", type=float, default=0.5)
    parser.add_argument("--spatial-eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = load_jsonl(args.traces)
    spatial_calibration, _ = split_halves(
        spatial_pool(traces), eval_fraction=args.spatial_eval_fraction, seed=args.seed
    )
    object_rows = [row for row in traces if str(row["shard"]) == OBJECT_SHARD]
    object_success = [row for row in object_rows if row["success"]]
    object_failure = [row for row in object_rows if not row["success"]]
    line_episodes, check_episodes = split_halves(
        object_success, eval_fraction=args.eval_fraction, seed=args.seed
    )

    trust = _score_trust(spatial_calibration, object_rows + spatial_calibration)
    spatial_trust_line = calibrate_threshold(
        np.asarray(
            [trust[episode_key(row)] for row in spatial_calibration if row["success"]],
            dtype=np.float64,
        ),
        args.budget,
    )
    banks: dict[str, dict[tuple[str, int], float]] = {
        name: {episode_key(row): _peak(row, name) for row in object_rows} for name in TRACE_SIGNALS
    }
    banks["trust_head"] = {episode_key(row): trust[episode_key(row)] for row in object_rows}

    print(
        f"object successes: {len(line_episodes)} set the cutoff, "
        f"{len(check_episodes)} are checked, {len(object_failure)} failures are scored"
    )
    print(f"spatial trust cutoff, frozen: {spatial_trust_line:.3f}")
    report: dict[str, object] = {
        "budget": args.budget,
        "seed": args.seed,
        "object_success_eval_fraction": args.eval_fraction,
        "n_line_success": len(line_episodes),
        "n_check_success": len(check_episodes),
        "n_failure": len(object_failure),
        "spatial_trust_threshold": spatial_trust_line,
        "signals": {},
    }
    signals = report["signals"]
    assert isinstance(signals, dict)
    for name, scores in banks.items():
        own = _measure(
            name,
            scores,
            line_episodes,
            check_episodes,
            object_failure,
            budget=args.budget,
            seed=args.seed,
        )
        frozen_scores = np.asarray([scores[episode_key(row)] for row in check_episodes])
        failure_scores = np.asarray([scores[episode_key(row)] for row in object_failure])
        frozen = evaluate(frozen_scores, failure_scores, spatial_trust_line)
        if name != "trust_head":
            frozen_threshold = calibrate_threshold(
                np.asarray([_peak(row, name) for row in spatial_calibration if row["success"]]),
                args.budget,
            )
            frozen = evaluate(frozen_scores, failure_scores, frozen_threshold)
        else:
            frozen_threshold = spatial_trust_line
        own["spatial_threshold"] = frozen_threshold
        own["spatial_cutoff_on_same_check"] = {
            "false_stops": frozen.n_false_stops,
            "caught": frozen.n_detected,
        }
        signals[name] = own
        print(_line(own), end="")
        print(
            f"   spatial cutoff on this same check: "
            f"det={frozen.n_detected}/{frozen.n_eval_failure} "
            f"fs={frozen.n_false_stops}/{frozen.n_eval_success}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
