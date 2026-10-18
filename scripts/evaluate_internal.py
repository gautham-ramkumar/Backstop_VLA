"""Score the saved last-layer vectors and measure lead time.

The mean and variance come from calibration successes only. A step scores
high when it sits far from that cloud. The stop rule is the same persistence
rule as the other signals. Onset is used only on the held-out half.

Usage:
    PYTHONPATH=src python scripts/evaluate_internal.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backstop.guard.internal import diagonal_distance, fit_diagonal, internal_path, load_internal
from backstop.guard.scoring import evaluate_persistence_rule, fit_persistence_rule
from backstop.guard.traces import (
    episode_key,
    labels_by_key,
    load_jsonl,
    spatial_pool,
    split_halves,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/guard/internal"))
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = load_jsonl(args.traces)
    labels = labels_by_key(args.labels)
    calibration, eval_eps = split_halves(
        spatial_pool(traces), eval_fraction=args.eval_fraction, seed=args.seed
    )

    missing = []
    for row in calibration + eval_eps:
        if not internal_path(args.features, *episode_key(row)).is_file():
            missing.append(row)
    if missing:
        raise SystemExit(
            f"{len(missing)} episodes still have no internal vector; extraction is not finished"
        )

    success_rows = [
        load_internal(args.features, shard, episode)
        for shard, episode in (episode_key(row) for row in calibration if row["success"])
    ]
    mean, variance = fit_diagonal(np.concatenate(success_rows, axis=0))
    series = {
        episode_key(row): diagonal_distance(
            load_internal(args.features, *episode_key(row)), mean, variance
        )
        for row in calibration + eval_eps
    }
    rule = fit_persistence_rule(calibration, series, args.budget)
    measured = evaluate_persistence_rule(eval_eps, series, labels, rule.threshold, rule.hold)
    print(
        f"internal hold={rule.hold} line={rule.threshold:.3f} "
        f"det={measured['caught']}/{measured['n_eval_failure']} "
        f"fs={measured['false_stops']}/{measured['n_eval_success']} "
        f"lead {measured['median_lead_vs_onset_start']}/{measured['median_lead_vs_onset_end']} "
        f"before/in/after "
        f"{measured['before_onset']}/{measured['inside_onset']}/{measured['after_onset']}"
    )


if __name__ == "__main__":
    main()
