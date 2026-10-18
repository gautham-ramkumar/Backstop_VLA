"""Fit the per-step model on the calibration half and score the held-out half.

Each step is a row. The label is the episode outcome, copied onto every
step. Onset ranges are read only to measure lead time. The stop line is the
persistence rule on calibration successes, inside the 5% false-stop budget.

Usage:
    PYTHONPATH=src python scripts/train_step_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.guard.extract import load_chunk_series
from backstop.guard.internal import diagonal_distance, fit_diagonal, internal_path, load_internal
from backstop.guard.scoring import (
    evaluate_persistence_rule,
    fit_persistence_rule,
    format_rule_row,
)
from backstop.guard.step_model import (
    STEP_SIGNALS,
    fit_step_model,
    score_episode,
    step_feature_names,
    step_features,
)
from backstop.guard.traces import (
    OBJECT_SHARD,
    episode_key,
    labels_by_key,
    load_jsonl,
    spatial_pool,
    split_halves,
)


def _series_for(
    episode: dict[str, object],
    chunks: dict[tuple[str, int], dict[str, np.ndarray]],
    distance: np.ndarray,
) -> dict[str, np.ndarray]:
    per_step = episode["per_step"]
    if not isinstance(per_step, dict):
        raise ValueError("trace is missing per_step")
    series = {
        name: np.asarray(per_step[name], dtype=np.float64)
        for name in STEP_SIGNALS
        if name in per_step
    }
    extra = chunks[episode_key(episode)]
    series["action_chunk_magnitude"] = np.asarray(extra["action_chunk_magnitude"], dtype=np.float64)
    series["velocity_normalized_consistency"] = np.asarray(
        extra["velocity_normalized_consistency"], dtype=np.float64
    )
    series["internal_distance"] = np.asarray(distance, dtype=np.float64)
    length = {name: int(values.shape[0]) for name, values in series.items()}
    if len(set(length.values())) != 1:
        raise ValueError(f"{episode_key(episode)} series lengths differ: {length}")
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/guard/internal"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/guard/persistence/step_model_report.json"),
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
    object_eps = [row for row in traces if str(row["shard"]) == OBJECT_SHARD]
    needed = calibration + eval_eps + object_eps

    shards = sorted({str(row["shard"]) for row in needed})
    print(f"reading action chunks for {len(shards)} shards")
    chunks = load_chunk_series(shards, lerobot_root=args.lerobot_root, eval_root=args.eval_root)
    missing_chunk = [row for row in needed if episode_key(row) not in chunks]
    missing_internal = [
        row for row in needed if not internal_path(args.features, *episode_key(row)).is_file()
    ]
    if missing_chunk or missing_internal:
        raise SystemExit(
            f"{len(missing_chunk)} episodes have no action chunk and "
            f"{len(missing_internal)} have no internal vector"
        )

    success_rows = [
        load_internal(args.features, shard, episode)
        for shard, episode in (episode_key(row) for row in calibration if row["success"])
    ]
    mean, variance = fit_diagonal(np.concatenate(success_rows, axis=0))
    print(f"internal distance fit on {sum(row.shape[0] for row in success_rows)} success steps")

    names = step_feature_names()
    calibration_keys = {episode_key(row) for row in calibration}
    built: dict[tuple[str, int], np.ndarray] = {}
    training: list[tuple[np.ndarray, bool]] = []
    for episode in needed:
        distance = diagonal_distance(
            load_internal(args.features, *episode_key(episode)), mean, variance
        )
        features = step_features(_series_for(episode, chunks, distance), STEP_SIGNALS)
        built[episode_key(episode)] = features
        if episode_key(episode) in calibration_keys:
            training.append((features, not bool(episode["success"])))
    head = fit_step_model(training, names)
    scored = {key: score_episode(head, features) for key, features in built.items()}

    rule = fit_persistence_rule(calibration, scored, args.budget)
    measured = evaluate_persistence_rule(eval_eps, scored, labels, rule.threshold, rule.hold)
    row = {
        "threshold": rule.threshold,
        "hold": rule.hold,
        "calibration_detection": rule.calibration_detection,
        "calibration_false_stop": rule.calibration_false_stop,
        **measured,
    }
    print("lead is onset minus stop, in steps. Positive is before that end of the onset.")
    print(format_rule_row("step_model", row, width=20))
    ranked = sorted(
        zip(names, head.weights[:-1], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    print("largest coefficients (standardized features):")
    for name, weight in ranked[:8]:
        print(f"  {name:48} {float(weight):+.3f}")

    object_row = evaluate_persistence_rule(object_eps, scored, labels, rule.threshold, rule.hold)
    print(
        "object_clean frozen "
        f"det={object_row['caught']}/{object_row['n_eval_failure']} "
        f"fs={object_row['false_stops']}/{object_row['n_eval_success']}"
    )

    report = {
        "threshold": rule.threshold,
        "hold": rule.hold,
        "calibration_detection": rule.calibration_detection,
        "calibration_false_stop": rule.calibration_false_stop,
        "coefficients": {
            name: float(weight) for name, weight in zip(names, head.weights[:-1], strict=True)
        },
        "bias": float(head.weights[-1]),
        "eval": measured,
        "object_clean_frozen": object_row,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
