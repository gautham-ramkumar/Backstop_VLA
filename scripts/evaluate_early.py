"""Try the paper signals that can be scored from the recordings already on disk.

Action-chunk magnitude and speed-adjusted overlap come from the stored action
chunks. The time-varying line subtracts the typical success score at each
step, fit on the calibration half only, then uses the same persistence stop.
Onset ranges are read only to measure lead time on the held-out half.

Usage:
    PYTHONPATH=src python scripts/evaluate_early.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.guard.combine import TRUST_FEATURES, apply_trust_head, fit_trust_head
from backstop.guard.extract import load_chunk_series
from backstop.guard.scoring import (
    evaluate_persistence_rule,
    fit_and_evaluate_persistence,
    format_rule_row,
)
from backstop.guard.signals import episode_score
from backstop.guard.timevary import subtract_profile, success_profile
from backstop.guard.traces import (
    OBJECT_SHARD,
    episode_key,
    labels_by_key,
    load_jsonl,
    running_trust,
    spatial_pool,
    split_halves,
)


def _running(columns: list[np.ndarray], head: object) -> np.ndarray:
    matrix = np.stack(columns, axis=1)
    running = np.maximum.accumulate(matrix, axis=0)
    return apply_trust_head(head, running)  # type: ignore[arg-type]


def _excess(
    series: dict[tuple[str, int], np.ndarray],
    calibration: list[dict[str, object]],
    quantile: float,
) -> dict[tuple[str, int], np.ndarray]:
    successes = [series[episode_key(episode)] for episode in calibration if episode["success"]]
    profile = success_profile(successes, quantile=quantile)
    return {key: subtract_profile(values, profile) for key, values in series.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/guard/persistence/early_report.json"),
    )
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = load_jsonl(args.traces)
    labels = labels_by_key(args.labels)
    shards = sorted({str(row["shard"]) for row in traces})
    print(f"reading action chunks for {len(shards)} shards")
    chunks = load_chunk_series(shards, lerobot_root=args.lerobot_root, eval_root=args.eval_root)

    calibration, eval_eps = split_halves(
        spatial_pool(traces), eval_fraction=args.eval_fraction, seed=args.seed
    )

    def _have(episode: dict[str, object]) -> bool:
        return episode_key(episode) in chunks

    missing = [episode for episode in calibration + eval_eps if not _have(episode)]
    if missing:
        raise RuntimeError(f"{len(missing)} calibration/eval episodes have no action chunk")

    base_names = TRUST_FEATURES
    extra_names = ("action_chunk_magnitude", "velocity_normalized_consistency")

    def _columns(episode: dict[str, object], names: tuple[str, ...]) -> list[np.ndarray]:
        per_step = episode["per_step"]
        if not isinstance(per_step, dict):
            raise ValueError("trace is missing per_step")
        columns = [
            np.asarray(per_step[name], dtype=np.float64) for name in names if name in per_step
        ]
        extra = chunks[episode_key(episode)]
        for name in names:
            if name in extra:
                columns.append(extra[name])
        return columns

    features = []
    failure = []
    for episode in calibration:
        per_step = episode["per_step"]
        assert isinstance(per_step, dict)
        row_features = [float(episode["scores"][name]) for name in base_names]  # type: ignore[index]
        extra = chunks[episode_key(episode)]
        row_features.extend(episode_score(extra[name]) for name in extra_names)
        features.append(row_features)
        failure.append(0.0 if episode["success"] else 1.0)
    head4 = fit_trust_head(
        np.asarray(
            [[float(episode["scores"][name]) for name in base_names] for episode in calibration],  # type: ignore[index]
            dtype=np.float64,
        ),
        np.asarray([0.0 if episode["success"] else 1.0 for episode in calibration]),
        base_names,
    )
    head6 = fit_trust_head(
        np.asarray(features, dtype=np.float64),
        np.asarray(failure),
        base_names + extra_names,
    )

    def _map_from(builder) -> dict[tuple[str, int], np.ndarray]:
        return {
            episode_key(episode): builder(episode)
            for episode in traces
            if episode_key(episode) in chunks
        }

    banks: dict[str, dict[tuple[str, int], np.ndarray]] = {
        "action_chunk_magnitude": _map_from(
            lambda ep: chunks[episode_key(ep)]["action_chunk_magnitude"]
        ),
        "velocity_normalized_consistency": _map_from(
            lambda ep: chunks[episode_key(ep)]["velocity_normalized_consistency"]
        ),
        "action_chunk_disagreement": _map_from(
            lambda ep: np.asarray(ep["per_step"]["action_chunk_disagreement"], dtype=np.float64)
        ),
        "trust_running": _map_from(lambda ep: running_trust(ep["per_step"], head4)),
        "trust_plus_chunks": _map_from(
            lambda ep: _running(_columns(ep, base_names + extra_names), head6)
        ),
    }

    report: dict[str, object] = {"signals": {}}
    signals = report["signals"]
    assert isinstance(signals, dict)
    print("lead is onset minus stop, in steps. Positive is before that end of the onset.")
    for name, series in banks.items():
        fitted = fit_and_evaluate_persistence(calibration, eval_eps, series, labels, args.budget)
        signals[name] = fitted
        print(format_rule_row(name, fitted))
        for quantile in (0.5, 0.9):
            excess = _excess(series, calibration, quantile)
            label = f"{name} above success q{quantile:.1f}"
            fitted_excess = fit_and_evaluate_persistence(
                calibration, eval_eps, excess, labels, args.budget
            )
            signals[label] = fitted_excess
            print(format_rule_row(label, fitted_excess))

    object_eps = [
        row for row in traces if str(row["shard"]) == OBJECT_SHARD and episode_key(row) in chunks
    ]
    print("\nobject_clean with the spatial rule, frozen")
    frozen: dict[str, object] = {}
    for name, block in signals.items():
        assert isinstance(block, dict)
        source = name.split(" above ")[0]
        series = banks[source]
        if name.startswith(source) and " above " in name:
            quantile = float(name.rsplit("q", 1)[1])
            series = _excess(banks[source], calibration, quantile)
        row = evaluate_persistence_rule(
            object_eps,
            series,
            labels,
            float(block["threshold"]),
            int(block["hold"]),
        )
        frozen[name] = row
        det = f"{row['caught']}/{row['n_eval_failure']}"
        fs = f"{row['false_stops']}/{row['n_eval_success']}"
        print(f"  {name:44} det={det} fs={fs}")
    report["object_clean_frozen"] = frozen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
