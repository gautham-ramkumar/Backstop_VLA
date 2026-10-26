"""Re-selection pre-test (docs/recovery.md, "The cheap pre-test").

Compares `backstop.recover.reselect` metrics inside each failure's reviewed
onset window with good runs at the same steps. See that module for what the
three metrics mean and why they decide whether re-selection is worth
building. No GPU, no fitting. One episode's samples in memory at a time.

    PYTHONPATH=src python scripts/pretest_reselect.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.corpus.episode import EpisodeKey
from backstop.corpus.layout import ShardLayout
from backstop.guard.extract import iter_episode_samples
from backstop.guard.traces import EXCLUDED_SHARDS, LabelView, load_labels
from backstop.recover.reselect import RESELECT_METRICS, reselect_metrics

Pooled = dict[str, list[float]]


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_steps": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def onset_step_mask(labels: dict[EpisodeKey, LabelView]) -> np.ndarray:
    """True at every step index inside at least one failure's onset window."""
    mask = np.zeros(max(label.n_steps for label in labels.values()), dtype=bool)
    for label in labels.values():
        if label.onset_range is not None and not label.success:
            mask[label.onset_range[0] : label.onset_range[1] + 1] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/recovery/pretest.json"))
    args = parser.parse_args()

    labels = load_labels(args.labels)
    matched = onset_step_mask(labels)
    shards = sorted({shard for shard, _ in labels} - EXCLUDED_SHARDS)
    layout = ShardLayout(lerobot_root=args.lerobot_root, eval_root=args.eval_root)

    failure: Pooled = {name: [] for name in RESELECT_METRICS}
    success: Pooled = {name: [] for name in RESELECT_METRICS}
    failure_spread: list[float] = []
    n_success = 0
    for shard in shards:
        for episode, samples in iter_episode_samples(layout, shard):
            label = labels.get(EpisodeKey(shard, episode))
            if label is None:
                continue
            metrics = reselect_metrics(samples)
            if label.success:
                # Good runs at matched steps: those inside any failure's onset window.
                n_success += 1
                mask = matched[: samples.shape[0]]
                for name, values in metrics.items():
                    success[name].extend(values[mask].tolist())
                continue
            onset = label.onset_range
            if onset is None:
                continue
            window = slice(onset[0], onset[1] + 1)
            for name, values in metrics.items():
                failure[name].extend(values[window].tolist())
            failure_spread.append(float(metrics["spread"][window].mean()))

    success_p90 = float(np.percentile(success["spread"], 90))
    report = {
        "note": (
            "Reviewed failure onset windows vs good runs at matched steps. "
            "outlier_ratio near 1 means chunk 0 is just one more sample."
        ),
        "n_failure_episodes": len(failure_spread),
        "n_success_episodes": n_success,
        "failure_onset": {name: _summary(values) for name, values in failure.items()},
        "success_matched": {name: _summary(values) for name, values in success.items()},
        "failure_episodes_with_spread_above_success_p90": float(
            np.mean(np.asarray(failure_spread) > success_p90)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name in RESELECT_METRICS:
        f, s = report["failure_onset"][name], report["success_matched"][name]
        print(
            f"{name:14s} failure onset mean {f['mean']:.3f} (median {f['median']:.3f})"
            f" | good runs mean {s['mean']:.3f} (median {s['median']:.3f})"
        )
    share = report["failure_episodes_with_spread_above_success_p90"]
    print(
        f"failures whose onset spread beats the good-run 90th percentile: "
        f"{share:.0%} of {len(failure_spread)}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
