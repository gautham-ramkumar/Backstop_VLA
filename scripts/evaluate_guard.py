#!/usr/bin/env python
"""Score the guard's signal on corpus A and write the traces every other script reads.

Reads corpus A, scores action-chunk disagreement per step, and writes:

- `artifacts/guard/traces.jsonl`: one row per episode with its per-step
  series. `evaluate_stopwatch.py` (the headline), `freeze_guard.py` and
  `evaluate_presence.py` all start from it.
- The secondary single-split report (ADR-001, ADR-004): calibrated on one
  stratified half of corpus A, measured on the other, with a bootstrap
  interval and 5-fold cross-validation.
- The unseen suite (`object_clean`), scored with the frozen spatial line.

`language_absent` stays out of the pooled number and is reported on its own.
The other signals and the trust head that this script used to score are in
docs/experiments.md.

    uv run python scripts/evaluate_guard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.corpus.layout import ShardLayout
from backstop.eval.report import write_secondary_report
from backstop.guard.calibrate import bootstrap_rates
from backstop.guard.calibrate import evaluate as eval_metrics
from backstop.guard.extract import SIGNAL, EpisodeSignal, load_corpus_signals
from backstop.guard.summarize import report_signal
from backstop.guard.traces import as_float
from backstop.schema import DETECTION_METRIC, load_metric_contract

QUARANTINED_SHARDS = frozenset({"language_absent"})
# Held-out suites: scored with the frozen spatial calibration, never fit on.
HOLDOUT_SHARDS = frozenset({"object_clean"})


def _write_traces(path: Path, episodes: list[EpisodeSignal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for episode in episodes:
            per_step = {
                name: [float(value) for value in scores]
                for name, scores in episode.per_step.items()
            }
            row = {
                "shard": episode.shard,
                "episode_index": episode.episode_index,
                "success": episode.success,
                "n_steps": episode.n_steps,
                "task": episode.task,
                "scores": episode.scores,
                "per_step": per_step,
            }
            handle.write(json.dumps(row) + "\n")


def _score_holdout(
    episodes: list[EpisodeSignal], scores: list[float], threshold: float, *, seed: int
) -> dict[str, object]:
    """Frozen-threshold numbers for a held-out shard. Every episode is eval."""
    success = np.asarray(
        [score for score, ep in zip(scores, episodes, strict=True) if ep.success],
        dtype=np.float64,
    )
    failure = np.asarray(
        [score for score, ep in zip(scores, episodes, strict=True) if not ep.success],
        dtype=np.float64,
    )
    metrics = eval_metrics(success, failure, threshold)
    per_task: dict[str, dict[str, int]] = {}
    for score, episode in zip(scores, episodes, strict=True):
        row = per_task.setdefault(
            episode.task, {"n_success": 0, "n_failure": 0, "n_false_stops": 0, "n_detected": 0}
        )
        flagged = score >= threshold
        if episode.success:
            row["n_success"] += 1
            row["n_false_stops"] += int(flagged)
        else:
            row["n_failure"] += 1
            row["n_detected"] += int(flagged)
    return {
        "threshold": threshold,
        "detection_rate": metrics.detection_rate,
        "false_stop_rate": metrics.false_stop_rate,
        "n_success": metrics.n_eval_success,
        "n_failure": metrics.n_eval_failure,
        "n_detected": metrics.n_detected,
        "n_false_stops": metrics.n_false_stops,
        "bootstrap": bootstrap_rates(success, failure, threshold, seed=seed),
        "per_task": per_task,
    }


def _print_holdout(signal: str, shard: str, row: dict[str, object]) -> None:
    bootstrap = row["bootstrap"]
    assert isinstance(bootstrap, dict)
    print(
        f"  {shard} (held out, frozen threshold): detection={row['detection_rate']:.3f} "
        f"({row['n_detected']}/{row['n_failure']}), "
        f"false_stop={row['false_stop_rate']:.3f} ({row['n_false_stops']}/{row['n_success']}), "
        f"95% detection interval "
        f"[{bootstrap['detection_rate_low']:.3f}, {bootstrap['detection_rate_high']:.3f}]"
    )


def _print_report(values: dict[str, object]) -> None:
    per_family = values["per_family"]
    assert isinstance(per_family, dict)
    width = max((len(str(shard)) for shard in per_family), default=8) + 2
    bootstrap = values["bootstrap"]
    assert isinstance(bootstrap, dict)
    print(
        f"{values['signal']}: detection={values['detection_rate']:.3f} "
        f"({values['n_detected']}/{values['n_eval_failure']}), "
        f"false_stop={values['false_stop_rate']:.3f} "
        f"({values['n_false_stops']}/{values['n_eval_success']}), "
        f"95% detection interval "
        f"[{bootstrap['detection_rate_low']:.3f}, {bootstrap['detection_rate_high']:.3f}]"
    )
    print(f"{'shard'.ljust(width)}n_succ  n_fail  false_stop  detection")
    for shard, row in per_family.items():
        assert isinstance(row, dict)
        print(
            f"{str(shard).ljust(width)}{row['n_eval_success']:>6}  {row['n_eval_failure']:>6}  "
            f"{row['false_stop_rate']:>10.3f}  {row['detection_rate']:>9.3f}"
        )


def _markdown(
    reports: list[dict[str, object]],
    holdout: dict[str, dict[str, dict[str, object]]],
) -> str:
    lines = [
        "## Detection at 5%, one split (ADR-001, secondary since ADR-005)",
        "",
        "Detection rate at a 5% false-stop budget, calibrated on one",
        "stratified half of corpus A and measured on the other (ADR-004).",
        "`language_absent` is reported on its own and is not in the pooled number.",
        "The table is the single-crossing rule: stop if the episode peak crosses",
        "the line.",
        'One split (seed 0) only. "Against a stopwatch" below runs the headline',
        "rule over 20 splits.",
        "LIBERO-Object is not in this table. It is the unseen suite, scored",
        "below with the frozen spatial threshold. LIBERO-plus is not recorded.",
        "",
        "| signal | detection | 95% interval | false-stop | 5-fold mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for values in reports:
        bootstrap = values.get("bootstrap")
        cross = values.get("cross_validation")
        interval = "—"
        folded = "—"
        if isinstance(bootstrap, dict):
            interval = (
                f"{bootstrap['detection_rate_low']:.3f}–{bootstrap['detection_rate_high']:.3f}"
            )
        if isinstance(cross, dict):
            folded = f"{cross['detection_rate_mean']:.3f} ± {cross['detection_rate_std']:.3f}"
        lines.append(
            f"| {values['signal']} | {values['detection_rate']:.3f} | {interval} | "
            f"{values['false_stop_rate']:.3f} | {folded} |"
        )
    lines.extend(
        [
            "",
            "Per-family counts are in `artifacts/guard/action_chunk_disagreement/guard_eval.json`.",
            "A family cell with a handful of eval failures is a direction, not a rate.",
            "",
        ]
    )
    for shard, by_signal in holdout.items():
        sample = next(iter(by_signal.values()))
        lines.extend(
            [
                f"## Unseen suite: `{shard}`",
                "",
                "The threshold was fit on spatial only and",
                f"applied here unchanged. All {sample['n_success']} successes and",
                f"{sample['n_failure']} failures are eval episodes.",
                "",
                "| signal | detection | 95% interval | false-stop |",
                "|---|---:|---:|---:|",
            ]
        )
        for signal, row in by_signal.items():
            bootstrap = row["bootstrap"]
            assert isinstance(bootstrap, dict)
            lines.append(
                f"| {signal} | {row['detection_rate']:.3f} "
                f"({row['n_detected']}/{row['n_failure']}) | "
                f"{bootstrap['detection_rate_low']:.3f}–{bootstrap['detection_rate_high']:.3f} | "
                f"{row['false_stop_rate']:.3f} ({row['n_false_stops']}/{row['n_success']}) |"
            )
        lines.extend(
            [
                "",
                f"Per-task counts are in `artifacts/guard/holdout/{shard}.json`.",
                "",
            ]
        )
    return "\n".join(lines)


BEGIN_MARKER = "<!-- BEGIN generated by scripts/evaluate_guard.py -->"
END_MARKER = "<!-- END generated by scripts/evaluate_guard.py -->"


def _splice(path: Path, generated: str) -> str:
    """Replace only the generated block, keeping every hand-written section.

    A report that exists without the markers is refused rather than
    overwritten: the sections after this block are written by hand.
    """
    block = f"{BEGIN_MARKER}\n{generated.rstrip()}\n{END_MARKER}"
    if not path.exists():
        return f"# Guard evaluation\n\n{block}\n"
    current = path.read_text()
    start = current.find(BEGIN_MARKER)
    end = current.find(END_MARKER)
    if start < 0 or end < start:
        raise SystemExit(
            f"{path} has no {BEGIN_MARKER!r} / {END_MARKER!r} pair. Add them around "
            "the generated tables, or pass --report-path to write elsewhere."
        )
    return current[:start] + block + current[end + len(END_MARKER) :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--privileged-root", type=Path, default=Path("data/privileged"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/guard"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/evaluation.md"))
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    contract = load_metric_contract()
    by_shard = load_corpus_signals(
        ShardLayout(args.lerobot_root, args.privileged_root, args.eval_root)
    )
    if not by_shard:
        print(f"no shards found under {args.privileged_root}")
        return 1
    for shard in sorted(HOLDOUT_SHARDS):
        if (args.privileged_root / shard).is_dir() and shard not in by_shard:
            print(f"held-out shard {shard!r} is recorded but did not load (run.json or dataset)")
            return 1

    quarantined = [
        ep for shard, eps in by_shard.items() if shard in QUARANTINED_SHARDS for ep in eps
    ]
    excluded = QUARANTINED_SHARDS | HOLDOUT_SHARDS
    pooled = [ep for shard, eps in by_shard.items() if shard not in excluded for ep in eps]
    held_out = {shard: eps for shard, eps in by_shard.items() if shard in HOLDOUT_SHARDS}
    holdout: dict[str, dict[str, dict[str, object]]] = {shard: {} for shard in held_out}
    values = report_signal(
        pooled,
        SIGNAL,
        metric=contract,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    threshold = as_float(values["threshold"])
    quarantine_report = {}
    if quarantined:
        metrics = eval_metrics(
            np.asarray([ep.scores[SIGNAL] for ep in quarantined if ep.success], dtype=np.float64),
            np.asarray(
                [ep.scores[SIGNAL] for ep in quarantined if not ep.success], dtype=np.float64
            ),
            threshold,
        )
        quarantine_report["language_absent"] = {
            "n_failure": metrics.n_eval_failure,
            "detection_rate": metrics.detection_rate,
            "note": (
                "0% success by construction (Week 2, quarantined) -- excluded "
                "from calibration and pooled detection."
            ),
        }
    values["quarantined"] = quarantine_report
    for shard, episodes in held_out.items():
        holdout[shard][SIGNAL] = _score_holdout(
            episodes, [episode.scores[SIGNAL] for episode in episodes], threshold, seed=args.seed
        )
    report_path = write_secondary_report(
        output_dir=args.output_dir / SIGNAL,
        metric_name=DETECTION_METRIC,
        values=values,
        metric=contract,
    )
    _print_report(values)
    for shard, row in quarantine_report.items():
        print(f"  {shard} quarantined detection={row['detection_rate']:.3f}")
    for shard, by_signal in holdout.items():
        _print_holdout(SIGNAL, shard, by_signal[SIGNAL])
    print(f"  wrote {report_path}")
    reports = [values]

    for shard, by_signal in holdout.items():
        holdout_path = args.output_dir / "holdout" / f"{shard}.json"
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        holdout_path.write_text(
            json.dumps(
                {
                    "metric": DETECTION_METRIC,
                    "role": "secondary",
                    "false_stop_budget": contract.false_stop_budget,
                    "shard": shard,
                    "calibrated_on": "spatial (pooled, quarantine and holdout excluded)",
                    "signals": by_signal,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {holdout_path}")

    all_episodes = [episode for episodes in by_shard.values() for episode in episodes]
    trace_path = args.output_dir / "traces.jsonl"
    _write_traces(trace_path, all_episodes)
    print(f"wrote {trace_path}")

    text = _splice(args.report_path, _markdown(reports, holdout))
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(text)
    print(f"wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
