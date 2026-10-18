#!/usr/bin/env python
"""Evaluate every guard signal that does not need the taxonomy.

Reads corpus A, scores action-chunk disagreement, temporal self-consistency,
and progress monitoring from the dataset, and the visual precondition from
the first agentview frame (OWLv2 detector score for the instructed object).
Each signal is calibrated on one stratified half and measured on the other
(ADR-004). A logistic trust head combines the
signals, fit on the calibration half only. Bootstrap intervals and 5-fold
cross-validation are written beside the point estimates.

Lead time is computed by the later scripts, not here.
`language_absent` stays out of the pooled number.

`object_clean` (LIBERO-Object) is the unseen suite. It never touches a
threshold or the trust head: every signal and the head are fit on spatial
only, then applied frozen to it, and it is reported on its own.

    uv run python scripts/evaluate_guard.py
    uv run python scripts/evaluate_guard.py --skip-visual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.eval.report import write_secondary_report
from backstop.guard.calibrate import bootstrap_rates
from backstop.guard.calibrate import evaluate as eval_metrics
from backstop.guard.extract import (
    PARQUET_SIGNALS,
    EpisodeSignal,
    load_corpus_signals,
)
from backstop.guard.signals import (
    episode_score,
    instructed_object_phrase,
    visual_precondition,
)
from backstop.guard.summarize import report_signal, report_trust_head
from backstop.guard.traces import as_float
from backstop.label.extract import resolve_dataset_root
from backstop.schema import DETECTION_METRIC, load_metric_contract

QUARANTINED_SHARDS = frozenset({"language_absent"})
# Held-out suites: scored with the frozen spatial calibration, never fit on.
HOLDOUT_SHARDS = frozenset({"object_clean"})


def _attach_visual(
    by_shard: dict[str, list[EpisodeSignal]],
    *,
    lerobot_root: Path,
    eval_root: Path,
) -> None:
    from backstop.guard.vision import detector_scores, first_agentview_frame  # noqa: PLC0415

    images = []
    phrases = []
    owners: list[tuple[str, int]] = []
    for shard, episodes in by_shard.items():
        dataset_root, _provenance = resolve_dataset_root(eval_root, shard, lerobot_root)
        for episode in episodes:
            images.append(first_agentview_frame(dataset_root, episode.episode_index))
            phrases.append(instructed_object_phrase(episode.task))
            owners.append((shard, episode.episode_index))
    print(f"visual precondition: embedding {len(images)} first frames")
    # `visual_precondition` negates a presence score, so this stays "higher
    # means more likely missing". One line is fit on spatial successes, which
    # all name the black bowl; the per-object gate is scripts/evaluate_presence.py.
    scores = detector_scores(images, phrases)
    score_of = {owners[i]: float(scores[i]) for i in range(len(owners))}
    for shard, episodes in by_shard.items():
        updated = []
        for episode in episodes:
            value = score_of[(shard, episode.episode_index)]
            per_step = dict(episode.per_step)
            per_step["visual_precondition"] = visual_precondition(np_full(episode.n_steps, value))
            episode_scores = dict(episode.scores)
            episode_scores["visual_precondition"] = episode_score(per_step["visual_precondition"])
            updated.append(
                EpisodeSignal(
                    shard=episode.shard,
                    episode_index=episode.episode_index,
                    success=episode.success,
                    n_steps=episode.n_steps,
                    task=episode.task,
                    scores=episode_scores,
                    per_step=per_step,
                    perturbation=episode.perturbation,
                    perturbation_strength=episode.perturbation_strength,
                )
            )
        by_shard[shard] = updated


def np_full(n_steps: int, value: float) -> np.ndarray:
    return np.full(n_steps, value, dtype=np.float64)


def _write_traces(
    path: Path, episodes: list[EpisodeSignal], trust: dict[tuple[str, int], float]
) -> None:
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
                "trust": trust.get((episode.shard, episode.episode_index)),
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
    visual_note: str,
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
        'One split (seed 0) only: see "Against a stopwatch" for the same rules',
        "over 20 splits, where the false-stop rate averages above 5%.",
        "LIBERO-Object is not in this table. It is the unseen suite, scored",
        "below with the frozen spatial thresholds. LIBERO-plus is not recorded.",
        "",
        visual_note,
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
            "Per-family counts are in each `artifacts/guard/<signal>/guard_eval.json`.",
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
                "Every threshold and the trust head were fit on spatial only and",
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
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args(argv)

    contract = load_metric_contract()
    by_shard = load_corpus_signals(
        lerobot_root=args.lerobot_root,
        privileged_root=args.privileged_root,
        eval_root=args.eval_root,
    )
    if not by_shard:
        print(f"no shards found under {args.privileged_root}")
        return 1
    for shard in sorted(HOLDOUT_SHARDS):
        if (args.privileged_root / shard).is_dir() and shard not in by_shard:
            print(f"held-out shard {shard!r} is recorded but did not load (run.json or dataset)")
            return 1

    visual_note = "Visual precondition was not run (`--skip-visual`)."
    if not args.skip_visual:
        try:
            _attach_visual(by_shard, lerobot_root=args.lerobot_root, eval_root=args.eval_root)
            visual_note = (
                "Visual precondition is the negated OWLv2 detector score for the "
                "instructed object in the first agentview frame. One line, set on "
                "spatial successes; the per-object gate is reported further down."
            )
        except Exception as exc:  # noqa: BLE001 - recorded, and the other signals still report
            visual_note = f"Visual precondition failed to run ({exc}). The other signals did."
            print(visual_note)

    signal_names: list[str] = list(PARQUET_SIGNALS)
    sample = next(iter(by_shard.values()))[0]
    if "visual_precondition" in sample.scores:
        signal_names.append("visual_precondition")

    quarantined = [
        ep for shard, eps in by_shard.items() if shard in QUARANTINED_SHARDS for ep in eps
    ]
    excluded = QUARANTINED_SHARDS | HOLDOUT_SHARDS
    pooled = [ep for shard, eps in by_shard.items() if shard not in excluded for ep in eps]
    held_out = {shard: eps for shard, eps in by_shard.items() if shard in HOLDOUT_SHARDS}
    holdout: dict[str, dict[str, dict[str, object]]] = {shard: {} for shard in held_out}
    reports: list[dict[str, object]] = []
    for name in signal_names:
        values = report_signal(
            pooled,
            name,
            metric=contract,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
        quarantine_report = {}
        if quarantined:
            metrics = eval_metrics(
                np.asarray(
                    [episode.scores[name] for episode in quarantined if episode.success],
                    dtype=np.float64,
                ),
                np.asarray(
                    [episode.scores[name] for episode in quarantined if not episode.success],
                    dtype=np.float64,
                ),
                as_float(values["threshold"]),
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
            holdout[shard][name] = _score_holdout(
                episodes,
                [episode.scores[name] for episode in episodes],
                as_float(values["threshold"]),
                seed=args.seed,
            )
        report_path = write_secondary_report(
            output_dir=args.output_dir / name,
            metric_name=DETECTION_METRIC,
            values=values,
            metric=contract,
        )
        _print_report(values)
        for shard, row in quarantine_report.items():
            print(f"  {shard} quarantined detection={row['detection_rate']:.3f}")
        for shard, by_signal in holdout.items():
            _print_holdout(name, shard, by_signal[name])
        print(f"  wrote {report_path}")
        reports.append(values)

    trust_of: dict[tuple[str, int], float] = {}
    from backstop.guard.combine import TRUST_FEATURES  # noqa: PLC0415

    # The visual precondition keeps its own row above but is not a head input.
    head_names = [name for name in signal_names if name in TRUST_FEATURES]
    if len(head_names) >= 2:
        trust_values, head = report_trust_head(
            pooled,
            tuple(head_names),
            metric=contract,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
        trust_path = write_secondary_report(
            output_dir=args.output_dir / "trust_head",
            metric_name=DETECTION_METRIC,
            values=trust_values,
            metric=contract,
        )
        _print_report(trust_values)
        print(f"  wrote {trust_path}")
        reports.append(trust_values)
        from backstop.guard.combine import apply_trust_head  # noqa: PLC0415

        matrix = [[episode.scores[name] for name in head_names] for episode in pooled]
        probabilities = apply_trust_head(head, np.asarray(matrix, dtype=np.float64))
        for episode, probability in zip(pooled, probabilities, strict=True):
            trust_of[(episode.shard, episode.episode_index)] = float(probability)
        for shard, episodes in held_out.items():
            held_matrix = [[episode.scores[name] for name in head_names] for episode in episodes]
            held_trust = apply_trust_head(head, np.asarray(held_matrix, dtype=np.float64))
            for episode, probability in zip(episodes, held_trust, strict=True):
                trust_of[(episode.shard, episode.episode_index)] = float(probability)
            holdout[shard]["trust_head"] = _score_holdout(
                episodes,
                [float(value) for value in held_trust],
                as_float(trust_values["threshold"]),
                seed=args.seed,
            )
            _print_holdout("trust_head", shard, holdout[shard]["trust_head"])

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
    _write_traces(trace_path, all_episodes, trust_of)
    print(f"wrote {trace_path}")

    text = _splice(args.report_path, _markdown(reports, visual_note, holdout))
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(text)
    print(f"wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
