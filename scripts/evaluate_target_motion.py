"""Score the cup's own motion and measure lead time.

The cup's position comes from the simulator sidecar. ADR-003 forbids that as
a guard input, so this number is a ceiling: if seeing the cup move cannot
stop inside the pickup window, no camera guess of that motion will either.
The score at each step is how far the cup is behind the median successful
cup, fit on the calibration half only. The stop line is the same 5%
persistence rule. Distractor and the object suite use a different object
layout, so they are left out.

Usage:
    PYTHONPATH=src python scripts/evaluate_target_motion.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.guard.scoring import evaluate_persistence_rule, fit_persistence_rule
from backstop.guard.timevary import success_profile
from backstop.guard.traces import (
    EXCLUDED_SHARDS,
    episode_key,
    labels_by_key,
    load_jsonl,
    spatial_pool,
    split_halves,
)
from backstop.label.signals import (
    LIBERO_SPATIAL_SIM_STATE_WIDTH,
    motion_shortfall,
    target_displacement,
)
from backstop.record.privileged import load_episode

# Distractor adds an object, so its layout differs from the rest of spatial.
LAYOUT_EXCLUDED = EXCLUDED_SHARDS | {"distractor"}


def _displacement(root: Path, shard: str, episode: int) -> np.ndarray:
    state = load_episode(root, shard, episode)["sim_state"]
    if state.shape[1] != LIBERO_SPATIAL_SIM_STATE_WIDTH:
        raise ValueError(f"{shard} episode {episode} has sim state width {state.shape[1]}")
    return target_displacement(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument("--privileged-root", type=Path, default=Path("data/privileged"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/guard/persistence/target_motion.json"),
    )
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.5)
    args = parser.parse_args()

    traces = load_jsonl(args.traces)
    labels = labels_by_key(args.labels)
    calibration, eval_eps = split_halves(
        spatial_pool(traces, LAYOUT_EXCLUDED), eval_fraction=args.eval_fraction, seed=args.seed
    )

    moved = {
        episode_key(row): _displacement(args.privileged_root, *episode_key(row))
        for row in calibration + eval_eps
    }
    successes = [moved[episode_key(row)] for row in calibration if row["success"]]
    profile = success_profile(successes, quantile=args.quantile)
    scores = {key: motion_shortfall(values, profile) for key, values in moved.items()}
    rule = fit_persistence_rule(calibration, scores, args.budget)
    measured = evaluate_persistence_rule(eval_eps, scores, labels, rule.threshold, rule.hold)
    n_cal_success = sum(bool(row["success"]) for row in calibration)
    n_cal_failure = len(calibration) - n_cal_success
    print(
        f"spatial layout only, distractor left out. "
        f"calibration {n_cal_success} successes / {n_cal_failure} failures"
    )
    print("lead is onset minus stop, in steps. Positive is before that end of the onset.")
    lead_start = measured["median_lead_vs_onset_start"]
    lead_end = measured["median_lead_vs_onset_end"]
    print(
        f"target motion hold={rule.hold} line={rule.threshold:.3f} "
        f"det={measured['caught']}/{measured['n_eval_failure']} "
        f"fs={measured['false_stops']}/{measured['n_eval_success']} "
        f"lead {lead_start}/{lead_end} "
        f"before/in/after "
        f"{measured['before_onset']}/{measured['inside_onset']}/{measured['after_onset']}"
    )
    report = {
        "note": (
            "Reads the simulator's cup position. Not a guard input. "
            "Distractor and object_clean are excluded because their object layout differs."
        ),
        "quantile": args.quantile,
        "threshold": rule.threshold,
        "hold": rule.hold,
        "n_calibration_success": n_cal_success,
        "n_calibration_failure": n_cal_failure,
        "calibration_detection": rule.calibration_detection,
        "calibration_false_stop": rule.calibration_false_stop,
        "eval": measured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
