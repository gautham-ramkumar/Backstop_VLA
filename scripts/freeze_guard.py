"""Freeze the headline guard for live use, held out from the clean shard.

`evaluate_stopwatch.py` refits the rule on every random split and saves only
summaries. A recovery run needs one rule, fixed before it starts. This fits
"action_chunk_disagreement OR stopwatch" once, from good runs only, at two
budgets:

- 5%, the ADR-005 headline budget;
- 20%, the rate docs/recovery.md argues for once a false alarm costs a lift
  and a re-approach instead of a lost success.

Calibration uses the successes of the perturbed families only. The clean
shard is left out entirely, because the recovery experiment runs on the clean
shard's 100 starting states, and its guard-off arm is the recorded clean run.
`language_paraphrase` is left out too: its 3 successes score unlike every
other family's (evaluate_stopwatch's perturbation check, AUROC 0.99).

For each rule it also predicts, offline, where the guard fires on every clean
episode. The recovery runner branches only those episodes; the rest finish
the same with the guard on. Each rule file records the traces hash and git
commit it came from.

    PYTHONPATH=src python scripts/freeze_guard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.guard.online import save_rule
from backstop.guard.scoring import fit_stopwatch_from_rows
from backstop.guard.stopwatch import stop_step
from backstop.guard.traces import load_traces, series_map, spatial_pool
from backstop.provenance import file_sha256, git_dirty, git_sha

SIGNAL = "action_chunk_disagreement"
TARGET_SHARD = "clean"
NOT_CALIBRATION = frozenset({TARGET_SHARD, "language_paraphrase"})
DEFAULT_BUDGETS = (0.05, 0.20)


def rule_name(budget: float) -> str:
    return f"rule_{round(budget * 100):02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/guard/frozen"))
    parser.add_argument("--budget", type=float, action="append", default=None)
    args = parser.parse_args()

    traces = load_traces(args.traces)
    calibration = [row for row in spatial_pool(traces) if row.shard not in NOT_CALIBRATION]
    target = sorted((row for row in traces if row.shard == TARGET_SHARD), key=lambda row: row.key)
    series = series_map(calibration + target, SIGNAL)
    provenance = {
        "signal": SIGNAL,
        "calibration_shards": sorted({row.shard for row in calibration}),
        "n_calibration_successes": sum(row.success for row in calibration),
        "target_shard": TARGET_SHARD,
        "traces": str(args.traces),
        "traces_sha256": file_sha256(args.traces),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
    }

    for budget in args.budget or DEFAULT_BUDGETS:
        rule = fit_stopwatch_from_rows(calibration, series, budget)
        triggers = [
            {
                "episode_index": row.episode_index,
                "success": row.success,
                "n_steps": row.n_steps,
                "trigger_step": stop_step(series[row.key], row.n_steps, rule),
            }
            for row in target
        ]
        good = [t for t in triggers if t["success"]]
        bad = [t for t in triggers if not t["success"]]
        false_stop = sum(t["trigger_step"] is not None for t in good) / len(good)
        name = rule_name(budget)
        save_rule(
            rule,
            args.output_dir / f"{name}.json",
            budget=budget,
            target_false_stop_rate=false_stop,
            **provenance,
        )
        triggers_path = args.output_dir / f"{name}_triggers.json"
        triggers_path.write_text(json.dumps(triggers, indent=2) + "\n")
        stopped = [t["trigger_step"] for t in bad if t["trigger_step"] is not None]
        print(
            f"budget {budget:.0%}: line {rule.line:.4f}, clock {rule.stop_step}, "
            f"bound {rule.bound:.3f} | clean: false stops {false_stop:.1%} of {len(good)}, "
            f"failures stopped {len(stopped)}/{len(bad)} at median step {np.median(stopped):.0f}"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
