#!/usr/bin/env python
"""Week-2 strength calibration sweep.

Each perturbation family has a `strength` in [0, 1] with no known mapping yet to
success rate -- Week2_plan.md's own families.py docstring says the starting
ranges "are guesses." The sweep's job is to pick, per family, the strength whose
success rate lands in 30-70%: the structural defense against a guard that learns
to detect the perturbation instead of the impending failure (see the plan's
"Why every family targets ~50% success" section).

Design, per the plan's "Strength calibration" section:
    3 tasks (spanning measured week-1 difficulty: task 8 at 9/10, task 0 at 8/10,
    task 3 at 4/10) x 5 episodes x 3 candidate strengths x N families.

    N=4 here, not the plan's 5: `lighting`, `camera`, `object_pose`, `action_noise`
    have a continuous strength knob. `language` (both sub-modes) is categorical --
    `absent_target` is 0% by construction and quarantined, `paraphrase` is a
    control that must NOT move the success rate -- so sweeping either against a
    strength target is meaningless. `distractor` is unbuilt (timeboxed, per plan).
    180 episodes, not 225.

    Candidate strengths [0.3, 0.6, 1.0]: low, medium, and the ceiling, so a family
    that is still ~90% success at 1.0 and one that is already 0% at 0.3 both show
    up clearly rather than landing three similar-looking midpoints.

    Same 5 seeds (0-4) reused across every cell for a given task: LIBERO ties
    init_state_id to episode index within a run, so this holds the init state
    constant across every family/strength cell for that task -- a controlled
    comparison, not just a bigger sample.

`record.enabled=False` for every cell: `run_episodes` builds and attaches the
perturbation (`make_perturbation` / `attach`) unconditionally on whether a
dataset is open (see `record/loop.py`), so eval-only exercises the family
exactly as a real recording would, without paying for video encoding, parquet
writes, or privileged-state capture that this sweep does not need.

Resumable: writes one row to the CSV per completed cell and skips any cell
already present on restart, matching every other Week-2 recording job's
"a crash costs the current cell, not the sweep" design.

    uv run python scripts/calibration_sweep.py                 # full sweep
    uv run python scripts/calibration_sweep.py --smoke          # 1 cell, 1 episode
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from backstop.config import (
    EnvYamlConfig,
    EvalYamlConfig,
    PipelineConfig,
    PolicyConfig,
    RecordYamlConfig,
    ToleranceConfig,
    WandbYamlConfig,
)
from backstop.eval.pipeline import run_pipeline

logger = logging.getLogger(__name__)

TASKS = [8, 0, 3]  # week-1 difficulty: 9/10, 8/10, 4/10 successes respectively
FAMILIES = ["lighting", "camera", "object_pose", "action_noise"]
STRENGTHS = [0.3, 0.6, 1.0]
EPISODES_PER_CELL = 5
SEED = 0  # start_seed; episodes within a cell get seed, seed+1, ..., seed+n-1

RESULTS_CSV = Path("artifacts/calibration/sweep_results.csv")
FIELDNAMES = ["family", "task_id", "strength", "n_episodes", "n_success", "success_rate"]


def _cell_config(family: str, task_id: int, strength: float, n_episodes: int) -> PipelineConfig:
    tag = f"{family}_task{task_id}_s{strength:.2f}".replace(".", "")
    return PipelineConfig(
        seed=SEED,
        output_dir=f"artifacts/calibration/{tag}",
        policy=PolicyConfig(
            path="HuggingFaceVLA/smolvla_libero",
            revision="6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
            device="cuda",
            n_action_steps=1,
            num_steps=10,
        ),
        env=EnvYamlConfig(
            task="libero_spatial",
            task_ids=[task_id],
            control_mode="relative",
            init_states=True,
            mujoco_gl="osmesa",
        ),
        eval=EvalYamlConfig(n_episodes=n_episodes, batch_size=1, max_parallel_tasks=1),
        record=RecordYamlConfig(
            enabled=False,
            k_samples=1,
            perturbation=family,
            perturbation_strength=strength,
        ),
        wandb=WandbYamlConfig(enabled=False, mode="disabled"),
        tolerance=ToleranceConfig(enforce=False),
    )


def _load_done_cells() -> set[tuple[str, int, float]]:
    if not RESULTS_CSV.is_file():
        return set()
    with RESULTS_CSV.open() as handle:
        return {
            (row["family"], int(row["task_id"]), float(row["strength"]))
            for row in csv.DictReader(handle)
        }


def _append_result(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.is_file()
    with RESULTS_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_sweep(
    families: list[str], tasks: list[int], strengths: list[float], episodes_per_cell: int
) -> None:
    done = _load_done_cells()
    cells = [(f, t, s) for f in families for t in tasks for s in strengths]
    logger.info("sweep: %d cells, %d already done", len(cells), len(done))

    for family, task_id, strength in cells:
        if (family, task_id, strength) in done:
            logger.info("skip (already done): %s task=%d strength=%.2f", family, task_id, strength)
            continue
        logger.info("cell: family=%s task=%d strength=%.2f", family, task_id, strength)
        cfg = _cell_config(family, task_id, strength, episodes_per_cell)
        stats = run_pipeline(cfg)
        _append_result(
            {
                "family": family,
                "task_id": task_id,
                "strength": strength,
                "n_episodes": stats["n_total"],
                "n_success": stats["n_success"],
                "success_rate": stats["success_rate"],
            }
        )
        logger.info(
            "cell done: family=%s task=%d strength=%.2f success_rate=%.3f",
            family,
            task_id,
            strength,
            stats["success_rate"],
        )


def print_summary() -> None:
    """Discovers families/tasks/strengths actually present in the CSV, rather
    than iterating the module's default lists -- a refinement sweep at strengths
    or families outside the original defaults must still show up here."""
    if not RESULTS_CSV.is_file():
        print("no results yet")
        return
    with RESULTS_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    print(f"\n{len(rows)} cells recorded\n")
    for family in sorted(by_family):
        cell_rows = by_family[family]
        tasks = sorted({int(r["task_id"]) for r in cell_rows})
        strengths = sorted({float(r["strength"]) for r in cell_rows})
        print(f"## {family}")
        print("| strength | " + " | ".join(f"task {t}" for t in tasks) + " | pooled |")
        print("|---" * (len(tasks) + 2) + "|")
        for strength in strengths:
            per_task = {int(r["task_id"]): r for r in cell_rows if float(r["strength"]) == strength}
            cells_for_row = [per_task.get(t) for t in tasks]
            n_total = sum(int(c["n_episodes"]) for c in cells_for_row if c)
            n_success = sum(int(c["n_success"]) for c in cells_for_row if c)
            pooled = f"{n_success}/{n_total}" if n_total else "-"
            per_task_str = [
                f"{c['n_success']}/{c['n_episodes']} ({float(c['success_rate']):.0%})" if c else "-"
                for c in cells_for_row
            ]
            print(f"| {strength:.2f} | " + " | ".join(per_task_str) + f" | {pooled} |")
        print()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 family x 1 task x 1 strength x 1 episode, to check plumbing before the full sweep",
    )
    parser.add_argument("--summary-only", action="store_true", help="print the table, run nothing")
    parser.add_argument(
        "--families", nargs="+", default=None, help="override FAMILIES, e.g. for a refinement pass"
    )
    parser.add_argument("--tasks", nargs="+", type=int, default=None, help="override TASKS")
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=None, help="override STRENGTHS"
    )
    parser.add_argument(
        "--episodes-per-cell", type=int, default=None, help="override EPISODES_PER_CELL"
    )
    args = parser.parse_args(argv)

    if args.summary_only:
        print_summary()
        return 0

    if args.smoke:
        run_sweep(families=["lighting"], tasks=[8], strengths=[0.6], episodes_per_cell=1)
    else:
        run_sweep(
            args.families or FAMILIES,
            args.tasks or TASKS,
            args.strengths or STRENGTHS,
            args.episodes_per_cell or EPISODES_PER_CELL,
        )

    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
