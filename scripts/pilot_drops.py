#!/usr/bin/env python
"""Did a perturbation pilot's failures drop the object, and how early?

For each recorded episode, from its privileged sim state (labelling only, never
guard input; ADR-003): when the instructed bowl was first lifted, and every fall,
meaning a lift of more than `--lift-m` that comes back down away from the plate.

The question it answers is the stop rule for a new failure family: the stopwatch
ends every failure at about step 132, so a family only gives the guard room to
beat it if its failures go wrong before then. A family whose failures still stall
with no early fall has not changed the failure mode.

    uv run python scripts/pilot_drops.py artifacts/pilot/grasp_slip/s10_m60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backstop.corpus.privileged import load_sim_state
from backstop.label.signals import (
    BOWL_1_POS_SLICE,
    LIBERO_SPATIAL_SIM_STATE_WIDTH,
    NEAR_PLATE_THRESHOLD_M,
    PLATE_1_POS_SLICE,
)

STOPWATCH_STEP = 132  # where the clock alone ends every corpus failure (ADR-005)
# `obstacle`'s carton is the one object after the plate in the 105-wide layout
# (qpos address 44, probed on the real model), so the slices above still hold.
CARTON_POS_SLICE = slice(1 + 44, 1 + 47)
OBSTACLE_SIM_STATE_WIDTH = LIBERO_SPATIAL_SIM_STATE_WIDTH + 7 + 6  # one more free joint
CARTON_MOVED_M = 0.02


def falls(sim_state: np.ndarray, lift_m: float) -> tuple[int | None, list[int]]:
    """(first step the bowl is lifted, steps where a lifted bowl comes down off the plate)."""
    bowl, plate = sim_state[:, BOWL_1_POS_SLICE], sim_state[:, PLATE_1_POS_SLICE]
    lift = bowl[:, 2] - bowl[0, 2]
    off_plate = np.linalg.norm(bowl[:, :2] - plate[:, :2], axis=1) > NEAR_PLATE_THRESHOLD_M
    lifted = np.flatnonzero(lift > lift_m)
    down = [t for t in range(1, len(lift)) if lift[t - 1] > lift_m >= lift[t] and off_plate[t]]
    return (int(lifted[0]) if len(lifted) else None), down


def carton_hit(sim_state: np.ndarray) -> int | None:
    """First step the obstacle carton has been pushed or knocked from where it stood."""
    if sim_state.shape[1] != OBSTACLE_SIM_STATE_WIDTH:
        return None
    carton = sim_state[:, CARTON_POS_SLICE]
    moved = np.flatnonzero(np.linalg.norm(carton - carton[0], axis=1) > CARTON_MOVED_M)
    return int(moved[0]) if len(moved) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pilots", type=Path, nargs="+", help="pilot output dirs (eval/, privileged/)"
    )
    parser.add_argument("--shard", default="grasp_slip", help="grasp_slip or obstacle")
    parser.add_argument("--lift-m", type=float, default=0.03)
    args = parser.parse_args(argv)

    for pilot in args.pilots:
        episodes = json.loads((pilot / "eval" / "eval.json").read_text())["episodes"]
        rows = []
        for episode in episodes:
            states = load_sim_state(pilot / "privileged", args.shard, episode["episode_index"])
            assert states is not None, (
                f"{pilot}: episode {episode['episode_index']} has no sim state"
            )
            first_lift, down = falls(states, args.lift_m)
            hit = carton_hit(states)
            rows.append({**episode, "first_lift": first_lift, "falls": down, "carton_hit": hit})
        failures = [row for row in rows if not row["success"]]
        early = [row for row in failures if row["falls"] and row["falls"][0] < STOPWATCH_STEP]
        print(
            f"{pilot.name}: success {len(rows) - len(failures)}/{len(rows)}; "
            f"failures that dropped the bowl before step {STOPWATCH_STEP}: "
            f"{len(early)}/{len(failures)}; failures never lifting it: "
            f"{sum(row['first_lift'] is None for row in failures)}/{len(failures)}; "
            f"failures that hit the carton before step {STOPWATCH_STEP}: "
            f"{sum((row['carton_hit'] or STOPWATCH_STEP) < STOPWATCH_STEP for row in failures)}"
            f"/{len(failures)}"
        )
        for row in rows:
            print(
                f"  ep {row['episode_index']:>2} task {row['task_id']} "
                f"{'ok  ' if row['success'] else 'FAIL'} {row['n_steps']:>3} steps, "
                f"first lift {row['first_lift']}, falls {row['falls']}, "
                f"carton hit {row['carton_hit']}"
            )
        (pilot / "drops.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
