"""The auto proto-labeller's rows: computes signals for a recorded shard (read
through `backstop.corpus`) and merges rows into `artifacts/corpus/labels.jsonl`
(ADR-003's label sidecar).

Merge rule: a row a human has already reviewed (`review.status == "reviewed"`)
is never touched by a re-run. Everything else -- new episodes, and episodes
still `pending` -- is recomputed fresh. This is what makes it safe to re-run
after recording a new shard without re-litigating finished review work.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backstop.corpus.layout import ShardLayout
from backstop.corpus.privileged import load_sim_state
from backstop.corpus.read import load_dataset_frames, load_episode_meta
from backstop.label.signals import (
    GRIPPER_CLOSED_WIDTH_M,
    eef_path_length,
    eef_positions,
    gripper_close_events,
    gripper_width,
    object_signals,
    propose_category,
    propose_onset_range,
)

REVIEWED = "reviewed"
PENDING = "pending"


def build_episode_row(
    *,
    shard: str,
    episode_index: int,
    task: str,
    n_steps: int,
    success: bool,
    state: np.ndarray,
    sim_state: np.ndarray | None,
    perturbation: str,
    perturbation_strength: float,
    perturbation_params: dict[str, Any],
) -> dict[str, Any]:
    width = gripper_width(state)
    closed = width <= GRIPPER_CLOSED_WIDTH_M
    cycles = gripper_close_events(state)
    objects = object_signals(sim_state, eef_positions(state), closed)
    onset = propose_onset_range(objects, eef_positions(state), n_steps)

    return {
        "shard": shard,
        "episode_index": episode_index,
        "task": task,
        "n_steps": n_steps,
        "success": success,
        "perturbation": perturbation,
        "perturbation_strength": perturbation_strength,
        "perturbation_params": perturbation_params,
        "auto": {
            "proposed_category": propose_category(objects, success),
            "proposed_onset_range": list(onset) if onset is not None else None,
            "eef_path_length_m": eef_path_length(state),
            "gripper_cycle_count": cycles.n_cycles,
            "last_gripper_close_step": cycles.last_close_step,
            "object_signal_available": objects is not None,
            "grasped_ever": objects.grasped_ever if objects is not None else None,
            "bowl_displacement_m": objects.bowl_displacement_m if objects is not None else None,
            "terminal_bowl_to_plate_m": (
                objects.terminal_bowl_to_plate_m if objects is not None else None
            ),
        },
        "review": {
            "status": PENDING,
            "category": None,
            "onset_range": None,
            "notes": None,
            "reviewed_at": None,
        },
    }


def build_shard_rows(layout: ShardLayout, shard: str) -> list[dict[str, Any]]:
    dataset_path = layout.dataset_root(shard)
    provenance = layout.provenance(shard)
    assert provenance is not None  # dataset_root raised otherwise
    meta = load_episode_meta(dataset_path)
    frames = load_dataset_frames(dataset_path)
    params = layout.perturbation_params(shard)

    rows = []
    for episode_index in sorted(frames):
        frame = frames[episode_index]
        info = meta.get(episode_index, {"task": "", "length": frame["state"].shape[0]})
        sim_state = load_sim_state(layout.privileged_root, shard, episode_index)
        rows.append(
            build_episode_row(
                shard=shard,
                episode_index=episode_index,
                task=info["task"],
                n_steps=info["length"],
                success=frame["success"],
                state=frame["state"],
                sim_state=sim_state,
                perturbation=provenance.perturbation,
                perturbation_strength=provenance.perturbation_strength,
                perturbation_params=params,
            )
        )
    return rows


def load_labels(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    file = Path(path)
    if not file.is_file():
        return labels
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        labels[(row["shard"], row["episode_index"])] = row
    return labels


def merge_rows(
    existing: dict[tuple[str, int], dict[str, Any]], new_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reviewed rows are kept byte-for-byte; everything else is replaced with
    the freshly computed row. Rows in `existing` whose shard is not in this
    re-run's `new_rows` (a different shard entirely) pass through untouched."""
    merged: dict[tuple[str, int], dict[str, Any]] = dict(existing)
    for row in new_rows:
        key = (row["shard"], row["episode_index"])
        current = merged.get(key)
        if current is not None and current.get("review", {}).get("status") == REVIEWED:
            continue
        merged[key] = row
    return sorted(merged.values(), key=lambda r: (r["shard"], r["episode_index"]))


def write_labels(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(file.suffix + ".partial")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp.replace(file)


def mark_reviewed(
    row: dict[str, Any],
    *,
    category: str,
    onset_range: tuple[int, int] | None,
    notes: str,
) -> dict[str, Any]:
    row = dict(row)
    row["review"] = {
        "status": REVIEWED,
        "category": category,
        "onset_range": list(onset_range) if onset_range is not None else None,
        "notes": notes or None,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    return row
