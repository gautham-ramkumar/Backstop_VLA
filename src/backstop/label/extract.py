"""I/O for the auto proto-labeller: reads a recorded shard, computes signals,
and merges rows into `artifacts/corpus/labels.jsonl` (ADR-003's label sidecar).

Merge rule: a row a human has already reviewed (`review.status == "reviewed"`)
is never touched by a re-run. Everything else -- new episodes, and episodes
still `pending` -- is recomputed fresh. This is what makes it safe to re-run
after recording a new shard without re-litigating finished review work.

Parquet is imported lazily inside the functions that need it, matching
`scripts/verify_corpus.py`'s convention -- this package's pure-numpy half
(`signals.py`) must stay importable without pyarrow.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

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
from backstop.parquet_io import read_parquet_pydict

REVIEWED = "reviewed"
PENDING = "pending"


def discover_shards(privileged_root: str | Path) -> list[str]:
    """Every shard that was actually recorded with `privileged: true` -- the
    only ones the auto-labeller has anything to compute from."""
    root = Path(privileged_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_episode_meta(dataset_root: str | Path) -> dict[int, dict[str, Any]]:
    """episode_index -> {"task": str, "length": int}, from `meta/episodes`."""
    files = sorted((Path(dataset_root) / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no meta/episodes parquet under {dataset_root}")
    meta: dict[int, dict[str, Any]] = {}
    for file in files:
        d = read_parquet_pydict(file, columns=["episode_index", "tasks", "length"])
        for idx, tasks, length in zip(d["episode_index"], d["tasks"], d["length"], strict=True):
            meta[int(idx)] = {"task": str(tasks[0]) if tasks else "", "length": int(length)}
    return meta


_VIDEO_KEY = "observation.images.image"


def load_episode_video_range(
    dataset_root: str | Path, episode_index: int
) -> tuple[Path, float, float]:
    """(video_file, from_timestamp_s, to_timestamp_s) for one episode's
    agentview camera. LeRobot v3 concatenates every episode's frames into one
    video per chunk; `meta/episodes` is what records where one episode ends
    and the next begins."""
    columns = [
        "episode_index",
        f"videos/{_VIDEO_KEY}/chunk_index",
        f"videos/{_VIDEO_KEY}/file_index",
        f"videos/{_VIDEO_KEY}/from_timestamp",
        f"videos/{_VIDEO_KEY}/to_timestamp",
    ]
    files = sorted((Path(dataset_root) / "meta" / "episodes").rglob("*.parquet"))
    for file in files:
        d = read_parquet_pydict(file, columns=columns)
        for row_idx, idx in enumerate(d["episode_index"]):
            if int(idx) != episode_index:
                continue
            chunk = d[f"videos/{_VIDEO_KEY}/chunk_index"][row_idx]
            part = d[f"videos/{_VIDEO_KEY}/file_index"][row_idx]
            video_path = (
                Path(dataset_root)
                / "videos"
                / _VIDEO_KEY
                / f"chunk-{chunk:03d}"
                / f"file-{part:03d}.mp4"
            )
            return (
                video_path,
                float(d[f"videos/{_VIDEO_KEY}/from_timestamp"][row_idx]),
                float(d[f"videos/{_VIDEO_KEY}/to_timestamp"][row_idx]),
            )
    raise FileNotFoundError(f"episode {episode_index} not found in {dataset_root}/meta/episodes")


def load_dataset_frames(dataset_root: str | Path) -> dict[int, dict[str, Any]]:
    """episode_index -> {"state": (T, 8) float32 array, "success": bool}.

    Rows within one episode are contiguous and step-ordered in the parquet
    shard by construction (the record loop appends one row per step); grouping
    by first appearance preserves that order without a separate sort.
    """
    files = sorted((Path(dataset_root) / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {dataset_root}")
    states: dict[int, list[list[float]]] = {}
    success: dict[int, bool] = {}
    for file in files:
        columns = ["episode_index", "observation.state", "next.success"]
        d = read_parquet_pydict(file, columns=columns)
        for idx, state_row, succ in zip(
            d["episode_index"], d["observation.state"], d["next.success"], strict=True
        ):
            idx = int(idx)
            states.setdefault(idx, []).append([float(v) for v in state_row])
            success[idx] = success.get(idx, False) or bool(succ)
    return {
        idx: {"state": np.asarray(rows, dtype=np.float32), "success": success[idx]}
        for idx, rows in states.items()
    }


def load_privileged_sim_state(
    privileged_root: str | Path, shard: str, episode_index: int
) -> np.ndarray | None:
    from backstop.record.privileged import load_episode  # noqa: PLC0415

    try:
        payload = load_episode(privileged_root, shard, episode_index)
    except FileNotFoundError:
        return None
    return payload.get("sim_state")


def _shard_stem(root: str | Path, shard: str, suffix: str) -> Path:
    """Spatial shards are named `spatial_<shard>_k4`. A shard from another suite
    already carries its suite in the shard name (`object_clean` ->
    `object_clean_k4`), so that form is the fallback."""
    spatial = Path(root) / f"spatial_{shard}_k4{suffix}"
    return spatial if spatial.exists() else Path(root) / f"{shard}_k4{suffix}"


def load_run_provenance(eval_root: str | Path, shard: str) -> dict[str, Any]:
    path = _shard_stem(eval_root, shard, "") / "run.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def load_perturbation_params(configs_root: str | Path, shard: str) -> dict[str, Any]:
    """Best-effort: `perturbation_kwargs` lives in the recording config, not in
    `run.json` (`build_provenance` doesn't carry it -- see ADR-003). Only the
    categorical families (`absent_target`, `language_paraphrase`) have any."""
    import yaml  # noqa: PLC0415

    path = _shard_stem(configs_root, shard, ".yaml")
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    record = raw.get("record", {}) if isinstance(raw, dict) else {}
    kwargs = record.get("perturbation_kwargs", {}) if isinstance(record, dict) else {}
    return dict(kwargs) if isinstance(kwargs, dict) else {}


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


def resolve_dataset_root(
    eval_root: str | Path, shard: str, lerobot_root: str | Path
) -> tuple[Path, dict[str, Any]]:
    """(dataset_path, run_provenance) for a shard, from its `run.json`."""
    provenance = load_run_provenance(eval_root, shard)
    dataset_root = provenance.get("dataset_root")
    if not dataset_root:
        raise FileNotFoundError(
            f"no run.json (or no dataset_root in it) for shard {shard!r} under {eval_root}"
        )
    dataset_path = Path(dataset_root)
    if not dataset_path.is_absolute() and not dataset_path.is_dir():
        # run.json stores the path as it was at record time (relative to the repo
        # root the recording ran from); resolve against the caller's lerobot_root
        # if the plain relative path does not exist from the current cwd.
        candidate = Path(lerobot_root) / dataset_path.name
        dataset_path = candidate if candidate.is_dir() else dataset_path
    return dataset_path, provenance


def build_shard_rows(
    shard: str,
    *,
    lerobot_root: str | Path,
    privileged_root: str | Path,
    eval_root: str | Path,
    configs_root: str | Path,
) -> list[dict[str, Any]]:
    dataset_path, provenance = resolve_dataset_root(eval_root, shard, lerobot_root)
    meta = load_episode_meta(dataset_path)
    frames = load_dataset_frames(dataset_path)
    params = load_perturbation_params(configs_root, shard)
    perturbation = provenance.get("perturbation", "clean")
    strength = provenance.get("perturbation_strength", 0.0)

    rows = []
    for episode_index in sorted(frames):
        frame = frames[episode_index]
        info = meta.get(episode_index, {"task": "", "length": frame["state"].shape[0]})
        sim_state = load_privileged_sim_state(privileged_root, shard, episode_index)
        rows.append(
            build_episode_row(
                shard=shard,
                episode_index=episode_index,
                task=info["task"],
                n_steps=info["length"],
                success=frame["success"],
                state=frame["state"],
                sim_state=sim_state,
                perturbation=perturbation,
                perturbation_strength=strength,
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
