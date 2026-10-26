"""Reading a recorded LeRobot dataset: episode metadata, per-step columns, video.

Parquet goes through `backstop.parquet_io`, so nothing here needs LeRobot.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from backstop.parquet_io import read_parquet_pydict

VIDEO_KEY = "observation.images.image"


def feature_shape(dataset_root: Path, feature: str) -> tuple[int, ...]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    return tuple(info["features"][feature]["shape"])


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


def load_episode_video_range(
    dataset_root: str | Path, episode_index: int
) -> tuple[Path, float, float]:
    """(video_file, from_timestamp_s, to_timestamp_s) for one episode's
    agentview camera. LeRobot v3 concatenates every episode's frames into one
    video per chunk; `meta/episodes` is what records where one episode ends
    and the next begins."""
    columns = [
        "episode_index",
        f"videos/{VIDEO_KEY}/chunk_index",
        f"videos/{VIDEO_KEY}/file_index",
        f"videos/{VIDEO_KEY}/from_timestamp",
        f"videos/{VIDEO_KEY}/to_timestamp",
    ]
    files = sorted((Path(dataset_root) / "meta" / "episodes").rglob("*.parquet"))
    for file in files:
        d = read_parquet_pydict(file, columns=columns)
        for row_idx, idx in enumerate(d["episode_index"]):
            if int(idx) != episode_index:
                continue
            chunk = d[f"videos/{VIDEO_KEY}/chunk_index"][row_idx]
            part = d[f"videos/{VIDEO_KEY}/file_index"][row_idx]
            video_path = (
                Path(dataset_root)
                / "videos"
                / VIDEO_KEY
                / f"chunk-{chunk:03d}"
                / f"file-{part:03d}.mp4"
            )
            return (
                video_path,
                float(d[f"videos/{VIDEO_KEY}/from_timestamp"][row_idx]),
                float(d[f"videos/{VIDEO_KEY}/to_timestamp"][row_idx]),
            )
    raise FileNotFoundError(f"episode {episode_index} not found in {dataset_root}/meta/episodes")


def iter_episode_columns(
    dataset_root: Path, columns: list[str]
) -> Iterator[tuple[int, dict[str, list[Any]]]]:
    """(episode_index, column -> that episode's rows), one episode at a time.

    LeRobot writes episodes contiguously, so an episode is complete when the
    next one starts or the files end. Raises if an episode reappears after
    another began, rather than yield half of it.
    """
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {dataset_root}")
    current: int | None = None
    rows: dict[str, list[Any]] = {}
    finished: set[int] = set()
    for file in files:
        table = read_parquet_pydict(file, columns=["episode_index", *columns])
        for i, index in enumerate(table["episode_index"]):
            index = int(index)
            if index != current:
                if current is not None:
                    yield current, rows
                    finished.add(current)
                if index in finished:
                    raise ValueError(f"{dataset_root}: episode {index} is not contiguous on disk")
                current, rows = index, {name: [] for name in columns}
            for name in columns:
                rows[name].append(table[name][i])
    if current is not None:
        yield current, rows


def load_dataset_frames(dataset_root: Path) -> dict[int, dict[str, Any]]:
    """episode_index -> {"state": (T, 8) float32 array, "success": bool}."""
    return {
        index: {
            "state": np.asarray(
                [[float(v) for v in row] for row in rows["observation.state"]], dtype=np.float32
            ),
            "success": any(bool(flag) for flag in rows["next.success"]),
        }
        for index, rows in iter_episode_columns(dataset_root, ["observation.state", "next.success"])
    }
