"""I/O for the guard package.

Reads `action_chunk_samples` (the K sampled plans per step) and `next.success`
one episode at a time, and reduces each episode to its disagreement series
before reading the next, so a raw chunk-sample array is never held for more
than one episode. Uses `backstop.label.extract` for shard discovery, the
dataset root, and the task string. No privileged sim state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backstop.guard.signals import action_chunk_disagreement, episode_score
from backstop.label.extract import (
    discover_shards,
    load_episode_meta,
    load_run_provenance,
    resolve_dataset_root,
)
from backstop.parquet_io import as_array, read_parquet_pydict

logger = logging.getLogger(__name__)

SIGNAL = "action_chunk_disagreement"


@dataclass(frozen=True)
class EpisodeSignal:
    shard: str
    episode_index: int
    success: bool
    n_steps: int
    task: str
    scores: dict[str, float]
    per_step: dict[str, np.ndarray]
    perturbation: str
    perturbation_strength: float


def _feature_shape(dataset_root: Path, feature: str) -> tuple[int, ...]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    return tuple(info["features"][feature]["shape"])


def _iter_episode_rows(
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


def iter_episode_samples(
    shard: str, *, lerobot_root: str | Path, eval_root: str | Path
) -> Iterator[tuple[int, np.ndarray]]:
    """(episode_index, (T, K, chunk, dim) samples), one episode in memory at a time."""
    dataset_path, _provenance = resolve_dataset_root(eval_root, shard, lerobot_root)
    shape = _feature_shape(dataset_path, "action_chunk_samples")
    for index, rows in _iter_episode_rows(dataset_path, ["action_chunk_samples"]):
        yield index, as_array(rows["action_chunk_samples"], shape)


def load_shard_signals(
    shard: str,
    *,
    lerobot_root: str | Path,
    eval_root: str | Path,
) -> list[EpisodeSignal]:
    dataset_path, provenance = resolve_dataset_root(eval_root, shard, lerobot_root)
    shape = _feature_shape(dataset_path, "action_chunk_samples")
    tasks = load_episode_meta(dataset_path)
    perturbation = provenance.get("perturbation", "clean")
    strength = provenance.get("perturbation_strength", 0.0)

    results = []
    columns = ["action_chunk_samples", "next.success"]
    for index, rows in _iter_episode_rows(dataset_path, columns):
        samples = as_array(rows["action_chunk_samples"], shape)
        per_step = {SIGNAL: action_chunk_disagreement(samples)}
        results.append(
            EpisodeSignal(
                shard=shard,
                episode_index=index,
                success=any(bool(flag) for flag in rows["next.success"]),
                n_steps=samples.shape[0],
                task=str(tasks.get(index, {}).get("task", "")),
                scores={SIGNAL: episode_score(per_step[SIGNAL])},
                per_step=per_step,
                perturbation=str(perturbation),
                perturbation_strength=float(strength) if strength is not None else 0.0,
            )
        )
    return sorted(results, key=lambda episode: episode.episode_index)


def load_corpus_signals(
    *,
    lerobot_root: str | Path,
    privileged_root: str | Path,
    eval_root: str | Path,
    shards: list[str] | None = None,
) -> dict[str, list[EpisodeSignal]]:
    """shard -> its episodes' signals.

    A shard with no `run.json` is skipped with a warning: that is a directory
    like a renamed, incomplete recording sitting next to real shards under
    `privileged_root` (the same tolerance `scripts/auto_label.py` has). Any
    other missing file -- a real shard whose dataset or parquet is gone --
    raises, so corpus rows cannot silently drop out of a report.
    """
    shard_names = shards if shards is not None else discover_shards(privileged_root)
    by_shard: dict[str, list[EpisodeSignal]] = {}
    skipped: list[str] = []
    for shard in shard_names:
        if not load_run_provenance(eval_root, shard).get("dataset_root"):
            skipped.append(shard)
            continue
        by_shard[shard] = load_shard_signals(shard, lerobot_root=lerobot_root, eval_root=eval_root)
    if skipped:
        logger.warning("skipped shards with no run.json under %s: %s", eval_root, skipped)
    return by_shard
