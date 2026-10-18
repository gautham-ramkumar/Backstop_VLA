"""I/O for the guard package.

Reads the columns the three dataset signals need (`action_chunk_samples`,
`action_chunk`, `action`, `observation.state`) and reduces each episode
immediately, so a raw chunk-sample array is never held for more than one
episode. The visual precondition is filled in later from frames; it is not
in the parquet. Reuses `backstop.label.extract` for shard discovery, the
dataset root, and the task string. No privileged sim state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backstop.guard.signals import (
    action_chunk_disagreement,
    action_chunk_magnitude,
    episode_score,
    progress_monitoring,
    temporal_self_consistency,
    velocity_normalized_consistency,
)
from backstop.label.extract import (
    discover_shards,
    load_episode_meta,
    load_run_provenance,
    resolve_dataset_root,
)
from backstop.parquet_io import as_array, read_parquet_pydict

logger = logging.getLogger(__name__)

PARQUET_SIGNALS = (
    "action_chunk_disagreement",
    "temporal_self_consistency",
    "progress_monitoring",
)


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


def _reduce(rows: list[list[float]], trailing: tuple[int, ...]) -> np.ndarray:
    return as_array(rows, trailing)


def load_shard_signals(
    shard: str,
    *,
    lerobot_root: str | Path,
    eval_root: str | Path,
) -> list[EpisodeSignal]:
    dataset_path, provenance = resolve_dataset_root(eval_root, shard, lerobot_root)
    k, chunk_size, action_dim = _feature_shape(dataset_path, "action_chunk_samples")
    state_dim = _feature_shape(dataset_path, "observation.state")[0]
    tasks = load_episode_meta(dataset_path)

    files = sorted((dataset_path / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {dataset_path}")

    samples_by_ep: dict[int, list[list[float]]] = {}
    chunks_by_ep: dict[int, list[list[float]]] = {}
    actions_by_ep: dict[int, list[list[float]]] = {}
    states_by_ep: dict[int, list[list[float]]] = {}
    success: dict[int, bool] = {}
    columns = [
        "episode_index",
        "action_chunk_samples",
        "action_chunk",
        "action",
        "observation.state",
        "next.success",
    ]
    for file in files:
        table = read_parquet_pydict(file, columns=columns)
        zipped = zip(
            table["episode_index"],
            table["action_chunk_samples"],
            table["action_chunk"],
            table["action"],
            table["observation.state"],
            table["next.success"],
            strict=True,
        )
        for idx, samples, chunk, executed, state, succ in zipped:
            idx = int(idx)
            samples_by_ep.setdefault(idx, []).append(samples)
            chunks_by_ep.setdefault(idx, []).append(chunk)
            actions_by_ep.setdefault(idx, []).append(executed)
            states_by_ep.setdefault(idx, []).append(state)
            success[idx] = success.get(idx, False) or bool(succ)

    perturbation = provenance.get("perturbation", "clean")
    strength = provenance.get("perturbation_strength", 0.0)
    results = []
    for idx in sorted(samples_by_ep):
        sample_arr = _reduce(samples_by_ep[idx], (k, chunk_size, action_dim))
        chunk_arr = _reduce(chunks_by_ep[idx], (chunk_size, action_dim))
        action_arr = _reduce(actions_by_ep[idx], (action_dim,))
        state_arr = _reduce(states_by_ep[idx], (state_dim,))
        per_step = {
            "action_chunk_disagreement": action_chunk_disagreement(sample_arr),
            "temporal_self_consistency": temporal_self_consistency(chunk_arr),
            "progress_monitoring": progress_monitoring(state_arr, action_arr),
        }
        results.append(
            EpisodeSignal(
                shard=shard,
                episode_index=idx,
                success=success[idx],
                n_steps=sample_arr.shape[0],
                task=str(tasks.get(idx, {}).get("task", "")),
                scores={name: episode_score(values) for name, values in per_step.items()},
                per_step=per_step,
                perturbation=str(perturbation),
                perturbation_strength=float(strength) if strength is not None else 0.0,
            )
        )
    return results


def load_chunk_series(
    shards: list[str], *, lerobot_root: str | Path, eval_root: str | Path
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """(shard, episode) -> the two paper signals scored from `action_chunk`.

    Not in `traces.jsonl` (they were added after it), so read from parquet.
    One shard's chunks are held at a time.
    """
    series: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for shard in shards:
        dataset_path, _provenance = resolve_dataset_root(eval_root, shard, lerobot_root)
        chunk_size, action_dim = _feature_shape(dataset_path, "action_chunk")
        by_episode: dict[int, list[list[float]]] = {}
        for file in sorted((dataset_path / "data").rglob("*.parquet")):
            table = read_parquet_pydict(file, columns=["episode_index", "action_chunk"])
            for index, chunk in zip(table["episode_index"], table["action_chunk"], strict=True):
                by_episode.setdefault(int(index), []).append(chunk)
        for index, rows in by_episode.items():
            chunks = _reduce(rows, (chunk_size, action_dim))
            series[(shard, index)] = {
                "action_chunk_magnitude": action_chunk_magnitude(chunks),
                "velocity_normalized_consistency": velocity_normalized_consistency(chunks),
            }
    return series


def iter_episode_samples(
    shard: str, *, lerobot_root: str | Path, eval_root: str | Path
) -> Iterator[tuple[int, np.ndarray]]:
    """(episode_index, (T, K, chunk, dim) samples), one episode in memory at a time.

    LeRobot writes episodes contiguously, so an episode is complete when the
    next one starts (or the file list ends). Raises if an episode reappears
    after another one began, rather than yield half of it.
    """
    dataset_path, _provenance = resolve_dataset_root(eval_root, shard, lerobot_root)
    shape = _feature_shape(dataset_path, "action_chunk_samples")
    current: int | None = None
    rows: list[list[float]] = []
    finished: set[int] = set()
    for file in sorted((dataset_path / "data").rglob("*.parquet")):
        table = read_parquet_pydict(file, columns=["episode_index", "action_chunk_samples"])
        for index, samples in zip(
            table["episode_index"], table["action_chunk_samples"], strict=True
        ):
            index = int(index)
            if index != current:
                if current is not None:
                    yield current, _reduce(rows, shape)
                    finished.add(current)
                if index in finished:
                    raise ValueError(f"{shard}: episode {index} is not contiguous on disk")
                current, rows = index, []
            rows.append(samples)
    if current is not None:
        yield current, _reduce(rows, shape)


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
