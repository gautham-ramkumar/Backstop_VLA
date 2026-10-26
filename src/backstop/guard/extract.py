"""I/O for the guard package.

Reads `action_chunk_samples` (the K sampled plans per step) and `next.success`
one episode at a time, and reduces each episode to its disagreement series
before reading the next, so a raw chunk-sample array is never held for more
than one episode. Reads through `backstop.corpus`. No privileged sim state.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np

from backstop.corpus.layout import ShardLayout
from backstop.corpus.read import feature_shape, iter_episode_columns, load_episode_meta
from backstop.guard.signals import action_chunk_disagreement, episode_score
from backstop.guard.traces import TraceRow
from backstop.parquet_io import as_array

logger = logging.getLogger(__name__)

SIGNAL = "action_chunk_disagreement"


def iter_episode_samples(layout: ShardLayout, shard: str) -> Iterator[tuple[int, np.ndarray]]:
    """(episode_index, (T, K, chunk, dim) samples), one episode in memory at a time."""
    dataset_path = layout.dataset_root(shard)
    shape = feature_shape(dataset_path, "action_chunk_samples")
    for index, rows in iter_episode_columns(dataset_path, ["action_chunk_samples"]):
        yield index, as_array(rows["action_chunk_samples"], shape)


def load_shard_signals(layout: ShardLayout, shard: str) -> list[TraceRow]:
    dataset_path = layout.dataset_root(shard)
    shape = feature_shape(dataset_path, "action_chunk_samples")
    tasks = load_episode_meta(dataset_path)

    results = []
    columns = ["action_chunk_samples", "next.success"]
    for index, rows in iter_episode_columns(dataset_path, columns):
        samples = as_array(rows["action_chunk_samples"], shape)
        per_step = {SIGNAL: action_chunk_disagreement(samples)}
        results.append(
            TraceRow(
                shard=shard,
                episode_index=index,
                success=any(bool(flag) for flag in rows["next.success"]),
                n_steps=samples.shape[0],
                task=str(tasks.get(index, {}).get("task", "")),
                scores={SIGNAL: episode_score(per_step[SIGNAL])},
                per_step=per_step,
            )
        )
    return sorted(results, key=lambda episode: episode.episode_index)


def load_corpus_signals(
    layout: ShardLayout, shards: list[str] | None = None
) -> dict[str, list[TraceRow]]:
    """shard -> its episodes' signals.

    A shard with no `run.json` is skipped with a warning: that is a directory
    like a renamed, incomplete recording sitting next to real shards under
    the privileged root (the same tolerance `scripts/auto_label.py` has). Any
    other missing file -- a real shard whose dataset or parquet is gone --
    raises, so corpus rows cannot silently drop out of a report.
    """
    shard_names = shards if shards is not None else layout.discover_shards()
    by_shard: dict[str, list[TraceRow]] = {}
    skipped: list[str] = []
    for shard in shard_names:
        if layout.provenance(shard) is None:
            skipped.append(shard)
            continue
        by_shard[shard] = load_shard_signals(layout, shard)
    if skipped:
        logger.warning("skipped shards with no run.json under %s: %s", layout.eval_root, skipped)
    return by_shard
