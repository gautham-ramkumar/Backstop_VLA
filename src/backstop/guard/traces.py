"""`traces.jsonl` and `labels.jsonl` as typed rows, and the plumbing every
evaluation script repeats on top of them.

`scripts/evaluate_guard.py` writes one `TraceRow` per episode (shard, episode
index, success, episode score, per-step disagreement series). The other scripts
read them back, split them into calibration and held-out halves, and look up
reviewed onset ranges, all through this module, never through each other.

Labels are typed for reading only (`LabelView`). They are written by
`backstop.label.extract`, raw, so reviewed rows round-trip byte for byte.

No privileged sim state (ADR-003). Onset ranges are read only to report lead
time; nothing here fits on them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backstop.corpus.episode import EpisodeKey
from backstop.guard.split import stratified_split

# Left out of the pooled spatial corpus: `language_absent` has no task to
# fail at, and `object_clean` is a different suite (evaluated on its own).
EXCLUDED_SHARDS = frozenset({"language_absent", "object_clean"})
OBJECT_SHARD = "object_clean"


@dataclass(frozen=True, eq=False)
class TraceRow:
    """One episode's guard signal. `per_step` maps a signal to its per-step series.

    Compared by identity: value equality over arrays has no single answer.
    """

    shard: str
    episode_index: int
    success: bool
    n_steps: int
    task: str
    scores: dict[str, float]
    per_step: dict[str, np.ndarray]

    @property
    def key(self) -> EpisodeKey:
        return EpisodeKey(self.shard, self.episode_index)

    def series(self, name: str) -> np.ndarray:
        return self.per_step[name]

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> TraceRow:
        return cls(
            shard=str(row["shard"]),
            episode_index=int(row["episode_index"]),
            success=bool(row["success"]),
            n_steps=int(row["n_steps"]),
            task=str(row["task"]),
            scores={name: float(value) for name, value in row["scores"].items()},
            per_step={
                name: np.asarray(values, dtype=np.float64)
                for name, values in row["per_step"].items()
            },
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "episode_index": self.episode_index,
            "success": self.success,
            "n_steps": self.n_steps,
            "task": self.task,
            "scores": self.scores,
            "per_step": {
                name: [float(value) for value in series] for name, series in self.per_step.items()
            },
        }


def load_traces(path: Path) -> list[TraceRow]:
    return [TraceRow.from_json(row) for row in load_jsonl(path)]


def write_traces(path: Path, rows: Iterable[TraceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json()) + "\n")


@dataclass(frozen=True)
class LabelView:
    """What the guard scripts read from a label row."""

    key: EpisodeKey
    success: bool
    n_steps: int
    onset_range: tuple[int, int] | None
    """The human-reviewed onset window, or None if the episode has none."""

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> LabelView:
        review = row.get("review")
        onset = review.get("onset_range") if isinstance(review, dict) else None
        if isinstance(onset, list) and len(onset) == 2:
            onset = (int(onset[0]), int(onset[1]))
        return cls(
            key=EpisodeKey(str(row["shard"]), int(row["episode_index"])),
            success=bool(row["success"]),
            n_steps=int(row["n_steps"]),
            onset_range=onset if isinstance(onset, tuple) else None,
        )


def load_labels(path: Path) -> dict[EpisodeKey, LabelView]:
    views = (LabelView.from_json(row) for row in load_jsonl(path))
    return {view.key: view for view in views}


def onset_range(labels: dict[EpisodeKey, LabelView], key: EpisodeKey) -> tuple[int, int] | None:
    label = labels.get(key)
    return label.onset_range if label is not None else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def spatial_pool(
    traces: list[TraceRow], excluded: frozenset[str] = EXCLUDED_SHARDS
) -> list[TraceRow]:
    return [row for row in traces if row.shard not in excluded]


def split_halves(
    rows: list[TraceRow], *, eval_fraction: float, seed: int
) -> tuple[list[TraceRow], list[TraceRow]]:
    """(calibration, held-out), stratified by shard and outcome (ADR-004)."""
    keys = [(row.shard, row.episode_index, row.success) for row in rows]
    split = stratified_split(keys, eval_fraction=eval_fraction, seed=seed)
    calibration = [row for row in rows if row.key in split.calibration]
    held_out = [row for row in rows if row.key in split.eval]
    return calibration, held_out


def series_map(rows: list[TraceRow], name: str) -> dict[EpisodeKey, np.ndarray]:
    """key -> one per-step series."""
    return {row.key: row.series(name) for row in rows}
