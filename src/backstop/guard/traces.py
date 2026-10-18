"""Reading `traces.jsonl` and `labels.jsonl`, and the plumbing every
evaluation script repeats on top of them.

`scripts/evaluate_guard.py` writes one trace row per episode (shard, episode
index, success, episode scores, per-step series). The `evaluate_*` scripts,
`train_step_model.py`, `freeze_guard.py` and `pretest_reselect.py` all read
those rows back, key them by (shard, episode), split them into calibration
and held-out halves, and look up reviewed onset ranges. That code lived as
private copies in each script, and some scripts loaded others by file path
to borrow theirs. It lives here now, typed and tested, so a script imports
from the package and never from another script.

No privileged sim state (ADR-003). Onset ranges are read only to report lead
time; nothing here fits on them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backstop.guard.combine import TRUST_FEATURES, apply_trust_head
from backstop.guard.extract import PARQUET_SIGNALS
from backstop.guard.split import EpisodeKey, stratified_split

Row = dict[str, object]
Key = tuple[str, int]

FPS = 20
# Left out of the pooled spatial corpus: `language_absent` has no task to
# fail at, and `object_clean` is a different suite (evaluated on its own).
EXCLUDED_SHARDS = frozenset({"language_absent", "object_clean"})
OBJECT_SHARD = "object_clean"
# Every per-step series a trace row carries.
TRACE_SIGNALS = (*PARQUET_SIGNALS, "visual_precondition")


def as_float(value: object) -> float:
    """A JSON number as float; raises on anything else instead of guessing."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"expected a number, got {type(value).__name__}: {value!r}")


def as_int(value: object) -> int:
    """A JSON integer as int; raises on anything else (a float like 2.5 too)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"expected an integer, got {type(value).__name__}: {value!r}")


def load_jsonl(path: Path) -> list[Row]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_key(row: Row) -> Key:
    return str(row["shard"]), int(row["episode_index"])  # type: ignore[arg-type]


def labels_by_key(path: Path) -> dict[Key, Row]:
    return {episode_key(row): row for row in load_jsonl(path)}


def spatial_pool(traces: list[Row], excluded: frozenset[str] = EXCLUDED_SHARDS) -> list[Row]:
    return [row for row in traces if str(row["shard"]) not in excluded]


def onset_range(labels: dict[Key, Row], shard: str, episode: int) -> tuple[int, int] | None:
    """The human-reviewed onset window, or None if the episode has none."""
    row = labels.get((shard, episode))
    review = row.get("review") if row is not None else None
    if not isinstance(review, dict):
        return None
    onset = review.get("onset_range")
    if not isinstance(onset, list) or len(onset) != 2:
        return None
    return int(onset[0]), int(onset[1])


def split_halves(
    rows: list[Row], *, eval_fraction: float, seed: int
) -> tuple[list[Row], list[Row]]:
    """(calibration, held-out), stratified by shard and outcome (ADR-004)."""
    keys = [(*episode_key(row), bool(row["success"])) for row in rows]
    split = stratified_split(keys, eval_fraction=eval_fraction, seed=seed)
    calibration = [row for row in rows if EpisodeKey(*episode_key(row)) in split.calibration]
    held_out = [row for row in rows if EpisodeKey(*episode_key(row)) in split.eval]
    return calibration, held_out


def running_trust(per_step: dict[str, list[float]], head: object) -> np.ndarray:
    """Trust head applied to the running max of each feature, per step."""
    matrix = np.stack(
        [np.asarray(per_step[name], dtype=np.float64) for name in TRUST_FEATURES], axis=1
    )
    return apply_trust_head(head, np.maximum.accumulate(matrix, axis=0))  # type: ignore[arg-type]


def series_map(rows: list[Row], name: str, head: object | None = None) -> dict[Key, np.ndarray]:
    """key -> one per-step series. `trust_running` needs a fitted head."""
    series: dict[Key, np.ndarray] = {}
    for row in rows:
        per_step = row["per_step"]
        if not isinstance(per_step, dict):
            raise ValueError(f"{episode_key(row)}: trace is missing per_step")
        if name == "trust_running":
            if head is None:
                raise ValueError("trust_running needs a fitted head")
            series[episode_key(row)] = running_trust(per_step, head)
        else:
            series[episode_key(row)] = np.asarray(per_step[name], dtype=np.float64)
    return series
