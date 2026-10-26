"""Trace and label rows, typed, and the plumbing shared by every evaluation script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backstop.corpus.episode import EpisodeKey
from backstop.guard.split import stratified_split
from backstop.guard.traces import (
    LabelView,
    TraceRow,
    load_labels,
    load_traces,
    series_map,
    spatial_pool,
    split_halves,
    write_traces,
)

SIGNAL = "action_chunk_disagreement"


def _rows() -> list[TraceRow]:
    return [
        TraceRow(
            shard=shard,
            episode_index=i,
            success=i < n_success,
            n_steps=2,
            task="pick up the black bowl",
            scores={SIGNAL: 0.2 * i},
            per_step={SIGNAL: np.array([0.1 * i, 0.2 * i])},
        )
        for shard, n_success in (("clean", 14), ("lighting", 10), ("object_clean", 15))
        for i in range(20)
    ]


def test_traces_round_trip_byte_for_byte_with_the_field_order_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    write_traces(path, _rows())
    first = json.loads(path.read_text().splitlines()[0])
    assert list(first) == [
        "shard",
        "episode_index",
        "success",
        "n_steps",
        "task",
        "scores",
        "per_step",
    ]
    again = tmp_path / "again.jsonl"
    write_traces(again, load_traces(path))
    assert again.read_bytes() == path.read_bytes()


def test_label_view_reads_only_a_two_step_reviewed_window(tmp_path: Path) -> None:
    def row(index: int, review: object) -> str:
        return json.dumps(
            {"shard": "clean", "episode_index": index, "success": False, "n_steps": 280}
            | ({"review": review} if review is not None else {})
        )

    reviews = [{"onset_range": [30, 50]}, {"onset_range": None}, {"onset_range": [30]}, None]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(row(i, review) for i, review in enumerate(reviews)) + "\n\n")
    labels = load_labels(path)
    assert [labels[EpisodeKey("clean", i)].onset_range for i in range(4)] == [
        (30, 50),
        None,
        None,
        None,
    ]
    assert labels[EpisodeKey("clean", 0)] == LabelView(EpisodeKey("clean", 0), False, 280, (30, 50))


def test_episode_key_is_interchangeable_with_the_plain_tuple() -> None:
    key = EpisodeKey("clean", 3)
    assert key == ("clean", 3)
    assert {("clean", 3): "x"}[key] == "x"
    assert hash(key) == hash(("clean", 3))


def test_spatial_pool_drops_the_excluded_shards() -> None:
    assert {row.shard for row in spatial_pool(_rows())} == {"clean", "lighting"}
    assert {row.shard for row in spatial_pool(_rows(), frozenset({"clean"}))} == {
        "lighting",
        "object_clean",
    }


def test_split_halves_is_the_stratified_split_as_rows() -> None:
    rows = spatial_pool(_rows())
    calibration, held_out = split_halves(rows, eval_fraction=0.3, seed=3)
    split = stratified_split(
        [(r.shard, r.episode_index, r.success) for r in rows], eval_fraction=0.3, seed=3
    )
    assert {r.key for r in calibration} == split.calibration
    assert {r.key for r in held_out} == split.eval
    # Row order is preserved, so downstream output does not depend on set order.
    assert calibration == [r for r in rows if r in calibration]


@pytest.mark.parametrize("index", [0, 2])
def test_series_map_reads_one_per_step_series(index: int) -> None:
    series = series_map(_rows()[:3], SIGNAL)
    np.testing.assert_allclose(series[EpisodeKey("clean", index)], [0.1 * index, 0.2 * index])
    assert series[EpisodeKey("clean", index)].dtype == np.float64


def test_no_untyped_row_casts_remain(repo_root: Path) -> None:
    """The typed rows replaced every cast the raw dicts needed; keep it that way."""
    casts = [
        f"{path.relative_to(repo_root)}:{number}"
        for folder in ("src", "scripts")
        for path in sorted((repo_root / folder).rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if "type: ignore[arg-type]" in line
    ]
    assert not casts, casts
