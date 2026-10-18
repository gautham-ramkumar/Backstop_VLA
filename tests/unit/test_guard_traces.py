"""Trace-row plumbing shared by every evaluation script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backstop.guard.split import EpisodeKey, stratified_split
from backstop.guard.traces import (
    episode_key,
    labels_by_key,
    load_jsonl,
    onset_range,
    series_map,
    spatial_pool,
    split_halves,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shard, n_success in (("clean", 14), ("lighting", 10), ("object_clean", 15)):
        for i in range(20):
            rows.append(
                {
                    "shard": shard,
                    "episode_index": i,
                    "success": i < n_success,
                    "per_step": {"action_chunk_disagreement": [0.1 * i, 0.2 * i]},
                }
            )
    return rows


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"shard": "a", "episode_index": 1}\n\n{"shard": "b", "episode_index": 2}\n')
    assert [episode_key(row) for row in load_jsonl(path)] == [("a", 1), ("b", 2)]
    assert set(labels_by_key(path)) == {("a", 1), ("b", 2)}


def test_spatial_pool_drops_the_excluded_shards() -> None:
    assert {row["shard"] for row in spatial_pool(_rows())} == {"clean", "lighting"}
    assert {row["shard"] for row in spatial_pool(_rows(), frozenset({"clean"}))} == {
        "lighting",
        "object_clean",
    }


def test_split_halves_is_the_stratified_split_as_rows() -> None:
    rows = spatial_pool(_rows())
    calibration, held_out = split_halves(rows, eval_fraction=0.3, seed=3)
    split = stratified_split(
        [(*episode_key(r), bool(r["success"])) for r in rows], eval_fraction=0.3, seed=3
    )
    assert {EpisodeKey(*episode_key(r)) for r in calibration} == split.calibration
    assert {EpisodeKey(*episode_key(r)) for r in held_out} == split.eval
    # Row order is preserved, so downstream output does not depend on set order.
    assert calibration == [r for r in rows if r in calibration]


@pytest.mark.parametrize(
    ("review", "expected"),
    [
        ({"onset_range": [30, 50]}, (30, 50)),
        ({"onset_range": None}, None),
        ({"onset_range": [30]}, None),
        (None, None),
    ],
)
def test_onset_range_reads_only_a_reviewed_window(
    review: dict[str, object] | None, expected: tuple[int, int] | None
) -> None:
    labels = {("clean", 0): {"review": review}}
    assert onset_range(labels, "clean", 0) == expected  # type: ignore[arg-type]
    assert onset_range(labels, "clean", 1) is None  # type: ignore[arg-type]


def test_series_map_reads_one_per_step_series() -> None:
    series = series_map(_rows()[:3], "action_chunk_disagreement")
    np.testing.assert_allclose(series[("clean", 2)], [0.2, 0.4])
    assert series[("clean", 0)].dtype == np.float64
