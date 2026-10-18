"""Corpus loading: only a shard with no run.json may be skipped, and loudly."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from backstop.guard.extract import load_corpus_signals


def _roots(tmp_path: Path, shards: dict[str, str | None]) -> dict[str, Path]:
    """privileged/<shard>/ for each shard; eval/spatial_<shard>_k4/run.json when given."""
    roots = {name: tmp_path / name for name in ("privileged", "eval", "lerobot")}
    for root in roots.values():
        root.mkdir()
    for shard, dataset_root in shards.items():
        (roots["privileged"] / shard).mkdir()
        if dataset_root is not None:
            run_dir = roots["eval"] / f"spatial_{shard}_k4"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(json.dumps({"dataset_root": dataset_root}))
    return roots


def _load(roots: dict[str, Path]) -> dict[str, list]:
    return load_corpus_signals(
        lerobot_root=roots["lerobot"],
        privileged_root=roots["privileged"],
        eval_root=roots["eval"],
    )


def test_a_shard_without_run_json_is_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    roots = _roots(tmp_path, {"leftover": None})
    with caplog.at_level(logging.WARNING, logger="backstop.guard.extract"):
        assert _load(roots) == {}
    assert "leftover" in caplog.text


def test_a_real_shard_with_missing_data_raises(tmp_path: Path) -> None:
    """It used to be skipped like the leftover above, dropping its rows silently."""
    missing = tmp_path / "gone" / "backstop_spatial_clean_k4"
    roots = _roots(tmp_path, {"clean": str(missing)})
    with pytest.raises(FileNotFoundError):
        _load(roots)
