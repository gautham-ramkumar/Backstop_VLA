"""Corpus loading: only a shard with no run.json may be skipped, and loudly."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from backstop.corpus.layout import ShardLayout
from backstop.guard.extract import load_corpus_signals
from backstop.schema import RunProvenance


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
            (run_dir / "run.json").write_text(_run_json(dataset_root))
    return roots


def _run_json(dataset_root: str) -> str:
    provenance = RunProvenance(
        git_sha=None,
        policy_path="org/policy",
        policy_revision=None,
        policy_type="smolvla",
        device="cpu",
        mujoco_version=None,
        mujoco_gl="osmesa",
        seed=0,
        suite="libero_spatial",
        task_ids=None,
        n_episodes=1,
        n_action_steps=1,
        num_steps=10,
        k_samples=4,
        batch_size=1,
        dataset_root=dataset_root,
        output_dir="out",
    )
    return provenance.model_dump_json()


def _load(roots: dict[str, Path]) -> dict[str, list]:
    return load_corpus_signals(ShardLayout(roots["lerobot"], roots["privileged"], roots["eval"]))


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
