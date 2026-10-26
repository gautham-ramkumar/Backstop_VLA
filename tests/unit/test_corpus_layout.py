"""Where a shard's files are: run record, dataset and recording config."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from backstop.corpus.layout import ShardLayout


def test_run_stem_prefers_the_spatial_name_and_falls_back_to_the_suite_name(
    tmp_path: Path,
) -> None:
    (tmp_path / "spatial_clean_k4").mkdir()
    assert ShardLayout.run_stem(tmp_path, "clean") == tmp_path / "spatial_clean_k4"
    assert ShardLayout.run_stem(tmp_path, "object_clean") == tmp_path / "object_clean_k4"


def test_perturbation_params_come_from_the_recording_config(tmp_path: Path) -> None:
    (tmp_path / "spatial_language_absent_k4.yaml").write_text(
        "record:\n  perturbation_kwargs:\n    replacement: pick up the moka pot\n"
    )
    layout = ShardLayout(configs_root=tmp_path)
    assert layout.perturbation_params("language_absent") == {"replacement": "pick up the moka pot"}
    assert layout.perturbation_params("clean") == {}


def test_a_shard_without_a_run_record_has_no_dataset(tmp_path: Path) -> None:
    layout = ShardLayout(eval_root=tmp_path)
    assert layout.provenance("clean") is None
    with pytest.raises(FileNotFoundError, match="no run.json"):
        layout.dataset_root("clean")


def test_the_cli_flags_default_to_the_repo_layout_and_override_one_root() -> None:
    parser = argparse.ArgumentParser()
    ShardLayout.add_args(parser)
    assert ShardLayout.from_args(parser.parse_args([])) == ShardLayout()
    custom = ShardLayout.from_args(parser.parse_args(["--eval-root", "elsewhere"]))
    assert custom == ShardLayout(eval_root=Path("elsewhere"))
