"""scripts/compare_recordings.py: what has to match exactly, and what is only advisory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import compare_recordings as cmp


def _recording(root: Path, *, action: float = 0.5, state: float = 1.0, video: bytes = b"v") -> Path:
    data = root / cmp.DATASET / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 0], "action": [action, 0.1]}), data / "f.parquet")
    (root / cmp.DATASET / "meta").mkdir()
    (root / cmp.DATASET / "meta" / "info.json").write_text(json.dumps({"fps": 20}))
    videos = root / cmp.DATASET / "videos"
    videos.mkdir()
    (videos / "ep.mp4").write_bytes(video)
    sidecar = root / cmp.PRIVILEGED / "golden_smoke"
    sidecar.mkdir(parents=True)
    np.savez(sidecar / "000000.npz", sim_state=np.full((2, 3), state))
    return root


def test_identical_recordings_have_no_differences(tmp_path: Path) -> None:
    before, after = _recording(tmp_path / "a"), _recording(tmp_path / "b")
    assert cmp.compare_trees(before, after) == ([], [])


def test_a_changed_column_names_the_column(tmp_path: Path) -> None:
    before, after = _recording(tmp_path / "a"), _recording(tmp_path / "b", action=0.6)
    exact, _ = cmp.compare_trees(before, after)
    assert exact == ["dataset/data/chunk-000/f.parquet: column 'action' differs"]


def test_a_changed_sim_state_is_an_exact_difference(tmp_path: Path) -> None:
    before, after = _recording(tmp_path / "a"), _recording(tmp_path / "b", state=2.0)
    exact, _ = cmp.compare_trees(before, after)
    assert exact == ["privileged/golden_smoke/000000.npz: array 'sim_state' differs"]


def test_re_encoded_video_is_only_advisory(tmp_path: Path) -> None:
    before, after = _recording(tmp_path / "a"), _recording(tmp_path / "b", video=b"w")
    exact, advisory = cmp.compare_trees(before, after)
    assert exact == []
    assert advisory == ["dataset/videos/ep.mp4: bytes differ"]


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    before, after = _recording(tmp_path / "a"), _recording(tmp_path / "b")
    (after / cmp.DATASET / "meta" / "info.json").unlink()
    exact, _ = cmp.compare_trees(before, after)
    assert exact == ["dataset/meta/info.json: only in before"]


def test_the_run_config_writes_only_under_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    raw = cmp.redirected_config(cmp.ROOT / "configs" / "record" / "golden_smoke.yaml", out)
    written = [raw["output_dir"], raw["record"]["root"], raw["record"]["privileged_root"]]
    assert all(Path(path).is_relative_to(out) for path in written)
    assert raw["wandb"]["enabled"] is False
