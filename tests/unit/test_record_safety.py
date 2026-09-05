"""Recording must not destroy a corpus it did not create.

Week 1's `create_dataset` called `shutil.rmtree(root)` unconditionally, so any
re-run -- a typo'd config, a repeated `just record-spatial`, a resumed shell --
silently deleted hours of GPU output before writing its first frame. `data/` is
gitignored, so there was nothing to recover from.

These tests are CPU-only and never construct a real LeRobotDataset: they assert
on the decision `create_dataset` makes *before* it touches the filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# `backstop.record.loop` imports numpy at module scope, and numpy reaches this
# project through the `sim` extra. The CPU-only CI job must skip this module rather
# than fail to collect it -- the `record-contract` job runs it with sim installed.
pytest.importorskip("numpy")

from backstop.config import PipelineConfig  # noqa: E402
from backstop.record import loop as record_loop  # noqa: E402
from backstop.schema import ACTION_DIM, CHUNK_SIZE  # noqa: E402

FPS = 20


def _cfg(root: Path, *, on_existing: str = "fail", k_samples: int = 4, **extra: Any):
    return PipelineConfig.model_validate(
        {
            "seed": 0,
            "output_dir": str(root.parent / "out"),
            "policy": {"type": "smolvla", "path": "fake", "device": "cpu"},
            "env": {"type": "libero", "task": "libero_spatial"},
            "eval": {"n_episodes": 1},
            "record": {
                "enabled": True,
                "k_samples": k_samples,
                "root": str(root),
                "on_existing": on_existing,
                **extra,
            },
        }
    )


def _populate(root: Path, *, k_samples: int = 4, fps: int = FPS, episodes: int = 12) -> None:
    """Write just enough of a LeRobot v3 tree to look like a real corpus."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": episodes,
                "total_frames": episodes * 100,
                "fps": fps,
                "features": record_loop.dataset_features(k_samples),
            },
            default=list,
        )
    )
    (root / "data").mkdir()
    (root / "data" / "irreplaceable.parquet").write_bytes(b"12 hours of GPU")


class FakeMeta:
    def __init__(self, features: dict[str, dict], fps: int, episodes: int) -> None:
        self.features = features
        self.fps = fps
        self.total_episodes = episodes


class FakeDataset:
    def __init__(self, meta: FakeMeta) -> None:
        self.meta = meta


@pytest.fixture
def fake_lerobot(monkeypatch):
    """Stand in for `LeRobotDataset`; records which classmethod was reached."""
    calls: dict[str, Any] = {}
    on_disk: dict[str, Any] = {}

    class FakeLeRobotDataset:
        @classmethod
        def create(cls, **kwargs: Any) -> Any:
            calls["create"] = kwargs
            return FakeDataset(FakeMeta(kwargs["features"], kwargs["fps"], 0))

        @classmethod
        def resume(cls, **kwargs: Any) -> Any:
            calls["resume"] = kwargs
            return FakeDataset(
                FakeMeta(
                    on_disk.get("features", record_loop.dataset_features(4)),
                    on_disk.get("fps", FPS),
                    on_disk.get("episodes", 12),
                )
            )

    module = type("M", (), {"LeRobotDataset": FakeLeRobotDataset})
    monkeypatch.setitem(
        __import__("sys").modules,
        "lerobot.datasets.lerobot_dataset",
        module,  # type: ignore[arg-type]
    )
    return calls, on_disk


def test_fresh_root_creates_and_reports_zero_done(tmp_path, fake_lerobot) -> None:
    calls, _ = fake_lerobot
    _dataset, n_done = record_loop.create_dataset(_cfg(tmp_path / "ds"), fps=FPS)
    assert "create" in calls and "resume" not in calls
    assert n_done == 0


def test_existing_corpus_is_not_destroyed_by_default(tmp_path, fake_lerobot) -> None:
    """The regression that motivates this file."""
    root = tmp_path / "ds"
    _populate(root)
    canary = root / "data" / "irreplaceable.parquet"

    with pytest.raises(FileExistsError, match="on_existing"):
        record_loop.create_dataset(_cfg(root), fps=FPS)

    assert canary.is_file(), "existing corpus was deleted despite on_existing=fail"
    assert canary.read_bytes() == b"12 hours of GPU"


def test_default_is_fail_even_when_config_omits_the_key() -> None:
    """Safety must be the default, not something a config has to remember to set."""
    cfg = PipelineConfig.model_validate(
        {
            "seed": 0,
            "output_dir": "out",
            "policy": {"type": "smolvla", "path": "fake"},
            "env": {"type": "libero"},
            "eval": {"n_episodes": 1},
            "record": {"enabled": True, "root": "data/x"},
        }
    )
    assert cfg.record.on_existing == "fail"


def test_overwrite_is_honoured_when_asked_for_explicitly(tmp_path, fake_lerobot) -> None:
    calls, _ = fake_lerobot
    root = tmp_path / "ds"
    _populate(root)

    _dataset, n_done = record_loop.create_dataset(_cfg(root, on_existing="overwrite"), fps=FPS)

    assert not (root / "data" / "irreplaceable.parquet").exists()
    assert "create" in calls
    assert n_done == 0


def test_resume_reports_episodes_already_recorded(tmp_path, fake_lerobot) -> None:
    calls, on_disk = fake_lerobot
    root = tmp_path / "ds"
    _populate(root, episodes=12)
    on_disk["episodes"] = 12

    _dataset, n_done = record_loop.create_dataset(_cfg(root, on_existing="resume"), fps=FPS)

    assert "resume" in calls and "create" not in calls
    assert n_done == 12, "resume must tell the loop how many episodes to skip"
    assert (root / "data" / "irreplaceable.parquet").is_file()


def test_resume_refuses_a_shard_recorded_at_a_different_k(tmp_path, fake_lerobot) -> None:
    """`LeRobotDataset.resume` adopts the on-disk schema, so K silently disagrees.

    Without this check a K=4 run appends to a K=1 shard and the corpus ends up
    half-usable for chunk disagreement, discoverable only by reading it back.
    """
    _calls, on_disk = fake_lerobot
    root = tmp_path / "ds"
    _populate(root, k_samples=1)
    on_disk["features"] = record_loop.dataset_features(1)

    with pytest.raises(ValueError, match="action_chunk_samples"):
        record_loop.create_dataset(_cfg(root, on_existing="resume", k_samples=4), fps=FPS)


def test_resume_refuses_a_shard_recorded_at_a_different_fps(tmp_path, fake_lerobot) -> None:
    _calls, on_disk = fake_lerobot
    root = tmp_path / "ds"
    _populate(root, fps=30)
    on_disk["fps"] = 30

    with pytest.raises(ValueError, match="fps"):
        record_loop.create_dataset(_cfg(root, on_existing="resume"), fps=FPS)


def test_resume_shape_check_covers_the_chunk_columns(tmp_path, fake_lerobot) -> None:
    """Guards the check itself: the shapes compared must be the ones that matter."""
    features = record_loop.dataset_features(4)
    assert features["action_chunk_samples"]["shape"] == (4, CHUNK_SIZE, ACTION_DIM)
    assert features["action_chunk"]["shape"] == (CHUNK_SIZE, ACTION_DIM)


def test_a_directory_without_metadata_is_not_treated_as_a_corpus(tmp_path, fake_lerobot) -> None:
    """An interrupted mkdir should not wedge the next run into on_existing=fail."""
    calls, _ = fake_lerobot
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)

    _dataset, n_done = record_loop.create_dataset(_cfg(root), fps=FPS)
    assert "create" in calls
    assert n_done == 0


def test_a_named_shard_must_decide_about_privileged_state(tmp_path, fake_lerobot) -> None:
    """The one cost week 2 cannot pay later, and the quietest way to skip it.

    `privileged` defaults to false, so a shard config that omits the key records a
    complete, verifiable corpus with no sim state in it -- and nothing downstream
    complains until labelling, after the GPU time is spent. `spatial_clean_k4.yaml`
    shipped exactly that omission.
    """
    calls, _ = fake_lerobot
    with pytest.raises(ValueError, match="record.privileged is unset"):
        record_loop.create_dataset(_cfg(tmp_path / "ds", shard="clean"), fps=FPS)
    assert "create" not in calls, "refused before touching the dataset, not after"


def test_declining_privileged_state_on_purpose_is_allowed(tmp_path, fake_lerobot) -> None:
    """`privileged: false` written down is a decision; omitting it is not."""
    calls, _ = fake_lerobot
    _dataset, n_done = record_loop.create_dataset(
        _cfg(tmp_path / "ds", shard="clean", privileged=False), fps=FPS
    )
    assert "create" in calls
    assert n_done == 0


def test_an_unsharded_run_is_not_asked(tmp_path, fake_lerobot) -> None:
    """Smoke and eval runs are not corpus shards. Gating them would teach the
    reflex of setting the key to silence an error rather than to mean it."""
    calls, _ = fake_lerobot
    _dataset, n_done = record_loop.create_dataset(_cfg(tmp_path / "ds"), fps=FPS)
    assert "create" in calls
    assert n_done == 0


def test_every_shipped_record_config_has_decided(tmp_path) -> None:
    """The check is only worth having if the configs actually pass it.

    Loads each `configs/record/*.yaml` for real, so a shard config added later
    without the key fails here rather than at 2am on the GPU.
    """
    configs = sorted(Path("configs/record").glob("*.yaml"))
    assert configs, "no record configs found; this test is silently passing"
    for path in configs:
        cfg = PipelineConfig.from_yaml(path)
        if cfg.record.shard is None:
            continue
        assert "privileged" in cfg.record.model_fields_set, (
            f"{path} names shard {cfg.record.shard!r} but never decides about privileged sim state"
        )
