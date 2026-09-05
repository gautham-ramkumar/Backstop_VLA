"""Privileged sim-state sidecar: alignment, strictness, and where it must not go.

The sidecar is the only week-2 artefact that cannot be reconstructed offline -- a
missing or misaligned one costs a re-record. So every failure mode here raises
rather than warns, and these tests pin that down.

`docs/recovery.md` explains why the *full* flattened state is stored rather than a
curated pose dict: restoring it is what lets a later experiment branch from the
failure onset instead of replaying the whole episode.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from backstop.record.privileged import PrivilegedWriter, load_episode  # noqa: E402
from backstop.schema import lerobot_features  # noqa: E402

LIBERO_SPATIAL_STATE_DIM = 92


def _state(step: int, dim: int = LIBERO_SPATIAL_STATE_DIM) -> np.ndarray:
    return np.arange(dim, dtype=np.float64) + step


def test_round_trips_an_episode(tmp_path) -> None:
    writer = PrivilegedWriter(tmp_path, "clean")
    writer.begin_episode(7)
    for step in range(5):
        writer.add(_state(step))
    path = writer.end_episode()

    assert path == tmp_path / "clean" / "000007.npz"
    payload = load_episode(tmp_path, "clean", 7)
    assert payload["sim_state"].shape == (5, LIBERO_SPATIAL_STATE_DIM)
    assert payload["sim_state"].dtype == np.float64
    np.testing.assert_array_equal(payload["step"], np.arange(5))
    np.testing.assert_allclose(payload["sim_state"][3], _state(3))


def test_state_is_stored_at_full_float64_precision(tmp_path) -> None:
    """Restoring a downcast state lands the sim somewhere else.

    float32 would halve the file and silently break the branching use case, which
    is the entire reason the full state is captured rather than object poses.
    """
    writer = PrivilegedWriter(tmp_path, "clean")
    writer.begin_episode(0)
    exact = np.array([0.1234567890123456789] * LIBERO_SPATIAL_STATE_DIM, dtype=np.float64)
    writer.add(exact)
    writer.end_episode()

    restored = load_episode(tmp_path, "clean", 0)["sim_state"][0]
    assert restored.dtype == np.float64
    np.testing.assert_array_equal(restored, exact)


def test_episode_files_are_written_atomically(tmp_path) -> None:
    """A crash mid-write must not leave a short episode that reads back cleanly."""
    writer = PrivilegedWriter(tmp_path, "clean")
    writer.begin_episode(1)
    writer.add(_state(0))
    writer.end_episode()

    assert list((tmp_path / "clean").glob("*.partial")) == []
    assert (tmp_path / "clean" / "000001.npz").is_file()


def test_opening_an_episode_twice_raises_rather_than_dropping_states(tmp_path) -> None:
    writer = PrivilegedWriter(tmp_path, "clean")
    writer.begin_episode(0)
    writer.add(_state(0))
    with pytest.raises(RuntimeError, match="while episode 0 is open"):
        writer.begin_episode(1)


def test_an_episode_with_no_states_is_an_error_not_an_empty_file(tmp_path) -> None:
    """A silently-empty sidecar looks exactly like one that was never enabled."""
    writer = PrivilegedWriter(tmp_path, "clean")
    writer.begin_episode(0)
    with pytest.raises(RuntimeError, match="no sim states"):
        writer.end_episode()


def test_add_outside_an_episode_raises(tmp_path) -> None:
    writer = PrivilegedWriter(tmp_path, "clean")
    with pytest.raises(RuntimeError, match="outside an episode"):
        writer.add(_state(0))


def test_a_changing_state_width_is_rejected(tmp_path) -> None:
    """Fixed width per suite; a change means the scene model moved under us."""
    writer = PrivilegedWriter(tmp_path, "clean")
    writer.begin_episode(0)
    writer.add(_state(0, dim=92))
    with pytest.raises(ValueError, match="width changed from 92 to 96"):
        writer.add(_state(1, dim=96))


def test_shards_do_not_collide_on_episode_index(tmp_path) -> None:
    """Every shard restarts at episode 0; one flat directory would overwrite."""
    for shard in ("clean", "lighting"):
        writer = PrivilegedWriter(tmp_path, shard)
        writer.begin_episode(0)
        writer.add(_state(0) * (1 if shard == "clean" else -1))
        writer.end_episode()

    clean = load_episode(tmp_path, "clean", 0)["sim_state"]
    lighting = load_episode(tmp_path, "lighting", 0)["sim_state"]
    assert not np.array_equal(clean, lighting)


def test_sim_state_is_not_a_dataset_column() -> None:
    """ADR-003: privileged state is forbidden as guard input.

    The enforcement is structural -- week-3 signal code loads the LeRobot dataset,
    so anything absent from `lerobot_features` is unreachable by accident. If a
    future change adds it as a column, that prohibition is gone.
    """
    features = lerobot_features(k_samples=4)
    for key in features:
        assert "sim_state" not in key
        assert "privileged" not in key
