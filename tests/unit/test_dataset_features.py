"""LeRobot dataset schema expected by the recorder (no GPU)."""

from __future__ import annotations

from backstop.schema import ACTION_DIM, CHUNK_SIZE, STATE_DIM, lerobot_features


def test_dataset_features_include_chunk_and_samples() -> None:
    features = lerobot_features(k_samples=4)
    assert set(features) >= {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "action_chunk",
        "action_chunk_samples",
        "next.success",
    }
    assert features["observation.state"]["shape"] == (STATE_DIM,)
    assert features["action"]["shape"] == (ACTION_DIM,)
    assert features["action_chunk"]["shape"] == (CHUNK_SIZE, ACTION_DIM)
    assert features["action_chunk_samples"]["shape"] == (4, CHUNK_SIZE, ACTION_DIM)
    assert features["observation.images.image"]["dtype"] == "video"
