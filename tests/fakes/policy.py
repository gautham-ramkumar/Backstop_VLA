"""A numpy-only `ChunkPolicy`, for testing the loop's pieces without a checkpoint.

Deterministic like the real policy: every chunk is a fixed ramp plus the flow
noise for its (episode, step, sample index), so the executed chunk does not
depend on K and two runs with the same seeds agree exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backstop.schema import ACTION_DIM, CHUNK_SIZE

BASE = (
    np.arange(CHUNK_SIZE, dtype=np.float32)[:, None] * 0.01
    + np.arange(ACTION_DIM, dtype=np.float32)[None, :] * 0.1
)


class FakeChunkPolicy:
    def __init__(self) -> None:
        self.n_resets = 0
        self.observations: list[dict[str, Any]] = []
        self._last: np.ndarray | None = None

    def reset(self) -> None:
        self.n_resets += 1
        self._last = None

    def noise_for(self, *, episode_seed: int, step: int, index: int) -> np.ndarray:
        rng = np.random.default_rng([episode_seed, step, index])
        return rng.standard_normal((1, CHUNK_SIZE, ACTION_DIM)).astype(np.float32)

    def select_action(self, observation: dict[str, Any], *, noise: Any = None) -> np.ndarray:
        self.observations.append(observation)
        chunk = BASE + np.asarray(noise, dtype=np.float32)[0]
        self._last = chunk
        return chunk[:1].copy()

    def last_action_chunk(self) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("last_action_chunk before select_action")
        return self._last

    def sample_chunks(
        self, observation: dict[str, Any], k: int, *, episode_seed: int, step: int
    ) -> np.ndarray:
        extras = [
            BASE + self.noise_for(episode_seed=episode_seed, step=step, index=index)[0]
            for index in range(1, k)
        ]
        return np.stack([self.last_action_chunk(), *extras])
