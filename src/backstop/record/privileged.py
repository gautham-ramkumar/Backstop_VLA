"""Privileged per-episode sidecar, written outside the LeRobot dataset.

Contents are ground truth the policy never sees: the full flattened MuJoCo state
at every control step. Permitted uses are labelling, failure-onset annotation and
counterfactual branching (docs/recovery.md). **Forbidden as guard input** -- which
is why this lives in its own tree rather than as a dataset column. Week-3 signal
code loads the LeRobot dataset and therefore cannot reach any of it by accident.

One `.npz` per episode under `{root}/{shard}/{episode_index:06d}.npz`:

    sim_state  (T, D) float64   -- state *before* the action in dataset row t
    step       (T,)   int32     -- 0..T-1, so a truncated file is obvious

LIBERO-Spatial has D = 92, so an episode costs ~206 KB and a 400-episode corpus
~82 MB. Cheap enough that not recording it is the expensive choice: it cannot be
reconstructed later without re-running the episode on the GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PrivilegedWriter:
    """Accumulates one episode in memory, then writes it atomically on close.

    Deliberately strict. Every failure here raises instead of warning: a sidecar
    that silently stops writing looks exactly like one that was never enabled, and
    the difference only surfaces during week-2 labelling, after the GPU time is
    spent.
    """

    def __init__(self, root: str | Path, shard: str | None = None) -> None:
        self.dir = Path(root) / (shard or "default")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._episode: int | None = None
        self._states: list[np.ndarray] = []
        self._dim: int | None = None

    def begin_episode(self, episode_index: int) -> None:
        if self._episode is not None:
            raise RuntimeError(
                f"begin_episode({episode_index}) while episode {self._episode} is open; "
                "the previous episode's states would be lost."
            )
        self._episode = int(episode_index)
        self._states = []

    def add(self, state: Any) -> None:
        if self._episode is None:
            raise RuntimeError("add() called outside an episode; call begin_episode() first.")
        array = np.asarray(state, dtype=np.float64).reshape(-1)
        if array.size == 0:
            raise ValueError("empty sim state; the simulator returned nothing to record")
        if self._dim is None:
            self._dim = int(array.size)
        elif array.size != self._dim:
            # LIBERO states are fixed-width per task suite. A change mid-corpus means
            # the scene model changed, which would make the sidecar unstackable.
            raise ValueError(
                f"sim state width changed from {self._dim} to {array.size}; "
                "the scene model is not constant across this shard."
            )
        self._states.append(array)

    def end_episode(self) -> Path:
        if self._episode is None:
            raise RuntimeError("end_episode() called outside an episode.")
        if not self._states:
            raise RuntimeError(
                f"episode {self._episode} recorded no sim states; the capture hook did not run."
            )
        path = self.dir / f"{self._episode:06d}.npz"
        states = np.stack(self._states, axis=0)
        tmp = path.with_suffix(".npz.partial")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                sim_state=states,
                step=np.arange(states.shape[0], dtype=np.int32),
            )
        # Rename last: a crash mid-write leaves a .partial, never a truncated .npz
        # that would read back as a short episode.
        tmp.replace(path)
        logger.debug("privileged sidecar %s %s", path, states.shape)
        self._episode = None
        self._states = []
        return path

    @property
    def n_steps_open(self) -> int:
        return len(self._states)


def load_episode(root: str | Path, shard: str | None, episode_index: int) -> dict[str, np.ndarray]:
    """Read one sidecar back. Used by labelling, onset annotation and verification."""
    path = Path(root) / (shard or "default") / f"{episode_index:06d}.npz"
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}
