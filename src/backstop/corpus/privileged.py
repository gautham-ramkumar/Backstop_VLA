"""Reading the privileged sim-state sidecar (`backstop.record.privileged` writes it).

Ground truth the policy never sees, permitted for labelling, onset annotation and
counterfactual branching, and forbidden as guard input (ADR-003).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def episode_path(root: str | Path, shard: str | None, episode_index: int) -> Path:
    return Path(root) / (shard or "default") / f"{episode_index:06d}.npz"


def load_episode(root: str | Path, shard: str | None, episode_index: int) -> dict[str, np.ndarray]:
    with np.load(episode_path(root, shard, episode_index)) as payload:
        return {key: payload[key] for key in payload.files}


def load_sim_state(root: str | Path, shard: str, episode_index: int) -> np.ndarray | None:
    """The episode's (T, D) sim states, or None if it has no sidecar."""
    try:
        return load_episode(root, shard, episode_index).get("sim_state")
    except FileNotFoundError:
        return None
