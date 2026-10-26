"""Where one episode's rows go: the LeRobot dataset and the privileged sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backstop.env.libero import sim_state
from backstop.record.privileged import PrivilegedWriter
from backstop.record.step import Decision
from backstop.record.timing import StageTimer
from backstop.schema import ACTION_DIM, CHUNK_SIZE


@dataclass
class EpisodeSink:
    """Either destination may be absent: an eval run writes neither."""

    dataset: Any | None
    privileged: PrivilegedWriter | None
    timer: StageTimer
    k: int
    episode_index: int = -1
    n_frames: int = 0

    def begin(self, episode_index: int) -> None:
        self.episode_index, self.n_frames = episode_index, 0
        if self.privileged is not None:
            self.privileged.begin_episode(episode_index)

    def add_sim_state(self, vec_env: Any) -> None:
        """Sim state *before* this step's action, so dataset row t and sidecar row t
        describe the same instant: restore sim_state[t], apply action[t], and the
        transition reproduces. Capturing after `vec_env.step` would offset them by
        one and silently shift every onset annotation."""
        if self.privileged is not None:
            with self.timer("privileged"):
                self.privileged.add(sim_state(vec_env))

    def add_frame(
        self, frame_obs: dict[str, np.ndarray], decision: Decision, *, success: bool, task: str
    ) -> None:
        self.n_frames += 1
        if self.dataset is not None:
            with self.timer("frame_write"):
                self.dataset.add_frame(build_frame(frame_obs, decision, success, task, self.k))

    def end(self) -> None:
        if self.dataset is not None:
            with self.timer("save_episode"):
                self.dataset.save_episode()
        if self.privileged is not None:
            n_states = self.privileged.n_steps_open
            if n_states != self.n_frames:
                raise RuntimeError(
                    f"episode {self.episode_index}: {n_states} sim states for {self.n_frames} "
                    "recorded frames. The sidecar and the dataset would not line up row for row."
                )
            self.privileged.end_episode()


def build_frame(
    frame_obs: dict[str, np.ndarray], decision: Decision, success: bool, task: str, k: int
) -> dict[str, Any]:
    return {
        "observation.images.image": frame_obs["image"],
        "observation.images.image2": frame_obs["image2"],
        "observation.state": frame_obs["state"],
        "action": np.asarray(decision.action[0], dtype=np.float32).reshape(ACTION_DIM),
        "action_policy": np.asarray(decision.action_policy[0], dtype=np.float32).reshape(
            ACTION_DIM
        ),
        "action_chunk": np.asarray(decision.chunk, dtype=np.float32).reshape(
            CHUNK_SIZE, ACTION_DIM
        ),
        "action_chunk_samples": np.asarray(decision.samples, dtype=np.float32).reshape(
            k, CHUNK_SIZE, ACTION_DIM
        ),
        "next.success": np.array([success], dtype=bool),
        "task": task,
    }
