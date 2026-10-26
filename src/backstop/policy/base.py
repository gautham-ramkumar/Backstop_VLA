"""What the rollout loop needs from a policy, and the processors around it.

`ChunkPolicy` is the surface the loop actually calls, no more. It exists so the
loop and the recovery runner can be tested with a numpy-only fake
(`tests/fakes/policy.py`) instead of a 500M-parameter checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class UnsupportedPolicyError(NotImplementedError):
    """Raised for a policy type the config accepts but nothing implements yet."""


class ChunkPolicy(Protocol):
    def reset(self) -> None: ...

    def noise_for(self, *, episode_seed: int, step: int, index: int) -> Any:
        """The flow noise for sample `index` at `step`, a pure function of its arguments."""
        ...

    def select_action(self, observation: dict[str, Any], *, noise: Any = None) -> Any:
        """The executed action, in policy space, drawn with `noise`."""
        ...

    def last_action_chunk(self) -> Any:
        """The chunk the last `select_action` came from, `(chunk_size, action_dim)`."""
        ...

    def sample_chunks(
        self, observation: dict[str, Any], k: int, *, episode_seed: int, step: int
    ) -> Any:
        """K chunks, index 0 the executed one, without moving the executed trajectory."""
        ...


@dataclass(frozen=True)
class Processors:
    """LeRobot's four processing steps, in the order a step applies them.

    `env_pre` formats the env observation, `pre` normalizes it for the policy,
    `post` unnormalizes the action, and `env_post` formats it for the env.
    """

    env_pre: Any
    pre: Any
    post: Any
    env_post: Any
