"""One control step's decision: the executed action and the K chunks around it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from backstop.policy.base import ChunkPolicy, Processors, as_numpy
from backstop.record.timing import StageTimer


class UnnormalizeError(RuntimeError):
    """The postprocessor could not map a chunk from policy space to env space."""


@dataclass(frozen=True)
class Decision:
    """Everything in env space, batch of one.

    `action` is what the env executes, `action_policy` what the policy chose.
    They differ only under `action_noise` (ADR-003); every other family's
    `perturb_action` is the identity.
    """

    action: np.ndarray
    action_policy: np.ndarray
    chunk: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True)
class Decider:
    policy: ChunkPolicy
    processors: Processors
    k: int
    timer: StageTimer

    def __call__(
        self,
        observation: dict[str, Any],
        *,
        episode_seed: int,
        step: int,
        perturb: Callable[[np.ndarray], np.ndarray],
    ) -> Decision:
        import torch  # noqa: PLC0415
        from lerobot.utils.constants import ACTION  # noqa: PLC0415

        policy, timer = self.policy, self.timer
        with torch.inference_mode():
            # Timed apart because they answer different questions:
            # `policy_select` is the cost K cannot remove, `policy_samples` is the
            # cost K adds. The throughput benchmark reports the ratio between them.
            with timer("policy_select"):
                noise = policy.noise_for(episode_seed=episode_seed, step=step, index=0)
                action = policy.select_action(observation, noise=noise)
                chunk = policy.last_action_chunk()
            with timer("policy_samples"):
                samples = policy.sample_chunks(
                    observation, self.k, episode_seed=episode_seed, step=step
                )

        with timer("action_post"):
            post = self.processors.post
            transition = self.processors.env_post({ACTION: post(action)})
            action_policy = as_numpy(transition[ACTION])
            if action_policy.ndim == 1:
                action_policy = action_policy.reshape(1, -1)
            chunk = unnormalize_array(post, chunk)
            samples = unnormalize_array(post, samples)
            return Decision(perturb(action_policy), action_policy, chunk, samples)


def unnormalize_array(post: Any, array: np.ndarray) -> np.ndarray:
    """Policy-space -> env-space using the LeRobot postprocessor.

    A failure raises `UnnormalizeError` and stops the recording. Carrying on
    would store chunks in policy space next to executed actions in env space,
    so the two stop being comparable, and that cannot be repaired after the
    fact. `verify_corpus.py` checks the same thing on a finished corpus, by
    asserting `action == action_chunk[0]`.
    """
    import torch  # noqa: PLC0415

    try:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        if tensor.ndim == 3:
            k, t, a = tensor.shape
            out = post(tensor.reshape(k * t, a))
            return np.asarray(as_numpy(out), dtype=np.float32).reshape(k, t, -1)
        out = post(tensor)
        return np.asarray(as_numpy(out), dtype=np.float32).reshape(array.shape)
    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
        raise UnnormalizeError(
            f"unnormalize failed on shape {np.shape(array)} ({type(exc).__name__}: {exc}). "
            "Stored chunks would be in policy space while executed actions are in env "
            "space, which makes the corpus unusable for chunk-based signals."
        ) from exc
