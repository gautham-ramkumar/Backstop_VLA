"""Perturbation protocol and the RNG discipline every family follows.

A perturbation makes the environment harder in one named way, with a `strength`
in [0, 1] mapped to family-specific units. Week 2 tunes each family's strength so
roughly half its episodes still succeed: if a family always fails, a guard trained
on the corpus learns to detect the perturbation rather than the impending failure,
and a lighting-shifted image is trivially detectable.

Two hook points, deliberately in different places:

- `on_reset` mutates the simulator after the env resets. It runs inside a wrapper
  around each sub-env, so the vector env still produces the observation and the
  loop sees a scene that was already perturbed.
- `perturb_action` runs in the record loop, not the wrapper, because the loop has
  to store both what the policy asked for and what was executed. A wrapper would
  noise the action after the loop had already recorded it.
"""

from __future__ import annotations

import zlib
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Perturbation(Protocol):
    name: str
    strength: float
    # Whether this family needs `PerturbedEnv` at all -- i.e. whether `on_reset`
    # or `task_description` ever does anything. False for families that only
    # implement `perturb_action` (applied in the record loop, not the wrapper):
    # wrapping the env for them would be pure overhead, and would require every
    # vec_env, including a minimal test fake, to expose in-process sub-envs it has
    # no other reason to have. Read via `getattr(..., True)` at the call site, so
    # a family that omits this attribute defaults to the safe assumption.
    needs_env_hook: bool

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        """Mutate the simulator after reset. Return True if physics changed.

        A True return makes the wrapper re-settle before handing back an
        observation, because a moved object may be interpenetrating or floating.
        Purely visual changes return False and skip the settling.
        """
        ...

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The action actually sent to the env, given the one the policy chose."""
        ...

    def params(self) -> dict[str, Any]:
        """What was applied, for the label sidecar. Must be JSON-safe."""
        ...


class NoPerturbation:
    """The clean arm. Present so the clean and perturbed paths are the same code."""

    name = "clean"
    strength = 0.0
    needs_env_hook = False

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        return False

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return action

    def params(self) -> dict[str, Any]:
        return {}


def episode_rng(episode_seed: int, name: str) -> np.random.Generator:
    """A per-episode, per-family generator, independent of every other stream.

    Deliberately not the global numpy RNG and not torch's. The policy's flow noise
    comes from a dedicated torch generator keyed on (episode, step, sample) so that
    the executed trajectory does not depend on K; drawing perturbation randomness
    from a shared stream would reintroduce exactly that coupling, and a clean
    episode and its perturbed twin would stop being comparable.

    `crc32`, not `hash()`: Python salts string hashing per process, so a seed
    derived from `hash(name)` would differ between runs and a "reproducible"
    perturbed shard would silently not be.
    """
    digest = zlib.crc32(name.encode()) & 0xFFFFFFFF
    return np.random.default_rng([digest, int(episode_seed)])
