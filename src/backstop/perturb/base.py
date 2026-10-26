"""The perturbation base class and the RNG discipline every family follows.

A perturbation makes the environment harder in one named way, with a `strength`
in [0, 1] mapped to family-specific units. Each family's strength is tuned so
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
from typing import Any

import numpy as np


class Perturbation:
    """Base of every family. The defaults are the do-nothing behaviour, so a family
    overrides only the hooks it uses.

    `needs_env_hook` says whether the family needs `PerturbedEnv` at all, i.e.
    whether `prepare_env`, `on_reset` or `task_description` ever does anything.
    False for families that only implement `perturb_action` (applied in the record
    loop, not the wrapper): wrapping the env for them would be pure overhead, and
    would require every vec_env, including a minimal test fake, to expose
    in-process sub-envs it has no other reason to have.
    """

    name: str
    strength: float
    needs_env_hook: bool = True

    @classmethod
    def from_config(cls, strength: float, **kwargs: Any) -> Perturbation:
        """Build from a config's `perturbation_strength` and `perturbation_kwargs`."""
        return cls(strength, **kwargs)  # type: ignore[call-arg]

    def prepare_env(self, env: Any) -> None:
        """Edit a LeRobot `LiberoEnv` once, before its simulator is first built."""

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        """Mutate the simulator after reset. Return True if physics changed.

        A True return makes the wrapper re-settle before handing back an
        observation, because a moved object may be interpenetrating or floating.
        Purely visual changes return False and skip the settling.
        """
        return False

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The action actually sent to the env, given the one the policy chose."""
        return action

    def task_description(self, original: str) -> str:
        """The instruction the policy sees, given the task's own."""
        return original

    def params(self) -> dict[str, Any]:
        """What was applied, for the label sidecar. Must be JSON-safe."""
        return {}


class NoPerturbation(Perturbation):
    """The clean arm. Present so the clean and perturbed paths are the same code."""

    name = "clean"
    strength = 0.0
    needs_env_hook = False

    @classmethod
    def from_config(cls, strength: float, **kwargs: Any) -> Perturbation:
        return cls()


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
