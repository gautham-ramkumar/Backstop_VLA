"""Applies reset-time perturbations inside the vector env.

The perturbation has to land *inside* the sub-env rather than after
`vec_env.reset()` returns, because the observation the loop receives must show
the perturbed scene. Perturbing from outside would mutate the simulator while the
loop held an image of the world as it was before -- the policy would act on a
stale frame for one step, and the corpus would store that frame as if it were what
the policy saw.

`SyncVectorEnv` iterates `self.envs` on both reset and step, so replacing entries
in that list is enough; there is no factory to hook. Autoreset goes through
`self.envs[i].reset()` too, so the perturbation reapplies on its own.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym

from backstop.perturb.base import Perturbation, episode_rng

logger = logging.getLogger(__name__)

# LIBERO's own post-reset settle count. Matched on purpose: an object moved by a
# perturbation gets the same chance to come to rest as one placed by an init state.
SETTLE_STEPS = 10


class PerturbedEnv(gym.Wrapper):
    """Applies `perturbation.on_reset` after the wrapped env resets."""

    def __init__(self, env: gym.Env, perturbation: Perturbation, episode_seed: int) -> None:
        super().__init__(env)
        self.perturbation = perturbation
        self.episode_seed = episode_seed
        # A third, optional hook for the one family that cannot use `on_reset`:
        # `distractor` edits the BDDL and recompiles the MuJoCo model, which has
        # to happen *before* the wrapped env's first `reset()` ever builds it --
        # `on_reset` already runs too late, after `self.env.reset()` below has
        # called `LiberoEnv._ensure_env()`. `__init__` runs exactly once per
        # sub-env (`attach()` only constructs a `PerturbedEnv` the first time),
        # which is exactly the "before any reset" window this needs. Every other
        # family omits `prepare_env`, so this is a no-op for them.
        prepare_env = getattr(perturbation, "prepare_env", None)
        if prepare_env is not None:
            prepare_env(env.unwrapped)

    def set_episode_seed(self, episode_seed: int) -> None:
        """Vary the perturbation per episode. Same scene twice would make a shard
        50 samples of one draw rather than 50 draws."""
        self.episode_seed = int(episode_seed)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        inner = self.env.unwrapped
        control_env = getattr(inner, "_env", None)
        if control_env is None:
            raise RuntimeError(
                "PerturbedEnv wraps something that is not a LeRobot LiberoEnv: no `_env`."
            )
        rng = episode_rng(self.episode_seed, self.perturbation.name)
        changed = self.perturbation.on_reset(control_env, rng)
        observation = self._observe(inner, control_env, resettle=changed)
        # `task_description` is a plain attribute on the LeRobot LiberoEnv (set once
        # from the task suite, not recomputed per reset), not a method -- so this
        # sets an instance attribute on the wrapper itself rather than calling
        # through. `get_wrapper_attr` (what `vec_env.call("task_description")` uses)
        # checks `hasattr(self, name)` on the outermost wrapper first, so this value
        # is what the loop and the recorded frame both see; the language families
        # are the only ones that define `task_description`, so every other family
        # leaves the original untouched.
        self.task_description = self._perturbed_task_description(
            getattr(inner, "task_description")  # noqa: B009 -- duck-typed, like `_env` above
        )
        info = dict(info)
        info["perturbation"] = self.perturbation.name
        return observation, info

    def _perturbed_task_description(self, original: str) -> str:
        override = getattr(self.perturbation, "task_description", None)
        return original if override is None else override(original)

    def _observe(self, inner: Any, control_env: Any, *, resettle: bool) -> Any:
        """Rebuild the observation from the perturbed simulator.

        Re-settling steps the *inner* robosuite env, not the outer one, so it costs
        no control steps and cannot desynchronise the episode length from the clean
        arm -- a perturbed episode has to stay comparable to its clean twin.
        """
        if resettle:
            # LeRobot's helper, not LIBERO's -- the same no-op action its own reset
            # settles with, so a perturbed scene settles exactly like a clean one.
            from lerobot.envs.libero import get_libero_dummy_action  # noqa: PLC0415

            for _ in range(SETTLE_STEPS):
                control_env.step(get_libero_dummy_action())
        raw = control_env.env._get_observations(force_update=True)
        return inner._format_raw_obs(raw)


def attach(vec_env: Any, perturbation: Perturbation, episode_seed: int) -> Any:
    """Wrap every sub-env, or update the seed if they are already wrapped.

    Idempotent because the record loop calls this every episode: wrapping a wrapper
    would re-apply the perturbation once per layer, compounding a strength the
    calibration sweep chose exactly once.
    """
    envs = getattr(vec_env, "envs", None)
    if envs is None:
        raise RuntimeError(
            "Perturbations need in-process sub-envs (use_async_envs=False); this "
            f"{type(vec_env).__name__} exposes no `.envs`."
        )
    for index, env in enumerate(envs):
        if isinstance(env, PerturbedEnv):
            env.set_episode_seed(episode_seed)
        else:
            envs[index] = PerturbedEnv(env, perturbation, episode_seed)
    return vec_env
