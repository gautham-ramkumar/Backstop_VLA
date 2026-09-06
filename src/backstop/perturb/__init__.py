"""Perturbation families for week 2.

Week 1 produced 30 natural failures and every one of them was the same failure:
stall until the 280-step cap. One mode cannot support a taxonomy, so week 2 makes
the environment harder in named, tunable ways and records what breaks.

    from backstop.perturb import make_perturbation
    perturbation = make_perturbation("lighting", strength=0.6)
"""

from __future__ import annotations

from typing import Any

from backstop.perturb.base import NoPerturbation, Perturbation, episode_rng
from backstop.perturb.families import AbsentTarget, ActionNoise, Lighting, ObjectPose
from backstop.perturb.wrapper import PerturbedEnv, attach

FAMILIES: dict[str, Any] = {
    NoPerturbation.name: NoPerturbation,
    Lighting.name: Lighting,
    ObjectPose.name: ObjectPose,
    ActionNoise.name: ActionNoise,
    AbsentTarget.name: AbsentTarget,
}

__all__ = [
    "FAMILIES",
    "AbsentTarget",
    "ActionNoise",
    "Lighting",
    "NoPerturbation",
    "ObjectPose",
    "PerturbedEnv",
    "Perturbation",
    "attach",
    "episode_rng",
    "make_perturbation",
]


def make_perturbation(name: str, strength: float = 0.0, **kwargs: Any) -> Perturbation:
    """Build a family by name. Unknown names raise rather than silently no-op.

    A typo'd family that quietly recorded a clean shard would look like a
    perturbation that had no effect -- a finding, rather than a mistake.
    """
    if name not in FAMILIES:
        raise ValueError(f"unknown perturbation {name!r}; known: {sorted(FAMILIES)}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    family = FAMILIES[name]
    if family is NoPerturbation:
        return NoPerturbation()
    if family is AbsentTarget:
        return AbsentTarget(**kwargs)
    return family(strength, **kwargs)
