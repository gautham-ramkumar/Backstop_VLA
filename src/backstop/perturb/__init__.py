"""Perturbation families: named, tunable ways to make the environment harder.

The policy's 30 natural failures were all the same failure, a stall until the
280-step cap. One mode cannot support a taxonomy, so the corpus makes the
environment harder in named ways and records what breaks.

    from backstop.perturb import make_perturbation
    perturbation = make_perturbation("lighting", strength=0.6)
"""

from __future__ import annotations

from typing import Any

from backstop.perturb.base import NoPerturbation, Perturbation, episode_rng
from backstop.perturb.families import (
    AbsentTarget,
    ActionNoise,
    Camera,
    Distractor,
    Lighting,
    ObjectPose,
    Paraphrase,
)
from backstop.perturb.wrapper import PerturbedEnv, attach

FAMILIES: dict[str, Any] = {
    NoPerturbation.name: NoPerturbation,
    Lighting.name: Lighting,
    Camera.name: Camera,
    ObjectPose.name: ObjectPose,
    ActionNoise.name: ActionNoise,
    Distractor.name: Distractor,
    AbsentTarget.name: AbsentTarget,
    Paraphrase.name: Paraphrase,
}

# Families whose constructor takes a supplied value (the text to swap in) rather
# than a sampled `strength`. Both are categorical: there is nothing for a knob to
# scale, so treating them like the numeric families would need a fake strength
# that means nothing.
_CATEGORICAL: dict[str, Any] = {
    NoPerturbation.name: NoPerturbation,
    AbsentTarget.name: AbsentTarget,
    Paraphrase.name: Paraphrase,
}

__all__ = [
    "FAMILIES",
    "AbsentTarget",
    "ActionNoise",
    "Camera",
    "Distractor",
    "Lighting",
    "NoPerturbation",
    "ObjectPose",
    "Paraphrase",
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
    if name in _CATEGORICAL:
        return family(**kwargs)
    return family(strength, **kwargs)
