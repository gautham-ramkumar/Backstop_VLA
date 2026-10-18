"""Calibration/eval split for the headline metric (ADR-004).

Stratified by `(shard, success)`, not a flat random split: corpus A's
per-family success rates are deliberately different (that's the whole point
of calibrating each family to ~50%, Week2_plan.md's "why every family targets
~50%"), and per-family reporting is mandatory (README, Week2_plan.md's
risk table). A flat split could, by chance, put nearly all of one family's
failures in one half -- stratifying keeps every (shard, success) stratum
represented in both halves at roughly its overall proportion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeKey:
    shard: str
    episode_index: int


@dataclass(frozen=True)
class Split:
    calibration: frozenset[EpisodeKey]
    eval: frozenset[EpisodeKey]

    def __post_init__(self) -> None:
        overlap = self.calibration & self.eval
        if overlap:
            raise ValueError(f"{len(overlap)} episode(s) in both halves of the split")


def stratified_split(
    episodes: list[tuple[str, int, bool]], *, eval_fraction: float, seed: int
) -> Split:
    """`episodes`: (shard, episode_index, success) for every episode being
    split. Each `(shard, success)` stratum is shuffled independently under
    `seed` and cut at `eval_fraction`, so a stratum of size 1 always lands
    whole in one half rather than being silently dropped by rounding to 0."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    strata: dict[tuple[str, bool], list[tuple[str, int]]] = defaultdict(list)
    for shard, episode_index, success in episodes:
        strata[(shard, success)].append((shard, episode_index))

    rng = np.random.default_rng(seed)
    calibration: set[EpisodeKey] = set()
    eval_set: set[EpisodeKey] = set()
    for keys in strata.values():
        order = rng.permutation(len(keys))
        n_eval = max(1, round(len(keys) * eval_fraction)) if len(keys) > 1 else 0
        for rank, i in enumerate(order):
            shard, episode_index = keys[i]
            target = eval_set if rank < n_eval else calibration
            target.add(EpisodeKey(shard, episode_index))

    return Split(calibration=frozenset(calibration), eval=frozenset(eval_set))


def stratified_kfold(
    episodes: list[tuple[str, int, bool]], *, n_folds: int, seed: int
) -> list[Split]:
    """K disjoint eval folds, each stratified by `(shard, success)`.

    Week 2 sized the corpus and already said a single 30% holdout makes the
    detection rate jump in coarse steps. Each fold here is an eval set; the
    other folds are that fold's calibration set. A stratum smaller than
    `n_folds` simply misses some folds rather than being copied into two.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    strata: dict[tuple[str, bool], list[tuple[str, int]]] = defaultdict(list)
    for shard, episode_index, success in episodes:
        strata[(shard, success)].append((shard, episode_index))

    rng = np.random.default_rng(seed)
    fold_of: dict[EpisodeKey, int] = {}
    for keys in strata.values():
        order = rng.permutation(len(keys))
        for rank, index in enumerate(order):
            shard, episode_index = keys[int(index)]
            fold_of[EpisodeKey(shard, episode_index)] = rank % n_folds

    splits: list[Split] = []
    for fold in range(n_folds):
        evaluation = frozenset(key for key, assigned in fold_of.items() if assigned == fold)
        calibration = frozenset(key for key, assigned in fold_of.items() if assigned != fold)
        splits.append(Split(calibration=calibration, eval=evaluation))
    return splits
