"""Contract for the stratified calibration/eval split (ADR-004)."""

from __future__ import annotations

import pytest

from backstop.guard.split import EpisodeKey, Split, stratified_kfold, stratified_split


def _episodes() -> list[tuple[str, int, bool]]:
    # Two shards with different success rates -- a flat split could easily
    # starve one shard's failures from one half by chance; stratification must not.
    rows = []
    for i in range(20):
        rows.append(("clean", i, i < 14))  # 70% success
    for i in range(20):
        rows.append(("lighting", 100 + i, i < 10))  # 50% success
    return rows


def test_split_is_disjoint_and_covers_every_episode() -> None:
    split = stratified_split(_episodes(), eval_fraction=0.3, seed=0)
    assert not (split.calibration & split.eval)
    assert len(split.calibration) + len(split.eval) == 40


def test_split_preserves_per_stratum_proportions() -> None:
    split = stratified_split(_episodes(), eval_fraction=0.3, seed=0)

    def count(keys: frozenset[EpisodeKey], shard: str) -> int:
        return sum(1 for k in keys if k.shard == shard)

    # 14 clean successes * 0.3 rounds to 4; 6 clean failures * 0.3 rounds to 2.
    assert count(split.eval, "clean") == pytest.approx(6, abs=1)
    assert count(split.eval, "lighting") == pytest.approx(6, abs=1)


def test_split_is_reproducible_under_the_same_seed() -> None:
    a = stratified_split(_episodes(), eval_fraction=0.3, seed=42)
    b = stratified_split(_episodes(), eval_fraction=0.3, seed=42)
    assert a == b


def test_different_seeds_can_differ() -> None:
    a = stratified_split(_episodes(), eval_fraction=0.3, seed=1)
    b = stratified_split(_episodes(), eval_fraction=0.3, seed=2)
    assert a != b


def test_singleton_stratum_lands_whole_in_calibration() -> None:
    # A stratum of size 1 must not vanish under rounding (round(1*0.3) == 0).
    split = stratified_split([("language_absent", 0, False)], eval_fraction=0.3, seed=0)
    assert EpisodeKey("language_absent", 0) in split.calibration
    assert len(split.eval) == 0


def test_rejects_out_of_range_fraction() -> None:
    with pytest.raises(ValueError, match="eval_fraction"):
        stratified_split(_episodes(), eval_fraction=1.5, seed=0)


def test_kfold_folds_are_disjoint_and_cover_every_episode() -> None:
    folds = stratified_kfold(_episodes(), n_folds=5, seed=0)
    assert len(folds) == 5
    covered: set[EpisodeKey] = set()
    for fold in folds:
        assert not (fold.calibration & fold.eval)
        covered |= fold.eval
    assert len(covered) == 40


def test_split_rejects_overlapping_construction() -> None:
    key = EpisodeKey("clean", 0)
    with pytest.raises(ValueError, match="both halves"):
        Split(calibration=frozenset({key}), eval=frozenset({key}))
