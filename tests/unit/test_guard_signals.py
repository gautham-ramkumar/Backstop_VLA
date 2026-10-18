"""Contract for the action-chunk disagreement signal (no GPU, no corpus)."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.guard.signals import action_chunk_disagreement, episode_score


def test_identical_samples_have_zero_disagreement() -> None:
    chunk = np.ones((3, 4, 50, 7), dtype=np.float32)
    scores = action_chunk_disagreement(chunk)
    assert scores.shape == (3,)
    assert np.allclose(scores, 0.0)


def test_disagreement_scales_with_deviation() -> None:
    t, k, chunk_size, action_dim = 2, 4, 50, 7
    samples = np.zeros((t, k, chunk_size, action_dim), dtype=np.float32)
    samples[:, 1:] = 0.1  # every extra sample differs from the executed one (index 0) by 0.1
    scores = action_chunk_disagreement(samples)
    assert np.allclose(scores, 0.1, atol=1e-6)


def test_disagreement_uses_index_zero_as_reference_not_pairwise() -> None:
    # Two extra samples move in *opposite* directions around the executed one;
    # pairwise disagreement among all four would be nonzero even on average,
    # but deviation-from-executed here should be symmetric and equal.
    samples = np.zeros((1, 3, 2, 1), dtype=np.float32)
    samples[0, 1] = 1.0
    samples[0, 2] = -1.0
    scores = action_chunk_disagreement(samples)
    assert scores[0] == pytest.approx(1.0)


def test_k_less_than_two_returns_zeros() -> None:
    samples = np.random.default_rng(0).normal(size=(5, 1, 50, 7)).astype(np.float32)
    scores = action_chunk_disagreement(samples)
    assert scores.shape == (5,)
    assert np.array_equal(scores, np.zeros(5))


def test_wrong_ndim_raises() -> None:
    with pytest.raises(ValueError, match="expected"):
        action_chunk_disagreement(np.zeros((5, 7)))


def test_episode_score_is_the_max() -> None:
    assert episode_score(np.array([0.1, 0.9, 0.3])) == pytest.approx(0.9)


def test_episode_score_empty_is_zero() -> None:
    assert episode_score(np.array([])) == 0.0


def test_instructed_object_phrase_covers_corpus_wordings() -> None:
    from backstop.guard.signals import instructed_object_phrase

    bowl = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    paraphrase = "Grab the black bowl positioned between the plate and the ramekin, then place it"
    absent = "pick up the moka pot and place it on the plate"
    assert instructed_object_phrase(bowl) == "black bowl"
    assert instructed_object_phrase(paraphrase) == "black bowl"
    assert instructed_object_phrase(absent) == "moka pot"


def test_instructed_object_phrase_covers_libero_object() -> None:
    from backstop.guard.signals import instructed_object_phrase

    assert (
        instructed_object_phrase("pick up the alphabet soup and place it in the basket")
        == "alphabet soup"
    )
    assert instructed_object_phrase("pick up the bbq sauce and place it in the basket") == (
        "bbq sauce"
    )
    assert instructed_object_phrase("open the drawer") == "open the drawer"
