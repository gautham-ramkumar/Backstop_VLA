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


def test_temporal_consistency_is_zero_when_the_plan_does_not_change() -> None:
    from backstop.guard.signals import temporal_self_consistency

    # Each next chunk is the previous chunk shifted by one step.
    base = np.arange(8, dtype=np.float32).reshape(1, 8, 1)
    chunks = np.stack([base[0, i : i + 4] for i in range(4)], axis=0)
    scores = temporal_self_consistency(chunks)
    assert scores.shape == (4,)
    assert np.allclose(scores, 0.0)


def test_temporal_consistency_measures_the_overlap_only() -> None:
    from backstop.guard.signals import temporal_self_consistency

    chunks = np.zeros((2, 3, 1), dtype=np.float32)
    chunks[1, 0, 0] = 0.0  # overlap with chunks[0, 1]
    chunks[0, 1, 0] = 0.3
    chunks[0, 2, 0] = 0.3
    chunks[1, 1, 0] = 0.3
    # overlap positions: (0,1) vs (1,0) differs by 0.3; (0,2) vs (1,1) agrees
    scores = temporal_self_consistency(chunks)
    assert scores[0] == pytest.approx(0.15)
    assert scores[1] == 0.0


def test_progress_is_zero_when_the_hand_follows_the_command() -> None:
    from backstop.guard.signals import progress_monitoring

    length = 8
    state = np.zeros((length, 8), dtype=np.float64)
    state[:, 0] = np.arange(length) * 0.01
    state[:, 6] = 0.04
    state[:, 7] = -0.04
    action = np.zeros((length, 7), dtype=np.float64)
    action[:, 0] = 0.01
    scores = progress_monitoring(state, action)
    assert np.allclose(scores, 0.0)


def test_progress_flags_a_command_with_no_motion() -> None:
    from backstop.guard.signals import progress_monitoring

    state = np.zeros((5, 8), dtype=np.float64)
    state[:, 6] = 0.04
    state[:, 7] = -0.04
    action = np.zeros((5, 7), dtype=np.float64)
    action[:, 0] = 1.0
    scores = progress_monitoring(state, action)
    assert scores[0] == pytest.approx(1.0)
    assert scores[-1] == 0.0


def test_progress_flags_a_close_with_no_motion() -> None:
    from backstop.guard.signals import progress_monitoring

    state = np.zeros((6, 8), dtype=np.float64)
    state[:, 6] = 0.04
    state[:, 7] = -0.04
    state[1:, 6] = 0.0  # width drops from 0.08 to 0.04 at step 0
    action = np.zeros((6, 7), dtype=np.float64)
    scores = progress_monitoring(state, action, open_width_m=0.08)
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == 0.0


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


def test_chunk_magnitude_is_the_mean_action_size() -> None:
    from backstop.guard.signals import action_chunk_magnitude

    chunks = np.zeros((2, 3, 2), dtype=np.float32)
    chunks[0, :, 0] = 3.0  # each step has L2 norm 3
    chunks[1, 0, 1] = 4.0  # one step of size 4, two of size 0
    scores = action_chunk_magnitude(chunks)
    assert scores[0] == pytest.approx(3.0)
    assert scores[1] == pytest.approx(4.0 / 3.0)


def test_slow_plan_change_scores_higher_than_the_same_gap_on_a_fast_plan() -> None:
    from backstop.guard.signals import velocity_normalized_consistency

    slow = np.zeros((2, 3, 1), dtype=np.float32)
    fast = np.zeros((2, 3, 1), dtype=np.float32)
    # Overlap differs by 0.2 in both. The fast plan is a large motion.
    slow[0, 1, 0] = 0.2
    fast[0, 1, 0] = 1.2
    fast[0, 2, 0] = 1.0
    fast[1, 0, 0] = 1.0
    fast[1, 1, 0] = 1.0
    slow_score = velocity_normalized_consistency(slow)[0]
    fast_score = velocity_normalized_consistency(fast)[0]
    assert slow_score > fast_score


def test_visual_precondition_ranks_absence_above_presence() -> None:
    from backstop.guard.signals import visual_precondition

    scores = visual_precondition(np.array([0.9, 0.1]))
    assert scores[1] > scores[0]
