"""End-to-end rollout-loop contract, on CPU, with no simulator and no weights.

Every bug this file guards against is *silent*: the loop has a fallback for each
one (zeros, first-item, skipped step), so a broken run still writes a plausible
dataset and still passes lint and the schema tests. These assertions are the only
thing standing between a wiring regression and a wasted GPU run.

The fakes (`tests/fakes/rollout.py`) cover the policy and the vector env. The
environment preprocessor is the REAL `LiberoProcessorStep`, so the state layout
and the 180-degree image flip are checked against production behaviour rather
than against a mock of it.
"""

from __future__ import annotations

from typing import Any

import pytest

# torch and lerobot arrive with the `sim` extra, so the CPU-only CI job skips
# this module rather than failing to collect it.
pytest.importorskip("torch")
pytest.importorskip("lerobot")

import numpy as np  # noqa: E402

from backstop.record.step import UnnormalizeError, unnormalize_array  # noqa: E402
from backstop.schema import ACTION_DIM, CHUNK_SIZE, STATE_DIM  # noqa: E402
from fakes.rollout import (  # noqa: E402
    DONE_AT,
    EEF_POS,
    GRIPPER_QPOS,
    IMAGE_HW,
    POST_SCALE,
    SpyPostprocessor,
    run_fake_rollout,
)


@pytest.fixture
def rollout(tmp_path):
    return run_fake_rollout(tmp_path, k_samples=4)


def test_loop_records_one_frame_per_step(rollout) -> None:
    assert len(rollout.frames) == DONE_AT


def test_action_chunk_is_recorded_not_zeroed(rollout) -> None:
    """Guards the monkey-patch bypass: select_action never hits predict_action_chunk."""
    chunk = rollout.frames[0]["action_chunk"]
    assert chunk.shape == (CHUNK_SIZE, ACTION_DIM)
    assert np.abs(chunk).max() > 0.0, "action_chunk is all zeros -- the chunk was never captured"


def test_sample_zero_is_the_executed_chunk(rollout) -> None:
    frame = rollout.frames[0]
    np.testing.assert_allclose(frame["action_chunk_samples"][0], frame["action_chunk"])


def test_policy_postprocessor_is_applied_every_step(rollout) -> None:
    """Guards the missing unnormalize: LIBERO's env_post is empty, so post is the only one."""
    assert rollout.post.calls >= DONE_AT, (
        f"postprocessor called {rollout.post.calls}x for {DONE_AT} steps -- "
        "actions reach MuJoCo un-unnormalized"
    )
    noise = rollout.adapter.noise_for(episode_seed=0, step=0, index=0)
    raw_first_row = rollout.policy.chunk_for(noise)[0, 0].numpy()
    np.testing.assert_allclose(rollout.frames[0]["action"], raw_first_row * POST_SCALE, rtol=1e-5)


def test_recorded_chunk_is_in_the_same_units_as_the_executed_action(rollout) -> None:
    """Executed action must be row 0 of the stored chunk -- both post-unnormalize."""
    frame = rollout.frames[0]
    np.testing.assert_allclose(frame["action"], frame["action_chunk"][0], rtol=1e-5)


def test_action_policy_matches_action_with_no_perturbation_configured(rollout) -> None:
    """ADR-003: `record.perturbation` defaults to "clean", so `action_policy` (what
    the policy asked for) and `action` (what was executed) must be identical here --
    the divergence is `action_noise`-only, exercised in test_perturb.py, not here."""
    frame = rollout.frames[0]
    assert "action_policy" in frame
    np.testing.assert_allclose(frame["action_policy"], frame["action"], rtol=1e-5)
    np.testing.assert_allclose(frame["action_policy"], frame["action_chunk"][0], rtol=1e-5)


def test_action_noise_is_actually_invoked_from_the_loop(tmp_path) -> None:
    """The families' `perturb_action` is exercised against a bare rng in
    test_perturb.py; this proves the loop actually calls it end to end -- through
    `run_episodes`, not just unit-tested in isolation -- with a per-episode
    generator that advances step to step rather than redrawing the same noise."""
    result = run_fake_rollout(
        tmp_path,
        k_samples=1,
        record_overrides={"perturbation": "action_noise", "perturbation_strength": 1.0},
    )
    frames = result.frames
    assert len(frames) == DONE_AT

    frame = frames[0]
    assert not np.allclose(frame["action"], frame["action_policy"]), (
        "action_noise at strength 1.0 produced no divergence -- perturb_action was never reached"
    )
    np.testing.assert_allclose(frame["action_policy"], frame["action_chunk"][0], rtol=1e-5)
    # Gripper-untouched is exercised in test_perturb.py against a realistic
    # LIBERO-scale ([-1, 1]) action; this fixture's SpyPostprocessor multiplies
    # by POST_SCALE=10 to simulate unnormalize, so its fake gripper value lands
    # outside [-1, 1] and the family's own clip (mirroring what the real LIBERO
    # controller does to any out-of-range command) legitimately moves it. That
    # is a fixture-scale artifact, not something to assert against here.

    noise_0 = frames[0]["action"] - frames[0]["action_policy"]
    noise_1 = frames[1]["action"] - frames[1]["action_policy"]
    assert not np.allclose(noise_0, noise_1), (
        "same noise every step -- perturb_rng was rebuilt instead of advanced"
    )


def test_observation_state_carries_eef_pos_and_gripper(rollout) -> None:
    """Guards the raw_obs walk that silently grabs eef['mat'] instead."""
    state = rollout.frames[0]["observation.state"]
    assert state.shape == (STATE_DIM,)
    np.testing.assert_allclose(state[:3], EEF_POS[0], rtol=1e-5)
    np.testing.assert_allclose(state[6:8], GRIPPER_QPOS[0], rtol=1e-5)


def test_recorded_image_matches_what_the_policy_saw(rollout) -> None:
    """LiberoProcessorStep flips H and W; the corpus must store the flipped view."""
    recorded = rollout.frames[0]["observation.images.image"]
    assert recorded.shape == (IMAGE_HW, IMAGE_HW, 3)
    np.testing.assert_array_equal(recorded, rollout.image[::-1, ::-1, :])


def test_executed_trajectory_is_independent_of_k(tmp_path) -> None:
    """The whole point of seeded noise: K is an observer, not a participant.

    Without this, a K=4 week-2 corpus and a K=1 week-1 baseline diverge at the
    same seed, and any success-rate gap is unattributable.
    """
    k1 = run_fake_rollout(tmp_path / "k1", k_samples=1)
    k4 = run_fake_rollout(tmp_path / "k4", k_samples=4)
    assert len(k1.frames) == len(k4.frames)
    for step, (a, b) in enumerate(zip(k1.frames, k4.frames, strict=True)):
        np.testing.assert_allclose(
            a["action"], b["action"], rtol=1e-6, err_msg=f"trajectory diverged at step {step}"
        )
        np.testing.assert_allclose(a["action_chunk"], b["action_chunk"], rtol=1e-6)


def test_k_samples_are_distinct_from_each_other(rollout) -> None:
    """K identical chunks would make chunk-disagreement identically zero."""
    samples = rollout.frames[0]["action_chunk_samples"]
    assert samples.shape[0] == 4
    for index in range(1, 4):
        assert not np.allclose(samples[index], samples[0]), f"sample {index} equals the executed"


def test_resume_runs_exactly_the_tail_the_shard_is_missing(tmp_path) -> None:
    """A crash at episode 3 of 5 must resume as 3,4 -- same seeds, same init states.

    `seed = start_seed + episode_index`, so a resume that renumbered episodes would
    re-run the tail under different seeds and quietly break matched pairing against
    the clean arm.
    """
    full = run_fake_rollout(tmp_path / "full", k_samples=1, n_episodes=5)
    tail = run_fake_rollout(tmp_path / "tail", k_samples=1, n_episodes=5, skip_episodes=3)

    assert [row["episode_index"] for row in full.stats["episodes"]] == [0, 1, 2, 3, 4]
    assert [row["episode_index"] for row in tail.stats["episodes"]] == [3, 4]
    assert [row["seed"] for row in tail.stats["episodes"]] == [3, 4]
    assert tail.stats["n_skipped"] == 3
    assert tail.stats["partial"] is True

    for resumed, original in zip(tail.stats["episodes"], full.stats["episodes"][3:], strict=True):
        assert resumed == original, "resumed episode differs from the one it replaces"


def test_an_unresumed_run_is_not_marked_partial(rollout) -> None:
    """`partial` gates the tolerance check; a false positive would silence it."""
    assert rollout.stats["partial"] is False
    assert rollout.stats["n_skipped"] == 0


def test_timing_counts_match_what_was_actually_recorded(rollout) -> None:
    """A step counter that drifts from the frame count silently rescales every
    projection built on s/step -- which on day 1 of week 2 is the corpus size."""
    timing = rollout.stats["timing"]
    assert timing["n_steps"] == len(rollout.frames)
    assert timing["n_episodes"] == 1
    assert timing["episode_steps"] == [len(rollout.frames)]
    assert timing["stages"]["policy_select"]["calls"] == len(rollout.frames)
    assert timing["stages"]["frame_write"]["calls"] == len(rollout.frames)


def test_a_non_libero_env_does_not_take_down_the_run_for_want_of_a_stopwatch(rollout) -> None:
    """`instrument_render` reaches into the robosuite sim, which this fake has not
    got. Timing is diagnostic: unlike privileged capture it must degrade, not raise."""
    assert "env_render" not in rollout.stats["timing"]["stages"]
    assert rollout.stats["timing"]["derived"] == {}
    assert len(rollout.frames) == DONE_AT


def test_sampling_leaves_no_trace_on_the_executed_chunk(rollout) -> None:
    """`sample_chunks` runs through the same wrapped hook that captures the chunk."""
    for step, frame in enumerate(rollout.frames):
        np.testing.assert_allclose(
            frame["action"],
            frame["action_chunk"][0],
            rtol=1e-5,
            err_msg=f"stored chunk is not the executed one at step {step}",
        )


def test_a_failing_unnormalize_stops_the_run_instead_of_storing_policy_space() -> None:
    """It used to log a warning and return the chunk unconverted, which quietly
    made the corpus unusable for chunk-based signals."""

    def broken_post(action: Any) -> Any:
        raise RuntimeError("stats missing for 'action'")

    chunk = np.zeros((4, CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    with pytest.raises(UnnormalizeError, match="policy space") as raised:
        unnormalize_array(broken_post, chunk)
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_unnormalize_applies_the_postprocessor_to_every_sample() -> None:
    chunk = np.ones((4, CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    out = unnormalize_array(SpyPostprocessor(), chunk)
    assert out.shape == chunk.shape
    np.testing.assert_allclose(out, chunk * POST_SCALE)
