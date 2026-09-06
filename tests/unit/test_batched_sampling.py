"""Batched chunk sampling must be an optimisation, not a change of behaviour.

Unbatched, each extra chunk costs a full flow-matching integration and a K=4
corpus projects to 32.7 h against a ~20 h budget (docs/throughput.md). Batching
the K-1 extras into one forward pass is the fix, and it is the kind of fix that
can silently corrupt a corpus: sample the wrong noise, tile the observation
wrongly, or let the extras perturb the executed action, and the dataset still
looks complete and still passes the schema gate.

The fake policy is deliberately batch-aware and noise-sensitive, so a batched
call that dropped rows, reordered them, or reused one noise vector across the
batch produces visibly wrong numbers rather than plausible ones.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from backstop.policy.adapter import BatchedSamplingError, SmolVLAAdapter  # noqa: E402
from backstop.schema import ACTION_DIM, CHUNK_SIZE  # noqa: E402

K = 4
SEED, STEP = 7, 3


class BatchAwarePolicy:
    """Maps (noise, state) -> chunk, row by row, for any batch size.

    Every output row depends on that row's noise *and* on that row's observation,
    so tiling the observation incorrectly is as visible as mis-ordering the noise.
    """

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=32,
            n_action_steps=1,
            num_steps=10,
            output_features={"action": SimpleNamespace(shape=(ACTION_DIM,))},
        )
        self.batch_sizes: list[int] = []
        self.calls = 0

    def reset(self) -> None:
        return None

    def _get_action_chunk(self, batch: Any, noise: Any = None) -> Any:
        self.calls += 1
        state = batch["observation.state"]
        if noise is None:
            # What real SmolVLA does: draw its own from the global RNG, sized
            # from the observation batch (`sample_actions` reads state.shape[0]).
            noise = torch.randn(state.shape[0], CHUNK_SIZE, self.config.max_action_dim)
        bsize = noise.shape[0]
        self.batch_sizes.append(bsize)
        if state.shape[0] != bsize:
            raise AssertionError(
                f"observation batch {state.shape[0]} does not match noise batch {bsize}"
            )
        # Row i mixes its own noise with its own observation row.
        return noise[:, :, :ACTION_DIM] + state[:, None, :ACTION_DIM]

    def predict_action_chunk(self, batch: Any, noise: Any = None) -> Any:
        return self._get_action_chunk(batch, noise)

    def select_action(self, batch: Any, noise: Any = None) -> Any:
        return self._get_action_chunk(batch, noise)[:, 0]


def _observation() -> dict[str, Any]:
    return {
        "observation.state": torch.arange(ACTION_DIM, dtype=torch.float32).reshape(1, ACTION_DIM),
        "observation.images.image": torch.zeros(1, 3, 8, 8),
        "task": ["pick up the black bowl"],
    }


def _adapter(*, batch_samples: bool) -> tuple[SmolVLAAdapter, BatchAwarePolicy]:
    policy = BatchAwarePolicy()
    adapter = SmolVLAAdapter(policy, n_action_steps=1, num_steps=10, batch_samples=batch_samples)
    return adapter, policy


def _sample(adapter: SmolVLAAdapter, observation: dict[str, Any]) -> Any:
    adapter.select_action(
        observation, noise=adapter.noise_for(episode_seed=SEED, step=STEP, index=0)
    )
    return adapter.sample_chunks(observation, K, episode_seed=SEED, step=STEP)


def test_batched_matches_sequential_exactly() -> None:
    """The headline contract. Batching is a scheduling change, not a numerical one.

    On real hardware the two differ by float noise from a different reduction
    order; with an exact fake they must agree bit for bit, which is what isolates
    a wiring bug from a kernel difference.
    """
    observation = _observation()
    batched, _ = _adapter(batch_samples=True)
    sequential, _ = _adapter(batch_samples=False)

    np.testing.assert_array_equal(_sample(batched, observation), _sample(sequential, observation))


def test_the_extras_go_out_in_one_forward_pass() -> None:
    """The point of the change: K-1 integrations become one batch of K-1."""
    batched, batched_policy = _adapter(batch_samples=True)
    sequential, sequential_policy = _adapter(batch_samples=False)
    _sample(batched, _observation())
    _sample(sequential, _observation())

    # One call for select_action, one batched call of width K-1 for the extras.
    assert batched_policy.batch_sizes == [1, K - 1]
    assert sequential_policy.batch_sizes == [1] + [1] * (K - 1)
    assert batched_policy.calls < sequential_policy.calls


def test_sample_zero_is_still_the_executed_chunk() -> None:
    """Index 0 must stay the chunk the robot acted on, or every disagreement
    metric is computed against a chunk that was never executed."""
    adapter, _ = _adapter(batch_samples=True)
    observation = _observation()
    adapter.select_action(
        observation, noise=adapter.noise_for(episode_seed=SEED, step=STEP, index=0)
    )
    executed = adapter.last_action_chunk().copy()
    samples = adapter.sample_chunks(observation, K, episode_seed=SEED, step=STEP)

    np.testing.assert_array_equal(samples[0], executed)


def test_each_sample_carries_its_own_index_noise() -> None:
    """A batch built from one noise vector broadcast K-1 times would make chunk
    disagreement identically zero -- the exact defect K=4 exists to avoid."""
    adapter, _ = _adapter(batch_samples=True)
    samples = _sample(adapter, _observation())

    for index in range(1, K):
        expected = adapter.noise_for(episode_seed=SEED, step=STEP, index=index)[0, :, :ACTION_DIM]
        state = _observation()["observation.state"][0, :ACTION_DIM]
        np.testing.assert_allclose(samples[index], (expected + state).numpy(), rtol=1e-6)
    for index in range(1, K):
        assert not np.allclose(samples[index], samples[0])


def test_sampling_does_not_disturb_the_executed_chunk() -> None:
    """`_last_chunk` is captured by a hook the extras also trip."""
    adapter, _ = _adapter(batch_samples=True)
    observation = _observation()
    adapter.select_action(
        observation, noise=adapter.noise_for(episode_seed=SEED, step=STEP, index=0)
    )
    before = adapter.last_action_chunk().copy()
    adapter.sample_chunks(observation, K, episode_seed=SEED, step=STEP)

    np.testing.assert_array_equal(adapter.last_action_chunk(), before)


def test_the_observation_is_not_mutated_by_tiling() -> None:
    """The caller reuses this dict for the next step; tiling in place would grow
    the batch dimension every step until something far away failed."""
    adapter, _ = _adapter(batch_samples=True)
    observation = _observation()
    _sample(adapter, observation)

    assert observation["observation.state"].shape == (1, ACTION_DIM)
    assert observation["observation.images.image"].shape == (1, 3, 8, 8)
    assert observation["task"] == ["pick up the black bowl"]


def test_oom_falls_back_to_sequential_and_keeps_going() -> None:
    """A shard that dies at hour four is worse than one that runs slowly."""
    adapter, policy = _adapter(batch_samples=True)
    real = policy._get_action_chunk

    def oom_on_batches(batch: Any, noise: Any = None) -> Any:
        if noise is not None and noise.shape[0] > 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return real(batch, noise)

    policy._get_action_chunk = oom_on_batches
    policy.predict_action_chunk = oom_on_batches
    adapter._original_predict = oom_on_batches

    samples = _sample(adapter, _observation())

    assert samples.shape == (K, CHUNK_SIZE, ACTION_DIM)
    assert adapter.batch_samples is False, "should stop retrying a batch that will OOM again"
    sequential, _ = _adapter(batch_samples=False)
    np.testing.assert_array_equal(samples, _sample(sequential, _observation()))


def test_a_wrong_shaped_batch_raises_instead_of_falling_back() -> None:
    """The OOM fallback must not become a catch-all.

    A batched call that returns the wrong number of chunks means the observation
    did not tile. Silently sampling sequentially forever would hide that, and the
    corpus would be fine -- just recorded at a third of the intended speed with
    nobody the wiser.
    """
    adapter, policy = _adapter(batch_samples=True)

    def drops_rows(batch: Any, noise: Any = None) -> Any:
        return policy._get_action_chunk(batch, noise)[:1]

    adapter._original_predict = drops_rows
    adapter.select_action(
        _observation(), noise=adapter.noise_for(episode_seed=SEED, step=STEP, index=0)
    )

    with pytest.raises(BatchedSamplingError, match="did not tile"):
        adapter.sample_chunks(_observation(), K, episode_seed=SEED, step=STEP)


def test_k_of_one_still_short_circuits() -> None:
    """K=1 must not build a zero-width batch."""
    adapter, policy = _adapter(batch_samples=True)
    observation = _observation()
    adapter.select_action(
        observation, noise=adapter.noise_for(episode_seed=SEED, step=STEP, index=0)
    )
    calls = policy.calls
    samples = adapter.sample_chunks(observation, 1, episode_seed=SEED, step=STEP)

    assert samples.shape == (1, CHUNK_SIZE, ACTION_DIM)
    assert policy.calls == calls, "K=1 must not run another forward pass"


def test_unseeded_sampling_uses_the_sequential_path() -> None:
    """Without (episode_seed, step) there is no per-index noise to batch."""
    adapter, policy = _adapter(batch_samples=True)
    adapter.select_action(_observation())
    samples = adapter.sample_chunks(_observation(), K)

    assert samples.shape == (K, CHUNK_SIZE, ACTION_DIM)
    assert policy.batch_sizes == [1] * K
