"""Policy loading never drops a pinned revision, and refuses what it cannot load.

`backstop.policy.smolvla` imports torch and lerobot only inside functions, so
these run in the CPU job.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from backstop.config import PolicyConfig
from backstop.policy.base import ChunkPolicy, UnsupportedPolicyError
from backstop.policy.smolvla import load_pinned_config, load_policy_runtime
from fakes.policy import FakeChunkPolicy

REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def test_the_pin_is_passed_through() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def loader(path: str, **kwargs: Any) -> str:
        calls.append((path, kwargs))
        return "cfg"

    assert load_pinned_config(loader, "org/policy", REVISION) == "cfg"
    assert calls == [("org/policy", {"revision": REVISION})]


def test_no_pin_loads_without_a_revision() -> None:
    calls: list[dict[str, Any]] = []
    load_pinned_config(lambda path, **kwargs: calls.append(kwargs), "org/policy", None)
    assert calls == [{}]


def test_a_loader_that_rejects_the_pin_is_an_error_not_a_silent_retry() -> None:
    calls: list[dict[str, Any]] = []

    def loader(path: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        if "revision" in kwargs:
            raise TypeError("unexpected keyword argument 'revision'")
        return "HEAD config"

    with pytest.raises(TypeError, match="revision"):
        load_pinned_config(loader, "org/policy", REVISION)
    assert calls == [{"revision": REVISION}]  # never retried at HEAD


def test_openvla_is_refused_before_anything_loads() -> None:
    with pytest.raises(UnsupportedPolicyError, match="openvla"):
        load_policy_runtime(PolicyConfig(type="openvla", path="org/openvla"), env_cfg=None)


def test_the_fake_policy_keeps_the_executed_chunk_independent_of_k() -> None:
    """The property `noise_for` exists for, held by the fake the loop tests use."""
    policy: ChunkPolicy = FakeChunkPolicy()
    noise = policy.noise_for(episode_seed=3, step=5, index=0)
    action = policy.select_action({}, noise=noise)
    k1 = policy.sample_chunks({}, 1, episode_seed=3, step=5)
    k4 = policy.sample_chunks({}, 4, episode_seed=3, step=5)
    np.testing.assert_array_equal(k1[0], k4[0])
    np.testing.assert_array_equal(action[0], k4[0, 0])
    assert len({chunk.tobytes() for chunk in k4}) == 4
