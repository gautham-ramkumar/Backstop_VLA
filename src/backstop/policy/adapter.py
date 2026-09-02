"""Policy adapters. Backstop consumes chunks, not model internals."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from backstop.config import PolicyConfig


class UnsupportedPolicyError(NotImplementedError):
    """Raised when a deferred policy adapter is constructed."""


@runtime_checkable
class PolicyAdapter(Protocol):
    chunk_size: int
    action_dim: int

    def reset(self) -> None: ...

    def select_action(self, observation: dict[str, Any]) -> Any: ...

    def last_action_chunk(self) -> Any: ...

    def sample_chunks(self, observation: dict[str, Any], k: int) -> Any: ...


class OpenVLAAdapter:
    """Deferred until the SmolVLA record/eval loop is proven (ADR-002)."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise UnsupportedPolicyError(
            "OpenVLA is out of week-1 scope. Prove the SmolVLA loop first "
            "(see docs/adr/002-checkpoint.md)."
        )


class SmolVLAAdapter:
    """Wraps a LeRobot SmolVLA policy. Lazy-imports torch/lerobot."""

    def __init__(self, policy: Any, *, n_action_steps: int, num_steps: int) -> None:
        self._policy = policy
        cfg = policy.config
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        if num_steps is not None and hasattr(cfg, "num_steps"):
            cfg.num_steps = num_steps
        self.chunk_size = int(getattr(cfg, "chunk_size", 50))
        action_ft = cfg.output_features.get("action")
        self.action_dim = int(action_ft.shape[0]) if action_ft is not None else 7
        self._last_chunk: Any | None = None
        self._wrap_predict()

    def _wrap_predict(self) -> None:
        # select_action fills the queue via `_get_action_chunk`, not `predict_action_chunk`.
        get_chunk = getattr(self._policy, "_get_action_chunk", None)
        if get_chunk is not None:

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                chunk = get_chunk(*args, **kwargs)
                self._last_chunk = _chunk_to_numpy(chunk, self.chunk_size, self.action_dim)
                return chunk

            self._policy._get_action_chunk = wrapped  # type: ignore[method-assign]
        if hasattr(self._policy, "predict_action_chunk"):
            self._original_predict = self._policy.predict_action_chunk
        elif get_chunk is not None:
            self._original_predict = get_chunk

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        self._last_chunk = None

    def select_action(self, observation: dict[str, Any]) -> Any:
        """Raw policy tensor. Caller must run the LeRobot postprocessor before the env."""
        return self._policy.select_action(observation)

    def last_action_chunk(self) -> Any:
        np = _np()
        if self._last_chunk is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        return self._last_chunk

    def sample_chunks(self, observation: dict[str, Any], k: int) -> Any:
        """K independent flow-matching chunks. Index 0 is the executed chunk."""
        np = _np()
        executed = self.last_action_chunk()
        samples = [executed]
        predict = getattr(self, "_original_predict", None)
        if predict is None or k <= 1:
            while len(samples) < k:
                samples.append(executed)
            return np.stack(samples[:k], axis=0).astype(np.float32)
        saved_queues = getattr(self._policy, "_queues", None)
        queues = _copy_queues(saved_queues)
        if queues is not None:
            self._policy._queues = queues
        try:
            for _ in range(k - 1):
                chunk = predict(observation)
                samples.append(_chunk_to_numpy(chunk, self.chunk_size, self.action_dim))
        finally:
            if saved_queues is not None:
                self._policy._queues = saved_queues
        return np.stack(samples[:k], axis=0).astype(np.float32)

    @property
    def policy(self) -> Any:
        return self._policy


def make_adapter(cfg: PolicyConfig, policy: Any | None = None) -> PolicyAdapter:
    if cfg.type == "openvla":
        raise UnsupportedPolicyError(
            "OpenVLA is out of week-1 scope. Prove the SmolVLA loop first "
            "(see docs/adr/002-checkpoint.md)."
        )
    if policy is None:
        raise ValueError("SmolVLAAdapter requires a loaded LeRobot policy")
    return SmolVLAAdapter(policy, n_action_steps=cfg.n_action_steps, num_steps=cfg.num_steps)


def _np() -> Any:
    import numpy as np  # noqa: PLC0415

    return np


def _copy_queues(queues: Any) -> Any:
    if queues is None:
        return None
    from collections import deque  # noqa: PLC0415

    return {key: deque(value) for key, value in queues.items()}


def _chunk_to_numpy(chunk: Any, chunk_size: int, action_dim: int) -> Any:
    np = _np()
    array = np.asarray(_as_numpy(chunk), dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != action_dim:
        padded = np.zeros((array.shape[0], action_dim), dtype=np.float32)
        n = min(action_dim, array.shape[1])
        padded[:, :n] = array[:, :n]
        array = padded
    if array.shape[0] < chunk_size:
        pad = np.repeat(array[-1:], chunk_size - array.shape[0], axis=0)
        array = np.concatenate([array, pad], axis=0)
    return array[:chunk_size].astype(np.float32)


def _as_numpy(value: Any) -> Any:
    np = _np()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
