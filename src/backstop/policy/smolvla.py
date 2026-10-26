"""SmolVLA through LeRobot: loading it, and the adapter the rollout loop calls.

Every use of LeRobot's policy internals (`_get_action_chunk`, `_queues`) lives in
this module, so a LeRobot upgrade breaks here and nowhere else.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backstop.config import PolicyConfig
from backstop.policy.base import Processors, UnsupportedPolicyError

logger = logging.getLogger(__name__)


class BatchedSamplingError(RuntimeError):
    """Batched chunk sampling returned something structurally wrong.

    Distinct from an OOM on purpose: OOM falls back to sequential sampling, and a
    broken batch must not take that exit and record silently-degraded chunks.
    """


class SmolVLAAdapter:
    """Wraps a LeRobot SmolVLA policy. Lazy-imports torch/lerobot."""

    def __init__(
        self, policy: Any, *, n_action_steps: int, num_steps: int, batch_samples: bool = True
    ) -> None:
        self._policy = policy
        self.batch_samples = batch_samples
        self._oom_warned = False
        cfg = policy.config
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        if num_steps is not None and hasattr(cfg, "num_steps"):
            cfg.num_steps = num_steps
        self.chunk_size = int(getattr(cfg, "chunk_size", 50))
        action_ft = cfg.output_features.get("action")
        self.action_dim = int(action_ft.shape[0]) if action_ft is not None else 7
        self.max_action_dim = int(getattr(cfg, "max_action_dim", self.action_dim))
        self._device = str(getattr(cfg, "device", None) or "cpu")
        self._last_chunk: Any | None = None
        self._wrap_predict()

    def noise_for(self, *, episode_seed: int, step: int, index: int, batch_size: int = 1) -> Any:
        """Deterministic flow noise for sample `index` at `step`.

        Drawn from a dedicated generator rather than the global RNG so that the
        executed chunk (index 0) is identical whatever K is. Without this, a K=4
        corpus and a K=1 baseline diverge at the same seed and a success-rate
        shift cannot be attributed to the perturbation rather than the sampling.
        """
        import torch  # noqa: PLC0415

        seed = (episode_seed * 1_000_003 + step * 1_009 + index) % (2**31 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (batch_size, self.chunk_size, self.max_action_dim)
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        return noise.to(self._device)

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

    def select_action(self, observation: dict[str, Any], *, noise: Any = None) -> Any:
        """Raw policy tensor. Caller must run the LeRobot postprocessor before the env."""
        return self._policy.select_action(observation, noise=noise)

    def last_action_chunk(self) -> Any:
        np = _np()
        if self._last_chunk is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        return self._last_chunk

    def sample_chunks(
        self,
        observation: dict[str, Any],
        k: int,
        *,
        episode_seed: int | None = None,
        step: int | None = None,
    ) -> Any:
        """K flow-matching chunks. Index 0 is the executed chunk.

        Restores `_last_chunk` and the policy queues afterwards: the extra
        predictions run through the same wrapped `_get_action_chunk` that
        captures the executed chunk, so sampling must not leave a trace.

        The K-1 extras go through `select_action`'s *sibling*, never through
        `select_action` itself, so nothing here can move the executed trajectory.
        """
        np = _np()
        executed = self.last_action_chunk()
        predict = getattr(self, "_original_predict", None)
        if predict is None or k <= 1:
            return np.stack([executed] * k, axis=0).astype(np.float32)

        saved_queues = getattr(self._policy, "_queues", None)
        queues = _copy_queues(saved_queues)
        if queues is not None:
            self._policy._queues = queues
        try:
            extras = self._extra_samples(observation, k, episode_seed, step, predict)
        finally:
            if saved_queues is not None:
                self._policy._queues = saved_queues
            # The wrapped `_get_action_chunk` fired again for the extras, so
            # `_last_chunk` now holds a sample. Put the executed chunk back.
            self._last_chunk = executed
        return np.stack([executed, *extras], axis=0).astype(np.float32)

    def _extra_samples(
        self,
        observation: dict[str, Any],
        k: int,
        episode_seed: int | None,
        step: int | None,
        predict: Any,
    ) -> list[Any]:
        """The K-1 non-executed chunks, in one forward pass where possible.

        Unbatched, each extra costs a full flow-matching integration: measured at
        328 ms against 337 ms for the executed chunk, which put a K=4 corpus at
        32.7 h against a ~20 h budget (docs/throughput.md). Batching them shares
        the VLM prefix pass and the ten denoise steps across the batch dimension.

        Falls back to the sequential loop on CUDA OOM. Batch 3 of a 500M VLM with
        two images is not obviously safe on an 8 GB card, and a shard that dies at
        hour four is a worse outcome than one that runs slowly.
        """
        if not self.batch_samples or episode_seed is None or step is None:
            # Without a seed there is no per-index noise to assemble, so the
            # batched path has nothing to build its batch out of.
            return self._extra_samples_sequential(observation, k, episode_seed, step, predict)
        try:
            return self._extra_samples_batched(observation, k, episode_seed, step, predict)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            if not self._oom_warned:
                self._oom_warned = True
                logger.warning(
                    "Batched chunk sampling ran out of memory at K=%s (%s). Falling back to "
                    "sequential sampling for the rest of this run, which is roughly %sx slower "
                    "per step. Set policy.batch_samples=false to skip the retry.",
                    k,
                    type(exc).__name__,
                    k - 1,
                )
            self.batch_samples = False
            _empty_cuda_cache()
            return self._extra_samples_sequential(observation, k, episode_seed, step, predict)

    def _extra_samples_batched(
        self,
        observation: dict[str, Any],
        k: int,
        episode_seed: int,
        step: int,
        predict: Any,
    ) -> list[Any]:
        import torch  # noqa: PLC0415

        n = k - 1
        # Row i-1 is exactly `noise_for(index=i)`, so a batched sample carries the
        # same noise its sequential counterpart would have. Sample 0 is absent by
        # construction: it belongs to select_action, which is not on this path.
        noise = torch.cat(
            [
                self.noise_for(episode_seed=episode_seed, step=step, index=index)
                for index in range(1, k)
            ],
            dim=0,
        )
        chunks = _as_numpy(predict(_tile_batch(observation, n), noise=noise))
        if chunks.ndim != 3 or chunks.shape[0] != n:
            raise BatchedSamplingError(
                f"Batched sampling asked for {n} chunks and got shape {chunks.shape}. "
                "The observation did not tile along the batch dimension -- this path "
                "assumes eval.batch_size=1, one environment per rollout."
            )
        return [_chunk_to_numpy(chunks[i], self.chunk_size, self.action_dim) for i in range(n)]

    def _extra_samples_sequential(
        self,
        observation: dict[str, Any],
        k: int,
        episode_seed: int | None,
        step: int | None,
        predict: Any,
    ) -> list[Any]:
        samples = []
        for index in range(1, k):
            if episode_seed is None or step is None:
                noise = None
            else:
                noise = self.noise_for(episode_seed=episode_seed, step=step, index=index)
            chunk = predict(observation, noise=noise)
            samples.append(_chunk_to_numpy(chunk, self.chunk_size, self.action_dim))
        return samples

    @property
    def policy(self) -> Any:
        return self._policy


def load_pinned_config(from_pretrained: Callable[..., Any], path: str, revision: str | None) -> Any:
    """Load a policy config at exactly `revision`, or at HEAD only when none is pinned.

    There is deliberately no fallback: if the loader rejects `revision`, the
    error propagates. Retrying without it would load whatever the Hub repo
    holds today while provenance still records the pin.
    """
    if revision:
        return from_pretrained(path, revision=revision)
    return from_pretrained(path)


def load_policy_runtime(cfg: PolicyConfig, env_cfg: Any) -> tuple[SmolVLAAdapter, Processors]:
    """The pinned checkpoint, wrapped for the loop, and its four processors."""
    if cfg.type != "smolvla":
        raise UnsupportedPolicyError(
            f"policy.type {cfg.type!r} is not implemented; only SmolVLA is "
            "(see docs/adr/002-checkpoint.md)."
        )
    from lerobot.configs import PreTrainedConfig  # noqa: PLC0415
    from lerobot.envs import make_env_pre_post_processors  # noqa: PLC0415
    from lerobot.policies import make_policy, make_pre_post_processors  # noqa: PLC0415

    policy_cfg = load_pinned_config(PreTrainedConfig.from_pretrained, cfg.path, cfg.revision)
    # Declared as `Path | None`; lerobot only truth-tests it and forwards it as an
    # HF `pretrained_name_or_path`, so a Hub repo id round-trips through Path fine.
    policy_cfg.pretrained_path = Path(cfg.path)
    policy_cfg.device = cfg.device
    # n_action_steps/num_steps live on the concrete policy config (SmolVLAConfig),
    # not the PreTrainedConfig base -- the hasattr guard is the runtime contract.
    if hasattr(policy_cfg, "n_action_steps"):
        policy_cfg.n_action_steps = cfg.n_action_steps  # type: ignore[attr-defined]
    if hasattr(policy_cfg, "num_steps"):
        policy_cfg.num_steps = cfg.num_steps  # type: ignore[attr-defined]
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor_overrides = {"device_processor": {"device": str(policy.config.device)}}
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    adapter = SmolVLAAdapter(
        policy,
        n_action_steps=cfg.n_action_steps,
        num_steps=cfg.num_steps,
        batch_samples=cfg.batch_samples,
    )
    return adapter, Processors(env_pre=env_pre, pre=pre, post=post, env_post=env_post)


def _np() -> Any:
    import numpy as np  # noqa: PLC0415

    return np


def _tile_batch(observation: dict[str, Any], n: int) -> dict[str, Any]:
    """Repeat a batch-of-one observation `n` times along the batch dimension.

    Tiles every tensor whose leading dimension is 1 and every length-1 list (the
    language strings), and passes anything else through untouched. The
    alternative -- enumerating the keys SmolVLA reads -- silently drops whatever
    a future preprocessor adds, and a missing key surfaces as a shape error deep
    inside the VLM rather than here.

    `repeat`, not `expand`: an expanded view shares storage and is not
    contiguous, and the prefix embedding downstream reshapes. The copy is a few
    MB of images against the activations of a 500M VLM.
    """
    import torch  # noqa: PLC0415

    tiled: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 1:
            tiled[key] = value.repeat(n, *([1] * (value.ndim - 1)))
        elif isinstance(value, list) and len(value) == 1:
            tiled[key] = value * n
        else:
            tiled[key] = value
    return tiled


def _is_oom(exc: BaseException) -> bool:
    """True only for out-of-memory, never for a contract violation.

    Matching on `RuntimeError` alone would make the OOM fallback swallow
    `BatchedSamplingError` -- a wrong-shaped batch would then quietly record
    sequentially-sampled chunks forever instead of failing.
    """
    try:
        import torch  # noqa: PLC0415

        torch_oom = getattr(torch, "OutOfMemoryError", None)
        if torch_oom is not None and isinstance(exc, torch_oom):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
