"""Per-stage wall-clock accounting for the record loop.

Week 2 sizes its corpus from measured seconds per episode, and that decision
needs a *breakdown* rather than one number. The two candidate answers point in
opposite directions: if the flow-matching forward pass dominates the step then
K=4 roughly triples it and the corpus has to shrink, but if OSMesa rendering of
two 360x360 cameras dominates then extra chunk samples are nearly free and the
real lever is image size or camera count. Guessing which one it is, and then
optimising, is how a day gets spent on the wrong stage.

Always on. `time.perf_counter()` costs tens of nanoseconds against a step that
costs hundreds of milliseconds, so there is no fast path worth having and no
flag that can be left in the wrong position -- every overnight shard reports
its own throughput for free. The timer reads no RNG and mutates no simulator
state, so a timed run records the same trajectory an untimed one would.

Two accuracy caveats, both reported rather than papered over:

- **CUDA is asynchronous.** With `sync=False` (the default, and what recording
  runs use) a policy stage stops the clock when the kernels are *queued*, not
  when they finish, which would charge that time to whichever later stage
  happens to block. Here that is mostly moot -- the adapter calls `.cpu()` on
  every chunk it captures, which blocks -- but `sync=True` makes it exact by
  draining the queue at each stage boundary. Benchmarks set it; the price is
  that genuine CPU/GPU overlap is serialised, so the stage sum runs slightly
  high against an unsynced wall clock.
- **Stages do not tile the step.** The glue between them is deliberately not
  instrumented. Whatever is left over is reported as `unaccounted`, so a
  breakdown that fails to explain the wall clock says so out loud instead of
  quietly attributing the gap to the last stage measured.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

# Stages that partition one control step, in execution order.
STEP_STAGES: tuple[str, ...] = (
    "obs_prep",
    "privileged",
    "normalize",
    "policy_select",
    "policy_samples",
    "action_post",
    "env_step",
    "frame_write",
)

# Measured *inside* another stage. Summing these with their parent would count
# the same microseconds twice, so they are excluded from the accounted total and
# reported as a subdivision instead.
NESTED_STAGES: dict[str, str] = {"env_render": "env_step"}

# Not per-step. Dividing these by the step count would make them look free.
OUTER_STAGES: tuple[str, ...] = ("env_reset", "save_episode", "finalize")


class StageTimer:
    """Accumulates wall-clock per named stage.

    Usage is one context manager per stage:

        with timer("env_step"):
            observation, *_ = vec_env.step(action)

    Re-entering a stage accumulates; the call count is kept so that a stage
    entered twice per step (two cameras, say) reports an honest per-call cost
    alongside its per-step cost.
    """

    def __init__(self, *, sync: bool = False) -> None:
        self._totals: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._n_steps = 0
        self._n_episodes = 0
        self._start = time.perf_counter()
        self._end: float | None = None
        # Per-episode wall time and length. An aggregate mean hides the two things
        # worth seeing: the first episode carrying CUDA and asset warm-up, and any
        # drift over a long shard.
        self._episode_wall: list[float] = []
        self._episode_steps: list[int] = []
        self._episode_mark = self._start
        self._steps_at_mark = 0
        # Resolved once. Asking torch whether CUDA exists on every stage exit
        # would cost more than the stage in the cheap cases.
        self._sync: Callable[[], None] | None = _resolve_sync(sync)
        self._sync_requested = sync

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            # `finally`, so a stage that raises still reports the time it burned.
            # A six-hour run that dies in hour five should still say where it went.
            if self._sync is not None:
                self._sync()
            self._totals[stage] = self._totals.get(stage, 0.0) + (time.perf_counter() - start)
            self._calls[stage] = self._calls.get(stage, 0) + 1

    def start(self) -> None:
        """(Re)start the wall clock at the first episode.

        A timer is constructed before the pipeline loads a checkpoint and builds
        the simulator, and that setup is tens of seconds. Left in the wall clock
        it is charged to no stage, so it lands in `unaccounted` and -- far worse --
        divides into `s_per_step`, inflating every corpus projection built on it.
        Setup is a fixed cost per shard, not a per-step one.
        """
        self._start = time.perf_counter()
        self._episode_mark = self._start

    def step_done(self) -> None:
        self._n_steps += 1

    def episode_done(self) -> None:
        self._n_episodes += 1
        now = time.perf_counter()
        self._episode_wall.append(now - self._episode_mark)
        self._episode_steps.append(self._n_steps - self._steps_at_mark)
        self._episode_mark = now
        self._steps_at_mark = self._n_steps

    def finish(self) -> None:
        """Stop the wall clock. Idempotent; the first call is the one that counts."""
        if self._end is None:
            self._end = time.perf_counter()

    @property
    def wall_s(self) -> float:
        return (self._end if self._end is not None else time.perf_counter()) - self._start

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    def summary(self) -> dict[str, Any]:
        """A JSON-safe breakdown. Reports what it could not account for."""
        wall = self.wall_s
        stages = {name: self._stage_entry(name, wall) for name in self._ordered_names()}

        derived: dict[str, dict[str, Any]] = {}
        if "env_step" in self._totals and "env_render" in self._totals:
            # Robosuite runs physics substeps and camera sampling inside one
            # `step`, so physics is only reachable by subtraction.
            physics = self._totals["env_step"] - self._totals["env_render"]
            derived["env_physics"] = {
                "total_s": round(physics, 4),
                "ms_per_step": _per(physics * 1000.0, self._n_steps),
                "pc_of_wall": _pc(physics, wall),
                "note": "env_step minus env_render; not measured directly",
            }

        accounted = sum(t for name, t in self._totals.items() if name not in NESTED_STAGES)
        return {
            "wall_s": round(wall, 3),
            "n_episodes": self._n_episodes,
            "n_steps": self._n_steps,
            "s_per_step": _per(wall, self._n_steps),
            "s_per_episode": _per(wall, self._n_episodes),
            "steps_per_episode": _per(float(self._n_steps), self._n_episodes),
            # False when sync was asked for but torch/CUDA could not provide it --
            # otherwise the report would claim an attribution accuracy it lacks.
            "cuda_sync": self._sync is not None,
            "cuda_sync_requested": self._sync_requested,
            "episode_wall_s": [round(value, 3) for value in self._episode_wall],
            "episode_steps": list(self._episode_steps),
            "stages": stages,
            "derived": derived,
            "accounted_s": round(accounted, 3),
            "unaccounted_s": round(wall - accounted, 3),
            "unaccounted_pc": _pc(wall - accounted, wall),
        }

    def _stage_entry(self, name: str, wall: float) -> dict[str, Any]:
        total, calls = self._totals[name], self._calls[name]
        entry: dict[str, Any] = {
            "total_s": round(total, 4),
            "calls": calls,
            "ms_per_call": _per(total * 1000.0, calls),
            "pc_of_wall": _pc(total, wall),
        }
        if name in NESTED_STAGES:
            entry["nested_in"] = NESTED_STAGES[name]
        if name in STEP_STAGES or name in NESTED_STAGES:
            entry["ms_per_step"] = _per(total * 1000.0, self._n_steps)
        else:
            entry["ms_per_episode"] = _per(total * 1000.0, self._n_episodes)
        return entry

    def _ordered_names(self) -> list[str]:
        """Execution order, then anything the caller invented, so the table reads."""
        known = [*STEP_STAGES, *NESTED_STAGES, *OUTER_STAGES]
        ordered = [name for name in known if name in self._totals]
        return ordered + sorted(set(self._totals) - set(known))


def _resolve_sync(enabled: bool) -> Callable[[], None] | None:
    if not enabled:
        return None
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.synchronize


def _per(total: float, count: int) -> float:
    return round(total / count, 4) if count else 0.0


def _pc(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0
