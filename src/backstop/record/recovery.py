"""A guarded run that recovers: the policy acts, the guard watches, and when it
fires the arm lifts and backs off and the policy tries again.

The same pieces as the recording loop (`record.obs`, `record.step`), with
`recover.controller` deciding at each step between the policy, the recovery
motion and a halt. Nothing is written to a dataset: the result is the outcome,
the attempts, and optionally the agentview frames for checking by eye.

**Branching.** The guard reads only the recorded K samples, so where it would
first fire on a recorded episode is known offline, and every step before that
is the recording. A `Branch` restores the episode's sim state at that step
(`env.libero.restore_observation`) and starts the recovery there, instead of
re-running the prefix. The observation after a restore differs from the
recorded one by under 0.1 grey level (`scripts/verify_sim_state_roundtrip.py`),
and the recovery motion moves the arm before the policy looks again.
Sim state is privileged: it sets up the experiment and is never guard input
(ADR-003).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backstop.env.libero import (
    max_episode_steps,
    restore_observation,
    sim_state,
    task_description,
)
from backstop.record.loop import EpisodeSpec, read_success
from backstop.record.obs import ObsPipeline
from backstop.record.step import Decider
from backstop.recover.controller import Attempt, RecoveryController, Verdict


@dataclass(frozen=True)
class Branch:
    """Start at `step`, from the recorded `sim_state` there, as the guard fires.

    `recover=False` starts there with the guard quiet instead: a plain re-run
    from that state. `start_sim_state` is the episode's step-0 state, read
    once for where the arm started (a return-home recovery's goal).
    """

    step: int
    sim_state: np.ndarray
    recover: bool = True
    start_sim_state: np.ndarray | None = None


@dataclass(frozen=True)
class RecoveryOutcome:
    success: bool
    n_steps: int
    attempts: list[Attempt]
    halted_at: int | None
    frames: list[np.ndarray] = field(default_factory=list)
    # Sim state as each recovery motion ends, for labelling where the objects
    # were when the policy tried again. Privileged: analysis only (ADR-003).
    recovered_sim_states: list[np.ndarray] = field(default_factory=list)


def run_guarded_episode(
    obs: ObsPipeline,
    decide: Decider,
    controller: RecoveryController,
    spec: EpisodeSpec,
    *,
    branch: Branch | None = None,
    keep_frames: bool = False,
    noise_seed: int | None = None,
    max_steps: int | None = None,
) -> RecoveryOutcome:
    """One episode under the guard, from its reset or from `branch`.

        Steps are counted from the episode's start either way, so the step cap,
        the flow noise and the stopwatch see the same step numbers as the recording.
        `noise_seed` (default: the episode's seed) draws different flow noise from
        the same start, a fresh try. `max_steps` replaces the suite's step cap (280
    for LIBERO-Spatial), shorter for a quick test or longer so a recovery's
    re-attempt gets a full run's time; the guard's clock and halt still end it.
    """
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: PLC0415

    vec_env, seed = spec.vec_env, spec.seed
    decide.policy.reset()
    raw_obs, _info = vec_env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    task = task_description(vec_env, fallback=f"{spec.suite_name}/task_{spec.task_id}")
    max_steps = max_episode_steps(vec_env) if max_steps is None else max_steps
    noise_seed = seed if noise_seed is None else noise_seed
    step = 0
    if branch is not None and branch.start_sim_state is not None:
        start = obs.prepare(restore_observation(vec_env, branch.start_sim_state), task)
        controller.home_state = start.frame_obs["state"]
    if branch is not None:
        raw_obs, step = restore_observation(vec_env, branch.sim_state), branch.step

    success, done, frames, recovered = False, False, [], []
    while not done and step < max_steps:
        prepared = obs.prepare(raw_obs, task)
        state = prepared.frame_obs["state"]
        if keep_frames:
            frames.append(prepared.frame_obs["image"])
        if controller.home_state is None:
            controller.home_state = state  # the reset pose, when run from reset
        if branch is not None and branch.recover and step == branch.step:
            controller.fired(step, state, "offline")
        action = _recovery_action(controller, decide, state, step)
        if controller.attempts and controller.attempts[-1].end == step:
            recovered.append(np.asarray(sim_state(vec_env), dtype=np.float64))
        if action is None:
            decision = decide(
                obs.normalize(prepared), episode_seed=noise_seed, step=step, perturb=lambda a: a
            )
            verdict = controller.after_policy(decision.samples, step, state)
            if verdict is Verdict.HALT:
                break
            action = decision.action
            if verdict is Verdict.RECOVER:
                action = _recovery_action(controller, decide, state, step)
                assert action is not None, "a recovery motion must take at least one step"
        raw_obs, _reward, terminated, truncated, info = vec_env.step(action)
        success = read_success(info, success)
        done = bool(terminated[0] or truncated[0])
        step += 1
    return RecoveryOutcome(
        success, step, controller.attempts, controller.halted_at, frames, recovered
    )


def _recovery_action(
    controller: RecoveryController, decide: Decider, state: np.ndarray, step: int
) -> np.ndarray | None:
    """The motion's action, batched; None when not recovering or the motion just ended."""
    if not controller.recovering:
        return None
    action = controller.motion_action(state, step)
    if action is None:
        # The policy starts a fresh attempt from what it sees now.
        decide.policy.reset()
        return None
    return np.asarray(action, dtype=np.float32)[None]
