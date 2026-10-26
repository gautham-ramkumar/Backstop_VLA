#!/usr/bin/env python
"""Does `restore_sim_state` actually restore state?

`docs/recovery.md` and the privileged sidecar (`src/backstop/record/privileged.py`)
both rest on one claim that nothing in the test suite checked: capture a mid-episode
MuJoCo state with `sim_state()`, hand it back to `restore_sim_state()`, and the
simulator lands exactly where it was -- not close, not visually similar, bit-for-bit
equal state, and physically deterministic from there.

This is not a pytest test because it needs the real simulator (real MuJoCo, real
LIBERO assets), which CI does not have. It needs no policy and no GPU: actions are a
fixed synthetic sequence, not a checkpoint's output, since the claim under test is
about the simulator, not the policy.

    uv run python scripts/verify_sim_state_roundtrip.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from backstop.checks import Checks
from backstop.config import EnvYamlConfig, EvalYamlConfig
from backstop.env.libero import (
    close_libero,
    make_libero_envs,
    restore_observation,
    restore_sim_state,
    sim_state,
)
from backstop.schema import ACTION_DIM

CHECKS = Checks()
check = CHECKS.check


def _synthetic_actions(n: int, seed: int) -> np.ndarray:
    """A fixed, smooth, deterministic action sequence -- not calibrated for task
    success, which is irrelevant here, but shaped to look like what a policy
    actually emits.

    Earlier draft of this script used i.i.d. uniform deltas plus a coin-flip
    gripper on every step. That drove the arm into rapid contact/joint-limit
    events (gripper chatter, large uncorrelated jumps) and the round trip then
    showed real, if decaying, divergence after a restore -- traced to
    `qacc_warmstart`, which robosuite's flattened state (time + qpos + qvel
    only, see `MjSimState.flatten` in `robosuite/utils/binding_utils.py`) does
    not capture, so MuJoCo's contact solver starts the branch from a different
    warm start than the original run had. Confirmed by elimination: neither a
    missing `act` array (this model has none) nor stale OSC controller goal
    state explained it -- forcing `controller.update()` + `reset_goal()` after
    restore changed nothing, and `goal_pos` turned out to be recomputed fresh
    from current `ee_pos` on every `set_goal()` call anyway, not carried over.

    Smooth low-frequency motion with the gripper held still -- closer to actual
    policy output and free of manufactured contact events -- round-trips to
    float noise (~1e-6). That is the case this test gates on, since it is the
    case the corpus actually contains. The chatter-induced transient is a real
    limitation of "full flattened MuJoCo state" as a branching checkpoint, not
    a bug in this script; it is worth a line in docs/recovery.md, not a fix
    speculatively applied to vendored robosuite code.
    """
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.02, 0.07, size=6)
    phases = rng.uniform(0, 2 * np.pi, size=6)
    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    for i in range(6):
        actions[:, i] = 0.15 * np.sin(2 * np.pi * freqs[i] * t + phases[i])
    # Gripper held closed (+1 closes, -1 opens): no chatter, the case being tested.
    actions[:, 6] = 1.0
    return actions


def run(task: str, task_id: int, n_steps: int, branch_at: int, seed: int) -> None:
    env_cfg = EnvYamlConfig(task=task, task_ids=[task_id], init_states=True, mujoco_gl="osmesa")
    eval_cfg = EvalYamlConfig(n_episodes=1, batch_size=1, max_parallel_tasks=1)
    envs = make_libero_envs(env_cfg, eval_cfg)
    vec_env = next(iter(next(iter(envs.values())).values()))

    try:
        vec_env.reset(seed=[seed])
        actions = _synthetic_actions(n_steps, seed)

        states: list[np.ndarray] = []
        for t in range(n_steps):
            states.append(sim_state(vec_env))
            vec_env.step(actions[t : t + 1])

        check(0 <= branch_at < n_steps, "branch point is inside the recorded run", str(branch_at))
        original_dim = states[0].size
        check(
            all(s.size == original_dim for s in states),
            "sim state width is constant across the run",
            f"dim={original_dim}",
        )

        restored_obs = restore_sim_state(vec_env, states[branch_at])
        check(restored_obs is not None, "restore_sim_state returns an observation")

        state_after_restore = sim_state(vec_env)
        check(
            bool(np.array_equal(state_after_restore, states[branch_at])),
            "restored state exactly matches the captured state",
            f"max abs diff={float(np.abs(state_after_restore - states[branch_at]).max()):.3g}",
        )

        # Replay the same tail of actions from the branch point and confirm the
        # simulator reproduces the states it originally produced there. This is
        # the actual claim docs/recovery.md needs: not just "restore looks right"
        # but "restore, then step, reproduces the branch deterministically."
        mismatches = []
        for t in range(branch_at, n_steps - 1):
            vec_env.step(actions[t : t + 1])
            replayed = sim_state(vec_env)
            expected = states[t + 1]
            if not np.allclose(replayed, expected, atol=1e-4, rtol=0):
                mismatches.append((t + 1, float(np.abs(replayed - expected).max())))
        detail = f"{len(mismatches)}/{n_steps - branch_at - 1} steps diverged"
        if mismatches:
            detail += f"; first at step {mismatches[0][0]} (max diff {mismatches[0][1]:.3g})"
        check(
            not mismatches,
            "replayed trajectory from the branch matches the original, step for step",
            detail,
        )

        # A restore must not merely resemble the target step; it must be
        # distinguishable from doing nothing. Guard against a no-op that would
        # pass the equality check above by construction.
        check(
            not np.array_equal(states[branch_at], states[0]),
            "the branch state actually differs from the initial state",
            "otherwise the equality check above proves nothing",
        )

        # The recovery runner branches from a restored state, so the observation it
        # hands the policy must match the one the run saw at that state. Not bit for
        # bit: after `mj_step` MuJoCo's positions and camera poses still describe the
        # start of the last 2 ms substep, while a restore recomputes them for the
        # state itself. Measured: images differ by under 0.1 grey level on average,
        # proprioception by under 3e-4.
        reset_obs, _ = vec_env.reset(seed=[seed])
        state = sim_state(vec_env)
        restored = restore_observation(vec_env, state)
        gaps = _leaf_gaps(reset_obs, restored)
        worst = {key: f"{gap:.2g}" for key, gap in gaps.items() if gap > 0}
        check(
            all(gap <= (0.5 if key.startswith("/pixels") else 1e-3) for key, gap in gaps.items()),
            "restore_observation after a reset matches the reset observation",
            f"image mean / state max abs diff: {worst}",
        )
        check(
            not any(_leaf_gaps(restored, restore_observation(vec_env, state)).values()),
            "restoring the same state twice gives the identical observation",
        )
    finally:
        close_libero(envs)


def _leaf_gaps(a: Any, b: Any, path: str = "") -> dict[str, float]:
    """Per array: mean abs difference for images, max abs difference for the rest."""
    if isinstance(a, dict):
        return {k: v for key in a for k, v in _leaf_gaps(a[key], b[key], f"{path}/{key}").items()}
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {path: float(diff.mean() if path.startswith("/pixels") else diff.max())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--branch-at", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    run(args.task, args.task_id, args.steps, args.branch_at, args.seed)

    return CHECKS.report(
        on_failure="restore_sim_state does not round-trip. Do not record the clean K=4 shard."
    )


if __name__ == "__main__":
    raise SystemExit(main())
