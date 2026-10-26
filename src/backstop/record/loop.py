"""Rollout loop that records Backstop extras (action_chunk, K samples).

One episode is a short loop over three pieces with one job each:
`record.obs` (what the policy sees and the corpus stores), `record.step` (the
executed action and the K chunks) and `record.sink` (dataset and sidecar rows).
The recovery runner reuses the same pieces.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backstop.config import PipelineConfig
from backstop.env.libero import instrument_render, max_episode_steps, task_description
from backstop.perturb import Perturbation, attach, episode_rng, make_perturbation
from backstop.policy.base import ChunkPolicy, Processors
from backstop.record.obs import ObsPipeline
from backstop.record.privileged import PrivilegedWriter
from backstop.record.sink import EpisodeSink
from backstop.record.step import Decider
from backstop.record.timing import StageTimer
from backstop.schema import CONTROL_FPS, lerobot_features

logger = logging.getLogger(__name__)


def create_dataset(
    cfg: PipelineConfig, image_hw: tuple[int, int] = (360, 360), *, fps: int
) -> tuple[Any, int]:
    """Open the shard for writing. Returns `(dataset, n_already_recorded)`.

    `fps` must be the env control rate (LIBERO: 20 Hz), one row per control step.
    Not the 30 that was hardcoded here, and not `metadata["render_fps"]` (80) --
    a wrong value silently desyncs viewer playback and lead-time-in-seconds.

    `record.on_existing` decides what happens when the root is already populated.
    Always destroying it is how a stray re-run once cost a night of GPU.
    """
    _assert_privileged_decided(cfg)
    if fps != CONTROL_FPS:
        raise ValueError(f"the env steps at {fps} Hz; every corpus row assumes {CONTROL_FPS} Hz")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    root = Path(cfg.record.root)
    features = lerobot_features(cfg.record.k_samples, image_hw)
    mode = cfg.record.on_existing

    if _is_populated(root):
        if mode == "fail":
            raise FileExistsError(
                f"{root} already contains a dataset. Recording would destroy it.\n"
                "Choose deliberately in the config under `record.on_existing`:\n"
                "  resume    -- append to it, skipping episodes already recorded\n"
                "  overwrite -- delete it and start again\n"
                "  fail      -- (default) refuse, as now\n"
                "Or point `record.root` at a new shard."
            )
        if mode == "resume":
            dataset = LeRobotDataset.resume(repo_id=cfg.record.repo_id, root=str(root))
            _assert_resumable(dataset, features=features, fps=int(fps), root=root)
            n_done = int(dataset.meta.total_episodes)
            logger.warning(
                "RESUMING %s at episode %s; earlier episodes are not re-run", root, n_done
            )
            return dataset, n_done
        logger.warning("OVERWRITING %s -- %s episodes discarded", root, _episode_count(root))
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=cfg.record.repo_id,
        fps=int(fps),
        features=features,
        robot_type="panda",
        root=str(root),
        use_videos=True,
    )
    return dataset, 0


def _assert_privileged_decided(cfg: PipelineConfig) -> None:
    """A named shard must say, out loud, whether it captures privileged sim state.

    `record.privileged` defaults to false, so a shard config that simply
    forgets the key records a corpus with no sim state -- and every use of it
    (onset annotation, auto proto-labels, counterfactual branching) needs the
    GPU run repeated to get it back. Nothing else in the run would complain: the
    dataset is complete and passes verification, because sim state deliberately
    is not in it.

    Keyed on `shard` because that is what distinguishes a corpus shard from a
    smoke or debug run, and on `model_fields_set` so that writing
    `privileged: false` is an answer while omitting it is not.
    """
    if cfg.record.shard is None or "privileged" in cfg.record.model_fields_set:
        return
    raise ValueError(
        f"record.shard is {cfg.record.shard!r} but record.privileged is unset.\n"
        "A corpus shard has to decide: sim state cannot be added afterwards without "
        "re-running the episode on the GPU.\n"
        "  privileged: true   -- capture it (every week-2 corpus shard)\n"
        "  privileged: false  -- deliberately skip it"
    )


def _is_populated(root: Path) -> bool:
    """An empty or metadata-less directory is not a corpus worth protecting."""
    return (root / "meta" / "info.json").is_file()


def _episode_count(root: Path) -> int | str:
    try:
        return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])
    except (OSError, KeyError, ValueError, TypeError):
        return "an unknown number of"


def _assert_resumable(dataset: Any, *, features: dict[str, dict], fps: int, root: Path) -> None:
    """Resuming across a config change would interleave two incompatible corpora.

    `LeRobotDataset.resume` takes no `features`: it adopts whatever is on disk. So a
    K=4 run resuming a K=1 shard writes K=4 rows into a K=1 schema, and the mismatch
    surfaces (if at all) as a shape error thousands of frames later. Check up front.
    """
    existing = dataset.meta.features
    if int(dataset.meta.fps) != fps:
        raise ValueError(f"{root} was recorded at {dataset.meta.fps} fps, this run is {fps} fps.")
    for key, spec in features.items():
        if key not in existing:
            raise ValueError(
                f"{root} has no feature {key!r}; it was recorded by a different config."
            )
        want, got = tuple(spec["shape"]), tuple(existing[key]["shape"])
        if want != got:
            raise ValueError(
                f"{root} stores {key} with shape {got}, this run would write {want}. "
                "Resuming would mix two schemas in one shard. Record to a new root."
            )


def run_episodes(
    *,
    envs: dict[str, dict[int, Any]],
    policy: ChunkPolicy,
    processors: Processors,
    cfg: PipelineConfig,
    dataset: Any | None,
    start_seed: int,
    skip_episodes: int = 0,
    timer: StageTimer | None = None,
) -> dict[str, Any]:
    """Run n_episodes per task. Returns success stats and per-episode rows.

    `skip_episodes` resumes a partial shard. Episode order is deterministic --
    task ascending, then repeat index -- and `seed = start_seed + episode_index`,
    so skipping the first N reproduces exactly the tail that is missing.

    `timer` is always present in the returned stats; pass one in to control CUDA
    synchronisation or to read it after a run that raised. It never influences
    what is recorded.
    """
    timer = timer if timer is not None else StageTimer()
    # Here, not at construction: the caller built this before the checkpoint and the
    # simulator existed, and that setup must not divide into s/step.
    timer.start()
    k = cfg.record.k_samples
    privileged = (
        PrivilegedWriter(cfg.record.privileged_root, cfg.record.shard)
        if cfg.record.enabled and cfg.record.privileged
        else None
    )
    ctx = RolloutContext(
        obs=ObsPipeline(processors),
        decide=Decider(policy, processors, k, timer),
        sink=EpisodeSink(dataset, privileged, timer, k),
        perturbation=make_perturbation(
            cfg.record.perturbation,
            cfg.record.perturbation_strength,
            **cfg.record.perturbation_kwargs,
        ),
        timer=timer,
    )

    rows: list[dict[str, Any]] = []
    n_success = 0
    episode_index = 0
    for suite_name, task_map in envs.items():
        for task_id, vec_env in sorted(task_map.items()):
            for _ep in range(cfg.eval.n_episodes):
                seed = start_seed + episode_index
                if episode_index >= skip_episodes:
                    spec = EpisodeSpec(vec_env, episode_index, seed, suite_name, int(task_id))
                    outcome = rollout_episode(ctx, spec)
                    n_success += int(outcome.success)
                    rows.append(
                        {
                            "episode_index": episode_index,
                            "suite": suite_name,
                            "task_id": int(task_id),
                            "seed": seed,
                            "success": outcome.success,
                            "n_steps": outcome.n_steps,
                        }
                    )
                    logger.info(
                        "episode %s suite=%s task=%s seed=%s steps=%s success=%s running=%.1f%%",
                        episode_index + 1,
                        suite_name,
                        task_id,
                        seed,
                        outcome.n_steps,
                        outcome.success,
                        100.0 * n_success / len(rows),
                    )
                episode_index += 1

    if dataset is not None:
        with timer("finalize"):
            dataset.finalize()
    timer.finish()

    n_total = len(rows)
    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_failure": n_total - n_success,
        "success_rate": n_success / n_total if n_total else 0.0,
        "episodes": rows,
        # A resumed run only sees its own tail. The rate below is over that tail, not
        # over the shard, so callers must not compare it against the tolerance band.
        "n_skipped": int(skip_episodes),
        "partial": skip_episodes > 0,
        "timing": timer.summary(),
    }


@dataclass(frozen=True)
class RolloutContext:
    """What stays the same from one episode to the next."""

    obs: ObsPipeline
    decide: Decider
    sink: EpisodeSink
    perturbation: Perturbation
    timer: StageTimer


@dataclass(frozen=True)
class EpisodeSpec:
    vec_env: Any
    episode_index: int
    seed: int
    suite_name: str
    task_id: int


@dataclass(frozen=True)
class EpisodeOutcome:
    """Length separates a timeout (the step cap) from an early end."""

    success: bool
    n_steps: int


def rollout_episode(ctx: RolloutContext, spec: EpisodeSpec) -> EpisodeOutcome:
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: PLC0415

    vec_env, seed, timer, perturbation = spec.vec_env, spec.seed, ctx.timer, ctx.perturbation
    ctx.decide.policy.reset()
    # `attach` needs in-process sub-envs, which a family that only perturbs the
    # action (applied in `decide`, not through the wrapper) has no reason to force.
    if perturbation.needs_env_hook:
        attach(vec_env, perturbation, episode_seed=seed)
    with timer("env_reset"):
        raw_obs, _info = vec_env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    # After the reset: `hard_reset` rebuilds the simulator, discarding any earlier hook.
    instrument_render(vec_env, timer)
    ctx.sink.begin(spec.episode_index)
    # Read once: the perturbation wrapper sets it at reset, and nothing changes it later.
    task = task_description(vec_env, fallback=f"{spec.suite_name}/task_{spec.task_id}")
    max_steps = max_episode_steps(vec_env)
    # One generator per episode, advanced once per step, and separate from
    # `on_reset`'s stream (see backstop.perturb.base.episode_rng).
    perturb_rng = episode_rng(seed, perturbation.name)

    success, done, step = False, False, 0
    while not done and step < max_steps:
        with timer("obs_prep"):
            prepared = ctx.obs.prepare(raw_obs, task)
        ctx.sink.add_sim_state(vec_env)
        with timer("normalize"):
            observation = ctx.obs.normalize(prepared)
        decision = ctx.decide(
            observation,
            episode_seed=seed,
            step=step,
            perturb=lambda action: perturbation.perturb_action(action, perturb_rng),
        )
        with timer("env_step"):
            raw_obs, _reward, terminated, truncated, info = vec_env.step(decision.action)
        success = _read_success(info, success)
        done = bool(terminated[0] or truncated[0]) or step + 1 == max_steps
        ctx.sink.add_frame(prepared.frame_obs, decision, success=success and done, task=task)
        step += 1
        timer.step_done()
        if step % 50 == 0:
            logger.info(
                "task=%s seed=%s step=%s/%s success_so_far=%s",
                spec.task_id,
                seed,
                step,
                max_steps,
                success,
            )

    ctx.sink.end()
    timer.episode_done()
    return EpisodeOutcome(success, step)


def _read_success(info: dict[str, Any], current: bool) -> bool:
    if current:
        return True
    if "final_info" in info:
        final = info["final_info"]
        if isinstance(final, dict):
            flag = final.get("is_success", False)
            if hasattr(flag, "tolist"):
                return bool(np.any(flag))
            return bool(flag)
        for item in final:
            if isinstance(item, dict) and item.get("is_success"):
                return True
    if "is_success" in info:
        flag = info["is_success"]
        if hasattr(flag, "tolist"):
            return bool(np.any(flag))
        return bool(flag)
    return False
