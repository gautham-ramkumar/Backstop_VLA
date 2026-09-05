"""Rollout loop that records Backstop extras (action_chunk, K samples)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from backstop.config import PipelineConfig
from backstop.env.libero import instrument_render, sim_state
from backstop.policy.adapter import SmolVLAAdapter
from backstop.record.privileged import PrivilegedWriter
from backstop.record.timing import StageTimer
from backstop.schema import ACTION_DIM, CHUNK_SIZE, STATE_DIM, lerobot_features

logger = logging.getLogger(__name__)


def dataset_features(k_samples: int, image_hw: tuple[int, int] = (360, 360)) -> dict[str, dict]:
    return lerobot_features(k_samples, image_hw)


def create_dataset(
    cfg: PipelineConfig, image_hw: tuple[int, int] = (360, 360), *, fps: int
) -> tuple[Any, int]:
    """Open the shard for writing. Returns `(dataset, n_already_recorded)`.

    `fps` must be the env control rate (LIBERO: 20 Hz), one row per control step.
    Not the 30 that was hardcoded here, and not `metadata["render_fps"]` (80) --
    a wrong value silently desyncs viewer playback and lead-time-in-seconds.

    `record.on_existing` decides what happens when the root is already populated.
    Week 1 always destroyed it, which is how a stray re-run costs a night of GPU.
    """
    _assert_privileged_decided(cfg)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    root = Path(cfg.record.root)
    features = dataset_features(cfg.record.k_samples, image_hw)
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

    `record.privileged` defaults to false, so a week-2 shard config that simply
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
    adapter: SmolVLAAdapter,
    processors: dict[str, Any],
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
    from lerobot.envs import preprocess_observation  # noqa: PLC0415
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: PLC0415
    from lerobot.utils.constants import ACTION  # noqa: PLC0415

    env_pre = processors["env_preprocessor"]
    env_post = processors["env_postprocessor"]
    pre = processors["preprocessor"]
    post = processors["postprocessor"]

    rows: list[dict[str, Any]] = []
    n_success = 0
    n_total = 0
    episode_index = 0
    timer = timer if timer is not None else StageTimer()
    # Here, not at construction: the caller built this before the checkpoint and the
    # simulator existed, and that setup must not divide into s/step.
    timer.start()
    privileged = (
        PrivilegedWriter(cfg.record.privileged_root, cfg.record.shard)
        if cfg.record.enabled and cfg.record.privileged
        else None
    )

    for suite_name, task_map in envs.items():
        for task_id, vec_env in sorted(task_map.items()):
            for _ep in range(cfg.eval.n_episodes):
                seed = start_seed + episode_index
                if episode_index < skip_episodes:
                    episode_index += 1
                    continue
                success, n_steps = _rollout_one(
                    vec_env=vec_env,
                    adapter=adapter,
                    env_pre=env_pre,
                    env_post=env_post,
                    pre=pre,
                    post=post,
                    preprocess_observation=preprocess_observation,
                    action_key=ACTION,
                    new_rollout_option=NEW_ROLLOUT_OPTION,
                    cfg=cfg,
                    dataset=dataset,
                    privileged=privileged,
                    timer=timer,
                    episode_index=episode_index,
                    seed=seed,
                    suite_name=suite_name,
                    task_id=int(task_id),
                )
                n_total += 1
                n_success += int(success)
                rows.append(
                    {
                        "episode_index": episode_index,
                        "suite": suite_name,
                        "task_id": int(task_id),
                        "seed": seed,
                        "success": bool(success),
                        "n_steps": int(n_steps),
                    }
                )
                episode_index += 1
                logger.info(
                    "episode %s suite=%s task=%s seed=%s steps=%s success=%s running=%.1f%%",
                    episode_index,
                    suite_name,
                    task_id,
                    seed,
                    n_steps,
                    success,
                    100.0 * n_success / n_total,
                )

    if dataset is not None:
        with timer("finalize"):
            dataset.finalize()
    timer.finish()

    success_rate = n_success / n_total if n_total else 0.0
    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_failure": n_total - n_success,
        "success_rate": success_rate,
        "episodes": rows,
        # A resumed run only sees its own tail. The rate below is over that tail, not
        # over the shard, so callers must not compare it against the tolerance band.
        "n_skipped": int(skip_episodes),
        "partial": skip_episodes > 0,
        "timing": timer.summary(),
    }


def _rollout_one(
    *,
    vec_env: Any,
    adapter: SmolVLAAdapter,
    env_pre: Any,
    env_post: Any,
    pre: Any,
    post: Any,
    preprocess_observation: Any,
    action_key: str,
    new_rollout_option: Any,
    cfg: PipelineConfig,
    dataset: Any | None,
    privileged: PrivilegedWriter | None,
    timer: StageTimer,
    episode_index: int,
    seed: int,
    suite_name: str,
    task_id: int,
) -> tuple[bool, int]:
    """Returns (success, n_steps). Episode length separates timeouts from early ends."""
    adapter.reset()
    with timer("env_reset"):
        observation, _info = vec_env.reset(seed=[seed], options={new_rollout_option: True})
    # After the reset, not before: `hard_reset` rebuilds the simulator, discarding
    # any hook installed against the previous one.
    instrument_render(vec_env, timer)
    if privileged is not None:
        privileged.begin_episode(episode_index)
    try:
        task_desc = list(vec_env.call("task_description"))[0]
    except (AttributeError, NotImplementedError, TypeError):
        task_desc = f"{suite_name}/task_{task_id}"

    done = np.array([False])
    max_steps = vec_env.call("_max_episode_steps")[0]
    success = False
    step = 0
    k = cfg.record.k_samples
    import torch  # noqa: PLC0415

    while not bool(done[0]) and step < max_steps:
        with timer("obs_prep"):
            observation = preprocess_observation(observation)
            try:
                observation["task"] = list(vec_env.call("task_description"))
            except (AttributeError, NotImplementedError, TypeError):
                observation["task"] = [task_desc]

            observation = env_pre(observation)
            # Capture after env_pre (state assembled, images flipped) but before `pre`,
            # which normalizes. The corpus stores what the policy saw, in real units.
            frame_obs = _capture_obs(observation)
        # Sim state *before* this step's action, so dataset row t and sidecar row t
        # describe the same instant: restore sim_state[t], apply action[t], and the
        # transition reproduces. Capturing after `vec_env.step` would offset them by
        # one and silently shift every onset annotation.
        if privileged is not None:
            with timer("privileged"):
                privileged.add(sim_state(vec_env))
        with timer("normalize"):
            observation = pre(observation)
        with torch.inference_mode():
            # Split from the sampling below because they answer different questions:
            # `policy_select` is the cost K cannot remove, `policy_samples` is the
            # cost K adds. The day-1 gate is the ratio between them.
            with timer("policy_select"):
                action = adapter.select_action(
                    observation, noise=adapter.noise_for(episode_seed=seed, step=step, index=0)
                )
                chunk = adapter.last_action_chunk()
            with timer("policy_samples"):
                samples = adapter.sample_chunks(observation, k, episode_seed=seed, step=step)

        with timer("action_post"):
            action = post(action)
            transition = {action_key: action}
            transition = env_post(transition)
            stepped = _as_numpy(transition[action_key])
            if stepped.ndim == 1:
                stepped = stepped.reshape(1, -1)
            chunk = _unnormalize_array(post, chunk)
            samples = _unnormalize_array(post, samples)

        with timer("env_step"):
            observation, _reward, terminated, truncated, info = vec_env.step(stepped)
        success = _read_success(info, success)
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        if dataset is not None:
            with timer("frame_write"):
                frame = _build_frame(
                    frame_obs=frame_obs,
                    action=stepped[0],
                    chunk=chunk,
                    samples=samples,
                    success=bool(success and done[0]),
                    task=str(task_desc),
                    k=k,
                )
                dataset.add_frame(frame)
        step += 1
        timer.step_done()
        if step % 50 == 0:
            logger.info(
                "task=%s seed=%s step=%s/%s success_so_far=%s",
                task_id,
                seed,
                step,
                max_steps,
                success,
            )

    if dataset is not None:
        with timer("save_episode"):
            dataset.save_episode()
    timer.episode_done()
    if privileged is not None:
        n_states = privileged.n_steps_open
        if n_states != step:
            raise RuntimeError(
                f"episode {episode_index}: {n_states} sim states for {step} recorded frames. "
                "The sidecar and the dataset would not line up row for row."
            )
        privileged.end_episode()
    return bool(success), step


def _build_frame(
    *,
    frame_obs: dict[str, np.ndarray],
    action: np.ndarray,
    chunk: np.ndarray,
    samples: np.ndarray,
    success: bool,
    task: str,
    k: int,
) -> dict[str, Any]:
    return {
        "observation.images.image": frame_obs["image"],
        "observation.images.image2": frame_obs["image2"],
        "observation.state": frame_obs["state"],
        "action": np.asarray(action, dtype=np.float32).reshape(ACTION_DIM),
        "action_chunk": np.asarray(chunk, dtype=np.float32).reshape(CHUNK_SIZE, ACTION_DIM),
        "action_chunk_samples": np.asarray(samples, dtype=np.float32).reshape(
            k, CHUNK_SIZE, ACTION_DIM
        ),
        "next.success": np.array([success], dtype=bool),
        "task": task,
    }


def _capture_obs(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read the frame straight off the processed observation.

    `env_pre` (LiberoProcessorStep) has already assembled `observation.state` as
    eef_pos(3) + eef_axisangle(3) + gripper_qpos(2) and rotated the images 180
    degrees to match the checkpoint's camera convention. Rebuilding either of
    those from the raw env dict is how the corpus ends up storing rotation-matrix
    entries and upside-down video, so read them rather than reconstruct them.
    """
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: PLC0415

    state = _as_numpy(observation[OBS_STATE])[0].astype(np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(
            f"{OBS_STATE} has shape {state.shape}, expected ({STATE_DIM},). "
            "The env preprocessor changed; update schema.STATE_DIM deliberately."
        )
    image = _to_hwc_uint8(observation[f"{OBS_IMAGES}.image"])
    image2_key = f"{OBS_IMAGES}.image2"
    image2 = _to_hwc_uint8(observation[image2_key]) if image2_key in observation else image
    return {"image": image, "image2": image2, "state": state}


def _to_hwc_uint8(img: Any) -> np.ndarray:
    """(B, C, H, W) float32 in [0, 1] -> (H, W, C) uint8, exactly inverting preprocessing."""
    array = _as_numpy(img)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return np.ascontiguousarray(array)


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


def _unnormalize_array(post: Any, array: np.ndarray) -> np.ndarray:
    """Policy-space → env-space using the LeRobot postprocessor.

    On failure this returns the array unchanged, which leaves the stored chunk in
    policy space while the executed action is in env space -- the two silently
    stop being comparable. That is unrecoverable after the fact, so make the
    fallback loud. `verify_corpus.py` catches it too, by asserting
    `action == action_chunk[0]`.
    """
    try:
        import torch  # noqa: PLC0415

        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        if tensor.ndim == 3:
            k, t, a = tensor.shape
            out = post(tensor.reshape(k * t, a))
            return np.asarray(_as_numpy(out), dtype=np.float32).reshape(k, t, -1)
        out = post(tensor)
        return np.asarray(_as_numpy(out), dtype=np.float32).reshape(array.shape)
    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
        logger.warning(
            "UNNORMALIZE FAILED on shape %s (%s: %s). Stored chunks are in POLICY "
            "space while executed actions are in ENV space -- this corpus is not "
            "usable for chunk-based signals. Fix before recording further.",
            np.shape(array),
            type(exc).__name__,
            exc,
        )
        return np.asarray(array, dtype=np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
