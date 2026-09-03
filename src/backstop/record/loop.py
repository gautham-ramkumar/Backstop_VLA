"""Rollout loop that records Backstop extras (action_chunk, K samples)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from backstop.config import PipelineConfig
from backstop.policy.adapter import SmolVLAAdapter
from backstop.schema import ACTION_DIM, CHUNK_SIZE, STATE_DIM, lerobot_features

logger = logging.getLogger(__name__)


def dataset_features(k_samples: int, image_hw: tuple[int, int] = (360, 360)) -> dict[str, dict]:
    return lerobot_features(k_samples, image_hw)


def create_dataset(cfg: PipelineConfig, image_hw: tuple[int, int] = (360, 360), *, fps: int) -> Any:
    """`fps` must be the env control rate (LIBERO: 20 Hz), one row per control step.

    Not the 30 that was hardcoded here, and not `metadata["render_fps"]` (80) --
    a wrong value silently desyncs viewer playback and lead-time-in-seconds.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    root = Path(cfg.record.root)
    if root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=cfg.record.repo_id,
        fps=int(fps),
        features=dataset_features(cfg.record.k_samples, image_hw),
        robot_type="panda",
        root=str(root),
        use_videos=True,
    )


def run_episodes(
    *,
    envs: dict[str, dict[int, Any]],
    adapter: SmolVLAAdapter,
    processors: dict[str, Any],
    cfg: PipelineConfig,
    dataset: Any | None,
    start_seed: int,
) -> dict[str, Any]:
    """Run n_episodes per task. Returns success stats and per-episode rows."""
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

    for suite_name, task_map in envs.items():
        for task_id, vec_env in sorted(task_map.items()):
            for _ep in range(cfg.eval.n_episodes):
                seed = start_seed + episode_index
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
        dataset.finalize()

    success_rate = n_success / n_total if n_total else 0.0
    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_failure": n_total - n_success,
        "success_rate": success_rate,
        "episodes": rows,
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
    seed: int,
    suite_name: str,
    task_id: int,
) -> tuple[bool, int]:
    """Returns (success, n_steps). Episode length separates timeouts from early ends."""
    adapter.reset()
    observation, _info = vec_env.reset(seed=[seed], options={new_rollout_option: True})
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
        observation = preprocess_observation(observation)
        try:
            observation["task"] = list(vec_env.call("task_description"))
        except (AttributeError, NotImplementedError, TypeError):
            observation["task"] = [task_desc]

        observation = env_pre(observation)
        # Capture after env_pre (state assembled, images flipped) but before `pre`,
        # which normalizes. The corpus stores what the policy saw, in real units.
        frame_obs = _capture_obs(observation)
        observation = pre(observation)
        with torch.inference_mode():
            action = adapter.select_action(
                observation, noise=adapter.noise_for(episode_seed=seed, step=step, index=0)
            )
            chunk = adapter.last_action_chunk()
            samples = adapter.sample_chunks(observation, k, episode_seed=seed, step=step)

        action = post(action)
        transition = {action_key: action}
        transition = env_post(transition)
        stepped = _as_numpy(transition[action_key])
        if stepped.ndim == 1:
            stepped = stepped.reshape(1, -1)
        chunk = _unnormalize_array(post, chunk)
        samples = _unnormalize_array(post, samples)

        observation, _reward, terminated, truncated, info = vec_env.step(stepped)
        success = _read_success(info, success)
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        if dataset is not None:
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
        dataset.save_episode()
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
