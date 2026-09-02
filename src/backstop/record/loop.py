"""Rollout loop that records Backstop extras (action_chunk, K samples)."""

from __future__ import annotations

import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from backstop.config import PipelineConfig
from backstop.policy.adapter import SmolVLAAdapter
from backstop.schema import ACTION_DIM, CHUNK_SIZE, STATE_DIM, lerobot_features

logger = logging.getLogger(__name__)


def dataset_features(k_samples: int, image_hw: tuple[int, int] = (360, 360)) -> dict[str, dict]:
    return lerobot_features(k_samples, image_hw)


def create_dataset(cfg: PipelineConfig, image_hw: tuple[int, int] = (360, 360)) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    root = Path(cfg.record.root)
    if root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=cfg.record.repo_id,
        fps=30,
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
                success = _rollout_one(
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
                    }
                )
                episode_index += 1
                logger.info(
                    "episode %s suite=%s task=%s seed=%s success=%s running=%.1f%%",
                    episode_index,
                    suite_name,
                    task_id,
                    seed,
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
) -> bool:
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
        raw_obs = deepcopy(observation)
        observation = preprocess_observation(observation)
        try:
            observation["task"] = list(vec_env.call("task_description"))
        except (AttributeError, NotImplementedError, TypeError):
            observation["task"] = [task_desc]

        observation = env_pre(observation)
        observation = pre(observation)
        with torch.inference_mode():
            action = adapter.select_action(observation)
        chunk = adapter.last_action_chunk()
        samples = adapter.sample_chunks(observation, k)

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
                raw_obs=raw_obs,
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
    return bool(success)


def _build_frame(
    *,
    raw_obs: dict[str, Any],
    action: np.ndarray,
    chunk: np.ndarray,
    samples: np.ndarray,
    success: bool,
    task: str,
    k: int,
) -> dict[str, Any]:
    image, image2, state = _extract_obs(raw_obs)
    return {
        "observation.images.image": image,
        "observation.images.image2": image2,
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32).reshape(ACTION_DIM),
        "action_chunk": np.asarray(chunk, dtype=np.float32).reshape(CHUNK_SIZE, ACTION_DIM),
        "action_chunk_samples": np.asarray(samples, dtype=np.float32).reshape(
            k, CHUNK_SIZE, ACTION_DIM
        ),
        "next.success": np.array([success], dtype=bool),
        "task": task,
    }


def _extract_obs(raw_obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = raw_obs.get("pixels")
    image = None
    image2 = None
    if isinstance(pixels, dict):
        preferred = ["agentview_image", "image", "robot0_eye_in_hand_image", "image2"]
        ordered = [k for k in preferred if k in pixels] + [k for k in pixels if k not in preferred]
        first = pixels[ordered[0]]
        second = pixels[ordered[1]] if len(ordered) > 1 else first
        image = _to_hwc_uint8(first)
        image2 = _to_hwc_uint8(second)
    elif pixels is not None:
        image = _to_hwc_uint8(pixels)
        image2 = image
    if image is None:
        image = np.zeros((360, 360, 3), dtype=np.uint8)
    if image2 is None:
        image2 = image
    state = raw_obs.get("agent_pos")
    if state is None:
        state = raw_obs.get("robot_state")
    state_np = _pad_vec(_first_numeric(state), STATE_DIM)
    return image, image2, state_np


def _first_numeric(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    if isinstance(value, dict):
        for nested in value.values():
            try:
                return _first_numeric(nested)
            except (TypeError, ValueError):
                continue
        return np.zeros(STATE_DIM, dtype=np.float32)
    array = _as_numpy(value)
    return np.asarray(array, dtype=np.float32)


def _to_hwc_uint8(img: Any) -> np.ndarray:
    array = _as_numpy(img)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        max_v = float(array.max()) if array.size else 0.0
        if max_v <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return np.ascontiguousarray(array)


def _pad_vec(value: np.ndarray, dim: int) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    n = min(dim, flat.shape[0])
    out[:n] = flat[:n]
    return out


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
    """Best-effort policy-space → env-space using the LeRobot postprocessor."""
    try:
        import torch  # noqa: PLC0415

        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        if tensor.ndim == 3:
            k, t, a = tensor.shape
            out = post(tensor.reshape(k * t, a))
            return np.asarray(_as_numpy(out), dtype=np.float32).reshape(k, t, -1)
        out = post(tensor)
        return np.asarray(_as_numpy(out), dtype=np.float32).reshape(array.shape)
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return np.asarray(array, dtype=np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
