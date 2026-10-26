"""Raw env observation -> what the policy sees and what the corpus stores.

Two steps, kept apart because the privileged sim state is captured between them:
`prepare` formats the observation and reads the frame to record, in real units;
`normalize` then scales it for the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backstop.policy.base import Processors, as_numpy
from backstop.schema import STATE_DIM


@dataclass(frozen=True)
class PreparedObs:
    policy_obs: dict[str, Any]
    frame_obs: dict[str, np.ndarray]


@dataclass(frozen=True)
class ObsPipeline:
    processors: Processors

    def prepare(self, raw_obs: Any, task: str) -> PreparedObs:
        from lerobot.envs import preprocess_observation  # noqa: PLC0415

        observation: dict[str, Any] = preprocess_observation(raw_obs)
        observation["task"] = [task]
        observation = self.processors.env_pre(observation)
        # Captured after env_pre (state assembled, images flipped) but before
        # normalizing: the corpus stores what the policy saw, in real units.
        return PreparedObs(observation, capture_obs(observation))

    def normalize(self, prepared: PreparedObs) -> dict[str, Any]:
        return self.processors.pre(prepared.policy_obs)


def capture_obs(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read the frame straight off the processed observation.

    `env_pre` (LiberoProcessorStep) has already assembled `observation.state` as
    eef_pos(3) + eef_axisangle(3) + gripper_qpos(2) and rotated the images 180
    degrees to match the checkpoint's camera convention. Rebuilding either of
    those from the raw env dict is how the corpus ends up storing rotation-matrix
    entries and upside-down video, so read them rather than reconstruct them.
    """
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: PLC0415

    state = as_numpy(observation[OBS_STATE])[0].astype(np.float32)
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
    array = as_numpy(img)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return np.ascontiguousarray(array)
