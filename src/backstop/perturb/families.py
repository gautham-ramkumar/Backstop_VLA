"""The perturbation families.

Each takes `strength` in [0, 1] and maps it to its own units. The ranges below are
starting points for the calibration sweep, not tuned values -- the sweep picks the
strength whose success rate lands in 30-70%, and until it has run these numbers
are guesses.

Coverage against the four subsystems a failure can come from:

| family         | breaks        | mechanism                          |
|----------------|---------------|------------------------------------|
| `lighting`     | perception    | robosuite LightingModder at reset  |
| `object_pose`  | grounding     | object free-joint qpos at reset    |
| `absent_target`| precondition  | instruction names a missing object |
| `action_noise` | control       | Gaussian on the executed action    |
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Table bounds live at the top so the calibration sweep has one place to read.
LIGHT_DIFFUSE_DROP = 0.8  # at strength 1.0, diffuse falls to 20% of default
LIGHT_POS_JITTER_M = 1.5
OBJECT_XY_SIGMA_M = 0.08
OBJECT_YAW_SIGMA_RAD = 0.6
ACTION_NOISE_SIGMA = 0.5  # fraction of the [-1, 1] action range, at strength 1.0


class Lighting:
    """Dim and move the scene lights. Purely visual, so no re-settling."""

    name = "lighting"

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self._applied: dict[str, Any] = {}

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        from robosuite.utils.mjmod import LightingModder  # noqa: PLC0415

        modder = LightingModder(control_env.env.sim)
        scale = 1.0 - LIGHT_DIFFUSE_DROP * self.strength
        offset = LIGHT_POS_JITTER_M * self.strength
        applied = {}
        for light in modder.light_names:
            base = np.array(modder.get_diffuse(light), dtype=np.float64)
            modder.set_diffuse(light, base * scale)
            position = np.array(modder.get_pos(light), dtype=np.float64)
            shift = rng.normal(0.0, offset, size=3) if offset > 0 else np.zeros(3)
            modder.set_pos(light, position + shift)
            applied[light] = {"diffuse_scale": scale, "pos_shift": shift.tolist()}
        self._applied = applied
        return False

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return action

    def params(self) -> dict[str, Any]:
        return {"strength": self.strength, "lights": self._applied}


class ObjectPose:
    """Move the objects on the table. This is 'move the target object'.

    Writes each movable object's free-joint qpos directly, then reports that
    physics changed so the wrapper re-settles: a jittered object can end up
    interpenetrating the table or another object, and the settle steps are what
    turn an invalid pose into a resting one (or reveal that it cannot).
    """

    name = "object_pose"

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self._applied: dict[str, Any] = {}

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        sim = control_env.env.sim
        xy_sigma = OBJECT_XY_SIGMA_M * self.strength
        yaw_sigma = OBJECT_YAW_SIGMA_RAD * self.strength
        applied = {}
        for name, address in _free_joint_addresses(control_env).items():
            qpos = sim.data.qpos
            shift = rng.normal(0.0, xy_sigma, size=2)
            qpos[address : address + 2] += shift
            yaw = float(rng.normal(0.0, yaw_sigma))
            qpos[address + 3 : address + 7] = _rotate_quat_z(qpos[address + 3 : address + 7], yaw)
            applied[name] = {"xy_shift": shift.tolist(), "yaw": yaw}
        sim.forward()
        self._applied = applied
        return True

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return action

    def params(self) -> dict[str, Any]:
        return {"strength": self.strength, "objects": self._applied}


class ActionNoise:
    """Gaussian noise on the executed action. Breaks control, not perception.

    The only family where the executed action differs from the policy's, which is
    why the schema carries `action_policy` alongside `action`. The gripper
    dimension is left alone: it is effectively binary, and noising it produces
    random open/close chatter that fails every episode for a reason that has
    nothing to do with degraded control.
    """

    name = "action_noise"

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self.sigma = ACTION_NOISE_SIGMA * float(strength)

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        return False

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noisy = np.asarray(action, dtype=np.float64).copy()
        noisy[..., :-1] += rng.normal(0.0, self.sigma, size=noisy[..., :-1].shape)
        return np.clip(noisy, -1.0, 1.0).astype(np.asarray(action).dtype)

    def params(self) -> dict[str, Any]:
        return {"strength": self.strength, "sigma": self.sigma, "gripper_noised": False}


class AbsentTarget:
    """Instruct the robot to manipulate an object that is not in the scene.

    'Remove the target object', done through the instruction rather than through
    model surgery: same effect on the precondition the policy needs, and it does
    not require recompiling the MuJoCo model.

    0% success by construction, so this family is quarantined out of the balanced
    arm and reported separately. Its perturbation flag is perfectly predictive of
    the outcome, which is precisely what a guard must not be trained on.
    """

    name = "absent_target"
    strength = 1.0

    def __init__(self, replacement: str) -> None:
        self.replacement = replacement

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        return False

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return action

    def task_description(self, original: str) -> str:
        return self.replacement

    def params(self) -> dict[str, Any]:
        return {"replacement": self.replacement}


def _free_joint_addresses(control_env: Any) -> dict[str, int]:
    """qpos start index of every movable object's free joint.

    Only objects the task declares as interesting: `obj_of_interest` is LIBERO's
    own list, so this moves the things the instruction can refer to and leaves
    fixtures alone.
    """
    sim = control_env.env.sim
    addresses = {}
    for name in getattr(control_env.env, "obj_of_interest", []):
        joint = f"{name}_joint0"
        try:
            joint_id = sim.model.joint_name2id(joint)
        except (ValueError, KeyError):
            continue
        addresses[name] = int(sim.model.jnt_qposadr[joint_id])
    return addresses


def _rotate_quat_z(quat: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a (w, x, y, z) quaternion about the world z axis. MuJoCo order."""
    half = yaw / 2.0
    rot = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    w0, x0, y0, z0 = rot
    w1, x1, y1, z1 = quat
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )
