"""The perturbation families.

Each takes `strength` in [0, 1] and maps it to its own units. The ranges below are
starting points for the calibration sweep, not tuned values -- the sweep picks the
strength whose success rate lands in 30-70%, and until it has run these numbers
are guesses.

Coverage against the four subsystems a failure can come from:

| family              | breaks        | mechanism                          |
|---------------------|---------------|-------------------------------------|
| `lighting`           | perception    | robosuite LightingModder at reset  |
| `camera`             | perception    | robosuite CameraModder at reset    |
| `object_pose`        | grounding     | object free-joint qpos at reset    |
| `distractor`         | perception    | BDDL/MuJoCo model edit before compile |
| `absent_target`      | precondition  | instruction names a missing object |
| `language_paraphrase`| control (none)| benign instruction reword, control arm |
| `action_noise`       | control       | Gaussian on the executed action    |
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from backstop.env.libero import swap_task_files
from backstop.perturb.base import Perturbation, episode_rng

# Table bounds live at the top so the calibration sweep has one place to read.
LIGHT_DIFFUSE_DROP = 0.8  # at strength 1.0, diffuse falls to 20% of default
LIGHT_POS_JITTER_M = 1.5
CAMERA_POS_JITTER_M = 0.03  # at strength 1.0, per-axis sigma
CAMERA_ROT_JITTER_RAD = 0.15  # at strength 1.0, axis-angle magnitude (~8.6 deg)
CAMERA_FOVY_JITTER_DEG = 6.0  # at strength 1.0, sigma
OBJECT_XY_SIGMA_M = 0.08
OBJECT_YAW_SIGMA_RAD = 0.6
OBJECT_POSE_MAX_ATTEMPTS = 5
OBJECT_POSE_Z_TOLERANCE_M = 0.03  # resting height must stay this close to the pre-jitter height
OBJECT_POSE_VEL_TOLERANCE = 0.01  # m/s (linear) and rad/s (angular), after settling
OBJECT_POSE_PENETRATION_TOL_M = 0.003  # contact depth beyond this counts as interpenetrating
ACTION_NOISE_SIGMA = 0.5  # fraction of the [-1, 1] action range, at strength 1.0

# `distractor`'s object/region choice. LIBERO's own object registry has 50
# categories (`libero.libero.envs.objects.get_object_dict()`); `milk` is not used
# by any of libero_spatial's 5 declared objects or 2 fixtures (checked directly:
# grepped every task's `:objects` section). The region box is main_table-relative
# (x_min, y_min, x_max, y_max) -- checked against all ten libero_spatial task bddl
# files, which share one identical 11-region layout (diffed byte-for-byte), and
# clears every one of them with margin (the nearest declared region's x tops out
# at 0.21; this box starts at 0.30).
DISTRACTOR_CATEGORY = "milk"
DISTRACTOR_OBJECT_NAME = f"{DISTRACTOR_CATEGORY}_1"
DISTRACTOR_REGION_NAME = "distractor_region"
DISTRACTOR_REGION_RANGE = (0.30, -0.25, 0.36, -0.19)
DISTRACTOR_XY_JITTER_SIGMA_M = 0.02  # at strength 1.0, sigma of xy jitter around the sampled anchor
DISTRACTOR_YAW_JITTER_RAD = 0.6  # at strength 1.0, matches OBJECT_YAW_SIGMA_RAD's magnitude


class Lighting(Perturbation):
    """Dim and move the scene lights, and the camera headlight. Purely visual,
    so no re-settling.

    Scaling only the named scene lights (`light1`/`light2` in LIBERO-Spatial)
    barely changes the rendered image: MuJoCo's camera headlight
    (`model.vis.headlight`) is on by default, follows the camera, and is not
    part of `LightingModder`'s surface. Measured against the real simulator --
    not assumed -- dimming just the scene lights to 20% moved mean agentview
    pixel value 120 -> 87; scaling the headlight the same way took it to 11.
    Without the headlight, `strength=1.0` was barely darker than doing nothing.
    """

    name = "lighting"

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self._applied_lights: dict[str, Any] = {}
        self._applied_headlight: dict[str, Any] = {}

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        from robosuite.utils.mjmod import LightingModder  # noqa: PLC0415

        sim = control_env.env.sim
        modder = LightingModder(sim)
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
        self._applied_lights = applied

        headlight = sim.model.vis.headlight
        headlight_before = {
            channel: np.array(getattr(headlight, channel), dtype=np.float64).tolist()
            for channel in ("ambient", "diffuse", "specular")
        }
        for channel, base_values in headlight_before.items():
            getattr(headlight, channel)[:] = np.array(base_values) * scale
        self._applied_headlight = {"scale": scale, **headlight_before}

        return False

    def params(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "lights": self._applied_lights,
            "headlight": self._applied_headlight,
        }


AGENTVIEW_CAMERA = "agentview"


class Camera(Perturbation):
    """Jitter the agentview camera's pose and field of view. Purely visual, so no
    re-settling -- the scene itself is untouched, only how it is seen.

    The wrist camera is rigidly mounted to the gripper: jittering it would be
    jittering the robot's own proprioception, not an external sensor, so only
    `agentview` is ever touched.
    """

    name = "camera"

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self._applied: dict[str, Any] = {}

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        from robosuite.utils.mjmod import CameraModder  # noqa: PLC0415

        modder = CameraModder(control_env.env.sim, camera_names=[AGENTVIEW_CAMERA])
        pos_sigma = CAMERA_POS_JITTER_M * self.strength
        rot_angle_max = CAMERA_ROT_JITTER_RAD * self.strength
        fovy_sigma = CAMERA_FOVY_JITTER_DEG * self.strength

        base_pos = np.array(modder.get_pos(AGENTVIEW_CAMERA), dtype=np.float64)
        pos_shift = rng.normal(0.0, pos_sigma, size=3) if pos_sigma > 0 else np.zeros(3)
        modder.set_pos(AGENTVIEW_CAMERA, base_pos + pos_shift)

        base_quat = np.array(modder.get_quat(AGENTVIEW_CAMERA), dtype=np.float64)
        rot_angle = 0.0
        if rot_angle_max > 0:
            axis = rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            rot_angle = float(rng.uniform(-rot_angle_max, rot_angle_max))
            delta_quat = _axis_angle_to_quat(axis, rot_angle)
            new_quat = _quat_multiply(delta_quat, base_quat)
        else:
            new_quat = base_quat
        modder.set_quat(AGENTVIEW_CAMERA, new_quat)

        base_fovy = float(modder.get_fovy(AGENTVIEW_CAMERA))
        fovy_shift = float(rng.normal(0.0, fovy_sigma)) if fovy_sigma > 0 else 0.0
        modder.set_fovy(AGENTVIEW_CAMERA, base_fovy + fovy_shift)

        self._applied = {
            "pos_shift": pos_shift.tolist(),
            "rot_angle_rad": rot_angle,
            "fovy_shift_deg": fovy_shift,
        }
        return False

    def params(self) -> dict[str, Any]:
        return {"strength": self.strength, "camera": AGENTVIEW_CAMERA, **self._applied}


class ObjectPose(Perturbation):
    """Move the objects on the table. This is 'move the target object'.

    Writes each movable object's free-joint qpos directly, then settles and
    validates the result *itself* rather than asking the wrapper to: a jittered
    object can end up off the table, still sliding, or overlapping another
    object, and none of those are silent -- they are exactly the placements a
    later grasp attempt would fail against for a reason that has nothing to do
    with the perturbation being tested. Reject and resample, capped, and count
    both so a persistently high rejection rate is visible rather than baked
    into the corpus as an unusually large fraction of `distractor`-style
    failures with no distractor.

    Settling happens inside `on_reset` (not the wrapper's post-hoc resettle) so
    the validity check runs against the state a retry would actually replace --
    `on_reset` returns `False` here on purpose; asking the wrapper to settle
    again afterward would just repeat work already done.
    """

    name = "object_pose"

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self._applied: dict[str, Any] = {}
        self.n_attempts = 0
        self.n_rejections = 0
        self.last_attempts = 0
        self.last_accepted = True

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        from backstop.perturb.wrapper import SETTLE_STEPS  # noqa: PLC0415

        sim = control_env.env.sim
        xy_sigma = OBJECT_XY_SIGMA_M * self.strength
        yaw_sigma = OBJECT_YAW_SIGMA_RAD * self.strength
        addresses = _free_joint_addresses(control_env)
        joint_ids = {name: sim.model.joint_name2id(f"{name}_joint0") for name in addresses}
        body_ids = {name: int(sim.model.jnt_bodyid[jid]) for name, jid in joint_ids.items()}
        dof_addresses = {name: int(sim.model.jnt_dofadr[jid]) for name, jid in joint_ids.items()}
        base_qpos = {
            name: np.array(sim.data.qpos[addr : addr + 7]) for name, addr in addresses.items()
        }
        table_offset, table_half_xy = _table_bounds(control_env)

        applied: dict[str, Any] = {}
        valid = False
        attempt = 0
        while True:
            attempt += 1
            self.n_attempts += 1
            applied = {}
            for name, address in addresses.items():
                qpos = sim.data.qpos
                shift = rng.normal(0.0, xy_sigma, size=2) if xy_sigma > 0 else np.zeros(2)
                qpos[address : address + 2] = base_qpos[name][:2] + shift
                qpos[address + 2] = base_qpos[name][2]
                yaw = float(rng.normal(0.0, yaw_sigma)) if yaw_sigma > 0 else 0.0
                qpos[address + 3 : address + 7] = _rotate_quat_z(base_qpos[name][3:7], yaw)
                applied[name] = {"xy_shift": shift.tolist(), "yaw": yaw}
            sim.forward()
            for _ in range(SETTLE_STEPS):
                _settle(control_env)
            valid = all(
                _object_is_settled(
                    sim,
                    body_id=body_ids[name],
                    qpos_address=address,
                    dof_address=dof_addresses[name],
                    base_z=base_qpos[name][2],
                    table_offset=table_offset,
                    table_half_xy=table_half_xy,
                )
                for name, address in addresses.items()
            )
            if valid or attempt >= OBJECT_POSE_MAX_ATTEMPTS:
                break
        if not valid:
            self.n_rejections += 1
        for info in applied.values():
            info["accepted"] = valid
        self.last_attempts = attempt
        self.last_accepted = valid
        self._applied = applied
        return False

    def params(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "objects": self._applied,
            "attempts": self.last_attempts,
            "accepted": self.last_accepted,
            "n_attempts": self.n_attempts,
            "n_rejections": self.n_rejections,
        }


class ActionNoise(Perturbation):
    """Gaussian noise on the executed action. Breaks control, not perception.

    The only family where the executed action differs from the policy's, which is
    why the schema carries `action_policy` alongside `action`. The gripper
    dimension is left alone: it is effectively binary, and noising it produces
    random open/close chatter that fails every episode for a reason that has
    nothing to do with degraded control.
    """

    name = "action_noise"
    # Only `perturb_action` does anything (applied in the record loop).
    needs_env_hook = False

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self.sigma = ACTION_NOISE_SIGMA * float(strength)

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noisy = np.asarray(action, dtype=np.float64).copy()
        noisy[..., :-1] += rng.normal(0.0, self.sigma, size=noisy[..., :-1].shape)
        return np.clip(noisy, -1.0, 1.0).astype(np.asarray(action).dtype)

    def params(self) -> dict[str, Any]:
        return {"strength": self.strength, "sigma": self.sigma, "gripper_noised": False}


class AbsentTarget(Perturbation):
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

    @classmethod
    def from_config(cls, strength: float, **kwargs: Any) -> Perturbation:
        # Categorical: the text is supplied, so there is no strength to scale.
        return cls(**kwargs)

    def task_description(self, original: str) -> str:
        return self.replacement

    def params(self) -> dict[str, Any]:
        return {"replacement": self.replacement}


class Paraphrase(Perturbation):
    """Reword the instruction without changing what it asks for. The control arm

    for language: this family must leave the outcome distribution alone. Without
    it, a guard that scores well on `absent_target` could just as easily be
    detecting "the instruction text is unusual" rather than "the named object is
    missing" -- both look like an out-of-distribution instruction from the
    outside, and only a benign reword that doesn't change the task tells them
    apart.

    Categorical, like `absent_target`: the reword is supplied, not sampled, so
    there is nothing for a strength knob to scale.
    """

    name = "language_paraphrase"
    strength = 1.0

    def __init__(self, paraphrase: str) -> None:
        self.paraphrase = paraphrase

    @classmethod
    def from_config(cls, strength: float, **kwargs: Any) -> Perturbation:
        # Categorical: the text is supplied, so there is no strength to scale.
        return cls(**kwargs)

    def task_description(self, original: str) -> str:
        return self.paraphrase

    def params(self) -> dict[str, Any]:
        return {"paraphrase": self.paraphrase}


class Distractor(Perturbation):
    """Add a task-irrelevant object to the scene. The real surgery, not the
    `dynamics` stand-in the plan pre-approved as a fallback.

    Every other family perturbs an already-compiled MuJoCo model through
    `on_reset`. A distractor object cannot go through that hook: it has to be
    in the BDDL *before* MuJoCo compiles the model, and by the time `on_reset`
    runs, `LiberoEnv._ensure_env()` has already built the simulator from the
    original, unmodified BDDL. So this family uses a different hook --
    `prepare_env`, called once from `PerturbedEnv.__init__`, strictly before
    the wrapped env's first `reset()` -- to swap `LiberoEnv._task_bddl_file`
    and `LiberoEnv._init_states` while they are still plain, mutable instance
    attributes nobody has read yet. `_ensure_env()` itself is untouched; it
    just ends up building from a different (temp-file) path than it thinks it
    does. Confirmed against the real simulator, not assumed: the modified BDDL
    compiles, renders (a visibly new object appears in the agentview frame,
    ~50 mean-abs-pixel-value difference against the clean twin), and every
    pre-existing object's `obj_of_interest`/goal check is untouched.

    **The pairing problem, and how it is solved.** LIBERO's `.pruned_init`
    files are flattened MuJoCo states (`time + qpos + qvel`) sized for the
    *original* model. Adding an object with its own free joint changes that
    layout, so the clean corpus's init-state rows cannot be reused unmodified
    against the augmented model -- confirmed empirically (a 92-float
    libero_spatial row becomes 105 once the model gains one 7-qpos/6-qvel free
    joint). Rather than fall back to running this family unpaired (the
    `absent_target` precedent), the augmented model's own joint table is read
    back once (`_probe_distractor_model`, a throwaway env built and closed at
    attach time) to find exactly where MuJoCo placed the new free joint --
    empirically, right after the last object declared under `:objects` and
    before any of a fixture's own articulated joints (a cabinet's drawers, a
    stove's knob) -- and every row of the original init-state array is spliced
    at that address to insert the distractor's own qpos/qvel. Verified
    bit-identical against the real simulator for the robot and every
    pre-existing object on a real init-state row (not just the cold-reset
    default), so `distractor` is a **paired** family like every other one in
    this module: episode *j* still gets the same pre-existing scene as its
    clean twin, with one extra object added.

    **What the strength knob does, and what it honestly does not.** The
    distractor's category, table region, and rest orientation are fixed --
    only its exact xy/yaw within that region varies, jittered per episode
    (via `episode_rng(row_index, "distractor")`, baked into the spliced array
    once at attach time so no further RNG draw happens at reset) with a sigma
    that scales with strength, mirroring `ObjectPose`'s jitter shape. This
    does not scale *how* distracting the object is (e.g. its distance to the
    workspace or the target object) -- picking a second, closer anchor point
    that is verified collision-free across all ten libero_spatial tasks was
    out of the timebox for this spike. Calibration should treat this
    knowingly: strength here mostly controls placement variety, not
    distraction intensity, which is the honest limitation to check before
    this family goes into the calibration sweep.

    **Region and object choice.** `DISTRACTOR_REGION_RANGE` and
    `DISTRACTOR_CATEGORY` are fixed constants, not sampled -- checked directly
    against all ten libero_spatial task bddl files (they share one identical
    region layout and object set), not assumed to generalize. A task outside
    that suite would need the same check repeated before reuse.

    **Unpaired fallback.** If `init_states` is disabled for a run (not the
    corpus recording default), there is no init-state array to splice; the
    family still swaps the BDDL and lets the augmented model's own placement
    sampler place every object, including the pre-existing ones, on every
    reset. That is an explicitly unpaired mode, recorded in `params()` as
    `"paired": False`, analogous to `absent_target`'s quarantine -- it should
    not be compared episode-for-episode against a clean run.
    """

    name = "distractor"

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)
        self._qpos_adr: int | None = None
        self._applied: dict[str, Any] = {"paired": None}

    def prepare_env(self, env: Any) -> None:
        """Swap in an augmented BDDL and a matching init-state array, before
        `env` (a LeRobot `LiberoEnv`) ever builds its simulator.

        Called once, from `PerturbedEnv.__init__`. Raises if the simulator
        already exists -- that would mean this ran too late to matter, which
        should fail loudly rather than silently record a clean shard under
        `distractor`'s name.
        """
        swap_task_files(env, self._augment)

    def _augment(self, bddl_path: str, init_states: Any) -> tuple[str, Any]:
        modified_text = _inject_distractor(Path(bddl_path).read_text())
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- path outlives this function
            mode="w", suffix=".bddl", prefix="backstop_distractor_", delete=False
        )
        handle.write(modified_text)
        handle.close()

        nq_base, nv_base, qpos_adr, dof_adr, default_qpos = _probe_distractor_model(handle.name)
        self._qpos_adr = qpos_adr

        if init_states is None:
            # No init-state file to splice into: every reset falls back to the
            # augmented model's own placement sampler for every object, not
            # just the distractor. See the class docstring's unpaired fallback.
            self._applied["paired"] = False
            return handle.name, None

        original_states = np.asarray(init_states, dtype=np.float64)
        expected_len = 1 + nq_base + nv_base
        if original_states.shape[1] != expected_len:
            raise RuntimeError(
                f"init-state row length {original_states.shape[1]} != expected "
                f"{expected_len} (1 + nq_base {nq_base} + nv_base {nv_base}); "
                "the base model's joint layout does not match this init file."
            )
        spliced = np.stack(
            [
                _splice_distractor(
                    row,
                    nq_base=nq_base,
                    qpos_adr=qpos_adr,
                    dof_adr=dof_adr,
                    qpos=_jitter_distractor_qpos(
                        default_qpos, self.strength, episode_rng(index, self.name)
                    ),
                    qvel=np.zeros(6, dtype=np.float64),
                )
                for index, row in enumerate(original_states)
            ]
        )
        self._applied["paired"] = True
        return handle.name, spliced

    def on_reset(self, control_env: Any, rng: np.random.Generator) -> bool:
        """Nothing left to mutate: the distractor's placement for this episode
        was already baked into the init-state row `LiberoEnv.reset()` just
        applied, before this hook ever runs. Reads back what actually landed,
        for the label sidecar."""
        if self._qpos_adr is not None:
            sim = control_env.env.sim
            qpos = sim.data.qpos[self._qpos_adr : self._qpos_adr + 7]
            self._applied["qpos"] = np.asarray(qpos, dtype=np.float64).tolist()
        return False

    def params(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "category": DISTRACTOR_CATEGORY,
            "region": DISTRACTOR_REGION_RANGE,
            **self._applied,
        }


def _inject_distractor(text: str) -> str:
    """Text-splice one cosmetic object into a libero_spatial BDDL problem: a new
    region far from every declared one, an `:objects` entry, and an `:init`
    placement -- deliberately never added to `:obj_of_interest`, so it can
    never enter the goal check.

    Paren-depth matching, not fixed-offset string replacement: an offset
    computed from one file's exact whitespace would break the moment another
    task's bddl differs by a blank line. All ten libero_spatial task files
    share one identical `:regions`/`:objects` block (diffed directly), so one
    fixed region box and one fixed object category are injected the same way
    into any of them.
    """
    region_block = (
        f"      ({DISTRACTOR_REGION_NAME}\n"
        "          (:target main_table)\n"
        "          (:ranges (\n"
        f"              ({' '.join(str(v) for v in DISTRACTOR_REGION_RANGE)})\n"
        "            )\n"
        "          )\n"
        "      )\n"
    )
    text = _insert_before_section_close(text, "regions", region_block)
    text = _insert_before_section_close(
        text, "objects", f"    {DISTRACTOR_OBJECT_NAME} - {DISTRACTOR_CATEGORY}\n"
    )
    text = _insert_before_section_close(
        text,
        "init",
        f"    (On {DISTRACTOR_OBJECT_NAME} main_table_{DISTRACTOR_REGION_NAME})\n",
    )
    return text


def _insert_before_section_close(text: str, keyword: str, insertion: str) -> str:
    """Find `(:keyword ...)`'s own matching close paren by depth-counting, and
    splice `insertion` in immediately before it -- i.e. as the section's last
    entry. Whitespace-agnostic on purpose; BDDL is parenthesized like Lisp, and
    matching the exact indentation LIBERO happens to use would be a second,
    unnecessary way for this to be fragile."""
    marker = f"(:{keyword}"
    start = text.index(marker)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[:index] + insertion + text[index:]
    raise ValueError(f"Unbalanced parens in BDDL section {keyword!r}")


def _probe_distractor_model(bddl_path: str) -> tuple[int, int, int, int, np.ndarray]:
    """Build the augmented model once, off the hot path, to read back the
    joint layout MuJoCo actually produced -- not assumed. Closed before
    returning; `LiberoEnv._ensure_env()` builds the real one moments later.

    Returns `(nq_base, nv_base, qpos_adr, dof_adr, default_qpos)`:
    `nq_base`/`nv_base` are the *original* (pre-distractor) model's qpos/qvel
    counts, recovered by subtracting the new free joint's known size (7 qpos,
    6 qvel) rather than building a second, un-augmented model just to learn
    them. `default_qpos` is wherever robosuite's own placement sampler put the
    distractor on a cold reset -- a placement already validated by robosuite's
    in-region sampling -- used as the per-episode jitter anchor.
    """
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        hard_reset=True,
    )
    try:
        env.reset()
        sim = env.env.sim
        joint_name = f"{DISTRACTOR_OBJECT_NAME}_joint0"
        joint_id = sim.model.joint_name2id(joint_name)
        free_joint_type = 0  # mujoco.mjtJoint.mjJNT_FREE
        if int(sim.model.jnt_type[joint_id]) != free_joint_type:
            raise RuntimeError(
                f"{joint_name} is not a free joint (type {sim.model.jnt_type[joint_id]}); "
                "the qpos/qvel splice below assumes a 7-qpos/6-qvel free joint."
            )
        qpos_adr = int(sim.model.jnt_qposadr[joint_id])
        dof_adr = int(sim.model.jnt_dofadr[joint_id])
        default_qpos = np.array(sim.data.qpos[qpos_adr : qpos_adr + 7], dtype=np.float64)
        nq_base = int(sim.model.nq) - 7
        nv_base = int(sim.model.nv) - 6
        return nq_base, nv_base, qpos_adr, dof_adr, default_qpos
    finally:
        env.close()


def _splice_distractor(
    row: np.ndarray,
    *,
    nq_base: int,
    qpos_adr: int,
    dof_adr: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
    """Insert the distractor's 7 qpos and 6 qvel values into a flattened
    `time + qpos + qvel` MuJoCo state sized for the model *without* it.

    `qpos_adr`/`dof_adr` are addresses in the *augmented* model, but they are
    valid splice points for the base row too: everything before the
    distractor's own free joint is untouched by adding it (confirmed
    bit-identical against the real simulator), so the base row's qpos/qvel
    blocks share the same prefix layout up to that address.
    """
    time = row[0:1]
    qpos_base = row[1 : 1 + nq_base]
    qvel_base = row[1 + nq_base :]
    new_qpos = np.concatenate([qpos_base[:qpos_adr], qpos, qpos_base[qpos_adr:]])
    new_qvel = np.concatenate([qvel_base[:dof_adr], qvel, qvel_base[dof_adr:]])
    return np.concatenate([time, new_qpos, new_qvel])


def _jitter_distractor_qpos(
    default_qpos: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """Jitter the distractor's xy and yaw around the sampler-given anchor,
    scaled by strength -- the same shape as `ObjectPose`'s jitter, applied to
    an object nothing else depends on."""
    xy_sigma = DISTRACTOR_XY_JITTER_SIGMA_M * strength
    yaw_sigma = DISTRACTOR_YAW_JITTER_RAD * strength
    qpos = default_qpos.copy()
    shift = rng.normal(0.0, xy_sigma, size=2) if xy_sigma > 0 else np.zeros(2)
    qpos[0:2] = qpos[0:2] + shift
    yaw = float(rng.normal(0.0, yaw_sigma)) if yaw_sigma > 0 else 0.0
    qpos[3:7] = _rotate_quat_z(qpos[3:7], yaw)
    return qpos


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


def _settle(control_env: Any) -> None:
    """One LIBERO no-op settle step.

    Broken out to its own module-level function so CPU unit tests can
    monkeypatch it without `lerobot` installed -- this codebase's contract tests
    fake the simulator (see test_perturb.py's module docstring); the claim that
    the real settle path actually resolves objects to rest is checked against
    the real simulator separately, not with a fake standing in for MuJoCo
    contact dynamics.
    """
    from lerobot.envs.libero import get_libero_dummy_action  # noqa: PLC0415

    control_env.step(get_libero_dummy_action())


def _table_bounds(control_env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Table centre (xy) and half-extent (xy), read from the scene's own arena
    rather than guessed -- `table_full_size`/`table_offset` are what robosuite's
    arena was actually built from, not a proxy for it."""
    arena = control_env.env.model.mujoco_arena
    offset = np.array(arena.table_offset, dtype=np.float64)[:2]
    half_xy = np.array(arena.table_full_size, dtype=np.float64)[:2] / 2.0
    return offset, half_xy


def _object_is_settled(
    sim: Any,
    *,
    body_id: int,
    qpos_address: int,
    dof_address: int,
    base_z: float,
    table_offset: np.ndarray,
    table_half_xy: np.ndarray,
) -> bool:
    """After the settle steps: on the table, resting, not interpenetrating.

    Three independent checks, any of which failing means "resample," not
    "clip and continue" -- clipping would put an object at a table edge that
    was never itself a placement the settle steps validated.
    """
    xy = sim.data.qpos[qpos_address : qpos_address + 2]
    if np.any(np.abs(xy - table_offset) > table_half_xy):
        return False

    z = sim.data.qpos[qpos_address + 2]
    if abs(z - base_z) > OBJECT_POSE_Z_TOLERANCE_M:
        return False

    linear_vel = sim.data.qvel[dof_address : dof_address + 3]
    angular_vel = sim.data.qvel[dof_address + 3 : dof_address + 6]
    if np.linalg.norm(linear_vel) > OBJECT_POSE_VEL_TOLERANCE:
        return False
    if np.linalg.norm(angular_vel) > OBJECT_POSE_VEL_TOLERANCE:
        return False

    geom_bodyid = sim.model.geom_bodyid
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        if contact.dist >= -OBJECT_POSE_PENETRATION_TOL_M:
            continue
        if geom_bodyid[contact.geom1] == body_id or geom_bodyid[contact.geom2] == body_id:
            return False

    return True


def _quat_multiply(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions. MuJoCo order."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """(w, x, y, z) quaternion for a rotation of `angle` radians about a unit `axis`."""
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * np.asarray(axis))])


def _rotate_quat_z(quat: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a (w, x, y, z) quaternion about the world z axis. MuJoCo order."""
    return _quat_multiply(_axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), yaw), quat)
