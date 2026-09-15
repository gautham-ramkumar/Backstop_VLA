"""Contract for the perturbation families.

Week 2's whole deliverable is a corpus of failures that differ from each other.
The failures that would quietly ruin it are not crashes: a perturbation that
silently does nothing records a clean shard under a perturbed name, and one whose
randomness is not reproducible makes a perturbed episode incomparable to its clean
twin. Both produce a corpus that looks complete.

CPU only. The simulator is faked; the families are exercised against the real one
separately, because a fake cannot tell you whether an object ended up inside the
table.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

import backstop.perturb.families as families  # noqa: E402
from backstop.perturb import (  # noqa: E402
    AbsentTarget,
    ActionNoise,
    Camera,
    Distractor,
    Lighting,
    NoPerturbation,
    ObjectPose,
    Paraphrase,
    episode_rng,
    make_perturbation,
)
from backstop.perturb.families import _object_is_settled, _table_bounds  # noqa: E402

# A minimal but structurally real BDDL problem -- balanced parens, one region,
# one object, one init predicate, one obj_of_interest -- enough to exercise
# `_inject_distractor`'s paren-depth splicing without pulling in a real task
# file (this module stays CPU-only; the real files are checked in the spike's
# scratch scripts, not here).
FAKE_BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language pick up the bowl)
  (:regions
      (plate_region
          (:target main_table)
          (:ranges (
              (0.05 0.19 0.07 0.21)
            )
          )
      )
    )

  (:fixtures
    main_table - table
  )

  (:objects
    plate_1 - plate
  )

  (:obj_of_interest
    plate_1
  )

  (:init
    (On plate_1 main_table_plate_region)
  )

  (:goal
    (And (On plate_1 plate_1))
  )

)
"""

GRIPPER = -1


class FakeSim:
    """Enough MuJoCo surface for the qpos-writing families, including the
    contact/velocity/arena surface `object_pose`'s validity check reads."""

    def __init__(self, n_objects: int = 2) -> None:
        self.nq = 7 * n_objects
        self.data = SimpleNamespace(
            qpos=np.zeros(self.nq),
            qvel=np.zeros(6 * n_objects),
            ncon=0,
            contact=[],
        )
        # Identity quaternion (w=1) for every free joint, as MuJoCo stores them.
        for index in range(n_objects):
            self.data.qpos[index * 7 + 3] = 1.0
        self.model = SimpleNamespace(
            joint_name2id=lambda name: int(name.split("_")[0][-1]),
            jnt_qposadr=[index * 7 for index in range(n_objects)],
            jnt_dofadr=[index * 6 for index in range(n_objects)],
            jnt_bodyid=[index + 100 for index in range(n_objects)],
            geom_bodyid=[],
            # Real MuJoCo defaults (measured on the real simulator), so a test
            # that forgets to scale this notices against a realistic baseline
            # rather than an already-zeroed one.
            vis=SimpleNamespace(
                headlight=SimpleNamespace(
                    ambient=np.array([0.1, 0.1, 0.1]),
                    diffuse=np.array([0.4, 0.4, 0.4]),
                    specular=np.array([0.5, 0.5, 0.5]),
                )
            ),
        )
        self.forwarded = 0

    def forward(self) -> None:
        self.forwarded += 1


def _control_env(n_objects: int = 2) -> Any:
    sim = FakeSim(n_objects)
    # Objects start at xy=(0, 0), well inside these bounds -- table_full_size and
    # table_offset are the real robosuite arena field names and shapes (see
    # `_table_bounds`), not stand-ins for them.
    arena = SimpleNamespace(table_offset=(0.0, 0.0, 0.9), table_full_size=(1.0, 1.2, 0.05))
    env_model = SimpleNamespace(mujoco_arena=arena)
    inner = SimpleNamespace(
        sim=sim, obj_of_interest=[f"obj{i}" for i in range(n_objects)], model=env_model
    )
    return SimpleNamespace(env=inner)


@pytest.fixture
def no_op_settle(monkeypatch):
    """`object_pose` settles itself via `families._settle`, which imports real
    `lerobot` -- monkeypatched to a no-op so these stay CPU-only fakes. Real
    settling is checked against the real simulator, not reproduced here."""
    monkeypatch.setattr(families, "_settle", lambda control_env: None)


# --- reproducibility -------------------------------------------------------


def test_episode_rng_is_stable_across_processes() -> None:
    """Guards the `hash()` trap directly.

    Python salts string hashing per process unless PYTHONHASHSEED is fixed, so a
    seed derived from `hash(name)` gives a different perturbation on every run.
    The shard would still record, still look perturbed, and never reproduce.
    """
    code = (
        "from backstop.perturb import episode_rng; "
        "print(episode_rng(7, 'lighting').integers(0, 2**31))"
    )
    outputs = set()
    for hashseed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"episode_rng varies with PYTHONHASHSEED: {outputs}"


def test_different_episodes_get_different_draws() -> None:
    """A shard whose 50 episodes share one scene is 50 copies of one sample."""
    draws = {episode_rng(seed, "object_pose").normal() for seed in range(10)}
    assert len(draws) == 10


def test_different_families_do_not_share_a_stream() -> None:
    """Lighting and pose must not be correlated across a matched pair."""
    assert episode_rng(3, "lighting").normal() != episode_rng(3, "object_pose").normal()


# --- action noise ----------------------------------------------------------


def test_action_noise_leaves_the_gripper_alone() -> None:
    """The gripper dimension is effectively binary. Noising it produces random
    open/close chatter, which fails episodes for a reason unrelated to control."""
    action = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0]], dtype=np.float32)
    out = ActionNoise(1.0).perturb_action(action, np.random.default_rng(0))

    assert out[0, GRIPPER] == action[0, GRIPPER]
    assert not np.allclose(out[0, :GRIPPER], action[0, :GRIPPER])


def test_action_noise_stays_inside_the_action_range() -> None:
    """LIBERO's action space is [-1, 1]; an out-of-range command is clipped by the
    controller anyway, so leaving it unclipped would misreport what was executed."""
    action = np.full((1, 7), 0.95, dtype=np.float32)
    out = ActionNoise(1.0).perturb_action(action, np.random.default_rng(1))

    assert out.max() <= 1.0
    assert out.min() >= -1.0


def test_action_noise_scales_with_strength() -> None:
    action = np.zeros((1, 7), dtype=np.float32)
    weak = ActionNoise(0.1).perturb_action(action, np.random.default_rng(2))
    strong = ActionNoise(1.0).perturb_action(action, np.random.default_rng(2))

    assert np.abs(strong[0, :GRIPPER]).mean() > np.abs(weak[0, :GRIPPER]).mean()


def test_zero_strength_is_a_no_op() -> None:
    """The calibration sweep evaluates strength 0; it must equal the clean arm."""
    action = np.array([[0.3, -0.4, 0.5, 0.1, 0.0, -0.2, 1.0]], dtype=np.float32)
    out = ActionNoise(0.0).perturb_action(action, np.random.default_rng(3))

    np.testing.assert_array_equal(out, action)


def test_action_noise_preserves_dtype() -> None:
    """The env and the dataset both expect float32; float64 would silently widen."""
    action = np.zeros((1, 7), dtype=np.float32)
    assert ActionNoise(0.5).perturb_action(action, np.random.default_rng(4)).dtype == np.float32


# --- object pose -----------------------------------------------------------


def test_object_pose_moves_objects_and_settles_itself(no_op_settle) -> None:
    """`object_pose` settles and validates inside `on_reset` now, so it returns
    False -- asking the wrapper to resettle on top would just repeat work
    already done."""
    control_env = _control_env()
    before = control_env.env.sim.data.qpos.copy()

    changed = ObjectPose(1.0).on_reset(control_env, np.random.default_rng(0))

    assert changed is False, "already settled and validated itself"
    assert not np.allclose(control_env.env.sim.data.qpos[:2], before[:2])
    assert control_env.env.sim.forwarded == 1, "sim.forward() propagates the write"


def test_object_pose_keeps_quaternions_normalised(no_op_settle) -> None:
    """MuJoCo does not renormalise for you; a drifting quaternion skews the scene
    a little more with every episode and never raises."""
    control_env = _control_env()
    ObjectPose(1.0).on_reset(control_env, np.random.default_rng(5))

    for start in (0, 7):
        quat = control_env.env.sim.data.qpos[start + 3 : start + 7]
        assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-9)


def test_object_pose_only_touches_z_rotation(no_op_settle) -> None:
    """Objects sit on a table. Tipping them over is a different perturbation, and
    one that would fail for reasons the taxonomy should not conflate."""
    control_env = _control_env(n_objects=1)
    ObjectPose(1.0).on_reset(control_env, np.random.default_rng(6))
    quat = control_env.env.sim.data.qpos[3:7]

    assert np.isclose(quat[1], 0.0, atol=1e-12), "x component -- object tipped"
    assert np.isclose(quat[2], 0.0, atol=1e-12), "y component -- object tipped"


def test_object_pose_records_what_it_did(no_op_settle) -> None:
    """The label sidecar needs the actual offsets, not just the strength."""
    control_env = _control_env()
    pose = ObjectPose(0.7)
    pose.on_reset(control_env, np.random.default_rng(7))
    params = pose.params()

    assert params["strength"] == 0.7
    assert set(params["objects"]) == {"obj0", "obj1"}
    assert len(params["objects"]["obj0"]["xy_shift"]) == 2
    assert params["accepted"] is True
    assert params["attempts"] == 1


def test_object_pose_is_reproducible_for_a_given_episode(no_op_settle) -> None:
    first, second = _control_env(), _control_env()
    ObjectPose(1.0).on_reset(first, episode_rng(11, "object_pose"))
    ObjectPose(1.0).on_reset(second, episode_rng(11, "object_pose"))

    np.testing.assert_array_equal(first.env.sim.data.qpos, second.env.sim.data.qpos)


# --- object pose: validity check --------------------------------------------


def test_is_settled_accepts_a_resting_object_within_bounds() -> None:
    sim = FakeSim(n_objects=1)
    assert (
        _object_is_settled(
            sim,
            body_id=100,
            qpos_address=0,
            dof_address=0,
            base_z=0.0,
            table_offset=np.array([0.0, 0.0]),
            table_half_xy=np.array([0.5, 0.6]),
        )
        is True
    )


def test_is_settled_rejects_outside_table_bounds() -> None:
    sim = FakeSim(n_objects=1)
    sim.data.qpos[0:2] = [10.0, 10.0]
    assert (
        _object_is_settled(
            sim,
            body_id=100,
            qpos_address=0,
            dof_address=0,
            base_z=0.0,
            table_offset=np.array([0.0, 0.0]),
            table_half_xy=np.array([0.5, 0.6]),
        )
        is False
    )


def test_is_settled_rejects_a_height_change() -> None:
    """A resting object should stay near the height it started at; a big change
    means it fell off an edge or through the table, not that it came to rest."""
    sim = FakeSim(n_objects=1)
    sim.data.qpos[2] = 0.5
    assert (
        _object_is_settled(
            sim,
            body_id=100,
            qpos_address=0,
            dof_address=0,
            base_z=0.0,
            table_offset=np.array([0.0, 0.0]),
            table_half_xy=np.array([0.5, 0.6]),
        )
        is False
    )


def test_is_settled_rejects_nonzero_velocity() -> None:
    sim = FakeSim(n_objects=1)
    sim.data.qvel[0] = 1.0
    assert (
        _object_is_settled(
            sim,
            body_id=100,
            qpos_address=0,
            dof_address=0,
            base_z=0.0,
            table_offset=np.array([0.0, 0.0]),
            table_half_xy=np.array([0.5, 0.6]),
        )
        is False
    )


def test_is_settled_rejects_interpenetration_of_that_body() -> None:
    sim = FakeSim(n_objects=1)
    sim.model.geom_bodyid = [100, 200]
    sim.data.ncon = 1
    sim.data.contact = [SimpleNamespace(dist=-0.01, geom1=0, geom2=1)]
    assert (
        _object_is_settled(
            sim,
            body_id=100,
            qpos_address=0,
            dof_address=0,
            base_z=0.0,
            table_offset=np.array([0.0, 0.0]),
            table_half_xy=np.array([0.5, 0.6]),
        )
        is False
    )


def test_is_settled_ignores_shallow_contact_as_numerical_noise() -> None:
    """Real resting contact reports a tiny negative `dist` (measured ~-4e-6 on the
    real simulator); treating that as interpenetration would reject every
    ordinary resting placement."""
    sim = FakeSim(n_objects=1)
    sim.model.geom_bodyid = [100, 200]
    sim.data.ncon = 1
    sim.data.contact = [SimpleNamespace(dist=-1e-6, geom1=0, geom2=1)]
    assert (
        _object_is_settled(
            sim,
            body_id=100,
            qpos_address=0,
            dof_address=0,
            base_z=0.0,
            table_offset=np.array([0.0, 0.0]),
            table_half_xy=np.array([0.5, 0.6]),
        )
        is True
    )


def test_is_settled_ignores_contacts_not_involving_this_body() -> None:
    sim = FakeSim(n_objects=2)
    sim.model.geom_bodyid = [200, 300]
    sim.data.ncon = 1
    sim.data.contact = [SimpleNamespace(dist=-0.01, geom1=0, geom2=1)]
    assert (
        _object_is_settled(
            sim,
            body_id=100,
            qpos_address=0,
            dof_address=0,
            base_z=0.0,
            table_offset=np.array([0.0, 0.0]),
            table_half_xy=np.array([0.5, 0.6]),
        )
        is True
    )


def test_table_bounds_reads_the_real_arena_field_names() -> None:
    control_env = _control_env()
    offset, half_xy = _table_bounds(control_env)

    np.testing.assert_array_equal(offset, [0.0, 0.0])
    np.testing.assert_array_equal(half_xy, [0.5, 0.6])


# --- object pose: reject-and-resample ---------------------------------------


def test_object_pose_retries_until_a_valid_placement_and_does_not_count_a_rejection(
    monkeypatch, no_op_settle
) -> None:
    calls = {"n": 0}

    def flaky_valid(*_args, **_kwargs) -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    monkeypatch.setattr(families, "_object_is_settled", flaky_valid)
    control_env = _control_env(n_objects=1)
    pose = ObjectPose(1.0)

    changed = pose.on_reset(control_env, np.random.default_rng(0))

    assert changed is False
    assert pose.last_attempts == 3
    assert pose.last_accepted is True
    assert pose.n_rejections == 0
    assert pose.n_attempts == 3


def test_object_pose_gives_up_after_max_attempts_and_counts_one_rejection(
    monkeypatch, no_op_settle
) -> None:
    monkeypatch.setattr(families, "_object_is_settled", lambda *a, **k: False)
    control_env = _control_env(n_objects=1)
    pose = ObjectPose(1.0)

    pose.on_reset(control_env, np.random.default_rng(0))

    assert pose.last_attempts == families.OBJECT_POSE_MAX_ATTEMPTS
    assert pose.last_accepted is False
    assert pose.n_rejections == 1
    assert pose.params()["objects"]["obj0"]["accepted"] is False


def test_object_pose_rejection_rate_accumulates_across_episodes(monkeypatch, no_op_settle) -> None:
    """One `ObjectPose` instance records a whole shard's episodes -- the
    rejection *rate* the plan asks to be logged is only meaningful as a running
    total, not reset per reset()."""
    monkeypatch.setattr(families, "_object_is_settled", lambda *a, **k: False)
    pose = ObjectPose(1.0)

    pose.on_reset(_control_env(n_objects=1), np.random.default_rng(0))
    pose.on_reset(_control_env(n_objects=1), np.random.default_rng(1))

    assert pose.n_rejections == 2
    assert pose.n_attempts == 2 * families.OBJECT_POSE_MAX_ATTEMPTS


# --- lighting --------------------------------------------------------------


class FakeModder:
    light_names = ["light0", "light1"]

    def __init__(self, sim: Any) -> None:
        self.diffuse = {name: np.array([1.0, 1.0, 1.0]) for name in self.light_names}
        self.pos = {name: np.array([0.0, 0.0, 2.0]) for name in self.light_names}

    def get_diffuse(self, name: str) -> Any:
        return self.diffuse[name]

    def set_diffuse(self, name: str, value: Any) -> None:
        self.diffuse[name] = np.asarray(value)

    def get_pos(self, name: str) -> Any:
        return self.pos[name]

    def set_pos(self, name: str, value: Any) -> None:
        self.pos[name] = np.asarray(value)


@pytest.fixture
def fake_modder(monkeypatch):
    import robosuite.utils.mjmod as mjmod

    monkeypatch.setattr(mjmod, "LightingModder", FakeModder)
    return FakeModder


def test_lighting_dims_every_light_and_does_not_resettle(fake_modder) -> None:
    """Purely visual: settling would cost simulation time and change nothing."""
    control_env = _control_env()
    lighting = Lighting(1.0)

    changed = lighting.on_reset(control_env, np.random.default_rng(0))

    assert changed is False
    applied = lighting.params()["lights"]
    assert set(applied) == {"light0", "light1"}
    assert applied["light0"]["diffuse_scale"] == pytest.approx(0.2)


def test_lighting_at_zero_strength_changes_nothing(fake_modder) -> None:
    control_env = _control_env()
    lighting = Lighting(0.0)
    lighting.on_reset(control_env, np.random.default_rng(0))

    applied = lighting.params()["lights"]
    assert applied["light0"]["diffuse_scale"] == pytest.approx(1.0)
    assert applied["light0"]["pos_shift"] == [0.0, 0.0, 0.0]


def test_lighting_also_scales_the_camera_headlight(fake_modder) -> None:
    """Scaling only the two named scene lights barely changes the rendered
    image -- MuJoCo's camera headlight is a separate, always-on light source
    that `LightingModder` does not manage. Confirmed on the real simulator:
    scene lights alone took mean pixel value 120 -> 87; adding the headlight
    took it to 11."""
    control_env = _control_env()
    headlight = control_env.env.sim.model.vis.headlight
    before = {
        "ambient": headlight.ambient.copy(),
        "diffuse": headlight.diffuse.copy(),
        "specular": headlight.specular.copy(),
    }

    lighting = Lighting(1.0)
    lighting.on_reset(control_env, np.random.default_rng(0))

    for channel, base in before.items():
        np.testing.assert_allclose(getattr(headlight, channel), base * 0.2)
    assert lighting.params()["headlight"]["scale"] == pytest.approx(0.2)
    assert lighting.params()["headlight"]["diffuse"] == before["diffuse"].tolist()


def test_lighting_headlight_unchanged_at_zero_strength(fake_modder) -> None:
    control_env = _control_env()
    headlight = control_env.env.sim.model.vis.headlight
    before_diffuse = headlight.diffuse.copy()

    Lighting(0.0).on_reset(control_env, np.random.default_rng(0))

    np.testing.assert_allclose(headlight.diffuse, before_diffuse)


# --- camera ------------------------------------------------------------------


class FakeCameraModder:
    def __init__(self, sim: Any, camera_names: list[str] | None = None) -> None:
        self.camera_names = camera_names or ["agentview"]
        self.pos = {name: np.array([1.0, 0.0, 1.5]) for name in self.camera_names}
        self.quat = {name: np.array([1.0, 0.0, 0.0, 0.0]) for name in self.camera_names}
        self.fovy = dict.fromkeys(self.camera_names, 45.0)

    def get_pos(self, name: str) -> Any:
        return self.pos[name]

    def set_pos(self, name: str, value: Any) -> None:
        self.pos[name] = np.asarray(value)

    def get_quat(self, name: str) -> Any:
        return self.quat[name]

    def set_quat(self, name: str, value: Any) -> None:
        self.quat[name] = np.asarray(value)

    def get_fovy(self, name: str) -> Any:
        return self.fovy[name]

    def set_fovy(self, name: str, value: Any) -> None:
        self.fovy[name] = value


@pytest.fixture
def fake_camera_modder(monkeypatch):
    import robosuite.utils.mjmod as mjmod

    monkeypatch.setattr(mjmod, "CameraModder", FakeCameraModder)
    return FakeCameraModder


def test_camera_jitters_pose_and_fov_without_resettling(fake_camera_modder) -> None:
    """Purely visual, like lighting: the scene is untouched, only the sensor moved."""
    control_env = _control_env()
    camera = Camera(1.0)

    changed = camera.on_reset(control_env, np.random.default_rng(0))

    assert changed is False
    params = camera.params()
    assert params["camera"] == "agentview"
    assert any(abs(shift) > 0 for shift in params["pos_shift"])
    assert params["rot_angle_rad"] != 0.0
    assert params["fovy_shift_deg"] != 0.0


def test_camera_at_zero_strength_changes_nothing(fake_camera_modder) -> None:
    control_env = _control_env()
    camera = Camera(0.0)
    camera.on_reset(control_env, np.random.default_rng(0))

    params = camera.params()
    assert params["pos_shift"] == [0.0, 0.0, 0.0]
    assert params["rot_angle_rad"] == 0.0
    assert params["fovy_shift_deg"] == 0.0


def test_camera_keeps_the_jittered_quaternion_normalised(fake_camera_modder, monkeypatch) -> None:
    """A drifting camera quaternion would slowly skew where the corpus thinks the
    scene is, and MuJoCo does not renormalise for you."""
    instances: list[FakeCameraModder] = []
    original_init = FakeCameraModder.__init__

    def capturing_init(self, sim, camera_names=None):
        original_init(self, sim, camera_names=camera_names)
        instances.append(self)

    monkeypatch.setattr(FakeCameraModder, "__init__", capturing_init)

    Camera(1.0).on_reset(_control_env(), np.random.default_rng(1))

    assert np.isclose(np.linalg.norm(instances[0].quat["agentview"]), 1.0, atol=1e-9)


def test_camera_is_reproducible_for_a_given_episode(fake_camera_modder) -> None:
    first, second = Camera(1.0), Camera(1.0)
    first.on_reset(_control_env(), episode_rng(11, "camera"))
    second.on_reset(_control_env(), episode_rng(11, "camera"))

    assert first.params() == second.params()


def test_camera_never_touches_the_wrist_camera(fake_camera_modder, monkeypatch) -> None:
    """The wrist camera is rigidly mounted; jittering it would be jittering the
    robot's own proprioception, not an external sensor."""
    captured: dict[str, Any] = {}
    original_init = FakeCameraModder.__init__

    def spy_init(self, sim, camera_names=None):
        captured["camera_names"] = camera_names
        original_init(self, sim, camera_names=camera_names)

    monkeypatch.setattr(FakeCameraModder, "__init__", spy_init)
    Camera(1.0).on_reset(_control_env(), np.random.default_rng(0))

    assert captured["camera_names"] == ["agentview"]


# --- language ------------------------------------------------------------------


def test_absent_target_swaps_the_instruction_and_leaves_physics_alone() -> None:
    absent = AbsentTarget(replacement="pick up the nonexistent widget")
    action = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]], dtype=np.float32)

    assert absent.on_reset(_control_env(), np.random.default_rng(0)) is False
    np.testing.assert_array_equal(absent.perturb_action(action, np.random.default_rng(0)), action)
    assert absent.task_description("pick up the bowl") == "pick up the nonexistent widget"
    assert absent.params() == {"replacement": "pick up the nonexistent widget"}


def test_paraphrase_swaps_the_instruction_and_leaves_physics_alone() -> None:
    """The control arm: same mechanism as absent_target, so a guard cannot tell
    'the instruction changed' apart from 'the instruction changed AND the named
    object is gone' by any means other than actually reading the two."""
    paraphrase = Paraphrase(paraphrase="grab the bowl and set it on the plate")
    action = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]], dtype=np.float32)

    assert paraphrase.on_reset(_control_env(), np.random.default_rng(0)) is False
    np.testing.assert_array_equal(
        paraphrase.perturb_action(action, np.random.default_rng(0)), action
    )
    assert (
        paraphrase.task_description("pick up the bowl") == "grab the bowl and set it on the plate"
    )
    assert paraphrase.params() == {"paraphrase": "grab the bowl and set it on the plate"}


def test_paraphrase_is_selectable_through_the_factory() -> None:
    perturbation = make_perturbation("language_paraphrase", paraphrase="lift the mug")
    assert isinstance(perturbation, Paraphrase)
    assert perturbation.task_description("original") == "lift the mug"


# --- distractor --------------------------------------------------------------


def _section_text(text: str, keyword: str) -> str:
    """Test-only mirror of `_insert_before_section_close`'s paren matching, used
    to inspect a section's contents without depending on exact whitespace."""
    marker = f"(:{keyword}"
    start = text.index(marker)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index]
    raise AssertionError(f"unbalanced parens in {keyword!r}")


def test_inject_distractor_adds_region_object_and_init_predicate() -> None:
    modified = families._inject_distractor(FAKE_BDDL)

    assert f"{families.DISTRACTOR_OBJECT_NAME} - {families.DISTRACTOR_CATEGORY}" in _section_text(
        modified, "objects"
    )
    assert families.DISTRACTOR_REGION_NAME in _section_text(modified, "regions")
    assert (
        f"(On {families.DISTRACTOR_OBJECT_NAME} main_table_{families.DISTRACTOR_REGION_NAME})"
        in _section_text(modified, "init")
    )


def test_inject_distractor_never_touches_obj_of_interest() -> None:
    """The whole point of a cosmetic distractor: it must never enter the goal
    check, so it must never appear in `:obj_of_interest`."""
    modified = families._inject_distractor(FAKE_BDDL)

    assert families.DISTRACTOR_OBJECT_NAME not in _section_text(modified, "obj_of_interest")


def test_inject_distractor_leaves_existing_content_intact() -> None:
    modified = families._inject_distractor(FAKE_BDDL)

    assert "plate_1 - plate" in _section_text(modified, "objects")
    assert "plate_region" in _section_text(modified, "regions")
    assert "(On plate_1 main_table_plate_region)" in _section_text(modified, "init")
    assert "plate_1" in _section_text(modified, "obj_of_interest")


def test_inject_distractor_is_deterministic() -> None:
    assert families._inject_distractor(FAKE_BDDL) == families._inject_distractor(FAKE_BDDL)


def test_insert_before_section_close_raises_on_unbalanced_parens() -> None:
    with pytest.raises(ValueError, match="Unbalanced"):
        families._insert_before_section_close("(:regions (foo", "regions", "x")


def test_splice_distractor_inserts_qpos_and_qvel_at_the_given_addresses() -> None:
    # A tiny fabricated state: 1 time value + 2 qpos + 2 qvel (nq_base=2, nv_base=2).
    row = np.array([9.0, 1.0, 2.0, 10.0, 20.0])

    out = families._splice_distractor(
        row,
        nq_base=2,
        qpos_adr=1,
        dof_adr=1,
        qpos=np.array([100.0, 101.0]),
        qvel=np.array([200.0]),
    )

    np.testing.assert_array_equal(out, [9.0, 1.0, 100.0, 101.0, 2.0, 10.0, 200.0, 20.0])


def test_splice_distractor_at_the_start_of_each_block() -> None:
    row = np.array([9.0, 1.0, 2.0, 10.0, 20.0])

    out = families._splice_distractor(
        row, nq_base=2, qpos_adr=0, dof_adr=0, qpos=np.array([100.0]), qvel=np.array([200.0])
    )

    np.testing.assert_array_equal(out, [9.0, 100.0, 1.0, 2.0, 200.0, 10.0, 20.0])


def test_jitter_distractor_qpos_zero_strength_is_a_no_op() -> None:
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])

    out = families._jitter_distractor_qpos(default_qpos, 0.0, np.random.default_rng(0))

    np.testing.assert_array_equal(out, default_qpos)


def test_jitter_distractor_qpos_scales_with_strength() -> None:
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])

    weak = families._jitter_distractor_qpos(default_qpos, 0.1, np.random.default_rng(1))
    strong = families._jitter_distractor_qpos(default_qpos, 1.0, np.random.default_rng(1))

    assert np.abs(strong[:2] - default_qpos[:2]).sum() > np.abs(weak[:2] - default_qpos[:2]).sum()


def test_jitter_distractor_qpos_keeps_quaternion_normalised() -> None:
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])

    out = families._jitter_distractor_qpos(default_qpos, 1.0, np.random.default_rng(2))

    assert np.isclose(np.linalg.norm(out[3:7]), 1.0, atol=1e-9)


def test_jitter_distractor_qpos_only_touches_xy_and_yaw() -> None:
    """Like `object_pose`, the distractor sits flat on the table; z never moves."""
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])

    out = families._jitter_distractor_qpos(default_qpos, 1.0, np.random.default_rng(3))

    assert out[2] == default_qpos[2]


def _fake_probe(nq_base: int, nv_base: int, qpos_adr: int, dof_adr: int, default_qpos: np.ndarray):
    def probe(bddl_path: str):
        return nq_base, nv_base, qpos_adr, dof_adr, default_qpos

    return probe


def test_prepare_env_raises_if_the_simulator_already_exists() -> None:
    """`prepare_env` running after `_ensure_env()` has already built the model
    would silently do nothing useful -- the swapped bddl path would never be
    read. Fail loudly instead of recording a clean shard under `distractor`."""
    inner = SimpleNamespace(_task_bddl_file="/does/not/matter", _init_states=None, _env=object())

    with pytest.raises(RuntimeError, match="before the LIBERO simulator is built"):
        Distractor(0.5).prepare_env(inner)


def test_prepare_env_swaps_the_bddl_path_to_a_new_file(tmp_path, monkeypatch) -> None:
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text(FAKE_BDDL)
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(families, "_probe_distractor_model", _fake_probe(2, 2, 1, 1, default_qpos))
    inner = SimpleNamespace(_task_bddl_file=str(bddl_path), _init_states=None, _env=None)

    Distractor(0.5).prepare_env(inner)

    assert inner._task_bddl_file != str(bddl_path)
    written = Path(inner._task_bddl_file).read_text()
    assert families.DISTRACTOR_OBJECT_NAME in _section_text(written, "objects")
    # The original, installed-package bddl file is untouched.
    assert families.DISTRACTOR_OBJECT_NAME not in bddl_path.read_text()


def test_prepare_env_without_init_states_is_explicitly_unpaired(tmp_path, monkeypatch) -> None:
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text(FAKE_BDDL)
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(families, "_probe_distractor_model", _fake_probe(2, 2, 1, 1, default_qpos))
    inner = SimpleNamespace(_task_bddl_file=str(bddl_path), _init_states=None, _env=None)
    distractor = Distractor(0.5)

    distractor.prepare_env(inner)

    assert distractor.params()["paired"] is False
    assert inner._init_states is None


def test_prepare_env_splices_every_init_state_row(tmp_path, monkeypatch) -> None:
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text(FAKE_BDDL)
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(families, "_probe_distractor_model", _fake_probe(2, 2, 1, 1, default_qpos))
    original_row = np.array([9.0, 1.0, 2.0, 10.0, 20.0])
    original_states = np.stack([original_row, original_row * 2])
    inner = SimpleNamespace(
        _task_bddl_file=str(bddl_path), _init_states=original_states.copy(), _env=None
    )
    # strength 0.0 makes the jitter a deterministic no-op, so every row's
    # distractor block must equal `default_qpos` exactly, regardless of episode.
    distractor = Distractor(0.0)

    distractor.prepare_env(inner)

    assert inner._init_states.shape == (2, 1 + (2 + 7) + (2 + 6))
    expected_row0 = families._splice_distractor(
        original_row, nq_base=2, qpos_adr=1, dof_adr=1, qpos=default_qpos, qvel=np.zeros(6)
    )
    np.testing.assert_array_equal(inner._init_states[0], expected_row0)
    assert distractor.params()["paired"] is True


def test_prepare_env_gives_different_episodes_different_distractor_placements(
    tmp_path, monkeypatch
) -> None:
    """Baked in once at attach time via `episode_rng(row_index, ...)` -- if every
    row got the same draw, a shard's 50 episodes would be 50 copies of one
    distractor placement."""
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text(FAKE_BDDL)
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(families, "_probe_distractor_model", _fake_probe(2, 2, 1, 1, default_qpos))
    original_row = np.array([9.0, 1.0, 2.0, 10.0, 20.0])
    original_states = np.stack([original_row] * 5)
    inner = SimpleNamespace(
        _task_bddl_file=str(bddl_path), _init_states=original_states.copy(), _env=None
    )

    Distractor(1.0).prepare_env(inner)

    distractor_xy = inner._init_states[:, 1 + 1 : 1 + 1 + 2]  # qpos_adr=1, xy = first 2 of 7
    assert len({tuple(row) for row in distractor_xy}) == 5


def test_prepare_env_raises_on_init_state_length_mismatch(tmp_path, monkeypatch) -> None:
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text(FAKE_BDDL)
    default_qpos = np.array([0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(families, "_probe_distractor_model", _fake_probe(2, 2, 1, 1, default_qpos))
    wrong_length_row = np.zeros(6)  # expected 1 + 2 + 2 = 5
    inner = SimpleNamespace(
        _task_bddl_file=str(bddl_path), _init_states=np.stack([wrong_length_row]), _env=None
    )

    with pytest.raises(RuntimeError, match="init-state row length"):
        Distractor(0.5).prepare_env(inner)


def test_distractor_on_reset_reads_back_the_applied_placement_and_does_not_resettle() -> None:
    """The distractor's placement for this episode is already baked into the
    init-state row `LiberoEnv.reset()` applied before `on_reset` ever runs --
    there is nothing left to mutate, only to report."""
    sim = FakeSim(n_objects=1)
    sim.data.qpos[0:7] = [0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0]
    control_env = SimpleNamespace(env=SimpleNamespace(sim=sim))
    distractor = Distractor(0.5)
    distractor._qpos_adr = 0

    changed = distractor.on_reset(control_env, np.random.default_rng(0))

    assert changed is False
    assert distractor.params()["qpos"] == [0.31, -0.2, 0.97, 1.0, 0.0, 0.0, 0.0]


def test_distractor_on_reset_before_prepare_env_does_not_crash() -> None:
    """A unit test exercising `on_reset` directly, without ever calling
    `prepare_env` (which needs the real simulator) -- must degrade, not crash."""
    distractor = Distractor(0.5)

    changed = distractor.on_reset(_control_env(), np.random.default_rng(0))

    assert changed is False
    assert "qpos" not in distractor.params()


def test_distractor_perturb_action_is_a_no_op() -> None:
    action = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]], dtype=np.float32)

    out = Distractor(1.0).perturb_action(action, np.random.default_rng(0))

    np.testing.assert_array_equal(out, action)


def test_distractor_params_reports_strength_category_and_region() -> None:
    params = Distractor(0.4).params()

    assert params["strength"] == 0.4
    assert params["category"] == families.DISTRACTOR_CATEGORY
    assert params["region"] == families.DISTRACTOR_REGION_RANGE
    assert params["paired"] is None  # prepare_env has not run yet


def test_distractor_is_selectable_through_the_factory() -> None:
    perturbation = make_perturbation("distractor", strength=0.6)

    assert isinstance(perturbation, Distractor)
    assert perturbation.strength == 0.6
    assert perturbation.name == "distractor"


# --- factory ---------------------------------------------------------------


def test_unknown_family_raises_rather_than_recording_a_clean_shard() -> None:
    """A typo that no-ops would look like a perturbation with no effect -- a
    finding rather than a mistake, and one nobody would think to check."""
    with pytest.raises(ValueError, match="unknown perturbation"):
        make_perturbation("lightning", 0.5)


def test_strength_outside_the_unit_interval_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_perturbation("lighting", 1.5)


def test_clean_arm_is_the_same_code_path() -> None:
    """The clean shard runs the perturbation machinery too, so a bug in the
    plumbing shows up in the control arm instead of hiding there."""
    clean = make_perturbation("clean")
    action = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]], dtype=np.float32)

    assert isinstance(clean, NoPerturbation)
    assert clean.on_reset(_control_env(), np.random.default_rng(0)) is False
    np.testing.assert_array_equal(clean.perturb_action(action, np.random.default_rng(0)), action)
