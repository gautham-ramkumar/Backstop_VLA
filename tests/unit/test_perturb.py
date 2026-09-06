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
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from backstop.perturb import (  # noqa: E402
    ActionNoise,
    Lighting,
    NoPerturbation,
    ObjectPose,
    episode_rng,
    make_perturbation,
)

GRIPPER = -1


class FakeSim:
    """Enough MuJoCo surface for the qpos-writing families."""

    def __init__(self, n_objects: int = 2) -> None:
        self.nq = 7 * n_objects
        self.data = SimpleNamespace(qpos=np.zeros(self.nq))
        # Identity quaternion (w=1) for every free joint, as MuJoCo stores them.
        for index in range(n_objects):
            self.data.qpos[index * 7 + 3] = 1.0
        self.model = SimpleNamespace(
            joint_name2id=lambda name: int(name.split("_")[0][-1]),
            jnt_qposadr=[index * 7 for index in range(n_objects)],
        )
        self.forwarded = 0

    def forward(self) -> None:
        self.forwarded += 1


def _control_env(n_objects: int = 2) -> Any:
    sim = FakeSim(n_objects)
    inner = SimpleNamespace(sim=sim, obj_of_interest=[f"obj{i}" for i in range(n_objects)])
    return SimpleNamespace(env=inner)


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


def test_object_pose_moves_objects_and_asks_to_resettle() -> None:
    """A moved object may be interpenetrating; returning True is what buys the
    settle steps that resolve it."""
    control_env = _control_env()
    before = control_env.env.sim.data.qpos.copy()

    changed = ObjectPose(1.0).on_reset(control_env, np.random.default_rng(0))

    assert changed is True, "physics changed, so the wrapper must re-settle"
    assert not np.allclose(control_env.env.sim.data.qpos[:2], before[:2])
    assert control_env.env.sim.forwarded == 1, "sim.forward() propagates the write"


def test_object_pose_keeps_quaternions_normalised() -> None:
    """MuJoCo does not renormalise for you; a drifting quaternion skews the scene
    a little more with every episode and never raises."""
    control_env = _control_env()
    ObjectPose(1.0).on_reset(control_env, np.random.default_rng(5))

    for start in (0, 7):
        quat = control_env.env.sim.data.qpos[start + 3 : start + 7]
        assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-9)


def test_object_pose_only_touches_z_rotation() -> None:
    """Objects sit on a table. Tipping them over is a different perturbation, and
    one that would fail for reasons the taxonomy should not conflate."""
    control_env = _control_env(n_objects=1)
    ObjectPose(1.0).on_reset(control_env, np.random.default_rng(6))
    quat = control_env.env.sim.data.qpos[3:7]

    assert np.isclose(quat[1], 0.0, atol=1e-12), "x component -- object tipped"
    assert np.isclose(quat[2], 0.0, atol=1e-12), "y component -- object tipped"


def test_object_pose_records_what_it_did() -> None:
    """The label sidecar needs the actual offsets, not just the strength."""
    control_env = _control_env()
    pose = ObjectPose(0.7)
    pose.on_reset(control_env, np.random.default_rng(7))
    params = pose.params()

    assert params["strength"] == 0.7
    assert set(params["objects"]) == {"obj0", "obj1"}
    assert len(params["objects"]["obj0"]["xy_shift"]) == 2


def test_object_pose_is_reproducible_for_a_given_episode() -> None:
    first, second = _control_env(), _control_env()
    ObjectPose(1.0).on_reset(first, episode_rng(11, "object_pose"))
    ObjectPose(1.0).on_reset(second, episode_rng(11, "object_pose"))

    np.testing.assert_array_equal(first.env.sim.data.qpos, second.env.sim.data.qpos)


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
