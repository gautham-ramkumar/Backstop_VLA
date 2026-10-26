"""Golden fingerprints of one fixed rollout: proof the recording code still records the same thing.

Runs the real `run_episodes` on the CPU fakes (`tests/fakes/rollout.py`) with
every recording feature on: K=4 sampled chunks, two episodes, the `action_noise`
perturbation, and privileged sim-state capture. It then compares, against
`tests/unit/golden/rollout.json`:

- a sha256 of every recorded frame (every key: arrays by dtype, shape and bytes);
- a sha256 of every privileged sim-state sidecar;
- the per-episode rows (seed, steps, success) and the timer's stage names and call
  counts, kept readable because a change there should be easy to see.

A refactor of the recording code must leave all of it unchanged. When a change
*should* alter what is recorded, regenerate the file deliberately and review the
diff:

    BACKSTOP_UPDATE_GOLDEN=1 uv run pytest tests/unit/test_rollout_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

# torch and lerobot arrive with the `sim` extra; the sim CI job runs this module.
pytest.importorskip("torch")
pytest.importorskip("lerobot")

import numpy as np  # noqa: E402

from backstop.corpus.privileged import load_episode  # noqa: E402
from fakes.rollout import run_fake_rollout  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "rollout.json"
UPDATE_ENV = "BACKSTOP_UPDATE_GOLDEN"
SHARD = "golden"
N_EPISODES = 2


def _feed(digest: Any, value: Any) -> None:
    """Hash `value` canonically: same content, same digest, whatever the dict order."""
    if isinstance(value, np.ndarray):
        digest.update(f"ndarray:{value.dtype.str}:{value.shape}:".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict:")
        for key in sorted(value):
            digest.update(f"{key}=".encode())
            _feed(digest, value[key])
    elif isinstance(value, list | tuple):
        digest.update(f"seq:{len(value)}:".encode())
        for item in value:
            _feed(digest, item)
    elif isinstance(value, str | bool | int | float | np.generic) or value is None:
        digest.update(f"{type(value).__name__}:{value!r};".encode())
    else:
        raise TypeError(f"no canonical hash for {type(value).__name__}; extend _feed")


def fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _feed(digest, value)
    return digest.hexdigest()


def record_fingerprints(tmp_path: Path) -> dict[str, Any]:
    privileged_root = tmp_path / "privileged"
    rollout = run_fake_rollout(
        tmp_path,
        k_samples=4,
        n_episodes=N_EPISODES,
        with_sim=True,
        record_overrides={
            "perturbation": "action_noise",
            "perturbation_strength": 0.15,
            "privileged": True,
            "privileged_root": str(privileged_root),
            "shard": SHARD,
        },
    )
    timing = rollout.stats["timing"]
    return {
        "frames": [fingerprint(frame) for frame in rollout.frames],
        "privileged": [
            fingerprint(load_episode(privileged_root, SHARD, index)) for index in range(N_EPISODES)
        ],
        "episodes": rollout.stats["episodes"],
        "timing": {
            "n_episodes": timing["n_episodes"],
            "n_steps": timing["n_steps"],
            "episode_steps": timing["episode_steps"],
            "stage_calls": {name: entry["calls"] for name, entry in timing["stages"].items()},
        },
    }


def _first_difference(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    for section in sorted(expected.keys() | actual.keys()):
        want, got = expected.get(section), actual.get(section)
        if want == got:
            continue
        if isinstance(want, list) and isinstance(got, list):
            for index, (a, b) in enumerate(zip(want, got, strict=False)):
                if a != b:
                    return f"{section}[{index}] changed"
            return f"{section}: {len(want)} entries expected, {len(got)} recorded"
        return f"{section} changed: expected {want!r}, got {got!r}"
    return "no difference"


def test_recording_matches_the_golden_fingerprints(tmp_path: Path) -> None:
    actual = record_fingerprints(tmp_path)
    if os.environ.get(UPDATE_ENV):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        return
    assert GOLDEN.is_file(), f"{GOLDEN} is missing; generate it with {UPDATE_ENV}=1"
    expected = json.loads(GOLDEN.read_text())
    assert actual == expected, (
        f"The recording changed: {_first_difference(expected, actual)}. If that is intended, "
        f"regenerate with {UPDATE_ENV}=1 and review the diff of {GOLDEN.name}."
    )


def test_the_fingerprint_sees_every_kind_of_change() -> None:
    """The golden test is only as strict as `fingerprint`, so check its edges."""
    base = {"action": np.zeros(3, dtype=np.float32), "task": "x", "success": False}
    assert fingerprint(base) == fingerprint(dict(reversed(list(base.items()))))
    changed_value = {**base, "action": np.array([0, 0, 1e-7], dtype=np.float32)}
    changed_dtype = {**base, "action": np.zeros(3, dtype=np.float64)}
    changed_shape = {**base, "action": np.zeros((1, 3), dtype=np.float32)}
    changed_flag = {**base, "success": True}
    digests = {
        fingerprint(item)
        for item in (base, changed_value, changed_dtype, changed_shape, changed_flag)
    }
    assert len(digests) == 5
    with pytest.raises(TypeError, match="no canonical hash"):
        fingerprint({"bad": object()})
