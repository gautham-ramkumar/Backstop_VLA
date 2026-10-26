"""Which files may reach into LeRobot's or LIBERO's private attributes.

Each private attribute below belongs to a third-party class we do not control, so
an upgrade can rename or remove it. Keeping every use in one module per library
(`env/libero.py` for the simulator, the policy module for SmolVLA) means an
upgrade breaks one place, loudly.

Counts attribute access (`x._env`) and string names (`getattr(x, "_env")`,
`vec_env.call("_max_episode_steps")`), not mentions in docstrings or comments.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SIMULATOR = "src/backstop/env/libero.py"
POLICY = "src/backstop/policy/smolvla.py"
OWNERS: dict[str, str] = {
    "_env": SIMULATOR,
    "_format_raw_obs": SIMULATOR,
    "_get_observations": SIMULATOR,
    "_task_bddl_file": SIMULATOR,
    "_init_states": SIMULATOR,
    "_max_episode_steps": SIMULATOR,
    "_get_action_chunk": POLICY,
    "_queues": POLICY,
}


def _private_uses(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute) and node.attr in OWNERS:
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and node.value in OWNERS:
            found.add(str(node.value))
    return found


def test_private_attributes_are_used_only_by_their_owner() -> None:
    misplaced = {
        (str(path.relative_to(ROOT)), name)
        for folder in ("src", "scripts")
        for path in sorted((ROOT / folder).rglob("*.py"))
        for name in _private_uses(path)
        if OWNERS[name] != str(path.relative_to(ROOT))
    }
    assert not misplaced, "go through env/libero.py or the policy module instead:\n" + "\n".join(
        f"  {path}: {name}" for path, name in sorted(misplaced)
    )


def test_every_listed_attribute_is_still_used_by_its_owner() -> None:
    """A stale entry would let a later stray use pass unnoticed."""
    unused = {name for name, owner in OWNERS.items() if name not in _private_uses(ROOT / owner)}
    assert not unused, f"no longer used, so delete from OWNERS: {sorted(unused)}"
