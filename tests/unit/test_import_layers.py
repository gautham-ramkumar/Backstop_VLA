"""Which backstop packages may import which (docs/refactor_plan.md, "Target layout").

Dependencies point one way: foundation modules at the bottom, then the corpus
reader, then the packages that analyse recordings (label, guard, recover) and
the ones that run the simulator (env, policy, perturb), then recording, then
evaluation on top. A package that imports upward is how a cycle or a heavy
dependency creeps in unnoticed.

Function-level (lazy) imports count: they are still dependencies, only deferred.

`KNOWN_VIOLATIONS` lists the edges that break these rules today, each with the
refactor stage that removes it. The list may only shrink: a new violation fails,
and so does a listed one that no longer exists, so fixing an edge means deleting
its line here.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "backstop"
SCRIPTS = ROOT / "scripts"

FOUNDATION = frozenset({"backstop", "schema", "config", "provenance", "parquet_io"})
ALLOWED: dict[str, frozenset[str]] = {
    "backstop": frozenset({"schema"}),
    "checks": frozenset(),
    "schema": frozenset(),
    "config": frozenset(),
    "parquet_io": frozenset(),
    "provenance": frozenset({"schema"}),
    "corpus": FOUNDATION,
    "label": FOUNDATION | {"corpus"},
    "guard": FOUNDATION | {"corpus"},
    "recover": FOUNDATION | {"guard"},
    "env": FOUNDATION,
    "policy": FOUNDATION,
    "perturb": FOUNDATION | {"env"},
    "record": FOUNDATION | {"env", "policy", "perturb", "corpus", "guard", "recover"},
    "eval": FOUNDATION | {"record", "env", "policy"},
}
KNOWN_VIOLATIONS: dict[tuple[str, str], str] = {}
KNOWN_SCRIPT_IMPORTS: dict[tuple[str, str], str] = {}
# Must never be imported by the analysis packages: they run on the CPU job,
# without the simulator, the policy or a GPU.
HEAVY = ("torch", "transformers", "lerobot", "gymnasium", "mujoco", "robosuite", "libero")
ANALYSIS_PACKAGES = ("guard", "label", "recover")


def _imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _unit(module: str) -> str:
    """`backstop.guard.signals` -> `guard`; `backstop.schema` -> `schema`; `backstop` -> itself."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def _source_unit(path: Path) -> str:
    relative = path.relative_to(SRC)
    if len(relative.parts) == 1:
        return "backstop" if relative.name == "__init__.py" else relative.stem
    return relative.parts[0]


def package_edges() -> dict[tuple[str, str], set[str]]:
    """(importer, imported) -> the files where that edge appears."""
    edges: dict[tuple[str, str], set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        source = _source_unit(path)
        for module in _imported_modules(path):
            if module != "backstop" and not module.startswith("backstop."):
                continue
            target = _unit(module)
            if target != source:
                edges.setdefault((source, target), set()).add(str(path.relative_to(ROOT)))
    return edges


def _violations() -> dict[tuple[str, str], set[str]]:
    return {
        edge: files
        for edge, files in package_edges().items()
        if edge[1] not in ALLOWED.get(edge[0], frozenset())
    }


def test_every_package_has_a_layer() -> None:
    units = {_source_unit(path) for path in SRC.rglob("*.py")}
    assert units <= ALLOWED.keys(), f"give these a row in ALLOWED: {sorted(units - ALLOWED.keys())}"


def test_no_new_upward_imports() -> None:
    new = {edge: files for edge, files in _violations().items() if edge not in KNOWN_VIOLATIONS}
    assert not new, "imports against the layering:\n" + "\n".join(
        f"  {a} -> {b}: {sorted(files)}" for (a, b), files in sorted(new.items())
    )


def test_known_violations_still_exist() -> None:
    fixed = KNOWN_VIOLATIONS.keys() - _violations().keys()
    assert not fixed, f"fixed, so delete from KNOWN_VIOLATIONS: {sorted(fixed)}"


def test_scripts_import_the_package_never_each_other() -> None:
    script_names = {path.stem for path in SCRIPTS.glob("*.py")}
    found = {
        (path.stem, module.split(".")[0])
        for path in SCRIPTS.glob("*.py")
        for module in _imported_modules(path)
        if module.split(".")[0] in script_names
    }
    assert found == set(KNOWN_SCRIPT_IMPORTS), (
        f"new: {sorted(found - KNOWN_SCRIPT_IMPORTS.keys())}, "
        f"fixed (delete from KNOWN_SCRIPT_IMPORTS): {sorted(KNOWN_SCRIPT_IMPORTS.keys() - found)}"
    )


def test_analysis_packages_import_without_the_sim_stack() -> None:
    modules = sorted(
        ".".join(path.relative_to(SRC.parent).with_suffix("").parts).removesuffix(".__init__")
        for package in ANALYSIS_PACKAGES
        for path in (SRC / package).rglob("*.py")
    )
    probe = (
        "import importlib, sys\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        f"print(','.join(sorted(m for m in {HEAVY!r} if m in sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip() == "", f"importing analysis code loaded: {result.stdout.strip()}"
