"""Fingerprint every analysis output, to prove a change left all of them alone.

Reruns the analysis pipeline on the recorded corpus into `--out`, then writes
`--out/manifest.json`: one sha256 per output file. Take one snapshot before a
refactor and one after; the manifests must match.

    uv run python scripts/snapshot_outputs.py --out /tmp/before
    # ...change the code...
    uv run python scripts/snapshot_outputs.py --out /tmp/after --compare /tmp/before/manifest.json

Steps, in pipeline order. Nothing is written outside `--out`:

1. `evaluate_guard.py`: traces, the single-split report, the unseen suite, and
   the generated block of a copy of `docs/evaluation.md`.
2. `evaluate_stopwatch.py`: the headline and the stopwatch report, from those traces.
3. `freeze_guard.py`: the frozen rules and their trigger steps.
4. `pretest_reselect.py`: the recovery pre-test.
5. `auto_label.py`, on a copy of `labels.jsonl`.
6. With `--with-presence`: `evaluate_presence.py` (GPU, about 3 minutes).

Two things differ between any two runs by design, so they are normalised before
hashing: the `--out` path, which some outputs record, becomes `<OUT>`; and the
code version stamped on the frozen rules (`git_sha`, `git_dirty`) is dropped.

Needs the local `data/` and `artifacts/` recordings, so it is a local
pre-merge check, not a CI job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
VOLATILE_KEYS = frozenset({"git_sha", "git_dirty"})
MANIFEST = "manifest.json"
LOGS = "logs"
OUT_PLACEHOLDER = "<OUT>"


def steps(out: Path, *, labels: Path, report: Path, with_presence: bool) -> list[list[str]]:
    """Each step as script arguments, in the order the pipeline runs them."""
    guard = out / "guard"
    traces = guard / "traces.jsonl"
    plan = [
        ["evaluate_guard.py", "--output-dir", str(guard), "--report-path", str(report)],
        [
            "evaluate_stopwatch.py",
            "--traces",
            str(traces),
            "--labels",
            str(labels),
            "--output",
            str(guard / "stopwatch" / "report.json"),
        ],
        ["freeze_guard.py", "--traces", str(traces), "--output-dir", str(guard / "frozen")],
        [
            "pretest_reselect.py",
            "--labels",
            str(labels),
            "--output",
            str(out / "recovery" / "pretest.json"),
        ],
        ["auto_label.py", "--labels-path", str(out / "labels.jsonl")],
    ]
    if with_presence:
        plan.append(
            [
                "evaluate_presence.py",
                "--traces",
                str(traces),
                "--output",
                str(guard / "presence" / "report.json"),
            ]
        )
    return plan


def _drop_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_volatile(item) for key, item in value.items() if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_drop_volatile(item) for item in value]
    return value


def normalised(path: Path, out: Path) -> bytes:
    """The bytes that are hashed: `--out` replaced, volatile JSON keys dropped."""
    text = path.read_text().replace(str(out.resolve()), OUT_PLACEHOLDER)
    text = text.replace(str(out), OUT_PLACEHOLDER)
    if path.suffix == ".json":
        text = json.dumps(_drop_volatile(json.loads(text)), indent=2, sort_keys=True)
    return text.encode()


def manifest(out: Path) -> dict[str, str]:
    """Relative path -> sha256 of every output under `out`, logs and manifest excluded."""
    entries = {}
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if path.is_file() and relative.parts[0] != LOGS and relative.name != MANIFEST:
            entries[str(relative)] = hashlib.sha256(normalised(path, out)).hexdigest()
    return entries


def differences(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    lines = [f"missing: {name}" for name in sorted(expected.keys() - actual.keys())]
    lines += [f"new: {name}" for name in sorted(actual.keys() - expected.keys())]
    lines += [
        f"changed: {name}"
        for name in sorted(expected.keys() & actual.keys())
        if expected[name] != actual[name]
    ]
    return lines


def run(out: Path, *, labels: Path, report: Path, with_presence: bool) -> dict[str, str]:
    if out.exists():
        shutil.rmtree(out)
    (out / LOGS).mkdir(parents=True)
    # Each step writes into `out`; the inputs it edits are copied in first.
    labels_copy = out / "labels.jsonl"
    shutil.copyfile(labels, labels_copy)
    report_copy = out / "evaluation.md"
    shutil.copyfile(report, report_copy)
    for argv in steps(out, labels=labels_copy, report=report_copy, with_presence=with_presence):
        name = Path(argv[0]).stem
        print(f"running {name}", flush=True)
        with (out / LOGS / f"{name}.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(SCRIPTS / argv[0]), *argv[1:]],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    entries = manifest(out)
    (out / MANIFEST).write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {out / MANIFEST} ({len(entries)} files)")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("docs/evaluation.md"))
    parser.add_argument("--with-presence", action="store_true")
    parser.add_argument("--compare", type=Path, help="a manifest.json from an earlier snapshot")
    args = parser.parse_args(argv)

    entries = run(
        args.out, labels=args.labels, report=args.report, with_presence=args.with_presence
    )
    if args.compare is None:
        return 0
    found = differences(json.loads(args.compare.read_text()), entries)
    for line in found:
        print(line)
    print("identical" if not found else f"{len(found)} differences")
    return 0 if not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
