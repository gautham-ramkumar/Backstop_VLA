"""Record a short fixed run in the real simulator, and compare it with an earlier one.

The CPU golden test (`tests/unit/test_rollout_golden.py`) proves the recording
loop is unchanged on fakes. This is the same check on real LIBERO and SmolVLA:
record `configs/record/golden_smoke.yaml` (task 0, two episodes, K=4,
privileged sim state) before a change and again after it, then compare.

    uv run python scripts/compare_recordings.py --out /tmp/rec_before
    # ...change the code...
    uv run python scripts/compare_recordings.py --out /tmp/rec_after --compare /tmp/rec_before

What must match exactly: every column of every data and metadata parquet file,
every JSON metadata file, and every privileged sim-state array. Encoded videos
and intermediate image files are compared by hash but reported separately, as
advisory: an encoder can produce different bytes for the same frames.

Record twice *before* changing anything. If those two already differ (GPU
nondeterminism), this comparison is advisory and the CPU golden test is the gate.
Needs a GPU and the LIBERO assets, so it is a local check, not a CI job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "record" / "golden_smoke.yaml"
EXACT_SUFFIXES = (".parquet", ".json")
ADVISORY_SUFFIXES = (".mp4", ".png", ".jpg")
DATASET = "dataset"
PRIVILEGED = "privileged"


def redirected_config(config: Path, out: Path) -> dict[str, Any]:
    """The recording config with every output under `out` and wandb off."""
    raw = yaml.safe_load(config.read_text())
    raw["output_dir"] = str(out / "eval")
    raw["record"].update(
        root=str(out / DATASET), privileged_root=str(out / PRIVILEGED), on_existing="overwrite"
    )
    raw["wandb"] = {"enabled": False, "mode": "disabled"}
    return raw


def record(config: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    run_config = out / "config.yaml"
    run_config.write_text(yaml.safe_dump(redirected_config(config, out), sort_keys=False))
    with (out / "record.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "backstop.record.cli", "--config", str(run_config)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _compare_parquet(a: Path, b: Path) -> list[str]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    left, right = pq.read_table(a), pq.read_table(b)
    if left.schema != right.schema:
        return ["schema differs"]
    return [
        f"column {name!r} differs"
        for name in left.column_names
        if not left.column(name).equals(right.column(name))
    ]


def _compare_npz(a: Path, b: Path) -> list[str]:
    with np.load(a) as left, np.load(b) as right:
        if sorted(left.files) != sorted(right.files):
            return [f"keys differ: {sorted(left.files)} vs {sorted(right.files)}"]
        return [
            f"array {key!r} differs"
            for key in left.files
            if not np.array_equal(left[key], right[key], equal_nan=True)
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_trees(before: Path, after: Path) -> tuple[list[str], list[str]]:
    """(exact differences, advisory differences) between two recording folders."""
    exact: list[str] = []
    advisory: list[str] = []
    files = {
        path.relative_to(root)
        for root in (before, after)
        for sub in (DATASET, PRIVILEGED)
        for path in (root / sub).rglob("*")
        if path.is_file()
    }
    for relative in sorted(files):
        a, b = before / relative, after / relative
        if not a.is_file() or not b.is_file():
            exact.append(f"{relative}: only in {'after' if b.is_file() else 'before'}")
            continue
        if relative.suffix == ".parquet":
            found = _compare_parquet(a, b)
        elif relative.suffix == ".npz":
            found = _compare_npz(a, b)
        elif relative.suffix == ".json":
            found = [] if json.loads(a.read_text()) == json.loads(b.read_text()) else ["differs"]
        else:
            found = [] if _sha256(a) == _sha256(b) else ["bytes differ"]
        bucket = advisory if relative.suffix in ADVISORY_SUFFIXES else exact
        bucket.extend(f"{relative}: {problem}" for problem in found)
    return exact, advisory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--compare", type=Path, help="an earlier --out folder")
    parser.add_argument("--skip-record", action="store_true", help="only compare")
    args = parser.parse_args(argv)

    if not args.skip_record:
        print(f"recording {args.config} into {args.out}", flush=True)
        record(args.config, args.out)
    if args.compare is None:
        return 0
    exact, advisory = compare_trees(args.compare, args.out)
    for line in exact:
        print(f"DIFFERS  {line}")
    for line in advisory:
        print(f"advisory {line}")
    print("identical" if not exact else f"{len(exact)} exact differences")
    return 0 if not exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
