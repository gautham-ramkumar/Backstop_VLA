"""Check the C++ guard (cpp/) stops every recorded episode where Python does.

Builds `cpp/guard_replay.cpp`, then feeds each episode of the given shards
through it and through `OnlineGuard`, under each frozen rule. Every step's
trigger must match exactly; scores may differ only in the last few bits,
because C++ adds the deviations up in a different order than numpy.
`tests/unit/test_guard_cpp.py` runs the same comparison on synthetic episodes.

    PYTHONPATH=src python scripts/verify_guard_cpp.py --shard clean
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from backstop.corpus.layout import ShardLayout
from backstop.guard.extract import iter_episode_samples
from backstop.guard.online import OnlineGuard, load_rule, step_disagreement

CPP_DIR = Path(__file__).resolve().parents[1] / "cpp"
FROZEN = Path("artifacts/guard/frozen")
SCORE_TOLERANCE = 1e-12


def build(out_dir: Path) -> Path:
    binary = out_dir / "guard_replay"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(binary),
         str(CPP_DIR / "guard_replay.cpp")],
        check=True,
    )  # fmt: skip
    return binary


def cpp_replay(
    binary: Path, rule: Path, samples: np.ndarray, cooldown: int = 0, rearms: tuple[int, ...] = ()
) -> tuple[np.ndarray, list[str]]:
    """(per-step scores, per-step trigger or "-") from the C++ guard."""
    payload = np.asarray(samples.shape, dtype="<i8").tobytes()
    payload += np.ascontiguousarray(samples, dtype="<f8").tobytes()
    args = [str(binary), str(rule), str(cooldown), *map(str, rearms)]
    out = subprocess.run(args, input=payload, capture_output=True, check=True).stdout.decode()
    rows = [line.split() for line in out.splitlines()]
    return np.array([float(score) for score, _ in rows]), [reason for _, reason in rows]


def python_replay(
    rule: Path, samples: np.ndarray, cooldown: int = 0, rearms: tuple[int, ...] = ()
) -> tuple[np.ndarray, list[str]]:
    """The same, from `OnlineGuard`."""
    guard = OnlineGuard(load_rule(rule), cooldown=cooldown)
    scores, reasons = [], []
    for step, plan in enumerate(samples):
        if step in rearms:
            guard.rearm(step)
        scores.append(step_disagreement(plan))
        guard.check(scores[-1], step)
        reasons.append(guard.fired_reason or "-")
    return np.array(scores), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--rule", type=Path, action="append", default=None)
    parser.add_argument("--build-dir", type=Path, default=Path("artifacts/guard/cpp"))
    ShardLayout.add_args(parser)
    args = parser.parse_args()

    args.build_dir.mkdir(parents=True, exist_ok=True)
    binary = build(args.build_dir)
    rules = args.rule or sorted(FROZEN.glob("rule_??.json"))
    layout = ShardLayout.from_args(args)
    episodes = mismatches = 0
    worst = 0.0
    for shard in args.shard:
        for index, samples in iter_episode_samples(layout, shard):
            episodes += 1
            for rule in rules:
                cpp_scores, cpp_reasons = cpp_replay(binary, rule, samples)
                py_scores, py_reasons = python_replay(rule, samples)
                worst = max(worst, float(np.max(np.abs(cpp_scores - py_scores), initial=0.0)))
                if cpp_reasons != py_reasons:
                    mismatches += 1
                    print(f"MISMATCH {shard} episode {index} under {rule.name}")
    print(f"{episodes} episodes x {len(rules)} rules: {mismatches} trigger mismatches, "
          f"largest score difference {worst:.3g}")  # fmt: skip
    if mismatches or worst > SCORE_TOLERANCE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
