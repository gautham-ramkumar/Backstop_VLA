"""CLIs: `backstop-eval` evaluates a frozen VLA on LIBERO, `backstop-record` also
writes the LeRobot dataset with action chunks.

Recording is also the baseline pass: with seeded noise the trajectory is
identical whether or not the dataset is written, so one run yields both the
reproduced number and the corpus. Both exit on the pipeline's one tolerance
verdict.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backstop.config import PipelineConfig
from backstop.eval.pipeline import run_pipeline
from backstop.eval.tolerance import cli_outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen VLA on LIBERO")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/spatial_baseline.yaml"),
        help="Pipeline YAML",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Also write a LeRobot dataset (overrides YAML record.enabled)",
    )
    args = parser.parse_args(argv)
    return _run(args.config, record=args.record)


def record_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record Backstop LIBERO episodes")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/record/spatial_baseline.yaml"),
        help="Pipeline YAML",
    )
    args = parser.parse_args(argv)
    return _run(args.config, record=True)


def _run(config: Path, *, record: bool) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = PipelineConfig.from_yaml(config)
    if record:
        cfg.record.enabled = True
    result = run_pipeline(cfg)
    dataset = f" dataset={cfg.record.root}" if cfg.record.enabled else ""
    print(f"success_rate={result['success_rate']:.3f} n={result['n_total']}{dataset}")
    outcome = cli_outcome(result["success_rate"], result["in_tolerance"], cfg.tolerance)
    if outcome.message:
        print(outcome.message)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
