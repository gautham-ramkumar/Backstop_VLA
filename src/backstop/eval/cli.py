"""CLI: evaluate a frozen VLA on LIBERO."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backstop.config import PipelineConfig
from backstop.eval.pipeline import run_pipeline
from backstop.eval.tolerance import cli_outcome


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
    cfg = PipelineConfig.from_yaml(args.config)
    if args.record:
        cfg.record.enabled = True
    result = run_pipeline(cfg)
    print(f"success_rate={result['success_rate']:.3f} n={result['n_total']}")
    outcome = cli_outcome(result["success_rate"], result["in_tolerance"], cfg.tolerance)
    if outcome.message:
        print(outcome.message)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
