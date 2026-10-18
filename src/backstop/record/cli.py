"""CLI: record LIBERO episodes with action chunks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backstop.config import PipelineConfig
from backstop.eval.pipeline import run_pipeline
from backstop.eval.tolerance import cli_outcome


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Record Backstop LIBERO episodes")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/record/spatial_baseline.yaml"),
        help="Pipeline YAML",
    )
    args = parser.parse_args(argv)
    cfg = PipelineConfig.from_yaml(args.config)
    cfg.record.enabled = True
    result = run_pipeline(cfg)
    rate = result["success_rate"]
    print(f"success_rate={rate:.3f} n={result['n_total']} dataset={cfg.record.root}")
    # Recording is also the baseline pass: with K=1 and seeded noise the trajectory
    # is identical whether or not the dataset is written, so one run yields both the
    # reproduced number and the corpus. Tolerance is judged once, in the pipeline.
    outcome = cli_outcome(rate, result["in_tolerance"], cfg.tolerance)
    if outcome.message:
        print(outcome.message)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
