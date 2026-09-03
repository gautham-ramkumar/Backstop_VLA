"""CLI: record LIBERO episodes with action chunks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backstop.config import PipelineConfig
from backstop.eval.pipeline import run_pipeline


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
    # reproduced number and the corpus. Report tolerance here as `backstop-eval` does.
    lo, hi = cfg.tolerance.min_success_rate, cfg.tolerance.max_success_rate
    if not lo <= rate <= hi:
        print(f"OUT OF TOLERANCE [{lo}, {hi}].")
        if rate < lo:
            print(
                f"Retry with policy.num_steps={cfg.tolerance.fallback_num_steps} "
                "(docs/adr/002-checkpoint.md)."
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
