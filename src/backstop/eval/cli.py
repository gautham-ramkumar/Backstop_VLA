"""CLI: evaluate a frozen VLA on LIBERO."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backstop.config import PipelineConfig
from backstop.eval.pipeline import run_pipeline


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
    if cfg.tolerance.enforce and result["success_rate"] < cfg.tolerance.min_success_rate:
        print(
            f"BELOW TOLERANCE. Retry with policy.num_steps={cfg.tolerance.fallback_num_steps} "
            "(docs/adr/002-checkpoint.md)."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
