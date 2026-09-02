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
    print(f"success_rate={result['success_rate']:.3f} n={result['n_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
