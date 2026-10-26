#!/usr/bin/env python
"""Auto proto-labeller: proposes a failure category and onset range per episode.

Computes per-episode signals from a recorded shard (dataset + privileged sim
state) and proposes a failure category and an onset range, per
`Week2_plan.md`'s "Auto proto-labels" section. Writes/merges
`artifacts/corpus/labels.jsonl`, ADR-003's label sidecar.

The proposals are a starting point, not a verdict -- `scripts/review_server.py`
is where a human confirms or overrides them, and the agreement rate between
the two is itself one of the week-2 deliverables. This script never marks a
row `reviewed`; only the review server does that, and this script never
touches a row that already is.

    uv run python scripts/auto_label.py                  # every recorded shard
    uv run python scripts/auto_label.py --shard lighting  # one shard only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backstop.corpus.layout import ShardLayout
from backstop.label.extract import build_shard_rows, load_labels, merge_rows, write_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", default="data/lerobot")
    parser.add_argument("--privileged-root", default="data/privileged")
    parser.add_argument("--eval-root", default="artifacts/eval")
    parser.add_argument("--configs-root", default="configs/record")
    parser.add_argument("--labels-path", default="artifacts/corpus/labels.jsonl")
    parser.add_argument("--shard", action="append", help="limit to one or more shards (repeatable)")
    args = parser.parse_args(argv)

    layout = ShardLayout(
        Path(args.lerobot_root),
        Path(args.privileged_root),
        Path(args.eval_root),
        Path(args.configs_root),
    )
    shards = args.shard or layout.discover_shards()
    if not shards:
        print(f"no shards found under {args.privileged_root}")
        return 1

    existing = load_labels(args.labels_path)
    new_rows = []
    for shard in shards:
        try:
            rows = build_shard_rows(layout, shard)
        except FileNotFoundError as exc:
            print(f"SKIP {shard}: {exc}")
            continue
        new_rows.extend(rows)
        n_success = sum(1 for r in rows if r["success"])
        n_failure = len(rows) - n_success
        print(f"{shard}: {len(rows)} episodes ({n_success} success, {n_failure} failure)")

    merged = merge_rows(existing, new_rows)
    write_labels(args.labels_path, merged)

    n_reviewed = sum(1 for r in merged if r["review"]["status"] == "reviewed")
    print(f"\n{args.labels_path}: {len(merged)} rows total, {n_reviewed} already reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
