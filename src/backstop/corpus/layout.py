"""Which files belong to a shard: its run record, dataset and recording config."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backstop.schema import RunProvenance

# What the analysis scripts read and write, besides the shards themselves.
TRACES = Path("artifacts/guard/traces.jsonl")
LABELS = Path("artifacts/corpus/labels.jsonl")


@dataclass(frozen=True)
class ShardLayout:
    """The four roots a shard's files hang off. Defaults are the repo's own."""

    lerobot_root: Path = Path("data/lerobot")
    privileged_root: Path = Path("data/privileged")
    eval_root: Path = Path("artifacts/eval")
    configs_root: Path = Path("configs/record")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """`--lerobot-root`, `--privileged-root`, `--eval-root`, `--configs-root`."""
        for name, default in vars(ShardLayout()).items():
            flag = "--" + name.replace("_", "-")
            parser.add_argument(flag, type=Path, default=default)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ShardLayout:
        return cls(args.lerobot_root, args.privileged_root, args.eval_root, args.configs_root)

    def discover_shards(self) -> list[str]:
        """Every shard recorded with `privileged: true`, the corpus shards."""
        if not self.privileged_root.is_dir():
            return []
        return sorted(p.name for p in self.privileged_root.iterdir() if p.is_dir())

    @staticmethod
    def run_stem(root: Path, shard: str, suffix: str = "") -> Path:
        """Spatial shards are named `spatial_<shard>_k4`. A shard from another suite
        already carries its suite in the shard name (`object_clean` ->
        `object_clean_k4`), so that form is the fallback."""
        spatial = root / f"spatial_{shard}_k4{suffix}"
        return spatial if spatial.exists() else root / f"{shard}_k4{suffix}"

    def provenance(self, shard: str) -> RunProvenance | None:
        """The shard's `run.json`, or None when it has none (or names no dataset)."""
        path = self.run_stem(self.eval_root, shard) / "run.json"
        if not path.is_file():
            return None
        provenance = RunProvenance.model_validate(json.loads(path.read_text()))
        return provenance if provenance.dataset_root else None

    def dataset_root(self, shard: str) -> Path:
        provenance = self.provenance(shard)
        if provenance is None or provenance.dataset_root is None:
            raise FileNotFoundError(
                f"no run.json (or no dataset_root in it) for shard {shard!r} under {self.eval_root}"
            )
        dataset_path = Path(provenance.dataset_root)
        if not dataset_path.is_absolute() and not dataset_path.is_dir():
            # run.json stores the path as it was at record time (relative to the repo
            # root the recording ran from); resolve against lerobot_root if the plain
            # relative path does not exist from the current cwd.
            candidate = self.lerobot_root / dataset_path.name
            dataset_path = candidate if candidate.is_dir() else dataset_path
        return dataset_path

    def perturbation_params(self, shard: str) -> dict[str, Any]:
        """Best-effort: `perturbation_kwargs` lives in the recording config, not in
        `run.json` (`build_provenance` doesn't carry it -- see ADR-003). Only the
        categorical families (`absent_target`, `language_paraphrase`) have any."""
        import yaml  # noqa: PLC0415

        path = self.run_stem(self.configs_root, shard, ".yaml")
        if not path.is_file():
            return {}
        raw = yaml.safe_load(path.read_text()) or {}
        record = raw.get("record", {}) if isinstance(raw, dict) else {}
        kwargs = record.get("perturbation_kwargs", {}) if isinstance(record, dict) else {}
        return dict(kwargs) if isinstance(kwargs, dict) else {}
