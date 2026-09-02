"""Frozen week-1 data contracts.

These models are the source of truth for recorded episodes and for the
headline metric. Changing field names or the false-stop budget requires an ADR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ACTION_DIM = 7
STATE_DIM = 8
CHUNK_SIZE = 50

FORBIDDEN_HEADLINES = frozenset({"accuracy", "raw_accuracy", "success_rate"})
HEADLINE_METRIC = "detection_at_false_stop"
DEFAULT_FALSE_STOP_BUDGET = 0.05


class FrameRecord(BaseModel):
    """One environment step written into the LeRobot dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation_image_shape: tuple[int, int, int]
    observation_image2_shape: tuple[int, int, int]
    observation_state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    action_chunk_shape: tuple[int, int] = (CHUNK_SIZE, ACTION_DIM)
    action_chunk_samples_k: int = Field(ge=1)
    task: str
    success: bool

    @field_validator("observation_image_shape", "observation_image2_shape")
    @classmethod
    def _hwc(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3 or value[2] != 3:
            raise ValueError("images must be HWC with 3 channels")
        return value


class RunProvenance(BaseModel):
    """Sidecar written next to every eval/record run."""

    git_sha: str | None
    policy_path: str
    policy_revision: str | None
    policy_type: str
    device: str
    mujoco_version: str | None
    mujoco_gl: str
    seed: int
    suite: str
    task_ids: list[int] | None
    n_episodes: int
    n_action_steps: int
    num_steps: int
    k_samples: int
    batch_size: int
    success_rate: float | None = None
    n_success: int | None = None
    n_failure: int | None = None
    dataset_root: str | None = None
    output_dir: str
    in_tolerance: bool | None = None
    notes: str | None = None


class MetricContract(BaseModel):
    false_stop_budget: float = Field(gt=0.0, lt=1.0)
    headline_metric: Literal["detection_at_false_stop"] = HEADLINE_METRIC
    forbidden_headlines: list[str] = Field(default_factory=lambda: sorted(FORBIDDEN_HEADLINES))
    secondary_metrics: list[str] = Field(default_factory=list)

    @field_validator("headline_metric")
    @classmethod
    def _not_forbidden(cls, value: str) -> str:
        if value in FORBIDDEN_HEADLINES:
            raise ValueError(
                f"headline_metric {value!r} is forbidden; use {HEADLINE_METRIC} "
                "(see docs/adr/001-metric.md)"
            )
        return value


def load_metric_contract(path: str | Path | None = None) -> MetricContract:
    metric_path = Path(path) if path is not None else _default_metric_path()
    with metric_path.open() as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    contract = MetricContract.model_validate(raw)
    if abs(contract.false_stop_budget - DEFAULT_FALSE_STOP_BUDGET) > 1e-12:
        raise ValueError(
            f"false_stop_budget is {contract.false_stop_budget}; "
            f"week-1 lock is {DEFAULT_FALSE_STOP_BUDGET}. Update docs/adr/001-metric.md first."
        )
    return contract


def _default_metric_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] / "configs" / "metric.yaml"


def lerobot_features(k_samples: int, image_hw: tuple[int, int] = (360, 360)) -> dict[str, dict]:
    """LeRobot v3 feature dict matching FrameRecord."""
    height, width = image_hw
    image = {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.image": dict(image),
        "observation.images.image2": dict(image),
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "action_chunk": {"dtype": "float32", "shape": (CHUNK_SIZE, ACTION_DIM), "names": None},
        "action_chunk_samples": {
            "dtype": "float32",
            "shape": (k_samples, CHUNK_SIZE, ACTION_DIM),
            "names": None,
        },
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }
