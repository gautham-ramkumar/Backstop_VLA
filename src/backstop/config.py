"""YAML pipeline config (pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class PolicyConfig(BaseModel):
    type: Literal["smolvla", "openvla"] = "smolvla"
    path: str
    revision: str | None = None
    device: str = "cuda"
    n_action_steps: int = 1
    num_steps: int = 1


class EnvYamlConfig(BaseModel):
    type: Literal["libero"] = "libero"
    task: str = "libero_spatial"
    task_ids: list[int] | None = None
    control_mode: Literal["relative", "absolute"] = "relative"
    init_states: bool = True
    mujoco_gl: Literal["osmesa", "egl", "glfw"] = "osmesa"


class EvalYamlConfig(BaseModel):
    n_episodes: int = Field(ge=1)
    batch_size: int = 1
    max_parallel_tasks: int = 1


class RecordYamlConfig(BaseModel):
    enabled: bool = False
    k_samples: int = Field(default=1, ge=1)
    repo_id: str = "local/backstop"
    root: str = "data/lerobot/backstop"


class WandbYamlConfig(BaseModel):
    project: str = "backstop"
    enabled: bool = True
    mode: Literal["online", "offline", "disabled"] = "offline"


class ToleranceConfig(BaseModel):
    min_success_rate: float = 0.70
    max_success_rate: float = 0.90
    fallback_num_steps: int = 1
    enforce: bool = True


class PipelineConfig(BaseModel):
    seed: int = 0
    output_dir: str
    policy: PolicyConfig
    env: EnvYamlConfig
    eval: EvalYamlConfig
    record: RecordYamlConfig = Field(default_factory=RecordYamlConfig)
    wandb: WandbYamlConfig = Field(default_factory=WandbYamlConfig)
    tolerance: ToleranceConfig = Field(default_factory=ToleranceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with Path(path).open() as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        return cls.model_validate(raw)
