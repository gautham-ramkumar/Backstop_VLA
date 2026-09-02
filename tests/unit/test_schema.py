from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backstop.policy.adapter import OpenVLAAdapter, UnsupportedPolicyError
from backstop.schema import (
    ACTION_DIM,
    CHUNK_SIZE,
    DEFAULT_FALSE_STOP_BUDGET,
    FORBIDDEN_HEADLINES,
    HEADLINE_METRIC,
    STATE_DIM,
    FrameRecord,
    MetricContract,
    load_metric_contract,
)


def test_frame_record_accepts_week1_shapes() -> None:
    record = FrameRecord(
        observation_image_shape=(360, 360, 3),
        observation_image2_shape=(360, 360, 3),
        action_chunk_samples_k=4,
        task="pick up the black bowl",
        success=False,
    )
    assert record.observation_state_dim == STATE_DIM
    assert record.action_dim == ACTION_DIM
    assert record.action_chunk_shape == (CHUNK_SIZE, ACTION_DIM)


def test_frame_record_rejects_non_hwc() -> None:
    with pytest.raises(ValueError):
        FrameRecord(
            observation_image_shape=(3, 360, 360),
            observation_image2_shape=(360, 360, 3),
            action_chunk_samples_k=1,
            task="x",
            success=True,
        )


def test_metric_contract_from_yaml() -> None:
    contract = load_metric_contract(Path("configs/metric.yaml"))
    assert contract.false_stop_budget == DEFAULT_FALSE_STOP_BUDGET
    assert contract.headline_metric == HEADLINE_METRIC
    assert set(FORBIDDEN_HEADLINES) <= set(contract.forbidden_headlines)


def test_metric_contract_rejects_accuracy_headline() -> None:
    with pytest.raises(ValidationError):
        MetricContract(false_stop_budget=0.05, headline_metric="accuracy")  # type: ignore[arg-type]


def test_openvla_stub_is_unimplemented() -> None:
    with pytest.raises(UnsupportedPolicyError, match="week-1"):
        OpenVLAAdapter()
