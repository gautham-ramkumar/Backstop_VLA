"""The arithmetic that turns a measured s/step into a corpus size.

This is the number the day-1 gate rides on, so the failure that matters is not a
crash -- it is a projection that looks reasonable and is wrong by a third. Week 1
recorded 70% success, and its successes averaged 107 steps against a 280-step cap
for every failure. Projecting a perturbed shard tuned to ~50% success with week
1's *overall* mean would under-book roughly 22% of the GPU time.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_throughput.py"


def _load():
    spec = importlib.util.spec_from_file_location("benchmark_throughput", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load()


def _csv(tmp_path: Path, rows: list[tuple[bool, int]]) -> Path:
    lines = ["episode_index,suite,task_id,seed,success,n_steps"]
    for index, (success, steps) in enumerate(rows):
        lines.append(f"{index},libero_spatial,0,{index},{success},{steps}")
    path = tmp_path / "reference.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_length_model_splits_successes_from_failures(tmp_path: Path) -> None:
    path = _csv(tmp_path, [(True, 100), (True, 120), (False, 280), (False, 280)])
    model = benchmark.episode_length_model(path)

    assert model["observed_success_rate"] == 0.5
    assert model["mean_steps_success"] == 110.0
    assert model["mean_steps_failure"] == 280.0
    assert model["mean_steps_overall"] == 195.0


def test_projected_length_grows_as_success_falls(tmp_path: Path) -> None:
    """The whole reason for the split: perturbed shards fail more, so they run longer."""
    model = benchmark.episode_length_model(_csv(tmp_path, [(True, 100), (False, 280)]))

    assert benchmark.steps_at_success_rate(model, 1.0) == pytest.approx(100.0)
    assert benchmark.steps_at_success_rate(model, 0.0) == pytest.approx(280.0)
    assert benchmark.steps_at_success_rate(model, 0.5) == pytest.approx(190.0)
    assert benchmark.steps_at_success_rate(model, 0.5) > benchmark.steps_at_success_rate(model, 0.7)


def test_the_real_week1_reference_reproduces_its_published_numbers() -> None:
    """Guards the parse, not the arithmetic: a `success` column read as always-truthy
    would silently collapse the model to the success branch and halve every estimate."""
    model = benchmark.episode_length_model(benchmark.REFERENCE_CSV)

    assert model["n_episodes"] == 100
    assert model["observed_success_rate"] == 0.70
    assert model["mean_steps_failure"] == 280.0, "week 1's failures all ran to the cap"
    assert model["mean_steps_overall"] == 158.8
    # The plan's back-of-envelope said ~210 steps/episode; the frozen record says less.
    assert benchmark.steps_at_success_rate(model, 0.70) == pytest.approx(158.8, abs=0.1)


def test_a_reference_without_failures_is_refused(tmp_path: Path) -> None:
    """A model fitted to successes only would price a perturbed corpus at the clean
    arm's length -- the single most expensive way to be optimistic."""
    with pytest.raises(ValueError, match="failures"):
        benchmark.episode_length_model(_csv(tmp_path, [(True, 100), (True, 120)]))


def test_a_missing_reference_is_refused_rather_than_guessed() -> None:
    with pytest.raises(FileNotFoundError, match="frozen week-1"):
        benchmark.episode_length_model(Path("docs/does_not_exist.csv"))
