"""Contract for the record loop's stage timer.

The timer exists to answer one question on day 1 of week 2 -- is the step
policy-bound or render-bound -- and every failure mode here is a plausible-looking
table that answers it wrongly: a nested stage counted twice, a stage that raised
and vanished, or a breakdown that silently absorbs the time it cannot explain.

CPU only, no torch and no simulator: the arithmetic is what needs guarding.
"""

from __future__ import annotations

import json
import time

import pytest

from backstop.record.timing import NESTED_STAGES, StageTimer

TICK = 0.02


def _burn(seconds: float) -> None:
    """Busy-wait. `sleep` under-delivers on a loaded machine and this asserts on time."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_stage_totals_and_call_counts_accumulate() -> None:
    timer = StageTimer()
    for _ in range(3):
        with timer("env_step"):
            _burn(TICK)
    summary = timer.summary()

    entry = summary["stages"]["env_step"]
    assert entry["calls"] == 3
    assert entry["total_s"] >= 3 * TICK
    assert entry["ms_per_call"] == pytest.approx(entry["total_s"] * 1000 / 3, rel=0.01)


def test_nested_stage_is_not_double_counted() -> None:
    """`env_render` runs inside `env_step`. Summing both would invent time.

    A breakdown that over-accounts pushes `unaccounted` negative, which reads as
    "fully explained" while actually being the loudest possible symptom of a
    miscounted stage.
    """
    assert NESTED_STAGES["env_render"] == "env_step"
    timer = StageTimer()
    with timer("env_step"):
        with timer("env_render"):
            _burn(TICK)
        _burn(TICK)
    timer.step_done()
    timer.finish()
    summary = timer.summary()

    # env_step spans both; accounted must charge env_step once and env_render never.
    assert summary["accounted_s"] == pytest.approx(
        summary["stages"]["env_step"]["total_s"], abs=1e-3
    )
    assert summary["unaccounted_s"] >= 0.0
    assert summary["stages"]["env_render"]["nested_in"] == "env_step"


def test_physics_is_derived_by_subtracting_render() -> None:
    """Robosuite interleaves substeps and camera sampling; physics is only reachable
    by subtraction, so the report must label it as derived rather than measured."""
    timer = StageTimer()
    with timer("env_step"):
        with timer("env_render"):
            _burn(TICK)
        _burn(2 * TICK)
    timer.step_done()
    timer.finish()
    derived = timer.summary()["derived"]["env_physics"]

    assert derived["total_s"] == pytest.approx(2 * TICK, rel=0.4)
    assert "not measured directly" in derived["note"]


def test_untimed_work_shows_up_as_unaccounted() -> None:
    """The gap is the honest part: a breakdown must not attribute time it never saw."""
    timer = StageTimer()
    with timer("env_step"):
        _burn(TICK)
    _burn(3 * TICK)  # glue between stages, deliberately uninstrumented
    timer.finish()
    summary = timer.summary()

    assert summary["unaccounted_s"] == pytest.approx(3 * TICK, rel=0.4)
    assert summary["unaccounted_pc"] > 50.0


def test_a_stage_that_raises_still_reports_its_time() -> None:
    """A six-hour run that dies in hour five must still say where the hours went."""
    timer = StageTimer()
    with pytest.raises(RuntimeError), timer("env_step"):
        _burn(TICK)
        raise RuntimeError("simulator exploded")

    entry = timer.summary()["stages"]["env_step"]
    assert entry["calls"] == 1
    assert entry["total_s"] >= TICK


def test_sync_reports_false_when_it_could_not_be_honoured(monkeypatch) -> None:
    """Attribution is only exact with a drained CUDA queue.

    Reporting `cuda_sync: true` on a CPU box would claim a precision the numbers
    do not have, and the per-stage split is exactly what the day-1 gate turns on.
    """
    monkeypatch.setattr("backstop.record.timing._resolve_sync", lambda enabled: None)
    summary = StageTimer(sync=True).summary()

    assert summary["cuda_sync_requested"] is True
    assert summary["cuda_sync"] is False


def test_per_episode_wall_and_steps_are_split_by_episode() -> None:
    """Episode 0 carries CUDA and asset warm-up. A single mean hides it."""
    timer = StageTimer()
    for steps in (3, 2):
        for _ in range(steps):
            with timer("env_step"):
                _burn(TICK / 4)
            timer.step_done()
        timer.episode_done()
    timer.finish()
    summary = timer.summary()

    assert summary["episode_steps"] == [3, 2]
    assert len(summary["episode_wall_s"]) == 2
    assert summary["n_steps"] == 5
    assert summary["n_episodes"] == 2
    assert summary["steps_per_episode"] == pytest.approx(2.5)


def test_rates_are_zero_rather_than_dividing_by_zero() -> None:
    """A run that dies before its first step must still produce a report."""
    summary = StageTimer().summary()
    assert summary["s_per_step"] == 0.0
    assert summary["s_per_episode"] == 0.0
    assert summary["stages"] == {}


def test_summary_is_json_serialisable() -> None:
    """It is written to `timing.json` on every run; a non-serialisable field would
    take down the report writer at the end of a completed recording."""
    timer = StageTimer()
    with timer("env_step"):
        _burn(TICK)
    timer.step_done()
    timer.episode_done()
    timer.finish()

    assert json.loads(json.dumps(timer.summary()))["n_steps"] == 1


def test_start_excludes_setup_from_the_wall_clock() -> None:
    """Loading a checkpoint and building the simulator costs tens of seconds.

    Charged to the wall clock it divides into `s_per_step`, and a benchmark of a
    few episodes would over-book the corpus by hours. `run_episodes` restarts the
    clock once the setup is done.
    """
    timer = StageTimer()
    _burn(4 * TICK)  # policy load, env build
    timer.start()
    with timer("env_step"):
        _burn(TICK)
    timer.step_done()
    timer.finish()
    summary = timer.summary()

    assert summary["wall_s"] == pytest.approx(TICK, rel=0.5)
    assert summary["unaccounted_pc"] < 25.0


def test_finish_is_idempotent() -> None:
    """The wall clock stops when the run ends, not when the report is rendered."""
    timer = StageTimer()
    timer.finish()
    first = timer.wall_s
    _burn(TICK)
    timer.finish()

    assert timer.wall_s == first
