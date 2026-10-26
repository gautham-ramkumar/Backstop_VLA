"""Contract for `PerturbedEnv`'s task-description override.

`absent_target` and `language_paraphrase` only do anything if the instruction the
policy actually sees changes -- everything else about them is inert (`on_reset`
and `perturb_action` are both no-ops). `task_description` on the real LeRobot
LiberoEnv is a plain attribute, not a method, set once from the task suite, so
overriding it means setting an instance attribute on the wrapper itself rather
than calling through to the wrapped env. This is exactly the kind of thing that
can look wired up (the family has a `task_description()` method, the wrapper
looks like it should call it) while nothing on the path from the perturbation to
`vec_env.call("task_description")` -- what the record loop and `_build_frame`
both read -- actually invokes it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# `backstop.perturb.wrapper` imports gymnasium, which arrives with the `sim`
# extra. The CPU CI job skips this module; the sim job runs it.
pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402

from backstop.perturb import (  # noqa: E402
    AbsentTarget,
    ActionNoise,
    NoPerturbation,
    Paraphrase,
    Perturbation,
)
from backstop.perturb.wrapper import PerturbedEnv  # noqa: E402


class FakeControlEnv:
    """Enough of a LIBERO `ControlEnv` for `_observe`'s no-resettle path."""

    def __init__(self, raw_obs: dict[str, Any]) -> None:
        self.env = SimpleNamespace(_get_observations=lambda force_update=True: raw_obs)


class FakeInnerEnv:
    """Enough of a LeRobot `LiberoEnv` (post-`.unwrapped`) to drive `reset()`."""

    def __init__(self, task_description: str, raw_obs: dict[str, Any]) -> None:
        self.task_description = task_description
        self._env = FakeControlEnv(raw_obs)

    def _format_raw_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw


class FakeWrappedEnv(gym.Env):
    """What `PerturbedEnv` wraps: real code wraps a LeRobot `LiberoEnv` gym.Env;
    `PerturbedEnv.__init__` asserts `isinstance(env, gym.Env)`, so the fake must
    actually subclass it rather than just duck-type `.reset()`."""

    def __init__(self, inner: FakeInnerEnv) -> None:
        self._inner = inner

    @property
    def unwrapped(self) -> Any:  # type: ignore[override]
        return self._inner

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"dummy": True}, {}


def _perturbed_env(task_description: str, perturbation: Any) -> PerturbedEnv:
    inner = FakeInnerEnv(task_description, raw_obs={"observed": True})
    return PerturbedEnv(FakeWrappedEnv(inner), perturbation, episode_seed=0)


@pytest.mark.parametrize("perturbation", [NoPerturbation(), ActionNoise(1.0)])
def test_families_without_a_language_override_leave_the_instruction_alone(perturbation) -> None:
    wrapped = _perturbed_env("pick up the bowl", perturbation)
    wrapped.reset()

    assert wrapped.get_wrapper_attr("task_description") == "pick up the bowl"


def test_absent_target_overrides_what_vec_env_call_would_read() -> None:
    """The exact path `record/loop.py` uses: `vec_env.call("task_description")`
    resolves through `get_wrapper_attr`, not through a direct attribute read."""
    perturbation = AbsentTarget(replacement="pick up the nonexistent widget")
    wrapped = _perturbed_env("pick up the bowl", perturbation)
    wrapped.reset()

    assert wrapped.get_wrapper_attr("task_description") == "pick up the nonexistent widget"


def test_paraphrase_overrides_what_vec_env_call_would_read() -> None:
    perturbation = Paraphrase(paraphrase="grab the bowl and set it on the plate")
    wrapped = _perturbed_env("pick up the bowl", perturbation)
    wrapped.reset()

    assert wrapped.get_wrapper_attr("task_description") == "grab the bowl and set it on the plate"


def test_prepare_env_runs_once_before_the_first_reset() -> None:
    """`distractor` needs to mutate the LiberoEnv's bddl path before its lazy
    `_ensure_env()` ever builds a model from it -- by the time `on_reset` runs,
    `self.env.reset()` has already called `_ensure_env()`, which is too late.
    `prepare_env`, called from `PerturbedEnv.__init__`, is the only hook early
    enough, so it must run before `reset()` is ever invoked, not just before
    `on_reset`."""
    calls: list[Any] = []

    class FakePerturbation(Perturbation):
        name = "distractor"
        strength = 1.0

        def prepare_env(self, env: Any) -> None:
            calls.append(env)

    inner = FakeInnerEnv("pick up the bowl", raw_obs={"observed": True})
    wrapped_env = FakeWrappedEnv(inner)

    PerturbedEnv(wrapped_env, FakePerturbation(), episode_seed=0)

    assert calls == [inner], "prepare_env must run in __init__, before any reset()"


def test_families_without_prepare_env_are_unaffected() -> None:
    """Every family except `distractor` inherits the base class's no-op
    `prepare_env`; wrapping must leave the env as it was."""
    wrapped = _perturbed_env("pick up the bowl", NoPerturbation())

    wrapped.reset()

    assert wrapped.get_wrapper_attr("task_description") == "pick up the bowl"


def test_absent_target_does_not_leak_across_perturbation_instances() -> None:
    """A wrapper attribute is instance state -- if `_perturbed_task_description`
    ever read from a class-level default instead of `self.perturbation`, every
    sub-env would end up with whichever family's text was computed last."""
    clean_wrapped = _perturbed_env("pick up the bowl", NoPerturbation())
    absent_wrapped = _perturbed_env("pick up the bowl", AbsentTarget(replacement="find nothing"))

    clean_wrapped.reset()
    absent_wrapped.reset()

    assert clean_wrapped.get_wrapper_attr("task_description") == "pick up the bowl"
    assert absent_wrapped.get_wrapper_attr("task_description") == "find nothing"
