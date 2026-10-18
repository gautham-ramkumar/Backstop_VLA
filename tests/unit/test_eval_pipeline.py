"""Policy loading never drops a pinned revision."""

from __future__ import annotations

from typing import Any

import pytest

# The pipeline imports the rollout loop and its gymnasium-based wrapper (`sim`
# extra). The sim CI job runs this module.
pytest.importorskip("gymnasium")

from backstop.eval.pipeline import load_pinned_config

REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def test_the_pin_is_passed_through() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def loader(path: str, **kwargs: Any) -> str:
        calls.append((path, kwargs))
        return "cfg"

    assert load_pinned_config(loader, "org/policy", REVISION) == "cfg"
    assert calls == [("org/policy", {"revision": REVISION})]


def test_no_pin_loads_without_a_revision() -> None:
    calls: list[dict[str, Any]] = []
    load_pinned_config(lambda path, **kwargs: calls.append(kwargs), "org/policy", None)
    assert calls == [{}]


def test_a_loader_that_rejects_the_pin_is_an_error_not_a_silent_retry() -> None:
    calls: list[dict[str, Any]] = []

    def loader(path: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        if "revision" in kwargs:
            raise TypeError("unexpected keyword argument 'revision'")
        return "HEAD config"

    with pytest.raises(TypeError, match="revision"):
        load_pinned_config(loader, "org/policy", REVISION)
    assert calls == [{"revision": REVISION}]  # never retried at HEAD
