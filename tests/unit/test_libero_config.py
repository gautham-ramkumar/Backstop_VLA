"""LIBERO benchmark paths must be resolved, never guessed.

A config naming a non-existent assets directory suppresses LIBERO's Hub download.
The sim still runs and still renders -- the objects are simply missing, so every
episode times out at 0% success with no error anywhere. That cost a full smoke
run to find; these tests make it a one-second failure instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("libero")

from backstop.env.libero import ensure_libero_config  # noqa: E402


def _load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def test_every_required_path_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    payload = _load(ensure_libero_config())
    for key in ("bddl_files", "init_states", "assets"):
        assert Path(payload[key]).is_dir(), f"{key} -> {payload[key]} does not exist"


def test_assets_directory_is_populated(tmp_path, monkeypatch) -> None:
    """An empty assets dir renders scenes without objects -- existence is not enough."""
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    assets = Path(_load(ensure_libero_config())["assets"])
    assert any(assets.rglob("*.xml")) or any(assets.rglob("*.stl")), (
        f"{assets} has no scene or mesh files"
    )


def test_stale_config_with_bad_assets_is_repaired(tmp_path, monkeypatch) -> None:
    """The poisoned path lives in $HOME, so it survives a clean checkout."""
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    import yaml

    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump({"assets": str(tmp_path / "nope")}))
    assets = Path(_load(ensure_libero_config())["assets"])
    assert assets.is_dir() and assets != tmp_path / "nope"
