"""scripts/snapshot_outputs.py: what is hashed, and what counts as a difference."""

from __future__ import annotations

import json
from pathlib import Path

import snapshot_outputs as snap


def test_the_out_path_and_the_code_version_do_not_count(tmp_path: Path) -> None:
    def write_rule(out: Path, sha: str) -> Path:
        path = out / "guard" / "frozen" / "rule_05.json"
        path.parent.mkdir(parents=True)
        payload = {"rule": {"line": 0.25}, "git_sha": sha, "git_dirty": False, "traces": str(out)}
        path.write_text(json.dumps(payload))
        return path

    before, after = tmp_path / "before", tmp_path / "after"
    one = snap.normalised(write_rule(before, "aaa"), before)
    two = snap.normalised(write_rule(after, "bbb"), after)
    assert one == two
    assert b"git_sha" not in one and snap.OUT_PLACEHOLDER.encode() in one


def test_a_real_change_still_counts(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    path = out / "report.json"
    path.write_text(json.dumps({"saved_steps_mean": 13.37}))
    first = snap.normalised(path, out)
    path.write_text(json.dumps({"saved_steps_mean": 13.38}))
    assert snap.normalised(path, out) != first


def test_manifest_skips_logs_and_itself(tmp_path: Path) -> None:
    (tmp_path / snap.LOGS).mkdir()
    (tmp_path / snap.LOGS / "evaluate_guard.log").write_text("wrote ...")
    (tmp_path / snap.MANIFEST).write_text("{}")
    (tmp_path / "guard").mkdir()
    (tmp_path / "guard" / "traces.jsonl").write_text('{"shard": "clean"}\n')
    assert list(snap.manifest(tmp_path)) == ["guard/traces.jsonl"]


def test_differences_name_every_kind_of_change() -> None:
    expected = {"a": "1", "b": "2", "c": "3"}
    actual = {"a": "1", "b": "changed", "d": "4"}
    assert snap.differences(expected, actual) == ["missing: c", "new: d", "changed: b"]
    assert snap.differences(expected, dict(expected)) == []


def test_every_step_writes_only_inside_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    labels, report = out / "labels.jsonl", out / "evaluation.md"
    plan = snap.steps(out, labels=labels, report=report, with_presence=True)
    assert [argv[0] for argv in plan] == [
        "evaluate_guard.py",
        "evaluate_stopwatch.py",
        "freeze_guard.py",
        "pretest_reselect.py",
        "auto_label.py",
        "evaluate_presence.py",
    ]
    assert all((snap.SCRIPTS / argv[0]).is_file() for argv in plan)
    for argv in plan:
        paths = [Path(arg) for arg in argv[1:] if not arg.startswith("--")]
        assert all(path.is_relative_to(out) for path in paths), argv
