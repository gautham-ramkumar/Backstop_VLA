"""The shared HTTP base of the two local tools, and the guard viewer's routes.

A real `ThreadingHTTPServer` on a free port, with a fake clip source in place of
ffmpeg, so the bytes on the wire are what a browser would get.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import guard_viewer
from backstop.label.web import JsonRequestHandler, parse_range

CLIP = bytes(range(256)) * 4  # 1024 bytes, each position distinguishable


class FakeClips:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.write_bytes(CLIP)

    def clip(self, shard: str, episode_index: int) -> Path:
        return self.path


@contextmanager
def serving(handler: type[JsonRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def fetch(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 - localhost only
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("bytes=0-99", (0, 99)),
        ("bytes=100-", (100, 1023)),
        ("bytes=-", (0, 1023)),
        ("bytes=1000-5000", (1000, 1023)),
    ],
)
def test_parse_range(header: str, expected: tuple[int, int]) -> None:
    assert parse_range(header, total=1024) == expected


@pytest.fixture
def viewer(tmp_path: Path) -> Iterator[str]:
    """The guard viewer's real handler, on trace rows as evaluate_guard writes them now."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.html").write_text("<html>guard</html>")
    (assets / "guard.js").write_text("// js")
    (tmp_path / "secret.txt").write_text("not served")
    traces = tmp_path / "traces.jsonl"
    row: dict[str, Any] = {
        "shard": "clean",
        "episode_index": 3,
        "success": False,
        "n_steps": 2,
        "task": "pick up the black bowl",
        "scores": {"action_chunk_disagreement": 0.4},
        "per_step": {"action_chunk_disagreement": [0.1, 0.4]},
    }
    traces.write_text(json.dumps(row) + "\n")

    class Handler(guard_viewer.Handler):
        pass

    Handler.state = guard_viewer.GuardState(traces)
    Handler.assets_dir = assets
    Handler.clips = FakeClips(tmp_path / "clip.mp4")  # type: ignore[assignment]
    with serving(Handler) as url:
        yield url


def test_index_lists_episodes_without_the_series(viewer: str) -> None:
    status, _, body = fetch(f"{viewer}/api/index")
    assert status == 200
    assert json.loads(body) == [
        {
            "shard": "clean",
            "episode_index": 3,
            "success": False,
            "scores": {"action_chunk_disagreement": 0.4},
        }
    ]


def test_trace_route_returns_the_row_or_a_json_error(viewer: str) -> None:
    status, _, body = fetch(f"{viewer}/api/trace?shard=clean&episode=3")
    assert status == 200 and json.loads(body)["per_step"]["action_chunk_disagreement"] == [0.1, 0.4]
    status, _, body = fetch(f"{viewer}/api/trace?shard=clean&episode=9")
    assert status == 404 and json.loads(body) == {"error": "no trace"}
    status, _, _ = fetch(f"{viewer}/api/trace?shard=clean")
    assert status == 400


def test_assets_are_served_and_cannot_escape_their_folder(viewer: str) -> None:
    status, headers, body = fetch(f"{viewer}/")
    assert (status, headers["Content-Type"], body) == (200, "text/html", b"<html>guard</html>")
    status, headers, _ = fetch(f"{viewer}/static/guard.js")
    assert (status, headers["Content-Type"]) == (200, "application/javascript")
    status, _, body = fetch(f"{viewer}/static/%2E%2E/secret.txt")
    assert status == 404 and b"not served" not in body


def test_video_is_served_whole_or_by_range(viewer: str) -> None:
    status, headers, body = fetch(f"{viewer}/api/video?shard=clean&episode=3")
    assert (status, body, headers["Accept-Ranges"]) == (200, CLIP, "bytes")
    status, headers, body = fetch(
        f"{viewer}/api/video?shard=clean&episode=3", headers={"Range": "bytes=10-19"}
    )
    assert status == 206
    assert body == CLIP[10:20]
    assert headers["Content-Range"] == "bytes 10-19/1024"
    status, _, _ = fetch(f"{viewer}/api/video?shard=clean")
    assert status == 400


def test_unknown_route_is_a_json_404(viewer: str) -> None:
    status, headers, body = fetch(f"{viewer}/api/nope")
    assert status == 404 and headers["Content-Type"] == "application/json"
    assert json.loads(body) == {"error": "no route for /api/nope"}
