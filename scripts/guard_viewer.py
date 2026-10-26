#!/usr/bin/env python
"""Local viewer for guard scores: play an episode beside its per-step signal.

Reads `artifacts/guard/traces.jsonl` (written by `scripts/evaluate_guard.py`)
and shows each episode's video next to its disagreement series. It does not
show onset marks; those are in the label review tool. Video clips and the HTTP
plumbing are shared with that tool (`backstop.label.web`), on a different port
so both can run at once.

    uv run python scripts/guard_viewer.py               # http://localhost:8766
"""

from __future__ import annotations

import argparse
import contextlib
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from backstop.corpus.episode import EpisodeKey
from backstop.corpus.layout import ShardLayout
from backstop.guard.traces import TraceRow, load_traces
from backstop.label.web import JsonRequestHandler, VideoClips

ASSETS_DIR = Path(__file__).resolve().parent / "guard_assets"


class GuardState:
    """The trace rows, loaded once at startup."""

    def __init__(self, traces_path: Path) -> None:
        self.rows = load_traces(traces_path)
        self.by_key: dict[EpisodeKey, TraceRow] = {row.key: row for row in self.rows}

    def index(self) -> list[dict[str, object]]:
        """What the sidebar lists: one entry per episode, without the per-step series."""
        return [
            {
                "shard": row.shard,
                "episode_index": row.episode_index,
                "success": row.success,
                "scores": row.scores,
            }
            for row in self.rows
        ]


class Handler(JsonRequestHandler):
    state: GuardState  # set on the class before serving

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        try:
            if parts.path in {"/", "/index.html"}:
                self.serve_asset("index.html")
            elif parts.path.startswith("/static/"):
                self.serve_asset(parts.path.removeprefix("/static/"))
            elif parts.path == "/api/index":
                self.send_json(self.state.index())
            elif parts.path == "/api/trace":
                self._send_trace(query)
            elif parts.path == "/api/video":
                self.serve_video(query)
            else:
                self.send_json_error(404, f"no route for {parts.path}")
        except Exception as exc:  # noqa: BLE001 - shown in the page, not swallowed
            self.send_json_error(500, f"{type(exc).__name__}: {exc}")

    def _send_trace(self, query: dict[str, list[str]]) -> None:
        shard = query.get("shard", [None])[0]
        episode = query.get("episode", [None])[0]
        if shard is None or episode is None:
            self.send_json_error(400, "missing ?shard=&episode=")
            return
        row = self.state.by_key.get(EpisodeKey(shard, int(episode)))
        if row is None:
            self.send_json_error(404, "no trace")
            return
        self.send_json(row.to_json())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument(
        "--video-cache-dir", type=Path, default=Path("artifacts/corpus/video_cache")
    )
    args = parser.parse_args(argv)

    if not args.traces.is_file():
        print(f"{args.traces} does not exist yet -- run scripts/evaluate_guard.py first.")
        return 1
    Handler.state = GuardState(args.traces)
    Handler.assets_dir = ASSETS_DIR
    layout = ShardLayout(lerobot_root=args.lerobot_root, eval_root=args.eval_root)
    Handler.clips = VideoClips(layout, args.video_cache_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Guard viewer: http://127.0.0.1:{args.port}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
