#!/usr/bin/env python
"""Local viewer for guard scores. Does not show onset marks.

Onset marks wait on the taxonomy. This page plays the episode and draws the
per-step signal that `scripts/evaluate_guard.py` already computed. It reuses
the review tool's video cutting, on a different port so labeling stays on
8765.

    uv run python scripts/guard_viewer.py
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import review_server
from review_server import Store

GUARD_ASSETS = Path(__file__).resolve().parent / "guard_assets"


class GuardState:
    def __init__(self, traces_path: Path, store: Store) -> None:
        self.store = store
        lines = traces_path.read_text().splitlines()
        self.rows = [json.loads(line) for line in lines if line.strip()]
        self.by_key = {(row["shard"], row["episode_index"]): row for row in self.rows}

    def index(self) -> list[dict]:
        return [
            {
                "shard": row["shard"],
                "episode_index": row["episode_index"],
                "success": row["success"],
                "scores": row["scores"],
                "trust": row["trust"],
            }
            for row in self.rows
        ]


class Handler(BaseHTTPRequestHandler):
    state: GuardState

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        try:
            if parts.path in {"/", "/index.html"}:
                self._serve(GUARD_ASSETS / "index.html", "text/html")
            elif parts.path == "/static/guard.js":
                self._serve(GUARD_ASSETS / "guard.js", "application/javascript")
            elif parts.path == "/api/index":
                self._send_json(self.state.index())
            elif parts.path == "/api/trace":
                shard = query.get("shard", [None])[0]
                episode = query.get("episode", [None])[0]
                if shard is None or episode is None:
                    self._send_error_json(400, "missing ?shard=&episode=")
                    return
                row = self.state.by_key.get((shard, int(episode)))
                if row is None:
                    self._send_error_json(404, "no trace")
                    return
                self._send_json(row)
            elif parts.path == "/api/video":
                self.store = self.state.store
                self._parse_range = review_server.Handler._parse_range
                # Borrowed handler method: works because this Handler carries the same
                # attributes. Moving the shared HTTP code into a base class removes this.
                review_server.Handler._serve_video(self, query)  # type: ignore[arg-type]
            else:
                self._send_error_json(404, f"no route for {parts.path}")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _serve(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--labels-path", type=Path, default=Path("artifacts/corpus/labels.jsonl"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument(
        "--video-cache-dir", type=Path, default=Path("artifacts/corpus/video_cache")
    )
    args = parser.parse_args(argv)

    if not args.traces.is_file():
        print(f"{args.traces} does not exist yet -- run scripts/evaluate_guard.py first.")
        return 1
    store = Store(
        labels_path=args.labels_path,
        lerobot_root=args.lerobot_root,
        eval_root=args.eval_root,
        video_cache_dir=args.video_cache_dir,
    )
    Handler.state = GuardState(args.traces, store)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Guard viewer: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
