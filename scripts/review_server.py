#!/usr/bin/env python
"""Label review tool: a local page for confirming or overriding the auto-labels.

Serves `scripts/review_assets/index.html`: the episode video, the
auto-labeller's proposal, free-text category buttons (there is no frozen
codebook yet), and an onset-range scrubber. Confirmed or overridden labels are
written back to `artifacts/corpus/labels.jsonl` through a few JSON endpoints.
Video clips and the HTTP plumbing are shared with the guard viewer
(`backstop.label.web`).

    uv run python scripts/review_server.py               # http://localhost:8765
    uv run python scripts/review_server.py --port 9000
"""

from __future__ import annotations

import argparse
import contextlib
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from backstop.corpus.layout import ShardLayout
from backstop.label.extract import load_labels, mark_reviewed, write_labels
from backstop.label.web import JsonRequestHandler, VideoClips

ASSETS_DIR = Path(__file__).resolve().parent / "review_assets"


class Store:
    """The labels file, read and written per request. One instance, held by the
    handler class: `http.server` builds a fresh handler per request."""

    def __init__(self, labels_path: Path) -> None:
        self.labels_path = labels_path

    def episodes(self) -> list[dict]:
        return list(load_labels(self.labels_path).values())

    def shards(self) -> list[dict]:
        by_shard: dict[str, list[dict]] = {}
        for row in self.episodes():
            by_shard.setdefault(row["shard"], []).append(row)
        return [
            {
                "shard": shard,
                "n_episodes": len(rows),
                "n_reviewed": sum(1 for r in rows if r["review"]["status"] == "reviewed"),
                "n_failure": sum(1 for r in rows if not r["success"]),
            }
            for shard, rows in sorted(by_shard.items())
        ]

    def episode_row(self, shard: str, episode_index: int) -> dict | None:
        for row in self.episodes():
            if row["shard"] == shard and row["episode_index"] == episode_index:
                return row
        return None

    def _write(self, rows: dict[tuple[str, int], dict]) -> None:
        ordered = sorted(rows.values(), key=lambda r: (r["shard"], r["episode_index"]))
        write_labels(self.labels_path, ordered)

    def save_review(
        self,
        *,
        shard: str,
        episode_index: int,
        category: str,
        onset_range: tuple[int, int] | None,
        notes: str,
    ) -> dict:
        rows = load_labels(self.labels_path)
        key = (shard, episode_index)
        if key not in rows:
            raise KeyError(f"no label row for {key}; run scripts/auto_label.py first")
        rows[key] = mark_reviewed(
            rows[key], category=category, onset_range=onset_range, notes=notes
        )
        self._write(rows)
        return rows[key]

    def reset_review(self, *, shard: str, episode_index: int) -> dict:
        """Reopen a reviewed row -- the blind 20% re-code needs this, and the
        plan explicitly allows the codebook to be reopened once."""
        rows = load_labels(self.labels_path)
        key = (shard, episode_index)
        if key not in rows:
            raise KeyError(f"no label row for {key}")
        row = dict(rows[key])
        row["review"] = {
            "status": "pending",
            "category": None,
            "onset_range": None,
            "notes": None,
            "reviewed_at": None,
        }
        rows[key] = row
        self._write(rows)
        return row


class Handler(JsonRequestHandler):
    store: Store  # set on the class before serving

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        try:
            if parts.path == "/" or parts.path == "/index.html":
                self.serve_asset("index.html")
            elif parts.path.startswith("/static/"):
                self.serve_asset(parts.path.removeprefix("/static/"))
            elif parts.path == "/api/shards":
                self.send_json(self.store.shards())
            elif parts.path == "/api/episodes":
                shard = query.get("shard", [None])[0]
                if not shard:
                    self.send_json_error(400, "missing ?shard=")
                    return
                rows = sorted(
                    (r for r in self.store.episodes() if r["shard"] == shard),
                    key=lambda r: r["episode_index"],
                )
                self.send_json(rows)
            elif parts.path == "/api/video":
                self.serve_video(query)
            else:
                self.send_json_error(404, f"no route for {parts.path}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the reviewer, not swallowed
            self.send_json_error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        try:
            body = self.read_json_body()
            if parts.path == "/api/label":
                onset = body.get("onset_range")
                row = self.store.save_review(
                    shard=body["shard"],
                    episode_index=int(body["episode_index"]),
                    category=body["category"],
                    onset_range=tuple(onset) if onset else None,
                    notes=body.get("notes", ""),
                )
                self.send_json(row)
            elif parts.path == "/api/label/reset":
                row = self.store.reset_review(
                    shard=body["shard"], episode_index=int(body["episode_index"])
                )
                self.send_json(row)
            else:
                self.send_json_error(404, f"no route for {parts.path}")
        except KeyError as exc:
            self.send_json_error(400, f"missing field {exc}")
        except Exception as exc:  # noqa: BLE001
            self.send_json_error(500, f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--labels-path", default="artifacts/corpus/labels.jsonl")
    parser.add_argument("--lerobot-root", default="data/lerobot")
    parser.add_argument("--eval-root", default="artifacts/eval")
    parser.add_argument("--video-cache-dir", default="artifacts/corpus/video_cache")
    args = parser.parse_args(argv)

    labels_path = Path(args.labels_path)
    if not labels_path.is_file():
        print(f"{labels_path} does not exist yet -- run scripts/auto_label.py first.")
        return 1

    Handler.store = Store(labels_path)
    Handler.assets_dir = ASSETS_DIR
    layout = ShardLayout(lerobot_root=Path(args.lerobot_root), eval_root=Path(args.eval_root))
    Handler.clips = VideoClips(layout, Path(args.video_cache_dir))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review tool: http://127.0.0.1:{args.port}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
