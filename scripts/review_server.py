#!/usr/bin/env python
"""Review tool (Week 2 review tool, part 2 of 2): local HTML page + tiny server.

Serves `scripts/review_assets/index.html` -- episode video, the auto-labeller's
proposal, category buttons (free text at this stage: stage-1 open coding has
not happened yet, so there is no frozen codebook to constrain input to -- see
Week2_plan.md's `taxonomy` entry), and an onset-range scrubber -- and writes
confirmed/overridden labels back into `artifacts/corpus/labels.jsonl` through a
few JSON endpoints. Stdlib only, on purpose: this project's dependency surface
is `pydantic` + `pyyaml` plus the `sim` extra, and a review page does not need
a web framework to justify adding one.

    uv run python scripts/review_server.py               # http://localhost:8765
    uv run python scripts/review_server.py --port 9000

Video: LeRobot v3 concatenates every episode's frames into one video per
chunk, so "one episode's clip" does not exist on disk -- it is cut on first
request with ffmpeg and cached under `artifacts/corpus/video_cache/`. Source
clips are AV1 (SVT-AV1, this project's video codec); re-encoded to H.264 for
the cache so every browser's <video> tag can play them without a codec probe.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from backstop.label.extract import (
    load_episode_video_range,
    load_labels,
    mark_reviewed,
    resolve_dataset_root,
    write_labels,
)

ASSETS_DIR = Path(__file__).resolve().parent / "review_assets"
CONTROL_FPS = 20  # backstop.record's control rate; verify_corpus.py asserts the same.


class Store:
    """All server state. One instance, held by the request handler class --
    `http.server` builds a fresh handler per request, so anything that must
    survive between requests lives here, not on `self`.
    """

    def __init__(
        self,
        *,
        labels_path: Path,
        lerobot_root: Path,
        eval_root: Path,
        video_cache_dir: Path,
    ) -> None:
        self.labels_path = labels_path
        self.lerobot_root = lerobot_root
        self.eval_root = eval_root
        self.video_cache_dir = video_cache_dir
        self.video_cache_dir.mkdir(parents=True, exist_ok=True)

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

    def video_clip(self, shard: str, episode_index: int) -> Path:
        cache_path = self.video_cache_dir / shard / f"{episode_index:06d}.mp4"
        if cache_path.is_file():
            return cache_path
        dataset_path, _provenance = resolve_dataset_root(self.eval_root, shard, self.lerobot_root)
        source_video, from_ts, to_ts = load_episode_video_range(dataset_path, episode_index)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".mp4.partial")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            # `-ss`/`-to` *after* `-i` seek on the decoded output, not the input --
            # slower (decodes from the start of the source) but resets PTS to zero
            # in the trimmed file. Seeking before `-i` is faster but keeps the
            # original absolute timestamps -- worth avoiding on its own even
            # though it turned out not to be the cause of the real bug below.
            "-ss",
            f"{from_ts:.3f}",
            "-to",
            f"{to_ts:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            # moov (the index) at the front, not wherever ffmpeg's default muxer
            # order puts it -- see `_serve_video`'s Range support for the other
            # half of why Chrome reported `video.seekable` as [0, 0] otherwise.
            "-movflags",
            "+faststart",
            "-f",
            "mp4",  # `.partial` suffix defeats ffmpeg's extension-based muxer guess
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0 or not tmp_path.is_file():
            detail = result.stderr[-2000:]
            raise RuntimeError(f"ffmpeg failed for {shard}/{episode_index}: {detail}")
        tmp_path.replace(cache_path)
        return cache_path


class Handler(BaseHTTPRequestHandler):
    store: Store  # set on the class before serving

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

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        try:
            if parts.path == "/" or parts.path == "/index.html":
                self._serve_asset("index.html")
            elif parts.path.startswith("/static/"):
                self._serve_asset(parts.path.removeprefix("/static/"))
            elif parts.path == "/api/shards":
                self._send_json(self.store.shards())
            elif parts.path == "/api/episodes":
                shard = query.get("shard", [None])[0]
                if not shard:
                    self._send_error_json(400, "missing ?shard=")
                    return
                rows = sorted(
                    (r for r in self.store.episodes() if r["shard"] == shard),
                    key=lambda r: r["episode_index"],
                )
                self._send_json(rows)
            elif parts.path == "/api/video":
                self._serve_video(query)
            else:
                self._send_error_json(404, f"no route for {parts.path}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the reviewer, not swallowed
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        try:
            body = self._read_json_body()
            if parts.path == "/api/label":
                onset = body.get("onset_range")
                row = self.store.save_review(
                    shard=body["shard"],
                    episode_index=int(body["episode_index"]),
                    category=body["category"],
                    onset_range=tuple(onset) if onset else None,
                    notes=body.get("notes", ""),
                )
                self._send_json(row)
            elif parts.path == "/api/label/reset":
                row = self.store.reset_review(
                    shard=body["shard"], episode_index=int(body["episode_index"])
                )
                self._send_json(row)
            else:
                self._send_error_json(404, f"no route for {parts.path}")
        except KeyError as exc:
            self._send_error_json(400, f"missing field {exc}")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _serve_asset(self, name: str) -> None:
        path = (ASSETS_DIR / name).resolve()
        if ASSETS_DIR.resolve() not in path.parents or not path.is_file():
            self._send_error_json(404, f"no asset {name}")
            return
        content_type = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
        }.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_video(self, query: dict[str, list[str]]) -> None:
        """Range requests matter here, not just as an optimization: Chrome ties
        `HTMLMediaElement.seekable` to whether the server supports `Range`, not
        to how much of the file happens to be buffered. Without this, the
        onset scrubber's "seek" buttons silently no-op -- `duration` and
        `buffered` both look correct, `currentTime` assignment is just ignored.
        """
        shard = query.get("shard", [None])[0]
        episode = query.get("episode", [None])[0]
        if not shard or episode is None:
            self._send_error_json(400, "missing ?shard=&episode=")
            return
        clip_path = self.store.video_clip(shard, int(episode))
        total = clip_path.stat().st_size
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(total))
            self.end_headers()
            with clip_path.open("rb") as handle:
                self.wfile.write(handle.read())
            return

        start, end = self._parse_range(range_header, total)
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with clip_path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(end - start + 1))

    @staticmethod
    def _parse_range(range_header: str, total: int) -> tuple[int, int]:
        # "bytes=START-END", END optional (meaning "to the end of the file").
        spec = range_header.removeprefix("bytes=")
        start_str, _, end_str = spec.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else total - 1
        return start, min(end, total - 1)


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

    Handler.store = Store(
        labels_path=labels_path,
        lerobot_root=Path(args.lerobot_root),
        eval_root=Path(args.eval_root),
        video_cache_dir=Path(args.video_cache_dir),
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review tool: http://127.0.0.1:{args.port}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
