"""What the two local browser tools share: episode video clips and a small JSON server.

`scripts/review_server.py` (label review) and `scripts/guard_viewer.py` (guard
scores) each serve a static page, a few JSON endpoints, and episode videos.
Their own routes and state stay in those scripts. This module holds the parts
they have in common:

- `VideoClips` cuts one episode's clip out of LeRobot's per-chunk video, once,
  and caches it as H.264 so every browser can play it.
- `JsonRequestHandler` is a `BaseHTTPRequestHandler` with JSON replies,
  traversal-safe static assets, and video served with HTTP Range support.

Stdlib only (plus ffmpeg on PATH for cutting clips): a local review page does
not need a web framework.
"""

from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from backstop.corpus.layout import ShardLayout
from backstop.corpus.read import load_episode_video_range

CONTENT_TYPES = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}


class VideoClips:
    """One episode's video as its own file, cut on first request and cached.

    LeRobot v3 concatenates every episode's frames into one video per chunk, so
    a single episode's clip does not exist on disk until it is cut out. Source
    clips are AV1; the cache is H.264 so a browser's <video> tag plays it
    without a codec probe.
    """

    def __init__(self, layout: ShardLayout, cache_dir: Path) -> None:
        self.layout = layout
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clip(self, shard: str, episode_index: int) -> Path:
        cache_path = self.cache_dir / shard / f"{episode_index:06d}.mp4"
        if cache_path.is_file():
            return cache_path
        source_video, from_ts, to_ts = load_episode_video_range(
            self.layout.dataset_root(shard), episode_index
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".mp4.partial")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            # `-ss`/`-to` after `-i` seek on the decoded output: slower, but the
            # trimmed file's timestamps start at zero, which the player needs.
            "-ss",
            f"{from_ts:.3f}",
            "-to",
            f"{to_ts:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            # The index (moov) at the front: with Range support below, this is
            # what makes Chrome report the clip as seekable.
            "-movflags",
            "+faststart",
            "-f",
            "mp4",  # the `.partial` suffix defeats ffmpeg's extension-based muxer guess
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0 or not tmp_path.is_file():
            detail = result.stderr[-2000:]
            raise RuntimeError(f"ffmpeg failed for {shard}/{episode_index}: {detail}")
        tmp_path.replace(cache_path)
        return cache_path


def parse_range(range_header: str, total: int) -> tuple[int, int]:
    """`bytes=START-END` -> (start, end), inclusive. END may be omitted (to the end)."""
    spec = range_header.removeprefix("bytes=")
    start_str, _, end_str = spec.partition("-")
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else total - 1
    return start, min(end, total - 1)


class JsonRequestHandler(BaseHTTPRequestHandler):
    """Base handler for the local tools. Subclasses add `do_GET`/`do_POST` routes.

    `http.server` builds a fresh handler per request, so anything that must live
    between requests is set on the subclass before serving: `assets_dir` for
    static files and `clips` for video.
    """

    assets_dir: Path
    clips: VideoClips

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json_error(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def serve_asset(self, name: str) -> None:
        """A file from `assets_dir`. Paths that resolve outside it are a 404."""
        root = self.assets_dir.resolve()
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            self.send_json_error(404, f"no asset {name}")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_video(self, query: dict[str, list[str]]) -> None:
        """`?shard=&episode=` as video/mp4, honouring a Range header.

        Range support is not just an optimization: Chrome ties
        `HTMLMediaElement.seekable` to whether the server supports Range, so
        without it a page's seek controls silently do nothing.
        """
        shard = query.get("shard", [None])[0]
        episode = query.get("episode", [None])[0]
        if not shard or episode is None:
            self.send_json_error(400, "missing ?shard=&episode=")
            return
        clip_path = self.clips.clip(shard, int(episode))
        total = clip_path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = parse_range(range_header, total) if range_header else (0, total - 1)
        self.send_response(206 if range_header else 200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "video/mp4")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with clip_path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(end - start + 1))
