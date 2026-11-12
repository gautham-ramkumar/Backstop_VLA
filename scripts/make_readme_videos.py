#!/usr/bin/env python
"""The README's MP4 clips, cut from the recorded clean shard.

- `success_ep000.mp4`, `failure_ep001.mp4`: the two clips the GIFs show, as
  H.264 at the recording's 20 Hz.
- `guard_ep001.mp4` (and a GIF of it): the same failure with the frozen 5% guard watching. The
  camera view sits beside the policy's action-chunk disagreement score as it
  unfolds, with the guard's line, the step it fires on, and the stopwatch's
  step 140. Everything drawn comes from the recording: the score is the
  traced signal and the firing step is `stop_step` with the frozen rule, the
  same numbers `artifacts/guard/frozen/rule_05_triggers.json` holds.

    uv run python scripts/make_readme_videos.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backstop.corpus.layout import TRACES, ShardLayout  # noqa: E402
from backstop.guard.online import load_rule  # noqa: E402
from backstop.guard.stopwatch import stop_step  # noqa: E402
from backstop.guard.traces import load_traces  # noqa: E402
from backstop.label.web import VideoClips  # noqa: E402
from backstop.schema import CONTROL_FPS  # noqa: E402

SIGNAL = "action_chunk_disagreement"
SHARD = "clean"
FREEZE_S = 2.0  # hold on the frame the guard fires, so it can be read


def read_frames(path: Path) -> np.ndarray:
    import imageio.v3 as iio  # noqa: PLC0415

    return np.stack(list(iio.imiter(path)))


def encode(frames: list[np.ndarray], path: Path) -> None:
    height, width = frames[0].shape[:2]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(CONTROL_FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip
    subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True)


def guard_frames(video: np.ndarray, score: np.ndarray, line: float, clock: int, fired: int):
    n = len(score)
    steps = np.arange(n)
    fig, (cam, plot) = plt.subplots(
        1, 2, figsize=(9.6, 4.2), dpi=100, gridspec_kw={"width_ratios": [1, 1.25]}
    )
    fig.subplots_adjust(left=0.02, right=0.97, top=0.86, bottom=0.14, wspace=0.12)
    image = cam.imshow(video[0])
    cam.set_axis_off()
    plot.set_xlim(0, n)
    plot.set_ylim(0, max(float(score.max()), line) * 1.15)
    plot.axhline(line, color="#c0392b", ls="--", lw=1.2, label=f"guard line ({line:.3f})")
    plot.axvline(clock, color="#7f8c8d", ls=":", lw=1.2, label=f"stopwatch (step {clock})")
    plot.set_xlabel("step (20 Hz)")
    plot.set_title("action-chunk disagreement (K=4 samples)", fontsize=10)
    plot.legend(loc="upper right", fontsize=8, frameon=False)
    (trace,) = plot.plot([], [], color="#2c3e50", lw=1.4)
    (after,) = plot.plot([], [], color="#2c3e50", lw=1.0, alpha=0.3)
    marker = plot.scatter([], [], s=50, color="#c0392b", zorder=5)
    fire_line = plot.axvline(fired, color="#c0392b", lw=1.2, visible=False)
    title = fig.suptitle("", fontsize=12, weight="bold")

    def render(t: int, stopped: bool) -> np.ndarray:
        image.set_data(video[min(t, len(video) - 1)])
        upto = min(t, fired) if stopped else t
        trace.set_data(steps[: upto + 1], score[: upto + 1])
        after.set_data(
            steps[fired : t + 1] if stopped else [], score[fired : t + 1] if stopped else []
        )
        if stopped:
            marker.set_offsets([[fired, score[fired]]])
            fire_line.set_visible(True)
            title.set_text(
                f"Guard fires at step {fired} ({fired / CONTROL_FPS:.1f} s); "
                f"the stopwatch alone waits to step {clock}"
                if t <= fired
                else f"Without the guard the run stalls on to step {n} "
                f"(step {t}; greyed = after the stop)"
            )
            title.set_color("#c0392b" if t <= fired else "#555555")
        else:
            title.set_text(f"Failing run, guard watching: step {t}")
            title.set_color("#222222")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # pyright: ignore[reportAttributeAccessIssue]  # Agg canvas

    frames = [render(t, False) for t in range(fired)]
    held = render(fired, True)
    frames += [held] * int(FREEZE_S * CONTROL_FPS)
    frames += [render(t, True) for t in range(fired + 1, n, 2)]  # the rest at 2x
    plt.close(fig)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    parser.add_argument("--rule", type=Path, default=Path("artifacts/guard/frozen/rule_05.json"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/corpus/video_cache"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    clips = VideoClips(ShardLayout(), args.cache)
    for name, index in (("success_ep000", 0), ("failure_ep001", 1)):
        shutil.copyfile(clips.clip(SHARD, index), args.out / f"{name}.mp4")

    rule = load_rule(args.rule)
    row = next(r for r in load_traces(TRACES) if r.shard == SHARD and r.episode_index == 1)
    score = row.series(SIGNAL)
    fired = stop_step(score, row.n_steps, rule)
    assert fired is not None and fired < rule.stop_step, f"guard does not beat the clock: {fired}"
    video = read_frames(clips.clip(SHARD, 1))
    guard = args.out / "guard_ep001.mp4"
    encode(guard_frames(video, score, rule.line, rule.stop_step, fired), guard)
    # GitHub plays a README's GIFs inline but only links its MP4s.
    palette = "fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    gif = guard.with_suffix(".gif")
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(guard), "-vf", palette, str(gif)], check=True
    )
    for path in sorted([*args.out.glob("*.mp4"), args.out / "guard_ep001.gif"]):
        print(f"{path}  {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
