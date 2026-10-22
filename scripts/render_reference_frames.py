"""Render opening frames of LIBERO scenes as presence references.

The presence gate (`backstop.guard.presence`) sets each object's line from
frames that show that object. Corpus A never shows a moka pot, so the
`language_absent` arm cannot be scored without frames from somewhere else.
LIBERO-10 tasks 2 and 8 contain moka pots. This renders their opening frames
straight from the simulator: no policy, no rollout, one reset per LIBERO init
state. Each frame is rotated 180 degrees to match how the recorded dataset
stores the agentview (checked against `clean/0`: mean pixel difference 1.7
after the rotation, 70 before).

The objects in each scene are read from the task's BDDL file and stored
beside the frames. They label the references and are never a guard input.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=src \\
        python scripts/render_reference_frames.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from backstop.env.libero import ensure_libero_config, scene_objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--n", type=int, default=20, help="init states per task")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/guard/presence/reference_frames.npz")
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    ensure_libero_config()
    from lerobot.envs import make_env  # noqa: PLC0415
    from lerobot.envs.configs import LiberoEnv  # noqa: PLC0415

    config = LiberoEnv(
        task=args.suite, task_ids=args.task_ids, init_states=True, control_mode="relative"
    )
    envs = make_env(config, n_envs=1, use_async_envs=False)
    frames: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    for task_id, vec_env in envs[args.suite].items():
        language, objects = scene_objects(args.suite, task_id)
        for init in range(args.n):
            observation, _ = vec_env.reset(seed=init)
            pixels = observation["pixels"]
            image = pixels["image"] if isinstance(pixels, dict) else pixels
            frames.append(np.ascontiguousarray(np.asarray(image[0])[::-1, ::-1]))
            meta.append(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "init": init,
                    "task": language,
                    "objects": objects,
                }
            )
        vec_env.close()
        print(f"{args.suite} task {task_id}: {args.n} frames, objects {objects}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, images=np.stack(frames), meta=json.dumps(meta))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
