"""Replay saved frames through SmolVLA and store the last-layer summary.

The simulator is not started. Each recorded step is one prefix pass: cameras,
instruction, and arm state in, one vector out. Flow-matching is not run, so
this is the inside-the-model number the action signals do not have.

The first step of the first episode is checked against the stored policy
action. A mismatch means the replayed frame is not what the policy saw, and
the run stops.

Usage:
    PYTHONPATH=src python scripts/extract_internal.py --check-action --limit-steps 2
    PYTHONPATH=src python scripts/extract_internal.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from backstop.config import PipelineConfig
from backstop.env.libero import build_libero_env_config
from backstop.eval.pipeline import load_lerobot_policy
from backstop.guard.internal import pool_prefix
from backstop.label.extract import load_episode_meta, load_run_provenance, resolve_dataset_root
from backstop.policy.adapter import SmolVLAAdapter


def _observation(item: dict, task: str):
    import torch

    def _image(value: torch.Tensor) -> torch.Tensor:
        image = value.detach()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        if float(image.max()) > 1.5:
            image = image.float() / 255.0
        return image

    def _state(value: torch.Tensor) -> torch.Tensor:
        state = value.detach().float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return state

    return {
        "observation.images.image": _image(item["observation.images.image"]),
        "observation.images.image2": _image(item["observation.images.image2"]),
        "observation.state": _state(item["observation.state"]),
        "task": [task],
    }


def _prefix_vector(policy, batch) -> np.ndarray:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    prefix_embs, prefix_pad, prefix_att = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    attention = make_att_2d_masks(prefix_pad, prefix_att)
    position_ids = torch.cumsum(prefix_pad, dim=1) - 1
    (prefix_out, _), _cache = policy.model.vlm_with_expert.forward(
        attention_mask=attention,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    hidden = prefix_out[0].detach().float().cpu().numpy()
    mask = prefix_pad[0].detach().cpu().numpy().astype(bool)
    return pool_prefix(hidden, mask).astype(np.float32)


def _open_dataset(root: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=str(root))


def _episode_span(dataset, episode_index: int) -> tuple[int, int]:
    row = dataset.meta.episodes[episode_index]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def _check_action(adapter, post, batch, seed: int, step: int, stored: Any) -> float:
    import torch

    noise = adapter.noise_for(episode_seed=seed, step=step, index=0)
    with torch.inference_mode():
        raw = adapter.select_action(batch, noise=noise)
    executed: Any = post(raw)
    array = executed.detach().float().cpu().numpy().reshape(-1)
    if hasattr(stored, "detach"):
        stored = stored.detach().float().cpu().numpy()
    stored = np.asarray(stored, dtype=np.float32).reshape(-1)
    width = min(array.shape[0], stored.shape[0])
    return float(np.max(np.abs(array[:width] - stored[:width])))


def _shards(eval_root: Path) -> list[str]:
    names = []
    for path in sorted(eval_root.glob("*/run.json")):
        payload = json.loads(path.read_text())
        shard = payload.get("shard")
        if shard:
            names.append(str(shard))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/record/spatial_clean_k4.yaml"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/guard/internal"))
    parser.add_argument("--shard", action="append", default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--check-action", action="store_true")
    args = parser.parse_args()

    import torch

    cfg = PipelineConfig.from_yaml(args.config)
    env_cfg = build_libero_env_config(cfg.env, cfg.eval)
    policy, preprocessor, postprocessor, _policy_cfg = load_lerobot_policy(cfg, env_cfg)
    policy.eval()
    adapter = SmolVLAAdapter(
        policy,
        n_action_steps=cfg.policy.n_action_steps,
        num_steps=cfg.policy.num_steps,
        batch_samples=False,
    )
    shards = args.shard or _shards(args.eval_root)
    print(f"shards: {', '.join(shards)}", flush=True)
    started = time.perf_counter()
    done_steps = 0
    for shard in shards:
        provenance = load_run_provenance(args.eval_root, shard)
        dataset_path, _provenance = resolve_dataset_root(args.eval_root, shard, args.lerobot_root)
        meta = load_episode_meta(dataset_path)
        repo_id = f"local/{dataset_path.name}"
        print(f"opening {shard} ({len(meta)} episodes)", flush=True)
        dataset = _open_dataset(dataset_path, repo_id)
        seed0 = int(provenance.get("seed", 0))
        episodes = sorted(meta)
        if args.episode is not None:
            episodes = [args.episode]
        for episode_index in episodes:
            out_path = args.output / shard / f"{episode_index:04d}.npy"
            if out_path.is_file() and args.limit_steps is None and args.episode is None:
                done_steps += int(meta[episode_index]["length"])
                continue
            task = str(meta[episode_index]["task"])
            start, end = _episode_span(dataset, episode_index)
            if args.limit_steps is not None:
                end = min(end, start + args.limit_steps)
            vectors: list[np.ndarray] = []
            adapter.reset()
            tick = time.perf_counter()
            for step, frame in enumerate(range(start, end)):
                item = cast(dict[str, Any], dataset[frame])
                observation = _observation(item, task)
                batch = preprocessor(observation)
                first = step == 0 and episode_index == episodes[0] and shard == shards[0]
                if args.check_action and first:
                    stored = item.get("action_policy", item.get("action"))
                    gap = _check_action(
                        adapter,
                        postprocessor,
                        batch,
                        seed0 + episode_index,
                        step,
                        stored,
                    )
                    print(f"action check max abs diff {gap:.6f}", flush=True)
                    if gap > 0.05:
                        raise SystemExit(
                            "replayed action does not match the recording; features not written"
                        )
                    adapter.reset()
                with torch.inference_mode():
                    vectors.append(_prefix_vector(policy, batch))
            array = np.stack(vectors).astype(np.float16)
            if args.limit_steps is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, array)
            done_steps += array.shape[0]
            elapsed = time.perf_counter() - tick
            rate = array.shape[0] / elapsed if elapsed else 0.0
            print(
                f"{shard} ep {episode_index}: {array.shape[0]} steps, "
                f"{array.shape[1]} dims, {rate:.2f} steps/s, "
                f"total {done_steps} in {(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )
            if args.limit_steps is not None:
                return


if __name__ == "__main__":
    main()
