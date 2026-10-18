"""Visual precondition: is the instructed object in the first frame?

The guard's check is a per-object gate (`backstop.guard.presence`) on an
open-vocabulary detector's score (`vision.detector_scores`, OWLv2). Each
object's line is set from frames that show it, so the gate never needs to
see the object missing to be calibrated.

Reference frames, all set before any missing-object frame is scored:

- `black bowl`: first frames of the spatial calibration successes (ADR-004 split)
- `moka pot`: rendered LIBERO-10 scenes (`scripts/render_reference_frames.py`);
  corpus A never shows one
- each grocery item: the frames of a seeded half of the object-suite scenes
  that contain it (split by scene, since one task's runs share a layout)

Then four questions:

1. The guard's job: the 30 `language_absent` runs name a moka pot that is not
   there. How many does the moka-pot line stop at step 0?
2. False stops on the guard's own runs: spatial held-out good runs and the
   object-suite runs not used as references, each checked for its own
   instructed object, which is always present.
3. The detector's limits, same name, object present vs absent:
   across scenes (a grocery item asked of a spatial scene), and within one
   scene family (an object-suite scene that lacks that item, beside ones
   that have it). Scene contents come from the LIBERO BDDL files and are
   ground truth for scoring only, never a guard input. The `distractor`
   family adds a milk carton to spatial scenes, so those count as milk present.
4. With `--baselines`, the CLIP and SmolVLM2 scores this replaced, on the
   same 5% spatial line as before.

    PYTHONPATH=src python scripts/evaluate_presence.py
    PYTHONPATH=src python scripts/evaluate_presence.py --baselines   # +5 min
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from backstop.env.libero import scene_objects
from backstop.guard.calibrate import calibrate_threshold
from backstop.guard.presence import PresenceGate
from backstop.guard.scoring import auroc, fit_stopwatch_from_rows
from backstop.guard.signals import instructed_object_phrase
from backstop.guard.split import EpisodeKey, stratified_split
from backstop.guard.stopwatch import stop_step
from backstop.guard.traces import OBJECT_SHARD, episode_key, load_jsonl, series_map, spatial_pool
from backstop.guard.vision import (
    CLIP_MODEL,
    detector_score_matrix,
    detector_scores,
    first_agentview_frame,
)
from backstop.label.extract import resolve_dataset_root

VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Instruct"
SCENE_PROMPTS = ("a table", "a robot arm", "a floor", "an object")


def _crops(image: np.ndarray) -> list[np.ndarray]:
    """Whole frame, a 2x2 grid at 60%, and a 3x3 grid at 45%."""
    size = image.shape[0]
    out = [image]
    for n, scale in ((2, 0.6), (3, 0.45)):
        side = int(size * scale)
        stride = (size - side) // (n - 1)
        for i in range(n):
            for j in range(n):
                out.append(image[i * stride : i * stride + side, j * stride : j * stride + side])
    return out


def clip_scores(
    images: list[np.ndarray], phrases: list[str], device: str
) -> dict[str, list[float]]:
    import torch  # noqa: PLC0415
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    processor: Any = cast(Any, CLIPProcessor).from_pretrained(CLIP_MODEL)
    model: Any = cast(Any, CLIPModel).from_pretrained(CLIP_MODEL).to(device).eval()

    def _norm(features: Any) -> Any:
        features = getattr(features, "pooler_output", features)
        return features / features.norm(dim=-1, keepdim=True)

    def _text(texts: list[str]) -> Any:
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            return _norm(model.get_text_features(**inputs))

    unique = sorted(set(phrases))
    text = _text([f"a photo of a {phrase}" for phrase in unique])
    row_of = {phrase: i for i, phrase in enumerate(unique)}
    scene = _text([f"a photo of {prompt}" for prompt in SCENE_PROMPTS]).mean(0)
    scene = scene / scene.norm()

    out: dict[str, list[float]] = {"clip_whole": [], "clip_contrast": [], "clip_crops": []}
    for image, phrase in zip(images, phrases, strict=True):
        inputs = processor(images=_crops(image), return_tensors="pt").to(device)
        with torch.no_grad():
            features = _norm(model.get_image_features(**inputs))
        target = features @ text[row_of[phrase]]
        contrast = target - features @ scene
        out["clip_whole"].append(-float(target[0]))
        out["clip_contrast"].append(-float(contrast[0]))
        out["clip_crops"].append(-float(contrast.max()))
    return out


def vlm_scores(images: list[np.ndarray], phrases: list[str], device: str) -> list[float]:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    processor: Any = AutoProcessor.from_pretrained(VLM_MODEL)
    model: Any = (
        cast(Any, AutoModelForImageTextToText)
        .from_pretrained(VLM_MODEL, dtype=torch.bfloat16)
        .to(device)
    )
    model.eval()
    tokenizer = processor.tokenizer
    yes = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("Yes", " Yes", "yes")]
    no = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("No", " No", "no")]
    out: list[float] = []
    for image, phrase in zip(images, phrases, strict=True):
        question = f"Is there a {phrase} in this image? Answer yes or no."
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[Image.fromarray(image)], return_tensors="pt")
        inputs = inputs.to(device, torch.bfloat16)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        out.append(float(torch.logsumexp(log_probs[no], 0) - torch.logsumexp(log_probs[yes], 0)))
    return out


GROCERIES = (
    "alphabet soup",
    "bbq sauce",
    "butter",
    "chocolate pudding",
    "cream cheese",
    "ketchup",
    "milk",
    "orange juice",
    "salad dressing",
    "tomato sauce",
)
NAMES = ("black bowl", "moka pot", *GROCERIES)
BUDGET = 0.05


def _scene_objects(rows: list[dict[str, Any]]) -> list[set[str]]:
    """Objects in each corpus frame's scene, from the LIBERO task files."""
    spatial = set(scene_objects("libero_spatial", 0)[1])
    by_task = dict(scene_objects("libero_object", task_id) for task_id in range(10))
    out = []
    for row in rows:
        if row["shard"] == OBJECT_SHARD:
            out.append(set(by_task[row["task"]]))
        elif row["shard"] == "distractor":
            out.append(spatial | {"milk"})
        else:
            out.append(set(spatial))
    return out


def _rate(flags: np.ndarray) -> dict[str, float | int]:
    return {"n": int(len(flags)), "flagged": int(flags.sum())}


def gate_report(
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    objects: list[set[str]],
    reference_scores: dict[str, np.ndarray],
    split_calibration: frozenset[EpisodeKey],
    split_eval: frozenset[EpisodeKey],
    seed: int,
) -> dict[str, object]:
    column = {name: j for j, name in enumerate(NAMES)}
    phrases = [instructed_object_phrase(row["task"]) for row in rows]
    key = [EpisodeKey(row["shard"], row["episode_index"]) for row in rows]
    spatial_family = np.array([row["shard"] != OBJECT_SHARD for row in rows])

    references: dict[str, np.ndarray] = {}
    used_as_reference = np.zeros(len(rows), dtype=bool)
    bowl = [
        i
        for i, row in enumerate(rows)
        if row["success"] and key[i] in split_calibration and row["shard"] != "language_absent"
    ]
    references["black bowl"] = matrix[bowl, column["black bowl"]]
    used_as_reference[bowl] = True
    references["moka pot"] = reference_scores["moka pot"]
    rng = np.random.default_rng(seed)
    held_out_present: dict[str, np.ndarray] = {}
    for name in GROCERIES:
        # Split by scene, not by frame: the 10 runs of one object-suite task
        # open on near-identical layouts, so a frame split would put copies
        # of the same scene on both sides and flatter the held-out rate.
        frames = np.flatnonzero(~spatial_family & np.array([name in o for o in objects]))
        tasks = sorted({str(rows[i]["task"]) for i in frames})
        reference_tasks = set(rng.permutation(tasks)[: len(tasks) // 2].tolist())
        in_reference = np.array([str(rows[i]["task"]) in reference_tasks for i in frames])
        references[name] = matrix[frames[in_reference], column[name]]
        held_out_present[name] = frames[~in_reference]
        instructed = [i for i in frames[in_reference] if phrases[i] == name]
        used_as_reference[instructed] = True
    gate = PresenceGate.fit(references, BUDGET)

    def verdict(i: int, name: str) -> str:
        return gate.check(name, float(matrix[i, column[name]]))

    absent = [i for i, row in enumerate(rows) if row["shard"] == "language_absent"]
    spatial_good = [
        i
        for i, row in enumerate(rows)
        if row["success"] and key[i] in split_eval and row["shard"] != "language_absent"
    ]
    object_runs = [
        i for i, row in enumerate(rows) if row["shard"] == OBJECT_SHARD and not used_as_reference[i]
    ]
    guard = {
        "missing_moka_pot_stopped": _rate(
            np.array([verdict(i, phrases[i]) == "absent" for i in absent])
        ),
        "spatial_held_out_good_runs_stopped": _rate(
            np.array([verdict(i, phrases[i]) == "absent" for i in spatial_good])
        ),
        "object_suite_runs_stopped": _rate(
            np.array([verdict(i, phrases[i]) == "absent" for i in object_runs])
        ),
        "missing_moka_pot_scores": [float(matrix[i, column["moka pot"]]) for i in absent],
        "moka_pot_reference_scores": [float(v) for v in references["moka pot"]],
    }

    limits: dict[str, dict[str, object]] = {}
    for name in NAMES:
        j = column[name]
        has = np.array([name in o for o in objects])
        line = gate.lines[name]
        entry: dict[str, object] = {
            "line": line.line,
            "n_reference": line.n_reference,
            "false_stop_bound": line.bound,
        }
        if name in held_out_present:
            present = matrix[held_out_present[name], j]
            entry["held_out_present"] = _rate(present < line.line)
        else:
            present = references[name]
        # Frames that lack the object, split by whether the scene is the
        # kind that holds it. Moka pot is never in the corpus: every frame
        # lacks it, and the spatial ones include the 30 language_absent runs.
        if name in GROCERIES:
            scopes = {"other_scene": ~has & spatial_family, "same_scene": ~has & ~spatial_family}
        elif name == "black bowl":
            scopes = {"other_scene": ~has & ~spatial_family}
        else:
            scopes = {"spatial_scene": spatial_family, "object_scene": ~spatial_family}
        for scope, mask in scopes.items():
            missing = matrix[mask, j]
            entry[scope] = {**_rate(missing < line.line), "auroc": auroc(present, missing)}
        limits[name] = entry
    return {"guard": guard, "per_object": limits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/guard/traces.jsonl"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--eval-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("artifacts/guard/presence/reference_frames.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/guard/presence/report.json"))
    parser.add_argument("--eval-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", action="store_true", help="also score CLIP and SmolVLM2")
    args = parser.parse_args()

    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_jsonl(args.traces)
    images = []
    for row in rows:
        shard, episode = episode_key(row)
        root, _ = resolve_dataset_root(args.eval_root, shard, args.lerobot_root)
        images.append(first_agentview_frame(root, episode))
    print(f"scoring {len(images)} first frames x {len(NAMES)} object names on {device}")
    matrix = detector_score_matrix(images, list(NAMES))

    stored = np.load(args.references)
    reference_meta = json.loads(str(stored["meta"]))
    moka = [i for i, meta in enumerate(reference_meta) if "moka pot" in meta["objects"]]
    reference_scores = {
        "moka pot": detector_scores([stored["images"][i] for i in moka], ["moka pot"] * len(moka))
    }

    spatial = spatial_pool(rows)
    split = stratified_split(
        [(*episode_key(row), bool(row["success"])) for row in spatial],
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    objects = _scene_objects(rows)
    report: dict[str, object] = {
        "seed": args.seed,
        "budget": BUDGET,
        "detector": "google/owlv2-base-patch16-ensemble",
        **gate_report(
            rows, matrix, objects, reference_scores, split.calibration, split.eval, args.seed
        ),
    }
    report["headline_guard_on_missing"] = headline_guard_on_missing(
        rows, split.calibration, args.eval_fraction, args.seed
    )
    _print_gate(report)

    if args.baselines:
        report["baselines"] = baselines(rows, images, split, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


def headline_guard_on_missing(
    rows: list[dict[str, Any]],
    calibration: frozenset[EpisodeKey],
    eval_fraction: float,
    seed: int,
) -> dict[str, object]:
    """When ADR-005's headline guard stops the missing-object runs, for comparison.

    The gate decides at step 0. Disagreement OR stopwatch, fit on the same
    spatial calibration successes, has to wait for the policy to waver.
    """
    calibration_rows = [
        row for row in rows if EpisodeKey(row["shard"], row["episode_index"]) in calibration
    ]
    absent = [row for row in rows if row["shard"] == "language_absent"]
    series = series_map(calibration_rows + absent, "action_chunk_disagreement")
    rule = fit_stopwatch_from_rows(calibration_rows, series, BUDGET)
    steps = [stop_step(series[episode_key(row)], int(row["n_steps"]), rule) for row in absent]
    caught = np.array([step for step in steps if step is not None], dtype=np.float64)
    return {
        "guard": "action_chunk_disagreement OR stopwatch",
        "eval_fraction": eval_fraction,
        "seed": seed,
        "stop_steps": [None if step is None else int(step) for step in steps],
        "median_stop_step": float(np.median(caught)) if len(caught) else None,
        "signal_before_clock": int((caught < rule.stop_step).sum()),
    }


def _print_gate(report: dict[str, Any]) -> None:
    guard = report["guard"]
    for label, item in (
        ("missing moka pot stopped at step 0", "missing_moka_pot_stopped"),
        ("spatial held-out good runs stopped", "spatial_held_out_good_runs_stopped"),
        ("object-suite runs stopped", "object_suite_runs_stopped"),
    ):
        print(f"{label}: {guard[item]['flagged']}/{guard[item]['n']}")
    headline = report["headline_guard_on_missing"]
    print(
        f"headline guard on the same missing-object runs: median stop step "
        f"{headline['median_stop_step']:.0f}, signal before clock on "
        f"{headline['signal_before_clock']}/{len(headline['stop_steps'])}"
    )
    print("per object (line, then missing caught / missing frames, AUROC):")
    for name, entry in report["per_object"].items():
        cells = [f"line {entry['line']:.3f} from {entry['n_reference']}"]
        if "held_out_present" in entry:
            row = entry["held_out_present"]
            cells.append(f"present wrongly flagged {row['flagged']}/{row['n']}")
        for scope in ("other_scene", "same_scene", "spatial_scene", "object_scene"):
            if scope in entry:
                row = entry[scope]
                cells.append(f"{scope} {row['flagged']}/{row['n']} (AUROC {row['auroc']:.2f})")
        print(f"  {name:18s} " + ", ".join(cells))


def baselines(
    rows: list[dict[str, Any]], images: list[np.ndarray], split: Any, device: str
) -> dict[str, object]:
    """The replaced scores, on one 5% line set from spatial successes."""
    phrases = [instructed_object_phrase(row["task"]) for row in rows]
    scores = clip_scores(images, phrases, device)
    scores["vlm_no_minus_yes"] = vlm_scores(images, phrases, device)

    def _in(row: dict[str, Any], keys: frozenset[EpisodeKey]) -> bool:
        return EpisodeKey(row["shard"], row["episode_index"]) in keys

    out: dict[str, object] = {}
    for name, values in scores.items():
        value = np.asarray(values, dtype=np.float64)
        cal = [i for i, row in enumerate(rows) if row["success"] and _in(row, split.calibration)]
        line = calibrate_threshold(value[cal], BUDGET)
        absent = [i for i, row in enumerate(rows) if row["shard"] == "language_absent"]
        present = [i for i, row in enumerate(rows) if row["shard"] != "language_absent"]
        held = [i for i, row in enumerate(rows) if row["success"] and _in(row, split.eval)]
        obj = [i for i, row in enumerate(rows) if row["shard"] == OBJECT_SHARD]
        out[name] = {
            "line": line,
            "absent_caught": int((value[absent] >= line).sum()),
            "n_absent": len(absent),
            "spatial_held_out_success_stops": int((value[held] >= line).sum()),
            "n_spatial_held_out_success": len(held),
            "object_suite_stops": int((value[obj] >= line).sum()),
            "n_object_suite": len(obj),
            "auroc_absent_vs_present": auroc(value[absent], value[present]),
        }
        print(f"baseline {name}: {out[name]}")
    return out


if __name__ == "__main__":
    main()
