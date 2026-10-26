"""Detector scores for the presence gate: is the instructed object in view?

`detector_scores` and `detector_score_matrix` score the first agentview frame
of an episode with an open-vocabulary detector (OWLv2) for an object name.
`backstop.guard.presence` turns those scores into a per-object gate. One frame
per episode, because a precondition is a property of the scene the arm is
about to act in, not of a later step.

Privileged object poses are not read. Torch and the model weights are imported
inside each call, so importing this module never loads them. The CLIP and
SmolVLM2 scores this replaced are recorded in docs/experiments.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backstop.corpus.read import load_episode_video_range

DETECTOR_MODEL = "google/owlv2-base-patch16-ensemble"


def first_agentview_frame(dataset_root: Path, episode_index: int) -> np.ndarray:
    """RGB uint8 frame at the episode's first timestamp in the shard video."""
    import av  # noqa: PLC0415

    video_path, start_s, _end_s = load_episode_video_range(dataset_root, episode_index)
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base) if stream.time_base else 1.0 / 20.0
        container.seek(int(start_s / time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time + 1e-3 >= start_s:
                return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"no frame at {start_s:.3f}s in {video_path}")
    finally:
        container.close()


def detector_score_matrix(images: list[np.ndarray], names: list[str]) -> np.ndarray:
    """Best OWLv2 box score for every name in every image, shape (images, names).

    One forward pass per image with all names as queries. Each name's score
    is computed against the boxes independently of the other names, so this
    equals scoring the names one at a time. Scores are in [0, 1] and are only
    meaningful against a line fit for the same name
    (`presence.fit_presence_line`). CUDA is used when it is available.
    """
    if not images or not names:
        return np.zeros((len(images), len(names)), dtype=np.float64)

    import torch  # noqa: PLC0415
    from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor: Any = Owlv2Processor.from_pretrained(DETECTOR_MODEL)
    loaded: Any = Owlv2ForObjectDetection.from_pretrained(DETECTOR_MODEL)
    model: Any = loaded.to(device)
    model.eval()

    queries = [f"a photo of a {name}" for name in names]
    out = np.zeros((len(images), len(names)), dtype=np.float64)
    for i, image in enumerate(images):
        inputs = processor(text=[queries], images=np.ascontiguousarray(image), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        out[i] = torch.sigmoid(logits).max(dim=0).values.cpu().numpy()
    return out


def detector_scores(images: list[np.ndarray], phrases: list[str]) -> np.ndarray:
    """Best OWLv2 box score for each image's own phrase, in [0, 1]."""
    if len(images) != len(phrases):
        raise ValueError(f"{len(images)} images and {len(phrases)} phrases")
    names = list(dict.fromkeys(phrases))
    matrix = detector_score_matrix(images, names)
    column = {name: j for j, name in enumerate(names)}
    return np.array([matrix[i, column[p]] for i, p in enumerate(phrases)], dtype=np.float64)
