"""Visual precondition: is the instructed object in the agentview?

CLIP embeds the first frame of an episode and the object phrase from the
instruction (`signals.instructed_object_phrase`). The guard score is the
negated cosine similarity (`signals.visual_precondition`): higher means the
object looks absent. One frame per episode, because a precondition is a
property of the scene the arm is about to act in, not of a later step.

Privileged object poses are not read. Torch and the model weights are
imported inside each call so unit tests of the other signals never load them.

CLIP turned out not to work for this (docs/evaluation.md, "Visual
precondition"): whole-frame similarity is not comparable across object names
and the objects are small. `detector_scores` uses an open-vocabulary detector
instead, and `backstop.guard.presence` gives each object its own line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backstop.label.extract import load_episode_video_range

CLIP_MODEL = "openai/clip-vit-base-patch32"
DETECTOR_MODEL = "google/owlv2-base-patch16-ensemble"


def _pooled(output: object):
    """Newer transformers returns a pooled output object; older ones a tensor."""
    import torch  # noqa: PLC0415

    if torch.is_tensor(output):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        pooled = getattr(output, "text_embeds", None)
    if pooled is None:
        pooled = getattr(output, "image_embeds", None)
    if not torch.is_tensor(pooled):
        raise TypeError(f"CLIP features were {type(output).__name__}, not a tensor")
    return pooled


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


def absence_scores(images: list[np.ndarray], phrases: list[str]) -> np.ndarray:
    """Negated CLIP cosine similarity, one score per image.

    Text is encoded once per distinct phrase. Images are encoded in batches.
    CUDA is used when it is available; otherwise CPU.
    """
    if len(images) != len(phrases):
        raise ValueError(f"{len(images)} images and {len(phrases)} phrases")
    if not images:
        return np.zeros(0, dtype=np.float64)

    import torch  # noqa: PLC0415
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor: Any = CLIPProcessor.from_pretrained(CLIP_MODEL)
    loaded: Any = CLIPModel.from_pretrained(CLIP_MODEL)
    model: Any = loaded.to(device)
    model.eval()

    unique = list(dict.fromkeys(phrases))
    text_inputs = processor(
        text=[f"a photo of a {phrase}" for phrase in unique],
        return_tensors="pt",
        padding=True,
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.no_grad():
        text_features = _pooled(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_index = {phrase: i for i, phrase in enumerate(unique)}

    image_features: list[torch.Tensor] = []
    batch_size = 32
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        image_inputs = processor(images=batch, return_tensors="pt")
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        with torch.no_grad():
            features = _pooled(model.get_image_features(**image_inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        image_features.append(features)
    image_matrix = torch.cat(image_features, dim=0)

    phrase_ids = torch.tensor([text_index[phrase] for phrase in phrases], device=device)
    selected_text = text_features[phrase_ids]
    similarity = (image_matrix * selected_text).sum(dim=-1)
    return (-similarity).detach().cpu().numpy().astype(np.float64)


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
