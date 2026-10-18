"""Guard signals: pure numpy, one input array in, one score array out.

`action_chunk_disagreement` is the guard's signal (ADR-005): how much the K
flow-noise plans sampled at a step disagree with the one executed. The episode
score is its max over steps (`episode_score`), which matches a guard that
stops on the first crossing. `instructed_object_phrase` names the object the
presence gate looks for (`backstop.guard.presence`).

The other signals tried (temporal self-consistency, progress monitoring, a
visual precondition, chunk magnitude, speed-adjusted overlap) lost and were
removed; docs/experiments.md records them and where their code is kept.
"""

from __future__ import annotations

import re

import numpy as np


def action_chunk_disagreement(samples: np.ndarray) -> np.ndarray:
    """(T, K, chunk_size, action_dim) -> (T,) disagreement score.

    Mean absolute deviation of the K-1 extra flow-noise samples from the
    executed one (index 0, `adapter.select_action`'s own output -- see
    `record/loop.py`), averaged over the whole predicted chunk and every
    action dimension. Matches README's framing directly: "sample the policy K
    times under different flow noise; high spread means no confident opinion
    about what to do next." Index 0 is deliberately included as the
    reference, not dropped -- it is what the other K-1 are compared against,
    not one more sample among equals.

    Returns zeros when K < 2 (nothing to disagree with) rather than raising --
    a `clean`-arm episode recorded at K=1 is a legitimate input, just one with
    no signal.
    """
    if samples.ndim != 4:
        raise ValueError(f"expected (T, K, chunk_size, action_dim), got shape {samples.shape}")
    t, k = samples.shape[:2]
    if k < 2:
        return np.zeros(t, dtype=np.float64)
    executed = samples[:, 0:1]
    others = samples[:, 1:]
    return np.abs(others.astype(np.float64) - executed.astype(np.float64)).mean(axis=(1, 2, 3))


def instructed_object_phrase(task: str) -> str:
    """The object the instruction tells the arm to pick up.

    Corpus A only ever names two targets: ``black bowl`` (every spatial task,
    including paraphrases) and ``moka pot`` (the quarantined absent-object
    arm). Matching those phrases is the precondition's text side. LIBERO-Object
    names a different grocery item per task, always as ``pick up the <object>
    and place it ...``, so that pattern is the next fallback. A task that fits
    none of these returns the stripped instruction, so a new wording still
    produces a query instead of crashing.
    """
    lower = task.lower()
    if "moka pot" in lower:
        return "moka pot"
    if "black bowl" in lower:
        return "black bowl"
    match = _PICK_UP_OBJECT.search(lower)
    if match:
        return match.group(1).strip()
    return task.strip()


_PICK_UP_OBJECT = re.compile(r"pick up the (.+?) and place")


def episode_score(per_step_scores: np.ndarray) -> float:
    """Reduce a per-step signal to one number for the episode-level headline
    metric: the max. A guard that halts on the first threshold crossing
    behaves, for the yes/no "was this episode ever flagged" question the
    headline metric asks, exactly as if it looked at the max over the whole
    episode -- order doesn't matter for that question, only for lead time,
    which waits on the taxonomy's onset ranges (ADR-004)."""
    if per_step_scores.size == 0:
        return 0.0
    return float(np.max(per_step_scores))
