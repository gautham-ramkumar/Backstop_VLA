"""Guard signals: pure numpy, one input array in, one score array out.

The four signals named in README.md:

- action-chunk disagreement reads `action_chunk_samples` (K flow-noise plans).
- temporal self-consistency reads `action_chunk` at successive steps.
- progress monitoring reads `observation.state` and the executed `action`.
- visual precondition turns an image-text similarity (computed by the caller,
  never from privileged state) into an absence score.

Two further scores read the same `action_chunk` column. `action_chunk_magnitude`
is the size of the plan. `velocity_normalized_consistency` is the overlap
gap divided by that size. None of them read the taxonomy. Episode-level
reduction is `episode_score`, the max, which matches a guard that halts on
the first threshold crossing.
Lead time needs onset ranges and is not computed here.
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


def action_chunk_magnitude(chunks: np.ndarray) -> np.ndarray:
    """(T, chunk_size, action_dim) -> (T,) mean size of the planned actions.

    ActProbe's action-chunk magnitude: a large command is a different warning
    from a plan that merely disagrees with itself. Each step of the chunk is
    reduced to its L2 norm, then averaged, so a longer chunk is not a larger
    score by construction.
    """
    if chunks.ndim != 3:
        raise ValueError(f"expected (T, chunk_size, action_dim), got shape {chunks.shape}")
    step_norm = np.linalg.norm(chunks.astype(np.float64), axis=-1)
    return step_norm.mean(axis=1)


def velocity_normalized_consistency(chunks: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """(T, chunk_size, action_dim) -> (T,) overlap disagreement relative to motion size.

    The same absolute change of plan is a small slip in a fast move and a
    reversal in a slow one. VLA-FAIL divides the overlap gap by the size of
    the planned motion so those two are not scored the same. The last step
    has no successor, so its score is 0.
    """
    if chunks.ndim != 3:
        raise ValueError(f"expected (T, chunk_size, action_dim), got shape {chunks.shape}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    t, chunk_size, _action_dim = chunks.shape
    scores = np.zeros(t, dtype=np.float64)
    if t < 2 or chunk_size < 2:
        return scores
    previous = chunks[:-1, 1:].astype(np.float64)
    following = chunks[1:, :-1].astype(np.float64)
    gap = np.abs(previous - following).mean(axis=(1, 2))
    scale = np.abs(previous).mean(axis=(1, 2))
    scores[:-1] = gap / (scale + eps)
    return scores


def temporal_self_consistency(chunks: np.ndarray) -> np.ndarray:
    """(T, chunk_size, action_dim) -> (T,) disagreement between overlapping plans.

    The chunk at step t plans the next `chunk_size` actions. The chunk at
    step t+1 plans the same future, shifted by one. Where those overlap, a
    policy that is not changing its mind agrees with itself. The score is the
    mean absolute difference of that overlap. The last step has no successor,
    so its score is 0.
    """
    if chunks.ndim != 3:
        raise ValueError(f"expected (T, chunk_size, action_dim), got shape {chunks.shape}")
    t, chunk_size, _action_dim = chunks.shape
    scores = np.zeros(t, dtype=np.float64)
    if t < 2 or chunk_size < 2:
        return scores
    previous = chunks[:-1, 1:].astype(np.float64)
    following = chunks[1:, :-1].astype(np.float64)
    scores[:-1] = np.abs(previous - following).mean(axis=(1, 2))
    return scores


def progress_monitoring(
    state: np.ndarray,
    action: np.ndarray,
    *,
    horizon: int = 5,
    still_m: float = 1e-4,
    open_width_m: float = 0.08,
) -> np.ndarray:
    """(T, 8) proprio and (T, 7) executed action -> (T,) stall score.

    Two stalls, summed, both from the dataset columns a guard is allowed to
    see (ADR-003: no privileged object pose):

    - The commanded translation (`action[:3]`) and the observed end-effector
      step disagree. Direction is compared with cosine similarity so the
      score does not depend on action units, which are not meters. A command
      with no observed motion scores 1.
    - The gripper closes (finger width drops) while the end-effector stays
      put over `horizon` steps. Width is `state[:, 6] - state[:, 7]`, the
      same layout `label.signals` measured. The close is scaled by the open
      width (~0.08 m) so a full close-on-nothing adds about 1.

    The last step has no following observation, so its score is 0.
    """
    state = np.asarray(state)
    action = np.asarray(action)
    if state.ndim != 2 or state.shape[1] < 8:
        raise ValueError(f"expected state (T, 8), got shape {state.shape}")
    if action.ndim != 2 or action.shape[0] != state.shape[0] or action.shape[1] < 3:
        raise ValueError(
            f"expected action (T, >=3) matching state length {state.shape[0]}, got {action.shape}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    length = state.shape[0]
    scores = np.zeros(length, dtype=np.float64)
    if length < 2:
        return scores

    eef = state[:, :3].astype(np.float64)
    width = state[:, 6].astype(np.float64) - state[:, 7].astype(np.float64)
    commanded = action[:-1, :3].astype(np.float64)
    observed = np.diff(eef, axis=0)
    cmd_norm = np.linalg.norm(commanded, axis=1)
    obs_norm = np.linalg.norm(observed, axis=1)
    dot = np.sum(commanded * observed, axis=1)
    both = (cmd_norm > 1e-8) & (obs_norm > still_m)
    cosine = np.ones(length - 1, dtype=np.float64)
    cosine[both] = dot[both] / (cmd_norm[both] * obs_norm[both])
    stalled = (cmd_norm > 1e-8) & (obs_norm <= still_m)
    motion = np.where(stalled, 1.0, np.clip(1.0 - cosine, 0.0, 2.0))
    motion = np.where(cmd_norm > 1e-8, motion, 0.0)

    close = np.maximum(0.0, width[:-1] - width[1:])
    close_frac = close / open_width_m if open_width_m > 0 else close
    # How far the hand moves from this step across `horizon` steps.
    steps = np.arange(length - 1)
    future = np.minimum(length - 1, steps + horizon)
    travel = np.linalg.norm(eef[future] - eef[steps], axis=1)
    empty_close = np.where(travel <= still_m, close_frac, 0.0)

    scores[:-1] = motion + empty_close
    return scores


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


def visual_precondition(similarity: np.ndarray) -> np.ndarray:
    """(T,) presence score -> (T,) absence score.

    The caller scores the agentview for the instructed object
    (`instructed_object_phrase`): a detector's box score, or earlier a CLIP
    cosine similarity. Either is high when the object looks present. The
    guard score is the negation, so a missing object ranks above a visible
    one and the usual ``score >= threshold`` rule still means "flag". The
    score is not rescaled: calibration sets the threshold.
    """
    sim = np.asarray(similarity, dtype=np.float64)
    if sim.ndim != 1:
        raise ValueError(f"expected similarity (T,), got shape {sim.shape}")
    return -sim


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
