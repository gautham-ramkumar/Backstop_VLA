"""Does re-selection have anything to select? Per-step measures over K samples.

Re-selection means executing a different one of the K sampled chunks when the
guard fires. It is dead in two cases:

1. All K agree, so any other chunk is the same plan.
2. They disagree, but no more than in good runs at the same point. Then the
   spread is ordinary flow noise. The policy draws a fresh chunk every step
   (`n_action_steps: 1`), so picking another sample once is no different
   from what the next step does anyway.

Three numbers per step:

- `outlier_ratio`: chunk 0's mean distance to the others over the mean
  distance among the others. Chunk 0 is drawn the same way as the rest, so
  this should sit near 1. It checks that assumption; it is not a finding.
- `spread`: the largest pairwise distance between samples.
- `gripper_split`: 1 where the samples disagree on opening vs closing the
  gripper over the next `GRIPPER_HORIZON` actions -- a different decision,
  not just a different path.
"""

from __future__ import annotations

import numpy as np

GRIPPER_DIM = 6
GRIPPER_HORIZON = 10
RESELECT_METRICS = ("outlier_ratio", "spread", "gripper_split")


def reselect_metrics(samples: np.ndarray) -> dict[str, np.ndarray]:
    """(T, K, chunk, dim) -> {metric: (T,)} for the three measures above."""
    if samples.ndim != 4:
        raise ValueError(f"expected (T, K, chunk, dim), got shape {samples.shape}")
    t, k = samples.shape[:2]
    if k < 3:
        raise ValueError(f"need K >= 3 to compare chunk 0 with the others, got {k}")
    x = samples.astype(np.float64)
    # pair[t, i, j]: mean absolute difference between samples i and j at step t.
    pair = np.abs(x[:, :, None] - x[:, None, :]).mean(axis=(3, 4))
    to_zero = pair[:, 0, 1:].mean(axis=1)
    among = pair[:, 1:, 1:].sum(axis=(1, 2)) / ((k - 1) * (k - 2))
    gripper = np.sign(x[:, :, :GRIPPER_HORIZON, GRIPPER_DIM].mean(axis=2))
    return {
        "outlier_ratio": to_zero / np.maximum(among, 1e-12),
        "spread": pair.reshape(t, -1).max(axis=1),
        "gripper_split": (gripper.min(axis=1) != gripper.max(axis=1)).astype(np.float64),
    }
