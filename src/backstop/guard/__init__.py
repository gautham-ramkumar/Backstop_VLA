"""The guard: action-chunk disagreement OR a stopwatch (ADR-005).

- `signals.py`: the disagreement signal, pure numpy.
- `stopwatch.py`: the stop rule, both triggers set from good runs only.
- `online.py`: the same rule step by step, for use inside a rollout.
- `calibrate.py`, `split.py`: calibrate on one half, measure on the other (ADR-004).
- `presence.py`, `vision.py`: the per-object detector gate, kept beside the guard.
- `extract.py`, `traces.py`, `scoring.py`, `summarize.py`: reading recordings and
  trace rows, and the numbers the reports print.

No privileged sim state anywhere in this package: ADR-003 forbids it as guard
input, and a signal that needed it could not ship. The signals that lost are
written up in docs/experiments.md.
"""

from __future__ import annotations
