"""Recovery: what to do when the guard fires (docs/recovery.md).

`reselect.py` scores whether the K sampled chunks offer a different plan
(the offline pre-test). `reposition.py` is the scripted lift-and-back-off
motion. Neither reads privileged sim state (ADR-003); the recovery runner
may use sim state only to set up a branch, never to decide anything.
"""

from __future__ import annotations
