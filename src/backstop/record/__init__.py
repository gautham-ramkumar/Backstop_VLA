"""Recording episodes: the rollout loop and its pieces, the privileged sidecar
writer, and per-stage timing.

`timing` imports nothing outside the standard library, so the corpus-sizing
arithmetic it produces can be tested without the sim extra.
"""
