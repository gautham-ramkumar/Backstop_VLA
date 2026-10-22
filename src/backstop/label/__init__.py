"""Labelling: auto proto-labels, and the pieces of the local review tools.

`signals.py` is pure numpy, testable without a recorded corpus. `extract.py`
does the I/O -- reading a recorded shard's dataset, privileged sidecar, and
provenance, and merging computed rows into `artifacts/corpus/labels.jsonl`
without clobbering anything a human has already reviewed.
"""

from __future__ import annotations
