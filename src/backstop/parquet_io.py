"""Shared parquet-reading glue for `backstop.label` and `backstop.guard`.

Both packages read the same LeRobot v3 dataset shape and hit the same two
parquet quirks: pyarrow's `to_pydict()` output needs no further coercion for
scalar columns, but nested array columns (`action_chunk`, `observation.state`,
...) come back as Python lists that may already be shaped right or may be
flattened, depending on how the writer stored them. Imported lazily by
callers so the pure-numpy halves of those packages stay importable without
pyarrow (`scripts/verify_corpus.py` established this convention first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_parquet_pydict(path: Path, columns: list[str] | None = None) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    return pq.read_table(path, columns=columns).to_pydict()


def as_array(column: list[Any], trailing: tuple[int, ...]) -> np.ndarray:
    """Normalise a parquet list column to `(N, *trailing)`, whether pyarrow
    handed it back flat or already nested (`scripts/verify_corpus.py`'s
    original helper, promoted here once a second caller needed it)."""
    flat = np.array([np.asarray(row, dtype=np.float32).reshape(-1) for row in column])
    return flat.reshape((flat.shape[0], *trailing))
