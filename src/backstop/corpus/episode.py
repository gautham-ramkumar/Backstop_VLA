"""The one key every episode is looked up by."""

from __future__ import annotations

from typing import NamedTuple


class EpisodeKey(NamedTuple):
    """Hashes and compares like the `(shard, episode_index)` tuple it replaces."""

    shard: str
    episode_index: int
