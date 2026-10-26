"""Where recorded shards live and how to read them back.

The one package that knows the on-disk layout: `layout` finds a shard's run
record, dataset and recording config, `read` reads LeRobot datasets, and
`privileged` reads the sim-state sidecar. `guard` and `label` both read through
here, so neither imports the other.
"""
