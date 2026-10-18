"""Week 3: guard signals and the headline metric (ADR-001, ADR-004).

`signals.py` is pure numpy and holds all four README signals. Three of them
read columns already in the dataset (chunks, executed action, proprio). The
visual precondition reads an image-text similarity computed outside this
package. `split.py` and `calibrate.py` implement the calibrate-on-one-half /
measure-on-the-other methodology ADR-004 documents. `persist.py` is the
stop rule that requires a score to stay high, chosen without onset labels.
`step_model.py` is a per-step logistic fit on the calibration half.
`extract.py` does the I/O. No privileged sim state anywhere in this package
-- ADR-003 forbids it as guard input, and a signal that needed it couldn't
ship.
"""

from __future__ import annotations
