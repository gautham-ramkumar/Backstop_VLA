# Engineering notes: problems we hit, and how they were found

Most defects in this project produced no error. They produced a plausible-looking dataset
or a reasonable-looking number. Each check in the repository exists because one of these
shipped or nearly did.

## Simulator and rendering

- **Missing scene assets, 0% success, no error.** The LIBERO wheel ships no meshes or
  textures; it downloads them only when no config file exists. A config naming a wrong
  assets path suppressed the download permanently. The simulator still ran and rendered,
  with the objects missing, and every episode timed out. `ensure_libero_config()` now
  resolves the path from LIBERO itself and fails loudly on a broken install.
- **Restored states that drift.** Restoring the recorded state (time, positions,
  velocities) and replaying the same actions reproduces the run to float noise under
  smooth motion, but drifts for several steps after rapid gripper chatter. Traced by
  elimination (actuator state and controller goals ruled out) to MuJoCo's contact-solver
  warm start, which the saved state does not include. It matters for branching near a
  grasp, and is documented rather than patched in vendored code.
- **A restored observation is not the recorded one.** After a physics step, MuJoCo's body
  and camera poses still describe the start of the last 2 ms substep. A restore recomputes
  them for the state itself, so images differ by under 0.1 grey level on average. The check
  for this (`scripts/verify_sim_state_roundtrip.py`) uses that measured tolerance instead
  of pixel equality.
- **Recovery outcomes are chaotic.** That fraction of a grey level is enough for a
  recovered run to take a different path: one episode succeeded when branched and failed
  when run live. A single guard-on run is one sample of what recovery does from a state.
- **Rendering was not the bottleneck.** Per-stage timing showed the control step was
  policy-bound (56% inference, 29% rendering), so the lever was sampling, not image size.

## Policy integration

- **Zeroed action chunks.** SmolVLA fills its action queue through a private
  `_get_action_chunk`, not the public `predict_action_chunk`, so the first chunk capture
  recorded zeros. The test fakes now reproduce that routing on purpose.
- **A signal that was identically zero.** At K=1 the sampler short-circuits and stores
  four copies of the executed chunk, so disagreement is exactly 0. The clean shard was
  re-recorded at K=4.
- **Sampling must not move the trajectory.** Flow noise is drawn from a dedicated
  generator keyed on episode, step and sample index, so the executed chunk is the same
  whatever K is. The K=4 re-record reproduced the K=1 baseline on all 100 episodes,
  outcome and step count alike.
- **Batched sampling changes the numbers, not the signal.** Unbatched, each extra sample
  cost a full forward pass and put the K=4 corpus at 32.7 GPU hours against a 20-hour
  budget. The extras now share one batched pass, but batch-size-dependent GPU kernels move
  individual chunk values by up to 64% relative. The chunk spread, which is the signal,
  agrees to 0.4%, and the executed action never goes through the batched path. Out of
  memory falls back to sequential sampling automatically.
- **Silent unit and version errors.** A failed unnormalize step logged a warning and stored
  chunks in policy space next to executed actions in env space. A rejected model revision
  silently fell back to the latest weights while provenance still recorded the pin. Both
  now raise.

## Recording and data integrity

- **Wrong state and upside-down video.** Rebuilding the state vector and images from the
  raw env stored rotation-matrix entries and frames flipped 180°. Both are now read
  straight from the processed observation the policy sees.
- **Wrong frame rate.** A hardcoded 30 fps (and LIBERO's 80 Hz render rate) would desync
  playback and every lead time in seconds. The dataset is written at the 20 Hz control
  rate, and recording refuses any other.
- **A re-run that destroyed hours of recording.** Opening a dataset always deleted the
  old one. Recording now refuses an existing shard unless the config says to resume or
  overwrite, and a resume checks the schema on disk first.
- **Perturbations that did nothing.** Dimming the named scene lights barely changed the
  image (mean pixel 120 → 87), because MuJoCo's camera headlight is on by default and not
  part of the lighting modder. Scaling the headlight as well took it to 11. A distractor
  object changes the length of the saved state (92 → 105 values), so its init states are
  spliced at the address MuJoCo actually assigns, verified bit-identical for every other
  object.
- **`scripts/verify_corpus.py`** checks a finished recording against each of these
  failure modes. Run it after every recording.

## Evaluation methodology

- **A timer matched the guard.** Detection at 5% was the original headline until a
  stopwatch caught every failure inside the budget. The headline changed to time saved over
  the stopwatch (ADR-005).
- **Budgets that did not hold.** Rules searched to the edge of the budget averaged
  6.3–6.8% false stops over 20 splits and went over on 10–13 of them. One split had made
  them look fine. Rules are now order statistics of good runs only, reported over 20
  splits.
- **Levels that did not transfer.** The spatial rule stopped 17% of object-suite good
  runs; recalibration from the new suite's own good runs fixes it.

## Infrastructure

- **A green CI that ran a tenth of the tests.** Test modules skip themselves when an
  optional dependency is missing, and CI once passed while running 31 of 299 tests. CI now
  fails on any skip outside a declared list, in a job with the full simulator stack.
- **Offscreen OpenGL in CI.** The simulator job needed OSMesa installed and selected, as
  locally; without it, the lighting and camera tests had never run in CI.
- **Crashes in code no test reached.** The guard viewer crashed on a field removed from the
  traces. A refactor broke the GPU-only presence script in a way the type checker could not
  see (an untyped numpy index); the output-snapshot comparison caught it before it was
  committed. Both now have coverage.

## Verification

Changes to the recording or analysis code are proven not to alter results:

- **Golden rollout test.** The real recording loop runs on CPU fakes with every feature on
  and is fingerprinted frame by frame.
- **Real-simulator comparison.** `just compare-recordings OUT [EARLIER]` records a short
  fixed run on the GPU and compares every column, metadata file and sim-state array with an
  earlier recording. Two recordings of the same code are identical, videos included.
- **Analysis snapshot.** `just snapshot OUT [MANIFEST]` reruns the analysis pipeline and
  hashes every output.
- **Layering and ownership tests.** Which packages may import which, and which single module
  may touch each private LeRobot or LIBERO attribute, so an upgrade breaks in one place.
- **Sim-state round trip.** `just verify-sim-roundtrip`.

CI runs ruff, pyright and the test suite twice: on a core install without the simulator,
and with the full simulator stack, where no skip is allowed.
