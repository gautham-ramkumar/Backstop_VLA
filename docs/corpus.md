# The corpus

Every failure the policy makes on its own is the same stall, which is not enough to study.
The corpus therefore makes the environment harder in named ways, each tuned so that
roughly half its episodes still succeed. A family that always failed would teach a guard
to detect the perturbation rather than the failure.

| Shard | What changes | Episodes | Success |
|---|---|---:|---:|
| clean | nothing | 100 | 70 |
| lighting | scene lights and camera headlight dimmed and moved | 50 | 35 |
| camera | agentview camera pose and field of view jittered | 50 | 29 |
| object_pose | object positions and yaw jittered, then settled | 50 | 28 |
| distractor | an extra object compiled into the scene | 50 | 33 |
| action_noise | Gaussian noise on the executed action | 50 | 33 |
| language_absent | instruction names an object that is not there | 30 | 0 |
| language_paraphrase | instruction reworded, same task (control) | 20 | 3 |
| object_clean | unseen suite (LIBERO-Object), held out | 100 | 83 |

`language_absent` fails by construction and is reported separately, never pooled.
`language_paraphrase` was meant as a neutral control and was not: rewording the task-0
instruction cut success from 8/10 to 3/20, so the policy is sensitive to phrasing alone.

Each episode also stores the full MuJoCo state at every step, outside the dataset, for
labelling and for branching experiments. It is never a guard input
([ADR-003](adr/003-perturbation-schema.md)).

## Recorded data format

LeRobot v3 dataset, one row per control step at 20 Hz:

| Field | Shape | Notes |
|---|---|---|
| `observation.images.image` | (360, 360, 3) | agentview |
| `observation.images.image2` | (360, 360, 3) | wrist camera |
| `observation.state` | (8,) | eef position (3) + axis-angle (3) + gripper qpos (2) |
| `action` | (7,) | executed 6-DoF delta + gripper |
| `action_policy` | (7,) | what the policy chose; differs from `action` only under action noise |
| `action_chunk` | (50, 7) | the policy's full plan at this step |
| `action_chunk_samples` | (K, 50, 7) | K plans under different flow noise |
| `next.success` | (1,) | episode outcome |

This is why the repository owns its rollout loop instead of calling `lerobot-eval`:
LeRobot computes the action chunk, executes its first step and discards the rest. Backstop
is about what the policy was planning, so the chunk is kept.
