# Backstop

Runtime failure prediction for vision-language-action robot policies. Supervisory layer over a frozen VLA on LIBERO — not a replacement policy.

Week 1: rollout infrastructure. See [docs/runbook.md](docs/runbook.md), [docs/adr/001-metric.md](docs/adr/001-metric.md), and [docs/adr/002-checkpoint.md](docs/adr/002-checkpoint.md).

```bash
uv sync --extra dev --extra sim
just test
just smoke
```
