# Claim 5 — four-route audit

**Verdict: BLOCKED. Confidence: LOW.**

Exactly three materially different verification-oriented routes and the
mandatory fourth falsification route were completed. None supplied direct
five-seed regeneration or a valid assumption-satisfying counterexample.

## Observed resource evidence

| Dataset | Exact config | Batch | Timed update | Training-only linear projection |
| --- | --- | ---: | ---: | ---: |
| MNIST | `1x28x28-iddpm.json` | 500 | 8.160469 s | 226.6797 h/model |
| CelebA-HQ | `1x56x56-iddpm.json` | 500 | 32.910655 s | 914.1849 h/model |
| CIFAR-10 | `3x32x32-iddpm.json` | 500 | 9.636662 s | 535.3701 h/model |

These are one-update planning measurements on deterministic same-shape
tensors. They exclude real data, alignment transforms, full training,
generation, SW2, and five seeds, and are not claim evidence.

- Fixed command: `uv sync --locked && .venv/bin/python repro/run_campaign.py`
- Run: `8948914c-470d-4c41-9b56-44243d6545d8`
- Git: `6a9a52a024df44a8757c8345961f557fb798d6d9`
- HF `cpu-upgrade`; estimated 4 required cores
- Actual: 64 logical / 32 physical CPUs, affinity 64, four intra-op threads
- Peak RSS: 15,567 MB; no GPU
- Claim verifier: 114.457548 s wrapper; cumulative campaign: 146.942617 s
- Independent checker: passed all 11 checks
- Negative control: rejected the assertion that one timed update verifies Claim 5
- Cumulative regressions: Claims 2, 3, and 6 passed

The claim is unblocked only by exact public checkpoints plus transformed
datasets and raw five-seed outputs, or sufficient CPU resources for all 45
full training-and-generation runs.
