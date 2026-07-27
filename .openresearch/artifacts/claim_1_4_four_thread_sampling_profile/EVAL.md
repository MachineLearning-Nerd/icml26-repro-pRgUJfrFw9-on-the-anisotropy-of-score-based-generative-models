# Four-thread exact sampling calibration

Status: **CPU_RESOURCE_CALIBRATION_ONLY — Claims 1 and 4 remain BLOCKED**

The exact author D=256 iDDPM reverse path generated one finite batch of 1,000
samples through all 1,000 steps in `1220.295648` seconds with PyTorch intra-op
parallelism fixed to four threads. Ten sampling batches plus 2,000 optimizer
updates project to `4.918397886` hours before metrics and cumulative checks.

- Fixed command: `uv sync --locked && .venv/bin/python repro/run_campaign.py`
- HF flavor: `cpu-upgrade`; no GPU
- Estimated required cores: 4
- Actual allocation: 64 logical / 32 physical CPUs; affinity 64
- Run: `e443c307-fe3a-443a-81f3-1767bec3f4b0`
- Git commit: `7add11d5a2891b3cd957538af6dd1775e8e08409`
- Verifier: `1236.157199` seconds; campaign: `1257.999849` seconds
- Independent checker: passed all 12 checks
- Negative control: a 100-step shortcut was rejected as not the exact path
- Cumulative regressions: Claims 2, 3, and 6 passed

This run measured resources only. It used an untrained network, computed no
SW2/MSW2 value, and supplies no verification or falsification of Claims 1 or 4.
