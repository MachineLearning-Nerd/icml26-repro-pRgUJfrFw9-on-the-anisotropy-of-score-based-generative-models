# Claim 2 evaluation record

Verdict: **VERIFIED**, scoped to the exact assumptions of Theorem 1 in
arXiv:2510.22899v1.

The successful evidence is run `f696ab27-32f4-49e2-82b6-b7d9d71b144a` from
Git commit `6864b70ce711c738f306848dea56ffefbff83898`.

The fixed command was:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

The pre-run estimate was 2 CPU cores and 1–4 minutes. The selected compute was
Hugging Face `cpu-upgrade`; the job reported 64 logical CPUs, 32 physical
CPUs, and affinity 64. The full job took 32 seconds, the Claim 2 verifier
5.512556 seconds, and the cumulative suite 6.046443 seconds. No GPU was used.

See `raw_results.json`, `independent_checker_output.json`, and
`negative_control_output.json` for the machine-readable evidence.
