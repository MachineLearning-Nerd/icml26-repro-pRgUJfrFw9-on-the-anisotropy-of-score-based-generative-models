# Current verification run

This page supersedes the historical `d=6` NumPy proxy run. The exact judged
page remains unchanged under `historical/judged-33d98421/`.

## Fixed command

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

The command and locked environment are identical across every experiment
node. Current executable source and raw outputs are linked from each canonical
claim page.

## Accepted cumulative result

Paper-scale aggregate run:
`0c6c3210-941e-443c-ab16-dd524c804ac9`, Git
`26b31ae7f0bb19c7901c72f54413326ecd729d9e`.

| Checker | Exit |
| --- | ---: |
| SAD base checks | 0 |
| One-million geometry aggregate | 0 |
| Paper-scale five-pair endpoint aggregate | 0 |
| Claim 2 exact linear DSM theorem | 0 |
| Claim 3 alignment extrema theorem | 0 |
| Claim 6 architecture propositions | 0 |

`CAMPAIGN_SUMMARY.all_passed = true`. Actual allocation: 64 logical / 32
physical CPUs, affinity 64, no GPU. Cumulative runtime: `16.540458` seconds;
OpenResearch job duration: 42 seconds.

[Machine-readable cumulative summary](evidence/claim_1_4_five_pair_aggregate/campaign_run_summary.json) ·
[runtime record](evidence/claim_1_4_five_pair_aggregate/runtime_cpu.json) ·
[locked environment](evidence/environment/uv.lock)

Two prior launches from the same commit exited 127 because their images lacked
`uv`; no verifier executed. The unchanged commit then passed in the
project-standard official `uv` image. They are environment failures, not
scientific results.
