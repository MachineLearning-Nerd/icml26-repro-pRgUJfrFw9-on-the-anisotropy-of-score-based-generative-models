# Claim 6 evaluation record

Verdict: **VERIFIED** under the exact final-layer assumptions of Propositions
1–3.

Successful run `af45ac22-f33b-447a-94e4-3f9614b8f66b`, Git
`2d1c974bd148eb0088590b216581d48ea230d077`.

Actual PyTorch layers produced relative structural residuals `0.024981`
(nonlinear MLP), `0.030958` (convolution), and `0.029531` (shared-token
transformer). Independent complete finite parameter domains reproduced all
three structures with zero error. The transformer's derived geometry had
exactly 7 distinct eigenvalue groups for 7 tokens, with maximum within-group
spread `9.77e-15`. Assumption-breaking controls were all rejected: residuals
were `0.101464`, `0.208788`, and `0.925398`/`0.801893`.

Pre-run estimate: 8 cores, 2–5 minutes. Hugging Face `cpu-upgrade` reported 64
logical / 32 physical CPUs, affinity 64. Full job: 42 seconds. Claim 6 checker:
5.788714 seconds. Cumulative suite: 14.460175 seconds. No GPU was used.
