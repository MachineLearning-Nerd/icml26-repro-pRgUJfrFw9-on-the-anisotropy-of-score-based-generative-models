# Method — exact iDDPM endpoint

The verifier loads one endpoint direction from the frozen, variance-allocated
16,000-network geometry estimate. It reconstructs the author’s
`configs/1x16x16-iddpm.json` `Diffuser` without architectural substitutions.

For reproducible paired comparison, model initialization, training data,
dataloader shuffling, DSM randomness, reverse-diffusion randomness, metric
target data, and random projections have distinct predeclared seeds. The
largest- and smallest-endpoint siblings use the same seeds and differ only in
the committed geometry index.

Before a run can produce scientific evidence, a preregistered resource gate
measures wall time through optimizer step 10. The run exits nonzero if mean
throughput exceeds 12 seconds per update. This threshold was fixed from the
completed sibling calibration (4.93 seconds per update) and the observed
approximately ten-fold HF host-throughput split. The gate reads no loss,
sample, metric, direction, or scientific result; a rejected host contributes
no claim evidence and the identical commit may be relaunched.

Training uses Adam at `1e-4`, gradient clipping at 1, exactly 200 epochs over
10,000 rank-one samples with batch size 1,000, and therefore exactly 2,000
optimizer updates. Generation uses the model’s default DDPM-equivalent path:
1,000 reverse steps and eta 1, in ten batches of 1,000.

SW2 and MSW2 use exactly 16,384 normalized Gaussian projections. Projection
and sorting are chunked only to bound memory; every per-projection squared
Wasserstein value is retained in the raw JSON, so the aggregate is
independently recomputable.

The negative control deliberately violates both unit-direction and `D`
covariance scaling assumptions. It must be rejected before training evidence
is accepted.
