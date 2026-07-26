# Limitations and deviations

- This is one endpoint and one seed. It is deliberately labeled partial and
  cannot establish Claims 1 or 4.
- The 16,000-network variance-allocated geometry is far smaller than the
  paper’s one-million-network estimate. Only endpoints passed the predeclared
  direction-stability gate; intermediate individual eigenvectors remain
  unstable and require a separate design.
- Metric projections are processed in chunks rather than one huge tensor.
  This is mathematically the same estimator and all 16,384 projection values
  are retained.
- Full generated tensors are not printed into logs. Their deterministic
  SHA-256 digest, distribution diagnostics, all projection-level distances,
  seeds, and regeneration command are retained.
- HF `cpu-upgrade` hosts showed roughly ten-fold throughput variation despite
  identical reported CPU allocation. A wall-time-only gate rejects hosts above
  12 seconds per optimizer update at step 10. This affects resource
  feasibility, not scientific selection; rejected hosts yield no claim result.
- The paired endpoints use four PyTorch intra-op threads on hosts exposing 64
  logical and 32 physical CPUs. This changes only CPU reduction scheduling,
  which the paper does not specify, and is applied identically to both siblings.
