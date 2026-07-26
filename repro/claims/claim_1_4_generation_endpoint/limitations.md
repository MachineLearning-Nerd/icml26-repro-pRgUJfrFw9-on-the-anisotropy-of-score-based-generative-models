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
