# Variance-allocated geometry limitations

- The paper's one-million-network estimator is still larger than the 18,000
  pilot-plus-final networks here.
- The pilot chooses compute allocation from observed conditional variance.
  It does not choose a direction, threshold, or outcome from the anisotropy
  conjecture, and its samples are excluded from the final estimate.
- Readiness focuses on the largest- and smallest-eigenvalue endpoints. Interior
  vectors remain useful for exploratory rank association only when their band
  subspaces are stable.
- Even a stable geometry does not verify Claims 1 or 4; actual iDDPM
  training, generation, and Wasserstein evaluation remain mandatory.
