# Limitations and deviations

- One node is only half of the paper-scale estimator and is labeled partial.
  It cannot be called a one-million-network geometry until the disjoint sibling
  shard is aggregated.
- Per-sample seeds replace one uninterrupted global RNG stream. They still
  generate independent default network initializations and uniform timesteps,
  and they make every sample addressable and reproducible.
- Multiprocessing and streaming change floating-point reduction order relative
  to the author's tensor-matrix multiplication. The estimator, sample count,
  architecture, and probe distribution are unchanged; float64 accumulation and
  an independent eigensolver audit bound numerical risk.
- Stable geometry still does not verify Claims 1 or 4. Actual iDDPM training,
  generation, multi-direction rank association, and paired uncertainty remain
  required.
