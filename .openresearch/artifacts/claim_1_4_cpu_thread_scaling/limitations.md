# Limitations and deviations

- This is a resource diagnostic, not evidence for Claims 1 or 4.
- It benchmarks one exact batch and three updates rather than the full 2,000
  updates. The endpoint’s existing step-10 wall-time gate remains the final
  feasibility check.
- Thread settings can change floating-point reduction order. Exact final-state
  hashes, not merely similar losses, are required before a count is considered
  safe.
- Timings are host-specific. Two opposite condition orders reduce, but cannot
  eliminate, transient load and cache effects.
- Only intra-op thread count is varied. Inter-op threads, model, data, seeds,
  optimizer, and scientific design remain fixed.
