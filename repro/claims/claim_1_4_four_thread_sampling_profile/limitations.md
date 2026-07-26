# Limitations and deviations

- This is resource calibration only and cannot verify Claims 1 or 4.
- Sampling uses an untrained network. This preserves architecture and reverse
  computation cost but produces no meaningful scientific samples.
- Ten-batch runtime is a linear projection from one exact batch. The full
  endpoint must still report observed runtime for every batch.
- Timings are host-specific. The full endpoint retains its independent
  step-10 throughput gate.
- Four threads use a different floating-point reduction order than 32 threads.
  Both future endpoints must use four threads, and four-thread repetitions were
  bitwise deterministic in the preceding diagnostic.
