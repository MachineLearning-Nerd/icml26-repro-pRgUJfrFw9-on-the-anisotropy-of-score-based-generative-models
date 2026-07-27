# Evaluator note — CPU value-path diagnostic

Status: **diagnostic only; Claims 1 and 4 remain BLOCKED**.

The exact author iDDPM executed two deterministic DSM updates under four
conditions on the same 64-logical/32-physical-CPU HF host. Enabling
`torch.set_flush_denormal(True)` did not improve the largest-SAD path:
median step time changed from 49.3060 s to 50.6636 s (0.9732x), while losses
and final model hashes remained exact. The smallest-SAD path was also slow on
this host. This rejects the proposed subnormal-value mechanism and attributes
the earlier runtime split to host throughput variance, not a claim outcome.

The independent checker passed, the deliberately non-unit direction control
was rejected, and no GPU was used. See `raw_results.json` for every measured
forward, backward, and optimizer time.
