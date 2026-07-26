# Method — common four-thread sampling calibration

The preceding same-host diagnostic found that four intra-op threads were the
fastest setting (2.752 seconds/update versus 47.890 at 32), and that both
four-thread repetitions produced identical final model hashes. Cross-thread
rounding changed losses by at most `6.79e-8` relatively. The paper does not
specify CPU scheduling, so all future paired endpoints will use the same
four-thread setting; mixed-thread scientific comparisons are prohibited.

This verifier constructs the exact author iDDPM, generates a deterministic
batch of 1,000 initial Gaussian samples, and calls the unmodified
`model.sample(steps=None, eta=1.0)` path. That executes all 1,000 reverse
steps. Runtime is multiplied by ten and combined with the independently
measured four-thread training median to forecast the complete endpoint cost.

The checker independently audits shapes, step count, thread count, projection
arithmetic, within-setting determinism, finite output, CPU metadata, and the
negative control.
