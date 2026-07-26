# Method — CPU value-path diagnostic

The diagnostic uses the frozen stable largest and smallest SAD endpoint vectors,
the exact author D=256 iDDPM configuration, batch size 1,000, Adam at `1e-4`,
and gradient clipping at 1. Model initialization, rank-one data, timestep/noise,
and optimizer seeds are reset identically for every condition.

A separate batch-eight warmup initializes CPU kernels. Each condition then
starts from the same fresh model state and executes two exact DSM optimizer
updates. Forward, backward, optimizer, and total durations are recorded
separately, along with losses, unclipped gradient norms, and final model hashes.

The four predeclared conditions cross endpoint direction with default versus
`torch.set_flush_denormal(True)` behavior. The independent checker recomputes
median speed ratios and relative numerical differences without importing the
primary verifier.

This is an implementation/runtime diagnostic only. It produces no paper-claim
verdict.
