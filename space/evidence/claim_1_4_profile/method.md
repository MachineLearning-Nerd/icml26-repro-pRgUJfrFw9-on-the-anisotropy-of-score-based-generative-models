# Claims 1/4 CPU calibration method

This is planning evidence, not claim evidence.

The profiler instantiates the author implementation and unmodified
`1x16x16-iddpm` configuration. It times fresh-network default-probe geometry
calls, one DSM optimizer step at the paper's batch size 1000, and one
batch-1000 epsilon call of the exact model used inside each reverse diffusion
step. It also times a fixed projection-and-sort pilot for the paper's sliced
Wasserstein computation.

The resulting extrapolations are empirical cost estimates. They do not set a
sample count from the anisotropy formula and cannot verify either scientific
claim. The next child must commit its scientific design before seeing its
outcome.
