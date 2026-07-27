# Limitations and deviations

- Route 1 uses deterministic same-shape tensors for resource timing, not MNIST,
  CelebA-HQ, or CIFAR-10. It cannot measure the alignment effect.
- One timed update is not a training run. Linear horizon projections are
  planning estimates only and are never used as scientific acceptance evidence.
- No public checkpoint, transformed dataset, or raw five-seed Figure 9 outputs
  were located. CelebA-HQ also requires separately provisioned source data.
- Published aggregate bars do not expose seed-level uncertainty.
- The qualitative MNIST digit-1 mode-collapse statement cannot be independently
  quantified from the published montage.
- A failed or infeasible reproduction is not falsification. With no
  assumption-satisfying counterexample, the only honest verdict is BLOCKED.
