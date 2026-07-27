# Limitations and deviations

- Per-sample deterministic seeds replace the author's single uninterrupted RNG
  stream, while retaining independent default initializations and uniform
  timesteps.
- CPU multiprocessing and float64 streaming change reduction order but not the
  estimator; hashes and an independent reconstruction expose numerical errors.
- Near-degenerate interior eigenvalues may produce unstable individual
  eigenvectors even at one million networks. The verifier reports this rather
  than treating a stable local subspace as a stable individual direction.
- This artifact establishes paper-scale geometry only. Generation-preference
  evidence still requires training, sampling, Wasserstein metrics, seeds, and
  controls.
