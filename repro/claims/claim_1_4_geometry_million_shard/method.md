# Method — half-million geometry shard

Each integer seed deterministically initializes a fresh copy of the author's
`D=256` iDDPM U-Net. After initialization, that model draws one uniform
diffusion timestep through the author's `model.randint` path. The input is the
zero tensor, matching `geometry.py --sigma 0`; the score output is divided by
the model's `sqrt_one_minus_alphas_cumprod` factor exactly as in the author
implementation.

The verifier streams each `256`-vector outer product into float64 accumulators
instead of retaining a `500000 x 256` sample tensor. Thirty-two spawned worker
processes use one Torch thread each. The committed seed range is split into 128
deterministic tasks; results are merged in task order. Even and odd seeds form
independent 250,000-network halves for within-shard stability checks.

The raw artifact retains the complete matrix, both half matrices, selected
eigenvectors and eigenvalues, all 1,000 timestep counts, score-norm moments,
runtime and CPU allocation, and deterministic hashes. An independent script
decodes the payloads and reconstructs the eigendecomposition without importing
the primary verifier.
