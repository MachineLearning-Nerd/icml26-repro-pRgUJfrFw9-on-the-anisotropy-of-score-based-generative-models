# Claim 6 limitations and deviations

- The universal component is an independently reconstructed
  conditional-moment proof, not a proof-assistant certificate.
- Direct network dimensions (`D_out=20`, `C_out=4`, `T=7`) are large enough to
  make the matrix structures nontrivial but are smaller than image-scale
  diffusion U-Nets. Finite scale is not used to infer universality.
- The Monte Carlo geometries have sampling noise and therefore only
  approximately exhibit exact multiplicities. Exact multiplicity follows from
  the symbolic contraction and complete finite-domain checker.
- The CNN implementation uses one-dimensional zero-padded convolution to make
  every channel/spatial block inspectable. Proposition 2 is dimension-agnostic
  and its proof uses output-channel iid structure, not image dimensionality.
- These output-layer propositions do not establish the paper's empirical
  generation-preference conjecture.
