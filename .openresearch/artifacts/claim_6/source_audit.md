# Claim 6 source audit

This contract follows Propositions 1–3 in Appendix B and their proofs in
Appendices C.4–C.6 of arXiv:2510.22899v1.

## Proposition 1: MLP output

For `z=phi(W h+b)` with elementwise `phi`, the output rows are conditionally
iid given `h`. If one coordinate has conditional mean `mu_h` and variance
`s_h^2`, then

`E[z z^T | h] = s_h^2 I + mu_h^2 11^T`.

Averaging over any probe distribution for `h` preserves this form. No
Gaussian or linear-activation assumption is needed.

## Proposition 2: CNN output

For iid output-channel filters, each output channel has the same conditional
spatial mean vector `mu_h` and covariance `Sigma_h`, while distinct output
channels are conditionally independent. Consequently the diagonal channel
blocks are `Sigma_h+mu_h mu_h^T` and off-diagonal blocks are
`mu_h mu_h^T`. Averaging gives

`I_Cout tensor A + 11^T_Cout tensor B`.

The spatial matrices `A` and `B` need not be diagonal or circulant; boundary
conditions and the arbitrary hidden representation are absorbed into them.

## Proposition 3: transformer-style output

A shared zero-mean iid `W` acts separately on each of `T` token vectors and
the bias entries are zero-mean iid. The conditional `(i,j)` token block is a
scalar multiple of `I_Lout`, so before the fixed orthonormal map `Q` the
geometry is `A tensor I_Lout`. Its eigenvalues are the `T` eigenvalues of `A`,
each repeated `Lout` times; hence there are at most `T` distinct values.
Orthogonal congruence by `Q` preserves them.

These propositions apply to the specified final layers, not to arbitrary
unconstrained networks merely called MLPs, CNNs, or Transformers.
