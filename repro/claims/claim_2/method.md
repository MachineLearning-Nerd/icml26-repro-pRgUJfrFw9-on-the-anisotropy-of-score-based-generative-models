# Claim 2 method

## Route

1. Reconstruct the population DSM objective from the rank-one Gaussian model.
2. Use SymPy to certify the two strict differences that establish
   `rho_D > sigma^2 lambda_D`.
3. Compare the paper's closed form against exact rational GD iterates for every
   eigenvector direction in two fixed spectra.
4. Independently construct the full Kronecker Hessian and recover its smallest
   eigenvalue without using the paper's `rho_i` shortcut; also compare the
   analytical gradient with central finite differences.
5. Execute batch-one DSM SGD from zero initialization at `D=5`.
6. Estimate the per-example stochastic-gradient covariance trace at the exact
   optimum over 10 independently seeded blocks of 12,000 samples per direction.
7. Confirm that a deliberately wrong per-direction rate rule is rejected.

The sample count and tolerances were declared before execution and were not
computed from the theorem's formula.
