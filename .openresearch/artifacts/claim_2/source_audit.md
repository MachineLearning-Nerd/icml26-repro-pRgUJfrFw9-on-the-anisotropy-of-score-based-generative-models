# Claim 2 source audit

The contract uses the exact Theorem 1 statement at `main.tex` lines 182–185 and
the derivation in Appendix C.2. The theorem is narrower than the judge summary:
it treats a linear architecture `Omega = Phi Theta`, one fixed noise scale, and
rank-one Gaussian data aligned with an eigenvector of `Phi Phi^T`.

Let `A = Phi Phi^T`, `B_i = u_i u_i^T + sigma^2 I`. The population DSM gradient
gives the error recurrence

`E_t = E_(t-1) - 2 eta A E_(t-1) B_i`.

In the shared eigenbasis the relevant contraction matrix is diagonal with
entries `(sigma^2+1) lambda_i` at coordinate `i` and `sigma^2 lambda_j`
elsewhere. For a descending spectrum:

- when `i<D`, the minimum is exactly `sigma^2 lambda_D`;
- when `i=D`, both candidates exceed `sigma^2 lambda_D` because
  `lambda_D>0` and `lambda_(D-1)-lambda_D>0`.

At the optimum, the trace of the stochastic-gradient covariance is

`4 (1 + D sigma^2) lambda_i / (sigma^2 (sigma^2+1))`.

The verifier checks the universal algebraic differences symbolically, exact
rational matrix recurrences, a numerical batch-one DSM simulation, and a
separately implemented full-Hessian/finite-difference checker.
