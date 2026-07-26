# Claim 2 limitations and deviations

- The symbolic certificate reconstructs the algebra but is not a Lean/Coq proof.
- Numerical simulation uses `D=5`, matching the paper's illustrative Figure 4;
  the universal part is supported by the symbolic derivation, not inferred from
  this finite experiment.
- “Covariance proportional to lambda_i” is operationalized as covariance trace
  scale. The full covariance matrices also rotate with `u_i`, so literal
  elementwise proportionality across different directions would be false.
- The `u_D` mean-rate advantage relies on the theorem's zero-mean
  initialization: the unrestricted error operator contains slower
  off-diagonal modes, but those modes have zero coefficient in the stated mean
  error and are never entered by the population update.
