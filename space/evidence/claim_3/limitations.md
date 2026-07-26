# Claim 3 limitations and deviations

- The certificate is an independently reconstructed symbolic/combinatorial
  proof, not a Lean, Coq, or Isabelle proof object.
- Numerical eigendecompositions have sign ambiguity. The alignment only uses
  squared overlaps, so signs do not change the result.
- The random orthogonal sweep cannot prove a global extremum and is not used
  for that purpose.
- When either spectrum has repeated eigenvalues, the theorem's displayed
  transformations remain extrema but need not be unique. The verifier states
  and tests this boundary explicitly.
- This claim is a mathematical trace-alignment theorem. It does not by itself
  establish that alignment controls generation quality; that empirical
  question belongs to Claims 1, 4, and 5.
