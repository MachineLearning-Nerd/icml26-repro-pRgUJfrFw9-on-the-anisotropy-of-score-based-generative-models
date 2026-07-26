# Claim 3 source audit

The contract follows the alignment definition at `main.tex` lines 271–276,
Theorem 2 at lines 280–283, and Appendix C.3 of arXiv:2510.22899v1.

Write the two positive-semidefinite eigendecompositions as

`G_F = U diag(lambda) U^T` and `M = V diag(nu) V^T`,

with both spectra descending. For an arbitrary orthogonal transformation `W`,
set `R=U^T W V`. Then `R` is orthogonal and

`alpha(W) = sum_(i,j) lambda_i nu_j R_ij^2`.

The elementwise square `P_ij=R_ij^2` is doubly stochastic. The objective is
therefore linear over a subset of the Birkhoff polytope. Every permutation
matrix belongs to that subset, and the rearrangement inequality makes the
identity permutation maximal and the reversal permutation minimal. Thus the
polytope bounds are attainable by orthogonal matrices and are also the exact
orthogonal-group extrema.

The theorem does not require distinct eigenvalues. Degeneracy can make an
extremizer non-unique, but the displayed identity- and reversal-derived
transformations remain valid extrema.
