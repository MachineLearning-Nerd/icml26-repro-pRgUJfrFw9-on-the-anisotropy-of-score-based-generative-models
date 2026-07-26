# Claim 3 method

This verifier does not infer a universal theorem from a small numerical
example. It reconstructs the proof in three machine-checkable layers:

1. SymPy expands and factors the exchange difference for two descending
   weights and two descending assignments.
2. Exact-integer enumeration checks every permutation in each complete finite
   permutation domain from dimension 2 through 9.
3. An independent implementation constructs random PSD matrices, recovers
   their eigensystems, and asks SciPy's linear-assignment optimizer for both
   extrema without supplying the theorem's identity/reversal answer.

A predeclared Haar-orthogonal sweep probes the non-permutation interior of the
orthostochastic set. A wrong “identity minimizes” rule must fail on strict
spectra. A degenerate-spectrum case audits the non-uniqueness caveat.

The universal conclusion rests on the symbolic exchange and
doubly-stochastic/Birkhoff reduction. The finite checks are regression and
implementation evidence, not the logical basis for universality.
