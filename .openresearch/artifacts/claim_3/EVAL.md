# Claim 3 evaluation record

Verdict: **VERIFIED** for every orthogonal transformation under Theorem 2's
positive-semidefinite and eigenvalue-ordering assumptions.

Successful run: `843c3b36-403d-4e3a-83b1-bfdaf5b3aa58`; Git SHA:
`291c2adfa131af97b82b6ff3b617509f24ada30f`.

The symbolic exchange difference is exactly `(a-b)(x-y)`. Combined with
`P_ij=R_ij^2` being doubly stochastic for every orthogonal `R`, adjacent
exchanges and the Birkhoff-polytope bound prove that identity ordering is a
maximum and reversal is a minimum. Both bounds are orthogonal permutation
matrices, so the relaxation is tight.

Regression evidence checked all 409,112 permutations in complete domains
`D=2..9`. An independent assignment optimizer on five PSD cases (`D=8..12`)
agreed to `5.684341886080802e-14` at the maximum and
`2.1316282072803006e-14` at the minimum. A 384-matrix Haar sweep stayed within
the certified bounds. The wrong identity-is-minimum control was rejected with
gap 54.75. A repeated-spectrum case confirmed that extrema need not be unique.

Pre-run estimate: 2 CPU cores, 1–3 minutes. Selected compute: Hugging Face
`cpu-upgrade`. Actual allocation: 64 logical CPUs, 32 physical CPUs, affinity
64. Full job: 25 seconds. Claim 3 checker: 2.201216 seconds. Cumulative suite:
8.139307 seconds. No GPU was used.
