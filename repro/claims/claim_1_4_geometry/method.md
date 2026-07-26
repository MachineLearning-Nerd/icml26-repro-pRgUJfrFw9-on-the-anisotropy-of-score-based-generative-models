# Claims 1/4 geometry convergence method

Each independently seeded model is the author implementation configured by
`configs/1x16x16-iddpm.json`. The input is exactly zero. Its epsilon prediction
is divided by `sqrt(1-alpha_bar_t)` to recover the paper's score-network
normalization.

Every consecutive 1000-model block uses all 1000 timesteps exactly once in a
seed-independent permutation. This has the same uniform marginal as the paper
while preventing the very high-variance `t=0` term from appearing an
unbalanced number of times.

The verifier accumulates eight block geometry matrices and reports:

- eigenspectra at 1k, 2k, 4k, and 8k networks;
- selected-vector overlaps with the final estimate;
- independent alternating-block split-half vector and band-subspace overlaps;
- 200 block-bootstrap eigenvalue intervals;
- orthonormality, eigen residuals, trace, PSD, and symmetry;
- an independently implemented Torch eigendecomposition;
- random directions that must fail the eigenvector residual test.

The selected vectors and a compressed text encoding of the complete matrix
are printed to the run log so downstream branches and evaluators can reproduce
the calculation.
