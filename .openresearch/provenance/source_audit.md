# Paper and campaign source audit

Retrieval date: 2026-07-26 (Asia/Kolkata)

## Immutable sources

| Source | Retrieval | SHA-256 / revision |
| --- | --- | --- |
| ar5iv HTML, `https://ar5iv.labs.arxiv.org/html/2510.22899` | `curl -L` with user agent `OpenResearch-Reproduction/1.0 (+https://github.com/MachineLearning-Nerd)` | `c362d38e38228b7059ab918527f201cd6489057f2715ff34f26c37ca2a04d234` |
| arXiv source, `https://export.arxiv.org/e-print/2510.22899` | same explicit user agent | `781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c` |
| alphaXiv full extraction, `orx paper 2510.22899 --full` | alphaXiv public corpus | `8eb3391fca673ade8489403983d5c30a5f06fe48c870f1aacd60564c1185dc29` |
| Author implementation, `https://github.com/andreasfloros/score-anisotropy` | Git clone | `65b63d00f5e86c98290aa22d214dcfa0504fffd7` |
| Judged Space, `DineshAI/pRgUJfrFw9` | Git checkout at exact judge head | `33d98421a79a7686497639f2d4d6163fc5acfa3f` |
| Live verdict dataset, `ICML-2026-agent-repro/verdicts` | HF resolve API, filtered only by exact `space_id == "DineshAI/pRgUJfrFw9"` | dataset revision `3790325497824bcf196ac70bc3b97a97f4456350`; full JSON SHA-256 `094958ed84439855aadbe3fa49bac7f2d175c8342a08545af569c0bc168d0ad4` |

The exact filtered verdict is stored in `live_verdict_filtered.json`. The judged
Space file manifest was created before any candidate copy was changed and is
stored in `.openresearch/protected/judged_space_manifest.tsv`.

## Exact claim anchors, assumptions, and quantifiers

Line anchors below refer to `main.tex` in the arXiv source tarball.

### Claim 1: definitions and SAD conjecture

- Score Anisotropy Directions definition: lines 93–96.
- Average geometry definition: lines 221–228.
- Conjecture 1: lines 249–252.
- Domain: a family of networks mapping
  `R^D × R -> R^D`, parameter distribution `Theta`, and independently chosen
  probing distribution `P`.
- Exact geometry is the outer-product expectation
  `E[F_theta(x_sigma,sigma) F_theta(x_sigma,sigma)^T]`, not a scalar square.
- Quantifier/status: the eigenvector ordering is explicitly a conjecture, not a
  theorem. Under the paper's default probe, initialization is sampled from the
  architecture's default scheme; the eigenvectors in ascending eigenvalue order
  are hypothesized to be SADs, with small-eigenvalue aligned data modeled better.

### Claim 2: linear DSM theorem

- Theorem 1: lines 182–185; proof: Appendix C.2, beginning line 460.
- Assumptions: rank-one Gaussian data `N(0,vv^T)`, fixed `sigma>0`,
  unit `v`, linear score `Omega=Phi Theta`, fixed `Phi`, positive bottom two
  eigenvalues with `lambda_(D-1)>lambda_D`, zero-mean initialization, and small
  learning rate.
- Quantifier: for every eigenvector-aligned direction `v=u_i`, the mean-error
  rate is `O[(1-2 eta rho_i)^t]`; all `i<D` have the same limiting rate and
  `rho_D` is larger. Near optimality, the SGD-step covariance scales with
  `lambda_i`. The prose's “amplified” wording is an interpretation of this
  covariance result, not a separate theorem.

### Claim 3: extreme alignment theorem

- Alignment definition: lines 271–276.
- Theorem 2: lines 280–283; proof: Appendix C.3, beginning line 513.
- Assumptions: `W` is orthogonal; `G_F` and the data second moment are symmetric
  positive semidefinite; their eigenvectors are columns of `U,V` in descending
  eigenvalue order.
- Quantifier: over all orthogonal `D × D` matrices, `U J V^T` is a minimizer and
  `U V^T` is a maximizer. Degenerate spectra may make extrema non-unique; the
  displayed transforms remain extrema.

### Claim 4: iDDPM rank-one experiment

- Figure 6 caption/label `fig:dads`: lines 231–245.
- Interpretation: lines 254–258.
- Setup: lines 343–364.
- Exact setup: `D=256`, iDDPM U-Net, rank-one
  `N(0,Dvv^T)`, 10k samples, 2k optimizer steps (200 epochs at batch 1000),
  1000 diffusion steps, five seeds, `MSW_2`, `64D` projections, and geometry
  estimated from 1M initializations under the default zero-input/noise-level
  probe.
- Quantifier/status: the paper says “a clear trend,” with smallest-eigenvalue
  directions best and largest worst. It does not state strict point-by-point
  monotonicity as a theorem. A faithful contract must at minimum test the
  endpoint ordering and rank association with seed-level uncertainty.

### Claim 5: real-image alignment experiment

- Figure 8: lines 285–294.
- Figure 9: lines 298–305.
- Setup: lines 343–364.
- Exact setup: MNIST `28×28`, CelebA-HQ `56×56`, CIFAR-10 `3×32×32`;
  10k samples per dataset; five seeds; iDDPM; 1000 diffusion steps; batch 500;
  100k optimizer steps for MNIST/CelebA-HQ and 200k for CIFAR-10; `SW_2` with
  `64D` random projections.
- Quantifier/status: minimized alignment is reported to give substantial and
  consistent `SW_2` improvement on all three datasets. Maximized alignment is
  reported as similar to natural alignment and potentially not significantly
  different. Mode-collapse language is specifically attached to the MNIST
  samples (over-representation of digit “1”), not universally quantified over
  all three datasets.

### Claim 6: architecture geometry propositions

- MLP Proposition 1: lines 391–399.
- CNN Proposition 2: lines 401–409.
- Transformer Proposition 3: lines 411–419.
- Each statement concerns the explicitly specified final-layer family and its
  parameter-independence assumptions, not every unconstrained MLP/CNN/Transformer.
- Transformer parameters are additionally zero-mean and the fixed map `Q` is
  orthonormal. Under those assumptions, the geometry has at most `T` distinct
  eigenvalues.

## Non-circularity rule

No acceptance threshold, sample count, direction, or horizon may be selected
from the formula being checked. Theorem verifiers reconstruct identities from
the DSM objective or trace optimization; empirical checks use predeclared
directions/seeds and report uncertainty and controls.
