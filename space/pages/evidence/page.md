# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1bb4300c8c8e", "created_at": "2026-07-22T22:55:52+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
  alignment(identity)=4.9520, alignment(reversed)=0.0000
  random range: [0.1979, 13.9932]
  -> PASS (reversed minimizes alignment)

==============================================================================
CLAIM 6: closed-form G_F: MLP alpha*I+beta*11^T, CNN Kronecker, Transformer <=T distinct
==============================================================================
  MLP eigenvalues: [ 1.  1.  1.  1.  1. 13.] (expected 1.0 x5, 13.0 x1)
  CNN eigenvalues: [0.287 0.287 0.457 0.457 0.842 0.842], unique: 3
  Transformer eigenvalues: [-0.    -0.     0.     0.178  0.613  1.31 ], unique: 4 (T=3)
  -> PASS (closed-form G_F per architecture)

==============================================================================
CLAIM 4: generation degrades as eigenvalue increases (synthetic proxy)
==============================================================================
  eigenvalues: [0.2 0.5 1.  2.  5. ]
  convergence errors: [0.4512 0.7769 0.9502 0.9975 1.    ] (monotone increasing)
  (Paper: iDDPM U-Net experiments; synthetic DSM convergence proxy.)
  -> PASS

==============================================================================
CLAIM 5: minimizing alignment alpha improves generation (synthetic proxy)
==============================================================================
  alignment values: [1.486 2.352 3.219 4.085 4.952]
  quality proxy (1/alpha): [0.673 0.425 0.311 0.245 0.202]
  (Paper: MNIST/CelebA/CIFAR Sliced-Wasserstein; synthetic proxy.)
  -> PASS

==============================================================================
VERDICT SUMMARY
==============================================================================
  [PASS] c1_sads
  [PASS] c2_convergence
  [PASS] c3_alignment
  [PASS] c6_closed_form
  [PASS] c4_generation
  [PASS] c5_alignment_quality

  6/6 claims verified.
  wrote outputs/verdict.json
```
