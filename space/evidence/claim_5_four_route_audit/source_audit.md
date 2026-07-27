# Claim 5 source audit

Primary anchors are Figure 8, Figure 9, and Appendix A of
arXiv:2510.22899v1 (`main.tex` lines 285–305 and 343–364).

The exact setup is iDDPM on 10,000 examples per dataset, batch 500, 1,000
diffusion steps, five seeds, 100,000 optimizer steps for MNIST and CelebA-HQ,
200,000 for CIFAR-10, and SW2 with `64D` normalized random projections.

The primary Figure 9 image was retrieved on 2026-07-26 from
`https://ar5iv.labs.arxiv.org/html/2510.22899/assets/x15.png` with SHA-256
`b7e23901195f548379823dff020de27a4424cdccfc62bd2da60cad6142751922`.
Its `W_min / I / W_max` bars are:

| Dataset | W_min | I | W_max |
| --- | ---: | ---: | ---: |
| MNIST | 0.11 | 1.91 | 1.81 |
| CelebA-HQ | 0.27 | 1.18 | 1.11 |
| CIFAR-10 | 0.69 | 1.49 | 1.81 |

The paper’s “mode collapse” language is qualitative and specific to the MNIST
`W_max` panel, where digit 1 is said to be overrepresented. It is not a
universal claim about all three datasets.
