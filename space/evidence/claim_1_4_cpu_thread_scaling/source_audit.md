# Source and scope audit

This diagnostic does not test a paper claim. It addresses a compute obstruction
encountered while reproducing arXiv:2510.22899v1 Figure 6 under the required
CPU-only policy.

The scientific endpoint remains the author’s `1x16x16-iddpm.json` architecture,
rank-one `N(0, Dvv^T)` data, batch 1,000, Adam training, 2,000 optimizer
updates, 1,000 reverse steps, and 10,000 generated samples. Only PyTorch
intra-op scheduling is examined here. The paper does not quantify CPU thread
count, and no generation metric is computed by this diagnostic.
