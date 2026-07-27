# Source audit — five-pair endpoint aggregate

The frozen arXiv 2510.22899 source archive has SHA-256
`781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c`.
Figure 6 and its discussion are anchored at source lines 231–245 and 254–258;
the experimental setup is at lines 343–364.

The paper reports a clear trend rather than a universally quantified strict
monotonic theorem. Its setup uses a D=256 iDDPM U-Net, rank-one Gaussian data,
10,000 training examples, 2,000 updates, 1,000 reverse steps, 10,000 generated
samples, five seeds, and 16,384 MSW2 projections. Each frozen endpoint source
run implements those quantities and retains an independent metric checker.

This aggregate tests only the endpoint component. Its source directions came
from the earlier 16,000-network geometry estimate. The accepted paper-scale
audit later found the smallest individual direction unstable across two
500,000-network shards, so this result cannot settle the exact paper-scale
claim or substitute for the predeclared multi-direction rank test.
