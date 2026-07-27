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

This aggregate tests only the endpoint component using the accepted
one-million-network geometry estimate. Its ten endpoint runs retain the
paper's architecture and stated numerical quantities. The accepted geometry
audit found the leading direction stable but the smallest individual direction
unstable across the two 500,000-network shards. The deterministic
full-estimator direction is nevertheless the exact preregistered input; no
generation result is used to select or rotate it. Endpoint evidence cannot
substitute for the predeclared multi-direction rank test.
