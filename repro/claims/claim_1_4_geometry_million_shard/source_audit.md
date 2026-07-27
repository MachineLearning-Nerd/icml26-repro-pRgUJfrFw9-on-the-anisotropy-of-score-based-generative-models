# Source audit — paper-scale iDDPM geometry

The frozen arXiv source archive has SHA-256
`781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c`.
Figure 6 and its discussion are at `main.tex` lines 231–258 and its setup is at
lines 343–364. The author implementation's `geometry.py` defaults to
`num_samples=1000000`; for each sample it constructs a fresh `Diffuser`, draws
one uniform model timestep, probes the zero tensor when `sigma=0`, and averages
the resulting score outer products.

This node implements one prespecified 500,000-network half. A sibling uses a
disjoint seed range. Their equal-weight aggregate has exactly one million fresh
initializations and the same probe distribution. Streaming outer products and
CPU multiprocessing alter storage and execution order, not the estimand.

Geometry estimation alone cannot establish generation preference. It is a
prerequisite for identifying interior directions faithfully enough to run the
paper's multi-direction generation experiment.
