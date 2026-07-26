# Source audit — Claims 1 and 4 endpoint generation

The source is the frozen arXiv 2510.22899 source archive with SHA-256
`781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c`.
The complete campaign audit is in `.openresearch/provenance/source_audit.md`.

Figure 6 and its discussion are anchored at source lines 231–245 and 254–258;
the experimental setup is at lines 343–364. The paper uses a D=256 iDDPM
U-Net, rank-one Gaussian data `N(0,Dvv^T)`, 10,000 data points, 2,000
optimizer steps (200 epochs, batch 1,000), 1,000 diffusion steps, five seeds,
and MSW2 from `64D=16,384` projections. Its network geometry uses the
untrained architecture under the default zero-input/noise-level probe.

The paper describes a “clear trend”: small-eigenvalue SAD directions are
modeled better and large-eigenvalue directions worse. It does not assert a
strict pointwise monotonic theorem. The eventual contract therefore requires
both endpoint ordering with paired-seed uncertainty and a separately
predeclared multi-direction rank test.

This node is one half of the first paired endpoint seed. It cannot by itself
verify or falsify the trend.
