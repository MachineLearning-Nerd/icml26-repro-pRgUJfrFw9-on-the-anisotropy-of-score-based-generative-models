# Source audit — one-million-network aggregate

The frozen arXiv source archive has SHA-256
`781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c`.
Figure 6 and its discussion are at `main.tex` lines 231–258 and the setup is at
lines 343–364. The author implementation's `geometry.py` defaults to one
million fresh initialized networks.

Runs `9027461d-3dae-46b2-b88d-cef5ed0592a0` and
`d7ea7073-d087-4dc1-9268-969c08b31123` independently completed disjoint
500,000-network shards. This verifier checks their exact commits, run IDs,
counts, seed ranges, hashes, CPU/no-GPU records, controls, and matrices before
forming the equal-weight one-million aggregate.
