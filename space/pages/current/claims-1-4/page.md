# Claims 1 and 4 — paper-scale SAD geometry and iDDPM endpoints

**Current verdict for Claim 1: BLOCKED. Confidence: LOW.**

**Current verdict for Claim 4: BLOCKED. Confidence: LOW.**

This page supersedes the historical random-linear SAD check and synthetic
convergence proxy. It contains faithful paper-scale geometry and generation
evidence, but two endpoint directions cannot establish the paper's complete
SAD-rank ordering or multi-direction monotonic trend.

## Exact claims and source

Claim 1 defines SADs as the eigenvectors of
`G_F = E[F_theta F_theta^T]`, ordered by eigenvalue, and conjectures that this
ordering governs generation preference. Claim 4 is the Figure 6 observation
that the author iDDPM performs better when rank-one data align with
small-eigenvalue SADs and degrades as the associated eigenvalue increases.

Source: arXiv:2510.22899v1, Definition 2 and Figure 6 discussion
(`main.tex` lines 231–258), with experimental setup at lines 343–364.
Retrieved source archive SHA-256:
`781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c`.

[Endpoint contract](evidence/claim_1_4_five_pair_aggregate/claim_contract.json) ·
[endpoint source audit](evidence/claim_1_4_five_pair_aggregate/source_audit.md) ·
[geometry contract](evidence/claim_1_4_geometry_million_aggregate/claim_contract.json) ·
[geometry source audit](evidence/claim_1_4_geometry_million_aggregate/source_audit.md)

## Assumption and scale audit

| Requirement | Executed value |
| --- | --- |
| Architecture | author iDDPM U-Net, `configs/1x16x16-iddpm.json` |
| Dimension | `D=256`, shape `1×16×16` |
| Geometry estimator | two disjoint 500,000-network shards; one million total |
| Training distribution | `N(0, 256 vv^T)` |
| Training examples / updates | 10,000 / 2,000 |
| Batch / optimizer | 1,000 / Adam, learning rate `1e-4` |
| Generated examples / reverse steps | 10,000 / 1,000 |
| Metric projections | 16,384 normalized Gaussian projections |
| Repetitions | five predeclared paired seed blocks |
| Hardware | HF `cpu-upgrade`, CPU only; PyTorch intra-op threads `4` |

Every endpoint used separate committed seeds for initialization, data,
shuffling, DSM noise, sampling, target samples, and projections. No result was
used to select a direction, seed, horizon, tolerance, or sample count.

## Paper-scale geometry result

The equal-weight one-million estimator gives:

| Quantity | Result |
| --- | ---: |
| Largest eigenvalue | `134.9141661931137` |
| Smallest eigenvalue | `0.09433513949682383` |
| Leading cross-shard vector overlap | `0.9976952438160434` |
| Smallest cross-shard vector overlap | `0.1993975461110904` |
| Individually stable audited directions | index `0` only |

The preregistered individual-vector gate was overlap at least `0.85`, together
with the local-band criterion. Indices
`31, 63, 95, 127, 159, 191, 223, 255` all failed individual-direction
readiness. The low-eigenvalue local band is stable, but its individual bottom
vector is not uniquely identified.

[Geometry verifier](evidence/claim_1_4_geometry_million_aggregate/verify.py) ·
[independent checker](evidence/claim_1_4_geometry_million_aggregate/independent_check.py) ·
[raw geometry JSON](evidence/claim_1_4_geometry_million_aggregate/raw_results.json) ·
[independent output](evidence/claim_1_4_geometry_million_aggregate/independent_checker_output.json) ·
[shard A input](evidence/claim_1_4_geometry_million_aggregate/inputs/shard_a.json) ·
[shard B input](evidence/claim_1_4_geometry_million_aggregate/inputs/shard_b.json)

## Five paired iDDPM endpoint results

Lower SW2 is better. Every seed has a positive
`largest SAD − smallest SAD` difference:

| Seed | Largest SW2 | Smallest SW2 | Difference | Largest MSW2 | Smallest MSW2 | Difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12.341827 | 9.818525 | +2.523301 | 54.839661 | 35.656101 | +19.183560 |
| 1 | 11.568896 | 8.878388 | +2.690507 | 43.442307 | 29.750766 | +13.691540 |
| 2 | 11.349946 | 4.925080 | +6.424866 | 43.400477 | 19.025318 | +24.375159 |
| 3 | 11.619387 | 8.336563 | +3.282824 | 44.984068 | 28.456957 | +16.527111 |
| 4 | 10.211508 | 5.694858 | +4.516651 | 42.128257 | 21.370234 | +20.758023 |

The verifier exhaustively enumerates all `5^5 = 3,125` ordered paired
bootstrap resamples:

| Metric | Mean difference | Exact percentile 95% interval |
| --- | ---: | ---: |
| SW2 | `3.8876298653973436` | `[2.7420881249702798, 5.279937038942132]` |
| MSW2 | `18.907078830476827` | `[15.828264140870534, 22.082122379390484]` |

All ten endpoint exact-setup and independent-checker flags pass. The
independent implementation reconstructs every difference, mean, and interval
without importing the primary verifier.

[Primary verifier](evidence/claim_1_4_five_pair_aggregate/verify.py) ·
[independent checker](evidence/claim_1_4_five_pair_aggregate/independent_check.py) ·
[frozen endpoint JSON](evidence/claim_1_4_five_pair_aggregate/inputs/endpoints.json) ·
[raw aggregate JSON](evidence/claim_1_4_five_pair_aggregate/raw_results.json) ·
[independent output](evidence/claim_1_4_five_pair_aggregate/independent_checker_output.json) ·
[summary](evidence/claim_1_4_five_pair_aggregate/summary.json)

## Negative controls and failure behavior

Reversing largest and smallest labels changes the mean effects to
`−3.8876298653973436` SW2 and `−18.907078830476827` MSW2 and is rejected by
the same positive-effect rule.
[Control output](evidence/claim_1_4_five_pair_aggregate/negative_control_output.json).

The aggregate verifier exits nonzero unless all ten endpoint flags pass,
every observed difference is positive, both exhaustive intervals exclude
zero, the reversed-label control is rejected, the one-million geometry audit
passes, the leading direction is stable, the smallest-direction instability
is disclosed, and the independent checker passes.

## Four verification-oriented routes

| Route | Distinct method | Result |
| --- | --- | --- |
| 1 | Exact author iDDPM endpoints using the earlier 16,000-network estimator | All five pairs favored the reported endpoint ordering; retained only as historical/downscaled evidence |
| 2 | Paper-scale geometry reconstruction from two independent 500,000-network shards | Eigensystem checks pass, but only the leading individual direction is stable |
| 3 | Exact paper-scale iDDPM endpoint training and generation over five paired seeds | All ten runs and the paired aggregate pass; this raises endpoint evidence substantially |
| 4 | Dedicated falsification search over the audited multi-direction contract | No assumption-satisfying counterexample: non-leading individual directions fail identifiability, so neither confirmation nor falsification of rank monotonicity is valid |

A missing or unstable direction is not counted as falsification. The
predeclared nine-direction node remains unrun because executing it would assign
causal ranks to arbitrary rotations inside unstable eigenspaces.

## Reproduce, compute, and provenance

Fixed cumulative command:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

Pinned environment:
[pyproject.toml](evidence/environment/pyproject.toml) and
[uv.lock](evidence/environment/uv.lock).

Accepted aggregate Git SHA:
`26b31ae7f0bb19c7901c72f54413326ecd729d9e`.
Run: `0c6c3210-941e-443c-ab16-dd524c804ac9`.
Pre-run aggregate estimate: one core and about five seconds; the fixed
cumulative suite had uncertain library allocation, so it ran on HF
`cpu-upgrade`. Actual allocation: 64 logical / 32 physical CPUs, affinity 64,
no GPU. Aggregate verifier: `0.103713` seconds; cumulative suite:
`16.540458` seconds; OpenResearch run duration: 42 seconds.
[Machine-readable runtime](evidence/claim_1_4_five_pair_aggregate/runtime_cpu.json) ·
[cumulative summary](evidence/claim_1_4_five_pair_aggregate/campaign_run_summary.json).

## Reviewer verdict

**Claim 1: BLOCKED.** The SAD definition and paper-scale eigensystem are
implemented, and endpoint preference is strongly corroborated. The conjectured
complete ordering is not verified or falsified because only one audited
individual SAD is stable.

**Claim 4: BLOCKED.** The exact Figure 6 endpoint effect is faithfully
corroborated across five paired seeds. The reported multi-direction trend is
not verified or falsified because the required interior individual SADs fail
the preregistered cross-shard identifiability gate.
