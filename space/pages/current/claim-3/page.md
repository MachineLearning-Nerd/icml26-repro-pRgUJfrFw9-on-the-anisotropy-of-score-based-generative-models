# Claim 3 — exact alignment extrema

**Current verdict: VERIFIED.** This proof-level verifier supersedes the
historical `D=6` numerical example.

## Exact claim

Let `G_F=U diag(lambda) Uᵀ` and the data second moment
`M=V diag(nu) Vᵀ` be positive semidefinite, with both eigenvalue sequences in
descending order. Over **every** orthogonal `D×D` transformation `W`,

- `W=U Vᵀ` maximizes `alpha(W)=tr(G_F W M Wᵀ)`;
- `W=U J Vᵀ`, where `J` reverses the coordinates, minimizes it.

Repeated eigenvalues may make extrema non-unique; the displayed transforms
remain extrema. Source: arXiv:2510.22899v1, alignment definition
(`main.tex` lines 271–276), Theorem 2 (lines 280–283), and Appendix C.3.
[Exact contract](evidence/claim_3/claim_contract.json) ·
[source audit](evidence/claim_3/source_audit.md)

## Why this establishes the universal statement

Set `R=Uᵀ W V` and `P_ij=R_ij²`. Orthogonality makes every row and column sum
of `P` equal one, while

`alpha(W)=sum_(i,j) lambda_i nu_j P_ij`.

The verifier symbolically factors a two-assignment exchange:

```text
(a*x + b*y) - (a*y + b*x) = (a-b)(x-y).
```

For descending pairs, this is nonnegative. Adjacent exchanges yield the
identity maximum and reversal minimum among permutations. Linearity gives
those same bounds over the entire doubly-stochastic polytope; both bounds are
attained by orthogonal permutation matrices. The proof therefore covers the
whole orthogonal group rather than extrapolating a finite experiment.

## Executable and raw evidence

Fixed cumulative command:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

[Primary verifier](evidence/claim_3/verify.py) ·
[independent checker](evidence/claim_3/independent_check.py) ·
[raw JSON](evidence/claim_3/raw_results.json) ·
[independent output](evidence/claim_3/independent_checker_output.json) ·
[negative-control output](evidence/claim_3/negative_control_output.json) ·
[complete log](evidence/claim_3/run.log) ·
[method](evidence/claim_3/method.md) ·
[limitations](evidence/claim_3/limitations.md)

| Check | Raw result |
| --- | ---: |
| Symbolic exchange | exactly `(a-b)(x-y)` |
| Doubly-stochastic maximum error | `1.1102230246251565e-15` |
| Complete permutation domains | 409,112 permutations, `D=2…9` |
| Independent maximum error | `5.684341886080802e-14` |
| Independent minimum error | `2.1316282072803006e-14` |
| Haar-orthogonal sweep | 384/384 inside certified bounds |
| Degenerate-spectrum audit | tied ordering also achieves maximum `46.5` |
| Wrong identity-is-minimum control | rejected, gap `54.75` |

All cases, dimensions, and seeds are fixed in source. The executable exits
nonzero if any required check or the negative-control rejection fails.

## Compute and limitations

Successful Git SHA:
`291c2adfa131af97b82b6ff3b617509f24ada30f`. Run:
`843c3b36-403d-4e3a-83b1-bfdaf5b3aa58`. Pre-run estimate: 2 cores, 1–3
minutes. Hugging Face `cpu-upgrade` reported 64 logical / 32 physical CPUs and
affinity 64. Full job: 25 seconds; Claim 3 checker: 2.201216 seconds;
cumulative suite: 8.139307 seconds. GPU use: none.
[Runtime JSON](evidence/claim_3/runtime_cpu.json)

The certificate is a machine-checked symbolic/combinatorial reconstruction,
not a Lean/Coq proof object. The exhaustive and random checks are regression
evidence; they are not used as the basis for the universal quantifier. This
trace theorem alone makes no empirical claim about generation quality.
