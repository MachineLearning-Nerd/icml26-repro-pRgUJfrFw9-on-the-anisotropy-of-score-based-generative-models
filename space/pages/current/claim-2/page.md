# Claim 2 — exact linear DSM theorem

**Current verdict: VERIFIED.** This page supersedes the historical formula-only
sanity check. It verifies Theorem 1 under its stated assumptions; it does not
generalize the theorem to arbitrary initial error matrices or nonlinear
networks.

## Exact claim and source

For rank-one Gaussian data `x ~ N(0, vvᵀ)`, fixed `sigma>0`, linear
`Omega=Phi Theta`, `v=u_i` an eigenvector of `Phi Phiᵀ`, zero-mean
initialization, a sufficiently small learning rate, and
`lambda_(D-1)>lambda_D>0`, Theorem 1 gives

`||E[Omega_t]-Omega*|| = O((1-2 eta rho_i)^t)`,

with the same `rho_i` for `i<D` and a strictly larger `rho_D`. Near the
optimum, the SGD-step covariance trace scales with `lambda_i`.

Source: arXiv:2510.22899v1, Theorem 1 (`main.tex` lines 182–185) and Appendix
C.2. The retrieved arXiv source archive has SHA-256
`781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c`.
[Exact contract](evidence/claim_2/claim_contract.json) ·
[source audit](evidence/claim_2/source_audit.md)

## Assumption audit

| Assumption | Numerical realization |
| --- | --- |
| Rank-one Gaussian data | `x=z u_i`, `z~N(0,1)` |
| Fixed positive noise | `sigma=1` for training/covariance; `sigma=1.25` in the independent check |
| Aligned unit direction | Every standard-basis eigenvector in `D=5` |
| Positive, gapped bottom spectrum | `[8,5,3,2,1]`; second rational spectrum `[21/2,7,9/2,5/2,3/4]` |
| Zero-mean initialization | Deterministic zero initialization |
| Small learning rate | Population GD `eta=0.01`; batch-one SGD `eta=0.001` |

## Executable verification

The fixed cumulative command is:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

Pinned environment: [pyproject.toml](evidence/environment/pyproject.toml),
[uv.lock](evidence/environment/uv.lock), Python
[3.12](evidence/environment/python-version.txt).

Executable source: [primary verifier](evidence/claim_2/verify.py) and
[independent checker](evidence/claim_2/independent_check.py). The primary
checker writes all raw JSON and exits nonzero unless its symbolic certificate,
exact recurrence, population training, SGD covariance, batch-one SGD,
independent check, and negative control all satisfy their predeclared
acceptance criteria.

The central population recurrence actually iterated by the verifier is:

```python
error = error - 2.0 * eta * A @ error @ B
```

This is compared both with exact rational iterates and with the closed-form
rate; the proof conclusion is not inferred from a fitted finite trajectory.

## Raw observed evidence

| Test | Observed result | Acceptance |
| --- | ---: | --- |
| Exact rational recurrence | maximum error `0` | exact equality |
| Population-GD contraction | maximum factor error `2.3518798475397062e-07` | `<2e-5` |
| Covariance trace, 600,000 samples | maximum relative error `0.009539070308304845` | `<0.06` |
| Covariance trace vs `lambda_i` | Pearson `r=0.9999984740754329` | `>0.999` |
| Batch-one SGD, 96 repeats × 4,000 steps × 5 directions | late error/eigenvalue `r=0.9984768651353283` | supportive execution |
| Independent finite-difference gradient | maximum error `1.2695569040488408e-09` | `<1e-7` |
| Independent projected-Hessian rates | maximum error `0` | `<1e-10` |
| Wrong `rho_i=sigma² lambda_i` control | rejected; discrepancy `7` | must reject |

[Raw JSON](evidence/claim_2/raw_results.json) ·
[independent output](evidence/claim_2/independent_checker_output.json) ·
[negative-control output](evidence/claim_2/negative_control_output.json) ·
[complete run log](evidence/claim_2/run.log) ·
[evaluation record](evidence/claim_2/EVAL.md)

Seeds are fixed in source before execution: covariance blocks
`20000+100*i+block`, SGD `91000+i`, and independent checker `187`.

## Compute and provenance

Successful Git SHA: `6864b70ce711c738f306848dea56ffefbff83898`.
Run: `f696ab27-32f4-49e2-82b6-b7d9d71b144a`. Pre-run estimate: 2 CPU
cores, 1–4 minutes. Selected backend/flavor: Hugging Face `cpu-upgrade`.
Reported allocation: 64 logical CPUs, 32 physical CPUs, affinity 64. Full job:
32 seconds; Claim 2 checker: 5.512556 seconds; cumulative suite: 6.046443
seconds. GPU use: none. [Machine-readable runtime record](evidence/claim_2/runtime_cpu.json).

## Independent check, control, and limitation

The independent implementation constructs the full Kronecker Hessian and
finite-differences the objective rather than using the coordinatewise
derivation. Projected onto the invariant diagonal mean-error subspace reached
from the theorem's zero-mean initialization, it recovers
`[1.5625,1.5625,1.5625,1.5625,2.5625]` exactly.

The unrestricted Hessian instead has `[1.5625]` in all five directions.
Arbitrary off-diagonal perturbations therefore do **not** have the selected
direction advantage. This is a useful boundary of the theorem: those modes
are absent under its initialization and remain unreachable under its
population update. The verifier does not silently claim the stronger result.

The symbolic reconstruction is independently machine-checked algebra, not a
Lean/Coq proof object. The finite training experiment is `D=5`; the universal
conclusion comes from the symbolic rate identities and invariant-subspace
argument, not from extrapolating that finite run.
