# Claim 2 evaluation record

Verdict: **VERIFIED**, scoped to the exact assumptions of Theorem 1 in
arXiv:2510.22899v1.

The successful evidence is run `f696ab27-32f4-49e2-82b6-b7d9d71b144a` from
Git commit `6864b70ce711c738f306848dea56ffefbff83898`.

The fixed campaign command was:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

The run used Hugging Face `cpu-upgrade`. Before launch, the estimate was 2 CPU
cores and 1–4 minutes because the vectorized Monte Carlo and environment sync
had uncertain local runtime. The job reported 64 logical CPUs, 32 physical
CPUs, and an affinity of 64. The full job took 32 seconds; the Claim 2 checker
took 5.512556 seconds and the cumulative checker suite took 6.046443 seconds.

Acceptance evidence:

- the exact rational recurrence had maximum error 0;
- the population-GD observed contraction factors differed from their
  predictions by at most `2.3518798475397062e-07`;
- 600,000 independently seeded samples gave a maximum covariance-trace
  relative error of `0.009539070308304845` and Pearson correlation
  `0.9999984740754329` with the conditioning eigenvalues;
- batch-one SGD was actually executed for 96 repetitions per aligned
  direction and its late error/eigenvalue correlation was
  `0.9984768651353283`;
- an independent finite-difference gradient check had maximum absolute error
  `1.2695569040488408e-09`, and the independently constructed Hessian
  recovered the theorem's rates exactly on the invariant mean-error subspace;
- the wrong rule `rho_i = sigma^2 lambda_i` was rejected with maximum
  discrepancy 7.

The unrestricted Hessian result is retained in `raw_results.json`: arbitrary
off-diagonal perturbations do not have the selected-direction advantage. They
are excluded by the theorem's zero-mean initialization and are unreachable
under its population update, so this is an assumption-sensitivity result, not
a contradiction.

The verifier exits nonzero unless every required item passes. The independent
checker and negative control have separate outputs.
