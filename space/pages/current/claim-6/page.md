# Claim 6 — architecture-specific average geometries

**Current verdict: VERIFIED** for the exact final-layer families and parameter
assumptions stated in Propositions 1–3. This supersedes the historical script
that merely constructed matrices already having the claimed forms.

## Exact claims

- MLP: for `z=phi(W h+b)` with elementwise `phi` and iid output-coordinate
  parameters, `G=alpha I+beta 11ᵀ`.
- CNN: for iid output-channel convolutional filters,
  `G=I_Cout tensor A + 11ᵀ_Cout tensor B`.
- Transformer-style output: a shared zero-mean iid token map followed by
  fixed orthonormal `Q` gives `G=Q(A tensor I_Lout)Qᵀ`, hence at most `T`
  distinct eigenvalues.

Source: arXiv:2510.22899v1, Appendix B Propositions 1–3 and proofs C.4–C.6.
[Exact contract](evidence/claim_6/claim_contract.json) ·
[source audit](evidence/claim_6/source_audit.md)

## Proof reconstruction

Conditioned on an arbitrary hidden representation `h`, iid MLP output rows
have one common mean and variance; iid CNN output channels have one common
spatial mean vector and covariance. For the zero-mean transformer weights,

```text
E[(W a)_r (W c)_s] = delta_rs Var(W) aᵀc.
```

These conditional identities give the three matrix families exactly.
Expectation over any probe distribution preserves them. In the transformer
case every eigenvalue of the `T×T` factor repeats `Lout` times, and orthonormal
`Q` preserves the spectrum.

## Actual networks and uncertainty

| Architecture | Executed setup | Relative structural residual |
| --- | --- | ---: |
| MLP | tanh, 20 outputs, 12 hidden features, 36,000 networks | `0.02498127081761267` |
| CNN | tanh Conv1d, 4 outputs × 12 positions, 24,000 networks | `0.03095807180606085` |
| Transformer | 7 tokens × 4 outputs, shared linear map, orthonormal `Q`, 32,000 networks | `0.02953069280896221` |

The MLP's separately estimated conditional coefficients agree within
`0.009033` (`alpha`) and `0.004113` (`beta`). The CNN projected geometry is
positive definite with minimum eigenvalue `0.280398`. The transformer fitted
token factor differs from the analytical factor by `0.008991` relatively and
has exactly 7 eigenvalue groups for `T=7`; maximum exact within-group spread is
`9.77e-15`. Block residuals and their standard errors are included in raw
JSON.

An independent implementation exhaustively enumerates small Rademacher
parameter domains and obtains maximum error `0` for all three architectures.

## Controls

| Broken assumption | Residual | Rejected |
| --- | ---: | --- |
| MLP output coordinate 0 not identically distributed | `0.10146382325658827` | yes |
| CNN output channel 0 not identically distributed | `0.20878770000538335` | yes |
| Transformer weights have nonzero mean | theory `0.9253981874214255`; best structured projection `0.8018929141743302` | yes |

## Reproduce and inspect

Fixed cumulative command:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

[Verifier](evidence/claim_6/verify.py) ·
[independent checker](evidence/claim_6/independent_check.py) ·
[raw JSON](evidence/claim_6/raw_results.json) ·
[independent output](evidence/claim_6/independent_checker_output.json) ·
[negative controls](evidence/claim_6/negative_control_output.json) ·
[complete log](evidence/claim_6/run.log) ·
[method](evidence/claim_6/method.md) ·
[limitations](evidence/claim_6/limitations.md)

Git `2d1c974bd148eb0088590b216581d48ea230d077`; run
`af45ac22-f33b-447a-94e4-3f9614b8f66b`. Pre-run estimate: 8 cores, 2–5
minutes. Hugging Face `cpu-upgrade` reported 64 logical / 32 physical CPUs,
affinity 64. Runtime: 42 seconds for the full job, 5.788714 seconds for the
Claim 6 checker, and 14.460175 seconds for the cumulative suite; no GPU.
All dimensions, sample counts, thresholds, and deterministic seeds were fixed
in committed source before execution.

This proves the propositions for their specified output layers. It does not
claim that every unconstrained architecture with the same broad label has the
form, and it does not by itself verify generation preference.
