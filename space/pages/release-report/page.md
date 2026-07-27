# Final release report

Previous live judged score: `5/12`

Conservative projected score range after the proposed change: `6/12–8/12`

Best-supported possible new score: `8/12` **forecast, not a judge result**

The current total remains `5/12` until the live judge evaluates the new
Hugging Face revision. This report does not convert forecasts into earned
points.

## Claim summary

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 1 | 1 | LOW | BLOCKED | One-million SAD geometry and five paired iDDPM endpoints are faithful, but only the leading individual SAD is stable |
| 2 | 1 | 2 | HIGH | VERIFIED | Exact theorem reconstruction, population and SGD execution, independent checker, and a rejecting control |
| 3 | 1 | 2 | HIGH | VERIFIED | Universal symbolic exchange certificate, complete finite domains, independent assignment solver, and a rejecting control |
| 4 | 1 | 1 | LOW | BLOCKED | Every endpoint pair favors the small-eigenvalue direction, but unstable interior SADs prevent the full monotonic rank test |
| 5 | 0 | 0 | LOW | BLOCKED | Four materially distinct routes are complete; no direct 45-run reproduction or assumption-satisfying counterexample is available |
| 6 | 1 | 2 | HIGH | VERIFIED | Conditional derivations, actual executable architectures, exhaustive independent checks, and assumption-breaking controls |

Claims 2, 3, and 6 changed from judge-rated TOY evidence to current VERIFIED
evidence. Claims 1 and 4 gained faithful paper-scale corroboration but remain
BLOCKED on the exact multi-direction quantifier. Claim 5 replaced a tautology
with a four-route capability and falsification audit, but remains BLOCKED.

## Experiment tree and winning result

The campaign froze the baseline, developed exact theorem checks on stacked
children, reconstructed the author iDDPM geometry in two independent
500,000-network shards, ran ten paper-scale endpoints in five paired seed
blocks, and descended to an exact five-pair aggregate.

The strongest scientific release candidate is
`orx/evaluator-visible-paper-scale-release-candidate` at Git
`d5da257fd3d63e6f78ed1ac2b2f55d17f6cd444d`. Its formal run
`03e62e62-1240-434c-9a8d-5abe4e86e84c` completed in 2m02s and all seven
cumulative checkers exited zero.

All experiment nodes inherited this exact command:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

All CPU tasks with uncertain runtime used Hugging Face `cpu-upgrade`, CPU
only, via the official `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` image.
The accepted candidate received 64 logical / 32 physical CPUs, affinity 64,
and used no GPU. Its cumulative verifier runtime was `98.328770` seconds.
Hugging Face monetary cost was not exposed by the OpenResearch run record, so
no dollar estimate is invented. Local work was limited to short, one-core
static validation.

## Evidence and preservation

Canonical pages:

- [Claims 1 and 4](#/current-claims-1-4)
- [Claim 2](#/current-claim-2)
- [Claim 3](#/current-claim-3)
- [Claim 5](#/current-claim-5)
- [Claim 6](#/current-claim-6)
- [Visibility matrix](#/evidence-matrix)

The judged Space revision
`33d98421a79a7686497639f2d4d6163fc5acfa3f` is preserved under
`historical/judged-33d98421/`. Every entry in its protected manifest
recomputed to the recorded SHA-256 before release. The current navigation
labels that material exactly **Historical rejected baseline**.

## Remaining BLOCKED risks

- Claims 1 and 4 need individually identifiable interior SADs or a
  subspace-invariant reformulation supported by the paper; the current
  one-million estimator does not provide that.
- Claim 5 needs public exact checkpoints/transformed datasets/raw five-seed
  outputs, or CPU capacity for the 45 complete training-generation runs.

The exact publication action, after every gate passes, is a text-only commit
to the existing Space `DineshAI/pRgUJfrFw9`, using only the paths in
`release/upload_allowlist.txt`, followed by hash verification of the published
revision. The same published text paths and the illustrated report/notebook
will then be mirrored to GitHub `main`. No second Space will be created, and
no score increase will be claimed before a live judge verdict.
