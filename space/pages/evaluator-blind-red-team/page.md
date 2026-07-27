# Evaluator-blind pre-publication red-team

The review used only a fresh archive of the candidate `space/` tree, its
canonical `README.md`, `logbook.json`, `pages/index.md`, and pages reachable
from the logbook. Repository source, OpenResearch descriptions, dashboard
files, and unpublished branches were not used to fill gaps.

## First traversal — failed

Files opened:

```text
README.md
pages/index.md
pages/current/claims-1-4/page.md
pages/current/claim-2/page.md
pages/current/claim-3/page.md
pages/current/claim-5/page.md
pages/current/claim-6/page.md
pages/verification-run/page.md
pages/evidence/page.md
pages/historical-rejected-baseline/page.md
```

The reviewer could not verify three advertised “complete run log” links:
`evidence/claim_2/run.log`, `evidence/claim_3/run.log`, and
`evidence/claim_6/run.log` were absent. Claim 3 did not expose its assumptions
under an explicit heading. Claim 6 did not state runtime and deterministic
seeds in sufficiently discoverable provenance language. The candidate was
therefore not release-ready.

## Fixes

The three exact OpenResearch logs were added additively. Claim 3 now has an
explicit assumptions-and-quantifiers section. Claim 6 now explicitly names
runtime and deterministic seeds. No scientific verdict or numerical result
was changed.

## Repeated traversal

The same entrypoint-only traversal was repeated from a new fresh archive. For
all six claims, the reviewer directly located the exact claim and source,
assumptions, fixed command, pinned environment, executable source, inline raw
numbers, downloadable raw JSON, independent checker, negative control,
limitations, verdict, Git/run provenance, seeds, CPU allocation, and runtime.
Every advertised local link resolved. The visibility matrix has no missing
cells.

**Reviewer conclusion: evaluator-visible release gate PASS.** This conclusion
means the evidence is discoverable; it does not change the scientific BLOCKED
verdicts or the live judged score.
