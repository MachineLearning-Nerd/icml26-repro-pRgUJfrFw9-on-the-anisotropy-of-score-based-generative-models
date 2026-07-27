# Current paper-scale endpoint verification

This is the current five-pair verifier for the paper-scale Claims 1 and 4
endpoint experiment. It supersedes the 16,000-network historical endpoint
aggregate, which remains preserved on its completed experiment branch.

Run the fixed campaign command:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

The campaign invokes `verify.py`, which reads the ten frozen terminal endpoint
summaries in `inputs/endpoints.json`, enumerates all `5^5 = 3,125` paired
bootstrap resamples, runs the reversed-label negative control, and invokes
`independent_check.py`. The verifier exits nonzero unless every endpoint
integrity check passes, every observed largest-minus-smallest difference is
positive, both exhaustive percentile intervals exclude zero, the control is
rejected, and the independent checker passes.

The formal run must expose these log records:

- `CLAIM_1_4_FIVE_PAIR_SUMMARY`
- `CLAIM_1_4_FIVE_PAIR`
- `CLAIM_1_4_FIVE_PAIR_INDEPENDENT`
- `CAMPAIGN_SUMMARY`

The scoped scientific status is
`PAPER_SCALE_FIVE_PAIR_ENDPOINT_CORROBORATION`. Even when the verifier passes,
Claims 1 and 4 remain `BLOCKED_PENDING_MULTI_DIRECTION_RANK_ASSOCIATION`: two
endpoint directions do not establish the paper's full direction ordering, and
the smallest individual eigenvector fails the preregistered cross-shard
stability gate.
