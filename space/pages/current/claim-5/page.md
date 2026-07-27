# Claim 5 — real-image alignment experiment

**Current verdict: BLOCKED. Confidence: LOW.** This page supersedes the
historical `1/alignment` synthetic proxy. No toy, resource, or published-paper
value is presented as reproduction evidence.

## Exact claim and source

For MNIST, CelebA-HQ, and CIFAR-10, the alignment-minimizing transform
`W_min` is reported to substantially and consistently improve five-seed
average SW2 relative to natural alignment `I` under the exact iDDPM setup.
Figure 8 separately reports qualitative signs of mode collapse for the MNIST
`W_max` samples, specifically over-representation of digit 1.

Source: arXiv:2510.22899v1, Figures 8–9 (`main.tex` lines 285–305) and
Appendix A (`main.tex` lines 343–364). The arXiv source archive SHA-256 is
`781502b4dee07a2d43b8a6971312ec7678255655d24bb71d3f1c290dde7f0c7c`.
[Exact contract](evidence/claim_5_four_route_audit/claim_contract.json) ·
[source audit](evidence/claim_5_four_route_audit/source_audit.md)

## Assumptions and quantifiers

| Requirement | Exact value |
| --- | --- |
| Datasets | MNIST `1×28×28`; CelebA-HQ `1×56×56`; CIFAR-10 `3×32×32` |
| Dataset size | 10,000 each |
| Independent runs | 5 seeds per transform and dataset |
| Transforms | `W_min`, natural `I`, `W_max` |
| Architecture | author iDDPM |
| Batch / diffusion steps | 500 / 1,000 |
| Optimizer steps | 100,000 MNIST; 100,000 CelebA-HQ; 200,000 CIFAR-10 |
| Metric | SW2 with `64D` normalized random projections |
| Direct verification requirement | 45 full runs with raw seed values, uncertainty, controls, and exact transforms |

The MNIST mode-collapse observation is not universally quantified over
CelebA-HQ or CIFAR-10.

## Four completed routes

| Route | Method | Result | Why it does or does not resolve the claim |
| --- | --- | --- | --- |
| 1 | Exact author configs and exact batch 500; one warm-up plus one timed CPU update | All three executed | Resource evidence only; no real data, transform, horizon, generation, metric, or five seeds |
| 2 | Author repository release/tag/file audit plus HF model/dataset/Space searches | No checkpoint or raw seed artifact located | Missing capability is not evidence for or against the claim |
| 3 | Independent audit of primary Figure 9 image | Published `W_min` bar is lower for all three | Paper evidence, not a reproduction; seed-level uncertainty is unavailable |
| 4 | Dedicated assumption-satisfying falsification search | No valid counterexample | Published aggregates support the ordering; unavailability or incomplete execution cannot falsify |

[Method](evidence/claim_5_four_route_audit/method.md) ·
[public-artifact audit](evidence/claim_5_four_route_audit/public_artifact_audit.json) ·
[limitations](evidence/claim_5_four_route_audit/limitations.md)

## Paper evidence, kept separate

The primary Figure 9 image
(`https://ar5iv.labs.arxiv.org/html/2510.22899/assets/x15.png`, SHA-256
`b7e23901195f548379823dff020de27a4424cdccfc62bd2da60cad6142751922`)
contains:

| Dataset | `W_min` | `I` | `W_max` | Published reduction, `W_min` vs `I` |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 0.11 | 1.91 | 1.81 | 94.24% |
| CelebA-HQ | 0.27 | 1.18 | 1.11 | 77.12% |
| CIFAR-10 | 0.69 | 1.49 | 1.81 | 53.69% |

These numbers support the paper’s statement but do not provide an independent
reproduction or seed-level uncertainty.

## Observed evidence in this campaign

The exact-shape resource probe used the author networks and paper batch size:

| Dataset | Parameters | Timed update | Paper steps | Training-only linear projection |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 551,713 | 8.160469 s | 100,000 | 226.6797 h/model |
| CelebA-HQ | 718,049 | 32.910655 s | 100,000 | 914.1849 h/model |
| CIFAR-10 | 719,203 | 9.636662 s | 200,000 | 535.3701 h/model |

The projections exclude geometry estimation, dataset transformation, sampling,
SW2 evaluation, and repetition across 45 models. They are deliberately labeled
planning estimates and cannot earn scientific credit.

[Raw JSON](evidence/claim_5_four_route_audit/raw_results.json) ·
[independent output](evidence/claim_5_four_route_audit/independent_checker_output.json) ·
[negative-control output](evidence/claim_5_four_route_audit/negative_control_output.json) ·
[complete run log](evidence/claim_5_four_route_audit/run.log) ·
[evaluation record](evidence/claim_5_four_route_audit/EVAL.md)

## Executable audit and failure behavior

Fixed cumulative command:

```text
uv sync --locked && .venv/bin/python repro/run_campaign.py
```

Pinned environment:
[pyproject.toml](evidence/environment/pyproject.toml) and
[uv.lock](evidence/environment/uv.lock).
Executable source:
[primary verifier](evidence/claim_5_four_route_audit/verify.py) and
[independent checker](evidence/claim_5_four_route_audit/independent_check.py).

The audit exits nonzero if a route is missing or mislabeled, arithmetic
disagrees, a failed execution is called falsification, the invalid one-update
control is not rejected, GPU use appears, or the result is misrepresented as
anything other than `BLOCKED/LOW`.

## Compute, seeds, and reviewer verdict

Successful Git SHA:
`6a9a52a024df44a8757c8345961f557fb798d6d9`.
Run: `8948914c-470d-4c41-9b56-44243d6545d8`.
Seeds: MNIST `750001`, CelebA-HQ `750002`, CIFAR-10 `750003`.
Pre-run estimate: 4 required CPU cores and 5–45 minutes. Selected compute:
Hugging Face `cpu-upgrade`. Actual allocation: 64 logical / 32 physical CPUs,
affinity 64, four PyTorch intra-op threads. Peak RSS: 15,567 MB. Claim audit:
103.891004 seconds; wrapper 114.457548 seconds; cumulative suite 146.942617
seconds. GPU use: none.
[Machine-readable runtime](evidence/claim_5_four_route_audit/runtime_cpu.json).

**Reviewer verdict: BLOCKED.** The exact claim was neither verified nor
falsified. It requires public exact checkpoints/transformed datasets/raw
five-seed outputs, or CPU resources sufficient for the 45 complete
training-and-generation runs.
