# OpenResearch reproduction

## Collection classification and audit boundary

This repository is a **legacy/source workspace** for *On the Anisotropy of Score-Based Generative Models*
(arXiv `2510.22899`, OpenReview `pRgUJfrFw9`). It is preserved
separately from the standardized canonical record at
[`icml26-score-anisotropy`](https://github.com/MachineLearning-Nerd/icml26-score-anisotropy).

The claim results and scores recorded below are historical results of this
workspace. They are not new paper-level verifications performed while
organizing the collection. The collection audit did not run the scientific
implementation; the canonical record documents its own scoped status and
limitations.

### How the historical claim evidence is produced

The claim table and experiment log below are the authoritative mapping from
each paper claim to its producer, command, control, and evidence artifact. In
this workspace, the campaign runners generate the finite SAD geometry and endpoint diagnostics, then persist raw tables, figures, and the report surfaces named in the experiment log.

The former `orx/*` branches are historical workstreams, not additional final
publication claims. Their purposes and tips are preserved in
[`BRANCH_AUDIT.md`](BRANCH_AUDIT.md). Citation and author acknowledgment
details are in [`CITATION.cff`](CITATION.cff) and
[`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

## Thank you

Thank you to the paper authors for making this research available for study. The full acknowledgment is in [`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

This campaign tests the paper's central SAD-generation claim with the author
`D=256` iDDPM configuration on CPU. The paper reports that generation quality
degrades toward large-eigenvalue SADs. Across five exact paired endpoint runs,
we observe a mean largest-minus-smallest SW2 gap of `+3.887630` (exact paired
bootstrap 95% interval `[2.742088, 5.279937]`) and an MSW2 gap of `+18.907079`
(`[15.828264, 22.082122]`); lower is better.

**Assessment:** paper-scale endpoint corroboration, but Claims 1 and 4 remain
BLOCKED rather than verified. The one-million-network geometry audit found
that only the leading individual SAD is stable across independent shards, so
the paper's full multi-direction rank trend cannot be tested without assigning
meaning to unstable eigenvector rotations. Claims 2, 3, and 6 are VERIFIED;
Claim 5 is BLOCKED after four distinct routes.

The faithful endpoint setup uses 10,000 training samples, 2,000 optimizer
updates, 10,000 generated samples, 1,000 reverse steps, five paired seeds, and
16,384 projections. The only substitution is scope: extreme endpoints are
tested, while the unstable interior-direction sweep is not run. Compute is
Hugging Face `cpu-upgrade`, CPU only, with four PyTorch intra-op threads for
generation and no GPU.

[Read the illustrated technical report](reports/anisotropy-paper-scale/report.md) ·
[Open the tutorial notebook](notebooks/anisotropy_reproduction.py) ·
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-pRgUJfrFw9-on-the-anisotropy-of-score-based-generative-models/blob/main/notebooks/anisotropy_reproduction.py)

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| `main` | Publication surface | Not run as an experiment (publication surface) | README, report, notebook, and evaluator pages | none |
| [`orx/claims-1-and-4-one-million-geometry-aggregate`](https://github.com/MachineLearning-Nerd/icml26-repro-pRgUJfrFw9-on-the-anisotropy-of-score-based-generative-models/tree/orx/claims-1-and-4-one-million-geometry-aggregate) | Combine two independent 500k geometry shards | `uv sync --locked && .venv/bin/python repro/run_campaign.py` | one-million eigensystem passes; only index 0 individually stable | HF `cpu-upgrade`, 32 physical CPUs, 1m51s |
| [paper-scale endpoint branches](https://github.com/MachineLearning-Nerd/icml26-repro-pRgUJfrFw9-on-the-anisotropy-of-score-based-generative-models/branches) | Train and generate largest/smallest SAD pairs for seeds 0–4 | `uv sync --locked && .venv/bin/python repro/run_campaign.py` | all five paired SW2 and MSW2 gaps positive | HF `cpu-upgrade`, torch 4 threads, 2h27m–4h13m each |
| [`orx/claims-1-and-4-one-million-five-pair-aggregate`](https://github.com/MachineLearning-Nerd/icml26-repro-pRgUJfrFw9-on-the-anisotropy-of-score-based-generative-models/tree/orx/claims-1-and-4-one-million-five-pair-aggregate) | Exact paired aggregation and reversed-label control | `uv sync --locked && .venv/bin/python repro/run_campaign.py` | intervals exclude zero; Claims 1/4 remain BLOCKED on rank association | HF `cpu-upgrade`, 32 physical CPUs, 42s |

---

# [On the Anisotropy of Score-Based Generative Models](https://arxiv.org/abs/2510.22899)
Andreas Floros, Seyed-Mohsen Moosavi-Dezfooli, Pier Luigi Dragotti

## Abstract
We investigate the role of network architecture in shaping the inductive biases of modern score-based generative models. To this end, we introduce the Score Anisotropy Directions (SADs), architecture-dependent directions that reveal how different networks preferentially capture data structure. Our analysis shows that SADs form adaptive bases aligned with the architecture's output geometry, providing a principled way to predict generalization ability in score models prior to training. Through both synthetic data and standard image benchmarks, we demonstrate that SADs reliably capture fine-grained model behavior and correlate with downstream performance, as measured by Wasserstein metrics. Our work offers a new lens for explaining and predicting directional biases of generative models.

<p align="center">
  <img src="assets/spheres.svg" />
</p>

## Setup

```
pip install torch torchvision timm
```

## Organization

- ``geometry.py`` is used to precompute the average geometry for experiments that need it.
- ``main.py`` is the entry point for all experiments.

Example usage (run from README directory):

```sh
# Figure 1
sh scripts/sphere.sh
# Hadamard image of Figure 2
sh scripts/hadamard.sh
# DiT/4 SADs of Figure 6
sh scripts/eigen-dit-ps4.sh
```

## Citation
```BibTex
@misc{floros2025anisotropyscorebasedgenerativemodels,
      title={{On the Anisotropy of Score-Based Generative Models}},
      author={Andreas Floros and Seyed-Mohsen Moosavi-Dezfooli and Pier Luigi Dragotti},
      year={2025},
      eprint={2510.22899},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.22899},
}
```
