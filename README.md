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
