"""Small helpers reconstructed from the exact judged Space's embedded code.

These functions intentionally implement only the historical toy checks. They
must never be used as evidence that the paper's empirical claims were verified.
"""

from __future__ import annotations

import numpy as np


def average_geometry_matrix(score_fn, _probe, sigma, d, n_samples, rng):
    samples = rng.standard_normal((n_samples, d)) * sigma
    scores = np.stack([score_fn(sample, sigma) for sample in samples])
    return scores.T @ scores / n_samples


def compute_sads(geometry):
    values, vectors = np.linalg.eigh(geometry)
    order = np.argsort(np.abs(values))
    return values[order], vectors[:, order]


def convergence_rate_dsm(eigenvalues, sigma, steps):
    return np.exp(-np.asarray(eigenvalues) * steps / sigma**2)


def alignment(candidate, sads, eigenvalues):
    diagonal_overlaps = np.diag(sads.T @ candidate)
    return float(np.sum(np.asarray(eigenvalues) * diagonal_overlaps**2))


def mlp_geometry_matrix(d, alpha, beta):
    return alpha * np.eye(d) + beta * np.ones((d, d))


def cnn_geometry_matrix(d, kernel_size):
    del kernel_size
    spectrum = np.resize(np.array([0.287, 0.457, 0.842]), d)
    return np.diag(np.sort(spectrum))


def transformer_geometry_matrix(tokens, d):
    nonzero = np.linspace(0.178, 1.31, tokens)
    spectrum = np.concatenate([np.zeros(max(0, d - tokens)), nonzero])[:d]
    return np.diag(spectrum)
