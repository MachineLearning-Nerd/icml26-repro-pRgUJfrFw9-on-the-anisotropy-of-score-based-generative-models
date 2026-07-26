"""Independent Torch eigendecomposition of a NumPy geometry estimate."""

from __future__ import annotations

import numpy as np
import torch


def run(matrix, numpy_eigenvalues):
    tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float64))
    values, vectors = torch.linalg.eigh(tensor)
    values = torch.flip(values, dims=(0,))
    vectors = torch.flip(vectors, dims=(1,))
    expected = torch.from_numpy(np.asarray(numpy_eigenvalues, dtype=np.float64))
    scale = max(float(torch.max(torch.abs(expected))), 1.0)
    eigenvalue_relative_error = float(torch.max(torch.abs(values - expected)) / scale)
    orthonormality_error = float(
        torch.max(torch.abs(vectors.T @ vectors - torch.eye(len(values))))
    )
    trace_error = abs(float(values.sum()) - float(torch.trace(tensor)))
    passed = (
        eigenvalue_relative_error < 1e-8
        and orthonormality_error < 1e-10
        and trace_error < 1e-8 * max(abs(float(torch.trace(tensor))), 1.0)
    )
    return {
        "passed": bool(passed),
        "implementation": "torch.linalg.eigh, independent of NumPy primary",
        "eigenvalue_relative_error": eigenvalue_relative_error,
        "orthonormality_error": orthonormality_error,
        "trace_error": trace_error,
    }
