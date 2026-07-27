"""Independent Torch spectrum check for adaptive geometry."""

from __future__ import annotations

import numpy as np
import torch


def run(matrix, eigenvalues):
    tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float64))
    values = torch.flip(torch.linalg.eigvalsh(tensor), dims=(0,))
    expected = torch.from_numpy(np.asarray(eigenvalues, dtype=np.float64))
    scale = max(float(torch.max(torch.abs(expected))), 1.0)
    error = float(torch.max(torch.abs(values - expected)) / scale)
    trace_error = abs(float(values.sum()) - float(torch.trace(tensor)))
    return {
        "passed": bool(error < 1e-8 and trace_error < 1e-8 * max(abs(float(torch.trace(tensor))), 1.0)),
        "eigenvalue_relative_error": error,
        "trace_error": trace_error,
        "implementation": "torch.linalg.eigvalsh"
    }
