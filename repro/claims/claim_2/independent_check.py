"""Independent numerical checks for Theorem 1.

This implementation uses a full Kronecker operator and finite differences,
rather than the coordinatewise derivation used by verify.py.
"""

from __future__ import annotations

import json

import numpy as np


def objective(theta, phi, v, sigma):
    omega = phi @ theta
    return float(
        np.linalg.norm(omega @ v) ** 2
        + np.linalg.norm(sigma * omega + np.eye(theta.shape[0]) / sigma) ** 2
    )


def analytic_gradient(theta, phi, v, sigma):
    omega = phi @ theta
    return 2.0 * phi.T @ (
        omega @ (np.outer(v, v) + sigma**2 * np.eye(theta.shape[0]))
        + np.eye(theta.shape[0])
    )


def central_difference_gradient(theta, phi, v, sigma, step=1e-6):
    estimate = np.empty_like(theta)
    for row in range(theta.shape[0]):
        for col in range(theta.shape[1]):
            plus = theta.copy()
            minus = theta.copy()
            plus[row, col] += step
            minus[row, col] -= step
            estimate[row, col] = (
                objective(plus, phi, v, sigma)
                - objective(minus, phi, v, sigma)
            ) / (2.0 * step)
    return estimate


def run():
    spectrum = np.array([8.0, 5.0, 3.0, 2.0, 1.0])
    sigma = 1.25
    phi = np.diag(np.sqrt(spectrum))
    rng = np.random.default_rng(187)
    theta = rng.normal(scale=0.1, size=(5, 5))

    v = np.eye(5)[2]
    analytic = analytic_gradient(theta, phi, v, sigma)
    finite = central_difference_gradient(theta, phi, v, sigma)
    gradient_max_abs_error = float(np.max(np.abs(analytic - finite)))

    recovered_rho = []
    formula_rho = []
    for index in range(5):
        direction = np.eye(5)[index]
        left = phi @ phi.T
        right = np.outer(direction, direction) + sigma**2 * np.eye(5)
        operator = np.kron(right.T, left)
        recovered = float(np.linalg.eigvalsh(operator).min())
        expected = float(
            min(
                (sigma**2 + 1.0) * spectrum[index],
                sigma**2 * np.min(np.delete(spectrum, index)),
            )
        )
        recovered_rho.append(recovered)
        formula_rho.append(expected)

    hessian_max_abs_error = float(
        np.max(np.abs(np.asarray(recovered_rho) - np.asarray(formula_rho)))
    )
    passed = gradient_max_abs_error < 1e-7 and hessian_max_abs_error < 1e-10
    return {
        "passed": bool(passed),
        "gradient_max_abs_error": gradient_max_abs_error,
        "hessian_max_abs_error": hessian_max_abs_error,
        "recovered_rho": recovered_rho,
        "formula_rho": formula_rho,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
