"""Machine-checkable reconstruction of Theorem 1 in arXiv:2510.22899."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
ARTIFACT = Path(".openresearch/artifacts/claim_2")
ARTIFACT.mkdir(parents=True, exist_ok=True)


def load_independent_checker():
    spec = importlib.util.spec_from_file_location(
        "claim2_independent", HERE / "independent_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def symbolic_certificate():
    sigma2, smallest, gap = sp.symbols(
        "sigma2 smallest gap", positive=True, finite=True
    )
    selected_difference = sp.simplify(
        (sigma2 + 1) * smallest - sigma2 * smallest
    )
    competitor_difference = sp.simplify(
        sigma2 * (smallest + gap) - sigma2 * smallest
    )
    return {
        "selected_difference": str(selected_difference),
        "competitor_difference": str(competitor_difference),
        "passed": selected_difference == smallest
        and competitor_difference == gap * sigma2,
    }


def exact_recurrence_check():
    spectra = [
        [sp.Rational(8), sp.Rational(5), sp.Rational(3), sp.Rational(2), sp.Rational(1)],
        [sp.Rational(21, 2), sp.Rational(7), sp.Rational(9, 2), sp.Rational(5, 2), sp.Rational(3, 4)],
    ]
    sigma2 = sp.Rational(3, 2)
    eta = sp.Rational(1, 100)
    max_error = sp.Rational(0)
    rows = []
    for spectrum in spectra:
        dimension = len(spectrum)
        A = sp.diag(*spectrum)
        identity = sp.eye(dimension)
        for index in range(dimension):
            v = identity[:, index]
            B = v * v.T + sigma2 * identity
            omega_star = (v * v.T / (sigma2 + 1) - identity) / sigma2
            error = -omega_star
            contraction = identity - 2 * eta * A * B
            rho = min(
                (sigma2 + 1) * spectrum[index],
                *[sigma2 * value for j, value in enumerate(spectrum) if j != index],
            )
            for step in range(13):
                closed = -(contraction**step) * omega_star
                difference = error - closed
                local_error = max(abs(value) for value in difference)
                max_error = max(max_error, local_error)
                error = error - 2 * eta * A * error * B
            rows.append(
                {
                    "spectrum": [str(value) for value in spectrum],
                    "direction": index,
                    "rho": str(rho),
                }
            )
    return {"passed": max_error == 0, "max_exact_error": str(max_error), "rows": rows}


def population_gd_training():
    spectrum = np.array([8.0, 5.0, 3.0, 2.0, 1.0])
    sigma2 = 1.0
    eta = 0.01
    dimension = len(spectrum)
    A = np.diag(spectrum)
    errors = []
    predictions = []
    for index in range(dimension):
        v = np.eye(dimension)[:, index]
        B = np.outer(v, v) + sigma2 * np.eye(dimension)
        omega_star = (np.outer(v, v) / (sigma2 + 1.0) - np.eye(dimension)) / sigma2
        error = -omega_star
        trajectory = [float(np.linalg.norm(error))]
        for _ in range(300):
            error = error - 2.0 * eta * A @ error @ B
            trajectory.append(float(np.linalg.norm(error)))
        rho = min(
            (sigma2 + 1.0) * spectrum[index],
            sigma2 * np.min(np.delete(spectrum, index)),
        )
        predicted = 1.0 - 2.0 * eta * rho
        observed = float(np.median(np.asarray(trajectory[-50:])[1:] / np.asarray(trajectory[-50:])[:-1]))
        errors.append(
            {
                "direction": index,
                "rho": rho,
                "predicted_factor": predicted,
                "observed_factor": observed,
                "initial_error": trajectory[0],
                "final_error": trajectory[-1],
            }
        )
        predictions.append(abs(observed - predicted))
    passed = max(predictions) < 2e-5 and errors[-1]["observed_factor"] < errors[0]["observed_factor"]
    return {"passed": bool(passed), "directions": errors, "max_factor_error": max(predictions)}


def covariance_trace_blocks():
    spectrum = np.array([8.0, 5.0, 3.0, 2.0, 1.0])
    dimension = len(spectrum)
    sigma = 1.0
    sigma2 = sigma**2
    phi = np.diag(np.sqrt(spectrum))
    samples_per_block = 12_000
    blocks = 10
    rows = []
    for index in range(dimension):
        v = np.eye(dimension)[:, index]
        omega_star = (np.outer(v, v) / (sigma2 + 1.0) - np.eye(dimension)) / sigma2
        estimates = []
        for block in range(blocks):
            rng = np.random.default_rng(20_000 + 100 * index + block)
            x_scalar = rng.standard_normal(samples_per_block)
            x = x_scalar[:, None] * v[None, :]
            epsilon = rng.standard_normal((samples_per_block, dimension))
            q = x + sigma * epsilon
            residual = q @ omega_star.T + epsilon / sigma
            p = residual @ phi
            gradients = 2.0 * p[:, :, None] * q[:, None, :]
            flattened = gradients.reshape(samples_per_block, -1)
            centered = flattened - flattened.mean(axis=0, keepdims=True)
            estimates.append(float(np.sum(centered**2) / (samples_per_block - 1)))
        theory = (
            4.0
            * (1.0 + dimension * sigma2)
            * spectrum[index]
            / (sigma2 * (sigma2 + 1.0))
        )
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates, ddof=1) / math.sqrt(blocks))
        rows.append(
            {
                "direction": index,
                "lambda": float(spectrum[index]),
                "theory_trace": theory,
                "estimated_trace": mean,
                "standard_error": standard_error,
                "relative_error": abs(mean - theory) / theory,
                "block_estimates": estimates,
            }
        )
    lambdas = np.array([row["lambda"] for row in rows])
    estimates = np.array([row["estimated_trace"] for row in rows])
    correlation = float(np.corrcoef(lambdas, estimates)[0, 1])
    max_relative_error = max(row["relative_error"] for row in rows)
    return {
        "passed": correlation > 0.999 and max_relative_error < 0.06,
        "pearson_correlation": correlation,
        "max_relative_error": max_relative_error,
        "rows": rows,
    }


def batch_one_sgd_training():
    spectrum = np.array([8.0, 5.0, 3.0, 2.0, 1.0])
    dimension = len(spectrum)
    phi = np.diag(np.sqrt(spectrum))
    sigma = 1.0
    sigma2 = 1.0
    eta = 0.001
    repeats = 96
    steps = 4_000
    rows = []
    for index in range(dimension):
        rng = np.random.default_rng(91_000 + index)
        v = np.eye(dimension)[:, index]
        theta = np.zeros((repeats, dimension, dimension))
        omega_star = (np.outer(v, v) / (sigma2 + 1.0) - np.eye(dimension)) / sigma2
        checkpoints = {}
        for step in range(steps + 1):
            if step in {0, 100, 500, 1_000, 2_000, 4_000}:
                omega = np.einsum("ij,rjk->rik", phi, theta)
                checkpoints[str(step)] = float(
                    np.mean(np.linalg.norm(omega - omega_star, axis=(1, 2)) ** 2)
                )
            if step == steps:
                break
            x_scalar = rng.standard_normal(repeats)
            x = x_scalar[:, None] * v[None, :]
            epsilon = rng.standard_normal((repeats, dimension))
            q = x + sigma * epsilon
            omega = np.einsum("ij,rjk->rik", phi, theta)
            residual = np.einsum("rij,rj->ri", omega, q) + epsilon / sigma
            gradient = 2.0 * np.einsum(
                "ij,rj,rk->rik", phi.T, residual, q
            )
            theta -= eta * gradient
        rows.append(
            {
                "direction": index,
                "lambda": float(spectrum[index]),
                "mean_squared_error": checkpoints,
            }
        )
    late_errors = np.array([row["mean_squared_error"]["4000"] for row in rows])
    correlation = float(np.corrcoef(spectrum, late_errors)[0, 1])
    return {
        "executed": True,
        "supportive_not_required_for_proof": True,
        "late_error_lambda_correlation": correlation,
        "rows": rows,
    }


def negative_control():
    spectrum = np.array([8.0, 5.0, 3.0, 2.0, 1.0])
    sigma2 = 1.0
    true_rho = []
    wrong_rho = []
    for index in range(len(spectrum)):
        true_rho.append(
            min(
                (sigma2 + 1.0) * spectrum[index],
                sigma2 * np.min(np.delete(spectrum, index)),
            )
        )
        wrong_rho.append(sigma2 * spectrum[index])
    discrepancies = np.abs(np.asarray(true_rho) - np.asarray(wrong_rho))
    rejected = bool(np.max(discrepancies) > 1.0)
    return {
        "control_claim": "rho_i = sigma^2 lambda_i for every direction",
        "expected_to_fail": True,
        "rejected": rejected,
        "true_rho": true_rho,
        "wrong_rho": wrong_rho,
        "max_abs_discrepancy": float(np.max(discrepancies)),
    }


def main():
    symbolic = symbolic_certificate()
    recurrence = exact_recurrence_check()
    gd = population_gd_training()
    covariance = covariance_trace_blocks()
    sgd = batch_one_sgd_training()
    independent = load_independent_checker().run()
    control = negative_control()
    accepted = all(
        [
            symbolic["passed"],
            recurrence["passed"],
            gd["passed"],
            covariance["passed"],
            independent["passed"],
            control["rejected"],
            sgd["executed"],
        ]
    )
    result = {
        "claim": 2,
        "verdict": "VERIFIED" if accepted else "BLOCKED",
        "accepted": bool(accepted),
        "symbolic_certificate": symbolic,
        "exact_recurrence": recurrence,
        "population_gd": gd,
        "sgd_covariance": covariance,
        "batch_one_sgd": sgd,
        "independent_checker": independent,
        "negative_control": control,
        "seed_policy": "Explicit per-route seeds in source; no seed selected after observing results.",
    }
    (ARTIFACT / "raw_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(independent, indent=2) + "\n"
    )
    (ARTIFACT / "negative_control_output.json").write_text(
        json.dumps(control, indent=2) + "\n"
    )
    print("CLAIM_2_RESULT=" + json.dumps(result, sort_keys=True))
    print(
        "CLAIM_2_SUMMARY "
        f"verdict={result['verdict']} "
        f"max_covariance_relative_error={covariance['max_relative_error']:.6f} "
        f"covariance_lambda_r={covariance['pearson_correlation']:.9f} "
        f"gd_factor_error={gd['max_factor_error']:.3e} "
        f"independent_gradient_error={independent['gradient_max_abs_error']:.3e} "
        f"negative_control_rejected={control['rejected']}"
    )
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
