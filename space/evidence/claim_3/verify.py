"""Machine-checkable reconstruction of Theorem 2 in arXiv:2510.22899."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
ARTIFACT = Path(".openresearch/artifacts/claim_3")
ARTIFACT.mkdir(parents=True, exist_ok=True)


def load_independent_checker():
    spec = importlib.util.spec_from_file_location(
        "claim3_independent", HERE / "independent_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def symbolic_exchange_certificate():
    a, b, x, y = sp.symbols("a b x y", real=True)
    sorted_value = a * x + b * y
    swapped_value = a * y + b * x
    difference = sp.factor(sorted_value - swapped_value)
    expected = (a - b) * (x - y)
    # Under a>=b and x>=y, both factors are nonnegative. Repeated adjacent
    # exchanges bubble any permutation to the identity maximum; applying the
    # same argument from the opposite order gives the reversal minimum.
    return {
        "difference": str(difference),
        "expected": str(expected),
        "identity_exact": bool(sp.simplify(difference - expected) == 0),
        "order_certificate": "a>=b and x>=y implies (a-b)(x-y)>=0",
        "global_step": "finite adjacent exchanges yield identity maximum and reversal minimum",
        "passed": bool(sp.simplify(difference - expected) == 0),
    }


def doubly_stochastic_certificate():
    # Symbolically, orthogonality gives sum_j R_ij^2=1 and
    # sum_i R_ij^2=1. Numerically audit both identities on predeclared
    # dimensions and seeds without using the objective formula.
    rows = []
    maximum_error = 0.0
    for dimension, seed in [(3, 1203), (8, 1208), (32, 1232), (64, 1264)]:
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(dimension, dimension))
        orthogonal, _ = np.linalg.qr(raw)
        squared = orthogonal**2
        row_error = float(np.max(np.abs(squared.sum(axis=1) - 1.0)))
        column_error = float(np.max(np.abs(squared.sum(axis=0) - 1.0)))
        maximum_error = max(maximum_error, row_error, column_error)
        rows.append(
            {
                "dimension": dimension,
                "seed": seed,
                "row_sum_max_error": row_error,
                "column_sum_max_error": column_error,
            }
        )
    return {
        "symbolic_row_identity": "sum_j R_ij^2 = (R R^T)_ii = 1",
        "symbolic_column_identity": "sum_i R_ij^2 = (R^T R)_jj = 1",
        "maximum_numerical_error": maximum_error,
        "rows": rows,
        "passed": bool(maximum_error < 1e-12),
    }


def exhaustive_permutation_domains():
    rows = []
    passed = True
    total_permutations = 0
    for dimension in range(2, 10):
        lambdas = tuple(range(2 * dimension, dimension, -1))
        moments = tuple(range(3 * dimension, 2 * dimension, -1))
        identity = tuple(range(dimension))
        reversal = tuple(range(dimension - 1, -1, -1))
        identity_value = sum(
            lambdas[index] * moments[index] for index in range(dimension)
        )
        reversal_value = sum(
            lambdas[index] * moments[reversal[index]]
            for index in range(dimension)
        )
        observed_maximum = None
        observed_minimum = None
        maximum_permutations = []
        minimum_permutations = []
        count = 0
        for permutation in itertools.permutations(range(dimension)):
            count += 1
            value = sum(
                lambdas[index] * moments[permutation[index]]
                for index in range(dimension)
            )
            if observed_maximum is None or value > observed_maximum:
                observed_maximum = value
                maximum_permutations = [permutation]
            elif value == observed_maximum:
                maximum_permutations.append(permutation)
            if observed_minimum is None or value < observed_minimum:
                observed_minimum = value
                minimum_permutations = [permutation]
            elif value == observed_minimum:
                minimum_permutations.append(permutation)
        local_passed = (
            observed_maximum == identity_value
            and observed_minimum == reversal_value
            and maximum_permutations == [identity]
            and minimum_permutations == [reversal]
        )
        passed = passed and local_passed
        total_permutations += count
        rows.append(
            {
                "dimension": dimension,
                "permutations_checked": count,
                "identity_maximum": identity_value,
                "observed_maximum": observed_maximum,
                "reversal_minimum": reversal_value,
                "observed_minimum": observed_minimum,
                "unique_extrema": len(maximum_permutations) == 1
                and len(minimum_permutations) == 1,
                "passed": local_passed,
            }
        )
    return {
        "passed": bool(passed),
        "dimensions": "complete permutation domains D=2..9",
        "total_permutations_checked": total_permutations,
        "rows": rows,
    }


def orthogonal_sweep():
    dimension = 32
    lambdas = np.linspace(9.0, 1.0, dimension)
    moments = np.linspace(5.0, 0.5, dimension)
    certified_maximum = float(lambdas @ moments)
    certified_minimum = float(lambdas @ moments[::-1])
    values = []
    doubly_stochastic_errors = []
    for sample in range(384):
        rng = np.random.default_rng(70_000 + sample)
        raw = rng.normal(size=(dimension, dimension))
        orthogonal, _ = np.linalg.qr(raw)
        squared = orthogonal**2
        values.append(float(lambdas @ squared @ moments))
        doubly_stochastic_errors.append(
            max(
                float(np.max(np.abs(squared.sum(axis=0) - 1.0))),
                float(np.max(np.abs(squared.sum(axis=1) - 1.0))),
            )
        )
    minimum_sample = min(values)
    maximum_sample = max(values)
    return {
        "passed": bool(
            minimum_sample >= certified_minimum - 1e-10
            and maximum_sample <= certified_maximum + 1e-10
            and max(doubly_stochastic_errors) < 1e-12
        ),
        "dimension": dimension,
        "samples": len(values),
        "seed_range": [70_000, 70_000 + len(values) - 1],
        "certified_minimum": certified_minimum,
        "sampled_minimum": minimum_sample,
        "sampled_maximum": maximum_sample,
        "certified_maximum": certified_maximum,
        "maximum_doubly_stochastic_error": max(doubly_stochastic_errors),
        "supportive_not_proof": True,
    }


def degeneracy_audit():
    lambdas = np.array([5.0, 5.0, 2.0, 1.0])
    moments = np.array([4.0, 4.0, 3.0, 0.5])
    identity_value = float(lambdas @ moments)
    reversal_value = float(lambdas @ moments[::-1])
    swapped_top = np.array([1, 0, 2, 3])
    another_maximum = float(lambdas @ moments[swapped_top])
    return {
        "passed": bool(abs(identity_value - another_maximum) < 1e-12),
        "identity_is_maximum": identity_value,
        "top_tie_swap_value": another_maximum,
        "reversal_value": reversal_value,
        "interpretation": "displayed extrema remain valid but are not unique with repeated eigenvalues",
    }


def negative_control():
    lambdas = np.array([8.0, 5.0, 3.0, 2.0, 1.0])
    moments = np.array([7.0, 4.0, 2.5, 1.5, 0.25])
    identity_value = float(lambdas @ moments)
    reversal_value = float(lambdas @ moments[::-1])
    rejected = identity_value > reversal_value
    return {
        "control_claim": "identity ordering is a minimizer for strict descending spectra",
        "expected_to_fail": True,
        "identity_value": identity_value,
        "reversal_value": reversal_value,
        "gap": identity_value - reversal_value,
        "rejected": bool(rejected),
    }


def main():
    symbolic = symbolic_exchange_certificate()
    stochastic = doubly_stochastic_certificate()
    exhaustive = exhaustive_permutation_domains()
    independent = load_independent_checker().run()
    sweep = orthogonal_sweep()
    degeneracy = degeneracy_audit()
    control = negative_control()
    accepted = all(
        [
            symbolic["passed"],
            stochastic["passed"],
            exhaustive["passed"],
            independent["passed"],
            sweep["passed"],
            degeneracy["passed"],
            control["rejected"],
        ]
    )
    result = {
        "claim": 3,
        "verdict": "VERIFIED" if accepted else "BLOCKED",
        "accepted": bool(accepted),
        "symbolic_exchange_certificate": symbolic,
        "doubly_stochastic_certificate": stochastic,
        "complete_permutation_domains": exhaustive,
        "independent_checker": independent,
        "orthogonal_sweep": sweep,
        "degeneracy_audit": degeneracy,
        "negative_control": control,
        "seed_policy": "All dimensions, cases, and seeds were fixed in source before execution.",
    }
    (ARTIFACT / "raw_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(independent, indent=2, sort_keys=True) + "\n"
    )
    (ARTIFACT / "negative_control_output.json").write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_3_RESULT=" + json.dumps(result, sort_keys=True))
    print(
        "CLAIM_3_SUMMARY "
        f"verdict={result['verdict']} "
        f"permutations={exhaustive['total_permutations_checked']} "
        f"independent_max_error={independent['max_maximum_abs_error']:.3e} "
        f"independent_min_error={independent['max_minimum_abs_error']:.3e} "
        f"control_rejected={control['rejected']}"
    )
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
