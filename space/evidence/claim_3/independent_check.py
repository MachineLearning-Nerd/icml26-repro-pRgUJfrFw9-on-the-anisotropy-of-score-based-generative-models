"""Independent numerical optimizer for the Claim 3 alignment extrema."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def alignment(geometry, transformed_moment):
    return float(np.trace(geometry @ transformed_moment))


def run():
    rows = []
    maximum_errors = []
    minimum_errors = []
    selected_permutations = []
    for case in range(5):
        dimension = 8 + case
        rng = np.random.default_rng(31_000 + case)
        raw_u = rng.normal(size=(dimension, dimension))
        raw_v = rng.normal(size=(dimension, dimension))
        u, _ = np.linalg.qr(raw_u)
        v, _ = np.linalg.qr(raw_v)
        lambdas = np.sort(rng.uniform(0.2, 6.0, dimension))[::-1]
        moments = np.sort(rng.uniform(0.1, 4.0, dimension))[::-1]
        geometry = u @ np.diag(lambdas) @ u.T
        data_moment = v @ np.diag(moments) @ v.T

        cost = np.outer(lambdas, moments)
        max_rows, max_cols = linear_sum_assignment(cost, maximize=True)
        min_rows, min_cols = linear_sum_assignment(cost, maximize=False)
        optimized_max = float(cost[max_rows, max_cols].sum())
        optimized_min = float(cost[min_rows, min_cols].sum())

        reversal = np.eye(dimension)[:, ::-1]
        w_max = u @ v.T
        w_min = u @ reversal @ v.T
        direct_max = alignment(geometry, w_max @ data_moment @ w_max.T)
        direct_min = alignment(geometry, w_min @ data_moment @ w_min.T)
        max_error = abs(direct_max - optimized_max)
        min_error = abs(direct_min - optimized_min)
        maximum_errors.append(max_error)
        minimum_errors.append(min_error)
        selected_permutations.append(
            {
                "dimension": dimension,
                "max_columns": max_cols.tolist(),
                "min_columns": min_cols.tolist(),
            }
        )
        rows.append(
            {
                "dimension": dimension,
                "direct_maximum": direct_max,
                "assignment_maximum": optimized_max,
                "direct_minimum": direct_min,
                "assignment_minimum": optimized_min,
                "maximum_abs_error": max_error,
                "minimum_abs_error": min_error,
            }
        )

    passed = (
        max(maximum_errors) < 1e-9
        and max(minimum_errors) < 1e-9
        and all(
            item["max_columns"] == list(range(item["dimension"]))
            and item["min_columns"] == list(range(item["dimension"] - 1, -1, -1))
            for item in selected_permutations
        )
    )
    return {
        "passed": bool(passed),
        "implementation": "scipy.optimize.linear_sum_assignment on recovered PSD spectra",
        "max_maximum_abs_error": max(maximum_errors),
        "max_minimum_abs_error": max(minimum_errors),
        "selected_permutations": selected_permutations,
        "rows": rows,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
