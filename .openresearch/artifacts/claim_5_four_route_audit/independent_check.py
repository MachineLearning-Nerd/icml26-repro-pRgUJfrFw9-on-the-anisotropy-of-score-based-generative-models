"""Independent integrity checker for the Claim 5 four-route audit."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_5_four_route_audit"


def main() -> None:
    raw = json.loads((ARTIFACT / "raw_results.json").read_text())
    routes = raw["routes"]
    route_by_number = {route["route"]: route for route in routes}
    comparisons = route_by_number[3]["comparisons"]
    recomputed = {}
    for dataset, row in comparisons.items():
        difference = row["I"] - row["W_min"]
        recomputed[dataset] = {
            "I_minus_W_min": difference,
            "relative_reduction_vs_I": difference / row["I"],
            "minimum_better": row["W_min"] < row["I"],
        }

    projection_checks = []
    for row in route_by_number[1]["rows"]:
        if row["executed"]:
            expected = (
                row["seconds_per_optimizer_update"]
                * row["optimizer_steps_in_paper"]
                / 3_600
            )
            projection_checks.append(
                math.isclose(
                    expected,
                    row["linear_training_projection_hours"],
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )
        else:
            projection_checks.append(row["attempted"] is True)

    checks = {
        "exactly_four_routes": sorted(route_by_number) == [1, 2, 3, 4],
        "first_three_route_names_distinct": len(
            {route_by_number[index]["name"] for index in [1, 2, 3]}
        )
        == 3,
        "route_1_attempted_all_datasets": {
            row["dataset"] for row in route_by_number[1]["rows"]
        }
        == {"MNIST", "CelebA-HQ", "CIFAR-10"},
        "route_1_projection_math": all(projection_checks),
        "route_2_no_checkpoint_not_science": (
            "do not support or contradict"
            in route_by_number[2]["scientific_interpretation"]
        ),
        "route_3_recomputed": all(
            math.isclose(
                recomputed[dataset]["I_minus_W_min"],
                row["I_minus_W_min"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            and math.isclose(
                recomputed[dataset]["relative_reduction_vs_I"],
                row["relative_reduction_vs_I"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            and recomputed[dataset]["minimum_better"]
            == row["published_order_supports_minimum"]
            for dataset, row in comparisons.items()
        ),
        "route_4_dedicated_falsification": (
            route_by_number[4]["name"]
            == "assumption-satisfying falsification search"
        ),
        "no_invalid_falsification": (
            route_by_number[4]["valid_counterexample_found"] is False
            and route_by_number[4]["published_aggregate_counterexamples"] == []
        ),
        "negative_control_rejected": raw["negative_control"]["rejected"] is True,
        "honest_blocked_verdict": (
            raw["verdict"] == "BLOCKED" and raw["confidence"] == "LOW"
        ),
        "no_gpu": raw["compute"]["gpu_used"] is False,
    }
    passed = all(checks.values())
    output = {
        "implementation": (
            "independent route-label, arithmetic, verdict, and control audit; "
            "no primary-verifier imports"
        ),
        "checks": checks,
        "recomputed_figure_9": recomputed,
        "passed": passed,
    }
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_5_AUDIT_INDEPENDENT=" + json.dumps(output, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
