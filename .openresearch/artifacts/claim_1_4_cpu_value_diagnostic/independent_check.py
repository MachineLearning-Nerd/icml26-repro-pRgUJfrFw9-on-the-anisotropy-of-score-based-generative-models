"""Independent checker for the CPU value-path diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_cpu_value_diagnostic"


def median_total(condition: dict) -> float:
    return float(np.median([row["total_seconds"] for row in condition["steps"]]))


def main() -> None:
    raw = json.loads((ARTIFACT / "raw_results.json").read_text())
    conditions = raw["conditions"]
    largest_default = conditions["largest_default"]
    largest_flush = conditions["largest_flush"]
    smallest_default = conditions["smallest_default"]
    smallest_flush = conditions["smallest_flush"]

    largest_speedup = median_total(largest_default) / median_total(largest_flush)
    smallest_speedup = median_total(smallest_default) / median_total(smallest_flush)

    def loss_error(first: dict, second: dict) -> float:
        a = np.asarray([row["loss"] for row in first["steps"]], dtype=np.float64)
        b = np.asarray([row["loss"] for row in second["steps"]], dtype=np.float64)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))

    largest_loss_error = loss_error(largest_default, largest_flush)
    smallest_loss_error = loss_error(smallest_default, smallest_flush)
    checks = {
        "four_conditions": len(conditions) == 4,
        "two_steps_each": all(len(value["steps"]) == 2 for value in conditions.values()),
        "finite_positive_timings": all(
            np.isfinite(row["total_seconds"]) and row["total_seconds"] > 0
            for value in conditions.values()
            for row in value["steps"]
        ),
        "finite_losses": all(
            np.isfinite(row["loss"])
            for value in conditions.values()
            for row in value["steps"]
        ),
        "largest_direction_unit": abs(largest_default["direction_norm"] - 1.0) <= 1e-6,
        "smallest_direction_unit": abs(smallest_default["direction_norm"] - 1.0) <= 1e-6,
        "negative_control_rejected": raw["negative_control"]["rejected"] is True,
        "no_gpu": raw["compute"]["gpu_used"] is False,
    }
    passed = all(checks.values())
    output = {
        "implementation": "independent NumPy timing/loss audit; no primary-verifier imports",
        "checks": checks,
        "largest_speedup": largest_speedup,
        "smallest_speedup": smallest_speedup,
        "largest_loss_relative_error": largest_loss_error,
        "smallest_loss_relative_error": smallest_loss_error,
        "mechanism_supported": largest_speedup >= 2.0
        and largest_loss_error <= 1e-5,
        "passed": passed,
    }
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_CPU_DIAGNOSTIC_INDEPENDENT=" + json.dumps(output, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
