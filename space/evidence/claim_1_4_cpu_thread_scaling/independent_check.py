"""Independent NumPy checker for the CPU thread-scaling diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_cpu_thread_scaling"


def main() -> None:
    raw = json.loads((ARTIFACT / "raw_results.json").read_text())
    trials = raw["trials"]
    expected = [1, 2, 4, 8, 16, 32]
    reference = trials["32"][0]
    reference_losses = np.asarray(
        [row["loss"] for row in reference["steps"]], dtype=np.float64
    )
    reference_hash = reference["final_model_sha256"]
    recomputed = {}
    safe = []
    for count in expected:
        values = trials[str(count)]
        losses = np.asarray(
            [[row["loss"] for row in trial["steps"]] for trial in values],
            dtype=np.float64,
        )
        error = float(
            np.max(
                np.abs(losses - reference_losses)
                / np.maximum(np.abs(reference_losses), 1e-12)
            )
        )
        hashes_equal = all(
            trial["final_model_sha256"] == reference_hash for trial in values
        )
        median = float(
            np.median(
                [
                    np.median(
                        [
                            row["total_seconds"]
                            for row in trial["steps"]
                            if not row["warmup"]
                        ]
                    )
                    for trial in values
                ]
            )
        )
        recomputed[str(count)] = {
            "max_loss_relative_error_vs_32": error,
            "all_state_hashes_equal_32": hashes_equal,
            "median_seconds_per_update": median,
        }
        if hashes_equal and error <= 1e-7:
            safe.append(count)
    selected = min(
        safe,
        key=lambda count: recomputed[str(count)]["median_seconds_per_update"],
    )
    checks = {
        "six_thread_counts": raw["thread_counts"] == expected,
        "two_repetitions_each": all(
            len(trials[str(count)]) == 2 for count in expected
        ),
        "three_steps_each": all(
            len(trial["steps"]) == 3
            for count in expected
            for trial in trials[str(count)]
        ),
        "finite_positive_timings": all(
            np.isfinite(row["total_seconds"]) and row["total_seconds"] > 0
            for count in expected
            for trial in trials[str(count)]
            for row in trial["steps"]
        ),
        "finite_losses": all(
            np.isfinite(row["loss"])
            for count in expected
            for trial in trials[str(count)]
            for row in trial["steps"]
        ),
        "safe_set_matches": safe == raw["safe_thread_counts"],
        "selected_matches": selected == raw["selected_thread_count"],
        "diagnostics_match": all(
            abs(
                recomputed[str(count)]["median_seconds_per_update"]
                - raw["numerical_diagnostics"][str(count)][
                    "median_seconds_per_update"
                ]
            )
            <= 1e-9
            for count in expected
        ),
        "direction_unit": abs(raw["setup"]["direction_norm"] - 1.0) <= 1e-6,
        "negative_control_rejected": raw["negative_control"]["rejected"] is True,
        "no_gpu": raw["compute"]["gpu_used"] is False,
        "scientific_status_is_diagnostic": "DIAGNOSTIC_ONLY"
        in raw["scientific_status"],
    }
    passed = all(checks.values())
    output = {
        "implementation": (
            "independent NumPy recomputation from raw per-step records; "
            "no primary-verifier imports"
        ),
        "checks": checks,
        "recomputed": recomputed,
        "safe_thread_counts": safe,
        "selected_thread_count": selected,
        "selected_seconds_per_update": recomputed[str(selected)][
            "median_seconds_per_update"
        ],
        "endpoint_resource_gate_would_pass": recomputed[str(selected)][
            "median_seconds_per_update"
        ]
        <= 12.0,
        "passed": passed,
    }
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_THREAD_SCALING_INDEPENDENT=" + json.dumps(output, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
