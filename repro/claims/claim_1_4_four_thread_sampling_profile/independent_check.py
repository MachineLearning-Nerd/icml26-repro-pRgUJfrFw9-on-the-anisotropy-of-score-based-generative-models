"""Independent checker for the four-thread exact sampling calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_1_4_four_thread_sampling_profile"
)


def main() -> None:
    raw = json.loads((ARTIFACT / "raw_profile.json").read_text())
    sampling_seconds = raw["sampling"]["runtime_seconds_for_1k_samples"]
    training_seconds = (
        raw["projection"]["training_seconds_per_update"]
        * raw["projection"]["training_updates"]
    )
    projected_sampling = (
        sampling_seconds * raw["projection"]["sampling_batches"]
    )
    projected_hours = (training_seconds + projected_sampling) / 3_600
    checks = {
        "four_threads": raw["compute"]["torch_intraop_threads"] == 4,
        "allocated_cpu_recorded": raw["compute"]["affinity_cpus"] is not None,
        "no_gpu": raw["compute"]["gpu_used"] is False,
        "exact_batch": raw["sampling"]["shape"] == [1_000, 1, 16, 16],
        "exact_reverse_steps": raw["setup"]["diffusion_steps"] == 1_000
        and raw["setup"]["steps_argument"] is None,
        "eta_one": raw["setup"]["eta"] == 1.0,
        "finite_output": raw["sampling"]["finite"] is True
        and np.isfinite(raw["sampling"]["mean"])
        and np.isfinite(raw["sampling"]["standard_deviation"]),
        "positive_runtime": np.isfinite(sampling_seconds)
        and sampling_seconds > 0,
        "projection_math": abs(
            projected_hours
            - raw["projection"]["projected_training_plus_sampling_hours"]
        )
        <= 1e-12,
        "within_setting_deterministic": raw["policy"][
            "four_thread_repetitions_bitwise_identical"
        ]
        is True,
        "negative_control_rejected": raw["negative_control"]["rejected"] is True,
        "status_is_calibration": "CALIBRATION_ONLY"
        in raw["scientific_status"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    output = {
        "implementation": (
            "independent arithmetic/integrity audit from raw profile; "
            "no primary-verifier imports"
        ),
        "checks": checks,
        "recomputed_projected_training_seconds": training_seconds,
        "recomputed_projected_sampling_seconds": projected_sampling,
        "recomputed_projected_training_plus_sampling_hours": projected_hours,
        "passed": passed,
    }
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(
        "CLAIM_1_4_FOUR_THREAD_PROFILE_INDEPENDENT="
        + json.dumps(output, sort_keys=True)
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
